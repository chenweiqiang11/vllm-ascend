"""TurboQuant Hadamard-512 latent compress/dequant for REAL 4-bit MLA-latent storage.

Scheme = gcw tq_dequant_latent family (adopted per decision A): signed Hadamard-512,
single L2 norm per slot, 16-level Lloyd-Max codebook scaled to N(0,1/512).
Slot = 320 B uint8 = 256 nibble bytes (512x4bit) + 2 B vecNorm fp16 + 62 B pad.

compress(latent[N,512] fp16/bf16)  -> uint8[N,320]   (latent already rmsnorm'd)
dequant_paged(cache_u8[nb,bs,1,320]) -> fp16[nb,bs,1,512]   (full paged scratch for SFA)

Torch dequant (Phase 2a); the real tq_dequant_latent kernel swaps in later (Phase 2b).
"""
import math
import os as _os
import time as _time
import numpy as np
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

# host-side timing accumulators (env TQ_PROF); discriminates dispatch- vs device-bound
_PROF = {"compress": 0.0, "deq_total": 0.0, "deq_kernel": 0.0, "deq_had": 0.0,
         "deq_layout": 0.0, "nc": 0, "nd": 0}


def _prof_report():
    if _PROF["nd"] and _PROF["nd"] % 156 == 0:   # ~2 decode steps x 78 layers
        n = _PROF["nd"]
        print(f"[TQ_PROF] over {n} dequant / {_PROF['nc']} compress calls (host ms total): "
              f"compress={_PROF['compress']*1e3:.0f} deq_total={_PROF['deq_total']*1e3:.0f} "
              f"[kernel={_PROF['deq_kernel']*1e3:.0f} had={_PROF['deq_had']*1e3:.0f} "
              f"layout={_PROF['deq_layout']*1e3:.0f}]  per-call us: "
              f"compress={_PROF['compress']/max(_PROF['nc'],1)*1e6:.0f} deq={_PROF['deq_total']/n*1e6:.0f}",
              flush=True)

HEAD_DIM = 512
ROPE_HEAD_DIM = 64
SCALE_BYTES = 2
SLOT_ALIGN = 64
ROPE_DTYPE_BYTES = 2


def _is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


def _check_head_dim(head_dim):
    if not _is_power_of_2(int(head_dim)):
        raise ValueError(f"kv_lora_rank/head_dim must be a power of 2 for Sylvester Hadamard, got {head_dim}")


def _align_up(n, align):
    return ((int(n) + align - 1) // align) * align


def packed_bytes(head_dim=HEAD_DIM):
    _check_head_dim(head_dim)
    return int(head_dim) // 2


def base_slot_size(head_dim=HEAD_DIM):
    return _align_up(packed_bytes(head_dim) + SCALE_BYTES, SLOT_ALIGN)


def fused_slot_size(head_dim=HEAD_DIM, rope_head_dim=ROPE_HEAD_DIM):
    return packed_bytes(head_dim) + int(rope_head_dim) * ROPE_DTYPE_BYTES + SCALE_BYTES


SLOT_PAD = base_slot_size(HEAD_DIM)
PACKED = packed_bytes(HEAD_DIM)          # nibble bytes

_CENT = _PIT = _PI = _LUTSQ = None
_BUILT = None


def _npu_synchronize():
    if torch_npu is not None and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _require_npu():
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for this NPU-only path")


def _build(device, head_dim=HEAD_DIM):
    global _CENT, _PIT, _PI, _LUTSQ, _BUILT
    head_dim = int(head_dim)
    _check_head_dim(head_dim)
    key = (str(device), head_dim)
    if _CENT is not None and _BUILT == key:
        return
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400000) * (1.0 / math.sqrt(head_dim))
    c = np.linspace(-2 / math.sqrt(head_dim), 2 / math.sqrt(head_dim), 16)
    for _ in range(60):
        a = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        n = np.array([x[a == i].mean() if np.any(a == i) else c[i] for i in range(16)])
        if np.allclose(n, c):
            break
        c = n
    _CENT = torch.tensor(np.sort(c).astype(np.float32), device=device)
    signs = torch.tensor(rng.choice([-1.0, 1.0], head_dim).astype(np.float32), device=device)
    H = torch.tensor(np.array([[(-1) ** (bin(i & j).count("1")) for j in range(head_dim)]
                               for i in range(head_dim)], dtype=np.float32) / math.sqrt(head_dim), device=device)
    _PIT = (signs.unsqueeze(1) * H).contiguous()   # forward transform
    _PI = _PIT.t().contiguous()                    # inverse (orthonormal)
    _lc = _CENT.detach().float().cpu().numpy()
    _lt = np.zeros(256, dtype=np.float32)
    for _lb in range(256):
        _lt[_lb] = _lc[_lb & 0xF] ** 2 + _lc[(_lb >> 4) & 0xF] ** 2
    _LUTSQ = torch.tensor(_lt, device=device)  # [256] fp32 device-resident (byte -> sum of its 2 nibbles' c^2)
    _BUILT = key


@torch.no_grad()
def compress(latent, head_dim=None):
    """latent [N,head_dim] (rmsnorm'd) -> uint8 [N,base_slot_size(head_dim)] TQ slot."""
    head_dim = int(latent.shape[-1] if head_dim is None else head_dim)
    _check_head_dim(head_dim)
    packed = packed_bytes(head_dim)
    slot_pad = base_slot_size(head_dim)
    _prof = _os.environ.get("TQ_PROF")
    if _prof:
        _npu_synchronize(); _t0 = _time.perf_counter()
    _build(latent.device, head_dim)
    N = latent.shape[0]
    flat = latent.to(torch.float32)
    norms = flat.norm(dim=1, keepdim=True)                       # [N,1]
    y = (flat / (norms + 1e-8)) @ _PIT                           # [N,head_dim] Hadamard
    nib = torch.argmin((y.unsqueeze(1) - _CENT.view(1, 16, 1)).abs(), dim=1).to(torch.int32)  # [N,head_dim]
    nib4 = nib.view(N, head_dim // 4, 4)
    int16 = (nib4[:, :, 0] | (nib4[:, :, 1] << 4) | (nib4[:, :, 2] << 8) | (nib4[:, :, 3] << 12))  # [N,head_dim/4]
    lo = (int16 & 0xff).to(torch.uint8)
    hi = ((int16 >> 8) & 0xff).to(torch.uint8)
    slot = torch.zeros(N, slot_pad, dtype=torch.uint8, device=latent.device)
    slot[:, 0:packed:2] = lo
    slot[:, 1:packed:2] = hi
    norms_fp16 = norms.to(torch.float16).view(N)
    slot[:, packed:packed + SCALE_BYTES] = norms_fp16.view(torch.uint8).view(N, SCALE_BYTES)
    if _prof:
        _npu_synchronize(); _PROF["compress"] += _time.perf_counter() - _t0; _PROF["nc"] += 1
    return slot                                                  # [N,base_slot_size(head_dim)]


@torch.no_grad()
def dequant_slots(slot_u8, head_dim=HEAD_DIM):
    """slot_u8 [M,base_slot_size(head_dim)] uint8 -> fp16 [M,head_dim] (inverse-Hadamard, original basis)."""
    head_dim = int(head_dim)
    _check_head_dim(head_dim)
    packed = packed_bytes(head_dim)
    _build(slot_u8.device, head_dim)
    M = slot_u8.shape[0]
    u8 = slot_u8.view(torch.uint8)
    lo = u8[:, 0:packed:2].to(torch.int32)                       # [M,head_dim/4]
    hi = u8[:, 1:packed:2].to(torch.int32)
    int16 = (lo | (hi << 8))                                     # [M,head_dim/4]
    n0 = int16 & 0xf
    n1 = (int16 >> 4) & 0xf
    n2 = (int16 >> 8) & 0xf
    n3 = (int16 >> 12) & 0xf
    nib = torch.stack([n0, n1, n2, n3], dim=-1).reshape(M, head_dim)  # [M,head_dim]
    c = _CENT[nib.long()]                                        # [M,head_dim] post-Hadamard recon (unit-ish)
    z = c * (1.0 / torch.sqrt((c * c).sum(1, keepdim=True) + 1e-16))
    norm = u8[:, packed:packed + SCALE_BYTES].contiguous().view(torch.float16).to(torch.float32)  # [M,1]
    z = z * norm
    recon = (z @ _PI)                                            # inverse Hadamard -> orig basis
    return recon.to(torch.float16)


@torch.no_grad()
def dequant_paged(cache_u8, dtype=torch.bfloat16, head_dim=HEAD_DIM):
    """cache_u8 [nb,bs,1,base_slot_size(head_dim)] int8/uint8 -> [nb,bs,1,head_dim] paged scratch."""
    head_dim = int(head_dim)
    nb, bs, h, _ = cache_u8.shape
    flat = cache_u8.reshape(nb * bs * h, base_slot_size(head_dim))
    out = dequant_slots(flat, head_dim=head_dim).to(dtype)
    return out.reshape(nb, bs, h, head_dim)


import ctypes as _ct

# ---- ctypes aclnn binding for tq_dequant_latent (bypasses EXEC_NPU_CMD single-vendor
#      limit: dlopen OUR vendor's libcust_opapi.so directly; kernel binary via config.ini) ----
_libnn = _libcust = _libacl = None
_ACL = {torch.float16: 1, torch.int32: 3, torch.uint8: 4, torch.float32: 0}
_VENDOR_LIB = _os.environ.get("TQ_VENDOR_LIB", "libcust_opapi.so")  # all ops + dequant + compress


def _init_aclnn():
    global _libnn, _libcust, _libacl
    if _libcust is not None:
        return
    cann = "/usr/local/Ascend/cann-8.5.1"
    _libnn = _ct.CDLL(f"{cann}/aarch64-linux/lib64/libnnopbase.so")
    _libcust = _ct.CDLL(_VENDOR_LIB)
    _libacl = _ct.CDLL(f"{cann}/aarch64-linux/lib64/libascendcl.so")
    _libnn.aclCreateTensor.restype = _ct.c_void_p
    _libnn.aclCreateTensor.argtypes = [_ct.POINTER(_ct.c_int64), _ct.c_uint64, _ct.c_int,
        _ct.POINTER(_ct.c_int64), _ct.c_int64, _ct.c_int,
        _ct.POINTER(_ct.c_int64), _ct.c_uint64, _ct.c_void_p]
    _libnn.aclDestroyTensor.argtypes = [_ct.c_void_p]
    _libcust.aclnnTqDequantLatentGetWorkspaceSize.restype = _ct.c_int32
    _libcust.aclnnTqDequantLatentGetWorkspaceSize.argtypes = (
        [_ct.c_void_p] * 4 + [_ct.c_int64] * 4 + [_ct.c_void_p]
        + [_ct.POINTER(_ct.c_uint64), _ct.POINTER(_ct.c_void_p)])
    _libcust.aclnnTqDequantLatent.restype = _ct.c_int32
    _libcust.aclnnTqDequantLatent.argtypes = [_ct.c_void_p, _ct.c_uint64, _ct.c_void_p, _ct.c_void_p]
    # compress: GetWorkspaceSize(latent, centroids, slot, &ws, &exec)
    _libcust.aclnnTqCompressLatentGetWorkspaceSize.restype = _ct.c_int32
    _libcust.aclnnTqCompressLatentGetWorkspaceSize.argtypes = (
        [_ct.c_void_p] * 3 + [_ct.POINTER(_ct.c_uint64), _ct.POINTER(_ct.c_void_p)])
    _libcust.aclnnTqCompressLatent.restype = _ct.c_int32
    _libcust.aclnnTqCompressLatent.argtypes = [_ct.c_void_p, _ct.c_uint64, _ct.c_void_p, _ct.c_void_p]
    _libacl.aclrtMalloc.argtypes = [_ct.POINTER(_ct.c_void_p), _ct.c_uint64, _ct.c_int32]
    _libacl.aclrtFree.argtypes = [_ct.c_void_p]
    _libacl.aclrtSynchronizeStream.argtypes = [_ct.c_void_p]


_BINDING_LOADED = False
_TQ_CUSTOM_SO = _os.environ.get("TQ_CUSTOM_SO")


def _load_binding():
    global _BINDING_LOADED
    if _BINDING_LOADED:
        return
    if not _TQ_CUSTOM_SO:
        raise RuntimeError("TQ_CUSTOM_SO must be set when TQ_TORCHOP is enabled")
    torch.ops.load_library(_TQ_CUSTOM_SO)  # symbol resolved via LD_PRELOAD'd libcust_opapi.so
    _BINDING_LOADED = True


def _t2a(t):
    sh = t.shape
    vd = (_ct.c_int64 * len(sh))(*sh)
    st = (_ct.c_int64 * len(t.stride()))(*t.stride())
    return _libnn.aclCreateTensor(vd, len(sh), _ACL[t.dtype], st, 0, 2, vd, len(sh), _ct.c_void_p(t.data_ptr()))


def _kernel_call(kv, bt, cent, sl, kout, head_dim=HEAD_DIM):
    """Launch aclnnTqDequantLatent on the current torch stream. NO host sync: the
    kernel runs on the torch stream and subsequent torch ops on the same stream are
    FIFO-ordered after it. Workspace is a torch tensor (stream-ordered free), so no
    aclrtMalloc/Free race. Returns ws_t so the caller can keep it alive."""
    _init_aclnn()
    _require_npu()
    a = [_t2a(kv), _t2a(bt), _t2a(cent), _t2a(sl), _t2a(kout)]
    ws = _ct.c_uint64(0); exe = _ct.c_void_p()
    r = _libcust.aclnnTqDequantLatentGetWorkspaceSize(a[0], a[1], a[2], a[3], int(head_dim), 1, 4, 1, a[4],
                                                      _ct.byref(ws), _ct.byref(exe))
    assert r == 0, f"GetWorkspaceSize ret={r}"
    ws_t = torch.empty(int(ws.value) if ws.value > 0 else 1, dtype=torch.uint8, device=kout.device)
    stream = _ct.c_void_p(torch_npu.npu.current_stream().npu_stream)
    r = _libcust.aclnnTqDequantLatent(_ct.c_void_p(ws_t.data_ptr()), ws.value, exe, stream)
    assert r == 0, f"Execute ret={r}"
    for x in a:
        _libnn.aclDestroyTensor(x)
    return ws_t


@torch.no_grad()
def compress_kernel(latent, head_dim=None):
    """Fused compress via aclnnTqCompressLatent. latent [N,head_dim] (rmsnorm'd, fp16/bf16) ->
    (slot uint8 [N,base_slot_size(head_dim)], ws_t). Hadamard (1 matmul) in torch; norm/quantize/pack in kernel.
    Replaces the ~18-op torch compress with: 1 matmul + 1 kernel call."""
    _init_aclnn()
    _require_npu()
    head_dim = int(latent.shape[-1] if head_dim is None else head_dim)
    _check_head_dim(head_dim)
    _build(latent.device, head_dim)
    dev = latent.device
    N = latent.shape[0]
    z = (latent.float() @ _PIT.to(dev)).contiguous()        # Hadamard (un-normalized), fp32
    cent = _CENT.to(dev).contiguous()
    slot = torch.empty(N, base_slot_size(head_dim), dtype=torch.uint8, device=dev)
    a = [_t2a(z), _t2a(cent), _t2a(slot)]
    ws = _ct.c_uint64(0); exe = _ct.c_void_p()
    r = _libcust.aclnnTqCompressLatentGetWorkspaceSize(a[0], a[1], a[2], _ct.byref(ws), _ct.byref(exe))
    assert r == 0, f"CompressGetWorkspaceSize ret={r}"
    ws_t = torch.empty(int(ws.value) if ws.value > 0 else 1, dtype=torch.uint8, device=dev)
    stream = _ct.c_void_p(torch_npu.npu.current_stream().npu_stream)
    r = _libcust.aclnnTqCompressLatent(_ct.c_void_p(ws_t.data_ptr()), ws.value, exe, stream)
    assert r == 0, f"CompressExecute ret={r}"
    for x in a:
        _libnn.aclDestroyTensor(x)
    return slot, (ws_t, z)   # hold ws_t + z alive until caller scatters


@torch.no_grad()
def had_fwd(x, head_dim=None):
    """Forward Hadamard on the last dim: query -> Hadamard space (approach B)."""
    head_dim = int(x.shape[-1] if head_dim is None else head_dim)
    _check_head_dim(head_dim)
    _build(x.device, head_dim)
    return (x.float().reshape(-1, head_dim) @ _PIT.to(x.device)).to(x.dtype).reshape(x.shape)


@torch.no_grad()
def had_inv(x, head_dim=None):
    """Inverse Hadamard on the last dim: attention output -> orig basis (approach B)."""
    head_dim = int(x.shape[-1] if head_dim is None else head_dim)
    _check_head_dim(head_dim)
    _build(x.device, head_dim)
    return (x.float().reshape(-1, head_dim) @ _PI.to(x.device)).to(x.dtype).reshape(x.shape)


@torch.no_grad()
def dequant_for_sfa_kernel(cache_i8, rope_cache, block_table, seq_lens, dtype=torch.bfloat16, apply_inverse=True,
                           head_dim=HEAD_DIM):
    """Real-kernel variant (ctypes): tq_dequant_latent -> paged-logical latent + rope
    reordered to logical + identity block_table. kernel out [B,1,alloc,512] LOGICAL,
    pre-inverse-Hadamard. apply_inverse: A=True (inverse here); B=False (return k_had,
    caller Hadamard's q + inverse-Hadamard's the attention output)."""
    _prof = _os.environ.get("TQ_PROF")
    if _prof:
        _npu_synchronize(); _td0 = _time.perf_counter()
    head_dim = int(head_dim)
    _check_head_dim(head_dim)
    _build(cache_i8.device, head_dim)
    dev = cache_i8.device
    bs = cache_i8.shape[1]
    B, max_pages = block_table.shape
    alloc = max_pages * bs
    kv = cache_i8.view(torch.uint8).contiguous()
    bt = block_table.to(torch.int32).contiguous()
    cent = _CENT.to(dev).contiguous()
    sl = seq_lens.to(torch.int32).reshape(-1).contiguous()
    if _prof:
        _npu_synchronize(); _tk0 = _time.perf_counter()
    if _os.environ.get("TQ_TORCHOP"):   # graph-capturable torch.ops path (needs LD_PRELOAD our lib)
        _load_binding()
        # op returns [B, num_kv_heads, alloc, head_dim]; keep all batches (NO [0] batch-index,
        # which collapsed to B=1). Matches the ctypes path's [B,1,alloc,HEAD_DIM] for any B.
        kout = torch.ops.tq_custom.npu_tq_dequant_latent(kv, bt, cent, sl, head_dim, 1, 4, 1)
    else:                               # eager ctypes path
        kout = torch.zeros(B, 1, alloc, head_dim, dtype=torch.bfloat16, device=dev)  # Phase A: bf16 out
        _ws = _kernel_call(kv, bt, cent, sl, kout, head_dim=head_dim)  # hold ws alive through the ops below
    if _prof:
        _npu_synchronize(); _PROF["deq_kernel"] += _time.perf_counter() - _tk0; _th0 = _time.perf_counter()
    if apply_inverse:
        deq = (kout.reshape(B, alloc, head_dim).float() @ _PI.to(dev)).to(dtype)   # inverse Hadamard (A)
    else:
        # Phase A: kernel already emits bf16 -> no fp16->bf16 Cast (was .to(dtype))
        deq = kout.reshape(B, alloc, head_dim)                                     # k_had, no inverse (B)
    if _prof:
        _npu_synchronize(); _PROF["deq_had"] += _time.perf_counter() - _th0; _tl0 = _time.perf_counter()
    deq = deq.reshape(B * max_pages, bs, 1, head_dim).contiguous()
    bt_c = block_table.reshape(-1).clamp(min=0).to(torch.long)
    rope = rope_cache.index_select(0, bt_c).contiguous()                      # logical-ordered rope
    new_bt = torch.arange(B * max_pages, device=dev, dtype=block_table.dtype).reshape(B, max_pages)
    if _prof:
        _npu_synchronize(); _PROF["deq_layout"] += _time.perf_counter() - _tl0
        _PROF["deq_total"] += _time.perf_counter() - _td0; _PROF["nd"] += 1; _prof_report()
    if _os.environ.get("TQ_KDBG") and not globals().get("_KDBG_DONE"):
        globals()["_KDBG_DONE"] = True
        L = int(seq_lens.reshape(-1)[0].item())
        koutnz = float((kout.abs().sum(-1) > 0).float().sum().item())
        d2, r2, b2 = dequant_for_sfa(cache_i8, rope_cache, block_table, dtype=dtype, head_dim=head_dim)
        kk = deq.reshape(-1, head_dim)[:L].float(); tt = d2.reshape(-1, head_dim)[:L].float()
        cos = torch.nn.functional.cosine_similarity(kk, tt, dim=1).mean()
        print(f"[KDBG] B={B} max_pages={max_pages} bs={bs} L={L} koutnz_rows={koutnz} "
              f"bt={block_table[:1,:4].tolist()} sl={seq_lens.reshape(-1)[:4].tolist()} "
              f"deq{tuple(deq.shape)} torchdeq{tuple(d2.shape)} cos(kernel,torch)={float(cos):.4f}", flush=True)
    return deq, rope, new_bt


@torch.no_grad()
def dequant_for_sfa(cache_i8, rope_cache, block_table, dtype=torch.bfloat16, head_dim=HEAD_DIM):
    """Dequant ONLY the active blocks (those in block_table). Returns a compact
    dequanted latent + the SAME active slice of the rope cache + a remapped
    block_table (physical id -> local index). key, value(=latent) and key_rope must
    all share the compacted block dim, else SparseFlashAttention shape-checks fail.
      cache_i8 [nb,bs,1,base_slot_size(head_dim)] int8 ; rope_cache [nb,bs,1,rope] ; block_table [B,maxblk]
      -> (deq[A,bs,1,head_dim], rope[A,bs,1,rope], new_bt)
    """
    valid = block_table.reshape(-1)
    valid = valid[valid >= 0]
    if valid.numel() == 0:
        active = torch.zeros(1, dtype=torch.long, device=cache_i8.device)
    else:
        active = torch.unique(valid).to(torch.long)
    blk = cache_i8.index_select(0, active)                       # [A,bs,1,base_slot_size(head_dim)]
    deq = dequant_paged(blk, dtype=dtype, head_dim=head_dim)     # [A,bs,1,head_dim]
    rope = rope_cache.index_select(0, active).contiguous()       # [A,bs,1,rope]
    maxid = int(active.max().item()) + 1
    remap = torch.full((maxid,), -1, dtype=block_table.dtype, device=cache_i8.device)
    remap[active] = torch.arange(active.numel(), device=cache_i8.device, dtype=block_table.dtype)
    bt_c = block_table.clamp(min=0)
    new_bt = torch.where(block_table >= 0, remap[bt_c], block_table)
    return deq, rope, new_bt


def lutsq(device, head_dim=HEAD_DIM):
    """[256] fp32 LUT: byte -> _CENT[lo]^2 + _CENT[hi]^2 (its 2 nibbles). Folds 1/sqrt(sum c^2) with no
    per-nibble bitwise unpack -> graph-capture safe (gather+sum only, no RightShift/And aclop)."""
    _build(device, head_dim)
    return _LUTSQ
