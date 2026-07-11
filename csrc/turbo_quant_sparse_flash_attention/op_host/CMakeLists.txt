add_ops_compile_options(
        OP_NAME TurboQuantSparseFlashAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

target_sources(op_host_aclnn PRIVATE
        turbo_quant_sparse_flash_attention_def.cpp
)

target_sources(optiling PRIVATE
        turbo_quant_sparse_flash_attention_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        turbo_quant_sparse_flash_attention_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        turbo_quant_sparse_flash_attention_infershape.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
