# SPDX-License-Identifier: Apache-2.0

from tests.ut.attention.a2.test_turboquant_custom_ops import (
    test_turboquant_compress_npu_matches_reference as _test_compress,
)
from tests.ut.attention.a2.test_turboquant_custom_ops import (
    test_turboquant_sfa_npu_matches_reference as _test_sfa,
)


def test_turboquant_compress_npu_matches_reference() -> None:
    _test_compress()


def test_turboquant_sfa_npu_matches_reference() -> None:
    _test_sfa()
