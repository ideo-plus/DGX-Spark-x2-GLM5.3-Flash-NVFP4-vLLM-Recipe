"""NVFP4A16 (重みだけ、E2M1 4 bit + 16 要素ごとの FP8 スケール + テンソルごとの全体スケール) の
数値変換の試験 (C2)。

期待は、現行実装ではなく、計画の完了契約 C2 の式から導く:

- `global_scale_for(amax) = 448 * 6 / amax` (`amax <= 0` は 0 除算を避けて `1.0`)
- `quantize_tensor_group(W, G)`: 16 要素のブロックごとに
  `s8 = clamp(G * vmax / 6, 0, 448).to(float8_e4m3fn)`、
  `q = cast_to_fp4(clamp(W * (G / s8), -6, 6))` (`s8 == 0` の要素は 0 のまま)
- 逆量子化 `q * (s8.float() / G)` の誤差は、要素ごとに `|W - deq| <= s8.float() / G` (ブロックへ
  展開した値)。全体の相対 Frobenius 誤差は、`randn(64, 128)` (`manual_seed(1)`、bfloat16 を
  経由) で、実装 (`nvfp4.quantize_tensor_group`/`dequantize_tensor_group`) の実測値が
  0.09467066079378128 (2026-09-27) であり、上限の 0.2 に 2 倍以上の余裕がある。上限を超えたら、
  期待値を緩めず計画に戻す
- パック順は下位 nibble が偶数列、上位 nibble が奇数列。符号は bit 3。E2M1 の値は
  `(0, 0.5, 1, 1.5, 2, 3, 4, 6)`
- 対象の入力の次元 (列数) が 16 の倍数でなければ、`Nvfp4Error` (`ValueError` の派生) で断る
"""

from __future__ import annotations

import pytest
import torch

from k2_quant import nvfp4

FP4_MAX = 6.0
GROUP_SIZE = 16


def _weight(rows: int = 4, cols: int = 32, *, seed: int) -> torch.Tensor:
    """全 0 の行と、1 要素だけ大きい行を含む float32 の入力。"""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=generator)
    weight[0] = 0.0
    if rows >= 2:
        weight[1, -1] = 500.0
    return weight


def _amax(weight: torch.Tensor) -> float:
    return float(weight.abs().amax())


def _unpack_independently(
    packed: torch.Tensor, scale: torch.Tensor, global_scale: float
) -> torch.Tensor:
    """道具とは別に、下位 nibble = 偶数列・上位 nibble = 奇数列・bit 3 = 符号、という契約を
    直接実装して逆量子化する (道具の `dequantize_tensor_group` との一致を確かめるための、
    独立した参照実装)。"""
    out, half = packed.shape
    low = (packed.to(torch.int64)) & 0x0F
    high = (packed.to(torch.int64) >> 4) & 0x0F
    table = torch.tensor(nvfp4.E2M1_VALUES, dtype=torch.float32)
    values = torch.empty(out, half * 2, dtype=torch.float32)
    for nibble, columns in ((low, range(0, half * 2, 2)), (high, range(1, half * 2, 2))):
        magnitude = table[nibble & 0x07]
        sign = torch.where((nibble & 0x08) != 0, torch.tensor(-1.0), torch.tensor(1.0))
        values[:, list(columns)] = magnitude * sign
    block_scale = scale.to(torch.float32) / global_scale
    expanded = block_scale.repeat_interleave(GROUP_SIZE, dim=1)
    return values * expanded


def test_group_size_and_fp4_max_match_the_nvfp4_spec() -> None:
    """ブロックの大きさは 16、E2M1 の絶対値の上限は 6 (C2)。"""
    assert nvfp4.GROUP_SIZE == GROUP_SIZE
    assert nvfp4.FP4_MAX == FP4_MAX


def test_e2m1_values_are_the_eight_representable_magnitudes() -> None:
    """E2M1 の表現できる絶対値は `(0, 0.5, 1, 1.5, 2, 3, 4, 6)` の 8 個 (C2)。"""
    assert tuple(nvfp4.E2M1_VALUES) == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def test_nvfp4_error_is_a_value_error() -> None:
    """`Nvfp4Error` は `ValueError` の派生で、`__main__` が終了 1 にできる (C2)。"""
    assert issubclass(nvfp4.Nvfp4Error, ValueError)


def test_validate_shape_accepts_a_two_dimensional_shape_whose_input_is_a_multiple_of_16() -> None:
    """2 次元で、入力の次元 (列数) が 16 の倍数なら通す (C2)。"""
    nvfp4.validate_shape("module", (4, 32))
    nvfp4.validate_shape("module", (4, 16))


def test_validate_shape_rejects_an_input_dimension_that_is_not_a_multiple_of_16() -> None:
    """入力の次元が 16 の倍数でなければ `Nvfp4Error` で断る (C2)。"""
    with pytest.raises(nvfp4.Nvfp4Error):
        nvfp4.validate_shape("module", (4, 20))


def test_validate_shape_rejects_a_shape_that_is_not_two_dimensional() -> None:
    """2 次元でない形は `Nvfp4Error` で断る (C2)。"""
    with pytest.raises(nvfp4.Nvfp4Error):
        nvfp4.validate_shape("module", (16,))
    with pytest.raises(nvfp4.Nvfp4Error):
        nvfp4.validate_shape("module", (4, 16, 2))


def test_global_scale_for_non_positive_amax_returns_one_without_dividing_by_zero() -> None:
    """`amax <= 0` は 0 除算を避けて `1.0` を返す (C2)。"""
    assert nvfp4.global_scale_for(0.0) == 1.0


def test_global_scale_for_positive_amax_follows_the_reference_formula() -> None:
    """`amax` が正なら `448 * 6 / amax` (C2)。"""
    assert nvfp4.global_scale_for(12.0) == 448.0 * 6.0 / 12.0


def test_quantize_returns_the_contract_shapes_and_dtypes_for_packed_scale_and_global_scale() -> (
    None
):
    """`weight_packed` は U8 (out, in/2)、`weight_scale` は F8_E4M3 (out, in/16)、
    `weight_global_scale` は F32 (1,) で、渡した全体スケールの値を持つ (C2)。"""
    weight = _weight(rows=4, cols=32, seed=1)
    global_scale = nvfp4.global_scale_for(_amax(weight))

    packed, scale, global_scale_tensor = nvfp4.quantize_tensor_group(weight, global_scale)

    assert packed.dtype == torch.uint8
    assert tuple(packed.shape) == (4, 16)
    assert scale.dtype == torch.float8_e4m3fn
    assert tuple(scale.shape) == (4, 2)
    assert global_scale_tensor.dtype == torch.float32
    assert tuple(global_scale_tensor.shape) == (1,)
    # float32 に丸めた値と比べる (`global_scale` は Python の float64 で計算している)
    assert float(global_scale_tensor[0]) == pytest.approx(global_scale, rel=1e-6)


def test_quantize_does_not_modify_the_input() -> None:
    """量子化は入力を書き換えない (C2)。"""
    weight = _weight(rows=4, cols=32, seed=2)
    before = weight.clone()
    global_scale = nvfp4.global_scale_for(_amax(weight))

    nvfp4.quantize_tensor_group(weight, global_scale)

    assert torch.equal(weight, before)


def test_zero_block_dequantizes_to_zero_without_nan() -> None:
    """全 0 のブロックは 0 除算せず、有限のスケールと 0 の逆量子化を返す (C2)。"""
    weight = _weight(rows=4, cols=32, seed=3)
    weight[0] = 0.0
    global_scale = nvfp4.global_scale_for(_amax(weight))

    packed, scale, global_scale_tensor = nvfp4.quantize_tensor_group(weight, global_scale)
    dequantized = nvfp4.dequantize_tensor_group(packed, scale, global_scale_tensor)

    assert bool(torch.isfinite(scale.to(torch.float32)).all())
    assert bool((dequantized[0] == 0).all())


def test_dequantize_error_is_bounded_by_the_blocks_scale() -> None:
    """要素ごとの誤差は、その要素のブロックのスケール `weight_scale.float() / G` 以下 (C2)。"""
    weight = _weight(rows=4, cols=32, seed=4)
    global_scale = nvfp4.global_scale_for(_amax(weight))

    packed, scale, global_scale_tensor = nvfp4.quantize_tensor_group(weight, global_scale)
    dequantized = nvfp4.dequantize_tensor_group(packed, scale, global_scale_tensor)

    g = float(global_scale_tensor[0])
    block_scale = scale.to(torch.float32) / g
    bound = block_scale.repeat_interleave(GROUP_SIZE, dim=1)
    assert bool(((weight - dequantized).abs() <= bound + 1e-6).all())


def test_dequantize_relative_frobenius_error_is_below_the_limit() -> None:
    """全体の相対 Frobenius 誤差が 0.2 以下 (C2)。

    `randn(64, 128)` (`manual_seed(1)`、bfloat16 を経由) の実装での実測値は
    0.09467066079378128 (2026-09-27) で、上限 0.2 に 2 倍以上の余裕がある。
    """
    generator = torch.Generator().manual_seed(1)
    weight = torch.randn(64, 128, generator=generator).to(torch.bfloat16).to(torch.float32)
    global_scale = nvfp4.global_scale_for(_amax(weight))

    packed, scale, global_scale_tensor = nvfp4.quantize_tensor_group(weight, global_scale)
    dequantized = nvfp4.dequantize_tensor_group(packed, scale, global_scale_tensor)

    relative = float(
        torch.linalg.vector_norm(weight - dequantized) / torch.linalg.vector_norm(weight)
    )
    assert relative <= 0.2


def test_independent_unpack_of_the_packed_bytes_matches_the_tools_dequantization() -> None:
    """試験側で別実装した unpack (下位 nibble = 偶数列、上位 nibble = 奇数列、bit 3 = 符号) が、
    道具の `dequantize_tensor_group` と一致する (C2)。

    パック順の契約 (上位/下位 nibble のどちらが偶数列か、符号ビットの位置) を、逆量子化の式だけで
    検出できない退行から守る。
    """
    weight = _weight(rows=4, cols=32, seed=5)
    global_scale = nvfp4.global_scale_for(_amax(weight))

    packed, scale, global_scale_tensor = nvfp4.quantize_tensor_group(weight, global_scale)
    from_tool = nvfp4.dequantize_tensor_group(packed, scale, global_scale_tensor)
    from_independent_unpack = _unpack_independently(packed, scale, float(global_scale_tensor[0]))

    assert torch.equal(from_tool, from_independent_unpack)
