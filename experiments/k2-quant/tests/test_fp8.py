"""FP8 E4M3 (重みだけ、出力チャネルごとの対称スケール) の数値変換の試験 (C1)。

期待は、現行実装ではなく、計画の完了契約 C1 の式から導く:

- `scale = amax(|W|, dim=1, keepdim=True) / 448` (float32、0 の行は `finfo(float32).eps`)
- `Q = clamp(W / scale, -448, 448).to(float8_e4m3fn)`
- 逆量子化 `Q.float() * scale` の誤差は、要素ごとに
  `|W - deq| <= 2**-4 * |W| + 2**-10 * scale + 1e-6 * scale`、全体の相対 Frobenius 誤差 <= 0.05
"""

from __future__ import annotations

import pytest
import torch

from k2_quant import fp8

FP8_MAX = 448.0


def _weight(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """全 0 の行と、1 要素だけ大きい行を含む入力。"""
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(4, 6, generator=generator)
    weight[0] = 0.0
    weight[1, 2] = 500.0
    return weight.to(dtype)


def test_fp8_max_is_e4m3_range() -> None:
    """E4M3 の上限は 448 (C1)。"""
    assert fp8.FP8_MAX == FP8_MAX


def test_quantize_returns_weight_and_channel_scale() -> None:
    """`weight` は入力と同じ形の F8_E4M3、`weight_scale` は (out, 1) の float32 (C1)。"""
    weight = _weight()

    quantized, scale = fp8.quantize_per_channel(weight)

    assert quantized.dtype == torch.float8_e4m3fn
    assert tuple(quantized.shape) == tuple(weight.shape)
    assert scale.dtype == torch.float32
    assert tuple(scale.shape) == (weight.shape[0], 1)


def test_quantized_values_stay_finite_and_within_fp8_range() -> None:
    """448 を超える要素を clamp し、NaN や inf を出さない (C1)。"""
    weight = _weight()
    assert float(weight.abs().max()) > FP8_MAX

    quantized, _ = fp8.quantize_per_channel(weight)
    as_float = quantized.to(torch.float32)

    assert bool(torch.isfinite(as_float).all())
    assert float(as_float.abs().max()) <= FP8_MAX


def test_max_element_of_each_row_returns_to_amax() -> None:
    """各行の絶対最大の要素は ±448 に写り、逆量子化で amax に戻る (C1)。"""
    weight = _weight().to(torch.float32)
    quantized, scale = fp8.quantize_per_channel(weight)
    dequantized = fp8.dequantize_per_channel(quantized, scale)

    amax = weight.abs().amax(dim=1, keepdim=True)
    nonzero_rows = (amax[:, 0] > 0).nonzero().flatten()
    assert len(nonzero_rows) == 3

    for row in nonzero_rows.tolist():
        assert float(quantized.to(torch.float32)[row].abs().max()) == FP8_MAX
        assert abs(float(dequantized[row].abs().max()) - float(amax[row, 0])) <= 1e-5


def test_dequantize_error_is_bounded_per_element() -> None:
    """要素ごとの誤差が `2**-4 * |W| + 2**-10 * scale + 1e-6 * scale` に収まる (C1)。"""
    weight = _weight().to(torch.float32)
    quantized, scale = fp8.quantize_per_channel(weight)
    dequantized = fp8.dequantize_per_channel(quantized, scale)

    bound = 2**-4 * weight.abs() + 2**-10 * scale + 1e-6 * scale
    assert bool(((weight - dequantized).abs() <= bound).all())


def test_dequantize_relative_frobenius_error_is_below_limit() -> None:
    """全体の相対 Frobenius 誤差が 0.05 以下 (C1)。"""
    generator = torch.Generator().manual_seed(1)
    weight = torch.randn(64, 128, generator=generator).to(torch.bfloat16)

    quantized, scale = fp8.quantize_per_channel(weight)
    dequantized = fp8.dequantize_per_channel(quantized, scale)

    reference = weight.to(torch.float32)
    relative = float((reference - dequantized).norm() / reference.norm())
    assert relative <= 0.05


def test_zero_row_dequantizes_to_zero_without_nan() -> None:
    """全 0 の行は 0 除算せず、有限のスケールと 0 の逆量子化を返す (C1)。"""
    weight = _weight()
    weight[0] = 0.0

    quantized, scale = fp8.quantize_per_channel(weight)
    dequantized = fp8.dequantize_per_channel(quantized, scale)

    assert float(scale[0, 0]) > 0.0
    assert bool(torch.isfinite(scale).all())
    assert bool((dequantized[0] == 0).all())


def test_dequantize_multiplies_scale_after_cast_to_float32() -> None:
    """逆量子化は `Q.to(float32) * scale` と同じ (C1、compressed-tensors の `_dequantize`)。"""
    weight = _weight()
    quantized, scale = fp8.quantize_per_channel(weight)

    expected = quantized.to(torch.float32) * scale

    assert torch.equal(fp8.dequantize_per_channel(quantized, scale), expected)


def test_quantize_does_not_modify_input() -> None:
    """量子化は入力を書き換えない (C1)。"""
    weight = _weight()
    before = weight.clone()

    fp8.quantize_per_channel(weight)

    assert torch.equal(weight, before)


# --- FP8 の block 量子化を戻す (`--preset k2s4` の専門家の入力。C2b。Issue #99) ------------------


def _reference_dequantize_block(
    quantized: torch.Tensor, scale: torch.Tensor, block: tuple[int, int]
) -> torch.Tensor:
    """独立参照実装: 各要素を、その要素が属するブロックの scale で戻す (1 要素ずつの索引で計算し、
    `repeat_interleave` は使わない)。
    """
    block_h, block_w = block
    rows, cols = quantized.shape
    result = torch.zeros(rows, cols, dtype=torch.float32)
    for row in range(rows):
        for col in range(cols):
            result[row, col] = float(quantized[row, col].to(torch.float32)) * float(
                scale[row // block_h, col // block_w].to(torch.float32)
            )
    return result


def test_dequantize_block_matches_the_independent_reference_for_a_non_square_block() -> None:
    """`dequantize_block` は、各要素をそのブロックの scale で戻す (C2b)。

    非正方の形 (4, 6) と block (2, 3)、非対称な scale (行と列を入れ替えると違う値になる) を使い、
    行方向だけの展開や scale の転置といった誤りを検出できるようにする。
    """
    quantized = torch.tensor(
        [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0], [4.0, 6.0, -0.5, -1.0, -1.5, -2.0]] * 2,
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [10.0, 20.0]], dtype=torch.float32)

    result = fp8.dequantize_block(quantized, scale, (2, 3))

    assert torch.equal(result, _reference_dequantize_block(quantized, scale, (2, 3)))


def test_dequantize_block_does_not_modify_the_inputs() -> None:
    """`dequantize_block` は `quantized`・`scale` を書き換えない (C2b)。"""
    quantized = torch.zeros(2, 2, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, 1, dtype=torch.float32) * 3.0
    quantized_before = quantized.clone()
    scale_before = scale.clone()

    fp8.dequantize_block(quantized, scale, (2, 2))

    assert torch.equal(quantized.to(torch.float32), quantized_before.to(torch.float32))
    assert torch.equal(scale, scale_before)


def test_validate_block_shapes_accepts_a_matching_weight_and_scale_shape() -> None:
    """出力・入力の次元が block の倍数で、`scale` の形が `(out/bh, in/bw)` なら通す (C2b)。"""
    fp8.validate_block_shapes("m", (256, 128), (2, 1), (128, 128))


def test_validate_block_shapes_rejects_an_output_dimension_not_divisible_by_the_block() -> None:
    """出力の次元 (行数) が block の高さで割り切れなければ拒否する (C2b)。"""
    with pytest.raises(ValueError):
        fp8.validate_block_shapes("m", (200, 128), (2, 1), (128, 128))


def test_validate_block_shapes_rejects_an_input_dimension_not_divisible_by_the_block() -> None:
    """入力の次元 (列数) が block の幅で割り切れなければ拒否する (C2b)。"""
    with pytest.raises(ValueError):
        fp8.validate_block_shapes("m", (256, 100), (2, 1), (128, 128))


def test_validate_block_shapes_rejects_a_scale_shape_that_does_not_match_the_block_count() -> None:
    """`scale` の形が `(out/bh, in/bw)` と違えば拒否する (C2b)。"""
    with pytest.raises(ValueError):
        fp8.validate_block_shapes("m", (256, 128), (1, 1), (128, 128))
