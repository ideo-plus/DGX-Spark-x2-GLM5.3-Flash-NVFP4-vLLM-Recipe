"""FP8 E4M3 (重みだけ、出力チャネルごとの対称スケール) の数値変換の試験 (C1)。

期待は、現行実装ではなく、計画の完了契約 C1 の式から導く:

- `scale = amax(|W|, dim=1, keepdim=True) / 448` (float32、0 の行は `finfo(float32).eps`)
- `Q = clamp(W / scale, -448, 448).to(float8_e4m3fn)`
- 逆量子化 `Q.float() * scale` の誤差は、要素ごとに
  `|W - deq| <= 2**-4 * |W| + 2**-10 * scale + 1e-6 * scale`、全体の相対 Frobenius 誤差 <= 0.05
"""

from __future__ import annotations

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
