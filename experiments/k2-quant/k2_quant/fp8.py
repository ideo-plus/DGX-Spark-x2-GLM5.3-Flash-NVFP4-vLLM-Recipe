"""FP8 E4M3 (重みだけ、出力チャネルごとの対称スケール) の数値変換。

compressed-tensors の `float-quantized` の、チャネルごと静的スケールの形に合わせる:

- スケールは `amax(|W|, dim=1, keepdim=True) / 448` (float32)。0 の行は
  `torch.finfo(torch.float32).eps` に置き換える。
- 量子化は `clamp(W / scale, -448, 448).to(torch.float8_e4m3fn)`。
- 逆量子化は `Q.to(scale.dtype) * scale` (`_dequantize` と同じ式)。

第 4 段 (Issue #99) の FP8 の専門家は、上と別の block ごとの静的スケール (`strategy=block`。
`quant_config.fp8_block_structure` が `config.json` の `group_1` から読む) で量子化されている。
`validate_block_shapes`・`dequantize_block` は、この block 量子化を戻す。
"""

from __future__ import annotations

from typing import Final

import torch

FP8_DTYPE_NAME: Final = "F8_E4M3"
SCALE_DTYPE_NAME: Final = "F32"
FP8_MAX: Final[float] = float(torch.finfo(torch.float8_e4m3fn).max)


def quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """2 次元の重みを、F8_E4M3 の `weight` と `(out, 1)` の F32 `weight_scale` にする。"""
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-dimensional, got shape {tuple(weight.shape)}")
    as_float = weight.to(torch.float32)
    amax = as_float.abs().amax(dim=1, keepdim=True)
    scale = amax / FP8_MAX
    scale = torch.where(scale == 0, torch.full_like(scale, torch.finfo(torch.float32).eps), scale)
    quantized = torch.clamp(as_float / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized, scale


def dequantize_per_channel(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """`Q.to(float32) * scale` (compressed-tensors の `_dequantize` と同じ式)。"""
    return quantized.to(torch.float32) * scale


def validate_block_shapes(
    name: str,
    weight_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
    block_structure: tuple[int, int],
) -> None:
    """block 量子化の `weight` と `weight_scale` の形が、block の大きさと整合するかを確かめる
    (Issue #99)。

    `weight` が 2 次元で、出力の次元 (行数) が block の高さで、入力の次元 (列数) が block の幅で
    割り切れ、`weight_scale` の形がちょうど `(行数 // 高さ, 列数 // 幅)` でなければ `ValueError`。
    """
    block_height, block_width = block_structure
    if len(weight_shape) != 2:
        raise ValueError(f"{name}: weight is not 2-dimensional: {weight_shape}")
    out_features, in_features = weight_shape
    if out_features % block_height != 0:
        raise ValueError(
            f"{name}: output dimension {out_features} is not a multiple of "
            f"block height {block_height}"
        )
    if in_features % block_width != 0:
        raise ValueError(
            f"{name}: input dimension {in_features} is not a multiple of block width {block_width}"
        )
    expected_scale_shape = (out_features // block_height, in_features // block_width)
    if tuple(scale_shape) != expected_scale_shape:
        raise ValueError(
            f"{name}: weight_scale shape {tuple(scale_shape)} != expected {expected_scale_shape}"
        )


def dequantize_block(
    quantized: torch.Tensor, scale: torch.Tensor, block_structure: tuple[int, int]
) -> torch.Tensor:
    """block ごとの静的スケールで量子化された `weight` を戻す (Issue #99)。

    `quantized.to(float32) * scale.to(float32)` を、`scale` の各要素を block の大きさへ展開して
    掛ける。入力 (`quantized`・`scale`) は書き換えない。
    """
    block_height, block_width = block_structure
    expanded_scale = (
        scale.to(torch.float32)
        .repeat_interleave(block_height, dim=0)
        .repeat_interleave(block_width, dim=1)
    )
    return quantized.to(torch.float32) * expanded_scale
