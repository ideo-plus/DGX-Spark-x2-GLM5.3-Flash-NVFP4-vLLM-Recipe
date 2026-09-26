"""FP8 E4M3 (重みだけ、出力チャネルごとの対称スケール) の数値変換。

compressed-tensors の `float-quantized` の、チャネルごと静的スケールの形に合わせる:

- スケールは `amax(|W|, dim=1, keepdim=True) / 448` (float32)。0 の行は
  `torch.finfo(torch.float32).eps` に置き換える。
- 量子化は `clamp(W / scale, -448, 448).to(torch.float8_e4m3fn)`。
- 逆量子化は `Q.to(scale.dtype) * scale` (`_dequantize` と同じ式)。
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
