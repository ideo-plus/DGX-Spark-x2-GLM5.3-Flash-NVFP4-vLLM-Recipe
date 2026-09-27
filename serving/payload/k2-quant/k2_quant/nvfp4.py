"""NVFP4A16 (重みだけ、E2M1 4 bit を 2 つずつ詰めた U8 + 16 要素ごとの FP8 E4M3 のスケール +
テンソルごとの FP32 の全体スケール) の数値変換 (Issue #95)。

式の出典は、固定 vLLM (commit 0961bbae) の
`vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py` の参照実装
(`ref_nvfp4_quant`・`dequantize_to_dtype`・`break_fp4_bytes`)。要点:

- ブロック (16 要素) ごとのスケールは `s8 = clamp(G * vmax / 6, 0, 448)` を F8_E4M3 に丸めた値。
  全体スケール `G` は compressed-tensors の側で「除数」(`1/scale`) として保存されるので、
  `s = s8 / G` が実際の量子化ステップ幅になる (`dequantize_tensor_group` の `block_scale`)。
- パック順は下位 nibble が偶数列、上位 nibble が奇数列。符号は bit 3。
"""

from __future__ import annotations

from typing import Final

import torch

from k2_quant.fp8 import FP8_MAX

GROUP_SIZE: Final = 16
FP4_MAX: Final[float] = 6.0
PACKED_DTYPE_NAME: Final = "U8"
SCALE_DTYPE_NAME: Final = "F8_E4M3"
GLOBAL_SCALE_DTYPE_NAME: Final = "F32"
E2M1_VALUES: Final[tuple[float, ...]] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class Nvfp4Error(ValueError):
    """NVFP4A16 の前提 (形など) が崩れている。"""


def validate_shape(name: str, shape: tuple[int, ...]) -> None:
    """2 次元で、入力の次元 (列数) が `GROUP_SIZE` の倍数であることを確かめる。"""
    if len(shape) != 2 or shape[1] % GROUP_SIZE != 0:
        last = shape[-1] if shape else 0
        raise Nvfp4Error(f"{name}: input dimension {last} is not a multiple of {GROUP_SIZE}")


def global_scale_for(amax: float) -> float:
    """テンソル (または組) の全体スケール。`amax <= 0` は 0 除算を避けて `1.0`。"""
    if amax <= 0:
        return 1.0
    return FP8_MAX * FP4_MAX / amax


_FP4_BOUNDARIES: Final[tuple[tuple[float, bool, float], ...]] = (
    # (上限, 上限を含むか, その区間の値)。vLLM `nvfp4_emulation_utils.py` の境界 (同点は偶数側)。
    (0.25, True, 0.0),
    (0.75, False, 0.5),
    (1.25, True, 1.0),
    (1.75, False, 1.5),
    (2.5, True, 2.0),
    (3.5, False, 3.0),
    (5.0, True, 4.0),
)


def _cast_to_fp4_abs(magnitude: torch.Tensor) -> torch.Tensor:
    """E2M1 の絶対値への丸め (vLLM `nvfp4_emulation_utils.py` の境界。同点は偶数側)。"""
    out = torch.full_like(magnitude, FP4_MAX)
    for boundary, inclusive, value in reversed(_FP4_BOUNDARIES):
        within = magnitude <= boundary if inclusive else magnitude < boundary
        out = torch.where(within, torch.full_like(magnitude, value), out)
    return out


def quantize_tensor_group(
    weight: torch.Tensor, global_scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2 次元の重みを、NVFP4A16 の 3 テンソル (`weight_packed`, `weight_scale`,
    `weight_global_scale`) にする。入力は書き換えない。"""
    out, in_features = tuple(int(dim) for dim in weight.shape)
    validate_shape("weight", (out, in_features))
    x = weight.to(torch.float32).reshape(out, in_features // GROUP_SIZE, GROUP_SIZE)
    vmax = x.abs().amax(dim=-1, keepdim=True)
    s8 = torch.clamp(global_scale * vmax / FP4_MAX, 0.0, FP8_MAX).to(torch.float8_e4m3fn)
    s8f = s8.to(torch.float32)
    inv = torch.where(s8f == 0, torch.zeros_like(s8f), global_scale / s8f)
    scaled = torch.clamp(x * inv, -FP4_MAX, FP4_MAX)
    sign = torch.sign(scaled)
    magnitude = _cast_to_fp4_abs(scaled.abs())

    table = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    idx = torch.zeros(out, in_features // GROUP_SIZE, GROUP_SIZE, dtype=torch.uint8)
    for value_index, value in enumerate(table):
        idx = torch.where(
            torch.isclose(magnitude, value, atol=1e-4), torch.full_like(idx, value_index), idx
        )
    idx = torch.where(sign < 0, idx | 0x8, idx)
    idx = idx.reshape(out, in_features)
    low = idx[:, 0::2]
    high = idx[:, 1::2]
    packed = (low | (high << 4)).to(torch.uint8)

    return (
        packed,
        s8.reshape(out, in_features // GROUP_SIZE),
        torch.tensor([global_scale], dtype=torch.float32),
    )


def dequantize_tensor_group(
    packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """`quantize_tensor_group` の逆。`global_scale` は `(1,)` の float32 テンソル。"""
    out, half = packed.shape
    low = packed.to(torch.int64) & 0x0F
    high = (packed.to(torch.int64) >> 4) & 0x0F
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    values = torch.empty(out, half * 2, dtype=torch.float32)
    for nibble, columns in ((low, range(0, half * 2, 2)), (high, range(1, half * 2, 2))):
        magnitude = table[nibble & 0x07]
        sign = torch.where((nibble & 0x08) != 0, torch.tensor(-1.0), torch.tensor(1.0))
        values[:, list(columns)] = magnitude * sign
    block_scale = scale.to(torch.float32) / global_scale
    return values * block_scale.repeat_interleave(GROUP_SIZE, dim=1)
