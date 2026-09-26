"""対象の選択 (名前の規則と正規表現)。

正規表現は、入力 checkpoint のテンソル名 (`model.language_model.layers.N....`) から
`.weight` を除いたモジュール名に `re.match` で当てて、変換するモジュールを選ぶ。vLLM が
実行時の層名に当てる target ではない。`config.json` に足す target は、選ばれた
モジュールから `quant_config` が作る (この正規表現は書かない)。

既定は Issue #56 の第 1 段の範囲: dense の MLP (層 0〜2)、共有の専門家、lm_head。MTP の
`eh_proj` は既定から外す。vLLM (commit 0961bbae、`vllm/models/glm5next/common/mtp.py:49`)
の `eh_proj` は `self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size,
bias=False)` で、`quant_config` を受けない plain `nn.Linear` なので、FP8 の `eh_proj` を
読み込めない。一方、MTP の head (`shared_head.head`) は `SharedHead`
(`vllm/model_executor/models/deepseek_mtp.py`) の中の `ParallelLMHead` で、`quant_config`
を受けるので、既定に残す (Issue は MTP の head も「別の重みとしてあるなら対象にする」と
しているので、その名前の形も選ぶ。実機の index にその名前が無ければ、何も選ばれない。
名前は未確認)。`eh_proj` を含めたいときは、`--pattern` にその分岐を足して渡す。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from k2_quant.safetensors_file import TensorInfo

DEFAULT_PATTERN: Final[str] = (
    r".*\.layers\.(?:0|1|2)\.mlp\.(?:gate|up|down)_proj$"
    r"|.*\.layers\.\d+\.mlp\.shared_experts\.(?:gate|up|down)_proj$"
    r"|.*\.layers\.\d+\.shared_head\.head$"
    r"|(?:.*\.)?lm_head$"
)

WEIGHT_SUFFIX: Final = ".weight"
SUPPORTED_DTYPES: Final = frozenset({"BF16", "F16", "F32"})


class SelectionError(ValueError):
    """対象の選択または検証に失敗した。"""


def module_name(tensor_name: str) -> str | None:
    """`.weight` で終わればその前を返す。そうでなければ None。"""
    if tensor_name.endswith(WEIGHT_SUFFIX):
        return tensor_name[: -len(WEIGHT_SUFFIX)]
    return None


def select_modules(tensor_names: Iterable[str], pattern: str) -> tuple[str, ...]:
    """モジュール名に `re.match` で当てて、選ばれたモジュール名を名前順に返す。"""
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise SelectionError(f"invalid pattern: {pattern}") from exc
    modules: set[str] = set()
    for tensor_name in tensor_names:
        module = module_name(tensor_name)
        if module is not None and compiled.match(module):
            modules.add(module)
    return tuple(sorted(modules))


def validate_targets(modules: Sequence[str], tensors: Mapping[str, TensorInfo]) -> None:
    """対象が変換できる形かを確かめる。1 つでも崩れていれば `SelectionError`。"""
    if not modules:
        raise SelectionError("no target modules selected")
    for module in modules:
        prefix = f"{module}."
        weight_name = f"{module}{WEIGHT_SUFFIX}"
        for name in tensors:
            if name.startswith(prefix) and name != weight_name:
                raise SelectionError(f"{module}: unexpected parameter {name}")
        weight = tensors.get(weight_name)
        if weight is None:
            raise SelectionError(f"{module}: weight not found")
        if weight.dtype not in SUPPORTED_DTYPES:
            raise SelectionError(f"{module}: unsupported dtype {weight.dtype}")
        if len(weight.shape) != 2:
            raise SelectionError(f"{module}: weight is not 2-dimensional: {weight.shape}")
