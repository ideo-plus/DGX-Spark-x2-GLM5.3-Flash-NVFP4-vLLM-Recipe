"""対象の選択 (名前の規則と正規表現)。

正規表現は、入力 checkpoint のテンソル名 (`model.language_model.layers.N....`) から
`.weight` を除いたモジュール名に `re.match` で当てて、変換するモジュールを選ぶ。vLLM が
実行時の層名に当てる target ではない。`config.json` に足す target は、選ばれた
モジュールから `quant_config` が作る (この正規表現は書かない)。

既定は Issue #56 の第 1 段の範囲のうち、dense の MLP (層 0〜2) と共有の専門家。第 1 段の
範囲に挙がっている次の 2 種は、vLLM が FP8 の重みを読めないので、既定から外す。

- MTP の `eh_proj`: vLLM (commit 0961bbae、`vllm/models/glm5next/common/mtp.py:49`) の
  `eh_proj` は `self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size,
  bias=False)` で、`quant_config` を受けない plain `nn.Linear` なので、FP8 の `eh_proj` を
  読み込めない。
- `lm_head` と MTP の head (`shared_head.head`): vLLM 0961bbae は、FP8 (W8A16、
  compressed-tensors) の `ParallelLMHead` を humming の線形カーネルで読めず、
  `AttributeError: 'ParallelLMHead' object has no attribute 'output_partition_sizes'` で
  起動できなかった (2026-09-26 の実機。#68)。`lm_head` は `ParallelLMHead` (`model.py:947-951`。
  `docs/research/2026-09-26-k2-quant-survey.md` §1 の表)、`shared_head.head` は `SharedHead`
  (`vllm/model_executor/models/deepseek_mtp.py`) の中の同じ `ParallelLMHead`。同調査は、ソースから
  `lm_head` を「量子化できる」と読んでいたが、実機では読めなかった。例外を出したカーネルの
  ファイルと行は、この repo の記録に無く、未確認。

これらを含めたいとき (実験目的) は、`--pattern` にその分岐を足して渡す (README の「`--pattern` の
書き方」)。変換自体はできるが、vLLM 0961bbae では読み込めない。

第 2a 段 (Issue #74) の対象は、`config.json` の層種の並びから決める。`k2_quant.presets` が
`--preset k2s2a` の正規表現を作り、この module の `select_modules` に渡す。

vLLM で 1 つの線形層にまとまる組 (`FUSED_GROUPS`) は、`validate_fused_groups` が、選ばれたモジュール
のうち一部だけが選ばれた組を、変換の前に断る。FP8 の重みと BF16 の重みが、1 つの線形層に混ざるため。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from k2_quant.safetensors_file import TensorInfo

DEFAULT_PATTERN: Final[str] = (
    r".*\.layers\.(?:0|1|2)\.mlp\.(?:gate|up|down)_proj$"
    r"|.*\.layers\.\d+\.mlp\.shared_experts\.(?:gate|up|down)_proj$"
)

WEIGHT_SUFFIX: Final = ".weight"
SUPPORTED_DTYPES: Final = frozenset({"BF16", "F16", "F32"})

FUSED_GROUPS: Final[tuple[tuple[str, ...], ...]] = (
    # MLA: vLLM の `fused_qkv_a_proj`
    ("q_a_proj", "kv_a_proj_with_mqa"),
    # MLP (dense と共有の専門家): vLLM の `gate_up_proj`
    ("gate_proj", "up_proj"),
    # KDA: vLLM の `in_proj_qkvbfg_a`
    ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"),
)
"""vLLM で 1 つの線形層にまとまる組。組の名前は、同じ親 (`...self_attn`、`...mlp` など) の下の
モジュール名の末尾。組の定義はここだけに置く。"""


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


def validate_fused_groups(modules: Sequence[str], tensors: Mapping[str, TensorInfo]) -> None:
    """まとめた層の組が、checkpoint に実在する分について、全員選ばれているかを確かめる。

    組のうち一部だけが選ばれていれば `SelectionError` (選ばれていない相手の名前を全部示す)。
    相手の `.weight` が checkpoint に無ければ、混ざることがないので通す。組は同じ親の下だけで
    数える。
    """
    selected = set(modules)
    for module in modules:
        for group in FUSED_GROUPS:
            for member in group:
                suffix = f".{member}"
                if not module.endswith(suffix):
                    continue
                parent = module[: -len(suffix)]
                present = {
                    f"{parent}.{name}"
                    for name in group
                    if f"{parent}.{name}{WEIGHT_SUFFIX}" in tensors
                }
                missing = sorted(present - selected)
                if missing:
                    raise SelectionError(
                        f"{parent}: fused group {'+'.join(group)} is partially selected; "
                        f"missing {missing}"
                    )
