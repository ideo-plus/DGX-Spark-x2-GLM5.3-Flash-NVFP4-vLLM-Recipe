"""`quantization_config` に FP8 のチャネルごと group を足す。

入力の `config.json` は変えず、新しい dict を返す。既存の group と、それ以外の
`ignore` はそのまま残す。対象が既存 group の target に当たる場合は、変換の前に
`ConfigError` で止める。

新しい group の `targets` は、利用者の正規表現ではなく、変換したモジュールの集合から作る。
選択の正規表現は checkpoint のテンソル名に当てるもので、vLLM が実行時の層名に当てる
target とは名前の形が違う (`--pattern '.*gate_proj$'` は、実行時には専門家にも当たる)。
変換していない実行時のモジュールに target が当たると、その重みを FP8 として読もうとして
起動が壊れる。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

FLOAT_QUANTIZED: Final = "float-quantized"

CHECKPOINT_PREFIX: Final = "model.language_model."
"""checkpoint の本体と MTP のテンソル名の接頭辞。実行時の名前は、本体が `language_model.model.`、
MTP が `model.`、`lm_head` が `language_model.lm_head` になり、接頭辞が違う。"""

FP8_CHANNEL_WEIGHTS: Final[Mapping[str, object]] = {
    "actorder": None,
    "block_structure": None,
    "dynamic": False,
    "group_size": None,
    "num_bits": 8,
    "observer": "minmax",
    "observer_kwargs": {},
    "scale_dtype": None,
    "strategy": "channel",
    "symmetric": True,
    "type": "float",
    "zp_dtype": None,
}


class ConfigError(ValueError):
    """`quantization_config` の前提が崩れている。"""


def next_group_name(config_groups: Mapping[str, object]) -> str:
    """`group_0`, `group_1`, … の未使用の最小を返す。"""
    index = 0
    while f"group_{index}" in config_groups:
        index += 1
    return f"group_{index}"


def matches_target(module: str, target: str) -> bool:
    """`re:` 付きなら `re.match`、それ以外は等値で当てる。"""
    if target.startswith("re:"):
        return re.match(target[3:], module) is not None
    return module == target


def _existing_targets(config_groups: Mapping[str, Any]) -> list[str]:
    targets: list[str] = []
    for group in config_groups.values():
        if isinstance(group, dict):
            for target in group.get("targets", []):
                targets.append(str(target))
    return targets


def _runtime_suffix(module: str) -> str:
    """checkpoint のモジュール名から、実行時の名前の末尾を取り出す。

    `model.language_model.` の下の名前は、その接頭辞を除く。`.` を含まない最上位の名前
    (`lm_head`) はそのまま。どちらでもない名前は、実行時の名前の形が分からず、target を
    変換した集合に限れないので、`ConfigError` で断る。
    """
    if module.startswith(CHECKPOINT_PREFIX):
        return module[len(CHECKPOINT_PREFIX) :]
    if "." not in module:
        return module
    raise ConfigError(
        f"{module}: runtime name is unknown "
        f"(expected {CHECKPOINT_PREFIX}<name> or a top-level name)"
    )


def _target_for(modules: Sequence[str]) -> str:
    """変換したモジュールの実行時の名前にだけ当たる、`re:` 付きの target を作る。

    vLLM は `re.match` で実行時の層名に当てる。接頭辞は `(?:.*\\.)?` で受け、末尾は
    モジュールの名前で固定する。名前の順に並べ、重複を除くので、同じ入力から同じ target になる。
    """
    suffixes = sorted({_runtime_suffix(module) for module in modules})
    alternatives = "|".join(re.escape(suffix) for suffix in suffixes)
    return f"re:(?:.*\\.)?(?:{alternatives})$"


def add_fp8_channel_group(config: Mapping[str, Any], *, modules: Sequence[str]) -> dict[str, Any]:
    """FP8 (`strategy=channel`, `symmetric`, `dynamic=false`) の group を 1 つ足す。

    `targets` は、`modules` (変換したモジュール) の実行時の名前にだけ当たる 1 つの target。
    """
    if "quantization_config" not in config:
        raise ConfigError("quantization_config is missing")
    result: dict[str, Any] = json.loads(json.dumps(config, ensure_ascii=False))
    quantization_config = result["quantization_config"]
    if not isinstance(quantization_config, dict):
        raise ConfigError("quantization_config is not a JSON object")
    if quantization_config.get("quant_method") != "compressed-tensors":
        raise ConfigError(
            f"quant_method must be compressed-tensors: {quantization_config.get('quant_method')}"
        )
    if quantization_config.get("format") != "mixed-precision":
        raise ConfigError(f"format must be mixed-precision: {quantization_config.get('format')}")
    config_groups = quantization_config.get("config_groups")
    if not isinstance(config_groups, dict):
        raise ConfigError("config_groups is missing")
    existing_targets = _existing_targets(config_groups)
    for module in modules:
        for target in existing_targets:
            if matches_target(module, target):
                raise ConfigError(f"{module}: already matched by existing target {target}")
    name = next_group_name(config_groups)
    config_groups[name] = {
        "format": FLOAT_QUANTIZED,
        "input_activations": None,
        "output_activations": None,
        "targets": [_target_for(modules)],
        "weights": dict(FP8_CHANNEL_WEIGHTS),
    }
    ignore = quantization_config.get("ignore")
    if isinstance(ignore, list):
        converted = set(modules)
        quantization_config["ignore"] = [item for item in ignore if item not in converted]
    return result
