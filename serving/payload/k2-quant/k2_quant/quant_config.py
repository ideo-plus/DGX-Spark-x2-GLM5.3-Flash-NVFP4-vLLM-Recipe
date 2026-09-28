"""`quantization_config` に FP8 のチャネルごと group、または NVFP4A16 の group を足す。

入力の `config.json` は変えず、新しい dict を返す。既存の group と、それ以外の
`ignore` はそのまま残す。

新しい group の `targets` は、利用者の正規表現ではなく、変換したモジュールの集合から作る。
選択の正規表現は checkpoint のテンソル名に当てるもので、vLLM が実行時の層名に当てる
target とは名前の形が違う (`--pattern '.*gate_proj$'` は、実行時には専門家にも当たる)。
変換していない実行時のモジュールに target が当たると、その重みを FP8 として読もうとして
起動が壊れる。

`add_fp8_channel_group` と `add_nvfp4a16_group` (Issue #95) は、この検証・`targets` の
組み立て・`ignore` の更新を `_add_weight_only_group` として共有し、group の `format` と
`weights` だけが違う。

対象が既存 group の target に当たった場合の扱いは、その対象が FP8 入力 (Issue #99。MTP の層の
FP8 の専門家) かどうかで分かれる。FP8 入力でない対象が当たれば、従来どおり変換の前に
`ConfigError` で止める (base の契約のまま。`add_fp8_channel_group` は、この経路しか使わない)。
FP8 入力だけが当たった group (`group_1` など) は、その group の target に当たる候補
(checkpoint で `.weight` を持つモジュールのうち、その target に当たるもの) が全部変換された
ときだけ取り除いて新しい group に置き換える。一部だけなら、従来どおり `ConfigError`。
`fp8_block_structure` は、その専門家の元の量子化設定 (`strategy=block` など) を group から読む。
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Final

FLOAT_QUANTIZED: Final = "float-quantized"
FLOAT_BLOCK_STRATEGY: Final = "block"
"""第 4 段 (Issue #99) の FP8 の専門家の `weights.strategy` (block ごとの静的スケール)。"""

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


def _matching_target(group: object, name: str) -> str | None:
    """`group` の `targets` のうち `name` に当たるものを 1 つ返す (無ければ `None`)。

    `group` が dict でない、または `targets` が無い場合も `None`。
    """
    if not isinstance(group, dict):
        return None
    targets = [str(target) for target in group.get("targets", [])]
    for target in targets:
        if matches_target(name, target):
            return target
    return None


def fp8_block_structure(config: Mapping[str, Any], module: str) -> tuple[int, int]:
    """`module` を覆う既存 group の `weights` から、block 量子化の大きさ `(高さ, 幅)` を読む
    (Issue #99)。

    `module` に target が当たる group がちょうど 1 つで、`format == "float-quantized"`、
    `weights.type == "float"`、`weights.num_bits == 8`、`weights.strategy == "block"`、
    `weights.symmetric is True`、`weights.block_structure` が正の整数 2 つでなければ
    `ConfigError` (実機の値を憶測せず、知っている形と違えば明示して止める)。
    """
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise ConfigError("quantization_config is missing")
    config_groups = quantization_config.get("config_groups")
    if not isinstance(config_groups, dict):
        raise ConfigError("config_groups is missing")
    matched: list[tuple[str, dict[str, Any]]] = []
    for name, group in config_groups.items():
        if isinstance(group, dict) and _matching_target(group, module) is not None:
            matched.append((name, group))
    if len(matched) != 1:
        raise ConfigError(f"{module}: matched by {len(matched)} config_groups (expected exactly 1)")
    name, group = matched[0]
    if group.get("format") != FLOAT_QUANTIZED:
        raise ConfigError(f"{module}: group {name} format must be {FLOAT_QUANTIZED!r}")
    weights = group.get("weights")
    if not isinstance(weights, dict):
        raise ConfigError(f"{module}: group {name} weights is missing")
    if weights.get("type") != "float":
        raise ConfigError(f"{module}: group {name} weights.type must be 'float'")
    if weights.get("num_bits") != 8:
        raise ConfigError(f"{module}: group {name} weights.num_bits must be 8")
    if weights.get("strategy") != FLOAT_BLOCK_STRATEGY:
        raise ConfigError(
            f"{module}: group {name} weights.strategy must be {FLOAT_BLOCK_STRATEGY!r}"
        )
    if weights.get("symmetric") is not True:
        raise ConfigError(f"{module}: group {name} weights.symmetric must be true")
    block_structure = weights.get("block_structure")
    if (
        not isinstance(block_structure, list)
        or len(block_structure) != 2
        or not all(isinstance(value, int) and value > 0 for value in block_structure)
    ):
        raise ConfigError(
            f"{module}: group {name} weights.block_structure must be two positive integers"
        )
    return (int(block_structure[0]), int(block_structure[1]))


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


NVFP4_PACK_QUANTIZED: Final = "nvfp4-pack-quantized"

NVFP4_TENSOR_GROUP_WEIGHTS: Final[Mapping[str, object]] = {
    "actorder": None,
    "block_structure": None,
    "dynamic": False,
    "group_size": 16,
    "num_bits": 4,
    "observer": "memoryless_minmax",
    "observer_kwargs": {},
    "scale_dtype": "torch.float8_e4m3fn",
    "strategy": "tensor_group",
    "symmetric": True,
    "type": "float",
    "zp_dtype": None,
}
"""元の重みの専門家 (NVFP4) の group の `weights`。2026-09-27 に HF の rev `18d55bfd…` の
`config.json` の `group_0.weights` から逐語で確認した値。"""


def _add_weight_only_group(
    config: Mapping[str, Any],
    *,
    modules: Sequence[str],
    group_format: str,
    weights: Mapping[str, object],
    fp8_input_modules: Collection[str],
    candidates: Collection[str],
) -> dict[str, Any]:
    """重みだけの量子化の group を 1 つ足す (`input_activations`/`output_activations` は null)。

    `targets` は、`modules` (変換したモジュール) の実行時の名前にだけ当たる 1 つの target。

    `modules` のいずれかが既存 group の target に当たったとき、その対象が `fp8_input_modules`
    (Issue #99。FP8 の専門家を入力にしたモジュール) に無ければ、従来どおり即座に `ConfigError`。
    `fp8_input_modules` に有る対象だけが当たった group は、その group の target に当たる
    `candidates` (checkpoint で `.weight` を持つ全モジュール名) が全部 `modules` (変換対象) に
    含まれるときだけ、その group を取り除いて置き換える。一部だけなら、変換していない候補の
    名前を挙げて `ConfigError` (書き込みはしない)。新しい group の名前は、取り除く前の
    `next_group_name` (空いた番号を再利用しない)。
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

    module_set = set(modules)
    fp8_input_set = set(fp8_input_modules)
    to_remove: list[str] = []
    for group_name, group in config_groups.items():
        matched = [module for module in modules if _matching_target(group, module) is not None]
        if not matched:
            continue
        non_fp8_matched = [module for module in matched if module not in fp8_input_set]
        if non_fp8_matched:
            hit_target = _matching_target(group, non_fp8_matched[0])
            raise ConfigError(
                f"{non_fp8_matched[0]}: already matched by existing target {hit_target}"
            )
        group_candidates = {
            candidate for candidate in candidates if _matching_target(group, candidate) is not None
        }
        missing = sorted(group_candidates - module_set)
        if missing:
            raise ConfigError(
                f"{group_name}: not all modules matching its targets are converted; "
                f"missing {missing}"
            )
        to_remove.append(group_name)

    name = next_group_name(config_groups)
    for remove_name in to_remove:
        del config_groups[remove_name]
    config_groups[name] = {
        "format": group_format,
        "input_activations": None,
        "output_activations": None,
        "targets": [_target_for(modules)],
        "weights": dict(weights),
    }
    ignore = quantization_config.get("ignore")
    if isinstance(ignore, list):
        converted = set(modules)
        quantization_config["ignore"] = [item for item in ignore if item not in converted]
    return result


def add_fp8_channel_group(config: Mapping[str, Any], *, modules: Sequence[str]) -> dict[str, Any]:
    """FP8 (`strategy=channel`, `symmetric`, `dynamic=false`) の group を 1 つ足す。

    `targets` は、`modules` (変換したモジュール) の実行時の名前にだけ当たる 1 つの target。
    FP8 入力 (Issue #99) には関わらないため、`fp8_input_modules`・`candidates` は空にする。
    """
    return _add_weight_only_group(
        config,
        modules=modules,
        group_format=FLOAT_QUANTIZED,
        weights=FP8_CHANNEL_WEIGHTS,
        fp8_input_modules=(),
        candidates=(),
    )


def add_nvfp4a16_group(
    config: Mapping[str, Any],
    *,
    modules: Sequence[str],
    fp8_input_modules: Collection[str],
    candidates: Collection[str],
) -> dict[str, Any]:
    """NVFP4A16 (重みだけ NVFP4。`strategy=tensor_group`) の group を 1 つ足す (Issue #95)。

    `format`・`weights` は、元の重みの専門家の NVFP4 の group (`group_0`) と同じ形。
    `input_activations`/`output_activations` は null (重みだけの量子化のため、`group_0` の
    W4A4 とは異なる)。`fp8_input_modules`・`candidates` は Issue #99 (第 4 段。FP8 の専門家を
    入力にしたモジュールと、判定に使う checkpoint の全候補) で、`_add_weight_only_group` の
    契約をそのまま渡す。
    """
    return _add_weight_only_group(
        config,
        modules=modules,
        group_format=NVFP4_PACK_QUANTIZED,
        weights=NVFP4_TENSOR_GROUP_WEIGHTS,
        fp8_input_modules=fp8_input_modules,
        candidates=candidates,
    )
