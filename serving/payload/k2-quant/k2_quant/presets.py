"""選び方の設定 (`--preset`)。

`k2s1` は第 1 段の既定 (`selection.DEFAULT_PATTERN`)。`k2s2a` は、第 1 段の対象に、MLA の射影・
KDA のまとめていない射影・`lm_head` を足したもの (Issue #74、ADR 0007 の第 2a 段)。`k2s2b` は、
`k2s2a` の対象に、KDA のまとめた層 (vLLM の `in_proj_qkvbfg_a`) の 6 射影を足したもの (Issue #79、
ADR 0007 の第 2b 段)。読むには、vLLM の重ね合わせ `k2s2b` が前提。`k2s3` (Issue #95) は、対象は
`k2s2b` と同じで、形式だけを FP8 から NVFP4A16 (重みだけ NVFP4) にする。`k2s4` (Issue #99) は、
`k2s3` の対象に、MTP の層 (層 45) の MLA の射影 4 つと FP8 の専門家を NVFP4A16 で足す。

どの層が MLA でどの層が KDA かは、入力 `config.json` の `text_config.layer_types` (層ごとの種類の
並び) から決める。層番号は手で並べない。この並びは、本体の層 (0 から) だけを持ち、MTP の層 (層 45)
は載らないので、MTP の attention は `k2s1`〜`k2s3` では選ばれない。並びが無い、または知らない種類を
含むときは、層を推測して分けず、`SelectionError` で断る。MTP の層番号は、この並びの要素数
(`mtp_layer`) から決める。

KDA のまとめた層 (q・k・v・b・f_a・g_a) は、`k2s2a` では選ばず、`k2s2b` で組を丸ごと選ぶ。MLA の
`kv_b_proj` と indexer は、どの段でも選ばない (MTP の層も同じ)。vLLM で 1 つにまとまる組が崩れる
選び方は、`selection.validate_fused_groups` が変換の前に断る。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from k2_quant.selection import DEFAULT_PATTERN, KDA_FUSED_GROUP, SelectionError

DEFAULT_PRESET: Final = "k2s1"
STAGE2A_PRESET: Final = "k2s2a"
STAGE2B_PRESET: Final = "k2s2b"
STAGE3_PRESET: Final = "k2s3"
STAGE4_PRESET: Final = "k2s4"
PRESET_NAMES: Final[tuple[str, ...]] = (
    DEFAULT_PRESET,
    STAGE2A_PRESET,
    STAGE2B_PRESET,
    STAGE3_PRESET,
    STAGE4_PRESET,
)

FORMAT_FP8: Final = "fp8"
FORMAT_NVFP4A16: Final = "nvfp4a16"
FORMAT_NAMES: Final[tuple[str, ...]] = (FORMAT_FP8, FORMAT_NVFP4A16)
DEFAULT_FORMAT: Final = FORMAT_FP8

TEXT_CONFIG_KEY: Final = "text_config"
LAYER_TYPES_KEY: Final = "layer_types"
MLA_LAYER_TYPE: Final = "deepseek_sparse_attention"
KDA_LAYER_TYPE: Final = "linear_attention"

MLA_PROJECTIONS: Final[tuple[str, ...]] = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj")
KDA_UNMERGED_PROJECTIONS: Final[tuple[str, ...]] = ("o_proj", "f_b_proj", "g_b_proj")
LM_HEAD_PATTERN: Final = r"(?:.*\.)?lm_head$"
EXPERT_PROJECTIONS: Final[tuple[str, ...]] = ("gate_proj", "up_proj", "down_proj")
"""専門家の 3 射影 (`mlp.experts.N.{gate,up,down}_proj`)。第 4 段 (Issue #99) が、MTP の層の FP8
の専門家をこの 3 射影で選ぶ。"""


def layer_types(config: Mapping[str, Any]) -> tuple[str, ...]:
    """`config.json` の `text_config.layer_types` を、層番号の順に返す。

    無い、空、リストでない、知らない種類を含むときは `SelectionError`。最上位の `layer_types` は
    見ない。
    """
    text_config = config.get(TEXT_CONFIG_KEY)
    if not isinstance(text_config, dict):
        raise SelectionError(f"config.json: {TEXT_CONFIG_KEY} is missing")
    types = text_config.get(LAYER_TYPES_KEY)
    if not isinstance(types, list) or not types:
        raise SelectionError(f"config.json: {TEXT_CONFIG_KEY}.{LAYER_TYPES_KEY} is missing")
    for layer_type in types:
        if layer_type not in (MLA_LAYER_TYPE, KDA_LAYER_TYPE):
            raise SelectionError(
                f"config.json: {TEXT_CONFIG_KEY}.{LAYER_TYPES_KEY} has an unknown layer type: "
                f"{layer_type!r}"
            )
    return tuple(types)


def _attention_pattern(layers: Sequence[int], projections: Sequence[str]) -> str:
    """指定した層の `self_attn` の下の、指定した射影だけに当たる正規表現の分岐。"""
    layer_alternatives = "|".join(str(layer) for layer in layers)
    projection_alternatives = "|".join(re.escape(projection) for projection in projections)
    return rf".*\.layers\.(?:{layer_alternatives})\.self_attn\.(?:{projection_alternatives})$"


def _stage2_pattern(config: Mapping[str, Any], kda_projections: Sequence[str]) -> str:
    """第 2 段の選び方の正規表現を、`config.json` の層種の並びから作る。

    第 1 段の既定の対象、MLA の層の `MLA_PROJECTIONS`、KDA の層の `kda_projections`、`lm_head` の
    分岐を `|` でつなぐ。
    """
    types = layer_types(config)
    branches = [DEFAULT_PATTERN]
    for layer_type, projections in (
        (MLA_LAYER_TYPE, MLA_PROJECTIONS),
        (KDA_LAYER_TYPE, kda_projections),
    ):
        layers = [index for index, current in enumerate(types) if current == layer_type]
        if layers:
            branches.append(_attention_pattern(layers, projections))
    branches.append(LM_HEAD_PATTERN)
    return "|".join(branches)


def stage2a_pattern(config: Mapping[str, Any]) -> str:
    """第 2a 段の選び方。KDA の層は `KDA_UNMERGED_PROJECTIONS` だけを選ぶ。"""
    return _stage2_pattern(config, KDA_UNMERGED_PROJECTIONS)


def stage2b_pattern(config: Mapping[str, Any]) -> str:
    """第 2b 段の選び方。KDA の層は、第 2a 段の射影に `selection.KDA_FUSED_GROUP` を足して選ぶ。"""
    return _stage2_pattern(config, (*KDA_UNMERGED_PROJECTIONS, *KDA_FUSED_GROUP))


def stage3_pattern(config: Mapping[str, Any]) -> str:
    """第 3 段の選び方。対象は第 2b 段と同じ (差は形式だけ。Issue #95)。"""
    return stage2b_pattern(config)


def mtp_layer(config: Mapping[str, Any]) -> int:
    """MTP の層番号。`layer_types` (層 0 からの並び) には載らないので、その次の番号
    (要素数) にする (Issue #99)。"""
    return len(layer_types(config))


def _expert_pattern(layers: Sequence[int]) -> str:
    """指定した層の専門家 (`mlp.experts.N.{gate,up,down}_proj`) だけに当たる正規表現の分岐。"""
    layer_alternatives = "|".join(str(layer) for layer in layers)
    projection_alternatives = "|".join(re.escape(projection) for projection in EXPERT_PROJECTIONS)
    return (
        rf".*\.layers\.(?:{layer_alternatives})\.mlp\.experts\.\d+\.(?:{projection_alternatives})$"
    )


def stage4_pattern(config: Mapping[str, Any]) -> str:
    """第 4 段の選び方。第 3 段の対象に、MTP の層の MLA の射影 4 つと FP8 の専門家を足す
    (Issue #99)。`kv_b_proj` と indexer は、本体の層と同じく対象外のまま。"""
    mtp = mtp_layer(config)
    return "|".join(
        [
            stage3_pattern(config),
            _attention_pattern((mtp,), MLA_PROJECTIONS),
            _expert_pattern((mtp,)),
        ]
    )


def resolve_preset_pattern(name: str, config: Mapping[str, Any]) -> str:
    """preset の名前から、モジュール名に当てる正規表現を決める。`k2s1` は config を読まない。"""
    if name == DEFAULT_PRESET:
        return DEFAULT_PATTERN
    if name == STAGE2A_PRESET:
        return stage2a_pattern(config)
    if name == STAGE2B_PRESET:
        return stage2b_pattern(config)
    if name == STAGE3_PRESET:
        return stage3_pattern(config)
    if name == STAGE4_PRESET:
        return stage4_pattern(config)
    raise SelectionError(f"unknown preset: {name}")


def resolve_preset_format(name: str) -> str:
    """preset の名前から、数値形式を決める。`k2s3`・`k2s4` は NVFP4A16、それ以外は FP8。"""
    if name in (STAGE3_PRESET, STAGE4_PRESET):
        return FORMAT_NVFP4A16
    if name in (DEFAULT_PRESET, STAGE2A_PRESET, STAGE2B_PRESET):
        return FORMAT_FP8
    raise SelectionError(f"unknown preset: {name}")


def resolve_selection(
    name: str, config: Mapping[str, Any], *, pattern: str | None, weight_format: str | None
) -> tuple[str, str]:
    """正規表現と数値形式を、優先順位の規則ごと 1 か所で決める。

    正規表現は `pattern` があればそれを、無ければ `preset` の名前から解決する。数値形式は、
    `weight_format` が指定されていればそれを、無ければ `pattern` を明示したときは既定の
    `FORMAT_FP8`、`preset` から選ぶときは `resolve_preset_format` の結果を使う。数値形式の
    解決を呼び出し側 (CLI の層) に複製しない。
    """
    resolved_pattern = pattern if pattern is not None else resolve_preset_pattern(name, config)
    if weight_format is not None:
        resolved_format = weight_format
    elif pattern is not None:
        resolved_format = DEFAULT_FORMAT
    else:
        resolved_format = resolve_preset_format(name)
    return resolved_pattern, resolved_format
