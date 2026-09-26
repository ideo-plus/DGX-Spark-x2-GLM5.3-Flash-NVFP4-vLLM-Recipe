"""選び方の設定 (`--preset`)。

`k2s1` は第 1 段の既定 (`selection.DEFAULT_PATTERN`)。`k2s2a` は、第 1 段の対象に、MLA の射影・
KDA のまとめていない射影・`lm_head` を足したもの (Issue #74、ADR 0007 の第 2a 段)。

どの層が MLA でどの層が KDA かは、入力 `config.json` の `text_config.layer_types` (層ごとの種類の
並び) から決める。層番号は手で並べない。この並びは、本体の層 (0 から) だけを持ち、MTP の層 (層 45)
は載らないので、MTP の attention は選ばれない。並びが無い、または知らない種類を含むときは、層を
推測して分けず、`SelectionError` で断る。

まとめた層 (KDA の q・k・v・b・f_a・g_a、MLA の `kv_b_proj` と indexer) は選ばない。vLLM で 1 つに
まとまる組が崩れる選び方は、`selection.validate_fused_groups` が変換の前に断る。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from k2_quant.selection import DEFAULT_PATTERN, SelectionError

DEFAULT_PRESET: Final = "k2s1"
STAGE2A_PRESET: Final = "k2s2a"
PRESET_NAMES: Final[tuple[str, ...]] = (DEFAULT_PRESET, STAGE2A_PRESET)

TEXT_CONFIG_KEY: Final = "text_config"
LAYER_TYPES_KEY: Final = "layer_types"
MLA_LAYER_TYPE: Final = "deepseek_sparse_attention"
KDA_LAYER_TYPE: Final = "linear_attention"

MLA_PROJECTIONS: Final[tuple[str, ...]] = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj")
KDA_UNMERGED_PROJECTIONS: Final[tuple[str, ...]] = ("o_proj", "forget_gate.f_b_proj", "g_b_proj")
LM_HEAD_PATTERN: Final = r"(?:.*\.)?lm_head$"


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


def stage2a_pattern(config: Mapping[str, Any]) -> str:
    """第 2a 段の選び方の正規表現を、`config.json` の層種の並びから作る。

    第 1 段の既定の対象、MLA の層の `MLA_PROJECTIONS`、KDA の層の `KDA_UNMERGED_PROJECTIONS`、
    `lm_head` の分岐を `|` でつなぐ。
    """
    types = layer_types(config)
    branches = [DEFAULT_PATTERN]
    for layer_type, projections in (
        (MLA_LAYER_TYPE, MLA_PROJECTIONS),
        (KDA_LAYER_TYPE, KDA_UNMERGED_PROJECTIONS),
    ):
        layers = [index for index, current in enumerate(types) if current == layer_type]
        if layers:
            branches.append(_attention_pattern(layers, projections))
    branches.append(LM_HEAD_PATTERN)
    return "|".join(branches)


def resolve_preset_pattern(name: str, config: Mapping[str, Any]) -> str:
    """preset の名前から、モジュール名に当てる正規表現を決める。`k2s1` は config を読まない。"""
    if name == DEFAULT_PRESET:
        return DEFAULT_PATTERN
    if name == STAGE2A_PRESET:
        return stage2a_pattern(config)
    raise SelectionError(f"unknown preset: {name}")
