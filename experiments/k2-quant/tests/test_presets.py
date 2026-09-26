"""選び方の設定 (`--preset`) の試験 (第 2a 段。Issue #74 の C1 / C2 / C4)。

第 2a 段の対象 (MLA の射影・KDA のまとめていない射影・`lm_head`) は、`config.json` の
`text_config.layer_types` (層ごとの `linear_attention` = KDA、`deepseek_sparse_attention` = MLA)
から層番号を決めて選ぶ。層番号を手で並べない。確かめること:

- 層番号は `layer_types` から来る (並びを入れ替えると、選ばれる層が入れ替わる)
- 実機と同じ 45 層 (KDA 34 層、MLA 11 層) でも、層ごとの種類どおりに選び、二桁の層番号も取り違えない
- `layer_types` が無い、最上位にしか無い、知らない値を含む config では、推測せずに断る
- 第 1 段の設定 (`k2s1`) は、config を読まずに第 1 段の既定の正規表現になる
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

import synthetic
from k2_quant import presets, selection

LANGUAGE_MODEL = "model.language_model"


def _config(layer_types: Sequence[str]) -> dict[str, Any]:
    return {"text_config": {"layer_types": list(layer_types)}}


def _weights(*modules: str) -> list[str]:
    return [f"{module}.weight" for module in modules]


def test_stage2a_pattern_takes_layer_numbers_from_layer_types() -> None:
    """MLA の射影は MLA の層に、KDA の射影は KDA の層にだけ当たる。層番号は `layer_types` の
    並びで決まる。

    層 0 が MLA、層 1 が KDA の config では、`q_a_proj` は層 0 だけ、`g_b_proj` は層 1 だけが
    選ばれ、同じ字面が反対の層種にある名前は選ばれない。
    """
    config = _config([synthetic.MLA_LAYER_TYPE, synthetic.KDA_LAYER_TYPE])
    names = _weights(
        f"{LANGUAGE_MODEL}.layers.0.self_attn.q_a_proj",
        f"{LANGUAGE_MODEL}.layers.1.self_attn.q_a_proj",
        f"{LANGUAGE_MODEL}.layers.0.self_attn.g_b_proj",
        f"{LANGUAGE_MODEL}.layers.1.self_attn.g_b_proj",
    )

    selected = selection.select_modules(names, presets.stage2a_pattern(config))

    assert selected == (
        f"{LANGUAGE_MODEL}.layers.0.self_attn.q_a_proj",
        f"{LANGUAGE_MODEL}.layers.1.self_attn.g_b_proj",
    )


def test_stage2a_pattern_selects_each_layer_by_its_type_across_45_layers() -> None:
    """実機と同じ 45 層 (KDA 34 層、MLA 11 層) の並びで、層ごとの種類どおりに選び、それ以外は
    選ばない。

    MLA の層は `q_a_proj`・`kv_a_proj_with_mqa`・`q_b_proj`・`o_proj` の 4 つ、KDA の層は
    `o_proj`・`f_b_proj`・`g_b_proj` の 3 つだけが選ばれる。同じ字面が反対の層種に
    ある名前 (MLA の層の `g_b_proj`、KDA の層の `q_a_proj`)、KDA のまとめた層、`kv_b_proj`、
    indexer、並びに載らない層 45 の `o_proj` は選ばれない。二桁の層番号 (1 と 11、3 と 13 など)
    も取り違えない。
    """
    layer_types = [
        synthetic.MLA_LAYER_TYPE if layer % 4 == 3 else synthetic.KDA_LAYER_TYPE
        for layer in range(45)
    ]
    assert layer_types.count(synthetic.MLA_LAYER_TYPE) == 11
    assert layer_types.count(synthetic.KDA_LAYER_TYPE) == 34
    mla_selected = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj")
    mla_not_selected = ("kv_b_proj", "indexer.wq_b", "g_b_proj", "f_b_proj")
    kda_selected = ("o_proj", "f_b_proj", "g_b_proj")
    kda_not_selected = (
        "q_proj",
        "k_proj",
        "v_proj",
        "b_proj",
        "f_a_proj",
        "g_a_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
    )
    expected: set[str] = set()
    candidates: list[str] = []
    for layer, layer_type in enumerate(layer_types):
        is_mla = layer_type == synthetic.MLA_LAYER_TYPE
        selected_names = mla_selected if is_mla else kda_selected
        not_selected_names = mla_not_selected if is_mla else kda_not_selected
        for name in selected_names:
            expected.add(f"{LANGUAGE_MODEL}.layers.{layer}.self_attn.{name}")
        for name in (*selected_names, *not_selected_names):
            candidates.append(f"{LANGUAGE_MODEL}.layers.{layer}.self_attn.{name}")
    candidates.append(f"{LANGUAGE_MODEL}.layers.45.self_attn.o_proj")
    assert len(expected) == 11 * 4 + 34 * 3

    pattern = presets.stage2a_pattern(_config(layer_types))

    selected = selection.select_modules(_weights(*candidates), pattern)

    assert set(selected) == expected


def test_stage2a_pattern_selects_the_kda_f_b_proj_by_its_tensor_name() -> None:
    """KDA の層の `f_b_proj` は、実機のテンソル名の形 (`self_attn.f_b_proj`。`forget_gate.` が
    付かない) で選ばれる (#76)。

    `text_config.layer_types` の先頭 3 層が KDA、4 層目が MLA の config で、層 0 の
    `self_attn.f_b_proj.weight` が選ばれる。
    """
    config = _config([synthetic.KDA_LAYER_TYPE] * 3 + [synthetic.MLA_LAYER_TYPE])
    module = f"{LANGUAGE_MODEL}.layers.0.self_attn.f_b_proj"

    selected = selection.select_modules(_weights(module), presets.stage2a_pattern(config))

    assert selected == (module,)


def test_stage2a_pattern_does_not_select_the_old_form_the_mla_layer_or_merged_f_a_proj() -> None:
    """同じ字面の名前でも、次の 3 つは選ばれない (#76)。

    - 旧形式のテンソル名 (KDA の層 0 の `self_attn.forget_gate.f_b_proj`)。テンソル名に
      `forget_gate.` は付かないので、旧形式を残して選ぶことはしない
    - MLA の層 3 の `self_attn.f_b_proj`。`f_b_proj` は KDA の層の射影
    - KDA の層 0 の `self_attn.f_a_proj`。q・k・v・b・f_a・g_a のまとめた層で、選ばない
    """
    config = _config([synthetic.KDA_LAYER_TYPE] * 3 + [synthetic.MLA_LAYER_TYPE])
    names = _weights(
        f"{LANGUAGE_MODEL}.layers.0.self_attn.forget_gate.f_b_proj",
        f"{LANGUAGE_MODEL}.layers.3.self_attn.f_b_proj",
        f"{LANGUAGE_MODEL}.layers.0.self_attn.f_a_proj",
    )

    selected = selection.select_modules(names, presets.stage2a_pattern(config))

    assert selected == ()


def test_stage2a_pattern_also_selects_the_stage_1_range_and_lm_head() -> None:
    """第 2a 段の選び方は、第 1 段の既定の対象 (dense と共有の専門家) と `lm_head` も選ぶ。

    専門家 (`weight_packed` だけを持つ) と、第 1 段の範囲に入らない `mlp` (層 3 の dense の形) は
    選ばれない。
    """
    config = _config([synthetic.KDA_LAYER_TYPE] * 3 + [synthetic.MLA_LAYER_TYPE])
    stage_1 = (
        f"{LANGUAGE_MODEL}.layers.0.mlp.gate_proj",
        f"{LANGUAGE_MODEL}.layers.3.mlp.shared_experts.down_proj",
        f"{LANGUAGE_MODEL}.layers.45.mlp.shared_experts.up_proj",
    )
    names = [
        *_weights(*stage_1, "lm_head", f"{LANGUAGE_MODEL}.layers.3.mlp.gate_proj"),
        f"{LANGUAGE_MODEL}.layers.3.mlp.experts.0.gate_proj.weight_packed",
    ]

    selected = selection.select_modules(names, presets.stage2a_pattern(config))

    assert set(selected) == {*stage_1, "lm_head"}


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({}, id="no-text-config"),
        pytest.param({"text_config": {}}, id="no-layer-types"),
        pytest.param(
            {"layer_types": [synthetic.KDA_LAYER_TYPE, synthetic.MLA_LAYER_TYPE]},
            id="layer-types-only-at-top-level",
        ),
        pytest.param(_config([synthetic.KDA_LAYER_TYPE, "unknown"]), id="unknown-layer-type"),
    ],
)
def test_stage2a_pattern_refuses_a_config_without_usable_layer_types(
    config: dict[str, Any],
) -> None:
    """`text_config.layer_types` が無い (最上位にしか無いものは無いのと同じ)、または知らない値を含む
    config では、層を推測して分けず、選び方を作らずに断る。
    """
    with pytest.raises(selection.SelectionError):
        presets.stage2a_pattern(config)


def test_the_stage_1_preset_is_the_default_pattern_and_does_not_read_the_config() -> None:
    """`k2s1` は第 1 段の既定の正規表現になる。config を読まないので、`layer_types` が無い config
    でも断らない。
    """
    assert presets.resolve_preset_pattern("k2s1", {}) == selection.DEFAULT_PATTERN
