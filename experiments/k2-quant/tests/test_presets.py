"""選び方の設定 (`--preset`) の試験 (第 2a 段。Issue #74 の C1 / C2 / C4。第 2b 段。Issue #79)。

第 2a 段の対象 (MLA の射影・KDA のまとめていない射影・`lm_head`) は、`config.json` の
`text_config.layer_types` (層ごとの `linear_attention` = KDA、`deepseek_sparse_attention` = MLA)
から層番号を決めて選ぶ。層番号を手で並べない。確かめること:

- 層番号は `layer_types` から来る (並びを入れ替えると、選ばれる層が入れ替わる)
- 実機と同じ 45 層 (KDA 34 層、MLA 11 層) でも、層ごとの種類どおりに選び、二桁の層番号も取り違えない
- `layer_types` が無い、最上位にしか無い、知らない値を含む config では、推測せずに断る
- 第 1 段の設定 (`k2s1`) は、config を読まずに第 1 段の既定の正規表現になる

第 2b 段 (`k2s2b`) は、第 2a 段の対象に、KDA の層のまとめた層 (q・k・v・b・f_a・g_a の 6 射影) を
足して選ぶ。確かめること:

- KDA の層の 6 射影は、組ごと選ばれる (`validate_fused_groups` が断らない)。MLA の層・並びに載らない
  層・末尾だけ同じ名前・旧形式の名前は選ばれない
- 45 層の並びで、層ごとの種類どおりに選ぶ。第 2a 段の選び方との差は、KDA の層の 6 射影だけ
- 第 2a 段の選び方の文字列 (manifest の `--pattern` に残る) は変わらない
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

import synthetic
from k2_quant import presets, selection
from k2_quant.safetensors_file import TensorInfo

LANGUAGE_MODEL = "model.language_model"

KDA_MERGED_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")
"""KDA の層で vLLM の 1 つの線形層 (`in_proj_qkvbfg_a`) にまとまる 6 射影 (第 2b 段で選ぶ)。"""


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


# --- 第 2b 段 (`k2s2b`)。KDA のまとめた層 (q・k・v・b・f_a・g_a) を組ごと足す (Issue #79) ---------


def _tensors(*modules: str) -> dict[str, TensorInfo]:
    """モジュールごとの `.weight` (BF16 の 2 次元) を持つ、checkpoint のヘッダの見本。"""
    return {
        f"{module}.weight": TensorInfo(
            name=f"{module}.weight", dtype="BF16", shape=(4, 4), data_offsets=(0, 0)
        )
        for module in modules
    }


def _layer_types_across_45_layers() -> list[str]:
    """実機と同じ 45 層 (KDA 34 層、MLA 11 層。`layer % 4 == 3` が MLA) の層種の並び。"""
    layer_types = [
        synthetic.MLA_LAYER_TYPE if layer % 4 == 3 else synthetic.KDA_LAYER_TYPE
        for layer in range(45)
    ]
    assert layer_types.count(synthetic.MLA_LAYER_TYPE) == 11
    assert layer_types.count(synthetic.KDA_LAYER_TYPE) == 34
    return layer_types


def _candidates_across_45_layers(layer_types: Sequence[str]) -> list[str]:
    """45 層の、選ばれる名前と選ばれない名前 (同じ字面が反対の層種にあるものを含む) の `.weight`。

    MLA の層は、4 射影 (`q_a_proj`・`kv_a_proj_with_mqa`・`q_b_proj`・`o_proj`) と、`kv_b_proj`・
    indexer・KDA の射影の名前 (`q_proj`〜`g_a_proj`・`f_b_proj`・`g_b_proj`) を持つ。KDA の層は、
    9 射影 (`o_proj`・`f_b_proj`・`g_b_proj` と、まとめた層の 6 つ) と、MLA の射影の名前
    (`q_a_proj`・`q_b_proj`・`kv_a_proj_with_mqa`) を持つ。並びに載らない層 45 は、KDA・MLA の
    どちらの射影とも同じ名前の `o_proj` と `q_proj` を持つ。
    """
    mla_names = (
        "q_a_proj",
        "kv_a_proj_with_mqa",
        "q_b_proj",
        "o_proj",
        "kv_b_proj",
        "indexer.wq_b",
        *KDA_MERGED_PROJECTIONS,
        "f_b_proj",
        "g_b_proj",
    )
    kda_names = (
        "o_proj",
        "f_b_proj",
        "g_b_proj",
        *KDA_MERGED_PROJECTIONS,
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
    )
    modules = [
        f"{LANGUAGE_MODEL}.layers.{layer}.self_attn.{name}"
        for layer, layer_type in enumerate(layer_types)
        for name in (mla_names if layer_type == synthetic.MLA_LAYER_TYPE else kda_names)
    ]
    modules.append(f"{LANGUAGE_MODEL}.layers.45.self_attn.o_proj")
    modules.append(f"{LANGUAGE_MODEL}.layers.45.self_attn.q_proj")
    return _weights(*modules)


def test_stage2b_pattern_selects_the_six_kda_merged_projections_as_a_whole_group() -> None:
    """`--preset k2s2b` の選び方は、KDA の層 0 の q・k・v・b・f_a・g_a の 6 射影を、組ごと選ぶ
    (#79)。

    実機のテンソル名の形 (`self_attn.q_proj` など。`f_a_proj` に `forget_gate.` が付かない
    (#76)) の 6 つがすべて選ばれ、組の一部だけが選ばれる状態 (`validate_fused_groups` が断る)
    にならない。
    """
    config = _config([synthetic.KDA_LAYER_TYPE] * 3 + [synthetic.MLA_LAYER_TYPE])
    modules = [f"{LANGUAGE_MODEL}.layers.0.self_attn.{name}" for name in KDA_MERGED_PROJECTIONS]
    pattern = presets.resolve_preset_pattern("k2s2b", config)

    selected = selection.select_modules(_weights(*modules), pattern)

    assert set(selected) == set(modules)
    selection.validate_fused_groups(selected, _tensors(*modules))


def test_stage2b_pattern_does_not_select_lookalikes_of_the_merged_projections() -> None:
    """同じ字面の名前でも、次は選ばれない (第 2b 段が足すのは、KDA の層の、`self_attn` の直下の
    6 射影だけ)。

    - MLA の層 3 の `self_attn.q_proj`・`f_a_proj` (まとめた層の名前は、KDA の層のもの)
    - MLA の層 3 の `kv_b_proj` と indexer (`wq_b`)
    - KDA の層 0 の `self_attn.q_b_proj` (`q_proj` と末尾が似るだけの別の名前)
    - `layer_types` に載らない層 45 の `self_attn.q_proj`
    - 旧形式のテンソル名 (KDA の層 0 の `self_attn.forget_gate.f_a_proj`。テンソル名に
      `forget_gate.` は付かない (#76))
    - 専門家の `weight_packed` (`.weight` ではないので候補にならない)
    """
    config = _config([synthetic.KDA_LAYER_TYPE] * 3 + [synthetic.MLA_LAYER_TYPE])
    names = [
        *_weights(
            f"{LANGUAGE_MODEL}.layers.3.self_attn.q_proj",
            f"{LANGUAGE_MODEL}.layers.3.self_attn.f_a_proj",
            f"{LANGUAGE_MODEL}.layers.3.self_attn.kv_b_proj",
            f"{LANGUAGE_MODEL}.layers.3.self_attn.indexer.wq_b",
            f"{LANGUAGE_MODEL}.layers.0.self_attn.q_b_proj",
            f"{LANGUAGE_MODEL}.layers.45.self_attn.q_proj",
            f"{LANGUAGE_MODEL}.layers.0.self_attn.forget_gate.f_a_proj",
        ),
        f"{LANGUAGE_MODEL}.layers.3.mlp.experts.0.gate_proj.weight_packed",
    ]

    selected = selection.select_modules(names, presets.stage2b_pattern(config))

    assert selected == ()


def test_stage2b_pattern_selects_each_layer_by_its_type_across_45_layers() -> None:
    """実機と同じ 45 層 (KDA 34 層、MLA 11 層) の並びで、層ごとの種類どおりに選ぶ。

    MLA の層は 4 射影、KDA の層は 9 射影 (`o_proj`・`f_b_proj`・`g_b_proj` と、まとめた層の 6 つ)
    が選ばれ、数は `11 * 4 + 34 * 9`。MLA の層に `q_proj` などのまとめた層の名前、KDA の層に
    `q_a_proj` などの MLA の射影の名前は無く (選ばれない)、`kv_b_proj`・indexer・並びに載らない
    層 45 も選ばれない。二桁の層番号 (1 と 11、3 と 13 など) も取り違えない。
    """
    layer_types = _layer_types_across_45_layers()
    mla_selected = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj")
    kda_selected = ("o_proj", "f_b_proj", "g_b_proj", *KDA_MERGED_PROJECTIONS)
    expected = {
        f"{LANGUAGE_MODEL}.layers.{layer}.self_attn.{name}"
        for layer, layer_type in enumerate(layer_types)
        for name in (mla_selected if layer_type == synthetic.MLA_LAYER_TYPE else kda_selected)
    }
    assert len(expected) == 11 * 4 + 34 * 9

    pattern = presets.stage2b_pattern(_config(layer_types))

    selected = selection.select_modules(_candidates_across_45_layers(layer_types), pattern)

    assert set(selected) == expected


def test_stage2b_pattern_adds_only_the_kda_merged_projections_to_the_stage2a_selection() -> None:
    """第 2b 段の選び方は、第 2a 段の選び方の対象を、そのまま含み、足すのは KDA の層の 6 射影だけ。

    45 層の候補に、両方の選び方を当てると、第 2a 段の選択が第 2b 段の選択に含まれ、その差は、
    KDA の 34 層それぞれの、まとめた層の 6 射影 (34 * 6 個) になる。
    """
    layer_types = _layer_types_across_45_layers()
    config = _config(layer_types)
    candidates = _candidates_across_45_layers(layer_types)

    stage_2a = set(selection.select_modules(candidates, presets.stage2a_pattern(config)))
    stage_2b = set(selection.select_modules(candidates, presets.stage2b_pattern(config)))

    kda_layers = [
        layer
        for layer, layer_type in enumerate(layer_types)
        if layer_type == synthetic.KDA_LAYER_TYPE
    ]
    assert stage_2a < stage_2b
    assert stage_2b - stage_2a == {
        f"{LANGUAGE_MODEL}.layers.{layer}.self_attn.{name}"
        for layer in kda_layers
        for name in KDA_MERGED_PROJECTIONS
    }
    assert len(stage_2b - stage_2a) == 34 * 6


def test_the_preset_names_list_the_five_stages_in_order() -> None:
    """`--preset` に渡せる名前は、第 1 段 `k2s1`、第 2a 段 `k2s2a`、第 2b 段 `k2s2b`、
    第 3 段 `k2s3`、第 4 段 `k2s4` の 5 つ (C1。Issue #99)。"""
    assert presets.PRESET_NAMES == ("k2s1", "k2s2a", "k2s2b", "k2s3", "k2s4")


def test_stage2a_pattern_string_is_unchanged_by_adding_stage2b() -> None:
    """第 2a 段の選び方の文字列は、第 2b 段を足しても、1 文字も変わらない。

    manifest の `pattern` と `args` の `--pattern` に、この文字列が残り、`serve derived-import` が
    その `args` を 2 台目の変換にそのまま使う。第 2a 段の重み (2026-09-27 の探りで使った) を作り
    直すときに、同じ `args` で同じ選び方になるように、文字列を固定する。
    """
    config = _config([synthetic.KDA_LAYER_TYPE] * 3 + [synthetic.MLA_LAYER_TYPE])
    expected = "|".join(
        [
            selection.DEFAULT_PATTERN,
            r".*\.layers\.(?:3)\.self_attn\.(?:q_a_proj|kv_a_proj_with_mqa|q_b_proj|o_proj)$",
            r".*\.layers\.(?:0|1|2)\.self_attn\.(?:o_proj|f_b_proj|g_b_proj)$",
            r"(?:.*\.)?lm_head$",
        ]
    )

    assert presets.resolve_preset_pattern("k2s2a", config) == expected
    assert presets.stage2a_pattern(config) == expected


# --- 第 3 段 (`k2s3`)。対象は `k2s2b` と同じ、形式だけ NVFP4A16 にする (Issue #95) ------------


def test_stage3_pattern_is_identical_to_the_stage2b_pattern() -> None:
    """第 3 段の選び方の文字列は、第 2b 段と 1 文字も違わない (対象は `k2s2b` と同じ) (C1)。"""
    config = _config([synthetic.KDA_LAYER_TYPE] * 3 + [synthetic.MLA_LAYER_TYPE])

    assert presets.stage3_pattern(config) == presets.stage2b_pattern(config)


def test_resolve_preset_pattern_for_stage3_matches_stage2b_across_45_layers() -> None:
    """`resolve_preset_pattern("k2s3", …)` は、実機と同じ 45 層の並びでも、`k2s2b` と同じ選び方に
    なる (C1)。"""
    layer_types = _layer_types_across_45_layers()
    config = _config(layer_types)

    assert presets.resolve_preset_pattern("k2s3", config) == presets.resolve_preset_pattern(
        "k2s2b", config
    )
    selected = selection.select_modules(
        _candidates_across_45_layers(layer_types), presets.resolve_preset_pattern("k2s3", config)
    )
    expected = selection.select_modules(
        _candidates_across_45_layers(layer_types), presets.stage2b_pattern(config)
    )
    assert selected == expected


def test_resolve_preset_format_maps_fp8_presets_to_fp8_and_k2s3_k2s4_to_nvfp4a16() -> None:
    """`resolve_preset_format` は、`k2s1`/`k2s2a`/`k2s2b` を `fp8`、`k2s3`/`k2s4` を `nvfp4a16` に
    する (C1。Issue #99)。"""
    assert presets.resolve_preset_format("k2s1") == "fp8"
    assert presets.resolve_preset_format("k2s2a") == "fp8"
    assert presets.resolve_preset_format("k2s2b") == "fp8"
    assert presets.resolve_preset_format("k2s3") == "nvfp4a16"
    assert presets.resolve_preset_format("k2s4") == "nvfp4a16"


def test_resolve_preset_format_rejects_an_unknown_preset_name() -> None:
    """未知の preset 名は、形式を決めずに断る (C1)。"""
    with pytest.raises(selection.SelectionError):
        presets.resolve_preset_format("k2s9")


# --- 第 4 段 (`k2s4`)。層 45 (MTP) の MLA の射影と FP8 の専門家を k2s3 の対象に足す (Issue #99) --

STAGE4_MTP_MLA_PROJECTIONS = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj")
"""第 4 段が層 45 (MTP) の `self_attn` に足す MLA の射影 4 つ (要求シナリオ SCN-C1-P1 の Given)。"""


def test_stage4_pattern_selects_the_mtp_mla_projections_and_fp8_expert_added_to_stage3() -> None:
    """`--preset k2s4` は、`k2s3` の対象に、層 45 (MTP) の MLA の射影 4 つと FP8 の専門家 1 つの
    `gate_proj`・`up_proj`・`down_proj` の、計 7 個を足して選び、形式は NVFP4A16 になる
    (C1。SCN-C1-P1)。
    """
    layer_types = _layer_types_across_45_layers()
    config = _config(layer_types)
    candidates = _candidates_across_45_layers(layer_types)
    mtp = f"{LANGUAGE_MODEL}.layers.45"
    new_names = [f"{mtp}.self_attn.{name}" for name in STAGE4_MTP_MLA_PROJECTIONS] + [
        f"{mtp}.mlp.experts.0.{name}" for name in ("gate_proj", "up_proj", "down_proj")
    ]
    assert len(new_names) == 7
    stage3_selected = set(selection.select_modules(candidates, presets.stage3_pattern(config)))

    selected = selection.select_modules(
        [*candidates, *_weights(*new_names)], presets.stage4_pattern(config)
    )

    assert set(selected) == stage3_selected | set(new_names)
    assert presets.resolve_preset_format("k2s4") == "nvfp4a16"


def test_stage4_pattern_does_not_select_lookalikes_at_the_mtp_layer_or_elsewhere() -> None:
    """同じ字面でも、層 45 の `kv_b_proj`・indexer・`eh_proj`・`shared_head.head`、KDA の層 44 の
    `q_a_proj`、専門家の `weight_packed` だけ、層 145 の `o_proj` は選ばれない (C1。SCN-C1-N1)。
    """
    layer_types = _layer_types_across_45_layers()
    config = _config(layer_types)
    names = [
        *_weights(
            f"{LANGUAGE_MODEL}.layers.45.self_attn.kv_b_proj",
            f"{LANGUAGE_MODEL}.layers.45.self_attn.indexer.wq_b",
            f"{LANGUAGE_MODEL}.layers.45.eh_proj",
            f"{LANGUAGE_MODEL}.layers.45.shared_head.head",
            f"{LANGUAGE_MODEL}.layers.44.self_attn.q_a_proj",
            f"{LANGUAGE_MODEL}.layers.145.self_attn.o_proj",
        ),
        f"{LANGUAGE_MODEL}.layers.3.mlp.experts.0.gate_proj.weight_packed",
    ]

    selected = selection.select_modules(names, presets.stage4_pattern(config))

    assert selected == ()


def test_resolve_preset_pattern_for_stage4_matches_stage4_pattern_across_45_layers() -> None:
    """`resolve_preset_pattern(\"k2s4\", …)` は、実機と同じ 45 層の並びでも、`stage4_pattern` と
    同じ選び方になる (C1)。"""
    layer_types = _layer_types_across_45_layers()
    config = _config(layer_types)

    assert presets.resolve_preset_pattern("k2s4", config) == presets.stage4_pattern(config)
