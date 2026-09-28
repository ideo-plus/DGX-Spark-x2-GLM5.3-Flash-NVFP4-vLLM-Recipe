"""`quantization_config` の書き換えの試験 (C3)。

入力 `config.json` と比べ、未使用の最小の `group_N` が 1 つ増え、対象のモジュール名だけが
`ignore` から消え、既存 group とそれ以外の `ignore`、top-level の他のキーは変わらない。
対象が既存 group の target に当たる場合は、出力を書く前に `ConfigError` で止まる。ただし、
当たった対象が全部 FP8 入力 (Issue #99。`fp8_input_modules`) で、その group の候補
(`candidates`) が全部変換されていれば例外で、その group を取り除いて置き換える (第 4 段)。
新しい group の `targets` は、渡したモジュールの実行時の名前にだけ当たる (vLLM と同じ
`re.match` で当てて確かめる)。実行時の名前の形が分からないモジュールは、`ConfigError` で断る。
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

import synthetic
from k2_quant import quant_config

EXPECTED_WEIGHTS: dict[str, Any] = {
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


def _config(root: Path) -> dict[str, Any]:
    config: dict[str, Any] = json.loads((root / synthetic.CONFIG_NAME).read_text(encoding="utf-8"))
    return config


def _hits(target: str, runtime_name: str) -> bool:
    """vLLM と同じく、`re:` 付き target を実行時の名前に `re.match` で当てる。"""
    assert target.startswith("re:")
    return re.match(target.removeprefix("re:"), runtime_name) is not None


def _without(config: dict[str, Any], *keys: str) -> dict[str, Any]:
    quantization_config = config["quantization_config"]
    return {k: v for k, v in quantization_config.items() if k not in keys}


def test_matches_target_uses_re_match_for_regex() -> None:
    """`re:` 付き target はモジュール名に `re.match` で当てる (C3)。"""
    assert quant_config.matches_target(
        "model.language_model.layers.3.mlp.experts.0.gate_proj",
        r"re:.*\.experts\.\d+\.gate_proj$",
    )
    assert not quant_config.matches_target(
        "model.language_model.layers.3.mlp.experts.0.gate_proj",
        r"re:.*\.experts\.\d+\.down_proj$",
    )


def test_matches_target_uses_equality_without_re_prefix() -> None:
    """`re:` の無い target は等値で当てる (C3)。"""
    assert quant_config.matches_target("lm_head", "lm_head")
    assert not quant_config.matches_target("language_model.lm_head", "lm_head")


def test_next_group_name_is_smallest_unused() -> None:
    """未使用の最小の `group_N` を選ぶ (C3)。"""
    assert quant_config.next_group_name({"group_0": {}, "group_1": {}}) == "group_2"
    assert quant_config.next_group_name({"group_0": {}, "group_1": {}, "group_2": {}}) == "group_3"


def test_add_group_adds_only_the_new_group_and_removes_ignored_names(tmp_path: Path) -> None:
    """新しい group は未使用の最小の `group_N` になり、対象だけが `ignore` から消える (C3)。"""
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)
    original = copy.deepcopy(before)

    result = quant_config.add_fp8_channel_group(before, modules=["lm_head"])

    groups = result["quantization_config"]["config_groups"]
    new_group = groups["group_2"]
    assert {key: value for key, value in new_group.items() if key != "targets"} == {
        "format": "float-quantized",
        "input_activations": None,
        "output_activations": None,
        "weights": EXPECTED_WEIGHTS,
    }
    (target,) = new_group["targets"]
    assert _hits(target, "language_model.lm_head")
    assert not _hits(target, "language_model.model.layers.0.mlp.gate_proj")
    assert groups["group_0"] == before["quantization_config"]["config_groups"]["group_0"]
    assert groups["group_1"] == before["quantization_config"]["config_groups"]["group_1"]
    assert _without(result, "config_groups", "ignore") == _without(
        before, "config_groups", "ignore"
    )

    before_ignore = before["quantization_config"]["ignore"]
    after_ignore = result["quantization_config"]["ignore"]
    assert after_ignore == [name for name in before_ignore if name != "lm_head"]
    assert before == original


def test_add_group_does_not_overwrite_existing_group_name(tmp_path: Path) -> None:
    """既存の `group_2` を上書きせず、新しい group は `group_3` になる (C3)。"""
    extra_group = {"format": "float-quantized", "targets": ["re:placeholder"], "weights": {}}
    synthetic.build_checkpoint(tmp_path, config_groups_extra={"group_2": extra_group})
    before = _config(tmp_path)

    result = quant_config.add_fp8_channel_group(before, modules=["lm_head"])

    groups = result["quantization_config"]["config_groups"]
    assert set(groups) == {"group_0", "group_1", "group_2", "group_3"}
    assert groups["group_2"] == extra_group


def test_add_group_target_matches_only_the_runtime_names_of_the_given_modules(
    tmp_path: Path,
) -> None:
    """`targets` は、渡したモジュールの実行時の名前にだけ当たる (本体・MTP・`lm_head` の接頭辞)。

    層 0 と層 10 のように、名前が前方一致するだけのものや、渡していない射影・専門家・MTP の
    head には当たらない。
    """
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)
    modules = [
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.45.eh_proj",
        "lm_head",
    ]

    result = quant_config.add_fp8_channel_group(before, modules=modules)

    (target,) = result["quantization_config"]["config_groups"]["group_2"]["targets"]
    for name in (
        "language_model.model.layers.0.mlp.gate_proj",
        "model.layers.45.eh_proj",
        "language_model.lm_head",
        "lm_head",
    ):
        assert _hits(target, name), name
    for name in (
        "language_model.model.layers.10.mlp.gate_proj",
        "language_model.model.layers.0.mlp.up_proj",
        "language_model.model.layers.3.mlp.experts.0.gate_proj",
        "model.layers.45.shared_head.head",
    ):
        assert not _hits(target, name), name


def test_add_group_target_does_not_depend_on_the_order_or_repeats_of_modules(
    tmp_path: Path,
) -> None:
    """同じモジュールの集合からは、並びや重複に依らず同じ config になる (決定性)。"""
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)
    first = "model.language_model.layers.0.mlp.gate_proj"
    second = "lm_head"

    forward = quant_config.add_fp8_channel_group(before, modules=[first, second])
    shuffled = quant_config.add_fp8_channel_group(before, modules=[second, first, second])

    assert forward == shuffled


def test_add_group_rejects_a_module_whose_runtime_name_is_unknown(tmp_path: Path) -> None:
    """実行時の名前の形が分からないモジュールは断る (target を変換した集合に限れないため)。

    `model.language_model.` の下でも、最上位の名前でもない名前は、実行時の名前が決まらない。
    """
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)

    with pytest.raises(quant_config.ConfigError):
        quant_config.add_fp8_channel_group(before, modules=["model.visual.blocks.0.mlp.gate_proj"])


def test_add_group_rejects_module_matching_existing_target(tmp_path: Path) -> None:
    """既存 group の target に当たるモジュールは変換を断る (C3)。"""
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)

    with pytest.raises(quant_config.ConfigError):
        quant_config.add_fp8_channel_group(
            before,
            modules=["model.language_model.layers.3.mlp.experts.0.gate_proj"],
        )


def test_add_group_rejects_wrong_format(tmp_path: Path) -> None:
    """`format` が `mixed-precision` でなければ `ConfigError` (C3)。"""
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)
    before["quantization_config"]["format"] = "float-quantized"

    with pytest.raises(quant_config.ConfigError):
        quant_config.add_fp8_channel_group(before, modules=["lm_head"])


def test_add_group_rejects_wrong_quant_method(tmp_path: Path) -> None:
    """`quant_method` が `compressed-tensors` でなければ `ConfigError` (C3)。"""
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)
    before["quantization_config"]["quant_method"] = "awq"

    with pytest.raises(quant_config.ConfigError):
        quant_config.add_fp8_channel_group(before, modules=["lm_head"])


def test_add_group_rejects_missing_quantization_config(tmp_path: Path) -> None:
    """`quantization_config` が無ければ `ConfigError` (C3)。"""
    with pytest.raises(quant_config.ConfigError):
        quant_config.add_fp8_channel_group({"model_type": "glm5next"}, modules=["lm_head"])


def test_add_group_constant_matches_contract() -> None:
    """`FP8_CHANNEL_WEIGHTS` は契約の weights 辞書そのもの (C3)。"""
    assert dict(quant_config.FP8_CHANNEL_WEIGHTS) == EXPECTED_WEIGHTS


# --- 第 3 段 (`k2s3`): NVFP4A16 の group (`add_nvfp4a16_group`。Issue #95) ----------------------

EXPECTED_NVFP4_WEIGHTS: dict[str, Any] = {
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
"""元の重みの専門家の NVFP4 の group (`group_0`) の `weights` (2026-09-27 に HF の
rev `18d55bfd…` の `config.json` から逐語で確認した値)。"""


def test_add_nvfp4a16_group_matches_the_original_nvfp4_groups_format_and_weights(
    tmp_path: Path,
) -> None:
    """新しい group の `format` と `weights` は、元の重みの NVFP4 の group (`group_0`) と
    同じ形になる (C4)。
    """
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)
    original_group_0 = before["quantization_config"]["config_groups"]["group_0"]

    result = quant_config.add_nvfp4a16_group(
        before, modules=["lm_head"], fp8_input_modules=(), candidates=()
    )

    new_group = result["quantization_config"]["config_groups"]["group_2"]
    assert new_group["format"] == original_group_0["format"]
    assert new_group["weights"] == original_group_0["weights"]
    assert new_group["format"] == "nvfp4-pack-quantized"


def test_add_nvfp4a16_group_has_no_input_or_output_activations(tmp_path: Path) -> None:
    """新しい group は重みだけの量子化なので、`input_activations`/`output_activations` は
    null になる。元の専門家の group (`group_0`) は W4A4 (`input_activations` がある) だが、
    NVFP4A16 は重みだけなので null にする (C4)。
    """
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)
    group_0 = before["quantization_config"]["config_groups"]["group_0"]
    assert group_0["input_activations"] is not None

    result = quant_config.add_nvfp4a16_group(
        before, modules=["lm_head"], fp8_input_modules=(), candidates=()
    )

    new_group = result["quantization_config"]["config_groups"]["group_2"]
    assert new_group["input_activations"] is None
    assert new_group["output_activations"] is None


def test_add_nvfp4a16_group_target_matches_only_the_runtime_names_of_the_given_modules(
    tmp_path: Path,
) -> None:
    """`targets` は、渡したモジュールの実行時の名前にだけ当たる (FP8 と仕組みを共有する) (C4)。"""
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)

    result = quant_config.add_nvfp4a16_group(
        before,
        modules=["model.language_model.layers.0.mlp.gate_proj"],
        fp8_input_modules=(),
        candidates=(),
    )

    (target,) = result["quantization_config"]["config_groups"]["group_2"]["targets"]
    assert _hits(target, "language_model.model.layers.0.mlp.gate_proj")
    assert not _hits(target, "language_model.model.layers.0.mlp.up_proj")


def test_add_nvfp4a16_group_removes_only_the_converted_names_from_ignore(tmp_path: Path) -> None:
    """`ignore` からは、渡したモジュールの名前だけが消える (FP8 と同じ仕組みを共有する) (C4)。"""
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)

    result = quant_config.add_nvfp4a16_group(
        before, modules=["lm_head"], fp8_input_modules=(), candidates=()
    )

    ignore_before = before["quantization_config"]["ignore"]
    ignore_after = result["quantization_config"]["ignore"]
    assert "lm_head" not in ignore_after
    assert ignore_after == [name for name in ignore_before if name != "lm_head"]


def test_add_nvfp4a16_group_rejects_a_module_matching_an_existing_target(tmp_path: Path) -> None:
    """既存 group の target に当たるモジュールは、書く前に断る (FP8 と同じ検証を共有する) (C4)。"""
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)

    with pytest.raises(quant_config.ConfigError):
        quant_config.add_nvfp4a16_group(
            before,
            modules=["model.language_model.layers.3.mlp.experts.0.gate_proj"],
            fp8_input_modules=(),
            candidates=(),
        )


def test_nvfp4_tensor_group_weights_constant_matches_the_original_group_contract() -> None:
    """`NVFP4_TENSOR_GROUP_WEIGHTS` は、元の重みの NVFP4 の group の `weights` そのもの (C4)。"""
    assert dict(quant_config.NVFP4_TENSOR_GROUP_WEIGHTS) == EXPECTED_NVFP4_WEIGHTS


# --- 第 4 段 (`k2s4`): FP8 の block の設定を読む (`fp8_block_structure`。Issue #99) ------------

FP8_BLOCK_WEIGHTS: dict[str, Any] = {
    "actorder": None,
    "block_structure": [128, 128],
    "dynamic": False,
    "group_size": None,
    "num_bits": 8,
    "observer": "memoryless_minmax",
    "observer_kwargs": {},
    "scale_dtype": "torch.float32",
    "strategy": "block",
    "symmetric": True,
    "type": "float",
    "zp_dtype": None,
}
"""層 45 の FP8 の専門家を覆う `group_1` の `weights` (`synthetic._group_1` と同じ形)。"""


def _fp8_group(target: str) -> dict[str, Any]:
    return {
        "format": "float-quantized",
        "targets": [target],
        "weights": dict(FP8_BLOCK_WEIGHTS),
        "input_activations": {},
        "output_activations": None,
    }


def _config_with_groups(groups: dict[str, Any]) -> dict[str, Any]:
    return {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": groups,
            "ignore": [],
        }
    }


EXPERT_MODULE = "model.language_model.layers.45.mlp.experts.0.gate_proj"
EXPERT_TARGET = r"re:.*\.layers\.45\.mlp\.experts\.\d+\.gate_proj$"


def test_fp8_block_structure_reads_the_block_shape_from_the_matching_group() -> None:
    """当たる group がちょうど 1 つで `strategy=block` なら、`(高さ, 幅)` を返す (C4)。"""
    config = _config_with_groups({"group_1": _fp8_group(EXPERT_TARGET)})

    assert quant_config.fp8_block_structure(config, EXPERT_MODULE) == (128, 128)


def test_fp8_block_structure_rejects_a_module_matched_by_no_group() -> None:
    """当たる group が 1 つも無ければ断る (C4)。"""
    config = _config_with_groups(
        {"group_1": _fp8_group(r"re:.*\.layers\.3\.mlp\.experts\.\d+\.gate_proj$")}
    )

    with pytest.raises(quant_config.ConfigError):
        quant_config.fp8_block_structure(config, EXPERT_MODULE)


def test_fp8_block_structure_rejects_a_module_matched_by_more_than_one_group() -> None:
    """当たる group が 2 つ以上あれば断る (どの group の block か決まらない) (C4)。"""
    config = _config_with_groups(
        {"group_1": _fp8_group(EXPERT_TARGET), "group_2": _fp8_group(EXPERT_TARGET)}
    )

    with pytest.raises(quant_config.ConfigError):
        quant_config.fp8_block_structure(config, EXPERT_MODULE)


def test_fp8_block_structure_rejects_a_non_float_quantized_format() -> None:
    """当たる group の `format` が `float-quantized` でなければ断る (C4)。"""
    group = _fp8_group(EXPERT_TARGET)
    group["format"] = "nvfp4-pack-quantized"
    config = _config_with_groups({"group_1": group})

    with pytest.raises(quant_config.ConfigError):
        quant_config.fp8_block_structure(config, EXPERT_MODULE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strategy", "channel"),
        ("type", "int"),
        ("num_bits", 4),
        ("symmetric", False),
        ("block_structure", None),
        ("block_structure", [128, 0]),
    ],
)
def test_fp8_block_structure_rejects_weights_that_do_not_match_the_block_contract(
    field: str, value: object
) -> None:
    """当たる group の `weights` が block の FP8 の契約 (`strategy=block`・`type=float`・
    `num_bits=8`・`symmetric=True`・正の整数 2 つの `block_structure`) と違えば断る (C4)。
    """
    group = _fp8_group(EXPERT_TARGET)
    group["weights"][field] = value
    config = _config_with_groups({"group_1": group})

    with pytest.raises(quant_config.ConfigError):
        quant_config.fp8_block_structure(config, EXPERT_MODULE)


# --- 第 4 段: 既存 group を候補ごと置き換える (`candidates`。C4。Issue #99) --------------------


def test_add_nvfp4a16_group_rejects_a_non_fp8_input_module_even_if_candidates_are_fully_converted(
    tmp_path: Path,
) -> None:
    """当たった対象の一部が `fp8_input_modules` に無ければ、`candidates` が全部 `modules` に
    含まれていても取り除かず、即座に `ConfigError` で断り、元の config を書き換えない
    (C4。SCN-A-N2。Issue #99 の修正: FP8 入力でない対象には `candidates` の判定を適用しない)。
    """
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)
    original = copy.deepcopy(before)
    non_fp8_module = "model.language_model.layers.45.mlp.experts.1.gate_proj"

    with pytest.raises(quant_config.ConfigError) as excinfo:
        quant_config.add_nvfp4a16_group(
            before,
            modules=[EXPERT_MODULE, non_fp8_module],
            fp8_input_modules=[EXPERT_MODULE],
            candidates=[EXPERT_MODULE, non_fp8_module],
        )

    assert non_fp8_module in str(excinfo.value)
    assert "already matched by existing target" in str(excinfo.value)
    assert before == original


def test_add_nvfp4a16_group_removes_an_existing_group_when_all_its_candidates_are_converted(
    tmp_path: Path,
) -> None:
    """`candidates` を渡し、当たる既存 group (`group_1`。層 45 の専門家) の候補が全部 `modules` に
    含まれれば、その group を取り除いて新しい group に置き換える (C4。SCN-C4-P1)。

    新しい group の名前は、取り除く前の `next_group_name` (`group_0`・`group_1` が使用中なので
    `group_2`)。取り除いた `group_1` の空いた番号 (`group_1`) を再利用しない。
    """
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)

    result = quant_config.add_nvfp4a16_group(
        before,
        modules=[EXPERT_MODULE],
        fp8_input_modules=[EXPERT_MODULE],
        candidates=[EXPERT_MODULE],
    )

    groups = result["quantization_config"]["config_groups"]
    assert set(groups) == {"group_0", "group_2"}
    assert groups["group_0"] == before["quantization_config"]["config_groups"]["group_0"]
    (target,) = groups["group_2"]["targets"]
    assert _hits(target, "model.layers.45.mlp.experts.0.gate_proj")


def test_add_nvfp4a16_group_rejects_partial_conversion_of_an_existing_groups_candidates(
    tmp_path: Path,
) -> None:
    """`candidates` の一部だけが `modules` に含まれれば、変換していない候補の名前を挙げて断り、
    元の config を書き換えない (C4。SCN-C4-N2)。
    """
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)
    original = copy.deepcopy(before)
    converted = EXPERT_MODULE
    not_converted = "model.language_model.layers.45.mlp.experts.1.gate_proj"

    with pytest.raises(quant_config.ConfigError) as excinfo:
        quant_config.add_nvfp4a16_group(
            before,
            modules=[converted],
            fp8_input_modules=[converted],
            candidates=[converted, not_converted],
        )

    assert not_converted in str(excinfo.value)
    assert before == original


def test_add_nvfp4a16_group_leaves_an_unrelated_existing_group_and_avoids_its_name(
    tmp_path: Path,
) -> None:
    """候補が対象の group と無関係な既存 `group_2` は変わらず残り、新しい group は
    `group_3` になる (C4。SCN-C4-N1)。
    """
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)
    unrelated = _fp8_group(r"re:.*\.layers\.7\.mlp\.experts\.\d+\.gate_proj$")
    before["quantization_config"]["config_groups"]["group_2"] = unrelated

    result = quant_config.add_nvfp4a16_group(
        before,
        modules=[EXPERT_MODULE],
        fp8_input_modules=[EXPERT_MODULE],
        candidates=[EXPERT_MODULE],
    )

    groups = result["quantization_config"]["config_groups"]
    assert set(groups) == {"group_0", "group_2", "group_3"}
    assert groups["group_2"] == unrelated
