"""`quantization_config` の書き換えの試験 (C3)。

入力 `config.json` と比べ、未使用の最小の `group_N` が 1 つ増え、対象のモジュール名だけが
`ignore` から消え、既存 group とそれ以外の `ignore`、top-level の他のキーは変わらない。
対象が既存 group の target に当たる場合は、出力を書く前に `ConfigError` で止まる。
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

    result = quant_config.add_nvfp4a16_group(before, modules=["lm_head"])

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

    result = quant_config.add_nvfp4a16_group(before, modules=["lm_head"])

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
        before, modules=["model.language_model.layers.0.mlp.gate_proj"]
    )

    (target,) = result["quantization_config"]["config_groups"]["group_2"]["targets"]
    assert _hits(target, "language_model.model.layers.0.mlp.gate_proj")
    assert not _hits(target, "language_model.model.layers.0.mlp.up_proj")


def test_add_nvfp4a16_group_removes_only_the_converted_names_from_ignore(tmp_path: Path) -> None:
    """`ignore` からは、渡したモジュールの名前だけが消える (FP8 と同じ仕組みを共有する) (C4)。"""
    synthetic.build_checkpoint(tmp_path)
    before = _config(tmp_path)

    result = quant_config.add_nvfp4a16_group(before, modules=["lm_head"])

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
            before, modules=["model.language_model.layers.3.mlp.experts.0.gate_proj"]
        )


def test_nvfp4_tensor_group_weights_constant_matches_the_original_group_contract() -> None:
    """`NVFP4_TENSOR_GROUP_WEIGHTS` は、元の重みの NVFP4 の group の `weights` そのもの (C4)。"""
    assert dict(quant_config.NVFP4_TENSOR_GROUP_WEIGHTS) == EXPECTED_NVFP4_WEIGHTS
