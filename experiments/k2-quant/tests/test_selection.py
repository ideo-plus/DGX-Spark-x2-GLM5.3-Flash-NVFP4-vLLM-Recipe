"""対象の選択 (正規表現と名前の規則) の試験 (C2)。

既定の正規表現が拾う側・拾わない側を確かめる。既定の正規表現は Issue #56 の第 1 段の範囲
(dense の MLP、共有の専門家、`lm_head`) を、実機 `ignore` と同じ形
(`model.language_model.layers.N....`) の checkpoint のテンソル名で拾い、専門家・ルーター・
attention・visual・MTP の `eh_proj` などは拾わない。`eh_proj` は、vLLM
(`vllm/models/glm5next/common/mtp.py:49`、commit 0961bbae) では quant_config を受けない
plain `nn.Linear` なので、既定の対象から外している。この正規表現は checkpoint の名前だけに
当てる。実行時の層名に当てる `config.json` の target は、変換したモジュールから作る
(`tests/test_quant_config.py`)。
"""

from __future__ import annotations

import pytest

from k2_quant import selection
from k2_quant.safetensors_file import TensorInfo

LANGUAGE_MODEL = "model.language_model"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _first_stage_modules() -> set[str]:
    """Issue #56 の「第 1 段で FP8 にする `ignore` の中の名前」のうち、既定で拾う名前を、
    実機の名前の形で作る。

    dense (層 0〜2) の 3 射影、共有の専門家 (層 3〜45 の 43 層) の 3 射影、`lm_head`。MTP の
    `eh_proj` は Issue の第 1 段の範囲に挙がっているが、vLLM が quant_config を受けない
    plain `nn.Linear` として実装しているため、既定では拾わない (`_out_of_range_modules` の
    decoy として確かめる)。
    """
    dense = {
        f"{LANGUAGE_MODEL}.layers.{layer}.mlp.{projection}"
        for layer in range(3)
        for projection in PROJECTIONS
    }
    shared = {
        f"{LANGUAGE_MODEL}.layers.{layer}.mlp.shared_experts.{projection}"
        for layer in range(3, 46)
        for projection in PROJECTIONS
    }
    return dense | shared | {"lm_head"}


def _out_of_range_modules() -> set[str]:
    """第 1 段の範囲に入らない、近い名前 (同じ字面の射影、隣の層、専門家、MTP の別の部品)。"""
    modules: set[str] = set()
    # dense の形は層 0〜2 だけ。`(?:0|1|2)` の境界を外して 10・12・20・21 に当たる退行を検出する
    for layer in (3, 10, 12, 20, 21, 30, 45):
        for projection in PROJECTIONS:
            modules.add(f"{LANGUAGE_MODEL}.layers.{layer}.mlp.{projection}")
    for layer in (3, 45):
        for projection in PROJECTIONS:
            modules.add(f"{LANGUAGE_MODEL}.layers.{layer}.mlp.experts.0.{projection}")
            modules.add(f"{LANGUAGE_MODEL}.layers.{layer}.mlp.experts.255.{projection}")
        modules.add(f"{LANGUAGE_MODEL}.layers.{layer}.mlp.gate")
    for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        modules.add(f"{LANGUAGE_MODEL}.layers.0.self_attn.{name}")
    for projection in PROJECTIONS:
        modules.add(f"model.visual.blocks.0.mlp.{projection}")
    modules |= {
        f"{LANGUAGE_MODEL}.layers.45.enorm",
        f"{LANGUAGE_MODEL}.layers.45.hnorm",
        f"{LANGUAGE_MODEL}.layers.45.shared_head.norm",
        # MTP の eh_proj: vLLM が quant_config を受けない plain nn.Linear なので既定から外す
        # (vllm/models/glm5next/common/mtp.py:49、commit 0961bbae)
        f"{LANGUAGE_MODEL}.layers.45.eh_proj",
        f"{LANGUAGE_MODEL}.embed_tokens",
        f"{LANGUAGE_MODEL}.norm",
    }
    return modules


def _info(name: str, dtype: str, shape: tuple[int, ...]) -> TensorInfo:
    return TensorInfo(name=name, dtype=dtype, shape=shape, data_offsets=(0, 0))


def test_module_name_strips_weight_suffix() -> None:
    """`.weight` だけをモジュール名の候補にする (C2)。"""
    assert selection.module_name("model.language_model.layers.0.mlp.gate_proj.weight") == (
        "model.language_model.layers.0.mlp.gate_proj"
    )


@pytest.mark.parametrize(
    "tensor_name",
    [
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_packed",
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale",
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_global_scale",
        "model.language_model.layers.0.mlp.gate_proj.bias",
    ],
)
def test_module_name_is_none_for_other_parameters(tensor_name: str) -> None:
    """`.weight` 以外のパラメータは候補にならない (C2)。"""
    assert selection.module_name(tensor_name) is None


def test_default_pattern_selects_first_stage_names() -> None:
    """既定の正規表現は第 1 段の 3 種の名前を拾い、MTP の `eh_proj` は拾わない (C2)。

    `eh_proj` は、vLLM (`vllm/models/glm5next/common/mtp.py:49`、commit 0961bbae) では
    quant_config を受けない plain `nn.Linear` なので、テンソルは checkpoint にあっても
    既定では選ばれない。
    """
    tensor_names = [
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.3.mlp.shared_experts.down_proj.weight",
        "model.language_model.layers.45.mlp.shared_experts.up_proj.weight",
        "model.language_model.layers.45.eh_proj.weight",
        "lm_head.weight",
    ]

    selected = selection.select_modules(tensor_names, selection.DEFAULT_PATTERN)

    assert set(selected) == {
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.3.mlp.shared_experts.down_proj",
        "model.language_model.layers.45.mlp.shared_experts.up_proj",
        "lm_head",
    }


def test_default_pattern_selects_exactly_the_first_stage_range() -> None:
    """既定の正規表現は、第 1 段の 139 個の名前を全部拾い、近い名前は 1 つも拾わない (C2)。

    dense の層 0〜2 × 3 射影、共有の専門家の層 3〜45 × 3 射影、`lm_head` を実機の名前の形で
    作り、範囲外の近い名前と混ぜて、選ばれた集合が一致することを固定する。範囲外の近い名前は、
    dense の形の層 3・10・12・20・21・30・45、専門家、ルーター、attention、visual、MTP の
    `eh_proj` と別の部品。
    """
    expected = _first_stage_modules()
    assert len(expected) == 9 + 43 * 3 + 1
    decoys = _out_of_range_modules()
    assert not (expected & decoys)
    tensor_names = [f"{module}.weight" for module in sorted(expected | decoys)]

    selected = selection.select_modules(tensor_names, selection.DEFAULT_PATTERN)

    assert set(selected) == expected


def test_default_pattern_selects_the_mtp_shared_head_when_present() -> None:
    """MTP の head が別の重みとしてあれば、既定で選ぶ (Issue #56。名前は実機で未確認)。"""
    shared_head = "model.language_model.layers.45.shared_head.head"

    selected = selection.select_modules(
        [f"{shared_head}.weight", "model.language_model.layers.45.shared_head.norm.weight"],
        selection.DEFAULT_PATTERN,
    )

    assert selected == (shared_head,)


def test_default_pattern_rejects_non_dense_positions() -> None:
    """同じ字面 `gate_proj` でも dense 以外の位置は拾わない (C2)。"""
    tensor_names = [
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_packed",
        "model.visual.blocks.0.mlp.gate_proj.weight",
        "model.language_model.layers.30.mlp.gate_proj.weight",
        "model.language_model.layers.3.mlp.gate.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.embed_tokens.weight",
    ]

    assert selection.select_modules(tensor_names, selection.DEFAULT_PATTERN) == ()


def test_select_modules_returns_empty_tuple_when_nothing_matches() -> None:
    """当たらない正規表現では 1 つも選ばれない (C2)。"""
    assert (
        selection.select_modules(
            ["model.language_model.layers.9.mlp.gate_proj.weight"],
            r"model\.language_model\.layers\.0\.mlp\.gate_proj$",
        )
        == ()
    )


def test_validate_targets_accepts_two_dimensional_weight() -> None:
    """対象の `.weight` が 2 次元で BF16 なら通す (C2)。"""
    tensors = {
        "model.language_model.layers.0.mlp.gate_proj.weight": _info(
            "model.language_model.layers.0.mlp.gate_proj.weight", "BF16", (8, 4)
        )
    }

    selection.validate_targets(["model.language_model.layers.0.mlp.gate_proj"], tensors)


def test_validate_targets_rejects_empty_selection() -> None:
    """対象 0 件は拒否する (C2)。"""
    with pytest.raises(selection.SelectionError):
        selection.validate_targets([], {})


def test_validate_targets_rejects_extra_parameter() -> None:
    """対象モジュールに `.weight` 以外のパラメータがあれば拒否する (C2)。"""
    module = "model.language_model.layers.0.mlp.gate_proj"
    tensors = {
        f"{module}.weight": _info(f"{module}.weight", "BF16", (8, 4)),
        f"{module}.bias": _info(f"{module}.bias", "BF16", (8,)),
    }

    with pytest.raises(selection.SelectionError):
        selection.validate_targets([module], tensors)


def test_validate_targets_rejects_unsupported_dtype() -> None:
    """対象の `.weight` が BF16/F16/F32 でなければ拒否する (C2)。"""
    module = "model.language_model.layers.0.mlp.gate_proj"
    tensors = {f"{module}.weight": _info(f"{module}.weight", "F8_E4M3", (8, 4))}

    with pytest.raises(selection.SelectionError):
        selection.validate_targets([module], tensors)


def test_validate_targets_rejects_non_two_dimensional_weight() -> None:
    """対象の `.weight` が 2 次元でなければ拒否する (C2)。"""
    module = "model.language_model.layers.0.mlp.gate_proj"
    tensors = {f"{module}.weight": _info(f"{module}.weight", "BF16", (8,))}

    with pytest.raises(selection.SelectionError):
        selection.validate_targets([module], tensors)
