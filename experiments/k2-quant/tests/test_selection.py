"""対象の選択 (正規表現と名前の規則) の試験 (C2)。

既定の正規表現が拾う側・拾わない側を確かめる。既定の正規表現は Issue #56 の第 1 段の範囲のうち
dense の MLP と共有の専門家を、実機 `ignore` と同じ形 (`model.language_model.layers.N....`) の
checkpoint のテンソル名で拾い、専門家・ルーター・attention・visual・`lm_head`・MTP の
`eh_proj` と `shared_head.head` などは拾わない。`eh_proj` は、vLLM
(`vllm/models/glm5next/common/mtp.py:49`、commit 0961bbae) では quant_config を受けない
plain `nn.Linear` なので、既定の対象から外している。`lm_head` と `shared_head.head` は、
FP8 (W8A16、compressed-tensors) の `ParallelLMHead` を humming の線形カーネルで読めない
(#68) ので、既定の対象から外している。この正規表現は checkpoint の名前だけに当てる。実行時の層名に
当てる `config.json` の target は、変換したモジュールから作る (`tests/test_quant_config.py`)。

まとめた層の整合の検査 (Issue #74 の C3) も確かめる。vLLM で 1 つの線形層にまとまる組
(MLA の `q_a_proj` と `kv_a_proj_with_mqa`、MLP の `gate_proj` と `up_proj`、KDA の
q・k・v・b・f_a・g_a) は、組のうち checkpoint に実在する名前の一部だけが選ばれたら断り、
全員が選ばれるか、相手が checkpoint に無ければ通す。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from k2_quant import selection
from k2_quant.safetensors_file import TensorInfo

LANGUAGE_MODEL = "model.language_model"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

MLA_FUSED_GROUP = ("q_a_proj", "kv_a_proj_with_mqa")
MLP_FUSED_GROUP = ("gate_proj", "up_proj")
KDA_FUSED_GROUP = ("q_proj", "k_proj", "v_proj", "b_proj", "forget_gate.f_a_proj", "g_a_proj")
"""Issue #74 が挙げる、vLLM で 1 つの線形層にまとまる組 (MLA の `fused_qkv_a_proj`、MLP の
`gate_up_proj`、KDA の `in_proj_qkvbfg_a`)。"""

FUSED_GROUP_LOCATIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("mla-q_a-kv_a", f"{LANGUAGE_MODEL}.layers.3.self_attn", MLA_FUSED_GROUP),
    ("dense-gate-up", f"{LANGUAGE_MODEL}.layers.0.mlp", MLP_FUSED_GROUP),
    ("shared-expert-gate-up", f"{LANGUAGE_MODEL}.layers.3.mlp.shared_experts", MLP_FUSED_GROUP),
    ("kda-qkvbfg-a", f"{LANGUAGE_MODEL}.layers.0.self_attn", KDA_FUSED_GROUP),
)
"""(試験の名前, 組の親の名前, 組の名前)。組の全員は同じ親の下にある。"""

FUSED_GROUP_PARAMS = [
    pytest.param(parent, members, id=label) for label, parent, members in FUSED_GROUP_LOCATIONS
]


def _first_stage_modules() -> set[str]:
    """Issue #56 の「第 1 段で FP8 にする `ignore` の中の名前」のうち、既定で拾う名前を、
    実機の名前の形で作る。

    dense (層 0〜2) の 3 射影、共有の専門家 (層 3〜45 の 43 層) の 3 射影。MTP の `eh_proj`
    (vLLM が quant_config を受けない plain `nn.Linear` として実装している) と、`lm_head`・MTP の
    `shared_head.head` (FP8 の `ParallelLMHead` を読めない。#68) は、Issue の第 1 段の範囲に
    挙がっているが、既定では拾わない (`_out_of_range_modules` の decoy として確かめる)。
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
    return dense | shared


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
        # lm_head と MTP の shared_head.head: FP8 (W8A16) の ParallelLMHead を humming の線形
        # カーネルで読めないので既定から外す (#68)
        "lm_head",
        f"{LANGUAGE_MODEL}.layers.45.shared_head.head",
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


def test_default_pattern_selects_dense_and_shared_experts_only() -> None:
    """既定の正規表現は dense と共有の専門家の 2 種の名前を拾い、MTP の `eh_proj` と `lm_head` は
    拾わない (C2)。

    `eh_proj` は、vLLM (`vllm/models/glm5next/common/mtp.py:49`、commit 0961bbae) では
    quant_config を受けない plain `nn.Linear` なので、テンソルは checkpoint にあっても
    既定では選ばれない。`lm_head` は、FP8 の `ParallelLMHead` を読めない (#68) ので選ばれない。
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
    }


def test_default_pattern_selects_exactly_the_first_stage_range() -> None:
    """既定の正規表現は、dense と共有の専門家の 138 個を全部拾い、近い名前は 1 つも拾わない (C2)。

    dense の層 0〜2 × 3 射影、共有の専門家の層 3〜45 × 3 射影を実機の名前の形で作り、範囲外の
    近い名前と混ぜて、選ばれた集合が一致することを固定する。範囲外の近い名前は、dense の形の
    層 3・10・12・20・21・30・45、専門家、ルーター、attention、visual、`lm_head`、MTP の
    `eh_proj` と `shared_head.head` と別の部品。
    """
    expected = _first_stage_modules()
    assert len(expected) == 9 + 43 * 3
    decoys = _out_of_range_modules()
    assert not (expected & decoys)
    tensor_names = [f"{module}.weight" for module in sorted(expected | decoys)]

    selected = selection.select_modules(tensor_names, selection.DEFAULT_PATTERN)

    assert set(selected) == expected


def test_default_pattern_does_not_select_the_mtp_shared_head() -> None:
    """MTP の head が別の重みとしてあっても、既定では選ばない (#68)。

    `shared_head.head` は `SharedHead` (`vllm/model_executor/models/deepseek_mtp.py`) の中の
    `ParallelLMHead` で、FP8 (W8A16、compressed-tensors) の `ParallelLMHead` を humming の
    線形カーネルで読めない。
    """
    shared_head = "model.language_model.layers.45.shared_head.head"

    selected = selection.select_modules(
        [f"{shared_head}.weight", "model.language_model.layers.45.shared_head.norm.weight"],
        selection.DEFAULT_PATTERN,
    )

    assert selected == ()


@pytest.mark.parametrize(
    "tensor_name",
    [
        "lm_head.weight",
        "model.language_model.lm_head.weight",
    ],
)
def test_default_pattern_does_not_select_lm_head(tensor_name: str) -> None:
    """`lm_head` は、最上位の名前でも、接頭辞つきの名前でも、既定では選ばない (#68)。

    FP8 (W8A16、compressed-tensors) の `ParallelLMHead` を humming の線形カーネルで読めず、
    `AttributeError: 'ParallelLMHead' object has no attribute 'output_partition_sizes'` で
    起動できなかった。
    """
    assert selection.select_modules([tensor_name], selection.DEFAULT_PATTERN) == ()


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


def _group_tensors(parent: str, members: Sequence[str]) -> dict[str, TensorInfo]:
    """親の下に、組の名前ごとの BF16 の 2 次元の `.weight` を持つ checkpoint のテンソル。"""
    return {
        f"{parent}.{member}.weight": _info(f"{parent}.{member}.weight", "BF16", (4, 4))
        for member in members
    }


def _partial_selections() -> list[Any]:
    """組ごとに、「組の 1 人だけ」「組の 1 人を除く全員」を選んだ場合 (どちらも組の一部だけ)。"""
    cases: list[Any] = []
    for label, parent, members in FUSED_GROUP_LOCATIONS:
        seen: set[frozenset[str]] = set()
        for member in members:
            partial_selections = (
                (f"only-{member}", (member,)),
                (f"all-but-{member}", tuple(name for name in members if name != member)),
            )
            for partial_label, selected in partial_selections:
                if frozenset(selected) in seen:
                    continue
                seen.add(frozenset(selected))
                cases.append(pytest.param(parent, members, selected, id=f"{label}:{partial_label}"))
    return cases


def test_fused_groups_are_the_three_groups_of_the_issue() -> None:
    """組の定義は、Issue #74 が挙げる 3 つ (MLA の q_a と kv_a、MLP の gate と up、KDA の 6 つ)
    (C3)。
    """
    groups = {frozenset(group) for group in selection.FUSED_GROUPS}

    assert groups == {
        frozenset(MLA_FUSED_GROUP),
        frozenset(MLP_FUSED_GROUP),
        frozenset(KDA_FUSED_GROUP),
    }
    assert len(selection.FUSED_GROUPS) == 3


@pytest.mark.parametrize(("parent", "members"), FUSED_GROUP_PARAMS)
def test_validate_fused_groups_accepts_a_group_selected_whole(
    parent: str, members: Sequence[str]
) -> None:
    """組の全員が checkpoint にあり、全員が選ばれていれば通す (C3)。"""
    tensors = _group_tensors(parent, members)

    selection.validate_fused_groups([f"{parent}.{member}" for member in members], tensors)


@pytest.mark.parametrize(("parent", "members", "selected"), _partial_selections())
def test_validate_fused_groups_rejects_a_group_selected_partly(
    parent: str, members: Sequence[str], selected: Sequence[str]
) -> None:
    """組の全員が checkpoint にあるのに、一部だけが選ばれていれば断る。断る理由に、選ばれていない
    相手の名前が全部ある (C3)。

    FP8 になる線形層と BF16 のままの線形層が、vLLM の 1 つの線形層に混ざってしまう。
    """
    tensors = _group_tensors(parent, members)
    missing = [f"{parent}.{member}" for member in members if member not in selected]

    with pytest.raises(selection.SelectionError) as excinfo:
        selection.validate_fused_groups([f"{parent}.{member}" for member in selected], tensors)

    for name in missing:
        assert name in str(excinfo.value)


@pytest.mark.parametrize(("parent", "members"), FUSED_GROUP_PARAMS)
def test_validate_fused_groups_accepts_a_selected_member_whose_partners_are_not_in_the_checkpoint(
    parent: str, members: Sequence[str]
) -> None:
    """組の相手が checkpoint に `.weight` として無ければ、1 人だけが選ばれていても通す (C3)。

    相手の重みが無ければ、FP8 と BF16 が混ざることはない。第 1 段の合成 checkpoint の層 3 の
    共有の専門家 (`up_proj` だけがある) がこの形。
    """
    tensors = _group_tensors(parent, members[:1])

    selection.validate_fused_groups([f"{parent}.{members[0]}"], tensors)


def test_validate_fused_groups_does_not_pair_a_member_with_a_partner_of_another_parent() -> None:
    """組は同じ親の下だけで数える。別の層の相手が checkpoint にあっても、この層に相手が無ければ
    通す (C3)。
    """
    tensors = {
        **_group_tensors(f"{LANGUAGE_MODEL}.layers.0.mlp", ("gate_proj",)),
        **_group_tensors(f"{LANGUAGE_MODEL}.layers.1.mlp", ("up_proj",)),
    }

    selection.validate_fused_groups([f"{LANGUAGE_MODEL}.layers.0.mlp.gate_proj"], tensors)
