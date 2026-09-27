"""合成 checkpoint (実機の形を小さくしたもの) を作る試験の補助。

`tests/test_convert.py` が使う。索引付きの 3 shard と索引外の 1 shard、実機と
同じ形の `quantization_config`、`ignore`、shard 以外の通常ファイル、飛ばす
サブディレクトリを `tmp_path` の下に作る。乱数は `manual_seed` で固定し、全 0 の
行と 1 要素だけ大きい行を含める。

- 既定の正規表現が拾う (`DEFAULT_TARGET_MODULES`。dense・共有の専門家)
- 既定の正規表現が拾わない (`IGNORE_NAMES` のうち対象でないものと、専門家・ルーター・
  attention・visual・embed_tokens・norm・`lm_head`・MTP の `eh_proj` と `shared_head.head`)。
  `eh_proj` は、vLLM が quant_config を受けない plain `nn.Linear` として実装しているため、
  `lm_head` と `shared_head.head` は、FP8 (W8A16) の `ParallelLMHead` を humming の線形
  カーネルで読めない (#68) ため、既定の対象から外れる負例として、テンソルは残したまま
  `DEFAULT_TARGET_MODULES` から除いている。

第 2a 段の合成 checkpoint (`build_stage2_checkpoint`) は、KDA の層 0 と MLA の層 3、`lm_head`、
`layer_types` に載らない層 45 (MTP) を持ち、`config.json` に `text_config.layer_types` を持つ。
同じ字面 (`o_proj`、`gate_proj`) が、層種や親の名前によって対象・非対象に分かれる。KDA の gate
(`f_a_proj`・`f_b_proj`) は、実機と同じく、テンソル名が `self_attn.f_a_proj.weight` の形
(`forget_gate.` なし) で、`ignore` の名前だけが `self_attn.forget_gate.f_a_proj` の形になる (#76)。

第 2b 段 (`--preset k2s2b`。#79) は、同じ合成 checkpoint の層 0 の KDA のまとめた層
(`KDA_MERGED_MODULES`) を第 2a 段の対象に足して選ぶ (`STAGE2B_TARGET_MODULES`)。`f_a_proj` は、
`ignore` の名前が `self_attn.forget_gate.f_a_proj` の形のままなので、変換しても `ignore` に残る。

書き込みは `safetensors.torch.save_file` (参照実装) で行う。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

SHARD_NAMES: tuple[str, ...] = (
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
)
"""索引に載る shard の名前。"""

NON_INDEXED_SHARD: str = "model_mtp.safetensors"
"""索引に載らない shard の名前 (実機の `model_mtp.safetensors` に当たる)。"""

INDEX_NAME: str = "model.safetensors.index.json"
CONFIG_NAME: str = "config.json"
MANIFEST_NAME: str = "manifest.json"
OTHER_FILE_NAMES: tuple[str, ...] = ("tokenizer.json", "generation_config.json")
"""`config.json`・index・shard 以外の通常ファイル。"""

SKIPPED_DIR_NAME: str = ".cache"
"""写さず標準出力に列挙するサブディレクトリ。"""

DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.3.mlp.shared_experts.up_proj",
    "model.language_model.layers.45.mlp.shared_experts.down_proj",
    "model.language_model.layers.2.mlp.down_proj",
)
"""既定の正規表現が拾うモジュール名 (dense と共有の専門家)。MTP の `eh_proj` は、vLLM が
quant_config を受けない plain `nn.Linear` として実装しているため含めない
(`vllm/models/glm5next/common/mtp.py:49`、commit 0961bbae)。`lm_head` と `shared_head.head` は、
FP8 の `ParallelLMHead` を読めない (#68) ため含めない。"""

IGNORE_NAMES: tuple[str, ...] = (
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.2.mlp.down_proj",
    "model.language_model.layers.3.mlp.gate",
    "model.language_model.layers.3.mlp.shared_experts.up_proj",
    "model.language_model.layers.7.mlp.shared_experts.down_proj",
    "model.language_model.layers.45.eh_proj",
    "model.language_model.layers.45.mlp.shared_experts.down_proj",
    "model.visual.blocks.0.mlp.gate_proj",
    "lm_head",
)
"""入力 `config.json` の `ignore` に置く名前 (変換対象を取り除く前)。"""

SHARED_HEAD_MODULE: str = "model.language_model.layers.45.shared_head.head"
"""MTP の head (`shared_head.head`) のモジュール名。実機の index にあるかは未確認なので、
既定の合成 checkpoint には入れず、`build_checkpoint(include_shared_head=True)` のときだけ足す。
既定では変換しない (#68) 負例として、`--pattern` で明示したときだけ変換する正例として使う。"""

ARCHITECTURES: tuple[str, ...] = ("Glm5NextForConditionalGeneration",)

KDA_LAYER_TYPE: str = "linear_attention"
MLA_LAYER_TYPE: str = "deepseek_sparse_attention"
"""実機 `config.json` の `text_config.layer_types` の値 (`serving/config/configs.toml` の
`probe-pinned` の注記が、先頭 4 層をそのまま取ったものとして示している)。"""

STAGE2_LAYER_TYPES: tuple[str, ...] = (
    KDA_LAYER_TYPE,
    KDA_LAYER_TYPE,
    KDA_LAYER_TYPE,
    MLA_LAYER_TYPE,
)
"""第 2a 段の合成 checkpoint の層種の並び。層 0〜2 が KDA、層 3 が MLA。層 45 (MTP) は
並びに載らない。"""

_STAGE2_PREFIX: str = "model.language_model"
_KDA_LAYER: str = f"{_STAGE2_PREFIX}.layers.0"
_MLA_LAYER: str = f"{_STAGE2_PREFIX}.layers.3"
_MTP_LAYER: str = f"{_STAGE2_PREFIX}.layers.45"

STAGE2A_TARGET_MODULES: tuple[str, ...] = (
    # 第 2a 段の MLA の射影 (層 3)
    f"{_MLA_LAYER}.self_attn.q_a_proj",
    f"{_MLA_LAYER}.self_attn.kv_a_proj_with_mqa",
    f"{_MLA_LAYER}.self_attn.q_b_proj",
    f"{_MLA_LAYER}.self_attn.o_proj",
    # 第 2a 段の KDA のまとめていない射影 (層 0)
    f"{_KDA_LAYER}.self_attn.o_proj",
    f"{_KDA_LAYER}.self_attn.f_b_proj",
    f"{_KDA_LAYER}.self_attn.g_b_proj",
    # lm_head
    "lm_head",
    # 第 1 段の既定の対象 (dense と共有の専門家)
    f"{_KDA_LAYER}.mlp.gate_proj",
    f"{_KDA_LAYER}.mlp.up_proj",
    f"{_KDA_LAYER}.mlp.down_proj",
    f"{_MLA_LAYER}.mlp.shared_experts.gate_proj",
    f"{_MLA_LAYER}.mlp.shared_experts.up_proj",
    f"{_MLA_LAYER}.mlp.shared_experts.down_proj",
    f"{_MTP_LAYER}.mlp.shared_experts.down_proj",
)
"""第 2a 段の選び方が拾うモジュール名 (15 個)。"""

KDA_MERGED_MODULES: tuple[str, ...] = (
    f"{_KDA_LAYER}.self_attn.q_proj",
    f"{_KDA_LAYER}.self_attn.k_proj",
    f"{_KDA_LAYER}.self_attn.v_proj",
    f"{_KDA_LAYER}.self_attn.b_proj",
    f"{_KDA_LAYER}.self_attn.f_a_proj",
    f"{_KDA_LAYER}.self_attn.g_a_proj",
)
"""KDA のまとめた層 (層 0)。vLLM で 1 つの線形層 (`in_proj_qkvbfg_a`) にまとまる
q・k・v・b・f_a・g_a。第 2a 段では選ばれず、第 2b 段 (`--preset k2s2b`。#79) で組ごと選ばれる。"""

STAGE2_UNTOUCHED_MODULES: tuple[str, ...] = (
    # KDA のまとめた層 (vLLM で 1 つの線形層にまとまる q・k・v・b・f_a・g_a)
    *KDA_MERGED_MODULES,
    # MLA の kv_b_proj と indexer
    f"{_MLA_LAYER}.self_attn.kv_b_proj",
    f"{_MLA_LAYER}.self_attn.indexer.wq_b",
    # `layer_types` に載らない層 (MTP の層 45) の、KDA・MLA どちらの射影とも同じ名前の形
    f"{_MTP_LAYER}.self_attn.o_proj",
    f"{_MTP_LAYER}.eh_proj",
    f"{_STAGE2_PREFIX}.embed_tokens",
    f"{_STAGE2_PREFIX}.norm",
)
"""第 2a 段の選び方が拾わないモジュール名。同じ字面 (`o_proj`) が層種によって対象・非対象に
分かれるので、層 45 の `o_proj` を含める。"""

STAGE2B_TARGET_MODULES: tuple[str, ...] = (*STAGE2A_TARGET_MODULES, *KDA_MERGED_MODULES)
"""第 2b 段の選び方が拾うモジュール名 (21 個)。第 2a 段の 15 個と、KDA のまとめた層の 6 個。"""

STAGE2B_UNTOUCHED_MODULES: tuple[str, ...] = tuple(
    module for module in STAGE2_UNTOUCHED_MODULES if module not in KDA_MERGED_MODULES
)
"""第 2b 段の選び方が拾わないモジュール名。第 2a 段の拾わない名前から、KDA のまとめた層を除いたもの
(`kv_b_proj`・indexer・`layer_types` に載らない層 45・`embed_tokens`・`norm`・`eh_proj`)。"""

STAGE3_SHARED_SCALE_GROUPS: tuple[tuple[str, ...], ...] = (
    KDA_MERGED_MODULES,
    (f"{_MLA_LAYER}.self_attn.q_a_proj", f"{_MLA_LAYER}.self_attn.kv_a_proj_with_mqa"),
    (f"{_KDA_LAYER}.mlp.gate_proj", f"{_KDA_LAYER}.mlp.up_proj"),
    (f"{_MLA_LAYER}.mlp.shared_experts.gate_proj", f"{_MLA_LAYER}.mlp.shared_experts.up_proj"),
)
"""第 3 段 (`--preset k2s3`) の対象 (`STAGE2B_TARGET_MODULES`) のうち、`selection.FUSED_GROUPS` の
組に丸ごと当たる 4 組 (KDA のまとめた層、MLA の `q_a_proj`+`kv_a_proj_with_mqa`、層 0 dense の
`gate_proj`+`up_proj`、層 3 shared の `gate_proj`+`up_proj`)。組の全員は、`weight_global_scale`
を同じ値で共有する (#95)。"""

STAGE3_SINGLETON_SCALE_MODULES: tuple[str, ...] = tuple(
    module
    for module in STAGE2B_TARGET_MODULES
    if not any(module in group for group in STAGE3_SHARED_SCALE_GROUPS)
)
"""第 3 段の対象のうち、`STAGE3_SHARED_SCALE_GROUPS` のどの組にも属さない、単独で
`weight_global_scale` を持つモジュール (9 個)。"""

STAGE2_KDA_GATE_IGNORE_NAMES: Mapping[str, str] = {
    f"{_KDA_LAYER}.self_attn.f_a_proj": f"{_KDA_LAYER}.self_attn.forget_gate.f_a_proj",
    f"{_KDA_LAYER}.self_attn.f_b_proj": f"{_KDA_LAYER}.self_attn.forget_gate.f_b_proj",
}
"""実機の `config.json` の `ignore` は、KDA の gate を `self_attn.forget_gate.f_a_proj` の形で持ち、
テンソル名 (`self_attn.f_a_proj.weight`) と違う (#76)。テンソル名の形のモジュール名から、`ignore` に
置く名前へ写す。"""


def stage2_ignore_name(module: str) -> str:
    """第 2a 段の合成 checkpoint の `ignore` に置く名前。KDA の gate だけ `forget_gate.` 付き。"""
    return STAGE2_KDA_GATE_IGNORE_NAMES.get(module, module)


STAGE2_IGNORE_NAMES: tuple[str, ...] = tuple(
    stage2_ignore_name(module) for module in STAGE2A_TARGET_MODULES + STAGE2_UNTOUCHED_MODULES
)
"""第 2a 段の合成 checkpoint の `config.json` の `ignore` に置く名前 (変換の前は、BF16 の
モジュールがすべて `ignore` にある)。KDA の gate だけ、テンソル名と違う形になる。"""

STAGE2_EXPERT_MODULE: str = f"{_MLA_LAYER}.mlp.experts.0.gate_proj"
"""層 3 の専門家 (NVFP4)。`weight_packed`・`weight_scale`・`weight_global_scale` だけを持ち、
`.weight` が無いので、選び方の候補にならない。"""


def _matrix(rows: int, cols: int, *, seed: int) -> torch.Tensor:
    """全 0 の行と、1 要素だけ 100 倍大きい行を含む BF16 の行列。"""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=generator)
    weight[0] = 0.0
    if rows >= 2:
        weight[1, -1] = float(weight[1].abs().max()) * 100.0 + 1.0
    return weight.to(torch.bfloat16)


def _vector(length: int, *, seed: int) -> torch.Tensor:
    """BF16 の 1 次元テンソル (`norm.weight` に当たる)。"""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(length, generator=generator).to(torch.bfloat16)


def _group_0() -> dict[str, Any]:
    """層 3〜44 の専門家 (NVFP4、W4A4) の group。

    `format`・`weights`・`input_activations` は、2026-09-27 に HF の rev `18d55bfd…` の
    `config.json` から逐語で確認した値 (Issue #95 の計画「参照資料の調査結果」)。
    """
    return {
        "format": "nvfp4-pack-quantized",
        "targets": [
            "re:.*\\.layers\\.(?:[3-9]|[1-3][0-9]|4[0-4])\\.mlp\\.experts\\..*(gate|up|down)_proj$"
        ],
        "weights": {
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
        },
        "input_activations": {
            "actorder": None,
            "block_structure": None,
            "dynamic": "local",
            "group_size": 16,
            "num_bits": 4,
            "observer": "static_minmax",
            "observer_kwargs": {},
            "scale_dtype": "torch.float8_e4m3fn",
            "strategy": "tensor_group",
            "symmetric": True,
            "type": "float",
            "zp_dtype": None,
        },
        "output_activations": None,
    }


def _group_1() -> dict[str, Any]:
    """層 45 (MTP) の専門家 (FP8) の group。"""
    return {
        "format": "float-quantized",
        "targets": ["re:.*\\.layers\\.45\\.mlp\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$"],
        "weights": {
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
        },
        "input_activations": {
            "actorder": None,
            "block_structure": None,
            "dynamic": True,
            "group_size": 128,
            "num_bits": 8,
            "observer": None,
            "observer_kwargs": {},
            "scale_dtype": "torch.float32",
            "strategy": "group",
            "symmetric": True,
            "type": "float",
            "zp_dtype": None,
        },
        "output_activations": None,
    }


def _config(
    config_groups_extra: Mapping[str, Any] | None,
    ignore_extra: Sequence[str],
    *,
    ignore_base: Sequence[str] = IGNORE_NAMES,
    layer_types: Sequence[str] | None = None,
) -> dict[str, Any]:
    groups: dict[str, Any] = {"group_0": _group_0(), "group_1": _group_1()}
    if config_groups_extra:
        groups.update(config_groups_extra)
    ignore = list(ignore_base) + list(ignore_extra)
    config: dict[str, Any] = {
        "architectures": list(ARCHITECTURES),
        "model_type": "glm5next",
        "quantization_config": {
            "config_groups": groups,
            "format": "mixed-precision",
            "ignore": ignore,
            "kv_cache_scheme": None,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "sparsity_config": {},
            "transform_config": {},
            "version": "0.17.2.dev32+g46196ce",
        },
    }
    if layer_types is not None:
        config["text_config"] = {"layer_types": list(layer_types)}
    return config


def _shard_tensors(*, include_shared_head: bool) -> dict[str, dict[str, torch.Tensor]]:
    language_model = "model.language_model"
    layer0 = f"{language_model}.layers.0"
    layer3 = f"{language_model}.layers.3"
    expert3 = f"{layer3}.mlp.experts.0.gate_proj"
    layer45 = f"{language_model}.layers.45"
    shards = {
        SHARD_NAMES[0]: {
            f"{layer0}.mlp.gate_proj.weight": _matrix(8, 4, seed=1),
            f"{layer0}.self_attn.q_proj.weight": _matrix(4, 4, seed=2),
        },
        SHARD_NAMES[1]: {
            f"{layer3}.mlp.shared_experts.up_proj.weight": _matrix(6, 4, seed=3),
            f"{expert3}.weight_packed": torch.arange(16, dtype=torch.uint8).reshape(4, 4),
            f"{expert3}.weight_scale": torch.ones(4, 1).to(torch.float8_e4m3fn),
            f"{expert3}.weight_global_scale": torch.tensor([1.0], dtype=torch.float32),
            f"{layer3}.mlp.gate.weight": _matrix(2, 4, seed=5),
        },
        SHARD_NAMES[2]: {
            # 既定で変換しない `lm_head` (#68) だけでは、この shard に既定の対象が 1 つも
            # 無くなる。書き直しの経路を通り、書き直した shard の中で `lm_head` が保たれることを
            # 確かめるため、dense の層 2 の対象を足す
            f"{language_model}.layers.2.mlp.down_proj.weight": _matrix(4, 8, seed=12),
            "lm_head.weight": _matrix(16, 4, seed=6),
            f"{language_model}.embed_tokens.weight": _matrix(16, 4, seed=7),
            f"{language_model}.norm.weight": _vector(4, seed=8),
            f"{layer45}.eh_proj.weight": _matrix(4, 8, seed=9),
        },
        NON_INDEXED_SHARD: {
            f"{layer45}.mlp.shared_experts.down_proj.weight": _matrix(4, 6, seed=10),
        },
    }
    if include_shared_head:
        shards[SHARD_NAMES[2]][f"{SHARED_HEAD_MODULE}.weight"] = _matrix(16, 4, seed=11)
    return shards


def _stage2_shard_tensors() -> dict[str, dict[str, torch.Tensor]]:
    """第 2a 段の合成 checkpoint のテンソル。KDA の層 0 と MLA の層 3、`lm_head`、
    `layer_types` に載らない層 45 を持つ。形は小さい BF16 の 2 次元 (1 次元は `norm` だけ)。

    列数 (入力の次元) は、すべて 16 の倍数にする (`--preset k2s3` の NVFP4A16 は、対象の入力の
    次元が 16 の倍数であることを要求する。#95)。`norm` だけ 1 次元のまま。専門家の
    `weight_packed` 等の形は変えない。`_matrix` の性質 (全 0 の行、1 要素だけ 100 倍) は維持する。
    """
    kda = f"{_KDA_LAYER}.self_attn"
    mla = f"{_MLA_LAYER}.self_attn"
    shared = f"{_MLA_LAYER}.mlp.shared_experts"
    mtp = _MTP_LAYER
    return {
        SHARD_NAMES[0]: {
            f"{kda}.q_proj.weight": _matrix(4, 32, seed=21),
            f"{kda}.k_proj.weight": _matrix(4, 32, seed=22),
            f"{kda}.v_proj.weight": _matrix(4, 32, seed=23),
            f"{kda}.b_proj.weight": _matrix(2, 32, seed=24),
            f"{kda}.f_a_proj.weight": _matrix(2, 32, seed=25),
            f"{kda}.f_b_proj.weight": _matrix(4, 16, seed=26),
            f"{kda}.g_a_proj.weight": _matrix(2, 32, seed=27),
            f"{kda}.g_b_proj.weight": _matrix(4, 16, seed=28),
            f"{kda}.o_proj.weight": _matrix(4, 32, seed=29),
            f"{_KDA_LAYER}.mlp.gate_proj.weight": _matrix(8, 32, seed=30),
            f"{_KDA_LAYER}.mlp.up_proj.weight": _matrix(8, 32, seed=31),
            f"{_KDA_LAYER}.mlp.down_proj.weight": _matrix(4, 32, seed=32),
        },
        SHARD_NAMES[1]: {
            f"{mla}.q_a_proj.weight": _matrix(4, 32, seed=33),
            f"{mla}.kv_a_proj_with_mqa.weight": _matrix(6, 32, seed=34),
            f"{mla}.q_b_proj.weight": _matrix(4, 32, seed=35),
            f"{mla}.kv_b_proj.weight": _matrix(6, 32, seed=36),
            f"{mla}.o_proj.weight": _matrix(4, 32, seed=37),
            f"{mla}.indexer.wq_b.weight": _matrix(4, 32, seed=38),
            f"{shared}.gate_proj.weight": _matrix(6, 32, seed=39),
            f"{shared}.up_proj.weight": _matrix(6, 32, seed=40),
            f"{shared}.down_proj.weight": _matrix(4, 48, seed=41),
            f"{STAGE2_EXPERT_MODULE}.weight_packed": torch.arange(16, dtype=torch.uint8).reshape(
                4, 4
            ),
            f"{STAGE2_EXPERT_MODULE}.weight_scale": torch.ones(4, 1).to(torch.float8_e4m3fn),
            f"{STAGE2_EXPERT_MODULE}.weight_global_scale": torch.tensor([1.0], dtype=torch.float32),
        },
        SHARD_NAMES[2]: {
            "lm_head.weight": _matrix(16, 32, seed=42),
            f"{_STAGE2_PREFIX}.embed_tokens.weight": _matrix(16, 32, seed=43),
            f"{_STAGE2_PREFIX}.norm.weight": _vector(4, seed=44),
            f"{mtp}.eh_proj.weight": _matrix(4, 16, seed=45),
        },
        NON_INDEXED_SHARD: {
            f"{mtp}.self_attn.o_proj.weight": _matrix(4, 32, seed=46),
            f"{mtp}.mlp.shared_experts.down_proj.weight": _matrix(4, 48, seed=47),
        },
    }


def _write_checkpoint(
    root: Path,
    shards: Mapping[str, Mapping[str, torch.Tensor]],
    config: Mapping[str, Any],
) -> None:
    """shard・index (索引外の shard は載せない)・`config.json`・通常ファイル・飛ばす
    サブディレクトリを `root` に書く。"""
    root.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_name, tensors in shards.items():
        save_file(dict(tensors), str(root / shard_name), metadata={"format": "pt"})
        if shard_name == NON_INDEXED_SHARD:
            continue
        for tensor_name, tensor in tensors.items():
            weight_map[tensor_name] = shard_name
            total_size += tensor.numel() * tensor.element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (root / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (root / CONFIG_NAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (root / OTHER_FILE_NAMES[0]).write_text('{"dummy": true}\n', encoding="utf-8")
    (root / OTHER_FILE_NAMES[1]).write_text('{"dummy": true}\n', encoding="utf-8")
    cache = root / SKIPPED_DIR_NAME
    cache.mkdir(exist_ok=True)
    (cache / "junk.bin").write_bytes(b"junk")


def build_checkpoint(
    root: Path,
    *,
    config_groups_extra: Mapping[str, Any] | None = None,
    ignore_extra: Sequence[str] = (),
    include_shared_head: bool = False,
) -> None:
    """合成 checkpoint を `root` に作る (既存なら上書き)。

    `config_groups_extra` は `config_groups` に加える group (`group_2` の衝突試験に使う)。
    `ignore_extra` は `ignore` に足す名前。`include_shared_head` が真なら、MTP の head
    (`SHARED_HEAD_MODULE`) の重みを足し、その名前を `ignore` にも置く (実機の `ignore` と
    同じく、変換の前は `ignore` にある)。
    """
    shards = _shard_tensors(include_shared_head=include_shared_head)
    if include_shared_head:
        ignore_extra = (*ignore_extra, SHARED_HEAD_MODULE)
    _write_checkpoint(root, shards, _config(config_groups_extra, ignore_extra))


def build_stage2_checkpoint(root: Path) -> None:
    """第 2a 段の合成 checkpoint を `root` に作る (既存なら上書き)。

    KDA の層 0 (9 つの射影と dense の MLP)、MLA の層 3 (5 つの射影、indexer、共有の専門家、
    NVFP4 の専門家)、`lm_head`、`layer_types` に載らない層 45 (`o_proj`、`eh_proj`、共有の
    専門家) を持つ。`config.json` は `text_config.layer_types` に `STAGE2_LAYER_TYPES` を持ち、
    変換の前の BF16 のモジュール (`STAGE2_IGNORE_NAMES`) をすべて `ignore` に置く。実機と同じく、
    KDA の gate は、テンソル名が `self_attn.f_a_proj.weight` の形で、`ignore` の名前だけが
    `self_attn.forget_gate.f_a_proj` の形になる (#76)。
    """
    config = _config(None, (), ignore_base=STAGE2_IGNORE_NAMES, layer_types=STAGE2_LAYER_TYPES)
    _write_checkpoint(root, _stage2_shard_tensors(), config)
