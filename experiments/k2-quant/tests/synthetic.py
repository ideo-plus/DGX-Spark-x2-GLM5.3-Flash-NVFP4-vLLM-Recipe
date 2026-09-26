"""合成 checkpoint (実機の形を小さくしたもの) を作る試験の補助。

`tests/test_convert.py` が使う。索引付きの 3 shard と索引外の 1 shard、実機と
同じ形の `quantization_config`、`ignore`、shard 以外の通常ファイル、飛ばす
サブディレクトリを `tmp_path` の下に作る。乱数は `manual_seed` で固定し、全 0 の
行と 1 要素だけ大きい行を含める。

- 既定の正規表現が拾う (`DEFAULT_TARGET_MODULES`。dense・共有の専門家・`lm_head`)
- 既定の正規表現が拾わない (`IGNORE_NAMES` のうち対象でないものと、専門家・ルーター・
  attention・visual・embed_tokens・norm・MTP の `eh_proj`)。`eh_proj` は、vLLM が
  quant_config を受けない plain `nn.Linear` として実装しているため、既定の対象から外れる
  負例として、テンソルは残したまま `DEFAULT_TARGET_MODULES` から除いている。

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
    "lm_head",
)
"""既定の正規表現が拾うモジュール名 (Issue #56 の第 1 段の範囲)。MTP の `eh_proj` は、vLLM が
quant_config を受けない plain `nn.Linear` として実装しているため含めない
(`vllm/models/glm5next/common/mtp.py:49`、commit 0961bbae)。"""

IGNORE_NAMES: tuple[str, ...] = (
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.0.mlp.gate_proj",
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
既定の合成 checkpoint には入れず、`build_checkpoint(include_shared_head=True)` のときだけ足す。"""

ARCHITECTURES: tuple[str, ...] = ("Glm5NextForConditionalGeneration",)


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
    """層 3〜44 の専門家 (NVFP4) の group。"""
    return {
        "format": "nvfp4-pack-quantized",
        "targets": [
            "re:.*\\.layers\\.(?:[3-9]|[1-3][0-9]|4[0-4])\\.mlp\\.experts\\..*(gate|up|down)_proj$"
        ],
        "weights": {
            "actorder": None,
            "block_structure": None,
            "dynamic": False,
            "group_size": None,
            "num_bits": 4,
            "observer": "memoryless_minmax",
            "observer_kwargs": {},
            "scale_dtype": None,
            "strategy": "tensor_group",
            "symmetric": True,
            "type": "float",
            "zp_dtype": None,
        },
        "input_activations": None,
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
) -> dict[str, Any]:
    groups: dict[str, Any] = {"group_0": _group_0(), "group_1": _group_1()}
    if config_groups_extra:
        groups.update(config_groups_extra)
    ignore = list(IGNORE_NAMES) + list(ignore_extra)
    return {
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
    root.mkdir(parents=True, exist_ok=True)
    shards = _shard_tensors(include_shared_head=include_shared_head)
    if include_shared_head:
        ignore_extra = (*ignore_extra, SHARED_HEAD_MODULE)
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_name, tensors in shards.items():
        save_file(tensors, str(root / shard_name), metadata={"format": "pt"})
        if shard_name == NON_INDEXED_SHARD:
            continue
        for tensor_name, tensor in tensors.items():
            weight_map[tensor_name] = shard_name
            total_size += tensor.numel() * tensor.element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (root / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (root / CONFIG_NAME).write_text(
        json.dumps(_config(config_groups_extra, ignore_extra), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / OTHER_FILE_NAMES[0]).write_text('{"dummy": true}\n', encoding="utf-8")
    (root / OTHER_FILE_NAMES[1]).write_text('{"dummy": true}\n', encoding="utf-8")
    cache = root / SKIPPED_DIR_NAME
    cache.mkdir(exist_ok=True)
    (cache / "junk.bin").write_bytes(b"junk")
