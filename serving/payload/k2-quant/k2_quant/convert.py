"""変換の手順 (計画 → 実行)。

`plan_conversion` は読み取りと検証だけを行い、出力を 1 バイトも書かない。
`execute_plan` が shard ごとに写す／書き直し、index・`config.json`・manifest を書く。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from k2_quant import TOOL_NAME, TOOL_VERSION
from k2_quant.fp8 import FP8_DTYPE_NAME, SCALE_DTYPE_NAME, quantize_per_channel
from k2_quant.presets import resolve_preset_pattern
from k2_quant.quant_config import add_fp8_channel_group
from k2_quant.safetensors_file import (
    DTYPE_ITEMSIZE,
    ShardEntry,
    ShardHeader,
    TensorInfo,
    iter_tensor_bytes,
    read_header,
    read_tensor,
    tensor_to_chunks,
    write_shard,
)
from k2_quant.selection import (
    WEIGHT_SUFFIX,
    module_name,
    select_modules,
    validate_fused_groups,
    validate_targets,
)

INDEX_NAME: Final = "model.safetensors.index.json"
CONFIG_NAME: Final = "config.json"
MANIFEST_NAME: Final = "manifest.json"
SHARD_SUFFIX: Final = ".safetensors"
SCALE_SUFFIX: Final = ".weight_scale"
HASH_CHUNK_SIZE: Final = 64 << 20


class ConversionError(Exception):
    """入力の前提が崩れている、または変換を続けられない。"""


@dataclass(frozen=True)
class ShardPlan:
    name: str
    header: ShardHeader
    converted: tuple[str, ...]


@dataclass(frozen=True)
class ConversionPlan:
    source: Path
    pattern: str
    modules: tuple[str, ...]
    shards: tuple[ShardPlan, ...]
    indexed_shards: frozenset[str]
    index_weight_map: Mapping[str, str]
    config: dict[str, Any]
    other_files: tuple[str, ...]
    skipped: tuple[str, ...]


def _torch_reader(tensor: torch.Tensor) -> Callable[[], Iterator[bytes]]:
    def read() -> Iterator[bytes]:
        return tensor_to_chunks(tensor)

    return read


def _original_reader(
    path: Path, info: TensorInfo, data_start: int
) -> Callable[[], Iterator[bytes]]:
    def read() -> Iterator[bytes]:
        return iter_tensor_bytes(path, info, data_start=data_start)

    return read


def plan_conversion(source: Path, *, pattern: str | None, preset: str) -> ConversionPlan:
    """入力を読み、対象を選び、検証し、新しい config を組み立てる。書き込みはしない。

    `pattern` があればそれを、無ければ `preset` を、入力の `config.json` から正規表現に解決して
    使う。計画と manifest には、解決後の正規表現を持たせる。
    """
    config_path = source / CONFIG_NAME
    index_path = source / INDEX_NAME
    if not config_path.is_file():
        raise ConversionError(f"{CONFIG_NAME} not found in {source}")
    if not index_path.is_file():
        raise ConversionError(f"{INDEX_NAME} not found in {source}")
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map_value = index.get("weight_map")
    if not isinstance(weight_map_value, dict):
        raise ConversionError(f"{INDEX_NAME}: weight_map is missing")
    index_weight_map: dict[str, str] = {
        str(key): str(value) for key, value in weight_map_value.items()
    }
    indexed_shards = frozenset(index_weight_map.values())
    for shard_name in sorted(indexed_shards):
        if not (source / shard_name).is_file():
            raise ConversionError(f"indexed shard not found: {shard_name}")
    shard_paths = sorted(path for path in source.glob(f"*{SHARD_SUFFIX}") if path.is_file())
    headers: dict[str, ShardHeader] = {path.name: read_header(path) for path in shard_paths}
    tensors: dict[str, TensorInfo] = {}
    for header in headers.values():
        tensors.update(header.tensors)
    resolved_pattern = pattern if pattern is not None else resolve_preset_pattern(preset, config)
    modules = select_modules(tensors, resolved_pattern)
    validate_targets(modules, tensors)
    validate_fused_groups(modules, tensors)
    shards = tuple(
        ShardPlan(
            name=name,
            header=headers[name],
            converted=tuple(
                module for module in modules if f"{module}{WEIGHT_SUFFIX}" in headers[name].tensors
            ),
        )
        for name in sorted(headers)
    )
    other_files: list[str] = []
    skipped: list[str] = []
    for entry in sorted(source.iterdir(), key=lambda path: path.name):
        name = entry.name
        if entry.is_dir():
            skipped.append(name)
            continue
        if name.startswith("."):
            skipped.append(name)
            continue
        if name == MANIFEST_NAME:
            raise ConversionError(f"source already contains {MANIFEST_NAME}")
        if name in (CONFIG_NAME, INDEX_NAME) or name.endswith(SHARD_SUFFIX):
            continue
        other_files.append(name)
    return ConversionPlan(
        source=source,
        pattern=resolved_pattern,
        modules=tuple(modules),
        shards=shards,
        indexed_shards=indexed_shards,
        index_weight_map=index_weight_map,
        config=add_fp8_channel_group(config, modules=modules),
        other_files=tuple(other_files),
        skipped=tuple(skipped),
    )


def execute_plan(
    plan: ConversionPlan,
    output: Path,
    *,
    link: bool,
    source_repo: str,
    source_revision: str,
) -> None:
    """計画に従って出力ディレクトリを一度だけ作る。

    出力先は、存在しないパスか、存在する空のディレクトリ (`docker run --mount` の宛先は
    コンテナの中に必ず存在する)。中身があれば何も書かずに断る。
    """
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ConversionError(f"output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for shard in plan.shards:
        source = plan.source / shard.name
        destination = output / shard.name
        if shard.converted:
            _rewrite_shard(source, destination, shard)
        else:
            _copy_or_link(source, destination, link=link)
    for name in plan.other_files:
        _copy_or_link(plan.source / name, output / name, link=link)
    _write_index(plan, output)
    _write_config(plan, output)
    _write_manifest(plan, output, source_repo=source_repo, source_revision=source_revision)


def _copy_or_link(source: Path, destination: Path, *, link: bool) -> None:
    if link:
        os.link(source, destination)
    else:
        shutil.copyfile(source, destination)


def _rewrite_shard(source: Path, destination: Path, shard: ShardPlan) -> None:
    converted = set(shard.converted)
    data_start = shard.header.data_start
    entries: list[ShardEntry] = []
    for name, info in shard.header.tensors.items():
        module = module_name(name)
        if module is not None and module in converted:
            weight = read_tensor(source, info, data_start=data_start)
            quantized, scale = quantize_per_channel(weight)
            entries.append(
                ShardEntry(
                    name=f"{module}{WEIGHT_SUFFIX}",
                    dtype=FP8_DTYPE_NAME,
                    shape=tuple(int(dim) for dim in quantized.shape),
                    nbytes=quantized.numel() * quantized.element_size(),
                    chunks=_torch_reader(quantized),
                )
            )
            entries.append(
                ShardEntry(
                    name=f"{module}{SCALE_SUFFIX}",
                    dtype=SCALE_DTYPE_NAME,
                    shape=tuple(int(dim) for dim in scale.shape),
                    nbytes=scale.numel() * scale.element_size(),
                    chunks=_torch_reader(scale),
                )
            )
        else:
            entries.append(
                ShardEntry(
                    name=name,
                    dtype=info.dtype,
                    shape=info.shape,
                    nbytes=info.nbytes,
                    chunks=_original_reader(source, info, data_start),
                )
            )
    entries.sort(key=lambda entry: (-DTYPE_ITEMSIZE[entry.dtype], entry.name))
    write_shard(destination, entries, shard.header.metadata)


def _write_index(plan: ConversionPlan, output: Path) -> None:
    weight_map = dict(plan.index_weight_map)
    for shard in plan.shards:
        for module in shard.converted:
            weight_name = f"{module}{WEIGHT_SUFFIX}"
            if weight_name in weight_map:
                weight_map[f"{module}{SCALE_SUFFIX}"] = weight_map[weight_name]
    total_size = 0
    for shard_name in sorted(set(weight_map.values())):
        header = read_header(output / shard_name)
        total_size += sum(info.nbytes for info in header.tensors.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output / INDEX_NAME).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_config(plan: ConversionPlan, output: Path) -> None:
    (output / CONFIG_NAME).write_text(
        json.dumps(plan.config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_manifest(
    plan: ConversionPlan,
    output: Path,
    *,
    source_repo: str,
    source_revision: str,
) -> None:
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        size = path.stat().st_size
        files.append({"path": path.name, "size": size, "sha256": sha256_of(path)})
        total_bytes += size
    manifest: dict[str, Any] = {
        "conversion": {
            "tool": TOOL_NAME,
            "tool_version": TOOL_VERSION,
            "source": {"repo": source_repo, "revision": source_revision},
            "pattern": plan.pattern,
            # 変換の結果を決める引数だけを、`source`・`pattern` と同じ入力から作って記録する。
            # `--source` と `--output` は、マウントの位置に依存し、`--link` は出力のバイト列を
            # 変えないので、記録しない
            "args": [
                "--source-repo",
                source_repo,
                "--source-revision",
                source_revision,
                "--pattern",
                plan.pattern,
            ],
            "modules": list(plan.modules),
            "weight_dtype": FP8_DTYPE_NAME,
            "scale_dtype": SCALE_DTYPE_NAME,
            "strategy": "channel",
        },
        "files": files,
        "total_bytes": total_bytes,
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256_of(path: Path) -> str:
    """ファイルの SHA-256 を chunk で計算する。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
