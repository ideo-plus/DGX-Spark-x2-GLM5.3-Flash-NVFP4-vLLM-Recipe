"""safetensors の形式を自前で読み書きする。

公開仕様 (8 byte LE のヘッダ長 + JSON ヘッダ + バッファ) に従う。`safetensors` は
読み込まない。対象外テンソルをバイト範囲のまま chunk で写せること、ヘッダの JSON を
`sort_keys` で固定して同じ入力から同じ出力を作れることを狙う。
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch

DTYPE_ITEMSIZE: Final[Mapping[str, int]] = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

TORCH_DTYPES: Final[Mapping[str, torch.dtype]] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}

READ_CHUNK_SIZE: Final = 64 << 20


@dataclass(frozen=True)
class TensorInfo:
    """1 つのテンソルのヘッダの情報。`data_offsets` はバッファ先頭からの相対位置。"""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


@dataclass(frozen=True)
class ShardHeader:
    tensors: Mapping[str, TensorInfo]
    metadata: Mapping[str, str] | None
    data_start: int


@dataclass(frozen=True)
class ShardEntry:
    """書き出す 1 つのテンソル。`chunks` は呼ぶたびにバイト列を順に返す。"""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    chunks: Callable[[], Iterator[bytes]]


def _itemsize(path: Path, name: str, dtype: str) -> int:
    itemsize = DTYPE_ITEMSIZE.get(dtype)
    if itemsize is None:
        raise ValueError(f"{path}: {name}: unknown dtype {dtype}")
    return itemsize


def read_header(path: Path) -> ShardHeader:
    """ヘッダを読み、`__metadata__` と各テンソルの情報を返す。"""
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"{path}: too short for a safetensors header")
        header_length = struct.unpack("<Q", length_bytes)[0]
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError(f"{path}: truncated header")
    parsed = json.loads(header_bytes)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: header is not a JSON object")
    metadata_value = parsed.pop("__metadata__", None)
    metadata: Mapping[str, str] | None = None
    if metadata_value is not None:
        if not isinstance(metadata_value, dict):
            raise ValueError(f"{path}: __metadata__ is not a JSON object")
        metadata = {str(key): str(value) for key, value in metadata_value.items()}
    data_start = 8 + header_length
    buffer_size = file_size - data_start
    if buffer_size < 0:
        raise ValueError(f"{path}: header is larger than the file")
    tensors: dict[str, TensorInfo] = {}
    for name, value in parsed.items():
        if not isinstance(value, dict):
            raise ValueError(f"{path}: {name}: entry is not a JSON object")
        dtype = str(value["dtype"])
        shape = tuple(int(dim) for dim in value["shape"])
        offsets = value["data_offsets"]
        start = int(offsets[0])
        end = int(offsets[1])
        if start < 0 or end < start or end > buffer_size:
            raise ValueError(f"{path}: {name}: data_offsets out of range: {start}, {end}")
        itemsize = _itemsize(path, str(name), dtype)
        expected = itemsize
        for dim in shape:
            expected *= dim
        if end - start != expected:
            raise ValueError(f"{path}: {name}: size {end - start} != {expected}")
        tensors[str(name)] = TensorInfo(
            name=str(name), dtype=dtype, shape=shape, data_offsets=(start, end)
        )
    return ShardHeader(tensors=tensors, metadata=metadata, data_start=data_start)


def _iter_bytes(path: Path, offset: int, size: int, chunk_size: int) -> Iterator[bytes]:
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = size
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                raise ValueError(f"{path}: unexpected end of file")
            remaining -= len(chunk)
            yield chunk


def iter_tensor_bytes(
    path: Path,
    info: TensorInfo,
    *,
    chunk_size: int = READ_CHUNK_SIZE,
    data_start: int | None = None,
) -> Iterator[bytes]:
    """テンソルのバイト列を、バッファの範囲のまま chunk で返す。

    `data_start` を渡せばヘッダを読み直さない。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if data_start is None:
        data_start = read_header(path).data_start
    return _iter_bytes(path, data_start + info.data_offsets[0], info.nbytes, chunk_size)


def read_tensor(path: Path, info: TensorInfo, *, data_start: int | None = None) -> torch.Tensor:
    """BF16/F16/F32 のテンソルを値まで読む。`data_start` を渡せばヘッダを読み直さない。"""
    dtype = TORCH_DTYPES.get(info.dtype)
    if dtype is None:
        raise ValueError(f"{path}: unsupported dtype to read: {info.dtype}")
    if data_start is None:
        data_start = read_header(path).data_start
    raw = bytearray()
    offset = data_start + info.data_offsets[0]
    for chunk in _iter_bytes(path, offset, info.nbytes, READ_CHUNK_SIZE):
        raw.extend(chunk)
    return torch.frombuffer(raw, dtype=dtype).reshape(info.shape)


def tensor_to_chunks(tensor: torch.Tensor) -> Iterator[bytes]:
    """テンソルの生のバイト列を返す (F8 も BF16 も uint8 として見る)。"""
    yield tensor.contiguous().view(torch.uint8).numpy().tobytes()


def write_shard(
    path: Path, entries: Sequence[ShardEntry], metadata: Mapping[str, str] | None
) -> None:
    """entries の並びのまま、offsets を 0 から連続に振って 1 つの shard を書く。"""
    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = dict(metadata)
    offset = 0
    for entry in entries:
        itemsize = _itemsize(path, entry.name, entry.dtype)
        expected = itemsize
        for dim in entry.shape:
            expected *= dim
        if entry.nbytes != expected:
            raise ValueError(f"{path}: {entry.name}: nbytes {entry.nbytes} != {expected}")
        header[entry.name] = {
            "dtype": entry.dtype,
            "shape": list(entry.shape),
            "data_offsets": [offset, offset + entry.nbytes],
        }
        offset += entry.nbytes
    header_bytes = json.dumps(
        header, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for entry in entries:
            written = 0
            for chunk in entry.chunks():
                handle.write(chunk)
                written += len(chunk)
            if written != entry.nbytes:
                raise ValueError(
                    f"{path}: {entry.name}: wrote {written} bytes, expected {entry.nbytes}"
                )
