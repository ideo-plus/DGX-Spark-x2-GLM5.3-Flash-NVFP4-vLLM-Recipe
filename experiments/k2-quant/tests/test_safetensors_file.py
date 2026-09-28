"""safetensors の読み書き (形式) の試験 (C4)。

参照実装 (`safetensors.torch`) で書いた fixture を自前の `read_header` で読めること、
自前の `write_shard` の出力を参照実装で読めること、offsets が 0 始まりで連続して
いること、`tensor_to_chunks` がバイト列を往復することを確かめる。ネットワークも
GPU も使わない。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from k2_quant import safetensors_file as sf

DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.uint8: "U8",
}


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _fixture_tensors() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "layer.weight": torch.randn(3, 4, generator=generator).to(torch.bfloat16),
        "layer.scale": torch.rand(2, 1, generator=generator).to(torch.float32),
        "layer.fp8": (torch.rand(3, generator=generator) - 0.5).to(torch.float8_e4m3fn),
        "layer.packed": torch.arange(5, dtype=torch.uint8),
    }


def _write_fixture(path: Path) -> dict[str, torch.Tensor]:
    tensors = _fixture_tensors()
    save_file(tensors, str(path), metadata={"format": "pt", "custom": "x"})
    return tensors


def _entry(name: str, tensor: torch.Tensor) -> sf.ShardEntry:
    def chunks() -> Iterator[bytes]:
        return sf.tensor_to_chunks(tensor)

    return sf.ShardEntry(
        name=name,
        dtype=DTYPE_NAMES[tensor.dtype],
        shape=tuple(tensor.shape),
        nbytes=tensor.numel() * tensor.element_size(),
        chunks=chunks,
    )


def _header_offsets(path: Path) -> tuple[int, list[tuple[int, int]]]:
    raw = path.read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_len])
    offsets = [
        tuple(value["data_offsets"]) for key, value in header.items() if key != "__metadata__"
    ]
    return 8 + header_len, sorted((int(start), int(end)) for start, end in offsets)


def test_read_header_reports_dtype_shape_and_metadata(tmp_path: Path) -> None:
    """参照実装で書いた shard のヘッダを読める (C4)。"""
    path = tmp_path / "model.safetensors"
    tensors = _write_fixture(path)

    header = sf.read_header(path)

    assert set(header.tensors) == set(tensors)
    assert header.metadata == {"format": "pt", "custom": "x"}
    for name, tensor in tensors.items():
        info = header.tensors[name]
        assert info.dtype == DTYPE_NAMES[tensor.dtype]
        assert info.shape == tuple(tensor.shape)
        assert info.nbytes == tensor.numel() * tensor.element_size()


def test_read_header_data_start_points_after_header(tmp_path: Path) -> None:
    """`data_start` はヘッダの直後 (C4)。"""
    path = tmp_path / "model.safetensors"
    save_file(_fixture_tensors(), str(path))
    expected_start, _ = _header_offsets(path)

    assert sf.read_header(path).data_start == expected_start


@pytest.mark.parametrize("name", ["layer.weight", "layer.scale"])
def test_read_tensor_round_trips_supported_dtypes(tmp_path: Path, name: str) -> None:
    """BF16 と F32 のテンソルを値まで戻せる (C4)。"""
    path = tmp_path / "model.safetensors"
    tensors = _write_fixture(path)
    header = sf.read_header(path)

    result = sf.read_tensor(path, header.tensors[name])

    assert result.dtype == tensors[name].dtype
    assert torch.equal(result, tensors[name])


def test_read_tensor_round_trips_f8_e4m3(tmp_path: Path) -> None:
    """F8_E4M3 のテンソルも値まで読める (`.to(float32)` が入力と一致する) (C2a。Issue #99)。"""
    path = tmp_path / "model.safetensors"
    tensors = _write_fixture(path)
    header = sf.read_header(path)

    result = sf.read_tensor(path, header.tensors["layer.fp8"])

    assert result.dtype == torch.float8_e4m3fn
    assert torch.equal(result.to(torch.float32), tensors["layer.fp8"].to(torch.float32))


def test_iter_tensor_bytes_reconstructs_raw_bytes(tmp_path: Path) -> None:
    """chunk で読んだバイト列をつなぐと元のバイト列になる (C7)。"""
    path = tmp_path / "model.safetensors"
    tensors = _write_fixture(path)
    info = sf.read_header(path).tensors["layer.weight"]

    chunks = list(sf.iter_tensor_bytes(path, info, chunk_size=4))

    assert len(chunks) > 1
    assert b"".join(chunks) == _raw_bytes(tensors["layer.weight"])


def test_tensor_to_chunks_reconstructs_raw_bytes() -> None:
    """`tensor_to_chunks` は元のバイト列を返す (C4)。"""
    for tensor in _fixture_tensors().values():
        assert b"".join(sf.tensor_to_chunks(tensor)) == _raw_bytes(tensor)


def test_write_shard_is_readable_by_reference_implementation(tmp_path: Path) -> None:
    """自前で書いた shard を参照実装で読める (C4)。"""
    path = tmp_path / "written.safetensors"
    tensors = _fixture_tensors()
    entries = [_entry(name, tensor) for name, tensor in tensors.items()]

    sf.write_shard(path, entries, {"format": "pt", "custom": "x"})

    loaded = load_file(str(path))
    assert set(loaded) == set(tensors)
    for name, tensor in tensors.items():
        assert loaded[name].dtype == tensor.dtype
        assert torch.equal(loaded[name], tensor)
    with safe_open(str(path), framework="pt") as handle:
        assert handle.metadata() == {"format": "pt", "custom": "x"}


def test_write_shard_offsets_start_at_zero_and_are_contiguous(tmp_path: Path) -> None:
    """offsets は 0 始まりで連続し、最後がバッファ長に一致する (C4)。"""
    path = tmp_path / "written.safetensors"
    tensors = _fixture_tensors()
    entries = [_entry(name, tensor) for name, tensor in tensors.items()]

    sf.write_shard(path, entries, None)

    data_start, offsets = _header_offsets(path)
    assert offsets[0][0] == 0
    for (_, end), (start, _) in zip(offsets, offsets[1:], strict=False):
        assert end == start
    assert offsets[-1][1] == len(path.read_bytes()) - data_start
