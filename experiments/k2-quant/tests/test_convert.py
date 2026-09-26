"""変換の道具の端から端までの試験 (C4 / C5 / C6 / C7 / C8)。

合成 checkpoint (`synthetic.build_checkpoint`) を作り、CLI の `main` を直接呼ぶ。
実機にも `serving/var/` にも触らない。確かめること:

- 対象を含まない shard と shard 以外の通常ファイルはバイト単位で同じ (`--link` なら同じ inode)
- 対象を含む shard は対象だけが FP8 になり、他のテンソルはバイト列・dtype・`__metadata__` を保つ
- 既定では `lm_head` と MTP の `shared_head.head` を変換せず (#68)、`ignore` に残す。
  `--pattern` で明示したときだけ変換する
- index に `weight_scale` が増え、`total_size` を作り直す (索引外 shard は足さない)
- manifest が出力の全ファイルを SHA-256 と大きさで覆い、変換条件を持つ
- manifest の `conversion.args` が、出力を決める実際の引数 (元の repo・版・pattern) だけを持つ
- 同じ入力・同じ引数なら出力はバイト単位で同じ
- 出力先は、新しいディレクトリか、存在する空のディレクトリ (bind mount の宛先)
- 入力の不備では出力を 1 ファイルも書かずに終了 1
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import synthetic
from k2_quant import TOOL_NAME, TOOL_VERSION
from k2_quant import safetensors_file as sf
from k2_quant.__main__ import main
from k2_quant.fp8 import dequantize_per_channel
from k2_quant.selection import DEFAULT_PATTERN

SOURCE_REPO = "RedHatAI/GLM-5.3-Flash-NVFP4"
SOURCE_REVISION = "18d55bfd5a2194887738da73753975c9d3842f46"

NARROW_PATTERN = r"model\.language_model\.layers\.0\.mlp\.gate_proj$"
NARROW_MODULES = ("model.language_model.layers.0.mlp.gate_proj",)


def _run(
    source: Path,
    output: Path,
    *,
    pattern: str | None = None,
    link: bool = False,
) -> int:
    argv = [
        "--source",
        str(source),
        "--output",
        str(output),
        "--source-repo",
        SOURCE_REPO,
        "--source-revision",
        SOURCE_REVISION,
    ]
    if pattern is not None:
        argv += ["--pattern", pattern]
    if link:
        argv += ["--link"]
    return main(argv)


def _read_json(path: Path) -> dict[str, object]:
    parsed: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def _read_config(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def _manifest_conversion(output: Path) -> dict[str, Any]:
    """出力の `manifest.json` の `conversion` (変換の条件) を読む。"""
    conversion: dict[str, Any] = _read_config(output / synthetic.MANIFEST_NAME)["conversion"]
    return conversion


def _group_target(output: Path) -> str:
    """出力の `config.json` で足された group (`group_2`) の、唯一の target を返す。"""
    config = _read_config(output / synthetic.CONFIG_NAME)
    targets = config["quantization_config"]["config_groups"]["group_2"]["targets"]
    assert len(targets) == 1
    target: str = targets[0]
    return target


def _hits(target: str, runtime_name: str) -> bool:
    """vLLM と同じく、`re:` 付き target を実行時の名前に `re.match` で当てる。"""
    assert target.startswith("re:")
    return re.match(target.removeprefix("re:"), runtime_name) is not None


def _load_tensors(directory: Path) -> dict[str, torch.Tensor]:
    """ディレクトリのすべての shard のテンソルを、参照実装で 1 つの辞書に読む。"""
    tensors: dict[str, torch.Tensor] = {}
    for path in sorted(directory.glob("*.safetensors")):
        tensors.update(load_file(str(path)))
    return tensors


def _indexed_total_size(output: Path, weight_map: Mapping[str, str]) -> int:
    total = 0
    for shard in set(weight_map.values()):
        header = sf.read_header(output / shard)
        total += sum(info.nbytes for info in header.tensors.values())
    return total


def _tensor_identity(path: Path, name: str) -> tuple[torch.dtype, tuple[int, ...], bytes]:
    """テンソルの (dtype, 形, 生のバイト列)。参照実装で読む。"""
    with safe_open(str(path), framework="pt") as handle:
        tensor: torch.Tensor = handle.get_tensor(name)
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return tensor.dtype, tuple(tensor.shape), raw


def _tensor_bytes(path: Path, name: str) -> bytes:
    return _tensor_identity(path, name)[2]


def test_copied_shards_and_other_files_keep_identical_bytes(tmp_path: Path) -> None:
    """対象を含まない shard と通常ファイルはバイト単位で同じ (C4)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output, pattern=NARROW_PATTERN) == 0

    copied = [synthetic.SHARD_NAMES[1], synthetic.SHARD_NAMES[2], synthetic.NON_INDEXED_SHARD]
    copied += list(synthetic.OTHER_FILE_NAMES)
    for name in copied:
        assert (output / name).read_bytes() == (source / name).read_bytes()


def test_link_reuses_inode_for_copied_files(tmp_path: Path) -> None:
    """`--link` で写したファイルは入力と同じ inode、書き直した shard は別 (C4)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output, pattern=NARROW_PATTERN, link=True) == 0

    for name in [
        synthetic.SHARD_NAMES[1],
        synthetic.NON_INDEXED_SHARD,
        *synthetic.OTHER_FILE_NAMES,
    ]:
        assert (output / name).stat().st_ino == (source / name).stat().st_ino
    assert (output / synthetic.SHARD_NAMES[0]).stat().st_ino != (
        source / synthetic.SHARD_NAMES[0]
    ).stat().st_ino


def test_link_failure_stops_with_exit_1_without_copying_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--link` でハードリンクを張れないとき (例: 入力と出力が別のファイルシステム) は、
    黙って写さず、終了 1 になる (C4)。

    `os.link` が `EXDEV` で失敗する状況を作る。対象を含まない shard は、写されていない。
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    def cross_device(src: object, dst: object) -> None:
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV))

    monkeypatch.setattr(os, "link", cross_device)

    assert _run(source, output, pattern=NARROW_PATTERN, link=True) == 1
    assert not (output / synthetic.SHARD_NAMES[1]).exists()


def test_rewritten_shard_preserves_non_target_tensors_and_metadata(tmp_path: Path) -> None:
    """書き直した shard の対象外テンソルはバイト列と `__metadata__` を保つ (C4)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output, pattern=NARROW_PATTERN) == 0

    shard = synthetic.SHARD_NAMES[0]
    original_name = "model.language_model.layers.0.self_attn.q_proj.weight"
    assert _tensor_bytes(output / shard, original_name) == _tensor_bytes(
        source / shard, original_name
    )
    with safe_open(str(output / shard), framework="pt") as handle:
        assert handle.metadata() == {"format": "pt"}


def test_target_tensor_becomes_fp8_weight_and_channel_scale(tmp_path: Path) -> None:
    """対象は F8_E4M3 の `weight` と F32 の `weight_scale` (out, 1) になる (C4)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output, pattern=NARROW_PATTERN) == 0

    module = NARROW_MODULES[0]
    tensors = load_file(str(output / synthetic.SHARD_NAMES[0]))
    weight = tensors[f"{module}.weight"]
    scale = tensors[f"{module}.weight_scale"]

    assert weight.dtype == torch.float8_e4m3fn
    assert tuple(weight.shape) == (8, 4)
    assert scale.dtype == torch.float32
    assert tuple(scale.shape) == (8, 1)


def test_converted_weights_dequantize_close_to_the_input_values(tmp_path: Path) -> None:
    """出力の `weight` と `weight_scale` を戻すと、入力の値に近い (C1 / C4)。

    入力 shard を読む → 量子化 → 出力 shard に書く → 出力を読む、の経路を通った値で確かめる。
    別のテンソルを量子化する、`weight` と `weight_scale` の対応が入れ替わる、バイト列の解釈を
    誤る、といった退行を検出する。合成データは、全 0 の行と、1 要素だけ大きい行を含む。
    誤差の範囲は `tests/test_fp8.py` と同じ (要素ごとの上限と、相対 Frobenius 誤差 <= 0.05)。
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output) == 0

    original = _load_tensors(source)
    converted = _load_tensors(output)
    for module in synthetic.DEFAULT_TARGET_MODULES:
        weight = original[f"{module}.weight"].to(torch.float32)
        scale = converted[f"{module}.weight_scale"]
        restored = dequantize_per_channel(converted[f"{module}.weight"], scale)

        assert restored.shape == weight.shape, module
        bound = 2**-4 * weight.abs() + (2**-10 + 1e-6) * scale
        assert bool(((weight - restored).abs() <= bound).all()), module
        relative = torch.linalg.vector_norm(weight - restored) / torch.linalg.vector_norm(weight)
        assert float(relative) <= 0.05, module


def test_output_config_adds_only_the_fp8_group_and_removes_only_converted_names(
    tmp_path: Path,
) -> None:
    """出力の `config.json`: FP8 の group が 1 つ増え、変換した名前だけ `ignore` から消える (C3)。

    `plan_conversion` が既定の正規表現と実際に選ばれたモジュールから作った config が、
    そのまま出力に書かれること (配線) を、変換の出力で確かめる。既存の group と、それ以外の
    `ignore`、ほかのキーは、入力と同じ。
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output) == 0

    before = _read_config(source / synthetic.CONFIG_NAME)
    after = _read_config(output / synthetic.CONFIG_NAME)
    quantization_before = before["quantization_config"]
    quantization_after = after["quantization_config"]

    groups_before = quantization_before["config_groups"]
    groups_after = quantization_after["config_groups"]
    assert set(groups_after) - set(groups_before) == {"group_2"}
    for name, group in groups_before.items():
        assert groups_after[name] == group
    new_group = groups_after["group_2"]
    assert new_group["format"] == "float-quantized"
    assert len(new_group["targets"]) == 1
    assert new_group["input_activations"] is None
    assert new_group["output_activations"] is None
    weights = new_group["weights"]
    assert weights["num_bits"] == 8
    assert weights["type"] == "float"
    assert weights["strategy"] == "channel"
    assert weights["symmetric"] is True
    assert weights["dynamic"] is False

    converted = set(synthetic.DEFAULT_TARGET_MODULES)
    ignore_before = quantization_before["ignore"]
    ignore_after = quantization_after["ignore"]
    assert set(ignore_before) - set(ignore_after) == converted
    assert ignore_after == [name for name in ignore_before if name not in converted]
    # eh_proj は既定で変換しないので、ignore に残る (vLLM が quant_config を受けない
    # plain nn.Linear として実装しているため)
    assert "model.language_model.layers.45.eh_proj" in ignore_after
    # lm_head も既定で変換しないので、ignore に残る (FP8 の ParallelLMHead を読めない。#68)
    assert "lm_head" in ignore_after

    def without(config: dict[str, Any], *keys: str) -> dict[str, Any]:
        return {key: value for key, value in config.items() if key not in keys}

    assert without(quantization_after, "config_groups", "ignore") == without(
        quantization_before, "config_groups", "ignore"
    )
    assert without(after, "quantization_config") == without(before, "quantization_config")


def test_output_target_matches_only_the_runtime_names_of_converted_modules(tmp_path: Path) -> None:
    """出力の `targets` は、変換したモジュールの実行時の名前にだけ当たる (C3)。

    vLLM は `re:` 付き target を、実行時の層名に `re.match` で当てる。本体は
    `language_model.model.layers.N...`、MTP は `model.layers.45...`、lm_head は
    `language_model.lm_head`。checkpoint にあるが既定で変換しない `eh_proj` (vLLM が
    quant_config を受けない plain `nn.Linear` として実装しているため) と `lm_head`
    (FP8 の `ParallelLMHead` を読めない。#68)、checkpoint に無い (変換していない)
    `shared_head.head`、専門家、`ignore` にだけ名前がある層 7 の共有の専門家には、当たらない。
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output) == 0

    target = _group_target(output)
    for converted in (
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.3.mlp.shared_experts.up_proj",
        "model.layers.45.mlp.shared_experts.down_proj",
    ):
        assert _hits(target, converted), converted
    for not_converted in (
        "language_model.lm_head",
        "model.layers.45.shared_head.head",
        "model.layers.45.eh_proj",
        "language_model.model.layers.3.mlp.experts.0.gate_proj",
        "language_model.model.layers.7.mlp.shared_experts.down_proj",
    ):
        assert not _hits(target, not_converted), not_converted


def test_eh_proj_is_not_converted_by_default(tmp_path: Path) -> None:
    """MTP の `eh_proj` は既定では変換されず、バイト列を保ち `ignore` に残る (C3 / C4)。

    vLLM (`vllm/models/glm5next/common/mtp.py:49`、commit 0961bbae) の `eh_proj` は
    quant_config を受けない plain `nn.Linear` なので、FP8 の `eh_proj` を読み込めない。
    そのため既定の対象から外している。
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output) == 0

    module = "model.language_model.layers.45.eh_proj"
    shard = synthetic.SHARD_NAMES[2]
    assert _tensor_bytes(output / shard, f"{module}.weight") == _tensor_bytes(
        source / shard, f"{module}.weight"
    )
    with safe_open(str(output / shard), framework="pt") as handle:
        assert f"{module}.weight_scale" not in set(handle.keys())
    config = _read_config(output / synthetic.CONFIG_NAME)
    assert module in config["quantization_config"]["ignore"]


def test_lm_head_is_not_converted_by_default(tmp_path: Path) -> None:
    """`lm_head` は既定では変換されず、dtype・バイト列を保ち、`weight_scale` を足さず、
    `ignore` に残る。

    FP8 (W8A16、compressed-tensors) の `ParallelLMHead` を humming の線形カーネルで読めず、
    `AttributeError: 'ParallelLMHead' object has no attribute 'output_partition_sizes'` で
    起動できなかった (#68)。同じ shard の dense の対象は変換され、書き直した shard の中で
    `lm_head` のバイト列が保たれる。
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output) == 0

    shard = synthetic.SHARD_NAMES[2]
    dense = "model.language_model.layers.2.mlp.down_proj"
    assert _tensor_identity(output / shard, "lm_head.weight") == _tensor_identity(
        source / shard, "lm_head.weight"
    )
    assert _tensor_identity(output / shard, "lm_head.weight")[0] == torch.bfloat16
    with safe_open(str(output / shard), framework="pt") as handle:
        names = set(handle.keys())
        assert handle.get_tensor(f"{dense}.weight").dtype == torch.float8_e4m3fn
    assert "lm_head.weight_scale" not in names
    assert f"{dense}.weight_scale" in names
    config = _read_config(output / synthetic.CONFIG_NAME)
    assert "lm_head" in config["quantization_config"]["ignore"]


def test_shared_head_is_not_converted_by_default(tmp_path: Path) -> None:
    """MTP の head の重みがあっても、既定では変換されず、バイト列を保ち、`weight_scale` を足さず、
    `ignore` に残り、出力の `targets` もその実行時の名前に当たらない。

    `shared_head.head` は `SharedHead` (`vllm/model_executor/models/deepseek_mtp.py`) の中の
    `ParallelLMHead` で、`lm_head` と同じく FP8 では読めない (#68)。
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source, include_shared_head=True)

    assert _run(source, output) == 0

    module = synthetic.SHARED_HEAD_MODULE
    shard = synthetic.SHARD_NAMES[2]
    assert _tensor_identity(output / shard, f"{module}.weight") == _tensor_identity(
        source / shard, f"{module}.weight"
    )
    assert _tensor_identity(output / shard, f"{module}.weight")[0] == torch.bfloat16
    with safe_open(str(output / shard), framework="pt") as handle:
        assert f"{module}.weight_scale" not in set(handle.keys())
    assert not _hits(_group_target(output), "model.layers.45.shared_head.head")
    config = _read_config(output / synthetic.CONFIG_NAME)
    assert module in config["quantization_config"]["ignore"]


def test_lm_head_is_converted_when_the_pattern_names_it(tmp_path: Path) -> None:
    """`--pattern` に `lm_head` の分岐を足せば、`lm_head` も変換され、`ignore` から外れ、
    出力の `targets` がその実行時の名前 (`language_model.lm_head`) に当たる。

    既定から外しても、変換の能力は残る。
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output, pattern=DEFAULT_PATTERN + r"|(?:.*\.)?lm_head$") == 0

    converted = _load_tensors(output)
    assert converted["lm_head.weight"].dtype == torch.float8_e4m3fn
    assert "lm_head.weight_scale" in converted
    assert _hits(_group_target(output), "language_model.lm_head")
    config = _read_config(output / synthetic.CONFIG_NAME)
    assert "lm_head" not in config["quantization_config"]["ignore"]


def test_shared_head_is_converted_when_the_pattern_names_it(tmp_path: Path) -> None:
    """`--pattern` に `shared_head.head` の分岐を足せば、MTP の head も変換され、`ignore` から外れ、
    出力の `targets` がその実行時の名前に当たる。

    既定から外しても、変換の能力は残る。
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source, include_shared_head=True)

    assert (
        _run(source, output, pattern=DEFAULT_PATTERN + r"|.*\.layers\.\d+\.shared_head\.head$") == 0
    )

    converted = _load_tensors(output)
    assert converted[f"{synthetic.SHARED_HEAD_MODULE}.weight"].dtype == torch.float8_e4m3fn
    assert f"{synthetic.SHARED_HEAD_MODULE}.weight_scale" in converted
    assert _hits(_group_target(output), "model.layers.45.shared_head.head")
    config = _read_config(output / synthetic.CONFIG_NAME)
    assert synthetic.SHARED_HEAD_MODULE not in config["quantization_config"]["ignore"]


def test_output_target_is_limited_to_converted_modules_for_a_broad_pattern(tmp_path: Path) -> None:
    """`--pattern` が実行時には広く当たっても、`targets` は変換したモジュールだけに当たる (C3)。

    `.*gate_proj$` は、checkpoint では層 0 の dense の `gate_proj` だけを選ぶ (専門家は
    `weight_packed` で候補にならない)。実行時の名前では専門家の `gate_proj` にも当たるが、
    変換していないので、出力の `targets` は当たってはならない。
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output, pattern=r".*gate_proj$") == 0

    manifest_modules = _read_json(output / synthetic.MANIFEST_NAME)["conversion"]
    assert isinstance(manifest_modules, dict)
    assert manifest_modules["modules"] == ["model.language_model.layers.0.mlp.gate_proj"]
    target = _group_target(output)
    assert _hits(target, "language_model.model.layers.0.mlp.gate_proj")
    assert not _hits(target, "language_model.model.layers.3.mlp.experts.0.gate_proj")


def test_default_conversion_changes_only_the_targets_in_every_shard(tmp_path: Path) -> None:
    """既定の正規表現で変換すると、全 shard で、対象だけが変わる (C4)。

    合成 checkpoint は、対象と NVFP4 の専門家 (U8 の `weight_packed`、F8_E4M3 の `weight_scale`、
    F32 の `weight_global_scale`) が同じ shard にある実機の形を含む。書き直した各 shard で、
    出力のテンソル名の集合は「入力の名前 + 対象の `weight_scale`」に一致し、対象以外の全
    テンソルは、dtype・形・バイト列が入力と同じで、`__metadata__` も同じ。
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)
    targets = set(synthetic.DEFAULT_TARGET_MODULES)

    assert _run(source, output) == 0

    preserved_dtypes: set[torch.dtype] = set()
    for shard in (*synthetic.SHARD_NAMES, synthetic.NON_INDEXED_SHARD):
        with (
            safe_open(str(source / shard), framework="pt") as before,
            safe_open(str(output / shard), framework="pt") as after,
        ):
            before_names = set(before.keys())
            after_names = set(after.keys())
            assert after.metadata() == before.metadata(), shard
        converted = {f"{module}.weight" for module in targets if f"{module}.weight" in before_names}
        assert converted, f"{shard}: 対象が 1 つも無い (書き直しの経路を通らない)"
        added = {f"{name.removesuffix('.weight')}.weight_scale" for name in converted}
        assert after_names == before_names | added, shard
        for name in before_names - converted:
            unchanged = _tensor_identity(output / shard, name)
            assert unchanged == _tensor_identity(source / shard, name), f"{shard}: {name}"
            preserved_dtypes.add(unchanged[0])

    assert {torch.uint8, torch.float8_e4m3fn, torch.float32} <= preserved_dtypes


def test_index_adds_weight_scale_and_recomputes_total_size(tmp_path: Path) -> None:
    """index に `weight_scale` が増え、`total_size` を作り直す (C5)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output, pattern=NARROW_PATTERN) == 0

    index = _read_json(output / synthetic.INDEX_NAME)
    weight_map = index["weight_map"]
    assert isinstance(weight_map, dict)
    for module in NARROW_MODULES:
        assert weight_map[f"{module}.weight_scale"] == weight_map[f"{module}.weight"]
    assert synthetic.NON_INDEXED_SHARD not in set(weight_map.values())
    assert index["metadata"] == {"total_size": _indexed_total_size(output, weight_map)}


def test_manifest_lists_all_output_files_with_hashes(tmp_path: Path) -> None:
    """manifest が出力の全ファイルを覆い、変換の条件を持つ (C5)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output, pattern=NARROW_PATTERN) == 0

    manifest = _read_json(output / synthetic.MANIFEST_NAME)
    files = manifest["files"]
    assert isinstance(files, list)
    paths = [entry["path"] for entry in files]
    assert paths == sorted(paths)

    on_disk = sorted(
        path.name
        for path in output.iterdir()
        if path.is_file() and path.name != synthetic.MANIFEST_NAME
    )
    assert paths == on_disk
    for entry in files:
        path = output / entry["path"]
        assert entry["size"] == path.stat().st_size
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["total_bytes"] == sum(entry["size"] for entry in files)

    conversion = manifest["conversion"]
    assert isinstance(conversion, dict)
    assert conversion["tool"] == TOOL_NAME
    assert conversion["tool_version"] == TOOL_VERSION
    assert conversion["source"] == {"repo": SOURCE_REPO, "revision": SOURCE_REVISION}
    assert conversion["pattern"] == NARROW_PATTERN
    assert sorted(conversion["modules"]) == sorted(NARROW_MODULES)
    assert conversion["weight_dtype"] == "F8_E4M3"
    assert conversion["scale_dtype"] == "F32"
    assert conversion["strategy"] == "channel"


def test_manifest_records_the_arguments_that_shape_the_output(tmp_path: Path) -> None:
    """manifest の `conversion.args` は、元の repo・元の版・実際に使った pattern の列になる。

    取り込み側 (`serve derived-import`) は、この列をそのまま派生の重みの変換の引数にする。
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output, pattern=NARROW_PATTERN) == 0

    assert _manifest_conversion(output)["args"] == [
        "--source-repo",
        SOURCE_REPO,
        "--source-revision",
        SOURCE_REVISION,
        "--pattern",
        NARROW_PATTERN,
    ]


def test_manifest_records_the_default_pattern_when_pattern_is_omitted(tmp_path: Path) -> None:
    """`--pattern` を渡さなくても、`conversion.args` には実際に使った既定の pattern を明示する。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output) == 0

    conversion = _manifest_conversion(output)
    assert conversion["args"] == [
        "--source-repo",
        SOURCE_REPO,
        "--source-revision",
        SOURCE_REVISION,
        "--pattern",
        DEFAULT_PATTERN,
    ]
    assert conversion["pattern"] == DEFAULT_PATTERN


def test_manifest_does_not_depend_on_where_the_paths_are_or_on_link(tmp_path: Path) -> None:
    """`--source`・`--output` の場所と `--link` の有無は、manifest のバイト列を変えない。

    実機では入力と出力のマウント位置が違うので、これらが `conversion.args` に入ると、
    2 台の manifest が一致しなくなり、`--link` を使ったかどうかで出力も変わってしまう。
    """
    first_source = tmp_path / "first-source"
    second_source = tmp_path / "elsewhere" / "second-source"
    first = tmp_path / "first"
    second = tmp_path / "other" / "second"
    synthetic.build_checkpoint(first_source)
    synthetic.build_checkpoint(second_source)
    second.parent.mkdir()

    assert _run(first_source, first) == 0
    assert _run(second_source, second, link=True) == 0

    first_manifest = (first / synthetic.MANIFEST_NAME).read_bytes()
    assert (second / synthetic.MANIFEST_NAME).read_bytes() == first_manifest
    assert "--link" not in _manifest_conversion(second)["args"]


def test_conversion_is_deterministic(tmp_path: Path) -> None:
    """同じ入力・同じ引数なら全ファイルがバイト単位で同じ (C6)。"""
    source = tmp_path / "source"
    first = tmp_path / "first"
    second = tmp_path / "second"
    synthetic.build_checkpoint(source)

    assert _run(source, first) == 0
    assert _run(source, second) == 0

    first_files = sorted(path.name for path in first.iterdir())
    second_files = sorted(path.name for path in second.iterdir())
    assert first_files == second_files
    for name in first_files:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_non_indexed_shard_is_converted_but_not_indexed(tmp_path: Path) -> None:
    """索引外の shard も変換し、index には足さない (C5)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output) == 0

    tensors = load_file(str(output / synthetic.NON_INDEXED_SHARD))
    module = "model.language_model.layers.45.mlp.shared_experts.down_proj"
    assert tensors[f"{module}.weight"].dtype == torch.float8_e4m3fn
    assert f"{module}.weight_scale" in tensors

    index = _read_json(output / synthetic.INDEX_NAME)
    weight_map = index["weight_map"]
    assert isinstance(weight_map, dict)
    assert synthetic.NON_INDEXED_SHARD not in set(weight_map.values())


def test_other_files_are_copied_and_subdirectories_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """通常ファイルは写し、サブディレクトリは写さず標準出力に出す (C8)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output, pattern=NARROW_PATTERN) == 0

    for name in synthetic.OTHER_FILE_NAMES:
        assert (output / name).read_bytes() == (source / name).read_bytes()
    assert not (output / synthetic.SKIPPED_DIR_NAME).exists()
    captured = capsys.readouterr()
    assert synthetic.SKIPPED_DIR_NAME in captured.out


def test_rejects_a_non_empty_existing_output_directory(tmp_path: Path) -> None:
    """中身のある既存の出力先は終了 1 で、何も書き足さない (C7)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)
    output.mkdir()
    (output / "keep.txt").write_text("既存\n", encoding="utf-8")

    assert _run(source, output) == 1
    assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]
    assert (output / "keep.txt").read_text(encoding="utf-8") == "既存\n"


def test_accepts_an_existing_empty_output_directory(tmp_path: Path) -> None:
    """存在する空のディレクトリは、出力先として受け付け、全ファイルを書く。

    `docker run --mount type=bind,target=/derived` の宛先は、コンテナの中に必ず存在する。
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)
    output.mkdir()

    assert _run(source, output, pattern=NARROW_PATTERN) == 0

    written = sorted(path.name for path in output.iterdir())
    assert synthetic.MANIFEST_NAME in written
    assert synthetic.CONFIG_NAME in written
    assert synthetic.INDEX_NAME in written
    assert set(synthetic.SHARD_NAMES) | {synthetic.NON_INDEXED_SHARD} <= set(written)
    manifest_paths = [
        entry["path"] for entry in _read_config(output / synthetic.MANIFEST_NAME)["files"]
    ]
    assert manifest_paths == [name for name in written if name != synthetic.MANIFEST_NAME]


def test_rejects_zero_selection(tmp_path: Path) -> None:
    """対象 0 件は終了 1 で、出力ディレクトリを作らない (C7)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output, pattern=r"model\.language_model\.layers\.99\.mlp\.gate_proj$") == 1
    assert not output.exists()


def test_rejects_pattern_selecting_experts(tmp_path: Path) -> None:
    """専門家 (`weight_packed` を持つ) を拾う正規表現は終了 1 (C7)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)

    assert _run(source, output, pattern=r".*\.mlp\.experts\.\d+\.gate_proj$") == 1
    assert not output.exists()


def test_rejects_missing_indexed_shard(tmp_path: Path) -> None:
    """index が指す shard が無ければ終了 1 (C7)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)
    (source / synthetic.SHARD_NAMES[1]).unlink()

    assert _run(source, output) == 1
    assert not output.exists()


def test_rejects_wrong_format(tmp_path: Path) -> None:
    """`format` が `mixed-precision` でなければ終了 1 (C7)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)
    config_path = source / synthetic.CONFIG_NAME
    config = _read_json(config_path)
    quantization_config = config["quantization_config"]
    assert isinstance(quantization_config, dict)
    quantization_config["format"] = "float-quantized"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    assert _run(source, output) == 1
    assert not output.exists()


def test_rejects_source_containing_manifest(tmp_path: Path) -> None:
    """入力に `manifest.json` があれば終了 1 (出力の manifest と取り違えないため。C7)。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)
    (source / synthetic.MANIFEST_NAME).write_text("{}\n", encoding="utf-8")

    assert _run(source, output) == 1
    assert not output.exists()
