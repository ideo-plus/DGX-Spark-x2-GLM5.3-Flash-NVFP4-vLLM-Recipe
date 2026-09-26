"""変換の道具の manifest の取り込み (`serving_kit.derived`) の試験。

道具 (`experiments/k2-quant`) が書く `manifest.json` を、Mac で読み、serving の派生の重みの
マニフェスト (`DerivedWeightsManifest`) に変換する部分を、ファイルと値だけで確かめる
(Spark にも、道具の実行にも触れない)。確かめること:

- 道具の manifest の形 (実際の引数の記録 `conversion.args` を含む) を読み、形の誤りは
  `WeightsRefError` で断る。実際の引数のない入力は、補わずに断る
- 2 台ぶんの manifest の比較は、最上位の `generated_at` を除いて、違う箇所を列挙する
- 変換は、`args` を入力の記録から、`target_pattern` を入力の `pattern` から作る (手で書かない)。
  Hub のマニフェストと同じ除外規則 (モデルカードらしい名前、`.gitattributes`) を掛ける

サブコマンドとしての振る舞い (標準出力、終了コード、上書きの規則) は `test_cli.py` で見る。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from serving_kit import derived
from serving_kit import types as kt
from serving_kit.weights import WeightsRefError

REPO = "RedHatAI/GLM-5.3-Flash-NVFP4"
REVISION = "18d55bfd" + "0" * 32
PATTERN = r"^model\.language_model\.layers\.(?:0|3)\.mlp\.gate_proj$"
ARGS = ("--source-repo", REPO, "--source-revision", REVISION, "--pattern", PATTERN)
COMMIT = "a" * 40
TOOL_PATH = "experiments/k2-quant"
GENERATED_AT = datetime(2026, 9, 26, 3, 0, 0, tzinfo=UTC)

FILES: tuple[dict[str, Any], ...] = (
    {"path": "config.json", "size": 4_096, "sha256": "1" * 64},
    {"path": "model-00001-of-00003.safetensors", "size": 5_000, "sha256": "2" * 64},
    {"path": "model-00002-of-00003.safetensors", "size": 6_000, "sha256": "3" * 64},
)


def payload(*, files: Sequence[Mapping[str, Any]] = FILES) -> dict[str, Any]:
    """道具が書く manifest.json の形 (契約の形)。"""
    return {
        "conversion": {
            "tool": "k2-quant",
            "tool_version": "0.2.0",
            "source": {"repo": REPO, "revision": REVISION},
            "pattern": PATTERN,
            "args": list(ARGS),
            "modules": ["model.language_model.layers.0.mlp.gate_proj"],
            "weight_dtype": "F8_E4M3",
            "scale_dtype": "F32",
            "strategy": "channel",
        },
        "files": [dict(entry) for entry in files],
        "total_bytes": sum(int(entry["size"]) for entry in files),
    }


def write(directory: Path, name: str, body: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read(tmp_path: Path, name: str = "head.manifest.json", **kwargs: Any) -> derived.ToolManifest:
    return derived.read_tool_manifest(write(tmp_path, name, payload(**kwargs)))


# --- 読み取り -----------------------------------------------------------


def test_read_tool_manifest_reads_the_recorded_conversion_and_files(tmp_path: Path) -> None:
    tool = read(tmp_path)

    assert tool.source_repo == REPO
    assert tool.source_revision == REVISION
    assert tool.pattern == PATTERN
    assert tool.args == ARGS
    assert [(f.path, f.size, f.sha256) for f in tool.files] == [
        (str(f["path"]), int(f["size"]), str(f["sha256"])) for f in FILES
    ]


def test_read_tool_manifest_refuses_a_manifest_without_the_recorded_arguments(
    tmp_path: Path,
) -> None:
    """`conversion.args` のない入力 (実際の引数の記録がない形) は、補わずに断る。"""
    body = payload()
    del body["conversion"]["args"]
    path = write(tmp_path, "head.manifest.json", body)

    with pytest.raises(WeightsRefError) as caught:
        derived.read_tool_manifest(path)

    assert "args" in str(caught.value)
    assert str(path) in str(caught.value)


def test_read_tool_manifest_refuses_a_manifest_written_by_another_tool(tmp_path: Path) -> None:
    body = payload()
    body["conversion"]["tool"] = "other-tool"
    path = write(tmp_path, "head.manifest.json", body)

    with pytest.raises(WeightsRefError) as caught:
        derived.read_tool_manifest(path)

    assert "k2-quant" in str(caught.value)


@pytest.mark.parametrize("problem", ["missing", "not-json", "files-not-a-manifest-row"])
def test_read_tool_manifest_refuses_an_unreadable_or_misshapen_file(
    tmp_path: Path, problem: str
) -> None:
    """読めない・JSON でない・行の形が違う入力は、道筋を示して `WeightsRefError` で断る。"""
    path = tmp_path / "head.manifest.json"
    if problem == "not-json":
        path.write_text("{ 壊れた", encoding="utf-8")
    elif problem == "files-not-a-manifest-row":
        body = payload()
        body["files"][0]["sha256"] = "not-a-sha256"
        write(tmp_path, path.name, body)

    with pytest.raises(WeightsRefError) as caught:
        derived.read_tool_manifest(path)

    assert str(path) in str(caught.value)


# --- 2 台ぶんの比較 -----------------------------------------------------


def test_compare_finds_no_difference_between_equal_manifests(tmp_path: Path) -> None:
    first = read(tmp_path, "head.manifest.json")
    second = read(tmp_path, "worker.manifest.json")

    assert derived.compare_tool_manifests([first, second]) == ()


def test_compare_ignores_the_top_level_generated_at(tmp_path: Path) -> None:
    """最上位の `generated_at` だけの違いは、違いに数えない (書いた時刻は、2 台で必ず違う)。"""
    first = derived.read_tool_manifest(
        write(tmp_path, "head.manifest.json", {**payload(), "generated_at": "2026-09-25T00:00:00Z"})
    )
    second = derived.read_tool_manifest(
        write(
            tmp_path, "worker.manifest.json", {**payload(), "generated_at": "2026-09-26T09:00:00Z"}
        )
    )

    assert derived.compare_tool_manifests([first, second]) == ()


def test_compare_names_the_input_and_the_place_of_a_different_file_hash(tmp_path: Path) -> None:
    """違う箇所 (例: `files[2].sha256`) と、その入力の名前と、両方の値を挙げる。"""
    first = read(tmp_path, "head.manifest.json")
    changed = [*FILES[:2], {**FILES[2], "sha256": "f" * 64}]
    second = read(tmp_path, "worker.manifest.json", files=changed)

    differences = derived.compare_tool_manifests([first, second])

    assert len(differences) == 1
    assert "worker.manifest.json" in differences[0]
    assert "files[2].sha256" in differences[0]
    assert "3" * 64 in differences[0]
    assert "f" * 64 in differences[0]


def test_compare_compares_every_input_with_the_first_one(tmp_path: Path) -> None:
    """3 つ以上を渡しても、先頭だけでなく、すべての入力を先頭と比べる。"""
    first = read(tmp_path, "a.manifest.json")
    same = read(tmp_path, "b.manifest.json")
    changed = [{**FILES[0], "size": 1}, *FILES[1:]]
    different = read(tmp_path, "c.manifest.json", files=changed)

    differences = derived.compare_tool_manifests([first, same, different])

    assert differences
    assert all("c.manifest.json" in one for one in differences)


def test_compare_reports_a_key_present_on_one_side_only(tmp_path: Path) -> None:
    first = read(tmp_path, "head.manifest.json")
    body = payload()
    del body["conversion"]["strategy"]
    second = derived.read_tool_manifest(write(tmp_path, "worker.manifest.json", body))

    differences = derived.compare_tool_manifests([first, second])

    assert any("conversion.strategy" in one for one in differences)


@pytest.mark.parametrize(
    ("head_files", "worker_files", "side"),
    [
        (FILES, FILES[:2], "先頭の入力にしかない"),
        (FILES[:2], FILES, "この入力にしかない"),
    ],
    ids=["worker-lacks-a-shard", "worker-has-an-extra-shard"],
)
def test_compare_reports_a_file_present_in_one_input_only(
    tmp_path: Path,
    head_files: tuple[dict[str, Any], ...],
    worker_files: tuple[dict[str, Any], ...],
    side: str,
) -> None:
    """片方に分割ファイルが 1 つ足りない 2 台は、その入力の名前と、足りない `files[N]` を挙げる。"""
    first = read(tmp_path, "head.manifest.json", files=head_files)
    second = read(tmp_path, "worker.manifest.json", files=worker_files)

    differences = derived.compare_tool_manifests([first, second])

    assert f"worker.manifest.json: files[2]: {side}" in differences


# --- 派生のマニフェストへの変換 -----------------------------------------


def test_import_builds_the_derived_manifest_from_the_recorded_conversion(tmp_path: Path) -> None:
    """`args` は入力の記録、`target_pattern` は入力の `pattern`、`origin` は `source` から作る。

    手で書いた値や、定数の引数を使っていないことを、入力の値との一致で確かめる。
    """
    tool = read(tmp_path)

    result = derived.import_tool_manifest(
        tool, name="k2s1", tool_path=TOOL_PATH, commit=COMMIT, generated_at=GENERATED_AT
    )

    manifest = result.manifest
    assert manifest.kind == "derived"
    assert manifest.generated_at == GENERATED_AT
    assert manifest.derivation == kt.Derivation(
        name="k2s1",
        origin=kt.WeightsOrigin(repo=REPO, revision=REVISION),
        conversion=kt.ConversionSpec(
            tool=TOOL_PATH, commit=COMMIT, args=ARGS, target_pattern=PATTERN
        ),
    )
    assert [(f.path, f.size, f.sha256) for f in manifest.files] == [
        (str(f["path"]), int(f["size"]), str(f["sha256"])) for f in FILES
    ]
    assert manifest.total_bytes == sum(int(f["size"]) for f in FILES)
    assert result.excluded_paths == ()


def test_import_leaves_out_model_card_like_files_and_totals_the_rest(tmp_path: Path) -> None:
    """Hub のマニフェストと同じ規則で、モデルカードらしい名前と `.gitattributes` を除く。"""
    files = [
        {"path": ".gitattributes", "size": 10, "sha256": "4" * 64},
        *FILES,
        {"path": "README.md", "size": 20, "sha256": "5" * 64},
        {"path": "model_card.json", "size": 30, "sha256": "6" * 64},
    ]
    tool = read(tmp_path, files=files)

    result = derived.import_tool_manifest(
        tool, name="k2s1", tool_path=TOOL_PATH, commit=COMMIT, generated_at=GENERATED_AT
    )

    assert [f.path for f in result.manifest.files] == [str(f["path"]) for f in FILES]
    assert result.manifest.total_bytes == sum(int(f["size"]) for f in FILES)
    assert result.excluded_paths == (".gitattributes", "README.md", "model_card.json")


def test_import_orders_the_files_by_path(tmp_path: Path) -> None:
    tool = read(tmp_path, files=list(reversed(FILES)))

    result = derived.import_tool_manifest(
        tool, name="k2s1", tool_path=TOOL_PATH, commit=COMMIT, generated_at=GENERATED_AT
    )

    paths = [f.path for f in result.manifest.files]
    assert paths == sorted(paths)


def test_import_refuses_a_generated_at_without_a_timezone(tmp_path: Path) -> None:
    tool = read(tmp_path)

    with pytest.raises(ValueError, match="tz-aware"):
        derived.import_tool_manifest(
            tool,
            name="k2s1",
            tool_path=TOOL_PATH,
            commit=COMMIT,
            generated_at=datetime(2026, 9, 26, 3, 0, 0),  # noqa: DTZ001 - 素の日時を断ることの試験
        )


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("commit", "abc"),
        ("commit", "g" * 40),
        ("name", "k2/s1"),
        ("tool_path", "/abs/k2-quant"),
        ("tool_path", "../k2-quant"),
    ],
)
def test_import_refuses_a_value_that_the_derived_manifest_does_not_accept(
    tmp_path: Path, field: str, bad: str
) -> None:
    """名前・道具の道筋・commit の形の誤りは、`WeightsRefError` で断る (書く前に分かる)。"""
    tool = read(tmp_path)
    given = {"name": "k2s1", "tool_path": TOOL_PATH, "commit": COMMIT, field: bad}

    with pytest.raises(WeightsRefError):
        derived.import_tool_manifest(tool, generated_at=GENERATED_AT, **given)
