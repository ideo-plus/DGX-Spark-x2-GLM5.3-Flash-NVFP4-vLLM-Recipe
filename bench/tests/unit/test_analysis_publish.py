"""要約だけを公開の場所に置く (`analysis/publish.py`, task 4.4) の試験。

確かめること (タスク 4.4 の完了の状態、8.4、11.1):

- 公開のあと、公開先のディレクトリにあるのが `summary.json` と `summary.md` の
  2 つだけになる。どちらにも、生データ (試行のレコード、送った本文) に含めた
  目印の文字列 (sentinel) が現れない
- 公開先は `<公開の場所>/<manifest の run_id>/` (ディレクトリの名前からではなく、
  `RunStore.open(run_dir).manifest()` の `run_id` から決める)
- `summary.json` の写しは、読み込んだバイト列と 1 バイトも違わない
- 2 回公開しても同じ結果になり、一時ファイルを残さず、公開先にすでにある
  無関係なファイルには触れない
- 安全の確認 (8.3、11.1、fail closed): `summary.json` / `summary.md` が
  そろっていない、`summary.json` が `types.Summary` として検証できない
  (余分な項目、型の違い、途中で切れている)、`conditions.run_id` がこの
  計測ランと違う、どちらかがシンボリックリンクまたはハードリンクである、
  のどれか 1 つでも `PublishError` になり、何も写さない
- `<公開の場所>/<run_id>` に、公開の場所の外を指すシンボリックリンクが
  あらかじめ置かれていても、そこには書かない (独立レビュー 1 回目の指摘、
  必須 1。`_safe_dest_dir` の `is_relative_to` の確認)
- 公開の場所の既定値 (`default_docs_results_root`) は、git が見つからない、
  または 0 以外の終了コードを返したときに `Path.cwd()` へ黙って倒れず、
  `PublishError` になる (独立レビュー 1 回目の指摘、必須 2)
- `run_id` を経路の要素として使ってよいかの確認 (`_validate_run_id_component`)
  を、`RunStore` の外から直接試す (`RunStore` が払い出す `run_id` はすでに
  安全な形に整っているので、この道を通して危険な値を作れない)。空文字、
  `/`・`..`・先頭の `.`、長さの上限超え、NUL・改行のどれも拒む
- 公開の場所の既定値が `<このリポジトリの直下>/docs/results` になること
  (パスだけを確かめる。実際のリポジトリには公開しない)
- `publish.py` が、許した集まりの外を読み込んでいないこと (AST で確かめる)

`RunStore.create()` は `results_root` が git の管理の対象でないことを確かめる
ので、`tmp_path` (git のリポジトリの外) を計測ランの置き場所に使う。
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl

from bench_harness.analysis import publish
from bench_harness.store.rawstore import RunStore, new_run_id
from bench_harness.types import (
    ContentBlock,
    Profile,
    RunManifest,
    RunStatus,
    StreamResult,
    StreamTiming,
    SuiteName,
    Summary,
    TargetDef,
    TrialRecord,
)

AT = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)
SENTINEL = "sentinel-8f3c2e1a-do-not-leak-6b91"
"""生データにだけ現れ、要約にも公開先にも絶対に現れてはいけない目印。"""


# --- 組み立ての助け ---------------------------------------------------------


def _target(**overrides: Any) -> TargetDef:
    payload: dict[str, Any] = {
        "name": "candidate-d",
        "base_url": HttpUrl("http://localhost:8000"),
        "model": "glm-5.3-flash",
    }
    payload.update(overrides)
    return TargetDef.model_validate(payload)


def _manifest(*, run_id: str | None = None, target: TargetDef | None = None) -> RunManifest:
    return RunManifest(
        run_id=run_id or new_run_id("candidate-d", AT),
        status=RunStatus.RUNNING,
        target=target or _target(),
        started_at=AT,
        suites=[SuiteName.DECODE],
        profile_name="quick",
        profile=Profile(name="quick"),
        harness_version="0.1.0+test",
    )


def _make_run(tmp_path: Path, *, results_dirname: str = "results") -> RunStore:
    """計測ランを 1 つ作り、`SENTINEL` を含む試行を 1 つ書き足す。

    `trials.jsonl` (試行のレコード) と `bodies/` (送った本文) の両方に
    `SENTINEL` が入るので、要約と公開先のどちらにも現れないことを確かめられる。
    """
    store = RunStore.create(tmp_path / results_dirname, _manifest())
    body_ref = store.put_body({"model": "glm-5.3-flash", "messages": [], "sentinel": SENTINEL})
    trial = TrialRecord(
        run_id=store.manifest().run_id,
        suite=SuiteName.DECODE,
        condition="decode/code/en",
        trial_index=0,
        request_body_ref=body_ref,
        result=StreamResult(
            timing=StreamTiming(sent_at_utc=AT, sent_at_ns=0, end_ns=1),
            blocks=[ContentBlock(type="text", text=SENTINEL)],
        ),
    )
    store.append_trial(trial)
    return store


def _summary_for(conditions: RunManifest, *, incomplete: bool = False) -> Summary:
    return Summary(conditions=conditions, incomplete=incomplete, results=[])


def _write_summary_json(store: RunStore, raw: bytes) -> None:
    (store.run_dir / "summary.json").write_bytes(raw)


def _write_summary_md(store: RunStore, text: str = "# 要約\n\nダミーの要約。\n") -> None:
    (store.run_dir / "summary.md").write_text(text, encoding="utf-8")


def _write_valid_summary_files(store: RunStore, *, incomplete: bool = False) -> Summary:
    summary = _summary_for(store.manifest(), incomplete=incomplete)
    _write_summary_json(store, summary.model_dump_json().encode("utf-8"))
    _write_summary_md(store)
    return summary


def _docs_root(tmp_path: Path) -> Path:
    return tmp_path / "docs_results"


# --- 公開のあと、そのディレクトリにあるのが 2 つのファイルだけになる (tasks.md 4.4) ---


def test_publish_creates_exactly_the_two_summary_files(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    _write_valid_summary_files(store)
    docs_root = _docs_root(tmp_path)

    result = publish.publish_run(store.run_dir, docs_results_root=docs_root)

    entries = sorted(p.name for p in result.dest_dir.iterdir())
    assert entries == ["summary.json", "summary.md"]
    assert result.dest_dir == (docs_root / store.manifest().run_id).resolve()
    assert result.summary_json_path == result.dest_dir / "summary.json"
    assert result.summary_md_path == result.dest_dir / "summary.md"


def test_published_files_never_contain_the_sentinel(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    _write_valid_summary_files(store)
    docs_root = _docs_root(tmp_path)

    result = publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert SENTINEL not in result.summary_json_path.read_text(encoding="utf-8")
    assert SENTINEL not in result.summary_md_path.read_text(encoding="utf-8")
    # 生データの側には、確かに sentinel が入っている (試験そのものの前提の確認)
    assert SENTINEL in (store.run_dir / "trials.jsonl").read_text(encoding="utf-8")


def test_destination_is_root_slash_run_id_from_the_manifest_not_the_dirname(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    _write_valid_summary_files(store)
    docs_root = _docs_root(tmp_path)
    run_id = store.manifest().run_id

    # 計測ランのディレクトリの名前を変えても (run_dir.name != run_id)、
    # 公開先は manifest の run_id から決まる
    renamed = store.run_dir.parent / "renamed-run-dir"
    store.run_dir.rename(renamed)
    assert renamed.name != run_id

    result = publish.publish_run(renamed, docs_results_root=docs_root)

    assert result.dest_dir == (docs_root / run_id).resolve()


# --- summary.json の写しはバイト単位で同じ -----------------------------------


def test_summary_json_copy_is_byte_identical_to_the_source(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    _write_valid_summary_files(store)
    source_bytes = (store.run_dir / "summary.json").read_bytes()
    docs_root = _docs_root(tmp_path)

    result = publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert result.summary_json_path.read_bytes() == source_bytes


# --- 冪等性: 2 回公開しても同じ結果、一時ファイルを残さない ---------------------


def test_publish_twice_is_idempotent_and_leaves_no_temp_files(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    _write_valid_summary_files(store)
    docs_root = _docs_root(tmp_path)

    first = publish.publish_run(store.run_dir, docs_results_root=docs_root)
    second = publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert first == second
    entries = sorted(p.name for p in first.dest_dir.iterdir())
    assert entries == ["summary.json", "summary.md"]
    assert first.summary_json_path.read_bytes() == second.summary_json_path.read_bytes()
    assert first.summary_md_path.read_bytes() == second.summary_md_path.read_bytes()


def test_publish_leaves_unrelated_files_in_the_destination_untouched(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    _write_valid_summary_files(store)
    docs_root = _docs_root(tmp_path)
    dest_dir = docs_root / store.manifest().run_id
    dest_dir.mkdir(parents=True)
    (dest_dir / "notes.txt").write_text("keep me", encoding="utf-8")

    result = publish.publish_run(store.run_dir, docs_results_root=docs_root)

    entries = sorted(p.name for p in result.dest_dir.iterdir())
    assert entries == ["notes.txt", "summary.json", "summary.md"]
    assert (result.dest_dir / "notes.txt").read_text(encoding="utf-8") == "keep me"


# --- summary.json / summary.md がそろっていない -------------------------------


def test_missing_summary_json_is_refused_and_creates_nothing(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    _write_summary_md(store)  # summary.json は書かない
    docs_root = _docs_root(tmp_path)

    with pytest.raises(publish.PublishError, match="summarize"):
        publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert not docs_root.exists()


def test_missing_summary_md_is_refused_and_creates_nothing(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    summary = _summary_for(store.manifest())
    _write_summary_json(store, summary.model_dump_json().encode("utf-8"))  # summary.md は書かない
    docs_root = _docs_root(tmp_path)

    with pytest.raises(publish.PublishError, match="summarize"):
        publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert not docs_root.exists()


# --- summary.json が types.Summary として検証できない --------------------------


def test_summary_json_with_an_extra_field_is_refused(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    payload = json.loads(_summary_for(store.manifest()).model_dump_json())
    payload["not_a_real_field"] = True
    _write_summary_json(store, json.dumps(payload).encode("utf-8"))
    _write_summary_md(store)
    docs_root = _docs_root(tmp_path)

    with pytest.raises(publish.PublishError):
        publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert not docs_root.exists()


def test_summary_json_with_a_wrong_type_is_refused(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    payload = json.loads(_summary_for(store.manifest()).model_dump_json())
    payload["results"] = "not-a-list"
    _write_summary_json(store, json.dumps(payload).encode("utf-8"))
    _write_summary_md(store)
    docs_root = _docs_root(tmp_path)

    with pytest.raises(publish.PublishError):
        publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert not docs_root.exists()


def test_truncated_summary_json_is_refused(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    text = _summary_for(store.manifest()).model_dump_json()
    _write_summary_json(store, text.encode("utf-8")[: len(text) // 2])
    _write_summary_md(store)
    docs_root = _docs_root(tmp_path)

    with pytest.raises(publish.PublishError):
        publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert not docs_root.exists()


def test_summary_json_with_a_mismatched_run_id_is_refused(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    other_conditions = _manifest(run_id="20260101T000000Z-other-000001")
    assert other_conditions.run_id != store.manifest().run_id
    summary = _summary_for(other_conditions)
    _write_summary_json(store, summary.model_dump_json().encode("utf-8"))
    _write_summary_md(store)
    docs_root = _docs_root(tmp_path)

    with pytest.raises(publish.PublishError, match="run_id"):
        publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert not docs_root.exists()


# --- シンボリックリンクは拒む (sentinel を持ち出す経路を塞ぐ) --------------------


def test_summary_json_as_a_symlink_to_trials_jsonl_is_refused(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    _write_summary_md(store)
    os.symlink(store.run_dir / "trials.jsonl", store.run_dir / "summary.json")
    docs_root = _docs_root(tmp_path)

    with pytest.raises(publish.PublishError, match="symlink|リンク"):
        publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert not docs_root.exists()


def test_summary_md_as_a_symlink_to_trials_jsonl_is_refused(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    summary = _summary_for(store.manifest())
    _write_summary_json(store, summary.model_dump_json().encode("utf-8"))
    os.symlink(store.run_dir / "trials.jsonl", store.run_dir / "summary.md")
    docs_root = _docs_root(tmp_path)

    with pytest.raises(publish.PublishError, match="symlink|リンク"):
        publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert not docs_root.exists()


# --- ハードリンクも拒む (独立レビュー 1 回目の指摘 3。名前を変えて同じ中身を ---
# --- 指すファイルは、シンボリックリンクの確認だけでは防げない) -------------------


def test_summary_md_as_a_hardlink_to_trials_jsonl_is_refused(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    summary = _summary_for(store.manifest())
    _write_summary_json(store, summary.model_dump_json().encode("utf-8"))
    os.link(store.run_dir / "trials.jsonl", store.run_dir / "summary.md")
    docs_root = _docs_root(tmp_path)

    with pytest.raises(publish.PublishError, match="hardlink|ハードリンク"):
        publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert not docs_root.exists()
    # sentinel を含む中身が、公開の場所のどこにも現れていない (docs_root 自体が
    # 作られていないので、この確認は上の assert と合わせて二重に効く)
    if docs_root.exists():
        for path in docs_root.rglob("*"):
            if path.is_file():
                assert SENTINEL not in path.read_text(encoding="utf-8")


def test_summary_json_as_a_hardlink_to_trials_jsonl_is_refused(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    _write_summary_md(store)
    os.link(store.run_dir / "trials.jsonl", store.run_dir / "summary.json")
    docs_root = _docs_root(tmp_path)

    with pytest.raises(publish.PublishError, match="hardlink|ハードリンク"):
        publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert not docs_root.exists()


# --- 行き先が公開の場所の内側に収まることの確認 (独立レビュー 1 回目の指摘、必須 1) --


def test_pre_existing_symlink_at_the_destination_escaping_the_root_is_refused(
    tmp_path: Path,
) -> None:
    """`<公開の場所>/<run_id>` に、公開の場所の外を指すシンボリックリンクが
    あらかじめ置かれていても、そこには何も書かない。`run_id` の確認は文字列しか
    見ないので、`_safe_dest_dir` の `is_relative_to` の確認だけがこれを防ぐ
    (この確認をまるごと外すと、このテストだけが落ちる)。
    """
    store = _make_run(tmp_path)
    _write_valid_summary_files(store)
    docs_root = _docs_root(tmp_path)
    docs_root.mkdir(parents=True)
    outside = tmp_path / "outside-the-public-root"
    outside.mkdir()
    os.symlink(outside, docs_root / store.manifest().run_id)

    with pytest.raises(publish.PublishError):
        publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert list(outside.iterdir()) == []


# --- run_id を経路の要素として使ってよいかの確認 (RunStore を経由せず、直接試す) ---


@pytest.mark.parametrize(
    "hostile",
    ["../x", "a/b", ".hidden", "", "..", "x" * 201, "a\nb", "a\rb", "\x00"],
    ids=[
        "dotdot-slash",
        "slash",
        "leading-dot",
        "empty",
        "dotdot",
        "too-long",
        "newline",
        "carriage-return",
        "nul",
    ],
)
def test_validate_run_id_component_rejects_hostile_values(hostile: str) -> None:
    with pytest.raises(publish.PublishError):
        publish._validate_run_id_component(hostile)  # noqa: SLF001 -- 検証する関数そのものを試す


def test_validate_run_id_component_accepts_a_run_id_shaped_value() -> None:
    publish._validate_run_id_component("20260919T120000Z-candidate-d-ab12cd")  # 例外を投げない


def test_validate_run_id_component_accepts_the_length_upper_bound() -> None:
    publish._validate_run_id_component("x" * 200)  # 例外を投げない (200 文字ちょうど)


# --- 公開の場所の既定値 (パスだけを確かめ、実際のリポジトリには公開しない) --------


def test_default_docs_results_root_resolves_to_the_repository_docs_results() -> None:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    )
    expected = Path(proc.stdout.strip()) / "docs" / "results"

    assert publish.default_docs_results_root() == expected


def test_default_docs_results_root_raises_when_git_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """独立レビュー 1 回目の指摘、必須 2 (a)。`Path.cwd()` へは決して倒れない。"""
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))

    with pytest.raises(publish.PublishError):
        publish.default_docs_results_root()


def test_default_docs_results_root_raises_when_git_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """独立レビュー 1 回目の指摘、必須 2 (b)。`Path.cwd()` へは決して倒れない。"""
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))

    with pytest.raises(publish.PublishError):
        publish.default_docs_results_root()


# --- 未完了の計測ランでも公開でき、incomplete を報告する ------------------------


def test_incomplete_summary_publishes_and_reports_incomplete(tmp_path: Path) -> None:
    store = _make_run(tmp_path)
    _write_valid_summary_files(store, incomplete=True)
    docs_root = _docs_root(tmp_path)

    result = publish.publish_run(store.run_dir, docs_results_root=docs_root)

    assert result.incomplete is True


# --- 依存の向き: publish.py は許した集まりの外を読み込まない --------------------

_ALLOWED_STDLIB_ROOTS = frozenset(
    {"__future__", "json", "os", "subprocess", "tempfile", "pathlib", "typing"}
)
_ALLOWED_BENCH_HARNESS_MODULES = frozenset({"bench_harness.store", "bench_harness.types"})


def test_publish_module_imports_only_the_allowed_set() -> None:
    tree = ast.parse(inspect.getsource(publish))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root in _ALLOWED_STDLIB_ROOTS | {"pydantic"}, (
                    f"許していない読み込み: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "相対の読み込みがある"
            assert node.module is not None
            module = node.module
            root = module.split(".")[0]
            if root == "bench_harness":
                assert module in _ALLOWED_BENCH_HARNESS_MODULES, f"許していない読み込み: {module}"
            else:
                assert root in _ALLOWED_STDLIB_ROOTS | {"pydantic"}, (
                    f"許していない読み込み: {module}"
                )
