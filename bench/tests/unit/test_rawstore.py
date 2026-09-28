"""生データの保存と読み戻し (`store/rawstore.py`, task 2.3) の試験。

確かめること (タスク 2.3 の完了の状態):

- 書いている途中でプロセスを止めても、それまでの試行が読み戻せる (サブプロセスを
  `os._exit` で強制終了して確かめる)
- 同じ本文を 2 回保存してもファイルは 1 つ (バイト単位で同じ)
- 保存したどのファイルにも認証の情報の値が現れない (1.8)

そのほか、design.md と Implementation Notes、独立レビュー (1 回目) の指摘に沿って
確かめること: 重ならない計測ランの識別子 (1.5、長い対象サーバー名でも `OSError`
にならない)、`results_root` の git の検査が最初に走ること (8.5, fail closed)、
manifest の別名書き→置き換え (atomic write) と状態の遷移 (終わりの状態 3 つ ×
行き先 4 つを全部拒否)、`trials.jsonl` **と** `metrics/deltas.jsonl` の両方で
末尾破損の扱いが同じであること、「JSON として読めない最後の行 (捨てて警告)」と
「JSON としては読めるのに検証に落ちる行 (最後でも途中でもハードエラー)」を区別
すること、`read_trials()` / `read_metric_deltas()` (レコードと警告を一緒に返す、
分析が使う主な口)、本文のハッシュの検証と gzip の mtime が固定されていることの
直接の確認、条件の鍵とファイル名の可逆な対応。

`RunStore.create()` は `config.resolve_results_root()` を通すので、既定では
git のリポジトリの外にある `tmp_path` を使う (最初の試験でその前提を確かめる)。
git の管理下であることを確かめる試験だけ、その場で `git init` したリポジトリを使う。
"""

from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl, JsonValue, ValidationError

from bench_harness import config as c
from bench_harness.store.rawstore import (
    RunStore,
    StoreError,
    decode_condition_filename,
    encode_condition_filename,
    list_run_dirs,
    new_run_id,
)
from bench_harness.types import (
    DatasetRef,
    DerivedMetrics,
    MetricSnapshot,
    MetricsUnavailable,
    Profile,
    RunManifest,
    RunStatus,
    SkippedCondition,
    StreamResult,
    StreamTiming,
    SuiteName,
    TargetDef,
    TrialRecord,
)

AT = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)


# --- 組み立ての助け ---------------------------------------------------------


def _target(**overrides: Any) -> TargetDef:
    payload: dict[str, Any] = {
        "name": "candidate-d",
        "base_url": HttpUrl("http://localhost:8000"),
        "model": "glm-5.3-flash",
    }
    payload.update(overrides)
    return TargetDef.model_validate(payload)


def _manifest(
    *,
    run_id: str | None = None,
    target: TargetDef | None = None,
    status: RunStatus = RunStatus.RUNNING,
) -> RunManifest:
    return RunManifest(
        run_id=run_id or new_run_id("candidate-d", AT),
        status=status,
        target=target or _target(),
        started_at=AT,
        suites=[SuiteName.DECODE],
        profile_name="quick",
        profile=Profile(name="quick"),
        harness_version="0.1.0+test",
    )


def _create_store(tmp_path: Path, **manifest_overrides: Any) -> RunStore:
    return RunStore.create(tmp_path / "results", _manifest(**manifest_overrides))


def _trial_record(store: RunStore, index: int, *, condition: str = "decode/code/en") -> TrialRecord:
    return TrialRecord(
        run_id=store.manifest().run_id,
        suite=SuiteName.DECODE,
        condition=condition,
        trial_index=index,
        request_body_ref="0" * 64,
        result=StreamResult(
            timing=StreamTiming(sent_at_utc=AT, sent_at_ns=index, end_ns=index + 1),
        ),
    )


def _snapshot(raw_text: str) -> MetricSnapshot:
    return MetricSnapshot(taken_at_utc=AT, raw_text=raw_text, values={}, missing=[])


# --- この試験ファイルの前提 --------------------------------------------------


def test_tmp_path_fixture_is_outside_any_git_repository(tmp_path: Path) -> None:
    """この試験ファイルの大半は、`tmp_path` が git のリポジトリの外にあることを前提にする。"""
    proc = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, f"tmp_path が git のリポジトリの中にある: {tmp_path}"


# --- 1.5 計測ランの識別子 ----------------------------------------------------


def test_new_run_id_matches_the_expected_shape() -> None:
    run_id = new_run_id("candidate-d", AT)

    assert re.fullmatch(r"20260919T120000Z-candidate-d-[a-z0-9]{6}", run_id)


def test_new_run_id_sanitizes_unsafe_characters() -> None:
    run_id = new_run_id("weird name/with:chars?", AT)

    assert "/" not in run_id
    assert ":" not in run_id
    assert "?" not in run_id
    assert " " not in run_id


def test_new_run_id_is_unique_within_the_same_second_for_the_same_target() -> None:
    ids = {new_run_id("candidate-d", AT) for _ in range(300)}

    assert len(ids) == 300


def test_new_run_id_truncates_long_target_names(tmp_path: Path) -> None:
    """整えたあとの名前が長すぎると、`RunStore.create()` の `mkdir` が生の
    `OSError` (ファイル名が長すぎる) になる。切り詰めても、乱数の部分で一意性は
    保たれる (レビュー指摘 6)。
    """
    run_id = new_run_id("x" * 300, AT)

    assert len(run_id) < 200  # ファイル名の長さの上限に余裕を残す
    store = RunStore.create(tmp_path / "results", _manifest(run_id=run_id))
    assert store.run_dir.name == run_id


# --- create() / open(): git の検査が最初、schema_version の拒否 --------------


def test_create_verifies_results_root_before_creating_anything(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    results_root = repo / "results"
    manifest = _manifest()

    with pytest.raises(c.ConfigError):
        RunStore.create(results_root, manifest)

    assert not results_root.exists()
    assert not (repo / manifest.run_id).exists()


def test_create_accepts_a_results_root_outside_any_repo(tmp_path: Path) -> None:
    store = _create_store(tmp_path)

    assert store.run_dir.is_dir()
    assert (store.run_dir / "manifest.json").is_file()
    assert (store.run_dir / "trials.jsonl").is_file()
    assert (store.run_dir / "bodies").is_dir()
    assert (store.run_dir / "metrics").is_dir()
    assert (store.run_dir / "metrics" / "deltas.jsonl").is_file()


def test_create_rejects_an_already_existing_run_directory(tmp_path: Path) -> None:
    manifest = _manifest()
    RunStore.create(tmp_path / "results", manifest)

    with pytest.raises(StoreError):
        RunStore.create(tmp_path / "results", manifest)


def test_create_rejects_a_manifest_that_is_not_running(tmp_path: Path) -> None:
    manifest = _manifest(status=RunStatus.COMPLETED)

    with pytest.raises(StoreError):
        RunStore.create(tmp_path / "results", manifest)

    assert not (tmp_path / "results").exists()


def test_open_round_trips_the_manifest(tmp_path: Path) -> None:
    store = _create_store(tmp_path)

    reopened = RunStore.open(store.run_dir)

    assert reopened.manifest() == store.manifest()


def test_legacy_manifest_without_retokenization_settings_remains_readable(tmp_path: Path) -> None:
    store = _create_store(tmp_path)
    manifest_path = store.run_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.pop("output_retokenization", None)
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    reopened = RunStore.open(store.run_dir)
    assert reopened.manifest().schema_version == data["schema_version"]
    assert reopened.manifest().model_dump().get("output_retokenization") is None


def test_legacy_manifest_is_not_silently_treated_as_explicit_retokenization_off(
    tmp_path: Path,
) -> None:
    store = _create_store(tmp_path)
    data = json.loads((store.run_dir / "manifest.json").read_text(encoding="utf-8"))
    data.pop("output_retokenization", None)
    old = RunManifest.model_validate(data)
    explicitly_off = RunManifest.model_validate(
        {**data, "output_retokenization": {"enabled": False}}
    )

    assert old.model_dump().get("output_retokenization") != explicitly_off.model_dump().get(
        "output_retokenization"
    )


def test_open_missing_manifest_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"
    missing.mkdir()

    with pytest.raises(StoreError):
        RunStore.open(missing)


def test_open_rejects_a_manifest_with_a_different_schema_version(tmp_path: Path) -> None:
    store = _create_store(tmp_path)
    manifest_path = store.run_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["schema_version"] = 999
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(StoreError) as exc_info:
        RunStore.open(store.run_dir)

    assert "999" in str(exc_info.value)


# --- manifest: 別名書き→置き換え、状態の遷移、再検証 ---------------------------


def test_manifest_update_leaves_no_temp_files(tmp_path: Path) -> None:
    store = _create_store(tmp_path)

    store.set_status(RunStatus.COMPLETED, finished_at=AT)

    leftovers = [p for p in store.run_dir.iterdir() if p.name.startswith(".manifest.json.")]
    assert leftovers == []


def test_manifest_write_failure_leaves_the_old_manifest_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _create_store(tmp_path)
    original_bytes = (store.run_dir / "manifest.json").read_bytes()

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated disk failure")

    # `rawstore` は `os.replace` を module 単位で参照するので、標準の `os` を直接
    # 差し替えれば、そこから見える呼び出しも同じように失敗する。
    monkeypatch.setattr(os, "replace", _boom)

    with pytest.raises(OSError):
        store.set_status(RunStatus.COMPLETED, finished_at=AT)

    assert store.manifest().status == RunStatus.RUNNING  # メモリ上も変わっていない
    assert (store.run_dir / "manifest.json").read_bytes() == original_bytes
    leftovers = [p for p in store.run_dir.iterdir() if p.name.startswith(".manifest.json.")]
    assert leftovers == []


@pytest.mark.parametrize(
    "target_status", [RunStatus.COMPLETED, RunStatus.ABORTED, RunStatus.INTERRUPTED]
)
def test_set_status_allows_the_three_terminal_transitions(
    tmp_path: Path, target_status: RunStatus
) -> None:
    store = _create_store(tmp_path)

    store.set_status(target_status, finished_at=AT)

    assert store.manifest().status is target_status
    assert RunStore.open(store.run_dir).manifest().status is target_status


_TERMINAL_STATUSES = [RunStatus.COMPLETED, RunStatus.ABORTED, RunStatus.INTERRUPTED]


@pytest.mark.parametrize("current_status", _TERMINAL_STATUSES)
@pytest.mark.parametrize("target_status", list(RunStatus))
def test_set_status_rejects_every_transition_out_of_a_terminal_state(
    tmp_path: Path, current_status: RunStatus, target_status: RunStatus
) -> None:
    """終わりの状態 3 つ × 行き先の状態 4 つ (`running` を含む) を、全部拒否されると
    確かめる。狭い試験 (`completed -> aborted` だけ) だと、`_ALLOWED_TRANSITIONS`
    の `COMPLETED` に `RUNNING` を足す変異が生き残ってしまう (レビュー指摘 5)。
    """
    store = _create_store(tmp_path)
    store.set_status(current_status, finished_at=AT)

    with pytest.raises(StoreError):
        store.set_status(target_status, finished_at=AT)


def test_set_status_rejects_running_to_running(tmp_path: Path) -> None:
    store = _create_store(tmp_path)

    with pytest.raises(StoreError):
        store.set_status(RunStatus.RUNNING, finished_at=AT)


def test_update_manifest_revalidates_and_rejects_bad_data(tmp_path: Path) -> None:
    store = _create_store(tmp_path)

    with pytest.raises(StoreError):
        store.update_manifest(status="not-a-real-status")

    assert store.manifest().status == RunStatus.RUNNING  # 拒否のあとも変わっていない


def test_update_manifest_appends_warnings_skipped_and_datasets(tmp_path: Path) -> None:
    store = _create_store(tmp_path)
    dataset = DatasetRef(
        name="HumanEval+",
        version="v1",
        source_url="https://example.invalid",
        license="Apache-2.0",
        scoring_method="pass@1",
        sha256="0" * 64,
    )
    skipped = SkippedCondition(
        suite=SuiteName.PREFILL, key="prefill/cold/128k", reason="context_limit"
    )

    updated = store.update_manifest(
        warnings=["running requests were 1 before start"],
        skipped=[skipped],
        datasets=[dataset],
    )

    assert updated.warnings == ["running requests were 1 before start"]
    assert updated.skipped == [skipped]
    assert updated.datasets == [dataset]
    reopened = RunStore.open(store.run_dir)
    assert reopened.manifest().datasets == [dataset]


# --- 試行のレコード: 追記、読み戻し、末尾破損、途中破損 -------------------------


def test_append_trial_and_iter_trials_round_trip(tmp_path: Path) -> None:
    store = _create_store(tmp_path)
    records = [_trial_record(store, i) for i in range(5)]
    for record in records:
        store.append_trial(record)

    read_back = list(store.iter_trials())

    assert read_back == records


def test_iter_trials_on_an_empty_run_yields_nothing(tmp_path: Path) -> None:
    store = _create_store(tmp_path)

    assert list(store.iter_trials()) == []


def test_iter_trials_reads_a_complete_last_line_without_a_trailing_newline(
    tmp_path: Path,
) -> None:
    store = _create_store(tmp_path)
    r0 = _trial_record(store, 0)
    r1 = _trial_record(store, 1)
    store.append_trial(r0)
    with (store.run_dir / "trials.jsonl").open("ab") as fh:
        fh.write(r1.model_dump_json().encode("utf-8"))  # わざと改行を付けない

    read_back = list(store.iter_trials())

    assert read_back == [r0, r1]


def test_iter_trials_skips_a_truncated_last_line_and_warns(tmp_path: Path) -> None:
    store = _create_store(tmp_path)
    good = _trial_record(store, 0)
    store.append_trial(good)
    with (store.run_dir / "trials.jsonl").open("ab") as fh:
        fh.write(b'{"schema_version": 1, "run_id": "trunc')  # 書いている途中で切れた跡

    iterator = store.iter_trials()
    read_back = list(iterator)

    assert read_back == [good]
    assert len(iterator.warnings) == 1
    assert "trials.jsonl" in iterator.warnings[0]


def test_iter_trials_raises_on_a_corrupt_middle_line_naming_the_line_number(
    tmp_path: Path,
) -> None:
    store = _create_store(tmp_path)
    store.append_trial(_trial_record(store, 0))
    with (store.run_dir / "trials.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
    store.append_trial(_trial_record(store, 1))

    with pytest.raises(StoreError) as exc_info:
        list(store.iter_trials())

    assert "2" in str(exc_info.value)


def test_iter_trials_raises_when_the_last_line_is_valid_json_but_fails_validation(
    tmp_path: Path,
) -> None:
    """「途中で切れた」と見なして捨ててよいのは、最後の行が JSON として読めない
    場合だけ。JSON としては完全に読めるのに `TrialRecord` の検証に落ちる行は、
    本物の破損か版の食い違いなので、最後の行でもハードエラーにする (レビュー指摘
    必須 2)。
    """
    store = _create_store(tmp_path)
    good = _trial_record(store, 0)
    store.append_trial(good)
    with (store.run_dir / "trials.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema_version": 1, "bogus_field": "boom"}) + "\n")

    with pytest.raises(StoreError) as exc_info:
        list(store.iter_trials())

    assert "2" in str(exc_info.value)


def test_read_trials_returns_records_and_warnings_together(tmp_path: Path) -> None:
    """分析 (4.1〜4.3) が使う主な口。読み切ったレコードと、捨てた行の警告を、
    一緒に返す (レビュー指摘 必須 3)。
    """
    store = _create_store(tmp_path)
    good = _trial_record(store, 0)
    store.append_trial(good)
    with (store.run_dir / "trials.jsonl").open("ab") as fh:
        fh.write(b'{"schema_version": 1, "run_id": "trunc')

    records, warnings = store.read_trials()

    assert records == [good]
    assert len(warnings) == 1
    assert "trials.jsonl" in warnings[0]


def test_read_trials_on_an_empty_run_returns_empty(tmp_path: Path) -> None:
    store = _create_store(tmp_path)

    records, warnings = store.read_trials()

    assert records == []
    assert warnings == []


_CRASH_SCRIPT = """
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from bench_harness.store.rawstore import RunStore
from bench_harness.types import StreamResult, StreamTiming, SuiteName, TrialRecord

run_dir = Path(sys.argv[1])
n = int(sys.argv[2])
store = RunStore.open(run_dir)
run_id = store.manifest().run_id
for i in range(n):
    record = TrialRecord(
        run_id=run_id,
        suite=SuiteName.DECODE,
        condition="decode/code/en",
        trial_index=i,
        request_body_ref="0" * 64,
        result=StreamResult(
            timing=StreamTiming(sent_at_utc=datetime.now(UTC), sent_at_ns=i, end_ns=i + 1),
        ),
    )
    store.append_trial(record)
os._exit(0)  # 何も close せず、後片づけもせずに終わる (SIGKILL 相当)
"""


def test_trials_survive_a_hard_kill_in_a_subprocess(tmp_path: Path) -> None:
    store = _create_store(tmp_path)
    n = 25

    result = subprocess.run(
        [sys.executable, "-c", _CRASH_SCRIPT, str(store.run_dir), str(n)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    reopened = RunStore.open(store.run_dir)
    trials = list(reopened.iter_trials())
    assert [t.trial_index for t in trials] == list(range(n))


# --- 送った要求の本文: 重複の除去とハッシュの検証 -------------------------------


def test_put_body_dedupes_identical_bodies(tmp_path: Path) -> None:
    store = _create_store(tmp_path)
    body: dict[str, JsonValue] = {"model": "m", "max_tokens": 10, "messages": []}

    ref1 = store.put_body(body)
    bytes1 = (store.run_dir / "bodies" / f"{ref1}.json.gz").read_bytes()
    ref2 = store.put_body(dict(body))  # 別の dict インスタンス、同じ中身
    bytes2 = (store.run_dir / "bodies" / f"{ref2}.json.gz").read_bytes()

    assert ref1 == ref2
    assert bytes1 == bytes2
    assert len(list((store.run_dir / "bodies").iterdir())) == 1


def test_get_body_round_trips(tmp_path: Path) -> None:
    store = _create_store(tmp_path)
    body: dict[str, JsonValue] = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }

    ref = store.put_body(body)
    restored = store.get_body(ref)

    assert restored == body


def test_put_body_round_trips_a_roughly_500kb_body(tmp_path: Path) -> None:
    store = _create_store(tmp_path)
    big_text = "あ" * 500_000
    body: dict[str, JsonValue] = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": big_text}],
    }

    ref = store.put_body(body)
    restored = store.get_body(ref)

    assert restored == body


def test_get_body_detects_a_tampered_file(tmp_path: Path) -> None:
    store = _create_store(tmp_path)
    ref = store.put_body({"model": "m"})
    path = store.run_dir / "bodies" / f"{ref}.json.gz"
    path.write_bytes(gzip.compress(b'{"model": "tampered"}', mtime=0))

    with pytest.raises(StoreError):
        store.get_body(ref)


def test_get_body_missing_ref_raises(tmp_path: Path) -> None:
    store = _create_store(tmp_path)

    with pytest.raises(StoreError):
        store.get_body("0" * 64)


def test_put_body_gzip_header_has_a_fixed_mtime(tmp_path: Path) -> None:
    """gzip の決定性を直接確かめる。ヘッダーの 4〜7 バイト目 (0-indexed、リトル
    エンディアンの符号なし 32bit) が mtime で、固定 (0) でなければ、同じ本文でも
    毎回違うバイト列になる (レビュー指摘 4)。
    """
    store = _create_store(tmp_path)

    ref = store.put_body({"model": "m"})
    header = (store.run_dir / "bodies" / f"{ref}.json.gz").read_bytes()[:10]

    assert header[:2] == b"\x1f\x8b"  # gzip の magic number
    assert header[4:8] == b"\x00\x00\x00\x00"


# --- 内部の指標: 加工前のテキスト、得られなかった側、増分の行 ---------------------


def test_write_metrics_stores_raw_text_verbatim(tmp_path: Path) -> None:
    store = _create_store(tmp_path)
    before = _snapshot("# before\nfoo 1\n")
    after = _snapshot("# after\nfoo 2\n")

    store.write_metrics("prefill/cold/32k", before, after, DerivedMetrics())

    encoded = encode_condition_filename("prefill/cold/32k")
    before_path = store.run_dir / "metrics" / f"{encoded}.before.prom"
    after_path = store.run_dir / "metrics" / f"{encoded}.after.prom"
    assert before_path.read_text(encoding="utf-8") == before.raw_text
    assert after_path.read_text(encoding="utf-8") == after.raw_text


def test_write_metrics_records_an_unavailable_side_and_writes_nothing_for_it(
    tmp_path: Path,
) -> None:
    store = _create_store(tmp_path)
    before: MetricSnapshot | MetricsUnavailable = MetricsUnavailable(reason="/metrics が 404")
    after = _snapshot("foo 1\n")

    store.write_metrics("decode/code/en", before, after, DerivedMetrics())

    encoded = encode_condition_filename("decode/code/en")
    assert not (store.run_dir / "metrics" / f"{encoded}.before.prom").exists()
    assert (store.run_dir / "metrics" / f"{encoded}.after.prom").exists()

    rows = list(store.iter_metric_deltas())
    assert len(rows) == 1
    assert rows[0].before_available is False
    assert rows[0].before_unavailable_reason == "/metrics が 404"
    assert rows[0].after_available is True
    assert rows[0].after_unavailable_reason is None


def test_metric_deltas_round_trip_in_order(tmp_path: Path) -> None:
    store = _create_store(tmp_path)
    before = _snapshot("foo 1\n")
    after = _snapshot("foo 2\n")
    derived = DerivedMetrics(kv_usage_peak=0.5)

    store.write_metrics("decode/code/en", before, after, derived)
    store.write_metrics("decode/code/ja", before, after, derived)

    rows = list(store.iter_metric_deltas())

    assert [row.condition for row in rows] == ["decode/code/en", "decode/code/ja"]
    assert rows[0].derived == derived


def test_iter_metric_deltas_on_an_empty_run_yields_nothing(tmp_path: Path) -> None:
    store = _create_store(tmp_path)

    assert list(store.iter_metric_deltas()) == []


def test_iter_metric_deltas_skips_a_truncated_last_line_and_warns(tmp_path: Path) -> None:
    """`write_metrics` も `append_trial` と同じ `_append_line` で書くので、書いて
    いる途中でプロセスが落ちる危険は同じ。`trials.jsonl` と同じ決まりにする
    (レビュー指摘 必須 1)。
    """
    store = _create_store(tmp_path)
    store.write_metrics(
        "decode/code/en", _snapshot("foo 1\n"), _snapshot("foo 2\n"), DerivedMetrics()
    )
    with (store.run_dir / "metrics" / "deltas.jsonl").open("ab") as fh:
        fh.write(b'{"schema_version": 1, "condition": "trunc')  # 途中で切れた跡

    iterator = store.iter_metric_deltas()
    rows = list(iterator)

    assert len(rows) == 1
    assert len(iterator.warnings) == 1
    assert "deltas.jsonl" in iterator.warnings[0]


def test_iter_metric_deltas_raises_on_a_corrupt_middle_line_naming_the_line_number(
    tmp_path: Path,
) -> None:
    store = _create_store(tmp_path)
    store.write_metrics(
        "decode/code/en", _snapshot("foo 1\n"), _snapshot("foo 2\n"), DerivedMetrics()
    )
    with (store.run_dir / "metrics" / "deltas.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
    store.write_metrics(
        "decode/code/ja", _snapshot("foo 1\n"), _snapshot("foo 2\n"), DerivedMetrics()
    )

    with pytest.raises(StoreError) as exc_info:
        list(store.iter_metric_deltas())

    assert "2" in str(exc_info.value)


def test_iter_metric_deltas_raises_when_the_last_line_is_valid_json_but_fails_validation(
    tmp_path: Path,
) -> None:
    """JSON としては読めるのに `MetricDeltaRecord` の検証に落ちる最後の行は、
    途中で切れた行ではなくハードエラー (レビュー指摘 必須 2、deltas 側)。
    """
    store = _create_store(tmp_path)
    store.write_metrics(
        "decode/code/en", _snapshot("foo 1\n"), _snapshot("foo 2\n"), DerivedMetrics()
    )
    with (store.run_dir / "metrics" / "deltas.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema_version": 1, "bogus_field": True}) + "\n")

    with pytest.raises(StoreError) as exc_info:
        list(store.iter_metric_deltas())

    assert "2" in str(exc_info.value)


def test_read_metric_deltas_returns_records_and_warnings_together(tmp_path: Path) -> None:
    """分析 (4.1) が使う主な口 (レビュー指摘 必須 3)。"""
    store = _create_store(tmp_path)
    store.write_metrics(
        "decode/code/en", _snapshot("foo 1\n"), _snapshot("foo 2\n"), DerivedMetrics()
    )
    with (store.run_dir / "metrics" / "deltas.jsonl").open("ab") as fh:
        fh.write(b'{"schema_version": 1, "condition": "trunc')

    rows, warnings = store.read_metric_deltas()

    assert len(rows) == 1
    assert len(warnings) == 1
    assert "deltas.jsonl" in warnings[0]


def test_read_metric_deltas_on_an_empty_run_returns_empty(tmp_path: Path) -> None:
    store = _create_store(tmp_path)

    rows, warnings = store.read_metric_deltas()

    assert rows == []
    assert warnings == []


@pytest.mark.parametrize(
    "condition",
    ["prefill/cold/32k", "decode/code/en", "agent/stage/020k", "a/b", "a%2Fb", "plain"],
)
def test_condition_filename_encoding_round_trips(condition: str) -> None:
    encoded = encode_condition_filename(condition)

    assert "/" not in encoded
    assert decode_condition_filename(encoded) == condition


def test_condition_filename_encoding_has_no_collisions() -> None:
    conditions = ["prefill/cold/32k", "decode/code/en", "agent/stage/020k", "a/b", "a%2Fb"]

    encoded = {encode_condition_filename(c) for c in conditions}

    assert len(encoded) == len(conditions)


# --- 計測ランの一覧 ----------------------------------------------------------


def test_list_run_dirs_returns_only_directories_with_a_manifest(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    store_a = RunStore.create(results_root, _manifest())
    store_b = RunStore.create(results_root, _manifest())
    (results_root / "not-a-run").mkdir()

    found = list_run_dirs(results_root)

    assert set(found) == {store_a.run_dir, store_b.run_dir}


def test_list_run_dirs_on_a_missing_root_returns_empty(tmp_path: Path) -> None:
    assert list_run_dirs(tmp_path / "does-not-exist") == []


# --- 1.8 認証の情報の値が、どのファイルにも現れない -----------------------------

_SENTINEL = "sentinel-9f3c7a1e-do-not-leak"


def test_no_file_under_the_run_dir_contains_the_api_key_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """保存の部品には認証の情報の値そのものが渡らない。`TargetDef` が持つのは
    環境変数の名前だけで、値は環境変数に置いたまま (実際に読むのは `config.resolve_api_key`
    の仕事で、この module は呼ばない)。計測ランのすべてのファイルを走査して確かめる。
    """
    monkeypatch.setenv("BENCH_TEST_STORE_API_KEY", _SENTINEL)
    target = _target(api_key_env="BENCH_TEST_STORE_API_KEY")
    store = RunStore.create(tmp_path / "results", _manifest(target=target))

    body: dict[str, JsonValue] = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    store.put_body(body)
    store.append_trial(_trial_record(store, 0))
    store.write_metrics(
        "decode/code/en", _snapshot("foo 1\n"), _snapshot("foo 2\n"), DerivedMetrics()
    )
    store.update_manifest(warnings=["started with BENCH_TEST_STORE_API_KEY set"])
    store.set_status(RunStatus.COMPLETED, finished_at=AT)

    scanned = 0
    for path in store.run_dir.rglob("*"):
        if not path.is_file():
            continue
        scanned += 1
        content = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        assert _SENTINEL.encode("utf-8") not in content, f"{path} に認証の情報が漏れている"
    assert scanned > 0


# --- MetricDeltaRecord / manifest の再検証が pydantic の ValidationError を包む -----


def test_update_manifest_wraps_validation_error_as_store_error(tmp_path: Path) -> None:
    store = _create_store(tmp_path)

    try:
        store.update_manifest(status="bogus")
    except StoreError as exc:
        assert isinstance(exc.__cause__, ValidationError)
    else:
        pytest.fail("StoreError が発生しなかった")
