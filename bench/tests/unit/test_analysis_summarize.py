"""速さの要約 (`analysis/summarize.py`, task 4.1) の試験。

生データは、`RunStore.create` と手で組み立てた `TrialRecord` で作る (置き場所は
`tmp_path` の下で、git のリポジトリの外)。時刻とトークン数は試行ごとにすべて
違う値にしてあるので、式の変数を取り違える壊れ方 (`− 1` の抜け、分母と分子の
入れ替え、最初と最後の時刻の入れ替え) が、期待の値との食い違いとして出る
(Implementation Notes 2.2「式を確かめる試験では、関係する量をすべて違う値に
する」)。

確かめること (tasks.md 4.1 の完了の状態):

- 用意した生データの例から、2 つの要約 (`summary.json` と `summary.md`) ができる
- どちらにも、送った内容と応答の本文の目印の文字列が含まれない (8.3)
- 主な結果と参考が分かれて表示される (4.4)

そのほか、要件 2.3〜2.6、3.7、4.2、4.5、7.3、8.2、8.6、10.1、10.2、10.4、10.7 と、
Implementation Notes 2.3 (警告を落とさない)、3.4 (同時処理のまとめ方) に沿った
振る舞いを確かめる。
"""

from __future__ import annotations

import ast
import json
import math
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl, JsonValue

from bench_harness.analysis import summarize as summarize_module
from bench_harness.analysis.stats import describe
from bench_harness.analysis.summarize import summarize_run, trial_values, write_summary
from bench_harness.runner import execute_run
from bench_harness.store.rawstore import RunStore, StoreError
from bench_harness.suites.decode import DecodeSuite
from bench_harness.types import (
    ContentBlock,
    DerivedMetrics,
    LogicalMetric,
    MetricFlag,
    MetricResult,
    MetricSnapshot,
    MetricsUnavailable,
    Profile,
    RequestError,
    RunManifest,
    RunRequest,
    RunStatus,
    StreamResult,
    StreamTiming,
    SuiteName,
    Summary,
    TargetDef,
    Tier,
    TrialFlag,
    TrialRecord,
    Usage,
)
from fake_server import FakeServer, token_stream_response

AT = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)
AT_END = datetime(2026, 9, 19, 12, 30, 0, tzinfo=UTC)
RUN_ID = "20260919T120000Z-fake-abcdef"
SENTINEL_REQUEST = "SENTINEL-REQUEST-XYZZY"
"""送った本文だけに置く目印。要約に漏れてはいけない (8.3)。"""
SENTINEL_RESPONSE = "SENTINEL-RESPONSE-PLUGH"
"""応答の本文だけに置く目印。要約に漏れてはいけない (8.3)。"""


# --- 生データの組み立て -----------------------------------------------------


def _manifest(
    *,
    status: RunStatus = RunStatus.RUNNING,
    min_successes: int = 1,
    suites: Sequence[SuiteName] = (SuiteName.DECODE,),
    **overrides: Any,
) -> RunManifest:
    payload: dict[str, Any] = {
        "run_id": RUN_ID,
        "status": status,
        "target": TargetDef(
            name="fake",
            base_url=HttpUrl("http://127.0.0.1:8000"),
            model="glm-5.3-flash",
            notes="試験用の対象サーバー",
        ),
        "server_model": "glm-5.3-flash-server",
        "server_version": "0.0.0-test",
        "started_at": AT,
        "suites": list(suites),
        "profile_name": "test",
        "profile": Profile(name="test", min_successes=min_successes),
        "harness_version": "0.1.0+gtest",
        "context_limit": 32000,
    }
    payload.update(overrides)
    return RunManifest.model_validate(payload)


def _store(tmp_path: Path, **manifest_overrides: Any) -> RunStore:
    return RunStore.create(tmp_path / "results", _manifest(**manifest_overrides))


def _ns(seconds: float) -> int:
    """秒をナノ秒の整数にする (単調な時計の値のつもり)。"""
    return round(seconds * 1_000_000_000)


def _trial(
    *,
    condition: str,
    suite: SuiteName = SuiteName.DECODE,
    trial_index: int = 0,
    warmup: bool = False,
    tier: Tier = "primary",
    sent_s: float = 0.0,
    first_s: float | None = None,
    last_s: float | None = None,
    end_s: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int = 0,
    cache_read: int | None = None,
    cache_creation: int | None = None,
    failed: bool = False,
    flags: Sequence[TrialFlag] = (),
    round_id: int | None = None,
    stream_index: int | None = None,
    body_ref: str = "0" * 64,
    text: str = "",
) -> TrialRecord:
    """1 つの試行のレコードを、時刻とトークン数を直に指定して作る。"""
    usage = (
        None
        if input_tokens is None
        else Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
        )
    )
    tail = last_s if last_s is not None else (first_s if first_s is not None else sent_s)
    timing = StreamTiming(
        sent_at_utc=AT,
        sent_at_ns=_ns(sent_s),
        first_token_ns=None if first_s is None else _ns(first_s),
        last_token_ns=None if last_s is None else _ns(last_s),
        end_ns=_ns(end_s if end_s is not None else tail),
    )
    return TrialRecord(
        run_id=RUN_ID,
        suite=suite,
        condition=condition,
        tier=tier,
        trial_index=trial_index,
        warmup=warmup,
        round_id=round_id,
        stream_index=stream_index,
        request_body_ref=body_ref,
        result=StreamResult(
            timing=timing,
            usage=usage,
            stop_reason=None if failed else "max_tokens",
            blocks=[] if not text else [ContentBlock(type="text", text=text)],
            error=RequestError(kind="http", http_status=500, message="") if failed else None,
        ),
        flags=list(flags),
    )


def _append(store: RunStore, *records: TrialRecord) -> None:
    for record in records:
        store.append_trial(record)


def _finish(store: RunStore, status: RunStatus = RunStatus.COMPLETED) -> RunStore:
    store.set_status(status, AT_END)
    return store


def _summary(store: RunStore) -> Summary:
    return summarize_run(store.run_dir).summary


def _row(summary: Summary, condition: str, metric: str) -> MetricResult:
    matched = [r for r in summary.results if r.condition == condition and r.metric == metric]
    assert len(matched) == 1, f"{condition} の {metric} の行が {len(matched)} 個ある"
    return matched[0]


def _metrics_of(summary: Summary, condition: str) -> set[str]:
    return {r.metric for r in summary.results if r.condition == condition}


# --- 生成速度と最初のトークンまでの時間 (2.3、2.4、3.7、10.1) ----------------


def _decode_bed(store: RunStore) -> None:
    """1 つの条件に、慣らし 1 回、成功 2 回、失敗 1 回。すべて違う値にしてある。"""
    _append(
        store,
        # 慣らし: 集計に入れば、生成速度は (100 - 1) / 0.1 = 990 になり、すぐ分かる
        _trial(
            condition="decode/code/en",
            trial_index=0,
            warmup=True,
            sent_s=0.0,
            first_s=0.1,
            last_s=0.2,
            input_tokens=10,
            output_tokens=100,
        ),
        _trial(
            condition="decode/code/en",
            trial_index=0,
            sent_s=1.0,
            first_s=1.5,  # ttft = 0.5 秒
            last_s=3.5,  # 生成は 2.0 秒
            input_tokens=11,
            output_tokens=41,  # (41 - 1) / 2.0 = 20.0
        ),
        _trial(
            condition="decode/code/en",
            trial_index=1,
            sent_s=4.0,
            first_s=4.25,  # ttft = 0.25 秒
            last_s=5.25,  # 生成は 1.0 秒
            input_tokens=12,
            output_tokens=31,  # (31 - 1) / 1.0 = 30.0
        ),
        _trial(condition="decode/code/en", trial_index=2, sent_s=6.0, end_s=6.1, failed=True),
    )


def test_decode_speed_and_ttft_use_the_fixed_formulas(tmp_path: Path) -> None:
    """生成速度は `(出力 − 1) ÷ (最後 − 最初)`、ttft は `最初 − 送り始め` (design.md)。"""
    store = _store(tmp_path)
    _decode_bed(store)
    summary = _summary(_finish(store))

    tps = _row(summary, "decode/code/en", "decode_tps")
    assert tps.continuous is not None
    assert tps.continuous.n == 2
    assert tps.continuous.mean == pytest.approx(25.0)
    assert tps.continuous.median == pytest.approx(25.0)
    assert tps.continuous.min == pytest.approx(20.0)
    assert tps.continuous.max == pytest.approx(30.0)
    assert tps.continuous.iqr == pytest.approx(5.0)

    ttft = _row(summary, "decode/code/en", "ttft_s")
    assert ttft.continuous is not None
    assert ttft.continuous.n == 2
    assert ttft.continuous.mean == pytest.approx(0.375)
    assert ttft.continuous.min == pytest.approx(0.25)
    assert ttft.continuous.max == pytest.approx(0.5)


def test_legacy_text_block_does_not_acquire_unrecorded_phase_timestamps(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _append(
        store,
        _trial(
            condition="decode/json/en",
            first_s=1.0,
            last_s=3.0,
            input_tokens=10,
            output_tokens=41,
            text='{"ok": true}',
        ),
    )
    store = _finish(store)
    legacy, warnings = store.read_trials()
    assert warnings == []
    assert legacy[0].result.blocks[0].text == '{"ok": true}'
    assert legacy[0].result.model_dump().get("output_phases") is None

    summary = _summary(store)
    all_output = _row(summary, "decode/json/en", "decode_tps").continuous
    assert all_output is not None
    assert all_output.mean == pytest.approx(20.0)
    values = trial_values(*_read_back(store))
    assert values.get(("decode/json/en", "text_chars_per_s"), []) == []
    assert summary.output_phases["decode/json/en"]["text_arrived"] == 1
    assert summary.output_phases["decode/json/en"]["unknown"] == 1
    assert "不明" in summary.model_dump_json()


def test_legacy_trial_still_reproduces_the_total_output_decode_speed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _append(
        store,
        _trial(
            condition="decode/json/ja",
            first_s=2.0,
            last_s=4.0,
            input_tokens=10,
            output_tokens=41,
            text="旧記録の本文",
        ),
    )
    summary = _summary(_finish(store))
    all_output = _row(summary, "decode/json/ja", "decode_tps").continuous
    assert all_output is not None
    assert all_output.n == 1
    assert all_output.mean == pytest.approx(20.0)


def test_warmup_is_excluded_and_failures_are_counted(tmp_path: Path) -> None:
    """慣らしは集計にも失敗の件数にも入らない。失敗は値から外して数える (2.5、10.1)。"""
    store = _store(tmp_path)
    _decode_bed(store)
    summary = _summary(_finish(store))

    for metric in ("decode_tps", "ttft_s"):
        row = _row(summary, "decode/code/en", metric)
        assert row.failures == 1, f"{metric}: 失敗の件数が違う"
        assert row.continuous is not None
        assert row.continuous.n == 2, f"{metric}: 慣らしか失敗が混ざっている"


def test_a_failed_trial_with_partial_timings_never_reaches_the_speed_values(
    tmp_path: Path,
) -> None:
    """失敗した試行が、途中までの時刻とトークン数を持っていても、速さの集計に入れない (10.1)。

    実機で起きる形: `message_delta` のあと、`message_stop` の前に接続が切れると、
    クライアントは、失敗 (`protocol`) と一緒に、最初と最後のトークンの時刻と、16 以上の
    出力のトークン数を返す。これを集計に入れると、速さが何倍にも水増しされる。
    """
    store = _store(tmp_path)
    _append(
        store,
        _trial(  # (21 - 1) / 1.0 = 20.0 tok/s、ttft 0.20 s
            condition="decode/code/en",
            trial_index=0,
            sent_s=0.0,
            first_s=0.2,
            last_s=1.2,
            input_tokens=11,
            output_tokens=21,
        ),
        _trial(  # (31 - 1) / 1.5 = 20.0 tok/s、ttft 0.30 s
            condition="decode/code/en",
            trial_index=1,
            sent_s=2.0,
            first_s=2.3,
            last_s=3.8,
            input_tokens=12,
            output_tokens=31,
        ),
        _trial(  # 失敗。単独なら (50 - 1) / 0.1 = 490 tok/s、ttft 1.00 s
            condition="decode/code/en",
            trial_index=2,
            sent_s=5.0,
            first_s=6.0,
            last_s=6.1,
            input_tokens=100,
            output_tokens=50,
            failed=True,
        ),
        _trial(  # 入力の処理でも同じ形。成功: 8000 / 2.0 = 4000 tok/s
            condition="prefill/cold/8k",
            suite=SuiteName.PREFILL,
            trial_index=0,
            sent_s=10.0,
            first_s=12.0,
            last_s=12.1,
            input_tokens=8000,
            output_tokens=16,
        ),
        _trial(  # 失敗。単独なら 9000 / 0.5 = 18000 tok/s
            condition="prefill/cold/8k",
            suite=SuiteName.PREFILL,
            trial_index=1,
            sent_s=20.0,
            first_s=20.5,
            last_s=20.6,
            input_tokens=9000,
            output_tokens=16,
            failed=True,
        ),
    )
    finished = _finish(store)
    summary = _summary(finished)

    decode_tps = _row(summary, "decode/code/en", "decode_tps")
    assert decode_tps.continuous is not None
    assert decode_tps.continuous.n == 2
    assert decode_tps.continuous.mean == pytest.approx(20.0)
    assert decode_tps.continuous.max == pytest.approx(20.0)
    assert decode_tps.failures == 1

    ttft = _row(summary, "decode/code/en", "ttft_s")
    assert ttft.continuous is not None
    assert ttft.continuous.n == 2
    assert ttft.continuous.max == pytest.approx(0.30)

    prefill_tps = _row(summary, "prefill/cold/8k", "prefill_tps")
    assert prefill_tps.continuous is not None
    assert prefill_tps.continuous.n == 1
    assert prefill_tps.continuous.mean == pytest.approx(4000.0)
    assert prefill_tps.failures == 1

    per_trial = trial_values(*_read_back(finished))
    assert [v.trial_index for v in per_trial[("decode/code/en", "decode_tps")]] == [0, 1]
    assert [v.trial_index for v in per_trial[("prefill/cold/8k", "prefill_tps")]] == [0]


def test_too_few_output_tokens_leave_decode_tps_but_stay_in_ttft(tmp_path: Path) -> None:
    """出力が 16 トークン未満の試行は、生成速度からだけ外す (design.md 速さの定義)。"""
    store = _store(tmp_path)
    _append(
        store,
        _trial(
            condition="decode/prose/ja",
            trial_index=0,
            sent_s=0.0,
            first_s=0.2,
            last_s=1.2,
            input_tokens=7,
            output_tokens=21,  # (21 - 1) / 1.0 = 20.0
        ),
        _trial(
            condition="decode/prose/ja",
            trial_index=1,
            sent_s=2.0,
            first_s=2.5,
            last_s=2.6,
            input_tokens=8,
            output_tokens=8,
            flags=[TrialFlag.TOO_FEW_OUTPUT_TOKENS],
        ),
    )
    summary = _summary(_finish(store))

    tps = _row(summary, "decode/prose/ja", "decode_tps")
    assert tps.continuous is not None
    assert tps.continuous.n == 1
    assert tps.continuous.mean == pytest.approx(20.0)
    ttft = _row(summary, "decode/prose/ja", "ttft_s")
    assert ttft.continuous is not None
    assert ttft.continuous.n == 2
    assert tps.flag_counts["too_few_output_tokens"] == 1


# --- 入力の処理速度 (3.1、3.2) ----------------------------------------------


def test_prefill_tps_uses_the_total_input_tokens(tmp_path: Path) -> None:
    """入力の処理速度は、キャッシュの内訳を足した入力の全長 ÷ ttft (3.2)。"""
    store = _store(tmp_path, suites=[SuiteName.PREFILL])
    _append(
        store,
        _trial(
            condition="prefill/warm/8k",
            suite=SuiteName.PREFILL,
            trial_index=0,
            sent_s=0.0,
            first_s=0.5,  # ttft = 0.5 秒
            last_s=0.6,
            input_tokens=800,
            cache_read=200,
            cache_creation=100,  # 全長 1100 → 2200.0
            output_tokens=4,
        ),
        _trial(
            condition="prefill/warm/8k",
            suite=SuiteName.PREFILL,
            trial_index=1,
            sent_s=1.0,
            first_s=1.25,  # ttft = 0.25 秒
            last_s=1.3,
            input_tokens=400,
            cache_read=50,
            cache_creation=50,  # 全長 500 → 2000.0
            output_tokens=5,
        ),
    )
    summary = _summary(_finish(store))

    prefill = _row(summary, "prefill/warm/8k", "prefill_tps")
    assert prefill.continuous is not None
    assert prefill.continuous.n == 2
    assert prefill.continuous.mean == pytest.approx(2100.0)
    assert prefill.continuous.min == pytest.approx(2000.0)
    assert prefill.continuous.max == pytest.approx(2200.0)
    # 出力の上限が小さい条件なので、生成速度は出さない
    assert _metrics_of(summary, "prefill/warm/8k") == {"ttft_s", "prefill_tps"}


# --- 同時処理 (4.2、4.5、Implementation Notes 3.4) ---------------------------


def _concurrency_round(
    store: RunStore,
    *,
    condition: str = "concurrency/c2",
    tier: Tier = "primary",
    round_id: int,
    warmup: bool = False,
    streams: Sequence[dict[str, Any]],
) -> None:
    """1 回ぶん (n 本) を書く。

    `suites/concurrency` と同じ付け方にする: `trial_index` と `round_id` は
    どちらも回の番号で、慣らしと本番はそれぞれ 0 から数える (注 3.4)。だから
    慣らしの回と本番の回は、`round_id` がぶつかる。
    """
    _append(
        store,
        *[
            _trial(
                condition=condition,
                suite=SuiteName.CONCURRENCY,
                tier=tier,
                trial_index=round_id,
                warmup=warmup,
                round_id=round_id,
                stream_index=index,
                **stream,
            )
            for index, stream in enumerate(streams)
        ],
    )


def test_concurrency_warmup_round_is_excluded_from_every_value(tmp_path: Path) -> None:
    """慣らしの回は、回の合計にも 1 本あたりの値にも入らない (2.5)。

    慣らしと本番は、それぞれ 0 から番号を振るので (`iter_trials`)、実データでは
    `(warmup=True, round_id=0)` と `(warmup=False, round_id=0)` が必ずぶつかる。
    回をまとめる鍵が `round_id` だけだと、慣らしの 2 本が本番の回に混ざり、
    1 回ぶんの合計が黙って小さくなる。
    """
    store = _store(tmp_path, suites=[SuiteName.CONCURRENCY])
    _concurrency_round(  # 慣らし: 短くて速い
        store,
        round_id=0,
        warmup=True,
        streams=[
            # (17 - 1) / 0.2 = 80.0、ttft 0.2
            {"sent_s": 0.0, "first_s": 0.2, "last_s": 0.4, "input_tokens": 2, "output_tokens": 17},
            # (31 - 1) / 0.3 = 100.0、ttft 0.3
            {
                "sent_s": 0.05,
                "first_s": 0.35,
                "last_s": 0.65,
                "input_tokens": 3,
                "output_tokens": 31,
            },
        ],
    )
    _concurrency_round(  # 本番: 遅くて、トークンが多い
        store,
        round_id=0,
        streams=[
            # (21 - 1) / 4.0 = 5.0、ttft 1.0
            {
                "sent_s": 10.0,
                "first_s": 11.0,
                "last_s": 15.0,
                "input_tokens": 4,
                "output_tokens": 21,
            },
            # (41 - 1) / 5.0 = 8.0、ttft 1.9
            {
                "sent_s": 10.1,
                "first_s": 12.0,
                "last_s": 17.0,
                "input_tokens": 5,
                "output_tokens": 41,
            },
        ],
    )
    summary = _summary(_finish(store))

    total = _row(summary, "concurrency/c2", "round_total_tps")
    assert total.continuous is not None
    assert total.continuous.n == 1, "慣らしの回が、別の回として数えられている"
    # 本番の 2 本だけ: (21 + 41) / (17.0 - 11.0)。慣らしが混ざると 110 / 16.8 になる
    assert total.continuous.mean == pytest.approx(62.0 / 6.0)

    tps = _row(summary, "concurrency/c2", "decode_tps")
    assert tps.continuous is not None
    assert tps.continuous.n == 2
    assert tps.continuous.min == pytest.approx(5.0)
    assert tps.continuous.max == pytest.approx(8.0)

    ttft = _row(summary, "concurrency/c2", "ttft_s")
    assert ttft.continuous is not None
    assert ttft.continuous.n == 2
    assert ttft.continuous.min == pytest.approx(1.0)
    assert ttft.continuous.max == pytest.approx(1.9)


def test_concurrency_reports_per_stream_and_round_totals(tmp_path: Path) -> None:
    """1 本あたりの速さと、1 回ぶんの合計 (Σ出力 ÷ (最後の最大 − 最初の最小))。"""
    store = _store(tmp_path, suites=[SuiteName.CONCURRENCY])
    _concurrency_round(
        store,
        round_id=0,
        streams=[
            # (21 - 1) / 4.0 = 5.0、ttft 1.0
            {"sent_s": 0.0, "first_s": 1.0, "last_s": 5.0, "input_tokens": 3, "output_tokens": 21},
            # (41 - 1) / 5.0 = 8.0、ttft 1.9
            {"sent_s": 0.1, "first_s": 2.0, "last_s": 7.0, "input_tokens": 4, "output_tokens": 41},
        ],
    )
    _concurrency_round(
        store,
        round_id=1,
        streams=[
            # (17 - 1) / 2.0 = 8.0、ttft 1.5
            {
                "sent_s": 10.0,
                "first_s": 11.5,
                "last_s": 13.5,
                "input_tokens": 5,
                "output_tokens": 17,
            },
            # (41 - 1) / 4.0 = 10.0、ttft 2.0
            {
                "sent_s": 10.1,
                "first_s": 12.1,
                "last_s": 16.1,
                "input_tokens": 6,
                "output_tokens": 41,
            },
        ],
    )
    summary = _summary(_finish(store))

    tps = _row(summary, "concurrency/c2", "decode_tps")
    assert tps.continuous is not None
    assert tps.continuous.n == 4
    assert tps.continuous.mean == pytest.approx(7.75)
    assert tps.continuous.min == pytest.approx(5.0)
    assert tps.continuous.max == pytest.approx(10.0)

    total = _row(summary, "concurrency/c2", "round_total_tps")
    assert total.continuous is not None
    assert total.continuous.n == 2
    assert total.continuous.min == pytest.approx(62.0 / 6.0)
    assert total.continuous.max == pytest.approx(58.0 / 4.6)

    ttft = _row(summary, "concurrency/c2", "ttft_s")
    assert ttft.continuous is not None
    assert ttft.continuous.n == 4
    assert ttft.continuous.median == pytest.approx(1.7)
    assert MetricFlag.PARTIAL_FAILURES not in ttft.flags


def test_concurrency_round_with_a_failed_stream_keeps_the_rest(tmp_path: Path) -> None:
    """1 本が失敗した回も、残りの本から値を出し、条件に「一部が失敗」を付ける (4.5)。"""
    store = _store(tmp_path, suites=[SuiteName.CONCURRENCY])
    _concurrency_round(
        store,
        condition="concurrency/c4",
        tier="reference",
        round_id=0,
        streams=[
            {"sent_s": 0.0, "first_s": 1.0, "last_s": 5.0, "input_tokens": 3, "output_tokens": 21},
            {"sent_s": 0.1, "end_s": 0.3, "failed": True},
        ],
    )
    summary = _summary(_finish(store))

    total = _row(summary, "concurrency/c4", "round_total_tps")
    assert total.continuous is not None
    assert total.continuous.n == 1
    assert total.continuous.mean == pytest.approx(21.0 / 4.0)  # 成功した 1 本だけから
    assert total.failures == 1
    assert MetricFlag.PARTIAL_FAILURES in total.flags
    assert total.tier == "reference"
    per_stream = _row(summary, "concurrency/c4", "decode_tps")
    assert per_stream.continuous is not None
    assert per_stream.continuous.mean == pytest.approx(5.0)
    assert MetricFlag.PARTIAL_FAILURES in per_stream.flags


def test_concurrency_tolerates_a_round_cut_short(tmp_path: Path) -> None:
    """中断で本数がそろわなかった回があっても、例外にせず集計する。"""
    store = _store(tmp_path, suites=[SuiteName.CONCURRENCY])
    _concurrency_round(
        store,
        round_id=0,
        streams=[
            {"sent_s": 0.0, "first_s": 1.0, "last_s": 5.0, "input_tokens": 3, "output_tokens": 21},
            {"sent_s": 0.1, "first_s": 2.0, "last_s": 7.0, "input_tokens": 4, "output_tokens": 41},
        ],
    )
    _concurrency_round(
        store,
        round_id=1,
        streams=[
            # 2 本目が書かれる前に止まった回
            {
                "sent_s": 10.0,
                "first_s": 11.0,
                "last_s": 14.0,
                "input_tokens": 5,
                "output_tokens": 31,
            }
        ],
    )
    summary = _summary(_finish(store, RunStatus.INTERRUPTED))

    total = _row(summary, "concurrency/c2", "round_total_tps")
    assert total.continuous is not None
    assert total.continuous.n == 2
    assert total.continuous.min == pytest.approx(31.0 / 3.0)
    assert total.continuous.max == pytest.approx(62.0 / 6.0)
    assert summary.incomplete is True


def test_concurrency_without_round_ids_treats_each_trial_as_one_round(tmp_path: Path) -> None:
    """1 本だけの条件は、回の番号が残らないことがある。試行ごとに 1 回ぶんと見る。"""
    store = _store(tmp_path, suites=[SuiteName.CONCURRENCY])
    _append(
        store,
        _trial(
            condition="concurrency/c1",
            suite=SuiteName.CONCURRENCY,
            trial_index=0,
            sent_s=0.0,
            first_s=1.0,
            last_s=5.0,
            input_tokens=3,
            output_tokens=21,  # 21 / 4.0 = 5.25
        ),
        _trial(
            condition="concurrency/c1",
            suite=SuiteName.CONCURRENCY,
            trial_index=1,
            sent_s=6.0,
            first_s=7.0,
            last_s=9.0,
            input_tokens=4,
            output_tokens=31,  # 31 / 2.0 = 15.5
        ),
    )
    summary = _summary(_finish(store))

    total = _row(summary, "concurrency/c1", "round_total_tps")
    assert total.continuous is not None
    assert total.continuous.n == 2, "2 つの試行が、1 回ぶんにまとめられている"
    assert total.continuous.min == pytest.approx(5.25)
    assert total.continuous.max == pytest.approx(15.5)


# --- 条件ごとの印と件数 (10.1、10.2、10.7) ----------------------------------


def test_insufficient_trials_is_flagged_but_the_row_stays(tmp_path: Path) -> None:
    """成功した試行が最小の数に届かない条件にも、行を出す (10.2)。"""
    store = _store(tmp_path, min_successes=3)
    _append(
        store,
        _trial(
            condition="decode/code/ja",
            trial_index=0,
            sent_s=0.0,
            first_s=0.5,
            last_s=1.5,
            input_tokens=9,
            output_tokens=21,
        ),
        _trial(
            condition="decode/code/ja",
            trial_index=1,
            sent_s=2.0,
            first_s=2.25,
            last_s=3.25,
            input_tokens=10,
            output_tokens=41,
        ),
    )
    summary = _summary(_finish(store))

    row = _row(summary, "decode/code/ja", "decode_tps")
    assert row.continuous is not None
    assert row.continuous.n == 2
    assert MetricFlag.INSUFFICIENT_TRIALS in row.flags


def test_insufficient_trials_is_judged_row_by_row(tmp_path: Path) -> None:
    """印は、その行の値の数で判定する (行によって、数える単位が違うため)。

    同時処理では、`ttft_s` と `decode_tps` の n は本の数、`round_total_tps` の
    n は回の数になる。条件の本の数で判定すると、回の数が足りない行に印が
    付かないまま出る。
    """
    store = _store(tmp_path, min_successes=3, suites=[SuiteName.CONCURRENCY])
    _concurrency_round(
        store,
        round_id=0,
        streams=[
            {"sent_s": 0.0, "first_s": 1.0, "last_s": 5.0, "input_tokens": 3, "output_tokens": 21},
            {"sent_s": 0.1, "first_s": 2.0, "last_s": 7.0, "input_tokens": 4, "output_tokens": 41},
        ],
    )
    _concurrency_round(
        store,
        round_id=1,
        streams=[
            {
                "sent_s": 10.0,
                "first_s": 11.5,
                "last_s": 13.5,
                "input_tokens": 5,
                "output_tokens": 17,
            },
            {
                "sent_s": 10.1,
                "first_s": 12.1,
                "last_s": 16.1,
                "input_tokens": 6,
                "output_tokens": 41,
            },
        ],
    )
    summary = _summary(_finish(store))

    for metric in ("ttft_s", "decode_tps"):
        row = _row(summary, "concurrency/c2", metric)
        assert row.continuous is not None
        assert row.continuous.n == 4
        assert MetricFlag.INSUFFICIENT_TRIALS not in row.flags, f"{metric}: 4 ≥ 3 なのに印が付いた"
    total = _row(summary, "concurrency/c2", "round_total_tps")
    assert total.continuous is not None
    assert total.continuous.n == 2
    assert MetricFlag.INSUFFICIENT_TRIALS in total.flags, "回の数 2 < 3 なのに印が付かない"


def test_insufficient_trials_counts_the_values_left_after_the_exclusion(tmp_path: Path) -> None:
    """`decode_tps` は、16 トークン未満を除いたあとの数で判定する。"""
    store = _store(tmp_path, min_successes=2)
    _append(
        store,
        _trial(
            condition="decode/prose/en",
            trial_index=0,
            sent_s=0.0,
            first_s=0.2,
            last_s=1.2,
            input_tokens=7,
            output_tokens=21,
        ),
        _trial(
            condition="decode/prose/en",
            trial_index=1,
            sent_s=2.0,
            first_s=2.5,
            last_s=2.6,
            input_tokens=8,
            output_tokens=8,
            flags=[TrialFlag.TOO_FEW_OUTPUT_TOKENS],
        ),
    )
    summary = _summary(_finish(store))

    assert MetricFlag.INSUFFICIENT_TRIALS not in _row(summary, "decode/prose/en", "ttft_s").flags
    assert MetricFlag.INSUFFICIENT_TRIALS in _row(summary, "decode/prose/en", "decode_tps").flags


def test_a_condition_whose_trials_all_failed_still_appears(tmp_path: Path) -> None:
    """全部の試行が失敗した条件も、値なしの行として残す (10.1)。"""
    store = _store(tmp_path, min_successes=1)
    _append(
        store,
        _trial(condition="decode/prose/en", trial_index=0, sent_s=0.0, end_s=0.2, failed=True),
        _trial(condition="decode/prose/en", trial_index=1, sent_s=1.0, end_s=1.3, failed=True),
    )
    summary = _summary(_finish(store))

    assert _metrics_of(summary, "decode/prose/en") == {"decode_tps", "ttft_s"}
    for metric in ("decode_tps", "ttft_s"):
        row = _row(summary, "decode/prose/en", metric)
        assert row.continuous is None
        assert row.failures == 2
        assert MetricFlag.INSUFFICIENT_TRIALS in row.flags


def test_a_condition_with_only_warmups_appears_without_values(tmp_path: Path) -> None:
    """慣らししか残っていない条件でも、例外にせず値なしの行を出す。"""
    store = _store(tmp_path, min_successes=1)
    _append(
        store,
        _trial(
            condition="decode/code/en",
            trial_index=0,
            warmup=True,
            sent_s=0.0,
            first_s=0.5,
            last_s=1.5,
            input_tokens=9,
            output_tokens=21,
        ),
    )
    summary = _summary(_finish(store, RunStatus.ABORTED))

    row = _row(summary, "decode/code/en", "decode_tps")
    assert row.continuous is None
    assert row.failures == 0
    assert MetricFlag.INSUFFICIENT_TRIALS in row.flags


def test_flag_counts_and_the_count_based_flags(tmp_path: Path) -> None:
    """印の件数は、慣らしを除いた試行で数え、3 つの印に反映する (2.6、3.3、10.7)。"""
    store = _store(tmp_path, min_successes=1)
    _append(
        store,
        _trial(  # 慣らしの印は数えない
            condition="decode/code/en",
            trial_index=0,
            warmup=True,
            sent_s=0.0,
            first_s=0.1,
            last_s=0.2,
            input_tokens=9,
            output_tokens=21,
            flags=[TrialFlag.SHORT_OUTPUT],
        ),
        _trial(
            condition="decode/code/en",
            trial_index=0,
            sent_s=1.0,
            first_s=1.5,
            last_s=3.5,
            input_tokens=10,
            output_tokens=41,
            flags=[TrialFlag.SHORT_OUTPUT],
        ),
        _trial(
            condition="decode/code/en",
            trial_index=1,
            sent_s=4.0,
            first_s=4.25,
            last_s=5.25,
            input_tokens=11,
            output_tokens=31,
            flags=[TrialFlag.LENGTH_OFF_TARGET, TrialFlag.REPLACEMENT_CHAR],
        ),
        _trial(
            condition="decode/code/en",
            trial_index=2,
            sent_s=6.0,
            first_s=6.5,
            last_s=8.5,
            input_tokens=12,
            output_tokens=51,
            flags=[TrialFlag.REPETITION_LOOP],
        ),
    )
    summary = _summary(_finish(store))

    row = _row(summary, "decode/code/en", "decode_tps")
    assert row.flag_counts == {
        "short_output": 1,
        "length_off_target": 1,
        "too_few_output_tokens": 0,
        "replacement_char": 1,
        "repetition_loop": 1,
    }
    assert MetricFlag.SHORT_OUTPUTS in row.flags
    assert MetricFlag.LENGTH_OFF_TARGET in row.flags
    assert MetricFlag.SUSPECT_OUTPUTS in row.flags
    assert MetricFlag.INSUFFICIENT_TRIALS not in row.flags


# --- 未完了の計測ランと、読み取りの警告 (10.4、Implementation Notes 2.3) -----


@pytest.mark.parametrize(
    ("status", "incomplete"),
    [
        (RunStatus.COMPLETED, False),
        (RunStatus.ABORTED, True),
        (RunStatus.INTERRUPTED, True),
        (RunStatus.RUNNING, True),  # 状態が running のまま = 計測ランが落ちた跡
    ],
)
def test_incomplete_runs_are_marked_and_bannered(
    tmp_path: Path, status: RunStatus, incomplete: bool
) -> None:
    store = _store(tmp_path)
    _decode_bed(store)
    if status is not RunStatus.RUNNING:
        store.set_status(status, AT_END)

    result = summarize_run(store.run_dir)
    assert result.summary.incomplete is incomplete

    _, md_path = write_summary(store.run_dir)
    text = md_path.read_text(encoding="utf-8")
    assert ("未完了" in text) is incomplete


def test_truncated_last_line_is_dropped_and_the_warning_is_surfaced(tmp_path: Path) -> None:
    """末尾が途中で切れた `trials.jsonl` でも要約でき、警告が呼び出し側に届く。"""
    store = _store(tmp_path)
    _decode_bed(store)
    with (store.run_dir / "trials.jsonl").open("ab") as fh:
        fh.write(b'{"run_id": "20260919T120000Z-fake-abcdef", "suite": "dec')

    result = summarize_run(store.run_dir)
    assert result.warnings, "捨てた行の警告が返っていない"
    assert any("trials.jsonl" in warning for warning in result.warnings)
    row = _row(result.summary, "decode/code/en", "decode_tps")
    assert row.continuous is not None
    assert row.continuous.n == 2

    _, md_path = write_summary(store.run_dir)
    assert "trials.jsonl" in md_path.read_text(encoding="utf-8")


def test_corrupt_middle_line_raises_store_error(tmp_path: Path) -> None:
    """本物の破損 (途中の行) は `StoreError` のまま外へ出す (Implementation Notes 2.3)。"""
    store = _store(tmp_path)
    _decode_bed(store)
    path = store.run_dir / "trials.jsonl"
    lines = path.read_bytes().splitlines()
    path.write_bytes(b"\n".join([lines[0], b'{"unknown": 1}', *lines[1:]]) + b"\n")

    with pytest.raises(StoreError):
        summarize_run(store.run_dir)


# --- 内部の指標 (7.3、7.4) --------------------------------------------------


def test_server_metrics_are_carried_per_condition(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _decode_bed(store)
    store.write_metrics(
        "decode/code/en",
        MetricSnapshot(taken_at_utc=AT, raw_text="# before\n", values={}, missing=[]),
        MetricSnapshot(taken_at_utc=AT_END, raw_text="# after\n", values={}, missing=[]),
        DerivedMetrics(
            spec_acceptance_rate=0.75,
            mean_acceptance_length=1.5,
            decode_steps=400.0,
            tokens_per_step=1.75,
            prefix_cache_hit_rate=0.25,
            kv_usage_peak=0.5,
            preemptions=2.0,
        ),
    )
    store.write_metrics(
        "decode/prose/en",
        MetricsUnavailable(reason="/metrics が 404"),
        MetricsUnavailable(reason="/metrics が 404"),
        DerivedMetrics(missing=[LogicalMetric.SPEC_DRAFTS, LogicalMetric.KV_USAGE]),
    )
    summary = _summary(_finish(store))

    assert set(summary.server_metrics) == {"decode/code/en", "decode/prose/en"}
    got = summary.server_metrics["decode/code/en"]
    assert got.spec_acceptance_rate == pytest.approx(0.75)
    assert got.tokens_per_step == pytest.approx(1.75)
    assert got.kv_usage_peak == pytest.approx(0.5)
    assert summary.server_metrics["decode/prose/en"].missing == [
        LogicalMetric.SPEC_DRAFTS,
        LogicalMetric.KV_USAGE,
    ]

    _, md_path = write_summary(store.run_dir)
    text = md_path.read_text(encoding="utf-8")
    assert "0.750" in text
    assert "spec_drafts" in text


def test_server_metrics_section_heading_names_generation_tokens(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _decode_bed(store)
    store.write_metrics(
        "decode/code/en",
        MetricSnapshot(taken_at_utc=AT, raw_text="# before\n", values={}, missing=[]),
        MetricSnapshot(taken_at_utc=AT_END, raw_text="# after\n", values={}, missing=[]),
        DerivedMetrics(tokens_per_step=1.0),
    )
    _finish(store)

    _, md_path = write_summary(store.run_dir)

    lines = md_path.read_text(encoding="utf-8").splitlines()
    section = lines.index("## 対象サーバーの内部の指標")
    header_line = next(line for line in lines[section + 1 :] if line.startswith("|"))
    cells = [cell.strip() for cell in header_line.strip().strip("|").split("|")]
    assert len(cells) == 9
    assert cells[4] == "1 ステップあたりの生成トークン"


def test_non_finite_server_metrics_never_reach_the_summary(tmp_path: Path) -> None:
    """外から来た値は範囲も確かめる (注 2.7)。JSON に NaN や Infinity を書かない。

    `json.loads` は `Infinity` と `NaN` を読めてしまうので、手で書いた
    `metrics/deltas.jsonl` からは、有限でない値が分析に届きうる。
    """
    store = _store(tmp_path)
    _decode_bed(store)
    row = {
        "schema_version": 1,
        "condition": "decode/code/en",
        "before_available": True,
        "after_available": True,
        "derived": {
            "kv_usage_peak": math.inf,
            "tokens_per_step": math.nan,
            "decode_steps": 3.0,
            "missing": [],
        },
    }
    raw = json.dumps(row)
    assert "Infinity" in raw and "NaN" in raw, "試験の前提: 生データに有限でない値が入る"
    (store.run_dir / "metrics" / "deltas.jsonl").write_text(raw + "\n", encoding="utf-8")
    result = summarize_run(_finish(store).run_dir)

    got = result.summary.server_metrics["decode/code/en"]
    assert got.kv_usage_peak is None
    assert got.tokens_per_step is None
    assert got.decode_steps == pytest.approx(3.0)
    assert result.warnings, "有限でない値を捨てたことが伝わっていない"

    json_path, _ = write_summary(store.run_dir)
    text = json_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text


# --- 壊れた形の生データに耐える ---------------------------------------------


def test_missing_timings_and_usage_do_not_raise(tmp_path: Path) -> None:
    """時刻やトークン数が欠けた試行があっても、例外にせず値から外す。"""
    store = _store(tmp_path, min_successes=1)
    _append(
        store,
        # 成功しているが、トークンが 1 つも来なかった (時刻も来ない)
        _trial(condition="decode/code/en", trial_index=0, sent_s=0.0, end_s=0.5, input_tokens=9),
        # 成功しているが、トークン数の申告がない
        _trial(
            condition="decode/code/en",
            trial_index=1,
            sent_s=1.0,
            first_s=1.2,
            last_s=1.4,
            end_s=1.5,
        ),
        # 最初と最後が同じ時刻 (分母が 0)
        _trial(
            condition="decode/code/en",
            trial_index=2,
            sent_s=2.0,
            first_s=2.5,
            last_s=2.5,
            input_tokens=9,
            output_tokens=41,
        ),
    )
    result = summarize_run(_finish(store).run_dir)

    assert _row(result.summary, "decode/code/en", "decode_tps").continuous is None
    ttft = _row(result.summary, "decode/code/en", "ttft_s")
    assert ttft.continuous is not None
    assert ttft.continuous.n == 2  # 時刻が来た 2 件だけ
    assert result.warnings, "値にできなかった試行が伝わっていない"


def test_proportion_suites_stay_out_of_the_continuous_rows(tmp_path: Path) -> None:
    """品質と長い会話のまとまりは、速さの値を 1 つも出さない (4.2 で割合の行になった)。

    task 4.2 の前は、この 2 つのまとまりから行が 1 つも出なかった。いまは条件
    ごとに割合の行が 1 行だけ出る (中身は
    `test_analysis_summarize_proportions.py` が確かめる)。ここで確かめるのは、
    速さの値 (`decode_tps` など) にも `trial_values` にも混ざらないことである。
    """
    store = _store(tmp_path, suites=[SuiteName.QUALITY, SuiteName.AGENT])
    _append(
        store,
        _trial(
            condition="quality/toolcall",
            suite=SuiteName.QUALITY,
            trial_index=0,
            sent_s=0.0,
            first_s=0.5,
            last_s=1.5,
            input_tokens=9,
            output_tokens=41,
        ),
        _trial(
            condition="agent/stage/020k",
            suite=SuiteName.AGENT,
            trial_index=0,
            sent_s=2.0,
            first_s=2.5,
            last_s=3.5,
            input_tokens=20000,
            output_tokens=41,
        ),
    )
    finished = _finish(store)
    summary = _summary(finished)

    assert {(row.condition, row.metric) for row in summary.results} == {
        ("quality/toolcall", "accuracy"),
        ("agent/stage/020k", "agent_break_rate"),
    }
    assert all(row.continuous is None for row in summary.results)
    assert trial_values(*_read_back(finished)) == {}


# --- 2 つの要約 (8.2、8.3、8.6、4.4) ----------------------------------------


def _mixed_bed(tmp_path: Path) -> RunStore:
    """主な結果と参考、送った本文と応答の目印を含む、ひととおりの生データ。"""
    store = _store(
        tmp_path,
        min_successes=1,
        suites=[SuiteName.DECODE, SuiteName.PREFILL, SuiteName.CONCURRENCY],
        warnings=["計測の前に、実行中の要求が 1 件あった"],
        skipped=[
            {
                "suite": SuiteName.PREFILL,
                "key": "prefill/cold/128k",
                "reason": "入力の長さの上限を超えるので、送る前に飛ばした",
            }
        ],
    )
    body: dict[str, JsonValue] = {
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": SENTINEL_REQUEST}],
    }
    ref = store.put_body(body)
    _append(
        store,
        _trial(
            condition="decode/code/en",
            trial_index=0,
            sent_s=0.0,
            first_s=0.5,
            last_s=2.5,
            input_tokens=11,
            output_tokens=41,
            body_ref=ref,
            text=SENTINEL_RESPONSE,
        ),
        _trial(
            condition="decode/code/en",
            trial_index=1,
            sent_s=3.0,
            first_s=3.25,
            last_s=4.25,
            input_tokens=12,
            output_tokens=31,
            body_ref=ref,
            text=SENTINEL_RESPONSE,
        ),
        _trial(
            condition="prefill/cold/8k",
            suite=SuiteName.PREFILL,
            trial_index=0,
            sent_s=5.0,
            first_s=6.0,
            last_s=6.1,
            input_tokens=8100,
            output_tokens=4,
            body_ref=ref,
        ),
    )
    _concurrency_round(
        store,
        condition="concurrency/c1",
        round_id=0,
        streams=[
            {"sent_s": 7.0, "first_s": 8.0, "last_s": 12.0, "input_tokens": 3, "output_tokens": 21}
        ],
    )
    _concurrency_round(
        store,
        condition="concurrency/c8",
        tier="reference",
        round_id=0,
        streams=[
            {
                "sent_s": 13.0,
                "first_s": 14.0,
                "last_s": 19.0,
                "input_tokens": 4,
                "output_tokens": 41,
            }
        ],
    )
    store.write_metrics(
        "decode/code/en",
        MetricSnapshot(taken_at_utc=AT, raw_text="# before\n", values={}, missing=[]),
        MetricSnapshot(taken_at_utc=AT_END, raw_text="# after\n", values={}, missing=[]),
        DerivedMetrics(spec_acceptance_rate=0.75, prefix_cache_hit_rate=0.25, kv_usage_peak=0.5),
    )
    return _finish(store)


def test_write_summary_makes_two_files_without_any_body_text(tmp_path: Path) -> None:
    """2 つの要約ができ、どちらにも送った内容と応答の本文が含まれない (8.2、8.3)。"""
    store = _mixed_bed(tmp_path)
    json_path, md_path = write_summary(store.run_dir)

    assert json_path == store.run_dir / "summary.json"
    assert md_path == store.run_dir / "summary.md"
    json_text = json_path.read_text(encoding="utf-8")
    md_text = md_path.read_text(encoding="utf-8")
    for text in (json_text, md_text):
        assert SENTINEL_REQUEST not in text
        assert SENTINEL_RESPONSE not in text

    restored = Summary.model_validate_json(json_text)
    assert restored.conditions.run_id == RUN_ID
    assert restored.results
    # JSON に NaN / Infinity を書かない (ほかの道具が読めなくなる)
    json.loads(json_text, parse_constant=_reject_constant)


def _reject_constant(name: str) -> float:
    raise AssertionError(f"JSON に {name} が書かれている")


def test_markdown_separates_primary_from_reference(tmp_path: Path) -> None:
    """主な結果と参考が、別の節に分かれて表示される (4.4)。"""
    store = _mixed_bed(tmp_path)
    _, md_path = write_summary(store.run_dir)
    text = md_path.read_text(encoding="utf-8")

    assert "## 主な結果" in text
    assert "## 参考" in text
    _, _, results = text.partition("## 主な結果")
    primary, _, reference = results.partition("## 参考")
    assert "concurrency/c1" in primary
    assert "decode/code/en" in primary
    assert "concurrency/c8" not in primary
    assert "concurrency/c8" in reference


def test_markdown_header_shows_the_run_conditions(tmp_path: Path) -> None:
    """要約の先頭に、実行の条件と、飛ばした条件と、警告を出す (8.6、3.6)。"""
    store = _mixed_bed(tmp_path)
    _, md_path = write_summary(store.run_dir)
    text = md_path.read_text(encoding="utf-8")
    header = text.partition("## 主な結果")[0]

    assert RUN_ID in header
    assert "glm-5.3-flash" in header
    assert "0.1.0+gtest" in header  # 道具の版
    assert "completed" in header
    assert "32000" in header  # 入力の長さの上限
    assert "prefill/cold/128k" in header  # 飛ばした条件
    assert "実行中の要求が 1 件あった" in header  # manifest の警告


def test_write_summary_is_byte_identical_on_a_second_run(tmp_path: Path) -> None:
    """何度実行しても同じ結果になる (Batch の約束、4.1)。"""
    store = _mixed_bed(tmp_path)
    json_path, md_path = write_summary(store.run_dir)
    first = (json_path.read_bytes(), md_path.read_bytes())
    write_summary(store.run_dir)
    assert (json_path.read_bytes(), md_path.read_bytes()) == first


def test_results_keep_the_order_the_conditions_were_measured_in(tmp_path: Path) -> None:
    store = _mixed_bed(tmp_path)
    summary = summarize_run(store.run_dir).summary
    seen: list[str] = []
    for row in summary.results:
        if row.condition not in seen:
            seen.append(row.condition)
    assert seen == ["decode/code/en", "prefill/cold/8k", "concurrency/c1", "concurrency/c8"]


def test_markdown_shows_the_excluded_count_only_where_it_applies(tmp_path: Path) -> None:
    """16 トークン未満の除外は `decode_tps` にしか効かないので、その行にだけ出す。"""
    store = _store(tmp_path, min_successes=1)
    _append(
        store,
        _trial(
            condition="decode/prose/en",
            trial_index=0,
            sent_s=0.0,
            first_s=0.2,
            last_s=1.2,
            input_tokens=7,
            output_tokens=21,
        ),
        _trial(
            condition="decode/prose/en",
            trial_index=1,
            sent_s=2.0,
            first_s=2.5,
            last_s=2.6,
            input_tokens=8,
            output_tokens=8,
            flags=[TrialFlag.TOO_FEW_OUTPUT_TOKENS],
        ),
    )
    _, md_path = write_summary(_finish(store).run_dir)
    markdown = md_path.read_text(encoding="utf-8")
    result_tables = markdown.split("## 主な結果\n", 1)[1].split("## 品質", 1)[0]
    rows = [line for line in result_tables.splitlines() if line.startswith("| `decode/prose/en`")]
    assert len(rows) == 2, "表の行が見つからない"

    decode_line = next(line for line in rows if "`decode_tps`" in line)
    ttft_line = next(line for line in rows if "`ttft_s`" in line)
    assert "少なすぎる 1 件" in decode_line
    assert "少なすぎる" not in ttft_line


def test_markdown_explains_how_the_insufficient_flag_is_judged(tmp_path: Path) -> None:
    """行ごとに判定するという決まりを、人が読む要約にも書く。"""
    store = _mixed_bed(tmp_path)
    _, md_path = write_summary(store.run_dir)
    assert "その行の値の数" in md_path.read_text(encoding="utf-8")


# --- 試行ごとの値 (4.3 の対応のある比較が使う) -------------------------------


def _read_back(store: RunStore) -> tuple[list[TrialRecord], RunManifest]:
    reopened = RunStore.open(store.run_dir)
    records, _ = reopened.read_trials()
    return records, reopened.manifest()


def test_trial_values_back_every_row_of_the_summary(tmp_path: Path) -> None:
    """要約の各行は、公開の `trial_values` の値に `describe` をかけたものと一致する。

    速さの定義を 2 か所に分けないため、要約も比較 (4.3) も、同じ関数を通す。
    """
    store = _mixed_bed(tmp_path)
    summary = summarize_run(store.run_dir).summary
    records, manifest = _read_back(store)
    per_trial = trial_values(records, manifest)

    assert summary.results
    for row in summary.results:
        values = [item.value for item in per_trial.get((row.condition, row.metric), [])]
        if row.continuous is None:
            assert values == [], f"{row.condition} の {row.metric}: 値があるのに行は空"
        else:
            assert row.continuous == describe(values), f"{row.condition} の {row.metric}"
    assert set(per_trial) == {(row.condition, row.metric) for row in summary.results}


def test_trial_values_carry_the_identity_needed_for_pairing(tmp_path: Path) -> None:
    """対応のある比較のために、試行の番号とストリームの番号を添える。"""
    store = _mixed_bed(tmp_path)
    per_trial = trial_values(*_read_back(store))

    assert [
        (v.trial_index, v.stream_index) for v in per_trial[("decode/code/en", "decode_tps")]
    ] == [
        (0, None),
        (1, None),
    ]
    assert [
        (v.trial_index, v.stream_index) for v in per_trial[("concurrency/c1", "decode_tps")]
    ] == [(0, 0)]
    assert [
        (v.trial_index, v.stream_index) for v in per_trial[("concurrency/c1", "round_total_tps")]
    ] == [(0, None)], "1 回ぶんの合計は、回の番号を試行の番号にする"


def test_trial_values_leave_out_warmups_and_failures(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _decode_bed(store)  # 慣らし 1 回、成功 2 回、失敗 1 回
    per_trial = trial_values(*_read_back(_finish(store)))

    assert [v.trial_index for v in per_trial[("decode/code/en", "ttft_s")]] == [0, 1]
    assert [v.value for v in per_trial[("decode/code/en", "decode_tps")]] == pytest.approx(
        [20.0, 30.0]
    )


def test_trial_values_rejects_records_from_another_run(tmp_path: Path) -> None:
    """取り違えた計測ランのレコードで、対応のある比較を組み立てさせない。"""
    store = _store(tmp_path)
    _decode_bed(store)
    records, _ = _read_back(_finish(store))

    with pytest.raises(ValueError, match="run_id"):
        trial_values(records, _manifest(run_id="20260919T120000Z-fake-zzzzzz"))


# --- 依存の向き (design.md) -------------------------------------------------


def test_summarize_imports_nothing_outside_the_allowed_set() -> None:
    """`analysis` は `types`、`store`、`analysis.stats` 以外を読み込まない (design.md)。"""
    source = Path(summarize_module.__file__ or "")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "相対 import は使わない (依存の向きを見えなくする)"
            assert node.module is not None
            modules.add(node.module)

    allowed_internal = {
        "bench_harness.types",
        "bench_harness.store",
        "bench_harness.store.rawstore",
        "bench_harness.analysis.stats",
        "bench_harness.analysis.output_phases",
    }
    for module in modules:
        root = module.split(".")[0]
        if root == "bench_harness":
            assert module in allowed_internal, f"読み込んではいけない module: {module}"
        else:
            assert root in sys.stdlib_module_names or root == "pydantic", (
                f"標準ライブラリと pydantic 以外を読み込んでいる: {module}"
            )


# --- 端から端まで: 実際の計測ランを要約する ---------------------------------
#
# 分析は `runner` を読み込まない (上の試験が機械的に確かめている)。この試験
# **だけ**が、生データの作り手 (計測ランの進行と本物のまとまり) を呼ぶ。


def _write_targets(tmp_path: Path, base_url: str) -> Path:
    path = tmp_path / "targets.toml"
    path.write_text(
        f'[targets.fake]\nbase_url = "{base_url}"\nmodel = "fake-model"\n', encoding="utf-8"
    )
    return path


def _write_profiles(tmp_path: Path) -> Path:
    path = tmp_path / "profiles.toml"
    path.write_text(
        """
[profiles.e2e]
seed = 3
min_successes = 10
max_consecutive_failures = 5
metrics_interval_s = 60.0

[profiles.e2e.sampling]
temperature = 0.0
thinking = "server_default"

[profiles.e2e.timeout]
connect_s = 5.0
first_event_s = 5.0
idle_s = 5.0
total_s = 20.0

[profiles.e2e.decode]
trials = 10
warmup_trials = 1
max_tokens = 32
""",
        encoding="utf-8",
    )
    return path


class _SilentProgress:
    """進み具合を捨てる受け口 (この試験は表示を見ない)。"""

    def update(
        self, suite: SuiteName, condition: str, done: int, total: int, failures: int
    ) -> None:
        return None


async def test_summarizes_a_run_produced_by_the_real_decode_suite(
    tmp_path: Path, fake_server: FakeServer
) -> None:
    """偽のサーバーに実際に流した計測ランを、そのまま要約できる。"""
    fake_server.set_response(token_stream_response(output_tokens=20, gap_s=0.002))
    outcome = await execute_run(
        RunRequest(target_name="fake", suites=[SuiteName.DECODE], profile_name="e2e"),
        _SilentProgress(),
        targets_path=_write_targets(tmp_path, fake_server.base_url),
        profiles_path=_write_profiles(tmp_path),
        results_root=tmp_path / "results",
        registry={SuiteName.DECODE: DecodeSuite()},
        env={},
    )
    assert outcome.status is RunStatus.COMPLETED

    result = summarize_run(outcome.run_dir)
    assert result.summary.incomplete is False
    conditions = {row.condition for row in result.summary.results}
    assert conditions == {
        "decode/code/en",
        "decode/code/ja",
        "decode/prose/en",
        "decode/prose/ja",
        "decode/json/en",
        "decode/json/ja",
    }
    for condition in conditions:
        for metric in ("decode_tps", "ttft_s"):
            row = _row(result.summary, condition, metric)
            assert row.failures == 0
            assert row.continuous is not None
            assert row.continuous.n == 10, f"{condition} の {metric}: 慣らしが混ざっている"
            assert row.continuous.min > 0.0
            assert MetricFlag.INSUFFICIENT_TRIALS not in row.flags

    json_path, md_path = write_summary(outcome.run_dir)
    written = Summary.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert {row.condition for row in written.results} == conditions
    markdown = md_path.read_text(encoding="utf-8")
    assert all(condition in markdown for condition in conditions)
