"""割合で表す結果の要約 (`analysis/summarize.py`, task 4.2) の試験。

task 4.1 の試験 (`test_analysis_summarize.py`) が速さの行を受け持つので、この
ファイルは、品質の検査の割合 (5.6、5.7) と、長い会話の段階ごとの表 (6.4〜6.6)
と、公開の課題の出どころ (5.5) だけを見る。

確かめること (tasks.md 4.2 の完了の状態):

- 9 種類の分類を含む生データの例から、段階ごとの表ができる
- 崩れ 0 件で 50 回の段階が「試行の数が足りない」、300 回の段階が「下回った」
  になる

そのほか、要件 5.5、5.6、5.7、6.4、6.5、6.6 と、Implementation Notes 4.1
(失敗した試行にも、もっともらしい時刻とトークン数を持たせる。速さの行と
`trial_values` を変えない) に沿った振る舞いを確かめる。

数の作り方の決まり: 分子、分母、要求の失敗、採点できなかった件数は、すべて
違う値にする。同じ値にすると、量を取り違える壊れ方 (要求の失敗を分母に残す、
採点できなかったものを不正解に混ぜる) を検出できない (注 2.2、4.1)。
"""

from __future__ import annotations

import ast
import random
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl, JsonValue

from bench_harness.analysis import summarize as summarize_module
from bench_harness.analysis.summarize import summarize_run, trial_values, write_summary
from bench_harness.store.rawstore import RunStore
from bench_harness.types import (
    AgentSettings,
    ContentBlock,
    DatasetRef,
    MetricFlag,
    MetricResult,
    Profile,
    QualityOutcome,
    QualityTaskKind,
    QualityVerdict,
    RequestError,
    RunManifest,
    RunStatus,
    StreamResult,
    StreamTiming,
    SuiteName,
    Summary,
    TargetDef,
    ThresholdVerdict,
    Tier,
    ToolCallOutcome,
    ToolCallVerdict,
    TrialFlag,
    TrialRecord,
    Usage,
    Verdict,
)

AT = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)
AT_END = datetime(2026, 9, 19, 12, 30, 0, tzinfo=UTC)
RUN_ID = "20260919T120000Z-fake-abcdef"
SENTINEL_REQUEST = "SENTINEL-REQUEST-XYZZY"
"""送った本文だけに置く目印。要約に漏れてはいけない (8.3)。"""
SENTINEL_RESPONSE = "SENTINEL-RESPONSE-PLUGH"
"""応答の本文だけに置く目印。要約に漏れてはいけない (8.3)。"""

ACCURACY = "accuracy"
BREAK_RATE = "agent_break_rate"

ALL_OUTCOMES: tuple[ToolCallOutcome, ...] = tuple(ToolCallOutcome)


# --- 生データの組み立て -----------------------------------------------------


def _manifest(
    *,
    status: RunStatus = RunStatus.RUNNING,
    min_successes: int = 1,
    threshold: float = 0.01,
    suites: Sequence[SuiteName] = (SuiteName.AGENT,),
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
        "profile": Profile(
            name="test",
            min_successes=min_successes,
            agent=AgentSettings(threshold=threshold),
        ),
        "harness_version": "0.1.0+gtest",
        "context_limit": 131072,
    }
    payload.update(overrides)
    return RunManifest.model_validate(payload)


def _store(tmp_path: Path, *, root: str = "results", **manifest_overrides: Any) -> RunStore:
    return RunStore.create(tmp_path / root, _manifest(**manifest_overrides))


def _ns(seconds: float) -> int:
    return round(seconds * 1_000_000_000)


def _trial(
    *,
    condition: str,
    suite: SuiteName = SuiteName.AGENT,
    trial_index: int = 0,
    warmup: bool = False,
    tier: Tier = "primary",
    sent_s: float = 0.0,
    first_s: float | None = None,
    last_s: float | None = None,
    end_s: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int = 0,
    failed: bool = False,
    flags: Sequence[TrialFlag] = (),
    verdict: Verdict | None = None,
    target_input_tokens: int | None = None,
    body_ref: str = "0" * 64,
    text: str = "",
) -> TrialRecord:
    """1 つの試行のレコードを、時刻とトークン数を直に指定して作る。"""
    usage = (
        None
        if input_tokens is None
        else Usage(input_tokens=input_tokens, output_tokens=output_tokens)
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
        request_body_ref=body_ref,
        result=StreamResult(
            timing=timing,
            usage=usage,
            stop_reason=None if failed else "end_turn",
            blocks=[] if not text else [ContentBlock(type="text", text=text)],
            error=RequestError(kind="protocol", message="") if failed else None,
        ),
        flags=list(flags),
        verdict=verdict,
        target_input_tokens=target_input_tokens,
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


def _agent_records(
    *,
    key: str,
    outcomes: Sequence[ToolCallOutcome],
    target_tokens: int | None,
    actual_tokens: Sequence[int] | None = None,
    warmup: bool = False,
    base_s: float = 0.0,
) -> list[TrialRecord]:
    """1 つの段階の試行を作る。

    要求が失敗した試行にも、もっともらしい時刻とトークン数を持たせる (注 4.1)。
    時刻は試行ごとにすべて違う。
    """
    records: list[TrialRecord] = []
    for index, outcome in enumerate(outcomes):
        failed = outcome is ToolCallOutcome.REQUEST_FAILED
        start = base_s + index * 10.0
        tokens = (
            actual_tokens[index]
            if actual_tokens is not None
            else ((target_tokens or 1000) + index * 7)
        )
        records.append(
            _trial(
                condition=key,
                suite=SuiteName.AGENT,
                trial_index=index,
                warmup=warmup,
                sent_s=start,
                first_s=start + 0.5 + index * 0.001,
                last_s=start + 1.5 + index * 0.002,
                input_tokens=tokens,
                output_tokens=20 + index,
                failed=failed,
                verdict=ToolCallVerdict(outcome=outcome),
                target_input_tokens=target_tokens,
            )
        )
    return records


def _repeat(outcome: ToolCallOutcome, count: int) -> list[ToolCallOutcome]:
    return [outcome] * count


# --- 長い会話の段階ごとの表 (6.4) -------------------------------------------


def test_a_stage_with_all_nine_outcomes_produces_the_stage_table(tmp_path: Path) -> None:
    """9 種類の分類を含む生データの例から、段階ごとの表ができる (tasks.md 4.2)。

    崩れた割合は、`correct` と `request_failed` を除いた 7 種類 ÷ (試行 − 要求の
    失敗) である (design.md scoring/toolcall)。
    """
    store = _store(tmp_path)
    _append(
        store, *_agent_records(key="agent/stage/020k", outcomes=ALL_OUTCOMES, target_tokens=20000)
    )
    summary = _summary(_finish(store))

    assert summary.agent is not None
    assert summary.agent.threshold == pytest.approx(0.01)
    assert len(summary.agent.stages) == 1
    stage = summary.agent.stages[0]
    assert stage.stage_key == "agent/stage/020k"
    assert stage.target_input_tokens == 20000
    assert stage.trials == 9
    assert stage.request_failures == 1
    # 9 種類のすべての鍵が、0 件のものも含めて出る
    assert set(stage.outcome_counts) == set(ToolCallOutcome)
    assert all(count == 1 for count in stage.outcome_counts.values())
    assert stage.break_rate is not None
    assert stage.break_rate.numerator == 7, "correct か request_failed が崩れに混ざっている"
    assert stage.break_rate.denominator == 8, "要求の失敗が分母に残っている"
    assert stage.break_rate.rate == pytest.approx(7.0 / 8.0)
    assert stage.verdict is ThresholdVerdict.ABOVE
    assert stage.actual_input_tokens is not None
    assert stage.actual_input_tokens.n == 9


def test_all_nine_outcome_keys_appear_even_when_only_one_kind_occurred(tmp_path: Path) -> None:
    """起きなかった分類も、0 件として表に出す (6.4)。"""
    store = _store(tmp_path)
    _append(
        store,
        *_agent_records(
            key="agent/stage/020k",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 4),
            target_tokens=20000,
        ),
    )
    summary = _summary(_finish(store))

    assert summary.agent is not None
    counts = summary.agent.stages[0].outcome_counts
    assert set(counts) == set(ToolCallOutcome)
    assert counts[ToolCallOutcome.CORRECT] == 4
    assert counts[ToolCallOutcome.NO_CALL] == 0
    assert counts[ToolCallOutcome.REQUEST_FAILED] == 0


# --- しきい値の判定 (6.5) ---------------------------------------------------


def test_zero_breaks_needs_enough_trials_before_it_counts_as_below(tmp_path: Path) -> None:
    """崩れ 0 件で 50 回は「試行の数が足りない」、300 回は「下回った」(tasks.md 4.2)。

    「下回った」は片側 95% の上限で判定する。両側の上限で判定すると、300 回でも
    「試行の数が足りない」のままになる (stats の `threshold_verdict`)。
    """
    store = _store(tmp_path)
    _append(
        store,
        *_agent_records(
            key="agent/stage/020k",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 50),
            target_tokens=20000,
        ),
        *_agent_records(
            key="agent/stage/040k",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 300),
            target_tokens=40000,
            base_s=1000.0,
        ),
    )
    summary = _summary(_finish(store))

    assert summary.agent is not None
    short, long = summary.agent.stages
    assert short.stage_key == "agent/stage/020k"
    assert short.break_rate is not None
    assert (short.break_rate.numerator, short.break_rate.denominator) == (0, 50)
    assert short.verdict is ThresholdVerdict.UNDETERMINED
    assert long.stage_key == "agent/stage/040k"
    assert long.break_rate is not None
    assert (long.break_rate.numerator, long.break_rate.denominator) == (0, 300)
    assert long.verdict is ThresholdVerdict.BELOW
    assert summary.agent.first_exceeded_tokens is None


def test_a_stage_above_the_threshold_sets_the_first_exceeded_length(tmp_path: Path) -> None:
    """しきい値を初めて超えた長さを、実際の入力のトークン数 (中央値) で示す (6.6)。"""
    store = _store(tmp_path)
    _append(
        store,
        *_agent_records(
            key="agent/stage/020k",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 20),
            target_tokens=20000,
            actual_tokens=[20000] * 20,
        ),
        *_agent_records(
            key="agent/stage/040k",
            outcomes=[
                *_repeat(ToolCallOutcome.NO_CALL, 5),
                *_repeat(ToolCallOutcome.CORRECT, 15),
            ],
            target_tokens=40000,
            # 中央値が 40100 になる並び (狙いの 40000 とは違う値にする)
            actual_tokens=[40005 + index * 10 for index in range(20)],
            base_s=1000.0,
        ),
        *_agent_records(
            key="agent/stage/060k",
            outcomes=_repeat(ToolCallOutcome.WRONG_CALL, 10),
            target_tokens=60000,
            base_s=2000.0,
        ),
    )
    summary = _summary(_finish(store))

    assert summary.agent is not None
    stages = {stage.stage_key: stage for stage in summary.agent.stages}
    assert stages["agent/stage/040k"].verdict is ThresholdVerdict.ABOVE
    assert stages["agent/stage/040k"].break_rate is not None
    assert stages["agent/stage/040k"].break_rate.rate == pytest.approx(0.25)
    assert stages["agent/stage/040k"].actual_input_tokens is not None
    assert stages["agent/stage/040k"].actual_input_tokens.median == pytest.approx(40100.0)
    assert summary.agent.first_exceeded_tokens == 40100, "最初に超えた段階の実際の長さ"
    assert summary.agent.reached_tokens == 60000


def test_first_exceeded_needs_a_rate_strictly_greater_than_the_threshold(tmp_path: Path) -> None:
    """ちょうどしきい値と同じ段階は、「超えた」に数えない (6.6)。"""
    store = _store(tmp_path, threshold=0.01)
    _append(
        store,
        *_agent_records(  # 1 / 100 = 0.01。しきい値とちょうど同じ
            key="agent/stage/020k",
            outcomes=[ToolCallOutcome.NO_CALL, *_repeat(ToolCallOutcome.CORRECT, 99)],
            target_tokens=20000,
            actual_tokens=[20000] * 100,
        ),
        *_agent_records(  # 2 / 100 = 0.02。ここで初めて超える
            key="agent/stage/040k",
            outcomes=[
                *_repeat(ToolCallOutcome.WRONG_CALL, 2),
                *_repeat(ToolCallOutcome.CORRECT, 98),
            ],
            target_tokens=40000,
            actual_tokens=[40000] * 100,
            base_s=2000.0,
        ),
    )
    summary = _summary(_finish(store))

    assert summary.agent is not None
    first = summary.agent.stages[0]
    assert first.break_rate is not None
    assert first.break_rate.rate == pytest.approx(0.01)
    assert summary.agent.first_exceeded_tokens == 40000, "ちょうど同じ段階を超えたと数えている"


def test_request_failures_stay_out_of_the_denominator_and_are_shown_apart(tmp_path: Path) -> None:
    """要求そのものの失敗は、崩れた割合の分母から外し、件数を別に示す (6.4)。

    分子 5、分母 17、要求の失敗 3、試行 20 と、すべて違う値にしてある。
    """
    store = _store(tmp_path)
    _append(
        store,
        *_agent_records(
            key="agent/stage/020k",
            outcomes=[
                *_repeat(ToolCallOutcome.WRONG_CALL, 5),
                *_repeat(ToolCallOutcome.REQUEST_FAILED, 3),
                *_repeat(ToolCallOutcome.CORRECT, 12),
            ],
            target_tokens=20000,
        ),
    )
    summary = _summary(_finish(store))

    assert summary.agent is not None
    stage = summary.agent.stages[0]
    assert stage.trials == 20
    assert stage.request_failures == 3
    assert stage.break_rate is not None
    assert (stage.break_rate.numerator, stage.break_rate.denominator) == (5, 17)

    row = _row(summary, "agent/stage/020k", BREAK_RATE)
    assert row.proportion == stage.break_rate, "表の行と段階の割合が食い違っている"
    assert row.failures == 3
    assert row.flag_counts["not_scored"] == 3


def test_agent_warmups_never_reach_the_stage_table(tmp_path: Path) -> None:
    """慣らしの試行は、段階の集計に入れない (2.5)。"""
    store = _store(tmp_path)
    _append(
        store,
        *_agent_records(
            key="agent/stage/020k",
            outcomes=_repeat(ToolCallOutcome.NO_CALL, 4),
            target_tokens=20000,
            warmup=True,
        ),
        *_agent_records(
            key="agent/stage/020k",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 6),
            target_tokens=20000,
            base_s=500.0,
        ),
    )
    summary = _summary(_finish(store))

    assert summary.agent is not None
    stage = summary.agent.stages[0]
    assert stage.trials == 6, "慣らしが試行の数に混ざっている"
    assert stage.outcome_counts[ToolCallOutcome.NO_CALL] == 0
    assert stage.break_rate is not None
    assert (stage.break_rate.numerator, stage.break_rate.denominator) == (0, 6)
    assert stage.actual_input_tokens is not None
    assert stage.actual_input_tokens.n == 6


def test_stages_are_ordered_by_length_even_when_the_records_are_shuffled(tmp_path: Path) -> None:
    """段階は、会話の長さの順に並べる (実行の順ではない)。"""
    store = _store(tmp_path)
    records: list[TrialRecord] = []
    for index, tokens in enumerate((60000, 20000, 120000, 40000)):
        records.extend(
            _agent_records(
                key=f"agent/stage/{tokens // 1000:03d}k",
                outcomes=_repeat(ToolCallOutcome.CORRECT, 3),
                target_tokens=tokens,
                base_s=index * 1000.0,
            )
        )
    random.Random(7).shuffle(records)
    _append(store, *records)
    summary = _summary(_finish(store))

    assert summary.agent is not None
    assert [stage.target_input_tokens for stage in summary.agent.stages] == [
        20000,
        40000,
        60000,
        120000,
    ]


def test_a_run_stopped_by_the_context_limit_reports_the_reach_and_the_reason(
    tmp_path: Path,
) -> None:
    """上限に達して止まったら、到達した長さと理由を要約に残す (6.9)。"""
    store = _store(
        tmp_path,
        skipped=[
            {
                "suite": SuiteName.AGENT,
                "key": "agent/stage/080k",
                "reason": "入力の長さの上限に達したので、この段階で止めた",
            }
        ],
    )
    _append(
        store,
        *_agent_records(
            key="agent/stage/020k",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 3),
            target_tokens=20000,
        ),
        *_agent_records(
            key="agent/stage/060k",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 3),
            target_tokens=60000,
            base_s=1000.0,
        ),
    )
    summary = _summary(_finish(store, RunStatus.COMPLETED))

    assert summary.agent is not None
    assert summary.agent.reached_tokens == 60000
    assert summary.agent.stopped_reason == "入力の長さの上限に達したので、この段階で止めた"


def test_a_stage_that_only_failed_never_counts_as_reached(tmp_path: Path) -> None:
    """採点できた試行が 1 つもない段階は、到達した長さに数えない (6.9)。"""
    store = _store(tmp_path)
    _append(
        store,
        *_agent_records(
            key="agent/stage/020k",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 3),
            target_tokens=20000,
        ),
        *_agent_records(
            key="agent/stage/040k",
            outcomes=_repeat(ToolCallOutcome.REQUEST_FAILED, 4),
            target_tokens=40000,
            base_s=1000.0,
        ),
    )
    summary = _summary(_finish(store))

    assert summary.agent is not None
    late = summary.agent.stages[1]
    assert late.trials == 4
    assert late.request_failures == 4
    assert late.break_rate is None, "分母が 0 の段階に割合を出している"
    assert late.verdict is ThresholdVerdict.UNDETERMINED
    assert summary.agent.reached_tokens == 20000
    assert _row(summary, "agent/stage/040k", BREAK_RATE).proportion is None


def test_agent_data_that_is_only_a_skip_still_makes_a_summary(tmp_path: Path) -> None:
    """段階のレコードが 1 つもなくても、飛ばした理由だけは残す (6.9)。"""
    store = _store(
        tmp_path,
        suites=[SuiteName.DECODE, SuiteName.AGENT],
        skipped=[
            {
                "suite": SuiteName.AGENT,
                "key": "agent/stage/020k",
                "reason": "入力の長さの上限より長いので、送る前に飛ばした",
            }
        ],
    )
    _append(
        store,
        _trial(
            condition="decode/code/en",
            suite=SuiteName.DECODE,
            trial_index=0,
            sent_s=0.0,
            first_s=0.5,
            last_s=2.5,
            input_tokens=11,
            output_tokens=41,
        ),
    )
    summary = _summary(_finish(store))

    assert summary.agent is not None
    assert summary.agent.stages == []
    assert summary.agent.reached_tokens is None
    assert summary.agent.first_exceeded_tokens is None
    assert summary.agent.stopped_reason == "入力の長さの上限より長いので、送る前に飛ばした"


def test_a_run_without_any_agent_data_has_no_agent_summary(tmp_path: Path) -> None:
    """長い会話の検査を流していない計測ランには、段階の要約を出さない。"""
    store = _store(
        tmp_path,
        suites=[SuiteName.DECODE],
        skipped=[
            {
                "suite": SuiteName.PREFILL,
                "key": "prefill/cold/128k",
                "reason": "入力の長さの上限を超える",
            }
        ],
    )
    _append(
        store,
        _trial(
            condition="decode/code/en",
            suite=SuiteName.DECODE,
            trial_index=0,
            sent_s=0.0,
            first_s=0.5,
            last_s=2.5,
            input_tokens=11,
            output_tokens=41,
        ),
    )
    summary = _summary(_finish(store))

    assert summary.agent is None


# --- 品質の検査の割合 (5.6、5.7) --------------------------------------------


def _quality_records(
    *,
    key: str,
    task: QualityTaskKind,
    correct: int,
    incorrect: int,
    not_scored: int,
    request_failed: int,
    base_s: float = 0.0,
) -> list[TrialRecord]:
    """品質の検査の試行を、採点の内訳から作る。

    `not_scored` は、要求は成功したのに採点できなかったもの (隔離の実行環境が
    ない、など)。`request_failed` は、要求そのものが失敗したもの。どちらも
    「採点できなかった」に数えるが、要求の失敗の件数は別に出る。
    """
    outcomes: list[tuple[QualityOutcome, bool]] = [
        *[(QualityOutcome.CORRECT, False)] * correct,
        *[(QualityOutcome.INCORRECT, False)] * incorrect,
        *[(QualityOutcome.NOT_SCORED, False)] * not_scored,
        *[(QualityOutcome.NOT_SCORED, True)] * request_failed,
    ]
    records: list[TrialRecord] = []
    for index, (outcome, failed) in enumerate(outcomes):
        start = base_s + index * 10.0
        records.append(
            _trial(
                condition=key,
                suite=SuiteName.QUALITY,
                trial_index=index,
                sent_s=start,
                first_s=start + 0.5,
                last_s=start + 1.5 + index * 0.001,
                input_tokens=500 + index,
                output_tokens=30 + index,
                failed=failed,
                verdict=QualityVerdict(task=task, outcome=outcome),
            )
        )
    return records


def test_quality_accuracy_splits_the_unscored_from_the_wrong_answers(tmp_path: Path) -> None:
    """採点できなかった件数を、不正解と分けて示す (5.7)。分母は採点できた試行だけ。

    分子 5、分母 8、採点できなかった 4、要求の失敗 2、試行 12 と、すべて違う値に
    してある。
    """
    store = _store(tmp_path, suites=[SuiteName.QUALITY])
    _append(
        store,
        *_quality_records(
            key="quality/code/humaneval+",
            task="code",
            correct=5,
            incorrect=3,
            not_scored=2,
            request_failed=2,
        ),
    )
    summary = _summary(_finish(store))

    row = _row(summary, "quality/code/humaneval+", ACCURACY)
    assert row.proportion is not None
    assert row.proportion.numerator == 5
    assert row.proportion.denominator == 8, "採点できなかった試行が分母に残っている"
    assert row.proportion.rate == pytest.approx(0.625)
    assert 0.0 < row.proportion.ci95_low < 0.625 < row.proportion.ci95_high < 1.0
    assert row.flag_counts["not_scored"] == 4
    assert row.failures == 2
    assert row.continuous is None


def test_toolcall_quality_reads_the_tool_call_verdict(tmp_path: Path) -> None:
    """`quality/toolcall` の判定は `ToolCallVerdict`。正解は `CORRECT` だけ (5.1)。

    `REQUEST_FAILED` は採点できなかったもの、ほかの 7 種類は不正解である。
    """
    store = _store(tmp_path, suites=[SuiteName.QUALITY])
    outcomes = [
        *_repeat(ToolCallOutcome.CORRECT, 6),
        ToolCallOutcome.NO_CALL,
        ToolCallOutcome.WRONG_CALL,
        ToolCallOutcome.ARGS_SCHEMA_INVALID,
        *_repeat(ToolCallOutcome.REQUEST_FAILED, 2),
    ]
    _append(
        store,
        *[
            _trial(
                condition="quality/toolcall",
                suite=SuiteName.QUALITY,
                trial_index=index,
                sent_s=index * 10.0,
                first_s=index * 10.0 + 0.5,
                last_s=index * 10.0 + 1.5,
                input_tokens=400 + index,
                output_tokens=20 + index,
                failed=outcome is ToolCallOutcome.REQUEST_FAILED,
                verdict=ToolCallVerdict(outcome=outcome),
            )
            for index, outcome in enumerate(outcomes)
        ],
    )
    summary = _summary(_finish(store))

    row = _row(summary, "quality/toolcall", ACCURACY)
    assert row.proportion is not None
    assert (row.proportion.numerator, row.proportion.denominator) == (6, 9)
    assert row.flag_counts["not_scored"] == 2
    assert row.failures == 2


def test_needle_conditions_each_get_their_own_row(tmp_path: Path) -> None:
    """探す課題は、長さと位置ごとの条件が、それぞれ 1 行になる (5.3、5.6)。"""
    store = _store(tmp_path, suites=[SuiteName.QUALITY])
    _append(
        store,
        *_quality_records(
            key="quality/needle/8k/d0",
            task="needle",
            correct=3,
            incorrect=1,
            not_scored=0,
            request_failed=0,
        ),
        *_quality_records(
            key="quality/needle/8k/d50",
            task="needle",
            correct=1,
            incorrect=2,
            not_scored=0,
            request_failed=1,
            base_s=1000.0,
        ),
    )
    summary = _summary(_finish(store))

    first = _row(summary, "quality/needle/8k/d0", ACCURACY)
    assert first.proportion is not None
    assert (first.proportion.numerator, first.proportion.denominator) == (3, 4)
    second = _row(summary, "quality/needle/8k/d50", ACCURACY)
    assert second.proportion is not None
    assert (second.proportion.numerator, second.proportion.denominator) == (1, 3)
    assert second.failures == 1


def test_a_trial_without_a_verdict_is_counted_as_unscored_and_warned(tmp_path: Path) -> None:
    """判定のない試行は、黙って落とさず、採点できなかったものに数えて警告する。"""
    store = _store(tmp_path, suites=[SuiteName.QUALITY])
    _append(
        store,
        *_quality_records(
            key="quality/toolcall",
            task="toolcall",
            correct=2,
            incorrect=1,
            not_scored=0,
            request_failed=0,
        ),
        _trial(  # まとまりの不具合で、判定が付かなかった試行
            condition="quality/toolcall",
            suite=SuiteName.QUALITY,
            trial_index=99,
            sent_s=500.0,
            first_s=500.5,
            last_s=501.5,
            input_tokens=600,
            output_tokens=40,
        ),
    )
    result = summarize_run(_finish(store).run_dir)

    row = _row(result.summary, "quality/toolcall", ACCURACY)
    assert row.proportion is not None
    assert (row.proportion.numerator, row.proportion.denominator) == (2, 3)
    assert row.flag_counts["not_scored"] == 1
    assert any("quality/toolcall" in warning for warning in result.warnings), (
        "判定のない試行が、黙って落とされている"
    )


def test_a_quality_condition_with_nothing_scored_keeps_its_row(tmp_path: Path) -> None:
    """採点できた試行が 1 つもなくても、行は出す (割合は値なし)。"""
    store = _store(tmp_path, suites=[SuiteName.QUALITY])
    _append(
        store,
        *_quality_records(
            key="quality/code/humaneval+",
            task="code",
            correct=0,
            incorrect=0,
            not_scored=3,
            request_failed=0,
        ),
    )
    summary = _summary(_finish(store))

    row = _row(summary, "quality/code/humaneval+", ACCURACY)
    assert row.proportion is None
    assert row.flag_counts["not_scored"] == 3


def test_proportion_rows_never_carry_the_insufficient_trials_flag(tmp_path: Path) -> None:
    """割合の行には「試行の数が足りない」の印を付けない (不確かさは区間が表す)。"""
    store = _store(tmp_path, min_successes=100, suites=[SuiteName.QUALITY, SuiteName.AGENT])
    _append(
        store,
        *_quality_records(
            key="quality/toolcall",
            task="toolcall",
            correct=1,
            incorrect=1,
            not_scored=0,
            request_failed=0,
        ),
        *_agent_records(
            key="agent/stage/020k",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 2),
            target_tokens=20000,
            base_s=1000.0,
        ),
    )
    summary = _summary(_finish(store))

    for condition, metric in (("quality/toolcall", ACCURACY), ("agent/stage/020k", BREAK_RATE)):
        row = _row(summary, condition, metric)
        assert MetricFlag.INSUFFICIENT_TRIALS not in row.flags, f"{condition} に印が付いた"


# --- 壊れた形の生データに耐える ---------------------------------------------


def test_unknown_quality_and_agent_condition_keys_do_not_raise(tmp_path: Path) -> None:
    """後のタスクが足す条件の鍵でも、例外にせず 1 行にする。"""
    store = _store(tmp_path, suites=[SuiteName.QUALITY, SuiteName.AGENT])
    _append(
        store,
        *_quality_records(
            key="quality/something/new",
            task="toolcall",
            correct=1,
            incorrect=1,
            not_scored=0,
            request_failed=0,
        ),
        *_agent_records(
            key="agent/something/new",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 2),
            target_tokens=20000,
            base_s=1000.0,
        ),
    )
    result = summarize_run(_finish(store).run_dir)

    assert _row(result.summary, "quality/something/new", ACCURACY).proportion is not None
    assert _row(result.summary, "agent/something/new", BREAK_RATE).proportion is not None
    assert result.summary.agent is not None
    assert [stage.target_input_tokens for stage in result.summary.agent.stages] == [20000]


def test_a_stage_key_without_a_length_falls_back_to_the_records(tmp_path: Path) -> None:
    """鍵から長さが読めない段階は、レコードの狙いの長さを使う。"""
    store = _store(tmp_path)
    _append(
        store,
        *_agent_records(
            key="agent/stage/unknown",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 3),
            target_tokens=33000,
        ),
    )
    summary = _summary(_finish(store))

    assert summary.agent is not None
    assert [stage.target_input_tokens for stage in summary.agent.stages] == [33000]


def test_a_stage_without_any_length_is_left_out_with_a_warning(tmp_path: Path) -> None:
    """長さがまったく決められない段階は、表から外して警告する (黙って捨てない)。"""
    store = _store(tmp_path)
    _append(
        store,
        *_agent_records(
            key="agent/stage/unknown",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 3),
            target_tokens=None,
        ),
        *_agent_records(
            key="agent/stage/020k",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 3),
            target_tokens=20000,
            base_s=1000.0,
        ),
    )
    result = summarize_run(_finish(store).run_dir)

    assert result.summary.agent is not None
    assert [stage.stage_key for stage in result.summary.agent.stages] == ["agent/stage/020k"]
    assert any("agent/stage/unknown" in warning for warning in result.warnings)
    # 行そのものは残る (4.3 が比べられるように)
    assert _row(result.summary, "agent/stage/unknown", BREAK_RATE).proportion is not None


def test_a_verdict_of_the_wrong_kind_does_not_raise(tmp_path: Path) -> None:
    """まとまりに合わない種類の判定が来ても、例外にせず、分母から外して警告する。"""
    store = _store(tmp_path)
    _append(
        store,
        *_agent_records(
            key="agent/stage/020k",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 3),
            target_tokens=20000,
        ),
        _trial(  # 長い会話の検査に、品質の検査の判定が混ざった
            condition="agent/stage/020k",
            suite=SuiteName.AGENT,
            trial_index=50,
            sent_s=500.0,
            first_s=500.5,
            last_s=501.5,
            input_tokens=20500,
            output_tokens=40,
            verdict=QualityVerdict(task="toolcall", outcome=QualityOutcome.CORRECT),
            target_input_tokens=20000,
        ),
    )
    result = summarize_run(_finish(store).run_dir)

    assert result.summary.agent is not None
    stage = result.summary.agent.stages[0]
    assert stage.trials == 4
    assert stage.break_rate is not None
    assert (stage.break_rate.numerator, stage.break_rate.denominator) == (0, 3)
    assert any("agent/stage/020k" in warning for warning in result.warnings)


# --- 速さの行と `trial_values` を変えない (4.1 との両立) ---------------------


def _speed_records() -> list[TrialRecord]:
    """速さのまとまりの生データ。品質と長い会話の有無で、変わってはいけない。"""
    return [
        _trial(
            condition="decode/code/en",
            suite=SuiteName.DECODE,
            trial_index=0,
            sent_s=0.0,
            first_s=0.5,
            last_s=2.5,
            input_tokens=11,
            output_tokens=41,
            text=SENTINEL_RESPONSE,
        ),
        _trial(
            condition="decode/code/en",
            suite=SuiteName.DECODE,
            trial_index=1,
            sent_s=3.0,
            first_s=3.25,
            last_s=4.25,
            input_tokens=12,
            output_tokens=31,
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
        ),
    ]


def _proportion_records() -> list[TrialRecord]:
    return [
        *_quality_records(
            key="quality/toolcall",
            task="toolcall",
            correct=4,
            incorrect=2,
            not_scored=1,
            request_failed=3,
            base_s=100.0,
        ),
        *_agent_records(
            key="agent/stage/020k",
            outcomes=ALL_OUTCOMES,
            target_tokens=20000,
            base_s=2000.0,
        ),
    ]


_MIXED_SUITES = (
    SuiteName.DECODE,
    SuiteName.PREFILL,
    SuiteName.QUALITY,
    SuiteName.AGENT,
)


def _mixed_bed(tmp_path: Path, *, root: str = "results") -> RunStore:
    """速さ、品質、長い会話をひととおり含む生データ。"""
    store = _store(
        tmp_path,
        root=root,
        suites=_MIXED_SUITES,
        datasets=[
            DatasetRef(
                name="HumanEval+",
                version="v0.1.10",
                source_url="https://example.invalid/humanevalplus",
                license="Apache-2.0",
                scoring_method="生成したコードを隔離して動かし、課題の検査に通るかで採点する",
                sha256="a" * 64,
            )
        ],
    )
    body: dict[str, JsonValue] = {
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": SENTINEL_REQUEST}],
    }
    store.put_body(body)
    _append(store, *_speed_records(), *_proportion_records())
    return _finish(store)


def test_speed_rows_and_trial_values_are_untouched_by_the_new_rows(tmp_path: Path) -> None:
    """割合の行が増えても、速さの行と `trial_values` は 1 つも変わらない (4.1、4.3)。"""
    mixed = _mixed_bed(tmp_path, root="mixed")
    speed_only = _store(tmp_path, root="speed", suites=[SuiteName.DECODE, SuiteName.PREFILL])
    _append(speed_only, *_speed_records())
    _finish(speed_only)

    mixed_summary = summarize_run(mixed.run_dir).summary
    speed_summary = summarize_run(speed_only.run_dir).summary
    speed_metrics = {"ttft_s", "decode_tps", "prefill_tps", "round_total_tps"}
    assert [row for row in mixed_summary.results if row.metric in speed_metrics] == (
        speed_summary.results
    )
    assert speed_summary.results, "試験の前提: 速さの行がある"

    mixed_store = RunStore.open(mixed.run_dir)
    speed_store = RunStore.open(speed_only.run_dir)
    mixed_values = trial_values(mixed_store.read_trials()[0], mixed_store.manifest())
    speed_values = trial_values(speed_store.read_trials()[0], speed_store.manifest())
    assert mixed_values == speed_values, "`trial_values` に割合の行が漏れている"
    assert set(mixed_values) == {
        ("decode/code/en", "decode_tps"),
        ("decode/code/en", "ttft_s"),
        ("prefill/cold/8k", "ttft_s"),
        ("prefill/cold/8k", "prefill_tps"),
    }


def test_proportion_rows_are_in_the_order_the_conditions_were_measured(tmp_path: Path) -> None:
    """行の並びは、生データに現れた順のまま (何度実行しても同じ結果になるため)。"""
    store = _mixed_bed(tmp_path)
    summary = summarize_run(store.run_dir).summary
    seen: list[str] = []
    for row in summary.results:
        if row.condition not in seen:
            seen.append(row.condition)
    assert seen == [
        "decode/code/en",
        "prefill/cold/8k",
        "quality/toolcall",
        "agent/stage/020k",
    ]


# --- 2 つの要約 (5.5、8.2、8.3) ---------------------------------------------


def test_markdown_has_the_quality_and_agent_sections(tmp_path: Path) -> None:
    """品質の検査と、長い会話の段階の節が出る (5.6、5.7、6.4、6.6)。"""
    store = _mixed_bed(tmp_path)
    _, md_path = write_summary(store.run_dir)
    text = md_path.read_text(encoding="utf-8")

    assert "## 品質の検査" in text
    assert "## 長い会話でのツール呼び出し" in text
    assert "quality/toolcall" in text
    assert "agent/stage/020k" in text
    assert "採点できなかった" in text
    assert "崩れた割合" in text
    # 9 種類の分類の内訳
    for outcome in ToolCallOutcome:
        assert outcome.value in text, f"{outcome.value} の内訳が出ていない"
    # 判定は日本語で出す
    assert "上回った" in text
    # 表の読み方に、割合の決まりを書き足す
    assert "二項" in text
    assert "下回った" in text
    assert "試行の数が足りない" in text


def test_markdown_shows_how_many_trials_a_zero_break_stage_would_need(tmp_path: Path) -> None:
    """崩れ 0 件で「試行の数が足りない」なら、あと何回要るかを示す (6.5)。"""
    store = _store(tmp_path)
    _append(
        store,
        *_agent_records(
            key="agent/stage/020k",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 50),
            target_tokens=20000,
        ),
    )
    _, md_path = write_summary(_finish(store).run_dir)
    text = md_path.read_text(encoding="utf-8")

    assert "試行の数が足りない" in text
    assert "299" in text, "崩れ 0 件で要る試行の数 (しきい値 1% なら 299) が出ていない"


def test_markdown_shows_the_first_exceeded_and_the_reached_length(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        skipped=[
            {
                "suite": SuiteName.AGENT,
                "key": "agent/stage/060k",
                "reason": "入力の長さの上限に達した",
            }
        ],
    )
    _append(
        store,
        *_agent_records(
            key="agent/stage/020k",
            outcomes=_repeat(ToolCallOutcome.CORRECT, 20),
            target_tokens=20000,
            actual_tokens=[20000] * 20,
        ),
        *_agent_records(
            key="agent/stage/040k",
            outcomes=_repeat(ToolCallOutcome.NO_CALL, 20),
            target_tokens=40000,
            actual_tokens=[40000] * 20,
            base_s=1000.0,
        ),
    )
    _, md_path = write_summary(_finish(store).run_dir)
    text = md_path.read_text(encoding="utf-8")

    assert "40000" in text
    assert "入力の長さの上限に達した" in text


def test_markdown_shows_the_public_datasets(tmp_path: Path) -> None:
    """公開の課題の名前、版、ライセンス、採点の方法、入手先を載せる (5.5)。"""
    store = _mixed_bed(tmp_path)
    _, md_path = write_summary(store.run_dir)
    text = md_path.read_text(encoding="utf-8")

    assert "HumanEval+" in text
    assert "v0.1.10" in text
    assert "Apache-2.0" in text
    assert "https://example.invalid/humanevalplus" in text
    assert "隔離して動かし" in text


def test_the_summaries_never_carry_any_body_text(tmp_path: Path) -> None:
    """割合の行が増えても、送った内容と応答の本文は要約に出ない (8.3)。"""
    store = _mixed_bed(tmp_path)
    json_path, md_path = write_summary(store.run_dir)

    for path in (json_path, md_path):
        text = path.read_text(encoding="utf-8")
        assert SENTINEL_REQUEST not in text
        assert SENTINEL_RESPONSE not in text
    restored = Summary.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert restored.agent is not None
    assert restored.agent.stages


def test_write_summary_is_byte_identical_on_a_second_run(tmp_path: Path) -> None:
    """割合の行を含む要約も、何度書いても同じバイト列になる (Batch の約束)。"""
    store = _mixed_bed(tmp_path)
    json_path, md_path = write_summary(store.run_dir)
    first = (json_path.read_bytes(), md_path.read_bytes())
    write_summary(store.run_dir)
    assert (json_path.read_bytes(), md_path.read_bytes()) == first


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
    }
    for module in modules:
        root = module.split(".")[0]
        if root == "bench_harness":
            assert module in allowed_internal, f"読み込んではいけない module: {module}"
        else:
            assert root in sys.stdlib_module_names or root == "pydantic", (
                f"標準ライブラリと pydantic 以外を読み込んでいる: {module}"
            )
