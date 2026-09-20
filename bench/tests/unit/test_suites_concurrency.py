"""同時処理のまとまりの試験 (task 3.4: suites/concurrency)。

`fake_server` は、届いた要求の到着時刻 (`time.monotonic_ns()`) と、届いた
時点で処理中だった本数を記録する (`fake_server.py` の docstring「記録」)。
これを使って、1 回ぶんの n 本が実際に同時に送られたことを確かめる。

時刻を判定する試験 (送り始めの幅) は、tasks.md Implementation Notes 1.4 の
決まりに従う: まとまり自身の慣らしの回で接続を温め、複数の回を測って、個々の
回ではなく中央値で判定し、下限は確かめない。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic import JsonValue

from bench_harness.client.messages import HttpxMessagesClient
from bench_harness.corpus.synth import TemplateCorpus
from bench_harness.suites import concurrency as concurrency_module
from bench_harness.suites.base import (
    DEFAULT_CONNECTION_WARMUP_TRIALS,
    ConditionAborted,
    SuiteContext,
    make_suite_context,
    plan_condition,
    trial_seed,
)
from bench_harness.suites.concurrency import SUITE, ConcurrencySuite
from bench_harness.types import (
    ConditionPlan,
    Profile,
    SkippedCondition,
    SuiteName,
    TargetDef,
    TrialFlag,
    TrialRecord,
)
from fake_server import (
    FakeServer,
    Script,
    http_error_response,
    replace,
    token_stream_response,
)

RUN_ID = "20260920-000000-abcdef"


# --- 助け (test_suites_base.py と同じ形。別のテストファイルからは import しない) --


class BodySink:
    """`RunStore.put_body` の代わり。保存された本文をそのまま覚えておく。"""

    def __init__(self) -> None:
        self.bodies: list[dict[str, JsonValue]] = []
        self.refs: list[str] = []

    def put(self, body: dict[str, JsonValue]) -> str:
        canonical = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ref = hashlib.sha256(canonical).hexdigest()
        self.bodies.append(body)
        self.refs.append(ref)
        return ref


class RaisingSink:
    """`fail_at` 回目の呼び出しで、保存そのものが失敗したことを模す。"""

    def __init__(self, fail_at: int) -> None:
        self.calls = 0
        self.fail_at = fail_at

    def put(self, body: dict[str, JsonValue]) -> str:
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("put_body boom")
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()


class SlowSink(BodySink):
    """`slow_call` 回目の保存だけ、`delay_s` だけ遅く終わる。

    合図 (`_RoundGate`) が実際に「その回の全員が保存を終えるまで待つ」ことを
    確かめるための下ごしらえ (レビューの必須 1)。`put_body` は同期の呼び出し
    なので、`time.sleep` でイベントループごと止めてよい (本物の遅い保存を
    模す)。
    """

    def __init__(self, *, slow_call: int, delay_s: float = 0.2) -> None:
        super().__init__()
        self._slow_call = slow_call
        self._delay_s = delay_s
        self.calls = 0

    def put(self, body: dict[str, JsonValue]) -> str:
        self.calls += 1
        if self.calls == self._slow_call:
            time.sleep(self._delay_s)
        return super().put(body)


def make_profile(**overrides: Any) -> Profile:
    """試験用の設定。入力とトークンの比は小さくして、速く終わらせる。"""
    data: dict[str, Any] = {
        "name": "test",
        "seed": 7,
        "timeout": {"connect_s": 5.0, "first_event_s": 5.0, "idle_s": 5.0, "total_s": 20.0},
        "chars_per_token": {"prose_en": 4.0, "prose_ja": 1.6, "code": 3.2, "log": 3.4},
        "concurrency": {"levels": [4], "rounds": 1, "max_tokens": 8, "input_tokens": 40},
    }
    data.update(overrides)
    return Profile.model_validate(data)


def target_for(server: FakeServer, **overrides: Any) -> TargetDef:
    payload: dict[str, Any] = {"name": "fake", "base_url": server.base_url, "model": "fake-model"}
    payload.update(overrides)
    return TargetDef.model_validate(payload)


@asynccontextmanager
async def suite_ctx(
    server: FakeServer,
    *,
    profile: Profile | None = None,
    run_id: str = RUN_ID,
    put_body: Any = None,
) -> AsyncIterator[tuple[SuiteContext, BodySink]]:
    """偽のサーバーにつないだ `SuiteContext` と、本文の受け皿を渡す。"""
    sink = BodySink()
    async with HttpxMessagesClient(server.base_url) as client:
        ctx = make_suite_context(
            client=client,
            profile=profile if profile is not None else make_profile(),
            target=target_for(server),
            run_id=run_id,
            put_body=put_body if put_body is not None else sink.put,
        )
        yield ctx, sink


def _plan_for(
    ctx: SuiteContext,
    *,
    level: int,
    rounds: int,
    max_tokens: int,
    input_tokens: int,
    **overrides: Any,
) -> ConditionPlan:
    kwargs: dict[str, Any] = {
        "suite": SuiteName.CONCURRENCY,
        "key": f"concurrency/c{level}",
        "trials": rounds,
        "max_tokens": max_tokens,
        "concurrency": level,
        "target_input_tokens": input_tokens,
    }
    kwargs.update(overrides)
    plan = plan_condition(ctx, **kwargs)
    assert isinstance(plan, ConditionPlan)
    return plan


def _chunk[T](items: Sequence[T], size: int) -> list[list[T]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _first_message_text(body: dict[str, JsonValue]) -> str:
    """送った本文から、最初の発話の最初のブロックの文字列を取り出す (試験の下ごしらえ)。"""
    messages: Any = body["messages"]
    return str(messages[0]["content"][0]["text"])


async def _collect(
    suite: ConcurrencySuite, ctx: SuiteContext, plan: ConditionPlan
) -> list[TrialRecord]:
    return [record async for record in suite.run_condition(ctx, plan)]


# --- 計画: 鍵、主な結果と参考、設定の値 (4.1、4.4) ------------------------


async def test_plan_builds_one_condition_per_level_with_tiers_and_settings(
    fake_server: FakeServer,
) -> None:
    profile = make_profile(
        concurrency={"levels": [1, 2, 4, 8], "rounds": 3, "max_tokens": 32, "input_tokens": 500}
    )
    async with suite_ctx(fake_server, profile=profile) as (ctx, _):
        planned = SUITE.plan(ctx)

    assert [isinstance(p, ConditionPlan) for p in planned] == [True, True, True, True]
    plans = [p for p in planned if isinstance(p, ConditionPlan)]
    assert [p.key for p in plans] == [
        "concurrency/c1",
        "concurrency/c2",
        "concurrency/c4",
        "concurrency/c8",
    ]
    assert [p.tier for p in plans] == ["primary", "primary", "reference", "reference"]
    assert [p.concurrency for p in plans] == [1, 2, 4, 8]
    for plan in plans:
        assert plan.trials == 3
        assert plan.max_tokens == 32
        assert plan.target_input_tokens == 500
        assert plan.warmup_trials == DEFAULT_CONNECTION_WARMUP_TRIALS


async def test_plan_only_includes_the_selected_levels(fake_server: FakeServer) -> None:
    profile = make_profile(
        concurrency={"levels": [2, 4], "rounds": 2, "max_tokens": 16, "input_tokens": 100}
    )
    async with suite_ctx(fake_server, profile=profile) as (ctx, _):
        planned = SUITE.plan(ctx)

    keys = [p.key for p in planned if isinstance(p, ConditionPlan)]
    assert keys == ["concurrency/c2", "concurrency/c4"]


async def test_plan_skips_a_level_that_does_not_fit_the_context_limit(
    fake_server: FakeServer,
) -> None:
    """3.6 と同じ決まり: 上限に収まらない本数も、送る前に飛ばせる。"""
    profile = make_profile(
        concurrency={"levels": [4], "rounds": 1, "max_tokens": 16, "input_tokens": 120000}
    )
    async with HttpxMessagesClient(fake_server.base_url) as client:
        ctx = make_suite_context(
            client=client,
            profile=profile,
            target=target_for(fake_server),
            run_id=RUN_ID,
            put_body=BodySink().put,
            context_limit=32000,
        )
        planned = SUITE.plan(ctx)

    assert len(planned) == 1
    assert isinstance(planned[0], SkippedCondition)
    assert planned[0].key == "concurrency/c4"


def test_suite_module_exposes_a_ready_instance() -> None:
    assert SUITE.name is SuiteName.CONCURRENCY
    assert isinstance(SUITE, ConcurrencySuite)


# --- 1 回ぶんの入力: 先頭も中身も重ならない (4.1) --------------------------


async def test_streams_in_a_round_have_distinct_headers_and_documents(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(token_stream_response(output_tokens=4))
    n = 4
    profile = make_profile(
        concurrency={"levels": [n], "rounds": 1, "max_tokens": 8, "input_tokens": 40}
    )
    async with suite_ctx(fake_server, profile=profile) as (ctx, sink):
        plan = _plan_for(ctx, level=n, rounds=1, max_tokens=8, input_tokens=40, warmup_trials=0)
        records = await _collect(SUITE, ctx, plan)

    assert len(records) == n
    by_ref = dict(zip(sink.refs, sink.bodies, strict=True))
    bodies = [by_ref[record.request_body_ref] for record in records]
    first_lines = [str(body["system"]).splitlines()[0] for body in bodies]
    documents = [_first_message_text(body) for body in bodies]
    assert len(set(first_lines)) == n
    assert len(set(documents)) == n


async def test_measured_round_gets_the_same_document_but_a_different_header_across_runs(
    fake_server: FakeServer,
) -> None:
    """測った回 r・ストリーム k の文章は計測ランをまたいでも同じ (トークンの試験の RUN_ID の注)。"""
    n = 2
    profile = make_profile(
        concurrency={"levels": [n], "rounds": 2, "max_tokens": 8, "input_tokens": 40}
    )

    async def bodies_for(run_id: str) -> dict[tuple[int, int], dict[str, Any]]:
        fake_server.reset()
        fake_server.set_response(token_stream_response(output_tokens=4))
        async with suite_ctx(fake_server, profile=profile, run_id=run_id) as (ctx, sink):
            plan = _plan_for(ctx, level=n, rounds=2, max_tokens=8, input_tokens=40)
            records = await _collect(SUITE, ctx, plan)
        by_ref = dict(zip(sink.refs, sink.bodies, strict=True))
        return {
            (record.round_id, record.stream_index): by_ref[record.request_body_ref]  # type: ignore[misc]
            for record in records
            if not record.warmup
        }

    first = await bodies_for("20260920-000000-aaaaaa")
    second = await bodies_for("20260920-111111-bbbbbb")

    assert set(first) == set(second)
    assert len(first) == 2 * n  # 2 回 x n 本
    for key in first:
        system_a = str(first[key]["system"]).splitlines()[0]
        system_b = str(second[key]["system"]).splitlines()[0]
        assert system_a != system_b  # 先頭の識別子は run_id を含むので変わる
        message_a = first[key]["messages"][0]["content"][0]["text"]
        message_b = second[key]["messages"][0]["content"][0]["text"]
        assert message_a == message_b  # 文章そのものは run_id に依らない


# --- 送り始め: 同時に送り、時刻を記録する (4.3) -----------------------------


async def test_c4_round_sends_four_requests_with_a_small_median_arrival_spread(
    fake_server: FakeServer,
) -> None:
    """tasks.md 3.4 の完了の状態: 同時 4 本で、送り始めの時刻の幅が 50 ミリ秒以内。

    tasks.md Implementation Notes 1.4 の決まりに従い、まとまり自身の慣らしの
    回で接続を温めたあと、複数の回を測って中央値で判定する。下限は確かめない。
    """
    n = 4
    measured_rounds = 6
    # わずかな遅れを入れて、4 本が確実に重なって処理される時間を作る (in_flight の判定用)。
    # 送り始めの幅そのものには効かない (合図のあと、送る直前に時刻を打つため)
    fake_server.set_response(
        token_stream_response(output_tokens=3, first_delay_s=0.02, gap_s=0.005)
    )
    profile = make_profile(
        concurrency={
            "levels": [n],
            "rounds": measured_rounds,
            "max_tokens": 8,
            "input_tokens": 60,
        }
    )
    async with suite_ctx(fake_server, profile=profile) as (ctx, _):
        plan = _plan_for(ctx, level=n, rounds=measured_rounds, max_tokens=8, input_tokens=60)
        records = await _collect(SUITE, ctx, plan)

    total_rounds = DEFAULT_CONNECTION_WARMUP_TRIALS + measured_rounds
    assert len(records) == total_rounds * n

    requests = fake_server.requests_for("/v1/messages")
    assert len(requests) == len(records)
    request_groups = _chunk(requests, n)
    assert all(len(group) == n for group in request_groups)
    assert all(max(r.in_flight for r in group) == n for group in request_groups)

    measured_request_groups = request_groups[DEFAULT_CONNECTION_WARMUP_TRIALS:]
    assert len(measured_request_groups) == measured_rounds
    arrival_spreads_ms = [
        (max(r.arrived_at_ns for r in group) - min(r.arrived_at_ns for r in group)) / 1e6
        for group in measured_request_groups
    ]
    assert statistics.median(arrival_spreads_ms) < 50.0

    record_groups = _chunk(records, n)[DEFAULT_CONNECTION_WARMUP_TRIALS:]
    sent_spreads_ms = [
        (
            max(r.result.timing.sent_at_ns for r in group)
            - min(r.result.timing.sent_at_ns for r in group)
        )
        / 1e6
        for group in record_groups
    ]
    assert statistics.median(sent_spreads_ms) < 50.0


# --- 一部の失敗 (4.5 は分析の仕事。ここでは記録だけ確かめる) ---------------


async def test_one_failed_stream_still_yields_n_records(fake_server: FakeServer) -> None:
    n = 4
    fake_server.set_response_sequence(
        [
            token_stream_response(output_tokens=4),
            token_stream_response(output_tokens=4),
            http_error_response(500),
            token_stream_response(output_tokens=4),
        ]
    )
    profile = make_profile(
        concurrency={"levels": [n], "rounds": 1, "max_tokens": 8, "input_tokens": 40}
    )
    async with suite_ctx(fake_server, profile=profile) as (ctx, _):
        plan = _plan_for(ctx, level=n, rounds=1, max_tokens=8, input_tokens=40, warmup_trials=0)
        records = await _collect(SUITE, ctx, plan)

    assert len(records) == n
    failures = [record for record in records if record.result.error is not None]
    assert len(failures) == 1
    assert failures[0].result.error is not None
    assert failures[0].result.error.kind == "http"
    assert failures[0].result.error.http_status == 500
    successes = [record for record in records if record.result.error is None]
    assert len(successes) == n - 1
    assert all(record.round_id == 0 for record in records)
    stream_indexes = [record.stream_index for record in records if record.stream_index is not None]
    assert sorted(stream_indexes) == list(range(n))


# --- レコードの中身と、JSON への往復 ----------------------------------------


async def test_record_identity_fields_and_json_round_trip(fake_server: FakeServer) -> None:
    n = 3
    fake_server.set_response(token_stream_response(output_tokens=4))
    profile = make_profile(
        concurrency={"levels": [n], "rounds": 1, "max_tokens": 8, "input_tokens": 40}
    )
    async with suite_ctx(fake_server, profile=profile) as (ctx, _):
        # 計画には plan() の本数ごとの決めごと (4.4) と同じ "reference" を渡す
        # (ここで確かめたいのは、計画の tier がレコードにそのまま運ばれること)
        plan = _plan_for(ctx, level=n, rounds=1, max_tokens=8, input_tokens=40, tier="reference")
        records = await _collect(SUITE, ctx, plan)

    warmup_records = [r for r in records if r.warmup]
    measured_records = [r for r in records if not r.warmup]
    assert len(warmup_records) == DEFAULT_CONNECTION_WARMUP_TRIALS * n
    assert len(measured_records) == n
    assert sorted(r.stream_index for r in warmup_records if r.stream_index is not None) == list(
        range(n)
    )
    assert sorted(r.stream_index for r in measured_records if r.stream_index is not None) == list(
        range(n)
    )
    assert all(r.round_id == 0 for r in warmup_records)
    assert all(r.round_id == 0 for r in measured_records)
    assert all(r.trial_index == 0 for r in records)
    assert all(r.tier == "reference" for r in records)
    for record in records:
        assert TrialRecord.model_validate_json(record.model_dump_json()) == record


# --- 合図が固まらないこと: 合図の前で失敗する本があっても (4.3) -----------


async def test_a_put_body_failure_before_the_gate_does_not_hang_the_round(
    fake_server: FakeServer,
) -> None:
    n = 4
    profile = make_profile(
        concurrency={"levels": [n], "rounds": 1, "max_tokens": 8, "input_tokens": 40}
    )
    sink = RaisingSink(fail_at=2)

    async def consume() -> None:
        async with suite_ctx(fake_server, profile=profile, put_body=sink.put) as (ctx, _):
            plan = _plan_for(ctx, level=n, rounds=1, max_tokens=8, input_tokens=40, warmup_trials=0)
            async for _ in SUITE.run_condition(ctx, plan):
                pass

    with pytest.raises(RuntimeError, match="put_body boom"):
        await asyncio.wait_for(consume(), timeout=5.0)


# --- 中断: task を残さない (base.py の注、runner が SIGINT で打ち切る) -----


async def test_cancelling_run_condition_mid_round_leaves_no_pending_tasks(
    fake_server: FakeServer,
) -> None:
    n = 3
    fake_server.set_response(replace(token_stream_response(output_tokens=4), first_delay_s=5.0))
    profile = make_profile(
        concurrency={"levels": [n], "rounds": 1, "max_tokens": 8, "input_tokens": 40}
    )
    async with suite_ctx(fake_server, profile=profile) as (ctx, _):
        plan = _plan_for(ctx, level=n, rounds=1, max_tokens=8, input_tokens=40, warmup_trials=0)

        async def consume() -> None:
            async for _ in SUITE.run_condition(ctx, plan):
                pass

        before = {task for task in asyncio.all_tasks() if not task.done()}
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.2)
        # 打ち切りが実際に片付くまで、少しだけ道を譲る
        for _ in range(5):
            await asyncio.sleep(0)
        leftover = {task for task in asyncio.all_tasks() if not task.done()} - before
        assert leftover == set()


# --- 合図 (`_RoundGate`) が実際に効いていること (4.3、必須 1) ---------------


async def test_the_gate_holds_every_stream_until_the_slowest_one_has_stored_its_body(
    fake_server: FakeServer,
) -> None:
    """合図なし・最初の 1 本で解放・回をまたいだ使い回しを、まとめて検出する。

    速い保存と速い偽のサーバーでは、合図がなくても n 本はどのみち揃って
    送られてしまう (レビューの指摘)。1 本の保存だけをわざと遅くして、その回の
    送り始めの幅 (`sent_at_ns` の幅) が広がらないこと ―― つまり、遅れた本が
    保存を終えるまで、ほかの本が実際に待たされていること ―― を確かめる。

    「慣らしを含む 3 回目の回」(= 測った 2 回目の回) の最後のストリームを
    遅くすることで、次の 3 つの変異をまとめて検出する。
    (a) `start_gate` を渡さない (合図なし):
        遅い本を待たずに、残りがすぐ送り始めてしまう
    (b) 解放の条件を `>= n` ではなく `>= 1` にする (最初の 1 本で解放):
        遅い本より先に来た 1 本目が、即座に自分も含め全員を解放してしまう
        (2 本目以降は、既に立っている合図をすり抜けて待たずに進む)
    (c) `_RoundGate` を回の間で使い回す:
        1 回目の回 (慣らし) で合図が既に「立った」ままになり、2 回目以降の
        回では、誰も待たなくなる

    いずれの変異でも、遅くした回の送り始めの幅は約 200 ms (遅れの長さ) まで
    広がる。いまの実装は、待ち合わせが効いて数ミリ秒のまま (tasks.md
    Implementation Notes 1.4: 上限の判定であり、200 ms に対して 10 倍以上の
    余裕がある)。
    """
    assert DEFAULT_CONNECTION_WARMUP_TRIALS == 1, (
        "慣らしの回数が変わったら、下の「3 回目の回」の計算も見直すこと"
    )
    n = 4
    measured_rounds = 3
    fake_server.set_response(token_stream_response(output_tokens=3, gap_s=0.0))
    profile = make_profile(
        concurrency={
            "levels": [n],
            "rounds": measured_rounds,
            "max_tokens": 8,
            "input_tokens": 40,
        }
    )
    # 慣らし (1 回) + 測った 1 回目 (n 本) + 測った 2 回目の最後のストリーム、
    # で合わせて 3 * n 回目の保存が「慣らしを含む 3 回目の回」の最後の 1 本になる
    slow_call = n * 3
    sink = SlowSink(slow_call=slow_call)

    async with suite_ctx(fake_server, profile=profile, put_body=sink.put) as (ctx, _):
        plan = _plan_for(ctx, level=n, rounds=measured_rounds, max_tokens=8, input_tokens=40)
        records = await _collect(SUITE, ctx, plan)

    assert sink.calls == (DEFAULT_CONNECTION_WARMUP_TRIALS + measured_rounds) * n
    measured_records = [record for record in records if not record.warmup]
    record_groups = _chunk(measured_records, n)
    assert len(record_groups) == measured_rounds

    spreads_ms = [
        (
            max(record.result.timing.sent_at_ns for record in group)
            - min(record.result.timing.sent_at_ns for record in group)
        )
        / 1e6
        for group in record_groups
    ]
    slow_round_index = 1  # 測った 2 回目の回 (= 慣らしを含めて 3 回目の回)
    assert spreads_ms[slow_round_index] < 50.0
    assert max(spreads_ms) < 50.0


# --- `measures_decode_speed` と `expect_full_output` が固定であること (必須 2) --


async def test_expect_full_output_is_fixed_true_short_stop_reasons_get_flagged(
    fake_server: FakeServer,
) -> None:
    """`run_trial(..., expect_full_output=True)` が固定されていることを、印で確かめる (2.6)。"""
    n = 2
    fake_server.set_response(
        replace(token_stream_response(output_tokens=20), stop_reason="end_turn")
    )
    profile = make_profile(
        concurrency={"levels": [n], "rounds": 1, "max_tokens": 1024, "input_tokens": 40}
    )
    async with suite_ctx(fake_server, profile=profile) as (ctx, _):
        plan = _plan_for(ctx, level=n, rounds=1, max_tokens=1024, input_tokens=40, warmup_trials=0)
        records = await _collect(SUITE, ctx, plan)

    assert len(records) == n
    assert all(TrialFlag.SHORT_OUTPUT in record.flags for record in records)


async def test_measures_decode_speed_is_fixed_true_short_outputs_get_flagged(
    fake_server: FakeServer,
) -> None:
    """`run_trial(..., measures_decode_speed=True)` が固定されていることを、印で確かめる (2.6)。"""
    n = 2
    # output_tokens は 16 未満。stop_reason は既定の "max_tokens" のまま (SHORT_OUTPUT とは分ける)
    fake_server.set_response(token_stream_response(output_tokens=4))
    profile = make_profile(
        concurrency={"levels": [n], "rounds": 1, "max_tokens": 8, "input_tokens": 40}
    )
    async with suite_ctx(fake_server, profile=profile) as (ctx, _):
        plan = _plan_for(ctx, level=n, rounds=1, max_tokens=8, input_tokens=40, warmup_trials=0)
        records = await _collect(SUITE, ctx, plan)

    assert len(records) == n
    assert all(TrialFlag.TOO_FEW_OUTPUT_TOKENS in record.flags for record in records)


# --- 上限に当たったときの打ち切り (3.6、6.9、必須 3) ------------------------


async def test_context_limit_hit_yields_all_n_records_before_aborting(
    fake_server: FakeServer,
) -> None:
    """要件 8.1: 上限に当たっても、n 本のレコードが全部残ってから打ち切られる。

    `abort_if_context_limit` を丸ごと削除しても、`yield` の前に移しても
    (どちらも要件 8.1 を破る) 気づけるように、例外を捕まえたあとの受信済み
    レコードの数そのものを確かめる。
    """
    n = 3
    fake_server.set_chars_per_token(1.0)
    fake_server.set_context_limit(50, advertise=False)  # 小さい上限。申告はしない
    profile = make_profile(
        concurrency={"levels": [n], "rounds": 1, "max_tokens": 8, "input_tokens": 200}
    )
    async with suite_ctx(fake_server, profile=profile) as (ctx, _):
        assert ctx.context_limit is None  # 申告を知らないので、計画では飛ばさない
        plan = _plan_for(ctx, level=n, rounds=1, max_tokens=8, input_tokens=200, warmup_trials=0)

        collected: list[TrialRecord] = []
        with pytest.raises(ConditionAborted):
            async for record in SUITE.run_condition(ctx, plan):
                collected.append(record)

    assert len(collected) == n
    assert all(record.result.error is not None for record in collected)
    for record in collected:
        assert record.result.error is not None
        assert record.result.error.kind == "http"
        assert record.result.error.http_status == 400


# --- レコードの順序: stream_index の昇順 (完了の順ではない) -----------------


async def test_round_records_are_yielded_in_ascending_stream_index_order(
    fake_server: FakeServer,
) -> None:
    """完了の順に `yield` する変異を検出する。

    ストリームごとに違う遅れを付けて、完了の順を意図的に逆転させる
    (stream_index が大きいほど早く終わる)。届いた本文の中身 (文章そのもの)
    から、どのストリームの要求かを言い当てて遅れを割り当てるので、実際の
    到着の順序には依らない。`asyncio.gather` は渡した順 (= stream_index の順)
    で結果を返すので、正しい実装なら、完了の順がばらばらでも
    `records` は昇順のまま届く (`sorted()` を挟まずに確かめる)。
    """
    n = 4
    profile = make_profile(
        concurrency={"levels": [n], "rounds": 1, "max_tokens": 8, "input_tokens": 40}
    )
    condition_key = f"concurrency/c{n}"
    corpus = TemplateCorpus(profile.chars_per_token)
    expected_documents: dict[str, int] = {}
    for stream_index in range(n):
        seed = trial_seed(profile.seed, condition_key, 0, warmup=False, stream_index=stream_index)
        expected_documents[corpus.prose("en", 40, seed)] = stream_index

    def factory(body: dict[str, Any]) -> Script:
        text = str(body["messages"][0]["content"][0]["text"])
        stream_index = next(
            index for document, index in expected_documents.items() if text.startswith(document)
        )
        delay = (n - 1 - stream_index) * 0.03  # stream_index が大きいほど早く終わる
        return replace(token_stream_response(output_tokens=2, gap_s=0.0), first_delay_s=delay)

    fake_server.set_response_factory(factory)

    async with suite_ctx(fake_server, profile=profile) as (ctx, _):
        plan = _plan_for(ctx, level=n, rounds=1, max_tokens=8, input_tokens=40, warmup_trials=0)
        records = await _collect(SUITE, ctx, plan)

    assert len(records) == n
    assert [record.stream_index for record in records] == list(range(n))


# --- 長く書かせる指示が、本文に入っていること -------------------------------


async def test_the_long_output_instruction_is_included_in_every_request(
    fake_server: FakeServer,
) -> None:
    """長く書かせる指示を落とす変異を検出する。"""
    n = 2
    fake_server.set_response(token_stream_response(output_tokens=4))
    profile = make_profile(
        concurrency={"levels": [n], "rounds": 1, "max_tokens": 8, "input_tokens": 40}
    )
    async with suite_ctx(fake_server, profile=profile) as (ctx, sink):
        plan = _plan_for(ctx, level=n, rounds=1, max_tokens=8, input_tokens=40, warmup_trials=0)
        records = await _collect(SUITE, ctx, plan)

    by_ref = dict(zip(sink.refs, sink.bodies, strict=True))
    for record in records:
        text = _first_message_text(by_ref[record.request_body_ref])
        assert concurrency_module._LONG_OUTPUT_INSTRUCTION in text
