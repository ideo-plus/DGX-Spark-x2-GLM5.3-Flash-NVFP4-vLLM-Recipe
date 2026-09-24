"""同時処理の包みの計測と差し引き (issue #9)。

別のテストファイルからは import しない (助けは `test_suites_concurrency.py` から
要る分だけ写す)。concurrency の合成入力が、狙いのトークン数 (`target_input_tokens`)
より、要求の決まった分量 (包み: 識別子の行、指示、system、チャットテンプレート)
だけ長くなる (issue #9 の観測) 原因への対処を確かめる。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic import JsonValue

from bench_harness.client.messages import HttpxMessagesClient, build_request_body
from bench_harness.client.probe import ProbeError, TokenCount
from bench_harness.corpus.synth import TemplateCorpus
from bench_harness.suites.base import (
    ConditionAborted,
    SuiteContext,
    make_suite_context,
    trial_seed,
)
from bench_harness.suites.concurrency import _LONG_OUTPUT_INSTRUCTION, ConcurrencySuite
from bench_harness.types import (
    ConditionPlan,
    InputMessage,
    MessagesRequest,
    Profile,
    SuiteName,
    TargetDef,
    TextBlockParam,
    TrialFlag,
    TrialRecord,
)
from fake_server import FakeServer, estimator_aligned_counter, token_stream_response

RUN_A = "20260920-000000-aaaaaa"
RUN_B = "20260920-111111-bbbbbb"


# --- 助け (test_suites_concurrency.py と同じ形。別のテストファイルからは import しない) --


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


def make_profile(**overrides: Any) -> Profile:
    data: dict[str, Any] = {
        "name": "test",
        "seed": 7,
        "timeout": {"connect_s": 5.0, "first_event_s": 5.0, "idle_s": 5.0, "total_s": 20.0},
        "length_tolerance": 0.05,
        "chars_per_token": {"prose_en": 4.0, "prose_ja": 4.0, "code": 4.0, "log": 4.0},
        "concurrency": {"levels": [2], "rounds": 2, "max_tokens": 8, "input_tokens": 2000},
    }
    data.update(overrides)
    return Profile.model_validate(data)


def target_for(server: FakeServer) -> TargetDef:
    return TargetDef.model_validate(
        {"name": "fake", "base_url": server.base_url, "model": "fake-model"}
    )


@asynccontextmanager
async def suite_ctx(
    server: FakeServer,
    *,
    profile: Profile | None = None,
    run_id: str = RUN_A,
    count_input_tokens: Any = None,
) -> AsyncIterator[tuple[SuiteContext, BodySink]]:
    """偽のサーバーにつないだ `SuiteContext` と、本文の受け皿を渡す。"""
    sink = BodySink()
    async with HttpxMessagesClient(server.base_url) as client:
        kwargs: dict[str, Any] = {}
        if count_input_tokens is not None:
            kwargs["count_input_tokens"] = count_input_tokens
        ctx = make_suite_context(
            client=client,
            profile=profile if profile is not None else make_profile(),
            target=target_for(server),
            run_id=run_id,
            put_body=sink.put,
            **kwargs,
        )
        yield ctx, sink


def plans(suite: ConcurrencySuite, ctx: SuiteContext) -> list[ConditionPlan]:
    return [item for item in suite.plan(ctx) if isinstance(item, ConditionPlan)]


async def run_all(suite: ConcurrencySuite, ctx: SuiteContext) -> list[TrialRecord]:
    return [record for cond in plans(suite, ctx) async for record in suite.run_condition(ctx, cond)]


def first_text(body: dict[str, Any]) -> str:
    return str(body["messages"][0]["content"][0]["text"])


def system_first_line(body: dict[str, Any]) -> str:
    return str(body["system"]).splitlines()[0]


def frame_tokens_of(server: FakeServer) -> int:
    """偽のサーバーが記録した包みの計測 (count_tokens の 1 件目) のトークン数。"""
    return server.requests_for("/v1/messages/count_tokens")[0].input_tokens


# --- C1: 文章の狙いは target_input_tokens − 包み -----------------------------


async def test_document_targets_the_input_tokens_minus_the_measured_frame(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(token_stream_response(output_tokens=4))
    profile = make_profile()
    suite = ConcurrencySuite()

    async with suite_ctx(fake_server, profile=profile) as (ctx, sink):
        records = await run_all(suite, ctx)

    frame = frame_tokens_of(fake_server)
    by_ref = dict(zip(sink.refs, sink.bodies, strict=True))
    measured = [record for record in records if not record.warmup]
    assert measured
    for record in measured:
        assert record.round_id is not None
        assert record.stream_index is not None
        body = by_ref[record.request_body_ref]
        seed = trial_seed(
            profile.seed,
            "concurrency/c2",
            record.round_id,
            warmup=False,
            stream_index=record.stream_index,
        )
        expected = TemplateCorpus(profile.chars_per_token).prose("en", 2000 - frame, seed)
        assert first_text(body) == f"{expected}\n\n{_LONG_OUTPUT_INSTRUCTION}"


async def test_the_frame_is_measured_once_per_run_and_again_for_a_new_run(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(token_stream_response(output_tokens=4))
    profile = make_profile(
        concurrency={"levels": [1, 2], "rounds": 2, "max_tokens": 8, "input_tokens": 2000}
    )
    suite = ConcurrencySuite()

    async with suite_ctx(fake_server, profile=profile, run_id=RUN_A) as (ctx, _sink):
        await run_all(suite, ctx)
    assert fake_server.call_count("/v1/messages/count_tokens") == 1

    fake_server.reset()
    async with suite_ctx(fake_server, profile=profile, run_id=RUN_B) as (ctx, _sink):
        await run_all(suite, ctx)
    assert fake_server.call_count("/v1/messages/count_tokens") == 1


# --- C6: 修正後は許容に入り、修正前 (差し引かない) の再現は許容を超える -------


async def test_measured_trials_land_within_the_tolerance_with_a_realistic_frame(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(token_stream_response(output_tokens=4))
    counter = estimator_aligned_counter(fixed=150)
    fake_server.set_input_token_counter(counter)
    profile = make_profile()
    suite = ConcurrencySuite()

    async with suite_ctx(fake_server, profile=profile) as (ctx, _sink):
        records = await run_all(suite, ctx)

    measured = [record for record in records if not record.warmup]
    assert measured
    for record in measured:
        usage = record.result.usage
        assert usage is not None
        assert abs(usage.total_input_tokens - 2000) / 2000 <= 0.05
        assert TrialFlag.LENGTH_OFF_TARGET not in record.flags

    # 修正前 (差し引かずに 2000 トークンぶんの文章をそのまま合成する) を、同じ
    # 数え方で再現すると、許容 (5%) を超えることも確かめる (空回りの防止)。
    any_trial = fake_server.requests_for("/v1/messages")[0]
    assert any_trial.body is not None
    seed0 = trial_seed(profile.seed, "concurrency/c2", 0, warmup=False, stream_index=0)
    old_document = TemplateCorpus(profile.chars_per_token).prose("en", 2000, seed0)
    old = build_request_body(
        MessagesRequest(
            model="fake-model",
            max_tokens=8,
            system=system_first_line(any_trial.body) + "\n長く書くこと。",
            messages=[
                InputMessage(
                    role="user",
                    content=[TextBlockParam(text=f"{old_document}\n\n{_LONG_OUTPUT_INSTRUCTION}")],
                )
            ],
        )
    )
    assert counter(old) - 2000 > 0.05 * 2000


# --- SCN-C1-P1: 包みを測る要求の先頭は、どの試行の先頭とも重ならない ----------


async def test_the_frame_request_uses_its_own_nonce_and_an_empty_document(
    fake_server: FakeServer,
) -> None:
    fake_server.set_count_tokens_enabled(False)
    fake_server.set_response(token_stream_response(output_tokens=4))
    profile = make_profile(
        concurrency={"levels": [2], "rounds": 2, "max_tokens": 8, "input_tokens": 2000}
    )
    suite = ConcurrencySuite()

    async with suite_ctx(fake_server, profile=profile) as (ctx, _sink):
        await run_all(suite, ctx)

    requests = fake_server.requests_for("/v1/messages")
    # 測定 1 件 + (慣らし 1 回 + 本番 2 回) × 2 本
    assert len(requests) == 1 + 3 * 2
    assert requests[0].body is not None
    assert requests[0].body["max_tokens"] == 1
    assert first_text(requests[0].body) == "\n\n" + _LONG_OUTPUT_INSTRUCTION
    frame_first_line = system_first_line(requests[0].body)
    other_first_lines = {
        system_first_line(record.body) for record in requests[1:] if record.body is not None
    }
    assert frame_first_line not in other_first_lines


# --- SCN-C1-N1: 狙いが包み以下のときは、送る前に、理由つきでその条件を飛ばす ----


async def test_a_target_at_or_below_the_frame_is_skipped_with_a_reason_before_sending(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(token_stream_response(output_tokens=4))
    fake_server.set_input_token_counter(estimator_aligned_counter(fixed=50))
    profile = make_profile(
        concurrency={"levels": [2], "rounds": 2, "max_tokens": 8, "input_tokens": 10}
    )
    suite = ConcurrencySuite()

    async with suite_ctx(fake_server, profile=profile) as (ctx, _sink):
        with pytest.raises(ConditionAborted) as raised:
            await run_all(suite, ctx)

    skipped = raised.value.skipped
    assert skipped.suite is SuiteName.CONCURRENCY
    assert skipped.key == "concurrency/c2"
    assert "10" in skipped.reason
    assert str(frame_tokens_of(fake_server)) in skipped.reason
    assert fake_server.call_count("/v1/messages") == 0
    assert fake_server.call_count("/v1/messages/count_tokens") == 1


@pytest.mark.parametrize(("input_tokens", "skipped"), [(60, True), (61, False)])
async def test_the_frame_boundary_decides_between_skipping_and_sending(
    fake_server: FakeServer, input_tokens: int, skipped: bool
) -> None:
    """境界: 包みが 60 トークンなら、狙い 60 は飛ばし、狙い 61 (文章 1 トークン) は送る。"""
    fake_server.set_response(token_stream_response(output_tokens=4))
    fake_server.set_input_token_counter(lambda body: 60)
    profile = make_profile(
        concurrency={"levels": [2], "rounds": 1, "max_tokens": 8, "input_tokens": input_tokens}
    )
    suite = ConcurrencySuite()

    async with suite_ctx(fake_server, profile=profile) as (ctx, _sink):
        if skipped:
            with pytest.raises(ConditionAborted):
                await run_all(suite, ctx)
        else:
            assert await run_all(suite, ctx)

    # 送る条件は、(慣らし 1 回 + 本番 1 回) × 2 本
    assert fake_server.call_count("/v1/messages") == (0 if skipped else 2 * 2)
    assert fake_server.call_count("/v1/messages/count_tokens") == 1


# --- C3: ProbeError は推定に戻さず伝播する -----------------------------------


async def test_a_probe_error_is_not_turned_into_an_estimate(fake_server: FakeServer) -> None:
    fake_server.set_response(token_stream_response(output_tokens=4))

    async def failing(request: MessagesRequest) -> TokenCount:
        raise ProbeError("boom")

    suite = ConcurrencySuite()
    async with suite_ctx(fake_server, count_input_tokens=failing) as (ctx, _sink):
        with pytest.raises(ProbeError):
            await run_all(suite, ctx)

    assert fake_server.call_count("/v1/messages") == 0
