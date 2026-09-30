"""測る項目のまとまりに共通の約束事の試験 (task 3.1: suites/base)。

要求が絡む試験は、実際のソケット越しに偽のサーバーを相手に行う。時刻を判定
する試験は 1 つもない (このファイルが確かめるのは、条件の計画、印の付け方、
送った本文、レコードの中身、種と識別子の決まり方であって、速さではない)。
`start_gate` の試験だけは順序を見るが、時刻の値ではなく「合図の前に要求が
届いていないこと」だけを確かめる。
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic import JsonValue, SecretStr

from bench_harness.client.messages import HttpxMessagesClient, build_request_body
from bench_harness.client.probe import ProbeError, TokenCount, preflight
from bench_harness.corpus.synth import PREFIX_NONCE_HEX_LEN, TemplateCorpus, prefix_header_line
from bench_harness.suites.base import (
    DEFAULT_CONNECTION_WARMUP_TRIALS,
    ConditionAborted,
    FrameTokensPerRun,
    SuiteContext,
    abort_if_context_limit,
    cold_prefix_nonce,
    condition_key,
    iter_trials,
    make_suite_context,
    measure_frame_tokens,
    plan_condition,
    run_trial,
    single_user_message,
    system_with_prefix,
    tokens_label,
    trial_seed,
    warm_prefix_nonce,
)
from bench_harness.types import (
    SCHEMA_VERSION,
    ConditionPlan,
    InputMessage,
    MessagesRequest,
    PreflightOk,
    Profile,
    QualityOutcome,
    QualityVerdict,
    SkippedCondition,
    StreamResult,
    SuiteName,
    TargetDef,
    TextBlockParam,
    ToolCallOutcome,
    ToolCallVerdict,
    ToolDef,
    TrialFlag,
    TrialRecord,
)
from fake_server import (
    FakeServer,
    Script,
    UsageSpec,
    error_stream_response,
    replace,
    text_events,
    text_response,
    thinking_events,
    token_stream_response,
    tool_use_events,
)

REPLACEMENT_CHAR = "�"

RUN_ID = "20260920-000000-abcdef"


# --- 助け -----------------------------------------------------------------


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
    """試験用の設定。制限時間は短く、サンプリングは既定と違う値にしてある。"""
    data: dict[str, Any] = {
        "name": "test",
        "seed": 7,
        "sampling": {"temperature": 0.3, "top_p": 0.9, "top_k": 40},
        "timeout": {"connect_s": 5.0, "first_event_s": 5.0, "idle_s": 5.0, "total_s": 20.0},
        "length_tolerance": 0.04,
        "chars_per_token": {"prose_en": 8.0, "prose_ja": 2.0, "code": 3.0, "log": 3.0},
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
    context_limit: int | None = None,
    run_id: str = RUN_ID,
) -> AsyncIterator[tuple[SuiteContext, BodySink]]:
    """偽のサーバーにつないだ `SuiteContext` と、本文の受け皿を渡す。"""
    sink = BodySink()
    async with HttpxMessagesClient(server.base_url) as client:
        ctx = make_suite_context(
            client=client,
            profile=profile if profile is not None else make_profile(),
            target=target_for(server),
            run_id=run_id,
            put_body=sink.put,
            context_limit=context_limit,
        )
        yield ctx, sink


def make_plan(ctx: SuiteContext, **overrides: Any) -> ConditionPlan:
    """計画される (飛ばされない) ことが分かっている条件を作る。"""
    kwargs: dict[str, Any] = {
        "suite": SuiteName.DECODE,
        "key": "decode/code/en",
        "trials": 2,
        "warmup_trials": 0,
        "max_tokens": 64,
    }
    kwargs.update(overrides)
    plan = plan_condition(ctx, **kwargs)
    assert isinstance(plan, ConditionPlan)
    return plan


def user_message(text: str = "こんにちは") -> list[InputMessage]:
    return [InputMessage(role="user", content=[TextBlockParam(text=text)])]


# --- 条件の計画と、送る前に飛ばす上限 (3.6) --------------------------------


async def test_plan_skips_the_128k_condition_against_a_32000_limit_server(
    fake_server: FakeServer,
) -> None:
    """tasks.md 3.1 の完了の状態: 上限 3 万 2 千の偽のサーバーで、12 万 8 千が飛ぶ。

    上限は `preflight` が偽のサーバーの `/v1/models` から読む。出力の上限 16 を
    足すので、3 万 2 千ちょうどの条件も収まらない (3 万 2 千 16 > 3 万 2 千)。
    """
    fake_server.set_context_limit(32000)
    async with HttpxMessagesClient(fake_server.base_url) as client:
        target = target_for(fake_server)
        ok = await preflight(client, target)
        assert isinstance(ok, PreflightOk)
        assert ok.context_limit == 32000
        ctx = make_suite_context(
            client=client,
            profile=make_profile(),
            target=target,
            run_id=RUN_ID,
            put_body=BodySink().put,
            context_limit=ok.context_limit,
        )
        planned = [
            plan_condition(
                ctx,
                suite=SuiteName.PREFILL,
                key=f"prefill/cold/{tokens_label(tokens)}",
                trials=3,
                max_tokens=16,
                target_input_tokens=tokens,
            )
            for tokens in (8000, 32000, 128000)
        ]

    assert isinstance(planned[0], ConditionPlan)
    assert planned[0].target_input_tokens == 8000
    assert isinstance(planned[1], SkippedCondition)
    assert isinstance(planned[2], SkippedCondition)
    assert planned[2].suite is SuiteName.PREFILL
    assert planned[2].key == "prefill/cold/128k"
    assert "128000" in planned[2].reason
    assert "32000" in planned[2].reason


async def test_plan_keeps_every_condition_that_fits(fake_server: FakeServer) -> None:
    async with suite_ctx(fake_server, context_limit=132000) as (ctx, _):
        planned = [
            plan_condition(
                ctx,
                suite=SuiteName.PREFILL,
                key=f"prefill/cold/{tokens_label(tokens)}",
                trials=3,
                max_tokens=16,
                target_input_tokens=tokens,
            )
            for tokens in (8000, 32000, 128000)
        ]

    assert [isinstance(plan, ConditionPlan) for plan in planned] == [True, True, True]
    keys = [plan.key for plan in planned if isinstance(plan, ConditionPlan)]
    assert keys == ["prefill/cold/8k", "prefill/cold/32k", "prefill/cold/128k"]


async def test_plan_uses_the_profile_sampling_and_carries_the_settings(
    fake_server: FakeServer,
) -> None:
    profile = make_profile()
    async with suite_ctx(fake_server, profile=profile) as (ctx, _):
        plan = make_plan(
            ctx,
            suite=SuiteName.CONCURRENCY,
            key="concurrency/c4",
            trials=5,
            warmup_trials=2,
            max_tokens=256,
            tier="reference",
            concurrency=4,
            target_input_tokens=2000,
        )

    assert plan.sampling == profile.sampling
    assert (plan.suite, plan.key, plan.tier) == (
        SuiteName.CONCURRENCY,
        "concurrency/c4",
        "reference",
    )
    assert (plan.trials, plan.warmup_trials, plan.max_tokens) == (5, 2, 256)
    assert (plan.concurrency, plan.target_input_tokens) == (4, 2000)


async def test_plan_warmup_default_is_the_connection_warmup(fake_server: FakeServer) -> None:
    """慣らしの回数を渡さない条件でも、接続を温める 1 回が必ず入る (2.1 の注)。"""
    async with suite_ctx(fake_server) as (ctx, _):
        plan = plan_condition(ctx, suite=SuiteName.DECODE, key="decode/x", trials=1, max_tokens=8)
    assert isinstance(plan, ConditionPlan)
    assert plan.warmup_trials == DEFAULT_CONNECTION_WARMUP_TRIALS >= 1


async def test_plan_boundary_is_inclusive(fake_server: FakeServer) -> None:
    """入力 + 出力の上限が、上限とちょうど等しいなら収まる (2.7 の `fits_context`)。"""
    async with suite_ctx(fake_server, context_limit=32000) as (ctx, _):
        exact = plan_condition(
            ctx,
            suite=SuiteName.PREFILL,
            key="prefill/cold/exact",
            trials=1,
            max_tokens=16,
            target_input_tokens=31984,
        )
        over = plan_condition(
            ctx,
            suite=SuiteName.PREFILL,
            key="prefill/cold/over",
            trials=1,
            max_tokens=16,
            target_input_tokens=31985,
        )

    assert isinstance(exact, ConditionPlan)
    assert isinstance(over, SkippedCondition)


async def test_plan_without_a_known_limit_plans_normally(fake_server: FakeServer) -> None:
    async with suite_ctx(fake_server, context_limit=None) as (ctx, _):
        plan = plan_condition(
            ctx,
            suite=SuiteName.AGENT,
            key="agent/stage/120k",
            trials=2,
            max_tokens=256,
            target_input_tokens=120000,
        )
    assert isinstance(plan, ConditionPlan)


async def test_plan_rejects_values_outside_the_range(fake_server: FakeServer) -> None:
    async with suite_ctx(fake_server) as (ctx, _):
        with pytest.raises(ValueError, match="trials"):
            plan_condition(ctx, suite=SuiteName.DECODE, key="k", trials=-1, max_tokens=8)
        with pytest.raises(ValueError, match="max_tokens"):
            plan_condition(ctx, suite=SuiteName.DECODE, key="k", trials=1, max_tokens=0)
        with pytest.raises(ValueError, match="target_input_tokens"):
            plan_condition(
                ctx,
                suite=SuiteName.DECODE,
                key="k",
                trials=1,
                max_tokens=8,
                target_input_tokens=0,
            )
        with pytest.raises(ValueError, match="key"):
            plan_condition(ctx, suite=SuiteName.DECODE, key="", trials=1, max_tokens=8)


# --- 実行中に上限に当たったとき (3.6、6.9) --------------------------------


async def test_unknown_limit_records_the_400_and_aborts_the_condition(
    fake_server: FakeServer,
) -> None:
    """上限を申告しないが守るサーバー: レコードは残り、条件は打ち切られる。"""
    fake_server.set_chars_per_token(1.0)
    fake_server.set_context_limit(32000, advertise=False)
    long_text = "あ" * 33000

    async with HttpxMessagesClient(fake_server.base_url) as client:
        target = target_for(fake_server)
        ok = await preflight(client, target)
        assert isinstance(ok, PreflightOk)
        assert ok.context_limit is None  # 申告がなく、定義にも上限がない
        sink = BodySink()
        ctx = make_suite_context(
            client=client,
            profile=make_profile(),
            target=target,
            run_id=RUN_ID,
            put_body=sink.put,
            context_limit=ok.context_limit,
        )
        plan = plan_condition(
            ctx,
            suite=SuiteName.AGENT,
            key="agent/stage/120k",
            trials=2,
            warmup_trials=0,
            max_tokens=16,
            target_input_tokens=120000,
        )
        assert isinstance(plan, ConditionPlan)
        record = await run_trial(ctx, plan, trial_index=0, messages=user_message(long_text))

    # 8.1: 要求は 1 つ残らず記録する
    assert record.result.error is not None
    assert record.result.error.kind == "http"
    assert record.result.error.http_status == 400
    assert record.request_body_ref == sink.refs[0]

    with pytest.raises(ConditionAborted) as raised:
        abort_if_context_limit(plan, record)
    skipped = raised.value.skipped
    assert skipped.suite is SuiteName.AGENT
    assert skipped.key == "agent/stage/120k"
    assert "400" in skipped.reason
    assert "maximum context length" in skipped.reason  # サーバーの言い分をそのまま残す
    assert "120000" in skipped.reason  # 到達した長さ (狙いの値で代える)


async def test_abort_helper_does_nothing_for_other_results(fake_server: FakeServer) -> None:
    fake_server.set_response(text_response("ok"))
    async with suite_ctx(fake_server) as (ctx, _):
        plan = make_plan(ctx)
        ok_record = await run_trial(ctx, plan, trial_index=0, messages=user_message())
    abort_if_context_limit(plan, ok_record)  # 何も起きない


async def test_abort_helper_prefers_the_measured_reached_length(fake_server: FakeServer) -> None:
    fake_server.set_chars_per_token(1.0)
    fake_server.set_context_limit(2000, advertise=False)
    async with suite_ctx(fake_server) as (ctx, _):
        plan = make_plan(ctx, key="agent/stage/060k", suite=SuiteName.AGENT, max_tokens=16)
        record = await run_trial(ctx, plan, trial_index=0, messages=user_message("x" * 3000))
        with pytest.raises(ConditionAborted) as raised:
            abort_if_context_limit(plan, record, reached_input_tokens=61234)
    assert "61234" in raised.value.skipped.reason


# --- 共通の印: 狙った長さからの外れ (3.3) ---------------------------------


@pytest.mark.parametrize(
    ("actual_input_tokens", "flagged"),
    [(1950, False), (2080, False), (2081, True), (1919, True)],
)
async def test_length_off_target_flag_uses_a_strict_comparison(
    fake_server: FakeServer, actual_input_tokens: int, flagged: bool
) -> None:
    """狙い 2000、許容 4% (= ±80)。ちょうど境界は印を付けない (> で比べる)。"""
    fake_server.set_response(
        Script(
            events=text_events("ok"),
            usage=UsageSpec(input_tokens=actual_input_tokens, output_tokens=20),
        )
    )
    async with suite_ctx(fake_server) as (ctx, _):
        plan = make_plan(ctx, target_input_tokens=2000)
        record = await run_trial(ctx, plan, trial_index=0, messages=user_message())

    assert record.result.usage is not None
    assert record.result.usage.total_input_tokens == actual_input_tokens
    assert (TrialFlag.LENGTH_OFF_TARGET in record.flags) is flagged
    assert record.target_input_tokens == 2000


@pytest.mark.parametrize(
    ("total_input_tokens", "cache_read", "cache_creation", "flagged"),
    [
        # 全長 2000 (狙いちょうど)。応答の input_tokens は内訳を引いた 1200 で、それだけを
        # 見ると外れに見える
        (2000, 700, 100, False),
        # 全長 2100 (許容の 2080 を超える)。input_tokens は 1300
        (2100, 700, 100, True),
        # 全長 2090 (外れ)。input_tokens は 1950 で、それだけを見ると収まって見える
        (2090, 100, 40, True),
    ],
)
async def test_length_off_target_flag_uses_the_cache_inclusive_input_length(
    fake_server: FakeServer,
    total_input_tokens: int,
    cache_read: int,
    cache_creation: int,
    flagged: bool,
) -> None:
    """キャッシュが効いた試行 (3.5) では、`input_tokens` は内訳を引いた値になる。

    印は、内訳を足し戻した入力の全長 (3.2) で判定する。狙いは 2000、許容は 4%。
    偽のサーバーの `UsageSpec.input_tokens` は入力の全長で、応答には、本物のサーバーと
    同じく、内訳を引いた値が載る。
    """
    fake_server.set_response(
        Script(
            events=text_events("ok"),
            usage=UsageSpec(
                input_tokens=total_input_tokens,
                output_tokens=20,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_creation,
            ),
        )
    )
    async with suite_ctx(fake_server) as (ctx, _):
        plan = make_plan(ctx, target_input_tokens=2000)
        record = await run_trial(ctx, plan, trial_index=0, messages=user_message())

    assert record.result.usage is not None
    assert record.result.usage.input_tokens == total_input_tokens - cache_read - cache_creation
    assert record.result.usage.total_input_tokens == total_input_tokens
    assert (TrialFlag.LENGTH_OFF_TARGET in record.flags) is flagged


def test_run_trial_has_no_per_trial_sampling_or_output_limit_argument() -> None:
    """同じ条件の試行が同じ設定で送られること (2.7) を、引数の形で守る。"""
    forbidden = {"sampling", "temperature", "top_p", "top_k", "max_tokens", "model"}

    assert forbidden.isdisjoint(inspect.signature(run_trial).parameters)


def test_a_negative_stream_index_is_rejected() -> None:
    """負の番号は、「番号なし」の部品と区別できないので、受け付けない。"""
    with pytest.raises(ValueError, match="stream_index"):
        trial_seed(7, "concurrency/c4", 0, warmup=False, stream_index=-1)


async def test_length_flag_needs_both_a_target_and_usage(fake_server: FakeServer) -> None:
    fake_server.set_response(Script(events=text_events("ok"), usage=None))
    async with suite_ctx(fake_server) as (ctx, _):
        with_target = make_plan(ctx, target_input_tokens=2000)
        no_usage = await run_trial(ctx, with_target, trial_index=0, messages=user_message())

        fake_server.set_response(
            Script(events=text_events("ok"), usage=UsageSpec(input_tokens=99999, output_tokens=20))
        )
        without_target = make_plan(ctx)
        no_target = await run_trial(ctx, without_target, trial_index=0, messages=user_message())

    assert no_usage.result.usage is None
    assert no_usage.flags == []
    assert no_target.target_input_tokens is None
    assert no_target.flags == []


# --- 共通の印: 出力のトークンが少なすぎる、早く終わった -------------------


@pytest.mark.parametrize(
    ("output_tokens", "measures_decode_speed", "flagged"),
    [(8, True, True), (8, False, False), (16, True, False)],
)
async def test_too_few_output_tokens_only_for_speed_trials(
    fake_server: FakeServer, output_tokens: int, measures_decode_speed: bool, flagged: bool
) -> None:
    fake_server.set_response(token_stream_response(output_tokens=output_tokens))
    async with suite_ctx(fake_server) as (ctx, _):
        plan = make_plan(ctx, max_tokens=1024)
        record = await run_trial(
            ctx,
            plan,
            trial_index=0,
            messages=user_message(),
            measures_decode_speed=measures_decode_speed,
        )

    assert (TrialFlag.TOO_FEW_OUTPUT_TOKENS in record.flags) is flagged


@pytest.mark.parametrize(
    ("stop_reason", "expect_full_output", "flagged"),
    [("end_turn", True, True), ("max_tokens", True, False), ("end_turn", False, False)],
)
async def test_short_output_only_when_the_caller_expects_a_full_output(
    fake_server: FakeServer, stop_reason: str, expect_full_output: bool, flagged: bool
) -> None:
    fake_server.set_response(
        replace(token_stream_response(output_tokens=20), stop_reason=stop_reason)
    )
    async with suite_ctx(fake_server) as (ctx, _):
        plan = make_plan(ctx, max_tokens=1024)
        record = await run_trial(
            ctx,
            plan,
            trial_index=0,
            messages=user_message(),
            expect_full_output=expect_full_output,
        )

    assert record.result.stop_reason == stop_reason
    assert (TrialFlag.SHORT_OUTPUT in record.flags) is flagged


# --- 共通の印: 出力が壊れている疑い (10.7) --------------------------------


async def test_replacement_char_in_text(fake_server: FakeServer) -> None:
    fake_server.set_response(text_response(f"ふつうの本文 {REPLACEMENT_CHAR} の続き"))
    async with suite_ctx(fake_server) as (ctx, _):
        record = await run_trial(ctx, make_plan(ctx), trial_index=0, messages=user_message())
    assert record.flags == [TrialFlag.REPLACEMENT_CHAR]


async def test_replacement_char_in_thinking(fake_server: FakeServer) -> None:
    fake_server.set_response(
        Script(
            events=(*thinking_events(f"考え中 {REPLACEMENT_CHAR}"), *text_events("答え", index=1))
        )
    )
    async with suite_ctx(fake_server) as (ctx, _):
        record = await run_trial(ctx, make_plan(ctx), trial_index=0, messages=user_message())
    assert record.flags == [TrialFlag.REPLACEMENT_CHAR]


async def test_replacement_char_in_tool_input(fake_server: FakeServer) -> None:
    """ツールの引数は、置き換え文字の検査にだけかける (2.6 の注)。"""
    fake_server.set_response(
        Script(
            events=tool_use_events(
                "read_file", fragments=(f'{{"path": "/tmp/{REPLACEMENT_CHAR}.txt"}}',)
            ),
            stop_reason="tool_use",
        )
    )
    async with suite_ctx(fake_server) as (ctx, _):
        record = await run_trial(ctx, make_plan(ctx), trial_index=0, messages=user_message())
    assert record.flags == [TrialFlag.REPLACEMENT_CHAR]


async def test_repetition_loop_in_text(fake_server: FakeServer) -> None:
    fake_server.set_response(text_response("0123456789ab" * 8))
    async with suite_ctx(fake_server) as (ctx, _):
        record = await run_trial(ctx, make_plan(ctx), trial_index=0, messages=user_message())
    assert record.flags == [TrialFlag.REPETITION_LOOP]


async def test_clean_output_gets_no_flags(fake_server: FakeServer) -> None:
    fake_server.set_response(text_response("The nimbus cache reported 3 errors last night."))
    async with suite_ctx(fake_server) as (ctx, _):
        record = await run_trial(ctx, make_plan(ctx), trial_index=0, messages=user_message())
    assert record.flags == []


async def test_failed_requests_get_no_output_flags(fake_server: FakeServer) -> None:
    """失敗した要求には、出力の印を付けない (集計に入らないので意味がない)。

    入力の長さの印だけは別で、対象サーバーがトークン数を返していれば付く
    (出力ではなく、送った入力の長さについての事実であるため)。
    """
    fake_server.set_response(
        error_stream_response(before=text_events(REPLACEMENT_CHAR * 3 + "0123456789ab" * 8))
    )
    async with suite_ctx(fake_server) as (ctx, _):
        no_target = await run_trial(
            ctx,
            make_plan(ctx),
            trial_index=0,
            messages=user_message(),
            measures_decode_speed=True,
            expect_full_output=True,
        )
        with_target = await run_trial(
            ctx,
            make_plan(ctx, target_input_tokens=2000),
            trial_index=0,
            messages=user_message(),
            measures_decode_speed=True,
            expect_full_output=True,
        )

    assert no_target.result.error is not None
    assert no_target.flags == []
    assert with_target.result.usage is not None  # message_start のトークン数は来ている
    assert with_target.flags == [TrialFlag.LENGTH_OFF_TARGET]


# --- 同じ条件のすべての試行が、同じ設定で送られる (2.7) --------------------


async def test_every_trial_of_a_plan_sends_the_same_sampling_and_max_tokens(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(text_response("ok"))
    profile = make_profile()
    async with suite_ctx(fake_server, profile=profile) as (ctx, sink):
        plan = make_plan(ctx, trials=3, warmup_trials=2, max_tokens=64)
        records = [
            await run_trial(
                ctx,
                plan,
                trial_index=index,
                warmup=warmup,
                messages=user_message(f"試行 {index} ({warmup})"),
            )
            for index, warmup in iter_trials(plan)
        ]

    sent = [record.body for record in fake_server.requests_for("/v1/messages")]
    assert len(sent) == 5
    assert len(records) == 5
    for body in sent:
        assert body is not None
        assert body["temperature"] == pytest.approx(0.3)
        assert body["top_p"] == pytest.approx(0.9)
        assert body["top_k"] == 40
        assert body["max_tokens"] == 64
        assert body["stream"] is True
    assert [record.warmup for record in records] == [True, True, False, False, False]
    assert len(sink.bodies) == 5


async def test_stored_body_is_the_body_that_was_sent(fake_server: FakeServer) -> None:
    fake_server.set_response(text_response("ok"))
    async with suite_ctx(fake_server) as (ctx, sink):
        plan = make_plan(ctx, max_tokens=64)
        record = await run_trial(
            ctx,
            plan,
            trial_index=0,
            system="system line",
            messages=user_message("本文"),
            extra={"chat_template_kwargs": {"enable_thinking": False}},
        )

    expected = build_request_body(
        MessagesRequest(
            model="fake-model",
            max_tokens=64,
            messages=user_message("本文"),
            system="system line",
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
        )
    )
    assert sink.bodies == [expected]
    assert record.request_body_ref == sink.refs[0]
    received = fake_server.requests_for("/v1/messages")[0].body
    assert received == expected


async def test_run_trial_adds_the_chat_template_switch_when_thinking_is_off(
    fake_server: FakeServer,
) -> None:
    """`sampling.thinking = "off"` の条件では、`run_trial` が送る本文すべてに
    `chat_template_kwargs == {"enable_thinking": False}` が載る (issue #131、K9)。"""
    fake_server.set_response(text_response("ok"))
    profile = make_profile(
        sampling={"temperature": 0.3, "top_p": 0.9, "top_k": 40, "thinking": "off"}
    )
    async with suite_ctx(fake_server, profile=profile) as (ctx, sink):
        plan = make_plan(ctx, max_tokens=64)
        await run_trial(ctx, plan, trial_index=0, messages=user_message("本文"))

    assert sink.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    sent = fake_server.requests_for("/v1/messages")[0].body
    assert sent is not None
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


async def test_trial_record_is_complete_and_round_trips_through_json(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(token_stream_response(output_tokens=20))
    async with suite_ctx(fake_server) as (ctx, _):
        plan = make_plan(
            ctx,
            suite=SuiteName.CONCURRENCY,
            key="concurrency/c2",
            tier="reference",
            max_tokens=256,
            target_input_tokens=2000,
        )
        record = await run_trial(
            ctx,
            plan,
            trial_index=3,
            warmup=True,
            messages=user_message(),
            round_id=2,
            stream_index=1,
        )

    assert record.schema_version == SCHEMA_VERSION
    assert record.run_id == RUN_ID
    assert record.suite is SuiteName.CONCURRENCY
    assert record.condition == "concurrency/c2"
    assert record.tier == "reference"
    assert (record.trial_index, record.warmup) == (3, True)
    assert (record.round_id, record.stream_index) == (2, 1)
    assert record.target_input_tokens == 2000
    assert record.request_body_ref
    assert record.result.usage is not None
    assert TrialRecord.model_validate_json(record.model_dump_json()) == record


async def test_verdict_callback_is_attached(fake_server: FakeServer) -> None:
    fake_server.set_response(text_response("ok"))
    seen: list[StreamResult] = []

    def verdict(result: StreamResult) -> QualityVerdict:
        seen.append(result)
        return QualityVerdict(task="toolcall", outcome=QualityOutcome.CORRECT, detail="ok")

    async with suite_ctx(fake_server) as (ctx, _):
        record = await run_trial(
            ctx, make_plan(ctx), trial_index=0, messages=user_message(), verdict=verdict
        )

    assert len(seen) == 1
    assert isinstance(record.verdict, QualityVerdict)
    assert record.verdict.outcome is QualityOutcome.CORRECT


async def test_verdict_is_called_for_failed_requests_too(fake_server: FakeServer) -> None:
    fake_server.set_response(error_stream_response())

    def verdict(result: StreamResult) -> ToolCallVerdict:
        outcome = (
            ToolCallOutcome.REQUEST_FAILED if result.error is not None else ToolCallOutcome.CORRECT
        )
        return ToolCallVerdict(outcome=outcome)

    async with suite_ctx(fake_server) as (ctx, _):
        record = await run_trial(
            ctx, make_plan(ctx), trial_index=0, messages=user_message(), verdict=verdict
        )

    assert isinstance(record.verdict, ToolCallVerdict)
    assert record.verdict.outcome is ToolCallOutcome.REQUEST_FAILED


async def test_start_gate_runs_after_the_body_is_stored_and_before_the_request(
    fake_server: FakeServer,
) -> None:
    """3.4 が「合図で同時に送り始める」ために使う継ぎ目。"""
    fake_server.set_response(text_response("ok"))
    gate = asyncio.Event()

    async def wait_for_gate() -> None:
        await gate.wait()

    async with suite_ctx(fake_server) as (ctx, sink):
        task = asyncio.create_task(
            run_trial(
                ctx,
                make_plan(ctx),
                trial_index=0,
                messages=user_message(),
                start_gate=wait_for_gate,
            )
        )
        await asyncio.sleep(0.05)
        assert fake_server.call_count("/v1/messages") == 0  # まだ送っていない
        assert len(sink.bodies) == 1  # 本文の保存は、合図の前に済ませてある
        gate.set()
        record = await task

    assert fake_server.call_count("/v1/messages") == 1
    assert record.result.error is None


# --- 慣らしの決まりと、試行の番号 (2.5) -----------------------------------


async def test_iter_trials_puts_warmups_first_and_keys_are_unique(
    fake_server: FakeServer,
) -> None:
    async with suite_ctx(fake_server) as (ctx, _):
        plan = make_plan(ctx, trials=3, warmup_trials=2)
    pairs = list(iter_trials(plan))
    assert pairs == [(0, True), (1, True), (0, False), (1, False), (2, False)]
    assert len(set(pairs)) == len(pairs)


async def test_iter_trials_without_warmups(fake_server: FakeServer) -> None:
    async with suite_ctx(fake_server) as (ctx, _):
        plan = make_plan(ctx, trials=2, warmup_trials=0)
    assert list(iter_trials(plan)) == [(0, False), (1, False)]


# --- 種と、先頭の識別子 ----------------------------------------------------


async def test_cold_nonce_differs_per_trial_run_and_stream(fake_server: FakeServer) -> None:
    async with suite_ctx(fake_server) as (ctx, _):
        plan = make_plan(ctx, trials=2, warmup_trials=1)
        first = cold_prefix_nonce(ctx, plan, trial_index=0)
        second = cold_prefix_nonce(ctx, plan, trial_index=1)
        warmup = cold_prefix_nonce(ctx, plan, trial_index=0, warmup=True)
        streamed = cold_prefix_nonce(ctx, plan, trial_index=0, stream_index=1)
        other_key = cold_prefix_nonce(ctx, make_plan(ctx, key="decode/prose/ja"), trial_index=0)

    async with suite_ctx(fake_server, run_id="20260920-111111-ffffff") as (other_ctx, _):
        other_run = cold_prefix_nonce(other_ctx, make_plan(other_ctx), trial_index=0)

    nonces = [first, second, warmup, streamed, other_key, other_run]
    assert len(set(nonces)) == len(nonces)
    assert all(len(nonce) == PREFIX_NONCE_HEX_LEN for nonce in nonces)
    # 同じ引数なら、何度呼んでも同じ
    async with suite_ctx(fake_server) as (again_ctx, _):
        assert cold_prefix_nonce(again_ctx, make_plan(again_ctx), trial_index=0) == first


async def test_warm_nonce_is_the_same_for_every_trial_and_run(fake_server: FakeServer) -> None:
    async with suite_ctx(fake_server) as (ctx, _):
        plan = make_plan(ctx, key="prefill/warm/8k", suite=SuiteName.PREFILL)
        warm = warm_prefix_nonce(ctx, plan)
    async with suite_ctx(fake_server, run_id="20260920-222222-cccccc") as (other_ctx, _):
        other_plan = make_plan(other_ctx, key="prefill/warm/8k", suite=SuiteName.PREFILL)
        assert warm_prefix_nonce(other_ctx, other_plan) == warm
        assert warm_prefix_nonce(other_ctx, make_plan(other_ctx, key="prefill/warm/32k")) != warm


def test_trial_seed_depends_on_every_part() -> None:
    base = trial_seed(7, "decode/code/en", 1)
    assert base >= 0
    assert trial_seed(7, "decode/code/en", 1) == base
    assert trial_seed(8, "decode/code/en", 1) != base
    assert trial_seed(7, "decode/prose/en", 1) != base
    assert trial_seed(7, "decode/code/en", 2) != base
    assert trial_seed(7, "decode/code/en", 1, warmup=True) != base
    assert trial_seed(7, "decode/code/en", 1, stream_index=0) != base


_DETERMINISM_SNIPPET = """
import json

from bench_harness.suites.base import (
    cold_prefix_nonce,
    make_suite_context,
    plan_condition,
    trial_seed,
    warm_prefix_nonce,
)
from bench_harness.types import Profile, SuiteName, TargetDef


class NoClient:
    async def stream(self, request, timeout):
        raise NotImplementedError


ctx = make_suite_context(
    client=NoClient(),
    profile=Profile(name="p", seed=7),
    target=TargetDef(name="t", base_url="http://127.0.0.1:1/", model="m"),
    run_id="run-fixed",
    put_body=lambda body: "ref",
)
plan = plan_condition(ctx, suite=SuiteName.PREFILL, key="prefill/cold/8k", trials=2, max_tokens=16)
print(
    json.dumps(
        {
            "seed": trial_seed(7, "prefill/cold/8k", 1),
            "seed_warmup": trial_seed(7, "prefill/cold/8k", 1, warmup=True),
            "cold": cold_prefix_nonce(ctx, plan, trial_index=1),
            "cold_stream": cold_prefix_nonce(ctx, plan, trial_index=1, stream_index=2),
            "warm": warm_prefix_nonce(ctx, plan),
        }
    )
)
"""


def _run_snippet(hash_seed: str) -> dict[str, Any]:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    proc = subprocess.run(
        [sys.executable, "-c", _DETERMINISM_SNIPPET],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    result: dict[str, Any] = json.loads(proc.stdout)
    return result


def test_seeds_and_nonces_are_stable_across_processes() -> None:
    """`hash()` を使っていないこと (`PYTHONHASHSEED` で結果が変わらないこと)。"""
    first = _run_snippet("0")
    second = _run_snippet("1")
    assert first == second
    assert len({first["cold"], first["cold_stream"], first["warm"]}) == 3
    assert first["seed"] != first["seed_warmup"]


# --- 小さな助け ------------------------------------------------------------


def test_condition_key_and_tokens_label() -> None:
    assert condition_key(SuiteName.PREFILL, "cold", tokens_label(8000)) == "prefill/cold/8k"
    assert condition_key(SuiteName.DECODE, "code", "en") == "decode/code/en"
    assert tokens_label(128000) == "128k"
    assert tokens_label(1500) == "1500"
    with pytest.raises(ValueError, match="parts"):
        condition_key(SuiteName.DECODE, "")


def test_system_with_prefix_starts_with_the_prefix_line() -> None:
    nonce = "0" * PREFIX_NONCE_HEX_LEN
    system = system_with_prefix(nonce, "あなたは計測用の助手である。", "長く書くこと。")
    assert system.startswith(prefix_header_line(nonce))
    assert system.splitlines()[1] == "あなたは計測用の助手である。"


def test_single_user_message_builds_one_text_block() -> None:
    messages = single_user_message("本文")
    assert len(messages) == 1
    assert messages[0].role == "user"
    block = messages[0].content[0]
    assert isinstance(block, TextBlockParam)
    assert block.text == "本文"


async def test_make_suite_context_builds_the_corpus_from_the_profile(
    fake_server: FakeServer,
) -> None:
    """コーパスは `profile.chars_per_token` (英語の散文は 8.0) から作る。"""
    profile = make_profile()
    async with suite_ctx(fake_server, profile=profile) as (ctx, _):
        assert isinstance(ctx.corpus, TemplateCorpus)
        text = ctx.corpus.prose("en", 500, 1)
        assert 500 * 8.0 * 0.98 <= len(text) <= 500 * 8.0 * 1.02
        assert ctx.profile is profile
        assert ctx.run_id == RUN_ID


# --- 要求の包みを測る (issue #9: measure_frame_tokens) ----------------------


async def test_measure_frame_tokens_uses_count_tokens_and_returns_the_servers_count(
    fake_server: FakeServer,
) -> None:
    async with HttpxMessagesClient(fake_server.base_url) as client:
        ctx = make_suite_context(
            client=client,
            profile=make_profile(),
            target=target_for(fake_server),
            run_id=RUN_ID,
            put_body=BodySink().put,
        )
        tokens = await measure_frame_tokens(
            ctx, system="sys", messages=user_message("hello"), tools=[ToolDef(name="t")]
        )

    assert fake_server.call_count("/v1/messages/count_tokens") == 1
    assert fake_server.call_count("/v1/messages") == 0
    recorded = fake_server.requests_for("/v1/messages/count_tokens")[0]
    assert recorded.body is not None
    assert recorded.body["system"] == "sys"
    assert recorded.body["tools"][0]["name"] == "t"
    assert "max_tokens" not in recorded.body
    assert tokens == recorded.input_tokens


async def test_measure_frame_tokens_falls_back_to_a_one_token_request_when_the_endpoint_is_missing(
    fake_server: FakeServer,
) -> None:
    fake_server.set_count_tokens_enabled(False)
    fake_server.set_response(text_response("ok"))

    async with suite_ctx(fake_server) as (ctx, sink):
        await measure_frame_tokens(ctx, system="sys", messages=user_message("hello"))

    requests = fake_server.requests_for("/v1/messages")
    assert len(requests) == 1
    assert requests[0].body is not None
    assert requests[0].body["max_tokens"] == 1
    assert sink.bodies == []


async def test_measure_frame_tokens_does_not_swallow_probe_error(fake_server: FakeServer) -> None:
    async def failing(request: MessagesRequest) -> TokenCount:
        raise ProbeError("boom")

    async with HttpxMessagesClient(fake_server.base_url) as client:
        ctx = make_suite_context(
            client=client,
            profile=make_profile(),
            target=target_for(fake_server),
            run_id=RUN_ID,
            put_body=BodySink().put,
            count_input_tokens=failing,
        )
        with pytest.raises(ProbeError):
            await measure_frame_tokens(ctx, system="sys", messages=user_message("hello"))

    assert fake_server.call_count("/v1/messages") == 0


async def test_make_suite_context_passes_the_api_key_to_count_tokens(
    fake_server: FakeServer,
) -> None:
    async with HttpxMessagesClient(fake_server.base_url) as client:
        ctx = make_suite_context(
            client=client,
            profile=make_profile(),
            target=target_for(fake_server),
            run_id=RUN_ID,
            put_body=BodySink().put,
            api_key=SecretStr("k-frame"),
        )
        await measure_frame_tokens(ctx, system="sys", messages=user_message("hello"))

    recorded = fake_server.requests_for("/v1/messages/count_tokens")[-1]
    assert recorded.headers["x-api-key"] == "k-frame"


# --- 包みを、計測ランにつき 1 回だけ数える規則 (issue #9: FrameTokensPerRun) --------


class CountingMeasure:
    """`FrameTokensPerRun` に渡す測る中身の代わり。呼ばれた `run_id` を数える。"""

    def __init__(self, *, tokens: int = 100, failures: int = 0) -> None:
        self.tokens = tokens
        self.failures = failures
        self.run_ids: list[str] = []

    async def __call__(self, ctx: SuiteContext) -> int:
        self.run_ids.append(ctx.run_id)
        if self.failures > 0:
            self.failures -= 1
            raise ProbeError("boom")
        return self.tokens


async def test_frame_tokens_per_run_measures_once_for_the_same_run(
    fake_server: FakeServer,
) -> None:
    measure = CountingMeasure(tokens=123)
    frame = FrameTokensPerRun(measure)

    async with suite_ctx(fake_server) as (ctx, _):
        first = await frame.tokens_for(ctx)
        second = await frame.tokens_for(ctx)

    assert (first, second) == (123, 123)
    assert measure.run_ids == [RUN_ID]


async def test_frame_tokens_per_run_measures_again_for_a_new_run(
    fake_server: FakeServer,
) -> None:
    measure = CountingMeasure()
    frame = FrameTokensPerRun(measure)

    async with suite_ctx(fake_server, run_id="20260920-000000-aaaaaa") as (ctx_a, _):
        await frame.tokens_for(ctx_a)
    async with suite_ctx(fake_server, run_id="20260920-111111-bbbbbb") as (ctx_b, _):
        await frame.tokens_for(ctx_b)

    assert measure.run_ids == ["20260920-000000-aaaaaa", "20260920-111111-bbbbbb"]


async def test_frame_tokens_per_run_forget_makes_the_next_call_measure_again(
    fake_server: FakeServer,
) -> None:
    measure = CountingMeasure()
    frame = FrameTokensPerRun(measure)

    async with suite_ctx(fake_server) as (ctx, _):
        await frame.tokens_for(ctx)
        frame.forget()
        await frame.tokens_for(ctx)

    assert measure.run_ids == [RUN_ID, RUN_ID]


async def test_frame_tokens_per_run_does_not_remember_a_failed_measurement(
    fake_server: FakeServer,
) -> None:
    measure = CountingMeasure(tokens=77, failures=1)
    frame = FrameTokensPerRun(measure)

    async with suite_ctx(fake_server) as (ctx, _):
        with pytest.raises(ProbeError):
            await frame.tokens_for(ctx)
        tokens = await frame.tokens_for(ctx)

    assert tokens == 77
    assert measure.run_ids == [RUN_ID, RUN_ID]
