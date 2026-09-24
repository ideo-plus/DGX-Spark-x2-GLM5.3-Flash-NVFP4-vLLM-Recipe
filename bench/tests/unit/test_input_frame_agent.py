"""長い会話の検査の包みの計測と差し引き (issue #9)。

別のテストファイルからは import しない (助けは `test_suites_agent.py` から要る
分だけ写す)。agent の合成入力が、狙いのトークン数 (`target_input_tokens`) より、
要求の決まった分量 (包み: `SYSTEM_PROMPT` + `TOOL_CATALOG` + 最後の 1 手) だけ
長くなる (issue #9 の観測) 原因への対処を確かめる。
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic import JsonValue

from bench_harness.client.messages import HttpxMessagesClient, build_request_body
from bench_harness.client.probe import ProbeError, TokenCount
from bench_harness.corpus.conversation import build_conversation, derive_conversation_seed
from bench_harness.corpus.tools import make_tool_task
from bench_harness.suites.agent import AgentSuite, _measure_frame, stage_condition_key
from bench_harness.suites.base import ConditionAborted, SuiteContext, make_suite_context
from bench_harness.types import (
    ConditionPlan,
    ConversationPrefix,
    InputMessage,
    MessagesRequest,
    Profile,
    SkippedCondition,
    TargetDef,
    TextBlockParam,
    ToolTask,
    TrialRecord,
)
from fake_server import FakeServer, Script, estimator_aligned_counter, tool_use_response

RUN_A = "20260920-000000-aaaaaa"


# --- 助け (test_suites_agent.py と同じ形。別のテストファイルからは import しない) --


class BodySink:
    """`RunStore.put_body` の代わり。保存された本文をそのまま覚えておく。"""

    def __init__(self) -> None:
        self.bodies: list[dict[str, JsonValue]] = []

    def put(self, body: dict[str, JsonValue]) -> str:
        self.bodies.append(body)
        canonical = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def make_profile(**overrides: Any) -> Profile:
    """試験用の小さな設定。段階は 4000 / 6000 / 8000、会話 1 本、試行 2 回。"""
    data: dict[str, Any] = {
        "name": "test",
        "seed": 7,
        "sampling": {"temperature": 0.0},
        "timeout": {"connect_s": 5.0, "first_event_s": 5.0, "idle_s": 5.0, "total_s": 20.0},
        "length_tolerance": 0.05,
        "min_successes": 1,
        "chars_per_token": {"prose_en": 4.0, "prose_ja": 4.0, "code": 4.0, "log": 4.0},
        "agent": {
            "start_tokens": 4000,
            "end_tokens": 8000,
            "step_tokens": 2000,
            "trials_per_stage": 2,
            "conversations_per_stage": 1,
            "threshold": 0.01,
            "max_tokens": 256,
        },
    }
    data.update(overrides)
    return Profile.model_validate(data)


def agent_settings(**overrides: Any) -> dict[str, Any]:
    """`make_profile(agent=...)` に渡す、段階の設定の写像。"""
    data: dict[str, Any] = {
        "start_tokens": 4000,
        "end_tokens": 8000,
        "step_tokens": 2000,
        "trials_per_stage": 2,
        "conversations_per_stage": 1,
        "threshold": 0.01,
        "max_tokens": 256,
    }
    data.update(overrides)
    return data


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
    run_id: str = RUN_A,
    sink: BodySink | None = None,
    count_input_tokens: Any = None,
) -> AsyncIterator[SuiteContext]:
    """偽のサーバーにつないだ `SuiteContext`。"""
    async with HttpxMessagesClient(server.base_url) as client:
        kwargs: dict[str, Any] = {}
        if count_input_tokens is not None:
            kwargs["count_input_tokens"] = count_input_tokens
        yield make_suite_context(
            client=client,
            profile=profile if profile is not None else make_profile(),
            target=target_for(server),
            run_id=run_id,
            put_body=(sink if sink is not None else BodySink()).put,
            context_limit=context_limit,
            **kwargs,
        )


async def run_suite(
    suite: AgentSuite, ctx: SuiteContext
) -> tuple[list[TrialRecord], list[SkippedCondition]]:
    """計測ランの進行 (3.5) と同じ順で、まとまり全体を流す (`SUITE` の代わりに渡した実体を使う)。"""
    records: list[TrialRecord] = []
    skipped: list[SkippedCondition] = []
    for item in suite.plan(ctx):
        if isinstance(item, SkippedCondition):
            skipped.append(item)
            continue
        try:
            async for record in suite.run_condition(ctx, item):
                records.append(record)
        except ConditionAborted as exc:
            skipped.append(exc.skipped)
    return records, skipped


def planned_condition(suite: AgentSuite, ctx: SuiteContext, key: str) -> ConditionPlan:
    plan = next(item for item in suite.plan(ctx) if item.key == key)
    assert isinstance(plan, ConditionPlan)
    return plan


def sent_requests(server: FakeServer) -> list[Any]:
    return server.requests_for("/v1/messages")


def sent_bodies(server: FakeServer) -> list[dict[str, Any]]:
    bodies: list[dict[str, Any]] = []
    for request in sent_requests(server):
        assert request.body is not None
        bodies.append(dict(request.body))
    return bodies


def last_user_text(body: dict[str, Any]) -> str:
    """最後の 1 手 (課題の指示) の本文。"""
    messages = body["messages"]
    assert isinstance(messages, list)
    last = messages[-1]
    assert last["role"] == "user"
    content = last["content"]
    assert isinstance(content, list) and len(content) == 1
    text = content[0]["text"]
    assert isinstance(text, str)
    return text


def history_of(body: dict[str, Any]) -> list[Any]:
    """送った発話のうち、最後の 1 手を除いた履歴。"""
    messages = body["messages"]
    assert isinstance(messages, list)
    return list(messages[:-1])


def task_for(profile: Profile, stage_ordinal: int, trial_index: int) -> ToolTask:
    """まとまりが使う課題の番号の決まり (agent.py の docstring)。"""
    return make_tool_task(stage_ordinal * 10_000 + trial_index, profile.seed)


def expected_body(
    ctx: SuiteContext, cond: ConditionPlan, prefix: ConversationPrefix, task: ToolTask
) -> dict[str, JsonValue]:
    """まとまりが送るはずの本文を、クライアントの純粋な関数で組み立てる。"""
    request = MessagesRequest(
        model=ctx.target.model,
        max_tokens=cond.max_tokens,
        messages=[
            *prefix.messages,
            InputMessage(role="user", content=[TextBlockParam(text=task.prompt)]),
        ],
        system=prefix.system,
        tools=list(task.tools),
        temperature=cond.sampling.temperature,
        top_p=cond.sampling.top_p,
        top_k=cond.sampling.top_k,
    )
    return build_request_body(request)


def task_table(profile: Profile, stages: Sequence[int], trials: int) -> dict[str, ToolTask]:
    """この計測ランで送られる課題を、指示の文から引ける表にする。"""
    table: dict[str, ToolTask] = {}
    for ordinal in range(len(stages)):
        for trial_index in range(trials):
            task = task_for(profile, ordinal, trial_index)
            table[task.prompt] = task
    return table


def correct_responder(table: dict[str, ToolTask]) -> Callable[[dict[str, Any]], Script]:
    """どの試行にも、その課題の正解のツール呼び出しを返す。"""

    def factory(body: dict[str, Any]) -> Script:
        task = table[last_user_text(body)]
        return tool_use_response(task.expected_tool, task.expected_input)

    return factory


_STAGES: tuple[int, int, int] = (4000, 6000, 8000)


def _small_profile(**overrides: Any) -> Profile:
    return make_profile(
        agent=agent_settings(
            start_tokens=_STAGES[0],
            end_tokens=_STAGES[-1],
            step_tokens=_STAGES[1] - _STAGES[0],
            trials_per_stage=2,
            conversations_per_stage=1,
        ),
        **overrides,
    )


# --- C2: 包みは計測ランにつき 1 回、履歴は fixed_tokens=包み で組み立てる -----


async def test_the_frame_is_measured_once_for_all_stages(fake_server: FakeServer) -> None:
    profile = _small_profile()
    fake_server.set_response_factory(correct_responder(task_table(profile, _STAGES, 2)))
    suite = AgentSuite()

    async with suite_ctx(fake_server, profile=profile) as ctx:
        records, skipped = await run_suite(suite, ctx)

    assert skipped == []
    assert len(records) == 2 * len(_STAGES)
    assert fake_server.call_count("/v1/messages/count_tokens") == 1
    assert fake_server.call_count("/v1/messages") == 2 * len(_STAGES)


async def test_histories_are_built_with_the_measured_frame(fake_server: FakeServer) -> None:
    profile = _small_profile()
    fake_server.set_response_factory(correct_responder(task_table(profile, _STAGES, 2)))
    suite = AgentSuite()

    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = planned_condition(suite, ctx, stage_condition_key(4000))
        frame = await _measure_frame(ctx)  # この呼び出しも count_tokens を 1 回使う
        records = [record async for record in suite.run_condition(ctx, cond)]

    assert records
    prefix = build_conversation(
        4000,
        derive_conversation_seed(profile.seed, 0),
        chars_per_token=profile.chars_per_token,
        fixed_tokens=frame,
    )
    expected_history = json.loads(json.dumps([m.model_dump(mode="json") for m in prefix.messages]))
    for body in sent_bodies(fake_server):
        assert history_of(body) == expected_history


async def test_a_conversation_that_no_longer_fits_reports_its_approx_tokens_and_sends_nothing(
    fake_server: FakeServer,
) -> None:
    """会話の実際の長さ (`approx_tokens`) が段階の狙い (4000) を上回ることを前提にする。

    境界の丸めしだいで、必ず上回るとは限らないので、`profile.seed` を
    7、1、2、3 の順に試し、上回る組を探す (見つからなければ前提そのものが
    崩れているとして報告する)。
    """
    profile: Profile | None = None
    prefix: ConversationPrefix | None = None

    for seed in (7, 1, 2, 3):
        candidate = _small_profile(seed=seed)
        async with suite_ctx(fake_server, profile=candidate) as ctx:
            candidate_frame = await _measure_frame(ctx)
        candidate_prefix = build_conversation(
            4000,
            derive_conversation_seed(candidate.seed, 0),
            chars_per_token=candidate.chars_per_token,
            fixed_tokens=candidate_frame,
        )
        if candidate_prefix.approx_tokens > 4000:
            profile, prefix = candidate, candidate_prefix
            break

    assert profile is not None and prefix is not None, (
        "組み立てた会話が段階の狙い (4000) を上回る seed (7, 1, 2, 3) が見つからなかった"
    )

    fake_server.set_response_factory(correct_responder(task_table(profile, _STAGES, 2)))
    limit = max(4000, prefix.approx_tokens) + 256 - 1
    suite = AgentSuite()

    async with suite_ctx(fake_server, profile=profile, context_limit=limit) as ctx:
        cond = planned_condition(suite, ctx, stage_condition_key(4000))
        with pytest.raises(ConditionAborted) as raised:
            async for _ in suite.run_condition(ctx, cond):
                pass

    assert sent_requests(fake_server) == []
    assert str(prefix.approx_tokens) in raised.value.skipped.reason


# --- C6: 修正後は段階ごとの中央値が許容に入り、修正前の再現は許容を超える -----


async def test_stage_ladder_medians_land_within_the_tolerance_with_the_observed_frame(
    fake_server: FakeServer,
) -> None:
    stages = (20_000, 40_000, 60_000, 80_000, 100_000, 120_000)
    profile = make_profile(
        agent=agent_settings(
            start_tokens=20_000,
            end_tokens=120_000,
            step_tokens=20_000,
            trials_per_stage=5,
            conversations_per_stage=5,
        )
    )
    counter = estimator_aligned_counter(fixed=670, per_message=2)
    fake_server.set_input_token_counter(counter)
    fake_server.set_response_factory(correct_responder(task_table(profile, stages, 5)))
    suite = AgentSuite()

    async with suite_ctx(fake_server, profile=profile) as ctx:
        records, skipped = await run_suite(suite, ctx)

    assert skipped == []
    for target in stages:
        stage_records = [record for record in records if record.target_input_tokens == target]
        assert stage_records
        usages = [record.result.usage for record in stage_records]
        assert all(usage is not None for usage in usages)
        median = statistics.median(usage.total_input_tokens for usage in usages if usage)
        assert abs(median - target) / target <= 0.05

    # 修正前 (fixed_tokens なし) の見積もりが取りこぼす包みは、同じ数え方で許容を超える
    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond_20k = planned_condition(suite, ctx, stage_condition_key(20_000))
        gaps = [
            counter(
                expected_body(
                    ctx,
                    cond_20k,
                    build_conversation(
                        20_000,
                        derive_conversation_seed(profile.seed, conversation_index),
                        chars_per_token=profile.chars_per_token,
                    ),
                    task_for(profile, 0, 0),
                )
            )
            - build_conversation(
                20_000,
                derive_conversation_seed(profile.seed, conversation_index),
                chars_per_token=profile.chars_per_token,
            ).approx_tokens
            for conversation_index in range(5)
        ]
    assert statistics.median(gaps) > 0.05 * 20_000


# --- C3: ProbeError は推定に戻さず伝播する -----------------------------------


async def test_a_probe_error_is_not_turned_into_an_estimate(fake_server: FakeServer) -> None:
    profile = _small_profile()
    fake_server.set_response_factory(correct_responder(task_table(profile, _STAGES, 2)))

    async def failing(request: MessagesRequest) -> TokenCount:
        raise ProbeError("boom")

    suite = AgentSuite()
    async with suite_ctx(fake_server, profile=profile, count_input_tokens=failing) as ctx:
        with pytest.raises(ProbeError):
            await run_suite(suite, ctx)

    assert sent_requests(fake_server) == []
