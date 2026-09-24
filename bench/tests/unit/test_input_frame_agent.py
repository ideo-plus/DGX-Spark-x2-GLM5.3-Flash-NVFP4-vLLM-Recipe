"""長い会話の入力の長さの合わせ込み (issue #9、#24)。

別のテストファイルからは import しない (助けは `test_suites_agent.py` から要る
分だけ写す)。agent の合成入力が、狙いのトークン数 (`target_input_tokens`) に
合うかを確かめる。issue #9 は、手番以外の決まった分量 (包み) を対象サーバーに
数えさせて差し引く仕組みを足したが、実機では狙いより長い入力が残った (#24。原因は
特定されていない)。#24 は、段階 × 会話ごとに組み立てた会話そのものを数えさせ、その
実測で止める手番を決める有界の反復に置き換える。

確かめるのは 4 つのまとまり。

1. **実機の数え方の特徴** (#24): system と tools の有無の 4 通りで、`count_tokens`
   の答えと `max_tokens=1` の要求の `usage` が一致し、包みが system と tools の
   文字数に比例して加法に増える
2. **合わせ込みの規則** (#24): 段階 × 会話ごとに `system` + `tools` + 履歴 + 最後の
   1 手を `max_tokens=1` の要求として数えさせ、送る履歴は訪れた候補のうち数えた
   長さが狙いに最も近い手番の境界になる
3. **狙いへの到達**: 実機の数え方で、20k〜120k のすべての試行が許容の内側に入り、
   修正前の方法 (`fixed_tokens` だけ) では 20k の中央値が許容を超える
4. **失敗の扱い**: 数えられない `ProbeError` は見積もりに戻さず伝播し、上限判定と 400 の
   理由には見積もりではなく数えた長さを使う。ただし、口がなく、数える代わりの要求そのものが
   上限を超えて断られたとき (`InputOverContextLimitError`) だけは、段階を飛ばす扱いにする
   (1 本目の候補なら送らずに飛ばし、2 本目以降なら数え終えた候補を送る)
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
from bench_harness.client.probe import InputOverContextLimitError, ProbeError, TokenCount
from bench_harness.corpus.conversation import (
    MESSAGES_PER_ROUND,
    SYSTEM_PROMPT,
    build_conversation,
    derive_conversation_seed,
)
from bench_harness.corpus.tools import TOOL_CATALOG, make_tool_task
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
    TrialFlag,
    TrialRecord,
)
from fake_server import (
    FakeServer,
    InputTokenCounter,
    Script,
    estimator_aligned_counter,
    text_response,
    tool_use_response,
)

RUN_A = "20260920-000000-aaaaaa"

_CHARS_PER_TOKEN = 4.0
"""実機の特徴を作る偽の数え方の比 (偽のサーバーの既定と同じ)。"""


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


def count_token_requests(server: FakeServer) -> list[Any]:
    return server.requests_for("/v1/messages/count_tokens")


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


def final_turn(prompt: str) -> InputMessage:
    """まとまりが最後に足す、課題の指示 1 つぶんの `user` の発話。"""
    return InputMessage(role="user", content=[TextBlockParam(text=prompt)])


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
        messages=[*prefix.messages, final_turn(task.prompt)],
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


def _tools_json() -> str:
    """目録を、要求の本文に入る形 (compact な JSON) にする。"""
    return json.dumps(
        [tool.model_dump(mode="json") for tool in TOOL_CATALOG],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


# --- C3: 実機の数え方の特徴 (count_tokens と usage の一致、比例、加法) --------


async def test_the_realistic_counter_matches_count_tokens_and_a_max_tokens_request(
    fake_server: FakeServer,
) -> None:
    """実機の特徴: `count_tokens` と `max_tokens=1` の `usage` が一致し、包みが比例して増える。

    `estimator_aligned_counter` は system と tools の文字数を比で数え、発話ごとの
    目印を足す。実機のトークナイザーが持つ特徴 (数え方の口と `usage` が一致する、
    包みが system / tools の量に比例する) を偽のサーバーで再現する。
    """
    profile = _small_profile()
    counter = estimator_aligned_counter(fixed=15, per_message=10)
    fake_server.set_input_token_counter(counter)
    fake_server.set_response(text_response("ok"))
    task = task_for(profile, 0, 0)

    frames: dict[str, int] = {}
    async with suite_ctx(fake_server, profile=profile) as ctx:
        for label, system, tools in (
            ("none", None, None),
            ("system", SYSTEM_PROMPT, None),
            ("tools", None, list(TOOL_CATALOG)),
            ("both", SYSTEM_PROMPT, list(TOOL_CATALOG)),
        ):
            request = MessagesRequest(
                model=ctx.target.model,
                max_tokens=1,
                messages=[final_turn(task.prompt)],
                system=system,
                tools=tools,
            )
            counted = await ctx.count_input_tokens(request)
            result = await ctx.client.stream(request, profile.timeout)
            assert result.error is None
            assert result.usage is not None
            # 口の数え方と、実際に送った要求の usage が一致する
            assert counted.tokens == result.usage.total_input_tokens
            frames[label] = counted.tokens

    # 包みは system と tools で加法に増える (どちらか片方の効果を足したもの)
    assert frames["both"] - frames["none"] == (frames["system"] - frames["none"]) + (
        frames["tools"] - frames["none"]
    )
    # tools の分は、目録の JSON の文字数 ÷ 比 に比例する
    assert frames["tools"] - frames["none"] == round(len(_tools_json()) / _CHARS_PER_TOKEN)


# --- C3: 狙いへの到達 (修正後は許容の内側、修正前は 20k で外れる) -------------


async def test_the_stage_ladder_lands_within_the_tolerance_with_the_realistic_counter(
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
    counter = estimator_aligned_counter(fixed=15, per_message=10)
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
        assert abs(median - target) / target <= profile.length_tolerance
        assert all(TrialFlag.LENGTH_OFF_TARGET not in record.flags for record in stage_records)

    # 修正前 (測った包みを fixed_tokens に渡すだけ) の見積もりが取りこぼす、
    # 履歴に比例する分は、同じ数え方で 20k の中央値が許容 (5%) を超える
    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond_20k = planned_condition(suite, ctx, stage_condition_key(20_000))
        frame = await _measure_frame(ctx)
        gaps = [
            counter(
                expected_body(
                    ctx,
                    cond_20k,
                    build_conversation(
                        20_000,
                        derive_conversation_seed(profile.seed, conversation_index),
                        chars_per_token=profile.chars_per_token,
                        fixed_tokens=frame,
                    ),
                    task_for(profile, 0, 0),
                )
            )
            - build_conversation(
                20_000,
                derive_conversation_seed(profile.seed, conversation_index),
                chars_per_token=profile.chars_per_token,
                fixed_tokens=frame,
            ).approx_tokens
            for conversation_index in range(5)
        ]
    assert statistics.median(gaps) > profile.length_tolerance * 20_000


# --- C2: 段階 × 会話ごとに数えて、止める手番を決める ------------------------


async def test_the_conversation_is_fitted_by_counting_the_built_request(
    fake_server: FakeServer,
) -> None:
    profile = _small_profile()
    fake_server.set_response_factory(correct_responder(task_table(profile, _STAGES, 2)))
    suite = AgentSuite()
    calls: list[MessagesRequest] = []
    counter = estimator_aligned_counter(fixed=15, per_message=10)

    async def recording(request: MessagesRequest) -> TokenCount:
        calls.append(request)
        return TokenCount(tokens=counter(build_request_body(request)), method="count_tokens")

    async with suite_ctx(fake_server, profile=profile, count_input_tokens=recording) as ctx:
        records, skipped = await run_suite(suite, ctx)

    assert skipped == []
    assert records
    # 包み (発話 1 つ) は、計測ランにつき 1 回だけ数える
    frame_calls = [request for request in calls if len(request.messages) == 1]
    assert len(frame_calls) == 1
    assert frame_calls[0].max_tokens == 1
    assert frame_calls[0].system == SYSTEM_PROMPT
    assert frame_calls[0].tools == list(TOOL_CATALOG)
    assert all(request.max_tokens == 1 for request in calls)

    # 段階 × 会話ごとに、組み立てた会話 (最後の 1 手を含む) を数える
    for ordinal, _target in enumerate(_STAGES):
        prompt = task_for(profile, ordinal, 0).prompt
        stage_calls = [
            request
            for request in calls
            if len(request.messages) > 1 and request.messages[-1] == final_turn(prompt)
        ]
        assert 2 <= len(stage_calls) <= 4
        for request in stage_calls:
            assert request.system == SYSTEM_PROMPT
            assert request.tools == list(TOOL_CATALOG)

    # 送った履歴は、数えた候補の 1 つ (手番の境界) そのもの
    target = _STAGES[0]
    body0 = sent_bodies(fake_server)[0]
    sent_messages = body0["messages"]
    assert isinstance(sent_messages, list)
    best = next(
        request
        for request in calls
        if len(request.messages) > 1
        and [message.model_dump(mode="json") for message in request.messages] == sent_messages
    )
    frame = counter(build_request_body(frame_calls[0]))
    measured = counter(body0)
    # 訪れた候補のうち、数えた長さが狙いに最も近いものを送る。隣 (±1 手番) は近くない
    minus_messages = best.messages[: -(MESSAGES_PER_ROUND + 1)] + [best.messages[-1]]
    minus = best.model_copy(update={"messages": minus_messages})
    longer = build_conversation(
        target * 4,
        derive_conversation_seed(profile.seed, 0),
        chars_per_token=profile.chars_per_token,
        fixed_tokens=frame,
    )
    plus_history = longer.messages[: len(best.messages) - 1 + MESSAGES_PER_ROUND]
    plus = best.model_copy(update={"messages": [*plus_history, best.messages[-1]]})
    assert abs(measured - target) <= abs(counter(build_request_body(minus)) - target)
    assert abs(measured - target) <= abs(counter(build_request_body(plus)) - target)


async def test_the_fit_stops_at_the_limit_and_sends_the_closest_candidate(
    fake_server: FakeServer,
) -> None:
    """収束しない数え方では上限で止まり、訪れた候補のうち狙いに最も近いものを送る。

    素直に収束する数え方だけでは、上限 (`_MAX_LENGTH_COUNTS`) も「最後ではなく
    最も近い候補を送る」選択も働かない。意図的に振動させて、その両方を確かめる。
    """
    profile = _small_profile()
    fake_server.set_response_factory(correct_responder(task_table(profile, _STAGES, 2)))
    suite = AgentSuite()

    # 段階ごとに、狙いから意図的に外した値を順に返す。後ろの候補ほど遠くする
    deltas = (9000, -900, 9000, -900)
    prompt_target = {
        task_for(profile, ordinal, 0).prompt: target for ordinal, target in enumerate(_STAGES)
    }
    call_index: dict[str, int] = {}
    calls: list[tuple[str, int, int]] = []  # (課題の指示, 数えた長さ, 履歴の手番の数)

    async def oscillating(request: MessagesRequest) -> TokenCount:
        body: Any = build_request_body(request)
        messages = body["messages"]
        assert isinstance(messages, list)
        if len(messages) == 1:
            return TokenCount(tokens=1000, method="count_tokens")  # 包み
        last = messages[-1]
        content = last["content"]
        assert isinstance(content, list)
        prompt = content[0]["text"]
        assert isinstance(prompt, str)
        index = call_index.get(prompt, 0)
        call_index[prompt] = index + 1
        counted = prompt_target[prompt] + deltas[min(index, len(deltas) - 1)]
        calls.append((prompt, counted, (len(messages) - 1) // MESSAGES_PER_ROUND))
        return TokenCount(tokens=counted, method="count_tokens")

    async with suite_ctx(fake_server, profile=profile, count_input_tokens=oscillating) as ctx:
        records, skipped = await run_suite(suite, ctx)

    assert skipped == []
    assert records
    capped = 0
    for ordinal, target in enumerate(_STAGES):
        prompt = task_for(profile, ordinal, 0).prompt
        prompt_calls = [call for call in calls if call[0] == prompt]
        # 上限を超えて数えない
        assert 1 <= len(prompt_calls) <= 4
        if len(prompt_calls) == 4 and len({call[2] for call in prompt_calls}) == 4:
            # 手番の数が繰り返さないので、上限で止まった (収束では止まっていない)
            capped += 1
            sent = next(body for body in sent_bodies(fake_server) if last_user_text(body) == prompt)
            sent_rounds = len(history_of(sent)) // MESSAGES_PER_ROUND
            closest = min(prompt_calls, key=lambda call: abs(call[1] - target))
            assert sent_rounds == closest[2]
    assert capped >= 1, "上限で止まる段階が 1 つも無かった (この試験が空回りしている)"


async def test_the_fit_stops_when_the_count_is_not_above_the_frame(
    fake_server: FakeServer,
) -> None:
    """会話の計測が包み以下なら、比を作れず、最初の候補を送って止まる。"""
    profile = _small_profile()
    fake_server.set_response_factory(correct_responder(task_table(profile, _STAGES, 2)))
    suite = AgentSuite()
    conversation_calls: dict[str, int] = {}

    async def below_the_frame(request: MessagesRequest) -> TokenCount:
        body: Any = build_request_body(request)
        messages = body["messages"]
        assert isinstance(messages, list)
        if len(messages) == 1:
            return TokenCount(tokens=1000, method="count_tokens")
        prompt = messages[-1]["content"][0]["text"]
        assert isinstance(prompt, str)
        conversation_calls[prompt] = conversation_calls.get(prompt, 0) + 1
        return TokenCount(tokens=500, method="count_tokens")  # 包み 1000 以下

    async with suite_ctx(fake_server, profile=profile, count_input_tokens=below_the_frame) as ctx:
        records, skipped = await run_suite(suite, ctx)

    assert skipped == []
    assert records
    for ordinal, target in enumerate(_STAGES):
        prompt = task_for(profile, ordinal, 0).prompt
        # 比を作れないので、1 回数えたところで止まる
        assert conversation_calls[prompt] == 1
        first = build_conversation(
            target,
            derive_conversation_seed(profile.seed, 0),
            chars_per_token=profile.chars_per_token,
            fixed_tokens=1000,
        )
        sent = next(body for body in sent_bodies(fake_server) if last_user_text(body) == prompt)
        assert len(history_of(sent)) == len(first.messages)


# --- C5: 包みの計測は計測ランにつき 1 回 -------------------------------------


async def test_the_frame_is_measured_once_for_all_stages(fake_server: FakeServer) -> None:
    profile = _small_profile()
    fake_server.set_response_factory(correct_responder(task_table(profile, _STAGES, 2)))
    suite = AgentSuite()

    async with suite_ctx(fake_server, profile=profile) as ctx:
        records, skipped = await run_suite(suite, ctx)

    assert skipped == []
    assert len(records) == 2 * len(_STAGES)
    assert fake_server.call_count("/v1/messages") == 2 * len(_STAGES)

    count_requests = count_token_requests(fake_server)
    frame_requests = [
        request
        for request in count_requests
        if request.body is not None and len(request.body["messages"]) == 1
    ]
    # 発話 1 つの要求 (包み) は 1 件。会話ごとの計測は段階 × 会話につき 2〜4 件
    assert len(frame_requests) == 1
    assert len(count_requests) >= 1 + len(_STAGES) * 1 * 2
    assert len(count_requests) <= 1 + len(_STAGES) * 1 * 4


# --- C4: 上限の判定と 400 の理由には、数えた長さを使う ----------------------


async def test_a_conversation_that_no_longer_fits_reports_the_counted_tokens_and_sends_nothing(
    fake_server: FakeServer,
) -> None:
    profile = _small_profile()
    fake_server.set_input_token_counter(estimator_aligned_counter(fixed=15, per_message=10))
    fake_server.set_response_factory(correct_responder(task_table(profile, _STAGES, 2)))
    suite = AgentSuite()

    # 1 回目: 上限なしで送り、段階 4000 の試行 0 の、対象サーバーが数えた長さを得る
    async with suite_ctx(fake_server, profile=profile) as ctx:
        await run_suite(suite, ctx)
    first = sent_requests(fake_server)[0]
    counted = first.input_tokens
    assert counted > _STAGES[0], "この試験の前提: 数えた長さが狙いを上回ること"

    fake_server.reset()
    limit = counted + profile.agent.max_tokens - 1
    async with suite_ctx(fake_server, profile=profile, context_limit=limit) as ctx:
        cond = planned_condition(suite, ctx, stage_condition_key(4000))
        with pytest.raises(ConditionAborted) as raised:
            async for _ in suite.run_condition(ctx, cond):
                pass

    assert sent_requests(fake_server) == []
    assert str(counted) in raised.value.skipped.reason


# --- C6: 数える要求そのものが上限を超えて断られたとき (口なし) --------------


def _counted_first_candidate(
    profile: Profile, counter: InputTokenCounter, target: int, ordinal: int
) -> tuple[int, int]:
    """段階の 1 本目の候補 (`scale=1.0`) の、偽のサーバーが数える長さと、文字数の見積もり。

    `_measure_frame` と同じ包みを、同じ偽の数え方で数えて `fixed_tokens` にし、まとまりが
    最初に組み立てる会話と同じものを作る。数える長さは、その会話を `max_tokens=1` の
    要求にして数えたもの。
    """
    frame_task = task_for(profile, 0, 0)
    frame_request = MessagesRequest(
        model="fake-model",
        max_tokens=1,
        messages=[final_turn(frame_task.prompt)],
        system=SYSTEM_PROMPT,
        tools=list(TOOL_CATALOG),
    )
    frame = counter(build_request_body(frame_request))
    prefix = build_conversation(
        target,
        derive_conversation_seed(profile.seed, 0),
        chars_per_token=profile.chars_per_token,
        fixed_tokens=frame,
    )
    request = MessagesRequest(
        model="fake-model",
        max_tokens=1,
        messages=[*prefix.messages, final_turn(task_for(profile, ordinal, 0).prompt)],
        system=prefix.system,
        tools=prefix.tools,
    )
    return counter(build_request_body(request)), prefix.approx_tokens


async def test_a_stage_over_an_unadvertised_limit_is_skipped_with_the_later_stages(
    fake_server: FakeServer,
) -> None:
    """口がなく、上限を申告しない対象で、数える要求が上限を超えた段階から後ろを飛ばす。

    上限を申告しないので、まとまりは上限を知らない (`ctx.context_limit=None`)。段階 6000 の
    会話は、数えさせる要求 (`max_tokens=1`) の時点で上限を超えて HTTP 400 になる。この
    退行 (issue #24 の合わせ込みの導入) は、その `ProbeError` が外に出て、計測ランを止めていた。
    """
    profile = _small_profile()
    limit = 5000
    counter = estimator_aligned_counter(fixed=15, per_message=10)
    fake_server.set_count_tokens_enabled(False)
    fake_server.set_context_limit(limit, advertise=False)
    fake_server.set_input_token_counter(counter)
    fake_server.set_response_factory(correct_responder(task_table(profile, _STAGES, 2)))
    counted_6000, approx_6000 = _counted_first_candidate(profile, counter, _STAGES[1], 1)
    assert counted_6000 + 1 > limit, "この試験の前提: 1 本目の候補は、数える要求で上限を超える"
    suite = AgentSuite()

    async with suite_ctx(fake_server, profile=profile) as ctx:
        assert ctx.context_limit is None
        records, skipped = await run_suite(suite, ctx)  # ProbeError は外に出ない

    # 上限に収まる段階の試行は残る。計測ランは止まらない
    assert [record.target_input_tokens for record in records] == [_STAGES[0]] * 2
    assert all(record.result.error is None for record in records)
    # その段階と、後ろの段階が飛ばされる
    assert [item.key for item in skipped] == [
        stage_condition_key(_STAGES[1]),
        stage_condition_key(_STAGES[2]),
    ]
    # 理由の到達した長さは、数えた値ではなく見積もりで、そうとわかる書き方になっている
    reason = skipped[0].reason
    assert "見積もり" in reason
    assert str(approx_6000) in reason
    assert "対象サーバーは数えていない" in reason

    # 試行を送ったのは 4000 の段階だけ。6000 と 8000 の段階には、試行を 1 件も送らない
    sent = sent_bodies(fake_server)
    trials = [body for body in sent if body["max_tokens"] != 1]
    assert len(trials) == 2
    prompts = {last_user_text(body) for body in trials}
    assert prompts == {task_for(profile, 0, trial).prompt for trial in range(2)}
    # 8000 の段階は、数える要求も含めて 1 件も送らない (すでに上限に当たっている)
    later_prompts = {task_for(profile, 2, trial).prompt for trial in range(2)}
    assert later_prompts.isdisjoint(last_user_text(body) for body in sent)


async def test_the_first_candidate_over_the_known_limit_is_skipped_without_sending_a_trial(
    fake_server: FakeServer,
) -> None:
    """上限はわかっていて、計画は通るが、最初の候補が上限を超えるなら、試行を 1 件も送らない。

    上限は 4000 + 出力の上限 256 = 4256 で、計画では段階 4000 が収まる。しかし、組み立てた
    会話は、発話ごとの目印の分だけ狙いより長く、数える要求の時点で上限を超える。目印の
    分を 40 トークンにしているのは、1 本目の候補が計画の余裕 (出力の上限の 256) を超える
    ほど長くなる場合を作るため (前提は、試験の中で確かめる)。
    """
    profile = _small_profile()
    limit = _STAGES[0] + profile.agent.max_tokens
    counter = estimator_aligned_counter(fixed=15, per_message=40)
    fake_server.set_count_tokens_enabled(False)
    fake_server.set_context_limit(limit, advertise=False)
    fake_server.set_input_token_counter(counter)
    fake_server.set_response_factory(correct_responder(task_table(profile, _STAGES, 2)))
    counted_4000, approx_4000 = _counted_first_candidate(profile, counter, _STAGES[0], 0)
    assert counted_4000 + 1 > limit, "この試験の前提: 1 本目の候補は、数える要求で上限を超える"
    suite = AgentSuite()

    async with suite_ctx(fake_server, profile=profile, context_limit=limit) as ctx:
        cond = planned_condition(suite, ctx, stage_condition_key(_STAGES[0]))  # 計画は通る
        with pytest.raises(ConditionAborted) as raised:
            async for _ in suite.run_condition(ctx, cond):
                pass

    assert [body for body in sent_bodies(fake_server) if body["max_tokens"] != 1] == []
    reason = raised.value.skipped.reason
    assert raised.value.skipped.key == stage_condition_key(_STAGES[0])
    assert "見積もり" in reason
    assert str(approx_4000) in reason


async def test_a_later_candidate_over_the_limit_keeps_the_candidate_already_counted(
    fake_server: FakeServer,
) -> None:
    """2 本目以降の候補が上限を超えて断られたら、段階は飛ばさず、数え終えた候補を送る。

    数えたことのある候補だけが送られる。数えられなかった候補は、送らない。
    """
    profile = _small_profile()
    fake_server.set_response_factory(correct_responder(task_table(profile, _STAGES, 2)))
    suite = AgentSuite()
    prompt_target = {
        task_for(profile, ordinal, 0).prompt: target for ordinal, target in enumerate(_STAGES)
    }
    conversation_calls: dict[str, int] = {}

    async def refused_on_the_second_count(request: MessagesRequest) -> TokenCount:
        body: Any = build_request_body(request)
        messages = body["messages"]
        assert isinstance(messages, list)
        if len(messages) == 1:
            return TokenCount(tokens=1000, method="count_tokens")  # 包み
        prompt = messages[-1]["content"][0]["text"]
        assert isinstance(prompt, str)
        conversation_calls[prompt] = conversation_calls.get(prompt, 0) + 1
        if conversation_calls[prompt] == 1:
            return TokenCount(tokens=prompt_target[prompt] + 900, method="count_tokens")
        raise InputOverContextLimitError("over the limit")

    async with suite_ctx(
        fake_server, profile=profile, count_input_tokens=refused_on_the_second_count
    ) as ctx:
        records, skipped = await run_suite(suite, ctx)

    assert skipped == []
    assert len(records) == 2 * len(_STAGES)
    sent = sent_bodies(fake_server)
    for ordinal, target in enumerate(_STAGES):
        prompt = task_for(profile, ordinal, 0).prompt
        # 2 回目の計測で断られた (1 回目の候補だけが、数えられた)
        assert conversation_calls[prompt] == 2
        first = build_conversation(
            target,
            derive_conversation_seed(profile.seed, 0),
            chars_per_token=profile.chars_per_token,
            fixed_tokens=1000,
        )
        stage_trials = [body for body in sent if last_user_text(body) == prompt]
        assert len(stage_trials) == 1
        assert history_of(stage_trials[0]) == [
            message.model_dump(mode="json") for message in first.messages
        ]


# --- C5: ProbeError は推定に戻さず伝播する -----------------------------------


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


async def test_a_conversation_probe_error_is_not_turned_into_an_estimate(
    fake_server: FakeServer,
) -> None:
    """包みは数えられても、会話の計測が失敗したら、そのまま伝播する (見積もりに戻さない)。"""
    profile = _small_profile()
    fake_server.set_response_factory(correct_responder(task_table(profile, _STAGES, 2)))
    calls = {"count": 0}

    async def failing_after_the_frame(request: MessagesRequest) -> TokenCount:
        calls["count"] += 1
        if calls["count"] > 1:
            raise ProbeError("boom")
        return TokenCount(tokens=1000, method="count_tokens")

    suite = AgentSuite()
    async with suite_ctx(
        fake_server, profile=profile, count_input_tokens=failing_after_the_frame
    ) as ctx:
        with pytest.raises(ProbeError):
            await run_suite(suite, ctx)

    assert sent_requests(fake_server) == []
