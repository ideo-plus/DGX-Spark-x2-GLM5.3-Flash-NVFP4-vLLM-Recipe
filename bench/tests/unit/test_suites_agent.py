"""長い会話でのツール呼び出しの検査のまとまりの試験 (task 7.2: suites/agent)。

2 万〜12 万トークンの代わりに、小さな段階 (4000 / 6000 / 8000 トークン) の
`Profile` を使う。`chars_per_token` は、内容の種類によらず同じ比 (4.0) に
そろえてある (`Profile` の初期値 (コード 3.2、ログ 3.4、日本語 1.6) とは
わざと変えてあるので、まとまりが `chars_per_token` を渡し忘れると、組み立て
られる会話が変わって試験が落ちる)。

時刻を判定する試験は 1 つもない (条件の計画、会話の割り当て、送った本文、
判定、上限で止まること、決まった結果になることを確かめる試験だけ)。

確かめること (tasks.md 7.2 の完了の状態):

- 設定した段階と試行の数のとおりに実行される (送った要求 == レコード == 保存
  した本文)
- 上限が 7 万の偽のサーバーで、6 万の段階まで進んで止まり、理由が残る
  (要約の `reached_tokens` と `stopped_reason` まで通っていることを確かめる)

そのほか、要件 6.1、6.3、6.7、6.8、6.9 と、1.6、8.1、10.7 に沿った振る舞いを
確かめる。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from bench_harness.analysis.summarize import _tokens_from_label, summarize_run
from bench_harness.client.messages import HttpxMessagesClient, build_request_body
from bench_harness.corpus.conversation import (
    SYSTEM_PROMPT,
    build_conversation,
    derive_conversation_seed,
)
from bench_harness.corpus.tools import TOOL_CATALOG, make_tool_task
from bench_harness.scoring.toolcall import ToolTaskError
from bench_harness.store import RunStore
from bench_harness.suites.agent import (
    SUITE,
    AgentSuite,
    _measure_frame,
    stage_condition_key,
)
from bench_harness.suites.base import ConditionAborted, SuiteContext, make_suite_context
from bench_harness.types import (
    ConditionPlan,
    ConversationPrefix,
    InputMessage,
    MessagesRequest,
    Profile,
    RunManifest,
    RunStatus,
    SkippedCondition,
    SuiteName,
    TargetDef,
    TextBlockParam,
    ToolCallOutcome,
    ToolCallVerdict,
    ToolTask,
    TrialRecord,
)
from fake_server import (
    FakeServer,
    RecordedRequest,
    Script,
    estimator_aligned_counter,
    http_error_response,
    over_limit_response,
    text_response,
    tool_use_response,
)

RUN_A = "20260920-000000-aaaaaa"
RUN_B = "20260920-111111-bbbbbb"
AT = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)

# 2 万〜12 万の代わりに使う、小さな段階。1 段階が前置き (約 1.6k トークン) より
# 十分に長く、かつ試験が速く終わる長さにしてある
_STAGES: tuple[int, int, int] = (4000, 6000, 8000)

_MARKUP_TEXT = "<tool_call>read_file</tool_call> と書いてしまった応答"
"""`TargetDef.tool_markup_markers` の既定の目印を含む本文 (MARKUP_LEAKED になる)。"""


# --- 助け (test_suites_prefill.py から、この試験に要る分だけを写す) ---------


class BodySink:
    """`RunStore.put_body` の代わり。保存された本文をそのまま覚えておく。"""

    def __init__(self) -> None:
        self.bodies: list[dict[str, JsonValue]] = []

    def put(self, body: dict[str, JsonValue]) -> str:
        self.bodies.append(body)
        canonical = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def make_profile(**overrides: Any) -> Profile:
    """試験用の小さな設定。段階は 4000 / 6000 / 8000、会話 2 本、試行 4 回。"""
    data: dict[str, Any] = {
        "name": "test",
        "seed": 7,
        "sampling": {"temperature": 0.0},
        "timeout": {"connect_s": 5.0, "first_event_s": 5.0, "idle_s": 5.0, "total_s": 20.0},
        "length_tolerance": 0.05,
        "min_successes": 1,
        "chars_per_token": {"prose_en": 4.0, "prose_ja": 4.0, "code": 4.0, "log": 4.0},
        "agent": {
            "start_tokens": _STAGES[0],
            "end_tokens": _STAGES[-1],
            "step_tokens": _STAGES[1] - _STAGES[0],
            "trials_per_stage": 4,
            "conversations_per_stage": 2,
            "threshold": 0.01,
            "max_tokens": 256,
        },
    }
    data.update(overrides)
    return Profile.model_validate(data)


def agent_settings(**overrides: Any) -> dict[str, Any]:
    """`make_profile(agent=...)` に渡す、段階の設定の写像。"""
    data: dict[str, Any] = {
        "start_tokens": _STAGES[0],
        "end_tokens": _STAGES[-1],
        "step_tokens": _STAGES[1] - _STAGES[0],
        "trials_per_stage": 4,
        "conversations_per_stage": 2,
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
) -> AsyncIterator[SuiteContext]:
    """偽のサーバーにつないだ `SuiteContext`。"""
    async with HttpxMessagesClient(server.base_url) as client:
        yield make_suite_context(
            client=client,
            profile=profile if profile is not None else make_profile(),
            target=target_for(server),
            run_id=run_id,
            put_body=(sink if sink is not None else BodySink()).put,
            context_limit=context_limit,
        )


async def run_suite(
    ctx: SuiteContext, *, store: RunStore | None = None
) -> tuple[list[TrialRecord], list[SkippedCondition]]:
    """計測ランの進行 (3.5) と同じ順で、まとまり全体を流す。

    `ConditionAborted` は捕まえて `skipped` に足し、次の条件に進む
    (`suites/base.py`「計測ランの進行への契約」)。
    """
    records: list[TrialRecord] = []
    skipped: list[SkippedCondition] = []
    for item in SUITE.plan(ctx):
        if isinstance(item, SkippedCondition):
            skipped.append(item)
            continue
        try:
            async for record in SUITE.run_condition(ctx, item):
                records.append(record)
                if store is not None:
                    store.append_trial(record)
        except ConditionAborted as exc:
            skipped.append(exc.skipped)
    return records, skipped


async def run_stage(ctx: SuiteContext, cond: ConditionPlan) -> list[TrialRecord]:
    return [record async for record in SUITE.run_condition(ctx, cond)]


def planned_condition(ctx: SuiteContext, key: str) -> ConditionPlan:
    plan = next(item for item in SUITE.plan(ctx) if item.key == key)
    assert isinstance(plan, ConditionPlan)
    return plan


def sent_requests(server: FakeServer) -> list[RecordedRequest]:
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


def prefix_for(
    profile: Profile, stage_tokens: int, conversation_index: int, *, frame_tokens: int
) -> ConversationPrefix:
    """まとまりが組み立てるのと同じ会話を、計測した包み (issue #9) で作る。"""
    return build_conversation(
        stage_tokens,
        derive_conversation_seed(profile.seed, conversation_index),
        chars_per_token=profile.chars_per_token,
        fixed_tokens=frame_tokens,
    )


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


# --- 応答の作り方 (偽のサーバーの responder) --------------------------------


def task_table(profile: Profile, stages: Sequence[int], trials: int) -> dict[str, ToolTask]:
    """この計測ランで送られる課題を、指示の文から引ける表にする。"""
    table: dict[str, ToolTask] = {}
    for ordinal in range(len(stages)):
        for trial_index in range(trials):
            task = task_for(profile, ordinal, trial_index)
            table[task.prompt] = task
    return table


def wrong_call_for(task: ToolTask, seed: int) -> ToolTask:
    """正解と違うツールを呼ぶ課題 (引数はそのツールの定義に合う → WRONG_CALL)。"""
    for index in range(900_000, 900_100):
        other = make_tool_task(index, seed)
        if other.expected_tool != task.expected_tool:
            return other
    raise AssertionError("違うツールの課題が見つからない")


def correct_responder(table: dict[str, ToolTask]) -> Callable[[dict[str, Any]], Script]:
    """どの試行にも、その課題の正解のツール呼び出しを返す。"""

    def factory(body: dict[str, Any]) -> Script:
        task = table[last_user_text(body)]
        return tool_use_response(task.expected_tool, task.expected_input)

    return factory


def mixed_responder(
    table: dict[str, ToolTask], seed: int, kinds: Sequence[str]
) -> Callable[[dict[str, Any]], Script]:
    """指示の文から、4 種類の分類を決まったとおりに出し分ける。

    どの指示にどの種類を返すかは、課題の番号 (= 表に入れた順) で決まるので、
    要求の届く順に依存しない。
    """
    order = {prompt: index for index, prompt in enumerate(table)}

    def factory(body: dict[str, Any]) -> Script:
        prompt = last_user_text(body)
        task = table[prompt]
        kind = kinds[order[prompt] % len(kinds)]
        if kind == "correct":
            return tool_use_response(task.expected_tool, task.expected_input)
        if kind == "wrong_call":
            other = wrong_call_for(task, seed)
            return tool_use_response(other.expected_tool, other.expected_input)
        if kind == "markup_leaked":
            return text_response(_MARKUP_TEXT)
        return http_error_response(500)

    return factory


# --- 計画: 段階の鍵、試行の数、慣らしなし (6.1、6.8) ------------------------


async def test_plan_makes_one_condition_per_stage_with_zero_padded_keys(
    fake_server: FakeServer,
) -> None:
    profile = make_profile()
    async with suite_ctx(fake_server, profile=profile) as ctx:
        planned = SUITE.plan(ctx)

    assert [item.key for item in planned] == [
        "agent/stage/004k",
        "agent/stage/006k",
        "agent/stage/008k",
    ]
    for item, length in zip(planned, _STAGES, strict=True):
        assert isinstance(item, ConditionPlan)
        assert item.suite is SuiteName.AGENT
        assert item.tier == "primary"
        assert item.trials == profile.agent.trials_per_stage
        assert item.max_tokens == profile.agent.max_tokens
        assert item.target_input_tokens == length
        # 慣らしは行わない (集計に入らない長い要求を、段階ごとに 1 回払わない)
        assert item.warmup_trials == 0


def test_stage_keys_are_readable_by_the_summary_even_past_a_thousand_k() -> None:
    """要約は鍵の末尾から段階の長さを読む (`analysis/summarize._tokens_from_label`)。"""
    for tokens, label in ((20_000, "020k"), (120_000, "120k"), (1_000_000, "1000k")):
        key = stage_condition_key(tokens)
        assert key == f"agent/stage/{label}"
        assert _tokens_from_label(key.rsplit("/", 1)[-1]) == tokens
    # 千で割り切れない段階も、要約が読める形にする
    assert stage_condition_key(2500) == "agent/stage/2500"
    assert _tokens_from_label("2500") == 2500


async def test_stage_list_follows_start_end_and_step(fake_server: FakeServer) -> None:
    profile = make_profile(
        agent=agent_settings(start_tokens=5000, end_tokens=16_000, step_tokens=5000)
    )
    async with suite_ctx(fake_server, profile=profile) as ctx:
        keys = [item.key for item in SUITE.plan(ctx)]
    # 5000、10000、15000 (16000 は刻みに乗らないので入らない)
    assert keys == ["agent/stage/005k", "agent/stage/010k", "agent/stage/015k"]


# --- 送った要求 == レコード == 保存した本文 (8.1、注 3.5) -------------------


async def test_sends_exactly_stages_times_trials_requests_and_keeps_every_one(
    fake_server: FakeServer,
) -> None:
    profile = make_profile()
    table = task_table(profile, _STAGES, profile.agent.trials_per_stage)
    fake_server.set_response_factory(correct_responder(table))
    sink = BodySink()

    async with suite_ctx(fake_server, profile=profile, sink=sink) as ctx:
        records, skipped = await run_suite(ctx)

    expected = len(_STAGES) * profile.agent.trials_per_stage
    assert skipped == []
    assert len(sent_requests(fake_server)) == expected
    assert len(records) == expected
    assert len(sink.bodies) == expected
    assert all(record.warmup is False for record in records)
    assert {record.condition for record in records} == {
        "agent/stage/004k",
        "agent/stage/006k",
        "agent/stage/008k",
    }
    assert all(record.target_input_tokens is not None for record in records)


async def test_stages_are_sent_in_ascending_order(fake_server: FakeServer) -> None:
    """段階は短いほうから送る (前の段階で温まった前置きのキャッシュに当てる)。"""
    profile = make_profile()
    fake_server.set_response_factory(
        correct_responder(task_table(profile, _STAGES, profile.agent.trials_per_stage))
    )

    async with suite_ctx(fake_server, profile=profile) as ctx:
        records, _ = await run_suite(ctx)

    sent_lengths = [len(body["messages"]) for body in sent_bodies(fake_server)]
    stage_order = [record.condition for record in records]
    assert stage_order == [
        key
        for key in ("agent/stage/004k", "agent/stage/006k", "agent/stage/008k")
        for _ in range(profile.agent.trials_per_stage)
    ]
    # 発話の数は段階が伸びるほど増えるので、降順に送る壊れ方はここで落ちる
    assert min(sent_lengths[:4]) < min(sent_lengths[4:8]) < min(sent_lengths[8:])


# --- 会話の割り当て: 均等、かたまりごと (6.1) -------------------------------


async def test_trials_are_spread_evenly_over_conversations_in_blocks(
    fake_server: FakeServer,
) -> None:
    profile = make_profile()
    fake_server.set_response_factory(
        correct_responder(task_table(profile, _STAGES, profile.agent.trials_per_stage))
    )

    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = planned_condition(ctx, "agent/stage/004k")
        await run_stage(ctx, cond)
        frame = await _measure_frame(ctx)
        histories = [history_of(body) for body in sent_bodies(fake_server)]
        first = prefix_for(profile, _STAGES[0], 0, frame_tokens=frame)
        second = prefix_for(profile, _STAGES[0], 1, frame_tokens=frame)

    expected_first = json.loads(
        json.dumps([message.model_dump(mode="json") for message in first.messages])
    )
    expected_second = json.loads(
        json.dumps([message.model_dump(mode="json") for message in second.messages])
    )
    assert expected_first != expected_second  # 会話の種が違えば、1 手番目から違う
    # 4 試行 ÷ 2 本 = 2 回ずつ。続けて同じ会話を使う (前置きのキャッシュに当てる)
    assert histories == [expected_first, expected_first, expected_second, expected_second]


async def test_the_remainder_goes_to_the_first_conversations(fake_server: FakeServer) -> None:
    """割り切れないときの決まり (4 試行 ÷ 3 本 = 2、1、1) を、実際に作って確かめる。"""
    profile = make_profile(agent=agent_settings(conversations_per_stage=3, trials_per_stage=4))
    fake_server.set_response_factory(
        correct_responder(task_table(profile, _STAGES, profile.agent.trials_per_stage))
    )

    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = planned_condition(ctx, "agent/stage/004k")
        await run_stage(ctx, cond)
        frame = await _measure_frame(ctx)
        histories = [history_of(body) for body in sent_bodies(fake_server)]
        prefixes = [
            prefix_for(profile, _STAGES[0], index, frame_tokens=frame) for index in range(3)
        ]

    expected = [
        json.loads(json.dumps([message.model_dump(mode="json") for message in prefix.messages]))
        for prefix in prefixes
    ]
    assert len({json.dumps(item) for item in expected}) == 3
    assert histories == [expected[0], expected[0], expected[1], expected[2]]


# --- 送る本文の形 (6.1、6.2、7.1 の申し送り) --------------------------------


async def test_every_request_is_the_history_plus_one_user_message_with_the_task(
    fake_server: FakeServer,
) -> None:
    """本文をまるごと突き合わせる (system、tools、max_tokens、履歴、最後の 1 手)。"""
    profile = make_profile()
    fake_server.set_response_factory(
        correct_responder(task_table(profile, _STAGES, profile.agent.trials_per_stage))
    )

    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = planned_condition(ctx, "agent/stage/006k")
        await run_stage(ctx, cond)
        frame = await _measure_frame(ctx)
        bodies = sent_bodies(fake_server)
        expected = [
            expected_body(
                ctx,
                cond,
                prefix_for(profile, _STAGES[1], 0 if trial_index < 2 else 1, frame_tokens=frame),
                task_for(profile, 1, trial_index),
            )
            for trial_index in range(profile.agent.trials_per_stage)
        ]

    assert len(bodies) == len(expected)
    for body, want in zip(bodies, expected, strict=True):
        assert body == json.loads(json.dumps(want, ensure_ascii=False))


async def test_system_is_the_conversation_prompt_without_a_cache_busting_nonce(
    fake_server: FakeServer,
) -> None:
    """長い会話の検査は、前置きのキャッシュに当てるのが目的なので、識別子を足さない。"""
    profile = make_profile()
    fake_server.set_response_factory(
        correct_responder(task_table(profile, _STAGES, profile.agent.trials_per_stage))
    )

    async with suite_ctx(fake_server, profile=profile) as ctx:
        await run_suite(ctx)

    bodies = sent_bodies(fake_server)
    assert {body["system"] for body in bodies} == {SYSTEM_PROMPT}
    tools = [tool.model_dump(mode="json", exclude_none=True) for tool in TOOL_CATALOG]
    for body in bodies:
        assert body["tools"] == json.loads(json.dumps(tools, ensure_ascii=False))
        assert body["max_tokens"] == profile.agent.max_tokens


async def test_a_longer_stage_starts_with_the_shorter_stage_history(
    fake_server: FakeServer,
) -> None:
    """同じ会話なら、長い段階の履歴が短い段階の履歴を、そのまま先頭に含む (6.1)。"""
    profile = make_profile()
    fake_server.set_response_factory(
        correct_responder(task_table(profile, _STAGES, profile.agent.trials_per_stage))
    )

    async with suite_ctx(fake_server, profile=profile) as ctx:
        await run_suite(ctx)

    bodies = sent_bodies(fake_server)
    trials = profile.agent.trials_per_stage
    # 各段階の 1 本目 (会話 0 の最初の試行) を取り出して比べる
    per_stage = [history_of(bodies[stage * trials]) for stage in range(len(_STAGES))]
    assert len(per_stage[0]) < len(per_stage[1]) < len(per_stage[2])
    assert per_stage[1][: len(per_stage[0])] == per_stage[0]
    assert per_stage[2][: len(per_stage[1])] == per_stage[1]


async def test_every_trial_asks_a_different_final_task(fake_server: FakeServer) -> None:
    profile = make_profile()
    fake_server.set_response_factory(
        correct_responder(task_table(profile, _STAGES, profile.agent.trials_per_stage))
    )

    async with suite_ctx(fake_server, profile=profile) as ctx:
        await run_suite(ctx)

    prompts = [last_user_text(body) for body in sent_bodies(fake_server)]
    assert len(prompts) == len(_STAGES) * profile.agent.trials_per_stage
    assert len(set(prompts)) == len(prompts)


# --- 判定 (6.3) と、要約への受け渡し (4.2) ----------------------------------


async def test_verdicts_cover_correct_wrong_call_markup_leaked_and_request_failed(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    profile = make_profile()
    table = task_table(profile, _STAGES, profile.agent.trials_per_stage)
    kinds = ("correct", "wrong_call", "markup_leaked", "request_failed")
    fake_server.set_response_factory(mixed_responder(table, profile.seed, kinds))
    store = RunStore.create(tmp_path / "results", make_manifest(profile))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        records, skipped = await run_suite(ctx, store=store)
    store.set_status(RunStatus.COMPLETED, AT)

    assert skipped == []
    outcomes = [record.verdict for record in records]
    assert all(isinstance(verdict, ToolCallVerdict) for verdict in outcomes)
    counted = [verdict.outcome for verdict in outcomes if isinstance(verdict, ToolCallVerdict)]
    assert counted.count(ToolCallOutcome.CORRECT) == 3
    assert counted.count(ToolCallOutcome.WRONG_CALL) == 3
    assert counted.count(ToolCallOutcome.MARKUP_LEAKED) == 3
    assert counted.count(ToolCallOutcome.REQUEST_FAILED) == 3

    summary = summarize_run(store.run_dir).summary
    assert summary.agent is not None
    assert [stage.stage_key for stage in summary.agent.stages] == [
        "agent/stage/004k",
        "agent/stage/006k",
        "agent/stage/008k",
    ]
    for stage in summary.agent.stages:
        assert stage.trials == 4
        assert stage.outcome_counts[ToolCallOutcome.CORRECT] == 1
        assert stage.outcome_counts[ToolCallOutcome.WRONG_CALL] == 1
        assert stage.outcome_counts[ToolCallOutcome.MARKUP_LEAKED] == 1
        assert stage.request_failures == 1
        assert stage.break_rate is not None
        # 要求そのものの失敗は分母から外す (4 − 1 = 3)。崩れは 2 件
        assert (stage.break_rate.numerator, stage.break_rate.denominator) == (2, 3)
        assert stage.actual_input_tokens is not None


async def test_a_failed_request_still_carries_a_verdict(fake_server: FakeServer) -> None:
    """要求が失敗しても、判定 (`REQUEST_FAILED`) を必ず付ける (4.2 の申し送り)。"""
    profile = make_profile(agent=agent_settings(trials_per_stage=2, conversations_per_stage=1))
    fake_server.set_response(http_error_response(500))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        records = await run_stage(ctx, planned_condition(ctx, "agent/stage/004k"))

    assert len(records) == 2
    for record in records:
        assert record.result.error is not None
        assert isinstance(record.verdict, ToolCallVerdict)
        assert record.verdict.outcome is ToolCallOutcome.REQUEST_FAILED


async def test_markup_markers_come_from_the_target_definition(fake_server: FakeServer) -> None:
    """目印を渡さないと、同じ応答が `NO_CALL` になる (分類が変わる)。"""
    profile = make_profile(agent=agent_settings(trials_per_stage=1, conversations_per_stage=1))
    fake_server.set_response(text_response(_MARKUP_TEXT))

    async with HttpxMessagesClient(fake_server.base_url) as client:
        with_markers = make_suite_context(
            client=client,
            profile=profile,
            target=target_for(fake_server),
            run_id=RUN_A,
            put_body=BodySink().put,
        )
        without = make_suite_context(
            client=client,
            profile=profile,
            target=target_for(fake_server, tool_markup_markers=[]),
            run_id=RUN_A,
            put_body=BodySink().put,
        )
        leaked = await run_stage(with_markers, planned_condition(with_markers, "agent/stage/004k"))
        plain = await run_stage(without, planned_condition(without, "agent/stage/004k"))

    assert isinstance(leaked[0].verdict, ToolCallVerdict)
    assert leaked[0].verdict.outcome is ToolCallOutcome.MARKUP_LEAKED
    assert isinstance(plain[0].verdict, ToolCallVerdict)
    assert plain[0].verdict.outcome is ToolCallOutcome.NO_CALL


async def test_a_broken_task_stops_the_suite_instead_of_becoming_a_verdict(
    fake_server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ToolTaskError` は、モデルの誤りとして握りつぶさない (注 6.3)。"""
    profile = make_profile(agent=agent_settings(trials_per_stage=2, conversations_per_stage=1))
    fake_server.set_response(text_response("ok"))

    def broken(index: int, seed: int) -> ToolTask:
        task = make_tool_task(index, seed)
        return ToolTask(
            prompt=task.prompt,
            tools=list(TOOL_CATALOG),
            expected_tool="no_such_tool",
            expected_input={},
        )

    monkeypatch.setattr("bench_harness.suites.agent.make_tool_task", broken)

    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = planned_condition(ctx, "agent/stage/004k")
        with pytest.raises(ToolTaskError):
            await run_stage(ctx, cond)


# --- 出力の印 (10.7): ふつうの試行に、余計な印を付けない -------------------


async def test_a_normal_tool_call_trial_carries_no_flags(fake_server: FakeServer) -> None:
    """ツール呼び出しの応答は短く、`tool_use` で終わる。印は 1 つも付かない。

    段階は 20k にする (会話の丸めの最悪が、4k では 5% の許容を超えうるため)。
    """
    profile = make_profile(
        agent=agent_settings(
            start_tokens=20_000,
            end_tokens=20_000,
            step_tokens=20_000,
            trials_per_stage=2,
            conversations_per_stage=1,
        )
    )
    fake_server.set_input_token_counter(estimator_aligned_counter(fixed=0))
    table = task_table(profile, (20_000,), profile.agent.trials_per_stage)
    fake_server.set_response_factory(correct_responder(table))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = planned_condition(ctx, "agent/stage/020k")
        records = await run_stage(ctx, cond)

    for record in records:
        assert record.result.usage is not None
        assert record.result.stop_reason == "tool_use"
        assert record.flags == []


# --- 上限 (6.8、6.9): 申告された上限で、送る前に止める ---------------------


async def test_stops_at_the_stage_beyond_an_advertised_limit_and_reports_it(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """tasks.md 7.2 の完了の状態: 上限 7 万で、6 万の段階まで進んで止まる。"""
    profile = make_profile(
        agent=agent_settings(
            start_tokens=20_000,
            end_tokens=120_000,
            step_tokens=20_000,
            trials_per_stage=1,
            conversations_per_stage=1,
        )
    )
    table = task_table(profile, (20_000, 40_000, 60_000), 1)
    fake_server.set_context_limit(70_000)
    fake_server.set_response_factory(correct_responder(table))
    store = RunStore.create(tmp_path / "results", make_manifest(profile, context_limit=70_000))

    async with suite_ctx(fake_server, profile=profile, context_limit=70_000) as ctx:
        fake_server.set_input_token_counter(estimator_aligned_counter(fixed=0))
        records, skipped = await run_suite(ctx, store=store)
    store.update_manifest(skipped=[item.model_dump(mode="json") for item in skipped])
    store.set_status(RunStatus.COMPLETED, AT)

    assert [record.condition for record in records] == [
        "agent/stage/020k",
        "agent/stage/040k",
        "agent/stage/060k",
    ]
    assert [item.key for item in skipped] == [
        "agent/stage/080k",
        "agent/stage/100k",
        "agent/stage/120k",
    ]
    assert "80000" in skipped[0].reason and "70000" in skipped[0].reason
    # 80k 以上の段階では、要求を 1 つも送っていない
    assert len(sent_requests(fake_server)) == 3

    summary = summarize_run(store.run_dir).summary
    assert summary.agent is not None
    assert summary.agent.reached_tokens == 60_000
    assert summary.agent.stopped_reason == skipped[0].reason
    assert [stage.target_input_tokens for stage in summary.agent.stages] == [
        20_000,
        40_000,
        60_000,
    ]


async def test_a_runtime_context_limit_aborts_the_stage_and_skips_the_longer_ones(
    fake_server: FakeServer,
) -> None:
    """上限を申告しないサーバー: 400 のレコードは残し、長い段階には送らない。"""
    profile = make_profile(agent=agent_settings(trials_per_stage=2, conversations_per_stage=1))
    table = task_table(profile, _STAGES, profile.agent.trials_per_stage)
    correct = correct_responder(table)

    async with suite_ctx(fake_server, profile=profile, context_limit=None) as ctx:
        frame = await _measure_frame(ctx)
        # 6000 の段階から、発話の数が 4000 の段階を超える。その境目で 400 を返す
        boundary = len(prefix_for(profile, _STAGES[0], 0, frame_tokens=frame).messages) + 1

        def factory(body: dict[str, Any]) -> Script:
            messages = body["messages"]
            assert isinstance(messages, list)
            if len(messages) > boundary:
                return over_limit_response(70_000)
            return correct(body)

        fake_server.set_response_factory(factory)
        records, skipped = await run_suite(ctx)

    conditions = [record.condition for record in records]
    assert conditions == ["agent/stage/004k"] * 2 + ["agent/stage/006k"]
    assert records[-1].result.error is not None
    assert records[-1].result.error.http_status == 400
    assert isinstance(records[-1].verdict, ToolCallVerdict)
    assert records[-1].verdict.outcome is ToolCallOutcome.REQUEST_FAILED
    assert [item.key for item in skipped] == ["agent/stage/006k", "agent/stage/008k"]
    # 8k の段階には 1 つも送っていない (4k が 2 本 + 6k が 1 本)
    assert len(sent_requests(fake_server)) == 3


async def test_a_conversation_that_no_longer_fits_is_not_sent_at_all(
    fake_server: FakeServer,
) -> None:
    """組み立てた会話が上限に収まらなければ、送る前に止める (注 7.1 → 7.2)。

    計画の判定は段階の**狙い**の長さで行うので、実際の会話が狙いより長い
    ときは、ここだけが要求を止められる。
    """
    profile = make_profile(agent=agent_settings(trials_per_stage=2, conversations_per_stage=1))
    fake_server.set_response(text_response("ok"))
    # 狙い (4000) + 出力の上限 はちょうど収まるが、実際の会話 (包みを含む) は収まらない
    limit = _STAGES[0] + profile.agent.max_tokens

    async with suite_ctx(fake_server, profile=profile, context_limit=limit) as ctx:
        frame = await _measure_frame(ctx)
        prefix = prefix_for(profile, _STAGES[0], 0, frame_tokens=frame)
        estimated = prefix.approx_tokens
        assert estimated > _STAGES[0], "この試験の前提: 実際の会話が狙いより長いこと"
        assert estimated + profile.agent.max_tokens > limit

        planned = SUITE.plan(ctx)
        assert isinstance(planned[0], ConditionPlan)  # 計画では飛ばされない
        with pytest.raises(ConditionAborted) as raised:
            await run_stage(ctx, planned[0])

    assert sent_requests(fake_server) == []
    assert raised.value.skipped.key == "agent/stage/004k"
    assert str(prefix.approx_tokens) in raised.value.skipped.reason


# --- 決まった結果になること (11.4) ------------------------------------------


async def test_two_runs_with_the_same_seed_send_identical_bodies(
    fake_server: FakeServer,
) -> None:
    profile = make_profile()
    fake_server.set_response_factory(
        correct_responder(task_table(profile, _STAGES, profile.agent.trials_per_stage))
    )

    fake_server.reset()
    async with suite_ctx(fake_server, profile=profile, run_id=RUN_A) as ctx:
        await run_suite(ctx)
    bodies_a = sent_bodies(fake_server)

    fake_server.reset()
    async with suite_ctx(fake_server, profile=profile, run_id=RUN_B) as ctx:
        await run_suite(ctx)
    bodies_b = sent_bodies(fake_server)

    assert len(bodies_a) == len(_STAGES) * profile.agent.trials_per_stage
    assert bodies_a == bodies_b


async def test_the_frame_is_measured_once_per_run_and_again_for_a_new_run(
    fake_server: FakeServer,
) -> None:
    """包みの計測は計測ランにつき 1 回で、run_id が変われば測り直す (issue #9)。

    同時処理の `test_the_frame_is_measured_once_per_run_and_again_for_a_new_run` と
    同じ契約を、長い会話の側でも確かめる (`AgentSuite` は同じ覚え書きの形を別に
    実装している)。`plan()` が `_forget_run_state()` を通り、`run_id` が変われば
    `_frame_tokens_for` がもう一度数えるので、`count_tokens` はランごとに 1 回になる。
    """
    profile = make_profile(agent=agent_settings(trials_per_stage=2, conversations_per_stage=1))
    fake_server.set_response_factory(
        correct_responder(task_table(profile, _STAGES, profile.agent.trials_per_stage))
    )

    async with suite_ctx(fake_server, profile=profile, run_id=RUN_A) as ctx:
        records_a, skipped_a = await run_suite(ctx)
    assert skipped_a == []
    assert records_a
    assert fake_server.call_count("/v1/messages/count_tokens") == 1

    fake_server.reset()
    async with suite_ctx(fake_server, profile=profile, run_id=RUN_B) as ctx:
        records_b, skipped_b = await run_suite(ctx)
    assert skipped_b == []
    assert records_b
    assert fake_server.call_count("/v1/messages/count_tokens") == 1


# --- 中断 (10.4): 途中で打ち切っても、レコードが食い違わない ---------------


async def test_cancelling_mid_stage_leaves_records_that_match_the_sent_requests(
    fake_server: FakeServer,
) -> None:
    profile = make_profile(agent=agent_settings(trials_per_stage=4, conversations_per_stage=1))
    table = task_table(profile, _STAGES, profile.agent.trials_per_stage)
    correct = correct_responder(table)
    order: list[str] = []

    def factory(body: dict[str, Any]) -> Script:
        order.append(last_user_text(body))
        script = correct(body)
        # 3 本目から止める (2 本ぶんのレコードを残したまま、打ち切らせる)
        if len(order) >= 3:
            return Script(events=script.events, first_delay_s=5.0)
        return script

    fake_server.set_response_factory(factory)
    sink = BodySink()
    records: list[TrialRecord] = []

    async with suite_ctx(fake_server, profile=profile, sink=sink) as ctx:
        cond = planned_condition(ctx, "agent/stage/004k")

        async def consume() -> None:
            async for record in SUITE.run_condition(ctx, cond):
                records.append(record)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.5)

    sent = len(sent_requests(fake_server))
    assert 0 < len(records) < profile.agent.trials_per_stage
    # 打ち切られた 1 本はレコードにならないが、本文は送る前に保存されている
    assert len(records) == sent - 1
    assert len(sink.bodies) == sent
    assert [record.trial_index for record in records] == list(range(len(records)))


def test_agent_suite_exposes_a_module_level_instance() -> None:
    assert isinstance(SUITE, AgentSuite)
    assert SUITE.name is SuiteName.AGENT


# --- 生データの入れ物 -------------------------------------------------------


def make_manifest(profile: Profile, **overrides: Any) -> RunManifest:
    payload: dict[str, Any] = {
        "run_id": RUN_A,
        "status": RunStatus.RUNNING,
        "target": {
            "name": "fake",
            "base_url": "http://127.0.0.1:1",
            "model": "fake-model",
        },
        "server_model": "fake-model",
        "started_at": AT,
        "suites": [SuiteName.AGENT],
        "profile_name": profile.name,
        "profile": profile,
        "harness_version": "0.1.0+gtest",
    }
    payload.update(overrides)
    return RunManifest.model_validate(payload)


def test_more_trials_than_the_task_index_stride_are_refused_before_anything_is_sent() -> None:
    """課題の番号は「段階の番号 × 10,000 + 試行の番号」。10,000 までは、次の段階とぶつからない。"""
    from bench_harness.suites import agent as agent_module
    from bench_harness.types import AgentSettings

    ok = AgentSettings(trials_per_stage=agent_module._TASK_INDEX_STRIDE)
    agent_module._check_trials_fit_the_index_stride(ok)
    last_of_stage_0 = agent_module._task_index(0, agent_module._TASK_INDEX_STRIDE - 1)
    assert last_of_stage_0 < agent_module._task_index(1, 0)

    too_many = AgentSettings(trials_per_stage=agent_module._TASK_INDEX_STRIDE + 1)
    with pytest.raises(ValueError, match="trials_per_stage"):
        agent_module._check_trials_fit_the_index_stride(too_many)
