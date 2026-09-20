"""品質の検査のまとまりの試験 (task 6.7: suites/quality)。

偽のサーバーを相手にして、3 種類の条件 (ツール呼び出し、コードの課題、探す
課題) が実行され、正解・不正解・採点できなかった、が分かれて記録されることを
確かめる (tasks.md 6.7 の完了の状態)。

コードの課題は、単体の試験では**隔離の実行環境と課題の取得を差し替える**
(`FakeSandbox`、`loader_for`)。Docker にもネットワークにも触れない。実物の
コンテナを使う確認は `tests/integration/test_quality_with_sandbox.py` にある。

`chars_per_token` は、内容の種類によらず 3.7 にそろえてある (`Profile` の初期値
(4.0 / 1.6 / 3.2 / 3.4) とわざと変えてあるので、まとまりが比を渡し忘れると、
組み立てられる干し草が変わって試験が落ちる)。

時刻を判定する試験は 1 つもない (条件の計画、送った本文、判定、飛ばした理由、
上限で止まること、決まった結果になることを確かめる試験だけ)。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from bench_harness.analysis.summarize import (
    METRIC_ACCURACY,
    NOT_SCORED_COUNT,
    summarize_run,
    write_summary,
)
from bench_harness.client.messages import HttpxMessagesClient
from bench_harness.corpus import humaneval
from bench_harness.corpus.needle import make_needle_case
from bench_harness.corpus.tools import make_tool_task
from bench_harness.scoring.code import SENTINEL_PREFIX
from bench_harness.scoring.sandbox import SandboxError
from bench_harness.scoring.toolcall import ToolTaskError
from bench_harness.store import RunStore
from bench_harness.suites.base import ConditionAborted, SuiteContext, make_suite_context
from bench_harness.suites.quality import (
    CODE_CONDITION_KEY,
    MAX_CONSECUTIVE_SANDBOX_ERRORS,
    NEEDLE_MAX_TOKENS,
    TOOLCALL_CONDITION_KEY,
    TOOLCALL_TASK_INDEX_BASE,
    ProblemsLoader,
    QualitySuite,
    needle_condition_key,
)
from bench_harness.types import (
    CodeProblem,
    ConditionPlan,
    ContentKind,
    DatasetRef,
    NeedleCase,
    Profile,
    QualityOutcome,
    QualityVerdict,
    RunManifest,
    RunStatus,
    SandboxResult,
    SandboxSettings,
    SandboxUnavailable,
    SkippedCondition,
    SuiteName,
    TargetDef,
    ToolCallOutcome,
    ToolCallVerdict,
    ToolTask,
    TrialFlag,
    TrialRecord,
)
from fake_server import (
    FakeServer,
    RecordedRequest,
    Script,
    http_error_response,
    text_response,
    thinking_response,
    tool_use_response,
)

RUN_A = "20260920-000000-aaaaaa"
RUN_B = "20260920-111111-bbbbbb"
AT = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)

_NEEDLE_LENGTH = 800
"""探す課題の長さ (8k などの代わりに使う、試験用の小さな長さ)。"""

_NEEDLE_PREFIX = "quality/needle/"
_IMAGE_REF = "sha256:" + "ab" * 32
_WRONG_ANSWER = "SIGIL-0BADC0DE"
"""探す課題の、正解でない答え (`SIGIL-` + 16 進 8 桁の書式は同じ)。"""

_MARKUP_TEXT = "<tool_call>read_file</tool_call> と書いてしまった応答"
"""`TargetDef.tool_markup_markers` の既定の目印を含む本文 (MARKUP_LEAKED になる)。"""

_CUSTOM_MARKER = "[[call]]"
"""対象サーバーの定義で上書きする、記法の目印 (既定の目印とは重ならない)。"""

DATASET: DatasetRef = humaneval.dataset_ref()


# --- 助け (test_suites_agent.py から、この試験に要る分だけを写す) -----------


class BodySink:
    """`RunStore.put_body` の代わり。保存された本文をそのまま覚えておく。"""

    def __init__(self) -> None:
        self.bodies: list[dict[str, JsonValue]] = []

    def put(self, body: dict[str, JsonValue]) -> str:
        self.bodies.append(body)
        canonical = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def quality_settings(**overrides: Any) -> dict[str, Any]:
    """`make_profile(quality=...)` に渡す、品質の検査の設定の写像。"""
    data: dict[str, Any] = {
        "toolcall_tasks": 3,
        "needle_lengths": [_NEEDLE_LENGTH],
        "needle_depths": [0, 50, 100],
        "trials_per_cell": 1,
        "code_max_tokens": 256,
    }
    data.update(overrides)
    return data


def make_profile(**overrides: Any) -> Profile:
    """試験用の小さな設定。"""
    data: dict[str, Any] = {
        "name": "test",
        "seed": 7,
        "sampling": {"temperature": 0.0},
        "timeout": {"connect_s": 5.0, "first_event_s": 5.0, "idle_s": 5.0, "total_s": 20.0},
        "length_tolerance": 0.05,
        "min_successes": 1,
        "chars_per_token": {"prose_en": 3.7, "prose_ja": 3.7, "code": 3.7, "log": 3.7},
        "sandbox": {"timeout_s": 7.0},
        "quality": quality_settings(),
    }
    data.update(overrides)
    return Profile.model_validate(data)


def one_of_each(**overrides: Any) -> Profile:
    """1 条件あたり 1 試行だけの設定 (1 種類の条件を取り出して流す試験のため)。"""
    data: dict[str, Any] = {"toolcall_tasks": 1, "needle_depths": [50]}
    data.update(overrides)
    return make_profile(quality=quality_settings(**data))


def target_for(server: FakeServer, **overrides: Any) -> TargetDef:
    payload: dict[str, Any] = {"name": "fake", "base_url": server.base_url, "model": "fake-model"}
    payload.update(overrides)
    return TargetDef.model_validate(payload)


@asynccontextmanager
async def suite_ctx(
    server: FakeServer,
    *,
    profile: Profile | None = None,
    target: TargetDef | None = None,
    context_limit: int | None = None,
    run_id: str = RUN_A,
    sink: BodySink | None = None,
) -> AsyncIterator[SuiteContext]:
    """偽のサーバーにつないだ `SuiteContext`。"""
    async with HttpxMessagesClient(server.base_url) as client:
        yield make_suite_context(
            client=client,
            profile=profile if profile is not None else make_profile(),
            target=target if target is not None else target_for(server),
            run_id=run_id,
            put_body=(sink if sink is not None else BodySink()).put,
            context_limit=context_limit,
        )


async def run_suite(
    suite: QualitySuite,
    ctx: SuiteContext,
    *,
    store: RunStore | None = None,
    prefix: str = "",
) -> tuple[list[TrialRecord], list[SkippedCondition]]:
    """計測ランの進行 (3.5) と同じ順で、まとまりを流す。

    `prefix` を渡すと、その鍵で始まる条件だけを流す (1 種類の条件の試験のため)。
    """
    records: list[TrialRecord] = []
    skipped: list[SkippedCondition] = []
    for item in suite.plan(ctx):
        if not item.key.startswith(prefix):
            continue
        if isinstance(item, SkippedCondition):
            skipped.append(item)
            continue
        try:
            async for record in suite.run_condition(ctx, item):
                records.append(record)
                if store is not None:
                    store.append_trial(record)
        except ConditionAborted as exc:
            skipped.append(exc.skipped)
    return records, skipped


def planned_condition(suite: QualitySuite, ctx: SuiteContext, key: str) -> ConditionPlan:
    plan = next(item for item in suite.plan(ctx) if item.key == key)
    assert isinstance(plan, ConditionPlan)
    return plan


async def run_one(suite: QualitySuite, ctx: SuiteContext, key: str) -> list[TrialRecord]:
    """1 つの条件だけを流す。"""
    cond = planned_condition(suite, ctx, key)
    return [record async for record in suite.run_condition(ctx, cond)]


def sent_requests(server: FakeServer) -> list[RecordedRequest]:
    return server.requests_for("/v1/messages")


def last_user_text(body: dict[str, Any]) -> str:
    messages = body["messages"]
    assert isinstance(messages, list)
    last = messages[-1]
    assert last["role"] == "user"
    content = last["content"]
    assert isinstance(content, list) and len(content) == 1
    text = content[0]["text"]
    assert isinstance(text, str)
    return text


def sent_texts(server: FakeServer) -> list[str]:
    texts: list[str] = []
    for request in sent_requests(server):
        assert request.body is not None
        texts.append(last_user_text(dict(request.body)))
    return texts


def make_manifest(profile: Profile, **overrides: Any) -> RunManifest:
    payload: dict[str, Any] = {
        "run_id": RUN_A,
        "status": RunStatus.RUNNING,
        "target": {"name": "fake", "base_url": "http://127.0.0.1:1", "model": "fake-model"},
        "server_model": "fake-model",
        "started_at": AT,
        "suites": [SuiteName.QUALITY],
        "profile_name": profile.name,
        "profile": profile,
        "harness_version": "0.1.0+gtest",
    }
    payload.update(overrides)
    return RunManifest.model_validate(payload)


# --- 隔離の実行環境の代わり ---------------------------------------------------


_SENTINEL_RE = re.compile(re.escape(SENTINEL_PREFIX) + r"[0-9a-f]+__")


def default_decide(source: str) -> SandboxResult:
    """組み立てたプログラムの中身を見て、合格・不合格を決める。

    `999` を含む解答 (試験が「誤った関数」として返すもの) だけを落とす。合格の
    ときは、実物のコンテナと同じように、印 (`build_checked_program` が末尾に
    足したもの) を標準エラーの最後の行として返す。
    """
    passed = "999" not in source
    if not passed:
        return SandboxResult(
            passed=False, timed_out=False, exit_code=1, stderr_tail="AssertionError\n"
        )
    found = _SENTINEL_RE.search(source)
    assert found is not None, "印つきのプログラムを組み立てていない"
    return SandboxResult(
        passed=True, timed_out=False, exit_code=0, stderr_tail=f"\n{found.group(0)}\n"
    )


class FakeSandbox:
    """`ContainerSandbox` の代わり。Docker にもネットワークにも触れない。"""

    def __init__(
        self,
        decide: Callable[[str], SandboxResult] | None = None,
        *,
        unavailable: SandboxUnavailable | None = None,
        image_ref: str | None = _IMAGE_REF,
    ) -> None:
        self._decide = decide if decide is not None else default_decide
        self._unavailable = unavailable
        self._image_ref = image_ref
        self.sources: list[str] = []
        self.timeouts: list[float] = []
        self.thread_ids: list[int] = []

    def available(self) -> bool | SandboxUnavailable:
        return True if self._unavailable is None else self._unavailable

    def image_ref(self) -> str | None:
        return None if self._unavailable is not None else self._image_ref

    def run_python(self, source: str, timeout_s: float) -> SandboxResult:
        self.sources.append(source)
        self.timeouts.append(timeout_s)
        self.thread_ids.append(threading.get_ident())
        return self._decide(source)


# --- コードの課題 (手で書く。6.5 の取得を待たない) ----------------------------


def code_problem(task_id: str, entry_point: str) -> CodeProblem:
    return CodeProblem(
        task_id=task_id,
        prompt=f'def {entry_point}(x: int) -> int:\n    """Return x plus one."""\n',
        entry_point=entry_point,
        test="def check(candidate):\n    assert candidate(1) == 2\n",
        canonical_solution="    return x + 1\n",
    )


PROBLEMS: tuple[CodeProblem, ...] = (
    code_problem("HumanEval/0", "add_one"),
    code_problem("HumanEval/1", "add_two"),
    code_problem("HumanEval/2", "add_three"),
)


def loader_for(
    problems: Sequence[CodeProblem] = PROBLEMS, ref: DatasetRef = DATASET
) -> ProblemsLoader:
    def load() -> tuple[DatasetRef, list[CodeProblem]]:
        return ref, list(problems)

    return load


def failing_loader(error: Exception) -> ProblemsLoader:
    def load() -> tuple[DatasetRef, list[CodeProblem]]:
        raise error

    return load


def make_suite(
    *, sandbox: FakeSandbox | None = None, loader: ProblemsLoader | None = None
) -> QualitySuite:
    box = sandbox if sandbox is not None else FakeSandbox()

    def factory(_settings: SandboxSettings) -> FakeSandbox:
        return box

    return QualitySuite(
        sandbox_factory=factory, problems_loader=loader if loader is not None else loader_for()
    )


# --- 偽のサーバーの応答の作り方 ----------------------------------------------


def tool_tasks(profile: Profile) -> dict[str, ToolTask]:
    """この計測ランで送られるツールの課題を、指示の文から引ける表にする。"""
    table: dict[str, ToolTask] = {}
    for index in range(profile.quality.toolcall_tasks):
        task = make_tool_task(TOOLCALL_TASK_INDEX_BASE + index, profile.seed)
        table[task.prompt] = task
    return table


def needle_cases(profile: Profile) -> list[NeedleCase]:
    """この計測ランで送られる探す課題 (長さ → 位置 → 試行の順)。"""
    settings = profile.quality
    ratio = profile.chars_per_token[ContentKind.PROSE_EN]
    return [
        make_needle_case(
            target_tokens=length,
            depth_pct=depth,
            index=index,
            seed=profile.seed,
            chars_per_token=ratio,
        )
        for length in sorted(set(settings.needle_lengths))
        for depth in sorted(set(settings.needle_depths))
        for index in range(settings.trials_per_cell)
    ]


def wrong_call_for(task: ToolTask, seed: int) -> ToolTask:
    """正解と違うツールを呼ぶ課題 (引数はそのツールの定義に合う → WRONG_CALL)。"""
    for index in range(900_000, 900_100):
        other = make_tool_task(index, seed)
        if other.expected_tool != task.expected_tool:
            return other
    raise AssertionError("違うツールの課題が見つからない")


def _toolcall_script(task: ToolTask, kind: str, seed: int) -> Script:
    if kind == "correct":
        return tool_use_response(task.expected_tool, task.expected_input)
    if kind == "wrong":
        other = wrong_call_for(task, seed)
        return tool_use_response(other.expected_tool, other.expected_input)
    if kind == "markup":
        return text_response(_MARKUP_TEXT)
    if kind == "custom_markup":
        return text_response(f"{_CUSTOM_MARKER} read_file")
    return http_error_response(500)


def _needle_script(case: NeedleCase, kind: str) -> Script:
    if kind == "correct":
        return text_response(f"The access code is {case.answer}.")
    if kind == "wrong":
        return text_response(f"The access code is {_WRONG_ANSWER}.")
    if kind == "thinking_only":
        # 考えの中には正解があるが、本文にはない。ブロックはそのままつながれる
        # ので、正解のうしろに空白を置いて、語の切れ目の守り (scoring/needle) に
        # 引っかからないようにする (引っかかると、この試験が意味を失う)
        return thinking_response(
            f"I recall {case.answer} from the document.", answer="I could not find it."
        )
    return http_error_response(500)


def _code_answer(entry_point: str, *, correct: bool) -> str:
    body = "    return x + 1" if correct else "    return x + 999"
    return f"```python\ndef {entry_point}(x: int) -> int:\n{body}\n```"


def _code_script(problem: CodeProblem, kind: str) -> Script:
    if kind == "correct":
        return text_response(_code_answer(problem.entry_point, correct=True))
    if kind == "wrong":
        return text_response(_code_answer(problem.entry_point, correct=False))
    if kind == "prose":
        return text_response("I would use a loop for this, but here is no code.")
    if kind == "thinking_only":
        return thinking_response(
            _code_answer(problem.entry_point, correct=True),
            answer="I am not going to write it out.",
        )
    return http_error_response(500)


def make_responder(
    profile: Profile,
    *,
    toolcall: Sequence[str] = ("correct",),
    needle: Sequence[str] = ("correct",),
    code: Sequence[str] = ("correct",),
    problems: Sequence[CodeProblem] = PROBLEMS,
) -> Callable[[dict[str, Any]], Script]:
    """指示の文から、どの課題への応答かを見分けて台本を返す。

    どの課題にどの種類を返すかは、課題の番号 (= 表に入れた順) で決まるので、
    要求の届く順には依存しない。
    """
    tasks = tool_tasks(profile)
    task_order = {prompt: index for index, prompt in enumerate(tasks)}
    # 位置 (depth) が違っても、正解・コード名・問いは同じ (corpus/needle.py の
    # 決めごと) なので、干し草を含めた本文の全体で引く
    cases = {
        f"{case.haystack}\n\n{case.question}": (index, case)
        for index, case in enumerate(needle_cases(profile))
    }
    seed = profile.seed

    def factory(body: dict[str, Any]) -> Script:
        text = last_user_text(body)
        task = tasks.get(text)
        if task is not None:
            return _toolcall_script(task, toolcall[task_order[text] % len(toolcall)], seed)
        found = cases.get(text)
        if found is not None:
            index, case = found
            return _needle_script(case, needle[index % len(needle)])
        for index, problem in enumerate(problems):
            if f"def {problem.entry_point}(" in text:
                return _code_script(problem, code[index % len(code)])
        raise AssertionError(f"どの課題の指示か分からない: {text[:80]!r}")

    return factory


def align_needle_token_counting(server: FakeServer, ctx: SuiteContext, length: int) -> float:
    """偽のサーバーの数え方を、探す課題の狙いの長さにそろえる。

    偽のサーバーは `system` / `messages` / `tools` の JSON の**文字数**を
    `chars_per_token` で割る。干し草は本文の文字数だけを見積もるので、JSON の
    記号のぶんだけ食い違う (`fake_server.py`「入力のトークン数の決め方」)。
    """
    case = make_needle_case(
        target_tokens=length,
        depth_pct=sorted(set(ctx.profile.quality.needle_depths))[0],
        index=0,
        seed=ctx.profile.seed,
        chars_per_token=ctx.profile.chars_per_token[ContentKind.PROSE_EN],
    )
    payload = {
        "messages": [
            {
                "content": [{"text": f"{case.haystack}\n\n{case.question}", "type": "text"}],
                "role": "user",
            }
        ]
    }
    ratio = len(json.dumps(payload, ensure_ascii=False, sort_keys=True)) / length
    server.set_chars_per_token(ratio)
    return ratio


# --- 計画 (5.1、5.2、5.3、design.md suites の条件の鍵) -----------------------


async def test_plan_makes_the_three_kinds_of_conditions_in_the_design_order(
    fake_server: FakeServer,
) -> None:
    suite = make_suite()
    async with suite_ctx(fake_server) as ctx:
        planned = suite.plan(ctx)

    assert [item.key for item in planned] == [
        "quality/toolcall",
        "quality/code/humaneval+",
        "quality/needle/800/d0",
        "quality/needle/800/d50",
        "quality/needle/800/d100",
    ]
    assert TOOLCALL_CONDITION_KEY == "quality/toolcall"
    assert CODE_CONDITION_KEY == "quality/code/humaneval+"
    assert needle_condition_key(8000, 25) == "quality/needle/8k/d25"
    assert all(isinstance(item, ConditionPlan) for item in planned)
    # 計画を作っただけでは、要求を 1 つも送らない
    assert sent_requests(fake_server) == []


async def test_plan_uses_the_profile_counts_and_no_warmup(fake_server: FakeServer) -> None:
    suite = make_suite()
    async with suite_ctx(fake_server) as ctx:
        planned = [item for item in suite.plan(ctx) if isinstance(item, ConditionPlan)]

    by_key = {item.key: item for item in planned}
    assert by_key[TOOLCALL_CONDITION_KEY].trials == 3
    assert by_key[CODE_CONDITION_KEY].trials == len(PROBLEMS)
    assert by_key[CODE_CONDITION_KEY].max_tokens == 256
    assert by_key["quality/needle/800/d0"].trials == 1
    assert by_key["quality/needle/800/d0"].target_input_tokens == _NEEDLE_LENGTH
    assert by_key["quality/needle/800/d0"].max_tokens == NEEDLE_MAX_TOKENS
    assert by_key[TOOLCALL_CONDITION_KEY].target_input_tokens is None
    assert all(item.warmup_trials == 0 for item in planned)


def test_toolcall_task_indices_cannot_collide_with_the_agent_suite() -> None:
    """課題の番号の住み分け: agent は 0〜(段階の数 × 10,000)、履歴は負の数。"""
    from bench_harness.suites import agent as agent_module

    assert TOOLCALL_TASK_INDEX_BASE >= 100 * agent_module._TASK_INDEX_STRIDE


# --- 3 種類の条件が、正解・不正解・採点できなかった、を分けて記録する --------


async def test_the_three_kinds_record_correct_incorrect_and_not_scored_separately(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """tasks.md 6.7 の完了の状態 (要約の割合の行まで通ることも確かめる)。"""
    profile = make_profile()
    sandbox = FakeSandbox()
    suite = make_suite(sandbox=sandbox)
    sink = BodySink()
    store = RunStore.create(tmp_path / "results", make_manifest(profile))
    fake_server.set_response_factory(
        make_responder(
            profile,
            toolcall=("correct", "wrong", "http_500"),
            code=("correct", "wrong", "http_500"),
            needle=("correct", "wrong", "http_500"),
        )
    )

    async with suite_ctx(fake_server, profile=profile, sink=sink) as ctx:
        align_needle_token_counting(fake_server, ctx, _NEEDLE_LENGTH)
        records, skipped = await run_suite(suite, ctx, store=store)
    datasets = [dataset.model_dump(mode="json") for dataset in suite.run_info().datasets]
    store.update_manifest(datasets=datasets)
    store.set_status(RunStatus.COMPLETED, AT)

    assert skipped == []
    # 送った要求 == 残ったレコード == 保存した本文
    assert len(records) == len(sent_requests(fake_server)) == len(sink.bodies) == 9

    conditions = {record.condition for record in records}
    assert conditions == {
        TOOLCALL_CONDITION_KEY,
        CODE_CONDITION_KEY,
        "quality/needle/800/d0",
        "quality/needle/800/d50",
        "quality/needle/800/d100",
    }

    toolcall = [record.verdict for record in records if record.condition == TOOLCALL_CONDITION_KEY]
    assert [verdict.outcome for verdict in toolcall if isinstance(verdict, ToolCallVerdict)] == [
        ToolCallOutcome.CORRECT,
        ToolCallOutcome.WRONG_CALL,
        ToolCallOutcome.REQUEST_FAILED,
    ]

    code = [record.verdict for record in records if record.condition == CODE_CONDITION_KEY]
    assert [verdict.outcome for verdict in code if isinstance(verdict, QualityVerdict)] == [
        QualityOutcome.CORRECT,
        QualityOutcome.INCORRECT,
        QualityOutcome.NOT_SCORED,
    ]
    assert all(isinstance(verdict, QualityVerdict) and verdict.task == "code" for verdict in code)
    # どの問題の結果かが、レコードから追える
    assert all(
        isinstance(verdict, QualityVerdict) and problem.task_id in verdict.detail
        for verdict, problem in zip(code, PROBLEMS, strict=True)
    )

    needles = [record.verdict for record in records if record.condition.startswith(_NEEDLE_PREFIX)]
    assert [verdict.outcome for verdict in needles if isinstance(verdict, QualityVerdict)] == [
        QualityOutcome.CORRECT,
        QualityOutcome.INCORRECT,
        QualityOutcome.NOT_SCORED,
    ]

    result = summarize_run(store.run_dir)
    assert result.warnings == []
    rows = {row.condition: row for row in result.summary.results if row.metric == METRIC_ACCURACY}
    assert set(rows) == conditions

    toolcall_row = rows[TOOLCALL_CONDITION_KEY]
    assert toolcall_row.proportion is not None
    assert (toolcall_row.proportion.numerator, toolcall_row.proportion.denominator) == (1, 2)
    assert toolcall_row.flag_counts[NOT_SCORED_COUNT] == 1
    assert toolcall_row.failures == 1

    code_row = rows[CODE_CONDITION_KEY]
    assert code_row.proportion is not None
    assert (code_row.proportion.numerator, code_row.proportion.denominator) == (1, 2)
    assert code_row.flag_counts[NOT_SCORED_COUNT] == 1

    d0 = rows["quality/needle/800/d0"].proportion
    d50 = rows["quality/needle/800/d50"].proportion
    assert d0 is not None and (d0.numerator, d0.denominator) == (1, 1)
    assert d50 is not None and (d50.numerator, d50.denominator) == (0, 1)
    assert rows["quality/needle/800/d100"].proportion is None
    assert rows["quality/needle/800/d100"].flag_counts[NOT_SCORED_COUNT] == 1

    # 公開の課題の名前・版・ライセンスが、要約に出る (5.5)
    assert [dataset.name for dataset in result.summary.datasets] == [DATASET.name]
    assert result.summary.datasets[0].license == DATASET.license
    assert result.summary.datasets[0].version == DATASET.version

    # 隔離を動かしたのは、要求が成功した 2 問だけ。制限時間は設定の値
    assert len(sandbox.sources) == 2
    assert sandbox.timeouts == [7.0, 7.0]


async def test_normal_trials_carry_no_flags_but_broken_output_is_still_flagged(
    fake_server: FakeServer,
) -> None:
    """ふつうの試行に余計な印が付かず、壊れた出力には印が付く (10.7)。"""
    profile = one_of_each()
    suite = make_suite()
    fake_server.set_response_factory(make_responder(profile))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        align_needle_token_counting(fake_server, ctx, _NEEDLE_LENGTH)
        records, skipped = await run_suite(suite, ctx)

    assert skipped == []
    assert len(records) == 1 + len(PROBLEMS) + 1
    assert [record.flags for record in records] == [[] for _ in records]

    # 置き換え文字を含む応答には、印が付く
    fake_server.reset()
    fake_server.set_response(text_response("答えは SIGIL� です"))
    suite2 = make_suite()
    async with suite_ctx(fake_server, profile=profile) as ctx:
        align_needle_token_counting(fake_server, ctx, _NEEDLE_LENGTH)
        flagged = await run_one(suite2, ctx, needle_condition_key(_NEEDLE_LENGTH, 50))
    assert flagged[0].flags == [TrialFlag.REPLACEMENT_CHAR]


async def test_a_failed_request_is_not_scored_for_every_kind(fake_server: FakeServer) -> None:
    """要求が失敗した課題は「採点できなかった」。不正解には数えない (5.7)。"""
    profile = one_of_each()
    sandbox = FakeSandbox()
    suite = make_suite(sandbox=sandbox)
    fake_server.set_response(http_error_response(500))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        records, skipped = await run_suite(suite, ctx)

    assert skipped == []
    by_condition: dict[str, list[TrialRecord]] = {}
    for record in records:
        by_condition.setdefault(record.condition, []).append(record)

    toolcall = by_condition[TOOLCALL_CONDITION_KEY][0].verdict
    assert isinstance(toolcall, ToolCallVerdict)
    assert toolcall.outcome is ToolCallOutcome.REQUEST_FAILED
    for key in (CODE_CONDITION_KEY, needle_condition_key(_NEEDLE_LENGTH, 50)):
        for record in by_condition[key]:
            verdict = record.verdict
            assert isinstance(verdict, QualityVerdict)
            assert verdict.outcome is QualityOutcome.NOT_SCORED
    # 失敗した要求では、隔離を動かさない
    assert sandbox.sources == []


async def test_a_sandbox_error_is_not_scored_while_a_wrong_answer_is_incorrect(
    fake_server: FakeServer,
) -> None:
    """設備の失敗は `NOT_SCORED`、モデルの誤りは `INCORRECT` (5.7)。"""
    profile = one_of_each()

    def decide(source: str) -> SandboxResult:
        if "add_one" in source:
            raise SandboxError("docker 自身が失敗した (終了コード 125)")
        return default_decide(source)

    suite = make_suite(sandbox=FakeSandbox(decide))
    fake_server.set_response_factory(make_responder(profile, code=("correct", "wrong", "correct")))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        records = await run_one(suite, ctx, CODE_CONDITION_KEY)

    assert [record.verdict.outcome for record in records if record.verdict is not None] == [
        QualityOutcome.NOT_SCORED,
        QualityOutcome.INCORRECT,
        QualityOutcome.CORRECT,
    ]


async def test_a_program_that_exits_before_the_check_is_incorrect(
    fake_server: FakeServer,
) -> None:
    """終了コードが 0 でも、検査を通った印がなければ合格にしない (`score_checked`)。"""
    profile = one_of_each()

    def decide(_source: str) -> SandboxResult:
        # `sys.exit(0)` で check(...) の手前で終わった答えの真似 (印が出ない)
        return SandboxResult(passed=True, timed_out=False, exit_code=0, stderr_tail="")

    suite = make_suite(sandbox=FakeSandbox(decide))
    fake_server.set_response_factory(make_responder(profile, code=("correct",)))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        records = await run_one(suite, ctx, CODE_CONDITION_KEY)

    assert all(
        isinstance(record.verdict, QualityVerdict)
        and record.verdict.outcome is QualityOutcome.INCORRECT
        for record in records
    )


async def test_a_response_without_code_is_incorrect_not_unscored(fake_server: FakeServer) -> None:
    """説明だけの応答は、モデルが答えた結果なので**不正解** (scoring/code.py)。"""
    profile = one_of_each()
    sandbox = FakeSandbox()
    suite = make_suite(sandbox=sandbox)
    fake_server.set_response_factory(make_responder(profile, code=("prose",)))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        records = await run_one(suite, ctx, CODE_CONDITION_KEY)

    assert all(
        isinstance(record.verdict, QualityVerdict)
        and record.verdict.outcome is QualityOutcome.INCORRECT
        for record in records
    )
    assert sandbox.sources == []


# --- 隔離が使えない / 課題が取れない (6.7 の飛ばし方) ------------------------


async def test_an_unavailable_sandbox_skips_only_the_code_condition(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """隔離が使えないときは、コードの課題だけを飛ばし、要求を 1 つも送らない。"""
    profile = one_of_each()
    reason = "docker の守護プロセスが応答しない"
    suite = make_suite(sandbox=FakeSandbox(unavailable=SandboxUnavailable(reason=reason)))
    store = RunStore.create(tmp_path / "results", make_manifest(profile))
    fake_server.set_response_factory(make_responder(profile))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        align_needle_token_counting(fake_server, ctx, _NEEDLE_LENGTH)
        records, skipped = await run_suite(suite, ctx, store=store)
    store.update_manifest(skipped=[item.model_dump(mode="json") for item in skipped])
    store.set_status(RunStatus.COMPLETED, AT)
    write_summary(store.run_dir)

    assert [item.key for item in skipped] == [CODE_CONDITION_KEY]
    assert reason in skipped[0].reason
    assert [record.condition for record in records] == [
        TOOLCALL_CONDITION_KEY,
        needle_condition_key(_NEEDLE_LENGTH, 50),
    ]
    # コードの課題の要求は 1 つも送っていない
    assert not any("def add_" in text for text in sent_texts(fake_server))
    assert suite.run_info().datasets == ()

    markdown = (store.run_dir / "summary.md").read_text(encoding="utf-8")
    assert CODE_CONDITION_KEY in markdown
    assert reason in markdown


async def test_an_unavailable_dataset_skips_only_the_code_condition(
    fake_server: FakeServer,
) -> None:
    profile = one_of_each()
    suite = make_suite(
        loader=failing_loader(humaneval.DatasetUnavailable("ネットワークに届かない"))
    )
    fake_server.set_response_factory(make_responder(profile))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        align_needle_token_counting(fake_server, ctx, _NEEDLE_LENGTH)
        records, skipped = await run_suite(suite, ctx)

    assert [item.key for item in skipped] == [CODE_CONDITION_KEY]
    assert "ネットワークに届かない" in skipped[0].reason
    assert len(records) == 2
    assert not any("def add_" in text for text in sent_texts(fake_server))


@pytest.mark.parametrize(
    "error",
    [
        humaneval.DatasetVerificationError("SHA-256 が合わない"),
        humaneval.DatasetFormatError("中身が期待の形でない"),
        humaneval.DatasetCacheError("置き場所が git の管理の対象"),
    ],
)
async def test_a_broken_dataset_is_not_skipped_but_propagates(
    fake_server: FakeServer, error: Exception
) -> None:
    """壊れと取り違えは、黙って飛ばさない (note 6.5 → 6.7)。"""
    suite = make_suite(loader=failing_loader(error))
    async with suite_ctx(fake_server) as ctx:
        with pytest.raises(type(error)):
            suite.plan(ctx)


async def test_select_problems_is_applied_so_the_unscorable_problem_is_never_sent(
    fake_server: FakeServer,
) -> None:
    """`select_problems` を通す (採点できない問題を外し、限りの数だけ採る)。"""
    unscorable_id = next(iter(humaneval.UNSCORABLE_PROBLEMS))
    problems = (code_problem(unscorable_id, "broken_one"), *PROBLEMS)
    profile = one_of_each(code_problem_limit=2)
    suite = make_suite(loader=loader_for(problems))
    fake_server.set_response_factory(make_responder(profile, problems=problems))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        records = await run_one(suite, ctx, CODE_CONDITION_KEY)

    assert len(records) == 2
    texts = sent_texts(fake_server)
    assert not any("broken_one" in text for text in texts)
    assert [text.count("def add_") for text in texts] == [1, 1]


# --- 隔離の失敗が続いたら、条件を打ち切る ------------------------------------


async def test_consecutive_sandbox_errors_abort_the_condition_and_keep_earlier_records(
    fake_server: FakeServer,
) -> None:
    """設備が壊れたまま最後まで「採点できなかった」を並べない (ブレーカー)。"""
    problems = tuple(
        code_problem(f"HumanEval/{index}", f"add_{index}")
        for index in range(MAX_CONSECUTIVE_SANDBOX_ERRORS + 3)
    )
    profile = one_of_each()

    def decide(source: str) -> SandboxResult:
        if "def add_0(" in source:
            return default_decide(source)
        raise SandboxError("docker 自身が失敗した (終了コード 125)")

    suite = make_suite(sandbox=FakeSandbox(decide), loader=loader_for(problems))
    fake_server.set_response_factory(make_responder(profile, problems=problems))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = planned_condition(suite, ctx, CODE_CONDITION_KEY)
        records: list[TrialRecord] = []
        with pytest.raises(ConditionAborted) as raised:
            async for record in suite.run_condition(ctx, cond):
                records.append(record)

    assert len(records) == 1 + MAX_CONSECUTIVE_SANDBOX_ERRORS
    first = records[0].verdict
    assert isinstance(first, QualityVerdict) and first.outcome is QualityOutcome.CORRECT
    assert all(
        isinstance(record.verdict, QualityVerdict)
        and record.verdict.outcome is QualityOutcome.NOT_SCORED
        for record in records[1:]
    )
    assert raised.value.skipped.key == CODE_CONDITION_KEY
    assert str(MAX_CONSECUTIVE_SANDBOX_ERRORS) in raised.value.skipped.reason
    # 打ち切ったあとの問題には、要求を送っていない
    assert len(sent_requests(fake_server)) == len(records)


# --- 隔離を動かす糸 (event loop を塞がない) ----------------------------------


async def test_the_sandbox_runs_off_the_event_loop_thread(fake_server: FakeServer) -> None:
    """`run_python` は同期で最大 `timeout_s` かかる。event loop の糸では動かさない。"""
    profile = one_of_each()
    sandbox = FakeSandbox()
    suite = make_suite(sandbox=sandbox)
    fake_server.set_response_factory(make_responder(profile))

    loop_thread = threading.get_ident()
    async with suite_ctx(fake_server, profile=profile) as ctx:
        await run_one(suite, ctx, CODE_CONDITION_KEY)

    assert sandbox.thread_ids
    assert all(thread_id != loop_thread for thread_id in sandbox.thread_ids)


async def test_cancelling_mid_condition_keeps_records_and_waits_for_the_sandbox(
    fake_server: FakeServer,
) -> None:
    """中断しても、動かしている隔離を放り出さない (コンテナを残さない)。"""
    profile = one_of_each()
    profile = make_profile(quality=profile.quality.model_dump(), sandbox={"timeout_s": 1.0})
    started = threading.Event()
    finished = threading.Event()

    def decide(source: str) -> SandboxResult:
        if "def add_two(" in source:
            started.set()
            time.sleep(0.5)
            finished.set()
        return default_decide(source)

    suite = make_suite(sandbox=FakeSandbox(decide))
    fake_server.set_response_factory(make_responder(profile))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        cond = planned_condition(suite, ctx, CODE_CONDITION_KEY)
        records: list[TrialRecord] = []

        async def drive() -> None:
            async for record in suite.run_condition(ctx, cond):
                records.append(record)

        task = asyncio.create_task(drive())
        await asyncio.to_thread(started.wait, 5.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert finished.is_set()
    assert len(records) == 1
    assert records[0].condition == CODE_CONDITION_KEY


# --- 探す課題 (5.3) -----------------------------------------------------------


async def test_needle_requests_carry_the_calibrated_haystack_and_question(
    fake_server: FakeServer,
) -> None:
    """較正した比を渡す (渡し忘れると、干し草が変わって落ちる)。"""
    profile = one_of_each(needle_depths=[25])
    suite = make_suite()
    fake_server.set_response(text_response("ok"))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        await run_one(suite, ctx, needle_condition_key(_NEEDLE_LENGTH, 25))

    case = make_needle_case(
        target_tokens=_NEEDLE_LENGTH,
        depth_pct=25,
        index=0,
        seed=profile.seed,
        chars_per_token=profile.chars_per_token[ContentKind.PROSE_EN],
    )
    assert sent_texts(fake_server) == [f"{case.haystack}\n\n{case.question}"]
    body = sent_requests(fake_server)[0].body
    assert body is not None
    # 探す課題は前置きを付けない (先頭の識別子も、システムプロンプトもない)
    assert "system" not in body
    assert "tools" not in body


async def test_the_needle_verdict_reads_text_blocks_not_thinking(fake_server: FakeServer) -> None:
    """採点は本文 (`text`) から。thinking に正解があっても、正解にしない。"""
    profile = one_of_each()
    suite = make_suite()
    fake_server.set_response_factory(make_responder(profile, needle=("thinking_only",)))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        records = await run_one(suite, ctx, needle_condition_key(_NEEDLE_LENGTH, 50))

    verdict = records[0].verdict
    assert isinstance(verdict, QualityVerdict)
    assert verdict.outcome is QualityOutcome.INCORRECT


async def test_the_code_verdict_reads_text_blocks_not_thinking(fake_server: FakeServer) -> None:
    """コードの取り出しも本文から (thinking の中のコードは採らない)。"""
    profile = one_of_each()
    sandbox = FakeSandbox()
    suite = make_suite(sandbox=sandbox)
    fake_server.set_response_factory(make_responder(profile, code=("thinking_only",)))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        records = await run_one(suite, ctx, CODE_CONDITION_KEY)

    assert sandbox.sources == []
    assert all(
        isinstance(record.verdict, QualityVerdict)
        and record.verdict.outcome is QualityOutcome.INCORRECT
        for record in records
    )


async def test_needle_conditions_longer_than_the_limit_are_skipped_before_sending(
    fake_server: FakeServer,
) -> None:
    """上限が分かっているときは、送る前に飛ばす (3.6)。"""
    profile = one_of_each(needle_lengths=[_NEEDLE_LENGTH, 4000])
    suite = make_suite()
    fake_server.set_response_factory(make_responder(profile))

    async with suite_ctx(fake_server, profile=profile, context_limit=2000) as ctx:
        align_needle_token_counting(fake_server, ctx, _NEEDLE_LENGTH)
        records, skipped = await run_suite(suite, ctx, prefix=_NEEDLE_PREFIX)

    assert [item.key for item in skipped] == ["quality/needle/4k/d50"]
    assert "2000" in skipped[0].reason
    assert [record.condition for record in records] == [needle_condition_key(_NEEDLE_LENGTH, 50)]


async def test_a_rejected_needle_request_stops_the_longer_lengths(fake_server: FakeServer) -> None:
    """上限が分からないときは、送ってみて 400 で止める。後ろの長さには送らない (6.9)。"""
    profile = one_of_each(needle_lengths=[_NEEDLE_LENGTH, 1600, 2400])
    suite = make_suite()
    fake_server.set_response_factory(make_responder(profile))

    async with suite_ctx(fake_server, profile=profile, context_limit=None) as ctx:
        align_needle_token_counting(fake_server, ctx, _NEEDLE_LENGTH)
        fake_server.set_context_limit(1400 + NEEDLE_MAX_TOKENS, advertise=False)
        records, skipped = await run_suite(suite, ctx, prefix=_NEEDLE_PREFIX)

    assert [record.condition for record in records] == [
        needle_condition_key(_NEEDLE_LENGTH, 50),
        needle_condition_key(1600, 50),
    ]
    assert [item.key for item in skipped] == [
        needle_condition_key(1600, 50),
        needle_condition_key(2400, 50),
    ]
    assert "1600" in skipped[1].reason
    assert len(sent_requests(fake_server)) == 2


# --- ツール呼び出しの課題 (5.1) ----------------------------------------------


async def test_toolcall_requests_have_no_preamble_and_carry_the_catalog(
    fake_server: FakeServer,
) -> None:
    """前置きの会話なし (tasks.md 6.7)。目録は課題の `tools` をそのまま渡す。"""
    profile = one_of_each(toolcall_tasks=2)
    suite = make_suite()
    fake_server.set_response_factory(make_responder(profile))

    async with suite_ctx(fake_server, profile=profile) as ctx:
        await run_one(suite, ctx, TOOLCALL_CONDITION_KEY)

    tasks = list(tool_tasks(profile).values())
    assert sent_texts(fake_server) == [task.prompt for task in tasks]
    body = sent_requests(fake_server)[0].body
    assert body is not None
    assert "system" not in body
    tools = body["tools"]
    assert isinstance(tools, list)
    assert [tool["name"] for tool in tools] == [tool.name for tool in tasks[0].tools]


async def test_the_markup_markers_come_from_the_target_definition(
    fake_server: FakeServer,
) -> None:
    """記法の目印は、対象サーバーの定義から渡す (6.3)。"""
    profile = one_of_each()
    suite = make_suite()
    fake_server.set_response_factory(make_responder(profile, toolcall=("custom_markup",)))
    target = target_for(fake_server, tool_markup_markers=[_CUSTOM_MARKER])

    async with suite_ctx(fake_server, profile=profile, target=target) as ctx:
        records = await run_one(suite, ctx, TOOLCALL_CONDITION_KEY)

    verdict = records[0].verdict
    assert isinstance(verdict, ToolCallVerdict)
    assert verdict.outcome is ToolCallOutcome.MARKUP_LEAKED


async def test_a_broken_tool_task_propagates(
    fake_server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ToolTaskError` (課題の側の誤り) は握りつぶさない (注 6.3)。"""
    profile = one_of_each()
    suite = make_suite()
    fake_server.set_response_factory(make_responder(profile))

    from bench_harness.suites import quality as quality_module

    def broken(index: int, seed: int) -> ToolTask:
        task = make_tool_task(index, seed)
        return ToolTask(
            prompt=task.prompt,
            tools=list(task.tools),
            expected_tool="この名前のツールは目録にない",
            expected_input={},
        )

    monkeypatch.setattr(quality_module, "make_tool_task", broken)

    async with suite_ctx(fake_server, profile=profile) as ctx:
        with pytest.raises(ToolTaskError):
            await run_one(suite, ctx, TOOLCALL_CONDITION_KEY)


# --- 決まった結果になること (11.4、9.2) --------------------------------------


async def test_the_same_seed_sends_byte_identical_bodies_across_runs(
    fake_server: FakeServer,
) -> None:
    """同じ種なら、計測ランをまたいでも同じ要求になる (対応のある比較の前提)。"""
    profile = make_profile()

    bodies: list[list[dict[str, JsonValue]]] = []
    for run_id in (RUN_A, RUN_B):
        fake_server.reset()
        fake_server.set_response_factory(make_responder(profile))
        sink = BodySink()
        suite = make_suite()
        async with suite_ctx(fake_server, profile=profile, run_id=run_id, sink=sink) as ctx:
            align_needle_token_counting(fake_server, ctx, _NEEDLE_LENGTH)
            await run_suite(suite, ctx)
        bodies.append(sink.bodies)

    assert bodies[0] == bodies[1]
    assert len(bodies[0]) == 9


async def test_run_info_carries_the_dataset_and_the_image_for_the_manifest(
    fake_server: FakeServer,
) -> None:
    """8.1 が manifest に書き写すもの (5.5、design.md「識別子を計測ランに記録する」)。"""
    suite = make_suite(sandbox=FakeSandbox(image_ref=_IMAGE_REF))
    async with suite_ctx(fake_server) as ctx:
        suite.plan(ctx)
        info = suite.run_info()

    assert [dataset.sha256 for dataset in info.datasets] == [DATASET.sha256]
    assert info.sandbox_image_ref == _IMAGE_REF


async def test_plan_forgets_the_previous_run(fake_server: FakeServer) -> None:
    """`plan()` は、計測ランごとの覚え書きを消す。**同じ実体**を使い回しても混ざらない。

    1 回目は課題が読めて、2 回目は読めない。覚え書きを消さないと、2 回目の計測
    ランが、使っていない公開の課題を「使った」と報せてしまう (5.7)。
    """
    calls = 0

    def loader() -> tuple[DatasetRef, list[CodeProblem]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return DATASET, list(PROBLEMS)
        raise humaneval.DatasetUnavailable("取れない")

    suite = make_suite(loader=loader)
    async with suite_ctx(fake_server, run_id=RUN_A) as ctx:
        first = suite.plan(ctx)
        assert suite.run_info().datasets != ()
    async with suite_ctx(fake_server, run_id=RUN_B) as ctx:
        second = suite.plan(ctx)
        assert suite.run_info().datasets == ()

    assert isinstance(first[1], ConditionPlan)
    assert isinstance(second[1], SkippedCondition)


async def test_a_plan_from_another_run_is_refused_by_the_code_condition(
    fake_server: FakeServer,
) -> None:
    """コードの条件は、同じ計測ランの `plan()` で確かめた隔離と課題しか使わない。"""
    suite = make_suite()
    async with suite_ctx(fake_server, run_id=RUN_A) as ctx:
        planned = suite.plan(ctx)
    code = planned[1]
    assert isinstance(code, ConditionPlan)

    async with suite_ctx(fake_server, run_id=RUN_B) as ctx:
        with pytest.raises(ValueError, match="plan"):
            _ = [record async for record in suite.run_condition(ctx, code)]
    assert fake_server.call_count("/v1/messages") == 0


async def test_a_limit_remembered_for_one_run_does_not_stop_the_next_run(
    fake_server: FakeServer,
) -> None:
    """上限に当たったことは、その計測ランだけの覚え書き。次の計測ランの探す課題を止めない。"""
    suite = make_suite()
    async with suite_ctx(fake_server, run_id=RUN_A) as ctx:
        suite.plan(ctx)
        suite._remember_limit(RUN_A, 1)  # どの長さも、この上限より長い
    async with suite_ctx(fake_server, run_id=RUN_B) as ctx:
        planned = [item for item in suite.plan(ctx) if isinstance(item, ConditionPlan)]
        needle = next(item for item in planned if item.key.startswith("quality/needle/"))
        # plan() を通さずに覚え書きだけが残っていても、計測ランが違えば効かない
        suite._remember_limit(RUN_A, 1)
        fake_server.set_response(text_response("no idea"))
        records = [record async for record in suite.run_condition(ctx, needle)]
    assert records, "前の計測ランの上限のせいで、1 つも送られなかった"
