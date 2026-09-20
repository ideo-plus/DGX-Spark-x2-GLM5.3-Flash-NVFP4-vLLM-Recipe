"""計測ランの進行の試験 (task 3.5: runner)。

要求が絡む試験は、実際のソケット越しに偽のサーバーを相手に行う。このファイル
が確かめるのは、前提が満たされないときに何も作らないこと、選ばれたまとまりが
順に走ること、レコードが 1 つずつ残ること、止め方 (連続の失敗、中断) と終了の
値であって、速さではない。時刻の値を判定する試験は 1 つもない (遅い応答を使う
のは「送っている途中」を作るためだけで、待った長さは判定しない)。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from bench_harness import runner as runner_module
from bench_harness.client.messages import HttpxMessagesClient
from bench_harness.runner import (
    EXIT_ABORTED,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_PRECONDITION,
    PreconditionError,
    ProgressSink,
    StderrProgressSink,
    SuiteRunInfo,
    SupportsRunInfo,
    default_suite_registry,
    execute_run,
    harness_version,
)
from bench_harness.store import RunStore, StoreError, list_run_dirs
from bench_harness.suites import agent as agent_module
from bench_harness.suites import quality as quality_module
from bench_harness.suites.agent import AgentSuite
from bench_harness.suites.base import (
    Suite,
    SuiteContext,
    abort_if_context_limit,
    condition_key,
    iter_trials,
    plan_condition,
    run_trial,
    single_user_message,
    skip_condition,
)
from bench_harness.suites.quality import QualitySuite
from bench_harness.types import (
    ConditionPlan,
    DatasetRef,
    MessagesRequest,
    QualityOutcome,
    QualityVerdict,
    RunOutcome,
    RunRequest,
    RunStatus,
    SkippedCondition,
    StreamResult,
    SuiteName,
    TimeoutPolicy,
    TrialRecord,
)
from fake_server import (
    FakeServer,
    Script,
    http_error_response,
    replace,
    stalled_response,
    text_response,
)

PROFILE_NAME = "test"
TARGET_NAME = "fake"
RUN_TASK_NAME = "bench-run"
"""runner が計測の本体に付ける task の名前 (中断のあとに残っていないことを見る)。"""


# --- 設定ファイルと置き場所の下ごしらえ -------------------------------------


@dataclass(frozen=True)
class Bed:
    """1 回の計測ランに要る、設定ファイルと生データの置き場所の一式。"""

    targets: Path
    profiles: Path
    results_root: Path


def write_targets(
    tmp_path: Path,
    base_url: str,
    *,
    api_key_env: str | None = None,
    max_context_tokens: int | None = None,
) -> Path:
    lines = [f"[targets.{TARGET_NAME}]", f'base_url = "{base_url}"', 'model = "fake-model"']
    if api_key_env is not None:
        lines.append(f'api_key_env = "{api_key_env}"')
    if max_context_tokens is not None:
        lines.append(f"max_context_tokens = {max_context_tokens}")
    path = tmp_path / "targets.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_profiles(
    tmp_path: Path,
    *,
    decode_trials: int = 10,
    decode_warmup: int = 0,
    max_consecutive_failures: int = 5,
    metrics_interval_s: float = 60.0,
    thinking: str = "server_default",
    concurrency_levels: Sequence[int] = (2,),
    concurrency_rounds: int = 1,
) -> Path:
    """試験用の計測の設定。制限時間は短く、出力の上限は小さくしてある。"""
    text = f"""
[profiles.{PROFILE_NAME}]
seed = 3
min_successes = 1
max_consecutive_failures = {max_consecutive_failures}
length_tolerance = 0.05
compare_tolerance = 0.02
metrics_interval_s = {metrics_interval_s}

[profiles.{PROFILE_NAME}.sampling]
temperature = 0.0
thinking = "{thinking}"

[profiles.{PROFILE_NAME}.timeout]
connect_s = 5.0
first_event_s = 5.0
idle_s = 5.0
total_s = 20.0

[profiles.{PROFILE_NAME}.decode]
trials = {decode_trials}
warmup_trials = {decode_warmup}
max_tokens = 32

[profiles.{PROFILE_NAME}.concurrency]
levels = [{", ".join(str(level) for level in concurrency_levels)}]
rounds = {concurrency_rounds}
max_tokens = 16
input_tokens = 400
"""
    path = tmp_path / "profiles.toml"
    path.write_text(text, encoding="utf-8")
    return path


def make_bed(
    tmp_path: Path,
    server: FakeServer | None = None,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
    max_context_tokens: int | None = None,
    decode_trials: int = 10,
    decode_warmup: int = 0,
    max_consecutive_failures: int = 5,
    metrics_interval_s: float = 60.0,
    thinking: str = "server_default",
    concurrency_levels: Sequence[int] = (2,),
    concurrency_rounds: int = 1,
    results_root: Path | None = None,
) -> Bed:
    if base_url is None:
        assert server is not None, "make_bed: server か base_url のどちらかが要る"
        base_url = server.base_url
    return Bed(
        targets=write_targets(
            tmp_path,
            base_url,
            api_key_env=api_key_env,
            max_context_tokens=max_context_tokens,
        ),
        profiles=write_profiles(
            tmp_path,
            decode_trials=decode_trials,
            decode_warmup=decode_warmup,
            max_consecutive_failures=max_consecutive_failures,
            metrics_interval_s=metrics_interval_s,
            thinking=thinking,
            concurrency_levels=concurrency_levels,
            concurrency_rounds=concurrency_rounds,
        ),
        results_root=results_root if results_root is not None else tmp_path / "results",
    )


class RecordingProgress:
    """進み具合を覚えておくだけの受け口。"""

    def __init__(self) -> None:
        self.updates: list[tuple[SuiteName, str, int, int, int]] = []

    def update(
        self, suite: SuiteName, condition: str, done: int, total: int, failures: int
    ) -> None:
        self.updates.append((suite, condition, done, total, failures))


async def execute(
    bed: Bed,
    *,
    suites: Sequence[SuiteName] = (SuiteName.DECODE,),
    registry: Mapping[SuiteName, Any] | None = None,
    progress: ProgressSink | None = None,
    env: Mapping[str, str] | None = None,
    trials_override: Mapping[str, int] | None = None,
    target_name: str = TARGET_NAME,
    profile_name: str = PROFILE_NAME,
) -> RunOutcome:
    """試験の既定値をまとめて `execute_run` を呼ぶ。環境の写像は既定で空。"""
    return await execute_run(
        RunRequest(
            target_name=target_name,
            suites=list(suites),
            profile_name=profile_name,
            trials_override=dict(trials_override if trials_override is not None else {}),
        ),
        progress if progress is not None else RecordingProgress(),
        targets_path=bed.targets,
        profiles_path=bed.profiles,
        results_root=bed.results_root,
        registry=registry,
        env=env if env is not None else {},
    )


# --- 試験から完全に制御できる、最小のまとまり -------------------------------


@dataclass(frozen=True)
class StubCondition:
    """`StubSuite` が計画する 1 つの条件。"""

    key: str
    trials: int | None = None
    """`None` なら `ctx.profile.decode.trials` を使う (上書きが届くことを見るため)。"""
    warmup_trials: int = 0
    text_chars: int = 8
    """送る本文の文字数 (入力の長さの上限に当てるときに増やす)。"""
    concurrency: int = 1
    """計画に載せる、1 回ぶんの本数。"""
    records_per_round: int | None = None
    """実際に返すレコードの数 (既定は `concurrency`)。少なく返す壊れ方を作れる。"""


class StubSuite:
    """`suites/base` の約束事に従う、試験専用のまとまり。

    本物のまとまり (decode など) は条件の数も試行の中身も自分で決めるので、
    「2 回目の失敗でちょうど止まる」といった細かい確かめには向かない。この
    まとまりは、条件と試行の数と本文の大きさを試験から直に決められる。
    """

    def __init__(
        self,
        conditions: Sequence[StubCondition],
        *,
        skipped: Sequence[str] = (),
        name: SuiteName = SuiteName.QUALITY,
        outcome: QualityOutcome | None = None,
    ) -> None:
        self.name = name
        self.closed_early: list[str] = []
        """途中で閉じられた (`aclose`) 条件の鍵。読み切って終わった条件は入らない。"""
        self._conditions = list(conditions)
        self._skipped = list(skipped)
        self._outcome = outcome
        """すべての試行に付ける採点の結果 (`None` なら判定を付けない)。"""
        self._chars = {condition_key(name, c.key): c.text_chars for c in conditions}
        self._per_round = {
            condition_key(name, c.key): (
                c.records_per_round if c.records_per_round is not None else c.concurrency
            )
            for c in conditions
        }

    def plan(self, ctx: SuiteContext) -> list[ConditionPlan | SkippedCondition]:
        planned: list[ConditionPlan | SkippedCondition] = [
            plan_condition(
                ctx,
                suite=self.name,
                key=condition_key(self.name, cond.key),
                trials=cond.trials if cond.trials is not None else ctx.profile.decode.trials,
                warmup_trials=cond.warmup_trials,
                max_tokens=16,
                concurrency=cond.concurrency,
            )
            for cond in self._conditions
        ]
        planned.extend(
            skip_condition(
                self.name, condition_key(self.name, key), "試験のために計画の段階で飛ばした"
            )
            for key in self._skipped
        )
        return planned

    async def run_condition(
        self, ctx: SuiteContext, cond: ConditionPlan
    ) -> AsyncIterator[TrialRecord]:
        text = "x" * self._chars[cond.key]
        per_round = self._per_round[cond.key]
        single = cond.concurrency == 1
        try:
            for trial_index, warmup in iter_trials(cond):
                for stream_index in range(per_round):
                    record = await run_trial(
                        ctx,
                        cond,
                        trial_index=trial_index,
                        warmup=warmup,
                        round_id=None if single else trial_index,
                        stream_index=None if single else stream_index,
                        messages=single_user_message(
                            f"{cond.key} {trial_index} {stream_index} {text}"
                        ),
                        verdict=self._verdict if self._outcome is not None else None,
                    )
                    yield record
                    abort_if_context_limit(cond, record)
        except GeneratorExit:
            # 進行の側が `aclose()` したときだけ来る (読み切った場合は来ない)
            self.closed_early.append(cond.key)
            raise

    def _verdict(self, result: StreamResult) -> QualityVerdict:
        """採点の結果 (要求が失敗しても、必ず判定を付ける決まり。注 4.2)。"""
        outcome = QualityOutcome.NOT_SCORED if result.error is not None else self._outcome
        assert outcome is not None
        return QualityVerdict(task="code", outcome=outcome, detail="試験のための判定")


@dataclass(frozen=True)
class StubRunInfo:
    """`QualityRunInfo` の代わり (計測ランに記録するもの)。"""

    datasets: tuple[DatasetRef, ...] = ()
    sandbox_image_ref: str | None = None


class InfoSuite(StubSuite):
    """`run_info()` を持つまとまり (品質の検査 6.7 の代わり)。

    `run_info()` は `Suite` の約束事にないので、進行の側 (8.1) は、持っている
    まとまりだけを見分けて、使った公開の課題を実行の条件に足す。
    """

    def __init__(
        self,
        conditions: Sequence[StubCondition],
        *,
        datasets: Sequence[DatasetRef] = (),
        skipped: Sequence[str] = (),
        name: SuiteName = SuiteName.QUALITY,
        outcome: QualityOutcome | None = None,
    ) -> None:
        super().__init__(conditions, skipped=skipped, name=name, outcome=outcome)
        self._info = StubRunInfo(datasets=tuple(datasets))

    def run_info(self) -> StubRunInfo:
        return self._info


def dataset(name: str = "HumanEval+", version: str = "v0.1.10") -> DatasetRef:
    """公開の課題の出どころ (中身は、名前と版で見分けられれば足りる)。"""
    return DatasetRef(
        name=name,
        version=version,
        source_url=f"https://example.invalid/{name}/{version}",
        license="Apache-2.0",
        scoring_method="検査のプログラムを隔離して動かす",
        sha256="0" * 64,
    )


def stub_registry(*suites: StubSuite) -> dict[SuiteName, Any]:
    return {suite.name: suite for suite in suites}


def opened(outcome: RunOutcome) -> RunStore:
    return RunStore.open(outcome.run_dir)


def trials_of(outcome: RunOutcome) -> list[TrialRecord]:
    records, warnings = opened(outcome).read_trials()
    assert warnings == []
    return records


def init_git_repo(root: Path, *, gitignore: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    if gitignore is not None:
        (root / ".gitignore").write_text(gitignore, encoding="utf-8")


# --- 前提が満たされないとき: 計測ランのディレクトリを作らない (1.4、8.5) ------


async def test_an_unknown_target_fails_without_creating_a_run_directory(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    bed = make_bed(tmp_path, fake_server)

    with pytest.raises(PreconditionError) as exc_info:
        await execute(bed, target_name="いない子")

    assert exc_info.value.exit_code == EXIT_PRECONDITION
    assert "いない子" in str(exc_info.value)
    assert not bed.results_root.exists()


async def test_an_unknown_profile_fails_without_creating_a_run_directory(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    bed = make_bed(tmp_path, fake_server)

    with pytest.raises(PreconditionError):
        await execute(bed, profile_name="いない設定")

    assert not bed.results_root.exists()


async def test_a_trials_override_below_the_minimum_fails_without_creating_a_run_directory(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    bed = make_bed(tmp_path, fake_server)

    with pytest.raises(PreconditionError) as exc_info:
        await execute(bed, trials_override={"decode.trials": 5})

    assert "decode.trials" in str(exc_info.value)
    assert not bed.results_root.exists()


async def test_an_override_path_that_is_not_a_profile_field_fails(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    bed = make_bed(tmp_path, fake_server)

    with pytest.raises(PreconditionError):
        await execute(bed, trials_override={"decode.nope": 11})

    assert not bed.results_root.exists()


async def test_a_missing_api_key_env_var_fails_without_creating_a_run_directory(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    bed = make_bed(tmp_path, fake_server, api_key_env="BENCH_TEST_RUNNER_KEY")

    with pytest.raises(PreconditionError) as exc_info:
        await execute(bed, env={})

    message = str(exc_info.value)
    assert "BENCH_TEST_RUNNER_KEY" in message
    assert not bed.results_root.exists()


async def test_the_api_key_value_never_appears_in_the_run(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """認証の情報が要る対象サーバーでも、値は生データに出ない (1.8)。"""
    bed = make_bed(tmp_path, fake_server, api_key_env="BENCH_TEST_RUNNER_KEY")
    fake_server.set_response(text_response("ok"))
    stub = StubSuite([StubCondition(key="a", trials=1)])

    outcome = await execute(
        bed,
        suites=[SuiteName.QUALITY],
        registry=stub_registry(stub),
        env={"BENCH_TEST_RUNNER_KEY": "s3cret-value"},
    )

    assert outcome.exit_code == EXIT_OK
    for path in outcome.run_dir.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".prom"}:
            assert "s3cret-value" not in path.read_text(encoding="utf-8")
    sent = fake_server.requests_for("/v1/messages")[-1]
    assert sent.headers["x-api-key"] == "s3cret-value"  # ヘッダーには付く


async def test_a_results_root_inside_a_repository_fails_without_creating_anything(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    init_git_repo(repo, gitignore="")
    bed = make_bed(tmp_path, fake_server, results_root=repo / "results")

    with pytest.raises(PreconditionError) as exc_info:
        await execute(bed)

    assert "results" in str(exc_info.value)
    assert not (repo / "results").exists()
    # 置き場所は、対象サーバーに触る前に確かめる (要求を 1 つも送っていない)
    assert fake_server.call_count("/v1/messages") == 0


async def test_a_server_without_token_counts_fails_with_no_usage(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(replace(text_response("ok"), usage=None))
    bed = make_bed(tmp_path, fake_server)

    with pytest.raises(PreconditionError) as exc_info:
        await execute(bed)

    assert "no_usage" in str(exc_info.value)
    assert not bed.results_root.exists()


async def test_an_unreachable_target_fails_without_creating_a_run_directory(
    tmp_path: Path,
) -> None:
    bed = make_bed(tmp_path, base_url="http://127.0.0.1:1")

    with pytest.raises(PreconditionError) as exc_info:
        await execute(bed)

    assert "unreachable" in str(exc_info.value)
    assert not bed.results_root.exists()


async def test_thinking_other_than_server_default_fails_without_creating_a_run_directory(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """渡し方が決まるまで (task 8.4)、thinking の切り替えは前提の不足にする (注 3.1)。"""
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server, thinking="on")

    with pytest.raises(PreconditionError) as exc_info:
        await execute(bed)

    assert "thinking" in str(exc_info.value)
    assert not bed.results_root.exists()
    # 使えない設定は、対象サーバーへ 1 件も送る前に弾く
    assert fake_server.call_count("/v1/messages") == 0


async def test_a_suite_that_is_not_in_the_registry_fails(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """対応表にない名前は、前提の不足にする。

    task 8.1 で 5 つすべてが既定の対応表に載ったので、対応表を絞って確かめる
    (既定の対応表に足りない名前がなくなっても、この判定は残す)。
    """
    bed = make_bed(tmp_path, fake_server)
    registry = stub_registry(StubSuite([StubCondition(key="a", trials=1)], name=SuiteName.DECODE))

    with pytest.raises(PreconditionError) as exc_info:
        await execute(bed, suites=[SuiteName.AGENT], registry=registry)

    assert "agent" in str(exc_info.value)
    assert "decode" in str(exc_info.value)  # 実行できるものを示す
    assert not bed.results_root.exists()


async def test_the_same_suite_twice_fails(fake_server: FakeServer, tmp_path: Path) -> None:
    """2 度実行すると、同じ (条件、試行の番号) のレコードが 2 つできてしまう。"""
    bed = make_bed(tmp_path, fake_server)

    with pytest.raises(PreconditionError) as exc_info:
        await execute(bed, suites=[SuiteName.DECODE, SuiteName.DECODE])

    assert "decode" in str(exc_info.value)
    assert not bed.results_root.exists()


async def test_an_empty_suite_list_fails(fake_server: FakeServer, tmp_path: Path) -> None:
    bed = make_bed(tmp_path, fake_server)

    with pytest.raises(PreconditionError):
        await execute(bed, suites=[])

    assert not bed.results_root.exists()


# --- ふつうに終わる計測ラン (1.6、7.1、7.5、8.7) ------------------------------


async def test_a_decode_run_completes_and_records_the_conditions(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """本物の `DecodeSuite` で、端から端まで 1 回流す。"""
    fake_server.set_response(replace(text_response("ok"), stop_reason="max_tokens"))
    fake_server.set_version("9.9.9-fake")
    fake_server.set_context_limit(200000)
    bed = make_bed(tmp_path, fake_server, decode_trials=10, metrics_interval_s=0.05)
    progress = RecordingProgress()

    outcome = await execute(bed, suites=[SuiteName.DECODE], progress=progress)

    assert outcome.status is RunStatus.COMPLETED
    assert outcome.exit_code == EXIT_OK
    assert outcome.run_dir == bed.results_root / outcome.run_id

    manifest = opened(outcome).manifest()
    assert manifest.status is RunStatus.COMPLETED
    assert manifest.target.name == TARGET_NAME
    assert manifest.server_model == "fake-model"
    assert manifest.server_version == "9.9.9-fake"
    assert manifest.started_at is not None
    assert manifest.finished_at is not None
    assert manifest.suites == [SuiteName.DECODE]
    assert manifest.profile_name == PROFILE_NAME
    assert manifest.profile.decode.trials == 10
    assert manifest.context_limit == 200000
    assert manifest.generator_version >= 1
    assert manifest.skipped == []
    assert re.fullmatch(r"[^+]+\+(g[0-9a-f]{7,}(\.dirty)?|unknown)", manifest.harness_version)

    records = trials_of(outcome)
    assert len(records) == 40  # 4 つの条件 × 10 回
    assert {record.condition for record in records} == {
        "decode/code/en",
        "decode/code/ja",
        "decode/prose/en",
        "decode/prose/ja",
    }
    assert all(record.run_id == outcome.run_id for record in records)
    assert all(record.result.error is None for record in records)

    deltas, delta_warnings = opened(outcome).read_metric_deltas()
    assert delta_warnings == []
    assert len(deltas) == 4
    assert all(row.before_available and row.after_available for row in deltas)
    assert any(row.derived.spec_acceptance_rate is not None for row in deltas)
    assert len(list((outcome.run_dir / "metrics").glob("*.before.prom"))) == 4
    assert len(list((outcome.run_dir / "metrics").glob("*.after.prom"))) == 4
    assert progress.updates[-1][2] == 10


async def test_a_concurrency_run_counts_every_stream_of_a_round(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """1 回ぶんが n 本のレコードになるまとまりでも、進み具合の全体の数が合う (4.2)。"""
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    progress = RecordingProgress()

    outcome = await execute(bed, suites=[SuiteName.CONCURRENCY], progress=progress)

    assert outcome.status is RunStatus.COMPLETED
    records = trials_of(outcome)
    # 慣らし 1 回ぶん + 本番 1 回ぶん、それぞれ 2 本
    assert len(records) == 4
    assert all(record.condition == "concurrency/c2" for record in records)
    assert sorted(record.stream_index or 0 for record in records) == [0, 0, 1, 1]
    assert [update[2] for update in progress.updates] == [1, 2, 3, 4]
    assert [update[3] for update in progress.updates] == [4] * 4


async def test_only_the_requested_suites_run_in_the_given_order(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    registry = stub_registry(
        StubSuite([StubCondition(key="a", trials=1)], name=SuiteName.DECODE),
        StubSuite([StubCondition(key="a", trials=1)], name=SuiteName.PREFILL),
        StubSuite([StubCondition(key="a", trials=1)], name=SuiteName.CONCURRENCY),
    )

    outcome = await execute(bed, suites=[SuiteName.PREFILL, SuiteName.DECODE], registry=registry)

    assert [record.suite for record in trials_of(outcome)] == [
        SuiteName.PREFILL,
        SuiteName.DECODE,
    ]
    assert opened(outcome).manifest().suites == [SuiteName.PREFILL, SuiteName.DECODE]


async def test_the_default_registry_holds_the_five_suites() -> None:
    """task 8.1: 品質の検査と長い会話の検査も、対応表に載る (5 つ全部)。"""
    assert set(default_suite_registry()) == set(SuiteName)
    for name, suite in default_suite_registry().items():
        assert suite.name is name


def test_the_registry_hands_out_a_fresh_quality_and_agent_suite_every_time() -> None:
    """計測ランごとに作り直す (共有の状態そのものをなくす。注 7.2 → 8.1、6.7 → 8.1)。

    module の実体 (`SUITE`) を配ると、1 つのプロセスで 2 つの計測ランを流したとき
    に、覚え書き (使った公開の課題、上限に当たった長さ) が持ち越されうる。
    """
    first = default_suite_registry()
    second = default_suite_registry()

    assert isinstance(first[SuiteName.QUALITY], QualitySuite)
    assert isinstance(first[SuiteName.AGENT], AgentSuite)
    assert first[SuiteName.QUALITY] is not second[SuiteName.QUALITY]
    assert first[SuiteName.AGENT] is not second[SuiteName.AGENT]
    assert first[SuiteName.QUALITY] is not quality_module.SUITE
    assert first[SuiteName.AGENT] is not agent_module.SUITE


def test_the_registry_can_be_given_a_problems_loader_for_the_quality_suite() -> None:
    """公開の課題の読み方 (置き場所、取得の可否) を、入口 (5.1) が差し替えられる。"""
    calls: list[int] = []

    def loader() -> tuple[DatasetRef, list[Any]]:
        calls.append(1)
        return dataset(), []

    registry = default_suite_registry(problems_loader=loader)
    suite = registry[SuiteName.QUALITY]
    assert isinstance(suite, QualitySuite)
    # 差し替えた読み方が、まとまりの中に届いている (呼ぶのはまとまりだけ)
    assert suite._problems_loader is loader
    assert calls == []


async def test_each_run_builds_its_own_registry(
    fake_server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """対応表は、プロセスに 1 つではなく、計測ランごとに 1 つ作る (8.1)。"""
    bed = make_bed(tmp_path, fake_server)
    seen: list[Mapping[SuiteName, Suite]] = []
    real = runner_module.default_suite_registry

    def spy(**kwargs: Any) -> dict[SuiteName, Suite]:
        registry = real(**kwargs)
        seen.append(registry)
        return registry

    monkeypatch.setattr(runner_module, "default_suite_registry", spy)

    for _ in range(2):
        with pytest.raises(PreconditionError):  # 対応表を作ったあと、設定の誤りで止まる
            await execute(bed, target_name="いない子")

    assert len(seen) == 2
    assert seen[0][SuiteName.QUALITY] is not seen[1][SuiteName.QUALITY]
    assert seen[0][SuiteName.AGENT] is not seen[1][SuiteName.AGENT]


# --- 使った公開の課題を、実行の条件に残す (5.5、5.7、注 6.7 → 8.1) -----------


async def test_the_datasets_a_suite_used_reach_the_manifest(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """まとまりを流したあとに `run_info()` を読んで、`datasets` に書き写す。"""
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    used = dataset()
    suite = InfoSuite([StubCondition(key="a", trials=1)], datasets=(used,))

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=stub_registry(suite))

    assert outcome.status is RunStatus.COMPLETED
    assert opened(outcome).manifest().datasets == [used]


async def test_no_datasets_are_recorded_when_the_suite_used_none(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """コードの条件を飛ばした計測ランに、使っていない課題を載せない (5.7)。"""
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    suite = InfoSuite([StubCondition(key="a", trials=1)], datasets=())

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=stub_registry(suite))

    assert opened(outcome).manifest().datasets == []


async def test_a_suite_without_run_info_leaves_the_datasets_empty(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """`run_info()` は `Suite` の約束事にない。持たないまとまりでも、進行は止まらない。"""
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    suite = StubSuite([StubCondition(key="a", trials=1)])

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=stub_registry(suite))

    assert not isinstance(suite, SupportsRunInfo)
    assert opened(outcome).manifest().datasets == []


async def test_the_datasets_are_recorded_even_when_the_run_is_aborted(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """止まった計測ランの要約にも、使った課題が出る (10.4: 未完了でも、それまでを残す)。"""
    fake_server.set_response_sequence([text_response("ok"), http_error_response(500)])
    bed = make_bed(tmp_path, fake_server, max_consecutive_failures=1)
    used = dataset()
    suite = InfoSuite([StubCondition(key="a", trials=3)], datasets=(used,))

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=stub_registry(suite))

    assert outcome.status is RunStatus.ABORTED
    assert opened(outcome).manifest().datasets == [used]


async def test_the_datasets_are_recorded_even_when_the_run_is_interrupted(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """中断した計測ランでも、そこまでに使った課題は残る (10.4)。"""
    fake_server.set_response(replace(text_response("ok"), first_delay_s=0.3))
    bed = make_bed(tmp_path, fake_server)
    used = dataset()
    suite = InfoSuite([StubCondition(key="a", trials=5)], datasets=(used,))

    outcome = await execute(
        bed,
        suites=[SuiteName.QUALITY],
        registry=stub_registry(suite),
        progress=SignalAfterFirstTrial(),
    )

    assert outcome.status is RunStatus.INTERRUPTED
    assert opened(outcome).manifest().datasets == [used]


async def test_datasets_from_several_suites_are_merged_without_duplicates(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """2 つのまとまりが課題を報せたら、上書きではなく足し合わせる (名前と版で重複を除く)。"""
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    shared = dataset()
    other = dataset(name="Another", version="v2")
    registry = stub_registry(
        InfoSuite([StubCondition(key="a", trials=1)], datasets=(shared,)),
        InfoSuite(
            [StubCondition(key="a", trials=1)],
            datasets=(shared, other),
            name=SuiteName.AGENT,
        ),
    )

    outcome = await execute(bed, suites=[SuiteName.QUALITY, SuiteName.AGENT], registry=registry)

    assert opened(outcome).manifest().datasets == [shared, other]


async def test_a_later_suite_adds_its_datasets_instead_of_replacing_the_earlier_ones(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """2 つ目のまとまりが自分の課題だけを報せても、1 つ目の課題は消えない (上書きしない)。"""
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    first = dataset()
    second = dataset(name="Another", version="v2")
    registry = stub_registry(
        InfoSuite([StubCondition(key="a", trials=1)], datasets=(first,)),
        InfoSuite([StubCondition(key="a", trials=1)], datasets=(second,), name=SuiteName.AGENT),
    )

    outcome = await execute(bed, suites=[SuiteName.QUALITY, SuiteName.AGENT], registry=registry)

    assert opened(outcome).manifest().datasets == [first, second]


def test_the_run_info_protocol_matches_the_quality_suite() -> None:
    """進行の側の約束事 (`SupportsRunInfo`) を、品質の検査がそのまま満たす。"""
    quality: SupportsRunInfo = QualitySuite()
    info: SuiteRunInfo = quality.run_info()
    stub: SupportsRunInfo = InfoSuite([])

    assert isinstance(quality, SupportsRunInfo)
    assert isinstance(stub, SupportsRunInfo)
    assert list(info.datasets) == []
    assert info.sandbox_image_ref is None
    # 長い会話の検査は、公開の課題を使わないので `run_info()` を持たない
    assert not isinstance(AgentSuite(), SupportsRunInfo)


async def test_a_trials_override_reaches_the_suite_and_the_manifest(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server, decode_trials=10)
    stub = StubSuite([StubCondition(key="a")])  # 試行の数は設定から取る

    outcome = await execute(
        bed,
        suites=[SuiteName.QUALITY],
        registry=stub_registry(stub),
        trials_override={"decode.trials": 11},
    )

    assert opened(outcome).manifest().profile.decode.trials == 11
    assert len(trials_of(outcome)) == 11


# --- 連続の失敗での停止 (10.3) ------------------------------------------------


async def test_the_run_aborts_after_the_configured_number_of_consecutive_failures(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    # 1 つ目は前提の確認が使う。以後、成功 2 回のあとは失敗が続く
    fake_server.set_response_sequence(
        [
            text_response("ok"),
            text_response("ok"),
            text_response("ok"),
            http_error_response(500),
        ]
    )
    bed = make_bed(tmp_path, fake_server, max_consecutive_failures=3)
    stub = StubSuite([StubCondition(key="a", trials=10), StubCondition(key="b", trials=10)])

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=stub_registry(stub))

    assert outcome.status is RunStatus.ABORTED
    assert outcome.exit_code == EXIT_ABORTED
    records = trials_of(outcome)
    assert [record.result.error is None for record in records] == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert {record.condition for record in records} == {"quality/a"}  # 2 つ目には進まない
    assert fake_server.call_count("/v1/messages") == 6  # 前提の確認 1 + 試行 5
    # 途中で止めた条件の反復子は、その場で閉じる (送りかけの要求を残さない)
    assert stub.closed_early == ["quality/a"]
    manifest = opened(outcome).manifest()
    assert manifest.status is RunStatus.ABORTED
    assert manifest.finished_at is not None
    assert any("応答していない" in warning for warning in manifest.warnings)


async def test_a_finished_round_is_written_whole_before_the_run_stops(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """止めると決まっても、応答を得た要求のレコードは捨てない (8.1、10.1)。

    同時処理のまとまりは、1 回ぶんの n 本をすべて終えてから順に返す。2 本目で
    止めて反復子を閉じると、すでに応答のあった残りの 2 本が消えてしまう。
    """
    fake_server.set_response_sequence([text_response("ok"), http_error_response(500)])
    bed = make_bed(
        tmp_path,
        fake_server,
        max_consecutive_failures=2,
        concurrency_levels=(4,),
        concurrency_rounds=1,
    )

    outcome = await execute(bed, suites=[SuiteName.CONCURRENCY])

    assert outcome.status is RunStatus.ABORTED
    assert outcome.exit_code == EXIT_ABORTED
    records = trials_of(outcome)
    assert len(records) == 4
    assert sorted(record.stream_index or 0 for record in records) == [0, 1, 2, 3]
    # 応答を得た要求は、1 つ残らず生データにある (前提の確認の 1 件を除く)
    assert fake_server.call_count("/v1/messages") - 1 == len(records)
    assert len(list((outcome.run_dir / "bodies").glob("*.json.gz"))) == len(records)


async def test_a_partly_successful_round_is_also_written_whole(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """1 本だけ成功して 3 本が失敗した 1 回ぶんでも、4 件が残る。"""
    fake_server.set_response_sequence(
        [
            text_response("ok"),  # 前提の確認
            http_error_response(500),
            text_response("ok"),
            http_error_response(500),
        ]
    )
    bed = make_bed(
        tmp_path,
        fake_server,
        max_consecutive_failures=2,
        concurrency_levels=(4,),
        concurrency_rounds=1,
    )

    outcome = await execute(bed, suites=[SuiteName.CONCURRENCY])

    assert outcome.status is RunStatus.ABORTED
    records = trials_of(outcome)
    assert len(records) == 4
    assert sum(1 for record in records if record.result.error is None) == 1
    assert fake_server.call_count("/v1/messages") - 1 == len(records)


async def test_a_round_that_returns_fewer_records_than_planned_still_stops(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """1 回ぶんの本数より少なく返すまとまりでも、次の組が来たら、それを保存して止める。"""
    fake_server.set_response_sequence([text_response("ok"), http_error_response(500)])
    bed = make_bed(tmp_path, fake_server, max_consecutive_failures=1)
    stub = StubSuite(
        [StubCondition(key="a", trials=5, concurrency=4, records_per_round=1)],
    )

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=stub_registry(stub))

    assert outcome.status is RunStatus.ABORTED
    records = trials_of(outcome)
    # 止めると決めた 1 件と、次の組の 1 件 (応答を得たので残す) だけで止まる
    assert len(records) == 2
    assert [record.round_id for record in records] == [0, 1]
    assert fake_server.call_count("/v1/messages") - 1 == len(records)


async def test_a_success_between_failures_resets_the_counter(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response_sequence(
        [
            text_response("ok"),  # 前提の確認
            http_error_response(500),
            http_error_response(500),
            text_response("ok"),
            http_error_response(500),
        ]
    )
    bed = make_bed(tmp_path, fake_server, max_consecutive_failures=3)
    registry = stub_registry(StubSuite([StubCondition(key="a", trials=10)]))

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=registry)

    assert outcome.status is RunStatus.ABORTED
    assert [record.result.error is None for record in trials_of(outcome)] == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]


async def test_scorer_outcomes_are_not_counted_as_request_failures(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """不正解が続いても止めない (10.3 が数えるのは、要求そのものの失敗だけ)。

    品質の検査と長い会話の検査は、成功した要求に採点の結果を付ける (6.7、7.2)。
    その結果を失敗として数えると、モデルの出来が悪いだけの計測ランが、対象
    サーバーの不調として止められてしまう。
    """
    fake_server.set_response(text_response("まちがった答え"))
    bed = make_bed(tmp_path, fake_server, max_consecutive_failures=2)
    suite = StubSuite([StubCondition(key="a", trials=5)], outcome=QualityOutcome.INCORRECT)

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=stub_registry(suite))

    assert outcome.status is RunStatus.COMPLETED
    assert outcome.exit_code == EXIT_OK
    records = trials_of(outcome)
    assert len(records) == 5
    assert all(record.result.error is None for record in records)
    assert [
        record.verdict.outcome for record in records if isinstance(record.verdict, QualityVerdict)
    ] == [QualityOutcome.INCORRECT] * 5


async def test_a_decode_run_that_starts_failing_keeps_the_earlier_trials(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """3.5 の完了の状態: 途中で止まった偽のサーバーでも、それまでの試行が残る。"""
    fake_server.set_response_sequence(
        [
            text_response("ok"),
            text_response("ok"),
            text_response("ok"),
            http_error_response(503),
        ]
    )
    bed = make_bed(tmp_path, fake_server, decode_trials=10, max_consecutive_failures=2)

    outcome = await execute(bed, suites=[SuiteName.DECODE])

    assert outcome.status is RunStatus.ABORTED
    assert outcome.exit_code == EXIT_ABORTED
    records = trials_of(outcome)
    assert len(records) == 4
    assert all(record.condition == "decode/code/en" for record in records)
    assert all(record.request_body_ref for record in records)


# --- 上限の超過は連続の失敗に数えない (3.6) -----------------------------------


async def test_a_context_limit_condition_is_skipped_and_does_not_count_as_a_failure(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(text_response("ok"))
    fake_server.set_context_limit(200, advertise=False)  # 申告しないので、送ってみて 400 を見る
    bed = make_bed(tmp_path, fake_server, max_consecutive_failures=1)
    registry = stub_registry(
        StubSuite(
            [
                StubCondition(key="big", trials=3, text_chars=4000),
                StubCondition(key="small", trials=2),
            ]
        )
    )

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=registry)

    # 上限の超過を連続の失敗に数えていたら、max_consecutive_failures=1 で止まっていた
    assert outcome.status is RunStatus.COMPLETED
    assert outcome.exit_code == EXIT_OK
    manifest = opened(outcome).manifest()
    assert [skipped.key for skipped in manifest.skipped] == ["quality/big"]
    assert manifest.skipped[0].suite is SuiteName.QUALITY
    assert "400" in manifest.skipped[0].reason
    assert [record.condition for record in trials_of(outcome)] == [
        "quality/big",  # 400 の試行も生データには残る (8.1)
        "quality/small",
        "quality/small",
    ]


async def test_a_skipped_condition_from_plan_is_recorded_in_the_manifest(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    registry = stub_registry(StubSuite([StubCondition(key="a", trials=1)], skipped=["long"]))

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=registry)

    assert outcome.status is RunStatus.COMPLETED
    assert [skipped.key for skipped in opened(outcome).manifest().skipped] == ["quality/long"]
    assert len(trials_of(outcome)) == 1


# --- 内部の指標 (7.1、7.4、7.5) -----------------------------------------------


async def test_a_missing_metrics_endpoint_does_not_stop_the_run(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(text_response("ok"))
    fake_server.set_metrics_enabled(False)
    bed = make_bed(tmp_path, fake_server)
    registry = stub_registry(StubSuite([StubCondition(key="a", trials=2)]))

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=registry)

    assert outcome.status is RunStatus.COMPLETED
    assert outcome.exit_code == EXIT_OK
    deltas, _ = opened(outcome).read_metric_deltas()
    assert len(deltas) == 1
    assert not deltas[0].before_available
    assert not deltas[0].after_available
    assert deltas[0].before_unavailable_reason is not None
    assert "404" in deltas[0].before_unavailable_reason
    assert deltas[0].derived.missing != []
    assert list((outcome.run_dir / "metrics").glob("*.prom")) == []


async def test_requests_in_flight_before_the_run_are_warned_about(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """ほかの利用者がいると、内部の指標の増分が混ざる (design.md 計測ランの進行)。"""

    def factory(body: dict[str, Any]) -> Script:
        if "HOLD" in json.dumps(body, ensure_ascii=False):
            return stalled_response(after_events=1, stall_s=3.0)
        return text_response("ok")

    fake_server.set_response_factory(factory)
    bed = make_bed(tmp_path, fake_server)
    registry = stub_registry(StubSuite([StubCondition(key="a", trials=1)]))

    async with HttpxMessagesClient(fake_server.base_url) as holder:
        request = MessagesRequest(
            model="fake-model", max_tokens=8, messages=single_user_message("HOLD")
        )
        held = asyncio.create_task(holder.stream(request, TimeoutPolicy(total_s=10.0)))
        await wait_for_requests(fake_server, 1)
        try:
            outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=registry)
        finally:
            held.cancel()
            await asyncio.gather(held, return_exceptions=True)

    warnings = opened(outcome).manifest().warnings
    assert any("処理していた" in warning for warning in warnings)


async def wait_for_requests(server: FakeServer, count: int, *, timeout_s: float = 5.0) -> None:
    """要求が届くまで、event loop を止めずに待つ (偽のサーバーの待ちは thread を塞ぐ)。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while server.call_count("/v1/messages") < count:
        if loop.time() > deadline:
            raise AssertionError(f"要求が {count} 件届かなかった")
        await asyncio.sleep(0.01)


# --- 進み具合 (1.7) ------------------------------------------------------------


async def test_the_progress_sink_sees_every_trial_with_the_right_total(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response_sequence(
        [
            text_response("ok"),  # 前提の確認
            text_response("ok"),
            http_error_response(500),
            text_response("ok"),
        ]
    )
    bed = make_bed(tmp_path, fake_server)
    registry = stub_registry(StubSuite([StubCondition(key="a", trials=3, warmup_trials=1)]))
    progress = RecordingProgress()

    await execute(bed, suites=[SuiteName.QUALITY], registry=registry, progress=progress)

    assert [update[0] for update in progress.updates] == [SuiteName.QUALITY] * 4
    assert [update[1] for update in progress.updates] == ["quality/a"] * 4
    assert [update[2] for update in progress.updates] == [1, 2, 3, 4]  # 慣らしも数える
    assert [update[3] for update in progress.updates] == [4] * 4
    assert [update[4] for update in progress.updates] == [0, 1, 1, 1]


def test_the_stderr_progress_sink_writes_one_line(capsys: pytest.CaptureFixture[str]) -> None:
    StderrProgressSink().update(SuiteName.DECODE, "decode/code/en", 3, 10, 1)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.strip().splitlines()) == 1
    for part in ("decode", "decode/code/en", "3", "10", "1"):
        assert part in captured.err


@pytest.mark.parametrize(
    ("suite", "condition"),
    [
        (SuiteName.QUALITY, "quality/needle/128k/d100"),
        (SuiteName.QUALITY, "quality/code/humaneval+"),
        (SuiteName.AGENT, "agent/stage/120k"),
    ],
)
def test_the_stderr_progress_sink_shows_long_condition_keys_whole(
    suite: SuiteName, condition: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """8.1 で足したまとまりの、長い条件の鍵も、切り詰めずに 1 行で出す (1.7)。"""
    StderrProgressSink().update(suite, condition, 7, 50, 2)

    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
    assert condition in err
    assert "…" not in err
    assert err.strip() == f"[{suite.value}] {condition} 7/50 失敗 2"


# --- 中断 (10.4) ---------------------------------------------------------------


class SignalAfterFirstTrial:
    """最初の試行が終わった時点で、自分のプロセスに中断の合図を送る受け口。"""

    def __init__(self, signum: int = signal.SIGINT) -> None:
        self.updates: list[tuple[SuiteName, str, int, int, int]] = []
        self._signum = signum
        self._fired = False

    def update(
        self, suite: SuiteName, condition: str, done: int, total: int, failures: int
    ) -> None:
        self.updates.append((suite, condition, done, total, failures))
        if not self._fired:
            self._fired = True
            os.kill(os.getpid(), self._signum)


async def test_an_interrupt_signal_stops_the_run_and_marks_it_interrupted(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    # 2 つ目の試行が「送っている途中」になるように、応答の頭を遅らせる
    fake_server.set_response(replace(text_response("ok"), first_delay_s=0.3))
    fake_server.set_kv_usage(0.42)
    bed = make_bed(tmp_path, fake_server, metrics_interval_s=0.02)
    registry = stub_registry(StubSuite([StubCondition(key="a", trials=5)]))
    progress = SignalAfterFirstTrial()

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=registry, progress=progress)

    assert outcome.status is RunStatus.INTERRUPTED
    assert outcome.exit_code == EXIT_INTERRUPTED
    records = trials_of(outcome)
    assert len(records) == 1  # 打ち切った試行のレコードは作らない
    manifest = opened(outcome).manifest()
    assert manifest.status is RunStatus.INTERRUPTED
    assert manifest.finished_at is not None
    assert any("SIGINT" in warning and "quality/a" in warning for warning in manifest.warnings)

    # 中断のあとは、あとの時点の /metrics を読みに行かない (読まなかった事実を残す)
    deltas, _ = opened(outcome).read_metric_deltas()
    assert len(deltas) == 1
    assert deltas[0].before_available
    assert not deltas[0].after_available
    assert deltas[0].after_unavailable_reason is not None
    assert "中断" in deltas[0].after_unavailable_reason
    # 中断した条件でも、定期の読み取りは止めて、追ってきた最大値を残す
    assert deltas[0].derived.kv_usage_peak == pytest.approx(0.42)

    lingering = {task.get_name() for task in asyncio.all_tasks()}
    assert RUN_TASK_NAME not in lingering
    assert "metrics-sampler" not in lingering

    loop = asyncio.get_running_loop()
    assert not loop.remove_signal_handler(signal.SIGINT)  # すでに外してある
    assert not loop.remove_signal_handler(signal.SIGTERM)


async def test_the_signal_handlers_are_removed_after_a_normal_run(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """合図の受け口を残したままにすると、次の計測ランや呼び出し側に漏れる。"""
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    registry = stub_registry(StubSuite([StubCondition(key="a", trials=1)]))

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=registry)

    assert outcome.status is RunStatus.COMPLETED
    loop = asyncio.get_running_loop()
    assert not loop.remove_signal_handler(signal.SIGINT)
    assert not loop.remove_signal_handler(signal.SIGTERM)


async def test_a_sigterm_also_marks_the_run_interrupted(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.set_response(replace(text_response("ok"), first_delay_s=0.3))
    bed = make_bed(tmp_path, fake_server)
    registry = stub_registry(StubSuite([StubCondition(key="a", trials=5)]))

    outcome = await execute(
        bed,
        suites=[SuiteName.QUALITY],
        registry=registry,
        progress=SignalAfterFirstTrial(signal.SIGTERM),
    )

    assert outcome.status is RunStatus.INTERRUPTED
    assert outcome.exit_code == EXIT_INTERRUPTED


# --- 生データを保存できないとき (8.1) -----------------------------------------


async def test_a_storage_failure_aborts_the_run_and_propagates(
    fake_server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    registry = stub_registry(StubSuite([StubCondition(key="a", trials=2)]))

    def boom(self: RunStore, body: dict[str, JsonValue]) -> str:
        raise StoreError("試験のための保存の失敗")

    monkeypatch.setattr(RunStore, "put_body", boom)

    with pytest.raises(StoreError):
        await execute(bed, suites=[SuiteName.QUALITY], registry=registry)

    run_dirs = list_run_dirs(bed.results_root)
    assert len(run_dirs) == 1
    manifest = RunStore.open(run_dirs[0]).manifest()
    assert manifest.status is RunStatus.ABORTED
    assert any("保存" in warning for warning in manifest.warnings)


async def test_a_metrics_write_failure_does_not_stop_the_run(
    fake_server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """内部の指標は補いの情報なので、保存できなくても計測は続く (7.4)。

    試行のレコードの保存の失敗 (上の試験) と対にして、扱いの違いを固定する。
    """
    fake_server.set_response(text_response("ok"))
    bed = make_bed(tmp_path, fake_server)
    registry = stub_registry(StubSuite([StubCondition(key="a", trials=2)]))

    def boom(
        self: RunStore,
        condition: str,
        before: object,
        after: object,
        derived: object,
    ) -> None:
        raise StoreError("試験のための指標の保存の失敗")

    monkeypatch.setattr(RunStore, "write_metrics", boom)

    outcome = await execute(bed, suites=[SuiteName.QUALITY], registry=registry)

    assert outcome.status is RunStatus.COMPLETED
    assert outcome.exit_code == EXIT_OK
    assert len(trials_of(outcome)) == 2
    warnings = opened(outcome).manifest().warnings
    assert any("内部の指標" in warning and "quality/a" in warning for warning in warnings)


# --- 終了の値 (design.md Monitoring) -------------------------------------------


def test_the_exit_codes_are_the_documented_numbers() -> None:
    assert (EXIT_OK, EXIT_PRECONDITION, EXIT_ABORTED, EXIT_INTERRUPTED) == (0, 1, 2, 130)


# --- 道具の版 (1.6) ------------------------------------------------------------


def commit_all(repo: Path, message: str = "初回") -> None:
    """試験用のリポジトリに 1 つコミットする (計測者の git の設定に依らせない)。"""
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            message,
        ],
        cwd=repo,
        check=True,
    )


def clean_repo(tmp_path: Path) -> Path:
    """コミットが 1 つあり、変更のないリポジトリ (`ignored.txt` は無視される)。"""
    repo = tmp_path / "repo"
    init_git_repo(repo, gitignore="ignored.txt\n")
    (repo / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    commit_all(repo)
    return repo


def test_harness_version_uses_the_git_commit_of_a_clean_repository(tmp_path: Path) -> None:
    repo = clean_repo(tmp_path)

    version = harness_version(repo)

    assert re.fullmatch(r"[^+]+\+g[0-9a-f]{7,}", version), version
    assert "unknown" not in version
    assert not version.endswith(".dirty")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert version.split("+g")[1] == head[: len(version.split("+g")[1])]


def test_harness_version_marks_a_modified_tracked_file_as_dirty(tmp_path: Path) -> None:
    repo = clean_repo(tmp_path)
    (repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")

    assert harness_version(repo).endswith(".dirty")


def test_harness_version_marks_an_untracked_file_as_dirty(tmp_path: Path) -> None:
    """まだ追加していない `.py` も、読み込まれれば振る舞いを変える (9.4 の守り)。"""
    repo = clean_repo(tmp_path)
    (repo / "new_module.py").write_text("y = 1\n", encoding="utf-8")

    assert harness_version(repo).endswith(".dirty")


def test_harness_version_ignores_files_that_git_ignores(tmp_path: Path) -> None:
    repo = clean_repo(tmp_path)
    (repo / "ignored.txt").write_text("生データや仮想環境の置き場所", encoding="utf-8")

    version = harness_version(repo)

    assert not version.endswith(".dirty"), version


def test_harness_version_has_the_documented_shape() -> None:
    version = harness_version()

    assert re.fullmatch(r"[^+]+\+(g[0-9a-f]{7,}(\.dirty)?|unknown)", version), version


def test_harness_version_degrades_when_git_cannot_answer(tmp_path: Path) -> None:
    outside = tmp_path / "no-git-here"
    outside.mkdir()

    assert harness_version(outside).endswith("+unknown")


def test_harness_version_degrades_when_git_is_not_on_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = clean_repo(tmp_path)
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))

    assert harness_version(repo).endswith("+unknown")
