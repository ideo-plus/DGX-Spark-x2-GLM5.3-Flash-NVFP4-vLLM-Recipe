"""品質の検査と長い会話の検査を、偽のサーバーで端から端まで確かめる (task 8.1)。

tasks.md 8.1 の完了の状態を、どれもコマンドの入口 (`cli.main`) から通して
確かめる。

1. **9 種類の分類が、要約の段階ごとの表まで届く** (6.3、6.4): 偽のサーバーに
   9 種類を出し分けさせ、`summary.json` の段階ごとの件数と、`summary.md` の
   内訳の表を突き合わせる。要求そのものの失敗は、崩れた割合の分母から外れる
2. **上限で止まった計測ランが要約できる** (6.9): 段階と段階の間に上限を申告
   する偽のサーバーで、途中の段階まで進んで 0 で終わり、要約に到達した長さと
   止めた理由が出る。そういう計測ランどうしの比較も通る
3. **採点できなかった件数が、不正解と分かれて表示される** (5.7): 一部の要求を
   HTTP 500 にして、割合の分母から外れ、`summary.md` に別の欄で出ることを
   確かめる。飛ばしたコードの条件の理由も、要約に出る
4. **5 つのまとまりのどれでも、単独で選んで実行できる** (1.2)。`--suite` を
   省いたときの既定は速さの 3 つ、`--suite all` で 5 つ全部
5. **外に出てはいけないもの** (1.8、8.3) と、送った要求 == レコード == 保存した
   本文 (8.1、注 3.5)

## 公開の課題とコンテナ

コマンドの入口からは、Python の値 (手で書いた課題) を差し込めない。そこで
CLI を通す試験では、課題の置き場所を空のディレクトリに向け (`--data-cache`)、
取得も許さない (`--no-download`)。コードの条件は理由つきで飛び、要求は 1 つも
送られない。ネットワークには触れない (`humaneval.default_cache_dir` を、
呼ばれたら落ちるものに差し替えて、既定の置き場所に触れていないことも確かめる)。

コードの条件が端から端まで動くことは、最後の 1 本だけが、対応表を差し込んだ
`execute_run` と**実物のコンテナ**で確かめる (実行環境がない機械では、理由を
添えて飛ばす)。

生データの置き場所は、必ず `tmp_path` の下にする (リポジトリの `results/` と
`bench/data-cache/` を汚さない。8.5、6.5)。要求を送る先は 127.0.0.1 の偽の
サーバーだけである。時刻の値を判定する試験は 1 つもない。
"""

from __future__ import annotations

import gzip
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest

from bench_harness import cli
from bench_harness.analysis.summarize import METRIC_ACCURACY, NOT_SCORED_COUNT, write_summary
from bench_harness.corpus import humaneval
from bench_harness.corpus.needle import make_needle_case
from bench_harness.corpus.tools import make_tool_task
from bench_harness.runner import EXIT_OK, StderrProgressSink, execute_run
from bench_harness.scoring.sandbox import ContainerSandbox
from bench_harness.store import RunStore, list_run_dirs
from bench_harness.suites.quality import (
    CODE_CONDITION_KEY,
    TOOLCALL_TASK_INDEX_BASE,
    QualitySuite,
    needle_condition_key,
)
from bench_harness.types import (
    CodeProblem,
    DatasetRef,
    MetricResult,
    QualityOutcome,
    QualityVerdict,
    RunRequest,
    RunStatus,
    SandboxSettings,
    SandboxUnavailable,
    SuiteName,
    Summary,
    ToolCallOutcome,
    ToolTask,
    TrialRecord,
)
from fake_server import (
    FakeServer,
    Script,
    http_error_response,
    replace,
    text_response,
    tool_use_response,
)

TARGET_NAME: Final[str] = "fake"
PROFILE_NAME: Final[str] = "e2e"
SEED: Final[int] = 11

API_KEY_ENV: Final[str] = "BENCH_TEST_QA_KEY"
API_KEY_VALUE: Final[str] = "sk-bench-quality-DO-NOT-LEAK-3f7a"
RESPONSE_SENTINEL: Final[str] = "RESPONSE-SENTINEL-bench-qa-5c2e"

PREFLIGHT_MESSAGE_REQUESTS: Final[int] = 1
"""前提の確認 (2.7) が `/v1/messages` に送る要求の数 (試行のレコードを持たない)。"""

STAGES: Final[tuple[int, ...]] = (4000, 6000, 8000)
"""長い会話の検査の段階 (2 万〜12 万の代わりに使う、小さな段階)。"""

AGENT_TRIALS: Final[int] = 9
"""1 段階の試行の数。9 種類の分類を、段階ごとに 1 回ずつ出すのにちょうど要る数。"""

AGENT_CONVERSATIONS: Final[int] = 3
AGENT_MAX_TOKENS: Final[int] = 256
TASK_INDEX_STRIDE: Final[int] = 10_000
"""課題の番号の、段階ごとの幅 (`suites/agent.py`「最後の 1 手」の決まり)。"""

TOOLCALL_TASKS: Final[int] = 3
NEEDLE_LENGTH: Final[int] = 800
NEEDLE_DEPTHS: Final[tuple[int, ...]] = (0, 50)

DECODE_TRIALS: Final[int] = 10
"""`DecodeSettings.trials` の下限 (2.2)。"""

CONTEXT_LIMIT: Final[int] = 7000
"""4k と 6k の段階は収まり、8k の段階は収まらない上限 (6.9)。"""

MARKUP_TEXT: Final[str] = "<tool_call>read_file\n<arg_key>path</arg_key> と書いてしまった"
"""`TargetDef.tool_markup_markers` の既定の目印を含む本文 (`MARKUP_LEAKED`)。"""


# --- 9 種類の出し分け --------------------------------------------------------


KIND_OUTCOMES: Final[dict[str, ToolCallOutcome]] = {
    "correct": ToolCallOutcome.CORRECT,
    "no_call": ToolCallOutcome.NO_CALL,
    "unknown_tool": ToolCallOutcome.UNKNOWN_TOOL,
    "args_unparseable": ToolCallOutcome.ARGS_UNPARSEABLE,
    "args_schema_invalid": ToolCallOutcome.ARGS_SCHEMA_INVALID,
    "markup_leaked": ToolCallOutcome.MARKUP_LEAKED,
    "wrong_call": ToolCallOutcome.WRONG_CALL,
    "empty_or_truncated": ToolCallOutcome.EMPTY_OR_TRUNCATED,
    "request_failed": ToolCallOutcome.REQUEST_FAILED,
}
"""応答の作り方 → 期待する分類 (`scoring/toolcall.py` の検査の順序どおり)。"""

NINE_KINDS: Final[tuple[str, ...]] = tuple(KIND_OUTCOMES)
"""1 段階の中で、試行の番号の順に出し分ける 9 種類。

要求そのものの失敗を最後に置くので、段階をまたいでも 2 回続かない (連続の失敗
で計測ランが止まらない。10.3)。
"""


def tool_call_script(task: ToolTask, kind: str) -> Script:
    """課題と種類から、その分類になる応答を作る (`scoring/toolcall.py` の 9 種類)。"""
    match kind:
        case "correct":
            return tool_use_response(task.expected_tool, task.expected_input)
        case "no_call":
            return text_response(f"No tool is needed here. {RESPONSE_SENTINEL}")
        case "unknown_tool":
            return tool_use_response("read_file_v2_is_not_in_the_catalog", {"path": "src/main.py"})
        case "args_unparseable":
            # 引数の JSON が途中で切れている (クライアントは tool_input=None にする)
            return tool_use_response(task.expected_tool, fragments=('{"path": "src/mai',))
        case "args_schema_invalid":
            # 目録のツールはどれも `additionalProperties: false` なので、定義に
            # ない引数だけの呼び出しは、必ずスキーマに合わない
            return tool_use_response(task.expected_tool, {"not_a_parameter_of_this_tool": 1})
        case "markup_leaked":
            return text_response(MARKUP_TEXT)
        case "wrong_call":
            other = other_tool_task(task)
            return tool_use_response(other.expected_tool, other.expected_input)
        case "empty_or_truncated":
            return replace(text_response("途中で切れた"), stop_reason="max_tokens")
        case "request_failed":
            return http_error_response(500)
    raise AssertionError(f"知らない応答の種類: {kind!r}")


def other_tool_task(task: ToolTask) -> ToolTask:
    """正解と違うツールを呼ぶ課題 (引数はそのツールの定義に合う → `WRONG_CALL`)。"""
    for index in range(900_000, 900_100):
        other = make_tool_task(index, SEED)
        if other.expected_tool != task.expected_tool:
            return other
    raise AssertionError("違うツールの課題が見つからない")


def agent_task(stage_ordinal: int, trial_index: int) -> ToolTask:
    """長い会話の検査が、その段階の試行で聞く課題 (`suites/agent.py` の番号の決まり)。"""
    return make_tool_task(stage_ordinal * TASK_INDEX_STRIDE + trial_index, SEED)


def toolcall_task(index: int) -> ToolTask:
    """品質の検査のツール呼び出しの課題 (`suites/quality.py` の番号の決まり)。"""
    return make_tool_task(TOOLCALL_TASK_INDEX_BASE + index, SEED)


def needle_text(depth_pct: int, index: int = 0) -> tuple[str, str]:
    """探す課題の、送られる本文と正解 (`suites/quality.py` と同じ作り方)。"""
    case = make_needle_case(
        target_tokens=NEEDLE_LENGTH,
        depth_pct=depth_pct,
        index=index,
        seed=SEED,
        chars_per_token=CHARS_PER_TOKEN,
    )
    return f"{case.haystack}\n\n{case.question}", case.answer


class Responder:
    """要求の中身を見て、課題ごとの台本を返す (偽のサーバーの `set_response_factory`)。

    ツール呼び出しの課題は、指示の文 (最後の `user` の発話) で引く。どの課題に
    どの応答を返すかは、課題の番号だけで決まるので、要求が届く順に依存しない。
    """

    def __init__(
        self,
        *,
        agent_kinds: Sequence[str] = ("correct",),
        agent_trials: int = AGENT_TRIALS,
        toolcall_kinds: Sequence[str] = ("correct",),
        needle_kinds: Sequence[str] = ("correct",),
    ) -> None:
        self._tasks: dict[str, ToolTask] = {}
        self._kinds: dict[str, str] = {}
        for ordinal in range(len(STAGES)):
            for trial_index in range(agent_trials):
                task = agent_task(ordinal, trial_index)
                self._tasks[task.prompt] = task
                self._kinds[task.prompt] = agent_kinds[trial_index % len(agent_kinds)]
        for index in range(TOOLCALL_TASKS):
            task = toolcall_task(index)
            self._tasks[task.prompt] = task
            self._kinds[task.prompt] = toolcall_kinds[index % len(toolcall_kinds)]
        self._needles: dict[str, tuple[str, str]] = {}
        for index, depth in enumerate(NEEDLE_DEPTHS):
            text, answer = needle_text(depth)
            self._needles[text] = (answer, needle_kinds[index % len(needle_kinds)])

    def __call__(self, body: dict[str, Any]) -> Script:
        text = last_user_text(body)
        task = self._tasks.get(text)
        if task is not None:
            return tool_call_script(task, self._kinds[text])
        needle = self._needles.get(text)
        if needle is not None:
            answer, kind = needle
            if kind == "request_failed":
                return http_error_response(500)
            if kind == "wrong":
                return text_response("The access code is SIGIL-0BADC0DE.")
            return text_response(f"The access code is {answer}. {RESPONSE_SENTINEL}")
        return text_response("ok")  # 前提の確認と、速さのまとまり

    def agent_outcomes(self, trials: int) -> dict[ToolCallOutcome, int]:
        """1 段階ぶんの、期待する分類ごとの件数 (試験の側で数えたもの)。"""
        counts: Counter[ToolCallOutcome] = Counter()
        for trial_index in range(trials):
            prompt = agent_task(0, trial_index).prompt
            counts[KIND_OUTCOMES[self._kinds[prompt]]] += 1
        return dict(counts)


def last_user_text(body: Mapping[str, Any]) -> str:
    """要求の本文から、最後の `user` の発話の文字列を取り出す。"""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return ""
    content = last.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block["text"])
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


# --- 設定ファイルと置き場所の下ごしらえ --------------------------------------

CHARS_PER_TOKEN: Final[float] = 4.0
"""1 トークンあたりの文字数 (偽のサーバーの既定と同じにそろえる)。"""

PROFILES_TOML: Final[Path] = Path(__file__).resolve().parents[2] / "config" / "profiles.toml"


def quick_sandbox_settings() -> SandboxSettings:
    """計測で実際に使う隔離の設定 (`quick`) を、そのまま読む。"""
    data: dict[str, Any] = tomllib.loads(PROFILES_TOML.read_text(encoding="utf-8"))
    return SandboxSettings.model_validate(data["profiles"]["quick"]["sandbox"])


def sandbox_is_available() -> bool:
    """この機械で、コードの隔離が使えるか (使えなければ、コードの条件は先に飛ぶ)。"""
    return ContainerSandbox(quick_sandbox_settings()).available() is True


@dataclass(frozen=True)
class Bed:
    """1 回のコマンドに要る、設定ファイルと置き場所の一式。"""

    targets: Path
    profiles: Path
    results_root: Path
    data_cache: Path

    def run_options(self) -> list[str]:
        return [
            "--targets",
            str(self.targets),
            "--profiles",
            str(self.profiles),
            "--results-root",
            str(self.results_root),
            "--data-cache",
            str(self.data_cache),
            "--no-download",
        ]


def write_bed(tmp_path: Path, base_url: str, *, api_key_env: str | None = None) -> Bed:
    """試験用の対象サーバーの定義と計測の設定を書く。

    5 つのまとまりを、どれも単独で流せる大きさにしてある。`length_tolerance`
    を大きく取っているのは、偽のサーバーが入力のトークン数を「JSON の文字数 ÷
    4.0」で数える一方、合成の文章は本文の文字数だけで長さを決めるので、狙いから
    ずれるためである (この試験は長さの印を見ない)。

    隔離の設定は、計測で実際に使うもの (`quick`) をそのまま書く。既定のまま
    (`python:3.12-slim`、識別子なし) にすると、隔離が使えないという理由だけで
    コードの条件が飛び、課題の置き場所 (`--data-cache`) を見に行く道が、一度も
    通らなくなる。
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    targets = tmp_path / "targets.toml"
    lines = [f"[targets.{TARGET_NAME}]", f'base_url = "{base_url}"', 'model = "fake-model"']
    if api_key_env is not None:
        lines.append(f'api_key_env = "{api_key_env}"')
    targets.write_text("\n".join(lines) + "\n", encoding="utf-8")

    profiles = tmp_path / "profiles.toml"
    profiles.write_text(_profile_toml(quick_sandbox_settings()), encoding="utf-8")
    data_cache = tmp_path / "data-cache"
    data_cache.mkdir(parents=True, exist_ok=True)
    return Bed(
        targets=targets,
        profiles=profiles,
        results_root=tmp_path / "results",
        data_cache=data_cache,
    )


def _profile_toml(sandbox: SandboxSettings) -> str:
    text = f"""
[profiles.{PROFILE_NAME}]
seed = {SEED}
min_successes = 1
max_consecutive_failures = 5
length_tolerance = 0.9
compare_tolerance = 0.02
metrics_interval_s = 60.0

[profiles.{PROFILE_NAME}.chars_per_token]
prose_en = {CHARS_PER_TOKEN}
prose_ja = {CHARS_PER_TOKEN}
code = {CHARS_PER_TOKEN}
log = {CHARS_PER_TOKEN}

[profiles.{PROFILE_NAME}.sampling]
temperature = 0.0
thinking = "server_default"

[profiles.{PROFILE_NAME}.timeout]
connect_s = 5.0
first_event_s = 5.0
idle_s = 5.0
total_s = 30.0

[profiles.{PROFILE_NAME}.decode]
trials = {DECODE_TRIALS}
warmup_trials = 0
max_tokens = 32

[profiles.{PROFILE_NAME}.prefill]
trials = 2
warmup_trials = 1
max_tokens = 16
target_input_tokens = [1000]

[profiles.{PROFILE_NAME}.concurrency]
levels = [2]
rounds = 1
max_tokens = 16
input_tokens = 400

[profiles.{PROFILE_NAME}.quality]
toolcall_tasks = {TOOLCALL_TASKS}
needle_lengths = [{NEEDLE_LENGTH}]
needle_depths = [{", ".join(str(depth) for depth in NEEDLE_DEPTHS)}]
trials_per_cell = 1
code_max_tokens = 256
code_problem_limit = 2

[profiles.{PROFILE_NAME}.agent]
start_tokens = {STAGES[0]}
end_tokens = {STAGES[-1]}
step_tokens = {STAGES[1] - STAGES[0]}
trials_per_stage = {AGENT_TRIALS}
conversations_per_stage = {AGENT_CONVERSATIONS}
threshold = 0.01
max_tokens = {AGENT_MAX_TOKENS}
"""
    digest = f'image_digest = "{sandbox.image_digest}"\n' if sandbox.image_digest else ""
    return (
        text
        + f"""
[profiles.{PROFILE_NAME}.sandbox]
runtime = "{sandbox.runtime}"
image = "{sandbox.image}"
{digest}timeout_s = {sandbox.timeout_s}
memory_mb = {sandbox.memory_mb}
cpus = {sandbox.cpus}
pids_limit = {sandbox.pids_limit}
"""
    )


def bench(*argv: str) -> int:
    """`bench` を、その場 (同じプロセス) で 1 回走らせる。"""
    return cli.main(list(argv))


def run_bench(bed: Bed, *argv: str) -> tuple[int, Path]:
    """`bench run` を 1 回流して、終了の値と、増えた計測ランのディレクトリを返す。"""
    before = {path.name for path in _run_dirs(bed)}
    code = bench(
        "run",
        "--target",
        TARGET_NAME,
        "--profile",
        PROFILE_NAME,
        *argv,
        *bed.run_options(),
    )
    added = [path for path in _run_dirs(bed) if path.name not in before]
    assert len(added) == 1, f"増えた計測ランが 1 つではない: {added}"
    return code, added[0]


def _run_dirs(bed: Bed) -> list[Path]:
    return list_run_dirs(bed.results_root) if bed.results_root.is_dir() else []


def forbid_the_default_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """既定の置き場所 (`bench/data-cache/`) に触れたら、その場で落ちるようにする。

    `--data-cache` が効いていないと、この試験が本物の置き場所を読み書きして
    しまう。読まれた瞬間に分かるようにしておく (6.5)。
    """

    def boom() -> Path:
        raise AssertionError("既定の課題の置き場所に触れた (--data-cache が効いていない)")

    monkeypatch.setattr(humaneval, "default_cache_dir", boom)


def read_summary(run_dir: Path) -> Summary:
    return Summary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))


def read_markdown(run_dir: Path) -> str:
    return (run_dir / "summary.md").read_text(encoding="utf-8")


def records_of(run_dir: Path) -> list[TrialRecord]:
    records, warnings = RunStore.open(run_dir).read_trials()
    assert warnings == [], warnings
    return records


def accuracy_rows(summary: Summary) -> dict[str, MetricResult]:
    return {row.condition: row for row in summary.results if row.metric == METRIC_ACCURACY}


def assert_requests_match_records(server: FakeServer, run_dir: Path) -> list[TrialRecord]:
    """送った要求 == 残ったレコード == 保存した本文 (8.1、注 3.5)。"""
    records = records_of(run_dir)
    assert server.call_count("/v1/messages") == PREFLIGHT_MESSAGE_REQUESTS + len(records)
    refs = [record.request_body_ref for record in records]
    assert all(refs), "本文の参照を持たないレコードがある"
    stored = sorted(path.name.removesuffix(".json.gz") for path in (run_dir / "bodies").iterdir())
    assert sorted(set(refs)) == stored
    return records


def read_all_text(root: Path) -> dict[str, str]:
    """`root` の下のすべてのファイルを、文字列にして返す (`.gz` は展開する)。"""
    texts: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if path.suffix == ".gz":
            data = gzip.decompress(data)
        texts[path.relative_to(root).as_posix()] = data.decode("utf-8", errors="replace")
    return texts


# --- 流れ 1: 9 種類の分類が、段階ごとの表まで届く (6.3、6.4) ------------------


def test_the_nine_outcomes_reach_the_agent_stage_table(
    fake_server: FakeServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """8.1 の完了の状態: 9 種類の分類が、要約の段階ごとの表まで届く。"""
    forbid_the_default_cache(monkeypatch)
    responder = Responder(agent_kinds=NINE_KINDS)
    fake_server.set_response_factory(responder)
    bed = write_bed(tmp_path, fake_server.base_url)

    code, run_dir = run_bench(bed, "--suite", "agent")
    capsys.readouterr()

    assert code == EXIT_OK
    assert fake_server.errors == []
    records = assert_requests_match_records(fake_server, run_dir)
    assert len(records) == len(STAGES) * AGENT_TRIALS

    expected = responder.agent_outcomes(AGENT_TRIALS)
    assert set(expected) == set(ToolCallOutcome), "9 種類すべてを出し分けていない"

    summary = read_summary(run_dir)
    assert summary.incomplete is False
    assert summary.agent is not None
    assert [stage.stage_key for stage in summary.agent.stages] == [
        "agent/stage/004k",
        "agent/stage/006k",
        "agent/stage/008k",
    ]
    for stage in summary.agent.stages:
        assert stage.trials == AGENT_TRIALS
        assert stage.outcome_counts == expected
        assert stage.request_failures == expected[ToolCallOutcome.REQUEST_FAILED]
        assert stage.break_rate is not None
        # 要求そのものの失敗は分母から外す。崩れは「正解」と「要求の失敗」以外の 7 種類
        assert stage.break_rate.denominator == AGENT_TRIALS - 1
        assert stage.break_rate.numerator == AGENT_TRIALS - 2
        assert stage.actual_input_tokens is not None

    markdown = read_markdown(run_dir)
    assert "## 長い会話でのツール呼び出し" in markdown
    assert "分類ごとの件数 (9 種類):" in markdown
    for outcome in ToolCallOutcome:
        assert outcome.value in markdown
    for stage in summary.agent.stages:
        assert f"`{stage.stage_key}`" in markdown


# --- 流れ 2: 上限で止まった計測ランが、要約できる (6.9) ----------------------


def test_a_run_stopped_at_the_context_limit_summarizes_and_compares(
    fake_server: FakeServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """上限に達した段階から先には送らず、到達した長さと理由が要約に出る (6.9)。"""
    forbid_the_default_cache(monkeypatch)
    trials = 2
    fake_server.set_response_factory(Responder(agent_trials=trials))
    # 上限を申告するが、要求そのものは断らない (飛ばすのは計画の段で決まる)
    fake_server.set_context_limit(CONTEXT_LIMIT, advertise=True, enforce=False)
    bed = write_bed(tmp_path, fake_server.base_url)
    options = ("--suite", "agent", "--set", f"agent.trials_per_stage={trials}")

    code_a, run_a = run_bench(bed, *options)
    fake_server.reset()  # 2 回目の計測ランの要求だけを数え直す (台本と口の設定は残る)
    code_b, run_b = run_bench(bed, *options)
    capsys.readouterr()

    assert (code_a, code_b) == (EXIT_OK, EXIT_OK)  # 上限で止まるのは、ふつうの完了
    store = RunStore.open(run_a)
    assert store.manifest().status is RunStatus.COMPLETED
    assert store.manifest().context_limit == CONTEXT_LIMIT
    assert [item.key for item in store.manifest().skipped] == ["agent/stage/008k"]
    # 8k の段階には、要求を 1 つも送っていない
    records = assert_requests_match_records(fake_server, run_b)
    assert {record.condition for record in records} == {
        "agent/stage/004k",
        "agent/stage/006k",
    }

    summary = read_summary(run_a)
    assert summary.agent is not None
    assert summary.agent.reached_tokens == STAGES[1]
    assert summary.agent.stopped_reason is not None
    assert str(STAGES[2]) in summary.agent.stopped_reason
    assert str(CONTEXT_LIMIT) in summary.agent.stopped_reason
    markdown = read_markdown(run_a)
    assert summary.agent.stopped_reason in markdown.replace("\\|", "|")

    # 上限で止まった計測ランどうしでも、比較が通る (9.1)
    assert bench("compare", str(run_a), str(run_b)) == EXIT_OK
    out = capsys.readouterr().out
    assert "# 計測ランの比較" in out
    assert "agent/stage/004k" in out


# --- 流れ 3: 採点できなかった件数が、不正解と分かれて出る (5.7) --------------


def test_not_scored_is_separated_from_incorrect_in_the_quality_summary(
    fake_server: FakeServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """要求が失敗した課題は分母から外し、別の欄で数える (5.6、5.7)。"""
    forbid_the_default_cache(monkeypatch)
    fake_server.set_response_factory(
        Responder(
            toolcall_kinds=("correct", "wrong_call", "request_failed"),
            needle_kinds=("correct", "request_failed"),
        )
    )
    bed = write_bed(tmp_path, fake_server.base_url)

    code, run_dir = run_bench(bed, "--suite", "quality")
    capsys.readouterr()

    assert code == EXIT_OK
    records = assert_requests_match_records(fake_server, run_dir)
    # コードの条件は飛んでいるので、要求は「ツール呼び出し 3 + 探す課題 2」だけ
    assert len(records) == TOOLCALL_TASKS + len(NEEDLE_DEPTHS)
    assert not any(record.condition == CODE_CONDITION_KEY for record in records)

    summary = read_summary(run_dir)
    rows = accuracy_rows(summary)
    toolcall = rows["quality/toolcall"]
    assert toolcall.proportion is not None
    # 3 問のうち 1 問は要求が失敗 → 分母は 2、正解は 1
    assert (toolcall.proportion.numerator, toolcall.proportion.denominator) == (1, 2)
    assert toolcall.flag_counts[NOT_SCORED_COUNT] == 1
    assert toolcall.failures == 1

    d0 = rows[needle_condition_key(NEEDLE_LENGTH, NEEDLE_DEPTHS[0])]
    d50 = rows[needle_condition_key(NEEDLE_LENGTH, NEEDLE_DEPTHS[1])]
    assert d0.proportion is not None
    assert (d0.proportion.numerator, d0.proportion.denominator) == (1, 1)
    assert d0.flag_counts[NOT_SCORED_COUNT] == 0
    assert d50.proportion is None  # 採点できた試行が 1 つもない
    assert d50.flag_counts[NOT_SCORED_COUNT] == 1

    markdown = read_markdown(run_dir)
    assert "## 品質の検査" in markdown
    assert "採点できなかった" in markdown

    # 飛ばしたコードの条件の理由が、要約に出る (5.7: 使っていない課題は載せない)
    skipped = RunStore.open(run_dir).manifest().skipped
    assert [item.key for item in skipped] == [CODE_CONDITION_KEY]
    assert CODE_CONDITION_KEY in markdown
    assert skipped[0].reason in markdown.replace("\\|", "|")
    if sandbox_is_available():
        # 隔離が使える機械では、飛ばす理由は「指定した置き場所に課題がなく、
        # 取得も許されていない」になる (--data-cache と --no-download が効いている)
        assert bed.data_cache.name in skipped[0].reason
        assert "取得も許されていない" in skipped[0].reason
    assert summary.datasets == []
    assert "## 使った公開の課題" not in markdown


# --- 流れ 4: 5 つのまとまりのどれでも、単独で選べる (1.2) --------------------


SUITE_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "agent": ("--set", "agent.trials_per_stage=1"),
}
"""まとまりごとの、試験を短くするための上書き。"""


@pytest.mark.parametrize("suite", [name.value for name in SuiteName])
def test_each_suite_can_be_selected_alone(
    suite: str,
    fake_server: FakeServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """8.1 の完了の状態: 5 つのまとまりのどれでも、単独で選んで実行できる。"""
    forbid_the_default_cache(monkeypatch)
    fake_server.set_response_factory(Responder(agent_trials=1))
    bed = write_bed(tmp_path, fake_server.base_url)

    code, run_dir = run_bench(bed, "--suite", suite, *SUITE_OPTIONS.get(suite, ()))
    capsys.readouterr()

    assert code == EXIT_OK
    store = RunStore.open(run_dir)
    assert store.manifest().suites == [SuiteName(suite)]
    records = records_of(run_dir)
    assert records, "試行が 1 つも残っていない"
    assert {record.suite for record in records} == {SuiteName(suite)}
    assert {record.condition.split("/")[0] for record in records} == {suite}
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "summary.md").is_file()


def test_the_default_is_the_speed_suites_and_all_selects_every_suite(
    fake_server: FakeServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--suite` を省くと速さの 3 つ、`--suite all` で 5 つ全部 (8.1 で決めた既定)。"""
    forbid_the_default_cache(monkeypatch)
    fake_server.set_response_factory(Responder(agent_trials=1))
    bed = write_bed(tmp_path, fake_server.base_url)

    code_default, default_dir = run_bench(bed)
    code_all, all_dir = run_bench(bed, "--suite", "all", "--set", "agent.trials_per_stage=1")
    capsys.readouterr()

    assert (code_default, code_all) == (EXIT_OK, EXIT_OK)
    assert RunStore.open(default_dir).manifest().suites == [
        SuiteName.DECODE,
        SuiteName.PREFILL,
        SuiteName.CONCURRENCY,
    ]
    assert {record.suite for record in records_of(default_dir)} == {
        SuiteName.DECODE,
        SuiteName.PREFILL,
        SuiteName.CONCURRENCY,
    }
    assert RunStore.open(all_dir).manifest().suites == list(SuiteName)
    assert {record.suite for record in records_of(all_dir)} == set(SuiteName)


# --- 流れ 5: 外に出てはいけないもの (1.8、8.3) -------------------------------


def test_no_secret_and_no_response_text_leaves_the_quality_and_agent_summaries(
    fake_server: FakeServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """認証の情報の値と、応答の本文が、要約と標準出力に出ない (1.8、8.3)。"""
    forbid_the_default_cache(monkeypatch)
    monkeypatch.setenv(API_KEY_ENV, API_KEY_VALUE)
    fake_server.set_response_factory(Responder(agent_kinds=("no_call",), agent_trials=1))
    bed = write_bed(tmp_path, fake_server.base_url, api_key_env=API_KEY_ENV)

    code_quality, quality_dir = run_bench(bed, "--suite", "quality")
    code_agent, agent_dir = run_bench(bed, "--suite", "agent", "--set", "agent.trials_per_stage=1")
    captured = capsys.readouterr()

    assert (code_quality, code_agent) == (EXIT_OK, EXIT_OK)
    # 仕掛けの裏取り: 認証の情報は、確かにヘッダーに付いている (1.8)
    assert fake_server.requests_for("/v1/messages")[-1].headers["x-api-key"] == API_KEY_VALUE

    raw = read_all_text(bed.results_root)
    for run_dir in (quality_dir, agent_dir):
        trials = f"{run_dir.name}/trials.jsonl"
        assert RESPONSE_SENTINEL in raw[trials], "応答の目印が trials.jsonl にない"
        checked = {
            "summary.json": (run_dir / "summary.json").read_text(encoding="utf-8"),
            "summary.md": read_markdown(run_dir),
        }
        for label, text in checked.items():
            assert RESPONSE_SENTINEL not in text, label
            assert API_KEY_VALUE not in text, label
    assert [name for name, text in raw.items() if API_KEY_VALUE in text] == []
    assert API_KEY_VALUE not in captured.out
    assert API_KEY_VALUE not in captured.err
    assert RESPONSE_SENTINEL not in captured.out


# --- コードの条件を、実物のコンテナで端から端まで ---------------------------

HANDWRITTEN: Final[tuple[CodeProblem, ...]] = (
    CodeProblem(
        task_id="Handwritten/correct",
        prompt='def add_one(x: int) -> int:\n    """Return x + 1."""\n',
        entry_point="add_one",
        test=(
            "def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(-3) == -2\n"
        ),
    ),
    CodeProblem(
        task_id="Handwritten/wrong",
        prompt='def double(x: int) -> int:\n    """Return x * 2."""\n',
        entry_point="double",
        test="def check(candidate):\n    assert candidate(3) == 6\n",
    ),
)

HANDWRITTEN_DATASET: Final[DatasetRef] = humaneval.dataset_ref()


def code_responder(body: dict[str, Any]) -> Script:
    """コードの課題には解答を、ほかの課題には当たり障りのない応答を返す。"""
    text = last_user_text(body)
    if f"def {HANDWRITTEN[0].entry_point}(" in text:
        return text_response("```python\ndef add_one(x: int) -> int:\n    return x + 1\n```")
    if f"def {HANDWRITTEN[1].entry_point}(" in text:
        return text_response("```python\ndef double(x: int) -> int:\n    return x + 2\n```")
    return text_response("ok")


def handwritten_loader() -> tuple[DatasetRef, list[CodeProblem]]:
    return HANDWRITTEN_DATASET, list(HANDWRITTEN)


async def test_the_code_condition_runs_end_to_end_and_records_the_dataset(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    """公開の課題の出どころが、計測ランと要約に残る (5.5、注 6.7 → 8.1)。

    コマンドの入口からは Python の値を差し込めないので、ここだけは対応表を
    `execute_run` に注入する。隔離は**実物のコンテナ**で、実行環境がない機械
    では理由を添えて飛ばす。
    """
    status = ContainerSandbox(quick_sandbox_settings()).available()
    if isinstance(status, SandboxUnavailable):
        pytest.skip(f"隔離の実行環境が使えない: {status.reason}")

    fake_server.set_response_factory(code_responder)
    bed = write_bed(tmp_path, fake_server.base_url)
    registry = {SuiteName.QUALITY: QualitySuite(problems_loader=handwritten_loader)}

    outcome = await execute_run(
        RunRequest(target_name=TARGET_NAME, suites=[SuiteName.QUALITY], profile_name=PROFILE_NAME),
        StderrProgressSink(),
        targets_path=bed.targets,
        profiles_path=bed.profiles,
        results_root=bed.results_root,
        registry=registry,
        env={},
    )
    write_summary(outcome.run_dir)

    assert outcome.status is RunStatus.COMPLETED
    store = RunStore.open(outcome.run_dir)
    assert store.manifest().skipped == []  # コードの条件は飛んでいない
    assert store.manifest().datasets == [HANDWRITTEN_DATASET]

    code_records = [
        record for record in records_of(outcome.run_dir) if record.condition == CODE_CONDITION_KEY
    ]
    assert [
        record.verdict.outcome
        for record in code_records
        if isinstance(record.verdict, QualityVerdict)
    ] == [QualityOutcome.CORRECT, QualityOutcome.INCORRECT]
    first = code_records[0].verdict
    assert isinstance(first, QualityVerdict)
    assert first.sandbox is not None and first.sandbox.passed

    summary = read_summary(outcome.run_dir)
    assert [dataset.name for dataset in summary.datasets] == [HANDWRITTEN_DATASET.name]
    assert summary.datasets[0].version == HANDWRITTEN_DATASET.version
    assert summary.datasets[0].license == HANDWRITTEN_DATASET.license
    markdown = read_markdown(outcome.run_dir)
    assert "## 使った公開の課題" in markdown
    assert HANDWRITTEN_DATASET.license in markdown
