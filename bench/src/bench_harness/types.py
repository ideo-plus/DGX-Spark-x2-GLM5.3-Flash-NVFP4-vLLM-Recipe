"""全体で共有する型。

依存の向きの出発点であり、この module はほかの `bench_harness` の module を
読み込まない。標準ライブラリと pydantic だけに依存する。後続の部品は関数と
クラスの実装だけを受け持ち、このファイルを変更しない。

守る決まり:

- 認証の情報の値は、どの型にも入れない。対象サーバーの定義が持つのは環境変数の
  名前 (`TargetDef.api_key_env`) だけである (1.8)
- `Summary` から辿れる型は、送った内容と応答の本文を入れる項目を持たない (8.3)
- 記録の時刻は UTC の `datetime`、経過の時間はナノ秒の整数 (単調な時計の値)
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    JsonValue,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

SCHEMA_VERSION: Final[int] = 1
"""生データと要約の形の版。意味を変える変更、項目を消す変更で上げる。"""

GENERATOR_VERSION: Final[int] = 1
"""合成データの生成器の版。生成の仕方を変えたら上げる (比較の警告の対象)。"""


class _Model(BaseModel):
    """この module の型に共通の設定。知らない項目は、綴りの誤りとして弾く。"""

    model_config = ConfigDict(extra="forbid")


class _Frozen(_Model):
    """書いたあとに変えない型。更新は `model_copy(update=...)` で行う。"""

    model_config = ConfigDict(frozen=True)


# --- 印、分類、判定 -----------------------------------------------------


class SuiteName(StrEnum):
    """測る項目のまとまり。"""

    DECODE = "decode"
    PREFILL = "prefill"
    CONCURRENCY = "concurrency"
    QUALITY = "quality"
    AGENT = "agent"


class RunStatus(StrEnum):
    """計測ランの状態。`COMPLETED` 以外はすべて未完了として扱う。"""

    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    INTERRUPTED = "interrupted"


class TrialFlag(StrEnum):
    """1 つの試行に付く印。"""

    SHORT_OUTPUT = "short_output"
    LENGTH_OFF_TARGET = "length_off_target"
    TOO_FEW_OUTPUT_TOKENS = "too_few_output_tokens"
    REPLACEMENT_CHAR = "replacement_char"
    REPETITION_LOOP = "repetition_loop"


class MetricFlag(StrEnum):
    """条件ごとの結果に付く印。"""

    INSUFFICIENT_TRIALS = "insufficient_trials"
    PARTIAL_FAILURES = "partial_failures"
    SHORT_OUTPUTS = "short_outputs"
    LENGTH_OFF_TARGET = "length_off_target"
    SUSPECT_OUTPUTS = "suspect_outputs"


class ToolCallOutcome(StrEnum):
    """ツール呼び出しの応答の分類。上から順に、最初に当てはまった 1 つになる。"""

    CORRECT = "correct"
    NO_CALL = "no_call"
    UNKNOWN_TOOL = "unknown_tool"
    ARGS_UNPARSEABLE = "args_unparseable"
    ARGS_SCHEMA_INVALID = "args_schema_invalid"
    MARKUP_LEAKED = "markup_leaked"
    WRONG_CALL = "wrong_call"
    EMPTY_OR_TRUNCATED = "empty_or_truncated"
    REQUEST_FAILED = "request_failed"


class QualityOutcome(StrEnum):
    """品質の検査の採点。要求の失敗は、不正解と分けて数える (5.7)。"""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    NOT_SCORED = "not_scored"


class ThresholdVerdict(StrEnum):
    """割合が、しきい値を下回ったと言えるかどうか。"""

    BELOW = "below"
    ABOVE = "above"
    UNDETERMINED = "undetermined"


class LogicalMetric(StrEnum):
    """内部の指標の論理名。実際の名前への対応は `TargetDef.metric_map` で上書きする。"""

    SPEC_DRAFTS = "spec_drafts"
    SPEC_DRAFT_TOKENS = "spec_draft_tokens"
    SPEC_ACCEPTED_TOKENS = "spec_accepted_tokens"
    PREFIX_QUERIES = "prefix_queries"
    PREFIX_HITS = "prefix_hits"
    KV_USAGE = "kv_usage"
    RUNNING_REQUESTS = "running_requests"
    PROMPT_TOKENS = "prompt_tokens"
    GENERATION_TOKENS = "generation_tokens"
    ITERATION_TOKENS_SUM = "iteration_tokens_sum"
    ITERATION_TOKENS_COUNT = "iteration_tokens_count"
    PREEMPTIONS = "preemptions"


class ContentKind(StrEnum):
    """合成の文章の種類。1 トークンあたりの文字数は、種類ごとに違う。"""

    PROSE_EN = "prose_en"
    PROSE_JA = "prose_ja"
    CODE = "code"
    LOG = "log"


Lang = Literal["en", "ja"]
"""散文の言語。"""

Tier = Literal["primary", "reference"]
"""主な結果か、参考か (4.4)。"""

ThinkingMode = Literal["server_default", "on", "off"]
"""thinking の切り替え。渡し方を実機で確かめるまでは `server_default` だけを使う。"""

SandboxRuntime = Literal["auto", "podman", "docker"]
"""コードの隔離に使うコンテナの実行環境。`auto` は podman、docker の順に探す。"""

RequestErrorKind = Literal[
    "connect",
    "http",
    "stream_error",
    "timeout_first",
    "timeout_idle",
    "timeout_total",
    "protocol",
]
"""要求が失敗した種類。例外ではなく、値として返す (10.1、10.6)。"""

UnmetPrecondition = Literal["unreachable", "http_error", "no_usage", "no_output"]
"""計測を始められない前提の不足 (1.4)。"""

DiffOutcome = Literal["within", "outside"]
"""差が、ばらつきの範囲に収まるかどうか (9.2)。"""

ProportionDiffOutcome = Literal["different", "not_distinguishable"]
"""割合の差が、意味のあるものと言えるかどうか (9.6)。"""

QualityTaskKind = Literal["toolcall", "code", "needle"]
"""品質の検査の課題の種類。"""

Tolerance = Annotated[float, Field(gt=0.0, lt=1.0)]
"""許容の幅としきい値。0 と 1 は含まない。"""

DepthPct = Annotated[int, Field(ge=0, le=100)]
"""探す課題で、情報を埋める位置 (百分率)。"""


# --- 設定 ---------------------------------------------------------------


class Sampling(_Frozen):
    """サンプリングの設定。同じ条件のすべての試行で同じものを使う (2.7)。"""

    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    thinking: ThinkingMode = "server_default"


class TimeoutPolicy(_Frozen):
    """3 種類の制限時間。`ping` が来ないので、`idle_s` で止まったことを見つける。"""

    connect_s: PositiveFloat = 10.0
    first_event_s: PositiveFloat = 120.0
    idle_s: PositiveFloat = 60.0
    total_s: PositiveFloat = 900.0


class TargetDef(_Frozen):
    """対象サーバーの定義。認証の情報は、環境変数の名前だけを持つ (1.8)。"""

    name: str = Field(min_length=1)
    base_url: HttpUrl
    model: str = Field(min_length=1)
    notes: str = ""
    api_key_env: str | None = None
    metrics_url: HttpUrl | None = None
    max_context_tokens: int | None = Field(default=None, ge=1)
    metric_map: dict[LogicalMetric, str] = Field(default_factory=dict)
    tool_markup_markers: list[str] = Field(
        default_factory=lambda: ["<tool_call>", "</tool_call>", "<arg_key>", "<arg_value>"]
    )


class OutputSanity(_Frozen):
    """出力が壊れている疑いの判定の設定 (10.7)。"""

    repeat_min_chars: PositiveInt = 12
    repeat_min_count: int = Field(default=8, ge=2)


class SandboxSettings(_Frozen):
    """コードの隔離の実行の設定 (5.4)。イメージはダイジェストで固定する。"""

    runtime: SandboxRuntime = "auto"
    image: str = "python:3.12-slim"
    image_digest: str | None = None
    timeout_s: PositiveFloat = 10.0
    memory_mb: int = Field(default=512, ge=64)
    cpus: PositiveFloat = 1.0
    pids_limit: PositiveInt = 128


class DecodeSettings(_Frozen):
    """生成速度のまとまりの設定。試行は 10 回以上 (2.2)。"""

    trials: int = Field(default=10, ge=10)
    warmup_trials: NonNegativeInt = 2
    max_tokens: PositiveInt = 1024


class PrefillSettings(_Frozen):
    """入力の処理と最初のトークンまでの時間のまとまりの設定。"""

    trials: PositiveInt = 10
    warmup_trials: NonNegativeInt = 1
    max_tokens: PositiveInt = 16
    target_input_tokens: list[PositiveInt] = Field(
        default_factory=lambda: [8000, 32000, 128000], min_length=1
    )


class ConcurrencySettings(_Frozen):
    """同時処理のまとまりの設定。"""

    levels: list[PositiveInt] = Field(default_factory=lambda: [1, 2, 4, 8], min_length=1)
    rounds: PositiveInt = 5
    max_tokens: PositiveInt = 256
    input_tokens: PositiveInt = 2000


class QualitySettings(_Frozen):
    """品質の検査のまとまりの設定。"""

    toolcall_tasks: PositiveInt = 50
    needle_lengths: list[PositiveInt] = Field(
        default_factory=lambda: [8000, 32000, 128000], min_length=1
    )
    needle_depths: list[DepthPct] = Field(
        default_factory=lambda: [0, 25, 50, 75, 100], min_length=1
    )
    trials_per_cell: PositiveInt = 3
    code_max_tokens: PositiveInt = 1024
    code_problem_limit: int | None = Field(default=None, ge=1)


class AgentSettings(_Frozen):
    """長い会話の検査のまとまりの設定 (6.8)。"""

    start_tokens: PositiveInt = 20000
    end_tokens: PositiveInt = 120000
    step_tokens: PositiveInt = 20000
    trials_per_stage: PositiveInt = 50
    conversations_per_stage: PositiveInt = 5
    threshold: Tolerance = 0.01
    max_tokens: PositiveInt = 256

    @model_validator(mode="after")
    def _check_stage_range(self) -> Self:
        if self.start_tokens > self.end_tokens:
            raise ValueError("start_tokens が end_tokens より大きい")
        return self


_DEFAULT_CHARS_PER_TOKEN: Final[dict[ContentKind, float]] = {
    ContentKind.PROSE_EN: 4.0,
    ContentKind.PROSE_JA: 1.6,
    ContentKind.CODE: 3.2,
    ContentKind.LOG: 3.4,
}
"""1 トークンあたりの文字数の初期値。`bench calibrate` で測り直す。"""


class Profile(_Frozen):
    """計測の設定 (`quick` と `full`)。解決済みの形で計測ランに記録する (1.6)。"""

    name: str = Field(min_length=1)
    seed: int = 0
    sampling: Sampling = Sampling()
    chars_per_token: dict[ContentKind, float] = Field(
        default_factory=lambda: dict(_DEFAULT_CHARS_PER_TOKEN)
    )
    timeout: TimeoutPolicy = TimeoutPolicy()
    min_successes: PositiveInt = 10
    max_consecutive_failures: PositiveInt = 5
    length_tolerance: Tolerance = 0.05
    compare_tolerance: Tolerance = 0.02
    metrics_interval_s: PositiveFloat = 1.0
    output_sanity: OutputSanity = OutputSanity()
    sandbox: SandboxSettings = SandboxSettings()
    decode: DecodeSettings = DecodeSettings()
    prefill: PrefillSettings = PrefillSettings()
    concurrency: ConcurrencySettings = ConcurrencySettings()
    quality: QualitySettings = QualitySettings()
    agent: AgentSettings = AgentSettings()

    @model_validator(mode="after")
    def _check_chars_per_token(self) -> Self:
        missing = [kind.value for kind in ContentKind if kind not in self.chars_per_token]
        if missing:
            raise ValueError(f"chars_per_token に足りない種類がある: {', '.join(missing)}")
        # nan と inf は `<= 0.0` をすり抜け、文章の組み立てが終わらなくなる
        values = self.chars_per_token.values()
        # `0 < 値 < inf` は、nan でも偽になる (nan との比較は、どれも偽)
        if not all(0.0 < value < float("inf") for value in values):
            raise ValueError("chars_per_token の値は、0 より大きい有限の数である必要がある")
        return self


# --- 要求 ---------------------------------------------------------------


class ToolDef(_Frozen):
    """要求に入れるツールの定義。引数の検証にも使う。"""

    name: str = Field(min_length=1)
    description: str = ""
    input_schema: dict[str, JsonValue] = Field(default_factory=dict)


class TextBlockParam(_Frozen):
    """送る本文のブロック。"""

    type: Literal["text"] = "text"
    text: str


class ToolUseBlockParam(_Frozen):
    """送る、モデルのツール呼び出しのブロック (会話の組み立てに使う)。"""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, JsonValue] = Field(default_factory=dict)


class ToolResultBlockParam(_Frozen):
    """送る、合成のツールの結果のブロック。実際には何も実行しない (6.2)。"""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


MessageContentBlock = Annotated[
    TextBlockParam | ToolUseBlockParam | ToolResultBlockParam,
    Field(discriminator="type"),
]
"""送る 1 つのブロック。"""


class InputMessage(_Frozen):
    """送る 1 つの発話。"""

    role: Literal["user", "assistant"]
    content: list[MessageContentBlock]


class MessagesRequest(_Frozen):
    """`POST /v1/messages` に送る本文。

    `extra` は、対象サーバーごとの項目 (`chat_template_kwargs` など) を入れる。
    送るときは、この写像を本文の最上位に混ぜる。`tool_choice` と
    `stop_sequences` は送らない。
    """

    model: str = Field(min_length=1)
    max_tokens: PositiveInt
    messages: list[InputMessage]
    system: str | None = None
    tools: list[ToolDef] | None = None
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    stream: Literal[True] = True
    extra: dict[str, JsonValue] = Field(default_factory=dict)


# --- 応答 ---------------------------------------------------------------


class Usage(_Frozen):
    """対象サーバーが返したトークン数。"""

    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cache_read_input_tokens: NonNegativeInt | None = None
    cache_creation_input_tokens: NonNegativeInt | None = None

    @property
    def total_input_tokens(self) -> int:
        """入力の全長。内訳が来なかった項は 0 として足す (3.2)。"""
        return (
            self.input_tokens
            + (self.cache_read_input_tokens or 0)
            + (self.cache_creation_input_tokens or 0)
        )


class StreamTiming(_Frozen):
    """1 つの要求の時刻。経過の計算には単調な時計 (`perf_counter_ns`) を使う。"""

    sent_at_utc: datetime
    sent_at_ns: int
    message_start_ns: int | None = None
    first_token_ns: int | None = None
    first_text_ns: int | None = None
    last_token_ns: int | None = None
    end_ns: int
    event_count: NonNegativeInt = 0

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        """来なかった時刻は飛ばして、残りの並びが前後していないことを確かめる。"""
        stages: list[tuple[str, int | None]] = [
            ("sent_at_ns", self.sent_at_ns),
            ("message_start_ns", self.message_start_ns),
            ("first_token_ns", self.first_token_ns),
            ("first_text_ns", self.first_text_ns),
            ("last_token_ns", self.last_token_ns),
            ("end_ns", self.end_ns),
        ]
        known = [(name, value) for name, value in stages if value is not None]
        for (before, earlier), (after, later) in zip(known, known[1:], strict=False):
            if earlier > later:
                raise ValueError(f"{after} ({later}) が {before} ({earlier}) より前になっている")
        return self


class ContentBlock(_Frozen):
    """受け取った 1 つのブロック。ツールの引数は、解析の前後の両方を持つ。"""

    type: Literal["text", "thinking", "tool_use"]
    text: str | None = None
    tool_name: str | None = None
    tool_input_raw: str | None = None
    tool_input: dict[str, JsonValue] | None = None


class RequestError(_Frozen):
    """要求の失敗。例外ではなく、この値として返す (10.6)。"""

    kind: RequestErrorKind
    http_status: int | None = None
    message: str = ""


class StreamResult(_Frozen):
    """1 つの要求の結果。認証の情報は含まない (1.8)。"""

    timing: StreamTiming
    usage: Usage | None = None
    stop_reason: str | None = None
    blocks: list[ContentBlock] = Field(default_factory=list)
    server_model: str | None = None
    error: RequestError | None = None


class PreflightOk(_Frozen):
    """計測の前提が満たされたときの、対象サーバーの情報 (1.3)。"""

    server_model: str | None = None
    server_version: str | None = None
    context_limit: int | None = Field(default=None, ge=1)
    running_requests: int | None = Field(default=None, ge=0)


class PreflightFailure(_Frozen):
    """満たされていない前提 (1.4)。計測ランのディレクトリは作らない。"""

    unmet: UnmetPrecondition
    detail: str = ""


# --- 内部の指標 ---------------------------------------------------------


class MetricSnapshot(_Frozen):
    """ある時点の `/metrics`。加工する前の形も残す (7.5)。"""

    taken_at_utc: datetime
    raw_text: str
    values: dict[LogicalMetric, float] = Field(default_factory=dict)
    missing: list[LogicalMetric] = Field(default_factory=list)


class MetricsUnavailable(_Frozen):
    """内部の指標が得られなかったこと。計測は止めない (7.4)。"""

    reason: str


class DerivedMetrics(_Frozen):
    """2 つの時点の増分から出す値。分母が 0 のときは値なしにする (7.3)。"""

    spec_acceptance_rate: float | None = None
    mean_acceptance_length: float | None = None
    decode_steps: float | None = None
    tokens_per_step: float | None = None
    prefix_cache_hit_rate: float | None = None
    kv_usage_peak: float | None = None
    preemptions: float | None = None
    missing: list[LogicalMetric] = Field(default_factory=list)


# --- 課題と採点 ---------------------------------------------------------


class ToolTask(_Frozen):
    """正解の決まったツール呼び出しの課題 (5.1)。"""

    prompt: str
    tools: list[ToolDef]
    expected_tool: str
    expected_input: dict[str, JsonValue] = Field(default_factory=dict)


class ConversationPrefix(_Frozen):
    """takt の作業を模した会話の前置き (6.1)。段階をまたいで先頭が一致する。"""

    system: str
    tools: list[ToolDef]
    messages: list[InputMessage]
    approx_tokens: NonNegativeInt
    conversation_seed: int


class NeedleCase(_Frozen):
    """長い入力から情報を探す課題 (5.3)。"""

    haystack: str
    question: str
    answer: str
    target_tokens: PositiveInt
    depth_pct: DepthPct
    index: NonNegativeInt
    seed: int
    inserted_char_offset: NonNegativeInt


class CodeProblem(_Frozen):
    """公開のコードの課題の 1 問 (5.2)。"""

    task_id: str
    prompt: str
    entry_point: str
    test: str
    canonical_solution: str | None = None


class DatasetRef(_Frozen):
    """公開の課題の出どころ。計測ランと要約に記録する (5.5、11.2)。"""

    name: str
    version: str
    source_url: str
    license: str
    scoring_method: str
    sha256: str


class SandboxResult(_Frozen):
    """隔離して動かした結果 (5.4)。"""

    passed: bool
    timed_out: bool
    exit_code: int | None = None
    stderr_tail: str = ""


class SandboxUnavailable(_Frozen):
    """隔離の実行環境がないこと。隔離なしでは動かさない (5.4)。"""

    reason: str


class ToolCallVerdict(_Frozen):
    """ツール呼び出しの分類の結果 (6.3)。"""

    kind: Literal["tool_call"] = "tool_call"
    outcome: ToolCallOutcome
    detail: str = ""


class QualityVerdict(_Frozen):
    """品質の検査の採点の結果 (5.1、5.2、5.3、5.7)。"""

    kind: Literal["quality"] = "quality"
    task: QualityTaskKind
    outcome: QualityOutcome
    detail: str = ""
    sandbox: SandboxResult | None = None


Verdict = Annotated[ToolCallVerdict | QualityVerdict, Field(discriminator="kind")]
"""試行に付く判定。`kind` で 2 つを読み分ける。"""


# --- 計測ラン -----------------------------------------------------------


class SkippedCondition(_Frozen):
    """飛ばした条件と、その理由 (3.6、6.9)。連続の失敗には数えない。"""

    suite: SuiteName
    key: str
    reason: str


class ConditionPlan(_Frozen):
    """1 つの条件の計画。同じ条件のすべての試行が、この設定を使う (2.7)。"""

    suite: SuiteName
    key: str = Field(min_length=1)
    tier: Tier = "primary"
    trials: NonNegativeInt
    warmup_trials: NonNegativeInt = 0
    sampling: Sampling
    max_tokens: PositiveInt
    concurrency: PositiveInt = 1
    target_input_tokens: int | None = Field(default=None, ge=1)


class TrialRecord(_Frozen):
    """1 つの試行のレコード。書いたあとは変えず、追記だけする (8.1、8.7)。"""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    suite: SuiteName
    condition: str
    tier: Tier = "primary"
    trial_index: NonNegativeInt
    warmup: bool = False
    round_id: int | None = None
    stream_index: int | None = None
    request_body_ref: str
    result: StreamResult
    flags: list[TrialFlag] = Field(default_factory=list)
    verdict: Verdict | None = None
    target_input_tokens: int | None = None


class RunManifest(_Frozen):
    """計測ランの実行の条件と状態 (1.6)。要求と応答の本文は入れない (8.3)。"""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    status: RunStatus
    target: TargetDef
    server_model: str | None = None
    server_version: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    suites: list[SuiteName]
    profile_name: str
    profile: Profile
    harness_version: str
    generator_version: int = GENERATOR_VERSION
    context_limit: int | None = None
    warnings: list[str] = Field(default_factory=list)
    skipped: list[SkippedCondition] = Field(default_factory=list)
    datasets: list[DatasetRef] = Field(default_factory=list)


class RunRequest(_Frozen):
    """計測の実行の指示 (1.1、1.2)。"""

    target_name: str
    suites: list[SuiteName]
    profile_name: str
    trials_override: dict[str, int] = Field(default_factory=dict)


class RunOutcome(_Frozen):
    """計測ランの結末。`exit_code` は 0、2、130 のいずれかになる。"""

    run_id: str
    status: RunStatus
    run_dir: Path
    exit_code: int


# --- 分析 ---------------------------------------------------------------


class Describe(_Frozen):
    """連続の値の記述統計 (2.4、3.7)。"""

    n: NonNegativeInt
    mean: float
    median: float
    min: float
    max: float
    stdev: float | None = None
    iqr: float | None = None
    cv: float | None = None


class ProportionStat(_Frozen):
    """割合と、正確な二項の区間 (5.6、6.5)。"""

    numerator: NonNegativeInt
    denominator: NonNegativeInt
    rate: float
    ci95_low: float
    ci95_high: float
    upper95_one_sided: float

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        if self.numerator > self.denominator:
            raise ValueError("numerator が denominator より大きい")
        return self


class DiffVerdict(_Frozen):
    """2 組の連続の値の差の判定 (9.2)。区間は再標本化で出す。"""

    verdict: DiffOutcome
    paired: bool
    median_diff: float
    relative_diff: float | None = None
    ci95_low: float
    ci95_high: float
    tolerance: float
    n: NonNegativeInt


class MetricResult(_Frozen):
    """条件と値ごとの、要約の 1 行。"""

    condition: str
    metric: str
    tier: Tier
    continuous: Describe | None = None
    proportion: ProportionStat | None = None
    failures: NonNegativeInt = 0
    flags: list[MetricFlag] = Field(default_factory=list)
    flag_counts: dict[str, int] = Field(default_factory=dict)


class AgentStageResult(_Frozen):
    """長い会話の検査の、1 段階の結果 (6.4)。"""

    stage_key: str
    target_input_tokens: PositiveInt
    actual_input_tokens: Describe | None = None
    trials: NonNegativeInt
    outcome_counts: dict[ToolCallOutcome, int] = Field(default_factory=dict)
    request_failures: NonNegativeInt = 0
    break_rate: ProportionStat | None = None
    verdict: ThresholdVerdict = ThresholdVerdict.UNDETERMINED


class AgentSummary(_Frozen):
    """長い会話の検査の要約 (6.4、6.6、6.9)。"""

    threshold: float
    stages: list[AgentStageResult] = Field(default_factory=list)
    first_exceeded_tokens: int | None = None
    reached_tokens: int | None = None
    stopped_reason: str | None = None


class Summary(_Frozen):
    """要約 (8.2、8.6)。送った内容と応答の本文を入れる項目を持たない (8.3)。"""

    schema_version: int = SCHEMA_VERSION
    conditions: RunManifest
    incomplete: bool
    results: list[MetricResult] = Field(default_factory=list)
    agent: AgentSummary | None = None
    server_metrics: dict[str, DerivedMetrics] = Field(default_factory=dict)
    datasets: list[DatasetRef] = Field(default_factory=list)


class ComparisonRow(_Frozen):
    """比較の 1 行。両方の値、差、差の割合、判定 (9.1、9.2、9.6)。"""

    condition: str
    metric: str
    tier: Tier
    value_a: float | None = None
    value_b: float | None = None
    diff: float | None = None
    relative_diff: float | None = None
    verdict: DiffVerdict | None = None
    proportion_verdict: ProportionDiffOutcome | None = None


class RepeatabilityConclusion(_Frozen):
    """同じ対象サーバーの定義どうしの比較の結論 (9.3)。"""

    all_within: bool
    outside: list[str] = Field(default_factory=list)


class ComparisonReport(_Frozen):
    """2 つの計測ランの比較の結果。警告を先頭に出す (9.4、10.5)。"""

    schema_version: int = SCHEMA_VERSION
    run_a: str
    run_b: str
    warnings: list[str] = Field(default_factory=list)
    rows: list[ComparisonRow] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    repeatability: RepeatabilityConclusion | None = None
