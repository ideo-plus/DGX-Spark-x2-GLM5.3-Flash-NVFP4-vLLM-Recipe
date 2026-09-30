"""生データから、速さと割合の要約を作る (task 4.1、4.2: analysis/summarize)。

`summarize_run(run_dir)` が計測ランの生データを読んで `Summary` を組み立て、
`write_summary(run_dir)` がそれを `summary.json` (道具が読む) と `summary.md`
(人が読む) として計測ランのディレクトリに書く (8.2)。

## 契約 (design.md analysis/summarize、Batch)

- **入力**: `RunStore.open(run_dir)` の読み取りだけ。対象サーバーにも
  ネットワークにも触れない。`schema_version` が違えば `RunStore.open` が
  `StoreError` を投げる
- **出力**: `summary.json` と `summary.md`。別名で書いてから置き換える
- **何度実行しても同じ結果になる**: 生成の時刻を入れず、並びを固定し、小数の
  書き方も固定してあるので、同じ生データに対して 2 回書けばバイト単位で同じ
  ファイルになる
- **未完了の計測ランも要約できる** (10.4)。状態が `completed` でなければ
  `Summary.incomplete` を真にし、`summary.md` の先頭に未完了の断りを出す
  (状態が `running` のままなのは、計測ランが落ちた跡である)
- **行の並び**: 条件は `trials.jsonl` に最初に現れた順 (= 実行した順)、条件の
  中の値は `_SUITE_METRICS` の並び (割合の行は、そのあと 1 行)。どちらも生
  データだけで決まるので、同じ生データからは必ず同じ並びになる

## 比較 (4.3) との共有

`trial_values(records, manifest)` が、試行ごとの値を
`{(条件, 値の名前): [TrialValue, ...]}` の形で返す。`TrialValue` は
`trial_index`、`stream_index`、`value` を持つので、対応のある再標本化
(9.2) に、そのまま使える。要約の集計も**この同じ道**を通る (`describe` を
かける前の値が、この関数が返すものと同じ)。速さの定義が 2 か所に分かれない
ようにするためなので、比較の側は式を書き写さないこと。

## 警告の受け渡し (Implementation Notes 2.3)

`read_trials()` / `read_metric_deltas()` は、末尾が途中で切れた行を捨てたとき
に警告を返す。`Summary` にはその置き場所がないので、`summarize_run` は
`SummaryResult` (要約と警告の組) を返し、警告を `summary.md` の先頭にも出す。
コマンドの入口 (5.1) は `SummaryResult.warnings` をそのまま表示する。警告には、
値にできなかった試行や、有限でない内部の指標を捨てたことも足す (黙って落とす
と、要約が静かに歪む)。

## 速さの定義 (design.md「速さの定義」が固定したもの)

| 値 | 式 |
|---|---|
| `ttft_s` | `(first_token_ns − sent_at_ns) / 1e9` |
| `decode_tps` | `(output_tokens − 1) ÷ ((last_token_ns − first_token_ns) / 1e9)` |
| `prefill_tps` | `usage.total_input_tokens ÷ ttft_s` |
| `round_total_tps` | 1 回ぶんの `Σ output_tokens ÷ (回の幅 / 1e9)` |

回の幅は、その回の `max(last_token_ns) − min(first_token_ns)` である。

- 集計に入れるのは、**成功して、慣らしでない**試行だけ (2.5、10.1)
- `output_tokens` が 16 未満の試行は `decode_tps` から外す (分母が小さすぎて
  値が荒れる)。その試行にも `ttft_s` はある。件数は
  `flag_counts["too_few_output_tokens"]` に出す
- `prefill_tps` の分子は、キャッシュに当たった分を含む入力の全長である。効く
  条件 (`prefill/warm/*`) では、実際に計算した量より大きくなるので、実効の
  速さとして読む (その旨を `summary.md` に書く)
- 同時処理は、`(condition, warmup, round_id)` で 1 回ぶんにまとめる (注 3.4)。
  慣らしと本番は、それぞれ 0 から番号を振るので (`iter_trials`)、`round_id` は
  必ずぶつかる。鍵に `warmup` を入れるのは、そのためである。慣らしの回は、
  値にしない
- 1 本が失敗した回も、成功した本から値を出し、条件に `PARTIAL_FAILURES` を
  付ける (4.5)。中断で本数がそろわなかった回も、残っている本から出す (計測ラン
  自体が未完了として表示されるので、少ない本数のまま隠さない)
- 慣らしと失敗を外すのは、値を出す関数の中である (呼び出し側の 1 行に任せない)。
  どの値も、その条件の**すべての**レコードを受け取って、自分で選ぶ

## 試行の数が足りない印 (10.2)

`INSUFFICIENT_TRIALS` は、**行ごとに**判定する。その行の値の数
(`MetricResult.continuous.n`、値がなければ 0) が `Profile.min_successes` に
届かなければ付ける。行によって数える単位が違う (同時処理の `round_total_tps`
は回の数、`ttft_s` は本の数) ので、条件の成功した試行の数で一律に判定すると、
回の数が足りない行に印が付かないまま出る。`decode_tps` は、16 トークン未満を
除いたあとの数で判定する。

## 割合で表す結果 (task 4.2)

`quality` と `agent` のレコードは、連続の値ではなく**割合**にする。行は
`MetricResult.proportion` に入り、`continuous` は空になる。

| まとまり | 値の名前 | 分子 | 分母 |
|---|---|---|---|
| `quality` | `accuracy` | 正解と判定された試行 | 採点できた試行 |
| `agent` | `agent_break_rate` | 崩れた 7 種類 | 試行 − 要求の失敗 |

- **採点できなかった試行は、分母から外す** (5.7、6.4)。品質の検査では
  `QualityOutcome.NOT_SCORED` と `ToolCallOutcome.REQUEST_FAILED`、長い会話の
  検査では `ToolCallOutcome.REQUEST_FAILED` が、それに当たる。件数は
  `MetricResult.flag_counts["not_scored"]` に出す (不正解と混ぜない)
- **判定のない試行**は、まとまりの側の契約違反である。採点できなかったものと
  して数え、**必ず警告にする** (黙って落とすと、割合が静かに良くなる)
- `INSUFFICIENT_TRIALS` は、割合の行には付けない。不確かさは区間そのものが
  表しているので、印で二重に示さない (「表の読み方」に書いてある)
- 割合の表 (品質の検査、段階) にも、速さの表と同じ**印の欄**を出す。出力が
  壊れている疑いの件数 (10.7) は、条件ごとに数えてあるのに、欄がないと人に
  届かない
- 区間は、正確な二項の区間 (`binomial_interval`)。しきい値の判定
  (`threshold_verdict`) は、「下回った」を片側 95% の上限で、「上回った」を
  両側 95% の下限で決める (2.4)
- 段階の表 (`Summary.agent`) は、会話の長さの順に並べる。長さは、条件の鍵の
  末尾 (`agent/stage/020k` → 20000) から読み、読めなければレコードの
  `target_input_tokens` を使う。どちらもなければ、その段階を表から外して
  警告する (行そのものは残すので、4.3 は比べられる)
- 「しきい値を初めて超えた長さ」(6.6) は、**点の推定** (崩れた割合そのもの)
  がしきい値より大きい、いちばん短い段階の、実際の入力のトークン数の中央値で
  ある。試行の数が足りているかは、段階ごとの判定の欄が示す

## まとまりが書き込むレコードの形 (6.7、7.2 への申し送り)

この module は、次の形を当てにして集計する。品質の検査 (6.7) と長い会話の検査
(7.2) は、この形で `TrialRecord` を書くこと。

| 項目 | `quality` | `agent` |
|---|---|---|
| `suite` | `SuiteName.QUALITY` | `SuiteName.AGENT` |
| `condition` | 下の一覧 | `agent/stage/{020k..120k}` |
| `verdict` | `toolcall` は `ToolCallVerdict`、ほかは `QualityVerdict` | `ToolCallVerdict` |
| `target_input_tokens` | (任意) | 段階の狙いの長さ |
| `result.usage` | (任意) | 実際の会話の長さを出すので、必ず入れる |

`quality` の条件の鍵は、`quality/toolcall`、`quality/code/humaneval+`、
`quality/needle/{長さ}/d{位置}` である (design.md suites の表)。

要求が失敗した試行にも、判定を付けること (`REQUEST_FAILED` か `NOT_SCORED`)。
付いていない試行は、採点できなかったものとして数え、警告に出す。

依存の向き (design.md) により、この module が読み込んでよいのは標準ライブラリ、
pydantic、`bench_harness.types`、`bench_harness.store`、
`bench_harness.analysis.stats` だけである。`client`、`suites`、`runner`、
`metrics`、`corpus`、`scoring`、`config`、`cli` は読み込まない (試験が、
構文木を見て機械的に確かめている)。
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from bench_harness.analysis.output_phases import phase_metrics
from bench_harness.analysis.stats import (
    binomial_interval,
    describe,
    threshold_verdict,
    trials_needed_for_zero_failures,
)
from bench_harness.store.rawstore import MetricDeltaRecord, RunStore
from bench_harness.types import (
    AgentStageResult,
    AgentSummary,
    DerivedMetrics,
    Describe,
    MetricFlag,
    MetricResult,
    ProportionStat,
    QualityOutcome,
    RunManifest,
    RunStatus,
    SuiteName,
    Summary,
    ThinkingMode,
    ThresholdVerdict,
    Tier,
    ToolCallOutcome,
    ToolCallVerdict,
    TrialFlag,
    TrialRecord,
    _RetokenizationCounts,
)

__all__ = [
    "METRIC_ACCURACY",
    "METRIC_AGENT_BREAK_RATE",
    "METRIC_DECODE_TPS",
    "METRIC_PREFILL_TPS",
    "METRIC_ROUND_TOTAL_TPS",
    "METRIC_TTFT",
    "MIN_SPEED_OUTPUT_TOKENS",
    "NOT_SCORED_COUNT",
    "SUMMARY_JSON_NAME",
    "SUMMARY_MD_NAME",
    "SummaryResult",
    "TrialValue",
    "render_json",
    "render_markdown",
    "summarize_run",
    "trial_values",
    "write_summary",
]

SUMMARY_JSON_NAME: Final[str] = "summary.json"
SUMMARY_MD_NAME: Final[str] = "summary.md"

MIN_SPEED_OUTPUT_TOKENS: Final[int] = 16
"""これ未満の出力のトークン数は、生成速度の集計に入れない (design.md「速さの定義」)。

`suites/base` の同じ名前の定数と揃える。分析は `suites` を読み込めない (依存の
向き) ので、値をここに写してある。片方だけ変えないこと。
"""

METRIC_TTFT: Final[str] = "ttft_s"
METRIC_DECODE_TPS: Final[str] = "decode_tps"
METRIC_PREFILL_TPS: Final[str] = "prefill_tps"
METRIC_ROUND_TOTAL_TPS: Final[str] = "round_total_tps"
METRIC_ACCURACY: Final[str] = "accuracy"
METRIC_AGENT_BREAK_RATE: Final[str] = "agent_break_rate"
METRIC_TEXT_CHARS_PER_S: Final[str] = "text_chars_per_s"
METRIC_THINKING_CHARS_PER_S: Final[str] = "thinking_chars_per_s"
METRIC_TEXT_WAIT: Final[str] = "text_wait_s"
METRIC_THINKING_DURATION: Final[str] = "thinking_duration_s"
METRIC_TEXT_DURATION: Final[str] = "text_duration_s"
METRIC_THINKING_CHARS: Final[str] = "thinking_chars"
METRIC_TEXT_CHARS: Final[str] = "text_chars"
METRIC_THINKING_RETOKENIZED_TPS: Final[str] = "thinking_retokenized_tps"
METRIC_TEXT_RETOKENIZED_TPS: Final[str] = "text_retokenized_tps"

RETOKENIZED_METRIC_NAMES: Final[tuple[str, ...]] = (
    METRIC_THINKING_RETOKENIZED_TPS,
    METRIC_TEXT_RETOKENIZED_TPS,
)

PHASE_METRIC_NAMES: Final[tuple[str, ...]] = (
    METRIC_TEXT_WAIT,
    METRIC_THINKING_DURATION,
    METRIC_TEXT_DURATION,
    METRIC_THINKING_CHARS,
    METRIC_TEXT_CHARS,
    METRIC_THINKING_CHARS_PER_S,
    METRIC_TEXT_CHARS_PER_S,
    *RETOKENIZED_METRIC_NAMES,
)

NOT_SCORED_COUNT: Final[str] = "not_scored"
"""採点できなかった件数を入れる `MetricResult.flag_counts` の鍵 (5.7、注 1.2)。"""

_NS_PER_S: Final[float] = 1e9

_SUITE_METRICS: Final[dict[SuiteName, tuple[str, ...]]] = {
    SuiteName.DECODE: (METRIC_DECODE_TPS, METRIC_TTFT, *PHASE_METRIC_NAMES),
    SuiteName.PREFILL: (METRIC_TTFT, METRIC_PREFILL_TPS),
    SuiteName.CONCURRENCY: (
        METRIC_DECODE_TPS,
        METRIC_ROUND_TOTAL_TPS,
        METRIC_TTFT,
        *PHASE_METRIC_NAMES,
    ),
}
"""まとまりごとに出す**連続の値**と、その並び。ここにないまとまりは出さない。"""

_SUITE_PROPORTION_METRIC: Final[dict[SuiteName, str]] = {
    SuiteName.QUALITY: METRIC_ACCURACY,
    SuiteName.AGENT: METRIC_AGENT_BREAK_RATE,
}
"""まとまりごとに出す**割合**の値 (4.2)。条件につき 1 行で、連続の行のあとに出る。

`_SUITE_METRICS` と分けてあるのは、`trial_values` (4.3 が使う) が、試行ごとの
連続の値だけを返す約束だからである。割合は試行ごとの値にならないので、比較は
`MetricResult.proportion` の側で行う。
"""

_PROPORTION_METRICS: Final[frozenset[str]] = frozenset(_SUITE_PROPORTION_METRIC.values())

_BROKEN_OUTCOMES: Final[tuple[ToolCallOutcome, ...]] = tuple(
    outcome
    for outcome in ToolCallOutcome
    if outcome not in (ToolCallOutcome.CORRECT, ToolCallOutcome.REQUEST_FAILED)
)
"""「崩れた」と数える 7 種類 (design.md scoring/toolcall)。

`CORRECT` は崩れていない。`REQUEST_FAILED` はモデルの崩れではないので、崩れ
にも分母にも入れない。
"""


@dataclass(frozen=True)
class SummaryResult:
    """`summarize_run` の返り値。要約と、生データを読んだときの警告の組。

    `Summary` には警告の置き場所がないので、ここで一緒に返す (module の
    docstring「警告の受け渡し」)。
    """

    summary: Summary
    warnings: list[str]


@dataclass(frozen=True)
class TrialValue:
    """試行 1 つぶんの値。対応のある比較 (9.2) が、番号で突き合わせるのに使う。

    `stream_index` は、同時処理の 1 本ごとの値だけが持つ。1 回ぶんの合計
    (`round_total_tps`) は、`trial_index` に回の番号を入れ、`stream_index` は
    `None` にする。
    """

    trial_index: int
    stream_index: int | None
    value: float


@dataclass(frozen=True)
class _MetricValues:
    """1 つの条件・1 つの値の、試行ごとの値と、値にできなかった試行の数。"""

    values: list[TrialValue]
    unusable: int

    def numbers(self) -> list[float]:
        return [item.value for item in self.values]


@dataclass(frozen=True)
class _ConditionValues:
    """1 つの条件の、すべてのレコードと、値の名前ごとの試行ごとの値。

    `quality` と `agent` の条件では、`values` は空になる (割合は試行ごとの値に
    ならない)。行は `_proportion_row` が作る。
    """

    suite: SuiteName
    condition: str
    records: list[TrialRecord]
    values: dict[str, _MetricValues]


@dataclass(frozen=True)
class _ScoreCounts:
    """1 つの品質の検査の条件の、採点の内訳 (5.6、5.7)。"""

    correct: int
    incorrect: int
    not_scored: int

    @property
    def scored(self) -> int:
        """採点できた試行の数 (割合の分母)。"""
        return self.correct + self.incorrect


@dataclass(frozen=True)
class _StageCounts:
    """1 つの `agent` の条件の、分類ごとの件数と崩れた割合 (6.4、6.5、6.7)。

    `length` は、その段階の会話の長さ (狙いの入力のトークン数)。決められな
    かったときは `None` で、段階の表からは外れる (行は残る)。
    """

    condition: str
    length: int | None
    trials: int
    outcome_counts: dict[ToolCallOutcome, int]
    request_failures: int
    broken: int
    scored: int
    break_rate: ProportionStat | None
    actual_input_tokens: Describe | None


def trial_values(
    records: Sequence[TrialRecord], manifest: RunManifest
) -> dict[tuple[str, str], list[TrialValue]]:
    """試行ごとの値を、`{(条件, 値の名前): [TrialValue, ...]}` で返す (4.3 が使う)。

    要約の集計も、この同じ値から作る (module の docstring「比較との共有」)。
    返すのは、**成功して、慣らしでない**試行の値だけで、要約の連続の行と同じ
    並びになる。`quality` と `agent` のレコードは、試行ごとの連続の値を持たない
    ので、何も返さない (割合の比較は `MetricResult.proportion` から行う)。

    `records` は `manifest` の計測ランのものに限る (違う計測ランのレコードを
    混ぜたまま対応のある比較を組み立てると、番号だけが合って中身が食い違う)。
    """
    for record in records:
        if record.run_id != manifest.run_id:
            raise ValueError(
                "trial_values: ほかの計測ランのレコードが混ざっている "
                f"(manifest の run_id={manifest.run_id!r}, レコードの run_id={record.run_id!r})"
            )
    return {
        (data.condition, metric): metric_values.values
        for data in _condition_values(records)
        for metric, metric_values in data.values.items()
    }


# --- 入口 -------------------------------------------------------------------


def summarize_run(run_dir: Path) -> SummaryResult:
    """計測ランの生データを読んで、要約を組み立てる (2.3〜2.6、4.2、7.3、8.6)。

    生データに対して純粋である (読むだけで、何も書かず、ネットワークにも触れ
    ない)。本物の破損 (途中の行が壊れている、`schema_version` が違う) は
    `StoreError` のまま外へ出す。
    """
    store = RunStore.open(run_dir)
    manifest = store.manifest()
    trials, trial_warnings = store.read_trials()
    deltas, delta_warnings = store.read_metric_deltas()

    notes: list[str] = []
    stages = _stage_counts_by_condition(trials, notes)
    results = _summarize_trials(trials, manifest.profile.min_successes, stages, notes)
    output_phases, phase_warnings = _phase_conditions(trials, manifest.profile.sampling.thinking)
    recount_counts = _recount_conditions(trials)
    server_metrics = _server_metrics(deltas, notes)
    summary = Summary(
        conditions=manifest,
        incomplete=manifest.status is not RunStatus.COMPLETED,
        results=results,
        agent=_agent_summary(manifest, stages),
        server_metrics=server_metrics,
        datasets=list(manifest.datasets),
        output_phases=output_phases,
        phase_warnings=phase_warnings,
        output_retokenization_counts=recount_counts,
    )
    return SummaryResult(summary=summary, warnings=[*trial_warnings, *delta_warnings, *notes])


def _recount_conditions(trials: Sequence[TrialRecord]) -> dict[str, _RetokenizationCounts]:
    groups: dict[str, list[TrialRecord]] = {}
    for record in trials:
        if record.warmup or record.suite not in (SuiteName.DECODE, SuiteName.CONCURRENCY):
            continue
        if record.result.output_token_counts is not None:
            groups.setdefault(record.condition, []).append(record)
    result: dict[str, _RetokenizationCounts] = {}
    for condition, records in groups.items():
        missing: dict[str, int] = {}
        thinking_counted = 0
        text_counted = 0
        for record in records:
            counts = record.result.output_token_counts
            assert counts is not None
            if counts.thinking.count is not None:
                thinking_counted += 1
            elif counts.thinking.reason is not None:
                key = f"thinking:{counts.thinking.reason}"
                missing[key] = missing.get(key, 0) + 1
            if counts.text.count is not None:
                text_counted += 1
            elif counts.text.reason is not None:
                key = f"text:{counts.text.reason}"
                missing[key] = missing.get(key, 0) + 1
        result[condition] = _RetokenizationCounts(
            thinking_counted=thinking_counted,
            text_counted=text_counted,
            missing_reasons=missing,
        )
    return result


def _phase_conditions(
    trials: Sequence[TrialRecord],
    thinking: ThinkingMode,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    groups: dict[str, list[TrialRecord]] = {}
    for record in trials:
        if record.warmup or record.suite not in (SuiteName.DECODE, SuiteName.CONCURRENCY):
            continue
        groups.setdefault(record.condition, []).append(record)
    counts_by_condition: dict[str, dict[str, int]] = {}
    warnings: list[str] = []
    for condition, records in groups.items():
        duration_reasons: dict[str, int] = {}
        counts = {
            "requests": len(records),
            "request_successes": sum(record.result.error is None for record in records),
            "request_failures": sum(record.result.error is not None for record in records),
            "text_arrived": 0,
            "text_not_reached": 0,
            "thinking_only": 0,
            "thinking_observed": 0,
            "no_text": 0,
            "unknown": 0,
            "text_speed_n": 0,
        }
        for record in records:
            phase = phase_metrics(record.result)
            if phase.duration_reason is not None:
                reason = phase.duration_reason
                duration_reasons[reason] = duration_reasons.get(reason, 0) + 1
            if not phase.observed:
                counts["unknown"] += 1
            if phase.status == "text":
                counts["text_arrived"] += 1
            elif phase.status == "thinking_only":
                counts["thinking_only"] += 1
            else:
                counts["no_text"] += 1
            if phase.thinking_observed:
                counts["thinking_observed"] += 1
            if record.result.error is None and phase.text_chars_per_s is not None:
                counts["text_speed_n"] += 1
        not_reached = counts["thinking_only"] + counts["no_text"]
        counts["text_not_reached"] = not_reached
        if not_reached:
            warnings.append(
                f"{condition}: 本文未到達 {not_reached} 件 "
                f"(うち思考のみ {counts['thinking_only']} 件)"
            )
        if counts["unknown"]:
            warnings.append(f"{condition}: 段階別計測情報が不明 {counts['unknown']} 件")
        if thinking == "off" and counts["thinking_observed"]:
            # off を指定したのに思考が出た試行 (issue #131、#130 の thinking_observed を使う)。
            # glm47 のパーサーは off なら思考のブロックを作らないので、glm47 の構成ではこの
            # 警告は出ない。効くのは、off でも思考を分けて返す相手 (enable_thinking を読まない
            # パーサーのサーバーなど) のとき (bench/README.md の制限の 1 番目)
            warnings.append(
                f"{condition}: thinking off なのに思考が出た {counts['thinking_observed']} 件"
            )
        for reason, count in duration_reasons.items():
            warnings.append(f"{condition}: {reason} {count} 件")
        counts_by_condition[condition] = counts
    return counts_by_condition, warnings


def write_summary(run_dir: Path) -> tuple[Path, Path]:
    """`summary.json` と `summary.md` を計測ランのディレクトリに書く (8.2)。

    別名で書いてから置き換えるので、途中で落ちても古い要約が壊れない。同じ
    生データに対しては、何度実行しても同じバイト列になる。
    """
    result = summarize_run(run_dir)
    json_path = run_dir / SUMMARY_JSON_NAME
    md_path = run_dir / SUMMARY_MD_NAME
    _atomic_write_text(json_path, render_json(result.summary))
    _atomic_write_text(md_path, render_markdown(result.summary, result.warnings))
    return json_path, md_path


def render_json(summary: Summary) -> str:
    """道具が読む要約 (`summary.json`) の中身。項目の並びは型の定義順で決まる。"""
    return summary.model_dump_json(indent=2) + "\n"


# --- 条件ごとの集計 ---------------------------------------------------------


def _condition_values(trials: Sequence[TrialRecord]) -> list[_ConditionValues]:
    """条件ごとに、レコードと、値の名前ごとの試行ごとの値を集める。

    条件の並びは、`trials.jsonl` に最初に現れた順 (= 実行した順) にする。
    同じ生データなら必ず同じ並びになる。値を出す関数には、その条件の
    **すべての**レコード (慣らしと失敗を含む) を渡す。慣らしと失敗を外すのは、
    それぞれの関数の中である (外の 1 行に任せると、その 1 行だけが壊れたとき、
    値が黙って変わる)。
    """
    groups: dict[tuple[SuiteName, str], list[TrialRecord]] = {}
    for record in trials:
        if record.suite not in _SUITE_METRICS and record.suite not in _SUITE_PROPORTION_METRIC:
            continue
        groups.setdefault((record.suite, record.condition), []).append(record)

    return [
        _ConditionValues(
            suite=suite,
            condition=condition,
            records=records,
            values={
                metric: _VALUE_FUNCS[metric](records)
                for metric in _condition_metric_names(suite, records)
            },
        )
        for (suite, condition), records in groups.items()
    ]


def _condition_metric_names(suite: SuiteName, records: Sequence[TrialRecord]) -> tuple[str, ...]:
    names = _SUITE_METRICS.get(suite, ())
    if any(record.result.output_phases is not None for record in records):
        if any(record.result.output_token_counts is not None for record in records):
            return names
        return tuple(name for name in names if name not in RETOKENIZED_METRIC_NAMES)
    return tuple(name for name in names if name not in PHASE_METRIC_NAMES)


def _summarize_trials(
    trials: Sequence[TrialRecord],
    min_successes: int,
    stages: dict[str, _StageCounts],
    notes: list[str],
) -> list[MetricResult]:
    """条件ごとに値の行を作る。値は `trial_values` と同じ道を通って来る。"""
    results: list[MetricResult] = []
    for data in _condition_values(trials):
        results.extend(_summarize_condition(data, min_successes, stages, notes))
    return results


def _summarize_condition(
    data: _ConditionValues,
    min_successes: int,
    stages: dict[str, _StageCounts],
    notes: list[str],
) -> list[MetricResult]:
    """1 つの条件の行を作る。値がまったくなくても、行は必ず出す (10.1、10.2)。"""
    suite = data.suite
    condition = data.condition
    records = data.records
    measured = [record for record in records if not record.warmup]
    successes = [record for record in measured if record.result.error is None]
    failures = len(measured) - len(successes)
    flag_counts = {
        flag.value: sum(1 for record in measured if flag in record.flags) for flag in TrialFlag
    }
    tier: Tier = records[0].tier

    rows: list[MetricResult] = []
    for metric in data.values:
        metric_values = data.values[metric]
        if metric_values.unusable:
            notes.append(
                f"{condition} の {metric}: 時刻かトークン数が足りず、値にできなかった試行が "
                f"{metric_values.unusable} 件あったので、集計から外した"
            )
        numbers = metric_values.numbers()
        continuous = describe(numbers) if numbers else None
        rows.append(
            MetricResult(
                condition=condition,
                metric=metric,
                tier=tier,
                continuous=continuous,
                failures=failures,
                flags=_row_flags(
                    suite,
                    value_count=continuous.n if continuous is not None else 0,
                    successes=len(successes),
                    failures=failures,
                    flag_counts=flag_counts,
                    min_successes=min_successes,
                ),
                flag_counts=flag_counts,
                count_source="retokenized" if metric in RETOKENIZED_METRIC_NAMES else None,
            )
        )
    proportion_metric = _SUITE_PROPORTION_METRIC.get(suite)
    if proportion_metric is not None:
        rows.append(
            _proportion_row(
                suite,
                condition=condition,
                metric=proportion_metric,
                tier=tier,
                measured=measured,
                failures=failures,
                flag_counts=flag_counts,
                stages=stages,
                notes=notes,
            )
        )
    return rows


def _proportion_row(
    suite: SuiteName,
    *,
    condition: str,
    metric: str,
    tier: Tier,
    measured: Sequence[TrialRecord],
    failures: int,
    flag_counts: dict[str, int],
    stages: dict[str, _StageCounts],
    notes: list[str],
) -> MetricResult:
    """割合の行を 1 つ作る (5.6、5.7、6.4、6.5)。

    採点できた試行が 1 つもなければ、割合は値なしにして、行だけを残す
    (件数が 0 であることも、結果である)。`INSUFFICIENT_TRIALS` は付けない
    (module の docstring「割合で表す結果」)。

    `failures` は、要求そのものが失敗した試行の数 (ほかの行と同じ数え方)。
    `flag_counts["not_scored"]` は、**分母から外した**試行の数で、品質の検査
    では要求の失敗と採点できなかったものの合計、長い会話の検査では
    `REQUEST_FAILED` の件数になる。
    """
    if suite is SuiteName.AGENT:
        stage = stages.get(condition)
        proportion = stage.break_rate if stage is not None else None
        not_scored = stage.request_failures if stage is not None else len(measured)
    else:
        score = _score_counts(condition, measured, notes)
        proportion = binomial_interval(score.correct, score.scored) if score.scored else None
        not_scored = score.not_scored
    return MetricResult(
        condition=condition,
        metric=metric,
        tier=tier,
        continuous=None,
        proportion=proportion,
        failures=failures,
        flags=_count_flags(flag_counts),
        flag_counts={**flag_counts, NOT_SCORED_COUNT: not_scored},
    )


def _row_flags(
    suite: SuiteName,
    *,
    value_count: int,
    successes: int,
    failures: int,
    flag_counts: dict[str, int],
    min_successes: int,
) -> list[MetricFlag]:
    """1 つの行に付く印 (10.2、4.5、2.6、3.3、10.7)。並びは `MetricFlag` の定義順。

    `INSUFFICIENT_TRIALS` だけは、行ごとの値の数 (`value_count`) で判定する
    (module の docstring「試行の数が足りない印」)。ほかの印は、条件の全体で
    数えたものを、その条件のすべての行に付ける。
    """
    flags: list[MetricFlag] = []
    if value_count < min_successes:
        flags.append(MetricFlag.INSUFFICIENT_TRIALS)
    if suite is SuiteName.CONCURRENCY and failures > 0 and successes > 0:
        # 4.5: 一部が失敗した回ぶん。成功した要求だけから求めた値であることを示す
        flags.append(MetricFlag.PARTIAL_FAILURES)
    return [*flags, *_count_flags(flag_counts)]


def _count_flags(flag_counts: dict[str, int]) -> list[MetricFlag]:
    """件数から決まる印 (2.6、3.3、10.7)。並びは `MetricFlag` の定義順。

    連続の行と割合の行で、同じものを付ける (割合の行に付かないのは、
    `INSUFFICIENT_TRIALS` と `PARTIAL_FAILURES` だけである)。
    """
    flags: list[MetricFlag] = []
    if flag_counts[TrialFlag.SHORT_OUTPUT.value] > 0:
        flags.append(MetricFlag.SHORT_OUTPUTS)
    if flag_counts[TrialFlag.LENGTH_OFF_TARGET.value] > 0:
        flags.append(MetricFlag.LENGTH_OFF_TARGET)
    replacement = flag_counts[TrialFlag.REPLACEMENT_CHAR.value]
    repetition = flag_counts[TrialFlag.REPETITION_LOOP.value]
    if replacement + repetition > 0:
        flags.append(MetricFlag.SUSPECT_OUTPUTS)
    return flags


# --- 品質の検査の採点 (5.6、5.7) --------------------------------------------


def _score_counts(
    condition: str, measured: Sequence[TrialRecord], notes: list[str]
) -> _ScoreCounts:
    """採点の内訳を数える。採点できなかったものを、不正解と混ぜない (5.7)。

    判定の種類は、条件によって 2 つある。`quality/toolcall` は
    `ToolCallVerdict` (正解は `CORRECT` だけ、`REQUEST_FAILED` は採点できな
    かったもの、ほかの 7 種類は不正解)、コードと探す課題は `QualityVerdict`
    である。どちらが来ても読めるようにしてあるので、まとまりの側の取り違えで
    要約が落ちることはない。
    """
    correct = 0
    incorrect = 0
    not_scored = 0
    missing = 0
    for record in measured:
        verdict = record.verdict
        if verdict is None:
            missing += 1
            not_scored += 1
        elif isinstance(verdict, ToolCallVerdict):
            if verdict.outcome is ToolCallOutcome.CORRECT:
                correct += 1
            elif verdict.outcome is ToolCallOutcome.REQUEST_FAILED:
                not_scored += 1
            else:
                incorrect += 1
        elif verdict.outcome is QualityOutcome.CORRECT:
            correct += 1
        elif verdict.outcome is QualityOutcome.NOT_SCORED:
            not_scored += 1
        else:
            incorrect += 1
    if missing:
        notes.append(
            f"{condition}: 採点の判定がない試行が {missing} 件あったので、採点できなかった"
            "ものとして数えた (まとまりが判定を付けていない)"
        )
    return _ScoreCounts(correct=correct, incorrect=incorrect, not_scored=not_scored)


# --- 長い会話の段階 (6.4、6.5、6.6、6.7、6.9) -------------------------------


def _stage_counts_by_condition(
    trials: Sequence[TrialRecord], notes: list[str]
) -> dict[str, _StageCounts]:
    """`agent` の条件ごとに、分類の件数と崩れた割合を数える。

    並びは、`trials.jsonl` に最初に現れた順 (段階の表に並べ替えるのは
    `_agent_summary`)。
    """
    groups: dict[str, list[TrialRecord]] = {}
    for record in trials:
        if record.suite is SuiteName.AGENT:
            groups.setdefault(record.condition, []).append(record)
    return {
        condition: _stage_counts_of(condition, records, notes)
        for condition, records in groups.items()
    }


def _stage_counts_of(
    condition: str, records: Sequence[TrialRecord], notes: list[str]
) -> _StageCounts:
    """1 つの段階を数える。慣らしは入れない (2.5)。

    崩れた割合の分母は、`試行 − 要求の失敗` である。要求そのものの失敗は、
    モデルの崩れではないので分母から外し、件数を別に示す (6.4)。
    """
    measured = [record for record in records if not record.warmup]
    counts = dict.fromkeys(ToolCallOutcome, 0)
    unclassified = 0
    for record in measured:
        outcome = _agent_outcome(record)
        if outcome is None:
            unclassified += 1
            outcome = ToolCallOutcome.REQUEST_FAILED
        counts[outcome] += 1
    if unclassified:
        notes.append(
            f"{condition}: ツール呼び出しの分類がない試行が {unclassified} 件あったので、"
            "崩れた割合の分母から外した (まとまりが判定を付けていない)"
        )
    request_failures = counts[ToolCallOutcome.REQUEST_FAILED]
    scored = len(measured) - request_failures
    broken = sum(counts[outcome] for outcome in _BROKEN_OUTCOMES)
    tokens = [
        float(record.result.usage.total_input_tokens)
        for record in measured
        if record.result.usage is not None
    ]
    return _StageCounts(
        condition=condition,
        length=_stage_length(condition, records, notes),
        trials=len(measured),
        outcome_counts=counts,
        request_failures=request_failures,
        broken=broken,
        scored=scored,
        break_rate=binomial_interval(broken, scored) if scored > 0 else None,
        actual_input_tokens=describe(tokens) if tokens else None,
    )


def _agent_outcome(record: TrialRecord) -> ToolCallOutcome | None:
    """1 つの試行の分類。読み取れなければ `None` (呼び出し側が警告にする)。

    判定がない、または種類が違う (`QualityVerdict` が入っている) のは、
    まとまりの側の契約違反である。要求が失敗していれば `REQUEST_FAILED` と
    見なせるが、そうでなければ、崩れているともいないとも言えないので、分母に
    入れない。
    """
    verdict = record.verdict
    if isinstance(verdict, ToolCallVerdict):
        return verdict.outcome
    if record.result.error is not None:
        return ToolCallOutcome.REQUEST_FAILED
    return None


def _stage_length(condition: str, records: Sequence[TrialRecord], notes: list[str]) -> int | None:
    """段階の会話の長さ (狙いの入力のトークン数) を決める。

    条件の鍵の末尾 (`agent/stage/020k` → 20000) から読み、読めなければ
    レコードの `target_input_tokens` を使う。どちらもなければ `None` にして
    警告する (段階の表からは外れるが、行そのものは残る)。
    """
    from_key = _tokens_from_label(condition.rsplit("/", 1)[-1])
    if from_key is not None:
        return from_key
    for record in records:
        if record.target_input_tokens is not None and record.target_input_tokens > 0:
            return record.target_input_tokens
    notes.append(
        f"{condition}: 会話の長さが決められないので、段階の表から外した "
        "(条件の鍵から読めず、target_input_tokens もない)"
    )
    return None


def _tokens_from_label(label: str) -> int | None:
    """`020k` を 20000 に、`20000` を 20000 にする。読めなければ `None`。"""
    digits = label.removesuffix("k")
    if not digits.isdecimal():
        return None
    value = int(digits) * (1000 if digits != label else 1)
    return value if value > 0 else None


def _agent_summary(manifest: RunManifest, stages: dict[str, _StageCounts]) -> AgentSummary | None:
    """段階の表を組み立てる (6.4、6.6、6.9)。

    レコードも、飛ばした段階もなければ、`None` を返す (長い会話の検査を流して
    いない計測ランに、空の節を出さない)。
    """
    skipped = [item for item in manifest.skipped if item.suite is SuiteName.AGENT]
    if not stages and not skipped:
        return None
    threshold = manifest.profile.agent.threshold
    ordered = sorted(
        ((counts.length, counts) for counts in stages.values() if counts.length is not None),
        key=lambda item: item[0],
    )
    return AgentSummary(
        threshold=threshold,
        stages=[_stage_result(length, counts, threshold) for length, counts in ordered],
        first_exceeded_tokens=_first_exceeded_tokens(ordered, threshold),
        reached_tokens=max(
            (length for length, counts in ordered if counts.scored > 0), default=None
        ),
        stopped_reason=skipped[0].reason if skipped else None,
    )


def _stage_result(length: int, counts: _StageCounts, threshold: float) -> AgentStageResult:
    return AgentStageResult(
        stage_key=counts.condition,
        target_input_tokens=length,
        actual_input_tokens=counts.actual_input_tokens,
        trials=counts.trials,
        outcome_counts=counts.outcome_counts,
        request_failures=counts.request_failures,
        break_rate=counts.break_rate,
        verdict=(
            threshold_verdict(counts.break_rate, threshold)
            if counts.break_rate is not None
            else ThresholdVerdict.UNDETERMINED
        ),
    )


def _first_exceeded_tokens(
    ordered: Sequence[tuple[int, _StageCounts]], threshold: float
) -> int | None:
    """しきい値を初めて超えた会話の長さ (6.6)。

    判定に使うのは**点の推定** (崩れた割合そのもの) で、区間ではない。試行の
    数が足りているかどうかは、段階ごとの判定の欄が示す。長さは、実際の入力の
    トークン数の中央値 (得られなければ狙いの長さ) で示す (6.7)。
    """
    for length, counts in ordered:
        if counts.break_rate is not None and counts.break_rate.rate > threshold:
            actual = counts.actual_input_tokens
            return round(actual.median) if actual is not None else length
    return None


# --- 値の計算 ---------------------------------------------------------------


def _seconds(start: int | None, end: int | None) -> float | None:
    """2 つの単調な時計の値の差を秒にする。片方が来ていなければ値なし。"""
    if start is None or end is None:
        return None
    delta = (end - start) / _NS_PER_S
    return delta if math.isfinite(delta) else None


def _ratio(numerator: float, denominator: float | None) -> float | None:
    """割り算。分母が 0 以下か、結果が有限でなければ値なしにする。

    有限でない値を `describe` に渡すと `ValueError` になり、要約そのものが
    作れなくなる。ここで止めて、件数だけを警告に残す。
    """
    if denominator is None or denominator <= 0.0:
        return None
    try:
        value = numerator / denominator
    except (ZeroDivisionError, OverflowError):  # pragma: no cover - 分母は 0 より大きい
        return None
    return value if math.isfinite(value) else None


def _ttft_s(record: TrialRecord) -> float | None:
    """最初のトークンまでの時間 (秒)。thinking の最初のトークンも数える。"""
    timing = record.result.timing
    return _seconds(timing.sent_at_ns, timing.first_token_ns)


def _decode_tps(record: TrialRecord) -> float | None:
    """生成速度 (トークン/秒)。最初のトークンまでの時間を含めない (2.3)。"""
    usage = record.result.usage
    if usage is None:
        return None
    timing = record.result.timing
    return _ratio(usage.output_tokens - 1, _seconds(timing.first_token_ns, timing.last_token_ns))


def _text_chars_per_s(record: TrialRecord) -> float | None:
    return phase_metrics(record.result).text_chars_per_s


def _phase_value(record: TrialRecord, metric: str) -> float | None:
    value = getattr(phase_metrics(record.result), metric)
    return float(value) if value is not None else None


def _phase_values(metric: str) -> Callable[[Sequence[TrialRecord]], _MetricValues]:
    return lambda records: _collect(records, lambda record: _phase_value(record, metric))


def _prefill_tps(record: TrialRecord) -> float | None:
    """入力の処理速度 (トークン/秒)。分子は、内訳を含む入力の全長 (3.1、3.2)。"""
    usage = record.result.usage
    if usage is None:
        return None
    return _ratio(usage.total_input_tokens, _ttft_s(record))


def _too_few_output_tokens(record: TrialRecord) -> bool:
    """生成速度の集計から、設計どおり外す試行か (`TOO_FEW_OUTPUT_TOKENS`)。"""
    usage = record.result.usage
    return usage is not None and usage.output_tokens < MIN_SPEED_OUTPUT_TOKENS


def _is_measured_success(record: TrialRecord) -> bool:
    """集計に入れる試行か。慣らしでなく、要求が成功したものだけ (2.5、10.1)。"""
    return not record.warmup and record.result.error is None


def _collect(
    records: Sequence[TrialRecord],
    value_of: Callable[[TrialRecord], float | None],
    *,
    skip: Callable[[TrialRecord], bool] | None = None,
) -> _MetricValues:
    """試行ごとの値と、値にできなかった試行の数を返す。

    `records` は、その条件の**すべての**レコード。慣らしと失敗は、ここで外す
    (外の 1 行に任せない)。`skip` は「設計どおりの除外」で、値にできなかった
    数には入れない (別の印で件数を示している)。
    """
    values: list[TrialValue] = []
    unusable = 0
    for record in records:
        if not _is_measured_success(record):
            continue
        if skip is not None and skip(record):
            continue
        value = value_of(record)
        if value is None:
            unusable += 1
        else:
            values.append(
                TrialValue(
                    trial_index=record.trial_index,
                    stream_index=record.stream_index,
                    value=value,
                )
            )
    return _MetricValues(values=values, unusable=unusable)


def _ttft_values(records: Sequence[TrialRecord]) -> _MetricValues:
    return _collect(records, _ttft_s)


def _decode_values(records: Sequence[TrialRecord]) -> _MetricValues:
    return _collect(records, _decode_tps, skip=_too_few_output_tokens)


def _text_chars_values(records: Sequence[TrialRecord]) -> _MetricValues:
    return _collect(records, _text_chars_per_s)


def _prefill_values(records: Sequence[TrialRecord]) -> _MetricValues:
    return _collect(records, _prefill_tps)


def _round_total_values(records: Sequence[TrialRecord]) -> _MetricValues:
    """1 回ぶんの合計の生成速度を、回ごとに出す (4.2、注 3.4)。

    `records` は、その条件のすべてのレコード。回をまとめる鍵は
    `(warmup, round_id が無ければ trial_index)` である。慣らしと本番は、
    それぞれ 0 から番号を振るので (`iter_trials`)、`warmup` を鍵に入れないと、
    慣らしの本が本番の回に混ざり、1 回ぶんの合計が黙って小さくなる。慣らしの
    回は、値にしない。

    要求が失敗した本は、合計にも時刻の幅にも入れない (4.5: 成功した要求だけ
    から求める)。成功した本が 1 つもない回は、値を出さない (失敗の件数に
    数えてある)。
    """
    rounds: dict[tuple[bool, int], list[TrialRecord]] = {}
    for record in records:
        if record.result.error is not None:
            continue
        round_id = record.round_id if record.round_id is not None else record.trial_index
        rounds.setdefault((record.warmup, round_id), []).append(record)

    values: list[TrialValue] = []
    unusable = 0
    for key in sorted(rounds):
        warmup, round_id = key
        if warmup:
            continue  # 慣らしの回 (2.5)
        output_tokens = 0
        firsts: list[int] = []
        lasts: list[int] = []
        for record in rounds[key]:
            usage = record.result.usage
            timing = record.result.timing
            if usage is None or timing.first_token_ns is None or timing.last_token_ns is None:
                continue
            output_tokens += usage.output_tokens
            firsts.append(timing.first_token_ns)
            lasts.append(timing.last_token_ns)
        if not firsts:
            continue
        value = _ratio(output_tokens, _seconds(min(firsts), max(lasts)))
        if value is None:
            unusable += 1
        else:
            values.append(TrialValue(trial_index=round_id, stream_index=None, value=value))
    return _MetricValues(values=values, unusable=unusable)


_VALUE_FUNCS: Final[dict[str, Callable[[Sequence[TrialRecord]], _MetricValues]]] = {
    METRIC_TTFT: _ttft_values,
    METRIC_DECODE_TPS: _decode_values,
    METRIC_TEXT_CHARS_PER_S: _text_chars_values,
    METRIC_THINKING_CHARS_PER_S: _phase_values(METRIC_THINKING_CHARS_PER_S),
    METRIC_TEXT_WAIT: _phase_values(METRIC_TEXT_WAIT),
    METRIC_THINKING_DURATION: _phase_values(METRIC_THINKING_DURATION),
    METRIC_TEXT_DURATION: _phase_values(METRIC_TEXT_DURATION),
    METRIC_THINKING_CHARS: _phase_values(METRIC_THINKING_CHARS),
    METRIC_TEXT_CHARS: _phase_values(METRIC_TEXT_CHARS),
    METRIC_THINKING_RETOKENIZED_TPS: _phase_values(METRIC_THINKING_RETOKENIZED_TPS),
    METRIC_TEXT_RETOKENIZED_TPS: _phase_values(METRIC_TEXT_RETOKENIZED_TPS),
    METRIC_PREFILL_TPS: _prefill_values,
    METRIC_ROUND_TOTAL_TPS: _round_total_values,
}


# --- 内部の指標 (7.3、7.4) --------------------------------------------------


def _server_metrics(
    deltas: Sequence[MetricDeltaRecord], notes: list[str]
) -> dict[str, DerivedMetrics]:
    """条件ごとの導出値。外から来た値なので、有限かどうかをここで確かめる (注 2.7)。"""
    metrics: dict[str, DerivedMetrics] = {}
    for row in deltas:
        derived, dropped = _finite_derived(row.derived)
        if dropped:
            notes.append(
                f"{row.condition} の内部の指標に、有限でない値が {dropped} 件あったので、"
                "値なしにした"
            )
        metrics[row.condition] = derived
    return metrics


def _finite_derived(derived: DerivedMetrics) -> tuple[DerivedMetrics, int]:
    """NaN や無限大の項を値なしにする。JSON に `NaN` や `Infinity` を書かないため。"""
    data = derived.model_dump()
    dropped = 0
    for key, value in list(data.items()):
        if isinstance(value, float) and not math.isfinite(value):
            data[key] = None
            dropped += 1
    if dropped == 0:
        return derived, 0
    return DerivedMetrics.model_validate(data), dropped


# --- 人が読む要約 (summary.md) ----------------------------------------------

_TABLE_HEADER: Final[str] = (
    "| 条件 | 値 | n | 平均 | 中央値 | 最小 | 最大 | 標準偏差 | 四分位範囲 | 変動係数 | 失敗 | 印 |"
)
_TABLE_RULE: Final[str] = "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"

_METRIC_LEGEND: Final[str] = (
    "値の意味: `ttft_s` = 最初のトークンまでの時間 (秒)、"
    "`decode_tps` = 思考等を含む全出力の生成速度 (tok/s、最初のトークンまでの時間を含めない)、"
    "`text_wait_s` = 本文の非空データに到達するまでの秒数、"
    "`thinking_duration_s`・`text_duration_s` = 各段階の非空データの区間 (秒)、"
    "`thinking_chars`・`text_chars` = 各段階の文字数、"
    "`thinking_chars_per_s`・`text_chars_per_s` = 各段階の文字/秒、"
    "`thinking_retokenized_tps`・`text_retokenized_tps` = 生文字列の再計数トークン/秒"
    " (生成時の段階別トークン数ではない)、"
    "`prefill_tps` = 入力の処理速度 (トークン/秒)、"
    "`round_total_tps` = 1 回ぶんの合計の生成速度 (トークン/秒)。"
    "集計に入れたのは、成功した、慣らしでない試行だけである。"
)
_N_NOTE: Final[str] = (
    "`n` の数える単位は行によって違う。`ttft_s`、`decode_tps`、`prefill_tps` は試行の数 "
    "(同時処理では 1 本が 1 つ)、`round_total_tps` は回の数である。"
    "`decode_tps` は、出力が 16 トークン未満の試行を除いたあとの数になる。"
)
_PREFILL_NOTE: Final[str] = (
    "`prefill_tps` の分子は、キャッシュに当たった分を含む入力の全長である。"
    "キャッシュが効く条件 (`prefill/warm/*`) では、実際に計算した量より大きくなるので、"
    "実効の速さとして読むこと。"
)
_ACCURACY_NOTE: Final[str] = (
    "`accuracy` (正解の割合) の分母は、**採点できた試行**だけである。"
    "要求そのものが失敗した試行と、採点できなかった試行 (コードの隔離の実行環境がない、"
    "など) は分母から外し、件数を「採点できなかった」の欄に出す (5.7)。"
    "不正解とは混ぜない。"
)
_BREAK_RATE_NOTE: Final[str] = (
    "崩れた割合 (`agent_break_rate`) は、`correct` と `request_failed` を除いた 7 種類の"
    "件数 ÷ (試行の数 − `request_failed` の件数) である。要求そのものの失敗は"
    "モデルの崩れではないので、分母から外して別に数える (6.4)。"
)
_INTERVAL_NOTE: Final[str] = (
    "割合に添えた区間は、正確な二項の区間 (Clopper-Pearson) の両側 95% である。"
)
_ONE_SIDED_NOTE: Final[str] = (
    "段階の表の「片側 95% の上限」は、しきい値を下回ったと言えるかの判定に使う、別の量である"
    " (両側の区間と同じものとして読まないこと)。"
)
"""片側の上限の説明。**段階の表を出すときだけ**添える (その欄は、そこにしかない)。"""
_VERDICT_NOTE: Final[str] = (
    "しきい値に対する判定は 3 つ。`下回った` = 片側 95% の上限がしきい値より小さい。"
    "`上回った` = 両側 95% の下限がしきい値より大きい。"
    "`試行の数が足りない` = どちらとも言えない (崩れが 0 件なら、下回ったと言うのに"
    "あと何回要るかを添える)。"
)
_PROPORTION_FLAG_NOTE: Final[str] = (
    "割合の行には、`試行の数が足りない` の印を付けない。不確かさは区間そのものが"
    "表しているので、印で二重に示さない。"
)
_EMPTY: Final[str] = "—"

_VERDICT_LABELS: Final[dict[ThresholdVerdict, str]] = {
    ThresholdVerdict.BELOW: "下回った",
    ThresholdVerdict.ABOVE: "上回った",
    ThresholdVerdict.UNDETERMINED: "試行の数が足りない",
}


def _insufficient_note(min_successes: int) -> str:
    """`試行の数が足りない` の判定の決まりを、人が読む要約にも書く (10.2)。"""
    return (
        "`試行の数が足りない` の印は、行ごとに判定する (その行の値の数が、"
        f"成功した試行の最小の数 {min_successes} に届かない)。"
    )


def render_markdown(summary: Summary, warnings: Sequence[str] = ()) -> str:
    """人が読む要約 (`summary.md`) の中身 (8.2、8.6、4.4)。

    送った内容と応答の本文は、`Summary` に入っていないので、ここにも出ない
    (8.3)。生成の時刻を入れないので、同じ要約からは同じ文字列になる。
    """
    manifest = summary.conditions
    lines: list[str] = [f"# 計測ランの要約: {manifest.run_id}", ""]
    if summary.incomplete:
        lines += [
            f"> **この計測ランは未完了である** (状態: `{manifest.status.value}`)。"
            "途中までの結果として読むこと (10.4)。",
            "",
        ]
    lines += _conditions_section(summary)
    lines += _warnings_section(manifest, list(dict.fromkeys([*warnings, *summary.phase_warnings])))
    lines += _output_phases_section(summary)
    lines += _skipped_section(manifest)
    lines += _legend_section(summary)
    lines += _retokenization_section(summary)
    lines += _results_section("主な結果", summary, "primary")
    lines += _results_section("参考", summary, "reference")
    lines += _quality_section(summary)
    lines += _agent_section(summary)
    lines += _server_metrics_section(summary)
    lines += _datasets_section(summary)
    return "\n".join(lines).rstrip("\n") + "\n"


def _retokenization_section(summary: Summary) -> list[str]:
    if not summary.output_retokenization_counts:
        return []
    lines = ["## 出力文字列の再計数", "", "再計数値は生成時のトークン内訳ではない。", ""]
    for condition, counts in summary.output_retokenization_counts.items():
        missing = (
            ", ".join(
                f"{reason} {count} 件" for reason, count in sorted(counts.missing_reasons.items())
            )
            or "なし"
        )
        lines.append(
            f"- `{_cell(condition)}`: count_source={counts.count_source}, "
            f"思考 {counts.thinking_counted} 件、本文 {counts.text_counted} 件、欠測 {missing}"
        )
    lines.append("")
    return lines


def _output_phases_section(summary: Summary) -> list[str]:
    if not summary.output_phases:
        return []
    lines = [
        "## 本文への到達と計測件数",
        "",
        "| 条件 | 要求 | 要求成功 | 要求失敗 | 要求成功率 | 本文あり | 本文未到達 | "
        "思考のみ | 思考あり | 段階別情報不明 | 本文速度の有効件数 |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for condition, counts in summary.output_phases.items():
        requests = counts["requests"]
        successes = counts["request_successes"]
        rate = f"{successes}/{requests}" if requests else "0/0"
        values = (
            f"`{_cell(condition)}`",
            str(requests),
            str(successes),
            str(counts["request_failures"]),
            rate,
            str(counts["text_arrived"]),
            str(counts["text_not_reached"]),
            str(counts["thinking_only"]),
            str(counts["thinking_observed"]),
            str(counts["unknown"]),
            str(counts["text_speed_n"]),
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("本文ありは本文の到達件数であり、JSONの完全性や妥当性の判定ではない。")
    lines.append("思考ありは、思考のブロックに1文字以上が届いた試行の数で、本文の有無を問わない。")
    lines.append("")
    return lines


def _legend_section(summary: Summary) -> list[str]:
    """表の読み方。どの表からも同じなので、1 回だけ出す。

    出てこない欄の説明は書かない。しきい値の判定と片側 95% の上限は、段階の表
    にしかない欄なので、その表を出すときだけ添える。
    """
    notes: list[str] = []
    if any(row.metric not in _PROPORTION_METRICS for row in summary.results):
        notes += [
            _METRIC_LEGEND,
            _N_NOTE,
            _insufficient_note(summary.conditions.profile.min_successes),
        ]
        if any(row.metric == METRIC_PREFILL_TPS for row in summary.results):
            notes.append(_PREFILL_NOTE)
    if any(row.metric == METRIC_ACCURACY for row in summary.results):
        notes.append(_ACCURACY_NOTE)
    has_stage_table = summary.agent is not None and bool(summary.agent.stages)
    if summary.agent is not None:
        notes.append(_BREAK_RATE_NOTE)
        if has_stage_table:
            notes.append(_VERDICT_NOTE)
    if any(row.metric in _PROPORTION_METRICS for row in summary.results):
        notes.append(_INTERVAL_NOTE)
        if has_stage_table:
            notes.append(_ONE_SIDED_NOTE)
        notes.append(_PROPORTION_FLAG_NOTE)
    if not notes:
        return []
    lines = ["## 表の読み方", ""]
    for note in notes:
        lines += [f"- {note}", ""]
    return lines


def _conditions_section(summary: Summary) -> list[str]:
    """要約の先頭に置く、実行の条件 (1.6、8.6)。"""
    manifest = summary.conditions
    target = manifest.target
    profile = manifest.profile
    rows: list[tuple[str, str]] = [
        ("計測ランの識別子", f"`{manifest.run_id}`"),
        ("対象サーバー", f"`{target.name}`"),
        ("メモ", target.notes or _EMPTY),
        ("接続先", str(target.base_url)),
        ("モデル (定義 / サーバーの申告)", f"{target.model} / {manifest.server_model or '(不明)'}"),
        ("サーバーの版", manifest.server_version or "(不明)"),
        ("状態", f"`{manifest.status.value}`" + (" (未完了)" if summary.incomplete else "")),
        ("開始", manifest.started_at.isoformat()),
        ("終了", manifest.finished_at.isoformat() if manifest.finished_at else _EMPTY),
        ("測ったまとまり", ", ".join(suite.value for suite in manifest.suites) or _EMPTY),
        ("計測の設定", f"`{manifest.profile_name}`"),
        ("サンプリング", _sampling_text(summary)),
        (
            "入力の長さの上限",
            f"{manifest.context_limit} トークン" if manifest.context_limit else "(不明)",
        ),
        ("成功した試行の最小の数", str(profile.min_successes)),
        ("道具の版", f"`{manifest.harness_version}`"),
        ("生成器の版", str(manifest.generator_version)),
        ("要約の形の版", str(summary.schema_version)),
    ]
    rows.extend(_suite_settings_rows(manifest))
    lines = ["## 実行の条件", "", "| 項目 | 値 |", "|---|---|"]
    lines += [f"| {_cell(name)} | {_cell(value)} |" for name, value in rows]
    lines.append("")
    return lines


def _sampling_text(summary: Summary) -> str:
    sampling = summary.conditions.profile.sampling
    return (
        f"temperature={sampling.temperature}, "
        f"top_p={sampling.top_p if sampling.top_p is not None else '(なし)'}, "
        f"top_k={sampling.top_k if sampling.top_k is not None else '(なし)'}, "
        f"thinking={sampling.thinking}"
    )


def _suite_settings_rows(manifest: RunManifest) -> list[tuple[str, str]]:
    """まとまりごとの、試行の回数と出力の上限 (1.6)。"""
    profile = manifest.profile
    rows: list[tuple[str, str]] = []
    for suite in manifest.suites:
        if suite is SuiteName.DECODE:
            detail = (
                f"試行 {profile.decode.trials} 回 (慣らし {profile.decode.warmup_trials} 回)、"
                f"出力の上限 {profile.decode.max_tokens} トークン"
            )
        elif suite is SuiteName.PREFILL:
            detail = (
                f"試行 {profile.prefill.trials} 回 (慣らし {profile.prefill.warmup_trials} 回)、"
                f"出力の上限 {profile.prefill.max_tokens} トークン、"
                f"狙いの入力 {profile.prefill.target_input_tokens} トークン"
            )
        elif suite is SuiteName.CONCURRENCY:
            detail = (
                f"{profile.concurrency.rounds} 回ぶん × {profile.concurrency.levels} 本、"
                f"出力の上限 {profile.concurrency.max_tokens} トークン"
            )
        elif suite is SuiteName.QUALITY:
            detail = (
                f"ツール呼び出し {profile.quality.toolcall_tasks} 問、"
                f"探す課題は 1 つの条件につき {profile.quality.trials_per_cell} 回"
            )
        else:
            detail = (
                f"1 段階につき {profile.agent.trials_per_stage} 回、"
                f"{profile.agent.start_tokens}〜{profile.agent.end_tokens} トークンを "
                f"{profile.agent.step_tokens} 刻み"
            )
        rows.append((f"まとまり `{suite.value}`", detail))
    return rows


def _warnings_section(manifest: RunManifest, warnings: Sequence[str]) -> list[str]:
    items = [*warnings, *manifest.warnings]
    if not items:
        return []
    lines = ["## 警告", ""]
    lines += [f"- {_cell(item)}" for item in items]
    lines.append("")
    return lines


def _skipped_section(manifest: RunManifest) -> list[str]:
    if not manifest.skipped:
        return []
    lines = ["## 飛ばした条件", "", "| まとまり | 条件 | 理由 |", "|---|---|---|"]
    lines += [
        f"| {_cell(item.suite.value)} | `{_cell(item.key)}` | {_cell(item.reason)} |"
        for item in manifest.skipped
    ]
    lines.append("")
    return lines


def _results_section(title: str, summary: Summary, tier: Tier) -> list[str]:
    """主な結果と参考を、別の節に分けて出す (4.4)。

    この表は連続の値だけを並べる。割合の行は、数える単位も欄も違うので、
    それぞれの節 (`品質の検査`、`長い会話でのツール呼び出し`) に出す。
    """
    lines = [f"## {title}", ""]
    rows = [
        row for row in summary.results if row.tier == tier and row.metric not in _PROPORTION_METRICS
    ]
    if not rows:
        lines += ["(この計測ランには、該当する条件がない)", ""]
        return lines
    groups: dict[str, list[MetricResult]] = {}
    for row in rows:
        groups.setdefault(_suite_label(row.condition), []).append(row)
    for label, group in groups.items():
        lines += [f"### {label}", "", _TABLE_HEADER, _TABLE_RULE]
        lines += [_result_row(row) for row in group]
        lines.append("")
    return lines


def _suite_label(condition: str) -> str:
    """条件の鍵の先頭 (まとまりの名前) で表をまとめる。"""
    head = condition.split("/", 1)[0]
    return head or condition


def _result_row(row: MetricResult) -> str:
    stat = row.continuous
    cells = [
        f"`{_cell(row.condition)}`",
        f"`{_cell(row.metric)}`",
        str(stat.n) if stat is not None else _EMPTY,
        _num(stat.mean if stat is not None else None),
        _num(stat.median if stat is not None else None),
        _num(stat.min if stat is not None else None),
        _num(stat.max if stat is not None else None),
        _num(stat.stdev if stat is not None else None),
        _num(stat.iqr if stat is not None else None),
        _num(stat.cv if stat is not None else None),
        str(row.failures),
        _flags_text(row),
    ]
    return "| " + " | ".join(cells) + " |"


def _flags_text(row: MetricResult) -> str:
    """印を、件数つきの日本語で並べる (2.6、3.3、10.2、10.7、4.5)。

    16 トークン未満の除外は `decode_tps` にしか効かないので、その件数は
    `decode_tps` の行にだけ出す (ほかの行に並べると、その行の n が減った理由
    だと読めてしまう)。`MetricResult.flag_counts` は、どの行も条件の全体の
    件数を持つ (4.2 と 4.3 が使う)。
    """
    counts = row.flag_counts
    parts: list[str] = []
    for flag in row.flags:
        if flag is MetricFlag.INSUFFICIENT_TRIALS:
            parts.append("試行の数が足りない")
        elif flag is MetricFlag.PARTIAL_FAILURES:
            parts.append("一部が失敗 (成功した要求だけから求めた値)")
        elif flag is MetricFlag.SHORT_OUTPUTS:
            parts.append(f"早く終わった {counts.get(TrialFlag.SHORT_OUTPUT.value, 0)} 件")
        elif flag is MetricFlag.LENGTH_OFF_TARGET:
            parts.append(f"長さが外れた {counts.get(TrialFlag.LENGTH_OFF_TARGET.value, 0)} 件")
        else:
            suspect = counts.get(TrialFlag.REPLACEMENT_CHAR.value, 0) + counts.get(
                TrialFlag.REPETITION_LOOP.value, 0
            )
            parts.append(f"出力が壊れている疑い {suspect} 件")
    too_few = counts.get(TrialFlag.TOO_FEW_OUTPUT_TOKENS.value, 0)
    if too_few and row.metric == METRIC_DECODE_TPS:
        parts.append(f"出力のトークンが少なすぎる {too_few} 件 (この行から除いた)")
    return ", ".join(parts) if parts else _EMPTY


def _quality_section(summary: Summary) -> list[str]:
    """品質の検査の割合 (5.6、5.7)。分母と分子と、不確かさの幅と、印を添える。

    印は速さの表と同じ `_flags_text` で書く。出力が壊れている疑いの件数 (10.7)
    は、この表にも出さないと、割合しか見ない読み手に届かない。
    """
    rows = [row for row in summary.results if row.metric == METRIC_ACCURACY]
    if not rows:
        return []
    lines = [
        "## 品質の検査",
        "",
        (
            "| 条件 | 正解 | 採点の対象 | 正解の割合 | 95% の区間 | 採点できなかった |"
            " 要求の失敗 | 印 |"
        ),
        "|---|--:|--:|--:|---|--:|--:|---|",
    ]
    for row in rows:
        stat = row.proportion
        cells = [
            f"`{_cell(row.condition)}`",
            str(stat.numerator) if stat is not None else _EMPTY,
            str(stat.denominator) if stat is not None else "0",
            _num(stat.rate) if stat is not None else _EMPTY,
            _interval(stat),
            str(row.flag_counts.get(NOT_SCORED_COUNT, 0)),
            str(row.failures),
            _flags_text(row),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _agent_section(summary: Summary) -> list[str]:
    """長い会話でのツール呼び出し (6.4、6.5、6.6、6.9)。"""
    agent = summary.agent
    if agent is None:
        return []
    lines = ["## 長い会話でのツール呼び出し", ""]
    first = (
        f"{agent.first_exceeded_tokens} トークン ({_first_exceeded_kind(agent)})"
        if agent.first_exceeded_tokens is not None
        else "どの段階も超えていない"
    )
    reached = (
        f"{agent.reached_tokens} トークン (狙いの長さ)"
        if agent.reached_tokens is not None
        else _EMPTY
    )
    bullets = [
        f"崩れのしきい値: {_num(agent.threshold)}",
        f"しきい値を初めて超えた会話の長さ: {first}",
        f"採点できた試行が残っている、いちばん長い段階: {reached}",
    ]
    if agent.stopped_reason is not None:
        bullets.append(f"止めた理由: {_cell(agent.stopped_reason)}")
    for bullet in bullets:
        lines += [f"- {bullet}", ""]
    if not agent.stages:
        lines += [_no_stage_text(summary), ""]
        return lines
    lines += _agent_stage_table(agent.stages, agent.threshold, _flags_by_condition(summary))
    lines += _agent_outcome_table(agent.stages)
    return lines


def _first_exceeded_kind(agent: AgentSummary) -> str:
    """「初めて超えた長さ」が、どちらの量なのか (6.6 の決め方と同じ順でたどる)。

    `_first_exceeded_tokens` は、実際の入力のトークン数の中央値を使い、それが
    残っていなければ狙いの長さで代える。どちらを出したのかを書かないと、
    狙いの長さと実際の長さを取り違えて読まれる。
    """
    for stage in agent.stages:
        if stage.break_rate is not None and stage.break_rate.rate > agent.threshold:
            if stage.actual_input_tokens is not None:
                return "実際の入力のトークン数の中央値"
            return "狙いの長さ (実際の入力のトークン数が残っていない)"
    return "狙いの長さ"


def _no_stage_text(summary: Summary) -> str:
    """段階の表が空のときに、何が起きたのかを書く。

    段階の表が空になるのは、試行が 1 つもない場合と、試行はあるのに会話の長さ
    が決められなかった場合の 2 つである。後者を「試行が残っていない」と書くと、
    残っている生データを人が探しに行かなくなる。
    """
    without_length = [
        row.condition for row in summary.results if row.metric == METRIC_AGENT_BREAK_RATE
    ]
    if not without_length:
        return "(段階の試行が 1 つも残っていない)"
    keys = "、".join(f"`{_cell(condition)}`" for condition in without_length)
    return (
        f"(試行は残っているが、会話の長さが決められないので、段階の表に出せない: {keys}。"
        "理由は警告にある)"
    )


def _flags_by_condition(summary: Summary) -> dict[str, str]:
    """条件 → 印の文字列 (段階の表に、速さの表と同じ印を出すために引く)。"""
    return {
        row.condition: _flags_text(row)
        for row in summary.results
        if row.metric == METRIC_AGENT_BREAK_RATE
    }


def _agent_stage_table(
    stages: Sequence[AgentStageResult], threshold: float, flags: dict[str, str]
) -> list[str]:
    lines = [
        (
            "| 段階 | 狙いの入力 | 実際の入力 (中央値) | 試行 | 崩れ | 分母 |"
            " 崩れた割合 | 95% の区間 | 片側 95% の上限 | 判定 | 印 |"
        ),
        "|---|--:|--:|--:|--:|--:|--:|---|--:|---|---|",
    ]
    needed = _trials_needed(threshold)
    for stage in stages:
        stat = stage.break_rate
        actual = stage.actual_input_tokens
        cells = [
            f"`{_cell(stage.stage_key)}`",
            str(stage.target_input_tokens),
            str(round(actual.median)) if actual is not None else _EMPTY,
            str(stage.trials),
            str(stat.numerator) if stat is not None else _EMPTY,
            str(stat.denominator) if stat is not None else "0",
            _num(stat.rate) if stat is not None else _EMPTY,
            _interval(stat),
            _num(stat.upper95_one_sided) if stat is not None else _EMPTY,
            _verdict_text(stage, needed),
            flags.get(stage.stage_key, _EMPTY),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _agent_outcome_table(stages: Sequence[AgentStageResult]) -> list[str]:
    """9 種類の分類の内訳 (6.3、6.4)。起きなかった分類も 0 件として出す。"""
    outcomes = list(ToolCallOutcome)
    lines = [
        "分類ごとの件数 (9 種類):",
        "",
        "| 段階 | " + " | ".join(outcome.value for outcome in outcomes) + " |",
        "|---" + "|--:" * len(outcomes) + "|",
    ]
    for stage in stages:
        counts = [str(stage.outcome_counts.get(outcome, 0)) for outcome in outcomes]
        lines.append("| " + " | ".join([f"`{_cell(stage.stage_key)}`", *counts]) + " |")
    lines.append("")
    return lines


def _verdict_text(stage: AgentStageResult, needed: int | None) -> str:
    """しきい値に対する判定を、日本語で書く (6.5)。

    崩れが 0 件で「試行の数が足りない」なら、下回ったと言うのに要る試行の数も
    添える (あと何回流せばよいかが、その場で分かるように)。
    """
    text = _VERDICT_LABELS[stage.verdict]
    zero_breaks = stage.break_rate is not None and stage.break_rate.numerator == 0
    if stage.verdict is ThresholdVerdict.UNDETERMINED and zero_breaks and needed is not None:
        text += f" (崩れ 0 件のままなら {needed} 回要る)"
    return text


def _trials_needed(threshold: float) -> int | None:
    """崩れ 0 件で、しきい値を下回ったと言うのに要る試行の数 (6.5)。"""
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        return None
    return trials_needed_for_zero_failures(threshold)


def _interval(stat: ProportionStat | None) -> str:
    """両側 95% の区間を、1 つの升目に書く。"""
    if stat is None:
        return _EMPTY
    return f"{_num(stat.ci95_low)} 〜 {_num(stat.ci95_high)}"


def _server_metrics_section(summary: Summary) -> list[str]:
    """条件ごとの、対象サーバーの内部の指標の導出値 (7.3、7.4)。"""
    if not summary.server_metrics:
        return []
    lines = [
        "## 対象サーバーの内部の指標",
        "",
        (
            "| 条件 | 投機の当たり率 | 平均の受理長 | 生成のステップ |"
            " 1 ステップあたりの生成トークン |"
            " プレフィックスキャッシュ | KV の使用率の最大 | 追い出し | 得られなかった指標 |"
        ),
        "|---|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for condition, derived in summary.server_metrics.items():
        missing = ", ".join(metric.value for metric in derived.missing) or _EMPTY
        cells = [
            f"`{_cell(condition)}`",
            _num(derived.spec_acceptance_rate),
            _num(derived.mean_acceptance_length),
            _num(derived.decode_steps),
            _num(derived.tokens_per_step),
            _num(derived.prefix_cache_hit_rate),
            _num(derived.kv_usage_peak),
            _num(derived.preemptions),
            _cell(missing),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _datasets_section(summary: Summary) -> list[str]:
    """公開の課題の出どころ (5.5、11.2)。"""
    if not summary.datasets:
        return []
    lines = [
        "## 使った公開の課題",
        "",
        "| 名前 | 版 | 入手先 | ライセンス | 採点の方法 |",
        "|---|---|---|---|---|",
    ]
    lines += [
        "| "
        + " | ".join(
            [
                _cell(dataset.name),
                _cell(dataset.version),
                _cell(dataset.source_url),
                _cell(dataset.license),
                _cell(dataset.scoring_method),
            ]
        )
        + " |"
        for dataset in summary.datasets
    ]
    lines.append("")
    return lines


def _num(value: float | None) -> str:
    """小数の書き方を固定する (何度書いても同じ文字列になるように)。"""
    if value is None or not math.isfinite(value):
        return _EMPTY
    if value == 0.0:
        return "0.000"  # -0.0 も 0.000 にする
    if abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.3f}"


def _cell(text: str) -> str:
    """表の升目に入れられる形にする (改行と縦棒を逃がす)。"""
    return " ".join(text.split()).replace("|", "\\|")


# --- 別名で書いてから置き換える ---------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    """同じディレクトリに別名で書いて `fsync` し、`os.replace` してから、
    ディレクトリ自体も `fsync` する。

    保存の部品 (`store/rawstore._atomic_write_bytes`) と同じ決まりにしてある。
    置き換えが失敗しても前の要約は壊れず、置き換えが済んだあとに電源が落ちても、
    ディレクトリの項目が古い名前を指したままにならない。
    """
    data = text.encode("utf-8")
    directory = path.parent
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
