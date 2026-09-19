"""生データから、速さの要約を作る (task 4.1: analysis/summarize)。

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
  中の値は `_SUITE_METRICS` の並び。どちらも生データだけで決まるので、同じ
  生データからは必ず同じ並びになる

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

## この版で集計するまとまり

`decode`、`prefill`、`concurrency` の 3 つ。`quality` と `agent` のレコードは、
ここでは 1 行も出さない (割合で表す結果は task 4.2 が足す)。

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

from bench_harness.analysis.stats import describe
from bench_harness.store.rawstore import MetricDeltaRecord, RunStore
from bench_harness.types import (
    DerivedMetrics,
    MetricFlag,
    MetricResult,
    RunManifest,
    RunStatus,
    SuiteName,
    Summary,
    Tier,
    TrialFlag,
    TrialRecord,
)

__all__ = [
    "METRIC_DECODE_TPS",
    "METRIC_PREFILL_TPS",
    "METRIC_ROUND_TOTAL_TPS",
    "METRIC_TTFT",
    "MIN_SPEED_OUTPUT_TOKENS",
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

_NS_PER_S: Final[float] = 1e9

_SUITE_METRICS: Final[dict[SuiteName, tuple[str, ...]]] = {
    SuiteName.DECODE: (METRIC_DECODE_TPS, METRIC_TTFT),
    SuiteName.PREFILL: (METRIC_TTFT, METRIC_PREFILL_TPS),
    SuiteName.CONCURRENCY: (METRIC_DECODE_TPS, METRIC_ROUND_TOTAL_TPS, METRIC_TTFT),
}
"""まとまりごとに出す値と、その並び。ここにないまとまりは、この版では出さない。"""


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
    """1 つの条件の、すべてのレコードと、値の名前ごとの試行ごとの値。"""

    suite: SuiteName
    condition: str
    records: list[TrialRecord]
    values: dict[str, _MetricValues]


def trial_values(
    records: Sequence[TrialRecord], manifest: RunManifest
) -> dict[tuple[str, str], list[TrialValue]]:
    """試行ごとの値を、`{(条件, 値の名前): [TrialValue, ...]}` で返す (4.3 が使う)。

    要約の集計も、この同じ値から作る (module の docstring「比較との共有」)。
    返すのは、**成功して、慣らしでない**試行の値だけで、要約に出る行と同じ
    並びになる。`quality` と `agent` のレコードは、この版では何も返さない。

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
    results = _summarize_trials(trials, manifest.profile.min_successes, notes)
    server_metrics = _server_metrics(deltas, notes)
    summary = Summary(
        conditions=manifest,
        incomplete=manifest.status is not RunStatus.COMPLETED,
        results=results,
        agent=None,
        server_metrics=server_metrics,
        datasets=list(manifest.datasets),
    )
    return SummaryResult(summary=summary, warnings=[*trial_warnings, *delta_warnings, *notes])


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
        if record.suite not in _SUITE_METRICS:
            continue  # quality と agent は task 4.2 が集計する
        groups.setdefault((record.suite, record.condition), []).append(record)

    return [
        _ConditionValues(
            suite=suite,
            condition=condition,
            records=records,
            values={metric: _VALUE_FUNCS[metric](records) for metric in _SUITE_METRICS[suite]},
        )
        for (suite, condition), records in groups.items()
    ]


def _summarize_trials(
    trials: Sequence[TrialRecord], min_successes: int, notes: list[str]
) -> list[MetricResult]:
    """条件ごとに値の行を作る。値は `trial_values` と同じ道を通って来る。"""
    results: list[MetricResult] = []
    for data in _condition_values(trials):
        results.extend(_summarize_condition(data, min_successes, notes))
    return results


def _summarize_condition(
    data: _ConditionValues, min_successes: int, notes: list[str]
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
    for metric in _SUITE_METRICS[suite]:
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
            )
        )
    return rows


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
    if flag_counts[TrialFlag.SHORT_OUTPUT.value] > 0:
        flags.append(MetricFlag.SHORT_OUTPUTS)
    if flag_counts[TrialFlag.LENGTH_OFF_TARGET.value] > 0:
        flags.append(MetricFlag.LENGTH_OFF_TARGET)
    replacement = flag_counts[TrialFlag.REPLACEMENT_CHAR.value]
    repetition = flag_counts[TrialFlag.REPETITION_LOOP.value]
    if replacement + repetition > 0:
        flags.append(MetricFlag.SUSPECT_OUTPUTS)
    return flags


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
    "`decode_tps` = 生成速度 (トークン/秒、最初のトークンまでの時間を含めない)、"
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
_EMPTY: Final[str] = "—"


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
    lines += _warnings_section(manifest, warnings)
    lines += _skipped_section(manifest)
    lines += _legend_section(summary)
    lines += _results_section("主な結果", summary, "primary")
    lines += _results_section("参考", summary, "reference")
    lines += _server_metrics_section(summary)
    lines += _datasets_section(summary)
    return "\n".join(lines).rstrip("\n") + "\n"


def _legend_section(summary: Summary) -> list[str]:
    """表の読み方。主な結果と参考で同じなので、1 回だけ出す。"""
    if not summary.results:
        return []
    notes = [_METRIC_LEGEND, _N_NOTE, _insufficient_note(summary.conditions.profile.min_successes)]
    if any(row.metric == METRIC_PREFILL_TPS for row in summary.results):
        notes.append(_PREFILL_NOTE)
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
    """主な結果と参考を、別の節に分けて出す (4.4)。"""
    lines = [f"## {title}", ""]
    rows = [row for row in summary.results if row.tier == tier]
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


def _server_metrics_section(summary: Summary) -> list[str]:
    """条件ごとの、対象サーバーの内部の指標の導出値 (7.3、7.4)。"""
    if not summary.server_metrics:
        return []
    lines = [
        "## 対象サーバーの内部の指標",
        "",
        (
            "| 条件 | 投機の当たり率 | 平均の受理長 | 生成のステップ | 1 ステップあたり |"
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
