"""2 つの計測ランを比べる (task 4.3: analysis/compare)。

`compare_runs(run_dir_a, run_dir_b)` が 2 つの計測ランの生データを読んで
`ComparisonReport` を組み立て、`render_comparison_markdown(report)` が人の読む
形にする。`bench compare <run_a> <run_b>` (5.1) から呼ばれる。

## 契約 (design.md analysis/summarize / compare / publish、Batch)

- **入力**: `RunStore.open` の読み取りと、`summarize_run` / `trial_values`
  (4.1) の呼び出しだけ。対象サーバーにもネットワークにも触れない。計測ランで
  ない場所を渡すと、`RunStore.open` の `StoreError` がそのまま外に出る
- **出力**: `ComparisonReport` (`warnings` → `rows` → `excluded` →
  `repeatability`)。人が読む形と、道具が読む JSON を、別々に出せる
- **何度実行しても同じ結果になる**: 生成の時刻を入れず、並びを固定し、再標本化
  の種を入力から決めるので、同じ 2 つの計測ランからは必ずバイト単位で同じ
  出力になる (`PYTHONHASHSEED` にも依存しない)

## 速さの式は書き写さない (Implementation Notes 4.1)

試行ごとの値は、要約 (4.1) の公開の関数 `trial_values(records, manifest)` から
もらう。値は `(条件, 値の名前)` ごとの `TrialValue(trial_index, stream_index,
value)` の列で、慣らしと失敗はすでに外れている。比較の側で速さを計算し直すと、
定義が 2 か所に分かれるので、決してしない。

## 表の埋め方 (Implementation Notes 2.4)

| 欄 | 量 |
|---|---|
| `value_a` / `value_b` | 条件の**中央値** (割合の行は割合そのもの) |
| `diff` | `value_b − value_a` |
| `relative_diff` | `diff ÷ value_a` (`value_a` が 0、または片方の値がないときは値なし) |
| `verdict.median_diff` と区間 | 再標本化の統計量。**別の欄**に出す |

`DiffVerdict.median_diff` は、対応のある比較では「試行ごとの差の中央値」という
別の量である。中央値どうしの差 (`diff`) と同じ欄に混ぜない。

## 対応のある比較にする条件 (9.2、research.md の判断)

次の 3 つがすべて成り立つときだけ、対応のある再標本化にする。

1. 2 つの計測ランの `generator_version` が同じ (同じ番号の試行が同じ入力になる)
2. `profile.seed` が同じ (同上)
3. その行の値の `(trial_index, stream_index)` の列が、両方にあって、完全に
   一致する (片方に失敗した試行があれば一致しない)

対応のある比較にするときは、両方の列を鍵で並べ替えてから `diff_verdict` に
渡す。書き足した順のまま渡すと、番号の違う試行どうしの差を取ってしまう。

同じ番号が 2 回入っている列も、対応のある比較にしない。両方に同じ重なりが
あると番号の列は見かけ上そろうが、どの値とどの値の差を取るのかが決まらない。

**対応なしへの退化は、行ごとに警告する** (Implementation Notes 4.3)。1 と 2 が
そろっていて対応のある比較になるはずだったのに、3 で落ちた行が対象である。
失敗が 1 件あるだけで退化し、検出力が大きく落ちる (同じ +3% の差が、対応あり
では「収まらない」、対応なしでは「収まる」) ので、黙って済ませない。1 か 2 が
食い違う比較には、それぞれの警告がすでにあるので、行ごとには重ねない。

値が 2 つに満たない行は判定しない (`verdict=None`)。「判定できない」を「収まっ
た」と読ませないため、警告にも、繰り返しの結論の一覧にも、理由つきで出す。

## 繰り返しの結論 (9.3、4.4)

同じ対象サーバーの定義 (名前) どうしの比較のときだけ出す。結論を決めるのは
**主な結果 (primary)** の行だけで、参考 (reference) の行は表に出すが結論には
数えない (4.4: 1 本と 2 本が主な結果、4 本と 8 本は余力を知るための参考)。
判定できなかった行は、結論を `all_within=False` にして、理由つきで一覧に出す。

名前が同じでも、接続先 (`target.base_url`) かモデル (`target.model`) が違えば、
同じものを 2 回測った証拠にはならない。9.3 が言う「同じ対象サーバーの定義」は
定義そのもの (= 名前) なので結論は出したままにするが、その食い違いは**参考
ではなく、ふつうの警告**として、表より先に出す (Implementation Notes 4.3)。

## 警告と、参考の警告の分け方 (9.4)

測った値に効きうる食い違いは、すべてふつうの警告にする。参考 (`(参考)` で
始まる文) に落とすのは、次の 3 つだけである。

1. 対象サーバーの定義の名前そのものが違う (構成どうしの比較なので、ふつうは
   これでよい) と、サーバーが申告したモデルや版の違い
2. どちらの計測ランも測っていないまとまりの設定 (`quality.*`、`agent.*`、
   コードの隔離の `sandbox.*` など)。比べる行が 1 つも出ないので、値に効かない
3. 内部の指標を読む間隔 (`metrics_interval_s`)。読む回数が変わるだけで、測る
   値の決め方は変わらない

サンプリングの設定、thinking、出力の上限、試行の回数、慣らしの回数、乱数の種、
1 トークンあたりの文字数、制限時間、許容の幅、出力が壊れている疑いのしきい値
は、測ったまとまりのものを 1 つも落とさない。

## 依存の向き (design.md)

読み込んでよいのは標準ライブラリ、`bench_harness.types`、
`bench_harness.store`、`bench_harness.analysis.stats`、同じ層の
`bench_harness.analysis.summarize` だけである (試験が、構文木を見て機械的に
確かめている)。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from bench_harness.analysis.stats import diff_verdict, proportion_diff_verdict
from bench_harness.analysis.summarize import (
    METRIC_ACCURACY,
    METRIC_AGENT_BREAK_RATE,
    PHASE_METRIC_NAMES,
    RETOKENIZED_METRIC_NAMES,
    TrialValue,
    summarize_run,
    trial_values,
)
from bench_harness.store.rawstore import RunStore
from bench_harness.types import (
    ComparisonReport,
    ComparisonRow,
    DiffVerdict,
    MetricResult,
    Profile,
    ProportionStat,
    RepeatabilityConclusion,
    RunManifest,
    RunStatus,
    SuiteName,
    Summary,
    TargetDef,
    Tier,
    _OutputRetokenization,
)

__all__ = [
    "ComparisonResult",
    "ProportionPairs",
    "ProportionSides",
    "TrialValues",
    "compare_runs",
    "compare_summaries",
    "render_comparison_json",
    "render_comparison_markdown",
    "write_comparison",
]

TrialValues = Mapping[tuple[str, str], Sequence[TrialValue]]
"""`trial_values` が返す形 (`(条件, 値の名前)` ごとの、試行ごとの値)。"""

ProportionPairs = Mapping[tuple[str, str], tuple[ProportionStat, ProportionStat]]
"""割合の行の、両方の計測ランの分子と分母 (人が読む表に添える)。"""

ProportionSides = Mapping[tuple[str, str], tuple[ProportionStat | None, ProportionStat | None]]
"""割合の行の、それぞれの計測ランの分子と分母。**片方にしかない行も入る**。

`ProportionPairs` は両方にそろった行だけを持つ。片方でしか採点できなかった行は、
判定できないが、分かっている側の件数は表に出したい。`ComparisonResult` は
こちらを持ち、`to_markdown()` がこれを使う。
"""

_PROPORTION_METRICS: Final[frozenset[str]] = frozenset({METRIC_ACCURACY, METRIC_AGENT_BREAK_RATE})
"""割合で表す値の名前 (4.2)。行を表に振り分けるのに使う。

`ComparisonRow` は、連続の値か割合かを持たない。片方にしか割合がない行は
`proportion_verdict` が空になるので、それだけを見ると連続の値の表に落ちる。
値の名前で見れば、片側しかない行も割合の表に出せる。
"""

_MIN_VALUES_FOR_VERDICT: Final[int] = 2
"""判定に要る、片側の値の数 (`diff_verdict` の下限と同じ)。"""

_PROFILE_PATHS_WARNED_ELSEWHERE: Final[frozenset[str]] = frozenset(
    {
        "name",  # `profile_name` の警告と同じ違いなので、二重に出さない
        "compare_tolerance",  # 判定に使う値。どちらを使ったかを含めて別の文で出す
    }
)
"""計測の設定の差分から外す経路。

外すのはこの 2 つだけである。**測り方に効きうる項目は 1 つも外さない**
(項目を手で並べると、設定が増えたときに黙って見落とすので、設定は丸ごと
突き合わせる)。この 2 つも「違いを報せない」のではなく、別の文で報せる。
"""

_MISSING_FIELD: Final[str] = "(項目なし)"
"""片方の計測ランにしかない設定の項目 (版が違う生データ) の表示。"""

_PROFILE_PATH_SUITES: Final[dict[str, SuiteName]] = {
    "decode": SuiteName.DECODE,
    "prefill": SuiteName.PREFILL,
    "concurrency": SuiteName.CONCURRENCY,
    "quality": SuiteName.QUALITY,
    "sandbox": SuiteName.QUALITY,  # コードの隔離は、品質の検査だけが使う
    "agent": SuiteName.AGENT,
}
"""計測の設定の経路の先頭 → その設定が効くまとまり (9.4 の絞り込みに使う)。

ここにない経路 (乱数の種、サンプリング、1 トークンあたりの文字数、制限時間、
許容の幅、出力が壊れている疑いのしきい値など) は、どのまとまりにも効きうる
ので、まとまりで絞り込まない。
"""

_PROFILE_PATHS_INFORMATIONAL: Final[frozenset[str]] = frozenset({"metrics_interval_s"})
"""測ったまとまりに関わらず、参考に落とす経路。

`metrics_interval_s` は、計測の間に内部の指標を読む間隔である。読む回数が
変わるだけで、速さも割合も、決め方は変わらない。
"""


# --- 返り値 -----------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonResult:
    """`compare_runs` の返り値。

    `report` が比較そのもので、`read_warnings_a` / `read_warnings_b` は、
    それぞれの計測ランの生データを読んだときの警告 (末尾が途中で切れた行を
    捨てた、値にできなかった試行があった、など) である。同じ内容は
    `report.warnings` の先頭にも、どちらの計測ランのものかを添えて入って
    いる (10.5)。`proportions` は、割合の行の分子と分母で、人が読む表に
    件数を添えるために使う (`ComparisonRow` は割合しか持たない)。

    書き出すときは `to_markdown()` と `to_json()` を使うこと。
    `render_comparison_markdown(result.report)` と書くと、割合の行の分子と
    分母が黙って消える (注 4.3)。この 2 つなら、渡し忘れようがない。
    """

    report: ComparisonReport
    read_warnings_a: list[str] = field(default_factory=list)
    read_warnings_b: list[str] = field(default_factory=list)
    proportions: dict[tuple[str, str], tuple[ProportionStat, ProportionStat]] = field(
        default_factory=dict
    )
    proportion_sides: dict[tuple[str, str], tuple[ProportionStat | None, ProportionStat | None]] = (
        field(default_factory=dict)
    )
    """片方にしか割合がない行も含む、それぞれの側の分子と分母 (`ProportionSides`)。"""

    def to_markdown(self) -> str:
        """人が読む比較の結果。件数を渡し忘れようがない口 (注 4.3)。"""
        return _render_markdown(self.report, self.proportion_sides)

    def to_json(self) -> str:
        """道具が読む比較の結果 (`render_comparison_json` と同じ)。"""
        return render_comparison_json(self.report)


@dataclass(frozen=True)
class _RunData:
    """1 つの計測ランから読んだもの。"""

    summary: Summary
    values: dict[tuple[str, str], list[TrialValue]]
    warnings: list[str]


# --- 入口 -------------------------------------------------------------------


def compare_runs(
    run_dir_a: Path, run_dir_b: Path, *, tolerance: float | None = None
) -> ComparisonResult:
    """2 つの計測ランを比べる (9.1〜9.6、10.5)。

    生データに対して純粋である (読むだけで、何も書かない)。`tolerance` を
    渡さなければ、計測ラン A の `profile.compare_tolerance` を使う。
    """
    data_a = _load_run(run_dir_a)
    data_b = _load_run(run_dir_b)
    report = compare_summaries(
        data_a.summary,
        data_b.summary,
        values_a=data_a.values,
        values_b=data_b.values,
        tolerance=tolerance,
        read_warnings_a=data_a.warnings,
        read_warnings_b=data_b.warnings,
    )
    sides = _proportion_sides(data_a.summary, data_b.summary)
    return ComparisonResult(
        report=report,
        read_warnings_a=list(data_a.warnings),
        read_warnings_b=list(data_b.warnings),
        proportions=_both_sides(sides),
        proportion_sides=sides,
    )


def compare_summaries(
    summary_a: Summary,
    summary_b: Summary,
    *,
    values_a: TrialValues | None = None,
    values_b: TrialValues | None = None,
    tolerance: float | None = None,
    read_warnings_a: Sequence[str] = (),
    read_warnings_b: Sequence[str] = (),
) -> ComparisonReport:
    """2 つの要約を比べる (`compare_runs` の中身)。

    試行ごとの値 (`values_a`、`values_b`) は `trial_values` が返す形で渡す。
    渡さなければ、連続の値の行は「値が足りないので判定できない」になる
    (要約だけからは、ばらつきの範囲を出せない)。
    """
    manifest_a = summary_a.conditions
    manifest_b = summary_b.conditions
    effective_tolerance = _effective_tolerance(tolerance, manifest_a)

    warnings = [
        *_incomplete_warnings(manifest_a, manifest_b),
        *_read_warning_lines(manifest_a, read_warnings_a, "A"),
        *_read_warning_lines(manifest_b, read_warnings_b, "B"),
        *_phase_warning_lines(summary_a, "A"),
        *_phase_warning_lines(summary_b, "B"),
        *_setting_warnings(manifest_a, manifest_b, tolerance=tolerance),
        *_informational_warnings(manifest_a, manifest_b),
        *_retokenization_warnings(manifest_a, manifest_b),
    ]

    rows, row_warnings, markers = _build_rows(
        summary_a,
        summary_b,
        values_a=values_a or {},
        values_b=values_b or {},
        tolerance=effective_tolerance,
    )
    warnings.extend(row_warnings)

    return ComparisonReport(
        run_a=manifest_a.run_id,
        run_b=manifest_b.run_id,
        warnings=warnings,
        rows=rows,
        excluded=_excluded(summary_a, summary_b),
        repeatability=_repeatability(rows, markers, manifest_a, manifest_b),
    )


def _load_run(run_dir: Path) -> _RunData:
    """計測ランを読んで、要約と試行ごとの値と警告を返す (2.3: 読むのは `RunStore` だけ)。"""
    result = summarize_run(run_dir)
    store = RunStore.open(run_dir)
    records, _ = store.read_trials()  # 警告は summarize_run のものと同じなので、重ねない
    return _RunData(
        summary=result.summary,
        values=trial_values(records, store.manifest()),
        warnings=list(result.warnings),
    )


def _effective_tolerance(tolerance: float | None, manifest_a: RunManifest) -> float:
    """判定に使う許容の幅。既定は計測ラン A の設定の値 (9.2)。"""
    if tolerance is None:
        return manifest_a.profile.compare_tolerance
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"tolerance は 0 以上の有限の値である必要がある (tolerance={tolerance})")
    return tolerance


# --- 警告 (9.4、10.5) --------------------------------------------------------


def _incomplete_warnings(manifest_a: RunManifest, manifest_b: RunManifest) -> list[str]:
    """未完了の計測ランを比較に使うことを、いちばん先に報せる (10.4、10.5)。"""
    warnings: list[str] = []
    for label, manifest in (("A", manifest_a), ("B", manifest_b)):
        if manifest.status is RunStatus.COMPLETED:
            continue
        note = (
            "(`running` のまま残っているのは、計測ランが異常終了した跡である)"
            if manifest.status is RunStatus.RUNNING
            else ""
        )
        warnings.append(
            f"計測ラン {label} (`{manifest.run_id}`) は未完了である "
            f"(状態: `{manifest.status.value}`){note}。"
            "途中までの結果どうしの比較として読むこと。"
        )
    return warnings


def _read_warning_lines(manifest: RunManifest, warnings: Sequence[str], label: str) -> list[str]:
    """生データを読んだときの警告を、どちらの計測ランのものかを添えて並べる。"""
    return [
        f"計測ラン {label} (`{manifest.run_id}`) の生データの読み取り: {warning}"
        for warning in warnings
    ]


def _phase_warning_lines(summary: Summary, label: str) -> list[str]:
    """段階別計測の警告を、どちらの計測ランのものかを添えて並べる。"""
    return [
        f"計測ラン {label} (`{summary.conditions.run_id}`) の段階別計測: {warning}"
        for warning in summary.phase_warnings
    ]


def _setting_warnings(
    manifest_a: RunManifest, manifest_b: RunManifest, *, tolerance: float | None
) -> list[str]:
    """測り方に効く条件の食い違い (9.4)。どちらの値かが分かるように、両方を書く。"""
    warnings: list[str] = []

    if manifest_a.harness_version != manifest_b.harness_version:
        warnings.append(
            f"道具の版が違う (A: `{manifest_a.harness_version}`, "
            f"B: `{manifest_b.harness_version}`)。測り方そのものが違う可能性がある。"
        )
    for label, manifest in (("A", manifest_a), ("B", manifest_b)):
        version = manifest.harness_version
        if ".dirty" in version:
            warnings.append(
                f"計測ラン {label} の道具の版 `{version}` に、未コミットの変更の印 "
                "(`.dirty`) が付いている。版で測り方を特定できない。"
            )
        if "unknown" in version:
            warnings.append(
                f"計測ラン {label} の道具の版 `{version}` に `unknown` が入っている "
                "(git から版を決められなかった)。版で測り方を特定できない。"
            )

    if manifest_a.generator_version != manifest_b.generator_version:
        warnings.append(
            f"生成器の版が違う (A: {manifest_a.generator_version}, "
            f"B: {manifest_b.generator_version})。同じ番号の試行でも入力が違うので、"
            "対応のある比較にしない。"
        )

    if manifest_a.profile_name != manifest_b.profile_name:
        warnings.append(
            f"計測の設定の名前が違う (A: `{manifest_a.profile_name}`, "
            f"B: `{manifest_b.profile_name}`)。"
        )

    warnings.extend(_target_warnings(manifest_a.target, manifest_b.target))
    warnings.extend(
        _profile_warnings(
            manifest_a.profile,
            manifest_b.profile,
            measured=frozenset(manifest_a.suites) | frozenset(manifest_b.suites),
        )
    )

    tolerance_a = manifest_a.profile.compare_tolerance
    tolerance_b = manifest_b.profile.compare_tolerance
    if tolerance_a != tolerance_b:
        used = (
            f"判定には A の値 ({tolerance_a}) を使う。"
            if tolerance is None
            else f"判定には、指示された値 ({tolerance}) を使う。"
        )
        warnings.append(f"比較の許容の幅が違う (A: {tolerance_a}, B: {tolerance_b})。{used}")

    return warnings


def _target_warnings(target_a: TargetDef, target_b: TargetDef) -> list[str]:
    """名前は同じなのに、実体が違う対象サーバー (注 4.3)。

    繰り返しの結論 (9.3) を出すかどうかは、対象サーバーの**定義の名前**で
    決める。名前が同じなら結論は出したままにするが、接続先かモデルが違えば
    「同じものを 2 回測った」証拠にはならないので、参考ではなく、ふつうの警告
    として表より先に出す。名前そのものが違う比較は、参考の警告で報せている
    (構成どうしの比較なので、ふつうはそれでよい)。

    出すのは接続先とモデルだけである。認証の情報は、設定に環境変数の**名前**
    しかなく、値はどこにも入っていない (1.8)。ここでも読まないし、書かない。
    """
    if target_a.name != target_b.name:
        return []
    note = "同じものを 2 回測ったとは限らないので、繰り返しの結論はそのつもりで読むこと。"
    warnings: list[str] = []
    if str(target_a.base_url) != str(target_b.base_url):
        warnings.append(
            f"対象サーバーの定義の名前は同じ (`{target_a.name}`) だが、接続先が違う "
            f"(A: {target_a.base_url}, B: {target_b.base_url})。{note}"
        )
    if target_a.model != target_b.model:
        warnings.append(
            f"対象サーバーの定義の名前は同じ (`{target_a.name}`) だが、モデルが違う "
            f"(A: {target_a.model}, B: {target_b.model})。{note}"
        )
    return warnings


def _profile_warnings(
    profile_a: Profile, profile_b: Profile, *, measured: frozenset[SuiteName]
) -> list[str]:
    """計測の設定を丸ごと突き合わせ、違う項目を点でつないだ経路で報せる (9.4)。

    サンプリングの設定、出力の上限、試行の回数、乱数の種は、すべてこの差分に
    含まれる。項目を手で並べないので、設定が増えても見落とさない。

    比べた値に効きようのない項目 (`_informational_reason` を見ること) だけは、
    `(参考)` に落とす。落とさないほうに寄せてあるので、迷う項目はふつうの警告
    のままになる。`measured` は、**どちらかの計測ランが測った**まとまりである
    (片方だけが測ったまとまりの設定も、読み手には見せる)。
    """
    flat_a = _flatten(profile_a.model_dump(mode="json"))
    flat_b = _flatten(profile_b.model_dump(mode="json"))
    warnings: list[str] = []
    for path in sorted(set(flat_a) | set(flat_b)):
        if path in _PROFILE_PATHS_WARNED_ELSEWHERE:
            continue
        value_a = flat_a.get(path, _MISSING_FIELD)
        value_b = flat_b.get(path, _MISSING_FIELD)
        if value_a == value_b:
            continue
        reason = _informational_reason(path, measured)
        head = "(参考) " if reason is not None else ""
        detail = f"計測の設定 `{path}` が違う (A: {value_a}, B: {value_b})。"
        warnings.append(f"{head}{detail}{reason or ''}")
    return warnings


def _informational_reason(path: str, measured: frozenset[SuiteName]) -> str | None:
    """参考に落としてよい設定の差か。落としてよければ、その理由を返す (9.4)。

    落とすのは、次の 2 つだけである。

    1. どちらの計測ランも測っていないまとまりの設定。比べる行が 1 つも出ない
       ので、並べた値には効きようがない。コードの隔離 (`sandbox.*`) は、品質の
       検査だけが使うので、品質の検査に従う
    2. 内部の指標を読む間隔 (`metrics_interval_s`)。読む回数が変わるだけで、
       測る値の決め方は変わらない

    ほかは、すべてふつうの警告のままにする。サンプリングの設定、thinking、
    出力の上限、試行の回数、慣らしの回数、乱数の種、1 トークンあたりの文字数、
    制限時間、許容の幅、出力が壊れている疑いのしきい値は、測ったまとまりのもの
    を 1 つも落とさない。
    """
    if path in _PROFILE_PATHS_INFORMATIONAL:
        return "内部の指標を読む間隔なので、比べた値の決め方には効かない。"
    suite = _PROFILE_PATH_SUITES.get(path.split(".", 1)[0])
    if suite is not None and suite not in measured:
        return f"どちらの計測ランも `{suite.value}` を測っていないので、比べた値には効かない。"
    return None


def _flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """入れ子の写像を、点でつないだ経路 → 表示の文字列にする (1.3 の上書きの鍵と同じ形)。"""
    flat: dict[str, str] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        elif isinstance(value, str):
            flat[path] = value
        else:
            flat[path] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return flat


def _informational_warnings(manifest_a: RunManifest, manifest_b: RunManifest) -> list[str]:
    """測り方の食い違いではないが、読み手が知っておく違い (参考)。"""
    warnings: list[str] = []
    if manifest_a.target.name != manifest_b.target.name:
        warnings.append(
            f"(参考) 対象サーバーの定義が違う (A: `{manifest_a.target.name}`, "
            f"B: `{manifest_b.target.name}`)。構成どうしの比較なので、ふつうはこれでよい。"
            "繰り返しの結論 (9.3) は出さない。"
        )
    if manifest_a.server_model != manifest_b.server_model:
        warnings.append(
            f"(参考) サーバーが申告したモデルが違う (A: {manifest_a.server_model or '(不明)'}, "
            f"B: {manifest_b.server_model or '(不明)'})。"
        )
    if manifest_a.server_version != manifest_b.server_version:
        warnings.append(
            f"(参考) サーバーの版が違う (A: {manifest_a.server_version or '(不明)'}, "
            f"B: {manifest_b.server_version or '(不明)'})。"
        )
    return warnings


def _retokenization_warnings(manifest_a: RunManifest, manifest_b: RunManifest) -> list[str]:
    left = manifest_a.output_retokenization
    right = manifest_b.output_retokenization
    if left == right:
        return []

    def label(setting: _OutputRetokenization | None) -> str:
        if setting is None:
            return "記録なし"
        if not setting.enabled:
            return "無効"
        return f"有効 ({setting.method}, add_special_tokens={setting.add_special_tokens})"

    return [
        f"再計数の方法が異なる (A: {label(left)}, B: {label(right)})。"
        "段階別の再計数速度は判定しない。"
    ]


# --- 行 (9.1、9.2、9.6) ------------------------------------------------------


def _index(summary: Summary) -> dict[tuple[str, str], MetricResult]:
    """`(条件, 値の名前)` → 要約の行。並びは要約のまま (= 生データの順)。"""
    index: dict[tuple[str, str], MetricResult] = {}
    for row in summary.results:
        index.setdefault((row.condition, row.metric), row)
    return index


def _build_rows(
    summary_a: Summary,
    summary_b: Summary,
    *,
    values_a: TrialValues,
    values_b: TrialValues,
    tolerance: float,
) -> tuple[list[ComparisonRow], list[str], dict[tuple[str, str], str]]:
    """共通する `(条件, 値の名前)` の行を作る。

    行の並びは、計測ラン A の要約の並び (= A の生データの順) にする。同じ 2 つの
    計測ランからは、必ず同じ並びになる。
    """
    index_a = _index(summary_a)
    index_b = _index(summary_b)
    manifest_a = summary_a.conditions
    manifest_b = summary_b.conditions

    rows: list[ComparisonRow] = []
    warnings: list[str] = []
    markers: dict[tuple[str, str], str] = {}

    conditions_a = {condition for condition, _ in index_a}
    conditions_b = {condition for condition, _ in index_b}
    for key, row_a in index_a.items():
        row_b = index_b.get(key)
        if row_b is None:
            if key[1] in PHASE_METRIC_NAMES and key[0] in conditions_b:
                row, warning = _missing_phase_row(key, row_a, None)
                rows.append(row)
                warnings.append(warning)
                markers[key] = "段階別指標が不明"
            continue
        condition, metric = key
        tier: Tier = row_a.tier
        if row_a.tier != row_b.tier:
            warnings.append(
                f"`{condition}` の `{metric}` の区分が違う (A: {row_a.tier}, "
                f"B: {row_b.tier})。A の区分 ({tier}) で表に出す。"
            )

        value_a = _row_value(row_a)
        value_b = _row_value(row_b)
        comparable = metric not in RETOKENIZED_METRIC_NAMES or (
            row_a.count_source == row_b.count_source == "retokenized"
            and manifest_a.output_retokenization == manifest_b.output_retokenization
        )
        diff = None if not comparable or value_a is None or value_b is None else value_b - value_a
        relative_diff = (
            None if diff is None or value_a is None or value_a == 0.0 else diff / value_a
        )

        verdict: DiffVerdict | None = None
        marker: str | None = None
        if comparable and row_a.continuous is not None and row_b.continuous is not None:
            verdict, marker, pairing_warning = _continuous_verdict(
                key,
                values_a=values_a,
                values_b=values_b,
                manifest_a=manifest_a,
                manifest_b=manifest_b,
                tolerance=tolerance,
            )
            if pairing_warning is not None:
                warnings.append(pairing_warning)

        proportion_verdict = None
        if row_a.proportion is not None and row_b.proportion is not None:
            proportion_verdict = proportion_diff_verdict(row_a.proportion, row_b.proportion)

        if verdict is None and proportion_verdict is None and marker is None:
            marker = (
                "再計数の方法または計数元が異なる"
                if not comparable
                else "両方の計測ランにそろった値がない"
            )
        if verdict is None and proportion_verdict is None and marker is not None:
            markers[key] = marker
            warnings.append(
                f"`{condition}` の `{metric}` は判定できない ({marker})。"
                "「収まった」とは読まないこと。"
            )

        rows.append(
            ComparisonRow(
                condition=condition,
                metric=metric,
                tier=tier,
                value_a=value_a,
                value_b=value_b,
                diff=diff,
                relative_diff=relative_diff,
                verdict=verdict,
                proportion_verdict=proportion_verdict,
                count_source_a=row_a.count_source,
                count_source_b=row_b.count_source,
            )
        )
    for key, row_b in index_b.items():
        if key in index_a or key[1] not in PHASE_METRIC_NAMES or key[0] not in conditions_a:
            continue
        row, warning = _missing_phase_row(key, None, row_b)
        rows.append(row)
        warnings.append(warning)
        markers[key] = "段階別指標が不明"
    return rows, warnings, markers


def _missing_phase_row(
    key: tuple[str, str], row_a: MetricResult | None, row_b: MetricResult | None
) -> tuple[ComparisonRow, str]:
    condition, metric = key
    present = row_a if row_a is not None else row_b
    assert present is not None
    row = ComparisonRow(
        condition=condition,
        metric=metric,
        tier=present.tier,
        value_a=_row_value(row_a) if row_a is not None else None,
        value_b=_row_value(row_b) if row_b is not None else None,
        count_source_a=row_a.count_source if row_a is not None else None,
        count_source_b=row_b.count_source if row_b is not None else None,
    )
    missing = "A" if row_a is None else "B"
    kind = "再計数指標" if metric in RETOKENIZED_METRIC_NAMES else "段階別指標"
    return (
        row,
        f"`{condition}` の `{metric}` は計測ラン {missing} の{kind}が不明なので、"
        "比較から除外した。",
    )


def _row_value(row: MetricResult) -> float | None:
    """表に並べる値。連続の値は中央値、割合はその割合 (9.1)。"""
    if row.continuous is not None:
        return row.continuous.median
    if row.proportion is not None:
        return row.proportion.rate
    return None


def _continuous_verdict(
    key: tuple[str, str],
    *,
    values_a: TrialValues,
    values_b: TrialValues,
    manifest_a: RunManifest,
    manifest_b: RunManifest,
    tolerance: float,
) -> tuple[DiffVerdict | None, str | None, str | None]:
    """連続の値の行の判定 (9.2)。

    返すのは `(判定, 判定できない理由, 対応なしへ退化した警告)` である。
    退化の警告は、対応のある比較になるはずだったのに、番号がそろわなかった
    行だけに付く (注 4.3)。
    """
    items_a = list(values_a.get(key, ()))
    items_b = list(values_b.get(key, ()))
    if len(items_a) < _MIN_VALUES_FOR_VERDICT or len(items_b) < _MIN_VALUES_FOR_VERDICT:
        return (
            None,
            (
                f"値の数が足りない (A: {len(items_a)} 件、B: {len(items_b)} 件。"
                f"{_MIN_VALUES_FOR_VERDICT} 件以上が要る)"
            ),
            None,
        )

    pairing = _pairing(manifest_a, manifest_b, items_a, items_b)
    if pairing.paired:
        # 番号で突き合わせるため、両方を同じ鍵で並べ替えてから渡す
        items_a.sort(key=_value_key)
        items_b.sort(key=_value_key)
    verdict = diff_verdict(
        [item.value for item in items_a],
        [item.value for item in items_b],
        pairing.paired,
        tolerance,
        _row_seed(manifest_a.run_id, manifest_b.run_id, key),
    )
    warning = None
    if pairing.expected and not pairing.paired:
        condition, metric = key
        warning = (
            f"`{condition}` の `{metric}` は、試行の番号がそろわないので、対応のない比較に"
            f"した (A: {len(items_a)} 件、B: {len(items_b)} 件)。"
            "検出力が下がるので、同じ差でも「収まる」と出やすくなる。"
        )
    return verdict, None, warning


@dataclass(frozen=True)
class _Pairing:
    """対応のある比較にできるか (9.2) と、そもそも期待できたか。

    `expected` は、生成器の版と乱数の種がそろっていて、同じ番号の試行が同じ
    入力になるはずだった、ということである。期待できた比較が対応なしへ落ちた
    ことだけを、行ごとの警告にする (版や種の食い違いには、別の警告がある)。
    """

    paired: bool
    expected: bool


def _pairing(
    manifest_a: RunManifest,
    manifest_b: RunManifest,
    items_a: Sequence[TrialValue],
    items_b: Sequence[TrialValue],
) -> _Pairing:
    """対応のある比較にしてよいか (9.2 の 3 つの条件)。

    同じ番号が 2 回入っている列は、対応のある比較にしない。両方に同じ重なりが
    あると、並べ替えた番号の列は見かけ上そろうが、どの値とどの値の差を取るのか
    が決まらない (守りの判定)。
    """
    expected = (
        manifest_a.generator_version == manifest_b.generator_version
        and manifest_a.profile.seed == manifest_b.profile.seed
    )
    if not expected:
        return _Pairing(paired=False, expected=False)
    keys_a = sorted(_value_key(item) for item in items_a)
    keys_b = sorted(_value_key(item) for item in items_b)
    distinct = len(set(keys_a)) == len(keys_a) and len(set(keys_b)) == len(keys_b)
    return _Pairing(paired=bool(keys_a) and distinct and keys_a == keys_b, expected=True)


def _value_key(item: TrialValue) -> tuple[int, int, int]:
    """試行を突き合わせる鍵。`stream_index` がない値 (1 回ぶんの合計) も並べられる形にする。"""
    if item.stream_index is None:
        return (item.trial_index, 1, 0)
    return (item.trial_index, 0, item.stream_index)


def _row_seed(run_a: str, run_b: str, key: tuple[str, str]) -> int:
    """再標本化の種を、入力から決める (design.md 9.2 の Follow-up)。

    `hash()` は実行のたびに変わる (`PYTHONHASHSEED`) ので使わない。2 つの計測
    ランの識別子と、条件と値の名前から `sha256` で作るので、同じ比較は何度
    実行しても同じ区間になり、行ごとには違う種になる。
    """
    payload = "\x00".join((run_a, run_b, *key)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


# --- 片方にしかない条件 (9.5) -------------------------------------------------


def _excluded(summary_a: Summary, summary_b: Summary) -> list[str]:
    """比較から外した `(条件, 値の名前)` を、どちらにしかないかを添えて並べる。"""
    keys_a = set(_index(summary_a))
    keys_b = set(_index(summary_b))
    items = [f"{condition} / {metric} (A のみ)" for condition, metric in keys_a - keys_b]
    items += [f"{condition} / {metric} (B のみ)" for condition, metric in keys_b - keys_a]
    return sorted(items)


# --- 繰り返しの結論 (9.3、4.4) ------------------------------------------------


def _repeatability(
    rows: Sequence[ComparisonRow],
    markers: Mapping[tuple[str, str], str],
    manifest_a: RunManifest,
    manifest_b: RunManifest,
) -> RepeatabilityConclusion | None:
    """同じ対象サーバーの定義どうしの比較の結論 (9.3)。

    決めるのは主な結果の行だけで、参考の行は数えない (4.4)。判定できなかった
    行は、理由つきで一覧に出し、結論を「すべて収まった」にしない。
    """
    if manifest_a.target.name != manifest_b.target.name:
        return None

    primary = [row for row in rows if row.tier == "primary"]
    outside: list[str] = []
    for row in primary:
        label = f"{row.condition} / {row.metric}"
        if row.verdict is not None:
            if row.verdict.verdict == "outside":
                outside.append(label)
        elif row.proportion_verdict is not None:
            if row.proportion_verdict == "different":
                outside.append(label)
        else:
            marker = markers.get((row.condition, row.metric), "判定できない")
            outside.append(f"{label} (判定できない: {marker})")
    if not primary:
        outside.append("(比較できる主な結果の条件がないので、収まったとは言えない)")
    return RepeatabilityConclusion(all_within=not outside, outside=outside)


def _proportion_sides(
    summary_a: Summary, summary_b: Summary
) -> dict[tuple[str, str], tuple[ProportionStat | None, ProportionStat | None]]:
    """割合の行の、それぞれの側の分子と分母 (人が読む表に件数を添えるために持ち回る)。

    片方でしか採点できなかった行も入れる。判定はできないが、分かっている側の
    件数は表に出す (5.6: 割合には必ず分母と分子を添える)。
    """
    index_b = _index(summary_b)
    sides: dict[tuple[str, str], tuple[ProportionStat | None, ProportionStat | None]] = {}
    for key, row_a in _index(summary_a).items():
        row_b = index_b.get(key)
        if row_b is None:
            continue
        if row_a.proportion is None and row_b.proportion is None:
            continue
        sides[key] = (row_a.proportion, row_b.proportion)
    return sides


def _both_sides(
    sides: ProportionSides,
) -> dict[tuple[str, str], tuple[ProportionStat, ProportionStat]]:
    """両方にそろった行だけを取り出す (`ComparisonResult.proportions` の形)。"""
    pairs: dict[tuple[str, str], tuple[ProportionStat, ProportionStat]] = {}
    for key, (stat_a, stat_b) in sides.items():
        if stat_a is not None and stat_b is not None:
            pairs[key] = (stat_a, stat_b)
    return pairs


# --- 人が読む比較の結果 ------------------------------------------------------

_EMPTY: Final[str] = "—"

_CONTINUOUS_HEADER: Final[str] = (
    "| 条件 | 値 | A | B | 差 (B − A) | 差の割合 | 判定 | 対応 |"
    " 再標本化の差の中央値 | その 95% 区間 | 許容の幅 | n |"
)
_CONTINUOUS_RULE: Final[str] = "|---|---|--:|--:|--:|--:|---|---|--:|---|--:|--:|"

_PROPORTION_HEADER: Final[str] = "| 条件 | 値 | A | B | 差 (B − A) | 差の割合 | 判定 |"
_PROPORTION_RULE: Final[str] = "|---|---|--:|--:|--:|--:|---|"

_READING_NOTES: Final[tuple[str, ...]] = (
    "「対応あり」は、2 つの計測ランで生成器の版と乱数の種が同じで、値のある試行の番号が"
    "そろっているときだけ付く。そのときは、同じ番号どうしの差を取り直して判定する。"
    "「対応なし」は、それぞれの計測ランから独立に引き直して、中央値の差で判定する "
    "(入力が違えば速さも違うので、対応なしのほうが差を見つけにくい)。",
    "「収まる」は、再標本化した差の 95% 区間が 0 を含むか、差の割合の絶対値が許容の幅より"
    "小さいときである。どちらも満たさなければ「収まらない」になる。",
    "`差 (B − A)` と `差の割合` は、条件の**中央値**どうしから出す。"
    "`再標本化の差の中央値` は別の量で、対応のある比較では「試行ごとの差の中央値」である。"
    "2 つを同じ量として読まないこと。",
    "割合の行の判定は保守的である。両側 95% の区間が重ならないときだけ「差が意味あり」に"
    "するので、重なっていても本当は差がある場合はある。",
    "試行 10 回の対応のある比較では、同じ分布どうしでも約 8% の確率で「収まらない」が出る "
    "(許容の幅が 0 のとき。既定の 2% があれば、測り方のばらつきが 2% で 1 条件あたり "
    "1.5% ほど)。「収まらない」を構成の違いだと決める前に、まず試行の回数を 20 回以上に"
    "増やして測り直すこと。",
    "「判定できない」は「収まった」ではない。値の数が足りない行は、警告に理由を出している。",
)


def render_comparison_markdown(
    report: ComparisonReport, *, proportions: ProportionPairs | None = None
) -> str:
    """人が読む比較の結果 (9.1〜9.6、10.5)。

    警告 → 繰り返しの結論 → 主な結果 → 参考 → 外した条件 → 読み方、の順に出す。
    警告は、どの表よりも先に出す (9.4)。`proportions` を渡すと、割合の行に
    分子と分母を添える (`ComparisonRow` は割合しか持たないため)。

    `compare_runs` の返り値があるなら、`ComparisonResult.to_markdown()` を
    使うこと。`proportions=` を渡し忘れようがなく、片方でしか採点できなかった
    行の件数も残る (注 4.3)。
    """
    return _render_markdown(report, proportions or {})


def _render_markdown(report: ComparisonReport, sides: ProportionSides) -> str:
    """人が読む比較の結果の中身 (`sides` は、片側しかない行も持てる形)。"""
    lines: list[str] = [
        "# 計測ランの比較",
        "",
        f"- A: `{report.run_a}`",
        f"- B: `{report.run_b}`",
        "",
    ]
    lines += _warnings_section(report)
    lines += _repeatability_section(report)
    lines += _tier_section("主な結果", report, "primary", sides)
    lines += _tier_section("参考", report, "reference", sides)
    lines += _excluded_section(report)
    lines += _reading_section()
    return "\n".join(lines).rstrip("\n") + "\n"


def render_comparison_json(report: ComparisonReport) -> str:
    """道具が読む比較の結果。項目の並びは型の定義順で決まる。"""
    return report.model_dump_json(indent=2) + "\n"


def _warnings_section(report: ComparisonReport) -> list[str]:
    """警告の節。表より先に出すために、ここで必ず 1 つの節を作る (9.4、10.5)。"""
    lines = ["## 警告", ""]
    if not report.warnings:
        lines += ["- (条件の食い違いは見つからなかった)", ""]
        return lines
    lines += [f"- {_cell(warning)}" for warning in report.warnings]
    lines.append("")
    return lines


def _repeatability_section(report: ComparisonReport) -> list[str]:
    """繰り返しの結論 (9.3)。参考の条件は、数に入れないことを添える (4.4)。"""
    lines = ["## 繰り返しの結論", ""]
    reference_outside = sum(
        1
        for row in report.rows
        if row.tier == "reference"
        and (
            (row.verdict is not None and row.verdict.verdict == "outside")
            or row.proportion_verdict == "different"
        )
    )
    if report.repeatability is None:
        lines += [
            "対象サーバーの定義が違う比較なので、繰り返しの結論は出さない (警告を見ること)。",
            "",
        ]
        return lines
    if report.repeatability.all_within:
        lines.append(
            "同じ対象サーバーの定義どうしの比較で、主な結果のすべての条件が、"
            "測り方のばらつきの範囲に収まった。"
        )
    else:
        lines.append(
            f"同じ対象サーバーの定義どうしの比較で、主な結果の "
            f"{len(report.repeatability.outside)} 件が、測り方のばらつきの範囲に"
            "収まらなかった (または判定できなかった)。"
        )
        lines.append("")
        lines += [f"- `{_cell(item)}`" for item in report.repeatability.outside]
    if reference_outside:
        lines.append("")
        lines.append(f"参考の条件では {reference_outside} 件が収まらなかった (結論には数えない)。")
    lines.append("")
    return lines


def _tier_section(
    title: str, report: ComparisonReport, tier: Tier, sides: ProportionSides
) -> list[str]:
    """主な結果と参考を、別の節に分けて出す (4.4)。値の種類ごとに表を分ける。"""
    lines = [f"## {title}", ""]
    rows = [row for row in report.rows if row.tier == tier]
    if not rows:
        lines += ["(この比較には、該当する条件がない)", ""]
        return lines

    proportional = [row for row in rows if _is_proportion_row(row, sides)]
    continuous = [row for row in rows if not _is_proportion_row(row, sides)]
    if continuous:
        lines += [_CONTINUOUS_HEADER, _CONTINUOUS_RULE]
        lines += [_continuous_row(row) for row in continuous]
        lines.append("")
    if proportional:
        lines += ["### 割合で表す結果", "", _PROPORTION_HEADER, _PROPORTION_RULE]
        lines += [_proportion_row(row, sides) for row in proportional]
        lines.append("")
    return lines


def _is_proportion_row(row: ComparisonRow, sides: ProportionSides) -> bool:
    """割合の表に出す行か (4.2、5.6)。

    判定が付いた行だけで決めると、片方でしか採点できなかった行が、連続の値の
    表に落ちる (数える単位も欄も違うのに、速さの行と同じ形で並んでしまう)。
    値の名前と、どちらかに割合があることでも振り分ける。
    """
    return (
        row.proportion_verdict is not None
        or row.metric in _PROPORTION_METRICS
        or (row.condition, row.metric) in sides
    )


def _continuous_row(row: ComparisonRow) -> str:
    verdict = row.verdict
    cells = [
        f"`{_cell(row.condition)}`",
        f"`{_cell(row.metric)}`",
        _num(row.value_a),
        _num(row.value_b),
        _num(row.diff),
        _pct(row.relative_diff),
        _verdict_text(verdict),
        _paired_text(verdict),
        _num(verdict.median_diff) if verdict is not None else _EMPTY,
        _interval(verdict),
        _pct(verdict.tolerance, signed=False) if verdict is not None else _EMPTY,
        str(verdict.n) if verdict is not None else _EMPTY,
    ]
    return "| " + " | ".join(cells) + " |"


def _proportion_row(row: ComparisonRow, sides: ProportionSides) -> str:
    pair = sides.get((row.condition, row.metric))
    stat_a, stat_b = pair if pair is not None else (None, None)
    cells = [
        f"`{_cell(row.condition)}`",
        f"`{_cell(row.metric)}`",
        _rate(row.value_a, stat_a),
        _rate(row.value_b, stat_b),
        _num(row.diff),
        _pct(row.relative_diff),
        _proportion_verdict_text(row.proportion_verdict),
    ]
    return "| " + " | ".join(cells) + " |"


def _excluded_section(report: ComparisonReport) -> list[str]:
    """片方にしかない条件 (9.5)。"""
    lines = ["## 比較から外した条件", ""]
    if not report.excluded:
        lines += ["- (両方の計測ランに、同じ条件がそろっている)", ""]
        return lines
    lines += [f"- `{_cell(item)}`" for item in report.excluded]
    lines.append("")
    return lines


def _reading_section() -> list[str]:
    lines = ["## 読み方", ""]
    for note in _READING_NOTES:
        lines += [f"- {note}", ""]
    return lines


def _verdict_text(verdict: DiffVerdict | None) -> str:
    if verdict is None:
        return "判定できない"
    return "収まる" if verdict.verdict == "within" else "収まらない"


def _paired_text(verdict: DiffVerdict | None) -> str:
    if verdict is None:
        return _EMPTY
    return "対応あり" if verdict.paired else "対応なし"


def _proportion_verdict_text(outcome: str | None) -> str:
    if outcome is None:
        return "判定できない"
    return "差が意味あり" if outcome == "different" else "区別できない"


def _interval(verdict: DiffVerdict | None) -> str:
    if verdict is None:
        return _EMPTY
    return f"[{_num(verdict.ci95_low)}, {_num(verdict.ci95_high)}]"


def _rate(value: float | None, stat: ProportionStat | None) -> str:
    """割合と、分かっていれば分子と分母 (5.6 と同じく、件数を添えて読む)。"""
    if stat is None:
        return _num(value)
    return f"{_num(value)} ({stat.numerator}/{stat.denominator})"


def _num(value: float | None) -> str:
    """小数の書き方を固定する (何度書いても同じ文字列になるように)。"""
    if value is None or not math.isfinite(value):
        return _EMPTY
    if value == 0.0:
        return "0.000"  # -0.0 も 0.000 にする
    if abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.3f}"


def _pct(value: float | None, *, signed: bool = True) -> str:
    """割合を百分率にする。差の割合は向きが要るので符号を付け、許容の幅には付けない。"""
    if value is None or not math.isfinite(value):
        return _EMPTY
    if value == 0.0:
        return "0.00%"  # -0.0 も 0.00% にする
    return f"{value * 100.0:+.2f}%" if signed else f"{value * 100.0:.2f}%"


def _cell(text: str) -> str:
    """表の升目と箇条書きに入れられる形にする (改行と縦棒を逃がす)。"""
    return " ".join(text.split()).replace("|", "\\|")


# --- 別名で書いてから置き換える ---------------------------------------------


def write_comparison(
    report: ComparisonReport | ComparisonResult,
    out_path: Path,
    *,
    proportions: ProportionPairs | None = None,
) -> Path:
    """比較の結果を、人が読む形でファイルに書く (design.md: 指示があればファイルに)。

    `compare_runs` の返り値 (`ComparisonResult`) をそのまま渡せる。そのときは
    件数を中から取るので、`proportions=` は要らない (渡しても使わない)。
    `ComparisonReport` だけを渡すと、割合の行の分子と分母は `proportions=` を
    渡したぶんしか出ない (注 4.3)。

    同じディレクトリに別名で書いて `fsync` し、`os.replace` してから、ディレクトリ
    自体も `fsync` する (`analysis/summarize` と同じ決まり)。途中で落ちても、前の
    ファイルが壊れた中身にならない。
    """
    text = (
        report.to_markdown()
        if isinstance(report, ComparisonResult)
        else render_comparison_markdown(report, proportions=proportions)
    )
    data = text.encode("utf-8")
    directory = out_path.parent
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{out_path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return out_path
