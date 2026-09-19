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

値が 2 つに満たない行は判定しない (`verdict=None`)。「判定できない」を「収まっ
た」と読ませないため、警告にも、繰り返しの結論の一覧にも、理由つきで出す。

## 繰り返しの結論 (9.3、4.4)

同じ対象サーバーの定義 (名前) どうしの比較のときだけ出す。結論を決めるのは
**主な結果 (primary)** の行だけで、参考 (reference) の行は表に出すが結論には
数えない (4.4: 1 本と 2 本が主な結果、4 本と 8 本は余力を知るための参考)。
判定できなかった行は、結論を `all_within=False` にして、理由つきで一覧に出す。

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
from bench_harness.analysis.summarize import TrialValue, summarize_run, trial_values
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
    Summary,
    Tier,
)

__all__ = [
    "ComparisonResult",
    "ProportionPairs",
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
    """

    report: ComparisonReport
    read_warnings_a: list[str] = field(default_factory=list)
    read_warnings_b: list[str] = field(default_factory=list)
    proportions: dict[tuple[str, str], tuple[ProportionStat, ProportionStat]] = field(
        default_factory=dict
    )


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
    return ComparisonResult(
        report=report,
        read_warnings_a=list(data_a.warnings),
        read_warnings_b=list(data_b.warnings),
        proportions=_proportion_pairs(data_a.summary, data_b.summary),
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
        *_setting_warnings(manifest_a, manifest_b, tolerance=tolerance),
        *_informational_warnings(manifest_a, manifest_b),
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

    warnings.extend(_profile_warnings(manifest_a.profile, manifest_b.profile))

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


def _profile_warnings(profile_a: Profile, profile_b: Profile) -> list[str]:
    """計測の設定を丸ごと突き合わせ、違う項目を点でつないだ経路で報せる (9.4)。

    サンプリングの設定、出力の上限、試行の回数、乱数の種は、すべてこの差分に
    含まれる。項目を手で並べないので、設定が増えても見落とさない。
    """
    flat_a = _flatten(profile_a.model_dump(mode="json"))
    flat_b = _flatten(profile_b.model_dump(mode="json"))
    warnings: list[str] = []
    for path in sorted(set(flat_a) | set(flat_b)):
        if path in _PROFILE_PATHS_WARNED_ELSEWHERE:
            continue
        value_a = flat_a.get(path, _MISSING_FIELD)
        value_b = flat_b.get(path, _MISSING_FIELD)
        if value_a != value_b:
            warnings.append(f"計測の設定 `{path}` が違う (A: {value_a}, B: {value_b})。")
    return warnings


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

    for key, row_a in index_a.items():
        row_b = index_b.get(key)
        if row_b is None:
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
        diff = None if value_a is None or value_b is None else value_b - value_a
        relative_diff = (
            None if diff is None or value_a is None or value_a == 0.0 else diff / value_a
        )

        verdict: DiffVerdict | None = None
        marker: str | None = None
        if row_a.continuous is not None and row_b.continuous is not None:
            verdict, marker = _continuous_verdict(
                key,
                values_a=values_a,
                values_b=values_b,
                manifest_a=manifest_a,
                manifest_b=manifest_b,
                tolerance=tolerance,
            )

        proportion_verdict = None
        if row_a.proportion is not None and row_b.proportion is not None:
            proportion_verdict = proportion_diff_verdict(row_a.proportion, row_b.proportion)

        if verdict is None and proportion_verdict is None and marker is None:
            marker = "両方の計測ランにそろった値がない"
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
            )
        )
    return rows, warnings, markers


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
) -> tuple[DiffVerdict | None, str | None]:
    """連続の値の行の判定 (9.2)。判定できないときは、理由を返す。"""
    items_a = list(values_a.get(key, ()))
    items_b = list(values_b.get(key, ()))
    if len(items_a) < _MIN_VALUES_FOR_VERDICT or len(items_b) < _MIN_VALUES_FOR_VERDICT:
        return None, (
            f"値の数が足りない (A: {len(items_a)} 件、B: {len(items_b)} 件。"
            f"{_MIN_VALUES_FOR_VERDICT} 件以上が要る)"
        )

    paired = _is_paired(manifest_a, manifest_b, items_a, items_b)
    if paired:
        # 番号で突き合わせるため、両方を同じ鍵で並べ替えてから渡す
        items_a.sort(key=_value_key)
        items_b.sort(key=_value_key)
    verdict = diff_verdict(
        [item.value for item in items_a],
        [item.value for item in items_b],
        paired,
        tolerance,
        _row_seed(manifest_a.run_id, manifest_b.run_id, key),
    )
    return verdict, None


def _is_paired(
    manifest_a: RunManifest,
    manifest_b: RunManifest,
    items_a: Sequence[TrialValue],
    items_b: Sequence[TrialValue],
) -> bool:
    """対応のある比較にしてよいか (9.2 の 3 つの条件)。"""
    if manifest_a.generator_version != manifest_b.generator_version:
        return False
    if manifest_a.profile.seed != manifest_b.profile.seed:
        return False
    keys_a = sorted(_value_key(item) for item in items_a)
    keys_b = sorted(_value_key(item) for item in items_b)
    return bool(keys_a) and keys_a == keys_b


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


def _proportion_pairs(
    summary_a: Summary, summary_b: Summary
) -> dict[tuple[str, str], tuple[ProportionStat, ProportionStat]]:
    """割合の行の、両方の分子と分母 (人が読む表に件数を添えるために持ち回る)。"""
    index_b = _index(summary_b)
    pairs: dict[tuple[str, str], tuple[ProportionStat, ProportionStat]] = {}
    for key, row_a in _index(summary_a).items():
        row_b = index_b.get(key)
        if row_b is None or row_a.proportion is None or row_b.proportion is None:
            continue
        pairs[key] = (row_a.proportion, row_b.proportion)
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
    """
    lines: list[str] = [
        "# 計測ランの比較",
        "",
        f"- A: `{report.run_a}`",
        f"- B: `{report.run_b}`",
        "",
    ]
    lines += _warnings_section(report)
    lines += _repeatability_section(report)
    lines += _tier_section("主な結果", report, "primary", proportions)
    lines += _tier_section("参考", report, "reference", proportions)
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
    title: str, report: ComparisonReport, tier: Tier, proportions: ProportionPairs | None
) -> list[str]:
    """主な結果と参考を、別の節に分けて出す (4.4)。値の種類ごとに表を分ける。"""
    lines = [f"## {title}", ""]
    rows = [row for row in report.rows if row.tier == tier]
    if not rows:
        lines += ["(この比較には、該当する条件がない)", ""]
        return lines

    continuous = [row for row in rows if row.proportion_verdict is None]
    proportional = [row for row in rows if row.proportion_verdict is not None]
    if continuous:
        lines += [_CONTINUOUS_HEADER, _CONTINUOUS_RULE]
        lines += [_continuous_row(row) for row in continuous]
        lines.append("")
    if proportional:
        lines += ["### 割合で表す結果", "", _PROPORTION_HEADER, _PROPORTION_RULE]
        lines += [_proportion_row(row, proportions) for row in proportional]
        lines.append("")
    return lines


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


def _proportion_row(row: ComparisonRow, proportions: ProportionPairs | None) -> str:
    pair = (proportions or {}).get((row.condition, row.metric))
    stat_a, stat_b = (pair[0], pair[1]) if pair is not None else (None, None)
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
    report: ComparisonReport, out_path: Path, *, proportions: ProportionPairs | None = None
) -> Path:
    """比較の結果を、人が読む形でファイルに書く (design.md: 指示があればファイルに)。

    同じディレクトリに別名で書いて `fsync` し、`os.replace` してから、ディレクトリ
    自体も `fsync` する (`analysis/summarize` と同じ決まり)。途中で落ちても、前の
    ファイルが壊れた中身にならない。
    """
    data = render_comparison_markdown(report, proportions=proportions).encode("utf-8")
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
