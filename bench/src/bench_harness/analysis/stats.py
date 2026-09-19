"""統計の純粋な関数 (task 2.4)。

標準ライブラリ (`statistics`、`math`、`random`) だけで書く。`numpy` と `scipy`
は入れない (design.md の技術の選択)。依存するのは標準ライブラリ、pydantic、
`bench_harness.types` だけで、ほかの `bench_harness` の module を読み込まない。

どの関数も純粋である。時計も環境変数も、大域の乱数も読まない。再標本化の乱数
は引数の種から作った `random.Random` だけを使うので、同じ引数と同じ種なら、
何度呼んでも同じ結果になる (9.2 の Follow-up)。

測り方の決まり:

- **四分位範囲**: 線形補間の inclusive 法 (`statistics.quantiles` の
  `method="inclusive"`、R の type 7、numpy の既定と同じ)。標本の最小と最大を
  端に置くので外挿せず、n が 2 以上なら出せる。n が 1 のときは値なし
- **変動係数**: 標準偏差 / 平均。標準偏差がない (n が 1)、または平均が 0 の
  ときは値なし
- **割合の区間**: 正確な (Clopper-Pearson の) 区間。両側 95% を表示に使い、
  しきい値を下回ったかの判定には片側 95% の上限を使う (research.md
  「割合の不確かさは、正確な二項の区間で出す」)
- **2 組の差**: 差の中央値の 95% 区間を、再標本化 (1 万回、百分位法) で出す。
  対応のある場合は、試行の番号ごとの差を引き直す (research.md
  「2 つの計測ランの比較は、対応のある差の再標本化で判定する」)
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from typing import Final

from bench_harness.types import (
    Describe,
    DiffOutcome,
    DiffVerdict,
    ProportionDiffOutcome,
    ProportionStat,
    ThresholdVerdict,
)

__all__ = [
    "ALPHA",
    "BOOTSTRAP_RESAMPLES",
    "binomial_interval",
    "describe",
    "diff_verdict",
    "proportion_diff_verdict",
    "threshold_verdict",
    "trials_needed_for_zero_failures",
]

ALPHA: Final[float] = 0.05
"""区間の有意水準。両側は片方の裾に `ALPHA / 2`、片側は `ALPHA` を置く。"""

BOOTSTRAP_RESAMPLES: Final[int] = 10_000
"""再標本化の回数 (design.md analysis/stats)。"""

_BISECT_TOL: Final[float] = 1e-12
"""二分法を止める幅。区間の表示は小数 4〜6 桁なので、十分に細かい。"""

_SUM_TOL: Final[float] = 1e-18
"""裾の足し合わせを止める、合計に対する項の大きさ。"""


# --- 連続の値 -------------------------------------------------------------


def describe(values: Sequence[float]) -> Describe:
    """連続の値の記述統計を出す (2.4、3.7)。

    空の列と、有限でない値 (NaN、無限大) は `ValueError` にする。計測の失敗を
    NaN として集計に混ぜると、要約が黙って壊れるため。
    """
    data = _checked_floats(values, name="values")
    n = len(data)
    if n == 0:
        raise ValueError("describe: values が空である")
    ordered = sorted(data)
    mean = statistics.fmean(ordered)
    stdev = statistics.stdev(ordered) if n >= 2 else None
    if n >= 2:
        quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
        iqr: float | None = quartiles[2] - quartiles[0]
    else:
        iqr = None
    cv = stdev / mean if stdev is not None and mean != 0.0 else None
    return Describe(
        n=n,
        mean=mean,
        median=statistics.median(ordered),
        min=ordered[0],
        max=ordered[-1],
        stdev=stdev,
        iqr=iqr,
        cv=cv,
    )


# --- 二項の分布 -----------------------------------------------------------


def _log_pmf(k: int, n: int, p: float) -> float:
    """二項の確率質量の対数。`p` は 0 と 1 の間の値に限る。

    `math.comb` をそのまま使うと n が大きいところで巨大な整数になるので、
    `math.lgamma` で対数のまま計算する。
    """
    return (
        math.lgamma(n + 1.0)
        - math.lgamma(k + 1.0)
        - math.lgamma(n - k + 1.0)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def _lower_tail(k: int, n: int, p: float) -> float:
    """`P(X <= k)` を、`k` から下へ足して出す。

    `k` が最頻値以下のときだけ呼ぶ。そのとき項は下へ向かって単調に小さくなる
    ので、合計に対して無視できる大きさになった時点で打ち切ってよい。
    """
    term = math.exp(_log_pmf(k, n, p))
    total = term
    ratio = (1.0 - p) / p
    i = k
    while i > 0:
        term *= (i / (n - i + 1)) * ratio
        i -= 1
        if term <= 0.0:
            break
        total += term
        if term < total * _SUM_TOL:
            break
    return min(total, 1.0)


def _upper_tail(k: int, n: int, p: float) -> float:
    """`P(X >= k)` を、`k` から上へ足して出す。`k` が最頻値以上のときだけ呼ぶ。"""
    term = math.exp(_log_pmf(k, n, p))
    total = term
    ratio = p / (1.0 - p)
    i = k
    while i < n:
        term *= ((n - i) / (i + 1)) * ratio
        i += 1
        if term <= 0.0:
            break
        total += term
        if term < total * _SUM_TOL:
            break
    return min(total, 1.0)


def _cdf(k: int, n: int, p: float) -> float:
    """`P(X <= k)`。桁落ちを避けるため、小さいほうの裾を直に足す。"""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    if k <= n * p:
        return _lower_tail(k, n, p)
    return max(0.0, 1.0 - _upper_tail(k + 1, n, p))


def _sf(k: int, n: int, p: float) -> float:
    """`P(X >= k)`。"""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    if k >= n * p:
        return _upper_tail(k, n, p)
    return max(0.0, 1.0 - _lower_tail(k - 1, n, p))


def _solve_lower(k: int, n: int, tail: float) -> float:
    """`P(X >= k | p) = tail` を満たす `p`。左辺は `p` について単調に増える。"""
    lo, hi = 0.0, 1.0
    while hi - lo > _BISECT_TOL:
        mid = (lo + hi) / 2.0
        if _sf(k, n, mid) < tail:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _solve_upper(k: int, n: int, tail: float) -> float:
    """`P(X <= k | p) = tail` を満たす `p`。左辺は `p` について単調に減る。"""
    lo, hi = 0.0, 1.0
    while hi - lo > _BISECT_TOL:
        mid = (lo + hi) / 2.0
        if _cdf(k, n, mid) > tail:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def binomial_interval(k: int, n: int) -> ProportionStat:
    """割合の、正確な (Clopper-Pearson の) 区間を出す (5.6、6.5)。

    両側 95% (`ci95_low`、`ci95_high`) は、それぞれの裾に `ALPHA / 2` を置く。
    `upper95_one_sided` は、裾に `ALPHA` を置いた片側の上限で、しきい値を
    下回ったと言えるかの判定に使う (`threshold_verdict`)。

    端では閉じた式を使う。`k = 0` の上限は `1 - 裾^(1/n)`、`k = n` の下限は
    `裾^(1/n)` で、二分法より正確に出せる。
    """
    if n < 1:
        raise ValueError(f"binomial_interval: n は 1 以上である必要がある (n={n})")
    if not 0 <= k <= n:
        raise ValueError(f"binomial_interval: k は 0 以上 n 以下である必要がある (k={k}, n={n})")
    half = ALPHA / 2.0
    if k == 0:
        low = 0.0
        high = -math.expm1(math.log(half) / n)
        upper_one_sided = -math.expm1(math.log(ALPHA) / n)
    elif k == n:
        low = math.exp(math.log(half) / n)
        high = 1.0
        upper_one_sided = 1.0
    else:
        low = _solve_lower(k, n, half)
        high = _solve_upper(k, n, half)
        upper_one_sided = _solve_upper(k, n, ALPHA)
    return ProportionStat(
        numerator=k,
        denominator=n,
        rate=k / n,
        ci95_low=low,
        ci95_high=high,
        upper95_one_sided=upper_one_sided,
    )


def threshold_verdict(stat: ProportionStat, threshold: float) -> ThresholdVerdict:
    """割合が、しきい値を下回ったと言えるかを判定する (6.5)。

    「下回った」は片側の主張なので、片側 95% の上限で判定する。「上回った」は
    反対向きの片側の主張だが、表示している両側の区間と食い違わないように、
    両側 95% の下限で判定する (どちらもより保守的な側に倒す)。
    """
    if stat.upper95_one_sided < threshold:
        return ThresholdVerdict.BELOW
    if stat.ci95_low > threshold:
        return ThresholdVerdict.ABOVE
    return ThresholdVerdict.UNDETERMINED


def trials_needed_for_zero_failures(threshold: float) -> int:
    """崩れ 0 件で、しきい値を下回ったと言うのに要る試行の数を返す (6.5)。

    `1 - ALPHA^(1/n) < threshold` を満たす最小の `n`。1% なら 299 になる。
    見積もりを閉じた式から出したあと、`threshold_verdict` が実際に「下回った」
    になるまで 1 ずつ進めるので、判定と必ず辻褄が合う。
    """
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError(
            f"trials_needed_for_zero_failures: threshold は 0 と 1 の間である必要がある "
            f"(threshold={threshold})"
        )
    estimate = math.log(ALPHA) / math.log1p(-threshold)
    n = max(1, math.floor(estimate) - 1)
    while threshold_verdict(binomial_interval(0, n), threshold) is not ThresholdVerdict.BELOW:
        n += 1
    return n


def proportion_diff_verdict(a: ProportionStat, b: ProportionStat) -> ProportionDiffOutcome:
    """2 つの割合の差が、意味のあるものかを判定する (9.6)。

    両側 95% の区間が重ならないときだけ `different` にする。区間が重なっても
    差がある場合はあるので、これは保守的な判定である (比較の結果にその旨を
    書く: design.md analysis/stats)。
    """
    if a.ci95_high < b.ci95_low or b.ci95_high < a.ci95_low:
        return "different"
    return "not_distinguishable"


# --- 2 組の差 -------------------------------------------------------------


def _median(values: list[float]) -> float:
    """並べ替えてから中央値を返す (`statistics.median` と同じ値、少し速い)。"""
    ordered = sorted(values)
    size = len(ordered)
    half = size // 2
    if size % 2 == 1:
        return ordered[half]
    return (ordered[half - 1] + ordered[half]) / 2.0


def diff_verdict(
    a: Sequence[float],
    b: Sequence[float],
    paired: bool,
    tolerance: float,
    seed: int,
) -> DiffVerdict:
    """2 組の値の差が、ばらつきの範囲に収まるかを判定する (9.2)。

    差の中央値の 95% 区間を、再標本化 (`BOOTSTRAP_RESAMPLES` 回、百分位法) で
    出す。区間が 0 を含むか、差の割合の絶対値が `tolerance` より小さければ
    `within`、そうでなければ `outside`。

    `paired` が真のときは、試行の番号ごとの差を引き直す (両方の計測ランで
    同じ番号の試行が同じ入力になるときに使う)。偽のときは、それぞれの組から
    独立に引き直して、中央値の差を取る。

    返す値の意味:

    - `median_diff` は、区間と同じ統計量の、元の標本での値。対応のある場合は
      試行ごとの差の中央値、対応のない場合は中央値どうしの差になる
    - `relative_diff` は、比較の表に並べる差の割合
      (`(median(b) - median(a)) / median(a)`)。`median(a)` が 0 のときは
      値なしにして、区間の決まりだけで判定する
    - `n` は、判定の根拠になった試行の数 (対応のある場合は組の数、ない場合は
      少ないほうの組の大きさ)
    """
    xs = _checked_floats(a, name="a")
    ys = _checked_floats(b, name="b")
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError(
            f"diff_verdict: それぞれ 2 つ以上の値が要る (len(a)={len(xs)}, len(b)={len(ys)})"
        )
    if paired and len(xs) != len(ys):
        raise ValueError(
            f"diff_verdict: 対応のある比較では長さが同じである必要がある "
            f"(len(a)={len(xs)}, len(b)={len(ys)})"
        )
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"diff_verdict: tolerance は 0 以上の有限の値である (tolerance={tolerance})"
        )

    median_a = _median(xs)
    median_b = _median(ys)
    relative_diff = None if median_a == 0.0 else (median_b - median_a) / median_a

    rng = random.Random(seed)
    samples: list[float] = []
    if paired:
        diffs = [y - x for x, y in zip(xs, ys, strict=True)]
        size = len(diffs)
        median_diff = _median(diffs)
        # 番号を引き直すのと、差の列から引き直すのは同じこと (どちらも、同じ
        # 番号の差を重複ありで size 個選ぶ)。`choices` のほうがずっと速い
        for _ in range(BOOTSTRAP_RESAMPLES):
            samples.append(_median(rng.choices(diffs, k=size)))
        n = size
    else:
        median_diff = median_b - median_a
        size_a, size_b = len(xs), len(ys)
        for _ in range(BOOTSTRAP_RESAMPLES):
            resampled_a = _median(rng.choices(xs, k=size_a))
            resampled_b = _median(rng.choices(ys, k=size_b))
            samples.append(resampled_b - resampled_a)
        n = min(size_a, size_b)

    samples.sort()
    low_index = math.floor(ALPHA / 2.0 * BOOTSTRAP_RESAMPLES)
    high_index = math.ceil((1.0 - ALPHA / 2.0) * BOOTSTRAP_RESAMPLES) - 1
    ci95_low = samples[low_index]
    ci95_high = samples[high_index]

    within = (ci95_low <= 0.0 <= ci95_high) or (
        relative_diff is not None and abs(relative_diff) < tolerance
    )
    verdict: DiffOutcome = "within" if within else "outside"
    return DiffVerdict(
        verdict=verdict,
        paired=paired,
        median_diff=median_diff,
        relative_diff=relative_diff,
        ci95_low=ci95_low,
        ci95_high=ci95_high,
        tolerance=tolerance,
        n=n,
    )


# --- 助け -----------------------------------------------------------------


def _checked_floats(values: Sequence[float], *, name: str) -> list[float]:
    """`float` の列に直し、有限でない値があれば `ValueError` にする。

    入れ物の種類 (list、tuple) で結果が変わらないように、ここで list にする。
    """
    data = [float(value) for value in values]
    for value in data:
        if not math.isfinite(value):
            raise ValueError(f"{name} に有限でない値がある ({value!r})")
    return data
