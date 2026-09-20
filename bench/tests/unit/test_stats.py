"""統計の関数の試験 (task 2.4)。

区間の参照値は、実装とは別の道筋で用意する。

- 閉じた式: 崩れ 0 件の Clopper-Pearson の上限は `1 - alpha^(1/n)` で、
  二分法を使わずに確かめられる
- 公表されている Clopper-Pearson の値: `k=5, n=100` の両側 95% は
  [0.0164, 0.1128]、`k=50, n=100` は [0.3983, 0.6017] (二項の正確な区間の表と、
  正規化された不完全ベータ関数 `I_x(a, b)` の連分数展開から出した値。
  区間は `low = BetaInv(0.025; k, n-k+1)`、`high = BetaInv(0.975; k+1, n-k)`)
- 被覆の性質: 区間が真の割合を含む確率を、二項の確率質量を `math.comb` で
  足し合わせて出す。裾を取り違えた実装は、ここで落ちる

再標本化の試験は、判定が種の当たり外れで変わらないように、差が区間の幅に
比べて十分に大きい (または十分に小さい) 組を使う。
"""

from __future__ import annotations

import math
import random
import statistics
import time
from collections.abc import Sequence

import pytest

from bench_harness.analysis.stats import (
    ALPHA,
    BOOTSTRAP_RESAMPLES,
    binomial_interval,
    describe,
    diff_verdict,
    proportion_diff_verdict,
    threshold_verdict,
    trials_needed_for_zero_failures,
)
from bench_harness.types import ProportionStat, ThresholdVerdict

# --- 助け -----------------------------------------------------------------


def _binom_pmf(k: int, n: int, p: float) -> float:
    """二項の確率質量 (被覆の確認に使う、独立した計算)。"""
    return math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def _gauss_sample(seed: int, n: int, mu: float, sigma: float) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


def _bootstrap_median_cdf(j: int, n: int) -> float:
    """相異なる n 個 (奇数) から n 個を引き直したとき、中央値が j 番目以下になる確率。

    中央値は (n+1)/2 番目の順序統計量なので、引いた n 個のうち j 番目以下の値が
    (n+1)/2 個以上あることと同じ。再標本化の区間の水準 (95% か 90% か) を、
    実装とは別に決めるために使う。
    """
    middle = n // 2
    p = j / n
    return sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(middle + 1, n + 1))


def _analytic_quantile(level: float, n: int) -> int:
    """再標本化した中央値の、`level` の分位に当たる順序 (1 始まり)。"""
    return next(j for j in range(1, n + 1) if _bootstrap_median_cdf(j, n) >= level)


# --- 設計で決まっている定数 -----------------------------------------------


def test_design_constants() -> None:
    # design.md analysis/stats: 区間は 95%、再標本化は 1 万回
    assert ALPHA == 0.05
    assert BOOTSTRAP_RESAMPLES == 10_000


# --- describe -------------------------------------------------------------


def test_describe_hand_computed() -> None:
    # 並べ替えた値: 2, 4, 4, 4, 5, 5, 7, 9
    # 平均 40/8 = 5.0、中央値 (4+5)/2 = 4.5、標本標準偏差 sqrt(32/7) = 2.13809...
    # 四分位 (inclusive、R の type 7): Q1 = 4.0、Q3 = 5.5 なので IQR = 1.5
    # 変動係数 = 2.13809.../5.0 = 0.4276...
    values = [4.0, 2.0, 7.0, 4.0, 9.0, 5.0, 4.0, 5.0]

    got = describe(values)

    assert got.n == 8
    assert got.mean == pytest.approx(5.0)
    assert got.median == pytest.approx(4.5)
    assert got.min == pytest.approx(2.0)
    assert got.max == pytest.approx(9.0)
    assert got.stdev == pytest.approx(math.sqrt(32.0 / 7.0))
    assert got.iqr == pytest.approx(1.5)
    assert got.cv == pytest.approx(math.sqrt(32.0 / 7.0) / 5.0)
    # 取り違えを見つけるため、8 つの値がすべて違うことを確かめる
    assert len({got.mean, got.median, got.min, got.max, got.stdev, got.iqr, got.cv}) == 7


def test_describe_single_value() -> None:
    got = describe([3.5])

    assert got.n == 1
    assert got.mean == pytest.approx(3.5)
    assert got.median == pytest.approx(3.5)
    assert got.min == pytest.approx(3.5)
    assert got.max == pytest.approx(3.5)
    assert got.stdev is None
    assert got.iqr is None
    assert got.cv is None


def test_describe_constant_values() -> None:
    got = describe([3.0] * 5)

    assert got.stdev == pytest.approx(0.0)
    assert got.iqr == pytest.approx(0.0)
    # 平均が 0 でないので、変動係数は 0.0 (値なしにはしない)
    assert got.cv == pytest.approx(0.0)


def test_describe_mean_zero_has_no_cv() -> None:
    got = describe([-1.0, 1.0])

    assert got.mean == pytest.approx(0.0)
    assert got.stdev == pytest.approx(math.sqrt(2.0))
    assert got.iqr == pytest.approx(1.0)
    assert got.cv is None


def test_describe_rejects_empty() -> None:
    with pytest.raises(ValueError):
        describe([])


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_describe_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError):
        describe([1.0, bad, 2.0])


def test_describe_accepts_any_sequence() -> None:
    values = [1.0, 2.0, 4.0, 8.0]

    assert describe(tuple(values)).model_dump() == describe(values).model_dump()


# --- binomial_interval ----------------------------------------------------


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        # 公表されている Clopper-Pearson の値 (両側 95%)
        (5, 100, 0.016431879, 0.112834911),
        (50, 100, 0.398321130, 0.601678870),
        (1, 50, 0.000506228, 0.106469546),
        (3, 20, 0.032070937, 0.378926827),
        (2, 7, 0.036692566, 0.709579136),
    ],
)
def test_binomial_interval_matches_reference(k: int, n: int, low: float, high: float) -> None:
    got = binomial_interval(k, n)

    assert got.numerator == k
    assert got.denominator == n
    assert got.rate == pytest.approx(k / n)
    assert got.ci95_low == pytest.approx(low, abs=1e-8)
    assert got.ci95_high == pytest.approx(high, abs=1e-8)


@pytest.mark.parametrize("n", [1, 10, 50, 299])
def test_binomial_interval_zero_successes_uses_closed_form(n: int) -> None:
    got = binomial_interval(0, n)

    assert got.ci95_low == 0.0
    assert got.ci95_high == pytest.approx(1.0 - 0.025 ** (1.0 / n), abs=1e-12)
    assert got.upper95_one_sided == pytest.approx(1.0 - 0.05 ** (1.0 / n), abs=1e-12)


@pytest.mark.parametrize("n", [1, 10, 100])
def test_binomial_interval_all_successes(n: int) -> None:
    got = binomial_interval(n, n)

    assert got.ci95_high == 1.0
    assert got.upper95_one_sided == 1.0
    assert got.ci95_low == pytest.approx(0.025 ** (1.0 / n), abs=1e-9)


def test_binomial_interval_299_is_below_one_percent_and_298_is_not() -> None:
    # tasks.md 2.4 の完了の状態
    assert binomial_interval(0, 299).upper95_one_sided < 0.01
    assert binomial_interval(0, 298).upper95_one_sided >= 0.01


@pytest.mark.parametrize(("k", "n"), [(0, 7), (1, 7), (3, 7), (7, 7), (11, 40), (29, 40)])
def test_binomial_interval_is_mirrored(k: int, n: int) -> None:
    got = binomial_interval(k, n)
    mirrored = binomial_interval(n - k, n)

    assert got.ci95_low == pytest.approx(1.0 - mirrored.ci95_high, abs=1e-9)
    assert got.ci95_high == pytest.approx(1.0 - mirrored.ci95_low, abs=1e-9)


@pytest.mark.parametrize(("n", "p"), [(20, 0.1), (15, 0.3), (30, 0.5), (25, 0.85)])
def test_binomial_interval_covers_at_least_95_percent(n: int, p: float) -> None:
    # 裾を取り違えた実装 (両側の上下を入れ替えた、片側の確率を使った) は、
    # 被覆が 95% を割るのでここで落ちる
    covered = sum(
        _binom_pmf(k, n, p)
        for k in range(n + 1)
        if binomial_interval(k, n).ci95_low <= p <= binomial_interval(k, n).ci95_high
    )

    assert covered >= 0.95


def test_binomial_interval_is_monotonic_in_k() -> None:
    n = 40
    stats = [binomial_interval(k, n) for k in range(n + 1)]

    for before, after in zip(stats, stats[1:], strict=False):
        assert before.ci95_low <= after.ci95_low
        assert before.ci95_high <= after.ci95_high
        assert before.upper95_one_sided <= after.upper95_one_sided


@pytest.mark.parametrize(("k", "n"), [(0, 40), (5, 40), (20, 40), (39, 40)])
def test_binomial_interval_one_sided_upper_is_tighter(k: int, n: int) -> None:
    got = binomial_interval(k, n)

    assert got.ci95_low <= got.rate <= got.ci95_high
    assert got.upper95_one_sided < got.ci95_high


def test_binomial_interval_large_n_is_finite_and_fast() -> None:
    started = time.perf_counter()
    got = binomial_interval(137, 10_000)
    elapsed = time.perf_counter() - started

    assert math.isfinite(got.ci95_low)
    assert math.isfinite(got.ci95_high)
    assert got.ci95_low < got.rate < got.ci95_high
    # 不完全ベータ関数から出した参照値 (BetaInv(0.025; 137, 9864)、BetaInv(0.975; 138, 9863))
    assert got.ci95_low == pytest.approx(0.011514170, abs=1e-8)
    assert got.ci95_high == pytest.approx(0.016175380, abs=1e-8)
    assert elapsed < 3.0


@pytest.mark.parametrize(("k", "n"), [(-1, 10), (11, 10), (0, 0), (0, -1), (1, 0)])
def test_binomial_interval_rejects_invalid_counts(k: int, n: int) -> None:
    with pytest.raises(ValueError):
        binomial_interval(k, n)


# --- threshold_verdict ----------------------------------------------------


def test_threshold_verdict_below() -> None:
    assert threshold_verdict(binomial_interval(0, 299), 0.01) is ThresholdVerdict.BELOW


def test_threshold_verdict_undetermined_when_too_few_trials() -> None:
    # quick の設定 (1 段階 50 回) では、崩れ 0 件でも 1% 未満とは言えない
    assert threshold_verdict(binomial_interval(0, 50), 0.01) is ThresholdVerdict.UNDETERMINED


def test_threshold_verdict_above() -> None:
    assert threshold_verdict(binomial_interval(30, 100), 0.01) is ThresholdVerdict.ABOVE


def test_threshold_verdict_uses_one_sided_upper_bound() -> None:
    # 両側の上限は 1% 以上だが、片側の上限は 1% 未満。両側で判定する実装は落ちる
    stat = binomial_interval(0, 299)

    assert stat.ci95_high >= 0.01
    assert stat.upper95_one_sided < 0.01
    assert threshold_verdict(stat, 0.01) is ThresholdVerdict.BELOW


def test_threshold_verdict_uses_two_sided_lower_bound_for_above() -> None:
    # 片側の上限より、両側の下限のほうが小さい。上回ったの判定は両側の下限で行う
    stat = binomial_interval(2, 100)

    assert stat.ci95_low == pytest.approx(0.0024, abs=1e-3)
    assert threshold_verdict(stat, 0.002) is ThresholdVerdict.ABOVE
    assert threshold_verdict(stat, 0.01) is ThresholdVerdict.UNDETERMINED


# --- trials_needed_for_zero_failures --------------------------------------


def test_trials_needed_for_one_percent_is_299() -> None:
    assert trials_needed_for_zero_failures(0.01) == 299


@pytest.mark.parametrize("threshold", [0.01, 0.05, 0.001])
def test_trials_needed_matches_closed_form(threshold: float) -> None:
    # 1 - 0.05^(1/n) < t となる最小の n は、n > ln(0.05) / ln(1 - t) の最小の整数
    expected = math.floor(math.log(0.05) / math.log(1.0 - threshold)) + 1

    assert trials_needed_for_zero_failures(threshold) == expected


@pytest.mark.parametrize("threshold", [0.01, 0.05, 0.001])
def test_trials_needed_is_the_smallest_n_that_is_below(threshold: float) -> None:
    n = trials_needed_for_zero_failures(threshold)

    assert threshold_verdict(binomial_interval(0, n), threshold) is ThresholdVerdict.BELOW
    assert threshold_verdict(binomial_interval(0, n - 1), threshold) is not ThresholdVerdict.BELOW


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.5, math.nan])
def test_trials_needed_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        trials_needed_for_zero_failures(threshold)


# --- diff_verdict ---------------------------------------------------------


def test_diff_verdict_same_distribution_is_within() -> None:
    # 許容の幅を 0 にして、区間の決まりだけで判定させる
    a = _gauss_sample(seed=11, n=40, mu=100.0, sigma=5.0)
    b = _gauss_sample(seed=12, n=40, mu=100.0, sigma=5.0)

    got = diff_verdict(a, b, paired=False, tolerance=0.0, seed=20260919)

    assert got.verdict == "within"
    assert got.ci95_low <= 0.0 <= got.ci95_high
    assert got.paired is False
    assert got.tolerance == pytest.approx(0.0)
    assert got.n == 40


@pytest.mark.parametrize("paired", [True, False])
def test_diff_verdict_ten_percent_shift_is_outside(paired: bool) -> None:
    a = _gauss_sample(seed=21, n=20, mu=100.0, sigma=3.0)
    b = [v * 1.1 for v in a]

    got = diff_verdict(a, b, paired=paired, tolerance=0.02, seed=20260919)

    assert got.verdict == "outside"
    assert got.paired is paired
    assert got.relative_diff == pytest.approx(0.1, abs=1e-9)
    assert not (got.ci95_low <= 0.0 <= got.ci95_high)
    assert got.ci95_low <= got.median_diff <= got.ci95_high


def test_diff_verdict_small_shift_is_within_by_tolerance() -> None:
    rng = random.Random(33)
    a = [rng.uniform(90.0, 110.0) for _ in range(24)]
    b = [v * 1.005 for v in a]

    lenient = diff_verdict(a, b, paired=True, tolerance=0.02, seed=20260919)
    strict = diff_verdict(a, b, paired=True, tolerance=0.001, seed=20260919)

    # 区間は 0 を含まない (統計としては、はっきりした差)
    assert lenient.ci95_low > 0.0
    assert lenient.relative_diff == pytest.approx(0.005, abs=1e-9)
    # 対応のある場合の代表値は、試行ごとの差の中央値 (偶数個なので真ん中 2 つの平均)
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    assert lenient.median_diff == pytest.approx(statistics.median(diffs))
    # 許容の幅の決まりで「収まる」になる
    assert lenient.verdict == "within"
    # 許容の幅を差より狭くすると「収まらない」になる
    assert strict.verdict == "outside"


def test_diff_verdict_paired_detects_shift_that_unpaired_misses() -> None:
    # 試行ごとの入力の違いによるばらつき (50-250) が、3% の差より大きい。
    # 対応のない比較では差が埋もれる (research.md の「対応のある差の再標本化」)
    rng = random.Random(44)
    a = [rng.uniform(50.0, 250.0) for _ in range(24)]
    b = [v * 1.03 for v in a]

    paired = diff_verdict(a, b, paired=True, tolerance=0.02, seed=20260919)
    unpaired = diff_verdict(a, b, paired=False, tolerance=0.02, seed=20260919)

    assert paired.verdict == "outside"
    assert paired.ci95_low > 0.0
    assert unpaired.verdict == "within"
    assert unpaired.ci95_low <= 0.0 <= unpaired.ci95_high


def test_diff_verdict_interval_is_the_95_percent_interval() -> None:
    # 引き直した中央値の分布が厳密に出せる組で、区間の端を確かめる。
    # 差が 1..9 の 9 個なので、中央値は 5 番目の順序統計量になり、
    # 2.5% の分位は 2 番目、97.5% の分位は 8 番目 (90% の区間なら 3 と 7)
    a = [10.0] * 9
    b = [10.0 + step for step in range(1, 10)]
    assert _analytic_quantile(0.025, 9) == 2
    assert _analytic_quantile(0.975, 9) == 8
    assert _analytic_quantile(0.05, 9) == 3
    assert _analytic_quantile(0.95, 9) == 7

    got = diff_verdict(a, b, paired=True, tolerance=0.0, seed=20260919)

    assert got.ci95_low == pytest.approx(2.0)
    assert got.ci95_high == pytest.approx(8.0)
    assert got.median_diff == pytest.approx(5.0)


def test_diff_verdict_is_deterministic_for_the_same_seed() -> None:
    a = _gauss_sample(seed=51, n=16, mu=80.0, sigma=4.0)
    b = _gauss_sample(seed=52, n=16, mu=80.0, sigma=4.0)

    first = diff_verdict(a, b, paired=True, tolerance=0.02, seed=7)
    second = diff_verdict(a, b, paired=True, tolerance=0.02, seed=7)

    assert first.model_dump() == second.model_dump()


def test_diff_verdict_is_stable_across_seeds_for_a_clear_difference() -> None:
    a = _gauss_sample(seed=61, n=20, mu=100.0, sigma=3.0)
    b = [v * 1.1 for v in a]

    verdicts = {diff_verdict(a, b, paired=True, tolerance=0.02, seed=s).verdict for s in (1, 2, 3)}

    assert verdicts == {"outside"}


def test_diff_verdict_does_not_depend_on_container_type() -> None:
    a = _gauss_sample(seed=71, n=12, mu=50.0, sigma=2.0)
    b = _gauss_sample(seed=72, n=12, mu=52.0, sigma=2.0)

    from_lists = diff_verdict(a, b, paired=True, tolerance=0.02, seed=5)
    from_tuples = diff_verdict(tuple(a), tuple(b), paired=True, tolerance=0.02, seed=5)

    assert from_lists.model_dump() == from_tuples.model_dump()


def test_diff_verdict_paired_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError):
        diff_verdict([1.0, 2.0, 3.0], [1.0, 2.0], paired=True, tolerance=0.02, seed=1)


@pytest.mark.parametrize(
    ("a", "b"),
    [([1.0], [1.0, 2.0]), ([1.0, 2.0], [1.0]), ([], [1.0, 2.0])],
)
def test_diff_verdict_rejects_too_few_values(a: Sequence[float], b: Sequence[float]) -> None:
    with pytest.raises(ValueError):
        diff_verdict(a, b, paired=False, tolerance=0.02, seed=1)


def test_diff_verdict_with_zero_median_has_no_relative_diff() -> None:
    a = [-2.0, -1.0, 0.0, 1.0, 2.0]
    b = [v + 5.0 for v in a]

    got = diff_verdict(a, b, paired=True, tolerance=0.5, seed=3)

    assert got.relative_diff is None
    # 割合が出せないので、区間の決まりだけで判定する
    assert got.verdict == "outside"
    assert got.median_diff == pytest.approx(5.0)


def test_diff_verdict_unpaired_allows_different_lengths() -> None:
    a = _gauss_sample(seed=81, n=12, mu=100.0, sigma=4.0)
    b = _gauss_sample(seed=82, n=18, mu=100.0, sigma=4.0)

    got = diff_verdict(a, b, paired=False, tolerance=0.02, seed=9)

    assert got.n == 12
    assert got.paired is False
    # 偶数個の中央値 (2 つの真ん中の平均) を使っていることを、ここで固定する
    assert got.median_diff == pytest.approx(statistics.median(b) - statistics.median(a))


def test_diff_verdict_is_fast_for_small_samples() -> None:
    a = _gauss_sample(seed=91, n=20, mu=100.0, sigma=3.0)
    b = _gauss_sample(seed=92, n=20, mu=101.0, sigma=3.0)

    started = time.perf_counter()
    diff_verdict(a, b, paired=True, tolerance=0.02, seed=1)
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0


def test_diff_verdict_is_fast_for_three_hundred_samples() -> None:
    a = _gauss_sample(seed=101, n=300, mu=100.0, sigma=3.0)
    b = _gauss_sample(seed=102, n=300, mu=101.0, sigma=3.0)

    started = time.perf_counter()
    diff_verdict(a, b, paired=True, tolerance=0.02, seed=1)
    elapsed = time.perf_counter() - started

    assert elapsed < 10.0


# --- proportion_diff_verdict ----------------------------------------------


def test_proportion_diff_verdict_disjoint_intervals_are_different() -> None:
    a = binomial_interval(1, 100)
    b = binomial_interval(50, 100)

    assert proportion_diff_verdict(a, b) == "different"
    assert proportion_diff_verdict(b, a) == "different"


def test_proportion_diff_verdict_overlapping_intervals_are_not_distinguishable() -> None:
    a = binomial_interval(45, 100)
    b = binomial_interval(55, 100)

    assert proportion_diff_verdict(a, b) == "not_distinguishable"


def test_proportion_diff_verdict_identical_stats() -> None:
    stat: ProportionStat = binomial_interval(3, 30)

    assert proportion_diff_verdict(stat, stat) == "not_distinguishable"


# --- 境界の等号 (厳密な不等号であることを固定する) -----------------------------


def _stat(low: float, high: float, upper_one_sided: float) -> ProportionStat:
    return ProportionStat(
        numerator=1,
        denominator=100,
        rate=0.01,
        ci95_low=low,
        ci95_high=high,
        upper95_one_sided=upper_one_sided,
    )


def test_threshold_verdict_is_undetermined_when_a_bound_equals_the_threshold() -> None:
    """片側の上限、または両側の下限が、しきい値とちょうど等しいときは、言い切らない。"""
    upper_equals = _stat(low=0.001, high=0.03, upper_one_sided=0.02)
    lower_equals = _stat(low=0.02, high=0.09, upper_one_sided=0.08)

    assert threshold_verdict(upper_equals, 0.02) is ThresholdVerdict.UNDETERMINED
    assert threshold_verdict(lower_equals, 0.02) is ThresholdVerdict.UNDETERMINED


def test_proportion_diff_verdict_treats_touching_intervals_as_not_distinguishable() -> None:
    """2 つの区間の端がちょうど接するときは、重なっていると見なす (保守的な側)。"""
    lower = _stat(low=0.01, high=0.05, upper_one_sided=0.045)
    upper = _stat(low=0.05, high=0.12, upper_one_sided=0.11)

    assert proportion_diff_verdict(lower, upper) == "not_distinguishable"
    assert proportion_diff_verdict(upper, lower) == "not_distinguishable"
