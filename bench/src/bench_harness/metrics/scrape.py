"""対象サーバーの内部の指標 (`/metrics`) を読み、増分と導出値を出す (task 2.2)。

`/metrics` の Prometheus のテキストを取得して解析し、論理名 (`LogicalMetric`)
ごとの値に直す。得られなくても計測は止めない (7.4): 接続できない、HTTP が
200 でない、テキストが解析できない、いずれも `MetricsUnavailable` という値で
返し、例外を投げない。`asyncio.CancelledError` だけは、そのまま外へ伝える。

名前の対応は vLLM V1 の既定値 (`research.md` の表) を使い、
`TargetDef.metric_map` で上書きできる (上書きが勝つ)。同じ名前が複数のラベル
(`engine` など) で出たときは、KV の使用率は最大、それ以外 (カウンターと、
ヒストグラムの `_sum` / `_count`、実行中の要求数) は合計する。

依存するのは標準ライブラリと httpx、prometheus_client、pydantic、
`bench_harness.types` だけ。ほかの `bench_harness` の module を読み込まない。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from datetime import UTC, datetime
from types import TracebackType
from typing import Final, Protocol, Self, runtime_checkable

import httpx
from prometheus_client.parser import text_string_to_metric_families

from bench_harness.types import (
    DerivedMetrics,
    LogicalMetric,
    MetricSnapshot,
    MetricsUnavailable,
    TargetDef,
)

__all__ = [
    "DEFAULT_METRIC_MAP",
    "HttpxMetricsScraper",
    "MetricsScraper",
    "derive",
]

METRICS_PATH: Final[str] = "/metrics"

_REQUEST_TIMEOUT_S: Final[float] = 5.0
"""ハングした `/metrics` で計測ランが止まらないよう、短く切る (design.md metrics/scrape)。"""

DEFAULT_METRIC_MAP: Final[dict[LogicalMetric, str]] = {
    LogicalMetric.SPEC_DRAFTS: "vllm:spec_decode_num_drafts_total",
    LogicalMetric.SPEC_DRAFT_TOKENS: "vllm:spec_decode_num_draft_tokens_total",
    LogicalMetric.SPEC_ACCEPTED_TOKENS: "vllm:spec_decode_num_accepted_tokens_total",
    LogicalMetric.PREFIX_QUERIES: "vllm:prefix_cache_queries_total",
    LogicalMetric.PREFIX_HITS: "vllm:prefix_cache_hits_total",
    LogicalMetric.KV_USAGE: "vllm:kv_cache_usage_perc",
    LogicalMetric.RUNNING_REQUESTS: "vllm:num_requests_running",
    LogicalMetric.PROMPT_TOKENS: "vllm:prompt_tokens_total",
    LogicalMetric.GENERATION_TOKENS: "vllm:generation_tokens_total",
    # ヒストグラムは `_sum` / `_count` の 2 つの sample として出てくる
    # (prometheus_client の parser が、TYPE の base 名にこの 2 つの接尾辞を足す)
    LogicalMetric.ITERATION_TOKENS_SUM: "vllm:iteration_tokens_total_sum",
    LogicalMetric.ITERATION_TOKENS_COUNT: "vllm:iteration_tokens_total_count",
    LogicalMetric.PREEMPTIONS: "vllm:num_preemptions_total",
}
"""vLLM V1 の既定の名前の対応 (`research.md` の「`/metrics` の名前」)。

値は、解析したあとの sample の名前そのもの。カウンターは `_total` で終わる
(TYPE が counter の sample は、parser が末尾に `_total` を補うため、対象サーバー
の定義で名前を上書きするときも、この接尾辞込みの名前を書くこと)。
"""

_MAX_AGGREGATED: Final[frozenset[LogicalMetric]] = frozenset({LogicalMetric.KV_USAGE})
"""複数のラベルで同じ名前が出たとき、最大を取る論理名。それ以外は合計する。"""


def _resolve_metric_map(target: TargetDef) -> dict[LogicalMetric, str]:
    """既定の対応に、対象サーバーの定義の上書きを重ねる (上書きが勝つ)。"""
    return {**DEFAULT_METRIC_MAP, **target.metric_map}


def _metrics_url(target: TargetDef) -> str:
    if target.metrics_url is not None:
        return str(target.metrics_url)
    return str(target.base_url).rstrip("/") + METRICS_PATH


def _parse_snapshot(raw_text: str, metric_map: Mapping[LogicalMetric, str]) -> MetricSnapshot:
    """Prometheus のテキストを解析し、論理名ごとの値と、見つからなかった名前に直す。"""
    samples: dict[str, list[float]] = {}
    for family in text_string_to_metric_families(raw_text):
        for sample in family.samples:
            samples.setdefault(sample.name, []).append(sample.value)

    values: dict[LogicalMetric, float] = {}
    missing: list[LogicalMetric] = []
    for metric, name in metric_map.items():
        found = samples.get(name)
        if not found:
            missing.append(metric)
            continue
        values[metric] = max(found) if metric in _MAX_AGGREGATED else sum(found)

    return MetricSnapshot(
        taken_at_utc=datetime.now(UTC), raw_text=raw_text, values=values, missing=missing
    )


# --- 導出値 (純粋な関数) --------------------------------------------------


def _delta(
    before: MetricSnapshot,
    after: MetricSnapshot,
    metric: LogicalMetric,
    missing: set[LogicalMetric],
) -> float | None:
    """2 つの時点の増分。得られない、または負の増分 (再起動) は `None` にする。

    負の増分は、対象サーバーが計測の途中で再起動して、増えていく値が巻き戻った
    ことを表す。例外にはせず、値なしとして扱う。
    """
    if metric in before.missing or metric in after.missing:
        missing.add(metric)
        return None
    before_value = before.values.get(metric)
    after_value = after.values.get(metric)
    if before_value is None or after_value is None:
        missing.add(metric)
        return None
    diff = after_value - before_value
    if diff < 0:
        missing.add(metric)
        return None
    return diff


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """分母が 0 以下、またはどちらかが得られないときは `None` (0 で割らない)。"""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def derive(before: MetricSnapshot, after: MetricSnapshot, kv_peak: float | None) -> DerivedMetrics:
    """2 つの時点の内部の指標から、測る項目のまとまりの導出値を出す (7.3)。

    純粋な関数。副作用も通信も持たない。`kv_peak` は、その間にサンプラーで
    読んだ KV の使用率の最大 (`start_sampler` / `stop_sampler`)。`None` なら、
    `before` と `after` の gauge の値の最大で代える。
    """
    missing: set[LogicalMetric] = set()

    def delta(metric: LogicalMetric) -> float | None:
        return _delta(before, after, metric, missing)

    accepted = delta(LogicalMetric.SPEC_ACCEPTED_TOKENS)
    draft_tokens = delta(LogicalMetric.SPEC_DRAFT_TOKENS)
    drafts = delta(LogicalMetric.SPEC_DRAFTS)
    # `iteration_tokens_total_sum` には prefill の回の入力トークンも入るので、
    # 1 ステップあたりの生成トークンは生成のカウンターから取る
    generation = delta(LogicalMetric.GENERATION_TOKENS)
    iteration_count = delta(LogicalMetric.ITERATION_TOKENS_COUNT)
    prefix_hits = delta(LogicalMetric.PREFIX_HITS)
    prefix_queries = delta(LogicalMetric.PREFIX_QUERIES)
    preemptions = delta(LogicalMetric.PREEMPTIONS)

    mean_acceptance_length = None
    if accepted is not None and drafts is not None and drafts > 0:
        mean_acceptance_length = 1 + accepted / drafts

    resolved_kv_peak = kv_peak
    if resolved_kv_peak is None:
        gauge_values = [
            snapshot.values[LogicalMetric.KV_USAGE]
            for snapshot in (before, after)
            if LogicalMetric.KV_USAGE in snapshot.values
        ]
        resolved_kv_peak = max(gauge_values) if gauge_values else None

    return DerivedMetrics(
        spec_acceptance_rate=_ratio(accepted, draft_tokens),
        mean_acceptance_length=mean_acceptance_length,
        decode_steps=iteration_count,
        tokens_per_step=_ratio(generation, iteration_count),
        prefix_cache_hit_rate=_ratio(prefix_hits, prefix_queries),
        kv_usage_peak=resolved_kv_peak,
        preemptions=preemptions,
        # 呼び出しの順序に依存させない (元の enum の並びで見せる)
        missing=[metric for metric in LogicalMetric if metric in missing],
    )


# --- scraper --------------------------------------------------------------


@runtime_checkable
class MetricsScraper(Protocol):
    """内部の指標を読み、増分と導出値を出す (design.md `metrics/scrape`)。"""

    async def snapshot(self) -> MetricSnapshot | MetricsUnavailable: ...

    def start_sampler(self, interval_s: float) -> None: ...

    async def stop_sampler(self) -> float | None: ...


class HttpxMetricsScraper:
    """httpx と `prometheus_client` で `/metrics` を読む `MetricsScraper`。

    1 つの `httpx.AsyncClient` を持つ。`async with` で使うか、使い終わったら
    `aclose()` を呼ぶ。`snapshot()` は、接続できない、200 以外の状態、解析
    できないテキストのいずれでも例外を投げず `MetricsUnavailable` を返す
    (7.4)。`asyncio.CancelledError` は、握りつぶさずそのまま伝える
    (`httpx.HTTPError` にも `Exception` にも含まれないので、何もしなくても
    素通りする)。
    """

    def __init__(self, target: TargetDef, *, timeout_s: float = _REQUEST_TIMEOUT_S) -> None:
        self._url = _metrics_url(target)
        self._metric_map = _resolve_metric_map(target)
        self._client = httpx.AsyncClient(timeout=timeout_s)
        self._sampler_task: asyncio.Task[None] | None = None
        self._kv_peak: float | None = None

    @classmethod
    def from_target(cls, target: TargetDef, *, timeout_s: float = _REQUEST_TIMEOUT_S) -> Self:
        return cls(target, timeout_s=timeout_s)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(url={self._url!r})"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """サンプラーを確実に止めてから、接続を閉じる。取り残す task は残さない。"""
        await self.stop_sampler()
        await self._client.aclose()

    async def snapshot(self) -> MetricSnapshot | MetricsUnavailable:
        try:
            response = await self._client.get(self._url)
        except httpx.HTTPError as exc:
            return MetricsUnavailable(reason=f"{type(exc).__name__}: {exc}")
        if response.status_code != 200:
            return MetricsUnavailable(reason=f"HTTP {response.status_code}")
        try:
            return _parse_snapshot(response.text, self._metric_map)
        except Exception as exc:  # noqa: BLE001 -- 解析の失敗も指標の不足として扱う (7.4)
            return MetricsUnavailable(reason=f"解析できない: {type(exc).__name__}: {exc}")

    def start_sampler(self, interval_s: float) -> None:
        """定期的に読み、KV の使用率の最大を追う task を始める。二重に始めても安全。"""
        if self._sampler_task is not None and not self._sampler_task.done():
            return
        self._kv_peak = None
        self._sampler_task = asyncio.create_task(
            self._sample_loop(interval_s), name="metrics-sampler"
        )

    async def stop_sampler(self) -> float | None:
        """task を止めて (未着手なら何もせず)、追ってきた最大値を返す。"""
        task, self._sampler_task = self._sampler_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return self._kv_peak

    async def _sample_loop(self, interval_s: float) -> None:
        """`snapshot()` は例外を投げないので、握りつぶす失敗の処理はここには要らない。"""
        while True:
            result = await self.snapshot()
            if isinstance(result, MetricSnapshot):
                value = result.values.get(LogicalMetric.KV_USAGE)
                if value is not None:
                    self._kv_peak = value if self._kv_peak is None else max(self._kv_peak, value)
            await asyncio.sleep(interval_s)
