"""内部の指標を読む部品の試験 (task 2.2)。

純粋な `derive` は、手で組んだ `MetricSnapshot` で確かめる。解析と収集は、
実際のソケット越しに偽のサーバーを相手に確かめる (`fake_server` fixture)。

カウンターの増分を確かめる試験だけ、時刻の判定が要る。偽のサーバーは、
ストリームを書き終えたあとにカウンターを進めるので (`fake_server.py` の
`_advance_counters`)、応答を読み終えた時刻とカウンターが進む時刻は厳密には
一致しない。狭い時間の一致では判定せず、増分が現れるまで緩く再試行する
(tasks.md の Implementation Notes 1.4 に沿い、正確な時間は測らない)。
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import httpx
import pytest

from bench_harness.metrics.scrape import DEFAULT_METRIC_MAP, HttpxMetricsScraper, derive
from bench_harness.types import (
    LogicalMetric,
    MetricSnapshot,
    MetricsUnavailable,
    TargetDef,
)
from fake_server import FakeServer, token_stream_response

# --- 助け -----------------------------------------------------------------


def _snapshot(
    values: dict[LogicalMetric, float], missing: list[LogicalMetric] | None = None
) -> MetricSnapshot:
    return MetricSnapshot.model_validate(
        {
            "taken_at_utc": "2026-01-01T00:00:00Z",
            "raw_text": "",
            "values": values,
            "missing": missing or [],
        }
    )


def _target(server: FakeServer, **overrides: Any) -> TargetDef:
    payload: dict[str, Any] = {"name": "fake", "base_url": server.base_url, "model": "fake-model"}
    payload.update(overrides)
    return TargetDef.model_validate(payload)


async def _wait_for_generation_increase(
    scraper: HttpxMetricsScraper,
    baseline: float,
    *,
    expected_increase: float,
    timeout_s: float = 2.0,
) -> MetricSnapshot:
    """カウンターの増分が現れるまで、緩く再試行する (上の module docstring を参照)。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        result = await scraper.snapshot()
        assert isinstance(result, MetricSnapshot)
        current = result.values.get(LogicalMetric.GENERATION_TOKENS, 0.0)
        if current - baseline >= expected_increase:
            return result
        if loop.time() >= deadline:
            raise AssertionError(f"generation_tokens が増えなかった: {baseline} -> {current}")
        await asyncio.sleep(0.01)


_LITERAL_METRICS_TEXT = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="glm-5.3-flash",engine="0"} 2.0
vllm:num_requests_running{model_name="glm-5.3-flash",engine="1"} 3.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="glm-5.3-flash",engine="0"} 0.25
vllm:kv_cache_usage_perc{model_name="glm-5.3-flash",engine="1"} 0.75
# HELP vllm:num_preemptions_total Cumulative number of preemptions from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model_name="glm-5.3-flash"} 4.0
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="glm-5.3-flash"} 1000.0
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="glm-5.3-flash"} 500.0
# HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
# TYPE vllm:iteration_tokens_total histogram
vllm:iteration_tokens_total_bucket{le="1.0",model_name="glm-5.3-flash"} 0.0
vllm:iteration_tokens_total_bucket{le="8.0",model_name="glm-5.3-flash"} 3.0
vllm:iteration_tokens_total_bucket{le="+Inf",model_name="glm-5.3-flash"} 10.0
vllm:iteration_tokens_total_sum{model_name="glm-5.3-flash"} 1500.0
vllm:iteration_tokens_total_count{model_name="glm-5.3-flash"} 10.0
# HELP vllm:prefix_cache_queries_total Prefix cache queries, in terms of number of queried tokens.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{model_name="glm-5.3-flash",engine="0"} 100.0
vllm:prefix_cache_queries_total{model_name="glm-5.3-flash",engine="1"} 150.0
# HELP vllm:prefix_cache_hits_total Prefix cache hits, in terms of number of cached tokens.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{model_name="glm-5.3-flash",engine="0"} 20.0
vllm:prefix_cache_hits_total{model_name="glm-5.3-flash",engine="1"} 30.0
"""
"""投機的デコードの一族をわざと含まない、vLLM V1 の形式の生テキスト。

HELP/TYPE の行、ラベル、バケツ付きのヒストグラム、1 つのカウンターに 2 つの
ラベルの組を含む (`engine="0"`/`"1"`)。
"""


# --- 既定の対応 -------------------------------------------------------------


def test_default_metric_map_covers_every_logical_metric() -> None:
    assert set(DEFAULT_METRIC_MAP) == set(LogicalMetric)


# --- derive (純粋な関数) -----------------------------------------------------


def test_derive_computes_every_formula_from_increasing_counters() -> None:
    before = _snapshot(
        {
            LogicalMetric.SPEC_DRAFTS: 100.0,
            LogicalMetric.SPEC_DRAFT_TOKENS: 100.0,
            LogicalMetric.SPEC_ACCEPTED_TOKENS: 60.0,
            LogicalMetric.PREFIX_QUERIES: 1000.0,
            LogicalMetric.PREFIX_HITS: 200.0,
            LogicalMetric.GENERATION_TOKENS: 4000.0,
            LogicalMetric.ITERATION_TOKENS_SUM: 5000.0,
            LogicalMetric.ITERATION_TOKENS_COUNT: 100.0,
            LogicalMetric.PREEMPTIONS: 2.0,
            LogicalMetric.KV_USAGE: 0.1,
        }
    )
    after = _snapshot(
        {
            # 下書きの回数と、下書きしたトークンの数は、増分をわざと違う値にする。
            # 同じ値だと、2 つの式の分母を取り違えても、試験が気づけない。
            LogicalMetric.SPEC_DRAFTS: 120.0,  # +20
            LogicalMetric.SPEC_DRAFT_TOKENS: 150.0,  # +50
            LogicalMetric.SPEC_ACCEPTED_TOKENS: 100.0,  # +40
            LogicalMetric.PREFIX_QUERIES: 1200.0,  # +200
            LogicalMetric.PREFIX_HITS: 240.0,  # +40
            LogicalMetric.GENERATION_TOKENS: 4090.0,  # +90
            LogicalMetric.ITERATION_TOKENS_SUM: 5800.0,  # +800
            LogicalMetric.ITERATION_TOKENS_COUNT: 150.0,  # +50
            LogicalMetric.PREEMPTIONS: 3.0,  # +1
            LogicalMetric.KV_USAGE: 0.4,
        }
    )

    result = derive(before, after, kv_peak=None)

    assert result.spec_acceptance_rate == pytest.approx(40 / 50)  # 当たり ÷ 下書きしたトークン
    assert result.mean_acceptance_length == pytest.approx(1 + 40 / 20)  # 1 + 当たり ÷ 下書きの回数
    assert result.decode_steps == pytest.approx(50)
    assert result.tokens_per_step == pytest.approx(90 / 50)  # 生成トークン ÷ ステップ数
    assert result.prefix_cache_hit_rate == pytest.approx(40 / 200)
    assert result.preemptions == pytest.approx(1)
    assert result.kv_usage_peak == pytest.approx(0.4)  # kv_peak なし -> 2 つの gauge の最大
    assert result.missing == []


def test_derive_zero_denominator_yields_none_without_raising() -> None:
    before = _snapshot(
        {
            LogicalMetric.SPEC_DRAFTS: 10.0,
            LogicalMetric.SPEC_DRAFT_TOKENS: 10.0,
            LogicalMetric.SPEC_ACCEPTED_TOKENS: 5.0,
            LogicalMetric.PREFIX_QUERIES: 10.0,
            LogicalMetric.PREFIX_HITS: 5.0,
            LogicalMetric.GENERATION_TOKENS: 400.0,
            LogicalMetric.ITERATION_TOKENS_SUM: 10.0,
            LogicalMetric.ITERATION_TOKENS_COUNT: 10.0,
        }
    )
    after = _snapshot(
        {
            LogicalMetric.SPEC_DRAFTS: 10.0,  # +0
            LogicalMetric.SPEC_DRAFT_TOKENS: 10.0,  # +0
            LogicalMetric.SPEC_ACCEPTED_TOKENS: 12.0,  # +2 (分子だけ増える)
            LogicalMetric.PREFIX_QUERIES: 10.0,  # +0
            LogicalMetric.PREFIX_HITS: 8.0,  # +3
            LogicalMetric.GENERATION_TOKENS: 410.0,  # +10 (分子はある)
            LogicalMetric.ITERATION_TOKENS_SUM: 20.0,  # +10
            LogicalMetric.ITERATION_TOKENS_COUNT: 10.0,  # +0
        }
    )

    result = derive(before, after, kv_peak=None)

    assert result.spec_acceptance_rate is None
    assert result.mean_acceptance_length is None
    assert result.tokens_per_step is None  # 分母 (ステップ数) が 0 だから
    assert LogicalMetric.GENERATION_TOKENS not in result.missing  # 分子の欠落ではない
    assert result.prefix_cache_hit_rate is None
    assert result.decode_steps == pytest.approx(0)  # 生の増分自体は 0 であって None ではない


def test_derive_negative_delta_from_restart_is_none_and_marks_missing() -> None:
    before = _snapshot(
        {
            LogicalMetric.SPEC_ACCEPTED_TOKENS: 100.0,
            LogicalMetric.SPEC_DRAFT_TOKENS: 100.0,
            LogicalMetric.PREEMPTIONS: 5.0,
        }
    )
    after = _snapshot(
        {
            LogicalMetric.SPEC_ACCEPTED_TOKENS: 10.0,  # 負の増分 (再起動)
            LogicalMetric.SPEC_DRAFT_TOKENS: 120.0,
            LogicalMetric.PREEMPTIONS: 1.0,  # 負の増分
        }
    )

    result = derive(before, after, kv_peak=None)

    assert result.spec_acceptance_rate is None
    assert result.mean_acceptance_length is None
    assert result.preemptions is None
    assert LogicalMetric.SPEC_ACCEPTED_TOKENS in result.missing
    assert LogicalMetric.PREEMPTIONS in result.missing


def test_derive_metric_missing_in_either_snapshot_is_none_and_listed() -> None:
    absent = [
        LogicalMetric.SPEC_DRAFTS,
        LogicalMetric.SPEC_DRAFT_TOKENS,
        LogicalMetric.SPEC_ACCEPTED_TOKENS,
    ]
    before = _snapshot({}, missing=absent)
    after = _snapshot({}, missing=absent)

    result = derive(before, after, kv_peak=None)

    assert result.spec_acceptance_rate is None
    assert result.mean_acceptance_length is None
    for metric in absent:
        assert metric in result.missing


def test_derive_kv_peak_uses_the_given_value_over_the_snapshots() -> None:
    before = _snapshot({LogicalMetric.KV_USAGE: 0.3})
    after = _snapshot({LogicalMetric.KV_USAGE: 0.2})

    result = derive(before, after, kv_peak=0.95)

    assert result.kv_usage_peak == pytest.approx(0.95)


def test_derive_kv_peak_falls_back_to_the_max_of_both_snapshots() -> None:
    before = _snapshot({LogicalMetric.KV_USAGE: 0.3})
    after = _snapshot({LogicalMetric.KV_USAGE: 0.2})

    result = derive(before, after, kv_peak=None)

    assert result.kv_usage_peak == pytest.approx(0.3)


def test_derive_kv_peak_is_none_when_the_gauge_is_missing_everywhere() -> None:
    before = _snapshot({})
    after = _snapshot({})

    result = derive(before, after, kv_peak=None)

    assert result.kv_usage_peak is None


# --- 1 ステップあたりの生成トークン (7.3) -----------------------------------


def test_tokens_per_step_is_one_without_speculation_despite_prefill_input_tokens() -> None:
    before = _snapshot(
        {
            LogicalMetric.GENERATION_TOKENS: 500.0,
            LogicalMetric.ITERATION_TOKENS_SUM: 1500.0,
            LogicalMetric.ITERATION_TOKENS_COUNT: 10.0,
        }
    )
    after = _snapshot(
        {
            # prefill の回があるので、sum の増分には入力トークン 1000 も入る。
            # 生成トークン ÷ ステップ数だけを見れば、投機なしでは 1.0 になる。
            LogicalMetric.GENERATION_TOKENS: 520.0,  # +20
            LogicalMetric.ITERATION_TOKENS_SUM: 2520.0,  # +1020 (入力 1000 + 生成 20)
            LogicalMetric.ITERATION_TOKENS_COUNT: 30.0,  # +20
        }
    )

    result = derive(before, after, kv_peak=None)

    assert result.tokens_per_step == pytest.approx(1.0)


def test_tokens_per_step_matches_mean_acceptance_length_with_speculation() -> None:
    before = _snapshot(
        {
            LogicalMetric.SPEC_DRAFTS: 100.0,
            LogicalMetric.SPEC_DRAFT_TOKENS: 100.0,
            LogicalMetric.SPEC_ACCEPTED_TOKENS: 60.0,
            LogicalMetric.GENERATION_TOKENS: 500.0,
            LogicalMetric.ITERATION_TOKENS_SUM: 1500.0,
            LogicalMetric.ITERATION_TOKENS_COUNT: 10.0,
        }
    )
    after = _snapshot(
        {
            LogicalMetric.SPEC_DRAFTS: 150.0,  # +50
            LogicalMetric.SPEC_DRAFT_TOKENS: 150.0,  # +50
            LogicalMetric.SPEC_ACCEPTED_TOKENS: 100.0,  # +40
            LogicalMetric.GENERATION_TOKENS: 590.0,  # +90
            LogicalMetric.ITERATION_TOKENS_SUM: 2590.0,  # +1090 (入力 1000 + 生成 90)
            LogicalMetric.ITERATION_TOKENS_COUNT: 60.0,  # +50
        }
    )

    result = derive(before, after, kv_peak=None)

    assert result.mean_acceptance_length == pytest.approx(1.8)  # 1 + 40 / 50
    assert result.tokens_per_step == pytest.approx(1.8)  # 90 / 50
    assert result.tokens_per_step == pytest.approx(result.mean_acceptance_length)


def test_mean_acceptance_length_keeps_its_own_formula_when_tokens_per_step_differs() -> None:
    before = _snapshot(
        {
            LogicalMetric.SPEC_DRAFTS: 100.0,
            LogicalMetric.SPEC_DRAFT_TOKENS: 100.0,
            LogicalMetric.SPEC_ACCEPTED_TOKENS: 60.0,
            LogicalMetric.GENERATION_TOKENS: 500.0,
            LogicalMetric.ITERATION_TOKENS_COUNT: 10.0,
        }
    )
    after = _snapshot(
        {
            LogicalMetric.SPEC_DRAFTS: 150.0,  # +50
            LogicalMetric.SPEC_DRAFT_TOKENS: 150.0,  # +50
            LogicalMetric.SPEC_ACCEPTED_TOKENS: 100.0,  # +40
            LogicalMetric.GENERATION_TOKENS: 600.0,  # +100
            LogicalMetric.ITERATION_TOKENS_COUNT: 60.0,  # +50
        }
    )

    result = derive(before, after, kv_peak=None)

    # 受理長は 1 + 当たり ÷ 下書きの回数、tokens_per_step は生成 ÷ ステップ数。
    # 片方から他方を作ると、この食い違いが消える。
    assert result.mean_acceptance_length == pytest.approx(1 + 40 / 50)
    assert result.tokens_per_step == pytest.approx(100 / 50)
    assert result.mean_acceptance_length != pytest.approx(result.tokens_per_step)


@pytest.mark.parametrize(
    "metric",
    [LogicalMetric.GENERATION_TOKENS, LogicalMetric.ITERATION_TOKENS_COUNT],
)
def test_tokens_per_step_is_none_and_lists_the_counter_marked_missing(
    metric: LogicalMetric,
) -> None:
    before = _snapshot({}, missing=[metric])
    after = _snapshot({}, missing=[metric])

    result = derive(before, after, kv_peak=None)

    assert result.tokens_per_step is None
    assert metric in result.missing


@pytest.mark.parametrize(
    "metric",
    [LogicalMetric.GENERATION_TOKENS, LogicalMetric.ITERATION_TOKENS_COUNT],
)
def test_tokens_per_step_is_not_defaulted_when_the_counter_is_absent(
    metric: LogicalMetric,
) -> None:
    values = {
        LogicalMetric.GENERATION_TOKENS: 500.0,
        LogicalMetric.ITERATION_TOKENS_COUNT: 10.0,
    }
    del values[metric]
    before = _snapshot(values)
    after = _snapshot(values)

    result = derive(before, after, kv_peak=None)

    # values にも missing にも無い値を 0.0 で埋めると、None にならず値が出てしまう
    assert result.tokens_per_step is None
    assert metric in result.missing


# --- 解析 (Prometheus のテキスト) --------------------------------------------


async def test_snapshot_parses_literal_vllm_text_and_aggregates_label_sets(
    fake_server: FakeServer,
) -> None:
    fake_server.set_metrics_text(_LITERAL_METRICS_TEXT)
    target = _target(fake_server)

    async with HttpxMetricsScraper(target) as scraper:
        result = await scraper.snapshot()

    assert isinstance(result, MetricSnapshot)
    assert result.raw_text == _LITERAL_METRICS_TEXT
    assert result.values[LogicalMetric.RUNNING_REQUESTS] == pytest.approx(5.0)  # 合計 (2.0 + 3.0)
    assert result.values[LogicalMetric.KV_USAGE] == pytest.approx(0.75)  # 最大 (0.25, 0.75)
    assert result.values[LogicalMetric.PREEMPTIONS] == pytest.approx(4.0)
    assert result.values[LogicalMetric.PROMPT_TOKENS] == pytest.approx(1000.0)
    assert result.values[LogicalMetric.GENERATION_TOKENS] == pytest.approx(500.0)
    assert result.values[LogicalMetric.ITERATION_TOKENS_SUM] == pytest.approx(1500.0)
    assert result.values[LogicalMetric.ITERATION_TOKENS_COUNT] == pytest.approx(10.0)
    assert result.values[LogicalMetric.PREFIX_QUERIES] == pytest.approx(250.0)  # 合計 (100 + 150)
    assert result.values[LogicalMetric.PREFIX_HITS] == pytest.approx(50.0)  # 合計 (20 + 30)
    assert LogicalMetric.SPEC_DRAFTS in result.missing
    assert LogicalMetric.SPEC_DRAFT_TOKENS in result.missing
    assert LogicalMetric.SPEC_ACCEPTED_TOKENS in result.missing


async def test_metric_map_override_finds_a_renamed_metric(fake_server: FakeServer) -> None:
    fake_server.set_metrics_options(rename={"vllm:prefix_cache_hits_total": "custom:hits"})
    # prometheus_client の parser は、counter の sample 名に `_total` を必ず補う
    target = _target(fake_server, metric_map={"prefix_hits": "custom:hits_total"})

    async with HttpxMetricsScraper(target) as scraper:
        result = await scraper.snapshot()

    assert isinstance(result, MetricSnapshot)
    assert LogicalMetric.PREFIX_HITS not in result.missing
    assert result.values[LogicalMetric.PREFIX_HITS] == pytest.approx(0.0)
    assert LogicalMetric.PREFIX_QUERIES not in result.missing  # 上書きしていない名前は既定のまま


# --- 偽のサーバーを相手にした試験 --------------------------------------------


async def test_snapshot_and_derive_reflect_a_streamed_request(fake_server: FakeServer) -> None:
    output_tokens = 20
    fake_server.set_response(token_stream_response(output_tokens=output_tokens, gap_s=0.0))
    target = _target(fake_server)

    async with HttpxMetricsScraper(target) as scraper:
        before = await scraper.snapshot()
        assert isinstance(before, MetricSnapshot)
        baseline = before.values.get(LogicalMetric.GENERATION_TOKENS, 0.0)

        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                f"{fake_server.base_url}/v1/messages",
                json={
                    "model": "fake-model",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert response.status_code == 200

        after = await _wait_for_generation_increase(
            scraper, baseline, expected_increase=float(output_tokens)
        )

    result = derive(before, after, kv_peak=None)

    assert result.spec_acceptance_rate == pytest.approx(0.8, abs=0.05)
    assert result.mean_acceptance_length == pytest.approx(1.8, abs=0.05)
    assert result.decode_steps == pytest.approx(float(output_tokens))
    assert result.tokens_per_step is not None
    assert result.tokens_per_step > 0
    assert result.missing == []


async def test_snapshot_without_spec_decode_family_has_no_exception_and_lists_missing(
    fake_server: FakeServer,
) -> None:
    fake_server.set_metrics_options(include_spec_decode=False)
    target = _target(fake_server)

    async with HttpxMetricsScraper(target) as scraper:
        before = await scraper.snapshot()
        after = await scraper.snapshot()

    assert isinstance(before, MetricSnapshot)
    assert isinstance(after, MetricSnapshot)
    for metric in (
        LogicalMetric.SPEC_DRAFTS,
        LogicalMetric.SPEC_DRAFT_TOKENS,
        LogicalMetric.SPEC_ACCEPTED_TOKENS,
    ):
        assert metric in before.missing
        assert metric in after.missing

    result = derive(before, after, kv_peak=None)

    assert result.spec_acceptance_rate is None
    assert result.mean_acceptance_length is None
    assert LogicalMetric.SPEC_DRAFTS in result.missing


async def test_snapshot_returns_unavailable_when_metrics_endpoint_is_404(
    fake_server: FakeServer,
) -> None:
    fake_server.set_metrics_enabled(False)
    target = _target(fake_server)

    async with HttpxMetricsScraper(target) as scraper:
        result = await scraper.snapshot()

    assert isinstance(result, MetricsUnavailable)


async def test_snapshot_returns_unavailable_when_connection_is_refused() -> None:
    # 空いている port を 1 つだけ確保して、すぐに閉じる (誰も待ち受けていない port になる)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    target = TargetDef.model_validate(
        {"name": "unreachable", "base_url": f"http://127.0.0.1:{port}", "model": "m"}
    )

    async with HttpxMetricsScraper(target, timeout_s=2.0) as scraper:
        result = await scraper.snapshot()

    assert isinstance(result, MetricsUnavailable)


async def test_snapshot_returns_unavailable_for_unparseable_text(fake_server: FakeServer) -> None:
    fake_server.set_metrics_text("this is not { valid Prometheus text @@@ \n\n garbage")
    target = _target(fake_server)

    async with HttpxMetricsScraper(target) as scraper:
        result = await scraper.snapshot()

    assert isinstance(result, MetricsUnavailable)


# --- サンプラー --------------------------------------------------------------


async def test_sampler_observes_a_kv_peak_set_mid_way(fake_server: FakeServer) -> None:
    target = _target(fake_server)
    async with HttpxMetricsScraper(target) as scraper:
        fake_server.set_kv_usage(0.1)
        scraper.start_sampler(0.01)
        await asyncio.sleep(0.15)
        fake_server.set_kv_usage(0.9)
        await asyncio.sleep(0.15)
        fake_server.set_kv_usage(0.2)
        await asyncio.sleep(0.15)

        peak = await scraper.stop_sampler()

    assert peak is not None
    assert peak == pytest.approx(0.9, abs=1e-6)


async def test_sampler_tolerates_a_server_that_starts_returning_404(
    fake_server: FakeServer,
) -> None:
    target = _target(fake_server)
    async with HttpxMetricsScraper(target) as scraper:
        fake_server.set_kv_usage(0.5)
        scraper.start_sampler(0.01)
        await asyncio.sleep(0.1)
        fake_server.set_metrics_enabled(False)
        await asyncio.sleep(0.1)  # 404 を挟んでも、例外にならず止まらない

        peak = await scraper.stop_sampler()

    assert peak == pytest.approx(0.5, abs=1e-6)


async def test_stop_sampler_without_start_is_safe(fake_server: FakeServer) -> None:
    target = _target(fake_server)
    async with HttpxMetricsScraper(target) as scraper:
        peak = await scraper.stop_sampler()

    assert peak is None


async def test_start_sampler_twice_does_not_create_a_second_task(fake_server: FakeServer) -> None:
    target = _target(fake_server)
    async with HttpxMetricsScraper(target) as scraper:
        before = asyncio.all_tasks()
        scraper.start_sampler(0.05)
        scraper.start_sampler(0.05)  # 二重に始めても安全
        during = asyncio.all_tasks()
        await scraper.stop_sampler()
        after = asyncio.all_tasks()

    assert len(during - before) == 1
    assert not (after - before)


async def test_no_task_leaks_after_close_without_explicit_stop(fake_server: FakeServer) -> None:
    target = _target(fake_server)
    before = asyncio.all_tasks()

    scraper = HttpxMetricsScraper(target)
    scraper.start_sampler(0.02)
    await asyncio.sleep(0.05)
    await scraper.aclose()  # stop_sampler を呼ばずに閉じても、task を残さない

    after = asyncio.all_tasks()
    assert not (after - before)


async def test_stop_sampler_cancels_promptly_even_with_a_long_interval(
    fake_server: FakeServer,
) -> None:
    target = _target(fake_server)
    async with HttpxMetricsScraper(target) as scraper:
        scraper.start_sampler(10.0)  # 長い間隔でも、stop が待たされず即座に戻ることを確かめる
        await asyncio.sleep(0.05)
        # cancel が伝わらず握りつぶされていたら、この wait_for がここで時間切れになる
        peak = await asyncio.wait_for(scraper.stop_sampler(), timeout=1.0)

    assert peak == pytest.approx(0.0)  # 開始直後の 1 回目の読み取り (KV の既定は 0.0) だけが効く
