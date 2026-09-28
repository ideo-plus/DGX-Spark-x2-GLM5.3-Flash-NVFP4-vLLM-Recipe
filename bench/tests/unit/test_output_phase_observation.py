"""SSE の実データと段階別時刻の契約。"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from httpx_sse import ServerSentEvent

from bench_harness.analysis.output_phases import phase_metrics
from bench_harness.client.messages import HttpxMessagesClient, _Collector
from bench_harness.types import (
    InputMessage,
    MessagesRequest,
    StreamResult,
    TextBlockParam,
    TimeoutPolicy,
)
from fake_server import (
    FakeServer,
    Script,
    SseEvent,
    UsageSpec,
    text_events,
    thinking_events,
    tool_use_events,
)

POLICY = TimeoutPolicy(connect_s=5, first_event_s=5, idle_s=5, total_s=20)
PHASE_FIELD = "output_phases"


async def _stream(server: FakeServer, script: Script) -> StreamResult:
    server.set_response(script)
    request = MessagesRequest(
        model="fake-model",
        max_tokens=256,
        messages=[InputMessage(role="user", content=[TextBlockParam(text="質問")])],
    )
    async with HttpxMessagesClient(server.base_url, None) as client:
        return await client.stream(request, POLICY)


async def test_block_start_and_multiple_text_blocks_keep_separate_phase_observations(
    fake_server: FakeServer,
) -> None:
    result = await _stream(
        fake_server,
        Script(
            events=(
                *thinking_events("思考", index=0),
                SseEvent(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {"type": "text", "text": "先頭"},
                    },
                ),
                SseEvent(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "text_delta", "text": "本文"},
                    },
                ),
                SseEvent("content_block_stop", {"type": "content_block_stop", "index": 1}),
                SseEvent(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 2,
                        "delta": {"type": "text_delta", "text": "末尾"},
                    },
                ),
            ),
            gap_s=0.001,
        ),
    )
    assert result.error is None
    phases = getattr(result, PHASE_FIELD)
    assert phases is not None
    assert phases.text.has_block is True
    assert phases.text.char_count == 6
    assert phases.thinking.char_count == 2
    assert phases.text.first_ns is not None and phases.text.last_ns is not None
    assert phases.thinking.first_ns is not None and phases.thinking.last_ns is not None
    assert phases.thinking.first_ns < phases.text.first_ns <= phases.text.last_ns


async def test_non_text_deltas_do_not_create_text_arrival_on_a_broken_stream(
    fake_server: FakeServer,
) -> None:
    result = await _stream(
        fake_server,
        Script(
            events=(
                *thinking_events("本文", signature="本文"),
                *tool_use_events("tool", fragments=('lambda("本文")',), index=1),
                SseEvent("future", {"type": "future", "text": "本文"}),
                SseEvent(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 2,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                SseEvent(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 2,
                        "delta": {"type": "text_delta", "text": ""},
                    },
                ),
            ),
            include_message_stop=False,
        ),
    )
    assert result.error is not None
    phases = getattr(result, PHASE_FIELD)
    assert phases is not None
    assert phases.text.has_block is True
    assert phases.text.char_count == 0
    assert phases.text.first_ns is None and phases.text.last_ns is None
    assert phases.thinking.char_count == 2
    assert result.timing.first_token_ns is not None


async def test_old_all_output_timestamps_are_not_redefined_as_text_timestamps(
    fake_server: FakeServer,
) -> None:
    result = await _stream(fake_server, Script(events=thinking_events("思考", chunks=("思", "考"))))
    assert result.error is None
    assert result.timing.first_token_ns is not None
    assert result.timing.last_token_ns is not None
    assert result.timing.first_token_ns < result.timing.last_token_ns
    assert result.timing.first_text_ns is None
    phases = getattr(result, PHASE_FIELD)
    assert phases is not None
    assert phases.text.first_ns is None


async def test_one_text_delta_cannot_be_treated_as_one_generated_token(
    fake_server: FakeServer,
) -> None:
    result = await _stream(
        fake_server,
        Script(events=text_events("複数トークン分の本文"), usage=UsageSpec(output_tokens=8)),
    )
    phases = getattr(result, PHASE_FIELD)
    assert result.error is None and phases is not None
    assert phases.text.char_count == len("複数トークン分の本文")
    assert phases.text.first_ns == phases.text.last_ns
    assert phases.thinking.char_count == 0
    assert result.usage is not None
    assert result.usage.output_tokens == 8


@pytest.mark.parametrize("middle_end", [10, 100])
@pytest.mark.parametrize("outer_kind,inner_kind", [("text", "thinking"), ("thinking", "text")])
def test_interleaved_sse_does_not_mix_phase_intervals(
    middle_end: int, outer_kind: str, inner_kind: str
) -> None:
    collector = _Collector(datetime(2026, 9, 29, tzinfo=UTC), 0)
    for index, kind, first, last in (
        (0, outer_kind, 1, 2),
        (1, inner_kind, 3, middle_end),
        (2, outer_kind, middle_end + 1, middle_end + 2),
    ):
        for event in (
            thinking_events("AB", index=index, chunks=("A", "B"))
            if kind == "thinking"
            else text_events("AB", index=index, chunks=("A", "B"))
        ):
            # 開始・署名・終了イベントに速度の観測時刻を足さない。
            stamp = first if event.type == "content_block_start" else last
            if event.type == "content_block_delta":
                assert event.data is not None
                delta = event.data["delta"]
                if isinstance(delta, dict) and delta.get(kind) == "A":
                    stamp = first
            collector.handle(
                ServerSentEvent(event=event.type, data=json.dumps(event.data)),
                stamp * 1_000_000_000,
            )
    result = collector.result((middle_end + 3) * 1_000_000_000, None)
    metrics = phase_metrics(result)
    assert metrics.text_duration_s is None
    assert metrics.thinking_duration_s is None
    assert metrics.text_chars_per_s is None
    assert metrics.thinking_chars_per_s is None
    assert metrics.duration_reason is not None
    assert metrics.status == "text"
    assert metrics.observed is True
    assert metrics.text_chars == (4 if outer_kind == "text" else 2)
    assert metrics.thinking_chars == (2 if outer_kind == "text" else 4)
    assert metrics.text_wait_s == (1 if outer_kind == "text" else 3)


async def test_equal_timestamp_deltas_keep_zero_span_without_infinite_speed(
    fake_server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("bench_harness.client.messages.time.perf_counter_ns", lambda: 100)
    result = await _stream(fake_server, Script(events=text_events("二文字", chunks=("二", "文字"))))
    phases = getattr(result, PHASE_FIELD)
    assert result.error is None and phases is not None
    assert phases.text.char_count == 3
    assert phases.text.first_ns == phases.text.last_ns
