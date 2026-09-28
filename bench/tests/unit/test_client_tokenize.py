"""同じ対象サーバーによる出力文字列の再計数を検証する。"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import HttpUrl

from bench_harness.types import ContentBlock, StreamResult, StreamTiming, TargetDef, Usage
from fake_server import FakeServer


def _result() -> StreamResult:
    return StreamResult(
        timing=StreamTiming(sent_at_utc=datetime.now(UTC), sent_at_ns=100, end_ns=200),
        usage=Usage(input_tokens=10, output_tokens=256),
        server_model="fake-model",
        stop_reason="max_tokens",
        blocks=[
            ContentBlock(type="thinking", text="思考"),
            ContentBlock(type="text", text="本文"),
        ],
    )


def _target(server: FakeServer) -> TargetDef:
    return TargetDef(name="fake", base_url=HttpUrl(server.base_url), model="fake-model")


async def _recount(server: FakeServer, *, timeout_s: float = 5.0) -> Any:
    recount_output = importlib.import_module("bench_harness.client.tokenize").recount_output
    return await recount_output(_result(), _target(server), api_key=None, timeout_s=timeout_s)


async def test_raw_phase_strings_are_counted_without_chat_wrapping(
    fake_server: FakeServer,
) -> None:
    fake_server.set_tokenize_response(200, {"count": 2, "tokens": [10, 11], "model": "fake-model"})
    counts = await _recount(fake_server)

    requests = fake_server.requests_for("/tokenize")
    assert len(requests) == 2
    assert {request.body["prompt"] for request in requests if request.body is not None} == {
        "思考",
        "本文",
    }
    assert all(
        request.body is not None
        and request.body["model"] == "fake-model"
        and request.body["add_special_tokens"] is False
        and "messages" not in request.body
        for request in requests
    )
    assert counts.thinking.count == 2
    assert counts.text.count == 2
    assert counts.thinking.count_source == "retokenized"
    assert counts.text.count_source == "retokenized"
    assert "256" not in counts.model_dump_json()


@pytest.mark.parametrize(
    ("status", "payload"),
    [
        (200, {"input_tokens": 256}),
        (200, {"count": True}),
        (200, {"count": -1}),
        (200, {"count": 1.5}),
        (200, {"count": "2"}),
        (200, {"count": 2, "tokens": [10]}),
        (200, {"count": 2, "tokens": [10, 11], "model": "other-model"}),
        (200, "{broken-json"),
        (404, {"error": "SENTINEL-PRIVATE-ERROR"}),
    ],
)
async def test_invalid_tokenize_results_remain_unknown_without_usage_fallback(
    fake_server: FakeServer, status: int, payload: dict[str, Any] | str
) -> None:
    fake_server.set_tokenize_response(status, payload)
    counts = await _recount(fake_server)

    assert counts.thinking.count is None
    assert counts.text.count is None
    assert counts.thinking.reason is not None
    assert counts.text.reason is not None
    assert "SENTINEL-PRIVATE-ERROR" not in counts.model_dump_json()
    assert "256" not in counts.model_dump_json()


async def test_tokenize_timeout_keeps_the_counts_unknown(fake_server: FakeServer) -> None:
    fake_server.set_tokenize_response(200, {"count": 2, "tokens": [10, 11]}, delay_s=0.2)
    counts = await _recount(fake_server, timeout_s=0.01)

    assert counts.thinking.count is None
    assert counts.text.count is None
    assert counts.thinking.reason is not None
    assert counts.text.reason is not None
