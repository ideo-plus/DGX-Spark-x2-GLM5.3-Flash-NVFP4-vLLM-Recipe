"""試験用の偽のサーバー自身の試験 (task 1.4)。

実際のソケット越しに、指定した遅れとイベントの列がそのとおり届くことと、
後続のタスクが要る振る舞いがすべて指定できることを確かめる。
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import httpx
import pytest

from fake_server import (
    FakeServer,
    Script,
    SseEvent,
    UsageSpec,
    dropped_response,
    empty_response,
    error_stream_response,
    estimator_aligned_counter,
    http_error_response,
    over_limit_response,
    ping_event,
    stalled_response,
    text_events,
    text_response,
    thinking_events,
    token_stream_response,
    tool_use_events,
    tool_use_response,
)

# --- 助け ---------------------------------------------------------------


def make_body(text: str = "hi", *, max_tokens: int = 64, **extra: Any) -> dict[str, Any]:
    """`/v1/messages` に送る最小の本文。"""
    body: dict[str, Any] = {
        "model": "fake-model",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": text}],
        "stream": True,
    }
    body.update(extra)
    return body


def parse_block(block: bytes) -> tuple[str, str]:
    """1 つの SSE のかたまりを、(イベントの種類, data の文字列) に分ける。"""
    event_type = ""
    data_lines: list[str] = []
    for line in block.decode("utf-8").split("\n"):
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    return event_type, "\n".join(data_lines)


def read_stream(response: httpx.Response) -> list[tuple[float, str, str]]:
    """(届いた時刻, イベントの種類, data の文字列) の並びを返す。"""
    out: list[tuple[float, str, str]] = []
    buffer = b""
    for chunk in response.iter_raw():
        now = time.perf_counter()
        buffer += chunk
        while b"\n\n" in buffer:
            block, buffer = buffer.split(b"\n\n", 1)
            if not block.strip():
                continue
            event_type, data = parse_block(block)
            out.append((now, event_type, data))
    return out


def stream_with(
    client: httpx.Client, body: dict[str, Any] | None = None
) -> list[tuple[float, str, str]]:
    """すでにある client で 1 つの要求を送る (接続の確立を測定から外せる)。"""
    with client.stream("POST", "/v1/messages", json=body or make_body()) as response:
        response.raise_for_status()
        return read_stream(response)


def stream_events(
    server: FakeServer, body: dict[str, Any] | None = None, **kwargs: Any
) -> list[tuple[float, str, str]]:
    """1 つの要求をストリームで送り、届いたイベントを返す。"""
    with httpx.Client(base_url=server.base_url, timeout=10.0, **kwargs) as client:
        return stream_with(client, body)


def types_of(events: list[tuple[float, str, str]]) -> list[str]:
    return [event_type for _, event_type, _ in events]


def data_of(events: list[tuple[float, str, str]], event_type: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, kind, data in events:
        if kind == event_type:
            parsed: dict[str, Any] = json.loads(data)
            out.append(parsed)
    return out


@pytest.fixture
def server(fake_server: FakeServer) -> FakeServer:
    return fake_server


# --- イベントの列と順序 (項目 1) ----------------------------------------


def test_scripted_event_order_arrives_verbatim(server: FakeServer) -> None:
    server.set_response(
        Script(
            events=(
                ping_event(),
                *text_events("Hello", index=0, chunks=("He", "llo")),
                SseEvent("surprise", {"type": "surprise", "unknown_field": 7}),
                *tool_use_events("read_file", {"path": "/tmp/a"}, index=1),
                *text_events("done", index=2),
            ),
            stop_reason="tool_use",
        )
    )

    events = stream_events(server)

    assert types_of(events) == [
        "message_start",
        "ping",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "surprise",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert json.loads(events[6][2])["unknown_field"] == 7
    assert data_of(events, "message_delta")[0]["delta"]["stop_reason"] == "tool_use"


def test_thinking_block_and_empty_response_are_scriptable(server: FakeServer) -> None:
    server.set_response(Script(events=thinking_events("考え中", chunks=("考え", "中"))))
    events = stream_events(server)
    deltas = data_of(events, "content_block_delta")
    assert [delta["delta"]["type"] for delta in deltas] == ["thinking_delta", "thinking_delta"]
    assert "".join(delta["delta"]["thinking"] for delta in deltas) == "考え中"

    server.set_response(Script(events=(), stop_reason="max_tokens"))
    events = stream_events(server)
    assert types_of(events) == ["message_start", "message_delta", "message_stop"]
    assert data_of(events, "message_delta")[0]["delta"]["stop_reason"] == "max_tokens"


def test_tool_use_fragments_arrive_verbatim_including_broken_json(server: FakeServer) -> None:
    fragments = ('{"pa', 'th": "/tmp/a"', "  <<broken")
    server.set_response(tool_use_response("read_file", fragments=fragments))

    events = stream_events(server)

    deltas = data_of(events, "content_block_delta")
    assert [delta["delta"]["partial_json"] for delta in deltas] == list(fragments)
    starts = data_of(events, "content_block_start")
    assert starts[0]["content_block"]["name"] == "read_file"


def test_text_after_tool_use_and_leaked_markup(server: FakeServer) -> None:
    server.set_response(
        tool_use_response("read_file", {"path": "/tmp/a"}, text_after="あとから来た本文")
    )
    events = stream_events(server)
    starts = data_of(events, "content_block_start")
    assert [start["content_block"]["type"] for start in starts] == ["tool_use", "text"]
    assert starts[1]["index"] == 1

    leaked = "<tool_call>read_file<arg_key>path<arg_value>/tmp/a</tool_call>"
    server.set_response(text_response(leaked))
    events = stream_events(server)
    assert data_of(events, "content_block_delta")[0]["delta"]["text"] == leaked


# --- 遅れ (項目 2) -------------------------------------------------------
#
# 時刻の判定の決まり: 偽のサーバーは、指定した時刻より早くは送れない。そこで
# 「早すぎないこと」は 1 回の測定でも確かめられる (下限は揺らがない)。ただし、
# これが成り立つのは、基準点を「試験自身が要求を送る前に打った時刻」にしたとき
# だけ。2 つのイベントの読み取り時刻の差は、前のイベントの読み取りが遅れると
# 縮むので、下限には使えない。
# 遅い側は OS の割り込みと、クライアント側で 2 つの塊が 1 回の read にまとまる
# ことで跳ねる。そこで遅い側は、中央値、全体の幅、外れた本数の上限で見る。
# 個々の値を狭い幅で全数判定すると、偽の失敗が混ざる。


def test_first_delay_is_observable(server: FakeServer) -> None:
    server.set_response(text_response("ok", first_delay_s=0.080))

    firsts: list[float] = []
    with httpx.Client(base_url=server.base_url, timeout=10.0) as client:
        stream_with(client)  # 接続を温める (TCP の確立を測定から外す)
        for _ in range(3):
            started = time.perf_counter()
            events = stream_with(client)
            firsts.append(events[0][0] - started)

    assert min(firsts) >= 0.078, f"指定より早く届いている: {firsts}"
    assert abs(statistics.median(firsts) - 0.080) < 0.010, (
        f"最初のイベントまでの遅れが外れている: {firsts}"
    )


def test_inter_event_gaps_are_observable(server: FakeServer) -> None:
    server.set_response(token_stream_response(output_tokens=10, first_delay_s=0.08, gap_s=0.030))

    with httpx.Client(base_url=server.base_url, timeout=10.0) as client:
        stream_with(client)  # 接続を温める (TCP の確立を測定から外す)
        events = stream_with(client)

    gaps = [later[0] - earlier[0] for earlier, later in zip(events, events[1:], strict=False)]
    assert len(gaps) == 14
    assert abs(statistics.median(gaps) - 0.030) < 0.003, f"イベントの間の遅れが外れている: {gaps}"
    outliers = [gap for gap in gaps if abs(gap - 0.030) >= 0.005]
    assert len(outliers) <= 2, f"遅れの外れが多すぎる: {gaps}"
    span = events[-1][0] - events[0][0]
    assert abs(span - 14 * 0.030) < 0.010, f"全体の長さが外れている: {span}"


def test_per_event_delay_overrides_the_default_gap(server: FakeServer) -> None:
    server.set_response(
        Script(
            events=(
                *text_events("a"),
                SseEvent("ping", {"type": "ping"}, delay_s=0.080),
            ),
            gap_s=0.0,
        )
    )

    # gap_s=0.0 なので、ping より前のイベントはすぐに届く。要求の送り始めから ping
    # までが、そのまま指定した遅れになる (基準点は、試験自身が打った時刻)。
    ping_arrivals: list[float] = []
    with httpx.Client(base_url=server.base_url, timeout=10.0) as client:
        stream_with(client)  # 接続を温める
        for _ in range(3):
            started = time.perf_counter()
            events = stream_with(client)
            ping_index = types_of(events).index("ping")
            assert ping_index == len(events) - 3, f"ping の位置が違う: {types_of(events)}"
            ping_arrivals.append(events[ping_index][0] - started)

    assert min(ping_arrivals) >= 0.078, f"指定した遅れより早く届いている: {ping_arrivals}"
    assert abs(statistics.median(ping_arrivals) - 0.080) < 0.010, (
        f"指定した遅れが効いていない: {ping_arrivals}"
    )


# --- トークン数 (項目 3) -------------------------------------------------


def test_usage_placement_follows_the_real_server_rule(server: FakeServer) -> None:
    server.set_response(
        replace(
            text_response("ok"),
            usage=UsageSpec(
                input_tokens=1000,
                output_tokens=7,
                cache_read_input_tokens=600,
                cache_creation_input_tokens=100,
            ),
        )
    )

    events = stream_events(server)

    start_usage = data_of(events, "message_start")[0]["message"]["usage"]
    assert start_usage["input_tokens"] == 1000
    assert "cache_read_input_tokens" not in start_usage
    delta_usage = data_of(events, "message_delta")[0]["usage"]
    assert delta_usage == {
        "input_tokens": 300,
        "output_tokens": 7,
        "cache_read_input_tokens": 600,
        "cache_creation_input_tokens": 100,
    }
    assert (
        delta_usage["input_tokens"]
        + delta_usage["cache_read_input_tokens"]
        + delta_usage["cache_creation_input_tokens"]
        == 1000
    )


def test_usage_can_be_absent_entirely(server: FakeServer) -> None:
    server.set_response(replace(text_response("ok"), usage=None))

    events = stream_events(server)

    assert "usage" not in data_of(events, "message_start")[0]["message"]
    assert "usage" not in data_of(events, "message_delta")[0]


def test_message_delta_can_be_missing(server: FakeServer) -> None:
    server.set_response(replace(text_response("ok"), include_message_delta=False))

    events = stream_events(server)

    assert "message_delta" not in types_of(events)
    assert types_of(events)[-1] == "message_stop"
    assert data_of(events, "message_start")[0]["message"]["usage"]["input_tokens"] > 0


def test_usage_can_be_left_out_of_message_start(server: FakeServer) -> None:
    server.set_response(
        replace(text_response("ok"), usage=UsageSpec(output_tokens=3, in_message_start=False))
    )

    events = stream_events(server)

    assert "usage" not in data_of(events, "message_start")[0]["message"]
    assert data_of(events, "message_delta")[0]["usage"]["output_tokens"] == 3


# --- 失敗 (項目 4) -------------------------------------------------------


def test_http_error_before_streaming(server: FakeServer) -> None:
    server.set_response(http_error_response(503, "ServiceUnavailableError", "落ちている"))

    response = httpx.post(f"{server.base_url}/v1/messages", json=make_body(), timeout=5.0)

    assert response.status_code == 503
    assert response.json() == {
        "type": "error",
        "error": {"type": "ServiceUnavailableError", "message": "落ちている"},
    }


def test_mid_stream_error_event(server: FakeServer) -> None:
    server.set_response(
        error_stream_response(
            before=text_events("途中まで"), error_type="InternalServerError", message="壊れた"
        )
    )

    events = stream_events(server)

    assert types_of(events)[-1] == "error"
    payload = json.loads(events[-1][2])
    assert payload["error"]["type"] == "InternalServerError"
    assert payload["error"]["message"] == "壊れた"
    assert "message_stop" not in types_of(events)


def test_connection_drop_mid_stream(server: FakeServer) -> None:
    server.set_response(dropped_response(after_events=2))

    with (
        httpx.Client(base_url=server.base_url, timeout=5.0) as client,
        pytest.raises((httpx.RemoteProtocolError, httpx.ReadError)),
        client.stream("POST", "/v1/messages", json=make_body()) as response,
    ):
        read_stream(response)


def test_stalled_stream_hits_the_client_timeout(server: FakeServer) -> None:
    server.set_response(stalled_response(after_events=1, stall_s=3.0))

    started = time.perf_counter()
    with (
        httpx.Client(base_url=server.base_url, timeout=httpx.Timeout(5.0, read=0.25)) as client,
        pytest.raises(httpx.ReadTimeout),
        client.stream("POST", "/v1/messages", json=make_body()) as response,
    ):
        read_stream(response)
    assert time.perf_counter() - started < 1.5


def test_stalled_stream_does_not_block_shutdown() -> None:
    server = FakeServer()
    server.start()
    server.set_response(stalled_response(after_events=1, stall_s=30.0))
    try:
        with (
            httpx.Client(base_url=server.base_url, timeout=httpx.Timeout(5.0, read=0.2)) as client,
            pytest.raises(httpx.ReadTimeout),
            client.stream("POST", "/v1/messages", json=make_body()) as response,
        ):
            read_stream(response)
    finally:
        started = time.perf_counter()
        server.stop()
        assert time.perf_counter() - started < 1.0


def test_slow_first_byte_delays_the_first_event(server: FakeServer) -> None:
    server.set_response(text_response("ok", first_delay_s=0.35))

    with (
        httpx.Client(base_url=server.base_url, timeout=httpx.Timeout(5.0, read=0.15)) as client,
        pytest.raises(httpx.ReadTimeout),
        client.stream("POST", "/v1/messages", json=make_body()) as response,
    ):
        read_stream(response)


# --- 入力の長さの上限とトークン数 (項目 5) ------------------------------


def test_over_limit_request_gets_http_400(server: FakeServer) -> None:
    server.set_context_limit(1000)

    response = httpx.post(
        f"{server.base_url}/v1/messages",
        json=make_body("x" * 8000, max_tokens=16),
        timeout=5.0,
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "BadRequestError"
    assert "maximum context length is 1000 tokens" in payload["error"]["message"]


def test_under_limit_request_is_served(server: FakeServer) -> None:
    server.set_context_limit(100000)

    events = stream_events(server, make_body("x" * 8000))

    assert types_of(events)[0] == "message_start"


def test_over_limit_can_also_be_scripted_directly(server: FakeServer) -> None:
    server.set_response(over_limit_response(32000, input_tokens=40000, max_tokens=16))

    response = httpx.post(f"{server.base_url}/v1/messages", json=make_body(), timeout=5.0)

    assert response.status_code == 400
    assert "maximum context length is 32000 tokens" in response.json()["error"]["message"]


def test_count_tokens_matches_the_streamed_input_tokens(server: FakeServer) -> None:
    body = make_body("x" * 4000)

    counted = httpx.post(
        f"{server.base_url}/v1/messages/count_tokens", json=body, timeout=5.0
    ).json()["input_tokens"]
    events = stream_events(server, body)

    assert data_of(events, "message_start")[0]["message"]["usage"]["input_tokens"] == counted


def test_empty_response_has_no_content_blocks(server: FakeServer) -> None:
    server.set_response(empty_response(stop_reason="max_tokens"))

    events = stream_events(server)

    assert types_of(events) == ["message_start", "message_delta", "message_stop"]
    assert data_of(events, "message_delta")[0]["usage"]["output_tokens"] == 0


def test_input_tokens_grow_with_the_size_of_the_input(server: FakeServer) -> None:
    small = stream_events(server, make_body("x" * 4000))
    large = stream_events(server, make_body("x" * 8000))

    small_tokens = data_of(small, "message_start")[0]["message"]["usage"]["input_tokens"]
    large_tokens = data_of(large, "message_start")[0]["message"]["usage"]["input_tokens"]
    assert small_tokens >= 1000
    assert 1.9 < large_tokens / small_tokens < 2.1
    assert server.requests_for("/v1/messages")[0].input_tokens == small_tokens


def test_scripted_usage_overrides_the_computed_input_tokens(server: FakeServer) -> None:
    server.set_response(
        replace(text_response("ok"), usage=UsageSpec(input_tokens=42, output_tokens=1))
    )

    events = stream_events(server, make_body("x" * 4000))

    assert data_of(events, "message_start")[0]["message"]["usage"]["input_tokens"] == 42


def test_count_tokens_endpoint_present_and_absent(server: FakeServer) -> None:
    response = httpx.post(
        f"{server.base_url}/v1/messages/count_tokens", json=make_body("x" * 4000), timeout=5.0
    )
    assert response.status_code == 200
    assert response.json()["input_tokens"] >= 1000

    server.set_count_tokens_enabled(False)
    response = httpx.post(
        f"{server.base_url}/v1/messages/count_tokens", json=make_body(), timeout=5.0
    )
    assert response.status_code == 404
    assert server.call_count("/v1/messages/count_tokens") == 2


def test_models_and_version_endpoints(server: FakeServer) -> None:
    server.set_context_limit(32000)
    payload = httpx.get(f"{server.base_url}/v1/models", timeout=5.0).json()
    assert payload["data"][0]["max_model_len"] == 32000
    assert payload["data"][0]["id"] == "fake-model"

    server.set_context_limit(32000, advertise=False)
    payload = httpx.get(f"{server.base_url}/v1/models", timeout=5.0).json()
    assert "max_model_len" not in payload["data"][0]

    assert httpx.get(f"{server.base_url}/version", timeout=5.0).json()["version"] != ""
    server.set_version(None)
    assert httpx.get(f"{server.base_url}/version", timeout=5.0).status_code == 404


# --- 応答を決める関数 (項目 6) ------------------------------------------


def test_response_factory_answers_per_request(server: FakeServer) -> None:
    def factory(body: dict[str, Any]) -> Script:
        text = str(body["messages"][0]["content"])
        if text == "fail":
            return http_error_response(500, "InternalServerError", "だめ")
        return tool_use_response("read_file", {"path": text})

    server.set_response_factory(factory)

    events = stream_events(server, make_body("/tmp/wanted"))
    assert json.loads(data_of(events, "content_block_delta")[0]["delta"]["partial_json"]) == {
        "path": "/tmp/wanted"
    }

    response = httpx.post(f"{server.base_url}/v1/messages", json=make_body("fail"), timeout=5.0)
    assert response.status_code == 500


def test_response_sequence_repeats_the_last_script(server: FakeServer) -> None:
    server.set_response_sequence(
        [text_response("1 番目"), text_response("2 番目"), text_response("最後")]
    )

    texts = [
        data_of(stream_events(server), "content_block_delta")[0]["delta"]["text"] for _ in range(4)
    ]

    assert texts == ["1 番目", "2 番目", "最後", "最後"]


def test_default_script_is_a_fixed_text(server: FakeServer) -> None:
    events = stream_events(server)
    assert types_of(events)[0] == "message_start"
    assert data_of(events, "content_block_delta")[0]["delta"]["text"] != ""


# --- プレフィックスキャッシュ (項目 7) ----------------------------------


def test_prefix_cache_knob(server: FakeServer) -> None:
    server.set_prefix_cache(True)
    server.set_response(text_response("ok", first_delay_s=0.20))
    system = "共通の前置き。" * 300

    started = time.perf_counter()
    cold = stream_events(server, make_body("1 問目", system=system))
    cold_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    warm = stream_events(server, make_body("2 問目", system=system))
    warm_elapsed = time.perf_counter() - started

    assert data_of(cold, "message_delta")[0]["usage"]["cache_read_input_tokens"] == 0
    warm_usage = data_of(warm, "message_delta")[0]["usage"]
    assert warm_usage["cache_read_input_tokens"] > 0
    assert warm_elapsed < cold_elapsed * 0.7
    metrics = httpx.get(f"{server.base_url}/metrics", timeout=5.0).text
    assert "vllm:prefix_cache_hits_total" in metrics
    hits = _metric_value(metrics, "vllm:prefix_cache_hits_total")
    assert hits > 0


def test_prefix_cache_is_off_by_default(server: FakeServer) -> None:
    system = "共通の前置き。" * 300
    stream_events(server, make_body("1 問目", system=system))
    warm = stream_events(server, make_body("2 問目", system=system))

    assert "cache_read_input_tokens" not in data_of(warm, "message_delta")[0]["usage"]
    metrics = httpx.get(f"{server.base_url}/metrics", timeout=5.0).text
    assert _metric_value(metrics, "vllm:prefix_cache_hits_total") == 0.0
    assert _metric_value(metrics, "vllm:prefix_cache_queries_total") > 0


def test_scripted_cache_fields_do_not_touch_the_delay_or_metrics(server: FakeServer) -> None:
    server.set_response(
        replace(
            text_response("ok", first_delay_s=0.10),
            usage=UsageSpec(input_tokens=1000, output_tokens=1, cache_read_input_tokens=900),
        )
    )

    started = time.perf_counter()
    events = stream_events(server)
    elapsed = time.perf_counter() - started

    assert data_of(events, "message_delta")[0]["usage"]["cache_read_input_tokens"] == 900
    assert elapsed >= 0.10, "台本のキャッシュの値が、遅れを縮めてしまっている"
    metrics = httpx.get(f"{server.base_url}/metrics", timeout=5.0).text
    assert _metric_value(metrics, "vllm:prefix_cache_hits_total") == 0.0


# --- /metrics (項目 8) ---------------------------------------------------


def _metric_value(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        metric_name = head.split("{", 1)[0]
        if metric_name == name:
            return float(value)
    raise AssertionError(f"{name} が /metrics にない")


def test_metrics_counters_advance_with_each_request(server: FakeServer) -> None:
    server.set_response(token_stream_response(output_tokens=5))
    server.set_metrics_options(acceptance_ratio=0.8)

    stream_events(server, make_body("x" * 400))
    input_tokens = server.requests_for("/v1/messages")[0].input_tokens
    text = httpx.get(f"{server.base_url}/metrics", timeout=5.0).text

    assert _metric_value(text, "vllm:generation_tokens_total") == 5.0
    assert _metric_value(text, "vllm:prompt_tokens_total") == float(input_tokens)
    assert _metric_value(text, "vllm:iteration_tokens_total_count") == 5.0
    assert _metric_value(text, "vllm:iteration_tokens_total_sum") == float(input_tokens + 5)
    assert _metric_value(text, "vllm:spec_decode_num_draft_tokens_total") == 5.0
    assert _metric_value(text, "vllm:spec_decode_num_accepted_tokens_total") == 4.0
    assert _metric_value(text, "vllm:num_requests_running") == 0.0
    assert _metric_value(text, "vllm:kv_cache_usage_perc") >= 0.0
    assert _metric_value(text, "vllm:num_preemptions_total") == 0.0

    stream_events(server, make_body("x" * 400))
    text = httpx.get(f"{server.base_url}/metrics", timeout=5.0).text
    assert _metric_value(text, "vllm:generation_tokens_total") == 10.0


def test_metrics_text_is_parseable_by_prometheus_client(server: FakeServer) -> None:
    from prometheus_client.parser import text_string_to_metric_families

    stream_events(server)
    text = httpx.get(f"{server.base_url}/metrics", timeout=5.0).text

    names = {
        sample.name for family in text_string_to_metric_families(text) for sample in family.samples
    }
    assert "vllm:prompt_tokens_total" in names
    assert "vllm:iteration_tokens_total_sum" in names
    assert "vllm:kv_cache_usage_perc" in names


def test_metrics_can_omit_the_spec_decode_family(server: FakeServer) -> None:
    server.set_metrics_options(include_spec_decode=False)

    text = httpx.get(f"{server.base_url}/metrics", timeout=5.0).text

    assert "spec_decode" not in text
    assert "vllm:prefix_cache_hits_total" in text


def test_metrics_names_can_be_renamed(server: FakeServer) -> None:
    server.set_metrics_options(rename={"vllm:prefix_cache_hits_total": "custom:cache_hits"})

    text = httpx.get(f"{server.base_url}/metrics", timeout=5.0).text

    assert "custom:cache_hits" in text
    assert "vllm:prefix_cache_hits_total" not in text


def test_metrics_can_be_raw_text_or_missing(server: FakeServer) -> None:
    server.set_metrics_text("# HELP x y\n# TYPE x counter\nx 1.0\n")
    assert httpx.get(f"{server.base_url}/metrics", timeout=5.0).text.endswith("x 1.0\n")

    server.set_metrics_enabled(False)
    assert httpx.get(f"{server.base_url}/metrics", timeout=5.0).status_code == 404
    assert server.call_count("/metrics") == 2


def test_preemptions_and_kv_usage_are_scriptable(server: FakeServer) -> None:
    server.bump_preemptions(3)
    server.set_kv_usage(0.75)

    text = httpx.get(f"{server.base_url}/metrics", timeout=5.0).text

    assert _metric_value(text, "vllm:num_preemptions_total") == 3.0
    assert _metric_value(text, "vllm:kv_cache_usage_perc") == 0.75


# --- 記録 (項目 9) -------------------------------------------------------


def test_headers_and_body_are_recorded(server: FakeServer) -> None:
    body = make_body("こんにちは", extra_field={"enable_thinking": False})
    httpx.post(
        f"{server.base_url}/v1/messages",
        json=body,
        headers={"Authorization": "Bearer secret-value", "x-api-key": "secret-value"},
        timeout=5.0,
    )

    record = server.requests_for("/v1/messages")[0]
    assert record.headers["authorization"] == "Bearer secret-value"
    assert record.headers["x-api-key"] == "secret-value"
    assert record.body == body
    assert record.body is not None and record.body["extra_field"] == {"enable_thinking": False}
    assert record.path == "/v1/messages"
    assert record.method == "POST"
    assert record.arrived_at_ns > 0


def test_wait_for_requests_and_reset(server: FakeServer) -> None:
    stream_events(server)
    stream_events(server)

    records = server.wait_for_requests(2, path="/v1/messages", timeout_s=1.0)
    assert len(records) == 2
    assert server.call_count("/v1/messages") == 2

    server.reset()
    assert server.requests == []
    assert server.call_count("/v1/messages") == 0
    text = httpx.get(f"{server.base_url}/metrics", timeout=5.0).text
    assert _metric_value(text, "vllm:generation_tokens_total") == 0.0


def test_wait_for_requests_times_out(server: FakeServer) -> None:
    with pytest.raises(TimeoutError):
        server.wait_for_requests(1, timeout_s=0.1)


async def test_eight_concurrent_streams_are_recorded(server: FakeServer) -> None:
    server.set_response(token_stream_response(output_tokens=6, first_delay_s=0.05, gap_s=0.02))

    async def one(index: int) -> int:
        async with (
            httpx.AsyncClient(base_url=server.base_url, timeout=10.0) as client,
            client.stream("POST", "/v1/messages", json=make_body(f"stream-{index}")) as response,
        ):
            count = 0
            async for _ in response.aiter_raw():
                count += 1
            return count

    counts = await asyncio.gather(*(one(index) for index in range(8)))

    assert all(count > 0 for count in counts)
    records = server.requests_for("/v1/messages")
    assert len(records) == 8
    assert max(record.in_flight for record in records) == 8
    arrivals = sorted(record.arrived_at_ns for record in records)
    assert arrivals[-1] - arrivals[0] < 500_000_000
    assert server.errors == []


def test_unknown_path_returns_404(server: FakeServer) -> None:
    assert httpx.get(f"{server.base_url}/nope", timeout=5.0).status_code == 404
    assert server.call_count("/nope") == 1


# --- fixture (項目 10) ---------------------------------------------------


def test_fixture_yields_a_started_server(fake_server: FakeServer) -> None:
    assert fake_server.base_url.startswith("http://127.0.0.1:")
    assert httpx.get(f"{fake_server.base_url}/v1/models", timeout=5.0).status_code == 200


def test_factory_fixture_can_start_several_servers(
    fake_server_factory: Callable[[], FakeServer],
) -> None:
    first = fake_server_factory()
    second = fake_server_factory()
    assert first.base_url != second.base_url
    assert httpx.get(f"{second.base_url}/v1/models", timeout=5.0).status_code == 200


# --- 入力トークン数の数え方の差し替え (issue #9) --------------------------


def test_input_token_counter_applies_to_messages_and_count_tokens(server: FakeServer) -> None:
    server.set_input_token_counter(estimator_aligned_counter(fixed=100))
    body = make_body("x" * 400)

    counted = httpx.post(
        f"{server.base_url}/v1/messages/count_tokens", json=body, timeout=5.0
    ).json()["input_tokens"]
    events = stream_events(server, body)
    started = data_of(events, "message_start")[0]["message"]["usage"]["input_tokens"]

    # "x" * 400 は文字列の content (400 / chars_per_token 4.0 = 100) + fixed 100
    assert counted == 200
    assert started == 200

    server.set_input_token_counter(None)
    reset_counted = httpx.post(
        f"{server.base_url}/v1/messages/count_tokens", json=body, timeout=5.0
    ).json()["input_tokens"]

    # 既定の JSON の文字数 ÷ 比に戻り、差し替え口の値とは変わる
    assert reset_counted != 200


def test_estimator_aligned_counter_counts_blocks_like_the_estimator() -> None:
    counter = estimator_aligned_counter(fixed=0, per_message=1, chars_per_token=1.0)
    tool_input = {"path": "/tmp/a"}
    body: dict[str, Any] = {
        "system": "SYS",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "read_file", "input": tool_input}],
            },
            {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]},
        ],
    }

    tool_use_chars = len("read_file") + len(
        json.dumps(tool_input, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )
    expected = (
        1 * len(body["messages"])  # per_message × 発話数
        + len("SYS")  # system
        + len("hello")  # text
        + tool_use_chars  # tool_use: name + json.dumps(input)
        + len("ok")  # tool_result: content
    )

    assert counter(body) == expected
