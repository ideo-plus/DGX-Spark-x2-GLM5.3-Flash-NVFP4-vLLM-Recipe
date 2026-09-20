"""`/v1/messages` のクライアントの試験 (task 2.1)。

実際のソケット越しに、偽のサーバーを相手に確かめる。時刻を判定する試験は
tasks.md の Implementation Notes (1.4) の決まりに従う: 先に 1 回流して接続を
温め、最初の遅れは 100 ミリ秒以上にし、遅い側は中央値で見て、下限は「試験自身が
要求を送る前に打った時刻」だけを基準にする。
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import SecretStr

from bench_harness.client.messages import (
    ANTHROPIC_VERSION,
    HttpxMessagesClient,
    MessagesClient,
    build_request_body,
)
from bench_harness.types import (
    ContentBlock,
    InputMessage,
    MessagesRequest,
    StreamResult,
    StreamTiming,
    TargetDef,
    TextBlockParam,
    TimeoutPolicy,
    Usage,
)
from fake_server import (
    FakeServer,
    Script,
    SseEvent,
    UsageSpec,
    dropped_response,
    error_stream_response,
    http_error_response,
    ping_event,
    replace,
    stalled_response,
    text_events,
    text_response,
    thinking_response,
    token_stream_response,
    tool_use_events,
    tool_use_response,
)

# --- 助け ---------------------------------------------------------------

POLICY = TimeoutPolicy(connect_s=5.0, first_event_s=5.0, idle_s=5.0, total_s=20.0)
"""ふつうの試験で使う、余裕のある制限時間。"""

SECRET = "sk-fake-super-secret-value"


def make_request(
    text: str = "こんにちは", *, max_tokens: int = 64, **kwargs: Any
) -> MessagesRequest:
    """最小の要求。"""
    return MessagesRequest(
        model="fake-model",
        max_tokens=max_tokens,
        messages=[InputMessage(role="user", content=[TextBlockParam(text=text)])],
        **kwargs,
    )


def client_for(server: FakeServer, api_key: SecretStr | None = None) -> HttpxMessagesClient:
    return HttpxMessagesClient(server.base_url, api_key)


def block_key(block: ContentBlock) -> tuple[str, str, str, str]:
    """ブロックを、並び順に依存せず比べるための鍵。"""
    return (block.type, block.text or "", block.tool_name or "", block.tool_input_raw or "")


def by_key(blocks: list[ContentBlock]) -> list[ContentBlock]:
    return sorted(blocks, key=block_key)


def observable(result: StreamResult) -> tuple[list[ContentBlock], Usage | None, str | None]:
    """イベントの並びに依存しないはずの、観測できる中身。"""
    return (by_key(result.blocks), result.usage, result.stop_reason)


def assert_timing_is_ordered(result: StreamResult) -> None:
    """来なかった時刻を飛ばして、残りの並びが前後していないこと (失敗の道でも)。"""
    timing = result.timing
    stages = [
        value
        for value in (
            timing.sent_at_ns,
            timing.message_start_ns,
            timing.first_token_ns,
            timing.first_text_ns,
            timing.last_token_ns,
            timing.end_ns,
        )
        if value is not None
    ]
    assert stages == sorted(stages), f"時刻の並びが前後している: {timing!r}"
    StreamTiming.model_validate(timing.model_dump())


def unknown_field_events() -> tuple[SseEvent, ...]:
    """知らない項目を混ぜた、本文のブロック 1 つ。"""
    return (
        SseEvent(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "", "future": 1},
                "unknown_field": "x",
            },
        ),
        SseEvent(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ok", "future": True},
                "unknown_field": [1, 2],
            },
        ),
        SseEvent("content_block_stop", {"type": "content_block_stop", "index": 0}),
    )


# --- 正常の道 -----------------------------------------------------------


async def test_text_response_returns_blocks_usage_and_stop_reason(fake_server: FakeServer) -> None:
    fake_server.set_response(text_response("こんばんは"))

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is None
    assert result.blocks == [ContentBlock(type="text", text="こんばんは")]
    assert result.stop_reason == "end_turn"
    assert result.server_model == "fake-model"
    assert result.usage is not None
    assert result.usage.output_tokens == 1
    assert result.timing.event_count >= 6
    assert result.timing.message_start_ns is not None
    assert result.timing.first_token_ns is not None
    assert result.timing.first_text_ns == result.timing.first_token_ns
    assert result.timing.last_token_ns == result.timing.first_token_ns
    assert_timing_is_ordered(result)
    assert fake_server.call_count("/v1/messages") == 1


async def test_client_satisfies_the_messages_client_protocol(fake_server: FakeServer) -> None:
    async with client_for(fake_server) as client:
        assert isinstance(client, MessagesClient)


async def test_from_target_builds_a_client_for_the_target_base_url(fake_server: FakeServer) -> None:
    target = TargetDef.model_validate(
        {"name": "fake", "base_url": fake_server.base_url, "model": "fake-model"}
    )

    async with HttpxMessagesClient.from_target(target) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is None
    assert fake_server.call_count("/v1/messages") == 1


# --- 速さ (2.3、3.1) ----------------------------------------------------
#
# 偽のサーバーは、指定した時刻より早くは送れない。だから「早すぎないこと」は、
# 試験自身が要求を送る前に打った時刻を基準にすれば 1 回でも判定できる。遅い側は
# OS の割り込みで跳ねるので、複数回の中央値で見る。

FIRST_DELAY_S = 0.150
GAP_S = 0.020
TOKENS = 8
MEASUREMENTS = 5


async def test_ttft_and_decode_speed_are_within_5_percent_of_the_script(
    fake_server: FakeServer,
) -> None:
    expected_ttft_s = (
        FIRST_DELAY_S + 2 * GAP_S
    )  # message_start → content_block_start → 最初の delta
    expected_speed = 1.0 / GAP_S

    ttfts: list[float] = []
    speeds: list[float] = []
    fake_server.set_response(text_response("warm"))
    async with client_for(fake_server) as client:
        await client.stream(make_request(), POLICY)  # 接続を温める (TCP の確立を測定から外す)
        fake_server.set_response(
            token_stream_response(output_tokens=TOKENS, first_delay_s=FIRST_DELAY_S, gap_s=GAP_S)
        )
        for _ in range(MEASUREMENTS):
            started_ns = time.perf_counter_ns()
            result = await client.stream(make_request(), POLICY)

            assert result.error is None
            timing = result.timing
            assert timing.first_token_ns is not None
            assert timing.last_token_ns is not None
            assert result.usage is not None and result.usage.output_tokens == TOKENS
            # 下限 (早すぎない) は、試験自身が打った時刻を基準にする
            assert (timing.first_token_ns - started_ns) / 1e9 >= expected_ttft_s - 0.002
            ttfts.append((timing.first_token_ns - timing.sent_at_ns) / 1e9)
            span_s = (timing.last_token_ns - timing.first_token_ns) / 1e9
            speeds.append((result.usage.output_tokens - 1) / span_s)

    ttft = statistics.median(ttfts)
    speed = statistics.median(speeds)
    assert abs(ttft - expected_ttft_s) <= expected_ttft_s * 0.05, f"TTFT が外れている: {ttfts}"
    assert abs(speed - expected_speed) <= expected_speed * 0.05, f"生成速度が外れている: {speeds}"


async def test_thinking_delta_is_the_first_token_but_not_the_first_text(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(thinking_response("考え中", answer="答え", gap_s=0.010))

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is None
    assert result.blocks == [
        ContentBlock(type="thinking", text="考え中"),
        ContentBlock(type="text", text="答え"),
    ]
    timing = result.timing
    assert timing.first_token_ns is not None and timing.first_text_ns is not None
    assert timing.first_token_ns < timing.first_text_ns, "thinking が最初のトークンになっていない"
    assert timing.last_token_ns == timing.first_text_ns, "最後の delta は本文の delta のはず"
    assert_timing_is_ordered(result)


# --- イベントの並び、ping、知らないイベント (10.1) -----------------------


async def test_block_order_does_not_change_the_observable_result(fake_server: FakeServer) -> None:
    tool_first = tool_use_response("read_file", {"path": "/tmp/a"}, text_after="あとがき")
    text_first = Script(
        events=(
            *text_events("あとがき", index=0),
            *tool_use_events("read_file", {"path": "/tmp/a"}, index=1),
        ),
        stop_reason="tool_use",
    )

    async with client_for(fake_server) as client:
        fake_server.set_response(tool_first)
        first = await client.stream(make_request(), POLICY)
        fake_server.set_response(text_first)
        second = await client.stream(make_request(), POLICY)

    assert first.error is None and second.error is None
    assert observable(first) == observable(second)
    assert by_key(first.blocks) == [
        ContentBlock(type="text", text="あとがき"),
        ContentBlock(
            type="tool_use",
            tool_name="read_file",
            tool_input_raw='{"path": "/tmp/a"}',
            tool_input={"path": "/tmp/a"},
        ),
    ]


async def test_ping_and_unknown_events_and_fields_do_not_change_the_result(
    fake_server: FakeServer,
) -> None:
    plain = text_response("ok")
    noisy = Script(
        events=(
            ping_event(),
            *unknown_field_events(),
            SseEvent("brand_new_event", {"type": "brand_new_event", "payload": {"a": 1}}),
            ping_event(),
        ),
        stop_reason="end_turn",
    )

    async with client_for(fake_server) as client:
        fake_server.set_response(plain)
        quiet_result = await client.stream(make_request(), POLICY)
        fake_server.set_response(noisy)
        noisy_result = await client.stream(make_request(), POLICY)

    assert quiet_result.error is None and noisy_result.error is None
    assert observable(quiet_result) == observable(noisy_result)
    assert noisy_result.timing.event_count > quiet_result.timing.event_count


async def test_message_start_after_the_first_delta_still_gives_an_ordered_timing(
    fake_server: FakeServer,
) -> None:
    """壊れた並び (delta のあとに message_start) でも、例外にせず不変条件を守る。"""
    fake_server.set_response(
        Script(
            events=(
                *text_events("ok"),
                SseEvent(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_fake",
                            "type": "message",
                            "role": "assistant",
                            "model": "fake-model",
                            "content": [],
                            "usage": {"input_tokens": 11, "output_tokens": 0},
                        },
                    },
                ),
            ),
            include_message_start=False,
        )
    )

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is None
    assert result.blocks == [ContentBlock(type="text", text="ok")]
    assert_timing_is_ordered(result)
    # 速さの計算に使う時刻は動かさず、message_start のほうを引き下げる
    timing = result.timing
    assert timing.first_token_ns is not None
    assert timing.message_start_ns is not None
    assert timing.message_start_ns <= timing.first_token_ns
    assert timing.first_token_ns - timing.sent_at_ns < 1_000_000_000


# --- トークン数 (3.2) ---------------------------------------------------


async def test_usage_falls_back_to_message_start_when_message_delta_is_missing(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(Script(events=text_events("ok"), include_message_delta=False))

    async with client_for(fake_server) as client:
        result = await client.stream(make_request("長めの入力の文章"), POLICY)

    assert result.error is None
    assert result.usage is not None
    recorded = fake_server.requests_for("/v1/messages")[0]
    assert result.usage.input_tokens == recorded.input_tokens
    assert result.usage.output_tokens == 0
    assert result.stop_reason is None


async def test_message_delta_usage_with_only_output_tokens_is_merged(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(
        Script(
            events=(
                *text_events("ok"),
                SseEvent(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
                        "usage": {"output_tokens": 7},
                    },
                ),
            ),
            include_message_delta=False,
            usage=UsageSpec(input_tokens=1234),
        )
    )

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is None
    assert result.usage == Usage(input_tokens=1234, output_tokens=7)
    assert result.stop_reason == "max_tokens"


async def test_cache_fields_feed_the_total_input_tokens(fake_server: FakeServer) -> None:
    fake_server.set_response(
        Script(
            events=text_events("ok"),
            usage=UsageSpec(
                input_tokens=1000,
                output_tokens=5,
                cache_read_input_tokens=600,
                cache_creation_input_tokens=100,
            ),
        )
    )

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is None
    assert result.usage is not None
    assert result.usage.input_tokens == 300  # 全長からキャッシュの内訳を引いた値
    assert result.usage.cache_read_input_tokens == 600
    assert result.usage.cache_creation_input_tokens == 100
    assert result.usage.total_input_tokens == 1000


async def test_usage_is_none_when_the_server_sends_none(fake_server: FakeServer) -> None:
    fake_server.set_response(Script(events=text_events("ok"), usage=None))

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is None
    assert result.usage is None


# --- ツール呼び出しのブロック -------------------------------------------


async def test_tool_use_keeps_the_raw_fragments_when_the_json_is_broken(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(tool_use_response("read_file", fragments=('{"pa', 'th": ')))

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is None
    assert result.blocks == [
        ContentBlock(
            type="tool_use", tool_name="read_file", tool_input_raw='{"path": ', tool_input=None
        )
    ]


async def test_tool_use_without_fragments_is_an_empty_object(fake_server: FakeServer) -> None:
    fake_server.set_response(tool_use_response("noop", fragments=()))

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is None
    assert result.blocks == [
        ContentBlock(type="tool_use", tool_name="noop", tool_input_raw="", tool_input={})
    ]


async def test_tool_use_input_that_is_not_an_object_is_not_parsed(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(tool_use_response("noop", fragments=("[1, 2]",)))

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is None
    assert result.blocks == [
        ContentBlock(type="tool_use", tool_name="noop", tool_input_raw="[1, 2]", tool_input=None)
    ]


# --- 失敗を、例外ではなく値で返す (10.1、10.6) --------------------------


async def test_connect_failure_returns_an_error_value(
    fake_server_factory: Callable[[], FakeServer],
) -> None:
    server = fake_server_factory()
    base_url = server.base_url
    server.stop()  # 誰も待ち受けていない port になる

    async with HttpxMessagesClient(base_url) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is not None
    assert result.error.kind == "connect"
    assert result.error.http_status is None
    assert result.error.message != ""
    assert result.blocks == []
    assert_timing_is_ordered(result)


@pytest.mark.parametrize("status", [400, 500])
async def test_http_error_status_is_reported_with_an_excerpt(
    fake_server: FakeServer, status: int
) -> None:
    fake_server.set_response(http_error_response(status, message="上限を超えている"))

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is not None
    assert result.error.kind == "http"
    assert result.error.http_status == status
    assert "上限を超えている" in result.error.message
    assert len(result.error.message) <= 520
    assert_timing_is_ordered(result)
    assert fake_server.call_count("/v1/messages") == 1, "やり直してはいけない (10.6)"


async def test_mid_stream_error_event_keeps_what_arrived_before_it(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(error_stream_response(before=text_events("途中まで")))

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is not None
    assert result.error.kind == "stream_error"
    assert "fake error" in result.error.message
    assert result.blocks == [ContentBlock(type="text", text="途中まで")]
    assert result.timing.first_token_ns is not None
    assert_timing_is_ordered(result)
    assert fake_server.call_count("/v1/messages") == 1


async def test_connection_drop_mid_stream_is_a_protocol_error(fake_server: FakeServer) -> None:
    fake_server.set_response(dropped_response(after_events=3))

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is not None
    assert result.error.kind == "protocol"
    assert result.timing.message_start_ns is not None
    assert_timing_is_ordered(result)
    assert fake_server.call_count("/v1/messages") == 1


async def test_stream_without_message_stop_is_a_protocol_error(fake_server: FakeServer) -> None:
    fake_server.set_response(Script(events=text_events("ok"), include_message_stop=False))

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is not None
    assert result.error.kind == "protocol"
    assert result.blocks == [ContentBlock(type="text", text="ok")]
    assert result.usage is not None
    assert result.stop_reason == "end_turn"
    assert_timing_is_ordered(result)
    assert fake_server.call_count("/v1/messages") == 1


async def test_malformed_sse_json_is_a_protocol_error(fake_server: FakeServer) -> None:
    fake_server.set_response(
        Script(events=(SseEvent("content_block_delta", raw_data="{壊れている"),))
    )

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is not None
    assert result.error.kind == "protocol"
    assert result.server_model == "fake-model"  # message_start までは読めている
    assert result.timing.message_start_ns is not None
    assert_timing_is_ordered(result)
    assert fake_server.call_count("/v1/messages") == 1


async def test_first_event_timeout_returns_a_value(fake_server: FakeServer) -> None:
    fake_server.set_response(replace(text_response("ok"), first_delay_s=1.0))
    policy = TimeoutPolicy(connect_s=5.0, first_event_s=0.15, idle_s=5.0, total_s=5.0)

    started = time.perf_counter()
    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), policy)
    elapsed = time.perf_counter() - started

    assert result.error is not None
    assert result.error.kind == "timeout_first"
    assert elapsed < 0.9, f"制限時間で打ち切れていない: {elapsed}"
    assert result.timing.event_count == 0
    assert_timing_is_ordered(result)
    assert fake_server.call_count("/v1/messages") == 1


async def test_idle_gap_timeout_returns_a_value_with_partial_data(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(stalled_response(after_events=3, stall_s=1.5))
    policy = TimeoutPolicy(connect_s=5.0, first_event_s=2.0, idle_s=0.15, total_s=5.0)

    started = time.perf_counter()
    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), policy)
    elapsed = time.perf_counter() - started

    assert result.error is not None
    assert result.error.kind == "timeout_idle"
    assert elapsed < 1.2, f"イベントの間隔で打ち切れていない: {elapsed}"
    assert result.timing.first_token_ns is not None
    assert result.blocks == [ContentBlock(type="text", text="tok ")]
    assert_timing_is_ordered(result)
    assert fake_server.call_count("/v1/messages") == 1


async def test_total_timeout_returns_a_value_with_partial_data(fake_server: FakeServer) -> None:
    fake_server.set_response(token_stream_response(output_tokens=30, gap_s=0.020))
    policy = TimeoutPolicy(connect_s=5.0, first_event_s=2.0, idle_s=2.0, total_s=0.25)

    started = time.perf_counter()
    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), policy)
    elapsed = time.perf_counter() - started

    assert result.error is not None
    assert result.error.kind == "timeout_total"
    assert 0.20 <= elapsed < 0.9, f"全体の制限時間で打ち切れていない: {elapsed}"
    assert result.blocks != []
    assert_timing_is_ordered(result)
    assert fake_server.call_count("/v1/messages") == 1


# --- 認証の情報 (1.8) ---------------------------------------------------


async def test_both_auth_headers_are_sent_and_the_secret_never_leaks(
    fake_server: FakeServer,
) -> None:
    fake_server.set_response(text_response("ok"))

    client = client_for(fake_server, SecretStr(SECRET))
    async with client:
        result = await client.stream(make_request(), POLICY)

    headers = fake_server.requests_for("/v1/messages")[0].headers
    assert headers["authorization"] == f"Bearer {SECRET}"
    assert headers["x-api-key"] == SECRET
    assert headers["anthropic-version"] == ANTHROPIC_VERSION
    assert headers["content-type"] == "application/json"

    assert SECRET not in result.model_dump_json()
    assert SECRET not in repr(result)
    assert SECRET not in repr(client)
    assert SECRET not in str(client)


@pytest.mark.parametrize("key", ["model", "max_tokens", "messages", "temperature", "stream"])
def test_extra_cannot_override_a_typed_field(key: str) -> None:
    """同じ条件の試行が同じ設定で送られること (2.7) を、extra が黙って崩さない。"""
    request = make_request(extra={key: "overridden"})

    with pytest.raises(ValueError, match=key):
        build_request_body(request)


async def test_the_secret_does_not_leak_into_the_error_message(fake_server: FakeServer) -> None:
    fake_server.set_response(http_error_response(401, message="unauthorized"))

    client = client_for(fake_server, SecretStr(SECRET))
    async with client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is not None
    assert result.error.kind == "http"
    assert SECRET not in result.error.message
    assert SECRET not in result.model_dump_json()


async def test_a_server_that_echoes_the_auth_header_cannot_leak_the_secret(
    fake_server: FakeServer,
) -> None:
    """対象サーバーが要求のヘッダーをエラーの本文に写しても、生データに落ちない (1.8)。"""
    echoed = f"bad request; headers were: authorization: Bearer {SECRET}, x-api-key: {SECRET}"
    fake_server.set_response(http_error_response(401, message=echoed))

    async with client_for(fake_server, SecretStr(SECRET)) as client:
        http_result = await client.stream(make_request(), POLICY)

    assert http_result.error is not None
    assert "headers were" in http_result.error.message  # 本文は残り、値だけが伏せられる
    assert SECRET not in http_result.model_dump_json()

    fake_server.set_response(error_stream_response(message=f"internal: key={SECRET}"))
    async with client_for(fake_server, SecretStr(SECRET)) as client:
        stream_result = await client.stream(make_request(), POLICY)

    assert stream_result.error is not None
    assert stream_result.error.kind == "stream_error"
    assert SECRET not in stream_result.model_dump_json()


async def test_no_auth_headers_when_no_secret_is_given(fake_server: FakeServer) -> None:
    async with client_for(fake_server) as client:
        await client.stream(make_request(), POLICY)

    headers = fake_server.requests_for("/v1/messages")[0].headers
    assert "authorization" not in headers
    assert "x-api-key" not in headers


# --- 送る本文 -----------------------------------------------------------


def test_build_request_body_omits_none_and_forces_stream() -> None:
    body = build_request_body(make_request(max_tokens=16))

    assert body["model"] == "fake-model"
    assert body["max_tokens"] == 16
    assert body["stream"] is True
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "こんにちは"}]}
    ]
    for omitted in ("system", "tools", "temperature", "top_p", "top_k", "extra"):
        assert omitted not in body, f"{omitted} は None なら送らない"


def test_build_request_body_keeps_the_given_sampling_fields() -> None:
    body = build_request_body(make_request(temperature=0.0, top_p=0.9, system="あなたは助手"))

    assert body["temperature"] == 0.0
    assert body["top_p"] == 0.9
    assert body["system"] == "あなたは助手"


async def test_extra_is_merged_into_the_top_level_of_the_sent_body(
    fake_server: FakeServer,
) -> None:
    request = make_request(
        extra={"chat_template_kwargs": {"enable_thinking": False}, "top_logprobs": 3}
    )
    expected = build_request_body(request)

    async with client_for(fake_server) as client:
        result = await client.stream(request, POLICY)

    assert result.error is None
    assert expected["chat_template_kwargs"] == {"enable_thinking": False}
    assert expected["top_logprobs"] == 3
    assert "extra" not in expected
    assert fake_server.requests_for("/v1/messages")[0].body == expected


# --- 同時の本数 (4.3) ---------------------------------------------------


async def test_eight_concurrent_streams_share_one_client(fake_server: FakeServer) -> None:
    fake_server.set_response(token_stream_response(output_tokens=4, gap_s=0.005))

    async with client_for(fake_server) as client:
        results = await asyncio.gather(
            *(client.stream(make_request(f"要求 {index}"), POLICY) for index in range(8))
        )

    assert all(result.error is None for result in results), [r.error for r in results]
    assert all(result.usage is not None and result.usage.output_tokens == 4 for result in results)
    assert fake_server.call_count("/v1/messages") == 8
    assert max(record.in_flight for record in fake_server.requests_for("/v1/messages")) > 1


# --- 接続の使い回し -----------------------------------------------------


async def test_the_connection_is_reused_across_requests(fake_server: FakeServer) -> None:
    """`message_stop` のあとを読み切って、keep-alive の接続を pool に戻す。

    使い回さないと、最初のトークンまでの時間に TCP の確立が毎回乗る。確かめる
    手立てが httpx の内部しかないので、この試験だけは私的な属性を見る (httpx の
    版は `pyproject.toml` で固定してある)。
    """
    fake_server.set_response(text_response("ok"))

    async with client_for(fake_server) as client:
        for _ in range(3):
            result = await client.stream(make_request(), POLICY)
            assert result.error is None
        pool = client._client._transport._pool  # type: ignore[attr-defined]
        connections = list(pool.connections)

    assert len(connections) == 1, f"接続が使い回されていない: {connections}"
    assert fake_server.call_count("/v1/messages") == 3


async def test_a_server_that_holds_the_stream_open_does_not_delay_the_result(
    fake_server: FakeServer,
) -> None:
    """`message_stop` のあとに閉じないサーバーでも、短い猶予で切り上げる。"""
    # text_response のイベントは 6 つ (message_start から message_stop まで)
    fake_server.set_response(replace(text_response("ok"), stall_after_events=6, stall_s=2.0))

    started = time.perf_counter()
    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)
    elapsed = time.perf_counter() - started

    assert result.error is None
    assert result.blocks == [ContentBlock(type="text", text="ok")]
    assert elapsed < 0.8, f"ストリームの終わりを待ちすぎている: {elapsed}"
