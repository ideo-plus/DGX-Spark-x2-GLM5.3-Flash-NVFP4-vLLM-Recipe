"""計測の前提を確かめる部品の試験 (task 2.7)。

実際のソケット越しに、偽のサーバーを相手に確かめる。時刻を判定する試験は
1 つだけ (実行中の要求の数を読む試験) で、tasks.md の Implementation Notes
(1.4) の決まりに従う: 個々の値を狭い幅で判定せず、値があるかどうかだけを見る。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from bench_harness.client import probe
from bench_harness.client.messages import HttpxMessagesClient
from bench_harness.client.probe import (
    InputOverContextLimitError,
    ProbeError,
    count_input_tokens,
    fits_context,
    is_context_limit_error,
    preflight,
)
from bench_harness.types import (
    InputMessage,
    MessagesRequest,
    PreflightFailure,
    PreflightOk,
    TargetDef,
    TextBlockParam,
    TimeoutPolicy,
)
from fake_server import (
    FakeServer,
    Script,
    UsageSpec,
    empty_response,
    http_error_response,
    replace,
    stalled_response,
    text_events,
    text_response,
    thinking_response,
)

# --- 助け -----------------------------------------------------------------

POLICY = TimeoutPolicy(connect_s=5.0, first_event_s=5.0, idle_s=5.0, total_s=20.0)
SECRET = "sk-fake-super-secret-value"


def make_request(text: str = "こんにちは", *, max_tokens: int = 64) -> MessagesRequest:
    return MessagesRequest(
        model="fake-model",
        max_tokens=max_tokens,
        messages=[InputMessage(role="user", content=[TextBlockParam(text=text)])],
    )


def client_for(server: FakeServer, api_key: SecretStr | None = None) -> HttpxMessagesClient:
    return HttpxMessagesClient(server.base_url, api_key)


def target_for(server: FakeServer, **overrides: Any) -> TargetDef:
    payload: dict[str, Any] = {"name": "fake", "base_url": server.base_url, "model": "fake-model"}
    payload.update(overrides)
    return TargetDef.model_validate(payload)


# --- 成功の道 (すべての項目が揃う) ----------------------------------------


async def test_preflight_ok_fills_all_four_fields(fake_server: FakeServer) -> None:
    fake_server.set_model("glm-5.3-flash")
    fake_server.set_version("1.2.3")
    fake_server.set_context_limit(32000)
    target = target_for(fake_server, model="glm-5.3-flash")

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightOk)
    assert result.server_model == "glm-5.3-flash"
    assert result.server_version == "1.2.3"
    assert result.context_limit == 32000
    assert result.running_requests == 0


# --- 上限の決まり方 (design.md client/probe) -------------------------------


async def test_context_limit_comes_from_v1_models_when_advertised(fake_server: FakeServer) -> None:
    fake_server.set_context_limit(32000)
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightOk)
    assert result.context_limit == 32000


async def test_context_limit_falls_back_to_target_def_when_not_advertised(
    fake_server: FakeServer,
) -> None:
    fake_server.set_context_limit(32000, advertise=False)
    target = target_for(fake_server, max_context_tokens=9000)

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightOk)
    assert result.context_limit == 9000


async def test_context_limit_is_none_when_neither_is_available(fake_server: FakeServer) -> None:
    target = target_for(fake_server)  # 上限を申告しない偽のサーバー、定義にも値なし

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightOk)
    assert result.context_limit is None


async def test_context_limit_prefers_the_server_over_the_target_def(
    fake_server: FakeServer,
) -> None:
    """優先順位の固定 (レビュー 2): 定義の値を先に見る変異を、この試験が検出する。"""
    fake_server.set_context_limit(32000, advertise=True)
    target = target_for(fake_server, max_context_tokens=5000)  # 32000 と違う値

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightOk)
    assert result.context_limit == 32000


# --- 範囲の外の補助の値は採用しない (レビュー BLOCKING) ----------------------


async def test_zero_context_limit_from_v1_models_falls_back_to_the_target_def(
    fake_server: FakeServer,
) -> None:
    """`max_model_len` が 0 (範囲の外) なら、対象サーバーの定義の値に進む。

    enforce=False にして、確認そのものの短い要求まで 400 にならないようにする。
    """
    fake_server.set_context_limit(0, advertise=True, enforce=False)
    target = target_for(fake_server, max_context_tokens=4096)

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightOk)
    assert result.context_limit == 4096


async def test_negative_context_limit_from_v1_models_is_ignored(fake_server: FakeServer) -> None:
    """`max_model_len` が負なら、対象サーバーの定義にも値がなければ `None` になる。"""
    fake_server.set_context_limit(-5, advertise=True, enforce=False)
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightOk)
    assert result.context_limit is None


@pytest.mark.parametrize("raw_value", ["-3", "NaN", "+Inf", "-Inf"])
async def test_out_of_range_running_requests_gauge_is_ignored(
    fake_server: FakeServer, raw_value: str
) -> None:
    """実行中の要求の数が、負や NaN、無限大なら `None` になる (`ge=0` の検証に落ちない)。"""
    fake_server.set_metrics_text(
        "# HELP vllm:num_requests_running Number of requests running\n"
        "# TYPE vllm:num_requests_running gauge\n"
        f"vllm:num_requests_running {raw_value}\n"
    )
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightOk)
    assert result.running_requests is None


def test_select_model_entry_matches_by_id() -> None:
    entries = [{"id": "a", "max_model_len": 1}, {"id": "b", "max_model_len": 2}]
    assert probe._select_model_entry(entries, "b") == {"id": "b", "max_model_len": 2}


def test_select_model_entry_falls_back_to_the_only_entry() -> None:
    entries = [{"id": "a", "max_model_len": 1}]
    assert probe._select_model_entry(entries, "does-not-exist") == entries[0]


def test_select_model_entry_is_none_when_ambiguous() -> None:
    entries = [{"id": "a"}, {"id": "b"}]
    assert probe._select_model_entry(entries, "c") is None


def test_select_model_entry_is_none_when_empty() -> None:
    assert probe._select_model_entry([], "a") is None


# --- version と metrics は、失敗しても前提の不足にしない --------------------


async def test_missing_version_endpoint_is_still_a_preflight_ok(fake_server: FakeServer) -> None:
    fake_server.set_version(None)
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightOk)
    assert result.server_version is None


async def test_metrics_404_still_a_preflight_ok_with_running_requests_none(
    fake_server: FakeServer,
) -> None:
    fake_server.set_metrics_enabled(False)
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightOk)
    assert result.running_requests is None


async def test_running_requests_reflects_an_in_flight_stream(fake_server: FakeServer) -> None:
    """始める前に実行中の要求があれば、その本数が読める (design.md 計測ランの進行)。

    `preflight` 自身の短い要求は速く終わるので、内部の gauge の読み取り部品
    (`_fetch_running_requests`) を直接呼び、そのあいだ別の要求を止めておく。
    """
    fake_server.set_response(stalled_response(after_events=1, stall_s=2.0))
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        task = asyncio.create_task(client.stream(make_request(), POLICY))
        try:
            # wait_for_requests は同期のブロッキング呼び出し。await せず直に呼ぶと
            # イベントループそのものを止めてしまい、上の task が一度も進めない
            await asyncio.to_thread(fake_server.wait_for_requests, 1, path="/v1/messages")
            running = await probe._fetch_running_requests(target, 2.0)
        finally:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task

    assert running is not None
    assert running > 0


# --- 満たされていない前提 (1.4) ---------------------------------------------


async def test_unreachable_when_the_server_is_unreachable(
    fake_server_factory: Callable[[], FakeServer],
) -> None:
    server = fake_server_factory()
    base_url = server.base_url
    target = target_for(server)
    server.stop()  # 誰も待ち受けていない port になる

    async with HttpxMessagesClient(base_url) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightFailure)
    assert result.unmet == "unreachable"
    assert result.detail != ""


@pytest.mark.parametrize("status", [401, 500])
async def test_http_error_status_is_reported_with_the_status_in_detail(
    fake_server: FakeServer, status: int
) -> None:
    fake_server.set_response(http_error_response(status, message="上限を超えている"))
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightFailure)
    assert result.unmet == "http_error"
    assert str(status) in result.detail
    assert "上限を超えている" in result.detail


async def test_no_usage_when_the_server_sends_no_token_count(fake_server: FakeServer) -> None:
    fake_server.set_response(Script(events=text_events("ok"), usage=None))
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightFailure)
    assert result.unmet == "no_usage"
    assert "トークン数" in result.detail


async def test_no_output_when_the_response_is_empty(fake_server: FakeServer) -> None:
    fake_server.set_response(empty_response())
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightFailure)
    assert result.unmet == "no_output"


async def test_thinking_only_response_is_still_a_preflight_ok(fake_server: FakeServer) -> None:
    """thinking だけの応答 (本文もツール呼び出しもない) も、出力とみなす (レビュー 5)。

    2.1 が thinking の最初のトークンを最初のトークンに数えるのと同じ読み方。
    """
    fake_server.set_response(thinking_response("考え中", answer=None))
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightOk)


async def test_timeout_first_maps_to_unreachable(fake_server: FakeServer) -> None:
    fake_server.set_response(replace(text_response("ok"), first_delay_s=1.0))
    policy = TimeoutPolicy(connect_s=5.0, first_event_s=0.15, idle_s=5.0, total_s=5.0)
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        result = await preflight(client, target, timeout=policy)

    assert isinstance(result, PreflightFailure)
    assert result.unmet == "unreachable"


async def test_protocol_error_maps_to_http_error(fake_server: FakeServer) -> None:
    fake_server.set_response(Script(events=text_events("ok"), include_message_stop=False))
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        result = await preflight(client, target)

    assert isinstance(result, PreflightFailure)
    assert result.unmet == "http_error"
    assert "protocol" in result.detail


async def test_preflight_sends_exactly_one_messages_request(fake_server: FakeServer) -> None:
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        await preflight(client, target)

    assert fake_server.call_count("/v1/messages") == 1


# --- 認証の情報 (1.8) -------------------------------------------------------


async def test_api_key_sentinel_is_sent_on_auxiliary_gets_and_never_returned(
    fake_server: FakeServer,
) -> None:
    target = target_for(fake_server)

    async with client_for(fake_server, SecretStr(SECRET)) as client:
        result = await preflight(client, target, api_key=SecretStr(SECRET))

    assert isinstance(result, PreflightOk)
    for path in ("/v1/models", "/version"):
        headers = fake_server.requests_for(path)[0].headers
        assert headers["authorization"] == f"Bearer {SECRET}"
        assert headers["x-api-key"] == SECRET

    assert SECRET not in result.model_dump_json()
    assert SECRET not in repr(result)


async def test_no_auth_headers_on_auxiliary_gets_when_no_key_is_given(
    fake_server: FakeServer,
) -> None:
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        await preflight(client, target)

    for path in ("/v1/models", "/version"):
        headers = fake_server.requests_for(path)[0].headers
        assert "authorization" not in headers
        assert "x-api-key" not in headers


# --- トークン数を数える (calibrate (5.1) が呼ぶ) ---------------------------


TOKEN_TEXT = "同じ文章を、口がある場合とない場合の両方で数える。tokens tokens tokens."


async def test_count_input_tokens_via_the_endpoint(fake_server: FakeServer) -> None:
    target = target_for(fake_server)
    request = make_request(TOKEN_TEXT)

    async with client_for(fake_server) as client:
        result = await count_input_tokens(client, target, request)

    assert result.method == "count_tokens"
    assert result.tokens > 0
    assert fake_server.call_count("/v1/messages") == 0, "count_tokens は要求そのものを送らない"


async def test_count_input_tokens_falls_back_when_the_endpoint_is_missing(
    fake_server: FakeServer,
) -> None:
    fake_server.set_count_tokens_enabled(False)
    target = target_for(fake_server)
    request = make_request(TOKEN_TEXT)

    async with client_for(fake_server) as client:
        result = await count_input_tokens(client, target, request)

    assert result.method == "one_token_request"
    assert result.tokens > 0
    assert fake_server.call_count("/v1/messages") == 1


async def test_count_input_tokens_agrees_between_the_two_methods(fake_server: FakeServer) -> None:
    target = target_for(fake_server)
    request = make_request(TOKEN_TEXT)

    async with client_for(fake_server) as client:
        via_endpoint = await count_input_tokens(client, target, request)
        fake_server.set_count_tokens_enabled(False)
        via_fallback = await count_input_tokens(client, target, request)

    assert via_endpoint.tokens == via_fallback.tokens


async def test_count_input_tokens_raises_probe_error_when_both_fail(
    fake_server: FakeServer,
) -> None:
    fake_server.set_count_tokens_enabled(False)
    fake_server.set_response(http_error_response(500))
    target = target_for(fake_server)
    request = make_request(TOKEN_TEXT)

    async with client_for(fake_server) as client:
        with pytest.raises(ProbeError) as excinfo:
            await count_input_tokens(client, target, request)

    # 上限の超過ではない失敗 (500) は、区別された型にならない (issue #24)
    assert not isinstance(excinfo.value, InputOverContextLimitError)


async def test_count_input_tokens_distinguishes_a_fallback_request_over_the_context_limit(
    fake_server: FakeServer,
) -> None:
    """口がなく、代わりの要求が上限を超えて HTTP 400 になったときだけ区別された型になる (#24)。

    `ProbeError` として受け取る側 (計測ランの進行、`calibrate`、同時処理の包みの計測) は、
    今までどおり `ProbeError` として扱える (派生型)。
    """
    fake_server.set_count_tokens_enabled(False)
    fake_server.set_context_limit(1)  # 入力 + max_tokens=1 が 1 を超える。偽のサーバーが 400 を返す
    target = target_for(fake_server)
    request = make_request(TOKEN_TEXT)

    async with client_for(fake_server) as client:
        with pytest.raises(ProbeError) as excinfo:
            await count_input_tokens(client, target, request)

    assert isinstance(excinfo.value, InputOverContextLimitError)
    assert fake_server.call_count("/v1/messages") == 1, "代わりの要求を 1 件だけ送った"


async def test_count_input_tokens_fallback_uses_the_total_input_tokens(
    fake_server: FakeServer,
) -> None:
    """代わりの数え方は、内訳を含む全長 (`usage.total_input_tokens`) を使う (レビュー 3)。

    `usage.input_tokens` だけに差し替える変異を、この試験が検出する。3 つの量
    (300、600、100) はすべて違う値にしてある。
    """
    fake_server.set_count_tokens_enabled(False)
    fake_server.set_response(
        Script(
            events=text_events("ok"),
            usage=UsageSpec(
                input_tokens=1000,  # 全長。表示される input_tokens は 1000-600-100=300
                output_tokens=1,
                cache_read_input_tokens=600,
                cache_creation_input_tokens=100,
            ),
        )
    )
    target = target_for(fake_server)
    request = make_request(TOKEN_TEXT)

    async with client_for(fake_server) as client:
        result = await count_input_tokens(client, target, request)

    assert result.method == "one_token_request"
    assert result.tokens == 1000


async def test_count_tokens_body_omits_generation_only_fields(fake_server: FakeServer) -> None:
    """count_tokens に送る本文は、公式の項目だけに絞る (レビュー 4)。

    `max_tokens`、`temperature`、`stream` は生成だけに効く項目で、厳密な
    サーバーには拒まれうるので落とす。
    """
    target = target_for(fake_server)
    request = MessagesRequest(
        model="fake-model",
        max_tokens=123,
        messages=[InputMessage(role="user", content=[TextBlockParam(text=TOKEN_TEXT)])],
        temperature=0.7,
        top_p=0.9,
        top_k=40,
    )

    async with client_for(fake_server) as client:
        await count_input_tokens(client, target, request)

    recorded = fake_server.requests_for("/v1/messages/count_tokens")[0]
    assert recorded.body is not None
    for omitted in ("max_tokens", "temperature", "top_p", "top_k", "stream"):
        assert omitted not in recorded.body, f"{omitted} は count_tokens に送らない"
    assert recorded.body["model"] == "fake-model"


async def test_count_tokens_keeps_extra_fields_that_can_affect_tokenization(
    fake_server: FakeServer,
) -> None:
    """`extra` (`chat_template_kwargs` など) は、トークン数に効きうるので残す。"""
    target = target_for(fake_server)
    request = MessagesRequest(
        model="fake-model",
        max_tokens=16,
        messages=[InputMessage(role="user", content=[TextBlockParam(text=TOKEN_TEXT)])],
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    )

    async with client_for(fake_server) as client:
        await count_input_tokens(client, target, request)

    recorded = fake_server.requests_for("/v1/messages/count_tokens")[0]
    assert recorded.body is not None
    assert recorded.body["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("status", [400, 401, 500])
async def test_count_tokens_non_missing_http_status_raises_probe_error_without_fallback(
    fake_server: FakeServer, status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """404/405/501 以外の失敗は、黙って代わりの数え方に落とさず `ProbeError` にする (レビュー 4)。

    偽のサーバーの `count_tokens` は 200 か 404 しか返せない (fake_server.py の
    契約) ので、この場面だけ `httpx.AsyncClient.post` を差し替えて、それ以外の
    状態を作る。認証の値はこの応答に含めていないので、漏れないことも確かめる。
    """

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "count_tokens が拒まれた"}})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    target = target_for(fake_server)
    request = make_request(TOKEN_TEXT)

    async with client_for(fake_server) as client:
        with pytest.raises(ProbeError, match=str(status)) as excinfo:
            await count_input_tokens(client, target, request)

    assert SECRET not in str(excinfo.value)
    assert fake_server.call_count("/v1/messages") == 0, "代わりの要求を送ってはいけない"
    # 口そのものの HTTP 400 は、設定の誤りとして素の ProbeError のまま (issue #24)
    assert not isinstance(excinfo.value, InputOverContextLimitError)


@pytest.mark.parametrize("payload", [{"input_tokens": -1}, {"input_tokens": -5}])
async def test_count_tokens_negative_value_raises_probe_error(
    fake_server: FakeServer, payload: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """200 の応答でも、負のトークン数は受け取らない (生の検証エラーを外に出さない)。"""

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    target = target_for(fake_server)

    async with client_for(fake_server) as client:
        with pytest.raises(ProbeError, match="input_tokens"):
            await count_input_tokens(client, target, make_request(TOKEN_TEXT))


# --- 上限に収まるかどうか (3.1、3.6、6.9) -----------------------------------


@pytest.mark.parametrize(
    ("context_limit", "input_tokens", "max_tokens", "expected"),
    [
        (None, 100, 50, None),
        (200, 100, 50, True),
        (150, 100, 50, True),  # ちょうど収まる境目
        (149, 100, 50, False),
    ],
)
def test_fits_context_truth_table(
    context_limit: int | None, input_tokens: int, max_tokens: int, expected: bool | None
) -> None:
    assert fits_context(context_limit, input_tokens, max_tokens) is expected


async def test_is_context_limit_error_true_for_the_fake_servers_enforced_limit(
    fake_server: FakeServer,
) -> None:
    fake_server.set_context_limit(1)

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is not None
    assert result.error.http_status == 400
    assert is_context_limit_error(result)


@pytest.mark.parametrize("status", [401, 500])
async def test_is_context_limit_error_false_for_other_http_errors(
    fake_server: FakeServer, status: int
) -> None:
    fake_server.set_response(http_error_response(status))

    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert not is_context_limit_error(result)


async def test_is_context_limit_error_false_for_success(fake_server: FakeServer) -> None:
    async with client_for(fake_server) as client:
        result = await client.stream(make_request(), POLICY)

    assert result.error is None
    assert not is_context_limit_error(result)
