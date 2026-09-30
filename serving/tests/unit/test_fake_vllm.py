"""試験用の偽の推論サーバーそのものの試験 (task 1.5)。

偽物 (`tests/fake_vllm.py`) は、あとのタスク (3.4、3.5、4.1、4.5、4.6、5.2、5.3) が
書き換えずに、台本だけで使うので、ここで振る舞いを固定する。

確かめること:

- 空きポートに立ち、台本どおりに応え、`stop()` のあとはその port につながらない (完了の状態)
- **5 つの道筋 (`/health`、`/v1/models`、`/version`、`/metrics`、`/v1/messages`) のどれでも、
  同じ形の `Fault` で、状態と「応答しない」を切り替えられる** (n 回目から 200、n 回目から
  503 や応答しない)。`/metrics` は、状態を変えずに本文だけ壊すこともできる (200 のまま、
  Prometheus の形でないテキスト)。`/v1/messages` は、状態が 200 以外のとき、既定で Anthropic
  の誤りの形の本文を返す
- `/v1/models` が名乗る名前と `max_model_len` を切り替えられる
- `/version` が版を返し、`None` にすると 404 になる
- `/metrics` が、処理中と待ちの要求の数、生成のトークンの数、投機的デコードの指標の有無を、
  design.md の watch と lifecycle が読む名前で、台本どおりに返す
- `/v1/messages` が、thinking のブロックの有無と長さ、本文、トークンの数、終わりの理由を、
  台本 (固定・列・本文に応じた関数) どおりに返す
- 届いた要求 (道筋、メソッド、ヘッダ、JSON の本文) が記録され、あとから読める
- 実際に HTTP (`httpx`) で叩く
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from fake_vllm import FakeVllm, Fault, MessagesReply, MetricsSample

TIMEOUT = 2.0

# --- 起動・停止 -------------------------------------------------------------


def test_start_serves_on_free_port_and_stop_closes_it() -> None:
    server = FakeVllm()
    server.start()
    url = server.base_url
    try:
        response = httpx.get(f"{url}/health", timeout=TIMEOUT)
        assert response.status_code == 200
    finally:
        server.stop()

    # 止まったあとは、同じ port につながらない (完了の状態)
    with pytest.raises(httpx.TransportError):
        httpx.get(f"{url}/health", timeout=TIMEOUT)


def test_two_servers_use_different_free_ports() -> None:
    a = FakeVllm()
    b = FakeVllm()
    a.start()
    b.start()
    try:
        assert a.base_url != b.base_url
        assert httpx.get(f"{a.base_url}/health", timeout=TIMEOUT).status_code == 200
        assert httpx.get(f"{b.base_url}/health", timeout=TIMEOUT).status_code == 200
    finally:
        a.stop()
        b.stop()


# --- /health -----------------------------------------------------------------


def test_health_default_is_200_with_no_body(fake_vllm: FakeVllm) -> None:
    response = httpx.get(f"{fake_vllm.base_url}/health", timeout=TIMEOUT)
    assert response.status_code == 200
    assert response.content == b""


def test_health_fault_sequence_switches_to_200_after_n_calls(fake_vllm: FakeVllm) -> None:
    fake_vllm.set_health_fault_sequence([Fault(status=503), Fault(status=503), Fault(status=200)])
    statuses = [
        httpx.get(f"{fake_vllm.base_url}/health", timeout=TIMEOUT).status_code for _ in range(4)
    ]
    assert statuses == [503, 503, 200, 200]


def test_health_fault_sequence_stops_responding_after_n_calls(fake_vllm: FakeVllm) -> None:
    fake_vllm.set_health_fault_sequence([Fault(status=200), Fault(status=None)])

    first = httpx.get(f"{fake_vllm.base_url}/health", timeout=TIMEOUT)
    assert first.status_code == 200

    with pytest.raises(httpx.TransportError):
        httpx.get(f"{fake_vllm.base_url}/health", timeout=TIMEOUT)
    with pytest.raises(httpx.TransportError):
        httpx.get(f"{fake_vllm.base_url}/health", timeout=TIMEOUT)

    # 応答しない台本でも、サーバー内部で想定外の例外として記録されていないこと
    assert fake_vllm.errors == []


# --- /v1/models、/version ------------------------------------------------------


def test_models_reports_name_and_max_model_len(fake_vllm: FakeVllm) -> None:
    fake_vllm.set_model("glm-5-3-flash")
    fake_vllm.set_max_model_len(163840)
    payload = httpx.get(f"{fake_vllm.base_url}/v1/models", timeout=TIMEOUT).json()
    assert payload["data"][0]["id"] == "glm-5-3-flash"
    assert payload["data"][0]["max_model_len"] == 163840


def test_models_fault_sequence_returns_503_first_then_200(fake_vllm: FakeVllm) -> None:
    fake_vllm.set_models_fault_sequence([Fault(status=503), Fault(status=200)])
    first = httpx.get(f"{fake_vllm.base_url}/v1/models", timeout=TIMEOUT)
    assert first.status_code == 503
    second = httpx.get(f"{fake_vllm.base_url}/v1/models", timeout=TIMEOUT)
    assert second.status_code == 200
    assert "data" in second.json()


def test_version_default_and_disabled(fake_vllm: FakeVllm) -> None:
    response = httpx.get(f"{fake_vllm.base_url}/version", timeout=TIMEOUT)
    assert response.status_code == 200
    assert response.json() == {"version": "0.0.0-fake"}

    fake_vllm.set_version(None)
    response = httpx.get(f"{fake_vllm.base_url}/version", timeout=TIMEOUT)
    assert response.status_code == 404


def test_version_fault_sequence_returns_503_first_then_200(fake_vllm: FakeVllm) -> None:
    fake_vllm.set_version_fault_sequence([Fault(status=503), Fault(status=200)])
    first = httpx.get(f"{fake_vllm.base_url}/version", timeout=TIMEOUT)
    assert first.status_code == 503
    second = httpx.get(f"{fake_vllm.base_url}/version", timeout=TIMEOUT)
    assert second.status_code == 200


# --- /metrics ------------------------------------------------------------------


def test_metrics_reports_running_and_waiting_requests(fake_vllm: FakeVllm) -> None:
    fake_vllm.set_model("glm-5-3-flash")
    fake_vllm.set_metrics(MetricsSample(running_requests=2, waiting_requests=3))
    text = httpx.get(f"{fake_vllm.base_url}/metrics", timeout=TIMEOUT).text
    assert 'vllm:num_requests_running{model_name="glm-5-3-flash"} 2.0' in text
    assert 'vllm:num_requests_waiting{model_name="glm-5-3-flash"} 3.0' in text
    assert "vllm:spec_decode_" not in text


def test_metrics_spec_decode_lines_appear_only_when_enabled(fake_vllm: FakeVllm) -> None:
    fake_vllm.set_metrics(MetricsSample(spec_decode=True))
    text = httpx.get(f"{fake_vllm.base_url}/metrics", timeout=TIMEOUT).text
    assert "vllm:spec_decode_num_drafts_total" in text
    assert "vllm:spec_decode_num_draft_tokens_total" in text
    assert "vllm:spec_decode_num_accepted_tokens_total" in text


def test_metrics_sequence_can_stop_increasing(fake_vllm: FakeVllm) -> None:
    """「生成のトークンの数がいつ止まるか」を、値を繰り返す列で表せることを確かめる (4.5)。"""
    fake_vllm.set_metrics_sequence(
        [
            MetricsSample(generation_tokens_total=10),
            MetricsSample(generation_tokens_total=20),
            MetricsSample(generation_tokens_total=20),
            MetricsSample(generation_tokens_total=20),
        ]
    )

    def read_tokens() -> float:
        text = httpx.get(f"{fake_vllm.base_url}/metrics", timeout=TIMEOUT).text
        for line in text.splitlines():
            if line.startswith("vllm:generation_tokens_total{"):
                return float(line.rsplit(" ", 1)[1])
        raise AssertionError("vllm:generation_tokens_total が見つからない")

    values = [read_tokens() for _ in range(5)]
    assert values == [10.0, 20.0, 20.0, 20.0, 20.0]


def test_metrics_fault_sequence_fails_with_503_n_times_then_succeeds(fake_vllm: FakeVllm) -> None:
    """design.md の lifecycle が言う「指標が読めない」の分岐 (3.4 が使う)。"""
    fake_vllm.set_metrics_fault_sequence([Fault(status=503), Fault(status=503), Fault(status=200)])
    statuses = [
        httpx.get(f"{fake_vllm.base_url}/metrics", timeout=TIMEOUT).status_code for _ in range(4)
    ]
    assert statuses == [503, 503, 200, 200]


def test_metrics_fault_sequence_stops_responding_n_times_then_succeeds(
    fake_vllm: FakeVllm,
) -> None:
    fake_vllm.set_metrics_fault_sequence(
        [Fault(status=None), Fault(status=None), Fault(status=200)]
    )
    with pytest.raises(httpx.TransportError):
        httpx.get(f"{fake_vllm.base_url}/metrics", timeout=TIMEOUT)
    with pytest.raises(httpx.TransportError):
        httpx.get(f"{fake_vllm.base_url}/metrics", timeout=TIMEOUT)
    response = httpx.get(f"{fake_vllm.base_url}/metrics", timeout=TIMEOUT)
    assert response.status_code == 200
    assert "vllm:num_requests_running" in response.text


def test_metrics_fault_sequence_returns_malformed_body_n_times_then_valid(
    fake_vllm: FakeVllm,
) -> None:
    """状態を 200 のまま、本文だけを Prometheus の形でないテキストに壊せることを確かめる。"""
    fake_vllm.set_metrics_fault_sequence(
        [Fault(status=200, body="not prometheus text"), Fault(status=200)]
    )
    broken = httpx.get(f"{fake_vllm.base_url}/metrics", timeout=TIMEOUT)
    assert broken.status_code == 200
    assert broken.text == "not prometheus text"
    valid = httpx.get(f"{fake_vllm.base_url}/metrics", timeout=TIMEOUT)
    assert valid.status_code == 200
    assert "vllm:num_requests_running" in valid.text


# --- /v1/messages -----------------------------------------------------------------


def test_messages_returns_thinking_and_text_blocks(fake_vllm: FakeVllm) -> None:
    fake_vllm.set_model("glm-5-3-flash")
    fake_vllm.set_messages_reply(
        MessagesReply(
            thinking="考え中" * 5,
            text="答え",
            input_tokens=42,
            output_tokens=7,
            stop_reason="end_turn",
        )
    )
    payload = httpx.post(
        f"{fake_vllm.base_url}/v1/messages", json={"model": "glm-5-3-flash"}, timeout=TIMEOUT
    ).json()
    assert payload["content"][0] == {"type": "thinking", "thinking": "考え中" * 5}
    assert payload["content"][1] == {"type": "text", "text": "答え"}
    assert payload["usage"] == {"input_tokens": 42, "output_tokens": 7}
    assert payload["stop_reason"] == "end_turn"
    assert payload["model"] == "glm-5-3-flash"


def test_messages_reply_without_thinking_omits_the_block(fake_vllm: FakeVllm) -> None:
    fake_vllm.set_messages_reply(MessagesReply(thinking=None, text="ok"))
    payload = httpx.post(f"{fake_vllm.base_url}/v1/messages", json={}, timeout=TIMEOUT).json()
    assert payload["content"] == [{"type": "text", "text": "ok"}]


def test_messages_sequence_returns_different_replies_in_order(fake_vllm: FakeVllm) -> None:
    fake_vllm.set_messages_sequence([MessagesReply(text="1回目"), MessagesReply(text="2回目")])
    texts = []
    for _ in range(3):
        payload = httpx.post(f"{fake_vllm.base_url}/v1/messages", json={}, timeout=TIMEOUT).json()
        texts.append(payload["content"][0]["text"])
    assert texts == ["1回目", "2回目", "2回目"]


def test_messages_factory_chooses_reply_from_request_body(fake_vllm: FakeVllm) -> None:
    """thinking の深さの確かめ (4.6) が使う、本文に応じた応答の選び方。"""

    def factory(body: dict[str, Any] | None) -> MessagesReply:
        assert body is not None
        effort = body.get("output_config", {}).get("effort")
        thinking = "深い思考" if effort == "low" else "浅い思考" * 10
        return MessagesReply(thinking=thinking)

    fake_vllm.set_messages_factory(factory)

    low = httpx.post(
        f"{fake_vllm.base_url}/v1/messages",
        json={"output_config": {"effort": "low"}},
        timeout=TIMEOUT,
    ).json()
    default = httpx.post(f"{fake_vllm.base_url}/v1/messages", json={}, timeout=TIMEOUT).json()
    assert low["content"][0]["thinking"] == "深い思考"
    assert default["content"][0]["thinking"] == "浅い思考" * 10


def test_messages_fault_returns_500_with_anthropic_error_body(fake_vllm: FakeVllm) -> None:
    """3.5 の短い要求と 4.1 の縮小の確認が、500 を受けたときの扱いを試験できること。"""
    fake_vllm.set_messages_fault(Fault(status=500))
    response = httpx.post(f"{fake_vllm.base_url}/v1/messages", json={}, timeout=TIMEOUT)
    assert response.status_code == 500
    payload = response.json()
    assert payload["type"] == "error"
    assert "message" in payload["error"]


def test_messages_fault_with_custom_body_returns_it_verbatim(fake_vllm: FakeVllm) -> None:
    fake_vllm.set_messages_fault(Fault(status=400, body='{"type": "error", "error": {}}'))
    response = httpx.post(f"{fake_vllm.base_url}/v1/messages", json={}, timeout=TIMEOUT)
    assert response.status_code == 400
    assert response.json() == {"type": "error", "error": {}}


def test_messages_fault_stops_responding(fake_vllm: FakeVllm) -> None:
    fake_vllm.set_messages_fault(Fault(status=None))
    with pytest.raises(httpx.TransportError):
        httpx.post(f"{fake_vllm.base_url}/v1/messages", json={}, timeout=TIMEOUT)


# --- 記録 ---------------------------------------------------------------------------


def test_requests_are_recorded_in_order_with_path_method_headers_and_body(
    fake_vllm: FakeVllm,
) -> None:
    httpx.get(f"{fake_vllm.base_url}/health", timeout=TIMEOUT)
    httpx.get(f"{fake_vllm.base_url}/v1/models", timeout=TIMEOUT)
    httpx.post(
        f"{fake_vllm.base_url}/v1/messages",
        json={"model": "glm-5-3-flash", "messages": []},
        headers={"anthropic-version": "2023-06-01"},
        timeout=TIMEOUT,
    )

    paths = [call.path for call in fake_vllm.requests]
    assert paths == ["/health", "/v1/models", "/v1/messages"]

    messages_call = fake_vllm.requests_for("/v1/messages")[0]
    assert messages_call.method == "POST"
    assert messages_call.body == {"model": "glm-5-3-flash", "messages": []}
    assert messages_call.headers["anthropic-version"] == "2023-06-01"

    assert fake_vllm.call_count("/health") == 1
    assert fake_vllm.call_count("/v1/models") == 1


def test_requests_are_recorded_even_when_a_fault_is_returned(fake_vllm: FakeVllm) -> None:
    """壊れた応答を返したときも、要求そのものは受け取っているので記録される。"""
    fake_vllm.set_metrics_fault(Fault(status=503))
    httpx.get(f"{fake_vllm.base_url}/metrics", timeout=TIMEOUT)
    assert fake_vllm.call_count("/metrics") == 1


def test_six_thinking_request_shapes_are_recorded_with_only_expected_fields(
    fake_vllm: FakeVllm,
) -> None:
    """4.6 が送る 6 通りの要求の形が、そのまま記録から読めることを確かめる (期待した項目だけ。
    issue #131 で `chat_template_off` (6 番目) を足した)。"""
    shapes: list[dict[str, Any]] = [
        {"model": "glm-5-3-flash", "messages": [], "temperature": 0},
        {
            "model": "glm-5-3-flash",
            "messages": [],
            "temperature": 0,
            "output_config": {"effort": "low"},
        },
        {
            "model": "glm-5-3-flash",
            "messages": [],
            "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        },
        {
            "model": "glm-5-3-flash",
            "messages": [],
            "temperature": 0,
            "output_config": {"effort": "medium"},
        },
        {
            "model": "glm-5-3-flash",
            "messages": [],
            "temperature": 0,
            "chat_template_kwargs": {"clear_thinking": True},
        },
        {
            "model": "glm-5-3-flash",
            "messages": [],
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ]
    for shape in shapes:
        httpx.post(f"{fake_vllm.base_url}/v1/messages", json=shape, timeout=TIMEOUT)

    recorded = fake_vllm.requests_for("/v1/messages")
    assert len(recorded) == 6
    for shape, call in zip(shapes, recorded, strict=True):
        assert call.body == shape
        assert set(call.body or {}) == set(shape)
