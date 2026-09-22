"""thinking の深さの確かめの試験 (tasks.md 4.6)。

確かめること (design.md 「確認 › thinking」、tasks.md 4.6 の完了の状態、requirements 9.2、10.5):

- 5 通り (`none`、`output_config_low`、`chat_template_low`、`output_config_medium`、
  `clear_thinking`) の要求の本文が、それぞれ「期待した項目だけ」を持つ (共通の 5 項目 + 通り
  ごとに 0 か 1 個の追加の項目)
- 5 通りを、回ごとに 1〜5 の順で送る (`trials` 回ぶん)
- 5 通りとも、同じ合成の会話 (`messages`) を送る。呼ぶ側は `messages=` で差し替えられる
- 記録するのは thinking の文字数、`input_tokens`/`output_tokens`、`stop_reason`、HTTP の状態
  だけで、応答の本文・thinking の文・誤りの応答の本文は、結果のファイルにも `detail` にも
  `report` にも入らない
- thinking のブロックが無い 200 の応答は `thinking_chars = 0`
- 1 回の HTTP の失敗 (2xx 以外、`usage` が丸ごと無い) は、値の欠けた `ThinkingTrial` になり、
  残りの回は続く
- `effective`: 2 対 1 か 3 対 1 のどちらかで範囲が重ならなければ `True`、両方判定できて両方
  重なれば `False`、それ以外 (値の欠け、`max_tokens` による打ち切りで判定できない) は `None`
- `max_tokens` で打ち切られた回が、範囲が低い側のグループにあると、その比べは判定できない
- 引数の確かめ (`trials`/`max_tokens`/`timeout_s`/`base_url`/`model`/`messages`) は、HTTP の
  要求を 1 つも出す前に行う
- `KeyboardInterrupt` は、1 回以上終わっていればそこまでの要約を書いてから、そのまま上げ直す

実物の推論サーバーには、どの段でもつながない (`fake_vllm.FakeVllm` を HTTP で叩くだけ)。試験は
実際の時間を待たない。
"""

from __future__ import annotations

import io
import json
import math
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from fake_vllm import FakeVllm, Fault, MessagesReply
from serving_kit import thinking as th

MODEL = "thinking-test-model"

_COMMON_KEYS = {"model", "max_tokens", "temperature", "messages", "stream"}
_EXPECTED_KEYS = {
    "none": _COMMON_KEYS,
    "output_config_low": _COMMON_KEYS | {"output_config"},
    "chat_template_low": _COMMON_KEYS | {"chat_template_kwargs"},
    "output_config_medium": _COMMON_KEYS | {"output_config"},
    "clear_thinking": _COMMON_KEYS | {"chat_template_kwargs"},
}
_VARIANT_ORDER = (
    "none",
    "output_config_low",
    "chat_template_low",
    "output_config_medium",
    "clear_thinking",
)


def variant_of(body: dict[str, Any] | None) -> str:
    """記録した要求の本文から、5 通りのどれかを見分ける (試験専用。実装の内部は見ない)。"""
    if body is None:
        return "unknown"
    if "output_config" in body:
        effort = body["output_config"].get("effort")
        return "output_config_low" if effort == "low" else "output_config_medium"
    if "chat_template_kwargs" in body:
        ctk = body["chat_template_kwargs"]
        if ctk.get("reasoning_effort") == "low":
            return "chat_template_low"
        if ctk.get("clear_thinking") is True:
            return "clear_thinking"
    return "none"


def sequenced_factory(
    table: dict[str, list[MessagesReply]],
) -> Callable[[dict[str, Any] | None], MessagesReply]:
    """通りごとに、呼ばれた順で `table` の値を返す (尽きたら最後を繰り返す)。"""
    counters: dict[str, int] = defaultdict(int)

    def factory(body: dict[str, Any] | None) -> MessagesReply:
        variant = variant_of(body)
        replies = table[variant]
        index = min(counters[variant], len(replies) - 1)
        counters[variant] += 1
        return replies[index]

    return factory


class _InterruptingTransport(httpx.BaseTransport):
    """`fail_at` 回目の要求で `KeyboardInterrupt` を投げ、それより前は実物の `HTTPTransport`
    に渡す (`test_watch.py` の見本と同じ形)。
    """

    def __init__(self, *, fail_at: int) -> None:
        self._inner = httpx.HTTPTransport()
        self._count = 0
        self._fail_at = fail_at

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._count += 1
        if self._count >= self._fail_at:
            raise KeyboardInterrupt
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


# --- 要求の本文 ----------------------------------------------------------------


def test_sends_the_expected_number_of_requests_with_exact_body_keys(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """5 通り x `trials` 回ぶんの要求が届き、本文が「期待した項目だけ」を持つ (完了の状態)。"""
    outcome = th.run_thinking(
        base_url=fake_vllm.base_url,
        model=MODEL,
        var_root=tmp_path / "var",
        report=io.StringIO(),
    )

    sent = fake_vllm.requests_for("/v1/messages")
    assert len(sent) == 5 * th.DEFAULT_TRIALS
    assert len(outcome.trials) == 5 * th.DEFAULT_TRIALS
    for request in sent:
        body = request.body
        assert body is not None, "要求の本文が JSON として届いていない"
        variant = variant_of(body)
        assert set(body.keys()) == _EXPECTED_KEYS[variant], f"{variant}: {sorted(body.keys())}"
        assert body["model"] == MODEL
        assert body["max_tokens"] == th.DEFAULT_MAX_TOKENS
        assert body["temperature"] == 0
        assert body["stream"] is False
    # seed、thinking、enable_thinking はどの本文にも出ない (research.md §g)
    for request in sent:
        assert request.body is not None
        assert "seed" not in request.body
        assert "thinking" not in request.body
        assert "enable_thinking" not in request.body


def test_sends_variants_round_robin_in_order(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """回ごとに 1〜5 の順で送る (決めごとの 4)。"""
    th.run_thinking(
        base_url=fake_vllm.base_url,
        model=MODEL,
        var_root=tmp_path / "var",
        trials=2,
        report=io.StringIO(),
    )

    order = [variant_of(r.body) for r in fake_vllm.requests_for("/v1/messages")]
    assert order == list(_VARIANT_ORDER) * 2


def test_uses_the_same_conversation_for_every_variant_in_a_round(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """5 通りとも同じ会話を送る (「同じ入力」)。会話は 3 発話、assistant は thinking と text の
    両方のブロックを持つ (過去の thinking を持つ会話)。"""
    th.run_thinking(
        base_url=fake_vllm.base_url,
        model=MODEL,
        var_root=tmp_path / "var",
        trials=1,
        report=io.StringIO(),
    )

    bodies = [r.body for r in fake_vllm.requests_for("/v1/messages") if r.body is not None]
    messages_seen = [body["messages"] for body in bodies]
    assert all(messages == messages_seen[0] for messages in messages_seen)
    conversation = messages_seen[0]
    assert [message["role"] for message in conversation] == ["user", "assistant", "user"]
    assistant_content = conversation[1]["content"]
    assert isinstance(assistant_content, list)
    assert {block["type"] for block in assistant_content} == {"thinking", "text"}
    assert isinstance(conversation[0]["content"], str)
    assert isinstance(conversation[2]["content"], str)


def test_messages_can_be_overridden(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """呼ぶ側が `messages=` で会話を差し替えられる。"""
    custom = [{"role": "user", "content": "hi"}]

    th.run_thinking(
        base_url=fake_vllm.base_url,
        model=MODEL,
        var_root=tmp_path / "var",
        trials=1,
        messages=custom,
        report=io.StringIO(),
    )

    bodies = [r.body for r in fake_vllm.requests_for("/v1/messages") if r.body is not None]
    assert all(body["messages"] == custom for body in bodies)


def test_sends_no_credentials(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """待ち受けに認証がないので、認証のヘッダを付けない。"""
    th.run_thinking(
        base_url=fake_vllm.base_url,
        model=MODEL,
        var_root=tmp_path / "var",
        trials=1,
        report=io.StringIO(),
    )

    for request in fake_vllm.requests_for("/v1/messages"):
        assert "authorization" not in request.headers
        assert "x-api-key" not in request.headers


@pytest.mark.parametrize("trials", [1, 2, 4])
def test_trials_controls_the_total_number_of_requests(
    tmp_path: Path, fake_vllm: FakeVllm, trials: int
) -> None:
    th.run_thinking(
        base_url=fake_vllm.base_url,
        model=MODEL,
        var_root=tmp_path / "var",
        trials=trials,
        report=io.StringIO(),
    )

    assert len(fake_vllm.requests_for("/v1/messages")) == 5 * trials


# --- 記録すること (requirements 10.5) ------------------------------------------

_MARKER = "MARKER-DO-NOT-LEAK-4c1c9b"


def test_never_persists_or_reports_response_bodies(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """応答の本文・thinking の文・誤りの応答の本文が、report にも結果にも出ない (requirements
    10.5)。1 回目は誤りの本文に目印を入れ、それ以外は正常な応答の thinking と text に目印を
    入れる。"""
    faults = [Fault()] * (5 * th.DEFAULT_TRIALS)
    faults[0] = Fault(status=500, body=json.dumps({"error": {"message": _MARKER}}))
    fake_vllm.set_messages_fault_sequence(faults)
    fake_vllm.set_messages_factory(lambda _body: MessagesReply(text=_MARKER, thinking=_MARKER))
    report = io.StringIO()
    var_root = tmp_path / "var"

    outcome = th.run_thinking(
        base_url=fake_vllm.base_url, model=MODEL, var_root=var_root, report=report
    )

    assert _MARKER not in report.getvalue()
    assert _MARKER not in outcome.detail
    assert _MARKER not in outcome.model_dump_json()
    written = [path for path in var_root.rglob("*") if path.is_file()]
    assert written, "書いたファイルが 1 つもないと、この試験は何も確かめていない"
    for path in written:
        assert _MARKER not in path.read_text(encoding="utf-8")


def test_thinking_chars_is_zero_when_absent_and_sums_when_present(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """thinking のブロックが無い 200 の応答は `thinking_chars = 0`。あれば文字数の合計。"""

    def factory(body: dict[str, Any] | None) -> MessagesReply:
        if variant_of(body) == "none":
            return MessagesReply(thinking=None, output_tokens=5, input_tokens=50)
        return MessagesReply(thinking="x" * 7, output_tokens=5, input_tokens=50)

    fake_vllm.set_messages_factory(factory)

    outcome = th.run_thinking(
        base_url=fake_vllm.base_url,
        model=MODEL,
        var_root=tmp_path / "var",
        trials=1,
        report=io.StringIO(),
    )

    by_variant = {trial.variant: trial for trial in outcome.trials}
    assert by_variant["none"].thinking_chars == 0
    assert by_variant["output_config_low"].thinking_chars == 7


def test_thinking_chars_sums_multiple_thinking_blocks(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """`content` に thinking のブロックが複数あれば、`thinking_chars` はその**合計**になる
    (最初の値でも最後の値でもない)。`fake_vllm.MessagesReply` は thinking のブロックを 1 つしか
    作れないので、生の応答の本文 (`Fault(body=...)`) を台本にする
    (`test_missing_usage_is_treated_as_a_failed_trial` と同じ手法)。`fake_vllm.py` は書き換えない。
    """
    first_len = 5
    second_len = 9
    assert first_len != second_len  # 最初/最後の値だけでは合計と区別できないことの前提
    fake_vllm.set_messages_fault(
        Fault(
            body=json.dumps(
                {
                    "id": "msg_fake",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "a" * first_len},
                        {"type": "thinking", "thinking": "b" * second_len},
                        {"type": "text", "text": "ok"},
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            )
        )
    )

    outcome = th.run_thinking(
        base_url=fake_vllm.base_url,
        model=MODEL,
        var_root=tmp_path / "var",
        trials=1,
        report=io.StringIO(),
    )

    assert outcome.trials, "この試験は何も確かめていない"
    for trial in outcome.trials:
        assert trial.thinking_chars == first_len + second_len


# --- HTTP の失敗 (決めごとの 7) -------------------------------------------------


def test_http_failure_is_recorded_and_the_run_continues(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """1 回の失敗 (2xx 以外) は、値の欠けた `ThinkingTrial` になり、残りの回は続く。"""
    faults = [Fault()] * 15
    faults[6] = Fault(status=500)  # 2 回目の output_config_low (0 始まりで 5+1=6)
    fake_vllm.set_messages_fault_sequence(faults)

    outcome = th.run_thinking(
        base_url=fake_vllm.base_url, model=MODEL, var_root=tmp_path / "var", report=io.StringIO()
    )

    assert len(fake_vllm.requests_for("/v1/messages")) == 15
    failed = [
        trial
        for trial in outcome.trials
        if trial.variant == "output_config_low" and trial.trial_index == 2
    ]
    assert len(failed) == 1
    trial = failed[0]
    assert trial.http_status == 500
    assert trial.thinking_chars is None
    assert trial.output_tokens is None
    assert trial.input_tokens is None
    assert trial.stop_reason is None
    # ほかの回は影響を受けない
    others = [t for t in outcome.trials if t is not trial]
    assert all(t.http_status == 200 for t in others)


def test_missing_usage_is_treated_as_a_failed_trial(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """`usage` が丸ごと無い 200 の応答は、値の欠けた `ThinkingTrial` になる。"""
    fake_vllm.set_messages_fault(
        Fault(
            body=json.dumps(
                {
                    "id": "msg_fake",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                }
            )
        )
    )

    outcome = th.run_thinking(
        base_url=fake_vllm.base_url,
        model=MODEL,
        var_root=tmp_path / "var",
        trials=1,
        report=io.StringIO(),
    )

    assert all(trial.http_status == 200 for trial in outcome.trials)
    assert all(trial.thinking_chars is None for trial in outcome.trials)
    assert all(trial.input_tokens is None for trial in outcome.trials)
    assert all(trial.output_tokens is None for trial in outcome.trials)
    assert all(trial.stop_reason is None for trial in outcome.trials)


def test_connection_failure_is_recorded_with_no_http_status(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """つながらない (応答しない) ときは、`http_status` も空になる。"""
    fake_vllm.set_messages_fault(Fault(status=None))

    outcome = th.run_thinking(
        base_url=fake_vllm.base_url,
        model=MODEL,
        var_root=tmp_path / "var",
        trials=1,
        timeout_s=2.0,
        report=io.StringIO(),
    )

    assert all(trial.http_status is None for trial in outcome.trials)


# --- 判定 (決めごとの 6) --------------------------------------------------------


def _table(**overrides: list[MessagesReply]) -> dict[str, list[MessagesReply]]:
    base: dict[str, list[MessagesReply]] = {
        "none": [
            MessagesReply(thinking="a" * 10, output_tokens=20, input_tokens=100),
            MessagesReply(thinking="a" * 11, output_tokens=21, input_tokens=100),
            MessagesReply(thinking="a" * 12, output_tokens=22, input_tokens=100),
        ],
        "output_config_low": [
            MessagesReply(thinking="a" * 9, output_tokens=19, input_tokens=100),
            MessagesReply(thinking="a" * 11, output_tokens=21, input_tokens=100),
            MessagesReply(thinking="a" * 13, output_tokens=23, input_tokens=100),
        ],
        "chat_template_low": [
            MessagesReply(thinking="a" * 9, output_tokens=19, input_tokens=100),
            MessagesReply(thinking="a" * 11, output_tokens=21, input_tokens=100),
            MessagesReply(thinking="a" * 13, output_tokens=23, input_tokens=100),
        ],
        "output_config_medium": [
            MessagesReply(thinking="a" * 10, output_tokens=20, input_tokens=100),
            MessagesReply(thinking="a" * 11, output_tokens=21, input_tokens=100),
            MessagesReply(thinking="a" * 12, output_tokens=22, input_tokens=100),
        ],
        "clear_thinking": [
            MessagesReply(thinking="a" * 10, output_tokens=20, input_tokens=100),
            MessagesReply(thinking="a" * 11, output_tokens=21, input_tokens=100),
            MessagesReply(thinking="a" * 12, output_tokens=22, input_tokens=100),
        ],
    }
    base.update(overrides)
    return base


def test_effective_is_true_when_the_second_variant_clearly_separates(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    table = _table(
        output_config_low=[
            MessagesReply(thinking="a" * 50, output_tokens=80, input_tokens=100),
            MessagesReply(thinking="a" * 55, output_tokens=85, input_tokens=100),
            MessagesReply(thinking="a" * 60, output_tokens=90, input_tokens=100),
        ],
        clear_thinking=[
            MessagesReply(thinking="a" * 10, output_tokens=20, input_tokens=60),
            MessagesReply(thinking="a" * 11, output_tokens=21, input_tokens=62),
            MessagesReply(thinking="a" * 12, output_tokens=22, input_tokens=65),
        ],
    )
    fake_vllm.set_messages_factory(sequenced_factory(table))

    outcome = th.run_thinking(
        base_url=fake_vllm.base_url, model=MODEL, var_root=tmp_path / "var", report=io.StringIO()
    )

    assert outcome.effective is True
    assert "2 対 1" in outcome.detail and "重ならない" in outcome.detail
    assert "4 対 1" in outcome.detail and "期待どおり" in outcome.detail
    assert "5 対 1" in outcome.detail and "減った" in outcome.detail


def test_effective_is_false_when_neither_variant_separates(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    table = _table()  # 2、3 とも baseline と重なり、5 の input_tokens も変わらない
    fake_vllm.set_messages_factory(sequenced_factory(table))

    outcome = th.run_thinking(
        base_url=fake_vllm.base_url, model=MODEL, var_root=tmp_path / "var", report=io.StringIO()
    )

    assert outcome.effective is False
    assert "変わらなかった" in outcome.detail


def test_effective_is_none_when_a_variant_has_a_missing_value(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """2 対 1 が値の欠けで判定できず、3 対 1 は判定できて重なる: 効いたとは言えるが判定できたと
    は言えないので `None`。"""
    faults = [Fault()] * 15
    faults[6] = Fault(status=500)  # 2 回目の output_config_low
    fake_vllm.set_messages_fault_sequence(faults)
    fake_vllm.set_messages_factory(
        lambda _body: MessagesReply(thinking="a" * 10, output_tokens=20, input_tokens=100)
    )

    outcome = th.run_thinking(
        base_url=fake_vllm.base_url, model=MODEL, var_root=tmp_path / "var", report=io.StringIO()
    )

    assert outcome.effective is None
    assert "判定できない" in outcome.detail


def test_effective_is_none_when_the_lower_group_was_truncated_by_max_tokens(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """範囲が重ならないように見えても、値が低い側 (ここでは baseline) が `max_tokens` で
    打ち切られていれば、その比べは判定できない (決めごとの 6)。"""
    table = _table(
        none=[
            MessagesReply(
                thinking="a" * 10, output_tokens=20, input_tokens=100, stop_reason="max_tokens"
            ),
            MessagesReply(thinking="a" * 11, output_tokens=21, input_tokens=100),
            MessagesReply(thinking="a" * 12, output_tokens=22, input_tokens=100),
        ],
        output_config_low=[
            MessagesReply(thinking="a" * 50, output_tokens=80, input_tokens=100),
            MessagesReply(thinking="a" * 55, output_tokens=85, input_tokens=100),
            MessagesReply(thinking="a" * 60, output_tokens=90, input_tokens=100),
        ],
        chat_template_low=[
            MessagesReply(thinking="a" * 10, output_tokens=20, input_tokens=100),
            MessagesReply(thinking="a" * 11, output_tokens=21, input_tokens=100),
            MessagesReply(thinking="a" * 12, output_tokens=22, input_tokens=100),
        ],
    )
    fake_vllm.set_messages_factory(sequenced_factory(table))

    outcome = th.run_thinking(
        base_url=fake_vllm.base_url, model=MODEL, var_root=tmp_path / "var", report=io.StringIO()
    )

    assert outcome.effective is None
    assert "打ち切られた回を含む" in outcome.detail
    assert "2 対 1" in outcome.detail


# --- 引数の確かめ (決めごとの 9) ------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"trials": 0},
        {"trials": -1},
        {"trials": True},
        {"trials": 1.5},
        {"trials": math.nan},
        {"trials": math.inf},
        {"max_tokens": 0},
        {"max_tokens": -5},
        {"max_tokens": True},
        {"max_tokens": 1.5},
        {"timeout_s": 0.0},
        {"timeout_s": -1.0},
        {"timeout_s": math.nan},
        {"timeout_s": math.inf},
        {"timeout_s": -math.inf},
        {"base_url": "ftp://example.com"},
        {"base_url": "http://"},
        {"model": ""},
        {"messages": []},
    ],
    ids=lambda v: ",".join(f"{k}={val}" for k, val in v.items()),
)
def test_degenerate_arguments_are_rejected_before_touching_the_network(
    tmp_path: Path, fake_vllm: FakeVllm, overrides: dict[str, Any]
) -> None:
    """退化した引数 (0、負、NaN、無限大、bool、小数、URL の形が悪い、空) は、どれも Spark にも
    推論サーバーにも触る前に `ValueError` で断る (決めごとの 9)。"""
    kwargs: dict[str, Any] = {
        "base_url": fake_vllm.base_url,
        "model": MODEL,
        "var_root": tmp_path / "var",
        "report": io.StringIO(),
        **overrides,
    }

    with pytest.raises(ValueError):
        th.run_thinking(**kwargs)

    assert fake_vllm.requests == ()


# --- 中断 (決めごとの 8) --------------------------------------------------------


def test_keyboard_interrupt_writes_a_partial_summary_after_some_trials(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """3 回ぶん終わったところで中断されたら、その 3 回の要約を書いてから、そのまま上げ直す。"""
    transport = _InterruptingTransport(fail_at=4)
    client = httpx.Client(transport=transport, timeout=httpx.Timeout(5.0))
    var_root = tmp_path / "var"

    try:
        with pytest.raises(KeyboardInterrupt):
            th.run_thinking(
                base_url=fake_vllm.base_url,
                model=MODEL,
                var_root=var_root,
                client=client,
                report=io.StringIO(),
            )
    finally:
        client.close()

    result_files = list(var_root.rglob(th.RESULT_FILE_NAME))
    assert len(result_files) == 1
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert len(payload["trials"]) == 3


def test_keyboard_interrupt_before_any_trial_completes_writes_nothing(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """1 回も終わらないうちの中断は、要約を書かずに、そのまま上げ直してよい。"""
    transport = _InterruptingTransport(fail_at=1)
    client = httpx.Client(transport=transport, timeout=httpx.Timeout(5.0))
    var_root = tmp_path / "var"

    try:
        with pytest.raises(KeyboardInterrupt):
            th.run_thinking(
                base_url=fake_vllm.base_url,
                model=MODEL,
                var_root=var_root,
                client=client,
                report=io.StringIO(),
            )
    finally:
        client.close()

    assert list(var_root.rglob(th.RESULT_FILE_NAME)) == []


# --- クライアントの扱い ---------------------------------------------------------


def test_provided_client_is_reused_and_left_open(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """呼ぶ側が渡した `client` は、この関数の中で閉じない。"""
    client = httpx.Client(timeout=httpx.Timeout(5.0))
    try:
        th.run_thinking(
            base_url=fake_vllm.base_url,
            model=MODEL,
            var_root=tmp_path / "var",
            trials=1,
            client=client,
            report=io.StringIO(),
        )
        response = client.get(f"{fake_vllm.base_url}/health")
        assert response.status_code == 200
    finally:
        client.close()


# --- 置き場所 --------------------------------------------------------------------


def test_result_is_written_under_a_utc_stamped_thinking_directory(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    var_root = tmp_path / "var"
    fixed_now = datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC)

    outcome = th.run_thinking(
        base_url=fake_vllm.base_url,
        model=MODEL,
        var_root=var_root,
        trials=1,
        now=lambda: fixed_now,
        report=io.StringIO(),
    )

    expected = var_root / "20260922T120000Z-thinking" / th.RESULT_FILE_NAME
    assert expected.is_file()
    payload = json.loads(expected.read_text(encoding="utf-8"))
    assert payload["model"] == MODEL
    assert len(payload["trials"]) == 5
    assert payload["effective"] == outcome.effective
