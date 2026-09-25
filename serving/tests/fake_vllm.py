"""試験専用の偽の推論サーバー (task 1.5)。

標準ライブラリ (`http.server`、`threading`、`json`) だけで、空きポートに立ち、
`/health`、`/v1/models`、`/metrics`、`/version`、`/v1/messages` に台本どおりに応える。
届いた要求 (道筋、メソッド、ヘッダ、JSON の本文) を記録し、試験から読める。HTTP のモックの
ライブラリは使わない。

呼び名について: テストダブルの分類では、これは Fake (中身を簡略にした実装) ではなく、
**Stub (台本どおりに返す) + Spy (呼び出しを記録して、試験があとから検査する)** である。期待を
先に仕込んで自分で検証する Mock でもない。`fake_` という名前は、P0 の `bench/tests/fake_server.py`
と、design.md の File Structure Plan に合わせている。

**あとの並行のタスク (4.1、4.5、4.6) と、3.4、3.5、5.2、5.3 は、この偽物を書き換えずに、台本だけで
使う。** 足りない台本の形が出てきたら、並行の作業を止めて、この偽物の変更を 1 つの作業として先に
行うこと (`fake_runner.py` と同じ扱い)。

`serving_kit` のどの部品も import しない。design.md が定める「`serving/` は `bench_harness` を
import しない」に加えて、この偽物は Spark にも実物の vLLM にも触れない、独立した道具にするため。

## 台本の決まり

- **5 つの道筋 (`/health`、`/v1/models`、`/version`、`/metrics`、`/v1/messages`) のどれでも、
  同じ形の `Fault` で、状態と「応答しない」を切り替えられる。** `set_X_fault` (常にこの 1 つ) か
  `set_X_fault_sequence` (呼び出しごとに順に使い、尽きたら最後を繰り返す) の、道筋ごとに同じ名前の
  口で切り替える。`Fault.status` が `None` のときは、応答を書かずに接続を切る (「応答しない」)。
  `Fault.status` が 200 以外のときは、その状態で、道筋ごとの既定の誤りの本文を返す。`Fault.body` を
  指定すると、状態が 200 のままでも、正常な中身の組み立てをやめて、それをそのまま返す (`/metrics`
  の「壊れた本文 (Prometheus の形でないテキスト)」のように、状態を変えずに中身だけ壊せる)。「何回目
  から 200 を返すか」も「何回目から 503、応答しない、壊れた本文になるか」も、この 1 つの口で表せる
- `Fault` が「正常」(既定の `Fault()`。状態が 200 で `body` なし) のときだけ、道筋ごとの正常な中身の
  台本に進む
- `/v1/models` の名前と入力の長さの上限、`/version` の版は、`set_model` / `set_max_model_len` /
  `set_version` で 1 つの値を決める (`Fault` とは別の軸)
- `/metrics` の値 (処理中と待ちの要求の数、生成のトークンの数、投機的デコードの指標の有無) は、
  `set_metrics` / `set_metrics_sequence` で切り替える。このモジュールは、時間や呼び出しの回数に
  応じて自動では増やさない。「生成のトークンの数がいつ止まるか」は、試験が渡す列の中で、値が同じ
  ところで折り返すことで表す
- `/v1/messages` の正常な中身は `set_messages_reply` / `set_messages_sequence` / さらに要求の
  本文で選べる `set_messages_factory` (thinking の深さの確かめ (4.6) の、5 通りの送り分けに使う)
  で切り替える。3 つは、最後に呼んだものだけが効く

## 記録

`requests` に、届いた要求が届いた順に入る (`RecordedRequest`)。`requests_for(path)`、
`call_count(path)` がある。複数の接続を別々の thread で捌くので、記録と台本の読み書きは、
すべて 1 つの鍵 (`threading.Lock`) で守る。
"""

from __future__ import annotations

import contextlib
import json
import socket
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.parse import urlsplit

__all__ = [
    "Fault",
    "FakeVllm",
    "MessagesFactory",
    "MessagesReply",
    "MetricsSample",
    "RecordedRequest",
]

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class Fault:
    """5 つの道筋のどれでも使う、1 回ぶんの「状態を変える」台本。

    - `status` が `None` のときは、応答を書かずに接続を切る (「応答しない」を表す)
    - `status` が 200 以外のときは、その状態で、道筋ごとの既定の誤りの本文を返す
      (`body` を指定すれば、それを返す)
    - `status` が 200 で `body` が `None` (既定の `Fault()`) のときだけ「正常」とみなし、
      道筋ごとの正常な中身の台本 (`MetricsSample`、`MessagesReply` など) に進む
    - `status` が 200 のまま `body` だけ指定すると、状態は変えずに中身だけ壊せる
      (`/metrics` の「Prometheus の形でないテキスト」のように)
    """

    status: int | None = 200
    body: str | None = None

    @property
    def is_normal(self) -> bool:
        """正常な中身の台本に進んでよいかどうか。"""
        return self.status == 200 and self.body is None


@dataclass(frozen=True)
class MetricsSample:
    """`/metrics` が正常なときの、1 回ぶんの値。design.md の「運転 › lifecycle」と
    「確認 › watch」が読む項目 (処理中と待ちの要求の数、生成のトークンの数、投機的デコードの
    指標の有無) だけを持つ。
    """

    running_requests: int = 0
    waiting_requests: int = 0
    generation_tokens_total: int = 0
    spec_decode: bool = False
    """`True` のとき、`vllm:spec_decode_` で始まる行を `/metrics` に足す。受け付けの開始の
    判定 (design.md 6.7) は、この行の有無が構成の `allow_speculative` と一致することを確かめる。"""


@dataclass(frozen=True)
class MessagesReply:
    """`/v1/messages` が正常なときの、1 回ぶんの返事 (ストリームでない応答だけ)。design.md の
    「運転 › lifecycle」の「短い要求」と「確認 › thinking」の節は、どちらもストリームの応答を
    要求していない。
    """

    text: str = "ok"
    thinking: str | None = None
    """`None` のときは、`content` に thinking のブロックを含めない。"""
    input_tokens: int = 10
    output_tokens: int = 5
    stop_reason: str = "end_turn"


MessagesFactory = Callable[[JsonDict | None], MessagesReply]
"""要求の本文 (JSON を解いたもの。解けなければ `None`) から、その要求への返事を決める関数。
5 通りの thinking の送り分け (4.6) のように、本文の中身で応答を変えたいときに使う。
"""


@dataclass(frozen=True)
class RecordedRequest:
    """記録した 1 つの要求。"""

    path: str
    method: str
    headers: Mapping[str, str]
    body: JsonDict | None


def _next[T](sequence: tuple[T, ...], position: int) -> tuple[T, int]:
    """`position` 番目 (0 始まり) の値と、次に使う位置を返す。尽きたら最後を繰り返す。"""
    index = min(position, len(sequence) - 1)
    return sequence[index], position + 1


class _Script[T]:
    """1 つの道筋の、呼び出しごとに進む台本 (列と、いまの位置)。呼び出し側の鍵の中で使うので、
    自分では鍵を持たない。`Fault` の台本にも、`MetricsSample` / `MessagesReply` の台本にも、
    同じ形で使う。
    """

    def __init__(self, default: T) -> None:
        self._sequence: tuple[T, ...] = (default,)
        self._pos = 0

    def set(self, values: Sequence[T]) -> None:
        if not values:
            raise ValueError("台本が空")
        self._sequence = tuple(values)
        self._pos = 0

    def next(self) -> T:
        value, self._pos = _next(self._sequence, self._pos)
        return value


class FakeVllm:
    """偽の推論サーバー。`start()` で待ち受け、`stop()` で止める。止まったあとは、その port に
    つながらない。
    """

    def __init__(
        self,
        *,
        model: str = "glm-5-3-flash",
        max_model_len: int = 4096,
        version: str | None = "0.0.0-fake",
    ) -> None:
        self._lock = threading.Lock()
        self._model = model
        self._max_model_len = max_model_len
        self._version = version

        # 5 つの道筋に共通の、状態を変える台本 (既定はすべて「正常」)
        self._health_fault: _Script[Fault] = _Script(Fault())
        self._models_fault: _Script[Fault] = _Script(Fault())
        self._version_fault: _Script[Fault] = _Script(Fault())
        self._metrics_fault: _Script[Fault] = _Script(Fault())
        self._messages_fault: _Script[Fault] = _Script(Fault())

        self._metrics_script: _Script[MetricsSample] = _Script(MetricsSample())

        self._messages_script: _Script[MessagesReply] = _Script(MessagesReply())
        self._messages_factory: MessagesFactory | None = None
        self._message_counter = 0

        self._requests: list[RecordedRequest] = []
        self.errors: list[str] = []
        """handler の中で起きた、想定していない例外の一覧 (試験の後始末で空であることを
        確かめられる)。"""

        self._server: _HttpServer | None = None
        self._thread: threading.Thread | None = None

    # --- 起動と停止 -------------------------------------------------------

    def start(self) -> None:
        """空いている port で待ち受けを始める。"""
        if self._server is not None:
            raise RuntimeError("すでに起動している")
        server = _HttpServer(("127.0.0.1", 0), _Handler, self)
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.02},
            name="fake-vllm",
            daemon=True,
        )
        thread.start()
        self._server = server
        self._thread = thread

    def stop(self) -> None:
        """待ち受けを止める。止まったあとは、この port につながらない。"""
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5.0)

    def __enter__(self) -> FakeVllm:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        """`http://127.0.0.1:<port>`。"""
        if self._server is None:
            raise RuntimeError("起動していない")
        host, port = self._server.server_address[0], self._server.server_address[1]
        return f"http://{host!s}:{port}"

    # --- 台本: /v1/models、/version の正常な中身 ------------------------------

    def set_model(self, name: str) -> None:
        """`/v1/models` と `/v1/messages` の応答が名乗るモデルの名前。"""
        with self._lock:
            self._model = name

    def set_max_model_len(self, n: int) -> None:
        """`/v1/models` の `max_model_len` (対応する入力の長さの上限)。"""
        with self._lock:
            self._max_model_len = n

    def set_version(self, version: str | None) -> None:
        """`/version` が返す版。`None` なら 404 にする。"""
        with self._lock:
            self._version = version

    # --- 台本: 5 つの道筋に共通の、状態を変える口 --------------------------------

    def set_health_fault(self, fault: Fault) -> None:
        """`/health` を、常にこの 1 つの状態にする。"""
        self.set_health_fault_sequence((fault,))

    def set_health_fault_sequence(self, faults: Sequence[Fault]) -> None:
        """`/health` の状態を、呼び出しごとに順に使う (尽きたら最後を繰り返す)。

        「何回目から 200 を返すか (それまでは接続できるが 503)」も「何回目から失敗するか
        (503、または応答しない)」も、この 1 つの口で表せる。
        """
        with self._lock:
            self._health_fault.set(faults)

    def set_models_fault(self, fault: Fault) -> None:
        """`/v1/models` を、常にこの 1 つの状態にする。"""
        self.set_models_fault_sequence((fault,))

    def set_models_fault_sequence(self, faults: Sequence[Fault]) -> None:
        """`/v1/models` の状態を、呼び出しごとに順に使う (尽きたら最後を繰り返す)。"""
        with self._lock:
            self._models_fault.set(faults)

    def set_version_fault(self, fault: Fault) -> None:
        """`/version` を、常にこの 1 つの状態にする。"""
        self.set_version_fault_sequence((fault,))

    def set_version_fault_sequence(self, faults: Sequence[Fault]) -> None:
        """`/version` の状態を、呼び出しごとに順に使う (尽きたら最後を繰り返す)。"""
        with self._lock:
            self._version_fault.set(faults)

    def set_metrics_fault(self, fault: Fault) -> None:
        """`/metrics` を、常にこの 1 つの状態にする。"""
        self.set_metrics_fault_sequence((fault,))

    def set_metrics_fault_sequence(self, faults: Sequence[Fault]) -> None:
        """`/metrics` の状態を、呼び出しごとに順に使う (尽きたら最後を繰り返す)。

        `Fault(status=503)` で「読めない」、`Fault(status=None)` で「応答しない」、
        `Fault(body="...")` で「200 のまま壊れた本文 (Prometheus の形でない)」を作れる。
        """
        with self._lock:
            self._metrics_fault.set(faults)

    def set_messages_fault(self, fault: Fault) -> None:
        """`/v1/messages` を、常にこの 1 つの状態にする。"""
        self.set_messages_fault_sequence((fault,))

    def set_messages_fault_sequence(self, faults: Sequence[Fault]) -> None:
        """`/v1/messages` の状態を、呼び出しごとに順に使う (尽きたら最後を繰り返す)。

        状態が 200 以外のときは、既定で Anthropic の誤りの形 (`{"type": "error", "error":
        {...}}`) を返す (3.5 の短い要求と 4.1 の縮小の確認が、500 や 400 を受けたときの扱いを
        試験できるように)。
        """
        with self._lock:
            self._messages_fault.set(faults)

    # --- 台本: /metrics の正常な中身 -------------------------------------------

    def set_metrics(self, sample: MetricsSample) -> None:
        """`/metrics` が正常なときの値を、常にこの 1 つにする。"""
        self.set_metrics_sequence((sample,))

    def set_metrics_sequence(self, samples: Sequence[MetricsSample]) -> None:
        """`/metrics` が正常なときの値を、呼び出しごとに順に使う (尽きたら最後を繰り返す)。

        生成のトークンの数を増やしていく列を作り、途中から同じ値を繰り返させれば、「途中で
        止まった」ことを表せる。
        """
        with self._lock:
            self._metrics_script.set(samples)

    # --- 台本: /v1/messages の正常な中身 -----------------------------------------

    def set_messages_reply(self, reply: MessagesReply) -> None:
        """`/v1/messages` が正常なときの返事を、常にこの 1 つにする。"""
        self.set_messages_sequence((reply,))

    def set_messages_sequence(self, replies: Sequence[MessagesReply]) -> None:
        """`/v1/messages` が正常なときの返事を、要求ごとに順に使う (尽きたら最後を繰り返す)。"""
        with self._lock:
            self._messages_script.set(replies)
            self._messages_factory = None

    def set_messages_factory(self, factory: MessagesFactory) -> None:
        """要求の本文から、その要求への正常な返事を決める関数を登録する
        (`set_messages_sequence` を上書きする)。
        """
        with self._lock:
            self._messages_factory = factory

    # --- 記録の読み出し ---------------------------------------------------------

    @property
    def requests(self) -> tuple[RecordedRequest, ...]:
        """届いたすべての要求 (届いた順)。"""
        with self._lock:
            return tuple(self._requests)

    def requests_for(self, path: str) -> tuple[RecordedRequest, ...]:
        """1 つの道筋あての要求だけ。"""
        return tuple(call for call in self.requests if call.path == path)

    def call_count(self, path: str) -> int:
        """1 つの道筋あての要求の数。"""
        return len(self.requests_for(path))

    # --- サーバー内部から呼ばれる口 -------------------------------------------------

    def _record(
        self, path: str, method: str, headers: Mapping[str, str], body: JsonDict | None
    ) -> None:
        with self._lock:
            self._requests.append(
                RecordedRequest(path=path, method=method, headers=dict(headers), body=body)
            )

    def _next_health_fault(self) -> Fault:
        with self._lock:
            return self._health_fault.next()

    def _next_models_fault(self) -> Fault:
        with self._lock:
            return self._models_fault.next()

    def _next_version_fault(self) -> Fault:
        with self._lock:
            return self._version_fault.next()

    def _next_metrics_fault(self) -> Fault:
        with self._lock:
            return self._metrics_fault.next()

    def _next_messages_fault(self) -> Fault:
        with self._lock:
            return self._messages_fault.next()

    def _models_payload(self) -> JsonDict:
        with self._lock:
            return {
                "object": "list",
                "data": [
                    {
                        "id": self._model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "fake",
                        "max_model_len": self._max_model_len,
                    }
                ],
            }

    def _version_payload(self) -> JsonDict | None:
        with self._lock:
            if self._version is None:
                return None
            return {"version": self._version}

    def _metrics_payload(self) -> str:
        with self._lock:
            sample = self._metrics_script.next()
            model = self._model
        return _render_metrics(sample, model=model)

    def _messages_payload(self, body: JsonDict | None) -> JsonDict:
        with self._lock:
            factory = self._messages_factory
            reply = factory(body) if factory is not None else self._messages_script.next()
            self._message_counter += 1
            counter = self._message_counter
            model = self._model
        content: list[JsonDict] = []
        if reply.thinking is not None:
            content.append({"type": "thinking", "thinking": reply.thinking})
        content.append({"type": "text", "text": reply.text})
        return {
            "id": f"msg_fake_{counter}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": reply.stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
            },
        }


def _render_metrics(sample: MetricsSample, *, model: str) -> str:
    """design.md の「運転 › lifecycle」(受け付けの開始の判定) と「確認 › watch」(見張りの指標)
    が読む名前で、Prometheus のテキストの形にする (research.md の投機的デコードの指標の節と、
    design.md の watch の節が挙げる、vLLM の実物の名前に合わせる)。
    """
    label = f'{{model_name="{model}"}}'
    lines = [
        "# HELP vllm:num_requests_running Number of requests in model execution batches.",
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running{label} {float(sample.running_requests)!r}",
        "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting{label} {float(sample.waiting_requests)!r}",
        "# HELP vllm:generation_tokens_total Number of generation tokens processed.",
        "# TYPE vllm:generation_tokens_total counter",
        f"vllm:generation_tokens_total{label} {float(sample.generation_tokens_total)!r}",
    ]
    if sample.spec_decode:
        for name, help_text in (
            ("vllm:spec_decode_num_drafts_total", "Number of speculative tokens drafted."),
            ("vllm:spec_decode_num_draft_tokens_total", "Number of draft tokens."),
            ("vllm:spec_decode_num_accepted_tokens_total", "Number of accepted tokens."),
        ):
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name}{label} 0.0")
    return "\n".join(lines) + "\n"


class _HttpServer(ThreadingHTTPServer):
    """接続ごとに daemon の thread を使う HTTP サーバー。"""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        fake: FakeVllm,
    ) -> None:
        self.fake = fake
        super().__init__(address, handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        # 「応答しない」台本で接続を切ったときに出る OSError は、試験の出力を汚さない
        exc = sys.exc_info()[1]
        if isinstance(exc, OSError | TimeoutError):
            return
        self.fake.errors.append(f"{type(exc).__name__}: {exc}")


class _Handler(BaseHTTPRequestHandler):
    """1 つの接続を扱う。"""

    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    timeout = 10.0
    """何も送ってこない接続を、いつまでも抱えないための上限。"""

    @property
    def _fake(self) -> FakeVllm:
        return cast(_HttpServer, self.server).fake

    def log_message(self, format: str, *args: Any) -> None:
        """試験の出力を汚さないよう、何も出さない。"""

    # --- 経路 -------------------------------------------------------------

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            self._handle_health()
        elif path == "/v1/models":
            self._handle_models()
        elif path == "/version":
            self._handle_version()
        elif path == "/metrics":
            self._handle_metrics()
        else:
            self._fake._record(path, "GET", self._header_map(), None)
            self._send_json(404, _error_body(f"unknown path {path}"))

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        raw = self._read_body()
        body = _parse_json(raw)
        if path == "/v1/messages":
            self._handle_messages(body)
        else:
            self._fake._record(path, "POST", self._header_map(), body)
            self._send_json(404, _error_body(f"unknown path {path}"))

    # --- 道筋ごとの扱い ---------------------------------------------------------
    #
    # どの道筋も、同じ順序で扱う: 記録する → `Fault` を引く → 正常でなければ `_apply_fault` に
    # まかせて終える → 正常なら、道筋ごとの中身を組み立てて返す。

    def _handle_health(self) -> None:
        # 実物と同じく、/health はどの状態でも本文を持たない
        self._fake._record("/health", "GET", self._header_map(), None)
        fault = self._fake._next_health_fault()
        if self._apply_fault(fault, content_type="text/plain", default_error=""):
            return
        self._send_status(200)

    def _handle_models(self) -> None:
        self._fake._record("/v1/models", "GET", self._header_map(), None)
        fault = self._fake._next_models_fault()
        if self._apply_fault(
            fault, content_type="application/json", default_error=_error_body("fake error")
        ):
            return
        self._send_json(200, self._fake._models_payload())

    def _handle_version(self) -> None:
        self._fake._record("/version", "GET", self._header_map(), None)
        fault = self._fake._next_version_fault()
        if self._apply_fault(
            fault, content_type="application/json", default_error=_error_body("fake error")
        ):
            return
        payload = self._fake._version_payload()
        if payload is None:
            self._send_json(404, _error_body("no version endpoint"))
        else:
            self._send_json(200, payload)

    def _handle_metrics(self) -> None:
        self._fake._record("/metrics", "GET", self._header_map(), None)
        fault = self._fake._next_metrics_fault()
        content_type = "text/plain; version=0.0.4; charset=utf-8"
        if self._apply_fault(fault, content_type=content_type, default_error="fake error"):
            return
        text = self._fake._metrics_payload()
        self._send_text(200, text, content_type)

    def _handle_messages(self, body: JsonDict | None) -> None:
        self._fake._record("/v1/messages", "POST", self._header_map(), body)
        fault = self._fake._next_messages_fault()
        default_error = {
            "type": "error",
            "error": {"type": "api_error", "message": "fake error"},
        }
        if self._apply_fault(fault, content_type="application/json", default_error=default_error):
            return
        self._send_json(200, self._fake._messages_payload(body))

    def _apply_fault(
        self, fault: Fault, *, content_type: str, default_error: str | JsonDict
    ) -> bool:
        """`fault` の指示にしたがって応答を送る。送り切ったら `True` を返す (呼び出し側は、
        正常な中身を組み立てずに、そこで終える)。`fault` が正常 (`Fault.is_normal`) なら、
        何もせずに `False` を返す。
        """
        if fault.status is None:
            self._drop_connection()
            return True
        if fault.body is not None:
            self._send_text(fault.status, fault.body, content_type)
            return True
        if fault.status != 200:
            if isinstance(default_error, str):
                self._send_text(fault.status, default_error, content_type)
            else:
                self._send_json(fault.status, default_error)
            return True
        return False

    # --- 助け ---------------------------------------------------------------

    def _header_map(self) -> dict[str, str]:
        return {key.lower(): value for key, value in self.headers.items()}

    def _read_body(self) -> bytes:
        length = _int_header(self.headers.get("Content-Length"))
        if length <= 0:
            return b""
        try:
            return self.rfile.read(length)
        except OSError:
            return b""

    def _send_status(self, status: int) -> None:
        """本文のない応答 (`/health` は、実物もボディを持たない)。"""
        try:
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()
        except OSError:
            self.close_connection = True

    def _send_json(self, status: int, payload: JsonDict) -> None:
        self._send_text(status, json.dumps(payload, ensure_ascii=False), "application/json")

    def _send_text(self, status: int, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        except OSError:
            self.close_connection = True

    def _drop_connection(self) -> None:
        """応答を書かずに接続を切る (「応答しない」を表す)。"""
        self.close_connection = True
        with contextlib.suppress(OSError):
            self.connection.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self.connection.close()


# --- 小さな助け -----------------------------------------------------------


def _parse_json(raw: bytes) -> JsonDict | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _int_header(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def _error_body(message: str) -> JsonDict:
    return {"error": {"message": message}}
