"""`POST /v1/messages` に 1 つの要求をストリームで送り、時刻と中身を返す (task 2.1)。

この module は「何を測っているか」を知らない。やり直しの仕組みも持たない (10.6)。
通信の失敗、HTTP のエラー、ストリームの途中のエラー、3 種類の時間切れ、ストリームの
形の誤りは、例外ではなく `StreamResult.error` として返す (10.1)。

時刻は 2 種類を持つ。経過の計算には単調な時計 (`time.perf_counter_ns`)、記録には
UTC の時刻。イベントの時刻は、受け取った直後、JSON を解析する前に打つ。

イベントの間隔の制限時間は、この module 自身が測る。実物の vLLM は `ping` を
送らないので、httpx の読み取りの制限時間では「イベントとイベントの間が空いた」
ことを表せない。

認証の情報は `Authorization: Bearer` と `x-api-key` の両方に付け、結果にも、
失敗のメッセージにも、この class の `repr` にも入れない (1.8)。

依存するのは標準ライブラリと httpx、httpx-sse、pydantic、`bench_harness.types`
だけ。ほかの `bench_harness` の module を読み込まない。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from types import TracebackType
from typing import Any, Final, Literal, Protocol, Self, runtime_checkable

import httpx
from httpx_sse import EventSource, ServerSentEvent, aconnect_sse
from pydantic import JsonValue, SecretStr, ValidationError

from bench_harness.types import (
    ContentBlock,
    MessagesRequest,
    RequestError,
    RequestErrorKind,
    StreamResult,
    StreamTiming,
    TargetDef,
    TimeoutPolicy,
    Usage,
)

__all__ = [
    "ANTHROPIC_VERSION",
    "HttpxMessagesClient",
    "MessagesClient",
    "build_request_body",
]

ANTHROPIC_VERSION: Final[str] = "2023-06-01"
"""`anthropic-version` のヘッダーの値。"""

MESSAGES_PATH: Final[str] = "/v1/messages"

_MAX_MESSAGE_CHARS: Final[int] = 500
"""失敗のメッセージに残す、本文の抜き書きの長さの上限。"""

_DEFAULT_MAX_CONNECTIONS: Final[int] = 32
"""1 つの client が張れる接続の数。同時 8 本のストリームに十分な余裕を取る。"""

_DRAIN_GRACE_S: Final[float] = 0.1
"""`message_stop` のあと、ストリームの終わりを待つ短い猶予 (下の `_drain` を参照)。"""

_DRAIN_MAX_EVENTS: Final[int] = 8
"""猶予の中で読み捨てる、後始末のイベントの数の上限。"""

_USAGE_FIELDS: Final[tuple[str, ...]] = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)

_BlockType = Literal["text", "thinking", "tool_use"]

_DELTA_BLOCK_TYPE: Final[dict[str, _BlockType]] = {
    "text_delta": "text",
    "thinking_delta": "thinking",
    "input_json_delta": "tool_use",
}
"""delta の種類から、ブロックの種類を推す (`content_block_start` が来なかった場合の備え)。"""


# --- 送る本文 -----------------------------------------------------------


def build_request_body(request: MessagesRequest) -> dict[str, JsonValue]:
    """実際に送る JSON の本文を、そのまま返す純粋な関数。

    `None` の項目は落とし、`stream` は必ず真にし、`extra` は本文の**最上位**に
    混ぜる (tasks.md の Implementation Notes 1.2)。`extra` の鍵が型のある項目
    (`model`、`max_tokens` など) と重なるときは、呼び出しの誤りとして `ValueError`。
    保存の部品 (2.3) は、この関数の返り値を「実際に送った本文」として残せる。
    """
    reserved = set(MessagesRequest.model_fields) - {"extra"}
    clashing = sorted(reserved & set(request.extra))
    if clashing:
        # 同じ条件の試行が同じ設定で送られること (2.7) を、extra が黙って崩さないようにする
        raise ValueError(f"extra は、型のある項目を上書きできない: {clashing}")
    dumped: dict[str, Any] = request.model_dump(mode="json", exclude_none=True, exclude={"extra"})
    body: dict[str, JsonValue] = dict(dumped)
    body.update(request.extra)
    body["stream"] = True
    return body


def _auth_headers(api_key: SecretStr | None) -> dict[str, str]:
    """要求ごとに組み立てる。認証の情報は、どこにも溜めない (1.8)。"""
    headers = {"content-type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
    if api_key is not None:
        value = api_key.get_secret_value()
        headers["authorization"] = f"Bearer {value}"
        headers["x-api-key"] = value
    return headers


def _excerpt(text: str, secret: str | None = None) -> str:
    """失敗のメッセージを、1 行の決まった長さに切り詰める。

    対象サーバーが、要求のヘッダーをエラーの本文に写して返すことがある。認証の
    値が生データに落ちないよう、切り詰める前に伏せる (1.8)。
    """
    if secret:
        text = text.replace(secret, "***")
    collapsed = " ".join(text.split())
    if len(collapsed) <= _MAX_MESSAGE_CHARS:
        return collapsed
    return collapsed[:_MAX_MESSAGE_CHARS] + "…"


# --- ブロックとトークン数の組み立て -------------------------------------


@dataclass
class _BlockAccumulator:
    """`index` ごとに、断片を集める入れ物。"""

    type: _BlockType | None
    tool_name: str | None = None
    text_parts: list[str] = field(default_factory=list)
    json_parts: list[str] = field(default_factory=list)

    def build(self) -> ContentBlock | None:
        if self.type is None:
            return None  # 知らない種類のブロックは捨てる
        if self.type == "tool_use":
            raw = "".join(self.json_parts)
            return ContentBlock(
                type="tool_use",
                tool_name=self.tool_name,
                tool_input_raw=raw,
                tool_input=_parse_tool_input(raw),
            )
        return ContentBlock(type=self.type, text="".join(self.text_parts))


def _parse_tool_input(raw: str) -> dict[str, JsonValue] | None:
    """`input_json_delta` をつないだ文字列を、写像として読めたときだけ返す。

    断片が 1 つも来なかった (空の) ツール呼び出しは、Anthropic の決まりに従って
    引数なし (`{}`) とみなす。読めない、または写像でないときは `None`。
    """
    if raw == "":
        return {}
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return {str(key): value for key, value in parsed.items()}
    return None


def _usage_fields(payload: Any) -> dict[str, Any]:
    """トークン数の写像から、知っている項目だけを取り出す。"""
    if not isinstance(payload, Mapping):
        return {}
    return {key: payload[key] for key in _USAGE_FIELDS if payload.get(key) is not None}


def _merge_usage(from_start: dict[str, Any], from_delta: dict[str, Any]) -> Usage | None:
    """終わりの側 (`message_delta`) の値を正とし、来なかった項目だけ始まりの側で補う。"""
    if not from_start and not from_delta:
        return None
    merged: dict[str, Any] = {**from_start, **from_delta}
    merged.setdefault("input_tokens", 0)
    merged.setdefault("output_tokens", 0)
    try:
        return Usage.model_validate(merged)
    except ValidationError:
        return None


# --- ストリームの読み取り -----------------------------------------------


class _Step(Enum):
    """1 つのイベントを読んだあと、どうするか。"""

    CONTINUE = auto()
    DONE = auto()
    FAILED = auto()


class _Collector:
    """届いたイベントから、時刻、トークン数、中身を組み立てる。"""

    def __init__(self, sent_at_utc: datetime, sent_at_ns: int, secret: str | None = None) -> None:
        self._secret = secret  # 失敗のメッセージから伏せるためだけに持つ。結果には入れない
        self.sent_at_utc = sent_at_utc
        self.sent_at_ns = sent_at_ns
        self.event_count = 0
        self.message_start_ns: int | None = None
        self.first_token_ns: int | None = None
        self.first_text_ns: int | None = None
        self.last_token_ns: int | None = None
        self.stop_reason: str | None = None
        self.server_model: str | None = None
        self.error: RequestError | None = None
        self._blocks: dict[int, _BlockAccumulator] = {}
        self._usage_start: dict[str, Any] = {}
        self._usage_delta: dict[str, Any] = {}

    # --- イベント ---

    def handle(self, sse: ServerSentEvent, stamp_ns: int) -> _Step:
        """1 つのイベントを取り込む。時刻はすでに打ってある (解析の前)。"""
        self.event_count += 1
        try:
            payload: Any = json.loads(sse.data) if sse.data else None
        except json.JSONDecodeError as exc:
            self.error = RequestError(
                kind="protocol",
                message=_excerpt(f"SSE の JSON を読み取れない ({sse.event}): {exc}", self._secret),
            )
            return _Step.FAILED
        if not isinstance(payload, dict):
            return _Step.CONTINUE  # 知らない形は無視する

        event = sse.event
        if event == "message_start":
            self._on_message_start(payload, stamp_ns)
        elif event == "content_block_start":
            self._on_block_start(payload)
        elif event == "content_block_delta":
            self._on_block_delta(payload, stamp_ns)
        elif event == "message_delta":
            self._on_message_delta(payload)
        elif event == "message_stop":
            return _Step.DONE
        elif event == "error":
            self.error = RequestError(
                kind="stream_error", message=_excerpt(_error_text(payload), self._secret)
            )
            return _Step.FAILED
        # `ping`、`content_block_stop`、知らないイベントは、何もしない
        return _Step.CONTINUE

    def _on_message_start(self, payload: dict[str, Any], stamp_ns: int) -> None:
        if self.message_start_ns is None:
            self.message_start_ns = stamp_ns
        message = payload.get("message")
        if not isinstance(message, dict):
            return
        model = message.get("model")
        if isinstance(model, str) and self.server_model is None:
            self.server_model = model
        if not self._usage_start:
            self._usage_start = _usage_fields(message.get("usage"))

    def _on_block_start(self, payload: dict[str, Any]) -> None:
        index = _index_of(payload)
        if index is None:
            return
        block = payload.get("content_block")
        if not isinstance(block, dict):
            return
        kind = block.get("type")
        if kind == "text":
            accumulator = _BlockAccumulator(type="text")
            _append_str(accumulator.text_parts, block.get("text"))
        elif kind == "thinking":
            accumulator = _BlockAccumulator(type="thinking")
            _append_str(accumulator.text_parts, block.get("thinking"))
        elif kind == "tool_use":
            name = block.get("name")
            accumulator = _BlockAccumulator(
                type="tool_use", tool_name=name if isinstance(name, str) else None
            )
        else:
            accumulator = _BlockAccumulator(type=None)  # 知らない種類は捨てる
        self._blocks[index] = accumulator

    def _on_block_delta(self, payload: dict[str, Any], stamp_ns: int) -> None:
        # 種類を問わず、`content_block_delta` はトークンとして数える (設計の速さの定義)
        if self.first_token_ns is None:
            self.first_token_ns = stamp_ns
        self.last_token_ns = stamp_ns

        delta = payload.get("delta")
        if not isinstance(delta, dict):
            return
        kind = delta.get("type")
        if kind == "text_delta" and self.first_text_ns is None:
            self.first_text_ns = stamp_ns

        index = _index_of(payload)
        if index is None:
            return
        accumulator = self._blocks.get(index)
        if accumulator is None:
            inferred = _DELTA_BLOCK_TYPE.get(str(kind))
            accumulator = _BlockAccumulator(type=inferred)
            self._blocks[index] = accumulator
        if accumulator.type is None:
            return
        if kind == "text_delta":
            _append_str(accumulator.text_parts, delta.get("text"))
        elif kind == "thinking_delta":
            _append_str(accumulator.text_parts, delta.get("thinking"))
        elif kind == "input_json_delta":
            _append_str(accumulator.json_parts, delta.get("partial_json"))
        # `signature_delta` と知らない種類は、中身に足さない

    def _on_message_delta(self, payload: dict[str, Any]) -> None:
        delta = payload.get("delta")
        if isinstance(delta, dict):
            stop_reason = delta.get("stop_reason")
            if isinstance(stop_reason, str):
                self.stop_reason = stop_reason
        self._usage_delta.update(_usage_fields(payload.get("usage")))

    # --- 結果 ---

    def blocks(self) -> list[ContentBlock]:
        """`index` の順に並べる (順序そのものは、採点にも集計にも使わない)。"""
        built = (self._blocks[index].build() for index in sorted(self._blocks))
        return [block for block in built if block is not None]

    def timing(self, end_ns: int) -> StreamTiming:
        """不変条件 (時刻が前後しない) を、失敗の道でも必ず守る形で組み立てる。

        時刻は到着の順に打つので、ふつうは何も起きない。壊れた並び (たとえば
        `content_block_delta` のあとに `message_start` が来る) のときは、
        `message_start_ns` のほうを最初のトークンまで引き下げる。最初のトークン
        と最後のトークンの時刻は、速さの計算に使うので動かさない。
        """
        message_start_ns = self.message_start_ns
        if message_start_ns is not None and self.first_token_ns is not None:
            message_start_ns = min(message_start_ns, self.first_token_ns)
        running = self.sent_at_ns
        ordered: list[int | None] = []
        for value in (
            message_start_ns,
            self.first_token_ns,
            self.first_text_ns,
            self.last_token_ns,
        ):
            if value is None:
                ordered.append(None)
                continue
            running = max(running, value)
            ordered.append(running)
        return StreamTiming(
            sent_at_utc=self.sent_at_utc,
            sent_at_ns=self.sent_at_ns,
            message_start_ns=ordered[0],
            first_token_ns=ordered[1],
            first_text_ns=ordered[2],
            last_token_ns=ordered[3],
            end_ns=max(running, end_ns),
            event_count=self.event_count,
        )

    def result(self, end_ns: int, error: RequestError | None) -> StreamResult:
        return StreamResult(
            timing=self.timing(end_ns),
            usage=_merge_usage(self._usage_start, self._usage_delta),
            stop_reason=self.stop_reason,
            blocks=self.blocks(),
            server_model=self.server_model,
            error=error,
        )

    def failed(
        self, kind: RequestErrorKind, message: str, status: int | None = None
    ) -> StreamResult:
        """失敗の値を、そこまでに集めた時刻と中身と一緒に返す。"""
        return self.result(
            time.perf_counter_ns(),
            RequestError(kind=kind, http_status=status, message=_excerpt(message, self._secret)),
        )


def _index_of(payload: Mapping[str, Any]) -> int | None:
    index = payload.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    return index


def _append_str(parts: list[str], value: Any) -> None:
    if isinstance(value, str):
        parts.append(value)


def _error_text(payload: Mapping[str, Any]) -> str:
    """`event: error` の中身を、1 行の文にする。"""
    error = payload.get("error")
    if isinstance(error, Mapping):
        kind = error.get("type")
        message = error.get("message")
        joined = ": ".join(str(part) for part in (kind, message) if isinstance(part, str))
        if joined:
            return joined
    return json.dumps(payload, ensure_ascii=False)


# --- クライアント -------------------------------------------------------


@runtime_checkable
class MessagesClient(Protocol):
    """1 つの要求をストリームで送り、結果を値として返す。例外を投げない。"""

    async def stream(self, request: MessagesRequest, timeout: TimeoutPolicy) -> StreamResult: ...


class HttpxMessagesClient:
    """httpx と httpx-sse で `/v1/messages` を話す `MessagesClient`。

    1 つの `httpx.AsyncClient` を持ち、接続を使い回す。`async with` で使うか、
    使い終わったら `aclose()` を呼ぶ。同時に 8 本のストリームを流せる。
    """

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr | None = None,
        *,
        max_connections: int = _DEFAULT_MAX_CONNECTIONS,
    ) -> None:
        self._base_url = str(base_url)
        self._api_key = api_key
        self._closed = False
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            limits=httpx.Limits(
                max_connections=max_connections, max_keepalive_connections=max_connections
            ),
            follow_redirects=False,
        )

    @classmethod
    def from_target(cls, target: TargetDef, api_key: SecretStr | None = None) -> Self:
        """対象サーバーの定義から作る。認証の情報の値は、呼ぶ側が環境変数から読む (1.8)。"""
        return cls(str(target.base_url), api_key)

    def __repr__(self) -> str:
        """認証の情報の値は、絶対に出さない (1.8)。"""
        auth = "set" if self._api_key is not None else "none"
        return f"{type(self).__name__}(base_url={self._base_url!r}, auth={auth!r})"

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
        self._closed = True
        await self._client.aclose()

    async def stream(self, request: MessagesRequest, timeout: TimeoutPolicy) -> StreamResult:
        """1 つの要求を送り、時刻と中身を返す。やり直しはしない (10.6)。"""
        if self._closed:  # 書き手の誤り。要求の失敗ではないので、例外にする
            raise RuntimeError("閉じた client では要求を送れない")

        body = build_request_body(request)
        headers = _auth_headers(self._api_key)
        httpx_timeout = httpx.Timeout(
            connect=timeout.connect_s,
            read=None,  # イベントの間隔は、この module 自身が測る
            write=timeout.connect_s,
            pool=timeout.connect_s,
        )

        sent_at_utc = datetime.now(UTC)
        sent_at_ns = time.perf_counter_ns()  # 送る直前に打つ
        secret = self._api_key.get_secret_value() if self._api_key is not None else None
        collector = _Collector(sent_at_utc, sent_at_ns, secret)
        total_deadline_ns = sent_at_ns + _to_ns(timeout.total_s)
        # 応答の頭を待つ時間も、「最初のイベントまで」に数える
        first_deadline_ns = sent_at_ns + _to_ns(timeout.first_event_s)

        result: StreamResult | None = None
        try:
            async with AsyncExitStack() as stack:
                try:
                    async with asyncio.timeout(_wait_s(first_deadline_ns, total_deadline_ns)):
                        event_source = await stack.enter_async_context(
                            aconnect_sse(
                                self._client,
                                "POST",
                                MESSAGES_PATH,
                                json=body,
                                headers=headers,
                                timeout=httpx_timeout,
                            )
                        )
                except TimeoutError:
                    kind = _timeout_kind(
                        first=True,
                        deadline_ns=first_deadline_ns,
                        total_deadline_ns=total_deadline_ns,
                    )
                    return collector.failed(kind, "応答の頭が届かなかった")
                result = await _read_stream(
                    collector, event_source, timeout, first_deadline_ns, total_deadline_ns
                )
        except httpx.HTTPError as exc:
            # 接続そのものの失敗。ストリームを閉じるときの失敗では、結果を捨てない
            if result is not None:
                return result
            return collector.failed("connect", f"{type(exc).__name__}: {exc}")
        return result


# --- 読み取り -----------------------------------------------------------


async def _read_stream(
    collector: _Collector,
    event_source: EventSource,
    timeout: TimeoutPolicy,
    first_deadline_ns: int,
    total_deadline_ns: int,
) -> StreamResult:
    """イベントを 1 つずつ読み、時刻を打ち、結果を組み立てる。例外を投げない。"""
    response = event_source.response
    if response.status_code >= 400:
        return await _http_error(collector, response, total_deadline_ns)

    iterator = event_source.aiter_sse()
    first = True
    deadline_ns = first_deadline_ns
    while True:
        wait_s = _wait_s(deadline_ns, total_deadline_ns)
        kind = _timeout_kind(
            first=first, deadline_ns=deadline_ns, total_deadline_ns=total_deadline_ns
        )
        if wait_s <= 0.0:
            return collector.failed(kind, "制限時間を超えた")
        try:
            async with asyncio.timeout(wait_s):
                sse = await anext(iterator)
                stamp_ns = time.perf_counter_ns()  # 解析の前に打つ
        except TimeoutError:
            return collector.failed(kind, f"{wait_s:.3f} 秒のあいだイベントが届かなかった")
        except StopAsyncIteration:
            break
        except httpx.HTTPError as exc:
            # ストリームの途中で切れた (`message_stop` を見ずに終わった)
            return collector.failed("protocol", f"{type(exc).__name__}: {exc}")

        first = False
        step = collector.handle(sse, stamp_ns)
        if step is _Step.DONE:
            await _drain(iterator)
            return collector.result(stamp_ns, None)
        if step is _Step.FAILED:
            return collector.result(stamp_ns, collector.error)
        # 次のイベントは、いま届いたイベントからの間隔で測る
        deadline_ns = stamp_ns + _to_ns(timeout.idle_s)

    return collector.failed("protocol", "ストリームが message_stop を送らずに終わった")


async def _drain(iterator: AsyncIterator[ServerSentEvent]) -> None:
    """`message_stop` のあとの残りを読み切る。時刻はもう打ち終えている。

    httpx は、終わりまで読まなかった応答の接続を捨てる。読み切ると keep-alive の
    接続が pool に戻り、次の要求で TCP の確立を省ける (最初のトークンまでの時間に
    毎回それが乗らない)。終わりが来なければ、短い猶予で諦める (そのときは、
    これまでどおり接続が捨てられるだけで、結果は変わらない)。
    """
    try:
        async with asyncio.timeout(_DRAIN_GRACE_S):
            for _ in range(_DRAIN_MAX_EVENTS):
                await anext(iterator)
    except (TimeoutError, StopAsyncIteration, httpx.HTTPError):
        return


async def _http_error(
    collector: _Collector, response: httpx.Response, total_deadline_ns: int
) -> StreamResult:
    """HTTP のエラーは、状態の番号と、本文の抜き書きを添えて返す。"""
    wait_s = max(_wait_s(total_deadline_ns, total_deadline_ns), _DRAIN_GRACE_S)
    try:
        async with asyncio.timeout(wait_s):
            raw = await response.aread()
        detail = raw.decode("utf-8", errors="replace")
    except TimeoutError:
        detail = "エラーの本文を読み切れなかった (制限時間)"
    except httpx.HTTPError as exc:
        detail = f"{type(exc).__name__}: {exc}"
    return collector.failed("http", detail, status=response.status_code)


def _to_ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


def _wait_s(deadline_ns: int, total_deadline_ns: int) -> float:
    """次のイベントを待てる秒数 (段階の期限と、全体の期限の早いほう)。"""
    return (min(deadline_ns, total_deadline_ns) - time.perf_counter_ns()) / 1_000_000_000


def _timeout_kind(*, first: bool, deadline_ns: int, total_deadline_ns: int) -> RequestErrorKind:
    """先に来た期限で、時間切れの種類を決める。"""
    if total_deadline_ns < deadline_ns:
        return "timeout_total"
    return "timeout_first" if first else "timeout_idle"
