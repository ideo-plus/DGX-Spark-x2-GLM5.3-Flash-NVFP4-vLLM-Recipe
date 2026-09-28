"""試験専用の偽の推論サーバー (task 1.4)。

実際のソケット (127.0.0.1 の空いている port) で待ち受け、`/v1/messages` の
ストリーム、`/v1/messages/count_tokens`、`/v1/models`、`/version`、`/metrics`
に応える。標準ライブラリの `http.server.ThreadingHTTPServer` だけで書いてある
ので、依存は増えない。

作りの約束:

- SSE のイベントは 1 つずつ HTTP の chunk として書き出し、その都度 socket に
  流す。だから「イベントの間の遅れ」がクライアント側からそのまま観測できる
- 遅れは、前のイベントの予定の時刻からの積み上げで決める (ずれが溜まらない)。
  OS の待ちは 10 ミリ秒ほど長引くことがあるので、最後のわずかは回して待つ。
  この Mac での実測は、イベントの間の遅れの誤差が、同時 8 本でも 1 ミリ秒未満
- 遅れを数え始めるのは、HTTP の頭を書いたあと。クライアントから見た最初の
  イベントまでの時間には、要求を送って届くまでの数ミリ秒が上乗せされる
- 接続ごとに thread を 1 つ使う (daemon)。同時に 8 本以上のストリームを扱える
- 止まった (stall) ストリームがあっても `stop()` は待たされない。`stop()` は
  停止の合図を立てるので、待っている handler はすぐ起きて接続を閉じる

## 使い方

```python
def test_example(fake_server: FakeServer) -> None:
    fake_server.set_response(token_stream_response(output_tokens=10, gap_s=0.03))
    ...  # fake_server.base_url に httpx で要求を送る
    assert fake_server.call_count("/v1/messages") == 1
```

`conftest.py` の fixture `fake_server` (1 台) と `fake_server_factory`
(何台でも) が、起動と後始末をする。

## 応答の台本 (`Script`)

1 つの要求への応答は、凍結の dataclass `Script` で表す。項目を変えるときは
`dataclasses.replace` を使う (この module から `replace` を再輸出している)。

```python
set_response(replace(text_response("ok"), gap_s=0.05, stop_reason="max_tokens"))
```

台本の渡し方は 3 つある。あとから渡したものが、前のものを取り消す。

- `set_response(script)`: すべての要求に、同じ台本で応える
- `set_response_sequence([a, b, c])`: 要求ごとに順に使い、尽きたら最後のものを
  繰り返す。「n 回目から失敗し続ける」を作れる (3.5 の連続の失敗での停止)

  ```python
  set_response_sequence([text_response("ok"), http_error_response(503)])
  ```

- `set_response_factory(fn)`: 要求の本文 (解析済みの写像) を受け取り、その要求
  への台本を返す関数を登録する (型は `ResponseFactory`)。要求ごとに違う応答を
  返せる。1 本だけ失敗させる (3.4)、課題に合った正しいツール呼び出しを返す、
  9 種類の分類を出し分ける (6.7) のに使う

  ```python
  def factory(body: dict[str, Any]) -> Script:
      text = str(body["messages"][-1]["content"])
      return tool_use_response("read_file", {"path": text})

  set_response_factory(factory)
  ```

- `events`: `message_start` と `message_delta` の間に流すイベントの並び。
  `SseEvent(type, data)` をそのまま書けるので、知らない種類のイベントも、
  知らない項目も、壊れた JSON も送れる。`SseEvent(type, raw_data="...")` は
  `data:` の行を文字どおり流す
- `include_message_start` / `include_message_delta` / `include_message_stop`:
  自動で付ける前後のイベントの有無。`False` にして `events` に自分で書けば、
  中身を完全に指定できる
- `stop_reason`: `message_delta` の終わり方 (`end_turn`、`max_tokens`、
  `tool_use` など)
- `usage`: トークン数の載せ方 (下記)。`None` にすると、どこにも載らない
  (前提の確認の「トークン数がない」の場合)
- `header_delay_s` / `first_delay_s` / `gap_s`: HTTP の頭までの遅れ、最初の
  イベントまでの遅れ (prefill を模す)、イベントの間の遅れ (decode を模す)。
  イベントごとの遅れは `SseEvent(..., delay_s=...)` で上書きできる
- `http_status` / `http_body`: ストリームを始めずに HTTP のエラーを返す
- `abort_after_events`: N 個のイベントを送ったあと、接続を切る
- `stall_after_events` / `stall_s`: N 個のイベントを送ったあと、何も送らない

イベントの数は、自動で付く `message_start` も数える。

## よく使う台本

- `text_response("本文")`: 本文のブロック 1 つ
- `token_stream_response(output_tokens=10, gap_s=0.03)`: 10 トークンを 30 ミリ秒
  おきに流す (生成速度の試験)
- `tool_use_response("read_file", {"path": "/tmp/a"})`: ツール呼び出し。
  `fragments=("{\\"pa", "th\\": 1}")` で引数の断片を文字どおり指定でき、
  `text_after="..."` でツール呼び出しのあとに本文のブロックが続く
- `thinking_response("考え中", answer="答え")`: thinking のブロックと本文
- `empty_response()`: ブロックが 1 つもない応答
- `error_stream_response(before=text_events("途中"))`: 途中で `event: error`
- `http_error_response(500)` / `over_limit_response(32000)`: HTTP のエラー
- `dropped_response(after_events=2)`: 途中で接続が切れる
- `stalled_response(after_events=1)`: 途中で止まる

イベントを自分で並べるときは `text_events` / `thinking_events` /
`tool_use_events` / `ping_event` / `error_event` が使える。

## トークン数 (`UsageSpec`)

実物の vLLM に合わせて、次のように載せる。

- `message_start` には `input_tokens` (入力の全長) と `output_tokens: 0`
- `message_delta` には `input_tokens` (全長からキャッシュの内訳を引いた値)、
  `output_tokens`、および内訳 (`cache_read_input_tokens`、
  `cache_creation_input_tokens`) を、指定があるときだけ
- `UsageSpec.input_tokens` が `None` なら、要求の本文から計算する (下記)。
  `output_tokens` が `None` なら、台本の `content_block_delta` の数にする
- `in_message_start` / `in_message_delta` で、載せる先を選べる

## 入力のトークン数の決め方

`system`、`messages`、`tools` だけを取り出して JSON にし
(`sort_keys=True`、`ensure_ascii=False`)、その文字数を
`chars_per_token` (既定 4.0) で割って四捨五入する (最低 1)。決まった規則なので、
入力が 2 倍になればトークン数もほぼ 2 倍になる。
`/v1/messages/count_tokens` も同じ値を返す。

`set_chars_per_token(v)` で比を変えられる。JSON の記号 (`{"role": "user", ...`)
も文字数に入るので、狙ったトークン数にきっちり合わせたい試験は、この比で
調整すること。

`set_input_token_counter(fn)` で、この決め方そのものを差し替えられる。差し替えた
関数は `/v1/messages`、`/v1/messages/count_tokens`、上限の判定のすべてに効く
(`_count_input_tokens` を経由するため)。

## 口の有無と、対象サーバーの申告

- `set_count_tokens_enabled(False)`: `POST /v1/messages/count_tokens` を 404 に
  する (この口を持たない対象サーバーの場合。2.7 の代わりの数え方の試験)
- `set_version(None)`: `GET /version` を 404 にする。文字列を渡すと、その版を
  返す (既定は `"0.0.0-fake"`)
- `set_model("glm-5.3-flash")`: `GET /v1/models` が返す名前と、ストリームの
  `message_start` に載る名前を変える (`Script.model` は 1 つの台本だけ変える)

## 入力の長さの上限

`set_context_limit(n)` を呼ぶと、`GET /v1/models` の `max_model_len` に `n` を
載せ、`入力のトークン数 + max_tokens > n` の要求に HTTP 400 を返す。
`advertise=False` を渡すと、上限を守らせるが `/v1/models` には載せない
(定義に書いた上限へ落ちることを確かめる場合)。`enforce=False` はその逆。

## プレフィックスキャッシュの真似 (既定は切)

`set_prefix_cache(True)` にすると、前に来た要求と先頭が同じところまでを
`cache_read_input_tokens` として返し、最初のイベントまでの遅れを
`first_delay_s * (1 - 当たった割合)` に縮め、`/metrics` の当たりの数を増やす。
当たりの長さは、先頭が一致した文字数をトークン数に直し、16 トークンの倍数に
切り下げた値 (ブロックの粒度の真似)。切のときは、当たりは常に 0 で、
`cache_read_input_tokens` は載せない。

`UsageSpec.cache_read_input_tokens` を直に指定した場合は、応答に載る値だけが
変わる。遅れと `/metrics` には効かない (2 つの knob を混ぜないため)。

## `/metrics`

vLLM の V1 の名前で、Prometheus のテキストを返す。要求を 1 つ捌くたびに、
入力と生成のトークン、ステップの数、投機的デコードの下書きと当たり
(`acceptance_ratio` の割合)、プレフィックスキャッシュの問い合わせと当たりが
増える。`vllm:num_requests_running` は、いま処理中の `/v1/messages` の本数。

- `set_metrics_enabled(False)`: 404 を返す
- `set_metrics_text("...")`: テキストをそのまま返す
- `set_metrics_options(include_spec_decode=False)`: 投機的デコードの一族を外す
- `set_metrics_options(rename={"vllm:prefix_cache_hits_total": "custom:hits"})`:
  名前を変える (`metric_map` の上書きの試験用)
- `bump_preemptions(n)` / `set_kv_usage(v)`: 追い出しの回数と KV の使用率

## 記録

`requests` に、届いた要求が順に入る (`RecordedRequest`)。本文 (解析済み)、
ヘッダー (鍵は小文字)、経路、届いた時刻 (`time.monotonic_ns()`)、届いた時点で
処理中だった `/v1/messages` の本数、計算した入力のトークン数を持つ。
`requests_for(path)`、`call_count(path)`、`wait_for_requests(n)` がある。
`errors` には、handler の中で起きた想定外の例外が入る (空であるべき)。

`reset()` は、記録、`/metrics` の増えていく値、プレフィックスキャッシュの記憶、
`set_response_sequence` の位置、`errors` を消す。台本と口の設定は残る。1 つの
試験の中で、条件を区切って数え直したいときに使う。

## 気をつけること

- `stop()` のあとに `start()` をすると、**別の port になる**。同じ URL のまま
  復活させる試験は書けない (「対象サーバーが落ちて、また上がる」を測るなら、
  新しい `FakeServer` を立てて、対象サーバーの定義ごと差し替えること)
- 遅れの終わりの回し待ち (`_SPIN_S`) は、コアの数が十分ある機械を前提にして
  いる。実測では 16 コアで同時 16 本まで飢えは起きなかったが、コアの少ない
  機械や、`pytest-xdist` と併せて使う場合は、遅れの精度が落ちうる
- プレフィックスキャッシュの真似は、過去の**全**要求と先頭を突き合わせる。
  その比較はロックを持ったまま走るので、12 万トークンの入力を何百回も流す
  試験では、条件ごとに `reset()` して記憶を空にすること
"""

from __future__ import annotations

import contextlib
import json
import socket
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final, cast
from urllib.parse import urlsplit

__all__ = [
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_MODEL",
    "FakeServer",
    "InputTokenCounter",
    "MetricsOptions",
    "RecordedRequest",
    "ResponseFactory",
    "Script",
    "SseEvent",
    "UsageSpec",
    "dropped_response",
    "empty_response",
    "error_event",
    "error_stream_response",
    "estimator_aligned_counter",
    "http_error_response",
    "over_limit_response",
    "ping_event",
    "replace",
    "stalled_response",
    "text_events",
    "text_response",
    "thinking_events",
    "thinking_response",
    "token_stream_response",
    "tool_use_events",
    "tool_use_response",
]

JsonDict = dict[str, Any]

InputTokenCounter = Callable[[JsonDict | None], int]
"""要求の本文から入力のトークン数を決める関数 (`set_input_token_counter` で差し替える)。"""

DEFAULT_MODEL: Final[str] = "fake-model"
DEFAULT_CHARS_PER_TOKEN: Final[float] = 4.0
_CACHE_BLOCK_TOKENS: Final[int] = 16
_SPIN_S: Final[float] = 0.012
"""遅れの最後のこのぶんは、OS の待ちに任せず、回して待つ (`_wait_until` を参照)。"""


# --- 台本の型 -----------------------------------------------------------


@dataclass(frozen=True)
class SseEvent:
    """SSE の 1 つのイベント。`event:` の行と `data:` の行になる。"""

    type: str
    data: JsonDict | None = None
    raw_data: str | None = None
    """`data:` の行を文字どおり流す (壊れた JSON を送る場合)。"""
    delay_s: float | None = None
    """このイベントを送る前の遅れ。`None` なら台本の既定を使う。"""

    def encode(self) -> bytes:
        if self.raw_data is not None:
            payload = self.raw_data
        else:
            payload = json.dumps(
                self.data if self.data is not None else {"type": self.type}, ensure_ascii=False
            )
        return f"event: {self.type}\ndata: {payload}\n\n".encode()


@dataclass(frozen=True)
class UsageSpec:
    """トークン数の載せ方。`None` の項目は、要求から計算するか、載せない。"""

    input_tokens: int | None = None
    """入力の全長。`None` なら要求の本文から計算する。"""
    output_tokens: int | None = None
    """`None` なら台本の `content_block_delta` の数にする。"""
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    in_message_start: bool = True
    in_message_delta: bool = True


@dataclass(frozen=True)
class Script:
    """1 つの要求への応答の台本。"""

    events: tuple[SseEvent, ...] = ()
    include_message_start: bool = True
    include_message_delta: bool = True
    include_message_stop: bool = True
    stop_reason: str | None = "end_turn"
    usage: UsageSpec | None = UsageSpec()
    model: str | None = None
    message_id: str = "msg_fake"
    header_delay_s: float = 0.0
    first_delay_s: float = 0.0
    gap_s: float = 0.0
    http_status: int | None = None
    http_body: JsonDict | None = None
    abort_after_events: int | None = None
    stall_after_events: int | None = None
    stall_s: float = 5.0


ResponseFactory = Callable[[JsonDict], Script]
"""要求の本文 (解析済み) を受け取り、その要求への台本を返す関数。"""


@dataclass(frozen=True)
class RecordedRequest:
    """届いた要求の記録。"""

    path: str
    method: str
    headers: Mapping[str, str]
    """鍵は小文字 (`authorization`、`x-api-key`)。"""
    body: JsonDict | None
    raw_body: bytes
    arrived_at_ns: int
    """届いた時刻 (`time.monotonic_ns()`)。"""
    in_flight: int
    """届いた時点で処理中だった `/v1/messages` の本数 (自分を含む)。"""
    input_tokens: int


@dataclass(frozen=True)
class MetricsOptions:
    """`/metrics` の出し方。"""

    include_spec_decode: bool = True
    acceptance_ratio: float = 0.8
    rename: Mapping[str, str] = field(default_factory=dict)
    model_label: str = DEFAULT_MODEL


@dataclass
class _Counters:
    """`/metrics` が返す、増えていく値。"""

    prompt_tokens: int = 0
    generation_tokens: int = 0
    iteration_sum: int = 0
    iteration_count: int = 0
    spec_drafts: int = 0
    spec_draft_tokens: int = 0
    spec_accepted_tokens: int = 0
    prefix_queries: int = 0
    prefix_hits: int = 0
    preemptions: int = 0


@dataclass(frozen=True)
class _Plan:
    """1 つの要求に対して、実際に流すものを決めた結果。"""

    script: Script
    events: tuple[SseEvent, ...]
    first_delay_s: float
    total_input_tokens: int
    output_tokens: int
    cache_read_tokens: int


# --- 台本を組み立てる助け -----------------------------------------------


def ping_event() -> SseEvent:
    """`ping` のイベント (実物の vLLM は送らないが、混ざっても困らないことの試験用)。"""
    return SseEvent("ping", {"type": "ping"})


def error_event(error_type: str = "InternalServerError", message: str = "fake error") -> SseEvent:
    """ストリームの途中のエラー。"""
    return SseEvent("error", {"type": "error", "error": {"type": error_type, "message": message}})


def text_events(
    text: str, *, index: int = 0, chunks: Sequence[str] | None = None
) -> tuple[SseEvent, ...]:
    """本文のブロック 1 つ。`chunks` を渡すと、その切れ目のとおりに流す。"""
    pieces = tuple(chunks) if chunks is not None else (text,)
    return (
        SseEvent(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        *(
            SseEvent(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": piece},
                },
            )
            for piece in pieces
        ),
        SseEvent("content_block_stop", {"type": "content_block_stop", "index": index}),
    )


def thinking_events(
    text: str,
    *,
    index: int = 0,
    chunks: Sequence[str] | None = None,
    signature: str | None = None,
) -> tuple[SseEvent, ...]:
    """thinking のブロック 1 つ。"""
    pieces = tuple(chunks) if chunks is not None else (text,)
    deltas: list[SseEvent] = [
        SseEvent(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "thinking_delta", "thinking": piece},
            },
        )
        for piece in pieces
    ]
    if signature is not None:
        deltas.append(
            SseEvent(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": signature},
                },
            )
        )
    return (
        SseEvent(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        *deltas,
        SseEvent("content_block_stop", {"type": "content_block_stop", "index": index}),
    )


def tool_use_events(
    name: str,
    tool_input: Mapping[str, Any] | None = None,
    *,
    index: int = 0,
    fragments: Sequence[str] | None = None,
    tool_id: str = "toolu_fake",
) -> tuple[SseEvent, ...]:
    """ツール呼び出しのブロック 1 つ。

    `fragments` を渡すと、`input_json_delta` の中身をその文字列のとおりに流す
    (途中で切れた JSON や、壊れた JSON を送れる)。渡さなければ `tool_input` を
    JSON にして 1 つの断片で流す。
    """
    if fragments is None:
        fragments = (json.dumps(dict(tool_input or {}), ensure_ascii=False),)
    return (
        SseEvent(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
            },
        ),
        *(
            SseEvent(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": fragment},
                },
            )
            for fragment in fragments
        ),
        SseEvent("content_block_stop", {"type": "content_block_stop", "index": index}),
    )


def text_response(
    text: str = "ok",
    *,
    chunks: Sequence[str] | None = None,
    first_delay_s: float = 0.0,
    gap_s: float = 0.0,
    stop_reason: str | None = "end_turn",
) -> Script:
    """本文だけの応答。"""
    return Script(
        events=text_events(text, chunks=chunks),
        first_delay_s=first_delay_s,
        gap_s=gap_s,
        stop_reason=stop_reason,
    )


def thinking_response(
    text: str = "考え中",
    *,
    chunks: Sequence[str] | None = None,
    answer: str | None = None,
    first_delay_s: float = 0.0,
    gap_s: float = 0.0,
) -> Script:
    """thinking のブロックと、続く本文のブロック。"""
    events = thinking_events(text, chunks=chunks)
    if answer is not None:
        events = (*events, *text_events(answer, index=1))
    return Script(events=events, first_delay_s=first_delay_s, gap_s=gap_s)


def token_stream_response(
    *,
    output_tokens: int,
    token: str = "tok ",
    first_delay_s: float = 0.0,
    gap_s: float = 0.0,
    stop_reason: str | None = "max_tokens",
) -> Script:
    """`output_tokens` 個の本文の delta を、`gap_s` の間隔で流す。

    生成速度の試験に使う。`output_tokens` はトークン数としても報告される。
    """
    return Script(
        events=text_events(token * output_tokens, chunks=(token,) * output_tokens),
        first_delay_s=first_delay_s,
        gap_s=gap_s,
        stop_reason=stop_reason,
        usage=UsageSpec(output_tokens=output_tokens),
    )


def tool_use_response(
    name: str,
    tool_input: Mapping[str, Any] | None = None,
    *,
    fragments: Sequence[str] | None = None,
    text_after: str | None = None,
    tool_id: str = "toolu_fake",
    first_delay_s: float = 0.0,
    gap_s: float = 0.0,
    stop_reason: str | None = "tool_use",
) -> Script:
    """ツール呼び出しの応答。`text_after` を渡すと、そのあとに本文のブロックが続く。"""
    events = tool_use_events(name, tool_input, fragments=fragments, tool_id=tool_id)
    if text_after is not None:
        events = (*events, *text_events(text_after, index=1))
    return Script(events=events, first_delay_s=first_delay_s, gap_s=gap_s, stop_reason=stop_reason)


def empty_response(stop_reason: str | None = "end_turn") -> Script:
    """ブロックが 1 つもない応答 (空、または途中で切れた応答)。"""
    return Script(events=(), stop_reason=stop_reason, usage=UsageSpec(output_tokens=0))


def http_error_response(
    status: int = 500, error_type: str = "InternalServerError", message: str = "fake error"
) -> Script:
    """ストリームを始めずに返す HTTP のエラー。"""
    return Script(http_status=status, http_body=_error_body(error_type, message))


def over_limit_response(limit: int, *, input_tokens: int = 0, max_tokens: int = 0) -> Script:
    """入力の長さの上限を超えた要求への HTTP 400。

    ふつうは `set_context_limit()` に任せればよい。この関数は、上限を
    そのつど変えずに 400 を返させたいときに使う。
    """
    return Script(
        http_status=400,
        http_body=_error_body(
            "BadRequestError", _over_limit_message(limit, input_tokens, max_tokens)
        ),
    )


def error_stream_response(
    *,
    before: Sequence[SseEvent] = (),
    error_type: str = "InternalServerError",
    message: str = "fake error",
    first_delay_s: float = 0.0,
    gap_s: float = 0.0,
) -> Script:
    """途中まで流してから `event: error` で終わる応答。"""
    return Script(
        events=(*before, error_event(error_type, message)),
        include_message_delta=False,
        include_message_stop=False,
        first_delay_s=first_delay_s,
        gap_s=gap_s,
    )


def dropped_response(
    *, after_events: int = 2, base: Script | None = None, gap_s: float = 0.0
) -> Script:
    """`after_events` 個のイベントを送ったあと、接続を切る応答。"""
    script = base if base is not None else token_stream_response(output_tokens=8, gap_s=gap_s)
    return replace(script, abort_after_events=after_events)


def stalled_response(
    *,
    after_events: int = 1,
    stall_s: float = 5.0,
    base: Script | None = None,
    gap_s: float = 0.0,
) -> Script:
    """`after_events` 個のイベントを送ったあと、何も送らなくなる応答。"""
    script = base if base is not None else token_stream_response(output_tokens=8, gap_s=gap_s)
    return replace(script, stall_after_events=after_events, stall_s=stall_s)


# --- /metrics -----------------------------------------------------------


_METRIC_HELP: Final[dict[str, tuple[str, str]]] = {
    "vllm:num_requests_running": ("gauge", "Number of requests in model execution batches."),
    "vllm:kv_cache_usage_perc": ("gauge", "KV-cache usage. 1 means 100 percent usage."),
    "vllm:num_preemptions_total": ("counter", "Cumulative number of preemptions from the engine."),
    "vllm:prompt_tokens_total": ("counter", "Number of prefill tokens processed."),
    "vllm:generation_tokens_total": ("counter", "Number of generation tokens processed."),
    "vllm:iteration_tokens_total": ("histogram", "Histogram of number of tokens per engine_step."),
    "vllm:prefix_cache_queries_total": (
        "counter",
        "Prefix cache queries, in terms of number of queried tokens.",
    ),
    "vllm:prefix_cache_hits_total": (
        "counter",
        "Prefix cache hits, in terms of number of cached tokens.",
    ),
    "vllm:spec_decode_num_drafts_total": ("counter", "Number of spec decoding drafts."),
    "vllm:spec_decode_num_draft_tokens_total": ("counter", "Number of draft tokens."),
    "vllm:spec_decode_num_accepted_tokens_total": ("counter", "Number of accepted tokens."),
}
_SPEC_DECODE_NAMES: Final[frozenset[str]] = frozenset(
    name for name in _METRIC_HELP if "spec_decode" in name
)


def _render_metrics(
    counters: _Counters, options: MetricsOptions, *, running: int, kv_usage: float
) -> str:
    """vLLM の V1 の名前で、Prometheus のテキストを組み立てる。"""
    label = f'{{model_name="{options.model_label}"}}'
    values: dict[str, float] = {
        "vllm:num_requests_running": float(running),
        "vllm:kv_cache_usage_perc": kv_usage,
        "vllm:num_preemptions_total": float(counters.preemptions),
        "vllm:prompt_tokens_total": float(counters.prompt_tokens),
        "vllm:generation_tokens_total": float(counters.generation_tokens),
        "vllm:prefix_cache_queries_total": float(counters.prefix_queries),
        "vllm:prefix_cache_hits_total": float(counters.prefix_hits),
        "vllm:spec_decode_num_drafts_total": float(counters.spec_drafts),
        "vllm:spec_decode_num_draft_tokens_total": float(counters.spec_draft_tokens),
        "vllm:spec_decode_num_accepted_tokens_total": float(counters.spec_accepted_tokens),
    }
    lines: list[str] = []
    for name, (kind, help_text) in _METRIC_HELP.items():
        if name in _SPEC_DECODE_NAMES and not options.include_spec_decode:
            continue
        shown = options.rename.get(name, name)
        lines.append(f"# HELP {shown} {help_text}")
        lines.append(f"# TYPE {shown} {kind}")
        if kind == "histogram":
            count = float(counters.iteration_count)
            lines.append(f'{shown}_bucket{{le="1.0",model_name="{options.model_label}"}} 0.0')
            lines.append(f'{shown}_bucket{{le="8.0",model_name="{options.model_label}"}} 0.0')
            lines.append(f'{shown}_bucket{{le="+Inf",model_name="{options.model_label}"}} {count}')
            lines.append(f"{shown}_sum{label} {float(counters.iteration_sum)!r}")
            lines.append(f"{shown}_count{label} {count!r}")
        else:
            lines.append(f"{shown}{label} {values[name]!r}")
    return "\n".join(lines) + "\n"


def estimator_aligned_counter(
    *, fixed: int, per_message: int = 0, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
) -> InputTokenCounter:
    """会話の組み立ての見積もりと同じ文字列を同じ比で数え、包みを足す数え方。

    数えるのは `corpus/conversation.py` の `_preamble_tokens` / `_build_round` が
    見積もる文字列そのもの (system、tools の JSON、text、tool_use の名前と引数の
    JSON、tool_result の本文)。`Profile.chars_per_token` を全種 `chars_per_token` に
    そろえた試験では、見積もりとの差が `fixed + per_message × 発話数` だけになる。
    `fixed` が要求の包み (テンプレート、識別子の割れ方)、`per_message` が発話ごとの
    役割の目印を模す。
    """

    def compact_json(value: object) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def count(body: JsonDict | None) -> int:
        if not body:
            return 1
        chars = 0
        system = body.get("system")
        if isinstance(system, str):
            chars += len(system)
        tools = body.get("tools")
        if tools is not None:
            chars += len(compact_json(tools))
        messages = body.get("messages") or []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                chars += len(content)
                continue
            for block in content or []:
                kind = block.get("type")
                if kind == "text":
                    chars += len(block.get("text", ""))
                elif kind == "tool_use":
                    chars += len(block.get("name", "")) + len(compact_json(block.get("input", {})))
                elif kind == "tool_result":
                    chars += len(block.get("content", ""))
        return fixed + per_message * len(messages) + round(chars / chars_per_token)

    return count


# --- サーバー本体 -------------------------------------------------------


class FakeServer:
    """試験用の偽の推論サーバー。`start()` で待ち受け、`stop()` で止める。"""

    def __init__(
        self, *, model: str = DEFAULT_MODEL, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
    ) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._model = model
        self._chars_per_token = chars_per_token
        self._input_token_counter: InputTokenCounter | None = None
        self._script: Script = text_response("ok")
        self._factory: ResponseFactory | None = None
        self._sequence: tuple[Script, ...] = ()
        self._sequence_pos = 0
        self._requests: list[RecordedRequest] = []
        self._in_flight = 0
        self._context_limit: int | None = None
        self._advertise_limit = True
        self._enforce_limit = True
        self._count_tokens_enabled = True
        self._tokenize_response: tuple[int, JsonDict | str, float] | None = None
        self._version: str | None = "0.0.0-fake"
        self._metrics_enabled = True
        self._metrics_text: str | None = None
        self._metrics_options = MetricsOptions(model_label=model)
        self._counters = _Counters()
        self._prefix_cache = False
        self._cache_block_tokens = _CACHE_BLOCK_TOKENS
        self._seen_prefixes: list[str] = []
        self._kv_usage: float | None = None
        self.errors: list[str] = []
        """handler の中で起きた、想定していない例外の一覧。"""
        self._shutdown = threading.Event()
        self._server: _HttpServer | None = None
        self._thread: threading.Thread | None = None

    # --- 起動と停止 ---

    def start(self) -> None:
        """空いている port で待ち受けを始める。"""
        if self._server is not None:
            raise RuntimeError("すでに起動している")
        self._shutdown.clear()
        server = _HttpServer(("127.0.0.1", 0), _Handler, self)
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.02},
            name="fake-server",
            daemon=True,
        )
        thread.start()
        self._server = server
        self._thread = thread

    def stop(self) -> None:
        """待ち受けを止める。止まっているストリームがあっても待たされない。"""
        self._shutdown.set()
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5.0)

    def __enter__(self) -> FakeServer:
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

    # --- 台本 ---

    def set_response(self, script: Script) -> None:
        """すべての要求に、同じ台本で応える。"""
        with self._lock:
            self._script = script
            self._factory = None
            self._sequence = ()
            self._sequence_pos = 0

    def set_response_sequence(self, scripts: Sequence[Script]) -> None:
        """要求ごとに、順に台本を使う。尽きたら最後のものを繰り返す。"""
        if not scripts:
            raise ValueError("台本が空")
        with self._lock:
            self._sequence = tuple(scripts)
            self._sequence_pos = 0
            self._factory = None

    def set_response_factory(self, factory: ResponseFactory) -> None:
        """要求の本文を見て、その要求への台本を決める関数を登録する。"""
        with self._lock:
            self._factory = factory
            self._sequence = ()
            self._sequence_pos = 0

    # --- 口の有無と、対象サーバーの申告 ---

    def set_model(self, model: str) -> None:
        with self._lock:
            self._model = model

    def set_context_limit(
        self, limit: int | None, *, advertise: bool = True, enforce: bool = True
    ) -> None:
        """入力の長さの上限を決める。`advertise` は `/v1/models` に載せるかどうか。"""
        with self._lock:
            self._context_limit = limit
            self._advertise_limit = advertise
            self._enforce_limit = enforce

    def set_count_tokens_enabled(self, enabled: bool) -> None:
        """`POST /v1/messages/count_tokens` の有無 (切ると 404)。"""
        with self._lock:
            self._count_tokens_enabled = enabled

    def set_tokenize_response(
        self, status: int, payload: JsonDict | str, *, delay_s: float = 0.0
    ) -> None:
        """出力文字列の再計数応答を指定する。未指定なら /tokenize は 404。"""
        with self._lock:
            self._tokenize_response = (status, payload, delay_s)

    def set_version(self, version: str | None) -> None:
        """`GET /version` が返す版 (`None` なら 404)。"""
        with self._lock:
            self._version = version

    def set_chars_per_token(self, chars_per_token: float) -> None:
        """入力のトークン数を決める、1 トークンあたりの文字数。"""
        if chars_per_token <= 0:
            raise ValueError("chars_per_token は 0 より大きい必要がある")
        with self._lock:
            self._chars_per_token = chars_per_token

    def set_input_token_counter(self, counter: InputTokenCounter | None) -> None:
        """入力のトークン数の数え方を差し替える (`None` で既定の JSON の文字数 ÷ 比に戻す)。

        `/v1/messages` の usage、`count_tokens` の答え、上限の判定のすべてに効く
        (`_count_input_tokens` を通るため)。プレフィックスキャッシュの真似の当たりの
        長さは、これまでどおり比で決める (2 つの knob を混ぜない)。
        """
        with self._lock:
            self._input_token_counter = counter

    def set_prefix_cache(self, enabled: bool, *, block_tokens: int = _CACHE_BLOCK_TOKENS) -> None:
        """プレフィックスキャッシュの真似の入り切り (既定は切)。"""
        with self._lock:
            self._prefix_cache = enabled
            self._cache_block_tokens = max(1, block_tokens)
            self._seen_prefixes = []

    # --- /metrics ---

    def set_metrics_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._metrics_enabled = enabled

    def set_metrics_text(self, text: str | None) -> None:
        """テキストをそのまま返す (`None` に戻すと、自動で組み立てる)。"""
        with self._lock:
            self._metrics_text = text

    def set_metrics_options(
        self,
        *,
        include_spec_decode: bool | None = None,
        acceptance_ratio: float | None = None,
        rename: Mapping[str, str] | None = None,
        model_label: str | None = None,
    ) -> None:
        """渡した項目だけを変える。"""
        with self._lock:
            current = self._metrics_options
            self._metrics_options = MetricsOptions(
                include_spec_decode=(
                    current.include_spec_decode
                    if include_spec_decode is None
                    else include_spec_decode
                ),
                acceptance_ratio=(
                    current.acceptance_ratio if acceptance_ratio is None else acceptance_ratio
                ),
                rename=dict(current.rename) if rename is None else dict(rename),
                model_label=current.model_label if model_label is None else model_label,
            )

    def bump_preemptions(self, count: int = 1) -> None:
        with self._lock:
            self._counters.preemptions += count

    def set_kv_usage(self, value: float | None) -> None:
        """KV の使用率。`None` なら、処理中の本数 × 0.05 を返す。"""
        with self._lock:
            self._kv_usage = value

    # --- 記録 ---

    @property
    def requests(self) -> list[RecordedRequest]:
        with self._lock:
            return list(self._requests)

    def requests_for(self, path: str) -> list[RecordedRequest]:
        with self._lock:
            return [record for record in self._requests if record.path == path]

    def call_count(self, path: str) -> int:
        with self._lock:
            return sum(1 for record in self._requests if record.path == path)

    def wait_for_requests(
        self, count: int, *, path: str | None = None, timeout_s: float = 5.0
    ) -> list[RecordedRequest]:
        """要求が `count` 件そろうまで待って、その一覧を返す。"""
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                matched = [
                    record for record in self._requests if path is None or record.path == path
                ]
                if len(matched) >= count:
                    return matched
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"要求が {count} 件そろわなかった (いまは {len(matched)} 件)"
                    )
                self._condition.wait(remaining)

    def reset(self) -> None:
        """記録、増えていく値、キャッシュの記憶を消す (台本と口の設定は残す)。"""
        with self._condition:
            self._requests = []
            self._counters = _Counters()
            self._seen_prefixes = []
            self._sequence_pos = 0
            self.errors = []
            self._condition.notify_all()

    # --- ここから下は handler が使う ---

    def _count_input_tokens(self, body: JsonDict | None) -> int:
        """`system`、`messages`、`tools` の JSON の文字数から、入力のトークン数を出す。"""
        with self._lock:
            counter = self._input_token_counter
            chars_per_token = self._chars_per_token
        if counter is not None:
            return max(1, counter(body))
        if not body:
            return 1
        payload = {key: body[key] for key in ("system", "messages", "tools") if key in body}
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return max(1, round(len(text) / chars_per_token))

    def _prefix_key(self, body: JsonDict | None) -> str:
        """先頭の一致を見るための文字列 (ツールの定義 → system → 発話の順)。"""
        if not body:
            return ""
        parts = [
            json.dumps(body.get("tools"), ensure_ascii=False, sort_keys=True),
            json.dumps(body.get("system"), ensure_ascii=False, sort_keys=True),
        ]
        messages = body.get("messages")
        if isinstance(messages, list):
            parts.extend(
                json.dumps(message, ensure_ascii=False, sort_keys=True) for message in messages
            )
        return "\x00".join(parts)

    def _record(
        self,
        *,
        path: str,
        method: str,
        headers: Mapping[str, str],
        raw_body: bytes,
        body: JsonDict | None,
        input_tokens: int,
        claim_in_flight: bool,
    ) -> RecordedRequest:
        with self._condition:
            if claim_in_flight:
                self._in_flight += 1
            record = RecordedRequest(
                path=path,
                method=method,
                headers=dict(headers),
                body=body,
                raw_body=raw_body,
                arrived_at_ns=time.monotonic_ns(),
                in_flight=self._in_flight,
                input_tokens=input_tokens,
            )
            self._requests.append(record)
            self._condition.notify_all()
            return record

    def _release_in_flight(self) -> None:
        with self._condition:
            self._in_flight = max(0, self._in_flight - 1)
            self._condition.notify_all()

    def _limits(self) -> tuple[int | None, bool]:
        with self._lock:
            return self._context_limit, self._enforce_limit

    def _next_script(self, body: JsonDict | None) -> Script:
        with self._lock:
            factory = self._factory
            if factory is None:
                if self._sequence:
                    index = min(self._sequence_pos, len(self._sequence) - 1)
                    self._sequence_pos += 1
                    return self._sequence[index]
                return self._script
        return factory(body or {})

    def _plan(self, script: Script, body: JsonDict | None) -> _Plan:
        """台本と要求から、実際に流すイベントとトークン数を決める。"""
        total_input = self._count_input_tokens(body)
        spec = script.usage
        if spec is not None and spec.input_tokens is not None:
            total_input = spec.input_tokens

        # 真似したキャッシュの当たりは、遅れの短縮と `/metrics` の当たりの数に効く。
        # 台本が `cache_read_input_tokens` を直に指定したときは、応答に載る値だけを
        # 差し替える (遅れと `/metrics` には効かせない)。
        cached = self._lookup_prefix_cache(body, total_input)
        if spec is not None and spec.cache_read_input_tokens is not None:
            cache_read_shown: int | None = spec.cache_read_input_tokens
        elif self._prefix_cache_enabled():
            cache_read_shown = cached
        else:
            cache_read_shown = None
        cache_creation = spec.cache_creation_input_tokens if spec is not None else None

        delta_count = sum(1 for event in script.events if event.type == "content_block_delta")
        output_tokens = delta_count
        if spec is not None and spec.output_tokens is not None:
            output_tokens = spec.output_tokens

        first_delay = script.first_delay_s
        if cached > 0 and total_input > 0:
            first_delay *= max(0.0, 1.0 - cached / total_input)

        with self._lock:
            model = script.model if script.model is not None else self._model

        events = self._render(
            script,
            model=model,
            total_input=total_input,
            output_tokens=output_tokens,
            cache_read=cache_read_shown,
            cache_creation=cache_creation,
        )
        return _Plan(
            script=script,
            events=events,
            first_delay_s=first_delay,
            total_input_tokens=total_input,
            output_tokens=output_tokens,
            cache_read_tokens=cached,
        )

    def _prefix_cache_enabled(self) -> bool:
        with self._lock:
            return self._prefix_cache

    def _lookup_prefix_cache(self, body: JsonDict | None, total_input: int) -> int:
        """前に来た要求と先頭が一致した長さを、トークン数で返す。"""
        with self._lock:
            if not self._prefix_cache:
                return 0
            key = self._prefix_key(body)
            best = 0
            for seen in self._seen_prefixes:
                limit = min(len(seen), len(key))
                common = 0
                while common < limit and seen[common] == key[common]:
                    common += 1
                best = max(best, common)
            self._seen_prefixes.append(key)
            block = self._cache_block_tokens
            chars_per_token = self._chars_per_token
        tokens = int(best / chars_per_token)
        tokens = (tokens // block) * block
        return max(0, min(tokens, max(0, total_input - 1)))

    def _render(
        self,
        script: Script,
        *,
        model: str,
        total_input: int,
        output_tokens: int,
        cache_read: int | None,
        cache_creation: int | None,
    ) -> tuple[SseEvent, ...]:
        """自動で付ける前後のイベントを足した、流すイベントの並びを作る。"""
        spec = script.usage
        events: list[SseEvent] = []
        if script.include_message_start:
            message: JsonDict = {
                "id": script.message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
            }
            if spec is not None and spec.in_message_start:
                message["usage"] = {"input_tokens": total_input, "output_tokens": 0}
            events.append(SseEvent("message_start", {"type": "message_start", "message": message}))
        events.extend(script.events)
        if script.include_message_delta:
            data: JsonDict = {
                "type": "message_delta",
                "delta": {"stop_reason": script.stop_reason, "stop_sequence": None},
            }
            if spec is not None and spec.in_message_delta:
                shown_input = total_input - (cache_read or 0) - (cache_creation or 0)
                usage: JsonDict = {
                    "input_tokens": max(0, shown_input),
                    "output_tokens": output_tokens,
                }
                if cache_read is not None:
                    usage["cache_read_input_tokens"] = cache_read
                if cache_creation is not None:
                    usage["cache_creation_input_tokens"] = cache_creation
                data["usage"] = usage
            events.append(SseEvent("message_delta", data))
        if script.include_message_stop:
            events.append(SseEvent("message_stop", {"type": "message_stop"}))
        return tuple(events)

    def _advance_counters(self, plan: _Plan) -> None:
        with self._lock:
            ratio = self._metrics_options.acceptance_ratio
            counters = self._counters
            counters.prompt_tokens += plan.total_input_tokens
            counters.generation_tokens += plan.output_tokens
            counters.iteration_count += plan.output_tokens
            counters.iteration_sum += plan.total_input_tokens + plan.output_tokens
            counters.spec_drafts += plan.output_tokens
            counters.spec_draft_tokens += plan.output_tokens
            counters.spec_accepted_tokens += int(plan.output_tokens * ratio)
            counters.prefix_queries += plan.total_input_tokens
            counters.prefix_hits += plan.cache_read_tokens

    def _models_payload(self) -> JsonDict:
        with self._lock:
            entry: JsonDict = {
                "id": self._model,
                "object": "model",
                "created": 0,
                "owned_by": "fake",
            }
            if self._advertise_limit and self._context_limit is not None:
                entry["max_model_len"] = self._context_limit
        return {"object": "list", "data": [entry]}

    def _version_payload(self) -> JsonDict | None:
        with self._lock:
            if self._version is None:
                return None
            return {"version": self._version}

    def _metrics_payload(self) -> str | None:
        with self._lock:
            if not self._metrics_enabled:
                return None
            if self._metrics_text is not None:
                return self._metrics_text
            running = self._in_flight
            kv = self._kv_usage if self._kv_usage is not None else min(1.0, running * 0.05)
            return _render_metrics(
                self._counters, self._metrics_options, running=running, kv_usage=kv
            )

    def _count_tokens_enabled_now(self) -> bool:
        with self._lock:
            return self._count_tokens_enabled

    def _wait(self, seconds: float) -> bool:
        """`seconds` だけ待つ。停止の合図が来たら、すぐ起きて `True` を返す。"""
        if seconds <= 0:
            return self._shutdown.is_set()
        return self._shutdown.wait(seconds)

    def _wait_until(self, deadline: float) -> bool:
        """`time.perf_counter()` が `deadline` を過ぎるまで待つ。

        macOS の待ちは 10 ミリ秒ほど長引くことがある。それではイベントの間の
        遅れが狂うので、最後の `_SPIN_S` だけは待たずに回して待つ。停止の合図が
        来たら、すぐ起きて `True` を返す。
        """
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return self._shutdown.is_set()
            if remaining > _SPIN_S:
                if self._shutdown.wait(remaining - _SPIN_S):
                    return True
            elif self._shutdown.is_set():
                return True
            else:
                time.sleep(0)  # ほかの thread に譲る


class _HttpServer(ThreadingHTTPServer):
    """接続ごとに daemon の thread を使う HTTP サーバー。"""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        fake: FakeServer,
    ) -> None:
        self.fake = fake
        super().__init__(address, handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
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
    def _fake(self) -> FakeServer:
        return cast("_HttpServer", self.server).fake

    def log_message(self, format: str, *args: Any) -> None:
        """試験の出力を汚さないよう、何も出さない。"""

    # --- 経路 ---

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        raw = self._read_body()
        body = _parse_json(raw)
        if path == "/v1/messages":
            self._handle_messages(raw, body)
        elif path == "/v1/messages/count_tokens":
            self._handle_count_tokens(path, raw, body)
        elif path == "/tokenize":
            self._handle_tokenize(raw, body)
        else:
            self._record(path, "POST", raw, body, 0, claim_in_flight=False)
            self._send_json(404, _error_body("NotFoundError", f"unknown path {path}"))

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        self._record(path, "GET", b"", None, 0, claim_in_flight=False)
        if path == "/v1/models":
            self._send_json(200, self._fake._models_payload())
        elif path == "/version":
            payload = self._fake._version_payload()
            if payload is None:
                self._send_json(404, _error_body("NotFoundError", "no version endpoint"))
            else:
                self._send_json(200, payload)
        elif path == "/metrics":
            text = self._fake._metrics_payload()
            if text is None:
                self._send_json(404, _error_body("NotFoundError", "no metrics endpoint"))
            else:
                self._send_text(200, text, "text/plain; version=0.0.4; charset=utf-8")
        else:
            self._send_json(404, _error_body("NotFoundError", f"unknown path {path}"))

    # --- /v1/messages ---

    def _handle_messages(self, raw: bytes, body: JsonDict | None) -> None:
        fake = self._fake
        input_tokens = fake._count_input_tokens(body)
        self._record("/v1/messages", "POST", raw, body, input_tokens, claim_in_flight=True)
        try:
            limit, enforce = fake._limits()
            max_tokens = _int_field(body, "max_tokens")
            if enforce and limit is not None and input_tokens + max_tokens > limit:
                self._send_json(
                    400,
                    _error_body(
                        "BadRequestError",
                        _over_limit_message(limit, input_tokens, max_tokens),
                    ),
                )
                return
            script = fake._next_script(body)
            if script.http_status is not None:
                if script.header_delay_s > 0:
                    fake._wait(script.header_delay_s)
                self._send_json(
                    script.http_status,
                    script.http_body
                    if script.http_body is not None
                    else _error_body("InternalServerError", "fake error"),
                )
                return
            plan = fake._plan(script, body)
            self._stream(plan)
            fake._advance_counters(plan)
        finally:
            fake._release_in_flight()

    def _handle_count_tokens(self, path: str, raw: bytes, body: JsonDict | None) -> None:
        fake = self._fake
        input_tokens = fake._count_input_tokens(body)
        self._record(path, "POST", raw, body, input_tokens, claim_in_flight=False)
        if not fake._count_tokens_enabled_now():
            self._send_json(404, _error_body("NotFoundError", "no count_tokens endpoint"))
            return
        self._send_json(200, {"input_tokens": input_tokens})

    def _handle_tokenize(self, raw: bytes, body: JsonDict | None) -> None:
        self._record("/tokenize", "POST", raw, body, 0, claim_in_flight=False)
        with self._fake._lock:
            response = self._fake._tokenize_response
        if response is None:
            self._send_json(404, _error_body("NotFoundError", "no tokenize endpoint"))
            return
        status, payload, delay_s = response
        if delay_s > 0:
            self._fake._wait(delay_s)
        if isinstance(payload, str):
            self._send_text(status, payload, "application/json")
        else:
            self._send_json(status, payload)

    def _stream(self, plan: _Plan) -> None:
        """イベントを 1 つずつ chunk にして流す。"""
        fake = self._fake
        script = plan.script
        if script.header_delay_s > 0 and fake._wait(script.header_delay_s):
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
        except OSError:
            return

        if self._break_stream(script, 0):
            return
        deadline = time.perf_counter()
        for index, event in enumerate(plan.events):
            if event.delay_s is not None:
                delay = event.delay_s
            else:
                delay = plan.first_delay_s if index == 0 else script.gap_s
            deadline += delay
            if fake._wait_until(deadline):
                return
            try:
                self._write_chunk(event.encode())
            except OSError:
                return
            if self._break_stream(script, index + 1):
                return
        with contextlib.suppress(OSError):
            self._write_raw(b"0\r\n\r\n")

    def _break_stream(self, script: Script, sent: int) -> bool:
        """接続を切る、または止まる場面かどうか。切った/止まったら `True`。"""
        if script.abort_after_events is not None and sent >= script.abort_after_events:
            self._drop_connection()
            return True
        if script.stall_after_events is not None and sent >= script.stall_after_events:
            self._fake._wait(script.stall_s)
            self._drop_connection()
            return True
        return False

    # --- 下回り ---

    def _read_body(self) -> bytes:
        length = _int_header(self.headers.get("Content-Length"))
        if length <= 0:
            return b""
        try:
            return self.rfile.read(length)
        except OSError:
            return b""

    def _header_map(self) -> dict[str, str]:
        return {key.lower(): value for key, value in self.headers.items()}

    def _record(
        self,
        path: str,
        method: str,
        raw: bytes,
        body: JsonDict | None,
        input_tokens: int,
        *,
        claim_in_flight: bool,
    ) -> RecordedRequest:
        return self._fake._record(
            path=path,
            method=method,
            headers=self._header_map(),
            raw_body=raw,
            body=body,
            input_tokens=input_tokens,
            claim_in_flight=claim_in_flight,
        )

    def _send_json(self, status: int, payload: JsonDict) -> None:
        self._send_text(status, json.dumps(payload, ensure_ascii=False), "application/json")

    def _send_text(self, status: int, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write_raw(body)
        except OSError:
            self.close_connection = True

    def _write_chunk(self, payload: bytes) -> None:
        self._write_raw(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")

    def _write_raw(self, payload: bytes) -> None:
        self.wfile.write(payload)
        self.wfile.flush()

    def _drop_connection(self) -> None:
        """終わりの印を書かずに接続を切る (途中で切れたストリームを模す)。"""
        self.close_connection = True
        with contextlib.suppress(OSError):
            self.connection.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self.connection.close()


# --- 小さな助け ---------------------------------------------------------


def _parse_json(raw: bytes) -> JsonDict | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _int_header(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _int_field(body: JsonDict | None, key: str) -> int:
    if not body:
        return 0
    value = body.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _error_body(error_type: str, message: str) -> JsonDict:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _over_limit_message(limit: int, input_tokens: int, max_tokens: int) -> str:
    """上限を超えたときの、実物の vLLM に合わせた言い回し。"""
    return (
        f"This model's maximum context length is {limit} tokens. "
        f"However, you requested {input_tokens + max_tokens} tokens "
        f"({input_tokens} in the messages, {max_tokens} in the completion). "
        "Please reduce the length of the messages or completion."
    )
