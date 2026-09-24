"""計測の前提を確かめ、対象サーバーの上限と版を得る (task 2.7)。

対象サーバーに短い要求を 1 回送り、応答とトークン数があることを確かめる
(1.3)。満たされていない前提は、例外ではなく `PreflightOk | PreflightFailure`
の値として返す (1.4)。`GET /v1/models`、`GET /version`、`/metrics` の実行中の
要求数は、それぞれ独立に失敗してよく、失敗しても前提の不足にはしない
(design.md `client/probe`)。

もう 1 つの役目は、`calibrate` コマンド (5.1) が呼び出して表示するだけの
`count_input_tokens`。`POST /v1/messages/count_tokens` を試し、その口がない
対象サーバーでは `max_tokens=1` の要求で代える (research.md の Decision
「入力の長さは、文字数とトークン数の比で狙う」の Follow-up)。この関数だけは
計測者が直接呼ぶ道具なので、両方の数え方に失敗したら `ProbeError` を投げる。
ただし、代わりの要求が入力の長さの上限を超えて HTTP 400 で断られたときだけは、
`ProbeError` の派生型 `InputOverContextLimitError` を投げ、ほかの失敗と区別
できるようにする (issue #24: 数える対象が上限を超えていたことを、段階を飛ばす
扱いにするため。口の側の HTTP 400 は設定の誤りとして、今までどおり素の
`ProbeError`)。

`fits_context` と `is_context_limit_error` は、条件が入力の長さの上限に収まる
かどうかを判定する純粋な関数 (3.1、3.6、6.9 で使う)。上限を超えた応答の
メッセージは、理由として残すだけで解析しない (research.md「入力の長さの上限
と、その見つけ方」: サーバーの差し替えで壊れるため)。

依存するのは標準ライブラリと httpx、pydantic、`bench_harness.types`、
`bench_harness.client.messages`、`bench_harness.metrics` (実行中の要求数の
gauge だけ) だけ。ほかの `bench_harness` の module は読み込まない。
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, NonNegativeInt, SecretStr, ValidationError

from bench_harness.client.messages import ANTHROPIC_VERSION, MessagesClient, build_request_body
from bench_harness.metrics import HttpxMetricsScraper
from bench_harness.types import (
    ContentBlock,
    InputMessage,
    LogicalMetric,
    MessagesRequest,
    MetricsUnavailable,
    PreflightFailure,
    PreflightOk,
    RequestError,
    RequestErrorKind,
    StreamResult,
    TargetDef,
    TextBlockParam,
    TimeoutPolicy,
    Usage,
)

__all__ = [
    "InputOverContextLimitError",
    "ProbeError",
    "TokenCount",
    "count_input_tokens",
    "fits_context",
    "is_context_limit_error",
    "preflight",
]

_PROBE_PROMPT: Final[str] = "ping"
_PROBE_MAX_TOKENS: Final[int] = 16
"""前提の確認に送る要求の大きさ。入力は数トークン、出力の上限は小さくする。"""

_AUX_TIMEOUT_S: Final[float] = 5.0
"""`/v1/models`、`/version`、`count_tokens` に使う、短い制限時間。"""

_UNREACHABLE_KINDS: Final[frozenset[RequestErrorKind]] = frozenset({"connect", "timeout_first"})
"""「到達できない」に分類する種類。接続できない場合と、応答の頭さえ届かない場合。"""

_DEFAULT_TIMEOUT: Final[TimeoutPolicy] = TimeoutPolicy()
"""既定の制限時間 (凍結の型なので、関数の既定値として使い回してよい)。"""


# --- 前提の確認 -----------------------------------------------------------


async def preflight(
    client: MessagesClient,
    target: TargetDef,
    *,
    timeout: TimeoutPolicy = _DEFAULT_TIMEOUT,
    api_key: SecretStr | None = None,
    aux_timeout_s: float = _AUX_TIMEOUT_S,
) -> PreflightOk | PreflightFailure:
    """計測の前提を確かめる (design.md `client/probe`、1.3、1.4)。

    `client` で短い要求 (数トークンの入力、小さい `max_tokens`) を 1 回だけ
    送る。`timeout` は、まとまりの実行で使うのと同じ `TimeoutPolicy` を渡して
    よい。`api_key` は `client` が使うのと同じ値を渡すこと。`/v1/models`、
    `/version` への短い GET に、`Authorization: Bearer` と `x-api-key` の
    両方として付ける (値そのものは、この関数の返り値にも例外にも出ない)。
    `/metrics` の実行中の要求数は `bench_harness.metrics` の scraper 越しに
    読むため、認証のヘッダーは付かない (その scraper 自身が認証を持たない
    ため。対象サーバーが `/metrics` にも認証を求める構成では、値は常に
    `None` になる)。

    `RequestError` から、満たされていない前提への分類:

    - `kind` が `connect` (接続できない) または `timeout_first` (応答の頭
      さえ届かない) → `unreachable`
    - `kind` が `http` (HTTP のエラーの状態) → `http_error` (状態の番号と、
      クライアントがすでに切り詰め・秘密を伏せたメッセージを詳細に残す)
    - それ以外 (`timeout_idle`、`timeout_total`、`protocol`、`stream_error`)
      → 対象サーバーには届いたが、応答の途中がおかしかった場合なので
      `http_error` に寄せる (メッセージは理由として残すが、解析はしない。
      3.6 と同じ方針)

    応答が来て `error` が `None` でも、`usage` がなければ `no_usage`。
    `usage` はあっても、出力のトークンが 0、または中身のブロックが 1 つも
    ない (本文もツール呼び出しの名前もない) ときは `no_output`。
    """
    request = MessagesRequest(
        model=target.model,
        max_tokens=_PROBE_MAX_TOKENS,
        messages=[InputMessage(role="user", content=[TextBlockParam(text=_PROBE_PROMPT)])],
    )
    result = await client.stream(request, timeout)

    if result.error is not None:
        return _classify_request_error(result.error)
    if result.usage is None:
        return PreflightFailure(
            unmet="no_usage", detail="対象サーバーの応答にトークン数が含まれていない"
        )
    if _is_no_output(result.usage, result.blocks):
        return PreflightFailure(
            unmet="no_output",
            detail="対象サーバーが出力のトークンも中身も1つも返さなかった",
        )

    headers = _aux_headers(api_key)
    async with httpx.AsyncClient(
        base_url=str(target.base_url), timeout=aux_timeout_s, headers=headers
    ) as http_client:
        models_entry, server_version, running_requests = await asyncio.gather(
            _fetch_models_entry(http_client, target),
            _fetch_version(http_client),
            _fetch_running_requests(target, aux_timeout_s),
        )

    server_model = result.server_model
    if server_model is None and models_entry is not None:
        candidate_model = models_entry.get("id")
        server_model = candidate_model if isinstance(candidate_model, str) else None

    context_limit: int | None = None
    if models_entry is not None:
        context_limit = _valid_context_limit(models_entry.get("max_model_len"))
    if context_limit is None:
        context_limit = target.max_context_tokens

    try:
        return PreflightOk(
            server_model=server_model,
            server_version=server_version,
            context_limit=context_limit,
            running_requests=running_requests,
        )
    except ValidationError:
        # 主の確認 (`_valid_context_limit`、`_valid_running_requests`) をすり抜けた、
        # 想定していない値のときだけの安全網。前提そのものは満たされているので、
        # 補助の情報を諦めて `PreflightOk` を返す (例外を外に出さない)
        return PreflightOk(server_model=server_model, server_version=server_version)


def _classify_request_error(error: RequestError) -> PreflightFailure:
    if error.kind in _UNREACHABLE_KINDS:
        return PreflightFailure(
            unmet="unreachable", detail=f"対象サーバーに接続できない: {error.message}"
        )
    if error.kind == "http":
        return PreflightFailure(
            unmet="http_error",
            detail=f"対象サーバーが HTTP {error.http_status} を返した: {error.message}",
        )
    return PreflightFailure(
        unmet="http_error",
        detail=f"対象サーバーの応答の途中がおかしい ({error.kind}): {error.message}",
    )


def _is_no_output(usage: Usage, blocks: list[ContentBlock]) -> bool:
    """トークン数はあるが、出力の中身が実質空のとき。"""
    if usage.output_tokens <= 0:
        return True
    return not any(block.text or block.tool_name for block in blocks)


def _aux_headers(api_key: SecretStr | None) -> dict[str, str]:
    """`/v1/models`、`/version`、`count_tokens` に付ける認証のヘッダー。

    値は、どこにも溜めない (1.8、messages.py の `_auth_headers` と同じ形)。
    """
    headers = {"anthropic-version": ANTHROPIC_VERSION}
    if api_key is not None:
        value = api_key.get_secret_value()
        headers["authorization"] = f"Bearer {value}"
        headers["x-api-key"] = value
    return headers


async def _get_json(http_client: httpx.AsyncClient, path: str) -> Any:
    """短い GET を送り、200 で JSON として読めたときだけ返す。それ以外は `None`。"""
    try:
        response = await http_client.get(path)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None


async def _fetch_models_entry(
    http_client: httpx.AsyncClient, target: TargetDef
) -> dict[str, Any] | None:
    payload = await _get_json(http_client, "/v1/models")
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        return None
    entries = [entry for entry in data if isinstance(entry, dict)]
    return _select_model_entry(entries, target.model)


def _select_model_entry(entries: list[dict[str, Any]], model: str) -> dict[str, Any] | None:
    """`id` が `model` と一致する項目を選ぶ。一致がなく 1 件しかなければ、それを使う。"""
    for entry in entries:
        if entry.get("id") == model:
            return entry
    if len(entries) == 1:
        return entries[0]
    return None


def _valid_context_limit(value: Any) -> int | None:
    """`max_model_len` を、`PreflightOk.context_limit` (`ge=1`) の範囲に合わせて採用する。

    範囲の外 (0 以下) や、int でない値は採用しない。呼び出し側は、次の候補
    (`TargetDef.max_context_tokens`、それもなければ `None`) に進む。
    """
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return None


async def _fetch_version(http_client: httpx.AsyncClient) -> str | None:
    payload = await _get_json(http_client, "/version")
    if not isinstance(payload, dict):
        return None
    version = payload.get("version")
    return version if isinstance(version, str) else None


async def _fetch_running_requests(target: TargetDef, timeout_s: float) -> int | None:
    """`/metrics` の、始める前の実行中の要求の数を読む。

    得られなくても前提の不足にはしない (7.4 と同じ扱い)。`HttpxMetricsScraper`
    自身が認証のヘッダーを持たないので、ここでも付けない。
    """
    async with HttpxMetricsScraper.from_target(target, timeout_s=timeout_s) as scraper:
        snapshot = await scraper.snapshot()
    if isinstance(snapshot, MetricsUnavailable):
        return None
    value = snapshot.values.get(LogicalMetric.RUNNING_REQUESTS)
    if value is None:
        return None
    return _valid_running_requests(value)


def _valid_running_requests(value: float) -> int | None:
    """gauge の値を、`PreflightOk.running_requests` (`ge=0`) の範囲に合わせて採用する。

    NaN、無限大、負の値は gauge としてありえない (対象サーバーの一時的な
    おかしな値、または解析の誤り) ので `None` にする。`round()` は NaN に
    `ValueError`、無限大に `OverflowError` を投げるので、先に `math.isfinite`
    で弾く。
    """
    if not math.isfinite(value) or value < 0:
        return None
    return round(value)


# --- トークン数を数える (`calibrate` (5.1) が呼ぶ) -------------------------


class TokenCount(BaseModel):
    """`count_input_tokens` の結果。`calibrate` コマンド (5.1) が表示するだけに使う。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens: NonNegativeInt
    method: Literal["count_tokens", "one_token_request"]


class ProbeError(Exception):
    """`count_input_tokens` が、計測者に知らせるべき失敗のときに投げる。

    `preflight` と違い、この関数は計測者が `calibrate` コマンドから直接呼ぶ
    道具で、上にやり直す仕組みがないので、例外にしてよい。投げる場面は 2 つ:
    (1) `count_tokens` が「口がない」(404/405/501) 以外の失敗をしたとき (黙って
    代わりの数え方に落とすと、設定の誤りに気づけない)、(2) 口がなく、代わりの
    数え方 (`max_tokens=1` の要求) も失敗したとき。(2) のうち、入力の長さの
    上限を超えた HTTP 400 だけは、派生型の `InputOverContextLimitError` で区別する。
    """


class InputOverContextLimitError(ProbeError):
    """代わりの数え方 (`max_tokens=1` の要求) が、入力の長さの上限を超えて断られたとき。

    口 (`count_tokens`) がなく、代わりの要求を送ったところ、対象サーバーが上限を
    超える入力として HTTP 400 を返した場合だけ投げる (`is_context_limit_error` と
    同じ判定)。数える対象の入力が長すぎたという事実で、対象サーバーの故障や設定の
    誤りとは別なので、呼び出し側 (7.2 の agent) は、この派生型だけを段階を飛ばす
    扱いにできる。`ProbeError` の派生型なので、区別しない呼び出し側 (計測ランの進行、
    `calibrate`、同時処理の包みの計測) は、今までどおり `ProbeError` として扱う。

    口そのものが返した HTTP 400 は、この型ではない (口が入力の長さで断ったのか、
    要求の形の誤りなのかを、状態の番号だけでは区別できないので、設定の誤りとして
    素の `ProbeError` のままにする)。
    """


async def count_input_tokens(
    client: MessagesClient,
    target: TargetDef,
    request: MessagesRequest,
    *,
    timeout: TimeoutPolicy = _DEFAULT_TIMEOUT,
    api_key: SecretStr | None = None,
    aux_timeout_s: float = _AUX_TIMEOUT_S,
) -> TokenCount:
    """与えた要求の入力のトークン数を数える (`calibrate` (5.1) が呼び出して表示する)。

    まず `POST /v1/messages/count_tokens` を試す。送る本文は、`request` から
    実際に送る形を組み立てたあと、公式の `count_tokens` にない、生成だけに
    効く項目 (`max_tokens`、`temperature`、`top_p`、`top_k`、`stream`) を外した
    もの (`extra` の `chat_template_kwargs` などは、トークン化に効きうるので
    残す)。口がない (404/405/501)、または接続そのものができない対象サーバー
    では、`max_tokens=1` にした同じ要求を `client` で実際に送り、返ってきた
    入力のトークン数の全長 (`Usage.total_input_tokens`。キャッシュの内訳を
    含む) で代える (research.md の Decision「入力の長さは、文字数とトークン数
    の比で狙う」の Follow-up)。口はあるのに 404/405/501 以外の理由で失敗した
    ときは、代わりの数え方に落とさず `ProbeError` を投げる (設定の誤りに
    気づけるように)。代わりの数え方も失敗したときも同様だが、その失敗が入力の
    長さの上限を超えた HTTP 400 (`is_context_limit_error`) のときだけは、
    `ProbeError` の派生型 `InputOverContextLimitError` を投げて区別する。
    """
    tokens = await _try_count_tokens_endpoint(target, request, api_key, aux_timeout_s)
    if tokens is not None:
        return TokenCount(tokens=tokens, method="count_tokens")

    # 1.2: 更新後は model_validate(model_dump()) で検証し直す (tasks.md Implementation Notes)
    fallback_request = MessagesRequest.model_validate(
        request.model_copy(update={"max_tokens": 1}).model_dump()
    )
    result = await client.stream(fallback_request, timeout)
    if result.error is None and result.usage is not None:
        return TokenCount(tokens=result.usage.total_input_tokens, method="one_token_request")

    detail = result.error.message if result.error is not None else "トークン数が返らなかった"
    message = (
        "入力のトークン数を数える口 (count_tokens) がなく、"
        f"代わりの要求 (max_tokens=1) も失敗した: {detail}"
    )
    if is_context_limit_error(result):
        raise InputOverContextLimitError(message)
    raise ProbeError(message)


_COUNT_TOKENS_ONLY_GENERATION_FIELDS: Final[frozenset[str]] = frozenset(
    {"max_tokens", "temperature", "top_p", "top_k", "stream"}
)
"""`count_tokens` に送らない項目。

公式の Anthropic の `count_tokens` の形は `model`、`messages`、`system`、
`tools` (と `tool_choice`、`thinking`) だけで、出力の量やサンプリングには
関わらない。これらを送ると、厳密なサーバーに 400/422 で拒まれうる。`extra`
(`chat_template_kwargs` の `enable_thinking` など) は、トークン化そのものに
効きうる (チャットテンプレートの展開が変わる) ので、ここでは落とさず残す。
"""

_MISSING_ENDPOINT_STATUSES: Final[frozenset[int]] = frozenset({404, 405, 501})
"""この状態のときだけ「口がない」とみなし、代わりの数え方に進む。

それ以外の失敗 (接続できた上での 400、401、500 など) は、設定の誤りに気づける
よう、黙って代わりの数え方に落とさず `ProbeError` にする。
"""


def _count_tokens_body(request: MessagesRequest) -> dict[str, JsonValue]:
    """`count_tokens` に送る本文。実際に送る形から、生成だけに効く項目を外す。"""
    body = build_request_body(request)
    for field in _COUNT_TOKENS_ONLY_GENERATION_FIELDS:
        body.pop(field, None)
    return body


async def _try_count_tokens_endpoint(
    target: TargetDef,
    request: MessagesRequest,
    api_key: SecretStr | None,
    timeout_s: float,
) -> int | None:
    """`POST /v1/messages/count_tokens` を試す。

    404/405/501 (この口がない) と、接続そのものができない場合 (口の有無を
    判定できない) だけを `None` にして、呼び出し側の代わりの数え方へ進ませる。
    それ以外の状態の番号がある失敗は、ここで `ProbeError` にする (認証の値は
    含めない)。
    """
    body = _count_tokens_body(request)
    headers = _aux_headers(api_key)
    try:
        async with httpx.AsyncClient(
            base_url=str(target.base_url), timeout=timeout_s, headers=headers
        ) as http_client:
            response = await http_client.post("/v1/messages/count_tokens", json=body)
    except httpx.HTTPError:
        return None  # 接続そのものができない。口の有無を判定できないので、代わりの数え方に進む

    if response.status_code in _MISSING_ENDPOINT_STATUSES:
        return None
    if response.status_code != 200:
        raise ProbeError(f"count_tokens が失敗した (HTTP {response.status_code})")
    try:
        payload = response.json()
    except ValueError:
        raise ProbeError("count_tokens の応答が JSON として読めなかった (HTTP 200)") from None
    tokens = payload.get("input_tokens") if isinstance(payload, dict) else None
    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
        return tokens
    raise ProbeError(
        "count_tokens の応答に、0 以上の整数の input_tokens が含まれていなかった (HTTP 200)"
    )


# --- 上限に収まるかどうか (3.1、3.6、6.9) -----------------------------------


def fits_context(context_limit: int | None, input_tokens: int, max_tokens: int) -> bool | None:
    """条件が入力の長さの上限に収まるか。上限がわからなければ `None`。

    `None` のときは、呼び出し側が送ってから `is_context_limit_error` で
    HTTP 400 を扱う (design.md `client/probe`: 「上限がわからないときは送り、
    HTTP 400 が返ったら、その条件を『上限に達した』として飛ばす」)。
    """
    if context_limit is None:
        return None
    return input_tokens + max_tokens <= context_limit


def is_context_limit_error(result: StreamResult) -> bool:
    """上限を超えたことを表す応答かどうか (3.6、6.9)。

    HTTP 400 かどうかだけを見る。メッセージは理由として残すが、正規表現などで
    は解析しない (research.md「入力の長さの上限と、その見つけ方」: 対象サーバー
    の差し替えで壊れるため)。
    """
    error = result.error
    return error is not None and error.kind == "http" and error.http_status == 400
