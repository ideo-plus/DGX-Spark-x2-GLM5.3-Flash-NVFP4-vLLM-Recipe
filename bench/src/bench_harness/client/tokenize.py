"""同一対象サーバーで思考・本文の生文字列を再計数する。"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import SecretStr

from bench_harness.types import (
    StreamResult,
    TargetDef,
    _CountReason,
    _OutputTokenCounts,
    _PhaseTokenCount,
)


def _unknown(reason: _CountReason) -> _PhaseTokenCount:
    return _PhaseTokenCount(reason=reason)


def _both_unknown(reason: _CountReason) -> _OutputTokenCounts:
    return _OutputTokenCounts(thinking=_unknown(reason), text=_unknown(reason))


def _phase_text(result: StreamResult, kind: str) -> str:
    return "".join(block.text or "" for block in result.blocks if block.type == kind)


def _validated_count(payload: Any, expected_model: str) -> int | None:
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    if model is not None and (not isinstance(model, str) or model != expected_model):
        return None
    count = payload.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    tokens = payload.get("tokens")
    if tokens is not None:
        if not isinstance(tokens, list) or len(tokens) != count:
            return None
        if any(isinstance(token, bool) or not isinstance(token, int) for token in tokens):
            return None
    return count


async def _count_one(
    client: httpx.AsyncClient, text: str, model: str, headers: dict[str, str]
) -> _PhaseTokenCount:
    if not text:
        return _unknown("no_phase")
    try:
        response = await client.post(
            "/tokenize",
            json={"model": model, "prompt": text, "add_special_tokens": False},
            headers=headers,
        )
    except httpx.TimeoutException:
        return _unknown("timeout")
    except httpx.HTTPError:
        return _unknown("unavailable")
    if response.status_code == 404:
        return _unknown("unsupported")
    if response.status_code != 200:
        return _unknown("http_error")
    try:
        payload = response.json()
    except ValueError:
        return _unknown("invalid_response")
    count = _validated_count(payload, model)
    if count is None:
        return _unknown("invalid_response")
    return _PhaseTokenCount(count=count)


async def recount_output(
    result: StreamResult,
    target: TargetDef,
    *,
    api_key: SecretStr | None,
    timeout_s: float,
) -> _OutputTokenCounts:
    """SSE終了後に呼び、元の時刻や要求の成否を変更しない。"""
    if result.error is not None:
        return _both_unknown("request_failed")
    if result.server_model is None:
        return _both_unknown("model_unknown")
    if result.server_model != target.model:
        return _both_unknown("model_mismatch")
    headers: dict[str, str] = {}
    if api_key is not None:
        secret = api_key.get_secret_value()
        headers = {"authorization": f"Bearer {secret}", "x-api-key": secret}
    async with httpx.AsyncClient(
        base_url=str(target.base_url), timeout=timeout_s, follow_redirects=False
    ) as client:
        thinking = await _count_one(client, _phase_text(result, "thinking"), target.model, headers)
        text = await _count_one(client, _phase_text(result, "text"), target.model, headers)
    return _OutputTokenCounts(thinking=thinking, text=text)
