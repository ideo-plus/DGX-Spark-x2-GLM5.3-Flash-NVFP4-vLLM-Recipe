"""試験の共通の下ごしらえ。

`tests/` は `pyproject.toml` の `[tool.pytest.ini_options] pythonpath` と
`[tool.mypy] mypy_path` に入れてあるので、`tests/unit/` と
`tests/integration/` のどちらからでも `from fake_server import ...` と書ける。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from fake_server import FakeServer


@pytest.fixture
def fake_server() -> Iterator[FakeServer]:
    """起動済みの偽のサーバーを 1 台渡し、試験が終わったら必ず止める。"""
    server = FakeServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def fake_server_factory() -> Iterator[Callable[[], FakeServer]]:
    """偽のサーバーを何台でも起動できる。試験が終わったら、すべて止める。"""
    servers: list[FakeServer] = []

    def start() -> FakeServer:
        server = FakeServer()
        server.start()
        servers.append(server)
        return server

    try:
        yield start
    finally:
        for server in servers:
            server.stop()


@pytest.fixture
def swallow_the_first_cancel_on(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str], asyncio.Event]:
    """HTTP の層が、外からの打ち切りを握りつぶした状態を作る (issue #55)。

    返す関数に要求の path を渡すと、`httpx.AsyncClient.send` を差し替える。その
    path への最初の要求だけ、送る前に止まって `Event` を立て (返す値)、そこで
    受けた `CancelledError` を握りつぶしてから、普通に送る。握りつぶされたあとの
    状態 (例外は消え、`Task.cancelling()` は 1 のまま) は、anyio 4.15.1 の
    `connect_tcp` が起こすものと同じ (原因は `probe._raise_if_cancel_was_swallowed`
    の docstring)。実物の競合は時刻に依存して再現しにくいので、ここで直に作る。
    """

    def install(path: str) -> asyncio.Event:
        stalled = asyncio.Event()
        swallowed = False
        real_send = httpx.AsyncClient.send

        async def send(
            self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
        ) -> httpx.Response:
            nonlocal swallowed
            if not swallowed and request.url.path == path:
                swallowed = True
                stalled.set()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(3600)
            return await real_send(self, request, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "send", send)
        return stalled

    return install
