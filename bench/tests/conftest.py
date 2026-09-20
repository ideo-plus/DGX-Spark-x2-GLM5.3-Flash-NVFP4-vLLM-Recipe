"""試験の共通の下ごしらえ。

`tests/` は `pyproject.toml` の `[tool.pytest.ini_options] pythonpath` と
`[tool.mypy] mypy_path` に入れてあるので、`tests/unit/` と
`tests/integration/` のどちらからでも `from fake_server import ...` と書ける。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

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
