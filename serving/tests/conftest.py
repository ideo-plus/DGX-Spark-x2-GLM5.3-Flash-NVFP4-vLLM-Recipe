"""試験の共通の下ごしらえ。

`tests/` は `pyproject.toml` の `[tool.pytest.ini_options] pythonpath` と
`[tool.mypy] mypy_path` に入れてあるので、`tests/unit/` と `tests/integration/`
のどちらからでも `tests/` 直下の助け (`fake_runner.py`、`fake_vllm.py`) を import できる。

`fake_runner.py` は、試験ごとに台本 (`script`) が大きく違うので、共通のフィクスチャは
持たず、各試験が `FakeRunner(...)` を直に組み立てる。`fake_vllm.py` は、起動と停止に
実際のソケットが要るので、ここで `fake_vllm` / `fake_vllm_factory` を用意する。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from fake_vllm import FakeVllm


@pytest.fixture
def fake_vllm() -> Iterator[FakeVllm]:
    """起動済みの偽の推論サーバーを 1 台渡し、試験が終わったら必ず止める。"""
    server = FakeVllm()
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def fake_vllm_factory() -> Iterator[Callable[[], FakeVllm]]:
    """偽の推論サーバーを何台でも起動できる。試験が終わったら、すべて止める。"""
    servers: list[FakeVllm] = []

    def start() -> FakeVllm:
        server = FakeVllm()
        server.start()
        servers.append(server)
        return server

    try:
        yield start
    finally:
        for server in servers:
            server.stop()
