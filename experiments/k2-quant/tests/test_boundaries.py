"""変換が GPU とネットワークを使わないことの試験 (C7)。

Issue #56 は、変換の道具を「CPU だけで動き、GPU とネットワークは使わない」ものと定める。
推論サーバーを動かしている実機で使うので、GPU に触れたり、通信や外部プロセスを起こしたり
してはならない。import の名前やソースの文言ではなく、変換を実際に実行したときの振る舞いで
確かめる。

- ソケットの作成・名前解決・外部プロセスの起動を、呼ばれたら失敗させた状態で、変換が最後まで
  終わる (許可した標準ライブラリを経由した通信でも検出できる)
- 変換のあと、CUDA が初期化されていない (GPU を使おうとすると、CUDA のない環境では例外に
  なり、ある環境では初期化済みになる)
"""

from __future__ import annotations

import socket
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import pytest
import torch

import synthetic
from k2_quant.__main__ import main


def _build(tmp_path: Path) -> tuple[Path, Path]:
    """合成 checkpoint を作り、(入力, 出力) の道筋を返す。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    synthetic.build_checkpoint(source)
    return source, output


def _convert(source: Path, output: Path) -> int:
    return main(
        [
            "--source",
            str(source),
            "--output",
            str(output),
            "--source-repo",
            "RedHatAI/GLM-5.3-Flash-NVFP4",
            "--source-revision",
            "18d55bfd5a2194887738da73753975c9d3842f46",
        ]
    )


def test_conversion_finishes_without_opening_a_socket_or_starting_a_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ソケットと外部プロセスを使えない状態でも、既定の変換が最後まで終わる (C7)。"""
    source, output = _build(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("conversion must not use the network or start a process")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)

    assert _convert(source, output) == 0
    assert (output / synthetic.MANIFEST_NAME).is_file()


def test_conversion_does_not_initialize_cuda(tmp_path: Path) -> None:
    """既定の変換のあと、CUDA が初期化されていない (C7)。

    GPU を使おうとする実装は、CUDA のない環境では変換が例外で止まり、ある環境では
    初期化済みになって、この試験が失敗する。
    """
    source, output = _build(tmp_path)
    # `torch.cuda.is_initialized` は型注釈がなく、strict の `mypy` が呼び出しを断る。型つきで受ける
    cuda_is_initialized: Callable[[], bool] = torch.cuda.is_initialized

    assert _convert(source, output) == 0

    assert not cuda_is_initialized()
