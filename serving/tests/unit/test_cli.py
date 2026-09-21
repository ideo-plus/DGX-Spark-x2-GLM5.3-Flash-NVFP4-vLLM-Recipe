"""コマンドの入口の疎通の試験 (task 1.1: 骨組みだけ)。

`main(argv)` をその場で呼ぶ (プロセスは起こさない)。サブコマンドの中身は
まだ 1 つもつながっていない (task 5.1 でつなぐ) ので、ここで確かめるのは
`--help`、`--version`、コマンドを選ばなかったときの扱いだけである。
"""

from __future__ import annotations

import pytest

from serving_kit import __version__
from serving_kit.cli import EXIT_PRECONDITION, build_parser, main


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """`serve --help` は 0 で終わり、使い方を標準出力に出す (design.md cli)。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "serve" in out


def test_version_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """`serve --version` は 0 で終わり、パッケージの版を標準出力に出す。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_no_command_returns_precondition(capsys: pytest.CaptureFixture[str]) -> None:
    """サブコマンドを選ばずに呼ぶと、断って `EXIT_PRECONDITION` (1) で終わる。

    サブコマンドの中身は task 5.1 でつなぐが、「選ばなかったときの扱い」は
    この入口だけの骨組みで決まる (design.md Error Handling: 1 = 前提の不足)。
    """
    exit_code = main([])
    assert exit_code == EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "実行するコマンドを 1 つ選ぶこと" in err


def test_unknown_argument_exits_with_usage_error() -> None:
    """`argparse` が扱わない引数は、`SystemExit(2)` としてそのまま伝播する。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["--no-such-flag"])
    assert excinfo.value.code == 2


def test_build_parser_prog_is_serve() -> None:
    """引数解析器の `prog` が `serve` であること (ヘルプの表示に使われる)。"""
    parser = build_parser()
    assert parser.prog == "serve"
