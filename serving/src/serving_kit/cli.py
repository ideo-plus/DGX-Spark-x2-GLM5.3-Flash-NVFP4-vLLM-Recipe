"""`serve` コマンドの入口 (task 1.1: 骨組みだけ)。

`bench` (`bench_harness.cli`) と同じ流儀 (argparse、`main(argv) -> int`、
誤りは stderr に 1 行の日本語) に合わせる。サブコマンドの中身は、まだ 1 つも
つながっていない。検査、配布、起動、停止などの実際のコマンドは、下の部品
(`config`、`guards`、`lifecycle` など) ができたあと、task 5.1 でここにつなぐ。

このタスクで持つのは、次の 3 つだけである。

- `--help`: `argparse` が使い方を標準出力に出し、`SystemExit(0)` で終わる
- `--version`: `argparse` が版を標準出力に出し、`SystemExit(0)` で終わる
- サブコマンドを選ばずに呼んだときの扱い: 使い方を標準エラーに出し、
  `EXIT_PRECONDITION` (1) で終わる (design.md Error Handling: 1 = 前提の不足)

終了コードの全体 (0 / 1 / 2 / 130) は design.md の Error Handling が決めている。
このタスクでは、まだ起こり得ない値 (2、130) の定数までは持たず、いま実際に
返し得る 2 つだけを定義する。残りは、対応するコマンドをつなぐタスクで足す。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Final

from serving_kit import __version__

__all__ = ["EXIT_OK", "EXIT_PRECONDITION", "build_parser", "main"]

EXIT_OK: Final[int] = 0
"""正常に終わった (design.md Error Handling)。"""

EXIT_PRECONDITION: Final[int] = 1
"""前提の不足、または断り (design.md Error Handling)。

サブコマンドを 1 つも選ばずに呼んだ場合は、ここに当たる。
"""


def build_parser() -> argparse.ArgumentParser:
    """`serve` の引数解析器を組み立てる。

    サブコマンドの入れ物 (`dest="command"`) だけをここで用意し、中身
    (`serve check` などの個々のサブコマンド) は task 5.1 で足す。
    `--help` と `--version` は `argparse` が `SystemExit` を送出する
    (呼び出し元にそのまま伝播させる)。
    """
    parser = argparse.ArgumentParser(
        prog="serve",
        description="DGX Spark 2 台に vLLM の推論サーバーを立てるための道具 (Serving Kit)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_subparsers(dest="command", title="commands", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI のエントリポイント。終了の値を返す (試験は、この関数をそのまま呼ぶ)。

    `--help`、`--version`、引数の使い方の誤りは、`argparse` が `SystemExit`
    を送出する (伝播させる)。サブコマンドを選ばなかったときだけ、ここで
    `EXIT_PRECONDITION` に変える。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    command: str | None = args.command
    if command is None:
        parser.print_help(sys.stderr)
        print("\n実行するコマンドを 1 つ選ぶこと。", file=sys.stderr)
        return EXIT_PRECONDITION
    # サブコマンドの中身は task 5.1 でつなぐ。それまでは、選べる名前が
    # 1 つもないので、argparse 自身が「知らない選択肢」として断る
    # (SystemExit(2)) ため、ここには到達しない。
    raise NotImplementedError(
        f"サブコマンド '{command}' はまだつながっていない (task 5.1 でつなぐ)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
