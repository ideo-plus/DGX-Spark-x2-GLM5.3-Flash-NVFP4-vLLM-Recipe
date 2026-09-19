"""`bench` コマンドの入口。

このモジュールは引数の骨組みだけを持つ。各サブコマンド (計測の実行、要約の
作り直し、比較、公開、文字数とトークン数の比の測定) は、タスク 5.1 で
`build_parser` にぶら下げる。ここでは、それらを後から足せる形にしておく。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from bench_harness import __version__


def build_parser() -> argparse.ArgumentParser:
    """`bench` の引数解析器を組み立てる。

    サブコマンドはまだ登録しない。後続のタスクが
    ``parser.add_subparsers(...)`` が返す ``subparsers`` に
    ``add_parser`` で足せるように、ここで枠だけを用意する。
    """
    parser = argparse.ArgumentParser(
        prog="bench",
        description=(
            "GLM-5.3-Flash 推論サーバーを Anthropic 互換 API 経由で計測するコマンドラインの道具"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_subparsers(dest="command", title="commands", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI のエントリポイント。

    `--help` と `--version` は `argparse` が処理の途中で
    `SystemExit` を送出する (呼び出し元に伝播させる)。
    """
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
