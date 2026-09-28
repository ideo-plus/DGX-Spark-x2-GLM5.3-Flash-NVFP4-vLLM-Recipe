"""`bench` コマンドの入口 (task 5.1)。

5 つのコマンドを持つ (design.md Technology Stack の CLI の行)。

| コマンド | すること |
|---|---|
| `bench run` | 計測ランを 1 回流し、終わりに要約を作る (1.1、1.2、1.4) |
| `bench summarize` | 生データから、2 つの要約を作り直す (8.2) |
| `bench compare` | 2 つの計測ランを比べて、人が読む形で出す (9.1〜9.6) |
| `bench publish` | 要約 2 つだけを公開の場所に写す (8.4) |
| `bench calibrate` | 内容の種類ごとの、1 トークンあたりの文字数を測る |

この module が受け持つのは、引数の解析と、部品の呼び出しと、表示と、終了の値
への変換だけである。計測の進行そのものは `runner`、集計は `analysis`、
トークン数の計数は `client.probe` にある (依存の向き: `... -> runner -> cli`)。

## 終了の値 (design.md Monitoring、注 3.5)

| 値 | 意味 | 定数 |
|---|---|---|
| 0 | 完了した | `EXIT_OK` |
| 1 | 前提の不足、または設定の誤り | `EXIT_PRECONDITION` |
| 2 | 連続の失敗、包みのトークン数を数えられずに停止。保存や公開の失敗もここ | `EXIT_ABORTED` |
| 130 | 中断 (`SIGINT`) | `EXIT_INTERRUPTED` |

数そのものは、この module では決めない (`runner` の定数をそのまま使う)。
例外から終了の値への対応は、`main` の 1 か所にまとめてある:

- `PreconditionError` (計測を始められない): 例外の文をそのまま出して、
  例外が持つ `exit_code` (1) で終わる。どの前提が満たされていないかは、
  例外の文に書いてある (1.4)
- `ConfigError` (設定の読み込みと検証の誤り)、`ProbeError` (トークン数を
  数えられない): 1
- `StoreError` / `PublishError` / `OSError` (計測ランでない場所、保存や公開の
  失敗): 2 (注 3.5、4.4)
- `KeyboardInterrupt`: 130。計測ランのディレクトリができる前の中断は、
  `execute_run` から `KeyboardInterrupt` がそのまま出てくる (注 3.5)

想定している失敗で、Python の traceback を計測者に見せることはない。想定して
いない例外は、直さずにそのまま外へ出す (黙って 0 でない値にしない)。

## 測る項目のまとまりの選び方 (1.2、task 8.1)

`--suite` は繰り返し書けて、**書いた順**に実行する。省いたときの既定は、速さの
3 つ (`decode`、`prefill`、`concurrency`) だけである。品質の検査と長い会話の
検査は、実機では 1 回の計測ランが何時間もかかる (注 7.2 → 8.3) ので、名指し
したときだけ流す。5 つ全部を流すときは `--suite all` と書く。

品質の検査が使う公開のコードの課題は、`--data-cache` で置き場所を変えられ、
`--no-download` で取得を禁じられる (`--no-download` のときに課題がなければ、
コードの条件だけが理由つきで飛ぶ)。どちらも、対応表を組み立てて `execute_run`
に渡す継ぎ目 (`default_suite_registry(problems_loader=…)`) を通る。

コマンドを 1 つも選ばずに `bench` とだけ打ったときは、使い方を標準エラーに
出して 1 (設定の誤り) で終わる。`bench compare` は、収まらない条件があっても
0 で終わる (判定は結果であって、この道具の失敗ではない。design.md Monitoring
の終了の値に、判定による値がない)。

## 認証の情報 (1.8)

この module は、認証の情報の**値**をどこにも出さない。表示してよいのは、値を
入れた環境変数の**名前**だけである。値を読むのは `runner` と
`config.resolve_api_key` で、この module は受け取った `SecretStr` を
`client` に渡すだけ (`calibrate`)。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

from pydantic import SecretStr

from bench_harness import __version__, config
from bench_harness.analysis.compare import (
    compare_runs,
    write_comparison,
)
from bench_harness.analysis.publish import PublishError, publish_run
from bench_harness.analysis.summarize import (
    SUMMARY_JSON_NAME,
    SUMMARY_MD_NAME,
    summarize_run,
    write_summary,
)
from bench_harness.client.messages import HttpxMessagesClient
from bench_harness.client.probe import ProbeError, count_input_tokens
from bench_harness.corpus import TemplateCorpus, humaneval
from bench_harness.runner import (
    EXIT_ABORTED,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_PRECONDITION,
    PreconditionError,
    StderrProgressSink,
    default_suite_registry,
    execute_run,
)
from bench_harness.store import StoreError, list_run_dirs
from bench_harness.suites.quality import ProblemsLoader
from bench_harness.types import (
    ContentKind,
    InputMessage,
    MessagesRequest,
    Profile,
    RunOutcome,
    RunRequest,
    RunStatus,
    SuiteName,
    TargetDef,
    TextBlockParam,
)

__all__ = ["ALL_SUITES", "DEFAULT_SUITES", "build_parser", "main"]

DEFAULT_SUITES: Final[tuple[SuiteName, ...]] = (
    SuiteName.DECODE,
    SuiteName.PREFILL,
    SuiteName.CONCURRENCY,
)
"""`--suite` を省いたときに流すまとまり (task 8.1 で決めた既定)。

8.1 で品質の検査と長い会話の検査を対応表に足したので、「実行できるものすべて」
を既定のままにすると、`quick` の設定でも何時間もかかる計測ラン (注 7.2 → 8.3)
が、うっかり始まってしまう。速さの 3 つ (P0 の終わりの条件に要るもの) だけを
既定にして、残りは名指し、または `--suite all` で選ぶ。
"""

ALL_SUITES: Final[str] = "all"
"""`--suite` に書くと、対応表のまとまりをすべて選ぶ名前。

`SuiteName` の値とは重ならない (重なると、まとまりの名前として読めなくなる)。
"""

COMPARISON_MD_NAME: Final[str] = "comparison.md"
"""`bench compare --out <ディレクトリ>` が書くファイルの名前。"""

CALIBRATE_TARGET_TOKENS: Final[int] = 2000
"""比の測定に使う、1 種類あたりの文章の長さ (狙いのトークン数)。

短すぎると要求の包み (役割の名前や JSON の記号) の重みが効きすぎ、長すぎると
測るのに時間がかかる。2 千トークンなら、包みの影響は 1% に満たない。
"""

CALIBRATE_MAX_TOKENS: Final[int] = 16
"""比の測定の要求に載せる出力の上限。

`count_tokens` の口に送るときは外される項目だが、`MessagesRequest` の必須の
項目なので値が要る。口がない対象サーバーでは、代わりの数え方
(`max_tokens=1` の要求) に置き換わる (注 2.7)。
"""

_RUN_ID_SHOWN_IN_ERROR: Final[int] = 10
"""知らない計測ランの名前を言われたときに、いくつまで名前を挙げるか。"""


# --- 引数の解析 --------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """`bench` の引数解析器を組み立てる。

    `--help` と `--version` は `argparse` が `SystemExit` を送出する
    (呼び出し元に伝播させる)。
    """
    parser = argparse.ArgumentParser(
        prog="bench",
        description=(
            "GLM-5.3-Flash 推論サーバーを Anthropic 互換 API 経由で計測するコマンドラインの道具"
        ),
        epilog=_EXIT_CODE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", title="commands", metavar="<command>")
    _add_run_parser(subparsers)
    _add_summarize_parser(subparsers)
    _add_compare_parser(subparsers)
    _add_publish_parser(subparsers)
    _add_calibrate_parser(subparsers)
    return parser


_EXIT_CODE_EPILOG: Final[str] = (
    "終了の値: 0 完了 / 1 前提の不足または設定の誤り"
    " / 2 連続の失敗、包みのトークン数を数えられずに停止 (保存・公開の失敗を含む)"
    " / 130 中断\n"
)

_RUN_ARG_HELP: Final[str] = (
    "計測ラン。ディレクトリの経路か、生データの置き場所 (--results-root) の下の識別子"
)


def _add_common_output_args(parser: argparse.ArgumentParser) -> None:
    """設定ファイルの場所の指定 (既定は `bench/config/` の中)。"""
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        metavar="PATH",
        help="対象サーバーの定義のファイル (既定: bench/config/targets.toml)",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=None,
        metavar="PATH",
        help="計測の設定のファイル (既定: bench/config/profiles.toml)",
    )


def _add_results_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "生データの置き場所 (既定: <リポジトリ>/results)。"
            "git の管理の対象になり得る場所は拒む (8.5)"
        ),
    )


def _add_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "run",
        help="計測ランを 1 回流し、終わりに要約を作る",
        description="対象サーバーとまとまりを選んで、計測ランを 1 回流す (1.1、1.2)。",
        epilog=_EXIT_CODE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", required=True, metavar="NAME", help="対象サーバーの定義の名前")
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            f"測る項目のまとまり (繰り返して複数選べる): {_runnable_suite_names()}。"
            f"省いたときの既定は、速さの 3 つ ({_default_suite_names()})。"
            f"'{ALL_SUITES}' と書くと、5 つ全部 (品質の検査と長い会話の検査は、"
            "実機では何時間もかかるので、名指ししたときだけ流す)"
        ),
    )
    parser.add_argument(
        "--profile", default="quick", metavar="NAME", help="計測の設定の名前 (既定: quick)"
    )
    parser.add_argument(
        "--retokenize-output",
        action="store_true",
        help="生成後、同じサーバーで思考と本文の生文字列を再計数する",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=None,
        metavar="経路=整数",
        help=(
            "試行の回数などの整数の項目を上書きする (例: --set decode.trials=20"
            " --set concurrency.rounds=2)。経路は Profile の項目を点でつないだもの。"
            "繰り返して複数指定でき、同じ経路を 2 回書いたら、あとのものが効く。"
            "長い会話の検査を 1 段階だけ流すなら --set agent.end_tokens=20000、"
            "試行を減らすなら --set agent.trials_per_stage=10、"
            "コードの課題を減らすなら --set quality.code_problem_limit=10"
        ),
    )
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "公開のコードの課題 (品質の検査) の置き場所 (既定: bench/data-cache)。"
            "git の管理の対象になり得る場所は拒む (8.5、11.1)"
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help=(
            "公開のコードの課題を取得しない。置き場所になければ、コードの条件だけを"
            "理由つきで飛ばす (ほかの条件は実行する)"
        ),
    )
    _add_common_output_args(parser)
    _add_results_root_arg(parser)


def _add_summarize_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "summarize",
        help="生データから、2 つの要約を作り直す",
        description="計測ランの生データから、summary.json と summary.md を作り直す (8.2)。",
        epilog=_EXIT_CODE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run", metavar="RUN", help=_RUN_ARG_HELP)
    _add_results_root_arg(parser)


def _add_compare_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "compare",
        help="2 つの計測ランを比べる",
        description=(
            "2 つの計測ランを比べて、人が読む形で標準出力に出す (9.1〜9.6)。"
            "収まらない条件があっても、それは結果であって失敗ではないので 0 で終わる"
        ),
        epilog=_EXIT_CODE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_a", metavar="RUN_A", help=_RUN_ARG_HELP)
    parser.add_argument("run_b", metavar="RUN_B", help=_RUN_ARG_HELP)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help=f"比較の結果を、このディレクトリの {COMPARISON_MD_NAME} にも書く",
    )
    _add_results_root_arg(parser)


def _add_publish_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "publish",
        help="要約 2 つだけを公開の場所に写す",
        description=(
            "指示されたときだけ、計測ランの summary.json と summary.md を公開の場所に写す"
            " (8.4)。生データは写さない"
        ),
        epilog=_EXIT_CODE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run", metavar="RUN", help=_RUN_ARG_HELP)
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=None,
        metavar="PATH",
        help="公開の場所 (既定: <リポジトリ>/docs/results)",
    )
    _add_results_root_arg(parser)


def _add_calibrate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "calibrate",
        help="1 トークンあたりの文字数を測る",
        description=(
            "内容の種類ごとに合成の文章を作り、対象サーバーにトークン数を数えさせて、"
            "1 トークンあたりの文字数を出す。結果は profiles.toml に書き写せる TOML の"
            "断片として、標準出力に出す"
        ),
        epilog=_EXIT_CODE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", required=True, metavar="NAME", help="対象サーバーの定義の名前")
    parser.add_argument(
        "--profile",
        default="quick",
        metavar="NAME",
        help="計測の設定の名前 (既定: quick)。文章の長さの狙いと制限時間に使う",
    )
    _add_common_output_args(parser)


# --- 入口 --------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI のエントリポイント。終了の値を返す (試験は、この関数をそのまま呼ぶ)。

    想定している失敗は、すべてここで 1 行の日本語にして、終了の値に直す
    (traceback は出さない)。`--help` と `--version`、引数の使い方の誤りは、
    `argparse` が `SystemExit` を送出する (伝播させる)。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(parser, args)
    except PreconditionError as exc:
        return _fail(exc, exc.exit_code)
    except config.ConfigError as exc:
        return _fail(exc, EXIT_PRECONDITION)
    except ProbeError as exc:
        return _fail(exc, EXIT_PRECONDITION)
    except (StoreError, PublishError, OSError) as exc:
        return _fail(exc, EXIT_ABORTED)
    except KeyboardInterrupt:
        print(
            "中断した (計測ランのディレクトリができていれば、それまでの結果は残っている)",
            file=sys.stderr,
        )
        return EXIT_INTERRUPTED


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    command: str | None = args.command
    if command is None:
        parser.print_help(sys.stderr)
        print("\n実行するコマンドを 1 つ選ぶこと。", file=sys.stderr)
        return EXIT_PRECONDITION
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "run": _cmd_run,
        "summarize": _cmd_summarize,
        "compare": _cmd_compare,
        "publish": _cmd_publish,
        "calibrate": _cmd_calibrate,
    }
    return handlers[command](args)


def _fail(exc: BaseException, code: int) -> int:
    print(f"エラー: {exc}", file=sys.stderr)
    return code


# --- bench run ---------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    """計測ランを 1 回流し、終わりに要約を作る (1.1、1.2、1.4、10.4)。"""
    target_name: str = args.target
    profile_name: str = args.profile
    targets_path: Path | None = args.targets
    profiles_path: Path | None = args.profiles
    results_root: Path | None = args.results_root

    request = RunRequest(
        target_name=target_name,
        suites=_selected_suites(args.suite),
        profile_name=profile_name,
        trials_override=_parse_overrides(args.overrides),
        retokenize_output=args.retokenize_output,
    )
    if SuiteName.QUALITY in request.suites:
        # 使えない置き場所は、計測を始める前に、設定の誤りとして示す。あとで気付くと、
        # 先に流したまとまりの試行だけが残り、計測ランが `running` のまま止まる
        try:
            humaneval.check_cache_dir(args.data_cache)
        except humaneval.DatasetCacheError as exc:
            raise PreconditionError(f"設定の誤り: --data-cache: {exc}") from exc
    registry = default_suite_registry(
        problems_loader=_problems_loader(args.data_cache, no_download=args.no_download)
    )
    outcome = asyncio.run(
        execute_run(
            request,
            StderrProgressSink(),
            targets_path=targets_path,
            profiles_path=profiles_path,
            results_root=results_root,
            registry=registry,
        )
    )
    return _report_run(outcome)


def _problems_loader(data_cache: Path | None, *, no_download: bool) -> ProblemsLoader | None:
    """公開のコードの課題の読み方 (`--data-cache` と `--no-download` の受け渡し)。

    どちらも指定がなければ `None` を返し、まとまり自身の読み方 (既定の置き場所、
    なければ取得する) に任せる。`RunRequest` は凍結の型で項目を足せないので、
    対応表を組み立てて `execute_run` に渡す継ぎ目を使う (注 1.2、6.5)。
    """
    if data_cache is None and not no_download:
        return None
    return partial(humaneval.load_humaneval_plus, data_cache, allow_download=not no_download)


def _report_run(outcome: RunOutcome) -> int:
    """計測ランの結末を表示し、要約を作って、終了の値を決める。

    要約は、止まった計測ランや中断した計測ランでも作る (10.4)。生データが
    読めずに要約を作れなかったときは、理由を標準エラーに出して、0 で終わらせ
    ない (完了と見分けがつかなくなるため)。
    """
    print(f"計測ラン: {outcome.run_id}")
    print(f"生データ: {outcome.run_dir}")
    print(f"状態: {outcome.status.value}")
    try:
        _emit_summary(outcome.run_dir)
    except (StoreError, OSError) as exc:
        print(f"エラー: 要約を作れなかった: {exc}", file=sys.stderr)
        return outcome.exit_code if outcome.exit_code != EXIT_OK else EXIT_ABORTED
    if outcome.status is not RunStatus.COMPLETED:
        print(
            f"警告: この計測ランは未完了 (状態: {outcome.status.value})。"
            "要約と比較の先頭に、その旨が出る (10.4、10.5)",
            file=sys.stderr,
        )
    return outcome.exit_code


def _selected_suites(values: list[str] | None) -> list[SuiteName]:
    """`--suite` の並びを `SuiteName` に直す (1.2)。

    省かれたときは、**速さの 3 つ**だけを選ぶ (task 8.1 で決めた既定)。品質の
    検査と長い会話の検査は、実機では 1 回の計測ランが何時間もかかる (注 7.2 →
    8.3 の見積もり) ので、名指ししたときだけ流す。5 つ全部を流すときは
    `--suite all` と書く。

    書いた順を保ち、繰り返しも受ける。知らない名前は、使える名前を添えた設定の
    誤りにする。同じまとまりを 2 回選んだときの判定は、対応表を持っている
    `runner` に任せる (同じ判定を 2 か所に書かない)。
    """
    if not values:
        return list(_default_suites())
    if ALL_SUITES in values:
        if len(values) > 1:
            raise PreconditionError(
                f"設定の誤り: まとまり '{ALL_SUITES}' は、ほかの名前と一緒には書けない"
                f" (受け取った値: {', '.join(values)})"
            )
        return _runnable_suites()
    selected: list[SuiteName] = []
    for value in values:
        try:
            selected.append(SuiteName(value))
        except ValueError:
            raise PreconditionError(
                f"設定の誤り: まとまり '{value}' は知らない名前"
                f" (実行できるもの: {_runnable_suite_names()}、"
                f"すべてなら '{ALL_SUITES}')"
            ) from None
    return selected


def _default_suites() -> list[SuiteName]:
    """`--suite` を省いたときに流すまとまり (速さの 3 つ)。"""
    return [name for name in DEFAULT_SUITES if name in default_suite_registry()]


def _default_suite_names() -> str:
    return ", ".join(name.value for name in _default_suites())


def _runnable_suites() -> list[SuiteName]:
    """いま実行できるまとまりを、`SuiteName` の定義順で返す。"""
    registry = default_suite_registry()
    return [name for name in SuiteName if name in registry]


def _runnable_suite_names() -> str:
    return ", ".join(name.value for name in _runnable_suites())


def _parse_overrides(values: list[str] | None) -> dict[str, int]:
    """`--set 経路=整数` の並びを、`RunRequest.trials_override` の形に直す。

    上書きの形式は 1 つだけにする (注 1.3: 点でつないだ経路。`config.apply_overrides`
    が上書きのあとに検証し直す)。経路そのものの正しさと、下限の検査は
    `config.apply_overrides` が行う (ここでは形だけを見る)。
    """
    overrides: dict[str, int] = {}
    for item in values or []:
        key, separator, raw = item.partition("=")
        key = key.strip()
        if not separator or not key:
            raise PreconditionError(
                f"設定の誤り: 上書きは '経路=整数' の形で書く (受け取った値: '{item}')"
            )
        try:
            overrides[key] = int(raw.strip())
        except ValueError:
            raise PreconditionError(
                f"設定の誤り: 上書き '{key}' の値が整数でない (受け取った値: '{raw.strip()}')"
            ) from None
    return overrides


# --- bench summarize ---------------------------------------------------------


def _cmd_summarize(args: argparse.Namespace) -> int:
    """生データから、2 つの要約を作り直す (8.2)。"""
    run_dir = _resolve_run_dir(args.run, args.results_root)
    _emit_summary(run_dir)
    return EXIT_OK


def _emit_summary(run_dir: Path) -> tuple[Path, Path]:
    """要約を作って、場所を標準出力に、読み取りの警告を標準エラーに出す (注 4.1)。"""
    result = summarize_run(run_dir)
    for warning in result.warnings:
        print(f"警告: {warning}", file=sys.stderr)
    json_path, md_path = write_summary(run_dir)
    print(f"要約: {md_path}")
    print(f"要約 (JSON): {json_path}")
    return json_path, md_path


# --- bench compare -----------------------------------------------------------


def _cmd_compare(args: argparse.Namespace) -> int:
    """2 つの計測ランを比べて、人が読む形で標準出力に出す (9.1〜9.6)。

    収まらない条件があっても 0 で終わる。判定は結果であって、この道具の失敗
    ではない (design.md Monitoring の終了の値に、判定による値がない)。
    """
    run_dir_a = _resolve_run_dir(args.run_a, args.results_root)
    run_dir_b = _resolve_run_dir(args.run_b, args.results_root)
    out_dir: Path | None = args.out

    result = compare_runs(run_dir_a, run_dir_b)
    # `to_markdown()` なら、割合の行の分子と分母を渡し忘れようがない (注 4.3)
    print(result.to_markdown(), end="")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = write_comparison(result, out_dir / COMPARISON_MD_NAME)
        print(f"比較の結果を書いた: {path}", file=sys.stderr)
    return EXIT_OK


# --- bench publish -----------------------------------------------------------


def _cmd_publish(args: argparse.Namespace) -> int:
    """要約 2 つだけを公開の場所に写す (8.4)。

    公開は、計測者がこのコマンドを打ったときだけ行う。`bench run` は公開しない。
    """
    run_dir = _resolve_run_dir(args.run, args.results_root)
    docs_root: Path | None = args.docs_root
    print(
        f"公開するのは {SUMMARY_JSON_NAME} と {SUMMARY_MD_NAME} の 2 つだけ (生データは写さない)。"
        f"公開の前に {SUMMARY_MD_NAME} を目で見て確かめること (注 4.4)",
        file=sys.stderr,
    )
    published = publish_run(run_dir, docs_root)
    if published.incomplete:
        print(
            "警告: この計測ランは未完了。要約の先頭に、その旨が出る (10.4)",
            file=sys.stderr,
        )
    print(f"公開先: {published.dest_dir}")
    print(published.summary_json_path)
    print(published.summary_md_path)
    return EXIT_OK


# --- bench calibrate ---------------------------------------------------------


@dataclass(frozen=True)
class _Calibration:
    """1 つの内容の種類についての、測った文字数とトークン数。"""

    kind: ContentKind
    chars: int
    tokens: int
    method: str

    @property
    def chars_per_token(self) -> float:
        return self.chars / self.tokens


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """内容の種類ごとに、1 トークンあたりの文字数を測って表示する。

    このコマンドは、部品 (`corpus` の合成の文章と、`client.probe` の計数) を
    呼んで表示するだけで、測り方を自分では持たない。標準出力に出すのは、
    `profiles.toml` にそのまま書き写せる TOML の断片だけにする (説明は標準
    エラーに出す)。
    """
    target = config.select_target(args.target, args.targets)
    profile = config.select_profile(args.profile, args.profiles)
    api_key = config.resolve_api_key(target)
    measured = asyncio.run(_measure_chars_per_token(target, profile, api_key))
    _print_calibration(target, profile, measured)
    return EXIT_OK


async def _measure_chars_per_token(
    target: TargetDef, profile: Profile, api_key: SecretStr | None
) -> list[_Calibration]:
    """内容の種類ごとに、合成の文章を作って、対象サーバーにトークン数を数えさせる。"""
    corpus = TemplateCorpus(profile.chars_per_token)
    measured: list[_Calibration] = []
    async with HttpxMessagesClient.from_target(target, api_key) as client:
        for kind in ContentKind:
            text = _sample_text(corpus, kind, profile.seed)
            request = MessagesRequest(
                model=target.model,
                max_tokens=CALIBRATE_MAX_TOKENS,
                messages=[InputMessage(role="user", content=[TextBlockParam(text=text)])],
            )
            count = await count_input_tokens(
                client, target, request, timeout=profile.timeout, api_key=api_key
            )
            if count.tokens <= 0:
                raise ProbeError(
                    f"入力のトークン数が 0 と返った ({kind.value})。比を出せない"
                    f" (対象サーバー '{target.name}')"
                )
            measured.append(
                _Calibration(kind=kind, chars=len(text), tokens=count.tokens, method=count.method)
            )
    return measured


def _sample_text(corpus: TemplateCorpus, kind: ContentKind, seed: int) -> str:
    """種類ごとの、測るための文章 (種が同じなら、いつでも同じ文章。11.4)。"""
    match kind:
        case ContentKind.PROSE_EN:
            return corpus.prose("en", CALIBRATE_TARGET_TOKENS, seed)
        case ContentKind.PROSE_JA:
            return corpus.prose("ja", CALIBRATE_TARGET_TOKENS, seed)
        case ContentKind.CODE:
            return corpus.code(CALIBRATE_TARGET_TOKENS, seed)
        case ContentKind.LOG:
            return corpus.log(CALIBRATE_TARGET_TOKENS, seed)


def _print_calibration(
    target: TargetDef, profile: Profile, measured: Sequence[_Calibration]
) -> None:
    """設定に書き写せる TOML の断片を、標準出力に出す。

    標準出力に出すのは、TOML として読める行だけにする (説明を足すときは、
    必ず `#` で始まるコメントにする)。計測者が、この出力をそのまま
    `profiles.toml` に貼れるようにするため。
    """
    print(
        f"対象サーバー '{target.name}' ({target.base_url}) で測った。"
        f"下の断片を {profile.name} の設定に書き写すこと",
        file=sys.stderr,
    )
    print(f"# bench calibrate --target {target.name} --profile {profile.name}")
    print(f"# 1 種類につき約 {CALIBRATE_TARGET_TOKENS} トークンの合成の文章で測った")
    print("# 要求の包み (役割や JSON の記号) のぶん、比はわずかに小さく出る")
    print(f"[profiles.{profile.name}.chars_per_token]")
    for item in measured:
        print(
            f"{item.kind.value} = {item.chars_per_token:.3f}"
            f"  # {item.chars} 文字 / {item.tokens} トークン ({item.method})"
        )


# --- 計測ランの場所 ----------------------------------------------------------


def _resolve_run_dir(value: str, results_root: Path | None) -> Path:
    """`RUN` の引数を、計測ランのディレクトリに直す。

    受け取るのは 2 つの形である (`--help` にも書いてある):

    - 生データの置き場所 (`--results-root`、既定は `<リポジトリ>/results`) の
      下にある、計測ランの識別子。シェルの補完で末尾に区切りが付いても
      (`<識別子>/`)、識別子として探す
    - ディレクトリの経路 (途中に経路の区切りを含む、または実在するディレクトリ)

    区切りを含まない名前は、**先に置き場所の下を探す**。いまいるディレクトリに
    同じ名前のディレクトリがあっても、置き場所の下の計測ランが勝つ (いまいる
    場所しだいで、別のものを黙って読まないため)。

    識別子として探して見つからなければ、ある計測ランの名前を添えた設定の誤りに
    する。経路として渡されたものが計測ランでなければ、ここでは判定せず、
    `RunStore.open` の `StoreError` (終了の値 2) に任せる。
    """
    name = _bare_name(value)
    if name is None:
        return Path(value)
    root = config.resolve_results_root(results_root)
    run_dir = root / name
    if run_dir.is_dir():
        return run_dir
    if Path(value).is_dir():
        return Path(value)
    known = [path.name for path in list_run_dirs(root)]
    shown = ", ".join(known[:_RUN_ID_SHOWN_IN_ERROR]) if known else "(なし)"
    more = (
        f" (ほか {len(known) - _RUN_ID_SHOWN_IN_ERROR} 件)"
        if len(known) > _RUN_ID_SHOWN_IN_ERROR
        else ""
    )
    raise PreconditionError(
        f"設定の誤り: 計測ラン '{value}' が '{root}' の下に見つからない。"
        f"ある計測ラン: {shown}{more}"
    )


def _bare_name(value: str) -> str | None:
    """識別子として探す名前を返す。経路として渡されたものなら `None`。

    識別子には経路の区切りが入らない。末尾の区切りだけは、シェルの補完が
    付けるので落とす。`.` と `..` と `~` で始まるものは、経路である。
    """
    separators = [os.sep] + ([os.altsep] if os.altsep else [])
    name = value.rstrip("".join(separators))
    if not name or any(separator in name for separator in separators):
        return None
    if name in {".", ".."} or name.startswith("~"):
        return None
    return name


if __name__ == "__main__":
    raise SystemExit(main())
