"""`serve` コマンドの入口 (task 5.1)。

`bench` (`bench_harness.cli`) と同じ流儀 (argparse、`main(argv) -> int`、誤りは stderr に
1 行の日本語) に合わせる。この module が受け持つのは、**引数の解析と、部品の呼び出しと、
表示と、終了コードへの変換だけ**である。Spark に触る判断 (関門、了承、片付け) は、すべて
下の部品 (`config`、`guards`、`image`、`weights`、`lifecycle`、`probe`、`netcheck`、`watch`、
`thinking`) にあり、依存の向きは `... -> cli` の一方向である。

## コマンド (design.md 「入口 › cli」の表)

| コマンド | すること | 状態を変えるか |
|---|---|---|
| `serve check <構成>` | 構成の検査と、9 つの関門を流して結果を並べる | 変えない |
| `serve push` | 2 台に 6 つの置き場所を作り、`payload/` を配る | 変える |
| `serve pull-image <構成>` | イメージを、ダイジェストで 2 台に取得する | 変える |
| `serve image-licenses <構成>` | イメージの中のライセンスの表記を読んで出す | 変える |
| `serve manifest <repo> <revision>` | Mac で、重みのマニフェストを作る | 変えない |
| `serve derived-import <manifest>...` | 道具の manifest を、Mac で取り込む | 変えない |
| `serve fetch <構成> [--probe-files]` | 重みを 2 台に取得して照合する | 変える |
| `serve verify <構成> [--probe-files]` | 2 台の重みをマニフェストと照合する | 変える |
| `serve start <構成> [--timeout]` | 2 台で起こし、受け付けの開始まで待つ | 変える |
| `serve stop` | 自分のコンテナを 2 台とも止めて消す | 変える |
| `serve status` | 2 台と推論サーバーの、いまの状態を示す | 変えない |
| `serve smoke <構成> [--max-tokens]` | 英語と日本語の短い要求を 1 つずつ送る | 変えない |
| `serve logs <構成>` | 2 台の記録を `serving/var/` に写す | 変えない |
| `serve probe <構成> [--timeout]` | 1 台の縮小の確認 | 変える |
| `serve netcheck links` | 直結のインターフェースを読む | 変えない |
| `serve netcheck bandwidth/sanity <構成>` | 帯域の計測、事前の確認 | 変える |
| `serve netcheck ab <構成> --env K=V` | 足す設定の A/B | 変える |
| `serve watch <構成>` | 連続の負荷の間の見張り | 変えない |
| `serve thinking <構成>` | thinking の深さの確かめ | 変えない |
| `serve autostart set <構成>` | Spark 上の見張りが自動で起こす構成を指定する | 変える |
| `serve autostart clear` | 自動で起こす構成の指定を解除する | 変える |
| `serve autostart status` | 両台の指定と見張りの状態を、読むだけで示す | 変えない |

## 終了コード (design.md 「Error Handling」)

| 値 | 意味 | 定数 |
|---|---|---|
| 0 | 正常 (すでに望む状態だった場合を含む) | `EXIT_OK` |
| 1 | 前提の不足、断り | `EXIT_PRECONDITION` |
| 2 | 実行して失敗した | `EXIT_FAILED` |
| 130 | 中断 (Ctrl-C) | `EXIT_INTERRUPTED` |

## design の文からの、意図した決めごと (design が明示しない細部を、ここで決めて残す)

1. **終了コードの写しは `main` の 1 か所**である (`_EXIT_BY_ERROR` の表を、上から順に見る)。
   「前提の不足・断り」= 1 は `ConfigError` (`plan.PlanError` を含む)、`ApprovalError`、
   `RemoteError` (ssh で入れない)、`ValueError`、`weights.WeightsRefError`。「実行しての失敗」
   = 2 は `ImageError`、`WeightsError` の残り (`WeightsMismatchError` を含む)、
   `LifecycleError`、`ProbeError`、`NetcheckError`、`PushError`。結末を表す型
   (`StartOutcome.status` など) の写し方は、各部品の docstring の表に従う
2. **traceback は出さない**。予期しない例外も、1 行にして 2 で終わる。`SERVE_DEBUG=1` の
   ときだけ、traceback を stderr に足す (`DEBUG_ENV`。口はこの 1 つだけ)
3. **例外の文と `detail` は、そのまま出す**。応答の本文や秘密が入っていないことは、各部品の
   責任である (`cli` は、そこを前提にしない。前提にすると、二重の検査で漏れが隠れる)
4. **標準出力は、後の処理が読める決まった形**だけにする。1 行に 1 つの `key=value` と、
   `|` で始まる表 (`serve status`) である。`detail` は、複数行でも 1 行に収める (改行、復帰、
   タブ、逆斜線を `\\n`、`\\r`、`\\t`、`\\\\` に書き換える)。進捗、計画、了承の問いかけ、警告、
   記録の末尾、誤りは、すべて stderr に出す。**例外は `serve image-licenses` の 1 つだけ**で、
   `key=value` の行のあとに、空の行を 1 つ置いて、読み取った表記の本文をそのまま出す
5. **`detail` は必ず出す**。stderr に全文を出し、標準出力の `key=value` にも `detail=` を
   入れる (片付けが終わらなかったことが、`detail` の末尾にだけ出る部品がある)
6. **了承は、状態を変えるコマンドだけ**が求める。計画は `guards.format_plan` が作り、
   `guards.make_confirmer(assume_yes=--yes)` が見せてから待つ。端末でなく `--yes` もなければ、
   `TerminalConfirmer` が `ApprovalError` にするので、状態を変える呼び出しは 1 つも出ない
   (終了コード 1)。`--yes` は、Claude が計測者の代わりに打つときは、会話で了承を得てから
   付ける (README)
7. **数の引数は `argparse` の `type=` で断る** (`seconds`、`count`、`repeats`、`percent`)。
   片側だけの比較 (`x < 下限`) は NaN と inf を素通しするので、`math.isfinite` と両側の比較で
   書く。`--duration 2h` のような単位 (`s`/`m`/`h`) も受ける
8. **時刻は、使うたびに取る** (`now()`)。1 回の回収や 1 回のジョブの中では同じ値を使うが、
   起こす時刻と照合の時刻のように、意味の違うものには別の値を渡す (Implementation Notes の
   2.4、4.4)
9. **`serving/` の位置は、パッケージの位置から 2 つ上**として求める (`serving_dir`)。
   リポジトリの最上位は、その 1 つ上である (`--configs` などの既定と、`payload/`、
   `weights/`、`var/` の位置を、ここから決める)
10. **`serve start` に渡す commit と、未コミットの変更の有無は、この module が `git` で
   調べる** (`repo_facts`。部品は `git` を呼ばない)。読めなければ `unknown` と「変更があるかも
   しれない」で通す (記録が欠けても、起動は止めない)
11. **`serve manifest` は、中身が同じなら、前の `generated_at` を保つ** (時刻だけの差分を
   出さない。Implementation Notes 3.2 の 6.1 への申し送り)
12. **`serve logs` に `--since` は持たない**。`logs.collect_logs` は、コンテナの記録の全量と
   `logs/` の全体を写す口で、範囲を選ぶ引数がない (design.md の `serve logs [--since]` との
   食い違い。効かない引数を足すと、範囲を絞れたと読み違えるので、足さない)
13. **`serve thinking` は、構成から宛先を組み立てる** (design.md の `--base-url` /
   `--model` のかわり)。head のホストと、構成の `--port` (`lifecycle.http_port`) と
   `served_model_name` から作るので、`bench` の対象サーバーの定義と食い違わない
14. **`serve netcheck links` は、読めても読めなくても 0 で終わる** (読み取りだけの報告で、
   合否を出す口ではない。本数を確定できなかったことは `cable_count` が空になり、理由は
   `detail` に出る)
15. **`serve push` は、了承のために見せた計画そのものを流す** (`_push_plan` が作った
   `ApprovedPlan.forward` を、書いた順に `_run_step` に渡す)。計画と実行を別々に書くと、
   片方にだけ宛先を足す誤りが、どちらの試験にも掛からない (requirements 2.1)
16. **`serve logs <構成>` は、構成の名前を受ける**。`logs.collect_logs` は、記録を読む相手を
   `plan.build_plans` が作った `ContainerPlan` でしか受けない (名前を文字列で渡す口がない)
   ためで、回収先は design.md 「記録の置き場所」の `serving/var/<日時>-logs-<構成>/` になる
17. **`serve verify` は、`kind = "probe"` の構成では `--probe-files` を書かなくても
   `probe_files` の範囲で照合する**。縮小の確認の構成が読む照合の記録は、`guards` が
   `<remote_root>/state/<slug>.probe.verified.json` に固定していて (design.md 「関門」)、
   全体の範囲の記録では、その構成の関門が通らないためである
18. **`serve derived-import` は、Spark にも Hub にも触らない** (`derived` の読み取りと変換と、
   `serving/weights/` への書き込みだけ)。2 台ぶんの manifest は、最上位の `generated_at` を除いて
   一致するときだけ取り込み、違う箇所を示して終了コード 1 にする。宛先に Hub のマニフェストが
   あれば上書きしない。中身が同じなら、`serve manifest` と同じく、前の `generated_at` を保つ
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TextIO

import httpx

from serving_kit import (
    __version__,
    derived,
    guards,
    image,
    lifecycle,
    logs,
    netcheck,
    plan,
    probe,
    thinking,
    watch,
)
from serving_kit import (
    autostart as autostart_mod,
)
from serving_kit import config as config_mod
from serving_kit import weights as weights_mod
from serving_kit.remote import RemoteError, RemoteRunner, SshRunner
from serving_kit.types import (
    AnyWeightsManifest,
    ConfigDef,
    DerivedWeightsManifest,
    GateResult,
    LaunchObservation,
    LinkReport,
    NodeDef,
    NodeRole,
    NodeStatus,
    PlannedCommand,
    PlannedPush,
    PlannedRun,
    ServiceStatus,
    WeightsManifest,
)

__all__ = [
    "DEBUG_ENV",
    "EXIT_FAILED",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "EXIT_PRECONDITION",
    "MKDIR_TIMEOUT_S",
    "SPARK_DIRS",
    "UNREADABLE_MARK",
    "PushError",
    "build_parser",
    "celsius",
    "count",
    "env_pair",
    "main",
    "percent",
    "repeats",
    "repo_facts",
    "seconds",
    "serving_dir",
]

EXIT_OK: Final[int] = 0
"""正常に終わった (design.md Error Handling)。"""

EXIT_PRECONDITION: Final[int] = 1
"""前提の不足、または断り (design.md Error Handling)。

サブコマンドを 1 つも選ばずに呼んだ場合も、ここに当たる。
"""

EXIT_FAILED: Final[int] = 2
"""実行して失敗した (design.md Error Handling)。"""

EXIT_INTERRUPTED: Final[int] = 130
"""中断 (Ctrl-C) (design.md Error Handling)。"""

DEBUG_ENV: Final[str] = "SERVE_DEBUG"
"""この環境変数が `1` のときだけ、予期しない例外の traceback を stderr に出す。"""

UNREADABLE_MARK: Final[str] = "読めなかった"
"""`serve status` の表で、一覧を読めなかった台に付ける印。"""

SPARK_DIRS: Final[tuple[str, ...]] = ("payload", "models", "probe", "cache", "logs", "state")
"""`serve push` が 2 台に作る 6 つの置き場所 (`remote_root` の下)。

名前は、部品が実際に使うもので固定してある: `payload` は配布の宛先 (`remote._PUSH_SUBDIRS`)、
`state` は起動の記録と照合の記録 (`lifecycle.LAUNCH_RECORD_SUBDIR`、`weights.RECORD_SUBDIR`)、
`logs` は通信の記録 (`logs.COMM_LOG_REMOTE_SUBDIR`)、`models` と `probe` は重みと縮小の確認用、
`cache` は JIT のキャッシュ (どれも、構成の `--mount` が指す)。
"""

MKDIR_TIMEOUT_S: Final[float] = 60.0
"""置き場所を作る `mkdir -p` の時間切れ。"""

_ROLE_ORDER: Final[tuple[NodeRole, ...]] = ("head", "worker")
"""台を見る順序 (head → worker)。"""

_PROBE_FILES: Final[str] = "probe_files"
_ALL_FILES: Final[str] = "all"

_UNKNOWN_COMMIT: Final[str] = "unknown"
"""`git` を読めなかったときに、起動の記録に書く commit。"""

_SECOND_SUFFIXES: Final[Mapping[str, float]] = {"s": 1.0, "m": 60.0, "h": 3600.0}


class PushError(Exception):
    """配布 (`serve push`) を実行して失敗した (終了コード 2)。

    置き場所を作れなかった、`payload/` を送れなかった、のどちらかである。届かなかったこと
    (ssh で入れない) は `remote.RemoteError` のままで、終了コード 1 になる。
    """


# --- 引数の型 (Spark にも推論サーバーにも触る前に断る) ----------------------


def seconds(text: str) -> float:
    """秒を読む (`30`、`30s`、`5m`、`2h`)。正で有限の値だけを受ける。

    **片側だけの比較では、NaN と inf が通る**ので、`math.isfinite` で見る
    (Implementation Notes 4.5 の決まり。`--timeout`、`--duration`、`--interval`、
    `--stall-window` が、これを使う)。
    """
    body, scale = text, 1.0
    if text and text[-1] in _SECOND_SUFFIXES:
        body, scale = text[:-1], _SECOND_SUFFIXES[text[-1]]
    try:
        value = float(body) * scale
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"秒の値として読めない (例: 30、30s、5m、2h): {text!r}"
        ) from None
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"秒の値は、正で有限の数にする: {text!r}")
    return value


def count(text: str) -> int:
    """回数を読む (1 以上の整数だけ)。小数、NaN、真偽の語は断る。"""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"1 以上の整数にする: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"1 以上の整数にする: {text!r}")
    return value


def repeats(text: str) -> int:
    """A/B の 1 腕あたりの回数 (`--repeat`。2 以上。1 回ずつでは、範囲が重なるかを見られない)。"""
    value = count(text)
    if value < 2:
        raise argparse.ArgumentTypeError(f"A/B の回数は 2 以上にする: {text!r}")
    return value


def percent(text: str) -> int:
    """割合を読む (0 から 100 の整数)。"""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"0 から 100 の整数にする: {text!r}") from None
    if not 0 <= value <= 100:
        raise argparse.ArgumentTypeError(f"0 から 100 の整数にする: {text!r}")
    return value


def celsius(text: str) -> float:
    """摂氏の温度を読む (`--thermal-threshold`)。正で有限の値だけを受ける。

    `percent` (0〜100 の割合) も `count` (回数) も意味が合わないので、専用の型にする。
    片側だけの比較では NaN と inf が通るので、`math.isfinite` で見る (`seconds` と同じ decision)。
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"摂氏の温度は、数値にする: {text!r}") from None
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"摂氏の温度は、正で有限の数にする: {text!r}")
    return value


def env_pair(text: str) -> tuple[str, str]:
    """`KEY=VALUE` を読む (名前の決まりは、構成の `env` と同じものを `plan` が掛ける)。"""
    name, _, value = text.partition("=")
    if not name or not value or name != name.strip():
        raise argparse.ArgumentTypeError(f"KEY=VALUE の形で書く: {text!r}")
    return name, value


# --- 場所 -----------------------------------------------------------------


def serving_dir() -> Path:
    """`serving/` の位置 (この module から見て、`src/serving_kit` の 2 つ上。決めごとの 9)。"""
    return Path(__file__).resolve().parents[2]


def repo_facts(repo_root: Path, *, timeout_s: float = 30.0) -> tuple[str, bool]:
    """リポジトリの commit と、未コミットの変更の有無を `git` から読む (決めごとの 10)。

    読めなければ `("unknown", True)` を返す (起動の記録のための事実で、起動の判定には使わない
    ので、読めないことを理由に止めない)。
    """
    head = _git(repo_root, ("rev-parse", "HEAD"), timeout_s)
    if head is None:
        return _UNKNOWN_COMMIT, True
    changed = _git(repo_root, ("status", "--porcelain"), timeout_s)
    return head.strip() or _UNKNOWN_COMMIT, changed is None or changed.strip() != ""


def _git(repo_root: Path, rest: Sequence[str], timeout_s: float) -> str | None:
    """`git` を 1 回だけ流し、読めなければ `None` を返す。"""
    try:
        done = subprocess.run(
            ["git", "-C", str(repo_root), *rest],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return None if done.returncode != 0 else done.stdout


# --- 表示 -----------------------------------------------------------------


def _one_line(text: str) -> str:
    """`key=value` の行に収める (改行、復帰、タブ、逆斜線を書き換える。決めごとの 4)。"""
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _shown(value: object) -> str:
    """値を、`key=value` の右側にする (空は空の文字列、真偽は `true` / `false`)。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return _one_line(str(value))


def _cell(value: object) -> str:
    """表の欄にする (`|` を含む値は、行の区切りを壊すので書き換える)。"""
    shown = _shown(value)
    return shown.replace("|", "/") if shown else "-"


@dataclass(frozen=True)
class _Out:
    """標準出力と標準エラーの、決まった書き方 (決めごとの 4)。"""

    out: TextIO
    err: TextIO

    def say(self, key: str, value: object) -> None:
        """`key=value` の行を 1 つ出す。"""
        self.out.write(f"{key}={_shown(value)}\n")

    def table(self, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        """表を出す (`|` で始まる行。人も、後の処理も読める)。"""
        self.out.write("| " + " | ".join(header) + " |\n")
        self.out.write("| " + " | ".join("---" for _ in header) + " |\n")
        for row in rows:
            self.out.write("| " + " | ".join(_cell(one) for one in row) + " |\n")

    def note(self, text: str) -> None:
        """進捗、警告、計画、記録の末尾を stderr に出す。"""
        self.err.write(f"{text}\n")
        self.err.flush()

    def detail(self, text: str) -> None:
        """`detail` を、stderr に全文、標準出力に 1 行で出す (決めごとの 5)。"""
        if text:
            self.note(text)
        self.say("detail", text)

    def gates(self, results: Sequence[GateResult], *, only_failed: bool) -> None:
        """関門の結果を `key=value` で並べる。"""
        shown = [one for one in results if not one.passed] if only_failed else list(results)
        for index, gate in enumerate(shown, start=1):
            self.say(f"gate.{index}.name", gate.gate)
            self.say(f"gate.{index}.node", gate.node)
            self.say(f"gate.{index}.passed", gate.passed)
            self.say(f"gate.{index}.detail", gate.detail)


# --- 1 回の実行の文脈 -------------------------------------------------------


@dataclass
class _Context:
    """1 回の `main` が使う、場所と口 (試験は、ここを差し替える)。"""

    args: argparse.Namespace
    show: _Out
    repo_root: Path
    var_root: Path
    runner_factory: Callable[[Path], RemoteRunner]
    client_factory: Callable[[float], httpx.Client] | None
    given_confirmer: guards.Confirmer | None
    stdin: TextIO
    now: Callable[[], datetime]
    sleep: Callable[[float], None] | None
    clock: Callable[[], float] | None
    given_facts: tuple[str, bool] | None
    _runner: RemoteRunner | None = field(default=None, init=False)
    _confirmer: guards.Confirmer | None = field(default=None, init=False)

    # --- 場所 ---

    @property
    def serving(self) -> Path:
        """このリポジトリの `serving/` (決めごとの 9)。"""
        return self.repo_root / "serving"

    def runner(self) -> RemoteRunner:
        """遠隔の実行役 (要るときに 1 つだけ作る。Spark に触らない道では作らない)。"""
        if self._runner is None:
            self._runner = self.runner_factory(self.var_root)
        return self._runner

    def confirmer(self) -> guards.Confirmer:
        """了承の口 (`--yes` の有無で、`guards` が 2 つの実装を選ぶ)。"""
        if self._confirmer is None:
            self._confirmer = self.given_confirmer or guards.make_confirmer(
                assume_yes=bool(self.args.yes), stdin=self.stdin, stderr=self.show.err
            )
        return self._confirmer

    def client(self, timeout_s: float) -> httpx.Client | None:
        """HTTP のクライアント (`None` なら、部品が自分で作って閉じる)。

        差し込んだ口が返したクライアントは、**部品も `cli` も閉じない** (部品は、自分で
        作ったものだけを閉じる)。差し込んだ側 (試験) が後始末をする。
        """
        return None if self.client_factory is None else self.client_factory(timeout_s)

    def facts(self) -> tuple[str, bool]:
        """起動の記録に書く、リポジトリの commit と未コミットの変更の有無。"""
        return self.given_facts if self.given_facts is not None else repo_facts(self.repo_root)

    # --- 定義の読み込み ---

    def configs(self) -> dict[str, ConfigDef]:
        path = self.args.configs or self.serving / "config" / "configs.toml"
        return config_mod.load_configs(Path(path), self.repo_root)

    def nodes(self) -> dict[NodeRole, NodeDef]:
        path = self.args.nodes or self.serving / "config" / "nodes.toml"
        return config_mod.load_nodes(Path(path), self.repo_root)

    def selected(self) -> tuple[ConfigDef, dict[NodeRole, NodeDef]]:
        """`<構成>` の位置の引数で選んだ構成と、ノードの定義。

        構成の定義を先に読む (位置の引数が指すものなので、誤りの文がわかりやすい)。
        """
        configs = self.configs()
        nodes = self.nodes()
        return config_mod.select_config(configs, str(self.args.config), nodes), nodes

    def manifest(self, config: ConfigDef) -> AnyWeightsManifest | None:
        """構成が指す重みのマニフェストを読む (重みを持たない構成では `None`)。"""
        if config.weights is None:
            return None
        return weights_mod.load_manifest(self.serving / "weights" / config.weights.manifest)

    def required_manifest(self, config: ConfigDef) -> AnyWeightsManifest:
        """重みを持つことを前提にして読む (持たない構成は、Spark に触る前に断る)。"""
        manifest = self.manifest(config)
        if manifest is None:
            raise ValueError(f"構成 '{config.name}' は重みを持たないので、このコマンドは使えない")
        return manifest

    def record_dir(self, command: str, config_name: str, started_at: datetime) -> Path:
        """記録と、配る元を置く場所 (`serving/var/<日時>-<コマンド>-<構成>/`)。"""
        return logs.var_dir(self.var_root, started_at, command, config_name)


def _ordered(nodes: Mapping[NodeRole, NodeDef]) -> tuple[NodeDef, ...]:
    """定義にある台を、head → worker の順に並べる。"""
    return tuple(nodes[role] for role in _ROLE_ORDER if role in nodes)


# --- 引数の解析 -------------------------------------------------------------

_EXIT_CODE_EPILOG: Final[str] = (
    "終了コード: 0 正常 / 1 前提の不足または断り / 2 実行しての失敗 / 130 中断\n"
)


def _common(parser: argparse.ArgumentParser) -> None:
    """すべてのコマンドに共通の引数 (design.md 「入口 › cli」)。"""
    parser.add_argument(
        "--configs",
        type=Path,
        default=None,
        metavar="PATH",
        help="構成の定義のファイル (既定: serving/config/configs.toml)",
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=None,
        metavar="PATH",
        help="ノードの定義のファイル (既定: serving/config/nodes.toml)",
    )
    parser.add_argument(
        "--var-root",
        type=Path,
        default=None,
        metavar="PATH",
        help="記録の置き場所の根 (既定: serving/var/。git の管理の外)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="計画を見せたうえで、了承を求めずに進む (会話で了承を得てから付ける)",
    )


def _leaf(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser], name: str, help_text: str
) -> argparse.ArgumentParser:
    """1 つのサブコマンドを足す (共通の引数と、終了コードの説明つき)。"""
    parser = subparsers.add_parser(
        name,
        help=help_text,
        description=help_text,
        epilog=_EXIT_CODE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _common(parser)
    return parser


def _with_config(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser], name: str, help_text: str
) -> argparse.ArgumentParser:
    """`<構成>` を 1 つ受けるサブコマンドを足す。"""
    parser = _leaf(subparsers, name, help_text)
    parser.add_argument("config", metavar="<構成>", help="構成の定義の名前")
    return parser


def build_parser() -> argparse.ArgumentParser:
    """`serve` の引数解析器を組み立てる。

    `--help` と `--version`、引数の使い方の誤りは、`argparse` が `SystemExit` を送出する
    (`bench` と同じく、呼び出し元にそのまま伝播させる。使い方の誤りは `argparse` の決まりで 2)。
    """
    parser = argparse.ArgumentParser(
        prog="serve",
        description="DGX Spark 2 台に vLLM の推論サーバーを立てるための道具 (Serving Kit)",
        epilog=_EXIT_CODE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", title="commands", metavar="<command>")

    _with_config(subparsers, "check", "構成を検査し、関門をすべて流して結果を並べる")
    _leaf(subparsers, "push", "2 台に 6 つの置き場所を作り、serving/payload/ を配る")
    _with_config(subparsers, "pull-image", "イメージを、ダイジェストで 2 台に取得する")
    _with_config(subparsers, "image-licenses", "イメージの中のライセンスの表記を読んで出す")

    manifest = _leaf(subparsers, "manifest", "重みのマニフェストを Mac で作る (Spark に触らない)")
    manifest.add_argument("repo", metavar="<repo>", help="Hugging Face Hub のリポジトリの名前")
    manifest.add_argument("revision", metavar="<revision>", help="40 桁の commit sha")

    derived_import = _leaf(
        subparsers,
        "derived-import",
        "変換の道具が書いた manifest.json を、派生の重みのマニフェストとして"
        " serving/weights/ に取り込む (Spark に触らない)",
    )
    derived_import.add_argument(
        "--name",
        required=True,
        metavar="<名前>",
        help="派生の重みの名前 (英数字とハイフン。例: k2s1)。マニフェストの名前になる",
    )
    derived_import.add_argument(
        "--commit",
        required=True,
        metavar="<40桁>",
        help="変換に使った道具を含むコミット (40 桁の 16 進)",
    )
    derived_import.add_argument(
        "--tool",
        default=derived.DEFAULT_TOOL_PATH,
        metavar="<道筋>",
        help=f"リポジトリの中の、変換の道具の道筋 (既定: {derived.DEFAULT_TOOL_PATH})",
    )
    derived_import.add_argument(
        "manifests",
        nargs="+",
        type=Path,
        metavar="<manifest.json>",
        help="2 台ぶんの manifest.json (最上位の generated_at を除いて一致しなければ断る)",
    )

    fetch = _with_config(subparsers, "fetch", "重みを 2 台に取得して照合する")
    fetch.add_argument(
        "--probe-files",
        action="store_true",
        help="safetensors を除いたファイル (設定とトークナイザ) だけを照合する",
    )
    verify = _with_config(subparsers, "verify", "2 台の重みを、マニフェストと照合する")
    verify.add_argument(
        "--probe-files",
        action="store_true",
        help="safetensors を除いたファイルだけを照合する (probe の構成では、つねにこの範囲)",
    )

    start = _with_config(subparsers, "start", "2 台で起こし、受け付けの開始まで待つ")
    start.add_argument(
        "--timeout",
        type=seconds,
        default=None,
        metavar="<秒>",
        help="受け付けの開始を待つ上限を、その回だけ上書きする (例: 2h)",
    )
    _leaf(subparsers, "stop", "自分のコンテナを 2 台とも止めて消す (kind を問わない)")
    _leaf(subparsers, "status", "2 台と推論サーバーの、いまの状態を示す")
    smoke = _with_config(subparsers, "smoke", "英語と日本語の短い要求を 1 つずつ送る")
    smoke.add_argument(
        "--max-tokens",
        type=count,
        default=lifecycle.SMOKE_MAX_TOKENS,
        metavar="<数>",
        help=f"応答の長さの上限を、その回だけ上書きする (既定: {lifecycle.SMOKE_MAX_TOKENS})",
    )
    _with_config(subparsers, "logs", "2 台の記録を serving/var/ に写す")

    probe_parser = _with_config(subparsers, "probe", "1 台の縮小の確認を流す")
    probe_parser.add_argument(
        "--timeout",
        type=seconds,
        default=None,
        metavar="<秒>",
        help="受け付けの開始を待つ上限を、その回だけ上書きする (例: 45m)",
    )

    _add_netcheck(subparsers)
    _add_watch(subparsers)
    _add_thinking(subparsers)
    _add_autostart(subparsers)
    return parser


def _add_netcheck(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """`serve netcheck links / bandwidth / sanity / ab`。"""
    parser = subparsers.add_parser(
        "netcheck",
        help="2 台の間の通信を確かめる",
        description="2 台の間の通信を確かめる (links は読み取りだけ)",
        epilog=_EXIT_CODE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inner = parser.add_subparsers(dest="netcheck_command", title="checks", metavar="<check>")
    _leaf(inner, "links", "直結のインターフェースを読み、ケーブルの本数を判断する")
    _with_config(inner, "bandwidth", "自前の all-reduce で、2 台の間の帯域を測る")
    _with_config(inner, "sanity", "推論サーバーの公式の資料が示す、事前の確認を流す")
    ab = _with_config(inner, "ab", "足す設定の A/B を、交互に流して比べる")
    ab.add_argument(
        "--env",
        type=env_pair,
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="足す環境変数 (複数書ける。1 つ以上が要る)",
    )
    ab.add_argument(
        "--repeat",
        type=repeats,
        default=netcheck.AB_REPEATS,
        metavar="<回>",
        help=f"腕ごとの回数 (2 以上。既定: {netcheck.AB_REPEATS})",
    )


def _add_autostart(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """`serve autostart set / clear / status` (issue #88 P6)。`_add_netcheck` と同じ形。"""
    parser = subparsers.add_parser(
        "autostart",
        help="Spark 上の見張りが自動で起こす構成を指定・解除・確認する",
        description="Spark 上の見張り (ops/vllm-autostart) が自動で起こす構成を指定・解除・確認",
        epilog=_EXIT_CODE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inner = parser.add_subparsers(dest="autostart_command", title="commands", metavar="<command>")
    _with_config(inner, "set", "指定した構成を、Spark 上の見張りが自動で起こす対象にする")
    _leaf(inner, "clear", "自動で起こす対象の指定を解除する (以後、見張りは何も起こさない)")
    _leaf(inner, "status", "両台の指定と見張りの状態を、読むだけで示す")


def _add_watch(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = _with_config(subparsers, "watch", "連続の負荷の間、外から見張る (読み取りだけ)")
    parser.add_argument(
        "--duration",
        type=seconds,
        default=watch.DEFAULT_DURATION_S,
        metavar="<秒>",
        help="見張りを続ける長さ (例: 2h)",
    )
    parser.add_argument(
        "--interval",
        type=seconds,
        default=watch.DEFAULT_INTERVAL_S,
        metavar="<秒>",
        help="観察の間隔 (例: 30s)",
    )
    parser.add_argument(
        "--stall-window",
        type=seconds,
        default=watch.DEFAULT_STALL_WINDOW_S,
        metavar="<秒>",
        help="「固まった」と判定する、トークンの数が増えない長さ (例: 5m)",
    )
    parser.add_argument(
        "--stall-gpu-threshold",
        type=percent,
        default=watch.DEFAULT_STALL_GPU_THRESHOLD_PCT,
        metavar="<%>",
        help="「固まった」と判定する、GPU の使用率のしきい値 (0〜100)",
    )
    parser.add_argument(
        "--unresponsive-threshold",
        type=count,
        default=watch.DEFAULT_UNRESPONSIVE_THRESHOLD,
        metavar="<回>",
        help="「応答しない」と判定する、続けての失敗の回数",
    )
    parser.add_argument(
        "--thermal-threshold",
        type=celsius,
        default=watch.DEFAULT_THERMAL_THRESHOLD_C,
        metavar="<℃>",
        help="熱区域の温度が、この値以上になった観察で thermal の出来事を出す (既定 90)",
    )


def _add_thinking(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = _with_config(subparsers, "thinking", "thinking の深さの渡し方を確かめる")
    parser.add_argument(
        "--trials",
        type=count,
        default=thinking.DEFAULT_TRIALS,
        metavar="<回>",
        help="6 通りのそれぞれを送る回数",
    )
    parser.add_argument(
        "--max-tokens",
        type=count,
        default=thinking.DEFAULT_MAX_TOKENS,
        metavar="<数>",
        help="応答の長さの上限",
    )
    parser.add_argument(
        "--timeout",
        type=seconds,
        default=thinking.DEFAULT_TIMEOUT_S,
        metavar="<秒>",
        help="1 要求あたりの時間切れ",
    )


# --- serve check ----------------------------------------------------------


def _cmd_check(ctx: _Context) -> int:
    """構成の検査と、9 つの関門の結果を並べる (requirements 3.7、2.2、2.5)。"""
    config, nodes = ctx.selected()
    plans = plan.build_plans(config, nodes, ctx.now())
    manifest = ctx.manifest(config)
    results = guards.run_gates(ctx.runner(), config, nodes, plans, manifest=manifest)
    failed = [one for one in results if not one.passed]
    ctx.show.say("config", config.name)
    ctx.show.say("kind", config.kind)
    ctx.show.gates(results, only_failed=False)
    ctx.show.say("gates_failed", len(failed))
    ctx.show.say("status", "refused" if failed else "passed")
    ctx.show.detail(
        f"{len(results)} 件の関門を流し、{len(failed)} 件が通らなかった"
        if failed
        else f"{len(results)} 件の関門がすべて通った"
    )
    return EXIT_PRECONDITION if failed else EXIT_OK


# --- serve push -----------------------------------------------------------


def _mkdir_argv(node: NodeDef) -> tuple[str, ...]:
    """1 台に、6 つの置き場所をまとめて作る列 (`remote` の許可の一覧を通る形)。"""
    return ("mkdir", "-p", *(f"{node.remote_root}/{name}" for name in SPARK_DIRS))


def _push_plan(targets: Sequence[NodeDef], payload: Path) -> tuple[PlannedCommand, ...]:
    """`serve push` の、前に進むコマンドの列 (これが、見せる計画であり、流す列でもある)。

    2 台に 6 つの置き場所を作ってから、`payload/` だけを配る。**宛先は `payload/` だけ**で、
    `state/` は `lifecycle` / `weights` が、自分の記録を置くときに配る。
    """
    made: list[PlannedCommand] = [
        PlannedRun(
            node=node.role,
            argv=_mkdir_argv(node),
            purpose=f"{node.role} に、6 つの置き場所を作る ({', '.join(SPARK_DIRS)})",
        )
        for node in targets
    ]
    made.extend(
        PlannedPush(
            node=node.role,
            local_dir=payload,
            remote_subdir="payload",
            delete=False,
            purpose=f"{node.role} に、serving/payload/ を配る",
        )
        for node in targets
    )
    return tuple(made)


def _run_step(
    runner: RemoteRunner, nodes: Mapping[NodeRole, NodeDef], step: PlannedCommand
) -> None:
    """了承を得た計画の 1 つを、そのまま流す (0 以外で終わったら `PushError`)。"""
    node = nodes[step.node]
    if isinstance(step, PlannedPush):
        result = runner.push(node, step.local_dir, step.remote_subdir, delete=step.delete)
    else:
        result = runner.run(node, step.argv, timeout_s=MKDIR_TIMEOUT_S, mutating=True)
    if result.exit_code != 0:
        raise PushError(
            f"{node.role} ({node.ssh_host}) で「{step.purpose}」ができなかった"
            f" (終了コード {result.exit_code}): {result.stderr.strip()}"
        )


def _cmd_push(ctx: _Context) -> int:
    """6 つの置き場所を作ってから、`payload/` だけを配る (requirements 1.1、1.2、2.1)。"""
    nodes = ctx.nodes()
    targets = _ordered(nodes)
    if not targets:
        raise ValueError("nodes に head も worker もない (配る先がない)")
    payload = ctx.serving / "payload"
    if not payload.is_dir():
        raise ValueError(f"配るものがない (ディレクトリがない): {payload}")

    # 巻き戻しは空 (置き場所は消さない)。`build_approved_plan` は、コンテナを起こす計画が
    # 空なら、巻き戻しも空にする
    approved = guards.build_approved_plan((), extra_forward=_push_plan(targets, payload))
    runner = ctx.runner()
    guards.request_approval(ctx.confirmer(), runner, approved, nodes)

    # **見せた計画そのものを、書いた順に流す** (決めごとの 15)
    for step in approved.forward:
        _run_step(runner, nodes, step)

    ctx.show.say("status", "pushed")
    ctx.show.say("nodes", ",".join(node.role for node in targets))
    ctx.show.say("directories", ",".join(SPARK_DIRS))
    ctx.show.say("payload_from", payload)
    ctx.show.say("payload_to", "payload")
    ctx.show.detail(
        f"{len(targets)} 台に、{len(SPARK_DIRS)} つの置き場所を作ってから、"
        "payload/ だけを配った (置き場所は消さない)"
    )
    return EXIT_OK


# --- serve pull-image / image-licenses ------------------------------------


def _cmd_pull_image(ctx: _Context) -> int:
    """イメージを、ダイジェストで 2 台に取得して照合する (requirements 3.2、3.3)。"""
    config, nodes = ctx.selected()
    outcome = image.pull_image(ctx.runner(), config, nodes, confirmer=ctx.confirmer())
    ctx.show.say("config", outcome.config_name)
    ctx.show.say("status", outcome.status)
    ctx.show.say("reference", outcome.reference)
    ctx.show.gates(outcome.gates, only_failed=True)
    ctx.show.detail(outcome.detail)
    return EXIT_OK if outcome.status == "pulled" else EXIT_PRECONDITION


def _cmd_image_licenses(ctx: _Context) -> int:
    """イメージの中のライセンスの表記を読んで出す (requirements 3.8、3.9)。"""
    config, nodes = ctx.selected()
    outcome = image.read_image_licenses(
        ctx.runner(),
        config,
        nodes,
        ctx.now(),
        confirmer=ctx.confirmer(),
        sleep=ctx.sleep,
        clock=ctx.clock,
        report=ctx.show.err,
    )
    ctx.show.say("config", outcome.config_name)
    ctx.show.say("status", outcome.status)
    ctx.show.say("readings", len(outcome.readings))
    ctx.show.gates(outcome.gates, only_failed=True)
    ctx.show.detail(outcome.detail)
    if outcome.status == "read":
        # 決めごとの 4 の例外: 読み取った表記の本文だけは、決まった形でなく、そのまま出す
        ctx.show.out.write(f"\n{outcome.text}\n")
        return EXIT_OK
    return EXIT_PRECONDITION


# --- serve manifest -------------------------------------------------------


def _cmd_manifest(ctx: _Context) -> int:
    """重みのマニフェストを Mac で作る (Spark に触らない。requirements 3.4)。"""
    repo = str(ctx.args.repo)
    result = weights_mod.build_manifest(repo, str(ctx.args.revision), generated_at=ctx.now())
    manifest = result.manifest
    path = weights_mod.manifest_path(repo, weights_dir=ctx.serving / "weights")
    reused = _reuse_generated_at(manifest, path)
    if reused is not None:
        manifest = reused
    path.parent.mkdir(parents=True, exist_ok=True)
    weights_mod.write_manifest(manifest, path)
    ctx.show.say("repo", manifest.repo)
    ctx.show.say("revision", manifest.revision)
    ctx.show.say("path", path)
    ctx.show.say("file_count", len(manifest.files))
    ctx.show.say("total_bytes", manifest.total_bytes)
    ctx.show.say("excluded", ",".join(result.excluded_paths))
    ctx.show.say("generated_at", manifest.generated_at.isoformat())
    ctx.show.say("reused_generated_at", reused is not None)
    ctx.show.say("status", "written")
    ctx.show.detail(
        f"{len(manifest.files)} ファイル、{manifest.total_bytes} バイトのマニフェストを書いた"
        + (f" (除いたもの: {', '.join(result.excluded_paths)})" if result.excluded_paths else "")
    )
    return EXIT_OK


def _reuse_generated_at[M: (WeightsManifest, DerivedWeightsManifest)](
    manifest: M, path: Path
) -> M | None:
    """前のマニフェストと中身が同じなら、前の `generated_at` を保つ (決めごとの 11)。

    前のマニフェストが別の種類 (Hub と派生) なら、中身は同じにならないので `None` を返す。
    """
    if not path.is_file():
        return None
    try:
        previous = weights_mod.load_manifest(path)
    except (weights_mod.WeightsError, ValueError, OSError):
        return None
    if type(previous) is not type(manifest):
        return None
    if previous.model_copy(update={"generated_at": manifest.generated_at}) != manifest:
        return None
    return manifest.model_copy(update={"generated_at": previous.generated_at})


# --- serve derived-import -------------------------------------------------


def _cmd_derived_import(ctx: _Context) -> int:
    """道具の manifest を、派生の重みのマニフェストとして取り込む (決めごとの 18)。"""
    tools = [derived.read_tool_manifest(source) for source in ctx.args.manifests]
    differences = derived.compare_tool_manifests(tools)
    if differences:
        raise weights_mod.WeightsRefError(
            "2 台のマニフェストが一致しない: " + " / ".join(differences)
        )
    name = str(ctx.args.name)
    result = derived.import_tool_manifest(
        tools[0],
        name=name,
        tool_path=str(ctx.args.tool),
        commit=str(ctx.args.commit),
        generated_at=ctx.now(),
    )
    manifest = result.manifest
    path = ctx.serving / "weights" / f"{name}.manifest.json"
    if path.is_file() and isinstance(weights_mod.load_manifest(path), WeightsManifest):
        raise weights_mod.WeightsRefError(
            f"'{path}' は Hub の重みのマニフェストなので、派生の重みで上書きしない (別の名前を選ぶ)"
        )
    reused = _reuse_generated_at(manifest, path)
    if reused is not None:
        manifest = reused
    path.parent.mkdir(parents=True, exist_ok=True)
    weights_mod.write_manifest(manifest, path)
    conversion = manifest.derivation.conversion
    ctx.show.say("name", name)
    ctx.show.say("path", path)
    ctx.show.say(
        "origin", manifest.derivation.origin.repo + "@" + manifest.derivation.origin.revision
    )
    ctx.show.say("tool", conversion.tool)
    ctx.show.say("commit", conversion.commit)
    ctx.show.say("target_pattern", conversion.target_pattern)
    ctx.show.say("file_count", len(manifest.files))
    ctx.show.say("total_bytes", manifest.total_bytes)
    ctx.show.say("excluded", ",".join(result.excluded_paths))
    ctx.show.say("inputs", len(tools))
    ctx.show.say("manifest_sha256", manifest.content_sha256)
    ctx.show.say("generated_at", manifest.generated_at.isoformat())
    ctx.show.say("reused_generated_at", reused is not None)
    ctx.show.say("status", "written")
    ctx.show.detail(
        f"{len(tools)} 台ぶんの manifest が一致した。{len(manifest.files)} ファイル、"
        f"{manifest.total_bytes} バイトの派生のマニフェストを書いた"
        + (f" (除いたもの: {', '.join(result.excluded_paths)})" if result.excluded_paths else "")
    )
    return EXIT_OK


# --- serve fetch / verify -------------------------------------------------


def _cmd_fetch(ctx: _Context) -> int:
    """重みを 2 台に取得して照合する (requirements 2.5、2.6、3.4、3.5)。"""
    config, nodes = ctx.selected()
    manifest = ctx.required_manifest(config)
    started_at = ctx.now()
    scope = _PROBE_FILES if ctx.args.probe_files else _ALL_FILES
    outcome = weights_mod.fetch_weights(
        ctx.runner(),
        config,
        nodes,
        manifest,
        started_at,
        confirmer=ctx.confirmer(),
        record_dir=ctx.record_dir("fetch", config.name, started_at),
        verified_at=ctx.now(),
        scope=scope,  # type: ignore[arg-type]
        sleep=ctx.sleep,
        clock=ctx.clock,
        report=ctx.show.err,
    )
    ctx.show.say("config", outcome.config_name)
    ctx.show.say("scope", outcome.scope)
    ctx.show.say("status", outcome.status)
    ctx.show.say("nodes", ",".join(one.node for one in outcome.fetches))
    ctx.show.say("mismatched", _mismatched(outcome.verifications))
    ctx.show.gates(outcome.gates, only_failed=True)
    ctx.show.detail(outcome.detail)
    return EXIT_OK if outcome.status == "fetched" else EXIT_PRECONDITION


def _cmd_verify(ctx: _Context) -> int:
    """2 台の重みを、マニフェストと照合する (requirements 3.5)。"""
    config, nodes = ctx.selected()
    manifest = ctx.required_manifest(config)
    started_at = ctx.now()
    narrow = bool(ctx.args.probe_files) or config.kind == "probe"
    outcome = weights_mod.verify_weights(
        ctx.runner(),
        config,
        nodes,
        manifest,
        started_at,
        confirmer=ctx.confirmer(),
        record_dir=ctx.record_dir("verify", config.name, started_at),
        verified_at=ctx.now(),
        scope=_PROBE_FILES if narrow else _ALL_FILES,  # type: ignore[arg-type]
    )
    ctx.show.say("config", outcome.config_name)
    ctx.show.say("scope", outcome.scope)
    ctx.show.say("status", outcome.status)
    ctx.show.say("nodes", ",".join(one.node for one in outcome.verifications))
    ctx.show.say("mismatched", _mismatched(outcome.verifications))
    ctx.show.gates(outcome.gates, only_failed=True)
    ctx.show.detail(outcome.detail)
    return EXIT_OK if outcome.status == "verified" else EXIT_PRECONDITION


def _mismatched(verifications: Sequence[weights_mod.NodeVerification]) -> str:
    """合わなかったファイルの名前 (台ごとに、そのまま並べる)。"""
    names: list[str] = []
    for one in verifications:
        names.extend(f"{one.node}:{path}" for path in one.mismatched)
    return ",".join(names)


# --- serve start / stop / status / smoke ----------------------------------


def _cmd_start(ctx: _Context) -> int:
    """2 台で起こし、受け付けの開始まで待つ (requirements 1.3、1.4、1.5、1.8)。"""
    config, nodes = ctx.selected()
    started_at = ctx.now()
    commit, dirty = ctx.facts()
    outcome = lifecycle.start(
        ctx.runner(),
        config,
        nodes,
        started_at,
        confirmer=ctx.confirmer(),
        var_root=ctx.var_root,
        record_dir=ctx.record_dir(lifecycle.COMMAND_NAME, config.name, started_at),
        repo_commit=commit,
        repo_dirty=dirty,
        manifest=ctx.manifest(config),
        timeout_s=ctx.args.timeout,
        sleep=ctx.sleep,
        clock=ctx.clock,
        report=ctx.show.err,
        client=ctx.client(lifecycle.HTTP_TIMEOUT_S),
    )
    ctx.show.say("config", config.name)
    ctx.show.say("status", outcome.status)
    ctx.show.gates(outcome.gates, only_failed=True)
    _say_observation(ctx, outcome.observation)
    _say_service(ctx, outcome.service)
    for role, tail in outcome.log_tails.items():
        if tail:
            ctx.show.note(f"--- {role} の記録の末尾 ---\n{tail}")
    ctx.show.detail(outcome.detail)
    if outcome.status in ("ready", "already_running"):
        return EXIT_OK
    return EXIT_PRECONDITION if outcome.status == "refused" else EXIT_FAILED


def _cmd_stop(ctx: _Context) -> int:
    """自分のコンテナを 2 台とも止めて消す (requirements 1.7、1.8)。"""
    nodes = ctx.nodes()
    outcome = lifecycle.stop(
        ctx.runner(),
        nodes,
        confirmer=ctx.confirmer(),
        report=ctx.show.err,
        sleep=ctx.sleep,
        clock=ctx.clock,
    )
    ctx.show.say("status", outcome.status)
    ctx.show.say("remaining_gpu_apps", len(outcome.remaining_gpu_apps))
    for index, app in enumerate(outcome.remaining_gpu_apps, start=1):
        ctx.show.say(f"gpu.{index}.process_name", app.process_name)
        ctx.show.say(f"gpu.{index}.used_memory_mib", app.used_memory_mib)
    ctx.show.detail(outcome.detail)
    return EXIT_OK if outcome.status in ("stopped", "already_stopped") else EXIT_FAILED


def _cmd_status(ctx: _Context) -> int:
    """2 台と推論サーバーの、いまの状態を示す (requirements 1.6、6.5)。

    読めなかった台があれば、表に印を付けて終了コード 1 にする (「何も動いていない」と読み
    違えないため。Implementation Notes 3.5 の申し送り)。
    """
    shown = lifecycle.read_status(
        ctx.runner(),
        ctx.configs(),
        ctx.nodes(),
        client=ctx.client(lifecycle.HTTP_TIMEOUT_S),
        report=ctx.show.err,
    )
    unreadable = set(shown.unreadable)
    ctx.show.table(
        ("node", "state", "config", "kind", "gpu", "fabric", "readable"),
        [
            (
                one.node,
                one.container_state,
                one.config_name,
                one.kind,
                _shown_gpu(one),
                _shown_link(one.fabric_link_up),
                UNREADABLE_MARK if one.node in unreadable else "ok",
            )
            for one in shown.service.nodes
        ],
    )
    for one in shown.service.nodes:
        ctx.show.say(f"node.{one.node}.container_state", one.container_state)
        ctx.show.say(f"node.{one.node}.config_name", one.config_name)
        ctx.show.say(f"node.{one.node}.kind", one.kind)
        ctx.show.say(f"node.{one.node}.config_sha256", one.config_sha256)
        ctx.show.say(f"node.{one.node}.gpu_apps", len(one.gpu_apps))
        ctx.show.say(f"node.{one.node}.fabric_link_up", one.fabric_link_up)
        ctx.show.say(f"node.{one.node}.readable", one.node not in unreadable)
    _say_service(ctx, shown.service)
    ctx.show.say("unreadable", ",".join(shown.unreadable))
    ctx.show.say("status", "partial" if shown.unreadable else "read")
    ctx.show.detail(
        "自分のコンテナの一覧を読めなかった台がある"
        f" ({', '.join(shown.unreadable)})。「何も動いていない」とは読めない"
        if shown.unreadable
        else f"{len(shown.service.nodes)} 台の状態を読んだ"
    )
    return EXIT_PRECONDITION if shown.unreadable else EXIT_OK


def _shown_gpu(one: NodeStatus) -> str:
    """GPU を使っているプロセスの数と、使っているメモリの合計。"""
    if not one.gpu_apps:
        return "-"
    total = sum(app.used_memory_mib or 0 for app in one.gpu_apps)
    return f"{len(one.gpu_apps)} 件 / {total} MiB"


def _shown_link(up: bool | None) -> str:
    return "-" if up is None else ("up" if up else "down")


def _cmd_smoke(ctx: _Context) -> int:
    """英語と日本語の短い要求を 1 つずつ送る (requirements 6.5、10.5)。

    応答の本文は `report` (stderr) にだけ出る。保存してよい事実だけを、標準出力に出す。
    """
    config, nodes = ctx.selected()
    outcome = lifecycle.smoke(
        config,
        nodes,
        client=ctx.client(lifecycle.SMOKE_TIMEOUT_S),
        report=ctx.show.err,
        max_tokens=int(ctx.args.max_tokens),
    )
    ctx.show.say("config", config.name)
    ctx.show.say("replies", len(outcome.replies))
    for reply in outcome.replies:
        ctx.show.say(f"reply.{reply.lang}.http_status", reply.http_status)
        ctx.show.say(f"reply.{reply.lang}.stop_reason", reply.stop_reason)
        ctx.show.say(f"reply.{reply.lang}.input_tokens", reply.input_tokens)
        ctx.show.say(f"reply.{reply.lang}.output_tokens", reply.output_tokens)
        ctx.show.say(f"reply.{reply.lang}.replacement_char", reply.replacement_char)
    answered = bool(outcome.replies) and all(reply.http_status == 200 for reply in outcome.replies)
    ctx.show.say("status", "answered" if answered else "failed")
    ctx.show.detail(outcome.detail)
    return EXIT_OK if answered else EXIT_FAILED


# --- serve logs -----------------------------------------------------------


def _cmd_logs(ctx: _Context) -> int:
    """2 台の記録を `serving/var/` に写す (requirements 1.9)。"""
    config, nodes = ctx.selected()
    started_at = ctx.now()
    plans = plan.build_plans(config, nodes, started_at)
    log_dir = logs.collect_logs(
        ctx.runner(),
        nodes,
        plans,
        var_root=ctx.var_root,
        started_at=started_at,
        command="logs",
        config_name=config.name,
    )
    missing = _missing_count(log_dir)
    ctx.show.say("config", config.name)
    ctx.show.say("log_dir", log_dir)
    ctx.show.say("missing", missing)
    ctx.show.say("status", "collected")
    ctx.show.detail(
        f"記録を {log_dir} に写した"
        + (f" (写せなかったもの {missing} 件は collect.json にある)" if missing else "")
    )
    return EXIT_OK


def _missing_count(log_dir: Path) -> int:
    """`collect.json` に書かれた、写せなかったものの数 (読めなければ 0 として示す)。"""
    path = log_dir / logs.MISSING_FILE_NAME
    if not path.is_file():
        return 0
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    missing = loaded.get("missing") if isinstance(loaded, dict) else None
    return len(missing) if isinstance(missing, list) else 0


# --- serve probe ----------------------------------------------------------


def _cmd_probe(ctx: _Context) -> int:
    """1 台の縮小の確認を流す (requirements 5.1〜5.5)。

    終了コードは `ready` = 0、`inconclusive` = 1、`failed` = 2 である (`serve start` は時間切れ
    を 2 にするが、`probe` は「判定が出たか」で分ける。Implementation Notes 4.1 の申し送り)。
    """
    config, nodes = ctx.selected()
    started_at = ctx.now()
    commit, dirty = ctx.facts()
    outcome = probe.run_probe(
        ctx.runner(),
        config,
        nodes,
        started_at,
        confirmer=ctx.confirmer(),
        var_root=ctx.var_root,
        record_dir=ctx.record_dir(lifecycle.COMMAND_NAME, config.name, started_at),
        repo_commit=commit,
        repo_dirty=dirty,
        manifest=ctx.manifest(config),
        timeout_s=ctx.args.timeout,
        sleep=ctx.sleep,
        clock=ctx.clock,
        report=ctx.show.err,
        client=ctx.client(lifecycle.HTTP_TIMEOUT_S),
        smoke_client=ctx.client(lifecycle.SMOKE_TIMEOUT_S),
    )
    ctx.show.say("config", outcome.config_name)
    ctx.show.say("status", outcome.status)
    _say_observation(ctx, outcome.observation)
    if outcome.reply is not None:
        ctx.show.say("reply.http_status", outcome.reply.http_status)
        ctx.show.say("reply.stop_reason", outcome.reply.stop_reason)
        ctx.show.say("reply.output_tokens", outcome.reply.output_tokens)
    ctx.show.detail(outcome.detail)
    if outcome.status == "ready":
        return EXIT_OK
    return EXIT_PRECONDITION if outcome.status == "inconclusive" else EXIT_FAILED


# --- serve netcheck -------------------------------------------------------


def _cmd_netcheck_links(ctx: _Context) -> int:
    """直結のインターフェースを読む (requirements 4.1、4.2。読み取りだけ)。"""
    nodes = ctx.nodes()
    reports = netcheck.read_link_reports(ctx.runner(), nodes)
    for role in _ROLE_ORDER:
        report = reports.get(role)
        if report is None:
            continue
        ctx.show.note(netcheck.format_link_report(report))
        _say_link_report(ctx, report)
    ctx.show.say("status", "read")
    ctx.show.detail(
        "; ".join(_shown_cables(reports[role]) for role in _ROLE_ORDER if role in reports)
    )
    return EXIT_OK


def _shown_cables(report: LinkReport) -> str:
    """1 台ぶんの、ケーブルの本数の言い方 (確定できなかったことも、言葉で示す)。"""
    if report.cable_count is None:
        return f"{report.node}: ケーブルの本数はわからない"
    return f"{report.node}: ケーブル {report.cable_count} 本"


def _say_link_report(ctx: _Context, report: LinkReport) -> None:
    ctx.show.say(f"cable_count.{report.node}", report.cable_count)
    ctx.show.say(f"interfaces.{report.node}", len(report.interfaces))
    ctx.show.say(f"tools_missing.{report.node}", ",".join(report.tools_missing))
    ctx.show.say(f"detail.{report.node}", report.detail)


def _cmd_netcheck_bandwidth(ctx: _Context) -> int:
    """2 台の間の帯域を測る (requirements 4.3、4.4、4.6)。"""
    config, nodes = ctx.selected()
    outcome = netcheck.run_bandwidth(
        ctx.runner(),
        config,
        nodes,
        ctx.now(),
        confirmer=ctx.confirmer(),
        var_root=ctx.var_root,
        sleep=ctx.sleep,
        clock=ctx.clock,
        report=ctx.show.err,
    )
    ctx.show.say("config", outcome.config_name)
    ctx.show.say("status", outcome.status)
    if outcome.run is not None:
        for sample in outcome.run.samples:
            ctx.show.say(f"sample.{sample.size_bytes}.busbw_gbps", f"{sample.busbw_gbps:.3f}")
            ctx.show.say(f"sample.{sample.size_bytes}.algbw_gbps", f"{sample.algbw_gbps:.3f}")
    for role, observed in outcome.nccl.items():
        ctx.show.say(f"network.{role}", None if observed is None else observed.network)
    ctx.show.say("comparison", outcome.comparison)
    ctx.show.gates(outcome.gates, only_failed=True)
    ctx.show.detail(outcome.detail)
    return _job_exit(outcome.status)


def _cmd_netcheck_sanity(ctx: _Context) -> int:
    """推論サーバーの公式の資料が示す、2 台での事前の確認を流す (requirements 4.6)。"""
    config, nodes = ctx.selected()
    outcome = netcheck.run_sanity(
        ctx.runner(),
        config,
        nodes,
        ctx.now(),
        confirmer=ctx.confirmer(),
        var_root=ctx.var_root,
        sleep=ctx.sleep,
        clock=ctx.clock,
        report=ctx.show.err,
    )
    ctx.show.say("config", outcome.config_name)
    ctx.show.say("status", outcome.status)
    for stage in outcome.stages:
        ctx.show.say(f"stage.{stage.index}.name", stage.name)
        ctx.show.say(f"stage.{stage.index}.passed", stage.passed)
        ctx.show.say(f"stage.{stage.index}.nodes_seen", ",".join(stage.nodes_seen))
    ctx.show.gates(outcome.gates, only_failed=True)
    ctx.show.detail(outcome.detail)
    return _job_exit(outcome.status)


def _cmd_netcheck_ab(ctx: _Context) -> int:
    """足す設定の A/B を、交互に流して比べる (requirements 4.5)。"""
    pairs: Sequence[tuple[str, str]] = ctx.args.env or ()
    if not pairs:
        raise ValueError("A/B には、足す環境変数が 1 つ以上要る (--env KEY=VALUE)")
    config, nodes = ctx.selected()
    report = netcheck.run_ab(
        ctx.runner(),
        config,
        nodes,
        ctx.now(),
        extra_env=dict(pairs),
        confirmer=ctx.confirmer(),
        var_root=ctx.var_root,
        repeats=int(ctx.args.repeat),
        sleep=ctx.sleep,
        clock=ctx.clock,
        report=ctx.show.err,
    )
    ctx.show.say("config", report.config_name)
    ctx.show.say("status", report.status)
    ctx.show.say("added_env", ",".join(f"{key}={value}" for key, value in sorted(pairs)))
    ctx.show.say("runs", len(report.runs))
    if report.outcome is not None:
        ctx.show.say("adopt", report.outcome.adopt)
        ctx.show.say("compared_size_bytes", report.outcome.compared_size_bytes)
    for route in report.routes:
        for role, observed in route.observed.items():
            ctx.show.say(
                f"route.{route.arm}.{route.repeat_index}.{role}",
                None if observed is None else observed.network,
            )
    ctx.show.gates(report.gates, only_failed=True)
    ctx.show.detail(report.detail)
    if report.status == "compared":
        return EXIT_OK
    return EXIT_PRECONDITION if report.status == "refused" else EXIT_FAILED


def _job_exit(status: str) -> int:
    """ジョブの結末を終了コードに写す (`passed` = 0、`refused` = 1、`failed` = 2)。"""
    if status == "passed":
        return EXIT_OK
    return EXIT_PRECONDITION if status == "refused" else EXIT_FAILED


# --- serve watch ----------------------------------------------------------


def _cmd_watch(ctx: _Context) -> int:
    """連続の負荷の間、外から見張る (requirements 7.7。読み取りだけ)。

    終了コードは、出来事なし = 0、出来事あり = 2 である (Implementation Notes 4.5 の申し送り)。
    """
    config, nodes = ctx.selected()
    outcome = watch.watch(
        ctx.runner(),
        config,
        nodes,
        var_root=ctx.var_root,
        duration_s=float(ctx.args.duration),
        interval_s=float(ctx.args.interval),
        unresponsive_threshold=int(ctx.args.unresponsive_threshold),
        stall_window_s=float(ctx.args.stall_window),
        stall_gpu_threshold_pct=int(ctx.args.stall_gpu_threshold),
        thermal_threshold_c=float(ctx.args.thermal_threshold),
        client=ctx.client(watch.DEFAULT_HEALTH_TIMEOUT_S),
        sleep=ctx.sleep,
        clock=ctx.clock,
        now=ctx.now,
        report=ctx.show.err,
    )
    ctx.show.say("config", outcome.config_name)
    ctx.show.say("samples_path", outcome.samples_path)
    ctx.show.say("sample_count", outcome.sample_count)
    ctx.show.say("gpu_utilization_available", outcome.gpu_utilization_available)
    ctx.show.say("events", len(outcome.events))
    for index, event in enumerate(outcome.events, start=1):
        ctx.show.say(f"event.{index}.finding", event.finding)
        ctx.show.say(f"event.{index}.at_utc", event.at_utc.isoformat())
        ctx.show.say(f"event.{index}.detail", event.detail)
    ctx.show.say("status", "events" if outcome.events else "quiet")
    ctx.show.detail(outcome.detail)
    return EXIT_FAILED if outcome.events else EXIT_OK


# --- serve thinking -------------------------------------------------------


def _cmd_thinking(ctx: _Context) -> int:
    """thinking の深さの渡し方を確かめる (requirements 9.2、10.5)。

    宛先は、選んだ構成から組み立てる (決めごとの 13)。`effective` が `None` (判定できなかった)
    のときは、終了コード 1 にする (`probe` の `inconclusive` と同じ扱い)。
    """
    config, nodes = ctx.selected()
    head = nodes.get("head")
    if head is None:
        raise ValueError("nodes.head の定義がない (thinking の宛先を決められない)")
    if config.served_model_name is None:
        raise ValueError(f"構成 '{config.name}' に served_model_name がない")
    base_url = f"http://{head.lan_addr}:{lifecycle.http_port(config)}"
    timeout_s = float(ctx.args.timeout)
    outcome = thinking.run_thinking(
        base_url=base_url,
        model=config.served_model_name,
        var_root=ctx.var_root,
        trials=int(ctx.args.trials),
        max_tokens=int(ctx.args.max_tokens),
        timeout_s=timeout_s,
        client=ctx.client(timeout_s),
        now=ctx.now,
        report=ctx.show.err,
    )
    ctx.show.say("config", config.name)
    ctx.show.say("base_url", base_url)
    ctx.show.say("model", outcome.model)
    ctx.show.say("trials", len(outcome.trials))
    ctx.show.say("effective", outcome.effective)
    ctx.show.say("status", "undecided" if outcome.effective is None else "decided")
    ctx.show.detail(outcome.detail)
    return EXIT_PRECONDITION if outcome.effective is None else EXIT_OK


# --- serve autostart --------------------------------------------------------


def _cmd_autostart_set(ctx: _Context) -> int:
    """指定した構成を、Spark 上の見張りが自動で起こす対象にする (issue #88 P6、要件 2、3)。

    両台の `launch.json` の `config_sha256` が Mac の計画と一致するときだけ配る
    (`autostart.set_autostart` が確かめる。一致しなければ、1 度も配らず終了コード 1)。
    """
    config, nodes = ctx.selected()
    started_at = ctx.now()
    commit, dirty = ctx.facts()
    outcome = autostart_mod.set_autostart(
        ctx.runner(),
        config,
        nodes,
        ctx.manifest(config),
        ctx.record_dir("autostart", config.name, started_at),
        started_at,
        repo_commit=commit,
        repo_dirty=dirty,
        confirmer=ctx.confirmer(),
    )
    ctx.show.say("config", config.name)
    ctx.show.say("status", outcome.status)
    ctx.show.say("nodes", ",".join(outcome.roles))
    ctx.show.detail(outcome.detail)
    return EXIT_OK if outcome.status == "designated" else EXIT_PRECONDITION


def _cmd_autostart_clear(ctx: _Context) -> int:
    """自動で起こす対象の指定を解除する (`config: null` を両台に配るだけ)。"""
    nodes = ctx.nodes()
    started_at = ctx.now()
    commit, dirty = ctx.facts()
    outcome = autostart_mod.clear_autostart(
        ctx.runner(),
        nodes,
        ctx.record_dir("autostart", "none", started_at),
        started_at,
        repo_commit=commit,
        repo_dirty=dirty,
        confirmer=ctx.confirmer(),
    )
    ctx.show.say("status", outcome.status)
    ctx.show.say("nodes", ",".join(outcome.roles))
    ctx.show.detail(outcome.detail)
    return EXIT_OK


def _cmd_autostart_status(ctx: _Context) -> int:
    """両台の指定と見張りの状態を、読むだけで示す (構成ファイルを読まなくても動く)。"""
    nodes = ctx.nodes()
    shown = autostart_mod.read_autostart(ctx.runner(), nodes)
    for node in shown.nodes:
        prefix = f"autostart.{node.role}"
        if node.designation is not None:
            for key, value in node.designation.model_dump(mode="json").items():
                ctx.show.say(f"{prefix}.designation.{key}", value)
        if node.status is not None:
            for key, value in node.status.model_dump(mode="json").items():
                ctx.show.say(f"{prefix}.status.{key}", value)
        ctx.show.say(f"{prefix}.readable", node.readable)
    unreadable = [node.role for node in shown.nodes if not node.readable]
    ctx.show.say("unreadable", ",".join(unreadable))
    ctx.show.say("status", "partial" if unreadable else "read")
    ctx.show.detail(
        f"読めなかった台がある ({', '.join(unreadable)})"
        if unreadable
        else f"{len(shown.nodes)} 台の指定と状態を読んだ"
    )
    return EXIT_PRECONDITION if unreadable else EXIT_OK


# --- 読み取った事実の表示 --------------------------------------------------


def _say_observation(ctx: _Context, observed: LaunchObservation | None) -> None:
    """起動の記録から読めた事実を並べる (読めなかった項目は空。requirements 5.2、6.3)。"""
    if observed is None:
        return
    ctx.show.say("observation.vllm_version", observed.vllm_version)
    ctx.show.say("observation.attention_backend", observed.attention_backend)
    ctx.show.say("observation.attention_candidates", ",".join(observed.attention_candidates))
    ctx.show.say("observation.moe_backend", observed.moe_backend)
    ctx.show.say("observation.kv_cache_tokens", observed.kv_cache_tokens)
    ctx.show.say("observation.kv_cache_gib", observed.kv_cache_gib)
    ctx.show.say("observation.model_loading_s", observed.model_loading_s)
    ctx.show.say("observation.engine_init_s", observed.engine_init_s)
    ctx.show.say("observation.speculative_config_seen", observed.speculative_config_seen)
    failure = observed.known_failure
    ctx.show.say("observation.known_failure", None if failure is None else failure.value)
    if observed.failure_excerpt:
        ctx.show.note("--- 誤りの前後 ---\n" + "\n".join(observed.failure_excerpt))


def _say_service(ctx: _Context, service: ServiceStatus | None) -> None:
    """推論サーバーから読めた事実を並べる (読めなかった項目は空。requirements 1.6、6.3)。"""
    if service is None:
        return
    ctx.show.say("health_ok", service.health_ok)
    ctx.show.say("served_model", service.served_model)
    ctx.show.say("max_model_len", service.max_model_len)
    ctx.show.say("running_requests", service.running_requests)
    ctx.show.say("waiting_requests", service.waiting_requests)


# --- 入口 -----------------------------------------------------------------

_EXIT_BY_ERROR: Final[tuple[tuple[type[Exception], int], ...]] = (
    # 前提の不足・断り (design.md Error Handling の 1)
    (weights_mod.WeightsRefError, EXIT_PRECONDITION),
    (config_mod.ConfigError, EXIT_PRECONDITION),  # plan.PlanError もここに入る
    (guards.ApprovalError, EXIT_PRECONDITION),
    (RemoteError, EXIT_PRECONDITION),
    (ValueError, EXIT_PRECONDITION),
    # 実行しての失敗 (design.md Error Handling の 2)
    (image.ImageError, EXIT_FAILED),
    (weights_mod.WeightsError, EXIT_FAILED),
    (lifecycle.LifecycleError, EXIT_FAILED),
    (probe.ProbeError, EXIT_FAILED),
    (netcheck.NetcheckError, EXIT_FAILED),
    (PushError, EXIT_FAILED),
    (autostart_mod.AutostartError, EXIT_FAILED),
)
"""例外から終了コードへの、ただ 1 つの表 (上から順に見る。決めごとの 1)。

`WeightsRefError` は `WeightsError` の一種なので、先に置く (前提の不足で 1)。
"""

_HANDLERS: Final[Mapping[str, Callable[[_Context], int]]] = {
    "check": _cmd_check,
    "push": _cmd_push,
    "pull-image": _cmd_pull_image,
    "image-licenses": _cmd_image_licenses,
    "manifest": _cmd_manifest,
    "derived-import": _cmd_derived_import,
    "fetch": _cmd_fetch,
    "verify": _cmd_verify,
    "start": _cmd_start,
    "stop": _cmd_stop,
    "status": _cmd_status,
    "smoke": _cmd_smoke,
    "logs": _cmd_logs,
    "probe": _cmd_probe,
    "watch": _cmd_watch,
    "thinking": _cmd_thinking,
}

_NETCHECK_HANDLERS: Final[Mapping[str, Callable[[_Context], int]]] = {
    "links": _cmd_netcheck_links,
    "bandwidth": _cmd_netcheck_bandwidth,
    "sanity": _cmd_netcheck_sanity,
    "ab": _cmd_netcheck_ab,
}

_AUTOSTART_HANDLERS: Final[Mapping[str, Callable[[_Context], int]]] = {
    "set": _cmd_autostart_set,
    "clear": _cmd_autostart_clear,
    "status": _cmd_autostart_status,
}

_NESTED_HANDLERS: Final[Mapping[str, tuple[str, Mapping[str, Callable[[_Context], int]]]]] = {
    "netcheck": ("netcheck_command", _NETCHECK_HANDLERS),
    "autostart": ("autostart_command", _AUTOSTART_HANDLERS),
}
"""入れ子のサブコマンドを持つコマンドの、属性名 (argparse の dest) と対応表の組。

案内文 (`err.write` の文言) は、この対応表の鍵からその都度組み立てる (鍵の並びは、
`_NETCHECK_HANDLERS`/`_AUTOSTART_HANDLERS` の定義順のまま。`dict` は挿入順を保つので、
既存の文言と完全に同じ出力になる)。
"""


def main(
    argv: Sequence[str] | None = None,
    *,
    runner_factory: Callable[[Path], RemoteRunner] | None = None,
    client_factory: Callable[[float], httpx.Client] | None = None,
    confirmer: guards.Confirmer | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    repo_root: Path | None = None,
    repo_facts: tuple[str, bool] | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> int:
    """CLI のエントリポイント。終了コードを返す (試験は、この関数をそのまま呼ぶ)。

    想定している失敗は、すべてここで 1 行の日本語にして、終了コードに直す (traceback は
    出さない。決めごとの 1、2)。`--help`、`--version`、引数の使い方の誤りは、`argparse` が
    `SystemExit` を送出する (伝播させる)。

    keyword の引数は、**試験のための差し込み口**である (`serve` の入口からは、どれも省く)。
    `runner_factory` は遠隔の実行役、`client_factory` は HTTP のクライアント、`confirmer` は
    了承の口、`repo_root` はリポジトリの最上位、`repo_facts` は `git` から読む事実、`now` は
    壁の時計、`sleep` / `clock` は待ちの口を差し替える。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    show = _Out(out=out, err=err)

    command: str | None = args.command
    if command is None:
        parser.print_help(err)
        err.write("\n実行するコマンドを 1 つ選ぶこと。\n")
        return EXIT_PRECONDITION
    handler = _HANDLERS.get(command)
    if handler is None:
        nested = _NESTED_HANDLERS.get(command)
        if nested is None:
            err.write(f"知らないコマンド: {command}\n")
            return EXIT_PRECONDITION
        dest, inner_handlers = nested
        inner: str | None = getattr(args, dest, None)
        handler = inner_handlers.get(inner) if inner is not None else None
        if handler is None:
            choices = " / ".join(inner_handlers)
            err.write(f"serve {command} は、{choices} の 1 つを選ぶこと。\n")
            return EXIT_PRECONDITION

    root = (serving_dir().parent if repo_root is None else repo_root).resolve()
    raw_var_root = args.var_root or root / "serving" / "var"
    ctx = _Context(
        args=args,
        show=show,
        repo_root=root,
        var_root=_absolute(Path(raw_var_root)),
        runner_factory=(
            runner_factory
            if runner_factory is not None
            else lambda var_root: SshRunner(var_root=var_root)
        ),
        client_factory=client_factory,
        given_confirmer=confirmer,
        stdin=sys.stdin if stdin is None else stdin,
        now=(lambda: datetime.now(UTC)) if now is None else now,
        sleep=sleep,
        clock=clock,
        given_facts=repo_facts,
    )
    try:
        return handler(ctx)
    except KeyboardInterrupt:
        err.write("エラー: 中断した (Spark にコンテナが残っていないか、serve status で確かめる)\n")
        return EXIT_INTERRUPTED
    except Exception as exc:  # noqa: BLE001 - 予期しない例外も 1 行にして写す (決めごとの 2)
        for kind, code in _EXIT_BY_ERROR:
            if isinstance(exc, kind):
                return _fail(err, exc, code)
        return _fail(err, exc, EXIT_FAILED, unexpected=True)


def _absolute(path: Path) -> Path:
    """相対の道筋を、いまの作業ディレクトリから見た絶対の道筋にする。

    `record_dir` は絶対でなければならない (部品が、配る元を配る前に空にする) ので、ここで
    そろえる。`resolve()` は使わない (symlink をたどると、`remote.CallGuard` が見る
    `var_root` と食い違いうる)。
    """
    return path if path.is_absolute() else Path.cwd() / path


def _fail(stream: TextIO, exc: BaseException, code: int, *, unexpected: bool = False) -> int:
    """誤りを 1 行にして stderr に出す (traceback は出さない。決めごとの 2)。"""
    shown = _one_line(str(exc)) or type(exc).__name__
    if unexpected:
        stream.write(f"エラー: 予期しない失敗 ({type(exc).__name__}): {shown}\n")
        stream.write(f"({DEBUG_ENV}=1 を付けると、詳しい出所を出す)\n")
        if os.environ.get(DEBUG_ENV) == "1":
            traceback.print_exception(exc, file=stream)
    else:
        stream.write(f"エラー: {shown}\n")
    stream.flush()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
