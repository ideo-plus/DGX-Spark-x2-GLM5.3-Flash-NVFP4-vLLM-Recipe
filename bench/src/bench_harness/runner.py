"""計測ランを、始めから終わりまで進める (task 3.5)。

`execute_run` が 1 回の計測ランのすべてを受け持つ。順序は design.md の
「計測ランの進行」のとおり:

1. 対象サーバーの定義と計測の設定を読み、試行の回数の上書きを当てて検証し直す
2. 認証の情報を環境から読む (値は持ち回るだけで、どこにも書き出さない。1.8)
3. 生データの置き場所が git の管理の対象でないことを確かめる (8.5、fail closed)
4. まとまりの文脈を組み立てる。使えない設定 (thinking) は、対象サーバーへ
   1 件も送る前に、ここで前提の不足になる
5. 前提を確かめる (1.3)。満たされていなければ、ここで終わる (1.4)。分かった
   入力の長さの上限を、文脈に入れ直す
6. 実行の条件 (`RunManifest`) を組み立てて、計測ランのディレクトリを作る (1.6)
7. 選ばれたまとまりを、**順に** (並行にはせず) 実行する (1.2)
8. 状態を確定して `RunOutcome` を返す

ここまでの 1〜6 で失敗したときは、計測ランのディレクトリを作らずに
`PreconditionError` を投げる (1.4)。`RunOutcome` が持てる終了の値は 0、2、130
だけなので (`types.RunOutcome` の docstring)、前提の不足 (終了の値 1) は例外で
表し、終了の値への変換は呼び出し側 (cli、task 5.1) が行う。

## 止まり方

| 何が起きたか | 状態 | 終了の値 |
|---|---|---|
| すべてのまとまりが終わった | `completed` | 0 |
| 要求の失敗が続いた (10.3) | `aborted` | 2 |
| 要求の入力のトークン数を数えられない (上限の超過を除く。issue #9、#24) | `aborted` | 2 |
| 中断の合図を受けた (10.4) | `interrupted` | 130 |
| 前提の不足、設定の誤り (1.4、8.5) | (作らない) | 1 (例外) |
| 生データを保存できない (8.1) | `aborted` | 例外をそのまま投げ直す |

- **連続の失敗** (10.3): 要求が失敗した試行を数える。上限の超過 (HTTP 400) は
  数えない (`is_context_limit_error`)。成功が 1 回あれば 0 に戻す。
  `Profile.max_consecutive_failures` に達したら止める。数えるのは**レコード
  1 件ごと**なので、同時処理の 1 回ぶん (n 本) が全滅すると、その 1 回で n 回
  数える (既定の 5 回なら、`concurrency/c8` の 1 回ぶんの全滅で止まる)。
  止めると決まっても、その 1 回ぶんのレコードは最後まで書く (下記)
- **上限の超過、送る前にわかる不成立** (3.6、6.9、issue #9): `plan()` が返した
  `SkippedCondition` と、`ConditionAborted` の `skipped` (上限の超過のほか、
  同時処理の狙いが要求の包み以下のとき、`agent` の会話の計測が上限を超えて
  断られたとき) を、どちらも `manifest.skipped` に足して、次の条件に進む
- **要求の入力のトークン数を数えられない** (issue #9、#24、#26): `concurrency` と
  `agent` は、条件の最初に要求の包みを数える。`concurrency` はさらに、回ごと・文書
  ごとに組み立てた要求を、その回の試行を送る前に数える。`agent` はさらに、段階 ×
  会話ごとに組み立てた会話を、その会話の最初の試行の直前に遅延して数える (どちらも
  対象サーバーに数えさせる)。数えられず `ProbeError` が出たときは、黙って文字数の
  見積もりに戻さず、その条件では、数えられなかった回または会話の試行を送らずに
  計測ランを止める (それより前に送った回または会話の試行は生データに残る)。理由 (条件の
  鍵と、数えられなかった訳) を `manifest.warnings` と標準エラーに残し、状態は
  `aborted`、終了の値は 2。ここまでに残した試行は生データにあるので、要約は作れる。
  例外は、`agent` の会話の計測が、入力の長さの上限を超えて断られたとき
  (`InputOverContextLimitError`) で、これは `ProbeError` として出ず、suite が
  `ConditionAborted` に変えるので、上の「上限の超過」と同じく、その段階を飛ばして
  次に進む (状態は `completed`、終了の値は 0)。包みの計測が上限を超えて断られた
  ときは、この例外に入らず、`ProbeError` のまま `aborted` になる
- **中断** (10.4): `SIGINT` と `SIGTERM` を受けたら、計測の本体の task を
  取り消す。送っている途中の要求は、そのまま打ち切られる。打ち切られた試行の
  レコードは**作らない** (レコードを作れるのは、応答が終わったときだけ) ので、
  どこまで進んでいたかは `manifest.warnings` に残す。1 回目の合図で自分の
  合図の受け口を外すので、2 回目の `SIGINT` はすぐに効く
- **生データの保存の失敗** (8.1): 要求の失敗ではない。状態を `aborted` にして
  印を残してから、例外をそのまま投げ直す (黙って捨てない)
- **終わった要求は、必ず残す** (8.1、10.1): 止めると決めたあとも、その 1 回ぶん
  のレコードは最後まで受け取って書く。「次の要求を新しく送らせない」ことより
  「すでに応答を得た要求を残す」ことを優先する (`_run_condition` を参照)

## コマンドの入口 (task 5.1) への申し送り

- 計測ランのディレクトリができる前 (設定の読み込みと、前提の確認の最中) は、
  合図の受け口をまだ入れていない。ここで `SIGINT` を受けると、`execute_run`
  からは `KeyboardInterrupt` がそのまま外に出る (記録する先がまだないため)。
  入口の側で捕まえて、終了の値 130 にすること
- `StoreError` / `OSError` が外に出たときは、実行の条件をすでに `aborted` に
  してある (できる限りの範囲で)。入口の側は、理由を標準エラーに出して、終了の
  値 2 にすること
- `PreconditionError` は、`exit_code` (1) をそのまま終了の値にし、例外の文を
  そのまま計測者に示すこと (どの前提が満たされていないかが書いてある)

## 使った公開の課題 (5.5、5.7、task 8.1)

まとまりの側からは実行の条件を書けない (`SuiteContext` の保存の口は `put_body`
だけで、数える口 `count_input_tokens` は別) ので、`run_info()` を持つまとまり
(`SupportsRunInfo`。いまは品質の検査だけ) については、**そのまとまりを流し
終えたあとに**進行の側が読み取り、`RunManifest.datasets` に足す。止まった
計測ランでも、中断した計測ランでも、そこまでに使った課題は残す (未完了の
要約にも、何で測ったかが要る)。コードの条件を飛ばした計測ランでは、使った
課題がないので、何も書かない (5.7)。

## 内部の指標 (7.1、7.4、7.5)

条件ごとに、前後で `/metrics` を読み、その間は `Profile.metrics_interval_s` で
定期的に読んで KV の使用率の最大を追う。得られなくても計測は止めない。まとまり
の前後の増分 (7.1) は、そのまとまりの条件の増分から作れる。

依存の向き (design.md) は `... -> suites -> runner -> cli`。この module は
`analysis` と `cli` を読み込まない (要約は、計測ランが終わったあとに cli が作る)。
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import signal
import subprocess
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol, TextIO, runtime_checkable

from pydantic import JsonValue, SecretStr

from bench_harness import __version__, config
from bench_harness.client.messages import HttpxMessagesClient, MessagesClient
from bench_harness.client.probe import ProbeError, is_context_limit_error, preflight
from bench_harness.client.tokenize import recount_output
from bench_harness.metrics import HttpxMetricsScraper, MetricsScraper, derive
from bench_harness.store import RunStore, StoreError, new_run_id
from bench_harness.suites import ConditionAborted, Suite, SuiteContext, make_suite_context
from bench_harness.suites.agent import AgentSuite
from bench_harness.suites.concurrency import ConcurrencySuite
from bench_harness.suites.decode import decode_suite as _DECODE_SUITE
from bench_harness.suites.prefill import SUITE as _PREFILL_SUITE
from bench_harness.suites.quality import ProblemsLoader, QualitySuite
from bench_harness.types import (
    ConditionPlan,
    DatasetRef,
    DerivedMetrics,
    LogicalMetric,
    MetricSnapshot,
    MetricsUnavailable,
    PreflightFailure,
    PreflightOk,
    Profile,
    RunManifest,
    RunOutcome,
    RunRequest,
    RunStatus,
    SkippedCondition,
    SuiteName,
    TargetDef,
    TrialRecord,
    _OutputRetokenization,
    _OutputTokenCounts,
    _PhaseTokenCount,
)

__all__ = [
    "EXIT_ABORTED",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "EXIT_PRECONDITION",
    "Clock",
    "PreconditionError",
    "ProgressSink",
    "StderrProgressSink",
    "SuiteRunInfo",
    "SupportsRunInfo",
    "default_suite_registry",
    "execute_run",
    "harness_version",
]

EXIT_OK: Final[int] = 0
"""すべてのまとまりが終わった (design.md Monitoring)。"""

EXIT_PRECONDITION: Final[int] = 1
"""前提の不足、または設定の誤り。`PreconditionError` が持つ値。"""

EXIT_ABORTED: Final[int] = 2
"""要求の失敗が続いたので止めた (10.3)。"""

EXIT_INTERRUPTED: Final[int] = 130
"""中断の合図で終わった (10.4)。128 + SIGINT の番号。"""

_RUN_TASK_NAME: Final[str] = "bench-run"
"""計測の本体の task の名前 (中断の合図で取り消す相手)。"""

_GIT_TIMEOUT_S: Final[float] = 5.0
"""道具の版を調べる git の制限時間。これは安全に関わる確認ではないので、
失敗したら版を `unknown` に落とすだけで、計測は止めない (1.6)。"""

_COMMIT_CHARS: Final[int] = 8
"""道具の版に載せる git のコミットの短い表記の長さ。"""

Clock = Callable[[], datetime]
"""いまの時刻を返す関数 (試験から差し替えられるようにしてある)。"""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class PreconditionError(Exception):
    """計測を始められない (前提の不足、または設定の誤り)。

    これが投げられたときは、計測ランのディレクトリを作っていない (1.4)。
    呼び出し側 (cli) は、`exit_code` をそのまま終了の値にし、どの前提が
    満たされていないかを、この例外の文として計測者に示す。
    """

    exit_code: Final[int] = EXIT_PRECONDITION


# --- 進み具合 (1.7) ---------------------------------------------------------


@runtime_checkable
class ProgressSink(Protocol):
    """いま測っている項目と、終わった試行の数を受け取る口 (1.7)。"""

    def update(
        self, suite: SuiteName, condition: str, done: int, total: int, failures: int
    ) -> None: ...


class StderrProgressSink:
    """標準エラーに 1 行ずつ出す `ProgressSink` (design.md Monitoring)。

    書き出し先は、呼ばれるたびに決める (試験が `sys.stderr` を差し替えても
    追従する)。
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream

    def update(
        self, suite: SuiteName, condition: str, done: int, total: int, failures: int
    ) -> None:
        stream = self._stream if self._stream is not None else sys.stderr
        print(
            f"[{suite.value}] {condition} {done}/{total} 失敗 {failures}",
            file=stream,
            flush=True,
        )


# --- まとまりの対応表 --------------------------------------------------------


def default_suite_registry(
    *, problems_loader: ProblemsLoader | None = None
) -> dict[SuiteName, Suite]:
    """名前から、まとまりの実体への対応表 (計測ランごとに 1 つ作る)。

    状態を持たないのは decode (3.2) と prefill (3.3) なので、module の実体を
    そのまま配る。同時処理 (3.4) は包みの計測の覚え書き (issue #9: 計測ランに
    つき 1 回数えたトークン数) を、品質の検査 (6.7) と長い会話の検査 (7.2) は
    計測ランごとの覚え書き (使った公開の課題、上限に当たった長さ、包みの計測の
    結果) を持つので、**呼ばれるたびに作り直す**。共有の状態そのものがなくなる
    ので、1 つのプロセスで 2 つの計測ランを流しても混ざらない
    (注 7.2 → 8.1、6.7 → 8.1)。

    `problems_loader` は、公開のコードの課題の読み方である。既定 (`None`) は
    まとまり自身の読み方 (`humaneval.load_humaneval_plus` の既定の置き場所)。
    置き場所を変える、取得を許さない、といった指定は、入口 (5.1) がここに
    渡す (`RunRequest` は凍結の型なので、項目を足さない。注 1.2)。
    """
    quality = (
        QualitySuite() if problems_loader is None else QualitySuite(problems_loader=problems_loader)
    )
    return {
        SuiteName.DECODE: _DECODE_SUITE,
        SuiteName.PREFILL: _PREFILL_SUITE,
        SuiteName.CONCURRENCY: ConcurrencySuite(),
        SuiteName.QUALITY: quality,
        SuiteName.AGENT: AgentSuite(),
    }


# --- 使った公開の課題 (5.5、注 6.7 → 8.1) ------------------------------------


class SuiteRunInfo(Protocol):
    """まとまりが、その計測ランについて報せるもの。

    いまのところ `QualitySuite.run_info()` だけが返す。読み取り専用の口として
    書いてあるので、`tuple` を返す実体もそのまま当てはまる。
    """

    @property
    def datasets(self) -> Sequence[DatasetRef]: ...

    @property
    def sandbox_image_ref(self) -> str | None: ...


@runtime_checkable
class SupportsRunInfo(Protocol):
    """`run_info()` を持つまとまり。

    `run_info()` は `Suite` の約束事 (design.md suites) にない。まとまりの側
    からは実行の条件を書けない (`SuiteContext` の保存の口は `put_body` だけで、
    数える口 `count_input_tokens` は別) ので、進行の側が、持っているまとまり
    だけを見分けて書き写す。
    """

    def run_info(self) -> SuiteRunInfo: ...


def _merge_datasets(recorded: Sequence[DatasetRef], used: Sequence[DatasetRef]) -> list[DatasetRef]:
    """すでに記録した出どころに、新しいものを足す (名前と版で重複を除く)。

    上書きにすると、まとまりが 2 つ以上、課題を報せたときに、先に流したほうの
    出どころが消える。
    """
    merged = list(recorded)
    seen = {(dataset.name, dataset.version) for dataset in merged}
    for dataset in used:
        key = (dataset.name, dataset.version)
        if key in seen:
            continue
        seen.add(key)
        merged.append(dataset)
    return merged


# --- 道具の版 (1.6) ----------------------------------------------------------


def harness_version(repo_dir: Path | None = None) -> str:
    """パッケージの版、git のコミット、未コミットの変更の有無から作る (1.6)。

    例: `0.1.0+g1a2b3c4` / `0.1.0+g1a2b3c4.dirty`。git が使えない (リポジトリの
    外、git がない、応答しない) ときは `0.1.0+unknown` に落とす。これは安全に
    関わる確認ではない (生データの置き場所の検査とは違う) ので、計測は止めない。
    """
    cwd = repo_dir if repo_dir is not None else Path(__file__).resolve().parent
    return f"{__version__}+{_git_suffix(cwd)}"


def _git_suffix(cwd: Path) -> str:
    commit = _git(["rev-parse", f"--short={_COMMIT_CHARS}", "HEAD"], cwd)
    if not commit:
        return "unknown"
    # まだ追加していないファイルも「未コミットの変更」に数える。パッケージの中の
    # 未追跡の `.py` は、そのまま読み込まれて振る舞いを変えるので、コミットだけ
    # では版を言い当てられない (9.4 の「道具の版が違えば警告」の守り)。git が
    # 無視するファイル (`.venv/`、`results/` など) は、既定で出てこないので
    # 数えない。見るのは作業木の全体 (この module があるリポジトリ全体)
    dirty = _git(["status", "--porcelain"], cwd)
    return f"g{commit}.dirty" if dirty else f"g{commit}"


def _git(args: list[str], cwd: Path) -> str | None:
    """引数の列で git を呼ぶ (シェルは使わない)。答えが得られなければ `None`。"""
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    env.pop("LANGUAGE", None)
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


# --- 計測ランを進める --------------------------------------------------------


async def execute_run(
    req: RunRequest,
    progress: ProgressSink,
    *,
    targets_path: Path | None = None,
    profiles_path: Path | None = None,
    results_root: Path | None = None,
    registry: Mapping[SuiteName, Suite] | None = None,
    env: Mapping[str, str] | None = None,
    clock: Clock = _utc_now,
) -> RunOutcome:
    """1 回の計測ランを、始めから終わりまで進める (3.5)。

    差し替えられるのは、設定ファイルの場所、生データの置き場所、まとまりの
    対応表、認証の情報を読む環境の写像、時計。既定は、どれも本番の値。

    投げるもの:

    - `PreconditionError`: 前提の不足、または設定の誤り (1.4)。計測ランの
      ディレクトリは作っていない
    - `StoreError` / `OSError`: 生データを保存できなかった (8.1)。状態を
      `aborted` にして印を残してから、投げ直す
    - `asyncio.CancelledError`: 呼び出し側から取り消されたとき (中断の合図で
      はないので、握りつぶさない)
    """
    suites = default_suite_registry() if registry is None else dict(registry)
    target, profile = _load_config(req, targets_path, profiles_path)
    _check_suites(req.suites, suites)
    api_key = _resolve_api_key(target, os.environ if env is None else env)
    root = _resolve_results_root(results_root)

    client = HttpxMessagesClient.from_target(target, api_key)
    scraper = HttpxMetricsScraper.from_target(target)
    try:
        started_at = clock()
        run_id = new_run_id(target.name, started_at)
        handle = _StoreHandle()
        # 文脈を、前提の確認より先に組み立てる。使えない設定 (thinking) は、
        # 対象サーバーへ 1 件も送る前に前提の不足になり、計測ランのディレクトリ
        # も作られない (注 3.1)。上限は、このあとの前提の確認で入れ直す
        ctx = _build_context(client, profile, target, run_id, handle, api_key)
        ok = await _preflight(client, target, profile, api_key)
        ctx = _with_context_limit(ctx, ok.context_limit)
        manifest = _build_manifest(req, target, profile, ok, run_id=run_id, started_at=started_at)
        store = _create_store(root, manifest)
        handle.store = store
        run = _Run(
            ctx=ctx,
            store=store,
            scraper=scraper,
            progress=progress,
            registry=suites,
            suites=list(req.suites),
            clock=clock,
            target=target,
            api_key=api_key,
            retokenize_output=req.retokenize_output,
        )
        return await run.execute()
    finally:
        await scraper.aclose()
        await client.aclose()


# --- 始める前の確かめ --------------------------------------------------------


def _load_config(
    req: RunRequest, targets_path: Path | None, profiles_path: Path | None
) -> tuple[TargetDef, Profile]:
    """対象サーバーの定義と計測の設定を読み、上書きを当てて検証し直す (1.1、1.3)。"""
    try:
        target = config.select_target(req.target_name, targets_path)
        profile = config.apply_overrides(
            config.select_profile(req.profile_name, profiles_path), req.trials_override
        )
    except config.ConfigError as exc:
        raise PreconditionError(f"設定の誤り: {exc}") from exc
    return target, profile


def _check_suites(selected: Sequence[SuiteName], registry: Mapping[SuiteName, Suite]) -> None:
    """選ばれたまとまりが、すべて対応表にあることを確かめる (1.2)。"""
    if not selected:
        raise PreconditionError("設定の誤り: 測る項目のまとまりが 1 つも選ばれていない")
    available = ", ".join(sorted(name.value for name in registry)) or "(なし)"
    seen: set[SuiteName] = set()
    for name in selected:
        if name not in registry:
            raise PreconditionError(
                f"設定の誤り: まとまり '{name.value}' は、まだ実行できない"
                f" (実行できるもの: {available})"
            )
        if name in seen:
            # 2 度実行すると、同じ (条件、試行の番号) のレコードが 2 つできて、
            # 試行の識別子が一意でなくなる (design.md Domain Model)
            raise PreconditionError(f"設定の誤り: まとまり '{name.value}' が 2 回以上選ばれている")
        seen.add(name)


def _resolve_api_key(target: TargetDef, env: Mapping[str, str]) -> SecretStr | None:
    """認証の情報の値を、環境の写像から読む (1.8)。

    決まりは `config.resolve_api_key` と同じ (設定に書くのは環境変数の名前だけ、
    空は「ない」と同じ) だが、試験から環境を差し替えられるように、`os.environ`
    ではなく渡された写像を見る。値は、この先どこにも書き出さない。
    """
    if target.api_key_env is None:
        return None
    value = env.get(target.api_key_env)
    if not value:
        raise PreconditionError(
            f"前提の不足: 環境変数 '{target.api_key_env}' が設定されていない"
            f" (対象サーバー '{target.name}' の認証の情報)"
        )
    return SecretStr(value)


def _resolve_results_root(explicit: Path | None) -> Path:
    """生データの置き場所を決め、git の管理の対象でないことを確かめる (8.5)。"""
    try:
        return config.resolve_results_root(explicit)
    except config.ConfigError as exc:
        raise PreconditionError(f"前提の不足: {exc}") from exc


async def _preflight(
    client: MessagesClient, target: TargetDef, profile: Profile, api_key: SecretStr | None
) -> PreflightOk:
    """短い要求を 1 回送って、計測に進んでよいかを確かめる (1.3、1.4)。"""
    outcome = await preflight(client, target, timeout=profile.timeout, api_key=api_key)
    if isinstance(outcome, PreflightFailure):
        raise PreconditionError(
            f"前提の不足 ({outcome.unmet}): {outcome.detail}"
            f" (対象サーバー '{target.name}' {target.base_url})"
        )
    return outcome


class _StoreHandle:
    """まとまりの文脈を、計測ランのディレクトリを作る前に組み立てるための間接参照。

    `make_suite_context` は本文を保存する口 (`put_body`) を要る (3.1) が、その
    本体 (`RunStore`) は文脈を組み立てたあとにしか作れない (文脈の組み立てで
    投げられたら、ディレクトリを作らずに終わるため。注 3.1)。この入れ物が、
    その順序の食い違いを吸収する。
    """

    def __init__(self) -> None:
        self.store: RunStore | None = None

    def put_body(self, body: dict[str, JsonValue]) -> str:
        if self.store is None:  # 書き手の誤り (計測ランを作る前に試行を始めた)
            raise RuntimeError("runner: 計測ランのディレクトリを作る前に、本文を保存しようとした")
        return self.store.put_body(body)


def _build_context(
    client: MessagesClient,
    profile: Profile,
    target: TargetDef,
    run_id: str,
    handle: _StoreHandle,
    api_key: SecretStr | None,
) -> SuiteContext:
    """まとまりの文脈を組み立てる。使えない設定は前提の不足にする (注 3.1)。

    入力の長さの上限は、まだ分からない (前提の確認で得る) ので `None` で作り、
    あとで `_with_context_limit` が入れ直す。使えない設定 (thinking) の判定を、
    通信より前に済ませるための順序。判定の決まりは `make_suite_context` が持ち、
    ここでは写さない (決まりを 2 か所に持たないため)。

    `api_key` は、包みの計測 (issue #9) の既定の口が、`count_tokens` の要求に
    付けるために使う (`make_suite_context` が閉じ込める。`SuiteContext` そのもの
    には残さない)。
    """
    try:
        return make_suite_context(
            client=client,
            profile=profile,
            target=target,
            run_id=run_id,
            put_body=handle.put_body,
            context_limit=None,
            api_key=api_key,
        )
    except ValueError as exc:
        raise PreconditionError(f"設定の誤り: {exc}") from exc


def _with_context_limit(ctx: SuiteContext, context_limit: int | None) -> SuiteContext:
    """前提の確認で分かった上限を、文脈に入れ直す (凍結の型なので作り直す)。

    `dataclasses.replace` は `__post_init__` を通るので、範囲の確かめ (注 2.7)
    も、もう一度効く。
    """
    try:
        return dataclasses.replace(ctx, context_limit=context_limit)
    except ValueError as exc:
        raise PreconditionError(f"前提の不足: {exc}") from exc


def _build_manifest(
    req: RunRequest,
    target: TargetDef,
    profile: Profile,
    ok: PreflightOk,
    *,
    run_id: str,
    started_at: datetime,
) -> RunManifest:
    """実行の条件を組み立てる (1.6)。"""
    warnings: list[str] = []
    if ok.running_requests is not None and ok.running_requests > 0:
        warnings.append(
            f"計測を始める前に、対象サーバーが {ok.running_requests} 件の要求を処理していた。"
            "ほかの利用者が使っていると、内部の指標の増分に、その分が混ざる"
        )
    return RunManifest(
        run_id=run_id,
        status=RunStatus.RUNNING,
        target=target,
        server_model=ok.server_model,
        server_version=ok.server_version,
        started_at=started_at,
        suites=list(req.suites),
        profile_name=profile.name,
        profile=profile,
        harness_version=harness_version(),
        context_limit=ok.context_limit,
        warnings=warnings,
        output_retokenization=_OutputRetokenization(enabled=req.retokenize_output),
    )


def _create_store(root: Path, manifest: RunManifest) -> RunStore:
    """計測ランのディレクトリを作る。失敗したら、何も作られていない (8.5)。"""
    try:
        return RunStore.create(root, manifest)
    except (config.ConfigError, StoreError) as exc:
        raise PreconditionError(f"前提の不足: 計測ランを作れない: {exc}") from exc


# --- 本体 --------------------------------------------------------------------


def _expected_records(cond: ConditionPlan) -> int:
    """その条件で残るはずのレコードの数 (同時処理は、1 回ぶんで n 本になる)。"""
    return (cond.trials + cond.warmup_trials) * cond.concurrency


async def _aclose(iterator: AsyncIterator[TrialRecord]) -> None:
    """途中で止めるときに、まとまりの反復子を確実に閉じる (送りかけを残さない)。"""
    if isinstance(iterator, AsyncGenerator):
        await iterator.aclose()


class _Run:
    """1 つの計測ランの進行。`execute_run` だけが作る。"""

    def __init__(
        self,
        *,
        ctx: SuiteContext,
        store: RunStore,
        scraper: MetricsScraper,
        progress: ProgressSink,
        registry: Mapping[SuiteName, Suite],
        suites: Sequence[SuiteName],
        clock: Clock,
        target: TargetDef,
        api_key: SecretStr | None,
        retokenize_output: bool,
    ) -> None:
        self._ctx = ctx
        self._store = store
        self._scraper = scraper
        self._progress = progress
        self._registry = registry
        self._suites = list(suites)
        self._clock = clock
        self._target = target
        self._api_key = api_key
        self._retokenize_output = retokenize_output
        self._consecutive_failures = 0
        self._abort_reason: str | None = None
        self._interrupt_signal: int | None = None
        self._installed: list[int] = []
        self._current_suite: SuiteName | None = None
        self._current_condition: str | None = None
        self._condition_done = 0
        self._trials_written = 0

    # --- 全体の進行 ---

    async def execute(self) -> RunOutcome:
        """まとまりを順に実行し、状態を確定して結末を返す。"""
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(self._run_all(), name=_RUN_TASK_NAME)
        self._install_signal_handlers(loop, task)
        try:
            await task
        except asyncio.CancelledError:
            if self._interrupt_signal is None:
                raise  # 自分の合図ではない (呼び出し側からの取り消し) ので、伝える
            return self._finish_interrupted()
        finally:
            self._remove_signal_handlers(loop)
        if self._abort_reason is not None:
            return self._finish(RunStatus.ABORTED, EXIT_ABORTED)
        return self._finish(RunStatus.COMPLETED, EXIT_OK)

    async def _run_all(self) -> None:
        """選ばれたまとまりを、順に実行する (並行にはしない。design.md runner)。"""
        for name in self._suites:
            suite = self._registry[name]
            try:
                await self._run_suite(suite)
            finally:
                # 止まった計測ラン (10.3) も、中断した計測ラン (10.4) も、そこまで
                # に使った公開の課題を残す。未完了の要約にも、何で測ったかが要る
                self._record_datasets(suite)
            if self._abort_reason is not None:
                return

    async def _run_suite(self, suite: Suite) -> None:
        """1 つのまとまりの条件を、計画の順に実行する。"""
        for item in suite.plan(self._ctx):
            if isinstance(item, SkippedCondition):
                self._record_skip(item)
                continue
            await self._run_condition(suite, item)
            if self._abort_reason is not None:
                return

    async def _run_condition(self, suite: Suite, cond: ConditionPlan) -> None:
        """1 つの条件を実行し、前後の内部の指標を残す (7.1)。

        止めると決まっても (10.3)、その場で反復子を閉じない。同時処理のまとまり
        は、1 回ぶんの n 本をすべて終えてから順に返すので、途中で閉じると、
        **すでに送って応答を得た**要求のレコードが消えてしまう (8.1、10.1)。
        止めると決まったレコードと同じ 1 回ぶん (`warmup` と `round_id` が同じ)
        を最後まで受け取ってから止める。「次の要求を新しく送らせない」ことより
        「終わった要求を残す」ことを優先する。1 回ぶんの本数は `cond.concurrency`
        なので、ふつうはその数だけ受け取れば、次の 1 回ぶんを走らせずに止まれる。
        まとまりがそれより少なく返した場合の備えとして、違う組のレコードが来た
        ときは、それも保存してから止める。
        """
        self._current_suite = suite.name
        self._current_condition = cond.key
        self._condition_done = 0
        total = _expected_records(cond)
        failures = 0
        cancelled = False
        # 止めると決まった 1 回ぶんの組 (`(warmup, round_id)`)。決まるまでは None
        stopping: tuple[bool, int | None] | None = None
        group: tuple[bool, int | None] | None = None
        group_count = 0

        before = await self._snapshot()
        self._scraper.start_sampler(self._ctx.profile.metrics_interval_s)
        try:
            iterator = suite.run_condition(self._ctx, cond)
            try:
                async for record in iterator:
                    key = (record.warmup, record.round_id)
                    if key != group:
                        group, group_count = key, 0
                    group_count += 1

                    try:
                        record = await self._recount_record(record)
                    except asyncio.CancelledError:
                        self._store.append_trial(self._cancelled_recount(record))
                        self._trials_written += 1
                        self._condition_done += 1
                        if record.result.error is not None:
                            failures += 1
                        self._progress.update(
                            suite.name, cond.key, self._condition_done, total, failures
                        )
                        for _ in range(max(0, cond.concurrency - group_count)):
                            try:
                                pending = await anext(iterator)
                            except StopAsyncIteration:
                                break
                            self._store.append_trial(self._cancelled_recount(pending))
                            self._trials_written += 1
                            self._condition_done += 1
                            if pending.result.error is not None:
                                failures += 1
                            self._progress.update(
                                suite.name, cond.key, self._condition_done, total, failures
                            )
                        raise
                    self._store.append_trial(record)  # 1 つ終わるたびに書く (8.7)
                    self._trials_written += 1
                    self._condition_done += 1
                    if record.result.error is not None:
                        failures += 1
                    self._progress.update(
                        suite.name, cond.key, self._condition_done, total, failures
                    )

                    if stopping is None:
                        self._count_failure(record)
                        if self._abort_reason is None:
                            continue
                        stopping = key
                    elif key != stopping:
                        break  # 次の 1 回ぶんまで来た。保存はしたので、ここで止める
                    if group_count >= cond.concurrency:
                        break  # この 1 回ぶんは、すべて受け取った
            except ProbeError as exc:
                # 要求の入力のトークン数を数えられなかった (issue #9、#24)。見積もりに戻さず、
                # 数えられなかった会話の試行は送らずに、計測ランを止める。ここ (内側の try) で
                # 扱うのは、警告を書く途中の保存の失敗を、外側の except が受け取る
                # ようにするため
                self._abort_because_input_cannot_be_counted(cond, exc)
            finally:
                await _aclose(iterator)
        except ConditionAborted as exc:
            # その条件を送れない、または続けられない (上限の超過、狙いが包み以下など)。
            # この条件だけを終わりにして、次の条件に進む (3.6、6.9)
            self._record_skip(exc.skipped)
        except (StoreError, OSError) as exc:
            # 生データを保存できなかった (`append_trial` か `put_body`)。要求の
            # 失敗ではないので、握りつぶさずに計測ランを止める (8.1)
            self._mark_storage_failure(exc)
            raise
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            await self._write_metrics(cond, before, cancelled=cancelled)

    async def _recount_record(self, record: TrialRecord) -> TrialRecord:
        if not self._retokenize_output:
            return record
        counts = await recount_output(
            record.result,
            self._target,
            api_key=self._api_key,
            timeout_s=self._ctx.profile.timeout.idle_s,
        )
        result = record.result.model_copy(update={"output_token_counts": counts})
        return record.model_copy(update={"result": result})

    @staticmethod
    def _cancelled_recount(record: TrialRecord) -> TrialRecord:
        counts = _OutputTokenCounts(
            thinking=_PhaseTokenCount(reason="cancelled"),
            text=_PhaseTokenCount(reason="cancelled"),
        )
        result = record.result.model_copy(update={"output_token_counts": counts})
        return record.model_copy(update={"result": result})

    # --- 要求の入力を数えられない (issue #9、#24) ---

    def _abort_because_input_cannot_be_counted(self, cond: ConditionPlan, exc: ProbeError) -> None:
        """要求の入力のトークン数を数えられなかったので、計測ランを止める合図を立てる。

        状態 (`aborted`)、終了の値 (2)、理由 (警告) を、連続の失敗と同じ経路
        (`_abort_reason` → `execute` の `_finish`) で作る。呼び出し元の
        `_run_condition` が `_run_suite` と `_run_all` を、この合図で止める。
        """
        self._abort_reason = (
            "要求の入力のトークン数を数えられなかったので、計測ランを止めた "
            f"(条件: {cond.key})。{exc}"
        )
        self._warn(self._abort_reason)

    # --- 連続の失敗 (10.3) ---

    def _count_failure(self, record: TrialRecord) -> None:
        """要求の失敗を数え、決まった回数続いたら止める合図を立てる (10.3)。"""
        if record.result.error is None:
            self._consecutive_failures = 0
            return
        if is_context_limit_error(record.result):
            return  # 上限の超過は、失敗にも成功にも数えない (数え直しもしない)
        self._consecutive_failures += 1
        limit = self._ctx.profile.max_consecutive_failures
        if self._consecutive_failures < limit:
            return
        self._abort_reason = (
            f"要求の失敗が {self._consecutive_failures} 回続いたので、計測ランを止めた"
            f" (条件: {record.condition})。対象サーバーが応答していない可能性がある"
        )
        self._warn(self._abort_reason)

    # --- 中断 (10.4) ---

    def _install_signal_handlers(
        self, loop: asyncio.AbstractEventLoop, task: asyncio.Task[None]
    ) -> None:
        """`SIGINT` と `SIGTERM` で、計測の本体の task を取り消す。

        合図を扱えない環境 (event loop が別の thread にある、Windows) では、
        何もせずにふつうに流す。
        """
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._on_signal, loop, task, sig)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            self._installed.append(sig)

    def _on_signal(
        self, loop: asyncio.AbstractEventLoop, task: asyncio.Task[None], sig: int
    ) -> None:
        self._interrupt_signal = sig
        # 受け口を外すと、既定の扱いに戻る。2 回目の SIGINT は、その場で終わる
        self._remove_signal_handlers(loop)
        task.cancel()

    def _remove_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        for sig in self._installed:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(sig)
        self._installed = []

    def _finish_interrupted(self) -> RunOutcome:
        """中断の合図で終わる。どこまで進んでいたかを、警告に残す (10.4)。"""
        name = signal.Signals(self._interrupt_signal or signal.SIGINT).name
        where = (
            f"まとまり {self._current_suite.value}、条件 {self._current_condition}"
            if self._current_suite is not None
            else "まとまりを始める前"
        )
        self._try_warn(
            f"{name} を受けたので、計測ランを中断した ({where}、"
            f"この条件で終わった試行 {self._condition_done} 件、"
            f"計測ラン全体で {self._trials_written} 件)。"
            "送っている途中だった要求は打ち切ったので、その試行のレコードは残っていない"
        )
        return self._finish(RunStatus.INTERRUPTED, EXIT_INTERRUPTED)

    # --- 内部の指標 (7.1、7.4、7.5) ---

    async def _snapshot(self) -> MetricSnapshot | MetricsUnavailable:
        """`/metrics` を 1 回読む。失敗しても計測は止めない (7.4)。"""
        try:
            return await self._scraper.snapshot()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- 指標の不足は、計測を止めない (7.4)
            return MetricsUnavailable(reason=f"{type(exc).__name__}: {exc}")

    async def _stop_sampler(self) -> float | None:
        try:
            return await self._scraper.stop_sampler()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- 指標の不足は、計測を止めない (7.4)
            return None

    async def _write_metrics(
        self,
        cond: ConditionPlan,
        before: MetricSnapshot | MetricsUnavailable,
        *,
        cancelled: bool,
    ) -> None:
        """定期の読み取りを止め、あとの時点を読んで、増分と導出値を残す (7.1、7.5)。"""
        kv_peak = await self._stop_sampler()
        after: MetricSnapshot | MetricsUnavailable
        if cancelled:
            # 中断のあとに読みに行くと、終わるのが遅れる。読まなかった事実を残す
            after = MetricsUnavailable(reason="計測ランが中断されたので、あとの時点は読まなかった")
        else:
            after = await self._snapshot()
        derived = _derive(before, after, kv_peak)
        try:
            self._store.write_metrics(cond.key, before, after, derived)
        except (StoreError, OSError) as exc:
            # 内部の指標は補いの情報なので、保存できなくても計測は止めない (7.4)。
            # 試行のレコードの保存の失敗 (8.1) とは、扱いを分ける。ここは条件の
            # `finally` から呼ばれるので、警告の書き込みまで投げると、元の失敗を
            # 覆い隠してしまう (だから `_try_warn`)
            self._try_warn(
                f"条件 {cond.key} の内部の指標を保存できなかった: {type(exc).__name__}: {exc}"
            )

    # --- 実行の条件の書き換え ---

    def _record_skip(self, skipped: SkippedCondition) -> None:
        """飛ばした条件を、実行の条件に足す (3.6、6.9)。"""
        self._store.update_manifest(skipped=[*self._store.manifest().skipped, skipped])
        _log(f"条件を飛ばした: {skipped.key} ({skipped.reason})")

    def _record_datasets(self, suite: Suite) -> None:
        """まとまりが使った公開の課題を、実行の条件に足す (5.5、注 6.7 → 8.1)。

        `run_info()` を持たないまとまり (速さの 3 つ、長い会話の検査) では、
        何もしない。使った課題が 1 つもない計測ラン (コードの条件を飛ばした
        場合) には、何も書かない (使っていない課題を載せない。5.7)。

        隔離のイメージの識別子 (`run_info().sandbox_image_ref`) は、`RunManifest`
        に欄がないので書かない。設定で固定した値 (`sandbox.image_digest`) は
        `RunManifest.profile` として記録済みで、実際に動かすイメージがそれと
        違えば、隔離の側 (6.6) が使えないものとして扱う。

        書き込みに失敗しても、計測は止めない (内部の指標 7.4 と同じ扱い)。
        ここはまとまりの後始末 (`finally`) から呼ばれるので、投げると、元の
        失敗 (保存の失敗や中断) を覆い隠してしまう。
        """
        if not isinstance(suite, SupportsRunInfo):
            return
        used = list(suite.run_info().datasets)
        if not used:
            return
        recorded = self._store.manifest().datasets
        merged = _merge_datasets(recorded, used)
        if len(merged) == len(recorded):
            return
        try:
            self._store.update_manifest(datasets=merged)
        except (StoreError, OSError) as exc:
            self._try_warn(
                f"まとまり {suite.name.value} が使った公開の課題を記録できなかった: "
                f"{type(exc).__name__}: {exc}"
            )

    def _warn(self, message: str) -> None:
        """警告を、標準エラーと実行の条件の両方に残す (design.md Monitoring)。"""
        _log(f"警告: {message}")
        self._store.update_manifest(warnings=[*self._store.manifest().warnings, message])

    def _try_warn(self, message: str) -> None:
        """警告を残す。実行の条件への書き込みが失敗しても、投げない。

        ほかの失敗の後始末から呼ぶための口。ここで投げると、元の失敗を覆い隠す。
        標準エラーには `_warn` が先に出しているので、警告そのものは消えない。
        """
        with contextlib.suppress(StoreError, OSError):
            self._warn(message)

    def _mark_storage_failure(self, exc: Exception) -> None:
        """生データを保存できなかった印を、できる限り残す (8.1)。

        ここでの書き込みも失敗しうる (同じ理由で壊れている) ので、失敗は伏せる。
        伏せてよいのは、呼び出し側が元の例外をそのまま投げ直すからで、計測者は
        標準エラーと終了の仕方から、失敗したことを必ず知る。
        """
        self._try_warn(
            f"生データを保存できなかったので、計測ランを止める: {type(exc).__name__}: {exc}"
        )
        with contextlib.suppress(StoreError, OSError):
            self._store.set_status(RunStatus.ABORTED, self._clock())

    def _finish(self, status: RunStatus, exit_code: int) -> RunOutcome:
        manifest = self._store.set_status(status, self._clock())
        return RunOutcome(
            run_id=manifest.run_id,
            status=manifest.status,
            run_dir=self._store.run_dir,
            exit_code=exit_code,
        )


def _derive(
    before: MetricSnapshot | MetricsUnavailable,
    after: MetricSnapshot | MetricsUnavailable,
    kv_peak: float | None,
) -> DerivedMetrics:
    """2 つの時点から導出値を出す。片方でも読めなければ、名前をすべて未取得にする (7.4)。"""
    if isinstance(before, MetricSnapshot) and isinstance(after, MetricSnapshot):
        return derive(before, after, kv_peak)
    return DerivedMetrics(kv_usage_peak=kv_peak, missing=list(LogicalMetric))


def _log(message: str) -> None:
    """標準エラーに 1 行出す (書き出し先は、呼ばれるたびに決める)。"""
    print(message, file=sys.stderr, flush=True)
