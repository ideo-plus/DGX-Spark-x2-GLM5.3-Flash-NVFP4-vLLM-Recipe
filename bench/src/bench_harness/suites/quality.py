"""品質の検査のまとまり (task 6.7: suites/quality)。

`suites/base.py` の上に薄く載る (Requirement 5、design.md「quality」の行)。
3 種類の条件を、同じ `Suite` の約束事で実行する。

| 条件の鍵 | 中身 | 判定の型 |
|---|---|---|
| `quality/toolcall` | 前置きの会話なしのツール呼び出しの課題 (5.1) | `ToolCallVerdict` |
| `quality/code/humaneval+` | 公開のコードの課題 (5.2、5.4、5.5) | `QualityVerdict` |
| `quality/needle/{長さ}/d{位置}` | 長い入力から情報を探す課題 (5.3) | `QualityVerdict` |

条件の並びは design.md の表のとおり (ツール呼び出し → コード → 探す課題)。
探す課題は、長さの短いほうから、位置の小さいほうから送る (上限に当たったとき、
「それより長い段」を確実に飛ばせるようにするため)。3 つの条件は互いに独立で、
コードの課題を飛ばしても、ほかの 2 つは実行される (6.7 の完了の状態)。

## 採点できなかったもの (5.7)

要求そのものが失敗した課題は、**不正解と分けて**「採点できなかった」にする。

- ツール呼び出し: `classify_tool_call` が `REQUEST_FAILED` を返す (要約は、
  これを分母から外す)
- コードと探す課題: `QualityOutcome.NOT_SCORED`

コードの課題では、隔離の仕組みの側の失敗 (`SandboxError`) も「採点できなかった」
である。設備の不調をモデルの成績にしない (`scoring/sandbox.py`)。一方、応答に
使えるコードがなかった場合は、モデルが答えた結果なので**不正解**である
(`no_code_verdict`)。

## コードの課題の前に確かめること (6.7)

**要求を 1 つも送る前に**、次の順で確かめる。どちらかがだめなら、この条件
だけを `SkippedCondition` として飛ばす (`plan()` が返したものは、そのまま
`manifest.skipped` に入る)。

1. 隔離の実行環境 (`ContainerSandbox.available()`)。使えなければ、その理由
2. 公開の課題の取得 (`humaneval.load_humaneval_plus`)。`DatasetUnavailable`
   だけが飛ばす理由になる。`DatasetVerificationError`、`DatasetFormatError`、
   `DatasetCacheError` は、そのまま外に出す (壊れと取り違えを、黙って見逃さない。
   注 6.5 → 6.7)

採点に回す問題は、必ず `select_problems` を通す (`UNSCORABLE_PROBLEMS` を外して
から、`Profile.quality.code_problem_limit` のぶんだけ先頭から採る。注 6.1 → 6.6)。
試行の番号は、選んだ並びの中の位置である。どの問題の結果かは、判定の
`detail` の先頭に課題の番号 (`HumanEval/0` など) を残して追えるようにする
(`TrialRecord` に課題の番号の欄はない)。

`plan()` は外部のコマンド (コンテナの確認) とファイルの読み取りを行うので、
この間だけ event loop が止まる。条件を 1 つ計画するだけの一度きりの費用で、
試行はまだ 1 つも走っていない (計測の数字には影響しない)。一方、**試行の間**に
走る `run_python` は、必ず別の糸で動かす (下記)。

## 隔離の実行は、別の糸で (event loop を塞がない)

`ContainerSandbox.run_python` は同期で、最大 `SandboxSettings.timeout_s` の間
戻らない。まとまりは非同期なので、そのまま呼ぶと、進み具合の表示、内部の指標の
サンプリング、中断の合図が、その間ずっと止まる。だから `asyncio.to_thread` で
別の糸に出す。

糸に出した同期の処理は、**取り消せない**。中断 (`CancelledError`) を受けたら、
`run_python` が自分で時間切れを迎えてコンテナを片付け終えるのを、
`timeout_s + SANDBOX_REAP_MARGIN_S` を上限に待ってから、取り消しを伝える
(`ContainerSandbox` は、時間切れ・中断・例外のいずれでも `docker kill` してから
戻る)。待たずに抜けると、コンテナが残る (注 6.1 → 6.6 で実機で確かめた壊れ方)。
`CancelledError` は必ず投げ直す (握りつぶすと、2 回目の SIGINT が効かなくなる。
注 3.5)。

## 隔離が途中で壊れたとき (ブレーカー)

`SandboxError` が `MAX_CONSECUTIVE_SANDBOX_ERRORS` 回**続けて**起きたら、その
条件を `ConditionAborted` で打ち切る。守護プロセスが落ちた計測で、残りの全問を
黙って「採点できなかった」で埋めないためである (正解の割合の分母が 0 に近づき、
区間だけが広がる結果は、読み手に何も伝えない)。1 回でも採点できれば、数え直す。
打ち切る前に、その試行のレコードは `yield` してある (8.1: すべての要求を残す)。

## 採点は本文 (`text`) から (注 6.4)

コードの取り出しも、探す課題の照合も、応答の `text` ブロックだけをつないで
行う。thinking の中に正解やコードがあっても、採点には使わない (答えとして
差し出されたものだけを採点する)。

## 先頭の識別子 (nonce) を足さない

速さのまとまり (3.3、3.4) は、キャッシュに当たらないことを確かめるために
システムプロンプトの 1 行目に識別子を入れる。品質の検査が測るのは**正解の
割合**で、速さではない。識別子を入れると、

- 干し草の本文に入れれば、埋めた情報の位置 (`depth_pct`) がずれる
- `run_id` を含む識別子を入れれば、同じ設定の 2 つの計測ランが違う本文を送る
  ことになり、対応のある比較 (9.2) の前提が崩れる

ので、どの条件にもシステムプロンプトを付けない。ツール呼び出しの課題は
「前置きの会話なし」(tasks.md 6.7) で、`user` の発話 1 つと `tools` だけを送る。

## 出力の上限

コードの課題は `Profile.quality.code_max_tokens` (既定 1024)。ツール呼び出しと
探す課題には設定の項目がないので、この module の定数を使う
(`TOOLCALL_MAX_TOKENS`、`NEEDLE_MAX_TOKENS`)。どちらも、呼び出しのあとに本文を
続けるモデルや、考えを述べてから答えるモデルが、`stop_reason = max_tokens` で
切れないだけの余裕を取ってある (注 6.3: 切れた応答は `EMPTY_OR_TRUNCATED` に
落ち、正確さを実際より悪く見せる)。

## 共通の印 (10.7)

`measures_decode_speed=False` (生成速度を測らないので、出力が短くても印を
付けない)、`expect_full_output=False` (上限まで書かせるつもりはない)。
置き換え文字と繰り返しの印は `run_trial` が付ける。探す課題だけは
`target_input_tokens` を持つので、狙った長さから外れれば `LENGTH_OFF_TARGET`
が付く (入力の長さの事実として残す)。慣らしの試行はない (速さを測らないので、
接続の確立を追い出す必要がない。agent 7.2 と同じ判断)。

## 計測ランへの記録 (5.5、design.md「識別子を計測ランに記録する」)

`run_info()` が、この計測ランで実際に使った公開の課題の出どころ
(`humaneval.dataset_ref()`) と、隔離のイメージの識別子 (`image_ref()`) を返す。
`SuiteContext` には保存の部品への口が `put_body` しかないので、まとまりからは
`manifest` を書き換えられない。計測の実行につなぐタスク (8.1) が、

```python
store.update_manifest(datasets=[d.model_dump(mode="json") for d in suite.run_info().datasets])
```

と書くこと (入れれば、要約の「使った公開の課題」の節にそのまま出る。注 6.5 →
6.7)。イメージの識別子は `RunManifest` に欄がないが、設定で固定した値
(`sandbox.image_digest`) は `RunManifest.profile` として記録済みである。

## 課題の番号の住み分け

ツール呼び出しの課題の番号は `TOOLCALL_TASK_INDEX_BASE + 試行の番号`。長い会話
の検査 (7.2) は「段階の番号 × 10,000 + 試行の番号」、その会話の履歴は負の番号を
使う (7.1)。まとまりが違うので番号が重なっても実害はない (同じ課題を別の文脈で
2 回聞くだけ) が、生データを読むときに取り違えないよう、桁で離してある。

依存するのは標準ライブラリと `bench_harness` の `types`、`corpus`、`scoring`、
`suites.base` だけ (design.md の依存の向き)。ほかのまとまりは読み込まない。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Final, Protocol

from bench_harness.corpus import humaneval
from bench_harness.corpus.needle import make_needle_case
from bench_harness.corpus.tools import make_tool_task
from bench_harness.scoring.code import (
    build_checked_program,
    extract_code,
    no_code_verdict,
    not_scored_verdict,
    score_checked,
)
from bench_harness.scoring.needle import score_needle
from bench_harness.scoring.sandbox import ContainerSandbox, SandboxError
from bench_harness.scoring.toolcall import classify_tool_call
from bench_harness.suites.base import (
    ConditionAborted,
    SuiteContext,
    VerdictFn,
    abort_if_context_limit,
    condition_key,
    iter_trials,
    plan_condition,
    run_trial,
    single_user_message,
    skip_condition,
    tokens_label,
)
from bench_harness.types import (
    CodeProblem,
    ConditionPlan,
    ContentKind,
    DatasetRef,
    NeedleCase,
    QualityOutcome,
    QualitySettings,
    QualityVerdict,
    SandboxResult,
    SandboxSettings,
    SandboxUnavailable,
    SkippedCondition,
    StreamResult,
    SuiteName,
    ToolCallVerdict,
    ToolTask,
    TrialRecord,
)

__all__ = [
    "CODE_CONDITION_KEY",
    "CODE_INSTRUCTION",
    "MAX_CONSECUTIVE_SANDBOX_ERRORS",
    "NEEDLE_MAX_TOKENS",
    "SANDBOX_REAP_MARGIN_S",
    "SUITE",
    "TOOLCALL_CONDITION_KEY",
    "TOOLCALL_MAX_TOKENS",
    "TOOLCALL_TASK_INDEX_BASE",
    "CodeSandbox",
    "ProblemsLoader",
    "QualityRunInfo",
    "QualitySuite",
    "SandboxFactory",
    "needle_condition_key",
]

_CODE_PART: Final[str] = "code"
_CODE_DATASET_PART: Final[str] = "humaneval+"
_NEEDLE_PART: Final[str] = "needle"
_DEPTH_PREFIX: Final[str] = "d"

TOOLCALL_CONDITION_KEY: Final[str] = condition_key(SuiteName.QUALITY, "toolcall")
"""ツール呼び出しの正確さの条件の鍵 (design.md suites)。"""

CODE_CONDITION_KEY: Final[str] = condition_key(SuiteName.QUALITY, _CODE_PART, _CODE_DATASET_PART)
"""コードの課題の条件の鍵 (design.md suites)。"""

_NEEDLE_KEY_PREFIX: Final[str] = condition_key(SuiteName.QUALITY, _NEEDLE_PART) + "/"
"""探す課題の条件の鍵の頭 (`quality/needle/`)。"""

TOOLCALL_MAX_TOKENS: Final[int] = 512
"""ツール呼び出しの課題の出力の上限。

`QualitySettings` に項目がないので、ここで決める (設定の項目を足す変更は、
土台のタスク 1.3 の受け持ち)。呼び出しのあとに本文を続けるモデルが
`stop_reason = max_tokens` で切れて `EMPTY_OR_TRUNCATED` に落ちないだけの
余裕を取る (注 6.3)。"""

NEEDLE_MAX_TOKENS: Final[int] = 512
"""探す課題の出力の上限。答えは短いが、考えを述べてから答えるモデルでも、
答えの手前で切れないだけの余裕を取る (切れた応答は不正解になる)。"""

TOOLCALL_TASK_INDEX_BASE: Final[int] = 10_000_000
"""ツール呼び出しの課題の番号の起点 (module の docstring「課題の番号の住み分け」)。"""

MAX_CONSECUTIVE_SANDBOX_ERRORS: Final[int] = 3
"""隔離の失敗がこの回数だけ続いたら、コードの課題の条件を打ち切る。"""

SANDBOX_REAP_MARGIN_S: Final[float] = 5.0
"""中断のあと、`run_python` の後始末 (`docker kill` とプロセスの回収) を待つ、
`timeout_s` への上乗せ。"""

CODE_INSTRUCTION: Final[str] = (
    "Complete the following Python function.\n"
    "Reply with the complete function definition inside a single Python code block.\n"
    "Do not include tests or usage examples.\n\n"
)
"""コードの課題の指示。どの問題でも同じ、決まった文である (11.4)。"""

_MAX_DETAIL_CHARS: Final[int] = 240
"""判定に残す理由の長さの上限 (生データが、長いエラーの本文で膨らまないように)。"""


# --- 差し替えられる継ぎ目 -----------------------------------------------------


class CodeSandbox(Protocol):
    """コードを隔離して動かす口 (`scoring/sandbox.ContainerSandbox` が満たす)。

    `SandboxRunner` (`scoring/sandbox`) に、計測ランへ記録するイメージの識別子
    (`image_ref`) を足したもの。単体の試験は、これを差し替えて Docker なしで
    まとまりを流す。
    """

    def available(self) -> bool | SandboxUnavailable: ...

    def run_python(self, source: str, timeout_s: float) -> SandboxResult: ...

    def image_ref(self) -> str | None: ...


SandboxFactory = Callable[[SandboxSettings], CodeSandbox]
"""設定から隔離の実行環境を作る口。既定は `ContainerSandbox`。"""

ProblemsLoader = Callable[[], tuple[DatasetRef, list[CodeProblem]]]
"""公開のコードの課題を読む口。既定は `humaneval.load_humaneval_plus`。"""


def _default_sandbox(settings: SandboxSettings) -> CodeSandbox:
    return ContainerSandbox(settings)


def _default_problems() -> tuple[DatasetRef, list[CodeProblem]]:
    return humaneval.load_humaneval_plus()


@dataclass(frozen=True)
class QualityRunInfo:
    """計測ランに記録するもの (module の docstring「計測ランへの記録」)。"""

    datasets: tuple[DatasetRef, ...] = ()
    sandbox_image_ref: str | None = None


@dataclass(frozen=True)
class _CodeSetup:
    """コードの課題の条件の、計画のときに決まったもの。"""

    run_id: str
    sandbox: CodeSandbox
    dataset: DatasetRef
    problems: tuple[CodeProblem, ...]
    image_ref: str | None


# --- まとまり -----------------------------------------------------------------


class QualitySuite:
    """品質の検査のまとまり (design.md suites)。

    計測ランごとの覚え書き (コードの課題の下ごしらえと、探す課題が上限に
    当たった長さ) を持つ。どちらも `plan()` が消すので、`SUITE` を計測ランを
    またいで使い回しても混ざらない。
    """

    name = SuiteName.QUALITY

    def __init__(
        self,
        *,
        sandbox_factory: SandboxFactory = _default_sandbox,
        problems_loader: ProblemsLoader = _default_problems,
    ) -> None:
        self._sandbox_factory = sandbox_factory
        self._problems_loader = problems_loader
        self._code: _CodeSetup | None = None
        self._limit_run_id: str | None = None
        self._limit_tokens: int | None = None

    # --- 計画 ---

    def plan(self, ctx: SuiteContext) -> list[ConditionPlan | SkippedCondition]:
        self._forget()
        settings = ctx.profile.quality
        planned: list[ConditionPlan | SkippedCondition] = [
            plan_condition(
                ctx,
                suite=self.name,
                key=TOOLCALL_CONDITION_KEY,
                trials=settings.toolcall_tasks,
                warmup_trials=0,
                max_tokens=TOOLCALL_MAX_TOKENS,
            ),
            self._plan_code(ctx, settings),
        ]
        planned.extend(self._plan_needle(ctx, settings))
        return planned

    def run_info(self) -> QualityRunInfo:
        """8.1 が `manifest` に書き写すもの (5.5)。`plan()` のあとに読む。"""
        setup = self._code
        if setup is None:
            return QualityRunInfo()
        return QualityRunInfo(datasets=(setup.dataset,), sandbox_image_ref=setup.image_ref)

    def _plan_code(
        self, ctx: SuiteContext, settings: QualitySettings
    ) -> ConditionPlan | SkippedCondition:
        """コードの課題を計画する。送る前に、隔離と課題を確かめる (6.7)。"""
        sandbox = self._sandbox_factory(ctx.profile.sandbox)
        status = sandbox.available()
        if status is not True:
            reason = status.reason if isinstance(status, SandboxUnavailable) else "理由は不明"
            return self._skip_code(f"コードの隔離の実行環境が使えない: {reason}")

        try:
            dataset, problems = self._problems_loader()
        except humaneval.DatasetUnavailable as exc:
            # 飛ばす理由にしてよいのは、これだけ (注 6.5 → 6.7)
            return self._skip_code(f"公開のコードの課題を用意できない: {exc}")

        selected = humaneval.select_problems(problems, settings.code_problem_limit)
        if not selected:
            return self._skip_code("採点できるコードの課題が 1 問もない")

        self._code = _CodeSetup(
            run_id=ctx.run_id,
            sandbox=sandbox,
            dataset=dataset,
            problems=tuple(selected),
            image_ref=sandbox.image_ref(),
        )
        return plan_condition(
            ctx,
            suite=self.name,
            key=CODE_CONDITION_KEY,
            trials=len(selected),
            warmup_trials=0,
            max_tokens=settings.code_max_tokens,
        )

    def _skip_code(self, reason: str) -> SkippedCondition:
        return skip_condition(
            self.name,
            CODE_CONDITION_KEY,
            f"{reason}。この条件には要求を 1 つも送らなかった (ほかの条件は実行する)",
        )

    def _plan_needle(
        self, ctx: SuiteContext, settings: QualitySettings
    ) -> Iterator[ConditionPlan | SkippedCondition]:
        """長さ × 位置のすべての組を計画する (短い長さから、小さい位置から)。"""
        for length in _needle_lengths(settings):
            for depth in _needle_depths(settings):
                yield plan_condition(
                    ctx,
                    suite=self.name,
                    key=needle_condition_key(length, depth),
                    trials=settings.trials_per_cell,
                    warmup_trials=0,
                    max_tokens=NEEDLE_MAX_TOKENS,
                    target_input_tokens=length,
                )

    # --- 実行 ---

    async def run_condition(
        self, ctx: SuiteContext, cond: ConditionPlan
    ) -> AsyncIterator[TrialRecord]:
        if cond.key == TOOLCALL_CONDITION_KEY:
            source = self._run_toolcall(ctx, cond)
        elif cond.key == CODE_CONDITION_KEY:
            source = self._run_code(ctx, cond)
        elif cond.key.startswith(_NEEDLE_KEY_PREFIX):
            source = self._run_needle(ctx, cond)
        else:
            raise ValueError(f"quality: 知らない条件の鍵 (key={cond.key!r})")
        async for record in source:
            yield record

    async def _run_toolcall(
        self, ctx: SuiteContext, cond: ConditionPlan
    ) -> AsyncIterator[TrialRecord]:
        """ツール呼び出しの課題 (5.1)。前置きの会話は付けない。"""
        for trial_index, warmup in iter_trials(cond):
            task = make_tool_task(TOOLCALL_TASK_INDEX_BASE + trial_index, ctx.profile.seed)
            record = await run_trial(
                ctx,
                cond,
                trial_index=trial_index,
                warmup=warmup,
                messages=single_user_message(task.prompt),
                tools=task.tools,
                measures_decode_speed=False,
                expect_full_output=False,
                verdict=_toolcall_verdict(ctx, task),
            )
            yield record
            abort_if_context_limit(cond, record)

    async def _run_code(self, ctx: SuiteContext, cond: ConditionPlan) -> AsyncIterator[TrialRecord]:
        """コードの課題 (5.2)。1 試行 = 1 問で、選んだ並びの位置が試行の番号。"""
        setup = self._code_setup(ctx)
        timeout_s = ctx.profile.sandbox.timeout_s
        consecutive_errors = 0

        for trial_index, warmup in iter_trials(cond):
            if trial_index >= len(setup.problems):
                raise ValueError(
                    f"quality: 課題の数より試行が多い (trials={cond.trials}, "
                    f"問題={len(setup.problems)})"
                )
            problem = setup.problems[trial_index]
            record = await run_trial(
                ctx,
                cond,
                trial_index=trial_index,
                warmup=warmup,
                messages=single_user_message(CODE_INSTRUCTION + problem.prompt),
                measures_decode_speed=False,
                expect_full_output=False,
            )

            broken = False
            if record.result.error is not None:
                verdict = not_scored_verdict(_request_failed_reason(record.result))
            else:
                try:
                    verdict = await _score_code_answer(
                        _text_of(record.result), problem, setup.sandbox, timeout_s
                    )
                except SandboxError as exc:
                    consecutive_errors += 1
                    broken = consecutive_errors >= MAX_CONSECUTIVE_SANDBOX_ERRORS
                    verdict = not_scored_verdict(f"隔離の実行に失敗した: {exc}")
                else:
                    consecutive_errors = 0

            record = _with_verdict(record, _with_task_id(verdict, problem.task_id))
            yield record
            abort_if_context_limit(cond, record)
            if broken:
                raise ConditionAborted(
                    skip_condition(
                        self.name,
                        cond.key,
                        "隔離の実行が "
                        f"{MAX_CONSECUTIVE_SANDBOX_ERRORS} 回続けて失敗したので、"
                        f"残りの問題には送らずに打ち切った。送った問題: {trial_index + 1} / "
                        f"{len(setup.problems)} 問",
                    )
                )

    async def _run_needle(
        self, ctx: SuiteContext, cond: ConditionPlan
    ) -> AsyncIterator[TrialRecord]:
        """探す課題 (5.3)。長さと位置は、条件の鍵と計画から読む。"""
        length, depth = _needle_cell(cond)
        self._abort_if_limit_already_reached(ctx, cond, length)
        ratio = ctx.profile.chars_per_token[ContentKind.PROSE_EN]

        for trial_index, warmup in iter_trials(cond):
            case = make_needle_case(
                target_tokens=length,
                depth_pct=depth,
                index=trial_index,
                seed=ctx.profile.seed,
                chars_per_token=ratio,
            )
            record = await run_trial(
                ctx,
                cond,
                trial_index=trial_index,
                warmup=warmup,
                messages=single_user_message(f"{case.haystack}\n\n{case.question}"),
                measures_decode_speed=False,
                expect_full_output=False,
                verdict=_needle_verdict(case),
            )
            yield record
            try:
                abort_if_context_limit(cond, record)
            except ConditionAborted:
                self._remember_limit(ctx.run_id, length)
                raise

    # --- 計測ランごとの覚え書き ---

    def _forget(self) -> None:
        self._code = None
        self._limit_run_id = None
        self._limit_tokens = None

    def _code_setup(self, ctx: SuiteContext) -> _CodeSetup:
        setup = self._code
        if setup is None or setup.run_id != ctx.run_id:
            raise ValueError("quality: コードの課題は、同じ計測ランの plan() のあとに実行すること")
        return setup

    def _remember_limit(self, run_id: str, length: int) -> None:
        self._limit_run_id = run_id
        self._limit_tokens = length

    def _abort_if_limit_already_reached(
        self, ctx: SuiteContext, cond: ConditionPlan, length: int
    ) -> None:
        """すでに上限に当たっていれば、この長さには 1 つも送らずに打ち切る (3.6)。"""
        reached = self._limit_tokens
        if self._limit_run_id != ctx.run_id or reached is None or length < reached:
            return
        raise ConditionAborted(
            skip_condition(
                self.name,
                cond.key,
                "すでに入力の長さの上限に達しているので、この長さには送らなかった。"
                f"到達した長さ: {reached} トークン (狙い)",
            )
        )


SUITE: Final[QualitySuite] = QualitySuite()
"""runner (3.5) が読み込む、この module のまとまりの実体。

計測の実行につなぐとき (8.1) は、`QualitySuite()` を毎回作ってもよい (計測ラン
ごとの覚え書きそのものがなくなる。注 7.2 → 8.1 と同じ考え方)。
"""


# --- 条件の鍵 -----------------------------------------------------------------


def needle_condition_key(tokens: int, depth_pct: int) -> str:
    """探す課題の条件の鍵 (`(8000, 25)` → `quality/needle/8k/d25`)。

    長さの表記は `tokens_label` (3.1 の決めごと) をそのまま使う。
    """
    if not 0 <= depth_pct <= 100:
        raise ValueError(f"needle_condition_key: depth_pct は 0〜100 (depth_pct={depth_pct})")
    return condition_key(
        SuiteName.QUALITY, _NEEDLE_PART, tokens_label(tokens), f"{_DEPTH_PREFIX}{depth_pct}"
    )


def _needle_cell(cond: ConditionPlan) -> tuple[int, int]:
    """条件から、探す課題の長さと位置を読み取る (鍵と計画が食い違えば投げる)。"""
    length = cond.target_input_tokens
    if length is None:
        raise ValueError(f"quality: 条件に target_input_tokens が必要 (key={cond.key!r})")
    parts = cond.key.split("/")
    if len(parts) != 4 or not parts[3].startswith(_DEPTH_PREFIX):
        raise ValueError(f"quality: 探す課題の条件の鍵の形が想定と違う (key={cond.key!r})")
    try:
        depth = int(parts[3][len(_DEPTH_PREFIX) :])
    except ValueError as exc:
        raise ValueError(f"quality: 条件の鍵から位置を読めない (key={cond.key!r})") from exc
    if cond.key != needle_condition_key(length, depth):
        raise ValueError(
            f"quality: 条件の鍵と狙いの長さが食い違う (key={cond.key!r}, "
            f"target_input_tokens={length})"
        )
    return length, depth


def _needle_lengths(settings: QualitySettings) -> tuple[int, ...]:
    """探す課題の長さ (短いほうから。同じ長さは 1 つにまとめる)。

    上限に当たったときに「それより長い長さ」を確実に飛ばせるよう、必ず昇順に
    する。同じ長さが 2 つあると条件の鍵がぶつかるので、重複は落とす。
    """
    return tuple(sorted(set(settings.needle_lengths)))


def _needle_depths(settings: QualitySettings) -> tuple[int, ...]:
    """埋める位置 (小さいほうから。同じ位置は 1 つにまとめる)。"""
    return tuple(sorted(set(settings.needle_depths)))


# --- 判定 ---------------------------------------------------------------------


def _text_of(result: StreamResult) -> str:
    """応答の `text` ブロックだけをつなぐ (thinking は採点に使わない。注 6.4)。"""
    return "".join(block.text or "" for block in result.blocks if block.type == "text")


def _bounded(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= _MAX_DETAIL_CHARS:
        return flat
    return flat[: _MAX_DETAIL_CHARS - 1] + "…"


def _request_failed_reason(result: StreamResult) -> str:
    """要求そのものが失敗した試行の理由 (採点できなかった側に数える。5.7)。"""
    error = result.error
    if error is None:  # pragma: no cover - 呼び出し元が確かめている
        return "要求そのものが失敗した"
    status = f" HTTP {error.http_status}" if error.http_status is not None else ""
    return _bounded(f"要求そのものが失敗したので採点できない: {error.kind}{status}")


def _toolcall_verdict(ctx: SuiteContext, task: ToolTask) -> VerdictFn:
    """応答を 9 種類に分ける関数 (`run_trial` が、失敗した試行にも呼ぶ)。

    記法の目印は、対象サーバーの定義から受け取る (6.3)。課題の側が壊れて
    いれば `ToolTaskError` が外に出る (握りつぶさない。注 6.3)。
    """
    markers = list(ctx.target.tool_markup_markers)

    def verdict(result: StreamResult) -> ToolCallVerdict:
        return classify_tool_call(result, task, markers)

    return verdict


def _needle_verdict(case: NeedleCase) -> VerdictFn:
    """探す課題の採点 (`run_trial` が、失敗した試行にも呼ぶ)。"""

    def verdict(result: StreamResult) -> QualityVerdict:
        if result.error is not None:
            return QualityVerdict(
                task="needle",
                outcome=QualityOutcome.NOT_SCORED,
                detail=_request_failed_reason(result),
            )
        return score_needle(_text_of(result), case)

    return verdict


async def _score_code_answer(
    text: str, problem: CodeProblem, sandbox: CodeSandbox, timeout_s: float
) -> QualityVerdict:
    """1 つの応答を採点する (取り出し → 組み立て → 隔離で実行 → 採点)。

    組み立ては印つきの `build_checked_program` を使う。`check(...)` のあとで
    しか印が書かれないので、`sys.exit(0)` で検査の手前で終わる答えを、合格に
    数えない (`scoring/code.py`「印」)。

    `scoring/code.py` の採点の口が変わっても、直すのはこの関数の中だけで済む。
    `SandboxError` (設備の失敗) と `CodeTaskError` (課題の側の誤り) は、そのまま
    外に出す。前者は呼び出し側が「採点できなかった」にし、続いたら条件を打ち
    切る。後者は、モデルの誤りとして握りつぶさない。
    """
    code = extract_code(text, problem.entry_point)
    if code is None:
        # モデルは答えたが、使えるコードがなかった → 不正解 (5.7)
        return no_code_verdict()
    program = build_checked_program(problem, code)
    result = await _run_off_the_loop(sandbox, program.source, timeout_s)
    return score_checked(result, program)


async def _run_off_the_loop(sandbox: CodeSandbox, program: str, timeout_s: float) -> SandboxResult:
    """`run_python` を別の糸で動かす (module の docstring「隔離の実行は、別の糸で」)。"""
    running = asyncio.ensure_future(asyncio.to_thread(sandbox.run_python, program, timeout_s))
    try:
        return await asyncio.shield(running)
    except asyncio.CancelledError:
        # 糸の中の同期の処理は取り消せない。コンテナを残さないために、後始末が
        # 終わるのを待ってから、取り消しを伝える
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.wait_for(
                asyncio.shield(running), timeout=timeout_s + SANDBOX_REAP_MARGIN_S
            )
        raise


def _with_task_id(verdict: QualityVerdict, task_id: str) -> QualityVerdict:
    """判定に、どの問題のものかを残す (`TrialRecord` に課題の番号の欄はない)。"""
    return QualityVerdict(
        task=verdict.task,
        outcome=verdict.outcome,
        detail=_bounded(f"{task_id} {verdict.detail}"),
        sandbox=verdict.sandbox,
    )


def _with_verdict(record: TrialRecord, verdict: QualityVerdict) -> TrialRecord:
    """レコードに判定を入れ直す (凍結の型なので、検証し直して作り直す。注 1.2)。

    コードの課題の採点は隔離の実行を待つので、`run_trial(verdict=…)` の同期の
    口には載らない。
    """
    data = record.model_dump()
    data["verdict"] = verdict.model_dump()
    return TrialRecord.model_validate(data)
