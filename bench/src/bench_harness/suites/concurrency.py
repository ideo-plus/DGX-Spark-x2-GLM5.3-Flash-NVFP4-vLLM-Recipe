"""同時処理のまとまり (task 3.4: suites/concurrency, Requirement 4)。

`Profile.concurrency.levels` (既定 1、2、4、8) の本数ごとに 1 つの条件を計画し
(`plan_condition` を使うので、上限に収まらない本数は、ほかの条件と同じように
送る前に飛ばせる)、1 回ぶん (`round`) につき `concurrency` 本の要求を、合図
(`_RoundGate`) で同時に送り始める (design.md concurrency「合図で同時に送り
始める」、4.3)。

## 条件の鍵と主な結果/参考の区別 (4.4)

`concurrency/c{n}` (`n` は本数)。1 本と 2 本は `primary`、4 本以上は
`reference`。

## 入力: 先頭も中身も重ならない (4.1)

1 回の中の n 本は、それぞれ別のストリーム (`stream_index`) として、
`cold_prefix_nonce` (システムプロンプトの 1 行目) と、文章そのものを作る種
(`trial_seed`) の両方に `stream_index` を混ぜる。どちらも `warmup` と
`round_index` (= `trial_index`) も含むので、同じ回の n 本どうし、慣らしと
本番、回をまたいでも、先頭も中身も重ならない (design.md corpus の注、
「同時処理で先頭を重ねないには、識別子の部品にストリームの番号を足す」)。
文章のあとに、出力の上限まで書かせる指示を足す
(`run_trial(..., expect_full_output=True)` が、届かなければ `SHORT_OUTPUT`
を付ける)。

文章の狙いは `target_input_tokens − 包み` である (issue #9)。`target_input_tokens`
だけでは、識別子の行、システムプロンプト 2 行目、指示、チャットテンプレートの
包みが足されて、実際の入力が狙いより長くなる。包みの分をあらかじめ引いてから
文章を作る。

## 包みの計測 (issue #9)

包み (文章と、それを包む決まった分量) は、計測ランにつき 1 回だけ、対象
サーバーに数えさせる。`_measure_frame` が、試行と同じ組み立て関数
(`_system_for` / `_user_text`) を使い、文章を空にした要求を
`measure_frame_tokens` に渡す。数える要求は、口があれば `count_tokens` の
1 件、口がなければ `max_tokens=1` の要求 1 件である (`put_body` には保存されず、
試行としては残らないが、その条件の `/metrics` の増分には入る)。数えられない
ときは `ProbeError` が伝播する (黙って文字数の見積もりに戻さない)。結果は
`run_id` ごとに覚えておき、`plan()` が消す (`ConcurrencySuite` は
`default_suite_registry` が計測ランごとに作り直す)。

包みを測る要求のシステムプロンプトの 1 行目は、`_FRAME_NONCE_KEY` から作る
名前空間に置く (条件の鍵 `concurrency/c{n}` とは別で、どの試行の先頭とも
重ならない)。狙いが包み以下なら、文章を入れられないので、送る前に `ValueError`
で止める。

## 識別子 (`trial_index` / `round_id` / `stream_index`)

`(condition, warmup, trial_index, stream_index)` の 4 つ組が、1 つのレコード
を一意に決める。`trial_index` と `round_id` は、どちらも回の番号 (0 から、
慣らしと本番は別々に数える。`iter_trials` の決まりのとおり)。測った回 r・
ストリーム k の文章は、`trial_seed` が `run_id` を含まないので、計測ランを
またいでも同じになる (`cold_prefix_nonce` の先頭の行だけが、`run_id` を含む
ので計測ランごとに変わる)。

## 慣らし (design.md「同じ回に n 本の接続がそろう前に測らない」)

`ConcurrencySettings` に慣らしの回数の項目がないので、`plan_condition` の
既定 (`DEFAULT_CONNECTION_WARMUP_TRIALS`、1 回) を使う。慣らしの 1 回も、
本番と同じ形の丸ごとの round (n 本を合図で同時に送る) にする。これで、
測る前に n 本ぶんの接続がすべて開いた状態になる (base.py の注「条件の最初の
要求は、接続の確立を最初のトークンまでの時間に混ぜないための慣らしでもある」
を、n 本ぶんに広げたもの)。

## 合図 (`_RoundGate`) と、失敗したときに固まらないこと (4.3)

n 個の task を作って `run_trial(..., start_gate=gate.wait)` で呼ぶ。
`run_trial` は、本文を保存したあと (`put_body`)、送る直前に `start_gate` を
待つ。`_RoundGate` は、n 本がそろうまで待ち、そろった瞬間に全員を同時に
起こす。合図の**前**で例外が起きた本 (`put_body` の失敗など) があると、
残りの本はいつまでも「n 本そろう」を待ち続けて固まる。それを防ぐため、
`asyncio.gather` が例外 (`asyncio.CancelledError` を含む) を投げたら、残りの
task をすべて打ち切り、後始末を待ってから、元の例外を投げ直す
(`_run_round`)。`asyncio.CancelledError` はここでも forward されるので、
中断の合図 (SIGINT、runner が task を打ち切る) を受けたときも、動いている
task を残さずに伝播する。

## `run_condition` の返し方

1 回ぶんが終わったら、n 本のレコードを `stream_index` の順に `yield` して
から (8.1: 上限に当たった本があっても、レコードはすべて残す)、その n 本の
それぞれに `abort_if_context_limit` を呼ぶ。1 本が要求そのものの失敗
(`StreamResult.error`) でも、その回は残りを含めて n 本のレコードを返す
(4.5 は分析の仕事。ここでは記録するだけ)。

依存するのは標準ライブラリと `bench_harness` の `types`、`suites.base` だけ。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Final

from bench_harness.suites.base import (
    SuiteContext,
    abort_if_context_limit,
    cold_prefix_nonce,
    condition_key,
    iter_trials,
    measure_frame_tokens,
    plan_condition,
    run_trial,
    single_user_message,
    system_with_prefix,
    trial_seed,
)
from bench_harness.types import (
    ConditionPlan,
    InputMessage,
    Lang,
    SkippedCondition,
    SuiteName,
    Tier,
    TrialRecord,
)

__all__ = ["SUITE", "ConcurrencySuite"]

_DOCUMENT_LANG: Final[Lang] = "en"
"""1 本ずつの文章に使う言語。決めごとに特別な意味はなく、決定的であればよい。"""

_LONG_OUTPUT_INSTRUCTION: Final[str] = (
    "上の文章をそのまま繰り返さず、関連する話題を広げながら、"
    "出力の上限に達するまでできるだけ長く書き続けること。"
)

_FRAME_NONCE_KEY: Final[str] = "concurrency/frame"
"""包みを測る要求の識別子の名前空間 (issue #9)。

条件の鍵 (`concurrency/c{n}`) とは別なので、包みを測る要求の先頭の行は、
どの試行の先頭の行とも重ならない。
"""

_SYSTEM_LINES: Final[tuple[str, ...]] = ("長く書くこと。",)
"""システムプロンプトの 1 行目 (識別子の行) のあとに置く行。試行と包みの計測で
同じものを使う (包みを取りこぼさないため)。"""


class _RoundGate:
    """1 回ぶんの合図 (design.md concurrency「合図で同時に送り始める」、4.3)。

    `n` 本がそろうまで待ち、そろった瞬間に `asyncio.Event` を立てて全員を
    同時に起こす。到着を数える処理 (`self._arrived += 1` と、そろったかの
    判定) に `await` を挟まないので、ロックなしで安全 (asyncio は協調的で、
    `await` のない区間は 1 つの task しか進まない)。
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"_RoundGate: n は 1 以上 (n={n})")
        self._n = n
        self._arrived = 0
        self._released = asyncio.Event()

    async def wait(self) -> None:
        self._arrived += 1
        if self._arrived >= self._n:
            self._released.set()
        await self._released.wait()


def _tier_for(level: int) -> Tier:
    """1 本と 2 本を主な結果、4 本以上を参考にする (4.4)。"""
    return "primary" if level <= 2 else "reference"


def _system_for(nonce: str) -> str:
    """試行と包みの計測で共通のシステムプロンプト。"""
    return system_with_prefix(nonce, *_SYSTEM_LINES)


def _user_text(document: str) -> str:
    """文章と、出力の上限まで書かせる指示をまとめた本文。"""
    return f"{document}\n\n{_LONG_OUTPUT_INSTRUCTION}"


async def _measure_frame(ctx: SuiteContext) -> int:
    """要求の包み (文章と、それを包む決まった分量) を 1 回数える (issue #9)。

    試行と同じ組み立て関数 (`_system_for` / `_user_text`) を使い、文章だけを
    空にする。これで、識別子の行・システムプロンプト 2 行目・指示・チャット
    テンプレートの包みが、すべて 1 回の計測に入る (`ProbeError` は伝播する)。
    """
    nonce = ctx.corpus.prefix_nonce(ctx.profile.seed, _FRAME_NONCE_KEY, ctx.run_id)
    return await measure_frame_tokens(
        ctx, system=_system_for(nonce), messages=single_user_message(_user_text(""))
    )


def _document_tokens(cond: ConditionPlan, frame_tokens: int) -> int:
    """文章に割り当てるトークン数 (`target_input_tokens − 包み`) (issue #9)。

    狙いが包み以下なら、文章を入れられないので、送る前に `ValueError` で止める。
    """
    target_tokens = cond.target_input_tokens
    if target_tokens is None:
        # plan() は必ず profile.concurrency.input_tokens を渡す。ここに来るのは
        # 呼び出し側が計画を自分で組み立てた誤りだけ (注 2.7 に合わせて早く落とす)
        raise ValueError("concurrency: target_input_tokens のない計画は扱えない")
    document_tokens = target_tokens - frame_tokens
    if document_tokens < 1:
        raise ValueError(
            f"concurrency: 狙いの入力 {target_tokens} トークンが要求の包み "
            f"{frame_tokens} トークン以下なので、文章を入れられない"
        )
    return document_tokens


def _round_input(
    ctx: SuiteContext,
    cond: ConditionPlan,
    *,
    document_tokens: int,
    round_index: int,
    warmup: bool,
    stream_index: int,
) -> tuple[str, list[InputMessage]]:
    """1 本ぶんの、先頭も中身も重ならない入力を作る (4.1)。

    先頭の識別子 (`cold_prefix_nonce`) と、文章そのものを作る種
    (`trial_seed`) の両方に `stream_index` を混ぜるので、同じ回の n 本は、
    システムプロンプトの 1 行目も、文章の中身も、互いに重ならない。
    `document_tokens` は、狙いから包みを引いたあとの、文章に割り当てる
    トークン数 (`_document_tokens`)。
    """
    nonce = cold_prefix_nonce(
        ctx, cond, trial_index=round_index, warmup=warmup, stream_index=stream_index
    )
    seed = trial_seed(
        ctx.profile.seed, cond.key, round_index, warmup=warmup, stream_index=stream_index
    )
    document = ctx.corpus.prose(_DOCUMENT_LANG, document_tokens, seed)
    return _system_for(nonce), single_user_message(_user_text(document))


async def _run_stream(
    ctx: SuiteContext,
    cond: ConditionPlan,
    *,
    document_tokens: int,
    round_index: int,
    warmup: bool,
    stream_index: int,
    gate: _RoundGate,
) -> TrialRecord:
    """1 本を、合図で送り始めて実行する。"""
    system, messages = _round_input(
        ctx,
        cond,
        document_tokens=document_tokens,
        round_index=round_index,
        warmup=warmup,
        stream_index=stream_index,
    )
    return await run_trial(
        ctx,
        cond,
        trial_index=round_index,
        warmup=warmup,
        system=system,
        messages=messages,
        round_id=round_index,
        stream_index=stream_index,
        start_gate=gate.wait,
        measures_decode_speed=True,
        expect_full_output=True,
    )


async def _run_round(
    ctx: SuiteContext,
    cond: ConditionPlan,
    *,
    document_tokens: int,
    round_index: int,
    warmup: bool,
) -> list[TrialRecord]:
    """1 回ぶんの `cond.concurrency` 本を、合図で同時に送り始める。

    1 本でも例外 (`put_body` の失敗、`asyncio.CancelledError` を含む) を
    投げたら、残りの task をすべて打ち切り、後始末 (`return_exceptions=True`)
    を待ってから、元の例外を投げ直す。合図の前で失敗した本があっても、まだ
    合図を待っている残りの本を固まらせないための決まり (この module の
    docstring「合図と、失敗したときに固まらないこと」)。
    """
    n = cond.concurrency
    gate = _RoundGate(n)
    tasks: list[asyncio.Task[TrialRecord]] = [
        asyncio.create_task(
            _run_stream(
                ctx,
                cond,
                document_tokens=document_tokens,
                round_index=round_index,
                warmup=warmup,
                stream_index=stream_index,
                gate=gate,
            )
        )
        for stream_index in range(n)
    ]
    try:
        records = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return list(records)


class ConcurrencySuite:
    """同時処理のまとまり (design.md suites の `concurrency` の行)。

    包みの計測の結果を、計測ランごとに 1 つだけ覚えておく (module の docstring
    「包みの計測」)。状態は `plan()` が消し、`default_suite_registry` は計測ラン
    ごとに新しい実体を作るので、計測ランをまたいで混ざらない。
    """

    name = SuiteName.CONCURRENCY

    def __init__(self) -> None:
        self._frame_run_id: str | None = None
        self._frame_tokens: int | None = None

    def plan(self, ctx: SuiteContext) -> list[ConditionPlan | SkippedCondition]:
        self._forget_frame()
        settings = ctx.profile.concurrency
        return [
            plan_condition(
                ctx,
                suite=self.name,
                key=condition_key(self.name, f"c{level}"),
                trials=settings.rounds,
                max_tokens=settings.max_tokens,
                tier=_tier_for(level),
                concurrency=level,
                target_input_tokens=settings.input_tokens,
            )
            for level in settings.levels
        ]

    async def run_condition(
        self, ctx: SuiteContext, cond: ConditionPlan
    ) -> AsyncIterator[TrialRecord]:
        frame_tokens = await self._frame_tokens_for(ctx)
        document_tokens = _document_tokens(cond, frame_tokens)
        for round_index, warmup in iter_trials(cond):
            records = await _run_round(
                ctx,
                cond,
                document_tokens=document_tokens,
                round_index=round_index,
                warmup=warmup,
            )
            for record in records:
                yield record
            for record in records:
                abort_if_context_limit(cond, record)

    def _forget_frame(self) -> None:
        self._frame_run_id = None
        self._frame_tokens = None

    async def _frame_tokens_for(self, ctx: SuiteContext) -> int:
        """包みを、同じ計測ランでは 1 回だけ数えて覚えておく (issue #9)。"""
        if self._frame_run_id == ctx.run_id and self._frame_tokens is not None:
            return self._frame_tokens
        frame_tokens = await _measure_frame(ctx)
        self._frame_run_id = ctx.run_id
        self._frame_tokens = frame_tokens
        return frame_tokens


SUITE: Final[ConcurrencySuite] = ConcurrencySuite()
"""runner (3.5) が import する、まとまりのインスタンス。"""
