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


def _round_input(
    ctx: SuiteContext,
    cond: ConditionPlan,
    *,
    round_index: int,
    warmup: bool,
    stream_index: int,
) -> tuple[str, list[InputMessage]]:
    """1 本ぶんの、先頭も中身も重ならない入力を作る (4.1)。

    先頭の識別子 (`cold_prefix_nonce`) と、文章そのものを作る種
    (`trial_seed`) の両方に `stream_index` を混ぜるので、同じ回の n 本は、
    システムプロンプトの 1 行目も、文章の中身も、互いに重ならない。
    """
    nonce = cold_prefix_nonce(
        ctx, cond, trial_index=round_index, warmup=warmup, stream_index=stream_index
    )
    seed = trial_seed(
        ctx.profile.seed, cond.key, round_index, warmup=warmup, stream_index=stream_index
    )
    target_tokens = cond.target_input_tokens
    if target_tokens is None:
        # plan() は必ず profile.concurrency.input_tokens を渡す。ここに来るのは
        # 呼び出し側が計画を自分で組み立てた誤りだけ (注 2.7 に合わせて早く落とす)
        raise ValueError("concurrency: target_input_tokens のない計画は扱えない")
    document = ctx.corpus.prose(_DOCUMENT_LANG, target_tokens, seed)
    system = system_with_prefix(nonce, "長く書くこと。")
    messages = single_user_message(f"{document}\n\n{_LONG_OUTPUT_INSTRUCTION}")
    return system, messages


async def _run_stream(
    ctx: SuiteContext,
    cond: ConditionPlan,
    *,
    round_index: int,
    warmup: bool,
    stream_index: int,
    gate: _RoundGate,
) -> TrialRecord:
    """1 本を、合図で送り始めて実行する。"""
    system, messages = _round_input(
        ctx, cond, round_index=round_index, warmup=warmup, stream_index=stream_index
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
    ctx: SuiteContext, cond: ConditionPlan, *, round_index: int, warmup: bool
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
    """同時処理のまとまり (design.md suites の `concurrency` の行)。"""

    name = SuiteName.CONCURRENCY

    def plan(self, ctx: SuiteContext) -> list[ConditionPlan | SkippedCondition]:
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
        for round_index, warmup in iter_trials(cond):
            records = await _run_round(ctx, cond, round_index=round_index, warmup=warmup)
            for record in records:
                yield record
            for record in records:
                abort_if_context_limit(cond, record)


SUITE: Final[ConcurrencySuite] = ConcurrencySuite()
"""runner (3.5) が import する、まとまりのインスタンス。"""
