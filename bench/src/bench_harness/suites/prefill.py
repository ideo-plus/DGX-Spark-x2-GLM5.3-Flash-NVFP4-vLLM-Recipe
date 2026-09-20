"""入力の処理と最初のトークンまでの時間のまとまり (task 3.3: suites/prefill)。

`suites/base.py` の上に薄く載る (Requirement 3、design.md「prefill」の行)。
`Profile.prefill.target_input_tokens` の長さ (既定 8k / 32k / 128k) ごとに、
キャッシュが効かない `prefill/cold/<label>` と、効く `prefill/warm/<label>` の
2 つの条件を作る (3.1)。上限に収まらない長さは `plan_condition` が送る前に
飛ばし (3.6)、上限が不明で対象サーバーが実行時に HTTP 400 を返したときは
`run_condition` の最後で `abort_if_context_limit` が `ConditionAborted` を
投げる (base.py の契約どおり)。

## 入力の形

システムプロンプトは `system_with_prefix(nonce)` (1 行目が先頭の識別子だけの
行で、それ以外の行はない)。ユーザーの発話は 1 つで、長い DOCUMENT のあとに
短い QUESTION を続けたものにする (design.md「prefill」の行、3.1)。

DOCUMENT は、コード・ログ・英語の散文の 3 節をほぼ均等な長さでつないだもの
(takt のエージェント作業――コードの変更、ログの確認、状況の報告――に近い
構成にするため。`corpus/synth.py` の `code` / `log` / `prose` をそのまま使う)。
条件の長さ `length` だけから決まる種 (`(profile.seed, "prefill", length)`、
`trial_seed` の形を借りる) で作るので、同じ `length` なら cold と warm の
どちらの条件からでも、また試行や計測ランをまたいでも、バイト単位で同じ
DOCUMENT になる (cold と warm の違いが、キャッシュの効き方だけになるように)。

### 長さの狙い方 (文字数の算段)

DOCUMENT のトークン数の狙いは、`length` から次の 3 つを引いて決める。差し
引く量はどれも固定で、実際に選ばれる QUESTION の中身には依存しない (だから
DOCUMENT 自体が試行ごとに変わらない)。

- `_HEADER_CHARS`: 先頭の識別子の行 (`[bench-prefix <32 桁の 16 進>]`) の
  文字数。常に 47 文字
- `_JSON_SKELETON_CHARS`: 要求の JSON のうち、本文を除いた骨組み
  (`{"messages": [{"content": [{"text": "", "type": "text"}], "role": "user"}],
  "system": ""}`) の文字数。89 文字
- `_QUESTION_ALLOWANCE_CHARS`: QUESTION の型紙のうち最長のものに、20 文字の
  余裕を足した値

この 3 つの合計を `Profile.chars_per_token[ContentKind.PROSE_EN]` (どれも
短い英語の文字列であるため) でトークン数に換算し、`length` から引いた値を
DOCUMENT の狙いにする。実際のトークン数は、JSON への書き出しで `"` や
`\n` が `\"` や `\\n` に展開される分だけ、この狙いより数 % 大きくなりうる
(狙った長さから数 % ずれることは、design.md の Decision で織り込み済み)。
`Profile.length_tolerance` (既定 5%) の範囲に収まることは、この module の
試験と、実機での校正 (task 8.2) で確かめる

## cold (3.4) と warm (3.5)

- **cold**: 試行ごと (慣らしを含む) に `cold_prefix_nonce` で違う識別子を
  使うので、先頭の行がそれまでに送ったどの入力とも重ならない。慣らしの回数
  は `DEFAULT_CONNECTION_WARMUP_TRIALS` (接続の確立を最初のトークンまでの
  時間に混ぜないためだけの慣らしで、キャッシュを温める意味はない――cold は
  そもそもキャッシュに当てないのが目的)。この慣らしで狙いの長さぶんの
  DOCUMENT を毎回作ると、128k の条件では慣らしのためだけに 128k トークンの
  前処理を払うことになるので、`_short_warmup_document` が作る数百トークンの
  短い DOCUMENT を使う。この要求のトークン数は条件の狙い (`length`) と大きく
  外れるので `TrialFlag.LENGTH_OFF_TARGET` が付くが、慣らしの記録は集計から
  除かれるので実害はない (base.py の共通の印を参照)
- **warm**: すべての要求 (慣らしを含む) が `warm_prefix_nonce` で同じ識別子
  を使い、同じ DOCUMENT を送る。末尾の QUESTION だけが試行ごとに違う。
  慣らしの回数は `max(Profile.prefill.warmup_trials, 1)` (少なくとも 1 回。
  1 回目がキャッシュを温める慣らしで、2 回目以降が集計の対象になる。
  `warm_prefix_nonce` は `run_id` を含まない (base.py の決めごとどおり) ので、
  別の計測ランでも同じ識別子になる。これは意図した振る舞いである。もし
  `run_id` を含めていたら、2 回目の計測ランの 1 回目の要求 (慣らし) がすでに
  対象サーバーのキャッシュに当たってしまう―― だが、それはまさに慣らしが
  存在する理由 (キャッシュの状態を、計測ランをまたいでもそろえること) と
  同じ結果になるので、問題ではない

## QUESTION

`_question_for` が、`trial_seed(profile.seed, cond.key, trial_index, warmup=…)`
(base.py の決めごとどおり `run_id` を含まない) から選ぶ。だから、同じ条件の
同じ試行の番号は、計測ランをまたいでも同じ QUESTION になる。短く、出力の
上限 (`Profile.prefill.max_tokens`、既定 16) の中で答えられる問いにしてある。
DOCUMENT の中身を実際に採点することはしない (prefill が測るのは速さだけ)。

依存するのは標準ライブラリと `bench_harness` の `types`、`corpus`、
`suites.base` だけ (design.md の依存の向き)。
"""

from __future__ import annotations

import math
import random
from collections.abc import AsyncIterator
from typing import Final

from bench_harness.corpus.synth import PREFIX_NONCE_HEX_LEN, prefix_header_line
from bench_harness.suites.base import (
    DEFAULT_CONNECTION_WARMUP_TRIALS,
    SuiteContext,
    abort_if_context_limit,
    cold_prefix_nonce,
    condition_key,
    iter_trials,
    plan_condition,
    run_trial,
    single_user_message,
    system_with_prefix,
    tokens_label,
    trial_seed,
    warm_prefix_nonce,
)
from bench_harness.types import (
    ConditionPlan,
    ContentKind,
    SkippedCondition,
    SuiteName,
    TrialRecord,
)

__all__ = ["SUITE", "PrefillSuite"]

_COLD_LABEL: Final[str] = "cold"
_WARM_LABEL: Final[str] = "warm"

_DOC_SEED_KEY: Final[str] = "prefill"
"""DOCUMENT の種を作るときの `trial_seed` の条件の鍵。実際の条件の鍵
(`prefill/cold/8k` など) を使わないのは、cold と warm とで同じ DOCUMENT に
するため (`length` (と、慣らし専用の短い DOCUMENT では warmup=True) だけで
決まるようにする)。"""

_MIN_DOCUMENT_TOKENS: Final[int] = 60
"""DOCUMENT のトークン数の狙いの下限。3 節に割るので、各節が空にならないため
の安全弁 (通常の長さ (8k 以上) では効かない)。"""

_COLD_WARMUP_DOCUMENT_TOKENS: Final[int] = 300
"""cold の接続の慣らしに使う、短い DOCUMENT のトークン数の狙い (「数百
トークン」、tasks.md 3.3 の完了の状態)。"""

_HEADER_CHARS: Final[int] = len(prefix_header_line("0" * PREFIX_NONCE_HEX_LEN))
"""先頭の識別子の行の文字数。nonce は常に `PREFIX_NONCE_HEX_LEN` 桁なので、
この行の長さは中身によらず一定 (system_with_prefix と同じ前提)。"""

_JSON_SKELETON_CHARS: Final[int] = len(
    '{"messages": [{"content": [{"text": "", "type": "text"}], "role": "user"}], "system": ""}'
)
"""要求の JSON のうち、本文 (system の中身とユーザーの発話の text) を除いた
骨組みの文字数。`fake_server._count_input_tokens` と `build_request_body` が
組み立てる形 (`system`、`messages` だけを見る。`tools` は渡さないので出ない)
に合わせてある。"""

_QUESTION_TEMPLATES: Final[tuple[str, ...]] = (
    "In one short sentence, what kind of material is this?",
    "In a few words, name one system mentioned above.",
    "Briefly, what does the log section describe?",
    "In one sentence, summarize the code section above.",
    "Name one outcome mentioned in the material above.",
    "In a few words, what is this document mostly about?",
)
"""DOCUMENT の末尾に足す、短い QUESTION の型紙。出力の上限 (既定 16 トークン)
の中で答えられる、短い問いにしてある。"""

_QUESTION_ALLOWANCE_CHARS: Final[int] = max(len(question) for question in _QUESTION_TEMPLATES) + 20
"""QUESTION が使いうる文字数の見積もり (最長の型紙 + 余裕)。DOCUMENT の狙いを
決めるときに使う固定値で、実際に選ばれた QUESTION の長さには依存させない
(そうしないと DOCUMENT が試行ごとに変わってしまう)。"""


class PrefillSuite:
    """入力の処理と最初のトークンまでの時間のまとまり (design.md suites)。"""

    name = SuiteName.PREFILL

    def plan(self, ctx: SuiteContext) -> list[ConditionPlan | SkippedCondition]:
        settings = ctx.profile.prefill
        planned: list[ConditionPlan | SkippedCondition] = []
        for length in settings.target_input_tokens:
            label = tokens_label(length)
            planned.append(
                plan_condition(
                    ctx,
                    suite=self.name,
                    key=condition_key(self.name, _COLD_LABEL, label),
                    trials=settings.trials,
                    warmup_trials=DEFAULT_CONNECTION_WARMUP_TRIALS,
                    max_tokens=settings.max_tokens,
                    target_input_tokens=length,
                )
            )
            planned.append(
                plan_condition(
                    ctx,
                    suite=self.name,
                    key=condition_key(self.name, _WARM_LABEL, label),
                    trials=settings.trials,
                    warmup_trials=max(settings.warmup_trials, 1),
                    max_tokens=settings.max_tokens,
                    target_input_tokens=length,
                )
            )
        return planned

    async def run_condition(
        self, ctx: SuiteContext, cond: ConditionPlan
    ) -> AsyncIterator[TrialRecord]:
        length = cond.target_input_tokens
        if length is None:
            raise ValueError(f"prefill: 条件に target_input_tokens が必要 (key={cond.key!r})")
        cold = _is_cold(cond)
        document = _document_for_length(ctx, length)
        short_document = _short_warmup_document(ctx) if cold else None

        for trial_index, warmup in iter_trials(cond):
            if cold and warmup:
                assert short_document is not None
                body_document = short_document
            else:
                body_document = document

            nonce = (
                cold_prefix_nonce(ctx, cond, trial_index=trial_index, warmup=warmup)
                if cold
                else warm_prefix_nonce(ctx, cond)
            )
            question = _question_for(ctx, cond, trial_index=trial_index, warmup=warmup)
            record = await run_trial(
                ctx,
                cond,
                trial_index=trial_index,
                warmup=warmup,
                system=system_with_prefix(nonce),
                messages=single_user_message(f"{body_document}\n\n{question}"),
                measures_decode_speed=False,
                expect_full_output=False,
            )
            yield record
            abort_if_context_limit(cond, record)


SUITE: Final[PrefillSuite] = PrefillSuite()
"""runner (3.5) が読み込む、この module のまとまりの実体。"""


# --- 条件の鍵から cold/warm を読み取る -------------------------------------


def _is_cold(cond: ConditionPlan) -> bool:
    """条件の鍵 (`prefill/cold/<label>` / `prefill/warm/<label>`) を読み取る。"""
    parts = cond.key.split("/")
    if (
        len(parts) != 3
        or parts[0] != SuiteName.PREFILL.value
        or parts[1]
        not in (
            _COLD_LABEL,
            _WARM_LABEL,
        )
    ):
        raise ValueError(f"prefill: 条件の鍵の形が想定と違う (key={cond.key!r})")
    return parts[1] == _COLD_LABEL


# --- DOCUMENT を作る ---------------------------------------------------------


def _document_target_tokens(ctx: SuiteContext, length: int) -> int:
    """DOCUMENT に割り当てるトークン数の狙い (この module の docstring を参照)。"""
    allowance_chars = _HEADER_CHARS + _JSON_SKELETON_CHARS + _QUESTION_ALLOWANCE_CHARS
    chars_per_token = ctx.profile.chars_per_token[ContentKind.PROSE_EN]
    allowance_tokens = math.ceil(allowance_chars / chars_per_token)
    return max(_MIN_DOCUMENT_TOKENS, length - allowance_tokens)


def _compose_document(
    ctx: SuiteContext, target_tokens: int, *, seed_trial_index: int, warmup: bool
) -> str:
    """コード・ログ・英語の散文の 3 節を、ほぼ均等な長さでつないで DOCUMENT にする。

    3 節の種は `trial_seed` を `stream_index` だけ変えて作る。`(seed_trial_index,
    warmup)` が同じなら、呼び出しのたびに同じ DOCUMENT になる (`run_id` にも、
    条件の鍵にも依存しない)。
    """
    base = max(1, target_tokens // 3)
    remainder = max(0, target_tokens - base * 3)
    seed = ctx.profile.seed
    code_seed = trial_seed(seed, _DOC_SEED_KEY, seed_trial_index, warmup=warmup, stream_index=0)
    log_seed = trial_seed(seed, _DOC_SEED_KEY, seed_trial_index, warmup=warmup, stream_index=1)
    prose_seed = trial_seed(seed, _DOC_SEED_KEY, seed_trial_index, warmup=warmup, stream_index=2)
    sections = (
        ctx.corpus.code(base + remainder, code_seed),
        ctx.corpus.log(base, log_seed),
        ctx.corpus.prose("en", base, prose_seed),
    )
    return "\n\n".join(sections)


def _document_for_length(ctx: SuiteContext, length: int) -> str:
    """狙いの長さ `length` に対応する DOCUMENT (cold と warm で共有)。"""
    target_tokens = _document_target_tokens(ctx, length)
    return _compose_document(ctx, target_tokens, seed_trial_index=length, warmup=False)


def _short_warmup_document(ctx: SuiteContext) -> str:
    """cold の接続の慣らし専用の、短い DOCUMENT (この module の docstring を参照)。"""
    return _compose_document(
        ctx,
        _COLD_WARMUP_DOCUMENT_TOKENS,
        seed_trial_index=_COLD_WARMUP_DOCUMENT_TOKENS,
        warmup=True,
    )


# --- QUESTION を作る ---------------------------------------------------------


def _question_for(ctx: SuiteContext, cond: ConditionPlan, *, trial_index: int, warmup: bool) -> str:
    """DOCUMENT の末尾に足す、短い QUESTION。試行の番号ごとに変わる。

    `trial_seed(profile.seed, cond.key, trial_index, warmup=…)` は `run_id` を
    含まない (base.py の決めごと) ので、同じ条件の同じ試行の番号は、計測ラン
    をまたいでも同じ QUESTION になる。
    """
    seed = trial_seed(ctx.profile.seed, cond.key, trial_index, warmup=warmup)
    return random.Random(seed).choice(_QUESTION_TEMPLATES)
