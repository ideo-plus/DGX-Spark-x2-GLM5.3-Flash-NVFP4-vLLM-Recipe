"""長い会話でのツール呼び出しの検査のまとまり (task 7.2: suites/agent)。

`suites/base.py` の上に薄く載る (Requirement 6、design.md「agent」の行と
「長い会話の検査の 1 段階」の流れ)。`Profile.agent` の刻みで段階を作り、
1 段階につき複数の会話 (`corpus/conversation.py`) を使って、会話の最後に
課題 (`corpus/tools.py` の `make_tool_task`) を 1 つ足して送り、応答を
`scoring/toolcall.py` の 9 種類に分ける。

## 段階 (6.1、6.8)

`start_tokens` から `end_tokens` まで `step_tokens` 刻み (既定は 2 万〜12 万
の 6 段階)。条件の鍵は `agent/stage/NNNk` で、`NNN` は千トークンを 3 桁で
0 埋めしたもの (design.md suites の表)。`tokens_label` (`20k`) を使わないのは、
要約の段階の表を鍵の文字列で並べたときにも順序が崩れないようにするため
(注 4.2)。千で割り切れない段階は、数をそのまま書く (`agent/stage/2500`)。
どちらの形も、要約の `_tokens_from_label` がそのまま読める。`1000k` のように
桁が増えても同じ (0 埋めは下限であって、切り詰めではない)。

`ConditionPlan.target_input_tokens` は段階の狙いの長さで、要約 (4.2) が段階の
表を作るときの鍵になる。

## 段階の順序と、会話の割り当て (6.1)

- 段階は**短いほうから**送る。同じ会話なら、長い段階の発話の列が短い段階の
  発話の列をそのまま先頭に含むので (7.1)、前の段階で対象サーバーに入った
  前置きのキャッシュに当たり、長い段階の 1 回目の待ち時間が縮む。takt の
  会話が伸びていく様子に合わせた順序でもある
- 1 段階につき `conversations_per_stage` 本の会話 (既定 5 本) を
  `derive_conversation_seed(Profile.seed, 会話の番号)` で作り、試行を**均等に
  かたまりで**割り振る。50 試行 ÷ 5 本なら、試行 0〜9 が会話 0、10〜19 が
  会話 1 ……となる。割り切れないときは、余りを先頭の会話から 1 つずつ配る
  (4 試行 ÷ 3 本なら 2、1、1)。続けて同じ会話を使うのは、前置きのキャッシュに
  当て続けるため (会話ごとに回すと、5 本の長い前置きがキャッシュを奪い合う)
- 1 本の会話は、1 段階につき**1 回だけ**組み立てて使い回す。12 万トークンの
  会話は約 1200 発話あるので、試行ごとに作り直さない

## 最後の 1 手 (課題)

履歴は必ず assistant の発話で終わる (7.1) ので、`user` の発話を 1 つ足すだけ
でよい。指示は `make_tool_task(index, Profile.seed).prompt` をそのまま使い、
`tools` は `task.tools` (= `TOOL_CATALOG`。履歴の前置きと同じ並び) を渡す。

課題の番号は `段階の番号 × _TASK_INDEX_STRIDE + 試行の番号` である。

- 計測ランの中で重ならない (どの試行も違う課題を解く)
- `Profile.seed` と段階と試行の番号だけで決まるので、計測ランをまたいでも
  同じになる (対応のある比較 9.2 が、これに依存する)
- 0 以上なので、会話の履歴が使う負の番号 (`history_task_index`) と住み分ける
  (7.1 の「課題の番号の住み分け」)

## 先頭の識別子 (nonce) を足さない

速さのまとまり (3.3、3.4) は、キャッシュに当たらないことを確かめるために
システムプロンプトの 1 行目に識別子を入れる。長い会話の検査は**逆**で、
takt と同じように前置きのキャッシュに当てるのが目的である (design.md
「長い会話の検査の 1 段階」、7.1「system と tools が、どの会話でも、どの段階
でも、まったく同じ」)。したがって `system` は `ConversationPrefix.system`
(= `SYSTEM_PROMPT`) をそのまま送り、`system_with_prefix` も
`cold_prefix_nonce` も使わない。送る本文が `run_id` に依存しないので、同じ
設定の 2 つの計測ランは、バイト単位で同じ要求を送る。

## 慣らしをしない

`AgentSettings` に慣らしの回数がなく、design.md の段階の表も「1 段階の試行の
数」だけを決めている。このまとまりは速さを測らない (測るのは崩れた割合だけ)
ので、接続の確立を最初のトークンまでの時間から追い出す必要がない。12 万
トークンの要求を、集計に入らない慣らしのために段階ごとに 1 回払うのは高く
つくので、`warmup_trials=0` にしてある。段階の 1 回目の試行が、その会話の
前置きを対象サーバーのキャッシュに載せる役も兼ねる。

## 共通の印 (10.7)

ツール呼び出しの応答は短く、`stop_reason` は `tool_use` である。だから
`measures_decode_speed=False` (出力のトークンが 16 未満でも、生成速度を測って
いないので印を付けない) と `expect_full_output=False` (出力の上限まで書かせる
つもりはないので、`max_tokens` で終わらなくても印を付けない) を渡す。
`LENGTH_OFF_TARGET` と出力が壊れている疑いの印は、`run_trial` が付ける
(段階の狙いの長さからの外れは、会話の組み立ての事実として残す)。

## 上限 (6.9)

3 つの段でふさぐ。どこで当たっても、**その段階から後ろの段階には要求を
送らない**。

1. **計画**: `plan_condition` が、段階の狙いの長さ + 出力の上限を
   `fits_context` にかけ、収まらない段階を送る前に飛ばす (`SkippedCondition`)。
   段階は単調に伸びるので、1 つ飛べば、それより後ろもすべて飛ぶ
2. **送る前**: 組み立てた会話の、対象サーバーが数えた長さ (`_FittedConversation.
   input_tokens`。issue #24 以降は包み (system + tools + 最後の 1 手) と履歴を
   含む会話全体) でもう一度 `fits_context` にかける。計画は**狙い**の長さで判定
   するので、実際の会話が狙いより長い段階は、ここだけが止められる。数えさせた
   要求そのものが上限を超えて HTTP 400 で断られたとき (口がなく、代わりの
   `max_tokens=1` の要求が断られた場合。`InputOverContextLimitError`) は、数えた
   長さがない。1 本目の候補なら、その段階には 1 件も送らずに `ConditionAborted` を
   投げ、理由の「到達した長さ」には、文字数からの見積もり (`approx_tokens`) を、
   対象サーバーが数えていないとわかる書き方で入れる。2 本目以降の候補なら、数え
   終えた候補のうち狙いに最も近いものを送る (数えた長さのある候補だけを送る)
3. **送ったあと**: 上限がわからない (`context_limit is None`) ときは、対象
   サーバーの HTTP 400 で初めてわかる。`abort_if_context_limit` がレコードを
   残したあとに `ConditionAborted` を投げる (base.py の契約)

2 と 3 で当たったときは、当たった段階を覚えておき、それより長い段階では要求を
送らずに `ConditionAborted` を投げる (base.py「段階を伸ばしていくまとまりは、
一度上限に当たったら、そのあとの段階では要求を送らずに `ConditionAborted` を
投げてよい」)。覚えているのは `run_id` ごとで、`plan()` が呼ばれるたびに
消す (`SUITE` は module に 1 つの実体なので、計測ランをまたいで残さない)。

飛ばした理由には、到達した長さを必ず入れる (6.9)。対象サーバーが数えた値でない
ときは (数える要求が断られた場合)、見積もりだとわかる書き方にする。要約の「止めた理由」は、
その計測ランで最初に飛ばした `agent` の条件の理由から作られる (4.2)。

## 包みの計測と、会話の長さの合わせ込み (issue #9、#24)

長い会話の入力が狙いより長くなる原因の候補は、2 つある。1 つは、手番以外の決まった
分量 (包み: `SYSTEM_PROMPT`、`TOOL_CATALOG`、最後の 1 手、チャットテンプレートの
展開) が `build_conversation` の前置きの見積もりからずれること (issue #9)。
もう 1 つは、履歴に比例する分 (発話ごとの役割の目印、`tool_use` / `tool_result` を
チャットテンプレートが包む分、結果の本文の生成器と較正の標本の比の差) で、文字数と
比からは導けない。後者は実装側の未確認の仮説で、issue #24 が示した推定ではない
(issue #24 の推定は、測った包みが組み立てに効いていない可能性と、試験の偽の数え方が
実機と違う可能性の 2 つ)。内訳を切り分ける計測はリポジトリに無く、実機での再計測は
未実施で、残るずれの原因は特定できていない。

`_measure_frame` が、`ConversationPrefix` と同じ `system` と `tools`、段階 0
試行 0 の課題の最後の 1 手をまとめた要求を、計測ランにつき 1 回だけ対象サーバーに
数えさせる (`count_tokens` の口、なければ `max_tokens=1` の要求 1 件。`put_body`
には保存されない)。数えた値を `build_conversation(..., fixed_tokens=…)` に渡すと、
前置きの見積もりが実測に置き換わる。ただし、issue #24 の実機の計測では、この置き換え
(issue #9) の前後で 6 段のうち 5 段の中央値がほとんど動かなかった (20k: 21,090 →
21,094)。これは前置きの見積もりと実測がほぼ等しい (E ≈ F) ためと推定されるが、
見積もりと実測を直接比べた記録はなく、未確認である。この置き換えだけでは止める位置が
動かないので、次の合わせ込みを足した。

そこで `_fit_conversation` が、段階 × 会話ごとに、組み立てた会話 (最後の 1 手を
含む要求そのもの) を対象サーバーに数えさせ、実測と履歴の見積もりの比
(`history_scale`) で止める手番を組み直す。手番の数が変わらなくなったら収束と
して止め、訪れた候補のうち数えた長さが狙いに最も近いものを送る (収束の判定は
数えたあとなので、止まる回も数えて 2〜4 回。上限 `_MAX_LENGTH_COUNTS` 回)。手番の
中身は変えないので `GENERATOR_VERSION` は据え置き、「同じ対象なら同じ入力」
(ADR 0001) と、段階をまたいだ先頭の一致は保たれる。数える要求は `put_body` には
残らず、その条件の `/metrics` の増分にだけ入る。

数えられないときは、包みでも会話でも `ProbeError` が伝播する (黙って見積もりに
戻さない。runner が計測ランを `aborted` にして止める)。例外は、会話の計測が
入力の長さの上限を超えて断られたとき (`InputOverContextLimitError`) だけで、
これは上限 (6.9) の扱いで段階を飛ばす (計測ランは止まらない)。包みの計測が上限を
超えて断られたときは、今までどおり `ProbeError` のまま計測ランを `aborted` にする。
包みの最後の 1 手の長さは段階・試行ごとに数十文字だけ違うが、20k の 0.1% 未満
なので、段階 0 試行 0 の課題で代表させる。包みの結果は `FrameTokensPerRun` が
`run_id` ごとに覚え、`plan()` が消す (`AgentSuite` は上限に当たった長さと同じく、
計測ランごとの状態として持つ)。

## 注意していること

- `AgentSettings.max_tokens` の既定は 256 である。呼び出しのあとに本文を続ける
  モデルでは、`stop_reason` が `max_tokens` になって `EMPTY_OR_TRUNCATED` に
  落ち、崩れた割合が実際より大きく出る (注 6.3)。実機の確認 (8.3) で分類の
  内訳を見て、必要なら設定を大きくすること。このまとまりは、段階をまたいで
  **同じ** `max_tokens` を使う (計画は条件ごとに作るが、値はどの段階でも
  `Profile.agent.max_tokens`。7.1 が言う「先頭のバイト列の一致」の前提)
- `ToolTaskError` (課題の側の誤り) は握りつぶさない。モデルの誤りとして数えて
  しまうと、崩れた割合が静かに歪む (注 6.3)

依存するのは標準ライブラリと `bench_harness` の `types`、`corpus`、
`scoring.toolcall`、`client.probe`、`suites.base` だけ (design.md の依存の
向き)。ほかのまとまりは読み込まない。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Final

from bench_harness.client.probe import InputOverContextLimitError, fits_context
from bench_harness.corpus.conversation import (
    MESSAGES_PER_ROUND,
    SYSTEM_PROMPT,
    build_conversation,
    derive_conversation_seed,
)
from bench_harness.corpus.tools import TOOL_CATALOG, make_tool_task
from bench_harness.scoring.toolcall import classify_tool_call
from bench_harness.suites.base import (
    ConditionAborted,
    FrameTokensPerRun,
    SuiteContext,
    VerdictFn,
    abort_if_context_limit,
    condition_key,
    iter_trials,
    measure_frame_tokens,
    plan_condition,
    run_trial,
    skip_condition,
)
from bench_harness.types import (
    AgentSettings,
    ConditionPlan,
    ConversationPrefix,
    InputMessage,
    SkippedCondition,
    StreamResult,
    SuiteName,
    TextBlockParam,
    ToolCallVerdict,
    ToolTask,
    TrialRecord,
)

__all__ = ["SUITE", "AgentSuite", "stage_condition_key"]

_STAGE_PART: Final[str] = "stage"
"""条件の鍵の 2 つ目の部品 (`agent/stage/020k`)。"""

_LABEL_DIGITS: Final[int] = 3
"""段階の鍵の、千トークンの桁数 (0 埋めの下限)。`1000k` のような桁の多い
段階は、そのまま 4 桁になる。"""

_TASK_INDEX_STRIDE: Final[int] = 10_000
"""課題の番号を、段階ごとに区切る幅。1 段階の試行の数は、これ以下であること
(試行の番号は 0 から始まるので、ちょうどこの数までは、次の段階とぶつからない)。"""

_MAX_LENGTH_COUNTS: Final[int] = 4
"""1 つの段階 × 会話で、組み立てた会話を対象サーバーに数えさせる回数の上限。

比の更新で手番の数が変われば組み立て直して数え直し、手番の数が変わらなく
なった回で止める。収束の判定は数えたあとに行う (変わらなかった回も数える) ので、
2〜4 回になる。上限は振動と無限ループの防止で、上限に達したら、訪れた候補の
うち最も近いものを送る。数える要求が入力の長さの上限を超えて断られたときは、
この回数の途中で止まる (`_fit_conversation`: 1 本目なら段階を飛ばし、2 本目以降
なら数え終えた候補を送る)。"""


async def _measure_frame(ctx: SuiteContext) -> int:
    """要求の包み (`SYSTEM_PROMPT` + `TOOL_CATALOG` + 最後の 1 手) を 1 回数える。

    `AgentSuite` が `FrameTokensPerRun` に渡す、測る中身。`SUITE` を作る前に
    定義しておく必要がある (module の上に置くのはそのため)。

    `ConversationPrefix.system` / `ConversationPrefix.tools` と同じ実体を渡す
    (どの会話でも、どの段階でも同じ。7.1)。最後の 1 手は段階・試行ごとに数十文字
    だけ違うが、20k の 0.1% 未満なので、段階 0 試行 0 の課題で代表させる。
    `ProbeError` は伝播する (issue #9)。包みが入力の長さの上限を超えて断られた
    ときも同じで、段階を飛ばさず、runner が計測ランを `aborted` にする。
    """
    task = make_tool_task(_task_index(0, 0), ctx.profile.seed)
    return await measure_frame_tokens(
        ctx, system=SYSTEM_PROMPT, tools=TOOL_CATALOG, messages=[_final_turn(task.prompt)]
    )


@dataclass(frozen=True)
class _FittedConversation:
    """数えて長さを合わせ込んだ 1 本の会話 (issue #24)。"""

    prefix: ConversationPrefix
    input_tokens: int
    """対象サーバーが数えた、最後の 1 手を含む要求全体の長さ。"""


class AgentSuite:
    """長い会話でのツール呼び出しの検査のまとまり (design.md suites)。

    上限に当たったことと、包みの計測の結果を、計測ランごとの状態として持つ
    (module の docstring「上限」「包みの計測」を参照)。包みの計測の覚え方は
    `FrameTokensPerRun` が持つ。状態は `plan()` が消すので、`SUITE` を計測ランを
    またいで使い回しても混ざらない。
    """

    name = SuiteName.AGENT

    def __init__(self) -> None:
        self._limit_run_id: str | None = None
        self._limit_tokens: int | None = None
        self._frame = FrameTokensPerRun(_measure_frame)

    def plan(self, ctx: SuiteContext) -> list[ConditionPlan | SkippedCondition]:
        self._forget_run_state()
        settings = ctx.profile.agent
        _check_trials_fit_the_index_stride(settings)
        return [
            plan_condition(
                ctx,
                suite=self.name,
                key=stage_condition_key(length),
                trials=settings.trials_per_stage,
                warmup_trials=0,
                max_tokens=settings.max_tokens,
                target_input_tokens=length,
            )
            for length in stage_lengths(settings)
        ]

    async def run_condition(
        self, ctx: SuiteContext, cond: ConditionPlan
    ) -> AsyncIterator[TrialRecord]:
        length = cond.target_input_tokens
        if length is None:
            raise ValueError(f"agent: 条件に target_input_tokens が必要 (key={cond.key!r})")
        settings = ctx.profile.agent
        self._abort_if_limit_already_reached(ctx, cond, length)

        frame_tokens = await self._frame.tokens_for(ctx)
        ordinal = _stage_ordinal(settings, length)
        sizes = conversation_sizes(cond.trials, settings.conversations_per_stage)
        built: dict[int, _FittedConversation] = {}

        for trial_index, warmup in iter_trials(cond):
            conversation_index = conversation_for(trial_index, sizes)
            fitted = built.get(conversation_index)
            if fitted is None:
                measuring_task = make_tool_task(_task_index(ordinal, 0), ctx.profile.seed)
                try:
                    fitted = await _fit_conversation(
                        ctx,
                        cond,
                        length,
                        conversation_index,
                        frame_tokens=frame_tokens,
                        final_turn=_final_turn(measuring_task.prompt),
                    )
                except ConditionAborted:
                    # 数えさせた会話が、対象サーバーに上限を超えるものとして断られた (6.9)
                    self._remember_limit(ctx.run_id, length)
                    raise
                built[conversation_index] = fitted
            task = make_tool_task(_task_index(ordinal, trial_index), ctx.profile.seed)
            self._abort_if_the_built_conversation_does_not_fit(
                ctx, cond, length, fitted.input_tokens
            )

            record = await run_trial(
                ctx,
                cond,
                trial_index=trial_index,
                warmup=warmup,
                system=fitted.prefix.system,
                messages=[*fitted.prefix.messages, _final_turn(task.prompt)],
                tools=task.tools,
                measures_decode_speed=False,
                expect_full_output=False,
                verdict=_verdict_for(ctx, task),
            )
            yield record
            try:
                # 到達した長さには、段階の狙いではなく、対象サーバーが数えた長さを残す
                abort_if_context_limit(
                    cond,
                    record,
                    reached_input_tokens=fitted.input_tokens,
                )
            except ConditionAborted:
                self._remember_limit(ctx.run_id, length)
                raise

    # --- 計測ランごとの状態 (上限と、包みの計測の結果) ---

    def _forget_run_state(self) -> None:
        self._limit_run_id = None
        self._limit_tokens = None
        self._frame.forget()

    def _remember_limit(self, run_id: str, length: int) -> None:
        self._limit_run_id = run_id
        self._limit_tokens = length

    def _abort_if_limit_already_reached(
        self, ctx: SuiteContext, cond: ConditionPlan, length: int
    ) -> None:
        """すでに上限に当たっていれば、この段階には 1 つも送らずに打ち切る (6.9)。"""
        reached = self._limit_tokens
        if self._limit_run_id != ctx.run_id or reached is None or length < reached:
            return
        raise ConditionAborted(
            skip_condition(
                self.name,
                cond.key,
                "すでに入力の長さの上限に達しているので、この段階には送らなかった。"
                f"到達した長さ: {reached} トークン (段階の狙い)",
            )
        )

    def _abort_if_the_built_conversation_does_not_fit(
        self,
        ctx: SuiteContext,
        cond: ConditionPlan,
        length: int,
        input_tokens: int,
    ) -> None:
        """組み立てた会話の実際の長さで、送る前にもう一度だけ確かめる (6.9)。

        issue #24 以降、ここで比べるのは対象サーバーが数えた長さ (`input_tokens`。
        包みと履歴と最後の 1 手を含む会話全体)。見積もり (`approx_tokens`) は
        数えた長さに置き換わっているので、そのまま比べる。
        """
        if fits_context(ctx.context_limit, input_tokens, cond.max_tokens) is not False:
            return
        self._remember_limit(ctx.run_id, length)
        raise ConditionAborted(
            skip_condition(
                self.name,
                cond.key,
                "組み立てた会話が入力の長さの上限を超えるので、送る前に止めた。"
                f"到達した長さ: {input_tokens} トークン (対象サーバーが数えた、包みを含む会話) + "
                f"出力の上限 {cond.max_tokens} トークン > 上限 {ctx.context_limit} トークン",
            )
        )


SUITE: Final[AgentSuite] = AgentSuite()
"""runner (3.5) が読み込む、この module のまとまりの実体。"""


# --- 段階 ---------------------------------------------------------------------


def stage_condition_key(tokens: int) -> str:
    """段階の条件の鍵 (`20000` → `agent/stage/020k`)。

    千で割り切れる長さは、千トークンを 3 桁に 0 埋めして `k` を付ける。
    割り切れない長さは、数をそのまま書く。どちらも、要約が鍵の末尾から段階の
    長さを読み戻せる形である (module の docstring「段階」を参照)。
    """
    if tokens < 1:
        raise ValueError(f"stage_condition_key: tokens は 1 以上 (tokens={tokens})")
    label = f"{tokens // 1000:0{_LABEL_DIGITS}d}k" if tokens % 1000 == 0 else str(tokens)
    return condition_key(SuiteName.AGENT, _STAGE_PART, label)


def stage_lengths(settings: AgentSettings) -> tuple[int, ...]:
    """段階の長さの並び (`start_tokens` から `end_tokens` まで `step_tokens` 刻み)。

    `end_tokens` が刻みに乗らないときは、超えない最後の段階で止める。
    `AgentSettings` が `start_tokens <= end_tokens` を保証するので、必ず
    1 つ以上になる。
    """
    lengths: list[int] = []
    length = settings.start_tokens
    while length <= settings.end_tokens:
        lengths.append(length)
        length += settings.step_tokens
    return tuple(lengths)


def _stage_ordinal(settings: AgentSettings, length: int) -> int:
    """段階の長さから、その段階の番号 (0 から) を求める。"""
    try:
        return stage_lengths(settings).index(length)
    except ValueError as exc:
        raise ValueError(
            f"agent: 設定にない段階の長さ (target_input_tokens={length}, "
            f"段階={stage_lengths(settings)})"
        ) from exc


def _task_index(stage_ordinal: int, trial_index: int) -> int:
    """課題の番号 (module の docstring「最後の 1 手」を参照)。"""
    return stage_ordinal * _TASK_INDEX_STRIDE + trial_index


def _check_trials_fit_the_index_stride(settings: AgentSettings) -> None:
    """課題の番号が段階をまたいでぶつからないことを、計画の前に確かめる。"""
    if settings.trials_per_stage > _TASK_INDEX_STRIDE:
        raise ValueError(
            f"agent: 1 段階の試行の数が多すぎる (trials_per_stage="
            f"{settings.trials_per_stage}, 上限={_TASK_INDEX_STRIDE})"
        )


# --- 会話の割り当て -----------------------------------------------------------


def conversation_sizes(trials: int, conversations: int) -> tuple[int, ...]:
    """会話ごとの試行の数 (均等に割り、余りは先頭の会話から 1 つずつ)。"""
    if conversations < 1:
        raise ValueError(f"agent: conversations は 1 以上 (conversations={conversations})")
    base, remainder = divmod(trials, conversations)
    return tuple(base + (1 if index < remainder else 0) for index in range(conversations))


def conversation_for(trial_index: int, sizes: Sequence[int]) -> int:
    """試行の番号から、使う会話の番号を求める (かたまりごとの割り当て)。"""
    remaining = trial_index
    for index, size in enumerate(sizes):
        if remaining < size:
            return index
        remaining -= size
    raise ValueError(f"agent: 試行の番号が会話の割り当てに収まらない (trial_index={trial_index})")


async def _fit_conversation(
    ctx: SuiteContext,
    cond: ConditionPlan,
    length: int,
    conversation_index: int,
    *,
    frame_tokens: int,
    final_turn: InputMessage,
) -> _FittedConversation:
    """1 本の会話を、対象サーバーに数えさせて狙いに合わせ込む (issue #24)。

    `chars_per_token` は必ず `Profile` の値を渡す (省くと `Profile` の初期値に
    なり、`profiles.toml` の較正した比と食い違う。注 7.1 → 7.2)。`frame_tokens`
    は、計測ランで 1 回数えた包みのトークン数 (`_measure_frame`)。

    `scale=1.0` で組み立てた会話 (`system` + `tools` + 履歴 + `final_turn`) を
    1 つの要求として数えさせ、実測と見積もりの比で止める手番を組み直す。手番の
    数が変わらなくなったら収束として止める (上限 `_MAX_LENGTH_COUNTS` 回)。訪れた
    候補のうち、数えた長さが狙いに最も近いものを返す。手番の中身は比によって
    変わらないので、`GENERATOR_VERSION` は上がらない。

    数える要求が、対象サーバーに入力の長さの上限を超えるものとして断られたとき
    (`InputOverContextLimitError`。口がなく、代わりの `max_tokens=1` の要求が
    HTTP 400 になった場合) だけは、`ProbeError` のまま伝えずに扱いを分ける (6.9)。

    - 1 本目の候補 (数え終えた候補がまだない) なら、その段階には 1 件も送らずに
      `ConditionAborted` を投げる。到達した長さには、数えられなかった候補の
      見積もり (`approx_tokens`) を、見積もりだとわかる書き方で入れる
    - 2 本目以降の候補なら、数え終えた候補のうち狙いに最も近いものを返す
      (反復の上限に達したときと同じ規則。数えた長さのある候補だけを返す)

    それ以外の数えられない理由の `ProbeError` は捕まえない (黙って見積もりに
    戻さない。runner が計測ランを `aborted` にして止める)。数える要求は `put_body`
    には保存されず、その条件の `/metrics` の増分にだけ入る (`measure_frame_tokens`)。
    """
    scale = 1.0
    best: _FittedConversation | None = None
    seen: set[int] = set()
    for _ in range(_MAX_LENGTH_COUNTS):
        prefix = build_conversation(
            length,
            derive_conversation_seed(ctx.profile.seed, conversation_index),
            chars_per_token=ctx.profile.chars_per_token,
            fixed_tokens=frame_tokens,
            history_scale=scale,
        )
        try:
            counted = await measure_frame_tokens(
                ctx,
                system=prefix.system,
                tools=prefix.tools,
                messages=[*prefix.messages, final_turn],
            )
        except InputOverContextLimitError as exc:
            if best is not None:
                break  # 数え終えた候補で止める。数えられなかった候補は送らない
            raise ConditionAborted(
                skip_condition(
                    cond.suite,
                    cond.key,
                    "組み立てた会話の計測が、対象サーバーに入力の長さの上限を超えるもの "
                    "(HTTP 400) として断られたので、この段階には送らなかった。"
                    f"到達した長さ: {prefix.approx_tokens} トークン "
                    "(文字数からの見積もり。対象サーバーは数えていない)。"
                    f"数える要求の結果: {exc}",
                )
            ) from exc
        if best is None or abs(counted - length) < abs(best.input_tokens - length):
            best = _FittedConversation(prefix, counted)

        rounds = len(prefix.messages) // MESSAGES_PER_ROUND
        if rounds in seen:
            break  # 手番の数が変わらなくなった。収束として止める
        seen.add(rounds)
        history = (prefix.approx_tokens - frame_tokens) / scale
        if history <= 0 or counted <= frame_tokens:
            break  # 比を作れない。訪れた候補で止める
        scale = (counted - frame_tokens) / history

    assert best is not None, "_fit_conversation は 1 回目で必ず数える"
    return best


# --- 最後の 1 手 ---------------------------------------------------------------


def _final_turn(prompt: str) -> InputMessage:
    """会話の最後に足す、課題の指示 1 つぶんの `user` の発話。"""
    return InputMessage(role="user", content=[TextBlockParam(text=prompt)])


def _verdict_for(ctx: SuiteContext, task: ToolTask) -> VerdictFn:
    """応答を 9 種類に分ける関数 (`run_trial` が、失敗した試行にも呼ぶ)。

    記法の目印は、対象サーバーの定義から受け取る (6.3)。課題の側が壊れて
    いれば `ToolTaskError` が外に出る (握りつぶさない。注 6.3)。
    """
    markers = list(ctx.target.tool_markup_markers)

    def verdict(result: StreamResult) -> ToolCallVerdict:
        return classify_tool_call(result, task, markers)

    return verdict
