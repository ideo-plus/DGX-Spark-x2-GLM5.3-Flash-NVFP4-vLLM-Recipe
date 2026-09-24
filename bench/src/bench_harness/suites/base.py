"""測る項目のまとまりに共通の約束事 (task 3.1: suites/base)。

5 つのまとまり (decode 3.2、prefill 3.3、concurrency 3.4、quality 6.7、
agent 7.2) は、この module の上に薄く載る。まとまりが自分で決めるのは
「どの条件を測るか」と「どんな入力を作るか」だけで、それ以外 (要求の組み立て、
送った本文の保存、試行のレコードの組み立て、共通の印、上限に当たったときの
打ち切り、慣らしの決まり、種と先頭の識別子の作り方) は、すべてここにある。
まとまりどうしは互いを読み込まず、この module だけを共有する (design.md の
「1 つのファイルに 1 つの責任」)。

依存の向き (design.md) は `types` → `config` → `corpus`/`scoring`/`client`/
`metrics`/`store` → `suites` → `runner` → `cli`。この module は `runner`、
`analysis`、`cli` を読み込まない。保存の部品 (`store`) も直接は読み込まず、
本文を保存する口を `SuiteContext.put_body` として受け取る (計測ランの進行
(3.5) が `RunStore.put_body` を渡す)。入力の長さを数える口も同じ形で、
`SuiteContext.count_input_tokens` として受け取る (issue #9: 包みの計測。
既定は `client.probe.count_input_tokens` を呼ぶ `make_suite_context` が作る)。

## 使い方 (まとまりの書き方)

```python
class DecodeSuite:
    name = SuiteName.DECODE

    def plan(self, ctx: SuiteContext) -> list[ConditionPlan | SkippedCondition]:
        return [
            plan_condition(
                ctx,
                suite=self.name,
                key=condition_key(self.name, kind, lang),
                trials=ctx.profile.decode.trials,
                warmup_trials=ctx.profile.decode.warmup_trials,
                max_tokens=ctx.profile.decode.max_tokens,
            )
            for kind in ("code", "prose")
            for lang in ("en", "ja")
        ]

    async def run_condition(
        self, ctx: SuiteContext, cond: ConditionPlan
    ) -> AsyncIterator[TrialRecord]:
        for trial_index, warmup in iter_trials(cond):
            nonce = cold_prefix_nonce(ctx, cond, trial_index=trial_index, warmup=warmup)
            seed = trial_seed(ctx.profile.seed, cond.key, trial_index, warmup=warmup)
            record = await run_trial(
                ctx,
                cond,
                trial_index=trial_index,
                warmup=warmup,
                system=system_with_prefix(nonce, "長く書くこと。"),
                messages=single_user_message(self._instruction(seed)),
                measures_decode_speed=True,
                expect_full_output=True,
            )
            yield record
            abort_if_context_limit(cond, record)
```

## 決めごと

- **同じ条件のすべての試行が、同じサンプリングと出力の上限を使う (2.7)**。
  `plan_condition` は `Profile.sampling` を必ず計画に写し、`run_trial` は
  サンプリングと `max_tokens` を**計画からしか**読まない。`run_trial` には、
  試行ごとにそれらを変える引数がない。まとまりが、計画を必ず `plan_condition`
  で作り、1 つの条件に 1 つの計画を使う限り、この決まりは破れない (計画を自前で
  組み立てて、試行ごとに差し替えることまでは、防いでいない)
- **慣らしの試行 (2.5)**。`iter_trials` は、慣らしを先に `warmup=True` で
  `0..warmup_trials-1`、続けて本番を `warmup=False` で `0..trials-1` と
  返す。試行の番号は慣らしと本番で別々に 0 から数え、`(条件, warmup,
  試行の番号)` の 3 つ組で一意になる。本番の `i` 番目は、計測ランをまたいでも
  同じ入力になる (`trial_seed` と `cold_prefix_nonce` が `run_id` 以外の
  同じ部品から決まるため)。対応のある比較 (9.2) は、これに依存する。
  慣らしと本番で `trial_index` が重なっても入力が重ならないように、種と
  識別子の部品には `warmup` が入っている (3.4「それまでに送ったどの入力とも
  先頭が重ならない」)。条件の最初の要求は、接続の確立を最初のトークンまでの
  時間に混ぜないための慣らしでもある (Implementation Notes 2.1)。設定に
  慣らしの回数がないまとまり (同時処理) は、`DEFAULT_CONNECTION_WARMUP_TRIALS`
  を使う
- **共通の印**。`run_trial` が、すべての試行に次の印を付ける。
  - `LENGTH_OFF_TARGET` (3.3): 計画に狙いの入力の長さがあり、対象サーバーが
    トークン数を返していて、`|実際 − 狙い| ÷ 狙い > Profile.length_tolerance`。
    境界 (ちょうど許容の幅) には**付けない** (`>` で比べる)。これは入力の
    長さの事実なので、要求が失敗した試行でも、トークン数が返っていれば付ける
  - `TOO_FEW_OUTPUT_TOKENS`: 要求が成功し、呼び出し側が
    `measures_decode_speed=True` と言った試行で、出力のトークンが
    `MIN_SPEED_OUTPUT_TOKENS` (16) 未満。生成速度の分母が小さすぎる試行を
    集計から外すための印なので (design.md「速さの定義」)、生成速度を測って
    いない試行 (出力の上限が 16 の prefill など) には付けない。既定は `False`
  - `SHORT_OUTPUT` (2.6): `expect_full_output=True` の試行で、要求が成功し、
    `stop_reason` が `max_tokens` でない (`None` を含む)
  - `REPLACEMENT_CHAR` / `REPETITION_LOOP` (10.7): 要求が成功した試行の本文
    (thinking を含む) を `detect_output_anomalies` にかける。ツールの引数は
    `collect_tool_input_text` で取り出し、置き換え文字の検査にだけ使う
    (JSON は構造が繰り返すので、繰り返しの検査にはかけない。注 2.6)
  出力の印は、要求が失敗した試行には付けない (失敗は集計に入らないので、
  印を足しても数えるものが増えるだけになる)。印は `TrialFlag` の定義順に
  並べ、同じ印を 2 つ入れない
- **上限の扱い (3.6、6.9)**。上限がわかっているときは `plan_condition` が
  送る前に飛ばして `SkippedCondition` を返す。上限が不明なときは送ってみて、
  対象サーバーが HTTP 400 を返したら `abort_if_context_limit` が
  `ConditionAborted` を投げる。レコードは投げる前に必ず作られている (8.1:
  すべての要求を残す) ので、まとまりはレコードを `yield` したあとに呼ぶ

## 計測ランの進行 (3.5) への契約

- `plan()` が返す `SkippedCondition` は、そのまま `manifest.skipped` に足す
- `run_condition()` から `ConditionAborted` が出たら、`exc.skipped` を
  `manifest.skipped` に足し、**その条件だけ**を終わりにして、次の条件に進む。
  上限の超過や、送る前にわかる条件の不成立 (同時処理の狙いが包み以下、issue #9)
  は「連続の失敗」に数えない (design.md Error Handling)
- `ConditionAborted` が出る前に `yield` されたレコードは、ふつうのレコードと
  同じように書く (400 の試行も 8.1 の対象)
- `ConditionAborted` は、1 つもレコードを `yield` せずに出ることもある。段階を
  伸ばしていくまとまり (7.2) は、一度上限に当たったら、そのあとの段階では
  要求を送らずに `ConditionAborted` を投げてよい (まとまりの実体が、当たった
  ことを覚えておく)
- `SuiteContext` は `make_suite_context` で作る。`put_body` には
  `RunStore.put_body` をそのまま渡す

依存するのは標準ライブラリと pydantic、`bench_harness` の `types`、`client`、
`corpus`、`scoring` だけ。
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from pydantic import JsonValue, SecretStr

from bench_harness.client.messages import MessagesClient, build_request_body
from bench_harness.client.probe import (
    TokenCount,
    fits_context,
    is_context_limit_error,
)
from bench_harness.client.probe import (
    count_input_tokens as probe_count_input_tokens,
)
from bench_harness.corpus.synth import SyntheticCorpus, TemplateCorpus, prefix_header_line
from bench_harness.scoring.sanity import (
    collect_text,
    collect_tool_input_text,
    detect_output_anomalies,
)
from bench_harness.types import (
    ConditionPlan,
    InputMessage,
    MessagesRequest,
    Profile,
    QualityVerdict,
    SkippedCondition,
    StreamResult,
    SuiteName,
    TargetDef,
    TextBlockParam,
    Tier,
    ToolCallVerdict,
    ToolDef,
    TrialFlag,
    TrialRecord,
)

__all__ = [
    "DEFAULT_CONNECTION_WARMUP_TRIALS",
    "MIN_SPEED_OUTPUT_TOKENS",
    "ConditionAborted",
    "CountInputTokens",
    "FrameTokensPerRun",
    "PutBody",
    "StartGate",
    "Suite",
    "SuiteContext",
    "VerdictFn",
    "abort_if_context_limit",
    "cold_prefix_nonce",
    "condition_key",
    "iter_trials",
    "make_suite_context",
    "measure_frame_tokens",
    "plan_condition",
    "run_trial",
    "single_user_message",
    "skip_condition",
    "system_with_prefix",
    "tokens_label",
    "trial_seed",
    "warm_prefix_nonce",
]

MIN_SPEED_OUTPUT_TOKENS: Final[int] = 16
"""これ未満の出力のトークン数は、生成速度の集計に入れない (design.md「速さの定義」)。"""

DEFAULT_CONNECTION_WARMUP_TRIALS: Final[int] = 1
"""慣らしの回数を設定に持たないまとまりが使う既定。

接続の確立 (TCP) を最初のトークンまでの時間に混ぜないため、条件の最初に
1 回は捨てる要求を送る (Implementation Notes 2.1)。
"""

MAX_REASON_CHARS: Final[int] = 500
"""飛ばした理由に残す文字数の上限 (生データが、長いエラーの本文で膨らまないように)。"""

_REPLACEMENT_CHAR: Final[str] = "�"
"""置き換え文字。`scoring/sanity` が本文に使うのと同じ文字。

ツールの引数には `detect_output_anomalies` をかけない (注 2.6: JSON は構造が
繰り返すので、繰り返しの検査に向かない)。置き換え文字だけをここで見る。
"""

_FLAG_ORDER: Final[dict[TrialFlag, int]] = {flag: order for order, flag in enumerate(TrialFlag)}
"""印の並びを、`TrialFlag` の定義順に固定する (レコードを見比べやすくする)。"""

_WARMUP_PART: Final[str] = "warmup"
_MEASURED_PART: Final[str] = "measured"
_NO_STREAM_PART: Final[int] = -1
"""ストリームの番号がないときの、識別子の部品 (番号 0 と区別する)。"""


PutBody = Callable[[dict[str, JsonValue]], str]
"""実際に送った本文を保存して、参照を返す口 (`RunStore.put_body`)。"""

VerdictFn = Callable[[StreamResult], ToolCallVerdict | QualityVerdict | None]
"""応答から判定を作る、まとまりごとの関数 (6.7、7.2 が渡す)。"""

StartGate = Callable[[], Awaitable[None]]
"""送り始める直前に待つ合図 (3.4 が、同時に送り始めるために渡す)。"""

CountInputTokens = Callable[[MessagesRequest], Awaitable[TokenCount]]
"""要求の入力のトークン数を、対象サーバーに数えさせる口 (issue #9)。

既定は `client.probe.count_input_tokens` を呼ぶ関数で、`make_suite_context` が
`api_key` を閉じ込めて作る。まとまりは `SuiteContext.count_input_tokens` 越しに
`measure_frame_tokens` を呼ぶ (probe の規約はそのまま: 口がなければ
`max_tokens=1` の要求で代え、それも失敗したら `ProbeError`)。
"""


# --- まとまりの文脈 -------------------------------------------------------


@dataclass(frozen=True)
class SuiteContext:
    """まとまりの実行に要るものを 1 つにまとめた入れ物 (design.md suites)。

    `put_body` は保存の部品への唯一の口で、まとまりは `store` を知らない。
    `corpus` は `make_suite_context` が `profile.chars_per_token` から作る。
    """

    client: MessagesClient
    profile: Profile
    target: TargetDef
    run_id: str
    put_body: PutBody
    corpus: SyntheticCorpus
    count_input_tokens: CountInputTokens
    context_limit: int | None = None

    def __post_init__(self) -> None:
        """外から来た値 (対象サーバーが申告した上限など) の範囲を確かめる (注 2.7)。"""
        if not self.run_id:
            raise ValueError("SuiteContext: run_id は空にできない")
        if self.context_limit is not None and self.context_limit < 1:
            raise ValueError(
                f"SuiteContext: context_limit は 1 以上か None (context_limit={self.context_limit})"
            )
        if self.profile.sampling.thinking != "server_default":
            # 設定の型は `server_default` しか許さない (task 8.4)。ここに来るのは、検証を
            # 通らない作り方 (`model_copy(update=…)` など) をしたときだけ。実行の条件と、
            # 実際に送る本文が食い違うので、送る前に止める
            raise ValueError(
                "SuiteContext: thinking は、対象サーバーの既定 (server_default) しか選べない。"
                f"sampling.thinking={self.profile.sampling.thinking!r}"
            )


def make_suite_context(
    *,
    client: MessagesClient,
    profile: Profile,
    target: TargetDef,
    run_id: str,
    put_body: PutBody,
    context_limit: int | None = None,
    corpus: SyntheticCorpus | None = None,
    api_key: SecretStr | None = None,
    count_input_tokens: CountInputTokens | None = None,
) -> SuiteContext:
    """`SuiteContext` を作る (3.5 と試験が使う)。

    `corpus` を渡さなければ、`profile.chars_per_token` から `TemplateCorpus` を
    作る。`context_limit` は `preflight` が返した値 (わからなければ `None`)。

    `count_input_tokens` を渡さなければ、`client.probe.count_input_tokens` を
    `api_key` で閉じた既定の口を作る (issue #9: 包みの計測)。`api_key` はこの
    既定の口だけが使い、`SuiteContext` そのものには残さない (秘密を下位層に
    持たせない。`runner._build_context` が渡す)。
    """
    if count_input_tokens is None:

        async def count_with_probe(request: MessagesRequest) -> TokenCount:
            return await probe_count_input_tokens(
                client, target, request, timeout=profile.timeout, api_key=api_key
            )

        count_input_tokens = count_with_probe

    return SuiteContext(
        client=client,
        profile=profile,
        target=target,
        run_id=run_id,
        put_body=put_body,
        corpus=corpus if corpus is not None else TemplateCorpus(profile.chars_per_token),
        count_input_tokens=count_input_tokens,
        context_limit=context_limit,
    )


async def measure_frame_tokens(
    ctx: SuiteContext,
    *,
    messages: Sequence[InputMessage],
    system: str | None = None,
    tools: Sequence[ToolDef] | None = None,
) -> int:
    """要求の包み (本文から文章・履歴を抜いた、決まった分量) を 1 回数える (issue #9)。

    `messages`、`system`、`tools` をそのまま 1 つの要求にまとめ、対象サーバーに
    入力のトークン数を数えさせる。数えるのは `count_tokens` の口 (なければ
    `max_tokens=1` の要求 1 件) で、probe の規約 (`client.probe.count_input_tokens`)
    をそのまま使う。口がない対象で代わりの要求を送ったときは、その要求は試行と
    しては保存されず (`put_body` を呼ばない)、その条件の `/metrics` の増分にだけ
    入る。

    文字数から比で見積もれない固定の分量 (チャットテンプレートの包み、識別子の
    割れ方、ツール定義の展開) があるので、比ではなく対象サーバーに数えさせる。

    `ProbeError` は捕まえない。黙って文字数の見積もりに戻すと、対象サーバーが
    数えられないまま、狙いの長さが静かに外れる (今の probe の規約に合わせる)。
    """
    request = MessagesRequest(
        model=ctx.target.model,
        max_tokens=1,
        messages=list(messages),
        system=system,
        tools=list(tools) if tools is not None else None,
    )
    return (await ctx.count_input_tokens(request)).tokens


class FrameTokensPerRun:
    """要求の包みのトークン数を、計測ランにつき 1 回だけ数えて覚える (issue #9)。

    測る中身 (どんな要求を数えさせるか) は、まとまりが `measure` として渡す。
    この部品が持つのは規則だけである。

    - 同じ `run_id` では、覚えた値を返し、数え直さない (測る要求の数を最小にする)
    - `run_id` が変われば、数え直す
    - `forget()` で消す (まとまりは `plan()` の最初に呼ぶ)

    `measure` が投げた例外 (`ProbeError` など) は、そのまま伝播し、何も覚えない。
    """

    def __init__(self, measure: Callable[[SuiteContext], Awaitable[int]]) -> None:
        self._measure = measure
        self._remembered: tuple[str, int] | None = None

    def forget(self) -> None:
        """覚えた値を消す。"""
        self._remembered = None

    async def tokens_for(self, ctx: SuiteContext) -> int:
        """`ctx.run_id` の計測ランの包みのトークン数を返す (なければ数えて覚える)。"""
        if self._remembered is not None and self._remembered[0] == ctx.run_id:
            return self._remembered[1]
        tokens = await self._measure(ctx)
        self._remembered = (ctx.run_id, tokens)
        return tokens


@runtime_checkable
class Suite(Protocol):
    """測る項目のまとまりの約束事 (design.md suites)。

    `plan` は条件の一覧 (飛ばしたものを含む) を返し、`run_condition` は 1 つの
    条件の試行を、終わった順に返す。統計はしない (分析の仕事)。
    """

    name: SuiteName

    def plan(self, ctx: SuiteContext) -> list[ConditionPlan | SkippedCondition]: ...

    def run_condition(
        self, ctx: SuiteContext, cond: ConditionPlan
    ) -> AsyncIterator[TrialRecord]: ...


class ConditionAborted(Exception):
    """その条件を、送れない、または続けられないので打ち切ること (3.6、6.9)。

    上限の超過 (HTTP 400、組み立てた会話が上限を超える) のほか、送る前にわかる
    設定と対象サーバーの組み合わせの不成立 (同時処理の狙いが包み以下、issue #9) や、
    隔離の実行が続けて失敗したとき (品質の検査) でも使う。

    `skipped` に、飛ばした条件と理由が入る。計測ランの進行 (3.5) は、これを
    捕まえて `manifest.skipped` に足し、連続の失敗には数えず、次の条件に進む。
    """

    def __init__(self, skipped: SkippedCondition) -> None:
        super().__init__(skipped.reason)
        self.skipped = skipped


# --- 条件の鍵と、飛ばす条件 -----------------------------------------------


def condition_key(suite: SuiteName, *parts: str) -> str:
    """条件の鍵を作る (例: `condition_key(SuiteName.PREFILL, "cold", "8k")`)。

    まとまりの名前を先頭に置き、`/` でつなぐ (design.md の例に合わせる)。
    """
    for part in parts:
        if not part or "/" in part:
            raise ValueError(f"condition_key: parts は空でなく、/ を含まない (parts={parts!r})")
    return "/".join([suite.value, *parts])


def tokens_label(tokens: int) -> str:
    """条件の鍵に入れる、トークン数の短い表し方 (8000 → `8k`)。

    千で割り切れないときは、そのままの数を返す (鍵が一意であり続けるように)。
    """
    if tokens < 1:
        raise ValueError(f"tokens_label: tokens は 1 以上 (tokens={tokens})")
    if tokens >= 1000 and tokens % 1000 == 0:
        return f"{tokens // 1000}k"
    return str(tokens)


def skip_condition(suite: SuiteName, key: str, reason: str) -> SkippedCondition:
    """飛ばした条件を作る。理由は決まった長さに切り詰める。"""
    return SkippedCondition(suite=suite, key=key, reason=_bounded(reason))


def _bounded(text: str) -> str:
    """理由を 1 行の決まった長さにする (対象サーバーの本文を、そのまま溜めない)。"""
    collapsed = " ".join(text.split())
    if len(collapsed) <= MAX_REASON_CHARS:
        return collapsed
    return collapsed[:MAX_REASON_CHARS] + "…"


def plan_condition(
    ctx: SuiteContext,
    *,
    suite: SuiteName,
    key: str,
    trials: int,
    max_tokens: int,
    warmup_trials: int = DEFAULT_CONNECTION_WARMUP_TRIALS,
    tier: Tier = "primary",
    concurrency: int = 1,
    target_input_tokens: int | None = None,
) -> ConditionPlan | SkippedCondition:
    """1 つの条件を計画する。上限に収まらなければ、送る前に飛ばす (3.6)。

    サンプリングは必ず `ctx.profile.sampling` から取る (2.7)。まとまりの側から
    は渡せない。`target_input_tokens` を渡した条件だけが、上限の判定と
    `LENGTH_OFF_TARGET` の対象になる。上限がわからない (`None`) ときは、ふつう
    に計画して、送ってから `abort_if_context_limit` で扱う。
    """
    if not key:
        raise ValueError("plan_condition: key は空にできない")
    if trials < 0:
        raise ValueError(f"plan_condition: trials は 0 以上 (key={key}, trials={trials})")
    if warmup_trials < 0:
        raise ValueError(
            f"plan_condition: warmup_trials は 0 以上 (key={key}, warmup_trials={warmup_trials})"
        )
    if max_tokens < 1:
        raise ValueError(f"plan_condition: max_tokens は 1 以上 (key={key}, {max_tokens=})")
    if concurrency < 1:
        raise ValueError(f"plan_condition: concurrency は 1 以上 (key={key}, {concurrency=})")
    if target_input_tokens is not None and target_input_tokens < 1:
        raise ValueError(
            f"plan_condition: target_input_tokens は 1 以上か None "
            f"(key={key}, target_input_tokens={target_input_tokens})"
        )

    if (
        target_input_tokens is not None
        and fits_context(ctx.context_limit, target_input_tokens, max_tokens) is False
    ):
        return skip_condition(
            suite,
            key,
            "入力の長さの上限を超えるので、送る前に飛ばした "
            f"(狙いの入力 {target_input_tokens} トークン + 出力の上限 {max_tokens} トークン > "
            f"上限 {ctx.context_limit} トークン)",
        )

    return ConditionPlan(
        suite=suite,
        key=key,
        tier=tier,
        trials=trials,
        warmup_trials=warmup_trials,
        sampling=ctx.profile.sampling,
        max_tokens=max_tokens,
        concurrency=concurrency,
        target_input_tokens=target_input_tokens,
    )


# --- 試行の番号、種、先頭の識別子 -----------------------------------------


def iter_trials(cond: ConditionPlan) -> Iterator[tuple[int, bool]]:
    """`(試行の番号, 慣らしかどうか)` を、慣らしを先に返す (2.5)。

    番号は慣らしと本番で別々に 0 から数える。`(条件, warmup, 番号)` の 3 つ組
    が一意で、本番の `i` 番目は計測ランをまたいでも同じ入力になる (この module
    の docstring「慣らしの試行」を参照)。
    """
    for index in range(cond.warmup_trials):
        yield index, True
    for index in range(cond.trials):
        yield index, False


def _encode_part(part: str | int) -> bytes:
    """種の部品を、型と長さを含めて一意に符号化する (corpus の `prefix_nonce` と同じ形)。"""
    if isinstance(part, str):
        data = part.encode("utf-8")
        return b"s" + str(len(data)).encode("ascii") + b":" + data
    data = str(part).encode("ascii")
    return b"i" + str(len(data)).encode("ascii") + b":" + data


def trial_seed(
    seed: int,
    condition_key: str,
    trial_index: int,
    *,
    warmup: bool = False,
    stream_index: int | None = None,
) -> int:
    """試行ごとの種を、`Profile.seed` と条件の鍵と試行の番号から決める。

    `hash()` を使わない (`PYTHONHASHSEED` で結果が変わるため)。同じ引数なら、
    プロセスをまたいでも同じ値を返す。慣らしと本番は、番号が同じでも違う種に
    なる (慣らしと同じ入力を、本番で送らないため)。
    """
    parts = _nonce_parts(seed, condition_key, trial_index, warmup, stream_index)
    digest = hashlib.sha256(b"".join(_encode_part(part) for part in parts)).digest()
    return int.from_bytes(digest[:8], "big") >> 1  # 63 bit (符号なしの正の整数)


def _nonce_parts(
    seed: int, condition_key: str, trial_index: int, warmup: bool, stream_index: int | None
) -> tuple[str | int, ...]:
    """試行を一意に表す部品の並び (種と、効かない条件の識別子で共通に使う)。"""
    if stream_index is not None and stream_index < 0:
        # 負の番号は、「番号なし」の部品 (-1) と区別できなくなる
        raise ValueError(f"stream_index は 0 以上である必要がある (stream_index={stream_index})")
    return (
        seed,
        condition_key,
        _WARMUP_PART if warmup else _MEASURED_PART,
        trial_index,
        _NO_STREAM_PART if stream_index is None else stream_index,
    )


def cold_prefix_nonce(
    ctx: SuiteContext,
    cond: ConditionPlan,
    *,
    trial_index: int,
    warmup: bool = False,
    stream_index: int | None = None,
) -> str:
    """キャッシュが効かない条件の、先頭の識別子 (3.4)。

    `(Profile.seed, 条件の鍵, 慣らしかどうか, 試行の番号, ストリームの番号,
    run_id)` から作る (注 2.5)。試行ごと、条件ごと、ストリームごと、計測ラン
    ごとに違うので、それまでに送ったどの入力とも先頭が重ならない。
    """
    parts = _nonce_parts(ctx.profile.seed, cond.key, trial_index, warmup, stream_index)
    return ctx.corpus.prefix_nonce(*parts, ctx.run_id)


def warm_prefix_nonce(ctx: SuiteContext, cond: ConditionPlan) -> str:
    """キャッシュが効く条件の、先頭の識別子 (3.5)。

    `(Profile.seed, 条件の鍵)` だけから作るので、試行をまたいでも、計測ランを
    またいでも同じになる (同じ前置きを送り続けて、キャッシュに当てる)。
    """
    return ctx.corpus.prefix_nonce(ctx.profile.seed, cond.key)


def system_with_prefix(nonce: str, *lines: str) -> str:
    """システムプロンプトを組み立てる。1 行目は、先頭の識別子の行 (注 2.5)。

    この行の長さは識別子の長さだけで決まるので、効く条件と効かない条件とで
    入力のトークン数が変わらない。
    """
    return "\n".join([prefix_header_line(nonce), *lines])


def single_user_message(text: str) -> list[InputMessage]:
    """本文のブロック 1 つだけを持つ、`user` の発話 1 つ。"""
    return [InputMessage(role="user", content=[TextBlockParam(text=text)])]


# --- 1 つの試行を実行する -------------------------------------------------


async def run_trial(
    ctx: SuiteContext,
    cond: ConditionPlan,
    *,
    trial_index: int,
    messages: Sequence[InputMessage],
    warmup: bool = False,
    system: str | None = None,
    tools: Sequence[ToolDef] | None = None,
    round_id: int | None = None,
    stream_index: int | None = None,
    measures_decode_speed: bool = False,
    expect_full_output: bool = False,
    verdict: VerdictFn | None = None,
    extra: Mapping[str, JsonValue] | None = None,
    start_gate: StartGate | None = None,
) -> TrialRecord:
    """1 つの試行を送り、共通の印を付けた `TrialRecord` を返す。

    サンプリングと出力の上限は `cond` からしか読まない (2.7)。送る本文は
    `build_request_body` で組み立て、`ctx.put_body` で保存してから送る
    (保存するのは、実際に送った形。注 1.2、2.3)。

    要求の失敗では投げない (クライアントが値として返す)。投げるのは、呼び出し
    の誤り (`extra` が型のある項目と重なる、`messages` が空、`trial_index` が
    負) とキャンセルだけ。

    引数:

    - `measures_decode_speed`: この試行で生成速度を測るか。`True` のときだけ
      `TOO_FEW_OUTPUT_TOKENS` を付ける。decode (3.2) と concurrency (3.4) が
      `True`、出力の上限が小さい prefill (3.3) や、採点が目的の quality (6.7)
      と agent (7.2) は `False` (既定)
    - `expect_full_output`: 出力の上限まで書かせるつもりか。`True` のときだけ
      `SHORT_OUTPUT` を付ける (2.6)
    - `verdict`: 応答から判定を作る関数 (6.7、7.2)。要求が失敗した試行でも
      呼ぶ (分類の 1 つが「要求そのものの失敗」であるため)
    - `start_gate`: 本文を保存し終えたあと、送り始める直前に待つ合図。同時に
      送る条件 (3.4) が、本文の保存で送り始めの時刻がずれないように使う
    - `round_id` / `stream_index`: 同時処理の 1 回ぶんと、その中の何本目か

    同じ条件の試行を、同時に (`asyncio.gather` で) 呼んでもよい。
    """
    if trial_index < 0:
        raise ValueError(f"run_trial: trial_index は 0 以上 (trial_index={trial_index})")
    if not messages:
        raise ValueError("run_trial: messages は空にできない")

    request = MessagesRequest(
        model=ctx.target.model,
        max_tokens=cond.max_tokens,
        messages=list(messages),
        system=system,
        tools=list(tools) if tools is not None else None,
        temperature=cond.sampling.temperature,
        top_p=cond.sampling.top_p,
        top_k=cond.sampling.top_k,
        extra=dict(extra) if extra is not None else {},
    )
    body_ref = ctx.put_body(build_request_body(request))
    if start_gate is not None:
        await start_gate()
    result = await ctx.client.stream(request, ctx.profile.timeout)

    return TrialRecord(
        run_id=ctx.run_id,
        suite=cond.suite,
        condition=cond.key,
        tier=cond.tier,
        trial_index=trial_index,
        warmup=warmup,
        round_id=round_id,
        stream_index=stream_index,
        request_body_ref=body_ref,
        result=result,
        flags=_trial_flags(
            ctx,
            cond,
            result,
            measures_decode_speed=measures_decode_speed,
            expect_full_output=expect_full_output,
        ),
        verdict=verdict(result) if verdict is not None else None,
        target_input_tokens=cond.target_input_tokens,
    )


def _trial_flags(
    ctx: SuiteContext,
    cond: ConditionPlan,
    result: StreamResult,
    *,
    measures_decode_speed: bool,
    expect_full_output: bool,
) -> list[TrialFlag]:
    """すべてのまとまりに共通の印 (この module の docstring「共通の印」を参照)。"""
    flags: list[TrialFlag] = []
    succeeded = result.error is None
    usage = result.usage

    # 入力の長さの事実なので、要求が失敗していてもトークン数が返っていれば見る
    target = cond.target_input_tokens
    if (
        target is not None
        and usage is not None
        and abs(usage.total_input_tokens - target) / target > ctx.profile.length_tolerance
    ):
        flags.append(TrialFlag.LENGTH_OFF_TARGET)

    if not succeeded:
        return _ordered(flags)

    if expect_full_output and result.stop_reason != "max_tokens":
        flags.append(TrialFlag.SHORT_OUTPUT)
    if (
        measures_decode_speed
        and usage is not None
        and usage.output_tokens < MIN_SPEED_OUTPUT_TOKENS
    ):
        flags.append(TrialFlag.TOO_FEW_OUTPUT_TOKENS)

    sanity = ctx.profile.output_sanity
    flags.extend(
        detect_output_anomalies(
            collect_text(result), sanity.repeat_min_chars, sanity.repeat_min_count
        )
    )
    if _REPLACEMENT_CHAR in collect_tool_input_text(result):
        flags.append(TrialFlag.REPLACEMENT_CHAR)
    return _ordered(flags)


def _ordered(flags: Sequence[TrialFlag]) -> list[TrialFlag]:
    """`TrialFlag` の定義順に並べ、同じ印を 1 つにまとめる。"""
    return sorted(set(flags), key=lambda flag: _FLAG_ORDER[flag])


# --- 実行中に上限に当たったとき -------------------------------------------


def abort_if_context_limit(
    cond: ConditionPlan, record: TrialRecord, *, reached_input_tokens: int | None = None
) -> None:
    """上限を超えた応答なら、その条件を打ち切る (3.6、6.9)。

    まとまりは、レコードを `yield` した**あと**に 1 行で呼ぶ (レコードは残す。
    8.1)。上限の超過でなければ、何もしない。

    理由には、対象サーバーの言い分 (解析しない。注 2.7) と、到達した入力の
    長さを残す。長さは `reached_input_tokens` → 対象サーバーが返したトークン数
    → 計画の狙いの長さ、の順に採る。
    """
    if not is_context_limit_error(record.result):
        return
    reached = reached_input_tokens
    if reached is None and record.result.usage is not None:
        reached = record.result.usage.total_input_tokens
    if reached is None:
        reached = cond.target_input_tokens

    parts = ["対象サーバーが入力の長さの上限を返した (HTTP 400) ので、この条件で止めた"]
    if reached is not None:
        parts.append(f"到達した入力の長さ: {reached} トークン")
    error = record.result.error
    if error is not None and error.message:
        parts.append(f"対象サーバーの応答: {error.message}")
    raise ConditionAborted(skip_condition(cond.suite, cond.key, "。".join(parts)))
