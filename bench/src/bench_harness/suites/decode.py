"""生成速度のまとまり (task 3.2: suites/decode)。

コード、散文、JSON と英語・日本語を掛け合わせた 6 つの条件
(`decode/{code,prose,json}/{en,ja}`)
で、試行ごとに違う、長く書かせる指示を送る。まとまりの共通の部品
(`suites/base.py`) の上に薄く載るだけで、条件の計画と、試行ごとの入力の
組み立てだけをこの module が受け持つ (design.md「suites」の decode の行)。

## 指示の作り方

話題は、領域・題材と切り口の 2 つの小さな型紙の掛け合わせ
(8 × 4 = 32 通り) から選ぶ。選び方 (`_topic_index`) は、試行の番号と慣らし
かどうかだけで決まる添字で、乱数を使わない。本番の試行は添字 0 から、
慣らしの試行は別の範囲 (`_WARMUP_TOPIC_OFFSET` から) を使うので、同じ番号
でも慣らしと本番で違う話題になる。32 通りが 20 を超えるので、同じ条件の
最初の 20 本の本番の試行は、必ず違う話題になる。

文中の架空の管理番号だけを、`random.Random(trial_seed(...))` で決める (code、prose、
json で同じ)。`trial_seed` は `run_id` を含まない (base.py) ので、計測ランをまたいでも、
同じ番号の試行は同じ指示になる。

話題も文中の題材も、すべて架空である (11.1: クリーンルーム。実在の企業や
人物、製品は登場しない)。英語の指示 (`*/en`) は英語の出力を、日本語の指示
(`*/ja`) は日本語の出力 (`code/ja` はコードのコメントと説明も日本語) を
求める。指示の文には、埋め込み忘れの型紙の記号 (`{`、`}`) を残さない。

## 慣らしとキャッシュ

先頭の識別子は、すべての試行で `cold_prefix_nonce` (3.4) を使う (`warm_prefix_nonce`
は使わない)。生成速度の計測で、プレフィックスキャッシュの当たりが最初の
トークンまでの時間を縮めてしまわないためである。`run_trial` に
`measures_decode_speed=True`、`expect_full_output=True` を渡し、共通の印
(`TOO_FEW_OUTPUT_TOKENS`、`SHORT_OUTPUT` など) は `suites/base.py` に任せる。

依存するのは標準ライブラリと `bench_harness.suites.base`、
`bench_harness.types` だけ。ほかの `suites/*.py` は読み込まない。
"""

from __future__ import annotations

import random
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
from bench_harness.types import ConditionPlan, Lang, SkippedCondition, SuiteName, TrialRecord

__all__ = ["DecodeSuite", "decode_suite"]

_KINDS: Final[tuple[str, ...]] = ("code", "prose", "json")
_LANGS: Final[tuple[Lang, ...]] = ("en", "ja")

_SYSTEM_LINE: Final[str] = (
    "You are a benchmarking assistant. Follow the user's instruction exactly and "
    "keep writing until it is fully satisfied; never stop early or ask a question back."
)

_CODE_MIN_UNITS: Final[int] = 5
_CODE_MIN_LINES: Final[int] = 150
_PROSE_MIN_WORDS_EN: Final[int] = 1200
_PROSE_MIN_CHARS_JA: Final[int] = 3000
"""日本語の散文に求める最小の文字数。1 トークンあたり約 1.5 文字なので、約 2000 トークン
ぶん。出力の上限 (既定 1024 トークン) を大きく超える量を求めて、上限まで書かせる (2.6)。"""

_WARMUP_TOPIC_OFFSET: Final[int] = 5000
"""慣らしの話題の添字を、本番とは別の範囲にずらす (「慣らしの索引空間」)。"""

# --- 話題の型紙 (架空。業務の実データや実在の企業・人物・製品ではない) --------

_CODE_DOMAINS_EN: Final[tuple[str, ...]] = (
    "a task queue for a fictional weather-balloon fleet",
    "an inventory tracker for a fictional deep-sea research station",
    "a scheduling library for a fictional lighthouse network",
    "a routing table for a fictional pneumatic-tube mail system",
    "a rate limiter for a fictional greenhouse irrigation grid",
    "a caching layer for a fictional archive of expedition logs",
    "a state machine for a fictional model-train signal yard",
    "a plugin registry for a fictional puzzle-box vending machine",
)
_CODE_FOCUSES_EN: Final[tuple[str, ...]] = (
    "with clear error handling throughout",
    "with a small internal event log",
    "with configurable retry limits",
    "with simple built-in metrics counters",
)

_CODE_DOMAINS_JA: Final[tuple[str, ...]] = (
    "架空の気球観測隊の作業待ち行列を管理する仕組み",
    "架空の深海調査基地の備品を管理する仕組み",
    "架空の灯台網の点検予定を管理する仕組み",
    "架空の空気管郵便の経路表を管理する仕組み",
    "架空の温室の散水を制御する仕組み",
    "架空の探検記録を保管するキャッシュの仕組み",
    "架空の模型鉄道の信号を制御する状態機械",
    "架空のからくり箱の自動販売機のプラグイン登録の仕組み",
)
_CODE_FOCUSES_JA: Final[tuple[str, ...]] = (
    "エラー処理を随所で丁寧に行う",
    "内部の操作ログを簡単に残す",
    "再試行の回数を設定で変えられる",
    "簡単な計数の仕組みを内蔵する",
)

_PROSE_TOPICS_EN: Final[tuple[str, ...]] = (
    "the annual migration patterns of the fictional Nimbus petrel",
    "the history of the fictional Tallow Strait lighthouse keepers",
    "the design tradeoffs of a fictional pneumatic mail network",
    "the maintenance routines of a fictional glacier research camp",
    "the founding of a fictional mountain observatory",
    "the daily operations of a fictional floating market cooperative",
    "the restoration of a fictional canal lock system",
    "the training program for a fictional volcano monitoring team",
)
_PROSE_ANGLES_EN: Final[tuple[str, ...]] = (
    "written for new trainees",
    "written for a funding committee",
    "written for a general audience",
    "written for a technical review board",
)

_PROSE_TOPICS_JA: Final[tuple[str, ...]] = (
    "架空のニンバス海燕の年ごとの渡りの様子",
    "架空のタロー海峡の灯台守の歴史",
    "架空の空気管郵便網の設計上の判断",
    "架空の氷河調査キャンプの保守作業",
    "架空の山岳天文台の創設の経緯",
    "架空の水上市場組合の日々の運営",
    "架空の運河の水門の修復作業",
    "架空の火山観測隊の研修の進め方",
)
_PROSE_ANGLES_JA: Final[tuple[str, ...]] = (
    "新人研修の受講者向けに",
    "助成金の審査委員会向けに",
    "一般の読者向けに",
    "技術の審査委員会向けに",
)

_JSON_TOPICS_EN: Final[tuple[str, ...]] = (
    "weather balloon inspection logs",
    "deep sea station supply transfers",
    "lighthouse lens maintenance visits",
    "pneumatic mail delivery rounds",
    "greenhouse irrigation checks",
    "expedition archive catalog entries",
    "model train signal inspections",
    "puzzle box vending machine restocks",
)
_JSON_TOPICS_JA: Final[tuple[str, ...]] = (
    "気球観測隊の点検記録",
    "深海調査基地の備品移送記録",
    "灯台のレンズの保守訪問記録",
    "空気管郵便の配達記録",
    "温室の散水点検記録",
    "探検資料の目録記録",
    "模型鉄道の信号点検記録",
    "からくり箱の自動販売機の補充記録",
)
_JSON_WARMUP_TOPICS_EN: Final[tuple[str, ...]] = (
    "kite festival registrations",
    "paper boat race entries",
    "lantern workshop reservations",
    "clock tower tour bookings",
    "cloud atlas annotations",
    "shell museum loans",
    "miniature garden visits",
    "snow globe repairs",
)
_JSON_WARMUP_TOPICS_JA: Final[tuple[str, ...]] = (
    "凧祭りの参加登録",
    "紙舟競走の参加記録",
    "ランタン工房の予約記録",
    "時計塔見学の予約記録",
    "雲図鑑の注記",
    "貝殻博物館の貸出記録",
    "箱庭の見学記録",
    "スノードームの修理記録",
)
_JSON_ANGLES_EN: Final[tuple[str, ...]] = (
    "organized by location",
    "organized by assigned team",
    "organized by inspection phase",
    "organized by priority",
)
_JSON_ANGLES_JA: Final[tuple[str, ...]] = (
    "場所ごとに整理した",
    "担当班ごとに整理した",
    "点検段階ごとに整理した",
    "優先度ごとに整理した",
)


class DecodeSuite:
    """生成速度のまとまり (design.md suites の decode の行)。"""

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
            for kind in _KINDS
            for lang in _LANGS
        ]

    async def run_condition(
        self, ctx: SuiteContext, cond: ConditionPlan
    ) -> AsyncIterator[TrialRecord]:
        kind, lang = _parse_condition_key(cond.key)
        for trial_index, warmup in iter_trials(cond):
            seed = trial_seed(ctx.profile.seed, cond.key, trial_index, warmup=warmup)
            topic_index = _topic_index(trial_index, warmup=warmup)
            if kind == "code":
                instruction = _code_instruction(lang, seed, topic_index)
            elif kind == "prose":
                instruction = _prose_instruction(lang, seed, topic_index)
            else:
                instruction = _json_instruction(lang, seed, topic_index, warmup=warmup)
            nonce = cold_prefix_nonce(ctx, cond, trial_index=trial_index, warmup=warmup)
            record = await run_trial(
                ctx,
                cond,
                trial_index=trial_index,
                warmup=warmup,
                system=system_with_prefix(nonce, _SYSTEM_LINE),
                messages=single_user_message(instruction),
                measures_decode_speed=True,
                expect_full_output=True,
            )
            yield record
            abort_if_context_limit(cond, record)


decode_suite: Final[DecodeSuite] = DecodeSuite()
"""計測ランの進行 (3.5) が直接 import する、まとまりの実体。"""


def _parse_condition_key(key: str) -> tuple[str, Lang]:
    """条件の鍵 (`decode/{kind}/{lang}`) から `kind` と `lang` を取り出す。"""
    parts = key.split("/")
    if len(parts) != 3 or parts[0] != SuiteName.DECODE.value:
        raise ValueError(f"decode: 未知の条件の鍵 (key={key!r})")
    kind, lang = parts[1], parts[2]
    if kind not in _KINDS or lang not in _LANGS:
        raise ValueError(f"decode: 未知の条件の鍵 (key={key!r})")
    return kind, lang


def _topic_index(trial_index: int, *, warmup: bool) -> int:
    """話題を選ぶ添字。試行の番号と慣らしかどうかだけで決まる (乱数を使わない)。

    型紙の掛け合わせ (8 × 4 = 32 通り) の範囲に入っている限り、番号ごとに
    違う話題になる。本番の試行は 0 から、慣らしの試行は
    `_WARMUP_TOPIC_OFFSET` から始まる、別々の添字の並びを使う。
    """
    return (_WARMUP_TOPIC_OFFSET + trial_index) if warmup else trial_index


def _code_instruction(lang: Lang, seed: int, index: int) -> str:
    rng = random.Random(seed)
    domains = _CODE_DOMAINS_EN if lang == "en" else _CODE_DOMAINS_JA
    focuses = _CODE_FOCUSES_EN if lang == "en" else _CODE_FOCUSES_JA
    domain = domains[index % len(domains)]
    focus = focuses[(index // len(domains)) % len(focuses)]
    tracking_number = rng.randint(1000, 9999)
    if lang == "en":
        return (
            f"Write one complete, self-contained Python module that implements "
            f"{domain}, {focus}. Somewhere in the module, use {tracking_number} as an "
            f"internal constant (for example a version or build number). Start with a "
            f"short module-level docstring, then define at least {_CODE_MIN_UNITS} "
            "classes or functions, each with its own docstring explaining what it does. "
            "Write only Python source code with English comments and docstrings; do not "
            "add any explanation before or after the code block. Keep adding classes and "
            f"functions until the module is at least {_CODE_MIN_LINES} lines long; do "
            "not stop early, and do not repeat the same class or function twice."
        )
    return (
        f"{domain}を実装する、完結した 1 つの Python モジュールを書いてください。"
        f"{focus}ようにしてください。モジュールのどこかに、版番号のような内部の定数と"
        f"して {tracking_number} を入れてください。先頭に短いモジュールの説明の "
        "docstring を置き、続けてそれぞれ docstring を持つクラスまたは関数を、少なくと"
        f"も {_CODE_MIN_UNITS} 個定義してください。コードのコメントと docstring はすべ"
        "て日本語で書き、コードのあとに日本語で短い説明を付け加えてください。少なくと"
        f"も {_CODE_MIN_LINES} 行になるまで、クラスや関数を増やしながら書き続けてくだ"
        "さい。途中で止めず、同じクラスや関数を繰り返さないでください。"
    )


def _prose_instruction(lang: Lang, seed: int, index: int) -> str:
    rng = random.Random(seed)
    topics = _PROSE_TOPICS_EN if lang == "en" else _PROSE_TOPICS_JA
    angles = _PROSE_ANGLES_EN if lang == "en" else _PROSE_ANGLES_JA
    topic = topics[index % len(topics)]
    angle = angles[(index // len(topics)) % len(angles)]
    tracking_number = rng.randint(1000, 9999)
    if lang == "en":
        return (
            f"Write one long, detailed report about {topic}, {angle}. Somewhere in the "
            f"text, mention {tracking_number} as a fictional internal case number. "
            "Organize the report into several sections, each with a short heading. Keep "
            f"writing until you have produced at least {_PROSE_MIN_WORDS_EN} words; do "
            "not stop early and do not summarize before you are done. Write only in "
            "English, in full sentences, with no code and no bullet lists."
        )
    return (
        f"{topic}について、長く詳しい報告書を 1 つ書いてください。{angle}書いてくださ"
        f"い。文中のどこかに、架空の管理番号として {tracking_number} を入れてくださ"
        "い。報告書は、それぞれ短い見出しを付けたいくつかの節に分けて構成してくださ"
        f"い。少なくとも {_PROSE_MIN_CHARS_JA} 字になるまで、途中で切り上げたり要約し"
        "たりせずに書き続けてください。日本語の文章だけで、コードや箇条書きを使わずに"
        "書いてください。"
    )


def _json_instruction(lang: Lang, seed: int, index: int, *, warmup: bool) -> str:
    if lang == "en":
        topics = _JSON_WARMUP_TOPICS_EN if warmup else _JSON_TOPICS_EN
        angles = _JSON_ANGLES_EN
    else:
        topics = _JSON_WARMUP_TOPICS_JA if warmup else _JSON_TOPICS_JA
        angles = _JSON_ANGLES_JA
    topic = topics[index % len(topics)]
    angle = angles[(index // len(topics)) % len(angles)]
    rng = random.Random(seed)
    tracking_number = rng.randint(1000, 9999)
    if lang == "en":
        return (
            f"Write a single long JSON array of fictional {topic}, {angle}. "
            f"Use {tracking_number} as the fictional internal case number for this dataset. "
            "Each array element must be a record object with the same fixed English "
            "field names: id, name, status, timestamp, quantity, and tags. Use a unique "
            "numeric id, a string name and status, an ISO 8601 timestamp, a numeric quantity, "
            "and an array of string tags for every record. Write at least 60 distinct "
            "records and keep writing until the array is complete; do not stop early. "
            "Output only the JSON array, with no explanation before or after it and no "
            "Markdown code fence. Use English for all string values."
        )
    return (
        f"架空の{topic}を、{angle}単一の長い JSON 配列として書いてください。"
        f"このデータ集合の架空の管理番号は {tracking_number} です。"
        "配列の各要素は同じ固定フィールドを持つ記録オブジェクトにしてください。"
        "キーは英語の id、name、status、timestamp、quantity、tags とし、"
        "id は一意の数値、name と status は文字列、timestamp は ISO 8601 形式の日時、"
        "quantity は数値、tags は文字列の配列にしてください。"
        "name、status、tags の文字列の値は日本語にしてください。"
        "少なくとも 60 件の異なる記録を書き、"
        "配列が完成するまで途中で止めずに書き続けてください。"
        "出力は JSON 配列だけにし、前後の説明や Markdown のコードブロックを付けないでください。"
    )
