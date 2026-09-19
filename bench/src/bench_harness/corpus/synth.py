"""種から決まる合成の文章 (task 2.5: corpus/synth)。

散文 (英語、日本語)、コード、ログの 4 種類の文章を、型紙に `random.Random(seed)`
で値を埋めて作る。語の並べ替え (word salad) ではなく、業務の報告やチケット、
ログ、コードのファイルを模した型紙に、コンポーネント名や数、日付などの値を
乱数で埋める。時刻、環境変数、`hash()`、大域の乱数、dict や set の反復順序に
は依存しない。同じ `chars_per_token` (constructor の引数) と、同じ呼び出しの
引数 (`lang`、`target_tokens`、`seed`) なら、プロセスをまたいでも、
`PYTHONHASHSEED` を変えても、バイト単位で同じ文字列を返す (11.4)。内容は
すべて型紙と乱数から作った架空のもので、業務の実データも、外部の文章の
引き写しもない (11.1)。コンポーネント名やチケット番号は、すべて架空である。

## 長さの狙い方

`target_tokens` は `Profile.chars_per_token[kind]` (constructor の引数) を
掛けて、文字数の狙い (`target_chars`) に変える。実際の出力は `target_chars`
の ±2% (`_CHAR_BUDGET_TOLERANCE`) に収める。target_tokens が 200〜128000 の
どの値でもこの精度で収める。これは `Profile.length_tolerance` (既定 ±5%、
対象サーバーが返した実際のトークン数との比較に使う、別の緩い許容) とは
別物である。

境界の決まり:

- **散文 (`prose`) とログ (`log`)**: 文、またはログの行を単位として、丸ごと
  足しても上限を超えないうちは足す。次の単位を足すと上限を超えるときは、
  その 1 単位だけを語 (日本語は文字。空白で区切れないため) に割って、区間に
  収まるところまで足す。語 1 つでも上限を超えるまれな場合は、文字まで割る
  (`_fit_unit_stream`)
- **コード (`code`)**: モジュールの docstring、1 つの関数、1 つのクラスを
  単位とし、単位の途中では決して切らない (構文を壊さないため)。次の単位を
  足すと上限を超えるときは、そこで単位を積むのをやめ、残りの文字数に
  ぴったり合う、`#` で始まる 1 行のコメントを足して閉じる。これは常に有効な
  Python の 1 行なので、`ast.parse` は常に通る (`_fit_code_stream`)

## 先頭の一致 (prefix stability)

`prose`、`log`、`code` はどれも、`target_tokens` に関わらず同じ乱数の列
(呼び出しのたびに `seed` から作り直す `random.Random(seed)`) から単位を
生成し、狙った長さのぶんだけ先頭から使う。そのため、同じ `(kind, lang, seed)`
では、大きい `target_tokens` の出力は、小さい `target_tokens` の出力で
始まる (`larger.startswith(smaller)`)。これは、境界の調整で語や文字に
割った断片が、まさに次の呼び出しでも同じ位置に現れる、同じ乱数の列の
続きだからである。

ただし `code` にだけ 1 点の例外がある。残りの文字数を埋めるために足す末尾の
コメント行 (`#` で始まる) は、`target_tokens` ごとに内容も長さも違う (その
時点での残りの文字数にぴったり合わせて作るため)。したがって `code` の先頭の
一致は、末尾のコメント行を除いた部分 (最後の完全な関数またはクラスまで) に
だけ保証される。呼び出し側は、末尾の行が `#` で始まっていれば、それを取り
除いてから先頭の一致を確かめること。

task 7.1 (長い会話) は、この先頭の一致をそのまま使ってよい。task 3.3
(キャッシュが効く条件) は、同じ `(kind, lang, seed)` の先頭が同じであること
を使う。

## `prefix_nonce`

入力の先頭に置く、決まった長さ (32 桁の 16 進、`PREFIX_NONCE_HEX_LEN`) の
識別子。渡した引数だけで決まる純粋な関数で、`chars_per_token` (constructor
の引数) にも、他の呼び出しの `seed` にも依存しない (種を効かせたいときは、
`seed` を引数の 1 つとして渡す)。引数の型 (`str` か `int`) と長さを符号化に
含めるので、`("a", "bc")` と `("ab", "c")`、`1` と `"1"` は違う nonce に
なる (`_encode_part`)。

呼び出し側 (task 3.3、3.4) の決まり (design.md corpus):

- **キャッシュが効かない条件**: `prefix_nonce(seed, condition_key, trial_index, run_id)`
  — 試行ごと、計測ランごとに違う nonce になる
- **キャッシュが効く条件**: `prefix_nonce(seed, condition_key)`
  — 同じ条件なら、試行や計測ランをまたいでも同じ nonce になる

`prefix_header_line(nonce)` は、この nonce をシステムプロンプトの 1 行目に
する。行の全長は `nonce` の長さだけで決まるので、効く条件と効かない条件とで
この行のトークン数が変わらない。

## GENERATOR_VERSION

型紙や生成の手順を変えて出力が変わる変更 (型紙の追加や削除、乱数の消費の
順序の変更、境界の決め方の変更など) は、必ず `bench_harness.types.GENERATOR_VERSION`
を上げること (design.md の Revalidation Triggers)。`test_corpus_synth.py` の
golden な sha256 の試験が、意図しない変更を検出する。

依存するのは標準ライブラリと `bench_harness.types` だけ。ほかの
`bench_harness` の module を読み込まない。
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Final, Protocol, runtime_checkable

from bench_harness.types import ContentKind, Lang

__all__ = [
    "PREFIX_NONCE_HEX_LEN",
    "SyntheticCorpus",
    "TemplateCorpus",
    "prefix_header_line",
]

PREFIX_NONCE_HEX_LEN: Final[int] = 32
"""`prefix_nonce` が返す、16 進の桁数 (design.md corpus)。"""

_CHAR_BUDGET_TOLERANCE: Final[float] = 0.02
"""文字数の狙いに対する許容 (±2%)。task 2.5 の完了の条件。"""

_PARAGRAPH_SENTENCES: Final[int] = 4
"""この数の文ごとに、段落の区切り (空行) を入れる。"""

_PREFIX_LINE_PREFIX: Final[str] = "[bench-prefix "
_PREFIX_LINE_SUFFIX: Final[str] = "]"
_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdef")


# --- 約束事 (Protocol) -----------------------------------------------------


@runtime_checkable
class SyntheticCorpus(Protocol):
    """設定と種から、いつでも同じ入力を作る (design.md corpus)。"""

    def prose(self, lang: Lang, target_tokens: int, seed: int) -> str: ...

    def code(self, target_tokens: int, seed: int) -> str: ...

    def log(self, target_tokens: int, seed: int) -> str: ...

    def prefix_nonce(self, *parts: str | int) -> str: ...


# --- 先頭の識別子 (prefix_nonce) --------------------------------------------


def _encode_part(part: str | int) -> bytes:
    """`prefix_nonce` の 1 つの引数を、型と長さを含めて明確に符号化する。

    型 (文字列 `s`、整数 `i`) と、値の符号化のあとの長さを前置きすることで、
    `("a", "bc")` と `("ab", "c")`、`1` と `"1"` が、違う符号化になり違う
    nonce になる (長さを前置きしない単純な連結だと、これらが同じ文字列に
    なってしまう)。
    """
    if isinstance(part, str):
        data = part.encode("utf-8")
        return b"s" + str(len(data)).encode("ascii") + b":" + data
    data = str(part).encode("ascii")
    return b"i" + str(len(data)).encode("ascii") + b":" + data


def _prefix_nonce(*parts: str | int) -> str:
    encoded = b"".join(_encode_part(part) for part in parts)
    return hashlib.sha256(encoded).hexdigest()[:PREFIX_NONCE_HEX_LEN]


def prefix_header_line(nonce: str) -> str:
    """`nonce` を、システムプロンプトの 1 行目にする。

    行の全長は `nonce` の長さ (常に `PREFIX_NONCE_HEX_LEN`) だけで決まるので、
    効かない条件 (試行ごとに違う nonce) と効く条件 (同じ nonce) とで、この行の
    トークン数が変わらない。
    """
    if len(nonce) != PREFIX_NONCE_HEX_LEN or any(ch not in _HEX_DIGITS for ch in nonce):
        raise ValueError(
            f"prefix_header_line: nonce は {PREFIX_NONCE_HEX_LEN} 桁の小文字の 16 進である"
            f"必要がある (nonce={nonce!r})"
        )
    return f"{_PREFIX_LINE_PREFIX}{nonce}{_PREFIX_LINE_SUFFIX}"


# --- 境界に合わせて単位をつなぐ、共通の仕組み -------------------------------

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\S+\s*")


def _word_pieces(text: str) -> list[str]:
    """`text` を、後ろに続く空白も込みで 1 語ずつ割る (英語の散文、ログの行)。"""
    return _WORD_RE.findall(text)


def _char_pieces(text: str) -> list[str]:
    """`text` を 1 文字ずつ割る (日本語。空白で区切れないため)。"""
    return list(text)


_SPLITTERS_CHAR_LEVEL: Final[tuple[Callable[[str], list[str]], ...]] = (_char_pieces,)
_SPLITTERS_WORD_LEVEL: Final[tuple[Callable[[str], list[str]], ...]] = (_word_pieces, _char_pieces)


def _closest_fit(
    parts: list[str],
    length: int,
    pieces: Iterable[str],
    target: float,
    lower: float,
    upper: float,
    splitters: tuple[Callable[[str], list[str]], ...],
) -> str:
    """`pieces` を `parts`/`length` に順に足しながら、`target` に最も近い、
    区間 `[lower, upper]` に収まる切れ目で止める。

    1 つの piece を丸ごと足すと上限を超えるときは、`splitters[0]` でそれを
    さらに細かい piece に割り、残りの `splitters` を引き継いで続ける
    (`splitters` が尽きたら、そこで打ち切る)。
    """
    for piece in pieces:
        candidate = length + len(piece)
        if candidate > upper:
            if length >= lower:
                return "".join(parts)
            if splitters:
                return _closest_fit(
                    parts, length, splitters[0](piece), target, lower, upper, splitters[1:]
                )
            return "".join(parts)
        if candidate >= target:
            # target をまたいだ (または、ちょうど届いた)。足す前と足した後の
            # うち、target に近いほうを選ぶ。足す前が区間に入っていない
            # (length < lower) ときは、選ぶ余地がないので必ず足す
            if length >= lower and (target - length) <= (candidate - target):
                return "".join(parts)
            parts.append(piece)
            return "".join(parts)
        parts.append(piece)
        length = candidate
    return "".join(parts)


def _fit_unit_stream(
    units: Iterator[str], target_chars: float, tolerance: float, *, char_level: bool
) -> str:
    """`units` (文やログの行) を順につなぎ、`target_chars` に最も近い、その
    ±`tolerance` に収まる文字列を返す。

    丸ごと 1 単位を足しても上限を超えず、かつまだ狙いに届いていないうちは、
    そのまま足す。次の単位を足すと上限を超える、または狙いをまたぐときは、
    足す前と足した後のうち、狙いに近いほうを選ぶ。丸ごと 1 単位が上限を
    超えてしまい、かつ足す前がまだ区間に入っていないときは、その 1 単位を
    語 (または文字) に割って、同じ規則で続きを積む。区間の幅
    (`upper - lower`) が 1 文字以上ある限り、必ず区間に収まったところで
    返る (`upper` の際まで 1 文字ずつ足せるため)。
    """
    upper = target_chars * (1.0 + tolerance)
    lower = target_chars * (1.0 - tolerance)
    splitters = _SPLITTERS_CHAR_LEVEL if char_level else _SPLITTERS_WORD_LEVEL
    return _closest_fit([], 0, units, target_chars, lower, upper, splitters)


def _pad_comment(length: int) -> str:
    """`length` 文字ちょうどの、有効な Python の 1 行のコメントを返す。"""
    if length <= 0:
        return ""
    if length == 1:
        return "#"
    # 同じ文字の連続にしない (出力の健全性の検査が、繰り返しと見なさないように)。
    # 通し番号を並べるので、どの長さでも周期を持たない。
    count = length // 5 + 2
    filler = "# pad" + "".join(f" {number:04d}" for number in range(1, count + 1))
    return filler[:length]


def _fit_code_stream(units: Iterator[str], target_chars: float, tolerance: float) -> str:
    """コードの完全な単位 (モジュールの docstring、1 つの関数、1 つのクラス) を
    順につなぎ、`target_chars` に最も近い、その ±`tolerance` に収まる文字列を
    返す。

    単位の途中では決して切らない (構文を壊さないため)。次の単位を足すと
    上限を超える、または狙いをまたぐときは、足す前と足した後のうち、狙いに
    近いほうを選ぶ (`_closest_fit` と同じ規則)。どちらの単位境界も区間に
    収まらないときは、そこで単位を積むのをやめ、残りの文字数にぴったり合う
    `#` で始まる 1 行のコメントを足して閉じる。
    """
    upper = target_chars * (1.0 + tolerance)
    lower = target_chars * (1.0 - tolerance)
    parts: list[str] = []
    length = 0
    for unit in units:
        candidate = length + len(unit)
        if candidate > upper:
            break
        if candidate >= target_chars:
            if length >= lower and (target_chars - length) <= (candidate - target_chars):
                break  # 足す前のほうが狙いに近い。足さずにパディングへ
            parts.append(unit)
            return "".join(parts)
        parts.append(unit)
        length = candidate
    target_len = round(target_chars)
    gap = target_len - length
    if gap > 0:
        parts.append(_pad_comment(gap))
    return "".join(parts)


# --- 語彙 (架空。業務の実データではない) ------------------------------------

_COMPONENT_ROOTS: Final[tuple[str, ...]] = (
    "nimbus cache",
    "echo broker",
    "quill router",
    "delta scheduler",
    "vault manager",
    "pulse collector",
    "arc drafter",
    "forge gateway",
    "tide balancer",
    "haven cache",
    "loom packer",
    "sable verifier",
    "ridge broker",
    "aster runner",
    "coral sweeper",
    "brisk collector",
)
_CODE_COMPONENTS: Final[tuple[str, ...]] = tuple(
    root.replace(" ", "_") for root in _COMPONENT_ROOTS
)

_EN_ROLES: Final[tuple[str, ...]] = (
    "on-call engineer",
    "release owner",
    "duty lead",
    "pipeline owner",
    "review lead",
    "capacity planner",
    "reliability lead",
    "build owner",
)
_EN_OUTCOMES: Final[tuple[str, ...]] = (
    "resolved",
    "mitigated",
    "rolled back",
    "under investigation",
    "reproduced in staging",
    "pending a follow-up",
    "closed as expected",
    "still degraded",
)
_EN_ADJECTIVES: Final[tuple[str, ...]] = (
    "priority",
    "bounded",
    "deferred",
    "sharded",
    "replicated",
    "fair",
)
_EN_DURATIONS: Final[tuple[str, ...]] = ("5-minute", "15-minute", "30-minute", "1-hour", "4-hour")

_JA_ROLES: Final[tuple[str, ...]] = (
    "当番のエンジニア",
    "リリース担当",
    "当直リーダー",
    "パイプライン担当",
    "レビュー担当",
    "キャパシティ担当",
    "信頼性担当",
    "ビルド担当",
)
_JA_OUTCOMES: Final[tuple[str, ...]] = (
    "解消した",
    "緩和された",
    "ロールバックされた",
    "調査中である",
    "ステージングで再現した",
    "対応待ちである",
    "想定どおり終了した",
    "依然として劣化している",
)
_JA_ADJECTIVES: Final[tuple[str, ...]] = (
    "優先度付きの",
    "上限付きの",
    "非同期の",
    "分割された",
    "複製された",
    "公平な",
)
_JA_DURATIONS: Final[tuple[str, ...]] = ("5分間", "15分間", "30分間", "1時間", "4時間")

_LOG_LEVELS: Final[tuple[str, ...]] = ("DEBUG", "INFO", "WARN", "ERROR")
_LOG_MESSAGES: Final[tuple[str, ...]] = (
    "request accepted",
    "cache hit",
    "cache miss",
    "retry scheduled",
    "batch flushed",
    "connection reset",
    "queue drained",
    "shard rebalanced",
    "checkpoint written",
    "eviction triggered",
    "lease renewed",
    "backoff applied",
    "healthcheck passed",
    "healthcheck failed",
    "watermark advanced",
    "snapshot taken",
)

_CODE_VERBS: Final[tuple[str, ...]] = (
    "route",
    "drain",
    "balance",
    "verify",
    "collect",
    "pack",
    "sweep",
    "warm",
)


def _fake_date(rng: random.Random) -> str:
    """実際の日時に依存しない、固定の起点からの架空の日付。"""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return (base + timedelta(days=rng.randint(0, 730))).strftime("%Y-%m-%d")


# --- 散文の型紙 (英語) -------------------------------------------------------


def _en_t01(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    n = rng.randint(1, 42)
    dur = rng.choice(_EN_DURATIONS)
    date = _fake_date(rng)
    return f"The {comp} reported {n} errors during the {dur} window ending {date}."


def _en_t02(rng: random.Random) -> str:
    role = rng.choice(_EN_ROLES)
    comp = rng.choice(_COMPONENT_ROOTS)
    outcome = rng.choice(_EN_OUTCOMES)
    minutes = rng.randint(2, 90)
    return (
        f"The {role} confirmed that the {comp} was {outcome} "
        f"after {minutes} minutes of investigation."
    )


def _en_t03(rng: random.Random) -> str:
    ticket = rng.randint(1000, 9999)
    comp = rng.choice(_COMPONENT_ROOTS)
    role = rng.choice(_EN_ROLES)
    return f"Ticket BH-{ticket} tracks a regression in the {comp} observed by the {role}."


def _en_t04(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    pct = rng.randint(2, 45)
    date = _fake_date(rng)
    return f"Latency on the {comp} increased by {pct}% after the {date} deployment."


def _en_t05(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    comp2 = rng.choice(_COMPONENT_ROOTS)
    adj = rng.choice(_EN_ADJECTIVES)
    return f"The design memo for the {comp} proposes replacing the {comp2} with a {adj} queue."


def _en_t06(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    passed = rng.randint(1, 40)
    total = passed + rng.randint(0, 10)
    return f"The {comp} passed {passed} of {total} checks in the nightly validation run."


def _en_t07(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    hours = rng.randint(1, 72)
    return f"No further action is required; the {comp} remained stable for {hours} hours."


def _en_t08(rng: random.Random) -> str:
    role = rng.choice(_EN_ROLES)
    comp = rng.choice(_COMPONENT_ROOTS)
    date = _fake_date(rng)
    return f"The {role} scheduled a maintenance window for the {comp} on {date}."


def _en_t09(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    n = rng.randint(1, 12)
    return (
        f"A capacity review found that the {comp} needs {n} additional replicas "
        "before next quarter."
    )


def _en_t10(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    comp2 = rng.choice(_COMPONENT_ROOTS)
    return (
        f"Requests routed through the {comp} now fall back to the {comp2} "
        "when retries are exhausted."
    )


def _en_t11(rng: random.Random) -> str:
    role = rng.choice(_EN_ROLES)
    comp = rng.choice(_COMPONENT_ROOTS)
    outcome = rng.choice(_EN_OUTCOMES)
    return f"The {role} marked the incident on the {comp} as {outcome}."


def _en_t12(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    dur = rng.choice(_EN_DURATIONS)
    n = rng.randint(1, 999)
    return f"During the {dur} soak test, the {comp} sustained {n} requests per second."


_EN_TEMPLATES: Final[tuple[Callable[[random.Random], str], ...]] = (
    _en_t01,
    _en_t02,
    _en_t03,
    _en_t04,
    _en_t05,
    _en_t06,
    _en_t07,
    _en_t08,
    _en_t09,
    _en_t10,
    _en_t11,
    _en_t12,
)


# --- 散文の型紙 (日本語) ------------------------------------------------------


def _ja_t01(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    n = rng.randint(1, 42)
    dur = rng.choice(_JA_DURATIONS)
    date = _fake_date(rng)
    return f"{comp} は {date} までの {dur} の間に {n} 件のエラーを記録した。"


def _ja_t02(rng: random.Random) -> str:
    role = rng.choice(_JA_ROLES)
    comp = rng.choice(_COMPONENT_ROOTS)
    outcome = rng.choice(_JA_OUTCOMES)
    return f"{role}は、{comp} の状態が{outcome}ことを確認した。"


def _ja_t03(rng: random.Random) -> str:
    ticket = rng.randint(1000, 9999)
    comp = rng.choice(_COMPONENT_ROOTS)
    role = rng.choice(_JA_ROLES)
    return f"チケット BH-{ticket} は、{role}が {comp} で見つけた不具合を追跡している。"


def _ja_t04(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    pct = rng.randint(2, 45)
    date = _fake_date(rng)
    return f"{date} のデプロイ後、{comp} のレイテンシが {pct}% 増加した。"


def _ja_t05(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    comp2 = rng.choice(_COMPONENT_ROOTS)
    adj = rng.choice(_JA_ADJECTIVES)
    return f"{comp} の設計メモでは、{comp2} を{adj}キューに置き換える案を検討している。"


def _ja_t06(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    passed = rng.randint(1, 40)
    total = passed + rng.randint(0, 10)
    return f"夜間の検証では、{comp} が {total} 件中 {passed} 件のチェックに合格した。"


def _ja_t07(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    hours = rng.randint(1, 72)
    return f"{comp} は {hours} 時間にわたり安定していたため、追加の対応は不要である。"


def _ja_t08(rng: random.Random) -> str:
    role = rng.choice(_JA_ROLES)
    comp = rng.choice(_COMPONENT_ROOTS)
    date = _fake_date(rng)
    return f"{role}は、{comp} の保守作業を {date} に予定した。"


def _ja_t09(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    n = rng.randint(1, 12)
    return f"容量の見直しにより、{comp} には来期までに {n} 台の増設が必要だとわかった。"


def _ja_t10(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    comp2 = rng.choice(_COMPONENT_ROOTS)
    return f"{comp} を通る要求は、再試行が尽きると {comp2} に切り替わるようになった。"


def _ja_t11(rng: random.Random) -> str:
    role = rng.choice(_JA_ROLES)
    comp = rng.choice(_COMPONENT_ROOTS)
    outcome = rng.choice(_JA_OUTCOMES)
    return f"{role}は、{comp} の障害を{outcome}として記録した。"


def _ja_t12(rng: random.Random) -> str:
    comp = rng.choice(_COMPONENT_ROOTS)
    dur = rng.choice(_JA_DURATIONS)
    n = rng.randint(1, 999)
    return f"{dur}の負荷試験では、{comp} は 1 秒あたり {n} 件の要求を処理し続けた。"


_JA_TEMPLATES: Final[tuple[Callable[[random.Random], str], ...]] = (
    _ja_t01,
    _ja_t02,
    _ja_t03,
    _ja_t04,
    _ja_t05,
    _ja_t06,
    _ja_t07,
    _ja_t08,
    _ja_t09,
    _ja_t10,
    _ja_t11,
    _ja_t12,
)


def _sentence_stream(rng: random.Random, lang: Lang) -> Iterator[str]:
    """型紙を組み合わせて、文を無限に生成する。段落ごとに空行を挟む。"""
    templates = _EN_TEMPLATES if lang == "en" else _JA_TEMPLATES
    count = 0
    while True:
        sentence = rng.choice(templates)(rng)
        count += 1
        separator = "\n\n" if count % _PARAGRAPH_SENTENCES == 0 else " "
        yield sentence + separator


# --- ログの型紙 ---------------------------------------------------------------


def _log_line_stream(rng: random.Random, seed: int) -> Iterator[str]:
    """時刻、レベル、コンポーネント、要求 ID、所要時間を持つログの行を無限に生成する。

    時計は、種から決めた固定の起点 (`seed` を使った、実際の日時に依存しない
    オフセット) から始め、毎回正の乱数だけ進める (単調に増える)。実際の時刻
    (`datetime.now()` など) は一切読まない。
    """
    clock = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=seed % 100_000)
    while True:
        clock += timedelta(milliseconds=rng.randint(5, 2500))
        level = rng.choice(_LOG_LEVELS)
        comp = rng.choice(_CODE_COMPONENTS)
        req_id = f"{rng.randint(0, 0xFFFFFFF):07x}"
        dur_ms = rng.randint(1, 4500)
        msg = rng.choice(_LOG_MESSAGES)
        ts = clock.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        yield f"{ts} {level:<5} {comp} req={req_id} dur={dur_ms}ms {msg}\n"


# --- コードの型紙 -------------------------------------------------------------


def _module_header(seed: int) -> str:
    return (
        f'"""Synthetic benchmark module (seed={seed}).\n\n'
        "Fictional components only; nothing here is executed.\n"
        '"""\n\n'
    )


def _fn_unit(rng: random.Random, index: int) -> str:
    verb = rng.choice(_CODE_VERBS)
    comp = rng.choice(_CODE_COMPONENTS)
    name = f"{verb}_{comp}_{index:05d}"
    n1 = rng.randint(1, 900)
    n2 = rng.randint(1, 900)
    op = rng.choice(("+", "-", "*"))
    shape = rng.randint(0, 2)
    lines = [
        f"def {name}(x: int, y: int = {n1}) -> int:",
        f'    """Fictional helper for {comp}, used only to pad benchmark context."""',
    ]
    if shape == 0:
        lines += [f"    total = x {op} y {op} {n2}", "    return total"]
    elif shape == 1:
        step = max(2, n1 % 7 + 2)
        lines += [
            f"    values = [i for i in range(y) if i % {step} == 0]",
            "    return sum(values) + x",
        ]
    else:
        lines += [f"    if x > {n1}:", f"        return x - {n2}", f"    return x + {n2}"]
    lines.append("")
    return "\n".join(lines) + "\n"


def _class_unit(rng: random.Random, index: int) -> str:
    comp = rng.choice(_CODE_COMPONENTS)
    limit = rng.randint(1, 500)
    name = f"Fictional{comp.title().replace('_', '')}{index:05d}"
    lines = [
        f"class {name}:",
        f'    """Fictional stand-in for {comp}, used only to vary benchmark shape."""',
        "",
        f"    def __init__(self, capacity: int = {limit}) -> None:",
        "        self.capacity = capacity",
        "        self.used = 0",
        "",
        "    def reserve(self, amount: int) -> bool:",
        "        if self.used + amount > self.capacity:",
        "            return False",
        "        self.used += amount",
        "        return True",
        "",
    ]
    return "\n".join(lines) + "\n"


def _code_unit_stream(rng: random.Random, seed: int) -> Iterator[str]:
    """モジュールの docstring のあと、関数とクラスを交互に無限に生成する。"""
    yield _module_header(seed)
    index = 0
    while True:
        yield _class_unit(rng, index) if index % 4 == 3 else _fn_unit(rng, index)
        index += 1


# --- 実装 (SyntheticCorpus) --------------------------------------------------


class TemplateCorpus:
    """`SyntheticCorpus` の実装。型紙に乱数で値を埋めて、決定的に文章を作る。

    決まりと使い方は、この module の docstring を参照。`chars_per_token` は、
    `Profile.chars_per_token` (内容の種類ごとの、1 トークンあたりの文字数) を
    渡す。
    """

    def __init__(self, chars_per_token: Mapping[ContentKind, float]) -> None:
        missing = [kind.value for kind in ContentKind if kind not in chars_per_token]
        if missing:
            raise ValueError(f"chars_per_token に足りない種類がある: {', '.join(missing)}")
        if any(value <= 0.0 for value in chars_per_token.values()):
            raise ValueError("chars_per_token の値は 0 より大きい必要がある")
        self._chars_per_token: dict[ContentKind, float] = dict(chars_per_token)

    def prose(self, lang: Lang, target_tokens: int, seed: int) -> str:
        kind = ContentKind.PROSE_EN if lang == "en" else ContentKind.PROSE_JA
        target_chars = self._budget(kind, target_tokens)
        rng = random.Random(seed)
        units = _sentence_stream(rng, lang)
        return _fit_unit_stream(
            units, target_chars, _CHAR_BUDGET_TOLERANCE, char_level=(lang == "ja")
        )

    def code(self, target_tokens: int, seed: int) -> str:
        target_chars = self._budget(ContentKind.CODE, target_tokens)
        rng = random.Random(seed)
        units = _code_unit_stream(rng, seed)
        return _fit_code_stream(units, target_chars, _CHAR_BUDGET_TOLERANCE)

    def log(self, target_tokens: int, seed: int) -> str:
        target_chars = self._budget(ContentKind.LOG, target_tokens)
        rng = random.Random(seed)
        units = _log_line_stream(rng, seed)
        return _fit_unit_stream(units, target_chars, _CHAR_BUDGET_TOLERANCE, char_level=False)

    def prefix_nonce(self, *parts: str | int) -> str:
        return _prefix_nonce(*parts)

    def _budget(self, kind: ContentKind, target_tokens: int) -> float:
        if target_tokens <= 0:
            raise ValueError(
                f"target_tokens は正の整数である必要がある (target_tokens={target_tokens})"
            )
        return target_tokens * self._chars_per_token[kind]
