"""長い入力から情報を探す課題の生成 (task 6.4: corpus/needle, design.md corpus, 5.3)。

英語の合成の散文 (`corpus/synth.py` の `TemplateCorpus.prose`) で狙った長さの
「干し草」を作り、そのどこか 1 か所に、正解が 1 つに決まる「探す情報」(架空の
コード名と、そのコード名に対応する架空の値) を 1 文だけ埋め込む。乱数は
`hashlib.sha256` から作った `random.Random` だけを使う (`hash()`、時刻、環境、
大域の乱数、辞書や集合の反復順序には依存しない)。同じ `(seed, index)` なら、
プロセスや `PYTHONHASHSEED` をまたいでも、同じコード名・同じ正解・同じ干し草の
土台になる (11.4)。

## 埋める位置 (`depth_pct`)

`depth_pct` は、狙う位置を干し草の文字数に対する百分率 (0〜100) で表す。

- `depth_pct <= 0`: 干し草の土台の直前 (先頭) にそのまま置く
- `depth_pct >= 100`: 干し草の土台の直後 (末尾) にそのまま置く
- それ以外: 土台のその割合に最も近い、文または段落の切れ目に置く。切れ目の
  探し方は `corpus/synth.py` の実装メモ (task 2.5) のとおり、この module の側で
  `. `、`。`、改行を目印に自分で見つける (`corpus/synth.py` は文や段落の切れ目を
  返す口を持たない)

挿入は、土台の文字列の途中に 1 文を差し込むだけで、文字を足しも引きもしない。
したがって、同じ土台と同じ挿入する文なら、`depth_pct` が変わっても `haystack`
全体の長さは変わらない。

## 挿入する文の書式 (task 6.4 の手直し、レビュー第 1 ラウンドの BLOCKING 指摘 4)

挿入する文はコード名を引用符で囲まない (`The access code for Copper-Wren-63
is SIGIL-....` のように、コード名の前後に `"` を置かない)。干し草の地の文
(`corpus/synth.py` の型紙) は二重引用符を一度も生成しないため、挿入した文だけに
引用符があると、モデルが本文を読まずに「引用符という珍しい文字を探す」だけで
埋め込んだ場所を見つけられてしまう恐れがあった。挿入する文は、地の文と同じ
文字種 (英字・数字・空白・ハイフン・ピリオド) だけでできている
(`tests/unit/test_corpus_needle.py` の
`test_needle_sentence_introduces_no_rare_characters` が、コード名と正解自身の
文字を除いて確かめる)。

`question` (別の項目。干し草には含まれない) は、引用符でコード名を囲んだまま
にしてある。モデルが読む「問い」の書式であり、干し草の中の目印にはならない
ため、変える理由がない。

挿入の継ぎ目は `_join_with_single_space` に任せる。土台の切れ目
(`. `、`。`、`\\n\\n`) は、すでに空白または改行で終わっている/始まっている
ことが多い (`corpus/synth.py` の `_fit_unit_stream` は、ほとんどの場合、単位
(文や段落) をまるごと含めたところで止まるため)。そこに単純に `f"{a} {b}"` で
継ぎ足すと、`…hours.\\n\\n The access code…` や `…exhausted.  The access
code…` のように、余分な半角空白が入ってしまう (`corpus/synth.py` の地の文は、
行の先頭に空白を置かず、半角空白が 2 つ連続することもない。
`tests/unit/test_corpus_needle.py` の
`test_no_line_in_haystack_starts_with_a_space` と
`test_no_double_space_anywhere_in_haystack` が確かめる)。
`_join_with_single_space` は、継ぎ目のどちらかがすでに空白なら、空白を足さない。

## 設計上の決定: 同じ `(seed, target_tokens, index)` では `depth_pct` を
## またいで、正解と干し草の土台を変えない

design.md はこの点について沈黙している。この module は、狙った位置の効果**だけ**
を切り分けられるように、`depth_pct` を、正解・問い・干し草の土台の生成には
一切使わない (種の導出にも混ぜない)。使うのは、挿入する位置を選ぶ最後の一歩
だけである。同じ `index` の下で、5 つの `depth_pct` (0、25、50、75、100) を
測るとき、正解や土台まで違えてしまうと、位置による差なのか、たまたま選ばれた
正解や土台の違いによる差なのかが区別できなくなる。一方で、`index` を変えれば
(品質の設定の `trials_per_cell`、同じ長さ×深さの繰り返しの試行)、正解も土台も
違う内容になる (試行どうしが丸ごと同じ課題を繰り返さないようにするため)。

## 正解の書式 (corpus が生成しうる書式との衝突を避ける)

正解は、固定の文字どおりの接頭辞 `SIGIL-` に、32 ビットの乱数を 8 桁の 16 進数
(大文字) にしたものを続けた文字列 (例: `SIGIL-4F82A1C9`)。`corpus/synth.py` の
型紙 (散文・コード・ログ、英語・日本語のどれも) は、`SIGIL-` という文字列を
一度も生成しない (`tests/unit/test_corpus_needle.py` が、複数の種と種類で
確かめている)。したがって、この接頭辞を含む文字列は、干し草の中に挿入した
1 箇所にしか現れない (正解が「言い当てられる」のではなく「探し出せる」ことを
保証する)。32 ビットの乱数 (2^32 通り) は、推測されない十分な情報量を持つ。
`scoring/needle.py` は、この `PREFIX-16進` という書式を前提にした、大量の候補を
並べる応答への対応を行う (module 側の docstring を参照)。

## `make_needle_case` の署名 (design.md corpus、および task 6.4 の手直し)

```
def make_needle_case(
    target_tokens: int, depth_pct: int, index: int, seed: int,
    *, chars_per_token: float | None = None,
) -> NeedleCase: ...
```

design.md の元の署名 (`chars_per_token` なし) は `make_tool_task`、
`build_conversation` と同様、比を引数に取らない。既定 (`chars_per_token=None`)
では、これまでどおり `Profile` の既定の比
(`chars_per_token` の default_factory と同じ値) を使う (golden なハッシュは
`chars_per_token` を省いた呼び出しと変わらない)。

task 6.4 の手直し (レビュー第 1 ラウンドの SHOULD FIX 指摘 3): 校正済みの比
(`bench calibrate` で測り直した値。既定の 4.0 と違うことがある) を無視すると、
狙った `target_tokens` が実際のトークン数と大きくずれ、長い条件 (128k など) が
対象サーバーの入力の上限を超えかねない。そこで、キーワード専用の
`chars_per_token` を追加した (位置引数の並びは変えていないので、design.md の
元の呼び出し方はそのまま動く)。品質の検査のまとまり (task 6.7) は、これに
`Profile.chars_per_token[ContentKind.PROSE_EN]` (校正済みの値) を渡すこと。
渡す値は有限かつ 0 より大きい必要があり、そうでなければ `ValueError` になる。

依存するのは標準ライブラリと `bench_harness.types`、`bench_harness.corpus.synth`
だけ。ほかの `bench_harness` の module (`client`、`suites`、`runner`、
`analysis`、`scoring`) は読み込まない (design.md の依存の向き)。
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Final

from bench_harness.corpus.synth import TemplateCorpus
from bench_harness.types import ContentKind, NeedleCase, Profile

__all__ = ["make_needle_case"]

_FACT_PURPOSE: Final[str] = "needle_fact"
"""コード名と正解を作る種の名前空間 (`_derive_seed` に渡す)。"""

_HAYSTACK_PURPOSE: Final[str] = "needle_haystack"
"""干し草の土台を作る種の名前空間。`_FACT_PURPOSE` と分けることで、コード名/
正解の乱数の消費と、干し草の乱数の消費が、互いに影響しない。"""

_MIN_BASE_TOKENS: Final[int] = 30
"""干し草の土台に割り当てるトークン数の狙いの下限 (`suites/prefill.py` の
`_MIN_DOCUMENT_TOKENS` と同じ考え方の安全弁。通常の長さ (500 トークン以上)
では効かない)。"""

_ANSWER_PREFIX: Final[str] = "SIGIL-"
"""正解の、固定の文字どおりの接頭辞。module の docstring を参照。"""

_ANSWER_HEX_DIGITS: Final[int] = 8
"""正解の 16 進部分の桁数 (32 ビット、2^32 通りの値)。"""

_DEFAULT_PROFILE: Final[Profile] = Profile(name="corpus.needle:default-chars-per-token")
"""`chars_per_token` の既定値だけを取り出すための、名前ばかりの `Profile`。
module の docstring「`make_needle_case` の署名」を参照。"""

_CODENAME_ADJECTIVES: Final[tuple[str, ...]] = (
    "Amber",
    "Cobalt",
    "Violet",
    "Slate",
    "Ember",
    "Quartz",
    "Coral",
    "Onyx",
    "Copper",
    "Ivory",
)
_CODENAME_NOUNS: Final[tuple[str, ...]] = (
    "Falcon",
    "Harbor",
    "Lantern",
    "Meridian",
    "Orbit",
    "Summit",
    "Thistle",
    "Vertex",
    "Wren",
    "Zephyr",
)
"""コード名の語彙。すべて架空 (既存の製品やデータの引き写しではない)。"""

_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(r"\. |。|\n\n|\n")
"""文・段落の切れ目 (task 2.5 の実装メモ: `. `、`。`、改行を目印に、この module
の側で切り分ける)。段落の区切り (`\\n\\n`) を、単独の改行より先に試す (2 つ
並んだ改行を、1 つの切れ目として扱うため)。"""


# --- 種を作る (`hash()` を使わない。corpus/tools.py と同じ形) -----------------


def _encode_part(part: str | int) -> bytes:
    """種の部品を、型と長さを含めて一意に符号化する (`corpus/tools.py` と同じ形)。"""
    if isinstance(part, str):
        data = part.encode("utf-8")
        return b"s" + str(len(data)).encode("ascii") + b":" + data
    data = str(part).encode("ascii")
    return b"i" + str(len(data)).encode("ascii") + b":" + data


def _derive_seed(seed: int, index: int, purpose: str) -> int:
    """`(seed, index, purpose)` から、決定的な `random.Random` の種を作る。"""
    parts: tuple[str | int, ...] = (seed, index, purpose)
    digest = hashlib.sha256(b"".join(_encode_part(part) for part in parts)).digest()
    return int.from_bytes(digest[:8], "big")


# --- コード名と正解 -----------------------------------------------------------


def _make_codename(rng: random.Random) -> str:
    adjective = rng.choice(_CODENAME_ADJECTIVES)
    noun = rng.choice(_CODENAME_NOUNS)
    number = rng.randint(1, 99)
    return f"{adjective}-{noun}-{number}"


def _make_answer(rng: random.Random) -> str:
    value = rng.getrandbits(_ANSWER_HEX_DIGITS * 4)
    return f"{_ANSWER_PREFIX}{value:0{_ANSWER_HEX_DIGITS}X}"


def _needle_sentence(codename: str, answer: str) -> str:
    """挿入する文。引用符を使わない (module の docstring「挿入する文の書式」を参照)。"""
    return f"The access code for {codename} is {answer}."


def _question_for(codename: str) -> str:
    return (
        f'According to the document above, what is the access code for "{codename}"? '
        "Answer with the code only."
    )


# --- 継ぎ目の空白を二重にしない -----------------------------------------------


def _join_with_single_space(before: str, after: str) -> str:
    """`before` と `after` を、区切りの空白をちょうど 1 つだけ挟んでつなぐ。

    どちらかがすでに空白 (半角空白、改行) で終わる/始まるなら、区切りの空白を
    足さない (二重にならないようにする)。module の docstring「挿入する文の
    書式」を参照。
    """
    if not before:
        return after
    if not after:
        return before
    if before[-1].isspace() or after[0].isspace():
        return before + after
    return f"{before} {after}"


# --- 文・段落の切れ目に挿入する ------------------------------------------------


def _sentence_boundaries(text: str) -> list[int]:
    """`text` の中の、文または段落の切れ目の位置 (その直後の位置) の一覧。

    先頭 (0) と末尾 (`len(text)`) も、それ自体が有効な挿入点として含める。
    """
    boundaries = {0, len(text)}
    for match in _BOUNDARY_RE.finditer(text):
        boundaries.add(match.end())
    return sorted(boundaries)


def _closest_boundary(boundaries: list[int], target: int) -> int:
    return min(boundaries, key=lambda boundary: abs(boundary - target))


def _insert_needle(base: str, sentence: str, depth_pct: int) -> tuple[str, int]:
    """`base` (干し草の土台) に `sentence` を挿入し、`(挿入後の全文, 開始位置)` を返す。

    `depth_pct` が 0 なら土台の直前、100 なら土台の直後にそのまま挿入する
    (module の docstring「埋める位置」を参照)。それ以外は、土台のその割合に
    最も近い、文または段落の切れ目に挿入する。継ぎ目の空白は
    `_join_with_single_space` に任せ、二重にならないようにする。
    """
    if depth_pct <= 0:
        combined = _join_with_single_space(sentence, base)
        return combined, 0
    if depth_pct >= 100:
        combined = _join_with_single_space(base, sentence)
        return combined, len(combined) - len(sentence)
    target = round(depth_pct / 100 * len(base))
    offset_in_base = _closest_boundary(_sentence_boundaries(base), target)
    prefix, suffix = base[:offset_in_base], base[offset_in_base:]
    with_sentence = _join_with_single_space(prefix, sentence)
    combined = _join_with_single_space(with_sentence, suffix)
    return combined, len(with_sentence) - len(sentence)


# --- 公開の関数 -----------------------------------------------------------------


def make_needle_case(
    target_tokens: int,
    depth_pct: int,
    index: int,
    seed: int,
    *,
    chars_per_token: float | None = None,
) -> NeedleCase:
    """`(seed, index)` から決まる、正解が 1 つに決まる探す課題を作る (design.md corpus, 5.3)。

    `target_tokens` に相当する文字数の干し草を作り、`depth_pct` (0〜100) の
    位置に、正解を含む 1 文を埋め込む。module の docstring の決まりのとおり、
    正解・問い・干し草の土台は `depth_pct` に依存しない (位置だけを変える)。

    `chars_per_token` (キーワード専用) は、英語の散文の 1 トークンあたりの
    文字数。省略または `None` なら、`Profile` の既定の比を使う (module の
    docstring「`make_needle_case` の署名」を参照)。
    """
    if target_tokens <= 0:
        raise ValueError(
            f"make_needle_case: target_tokens は正の整数である必要がある "
            f"(target_tokens={target_tokens})"
        )
    if not 0 <= depth_pct <= 100:
        raise ValueError(
            f"make_needle_case: depth_pct は 0 以上 100 以下である必要がある "
            f"(depth_pct={depth_pct})"
        )
    if index < 0:
        raise ValueError(f"make_needle_case: index は 0 以上である必要がある (index={index})")
    if chars_per_token is not None and (
        isinstance(chars_per_token, bool)  # True は 1.0 として通ってしまうので、先に弾く
        or not math.isfinite(chars_per_token)
        or chars_per_token <= 0
    ):
        raise ValueError(
            f"make_needle_case: chars_per_token は有限の正の数である必要がある "
            f"(chars_per_token={chars_per_token})"
        )

    fact_seed = _derive_seed(seed, index, _FACT_PURPOSE)
    haystack_seed = _derive_seed(seed, index, _HAYSTACK_PURPOSE)

    rng = random.Random(fact_seed)
    codename = _make_codename(rng)
    answer = _make_answer(rng)
    sentence = _needle_sentence(codename, answer)
    question = _question_for(codename)

    ratio = (
        _DEFAULT_PROFILE.chars_per_token[ContentKind.PROSE_EN]
        if chars_per_token is None
        else chars_per_token
    )
    target_chars = target_tokens * ratio
    needle_chars = len(sentence) + 1  # 挿入時に足しうる区切りの空白 1 文字ぶんの見積もり
    base_target_chars = max(_MIN_BASE_TOKENS * ratio, target_chars - needle_chars)
    base_target_tokens = max(_MIN_BASE_TOKENS, round(base_target_chars / ratio))

    profile_chars_per_token = dict(_DEFAULT_PROFILE.chars_per_token)
    profile_chars_per_token[ContentKind.PROSE_EN] = ratio
    corpus = TemplateCorpus(profile_chars_per_token)
    base_haystack = corpus.prose("en", base_target_tokens, haystack_seed)

    haystack, offset = _insert_needle(base_haystack, sentence, depth_pct)

    return NeedleCase(
        haystack=haystack,
        question=question,
        answer=answer,
        target_tokens=target_tokens,
        depth_pct=depth_pct,
        index=index,
        seed=seed,
        inserted_char_offset=offset,
    )
