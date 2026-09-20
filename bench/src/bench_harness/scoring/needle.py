"""長い入力から情報を探す課題の採点 (task 6.4: scoring/needle, design.md scoring/needle, 5.3)。

`score_needle(response_text, task)` は、応答の文字列 (すでに取り出した本文。
`StreamResult` そのものではない) に、課題 (`NeedleCase`) の正解が、数字/英字の
境界を守った形でちょうど現れるかどうかで、`CORRECT` か `INCORRECT` を返す。
要求そのものの失敗を `NOT_SCORED` にする判断は、呼び出し側 (品質の検査の
まとまり、task 6.7) の責任であり、この module は関わらない (5.7 は要求の失敗を
不正解と分けることを求めるが、それは呼び出し側が `score_needle` を呼ばずに
`NOT_SCORED` を付けることで満たされる)。

## 正規化の決め方 (design.md が沈黙しているため、ここで決める)

設計は、探す課題の採点の正規化 (大小文字、空白) を明記していない。この module
は、正解を水増ししない側 (より厳しい側) を選ぶ。一致は**文字どおり
(literal)、大小文字を区別する**。

- **大小文字は区別する (case-sensitive)。** 折りたたみ (case-folding) はしない。
  正解には 16 進の英字 (`A`〜`F`) を含むため、大小文字を区別しないと、たまたま
  別の大文字/小文字の並びが正解と紛れる余地が広がる。「引用符や逆引用符、
  句読点で囲う」ような書式の違いは罰さないが (下記)、文字そのものの大小の
  違いは、値の一部とみなして区別する
- **全角文字、Unicode の異体のハイフンは、別の文字として扱う。** 折りたたみを
  一切しないため、正解を小文字にした応答 (`sigil-...`)、全角のラテン文字にした
  応答 (`ＳＩＧＩＬ-...`)、ハイフンを Unicode の異体 (U+2010 HYPHEN、U+FF0D
  FULLWIDTH HYPHEN-MINUS など) に置き換えた応答は、正解の文字列とバイト単位で
  一致しないため、**設計どおり不正解になる**。これは意図した挙動であり、
  取りこぼしではない (`tests/unit/test_scoring_needle.py` の
  `test_lowercased_fullwidth_and_unicode_hyphen_variants_are_incorrect_by_design`
  が確かめる。使い方の文書 (8.5) にも明記すること)
- **空白は、特別な折りたたみをしない。** 正解の文字列をそのまま
  部分文字列として探すので、正解の前後にある半角の空白・引用符・逆引用符・
  括弧・句読点 (`.`、`,`、`!` など) は、そもそも判定に影響しない (下記「数字/
  英字の境界」を参照)。正解の**内部**の空白の有無や数は区別する (正解に空白は
  含まれない設計なので、実質的には影響しない)
- 応答が空文字列、巨大な文字列、制御文字や置き換え文字 (U+FFFD) を含んでいても、
  例外は投げない (`str.__contains__`/`re` の通常の扱いに任せる。特別な
  サニタイズはしない)

## 数字/英字の境界 (誤って正解にしない。ASCII に限る)

正解の文字列を、そのまま `str` の部分文字列として探すと、`4821` が `14821` や
`48210` の一部として誤って「見つかった」ことになってしまう。これを防ぐため、
正解の前後が、境界の文字 (`_BOUNDARY_CHARS`: 数字・英字・アンダースコア・
ハイフン) で続いていないことを、`(?<![境界の文字])` / `(?![境界の文字])` の
先読み・後読みで確かめる。

境界の文字の集合は、**ASCII に限った** `0-9A-Za-z_-` であり、Python の `\\w`
(既定では Unicode 対応で、漢字・ひらがな・カタカナも「語の文字」に含む) は
使わない (`re.ASCII` フラグを立てて `\\w` を使う方法も採らず、文字の集合を
明示的に書き下す。理由は下記)。`\\b` (Python の単語の境界) も使わない――
`\\b` は `\\w` を基準にしており、ハイフンを「語の文字」に含めないため、
`corpus/needle.py` の正解の書式 (`SIGIL-4F82A1C9` のような、ハイフンを含む
識別子) の境界を、`\\b` だけでは正しく扱えない。

**ASCII に限る理由 (レビュー第 1 ラウンドの BLOCKING 指摘 2):** 応答が日本語や
中国語で、正解の直前・直後に空白を置かずに続く場合 (`"コードは
SIGIL-1A2B3C4Dです"` のように、空白がない)、Unicode 対応の `\\w` では、
「は」や「で」も「語の文字」とみなされてしまい、境界の条件 (前後が境界の文字で
**ない**こと) を満たせず、正解が見つからなくなる。これは、日本語や中国語で
答えるモデルに対して、常に偽陰性になる致命的な壊れ方だった。CJK の文字は、
数字や ASCII の英字と地続きに「同じ識別子の続き」になることはない (`SIGIL-`
の 8 桁の 16 進数が、そのまま「あいう」に続いて 1 つの識別子になることは
あり得ない) ため、境界の文字から外してよい。全角の句読点・鉤括弧 (`「」`) は、
そもそも `\\w` にも ASCII の境界の文字にも含まれないため、この変更の前から
問題なく扱えていた。

この境界の定義のおかげで、正解を引用符・逆引用符・括弧・半角の句読点で
囲んだだけの応答 (`` `SIGIL-4F82A1C9` `` や `"4821"` など) や、空白なしで
CJK の文に続く応答は、境界の外側の文字が境界の文字ではないので、そのまま
正解として認められる (「害のない書式」を罰さない)。

## 大量に候補を並べる (shotgun) 応答への対応 (design.md が沈黙しているため、ここで決める)

design.md は、複数の候補を並べる応答の扱いを明記していない。この module は、
より厳しい側 (言い当てを許さない側) を選ぶが、レビュー第 1 ラウンドの
BLOCKING 指摘 1 を受けて、**候補とみなす条件を、正解の実際の書式に絞る**。

- 正解が `PREFIX-16進` の書式 (`re.fullmatch(r"([A-Za-z]+-)([0-9A-F]+)",
  answer)` に一致する。`corpus/needle.py` の正解 `SIGIL-4F82A1C9` はこれに
  当てはまる) のときだけ、shotgun の判定を行う: 応答の中に、正解と**同じ接頭辞
  (`SIGIL-` など)・同じ 16 進の桁数**の、境界を守った語 (正解自身を含む) が
  **2 つ以上、種類として**現れていたら、言い当てたのではなく候補を並べて
  (shotgun) 当てようとしたとみなし、正解が含まれていても `INCORRECT` にする。
  正解を 2 回以上繰り返すだけ (種類として 1 つ) は shotgun とは見なさない
- 正解がこの書式に**一致しない**とき (数字だけの正解など) は、shotgun の判定を
  一切行わない。境界を守った正解が見つかれば、それだけで `CORRECT` にする

**旧い実装との違い (レビュー第 1 ラウンドの指摘):** 以前は「正解と**同じ長さ**の、
`[\\w-]` だけでできた語」を候補とみなしていたため、正解 (`SIGIL-4F82A1C9`、
14 文字) と偶然同じ長さの、ふつうの英単語 (`implementation` なども 14 文字) や、
コード名 (`Copper-Wren-63` のような、たまたま 14 文字になる組み合わせ。
コード名の約 15.6% がこの長さになる) が応答に含まれるだけで、正直な正解を
`INCORRECT` にしてしまっていた (レビューの測定で、生成した課題の 16%、手で
書いた誠実な応答の 8/30 が誤って不正解になった)。新しい実装は、正解の接頭辞と
16 進の桁数が実際に一致する語だけを候補にするため、`SIGIL-` を含まない、
ふつうの英単語やコード名は、そもそも候補にならない
(`tests/unit/test_scoring_needle.py` の
`test_ordinary_same_length_english_word_does_not_trigger_shotgun` が確かめる)。
一方で、正解と同じ接頭辞・同じ桁数の別の値を並べる、本物の shotgun (`SIGIL-
00000000`、`SIGIL-FFFFFFFF` など) は、引き続き `INCORRECT` にする
(`test_shotgun_requires_matching_prefix_and_hex_length` が確かめる)。

依存するのは標準ライブラリと `bench_harness.types` だけ。`bench_harness.corpus`
は読み込まない (design.md の import の制約: 採点は corpus を読み込まない)。
"""

from __future__ import annotations

import re
from typing import Final

from bench_harness.types import NeedleCase, QualityOutcome, QualityVerdict

__all__ = ["score_needle"]

_SHOTGUN_CANDIDATE_LIMIT: Final[int] = 1
"""正解と同じ接頭辞・同じ 16 進の桁数の、種類としての候補の数が、この数を
超えたら shotgun とみなす。module の docstring「大量に候補を並べる応答への
対応」を参照。"""

_BOUNDARY_CHARS: Final[str] = "0-9A-Za-z_-"
"""数字/英字の境界に使う、ASCII に限った「境界の文字」の定義。module の
docstring「数字/英字の境界」を参照。CJK の文字は含めない。"""

_PREFIX_HEX_RE: Final[re.Pattern[str]] = re.compile(r"([A-Za-z]+-)([0-9A-F]+)")
"""正解が `PREFIX-16進` の書式かどうかを判定する (`fullmatch` で使う)。
`corpus/needle.py` の `SIGIL-4F82A1C9` の形。group(1) が接頭辞 (ハイフンを
含む)、group(2) が 16 進の部分。"""


def _boundary_pattern(literal: str) -> re.Pattern[str]:
    """`literal` を、数字/英字の境界を守って探す正規表現 (module の docstring
    「数字/英字の境界」を参照)。"""
    return re.compile(rf"(?<![{_BOUNDARY_CHARS}]){re.escape(literal)}(?![{_BOUNDARY_CHARS}])")


def _same_format_candidates(text: str, prefix: str, hex_digits: int) -> set[str]:
    """`text` の中の、`prefix` に続けて `hex_digits` 桁の 16 進数が来る、
    境界を守った語の集合 (module の docstring「大量に候補を並べる応答への
    対応」を参照)。
    """
    pattern = re.compile(
        rf"(?<![{_BOUNDARY_CHARS}]){re.escape(prefix)}[0-9A-F]{{{hex_digits}}}"
        rf"(?![{_BOUNDARY_CHARS}])"
    )
    return {match.group(0) for match in pattern.finditer(text)}


def score_needle(response_text: str, task: NeedleCase) -> QualityVerdict:
    """探す課題の応答を採点する (design.md corpus/scoring/needle, 5.3)。

    正解 (`task.answer`) が、数字/英字の境界を守った形で応答に現れなければ
    `INCORRECT`。現れても、正解が `PREFIX-16進` の書式で、かつ同じ接頭辞・
    同じ桁数の候補が複数種類あれば (module の docstring「大量に候補を並べる
    応答への対応」) `INCORRECT`。どちらでもなければ `CORRECT`。どんな
    `response_text` に対しても例外を投げない。
    """
    if not task.answer:
        return QualityVerdict(
            task="needle", outcome=QualityOutcome.INCORRECT, detail="empty_answer"
        )

    if not _boundary_pattern(task.answer).search(response_text):
        return QualityVerdict(
            task="needle", outcome=QualityOutcome.INCORRECT, detail="answer_not_found"
        )

    format_match = _PREFIX_HEX_RE.fullmatch(task.answer)
    if format_match is not None:
        prefix, hex_part = format_match.group(1), format_match.group(2)
        candidates = _same_format_candidates(response_text, prefix, len(hex_part))
        if len(candidates) > _SHOTGUN_CANDIDATE_LIMIT:
            return QualityVerdict(
                task="needle",
                outcome=QualityOutcome.INCORRECT,
                detail=f"shotgun_candidates={len(candidates)}",
            )

    return QualityVerdict(task="needle", outcome=QualityOutcome.CORRECT, detail="answer_found")
