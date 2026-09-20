"""出力が壊れている疑いの検出 (task 2.6: scoring/sanity, design.md scoring/sanity, 10.7)。

本文に置き換え文字 (U+FFFD、文字化けの跡) があるか、決まった長さ以上の同じ並びが
決まった回数以上、間を空けずに続いているかを判定する。どちらも純粋な関数で、
時計も環境変数も読まない。依存するのは標準ライブラリと `bench_harness.types` だけ
(design.md の import 制約)。

## `REPETITION_LOOP` の見つけ方

定義: `text` が、長さ `repeat_min_chars` 以上の `unit` について
`unit * repeat_min_count` を含む。短い周期の繰り返しも、この定義に含まれる
(`"ab"` が 48 回続けば、12 文字の単位 `"abababababab"` が 8 回続いている)。

言い換えると、ある単位の長さ `L` (`L >= repeat_min_chars`) について、
`text[i] == text[i - L]` が成り立つ位置 `i` が、`L * (repeat_min_count - 1)` 個
以上続いていればよい (その区間と、手前の `L` 文字を合わせると、`unit` が
`repeat_min_count` 回並ぶ)。近道 (窓のハッシュから周期の候補を推す、など) は
使わず、`L` を 1 つずつ調べる。定義と厳密に一致する (取りこぼしも、誤って印を
付けることもない。`tests/unit/test_scoring_sanity.py` が、定義どおりの総当たりと
突き合わせて確かめている)。

速さは、次の 2 つで保つ。

- **錨の位置だけを見る**: 必要な長さを `K = L * (repeat_min_count - 1)` とすると、
  長さ `K` 以上の一致の区間は、必ず `K` の倍数の位置を含む。そこで、`K` の
  倍数の位置 `q` で `text[q] == text[q - L]` が成り立つところだけを調べ、そこから
  左右に一致を伸ばす (伸ばすのは、スライスの比較を倍々に広げて行う)。調べる位置は
  `L` ごとに `n / K` 個なので、全体で `n * log(L の上限) / (repeat_min_count - 1)`
  程度で済む
- **印を付けない区間を飛ばす**: 空白と、英数字でない 1 種類の文字だけでできた
  長い区間 (改行が 100 万個、など) では、どの `L` でも一致が続く。錨の位置 `q` と、
  その手前の `L` 文字が、そうした区間にすっぽり入っているなら、`q` を通る繰り返しの
  単位は必ず「印を付けない並び」になるので、区間の終わりまで飛ばす

調べる単位の長さには、上限 `MAX_UNIT_CHARS` (4096 文字) を置く。これより長い
単位の繰り返しは見つけない (既定の 8 回なら、3 万文字を超える繰り返しに当たる)。

## 印を付けない並びの決まり

繰り返しの単位 (`text[start : start + period]`) に含まれる、空白でない文字の
種類で決める。

1. 空白でない文字が 1 つもない (単位が空白・タブ・改行だけでできている) →
   印を付けない
2. 空白でない文字がちょうど 1 種類で、それが英数字でない (`not ch.isalnum()`。
   句読点や記号、罫線素片の文字など) → 印を付けない。空白と混ざっていてもよい
   (例: Markdown の区切り線 `----------`、表の区切り `| | | |`、`=  =  =`)
3. 空白でない文字がちょうど 1 種類で、それが英数字である (`ch.isalnum()` が真。
   ラテン文字、数字、かな、漢字などを含む) → **印を付ける** (例:
   `"aaaaaaaa..."`、`"的的的的..."`、`"00000000..."`)
4. 空白でない文字が 2 種類以上ある → **印を付ける** (これまでどおり、長さと
   回数の条件を満たせば)

1、2 の決まりは、Markdown の区切り線、深い字下げ、空行の連続、罫線素片による
アスキーアートを、機械的な誤検出から守るために設ける。一方で、3 のとおり、
同じ英数字がひたすら続く出力 (壊れたモデルの典型的な症状の 1 つ) は、この
検査でいちばん見つけたいものなので、単一の文字であっても印を付ける。2 と 3
の違いは「その 1 種類の文字が英数字かどうか」だけであり、パディングや罫線は
英数字でない記号を使うことが多い一方、モデルの出力が壊れて同じ字が続くのは
たいてい英数字 (文字や数字) であるという、実際の使われ方の違いに対応させて
いる。見つけ方が機械的なため、この決まりを入れてもなお、コードやログのような
繰り返しの多い出力を誤って拾う余地は残る。そのため、印の付いた試行を集計から
除外はしない (design.md scoring/sanity): 条件ごとの件数だけを示し、計測者が
生データを見て判断する。
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Callable, Sequence
from itertools import compress, count
from operator import eq
from typing import Final

from bench_harness.types import StreamResult, TrialFlag

__all__ = [
    "collect_text",
    "collect_tool_input_text",
    "detect_output_anomalies",
]

_REPLACEMENT_CHAR: Final[str] = "�"

MAX_UNIT_CHARS: Final[int] = 4096
"""調べる繰り返しの単位の長さの上限。これより長い単位の繰り返しは見つけない。"""

_SOFT_RUN: Final[re.Pattern[str]] = re.compile(r"[\W_]+")
"""英数字でない文字 (空白を含む) の連続。正規表現の語の文字は、`str.isalnum()` と `_`。"""


def detect_output_anomalies(
    text: str, repeat_min_chars: int, repeat_min_count: int
) -> list[TrialFlag]:
    """本文が壊れている疑いを判定する (design.md scoring/sanity, 10.7)。

    - `TrialFlag.REPLACEMENT_CHAR`: `text` に U+FFFD が 1 つでもあれば付く
    - `TrialFlag.REPETITION_LOOP`: `repeat_min_chars` 文字以上の同じ並びが、間を
      空けずに `repeat_min_count` 回以上続けば付く (ただし、この module の
      docstring にある「印を付けない並びの決まり」に当てはまる区間は除く)

    返す列は、付いた印だけを `TrialFlag` の定義順 (`REPLACEMENT_CHAR` →
    `REPETITION_LOOP`) で並べ、同じ印を 2 つ以上入れない。
    """
    if repeat_min_chars < 1:
        raise ValueError(
            "detect_output_anomalies: repeat_min_chars は 1 以上である必要がある "
            f"(repeat_min_chars={repeat_min_chars})"
        )
    if repeat_min_count < 2:
        raise ValueError(
            "detect_output_anomalies: repeat_min_count は 2 以上である必要がある "
            f"(repeat_min_count={repeat_min_count})"
        )
    flags: list[TrialFlag] = []
    if _REPLACEMENT_CHAR in text:
        flags.append(TrialFlag.REPLACEMENT_CHAR)
    if _has_repetition_loop(text, repeat_min_chars, repeat_min_count):
        flags.append(TrialFlag.REPETITION_LOOP)
    return flags


def collect_text(result: StreamResult) -> str:
    """本文 (`text` と `thinking` のブロック) を、ブロックの順につなげる (design.md
    scoring/sanity: 「本文 (thinking を含む)」)。

    `tool_use` ブロックの生の引数 (`tool_input_raw`) は含まない。ツールの引数の
    JSON は、正当な理由で入れ子や配列などの繰り返しの多い構造になりうるため、
    `REPETITION_LOOP` の判定に混ぜない。引数の文字列の置き換え文字を確かめたい
    ときは、`collect_tool_input_text` を合わせて呼ぶこと。
    """
    return "".join(
        block.text
        for block in result.blocks
        if block.type in ("text", "thinking") and block.text is not None
    )


def collect_tool_input_text(result: StreamResult) -> str:
    """`tool_use` ブロックの生の引数文字列 (`tool_input_raw`) を、ブロックの順に
    つなげる (`collect_text` の docstring を参照)。

    置き換え文字 (U+FFFD) の検出だけに使う想定で、`REPETITION_LOOP` の判定
    (`detect_output_anomalies` を JSON の引数にそのまま使うこと) は勧めない。
    """
    return "".join(
        block.tool_input_raw
        for block in result.blocks
        if block.type == "tool_use" and block.tool_input_raw is not None
    )


# --- REPETITION_LOOP の実装 ------------------------------------------------


def _has_repetition_loop(text: str, repeat_min_chars: int, repeat_min_count: int) -> bool:
    """`text` のどこかに、印を付ける条件を満たす繰り返しがあるかを返す。"""
    return _scan_for_qualifying_run(
        text,
        repeat_min_chars,
        repeat_min_count,
        _run_not_excluded,
        _excluded_segments(text, repeat_min_chars),
    )


def _run_not_excluded(text: str, start: int, period: int) -> bool:
    """繰り返しの単位が「印を付けない並びの決まり」に当てはまらなければ True。"""
    unit = text[start : start + period]
    non_whitespace = {ch for ch in unit if not ch.isspace()}
    if not non_whitespace:
        return False  # 空白だけの単位
    if len(non_whitespace) == 1:
        (only,) = non_whitespace
        if not only.isalnum():
            return False  # 空白と、英数字でない 1 種類の文字だけの単位
    return True


def _scan_for_qualifying_run(
    text: str,
    repeat_min_chars: int,
    repeat_min_count: int,
    accept: Callable[[str, int, int], bool],
    excluded_segments: Sequence[tuple[int, int]] = (),
) -> bool:
    """長さと回数の条件を満たし、`accept` が認める繰り返しがあるかを返す。

    `accept(text, start, unit_len)` は、`text[start:]` から始まる長さ `unit_len` の
    単位の繰り返しに、印を付けてよいかを返す。`excluded_segments` は、`accept` が
    必ず偽を返すとわかっている区間 (`_excluded_segments`) で、飛ばすためだけに使う。
    空でも結果は変わらない (遅くなるだけ)。
    """
    n = len(text)
    segment_starts = [start for start, _ in excluded_segments]
    max_unit = min(n // repeat_min_count, MAX_UNIT_CHARS)
    for unit_len in range(repeat_min_chars, max_unit + 1):
        need = unit_len * (repeat_min_count - 1)
        # 錨の位置 (need の倍数) の文字と、その unit_len 文字前の文字を、まとめて取り出す
        anchors = text[need::need]
        earlier = text[need - unit_len :: need]
        resume_at = 0
        for index in compress(count(), map(eq, anchors, earlier)):
            q = need * (index + 1)
            if q < resume_at:
                continue
            segment_end = _covering_segment_end(excluded_segments, segment_starts, q, unit_len)
            if segment_end is not None:
                resume_at = segment_end
                continue
            low, high = _match_run(text, q, unit_len, need)
            if high - low >= need and accept(text, low - unit_len, unit_len):
                return True
            resume_at = high
    return False


def _match_run(text: str, q: int, unit_len: int, cap: int) -> tuple[int, int]:
    """`q` を含む、`text[i] == text[i - unit_len]` が続く区間 `[low, high)` を返す。

    左右とも、`q` から `cap` 文字までしか伸ばさない (それだけあれば、条件を満たすか
    どうかは決まる)。`text[q] == text[q - unit_len]` は、呼ぶ側が確かめてある。
    """
    right = _common_prefix(text, q, q - unit_len, min(cap, len(text) - q))
    left = _common_suffix(text, q, q - unit_len, min(cap, q - unit_len))
    return q - left, q + right


def _common_prefix(text: str, i: int, j: int, cap: int) -> int:
    """`text[i:]` と `text[j:]` の、先頭の一致の長さ (`cap` まで)。"""
    done = 0
    size = 1
    while done < cap:
        step = min(size, cap - done)
        if text[i + done : i + done + step] == text[j + done : j + done + step]:
            done += step
            size *= 2
            continue
        matched, mismatched = 0, step  # step 文字のどこかで食い違う。二分で探す
        while mismatched - matched > 1:
            mid = (matched + mismatched) // 2
            if text[i + done : i + done + mid] == text[j + done : j + done + mid]:
                matched = mid
            else:
                mismatched = mid
        return done + matched
    return done


def _common_suffix(text: str, i: int, j: int, cap: int) -> int:
    """`text[:i]` と `text[:j]` の、末尾の一致の長さ (`cap` まで)。"""
    done = 0
    size = 1
    while done < cap:
        step = min(size, cap - done)
        if text[i - done - step : i - done] == text[j - done - step : j - done]:
            done += step
            size *= 2
            continue
        matched, mismatched = 0, step
        while mismatched - matched > 1:
            mid = (matched + mismatched) // 2
            if text[i - done - mid : i - done] == text[j - done - mid : j - done]:
                matched = mid
            else:
                mismatched = mid
        return done + matched
    return done


def _excluded_segments(text: str, min_len: int) -> list[tuple[int, int]]:
    """空白と、英数字でない 1 種類の文字だけでできた、極大の区間を返す (始まりの順)。

    この区間の中に収まる単位は、必ず「印を付けない並び」になる。`min_len` 文字以下の
    区間は、飛ばす役に立たないので返さない。空白は、前後の 2 つの区間に同時に
    属しうる (`"--- ==="` の空白は、`"--- "` にも `" ==="` にも入る)。
    """
    segments: list[tuple[int, int]] = []
    for soft in _SOFT_RUN.finditer(text):
        if soft.end() - soft.start() <= min_len:
            continue
        start = soft.start()
        symbol: str | None = None
        whitespace_from = start  # 直前まで続いている空白の始まり
        for position in range(soft.start(), soft.end()):
            ch = text[position]
            if ch.isspace():
                continue_whitespace = position > soft.start() and text[position - 1].isspace()
                if not continue_whitespace:
                    whitespace_from = position
                continue
            if symbol is None or ch == symbol:
                symbol = ch
                whitespace_from = position + 1
                continue
            if position - start > min_len:
                segments.append((start, position))
            # 別の記号に変わった。直前の空白は、新しい区間にも入れる
            start = whitespace_from if whitespace_from <= position else position
            symbol = ch
            whitespace_from = position + 1
        if soft.end() - start > min_len:
            segments.append((start, soft.end()))
    return segments


def _covering_segment_end(
    segments: Sequence[tuple[int, int]], starts: Sequence[int], q: int, unit_len: int
) -> int | None:
    """`q` と、その手前の `unit_len` 文字が、印を付けない区間に収まるなら、区間の終わり。

    収まっていれば、`q` を通る長さ `unit_len` の単位は、その区間の文字だけでできて
    いるので、必ず「印を付けない並び」になる。区間の終わりまでの錨の位置も同じ。
    """
    index = bisect_right(starts, q) - 1
    for candidate in (index, index - 1):  # 空白を共有する、隣の区間も見る
        if candidate < 0:
            continue
        start, end = segments[candidate]
        if start <= q - unit_len and q < end:
            return end
    return None
