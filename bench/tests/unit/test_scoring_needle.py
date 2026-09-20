"""長い入力から情報を探す課題の採点の試験 (task 6.4: scoring/needle)。

課題の型 (`NeedleCase`) は、ほとんどの試験で手で作る (design.md「試験の例は
手で書く」の決まりにならう)。正解を含む/含まない応答、数字/英字の境界の誤検出
(`4821` が `14821` や `48210` で誤って正解にならない。CJK は境界の文字に数えない)、
大小文字の区別 (この module が選んだ、より厳しい側の規則)、引用符や句読点で
囲われた「害のない書式」を罰しないこと、`PREFIX-16進` の書式に限った、大量に
候補を並べる (shotgun) 応答への対応 (ふつうの同じ長さの英単語を誤って shotgun
扱いしない)、どんな応答文字列でも例外を投げないことを確かめる。round 2 の
2 つの試験 (末尾の「正直な応答の偽陰性がないこと」) だけは、`corpus/needle.py`
の生成器で本物の課題を作り、誠実な応答が必ず正解になることを、多くの組み合わせ
にわたって確かめる。
"""

from __future__ import annotations

import re
from typing import Final

import pytest

# `corpus.needle` は、round 2 の 2 つの試験 (誠実な応答に対する偽陰性がないこと)
# だけで、本物の課題を作るために使う。design.md の import の制約
# (「採点は corpus を読み込まない」) は `scoring/needle.py` 自身にかかるもので、
# この試験ファイルにはかからない (module の docstring を参照)。
from bench_harness.corpus.needle import make_needle_case
from bench_harness.scoring.needle import score_needle
from bench_harness.types import NeedleCase, QualityOutcome, QualityVerdict

# --- 助け -------------------------------------------------------------------

_DEFAULT_QUESTION: str = 'What is the code for "Amber-Falcon-7"?'


def _task(answer: str = "SIGIL-4F82A1C9", question: str = _DEFAULT_QUESTION) -> NeedleCase:
    return NeedleCase(
        haystack="filler haystack text. " * 5,
        question=question,
        answer=answer,
        target_tokens=100,
        depth_pct=50,
        index=0,
        seed=1,
        inserted_char_offset=10,
    )


_CODENAME_IN_QUESTION_RE: Final[re.Pattern[str]] = re.compile(r'"([^"]+)"')


def _codename_from_question(question: str) -> str:
    """`question` から、引用符で囲まれたコード名を取り出す (試験だけで使う。
    corpus/needle.py が作る本物の課題から、正直な応答を組み立てるために使う)。
    """
    match = _CODENAME_IN_QUESTION_RE.search(question)
    assert match is not None, f"問いの文からコード名を取り出せない: {question!r}"
    return match.group(1)


# --- 正解/不正解 ----------------------------------------------------------------


def test_correct_when_answer_present_verbatim() -> None:
    task = _task()
    verdict = score_needle(f"The access code is {task.answer}.", task)
    assert verdict.outcome == QualityOutcome.CORRECT
    assert verdict.kind == "quality"
    assert verdict.task == "needle"


def test_incorrect_when_answer_absent() -> None:
    task = _task()
    verdict = score_needle("I could not find any relevant code in the document.", task)
    assert verdict.outcome == QualityOutcome.INCORRECT


def test_incorrect_on_empty_response() -> None:
    task = _task()
    verdict = score_needle("", task)
    assert verdict.outcome == QualityOutcome.INCORRECT


def test_never_returns_not_scored() -> None:
    """要求そのものの失敗は、この module の責任範囲外 (呼び出し側が NOT_SCORED
    にする)。`score_needle` は、どんな応答文字列に対しても CORRECT か INCORRECT
    のどちらかしか返さない。
    """
    task = _task()
    for text in ("", "garbage", task.answer, "x" * 10_000):
        verdict = score_needle(text, task)
        assert verdict.outcome in (QualityOutcome.CORRECT, QualityOutcome.INCORRECT)


# --- 数字/英数字の境界 (誤って正解にしない) --------------------------------------


def test_boundary_rejects_prefix_extension() -> None:
    """`4821` は、より長い数字の並びの一部 (`14821`) では正解にならない。"""
    task = _task(answer="4821")
    verdict = score_needle("The recorded value was 14821 units.", task)
    assert verdict.outcome == QualityOutcome.INCORRECT


def test_boundary_rejects_suffix_extension() -> None:
    """`4821` は、より長い数字の並びの一部 (`48210`) では正解にならない。"""
    task = _task(answer="4821")
    verdict = score_needle("The recorded value was 48210 units.", task)
    assert verdict.outcome == QualityOutcome.INCORRECT


def test_boundary_accepts_exact_numeric_answer() -> None:
    task = _task(answer="4821")
    verdict = score_needle("The recorded value was 4821 units.", task)
    assert verdict.outcome == QualityOutcome.CORRECT


@pytest.mark.parametrize(
    "response",
    [
        "The value might be SIGIL-1A2B3C4D5 if I read it right.",  # 末尾に 1 桁余分
        "Maybe it is XSIGIL-1A2B3C4D, hard to tell.",  # 先頭に 1 文字余分
        "Could be SIGIL-1A2B3C4D-2, the second variant.",  # 末尾に「-2」余分
    ],
)
def test_boundary_rejects_prefix_hex_answer_extended_by_adjacent_characters(
    response: str,
) -> None:
    """`SIGIL-1A2B3C4D` (`PREFIX-16進` の書式) も、数字の答えと同じく、前後に
    英数字やハイフンが続く「一部」としては正解にならない (レビュー第 1 ラウンドの
    指摘 2 で見つかった、境界の再確認。round 2 の必須試験 (e))。
    """
    task = _task(answer="SIGIL-1A2B3C4D")
    verdict = score_needle(response, task)
    assert verdict.outcome == QualityOutcome.INCORRECT


# --- 害のない書式は罰しない ------------------------------------------------------


@pytest.mark.parametrize(
    "wrap",
    [
        '"{}"',
        "`{}`",
        "**{}**",
        "({})",
        "{}.",
        "{},",
        "{}!",
    ],
)
def test_surrounding_punctuation_and_quoting_is_tolerated(wrap: str) -> None:
    task = _task()
    response = f"The code is {wrap.format(task.answer)}"
    verdict = score_needle(response, task)
    assert verdict.outcome == QualityOutcome.CORRECT, response


# --- 大小文字の区別 (この module が選んだ、より厳しい側の規則) -------------------


def test_case_sensitive_rejects_lowercased_answer() -> None:
    """正解 (16 進の英字を含む) を小文字にした応答は、不正解にする (このモジュール
    の docstring に書いた、正規化を最小にする決定)。
    """
    task = _task(answer="SIGIL-4F82A1C9")
    verdict = score_needle(f"the code is {task.answer.lower()}", task)
    assert verdict.outcome == QualityOutcome.INCORRECT


def test_case_sensitive_accepts_exact_case() -> None:
    task = _task(answer="SIGIL-4F82A1C9")
    verdict = score_needle(f"the code is {task.answer}", task)
    assert verdict.outcome == QualityOutcome.CORRECT


# --- 大量に候補を並べる (shotgun) 応答は正解にしない -----------------------------
# task 6.4 の手直し (レビュー第 1 ラウンドの BLOCKING 指摘 1)。旧い実装は
# 「正解と同じ長さの `[\w-]` だけでできた語」を候補とみなしていたため、
# `implementation` のような、ふつうの 14 文字の英単語が正解 (`SIGIL-4F82A1C9`
# も 14 文字) と紛れて、正直な応答を不正解にしていた (48/300 で誤って不正解に
# なった、とレビューが測った)。新しい実装は、正解が `PREFIX-16進` の書式
# (`re.fullmatch(r"([A-Za-z]+-)([0-9A-F]+)", answer)`) のときだけ、同じ接頭辞・
# 同じ桁数の候補だけを数える。


_FOURTEEN_LETTER_ENGLISH_WORDS: Final[tuple[str, ...]] = (
    "recommendation",
    "implementation",
    "representative",
    "transformation",
    "administration",
    "categorization",
    "classification",
    "infrastructure",
)
"""`len("SIGIL-4F82A1C9") == 14` と同じ長さの、ふつうの英単語 (round 2 必須試験 (a))。"""


@pytest.mark.parametrize("word", _FOURTEEN_LETTER_ENGLISH_WORDS)
def test_ordinary_same_length_english_word_does_not_trigger_shotgun(word: str) -> None:
    """正解と同じ長さの、ふつうの英単語が周りにあるだけでは shotgun にしない
    (round 2 必須試験 (a)。正解の書式 `PREFIX-16進` に一致しない語は、そもそも
    候補に数えない)。
    """
    task = _task(answer="SIGIL-4F82A1C9")
    assert len(word) == len(task.answer)
    response = f"My {word} is that the access code is {task.answer}."
    verdict = score_needle(response, task)
    assert verdict.outcome == QualityOutcome.CORRECT, response


def test_shotgun_requires_matching_prefix_and_hex_length() -> None:
    """正解と同じ接頭辞・同じ桁数の候補が複数あれば、依然として shotgun にする
    (round 2 必須試験 (d): 本物の shotgun は、引き続き不正解のまま)。
    """
    task = _task(answer="SIGIL-4F82A1C9")
    response = f"It could be one of these: SIGIL-00000000, {task.answer}, or SIGIL-FFFFFFFF."
    verdict = score_needle(response, task)
    assert verdict.outcome == QualityOutcome.INCORRECT
    assert "shotgun" in verdict.detail


def test_single_mention_among_different_shaped_tokens_is_correct() -> None:
    """正解と長さの違う、ふつうの語や数字が周りにあっても、shotgun とは見なさない。"""
    task = _task(answer="SIGIL-4F82A1C9")
    response = (
        f"After checking ticket BH-4821 and node id 77, the code is {task.answer}, "
        "confirmed on 2024-05-01."
    )
    verdict = score_needle(response, task)
    assert verdict.outcome == QualityOutcome.CORRECT


def test_repeating_the_correct_answer_twice_is_not_a_shotgun() -> None:
    task = _task(answer="SIGIL-4F82A1C9")
    response = f"The code is {task.answer}. To confirm, it is {task.answer}."
    verdict = score_needle(response, task)
    assert verdict.outcome == QualityOutcome.CORRECT


def test_non_prefix_hex_answer_skips_shotgun_check_entirely() -> None:
    """正解が `PREFIX-16進` の書式でなければ (例: 数字だけ)、shotgun の判定は
    一切行わない。境界の判定 (数字/英数字の境界) だけは、引き続き働く
    (round 2 必須試験 (f))。
    """
    task = _task(answer="4821")
    # 正解と同じ長さの、別の数字を複数混ぜる (旧い実装ならここで shotgun 扱いに
    # なりかねない状況)。
    response = "Candidates include 1234, 5678, 4821, and 9012, but the code is 4821."
    verdict = score_needle(response, task)
    assert verdict.outcome == QualityOutcome.CORRECT


# --- 数字/英字の境界は ASCII に限る (CJK を「語の文字」に含めない) --------------
# task 6.4 の手直し (レビュー第 1 ラウンドの BLOCKING 指摘 2)。旧い実装は
# Python の `\w` (Unicode 対応) を使っていたため、正解の直前・直後が漢字や
# ひらがなだと、それを「語の文字が続いている」と誤認して、正解を見つけられな
# かった。


def test_answer_adjacent_to_japanese_text_without_spaces_is_correct() -> None:
    """正解の前後が、空白なしの日本語の文字に接していても、正解と認める
    (round 2 必須試験 (c))。
    """
    task = _task(answer="SIGIL-1A2B3C4D")
    verdict = score_needle("コードは" + task.answer + "です", task)
    assert verdict.outcome == QualityOutcome.CORRECT


def test_answer_wrapped_in_fullwidth_japanese_punctuation_is_correct() -> None:
    """全角の鉤括弧 (「」) で正解を囲んだ応答も、正解と認める
    (round 2 必須試験 (c))。
    """
    task = _task(answer="SIGIL-1A2B3C4D")
    verdict = score_needle(f"コードは「{task.answer}」です。", task)
    assert verdict.outcome == QualityOutcome.CORRECT


def test_lowercased_fullwidth_and_unicode_hyphen_variants_are_incorrect_by_design() -> None:
    """一致は文字どおり・大小文字を区別する。小文字化した正解、全角のラテン文字
    (`ＳＩＧＩＬ`)、Unicode のハイフン (U+2010、U+FF0D) はどれも、正解の文字列と
    バイト単位で一致しないため、設計どおり不正解になる (round 2 の指摘 6。
    この module の docstring にも明記する)。
    """
    task = _task(answer="SIGIL-1A2B3C4D")
    variants = [
        task.answer.lower(),  # sigil-1a2b3c4d
        "ＳＩＧＩＬ－1A2B3C4D",  # ＳＩＧＩＬ－1A2B3C4D (全角)
        task.answer.replace("-", "‐"),  # SIGIL‐1A2B3C4D (U+2010 HYPHEN)
        task.answer.replace("-", "－"),  # SIGIL－1A2B3C4D (U+FF0D FULLWIDTH HYPHEN)
    ]
    for variant in variants:
        verdict = score_needle(f"The code is {variant}.", task)
        assert verdict.outcome == QualityOutcome.INCORRECT, variant


# --- 正直な応答の偽陰性がないこと (false-negative-rate property) ----------------
# task 6.4 の手直し (round 2 必須試験 (b)、(g))。ここだけは、`corpus/needle.py`
# が作る本物の課題を使う (この試験ファイルは、生成器を使って「実際に起こる
# 状況」を作ってよい。`scoring/needle.py` 自身が corpus を読み込むわけではない
# ことに変わりはない。design.md の import の制約は、採点の module の実装に
# かかるものであり、この試験ファイルにはかからない)。


def test_honest_answer_quoting_matching_length_codename_is_correct() -> None:
    """コード名の長さが、たまたま正解と同じ長さになる組み合わせ (`(seed, index)
    ) = (1, 1)`: コード名 `Onyx-Zephyr-80` は 14 文字で、正解の書式
    `SIGIL-16進8桁` も常に 14 文字) で、コード名を引用符で示した正直な応答が、
    正解になることを確かめる (round 2 必須試験 (b)。レビューが見つけた、
    最も典型的な偽陰性の状況)。
    """
    task = make_needle_case(target_tokens=300, depth_pct=50, index=1, seed=1)
    codename = _codename_from_question(task.question)
    assert codename == "Onyx-Zephyr-80"
    assert len(codename) == len(task.answer), (
        f"この試験の前提 (コード名と正解が同じ長さ) が崩れている: "
        f"codename={codename!r} ({len(codename)}), answer={task.answer!r} ({len(task.answer)})"
    )

    response = f'The access code for "{codename}" is {task.answer}.'
    verdict = score_needle(response, task)
    assert verdict.outcome == QualityOutcome.CORRECT


def test_false_negative_rate_is_zero_over_many_generated_cases() -> None:
    """複数の種にわたる 100 件以上の生成された課題について、コード名を引用符で
    示した正直な応答が、必ず正解になることを確かめる (round 2 必須試験 (g):
    偽陰性率は 0 でなければならない)。
    """
    depths = (0, 25, 50, 75, 100)
    total = 0
    for seed in range(1, 6):
        for index in range(25):
            depth_pct = depths[index % len(depths)]
            task = make_needle_case(target_tokens=300, depth_pct=depth_pct, index=index, seed=seed)
            codename = _codename_from_question(task.question)
            response = f'The access code for "{codename}" is {task.answer}.'
            verdict = score_needle(response, task)
            assert verdict.outcome == QualityOutcome.CORRECT, (
                f"seed={seed} index={index} codename={codename!r} answer={task.answer!r}: "
                f"{verdict.outcome} ({verdict.detail})"
            )
            total += 1
    assert total >= 100, f"生成した課題の数が 100 件に届いていない (total={total})"


# --- 例外を投げない ---------------------------------------------------------------


def test_never_raises_on_pathological_input() -> None:
    task = _task()
    pathological = [
        "",
        "�" * 1000,  # 置き換え文字
        "\x00\x01\x02control chars",
        "x" * 500_000,  # 大きな応答
        "日本語の応答です。コードは見つかりません。",
        task.answer * 50,  # 正解の繰り返し
    ]
    for text in pathological:
        verdict = score_needle(text, task)
        assert isinstance(verdict, QualityVerdict)


def test_never_raises_when_answer_is_empty_string() -> None:
    task = _task(answer="")
    verdict = score_needle("anything", task)
    assert verdict.outcome == QualityOutcome.INCORRECT
