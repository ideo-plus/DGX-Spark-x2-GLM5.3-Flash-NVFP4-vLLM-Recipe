"""scoring/sanity の試験 (task 2.6)。

`detect_output_anomalies` は、単純な O(文字数 × 周期の候補数) の走査では
200,000 文字級の本文で遅すぎるため、周期的な区間を効率よく見つける実装にして
ある。ここでは、境界 (7 回では付かず 8 回で付く)、短い周期が長い単位に吸収
される場合、遠く離れた大きな周期、印を付けない並び (空白だけ、英数字でない
1 種類の文字だけ) の決まりと、その決まりでも英数字の 1 種類の文字の連続には
印が付くこと、性能の上限、アルゴリズムの取りこぼしの計測 (定義どおりの
総当たりとの突き合わせ) を確かめる。
"""

from __future__ import annotations

import random
import string
import time
from datetime import UTC, datetime

import pytest

from bench_harness.scoring.sanity import (
    MAX_UNIT_CHARS,
    collect_text,
    collect_tool_input_text,
    detect_output_anomalies,
)
from bench_harness.types import ContentBlock, StreamResult, StreamTiming, TrialFlag

# --- 助け -------------------------------------------------------------------


def _timing() -> StreamTiming:
    return StreamTiming(sent_at_utc=datetime.now(UTC), sent_at_ns=0, end_ns=1)


def _result(blocks: list[ContentBlock]) -> StreamResult:
    return StreamResult(timing=_timing(), blocks=blocks)


_WORDS = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
    "zulu",
)


def _filler_text(rng: random.Random, min_len: int) -> str:
    """ふつうの散文に似た、周期的な繰り返しを持たない文章を作る (性能の試験用)。"""
    parts: list[str] = []
    length = 0
    while length < min_len:
        word = rng.choice(_WORDS)
        parts.append(word)
        parts.append(" ")
        length += len(word) + 1
    return "".join(parts)


def _random_unit(rng: random.Random, length: int) -> str:
    """内部に短い繰り返しがほぼ生まれない、length 文字の並びを作る。"""
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choices(alphabet, k=length))


# --- REPLACEMENT_CHAR --------------------------------------------------------


def test_replacement_char_flags_once() -> None:
    text = "ordinary output " + "�" + " continues here"
    flags = detect_output_anomalies(text, repeat_min_chars=12, repeat_min_count=8)
    assert flags == [TrialFlag.REPLACEMENT_CHAR]


def test_replacement_char_absent_not_flagged() -> None:
    flags = detect_output_anomalies("ordinary text, nothing odd here.", 12, 8)
    assert TrialFlag.REPLACEMENT_CHAR not in flags


# --- REPETITION_LOOP: 境界 (12 文字 x 8 と x 7) -------------------------------


def test_repetition_loop_flags_at_exactly_min_count() -> None:
    unit = "Qz7mKpLr3sTn"  # 12 種類の異なる文字。内部に短い周期を持たない
    assert len(unit) == 12
    flags = detect_output_anomalies(unit * 8, 12, 8)
    assert flags == [TrialFlag.REPETITION_LOOP]


def test_repetition_loop_not_flagged_one_below_min_count() -> None:
    unit = "Qz7mKpLr3sTn"
    flags = detect_output_anomalies(unit * 7, 12, 8)
    assert flags == []


# --- REPETITION_LOOP: 日本語の文 ---------------------------------------------


def test_repetition_loop_japanese_sentence_boundary() -> None:
    sentence = "今日はとても天気が良いので散歩に出かけることにしました"
    assert len(sentence) >= 12
    flagged = detect_output_anomalies(sentence * 8, 12, 8)
    assert TrialFlag.REPETITION_LOOP in flagged
    not_flagged = detect_output_anomalies(sentence * 7, 12, 8)
    assert TrialFlag.REPETITION_LOOP not in not_flagged


# --- REPETITION_LOOP: 短い周期が長い単位に吸収される場合 -----------------------


def test_short_period_absorbed_into_min_chars_flags() -> None:
    # "ab" (周期 2) x 48 = 96 文字。12 文字の単位 "abababababab" が
    # ちょうど 8 回続くと見なせるので、印が付く
    flags = detect_output_anomalies("ab" * 48, 12, 8)
    assert TrialFlag.REPETITION_LOOP in flags


def test_short_period_absorbed_into_min_chars_not_flagged_below_boundary() -> None:
    # "ab" x 40 = 80 文字。12 文字の単位は 6 回にしかならない (6 * 12 = 72 <= 80)
    flags = detect_output_anomalies("ab" * 40, 12, 8)
    assert TrialFlag.REPETITION_LOOP not in flags


# --- REPETITION_LOOP: 長い周期、文章の途中 -------------------------------------


def test_long_period_loop_detected_in_middle_of_text() -> None:
    rng = random.Random(12345)
    prefix = _filler_text(rng, 20_000)
    unit = _random_unit(rng, 500)
    loop = unit * 8
    suffix = _filler_text(rng, 20_000)
    text = prefix + loop + suffix
    flags = detect_output_anomalies(text, 12, 8)
    assert TrialFlag.REPETITION_LOOP in flags


def test_repetition_loop_at_end_of_text() -> None:
    rng = random.Random(999)
    prefix = _filler_text(rng, 5_000)
    unit = _random_unit(rng, 30)
    text = prefix + unit * 8
    flags = detect_output_anomalies(text, 12, 8)
    assert TrialFlag.REPETITION_LOOP in flags


# --- REPETITION_LOOP: 印を付けない並び ----------------------------------------


def test_ordinary_english_prose_not_flagged() -> None:
    text = (
        "The on-call engineer reviewed the incident report and confirmed that "
        "the nimbus cache remained stable throughout the maintenance window. "
        "No further action is required, and the capacity review will proceed "
        "as scheduled next quarter. A follow-up ticket tracks the remaining "
        "cleanup work for the release owner."
    )
    flags = detect_output_anomalies(text, 12, 8)
    assert TrialFlag.REPETITION_LOOP not in flags


def test_ordinary_japanese_prose_not_flagged() -> None:
    text = (
        "当番のエンジニアは、夜間の検証で見つかった不具合を確認し、"
        "リリース担当と協力して原因を調査した。影響の範囲は限定的であり、"
        "追加の対応は不要と判断された。次回の見直しでは、"
        "容量の計画についても改めて検討する予定である。"
    )
    flags = detect_output_anomalies(text, 12, 8)
    assert TrialFlag.REPETITION_LOOP not in flags


def test_python_code_with_similar_lines_not_flagged() -> None:
    text = "\n".join(
        f"def route_echo_broker_{i:05d}(x: int, y: int = {i * 7}) -> int:\n"
        f'    """Fictional helper, used only to pad benchmark context."""\n'
        f"    total = x + y + {i}\n"
        "    return total\n"
        for i in range(60)
    )
    flags = detect_output_anomalies(text, 12, 8)
    assert TrialFlag.REPETITION_LOOP not in flags


def test_log_with_similar_lines_not_flagged() -> None:
    rng = random.Random(7)
    lines = []
    for i in range(400):
        req_id = f"{rng.randint(0, 0xFFFFFFF):07x}"
        dur_ms = rng.randint(1, 4500)
        ts = f"2024-01-01T00:00:{i % 60:02d}.000Z"
        lines.append(f"{ts} INFO nimbus_cache req={req_id} dur={dur_ms}ms cache hit")
    text = "\n".join(lines)
    flags = detect_output_anomalies(text, 12, 8)
    assert TrialFlag.REPETITION_LOOP not in flags


def test_markdown_table_not_flagged() -> None:
    rows = ["| Name | Value | Note |", "| --- | --- | --- |"]
    for i in range(40):
        rows.append(f"| component-{i:03d} | {i * 3} | status-{i % 5} |")
    text = "\n".join(rows)
    flags = detect_output_anomalies(text, 12, 8)
    assert TrialFlag.REPETITION_LOOP not in flags


def test_markdown_rule_of_dashes_not_flagged() -> None:
    text = "Section one.\n\n" + ("-" * 80) + "\n\nSection two follows the rule above."
    flags = detect_output_anomalies(text, 12, 8)
    assert TrialFlag.REPETITION_LOOP not in flags


def test_deep_indentation_not_flagged() -> None:
    text = "if True:\n" + "    if True:\n" * 1 + (" " * 96) + "value = 1\n"
    flags = detect_output_anomalies(text, 12, 8)
    assert TrialFlag.REPETITION_LOOP not in flags


def test_long_run_of_blank_lines_not_flagged() -> None:
    text = "Start of document.\n" + ("\n" * 120) + "End of document."
    flags = detect_output_anomalies(text, 12, 8)
    assert TrialFlag.REPETITION_LOOP not in flags


def test_template_corpus_samples_not_flagged() -> None:
    # task 2.5 の合成コード (f7bb36a で修正済み) は、長さの帳尻合わせに通し番号
    # 付きのコメント行を末尾に足す (周期を持たない)。字下げのような空白だけの
    # 並びは、印を付けない決まりで引き続き除かれることを確かめる
    from bench_harness.corpus import TemplateCorpus
    from bench_harness.types import Profile

    corpus = TemplateCorpus(Profile(name="quick").chars_per_token)
    samples = [
        corpus.prose("en", 32_000, seed=1),
        corpus.prose("ja", 32_000, seed=2),
        corpus.code(32_000, seed=3),
        corpus.log(32_000, seed=4),
    ]
    for sample in samples:
        flags = detect_output_anomalies(sample, 12, 8)
        assert TrialFlag.REPETITION_LOOP not in flags


# --- REPETITION_LOOP: 印を付けない並びの決まりを、英数字の 1 種類の文字までは
# --- 広げない (コントローラの指摘 1) -------------------------------------------


def test_single_alnum_char_run_flags_at_boundary() -> None:
    # 英数字の同じ 1 文字がひたすら続くのは、壊れたモデルの典型的な症状であり、
    # この検査がいちばん見つけたいものの 1 つなので、印を付ける
    assert detect_output_anomalies("a" * 96, 12, 8) == [TrialFlag.REPETITION_LOOP]
    assert detect_output_anomalies("的" * 96, 12, 8) == [TrialFlag.REPETITION_LOOP]
    assert detect_output_anomalies("0" * 96, 12, 8) == [TrialFlag.REPETITION_LOOP]


def test_single_alnum_char_run_not_flagged_below_boundary() -> None:
    # 12 * 8 = 96 に 1 文字だけ届かない (95 文字)
    assert detect_output_anomalies("a" * 95, 12, 8) == []


def test_single_non_alnum_or_whitespace_char_run_not_flagged() -> None:
    for text in ("-" * 500, "=" * 500, "─" * 500, "." * 500, " " * 500, "\n" * 500):
        flags = detect_output_anomalies(text, 12, 8)
        assert TrialFlag.REPETITION_LOOP not in flags, repr(text)


def test_whitespace_mixed_with_single_non_alnum_char_not_flagged() -> None:
    # 罫線や表の区切りは、空白と 1 種類の記号が混ざることが多い
    dashes_and_spaces = "- " * 60
    pipes_and_spaces = "| " * 60
    for text in (dashes_and_spaces, pipes_and_spaces):
        flags = detect_output_anomalies(text, 12, 8)
        assert TrialFlag.REPETITION_LOOP not in flags, repr(text)


# --- REPETITION_LOOP: アルゴリズムの取りこぼしの計測 (コントローラの指摘 2) ------


def _brute_force_has_qualifying_period(
    text: str, repeat_min_chars: int, repeat_min_count: int
) -> bool:
    """定義どおりの総当たり: すべての単位の長さと、すべての開始位置について、
    `text` が `unit * repeat_min_count` を含むかを調べる。遅いので、この試験の
    小さな入力にだけ使う。「印を付けない並びの決まり」は適用しない (この決まりは
    `test_single_*`/`test_whitespace_mixed_*` で別に確かめてある。ここで確かめる
    のは、長さと回数だけの、アルゴリズムの取りこぼしの有無)。
    """
    n = len(text)
    max_unit_len = n // repeat_min_count
    for unit_len in range(repeat_min_chars, max_unit_len + 1):
        span = unit_len * repeat_min_count
        for start in range(0, n - span + 1):
            unit = text[start : start + unit_len]
            if text[start : start + span] == unit * repeat_min_count:
                return True
    return False


@pytest.mark.parametrize("repeat_min_chars", [1, 2, 3, 8, 12])
@pytest.mark.parametrize("repeat_min_count", [2, 3, 8])
def test_algorithm_matches_brute_force_on_random_small_strings(
    repeat_min_chars: int, repeat_min_count: int
) -> None:
    """アルゴリズムの周期の探し方 (印を付けない決まりを適用する前の、長さと
    回数だけの判定) が、定義どおりの総当たりと一致するかを、受け付ける範囲の
    設定 (`repeat_min_chars` は 1、2、3、8、12、`repeat_min_count` は 2、3、8) と、
    2〜3 文字の小さな
    アルファベットで、種を固定した乱数から作った文字列数千個について確かめる。
    半分は「乱数の前置き + 単位 × 回数 + 乱数の後置き」の形で、繰り返しを
    十分な割合で仕込む。
    """
    # 内部の実装を直に試験する (「印を付けない決まり」を外した、長さと回数だけの
    # 判定を取り出すため)。取りこぼしの計測という、この試験だけの目的による
    from bench_harness.scoring.sanity import _scan_for_qualifying_run

    def algorithm_has_qualifying_period(
        text: str, repeat_min_chars: int, repeat_min_count: int
    ) -> bool:
        return _scan_for_qualifying_run(
            text, repeat_min_chars, repeat_min_count, lambda _text, _start, _period: True
        )

    rng = random.Random(20260919 + 10 * repeat_min_chars + repeat_min_count)
    alphabets = ("ab", "abc", "ab ")
    total = 1200
    mismatches: list[str] = []
    for trial in range(total):
        alphabet = alphabets[trial % len(alphabets)]
        length = rng.randint(10, 160)
        if trial % 2 == 0:
            # 繰り返しを仕込む: 乱数の前置き + 単位 x 回数 + 乱数の後置き
            unit_len = rng.randint(1, 24)
            unit = "".join(rng.choice(alphabet) for _ in range(unit_len))
            if trial % 10 == 0:
                # 単位の中で、同じ並びが周期でない距離にも現れるもの (近道の探し方が
                # 取りこぼした形)
                unit = rng.choice(("aabaaaab", "abababba", "abaaaabaabaaaabaabaaaabaaa"))
            count = rng.randint(repeat_min_count, repeat_min_count + 3)
            core = unit * count
            prefix_len = rng.randint(0, max(0, length - len(core)))
            prefix = "".join(rng.choice(alphabet) for _ in range(prefix_len))
            remaining = max(0, length - len(prefix) - len(core))
            suffix = "".join(rng.choice(alphabet) for _ in range(remaining))
            text = prefix + core + suffix
        else:
            text = "".join(rng.choice(alphabet) for _ in range(length))

        expected = _brute_force_has_qualifying_period(text, repeat_min_chars, repeat_min_count)
        actual = algorithm_has_qualifying_period(text, repeat_min_chars, repeat_min_count)
        if expected != actual:
            mismatches.append(text)

    assert not mismatches, (
        f"{len(mismatches)}/{total} 件で総当たりと食い違った (取りこぼし): {mismatches[:5]}"
    )


# --- REPETITION_LOOP: 公開の関数を、除外の決まりつきの総当たりと突き合わせる ------


def _unit_is_excluded(unit: str) -> bool:
    """「印を付けない並びの決まり」を、文書のとおりに書いたもの (実装とは別に)。"""
    symbols = {ch for ch in unit if not ch.isspace()}
    if not symbols:
        return True
    return len(symbols) == 1 and not next(iter(symbols)).isalnum()


def _brute_force_flags_a_loop(text: str, repeat_min_chars: int, repeat_min_count: int) -> bool:
    """定義どおりの総当たりに、除外の決まりを足したもの。"""
    n = len(text)
    for unit_len in range(repeat_min_chars, n // repeat_min_count + 1):
        span = unit_len * repeat_min_count
        for start in range(0, n - span + 1):
            unit = text[start : start + unit_len]
            if text[start : start + span] == unit * repeat_min_count and not _unit_is_excluded(
                unit
            ):
                return True
    return False


@pytest.mark.parametrize("repeat_min_chars", [1, 2, 3, 8, 12])
@pytest.mark.parametrize("repeat_min_count", [2, 3, 8])
def test_public_path_matches_brute_force_around_excluded_runs(
    repeat_min_chars: int, repeat_min_count: int
) -> None:
    """印を付けない区間を飛ばす道を、総当たりと突き合わせる。

    記号が長く続く入力を作らないと、この道は通らない。そこで、記号だけ、または記号と
    1 文字だけのアルファベットで、`記号 a が i 個 + 記号 b が j 個` の形の単位を仕込む。
    飛ばしすぎる壊れ方 (例: `_covering_segment_end` の `- unit_len` を落とす) は、
    「印が付くはずなのに付かない」として、ここで見つかる。
    """
    rng = random.Random(977 * repeat_min_chars + repeat_min_count)
    alphabets = ("-=", "-|", "._", "-a", "- ", "-= ")
    mismatches: list[str] = []
    total = 700
    for trial in range(total):
        alphabet = alphabets[trial % len(alphabets)]
        if trial % 3 != 2:
            first, second = rng.sample(alphabet, 2) if len(alphabet) > 1 else (alphabet, alphabet)
            unit = first * rng.randint(1, 18) + second * rng.randint(1, 6)
            core = unit * rng.randint(max(1, repeat_min_count - 1), repeat_min_count + 1)
            prefix = rng.choice(alphabet) * rng.randint(0, 30)
            suffix = rng.choice(alphabet) * rng.randint(0, 30)
            text = prefix + core + suffix
        else:
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(5, 120)))
        expected = _brute_force_flags_a_loop(text, repeat_min_chars, repeat_min_count)
        actual = TrialFlag.REPETITION_LOOP in detect_output_anomalies(
            text, repeat_min_chars, repeat_min_count
        )
        if expected != actual:
            mismatches.append(f"{text!r} (期待 {expected}, 実際 {actual})")

    assert not mismatches, f"{len(mismatches)}/{total} 件で総当たりと食い違った: {mismatches[:3]}"


def test_a_loop_whose_unit_starts_with_a_long_symbol_run_is_flagged() -> None:
    """錨の位置が記号の区間に入っていても、単位がその区間に収まらないなら、飛ばさない。

    `_covering_segment_end` の `start <= q - unit_len` の `- unit_len` を守る回帰試験。
    これを落とすと、既定の設定で、この繰り返しを見つけ損ねる。
    """
    text = ("|" * 15 + "###") * 8

    assert detect_output_anomalies(text, 12, 8) == [TrialFlag.REPETITION_LOOP]


# --- 両方の印 -----------------------------------------------------------------


def test_both_flags_together_in_fixed_order() -> None:
    unit = "Qz7mKpLr3sTn"
    text = "prefix " + "�" + " middle " + unit * 8
    flags = detect_output_anomalies(text, 12, 8)
    assert flags == [TrialFlag.REPLACEMENT_CHAR, TrialFlag.REPETITION_LOOP]


# --- 引数の検証 ----------------------------------------------------------------


def test_rejects_repeat_min_chars_below_one() -> None:
    with pytest.raises(ValueError):
        detect_output_anomalies("x", 0, 8)


@pytest.mark.parametrize(
    ("text", "repeat_min_chars", "repeat_min_count"),
    [
        # 単位 "aabb" が 3 回。窓が 1 文字だと、近道の探し方は取りこぼした
        ("aabbaabbaabb", 1, 3),
        # 26 文字の単位がちょうど 4 回。単位の中で、同じ 8 文字の並びが、周期 (26) より
        # 短い距離 (5、8、18、21) にも現れる。近道の探し方は、窓が 8 文字でも取りこぼした
        ("abaaaabaabaaaabaabaaaabaaa" * 4, 8, 4),
    ],
)
def test_units_that_repeat_a_window_internally_are_still_flagged(
    text: str, repeat_min_chars: int, repeat_min_count: int
) -> None:
    """レビューで見つかった取りこぼしの反例 (回帰試験)。"""
    assert detect_output_anomalies(text, repeat_min_chars, repeat_min_count) == [
        TrialFlag.REPETITION_LOOP
    ]


def test_units_longer_than_the_documented_cap_are_not_searched() -> None:
    """単位の長さの上限は、決まりとして文書に書いてある。上限ちょうどは見つける。"""
    rng = random.Random(7)
    at_cap = "".join(rng.choice(string.ascii_lowercase) for _ in range(MAX_UNIT_CHARS))
    over_cap = at_cap + "z"

    assert detect_output_anomalies(at_cap * 2, 12, 2) == [TrialFlag.REPETITION_LOOP]
    assert detect_output_anomalies(over_cap * 2, 12, 2) == []


def test_rejects_repeat_min_count_below_two() -> None:
    with pytest.raises(ValueError):
        detect_output_anomalies("x", 12, 1)


# --- 性能 (生成的、200,000 文字級) ---------------------------------------------


def test_performance_normal_text_completes_quickly() -> None:
    rng = random.Random(1)
    text = _filler_text(rng, 200_000)
    started = time.perf_counter()
    flags = detect_output_anomalies(text, 12, 8)
    elapsed = time.perf_counter() - started
    assert TrialFlag.REPETITION_LOOP not in flags
    assert elapsed < 2.0, f"200,000 文字の通常の文章の判定に {elapsed:.2f}s かかった"


def test_performance_text_ending_in_loop_completes_quickly() -> None:
    rng = random.Random(2)
    prefix = _filler_text(rng, 196_000)
    unit = _random_unit(rng, 500)
    text = prefix + unit * 8
    started = time.perf_counter()
    flags = detect_output_anomalies(text, 12, 8)
    elapsed = time.perf_counter() - started
    assert TrialFlag.REPETITION_LOOP in flags
    message = f"末尾に繰り返しがある 200,000 文字級の文章の判定に {elapsed:.2f}s かかった"
    assert elapsed < 2.0, message


def test_performance_long_excluded_run_completes_quickly() -> None:
    # 印を付けない決まりに当てはまる (空白だけの) 長い一様な並びでも、
    # O(文字数²) にならないことを確かめる。周期ごとに求めた区間を、同じ区間の
    # 中の候補では求め直さないようにする前は、この入力で処理が終わらなかった
    text = "\n" * 200_000
    started = time.perf_counter()
    flags = detect_output_anomalies(text, 12, 8)
    elapsed = time.perf_counter() - started
    assert TrialFlag.REPETITION_LOOP not in flags
    assert elapsed < 2.0, f"200,000 文字の空白だけの文章の判定に {elapsed:.2f}s かかった"


# --- collect_text / collect_tool_input_text -----------------------------------


def test_collect_text_includes_thinking_and_text_in_block_order() -> None:
    blocks = [
        ContentBlock(type="thinking", text="推論の過程。"),
        ContentBlock(type="text", text="本文です。"),
        ContentBlock(
            type="tool_use",
            tool_name="search",
            tool_input_raw='{"q": "test"}',
            tool_input={"q": "test"},
        ),
    ]
    result = _result(blocks)
    assert collect_text(result) == "推論の過程。本文です。"


def test_collect_tool_input_text_excludes_from_collect_text() -> None:
    blocks = [
        ContentBlock(type="text", text="本文"),
        ContentBlock(
            type="tool_use",
            tool_name="search",
            tool_input_raw='{"q": "test"}',
            tool_input={"q": "test"},
        ),
    ]
    result = _result(blocks)
    assert "tool_input" not in collect_text(result)
    assert collect_tool_input_text(result) == '{"q": "test"}'


def test_replacement_char_in_tool_input_detected_via_documented_api() -> None:
    blocks = [
        ContentBlock(
            type="tool_use",
            tool_name="write_file",
            tool_input_raw='{"content": "broken � byte"}',
            tool_input=None,
        ),
    ]
    result = _result(blocks)
    # 本文には現れないので、collect_text 側では検出されない
    assert TrialFlag.REPLACEMENT_CHAR not in detect_output_anomalies(collect_text(result), 12, 8)
    # ツールの生の引数文字列には現れるので、こちらを渡せば検出される
    assert TrialFlag.REPLACEMENT_CHAR in detect_output_anomalies(
        collect_tool_input_text(result), 12, 8
    )
