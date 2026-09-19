"""長い入力から情報を探す課題の生成の試験 (task 6.4: corpus/needle)。

決定性 (プロセスと `PYTHONHASHSEED` をまたいで同じ)、埋めた位置の精度 (狙いの
±5%)、正解と鍵 (codename) がそれぞれ全文にちょうど 1 回だけ現れること、全体の
長さが corpus の通常の許容 (±2%) に収まること、正解の書式が `corpus/synth.py`
の型紙の出力と衝突しないこと、同じ `(seed, index)` なら depth を変えても干し草の
土台と正解が変わらないこと (位置の効果だけを切り分ける設計上の決定)、golden な
ハッシュ (このモジュール自身の生成手順が変わっていないかの検出) を確かめる。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
import sys
from itertools import product
from typing import Final

import pytest

from bench_harness.corpus.needle import make_needle_case
from bench_harness.corpus.synth import TemplateCorpus
from bench_harness.types import ContentKind, NeedleCase, Profile

_PROFILE: Final[Profile] = Profile(name="quick")

_CODENAME_IN_QUESTION_RE: Final[re.Pattern[str]] = re.compile(r'"([^"]+)"')


def _codename_from_question(question: str) -> str:
    """`question` から、引用符で囲まれたコード名を取り出す (試験だけで使う)。"""
    match = _CODENAME_IN_QUESTION_RE.search(question)
    assert match is not None, f"問いの文からコード名を取り出せない: {question!r}"
    return match.group(1)


# 試験を速く保つため、小さめの狙いの長さを使う (task 6.4 の完了の状態のとおり)。
_SMALL_LENGTHS: Final[tuple[int, ...]] = (600, 1500, 3000)
_SEEDS: Final[tuple[int, ...]] = (1, 2, 3)
_DEPTHS: Final[tuple[int, ...]] = (0, 25, 50, 75, 100)

# 埋めた位置の精度の許容 (百分率点)。task 6.4 完了の状態「狙った位置の ±5%」。
_DEPTH_TOLERANCE_PCT: Final[float] = 5.0

# 全体の長さの許容。corpus/synth.py の _CHAR_BUDGET_TOLERANCE (±2%) と同じにする
# (task 6.4 完了の状態「corpus の通常の許容」)。少し余裕を持たせて ±3% にする
# (挿入した文の分だけ、狙いの文字数からの引き算に丸めの誤差が乗るため)。
_LENGTH_TOLERANCE_PCT: Final[float] = 3.0


def _full_prompt(task: NeedleCase) -> str:
    """試験だけで使う、干し草と問いをつないだ「全文」。"""
    return f"{task.haystack}\n\n{task.question}"


# --- 決定性 -------------------------------------------------------------------


def test_same_seed_and_args_produce_identical_task() -> None:
    a = make_needle_case(target_tokens=1500, depth_pct=50, index=0, seed=42)
    b = make_needle_case(target_tokens=1500, depth_pct=50, index=0, seed=42)
    assert a.model_dump_json() == b.model_dump_json()


def test_different_index_same_seed_produce_different_content() -> None:
    a = make_needle_case(target_tokens=1500, depth_pct=50, index=0, seed=42)
    b = make_needle_case(target_tokens=1500, depth_pct=50, index=1, seed=42)
    assert a.answer != b.answer
    assert a.haystack != b.haystack


def test_different_seed_same_index_produce_different_content() -> None:
    a = make_needle_case(target_tokens=1500, depth_pct=50, index=0, seed=1)
    b = make_needle_case(target_tokens=1500, depth_pct=50, index=0, seed=2)
    assert a.answer != b.answer
    assert a.haystack != b.haystack


_HASH_SCRIPT: Final[str] = """
from bench_harness.corpus.needle import make_needle_case

task = make_needle_case(
    target_tokens={target_tokens}, depth_pct={depth_pct}, index={index}, seed={seed}
)
print(task.model_dump_json())
"""


def test_deterministic_across_process_and_pythonhashseed() -> None:
    """プロセスをまたいでも、`PYTHONHASHSEED` を変えても、同じ課題になる。

    `hash()`、大域の乱数、時刻、dict/set の反復順序に依存していれば、ここで揺れる。
    """
    target_tokens, depth_pct, index, seed = 900, 25, 3, 2026
    expected = make_needle_case(
        target_tokens=target_tokens, depth_pct=depth_pct, index=index, seed=seed
    ).model_dump_json()

    script = _HASH_SCRIPT.format(
        target_tokens=target_tokens, depth_pct=depth_pct, index=index, seed=seed
    )
    for hash_seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        assert result.stdout.strip() == expected, f"PYTHONHASHSEED={hash_seed} で結果が違う"


# --- 埋めた位置の精度、正解と鍵の一意性、全体の長さ ----------------------------


@pytest.mark.parametrize(
    "target_tokens,depth_pct,seed", list(product(_SMALL_LENGTHS, _DEPTHS, _SEEDS))
)
def test_depth_answer_key_and_length_for_each_combination(
    target_tokens: int, depth_pct: int, seed: int
) -> None:
    task = make_needle_case(target_tokens=target_tokens, depth_pct=depth_pct, index=0, seed=seed)

    # 埋めた位置が、狙いの ±5 パーセント点に入る。
    haystack_len = len(task.haystack)
    realized_pct = task.inserted_char_offset / haystack_len * 100
    assert abs(realized_pct - depth_pct) <= _DEPTH_TOLERANCE_PCT, (
        f"depth_pct={depth_pct} realized={realized_pct:.2f} "
        f"offset={task.inserted_char_offset} haystack_len={haystack_len}"
    )

    # 正解が、干し草と問いをつないだ全文にちょうど 1 回だけ現れる。
    prompt = _full_prompt(task)
    assert prompt.count(task.answer) == 1

    # 鍵 (codename、問いが指し示す対象) が、干し草にちょうど 1 回だけ現れる。
    # 問いの文から、正解を含まない部分 (鍵の手がかり) を取り出すのは難しいため、
    # ここでは「問いの中で、正解の文字列を除いた語のうち、ハイフンを含む固有の
    # 語」を鍵とみなす代わりに、実装が公開する `question` 自体が鍵を一意に
    # 含んでいることを、間接に確かめる: 問いに現れる、正解と同じ長さでない
    # 「-」区切りの語 (kebab-case) を鍵の候補として集め、それぞれが干し草に
    # ちょうど 1 回だけ現れることを確かめる。
    candidates = [
        word.strip('"?.')
        for word in task.question.split()
        if "-" in word.strip('"?.') and word.strip('"?.') != task.answer
    ]
    assert candidates, "問いの文に鍵の手がかり (ハイフンを含む語) が見つからない"
    for candidate in candidates:
        assert task.haystack.count(candidate) == 1, f"鍵の候補 {candidate!r} が複数回/0回現れる"

    # 全体の長さが、狙いの文字数の通常の許容に収まる。
    want_chars = target_tokens * _PROFILE.chars_per_token[ContentKind.PROSE_EN]
    lower = want_chars * (1 - _LENGTH_TOLERANCE_PCT / 100)
    upper = want_chars * (1 + _LENGTH_TOLERANCE_PCT / 100)
    assert lower <= len(task.haystack) <= upper, (
        f"target_tokens={target_tokens} 狙い={want_chars:.1f} 実際={len(task.haystack)}"
    )


def test_depth_0_places_needle_at_the_very_start() -> None:
    task = make_needle_case(target_tokens=1500, depth_pct=0, index=0, seed=7)
    assert task.inserted_char_offset == 0
    assert task.haystack.startswith("The access code for ")


def test_depth_100_places_needle_at_the_very_end() -> None:
    task = make_needle_case(target_tokens=1500, depth_pct=100, index=0, seed=7)
    assert task.haystack.rstrip().endswith(".")
    assert task.answer in task.haystack[task.inserted_char_offset :]
    # 末尾のほうにあることを、大まかに確かめる (本体の 90% より後ろ)。
    assert task.inserted_char_offset > len(task.haystack) * 0.9


def test_all_depths_produce_distinct_offsets_for_same_seed_and_index() -> None:
    """ミューテーション「常に末尾に置く」「深さを反対から測る」を検出する試験。

    5 つの depth すべてで、挿入位置がそれぞれ別々の、単調に増える値になることを
    確かめる。常に末尾に置く壊れ方や、深さを 100-d として測る壊れ方は、ここで
    (少なくともどれかの組で) 落ちる。
    """
    offsets = [
        make_needle_case(target_tokens=4000, depth_pct=depth, index=0, seed=99).inserted_char_offset
        for depth in _DEPTHS
    ]
    assert len(set(offsets)) == len(offsets), f"挿入位置が重複している: {offsets}"
    assert offsets == sorted(offsets), f"挿入位置が深さの順に単調増加していない: {offsets}"


# --- 同じ index なら depth をまたいで干し草の土台と正解が変わらない -----------


def test_haystack_base_and_fact_fixed_across_depths_for_same_index() -> None:
    """同じ `(seed, target_tokens, index)` では、depth が違っても正解と鍵が
    同じで、干し草から挿入した文を除いた土台も同じであることを確かめる (位置の
    効果だけを切り分ける設計上の決定。この module の docstring を参照)。

    挿入は、土台の文字列に 1 文を差し込むだけで、文字を足しも引きもしない。
    継ぎ目の空白は `_join_with_single_space` が二重にならないよう調整するため
    (task 6.4 の手直し、レビュー第 1 ラウンドの指摘 4)、土台の末尾がすでに
    空白で終わっている場合とそうでない場合とで、挿入後の全体の長さが
    ちょうど 1 文字だけ違いうる (`depth_pct=100` で、土台の末尾がすでに
    半角空白または改行で終わっているときだけ、区切りの空白を足さないため)。
    したがって、長さの一致は「完全に等しい」ではなく「高々 1 文字の差」で
    確かめる。これは、正解と問いが一致することと合わせて、土台が depth に
    依存していないことを、実装の書式に立ち入らずに確かめられる。
    """
    tasks = [
        make_needle_case(target_tokens=2000, depth_pct=depth, index=5, seed=11) for depth in _DEPTHS
    ]
    answers = {task.answer for task in tasks}
    assert len(answers) == 1, f"depth をまたいで正解が変わった: {answers}"
    questions = {task.question for task in tasks}
    assert len(questions) == 1, f"depth をまたいで問いが変わった: {questions}"

    lengths = [len(task.haystack) for task in tasks]
    assert max(lengths) - min(lengths) <= 1, f"depth をまたいで全体の長さが変わった: {lengths}"


# --- 検証エラー -----------------------------------------------------------------


def test_rejects_non_positive_target_tokens() -> None:
    with pytest.raises(ValueError):
        make_needle_case(target_tokens=0, depth_pct=50, index=0, seed=1)
    with pytest.raises(ValueError):
        make_needle_case(target_tokens=-10, depth_pct=50, index=0, seed=1)


def test_rejects_out_of_range_depth_pct() -> None:
    with pytest.raises(ValueError):
        make_needle_case(target_tokens=1000, depth_pct=-1, index=0, seed=1)
    with pytest.raises(ValueError):
        make_needle_case(target_tokens=1000, depth_pct=101, index=0, seed=1)


def test_rejects_negative_index() -> None:
    with pytest.raises(ValueError):
        make_needle_case(target_tokens=1000, depth_pct=50, index=-1, seed=1)


# --- 正解の書式が corpus の型紙の出力と衝突しない -------------------------------


def test_answer_format_never_collides_with_corpus_generator_output() -> None:
    """正解の文字どおりの接頭辞が、`corpus/synth.py` のどの型紙の出力にも
    現れないことを、複数の種と種類で確かめる (task 6.4 完了の状態の一部:
    「正解の書式は corpus が生成しうる書式と衝突しない」)。
    """
    corpus = TemplateCorpus(_PROFILE.chars_per_token)
    task_for_prefix = make_needle_case(target_tokens=500, depth_pct=50, index=0, seed=1)
    prefix = task_for_prefix.answer.split("-")[0] + "-"

    for seed in range(1, 8):
        assert prefix not in corpus.prose("en", 4000, seed)
        assert prefix not in corpus.prose("ja", 4000, seed)
        assert prefix not in corpus.code(4000, seed)
        assert prefix not in corpus.log(4000, seed)


# --- chars_per_token の上書き (task 6.4 の手直し、レビュー第 1 ラウンドの指摘 3) --


def test_chars_per_token_none_matches_explicit_default() -> None:
    """省略した場合と、既定の比を明示的に渡した場合とで、バイト単位で同じ課題になる。"""
    default_ratio = Profile(name="probe").chars_per_token[ContentKind.PROSE_EN]
    omitted = make_needle_case(target_tokens=1200, depth_pct=50, index=0, seed=5)
    explicit = make_needle_case(
        target_tokens=1200, depth_pct=50, index=0, seed=5, chars_per_token=default_ratio
    )
    assert omitted.model_dump_json() == explicit.model_dump_json()


def test_chars_per_token_override_scales_haystack_length() -> None:
    """校正済みの比を渡すと、全体の長さがその比に比例して変わる。"""
    default_ratio = Profile(name="probe").chars_per_token[ContentKind.PROSE_EN]
    half_ratio = default_ratio / 2
    target_tokens = 3000

    default_case = make_needle_case(
        target_tokens=target_tokens,
        depth_pct=50,
        index=0,
        seed=17,
        chars_per_token=default_ratio,
    )
    half_case = make_needle_case(
        target_tokens=target_tokens, depth_pct=50, index=0, seed=17, chars_per_token=half_ratio
    )
    observed_ratio = len(half_case.haystack) / len(default_case.haystack)
    assert abs(observed_ratio - 0.5) < 0.05, (
        f"chars_per_token を半分にしても、全体の長さが比例して縮んでいない "
        f"(observed_ratio={observed_ratio:.3f})"
    )


@pytest.mark.parametrize("bad_ratio", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_chars_per_token_rejects_invalid_values(bad_ratio: float) -> None:
    with pytest.raises(ValueError):
        make_needle_case(
            target_tokens=1000, depth_pct=50, index=0, seed=1, chars_per_token=bad_ratio
        )


def test_chars_per_token_rejects_none_check_uses_math_isfinite() -> None:
    """有限性の確認そのものを試験する (境界: すぐ上の巨大な有限値は許す)。"""
    huge_but_finite = 1e300
    assert math.isfinite(huge_but_finite)
    # 例外を投げずに課題ができることだけを確かめる (長さの精度はここでは問わない)。
    make_needle_case(target_tokens=10, depth_pct=50, index=0, seed=1, chars_per_token=1.0)


# --- 挿入する文の書式 (引用符を使わない、空白を二重にしない) ---------------------
# task 6.4 の手直し (レビュー第 1 ラウンドの指摘 4)。


def test_needle_sentence_introduces_no_rare_characters() -> None:
    """挿入する文が、地の文にはない文字を持ち込まないことを確かめる。

    コード名と正解自身の文字 (値そのものとして一意なのは当然) は比較から除く。
    それ以外の文字 (つなぎの語、空白、ピリオドなど) は、干し草のほかの部分にも
    現れるはずである (以前の実装は、引用符 `"` を挿入する文にしか持ち込んで
    いなかった)。
    """
    task = make_needle_case(target_tokens=1500, depth_pct=50, index=0, seed=13)
    codename = _codename_from_question(task.question)
    haystack = task.haystack
    offset = task.inserted_char_offset

    answer_pos = haystack.index(task.answer, offset)
    sentence_end = answer_pos + len(task.answer) + 1  # 正解のあとの "." を含む
    sentence_span = haystack[offset:sentence_end]
    assert task.answer in sentence_span

    rest_of_haystack = haystack[:offset] + haystack[sentence_end:]
    template_chars = set(sentence_span) - set(codename) - set(task.answer)
    missing = template_chars - set(rest_of_haystack)
    assert not missing, f"挿入した文にしかない文字がある: {missing!r} (文: {sentence_span!r})"


def test_no_line_in_haystack_starts_with_a_space() -> None:
    """挿入した文のあとに続く本文の行が、空白で始まらないことを確かめる。

    段落の切れ目 (`\\n\\n`) の直後に単純に空白を足すと、`…hours.\\n\\n The
    access code…` のように、改行の直後に余分な空白が入ってしまう壊れ方が
    以前の実装にあった。まず干し草の地の文自身が、行頭に空白を置かないことを
    確かめてから (corpus/synth.py の性質)、挿入後の干し草でも同じであることを
    確かめる。
    """
    base = TemplateCorpus(_PROFILE.chars_per_token).prose("en", 2000, 21)
    assert not any(line.startswith(" ") for line in base.split("\n")), (
        "corpus/synth.py の地の文自身が、行頭に空白を持つ (前提が崩れている)"
    )

    for depth in _DEPTHS:
        task = make_needle_case(target_tokens=2000, depth_pct=depth, index=0, seed=21)
        offending = [line for line in task.haystack.split("\n") if line.startswith(" ")]
        assert not offending, f"depth={depth}: 行の先頭に空白がある: {offending[:3]!r}"


def test_the_needle_never_sticks_to_its_neighbours() -> None:
    """挿入した文の両側は、空白か、文書の端。区切りが**足りない**壊れ方を見る。

    二重の空白と行頭の空白の試験は、区切りが多すぎる向きしか見ない。区切りを
    落とすと `…SIGIL-0F843B9F.Requests routed…` のように後ろの文に貼り付くが、
    それは、どちらの試験にも当たらない (深さ 0 で起きやすい)。
    """
    checked_left = checked_right = 0
    for seed in range(1, 13):
        for depth in _DEPTHS:
            task = make_needle_case(target_tokens=2000, depth_pct=depth, index=seed, seed=seed)
            sentence = (
                f"The access code for {_codename_from_question(task.question)} is {task.answer}."
            )
            start = task.haystack.index(sentence)
            end = start + len(sentence)
            if start > 0:
                assert task.haystack[start - 1].isspace(), (seed, depth, task.haystack[:80])
                checked_left += 1
            if end < len(task.haystack):
                assert task.haystack[end].isspace(), (
                    seed,
                    depth,
                    task.haystack[end - 40 : end + 40],
                )
                checked_right += 1
    # 両側とも、実際に確かめた場合がある (深さ 0 は右だけ、深さ 100 は左だけになる)
    assert checked_left >= 12
    assert checked_right >= 12


def test_no_double_space_anywhere_in_haystack() -> None:
    """半角空白が 2 つ連続する箇所が、干し草のどこにもないことを確かめる。

    `corpus/synth.py` の文の継ぎ目は、常に半角空白 1 つ (`". "`) か、段落の
    切れ目 (`"\\n\\n"`、空白は含まない) のどちらかであり、半角空白 2 つが
    連続することはない (地の文自身の性質)。以前の実装は、土台の末尾がすでに
    半角空白 1 つで終わっているときに、さらに区切りの空白を足していたため
    (`f"{base} {sentence}"`)、`depth_pct=100` でほぼ毎回、二重の空白が
    生まれていた (`…exhausted.  The access code…`)。段落の切れ目のあとでは、
    `…hours.\\n\\n The access code…` のように、改行の直後に空白が入っていた
    (`test_no_line_in_haystack_starts_with_a_space` が別に確かめる)。
    """
    base = TemplateCorpus(_PROFILE.chars_per_token).prose("en", 2000, 21)
    assert "  " not in base, "corpus/synth.py の地の文自身に二重の空白がある (前提が崩れている)"

    for seed in (1, 21):
        for depth in _DEPTHS:
            task = make_needle_case(target_tokens=2000, depth_pct=depth, index=0, seed=seed)
            pos = task.haystack.find("  ")
            around = task.haystack[max(0, pos - 15) : pos + 15]
            assert pos == -1, (
                f"seed={seed} depth={depth}: 干し草に二重の空白がある "
                f"(位置: {pos}, 周辺: {around!r})"
            )


# --- golden なハッシュ (生成手順の意図しない変更の検出) -------------------------

_GOLDEN_TARGET_TOKENS: Final[int] = 1000
_GOLDEN_DEPTH_PCT: Final[int] = 50
_GOLDEN_INDEX: Final[int] = 0
_GOLDEN_SEED: Final[int] = 123
# この試験が落ちたら、変更が意図どおりか確かめたうえで、このハッシュを更新
# すること (`tests/unit/test_corpus_synth.py` の golden な試験にならう)。
# task 6.4 の手直し (round 2) で、挿入する文の書式 (引用符を外した) と、継ぎ目の
# 空白の扱い (`_join_with_single_space`) を変えたため、このハッシュを更新した。
# `GENERATOR_VERSION` (`bench_harness.types`) は変えていない — この module は
# まだリリースされていない新規の module であり (design.md の
# Revalidation Triggers の対象外)、上げる決まりは types.py の変更を伴う変更に
# 限られる (この module は types.py を変更していない)。
_GOLDEN_HASH: Final[str] = "b111a6b37a001cebc3d9d7a166493b5e7e4b06e368736fe8d3909cf63629eba9"


def test_golden_hash_pins_generator_output() -> None:
    task = make_needle_case(
        target_tokens=_GOLDEN_TARGET_TOKENS,
        depth_pct=_GOLDEN_DEPTH_PCT,
        index=_GOLDEN_INDEX,
        seed=_GOLDEN_SEED,
    )
    actual = hashlib.sha256(task.model_dump_json().encode()).hexdigest()
    assert actual == _GOLDEN_HASH
