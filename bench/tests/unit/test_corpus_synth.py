"""合成の文章を生成する部品の試験 (task 2.5)。

決定性 (プロセスと `PYTHONHASHSEED` をまたいで同じ)、長さの精度 (±2%)、
コードの構文の妥当性、先頭の一致 (prefix stability)、文の重複の少なさ、
`prefix_nonce` の性質を確かめる。golden なハッシュの試験は、型紙や生成の
手順を変えたときに `bench_harness.types.GENERATOR_VERSION` を上げ忘れて
いないかを検出する (失敗したら、変更が意図どおりか確かめてからハッシュを
更新し、GENERATOR_VERSION も上げること)。
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from typing import Final

import pytest

from bench_harness.corpus.synth import PREFIX_NONCE_HEX_LEN, TemplateCorpus, prefix_header_line
from bench_harness.types import ContentKind, Lang, Profile

PROFILE: Final[Profile] = Profile(name="quick")

# --- 助け -------------------------------------------------------------------


def _gen_prose_en(corpus: TemplateCorpus, target: int, seed: int) -> str:
    return corpus.prose("en", target, seed)


def _gen_prose_ja(corpus: TemplateCorpus, target: int, seed: int) -> str:
    return corpus.prose("ja", target, seed)


def _gen_code(corpus: TemplateCorpus, target: int, seed: int) -> str:
    return corpus.code(target, seed)


def _gen_log(corpus: TemplateCorpus, target: int, seed: int) -> str:
    return corpus.log(target, seed)


_KIND_GENERATORS: Final[list[tuple[ContentKind, Callable[[TemplateCorpus, int, int], str]]]] = [
    (ContentKind.PROSE_EN, _gen_prose_en),
    (ContentKind.PROSE_JA, _gen_prose_ja),
    (ContentKind.CODE, _gen_code),
    (ContentKind.LOG, _gen_log),
]

_ALL_GENERATE_FNS: Final[tuple[Callable[[TemplateCorpus, int, int], str], ...]] = tuple(
    fn for _, fn in _KIND_GENERATORS
)


def _is_japanese(ch: str) -> bool:
    return "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿"


def _strip_trailing_pad_comment(code_text: str) -> str:
    """`code()` の末尾が、長さ合わせのパディングの `#` コメント行なら取り除く。

    `synth.py` の module docstring「先頭の一致 (prefix stability)」に書いた、
    `code` だけの例外 (末尾のパディング行は target_tokens ごとに違う) に対応する。
    """
    lines = code_text.splitlines(keepends=True)
    if lines and lines[-1].lstrip().startswith("#"):
        return "".join(lines[:-1])
    return code_text


_SLOT_RE: Final[re.Pattern[str]] = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")


# --- 決定性 -------------------------------------------------------------------


def test_repeat_calls_are_byte_identical() -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    for generate in _ALL_GENERATE_FNS:
        first = generate(corpus, 1_500, 9)
        second = generate(corpus, 1_500, 9)
        assert first == second


def test_different_seeds_produce_different_output() -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    for generate in _ALL_GENERATE_FNS:
        a = generate(corpus, 1_500, 1)
        b = generate(corpus, 1_500, 2)
        assert a != b


_HASH_SCRIPT: Final[str] = """
import hashlib
from bench_harness.corpus.synth import TemplateCorpus
from bench_harness.types import Profile

profile = Profile(name="quick")
corpus = TemplateCorpus(profile.chars_per_token)
parts = [
    corpus.prose("en", {target}, {seed}),
    corpus.prose("ja", {target}, {seed}),
    corpus.code({target}, {seed}),
    corpus.log({target}, {seed}),
]
print(hashlib.sha256("".join(parts).encode()).hexdigest())
"""


def _combined_hash(corpus: TemplateCorpus, target: int, seed: int) -> str:
    parts = [
        corpus.prose("en", target, seed),
        corpus.prose("ja", target, seed),
        corpus.code(target, seed),
        corpus.log(target, seed),
    ]
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def test_deterministic_across_process_and_pythonhashseed() -> None:
    """プロセスをまたいでも、`PYTHONHASHSEED` を変えても、同じハッシュになる。

    `hash()`、大域の乱数、dict/set の反復順序に依存していれば、ここで揺れる。
    """
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    target, seed = 600, 2026
    expected = _combined_hash(corpus, target, seed)

    script = _HASH_SCRIPT.format(target=target, seed=seed)
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


# --- 長さの精度 (±2%) ---------------------------------------------------------


@pytest.mark.parametrize("target_tokens", [200, 8_000, 32_000, 128_000])
@pytest.mark.parametrize("kind,generate", _KIND_GENERATORS)
def test_length_within_two_percent(
    kind: ContentKind, generate: Callable[[TemplateCorpus, int, int], str], target_tokens: int
) -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    text = generate(corpus, target_tokens, 4242)
    want = target_tokens * PROFILE.chars_per_token[kind]
    lower, upper = want * 0.98, want * 1.02
    assert lower <= len(text) <= upper, (
        f"{kind}: target={target_tokens} 狙い={want:.1f} 実際={len(text)}"
    )


def test_generation_time_bound_for_128k() -> None:
    """128k トークンの生成が、十分に余裕を持った時間 (5 秒) に収まる。1 回の測定。"""
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    for kind, generate in _KIND_GENERATORS:
        start = time.perf_counter()
        generate(corpus, 128_000, 999)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"{kind}: 128k トークンの生成に {elapsed:.2f} 秒かかった"


# --- コードの構文 --------------------------------------------------------------


def test_code_parses_at_several_lengths_including_after_trimming() -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    for target in (200, 350, 8_000, 32_000, 128_000):
        source = corpus.code(target, 17)
        ast.parse(source)  # SyntaxError なら試験がここで落ちる


# --- 先頭の一致 (prefix stability) ---------------------------------------------


def test_prefix_stability_prose_and_log() -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    for generate in (_gen_prose_en, _gen_prose_ja, _gen_log):
        smaller = generate(corpus, 200, 55)
        larger = generate(corpus, 8_000, 55)
        assert larger.startswith(smaller)
        largest = generate(corpus, 32_000, 55)
        assert largest.startswith(larger)


def test_prefix_stability_code_up_to_last_complete_statement() -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    smaller = corpus.code(200, 55)
    larger = corpus.code(8_000, 55)
    assert larger.startswith(_strip_trailing_pad_comment(smaller))
    largest = corpus.code(32_000, 55)
    assert largest.startswith(_strip_trailing_pad_comment(larger))


# --- 文の重複 ------------------------------------------------------------------


def test_sentence_duplication_bound() -> None:
    """32k トークンの文書で、最も多い完全一致の文でも全体の 1% 未満。"""
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    cases: tuple[tuple[Lang, str], ...] = (("en", "."), ("ja", "。"))
    for lang, sep in cases:
        text = corpus.prose(lang, 32_000, 11)
        sentences = [s.strip() for s in text.replace("\n", " ").split(sep) if s.strip()]
        counts = Counter(sentences)
        total = len(sentences)
        _, most_common_count = counts.most_common(1)[0]
        assert most_common_count / total < 0.01, f"{lang}: 最頻の文が {most_common_count}/{total}"


# --- 型紙の埋め残しと、日本語らしさ ---------------------------------------------


def test_no_leftover_template_slots() -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    for generate in _ALL_GENERATE_FNS:
        text = generate(corpus, 1_000, 3)
        assert _SLOT_RE.search(text) is None, f"埋め残しの型紙のスロットがある: {text[:200]!r}"


def test_japanese_prose_contains_japanese_characters() -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    text = corpus.prose("ja", 500, 3)
    assert any(_is_japanese(ch) for ch in text)


# --- ログの時刻 ----------------------------------------------------------------


def test_log_timestamps_deterministic_and_monotonic() -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    text_a = corpus.log(3_000, 71)
    text_b = corpus.log(3_000, 71)
    assert text_a == text_b

    lines = [line for line in text_a.splitlines() if line]
    timestamps = [line.split(" ", 1)[0] for line in lines]
    assert len(timestamps) > 1
    # 連続する行を組にするため、長さが 1 つ違う。strict=False は意図的
    assert all(a < b for a, b in zip(timestamps, timestamps[1:], strict=False))


# --- prefix_nonce --------------------------------------------------------------


def test_prefix_nonce_length_and_charset() -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    nonce = corpus.prefix_nonce("seed", 42, "cond", 3)
    assert len(nonce) == PREFIX_NONCE_HEX_LEN == 32
    assert all(ch in "0123456789abcdef" for ch in nonce)


def test_prefix_nonce_deterministic() -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    assert corpus.prefix_nonce(1, "x", 2) == corpus.prefix_nonce(1, "x", 2)


def test_prefix_nonce_ambiguous_boundaries_differ() -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    assert corpus.prefix_nonce("a", "bc") != corpus.prefix_nonce("ab", "c")
    assert corpus.prefix_nonce(1) != corpus.prefix_nonce("1")


def test_cold_nonce_varies_warm_nonce_stays_same() -> None:
    """効かない条件 (cold) は試行ごとに違い、効く条件 (warm) は試行をまたいで同じ。"""
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    seed, condition_key, run_id = 1, "prefill/cold/8k", "run-abc"

    cold_trial_0 = corpus.prefix_nonce(seed, condition_key, 0, run_id)
    cold_trial_1 = corpus.prefix_nonce(seed, condition_key, 1, run_id)
    assert cold_trial_0 != cold_trial_1

    warm_a = corpus.prefix_nonce(seed, condition_key)
    warm_b = corpus.prefix_nonce(seed, condition_key)
    assert warm_a == warm_b


def test_prefix_header_line_constant_length() -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    short = prefix_header_line(corpus.prefix_nonce(1, "a", 0, "run"))
    long = prefix_header_line(corpus.prefix_nonce(2, "condition/key", 999, "another-run-id"))
    assert len(short) == len(long)


def test_prefix_header_line_rejects_bad_nonce() -> None:
    with pytest.raises(ValueError):
        prefix_header_line("too-short")
    with pytest.raises(ValueError):
        prefix_header_line("Z" * PREFIX_NONCE_HEX_LEN)


# --- constructor の検証 ---------------------------------------------------------


def test_constructor_rejects_incomplete_chars_per_token() -> None:
    with pytest.raises(ValueError):
        TemplateCorpus({ContentKind.PROSE_EN: 4.0})


def test_constructor_rejects_non_positive_chars_per_token() -> None:
    bad = dict(PROFILE.chars_per_token)
    bad[ContentKind.CODE] = 0.0
    with pytest.raises(ValueError):
        TemplateCorpus(bad)


# --- golden なハッシュ (GENERATOR_VERSION の上げ忘れを検出する) -----------------

_GOLDEN_SEED: Final[int] = 123
_GOLDEN_TARGET: Final[int] = 300
# 型紙や生成の手順 (乱数の消費のしかた、境界の決め方など) を変えて、ここが
# 落ちたら、変更が意図どおりか確かめたうえで、このハッシュを更新し、
# bench_harness.types.GENERATOR_VERSION も上げること。
_GOLDEN_HASHES: Final[dict[str, str]] = {
    "prose_en": "4f4b152bc381f3d68d7288e0e4ed153d7ea291e566affd0eff19651c0fe5953e",
    "prose_ja": "dc0e5a1055e59b53cd01145d9e90a1ae57ccfb340d6f6ced460c7aa7025ae616",
    "code": "11c947dd37c46ee156085437a58d0ec11574c3985d93ecbd3bb0be53ac302852",
    "log": "03e49db8bf8d0c0e4648941c51437ca9b419372f0268ceece8484784c47579a3",
}


def test_golden_hashes_pin_generator_output() -> None:
    corpus = TemplateCorpus(PROFILE.chars_per_token)
    actual = {
        "prose_en": hashlib.sha256(
            corpus.prose("en", _GOLDEN_TARGET, _GOLDEN_SEED).encode()
        ).hexdigest(),
        "prose_ja": hashlib.sha256(
            corpus.prose("ja", _GOLDEN_TARGET, _GOLDEN_SEED).encode()
        ).hexdigest(),
        "code": hashlib.sha256(corpus.code(_GOLDEN_TARGET, _GOLDEN_SEED).encode()).hexdigest(),
        "log": hashlib.sha256(corpus.log(_GOLDEN_TARGET, _GOLDEN_SEED).encode()).hexdigest(),
    }
    assert actual == _GOLDEN_HASHES
