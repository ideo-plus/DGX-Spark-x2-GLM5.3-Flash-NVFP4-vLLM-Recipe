"""MTP の計測手順書 (`docs/vllm-baseline/mtp-procedure.md`) の完了の状態を確かめる (C1、C8)。

文書のタスクなので、振る舞いの試験の代わりに、計画の完了契約 C1・C8 を固定する。確かめること:

1. 前提の節に、この版 (`0961bbae`) が MTP に対応していることと、既知の不具合 (`#58454`、
   `#57087`、`#55442`) の状態が書かれている (C1)
2. 先頭に「書かないこと」の決まり (10.5: 送った内容と応答の本文、認証の情報、`exl3-tp2` の
   中身) がある (C8)
3. 4 つの対象名 (`p2-nope-tp2-mtp1` / `mtp2` / `mtp3` / `mtp5`) がある (C8)
4. `decode` (`code/en`・`prose/ja` を含む) と `concurrency` の流し方がある (C8)
5. 受理長・1 ステップの時間・`tok/s` の比べ方がある (C8)
6. 出力が壊れていないかの確かめ方 (`repetition_loop`・`toolcall`) がある (C8)
7. 文脈が 2048 を超える条件の確認 (`#58454`) がある (C8)
8. 上限 (1800・3000) の確かめ方がある (C8)
9. 相対リンクの道筋が、すべて実在する (C8)
10. 本文に、既定の重みの読み込み方 (`instanttensor`、#50) と起動の目安 (`約 3 分`) がある (C6)

RED: 文書がなければ、`_read_text` で `pytest.fail` する。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest

SERVING_DIR: Final[Path] = Path(__file__).resolve().parents[2]
"""`serving/` (この試験から見て 2 つ上)。"""

REPO_ROOT: Final[Path] = SERVING_DIR.parent

DOC_PATH: Final[Path] = REPO_ROOT / "docs" / "vllm-baseline" / "mtp-procedure.md"

MTP_TARGET_NAMES: Final[tuple[str, ...]] = (
    "p2-nope-tp2-mtp1",
    "p2-nope-tp2-mtp2",
    "p2-nope-tp2-mtp3",
    "p2-nope-tp2-mtp5",
)
"""手順書が対象にする 4 つの構成名 (C7、C8)。"""

MTP_KNOWN_ISSUES: Final[tuple[str, ...]] = ("0961bbae", "58454", "57087", "55442")
"""前提の節に書く、この版と既知の不具合 (C1)。"""


def _read_text(path: Path) -> str:
    """文書の全文 (なければ、そこで落とす)。"""
    if not path.is_file():
        pytest.fail(f"文書がない: {path}")
    return path.read_text(encoding="utf-8")


# --- C1: 前提の節 ---------------------------------------------------------


@pytest.mark.parametrize("marker", MTP_KNOWN_ISSUES)
def test_the_procedure_names_the_build_and_the_known_issues(marker: str) -> None:
    """前提の節に、この版と、既知の不具合 3 つの番号がある (C1)。"""
    assert marker in _read_text(DOC_PATH)


# --- C8: 「書かないこと」の決まり -----------------------------------------

_WRITE_NOTHING_MARK: Final[str] = "書かないこと"

_FORBIDDEN_CONTENT_MARKERS: Final[tuple[str, ...]] = (
    "送った内容",
    "認証の情報",
    "exl3-tp2",
)

_LEADING_LINE_LIMIT: Final[int] = 20


def test_the_procedure_states_the_write_nothing_rule_up_front() -> None:
    """先頭に「書かないこと」の決まりと、その 3 項目がある (C8、10.5)。"""
    text = _read_text(DOC_PATH)
    mark_index = text.find(_WRITE_NOTHING_MARK)
    assert mark_index >= 0, f"「{_WRITE_NOTHING_MARK}」の決まりがない: {DOC_PATH}"

    leading_lines = text[:mark_index].count("\n")
    assert leading_lines < _LEADING_LINE_LIMIT, (
        f"「{_WRITE_NOTHING_MARK}」が文書の先頭にない ({leading_lines} 行目): {DOC_PATH}"
    )

    rest = text[mark_index:]
    next_heading = rest.find("\n## ", 1)
    window = rest if next_heading < 0 else rest[:next_heading]

    missing = [marker for marker in _FORBIDDEN_CONTENT_MARKERS if marker not in window]
    assert not missing, f"「書かないこと」の決まりに欠けている項目 {missing}: {DOC_PATH}"


# --- C8: 対象名と、流す suite ---------------------------------------------


@pytest.mark.parametrize("name", MTP_TARGET_NAMES)
def test_the_procedure_names_every_mtp_target(name: str) -> None:
    """4 つの対象名が、すべて書かれている (C8)。"""
    assert name in _read_text(DOC_PATH)


@pytest.mark.parametrize("marker", ["decode", "code/en", "prose/ja", "concurrency"])
def test_the_procedure_covers_the_decode_and_concurrency_suites(marker: str) -> None:
    """`decode` (code/en・prose/ja) と `concurrency` の流し方がある (C8)。"""
    assert marker in _read_text(DOC_PATH)


# --- C8: 比べる値と、壊れていないかの確かめ方 -----------------------------


@pytest.mark.parametrize("marker", ["受理長", "1 ステップ", "tok/s"])
def test_the_procedure_explains_how_to_compare_the_speed(marker: str) -> None:
    """受理長・1 ステップの時間・tok/s の比べ方がある (C8)。"""
    assert marker in _read_text(DOC_PATH)


@pytest.mark.parametrize("marker", ["repetition_loop", "toolcall"])
def test_the_procedure_explains_how_to_check_the_output_quality(marker: str) -> None:
    """出力が壊れていないかの確かめ方がある (C8)。"""
    assert marker in _read_text(DOC_PATH)


def test_the_procedure_covers_the_over_2048_context_check() -> None:
    """文脈が 2048 を超える条件の確認 (#58454) がある (C8)。"""
    text = _read_text(DOC_PATH)

    assert "2048" in text
    assert "58454" in text


@pytest.mark.parametrize("marker", ["1800", "3000"])
def test_the_procedure_explains_how_to_check_the_caps(marker: str) -> None:
    """上限 (GPU 1800、X925 3.0 GHz) の確かめ方がある (C8)。"""
    assert marker in _read_text(DOC_PATH)


def test_the_procedure_marks_the_mutating_blocks() -> None:
    """状態を変えるコマンドの塊に、⚠ の印がある (C8)。"""
    assert "⚠" in _read_text(DOC_PATH)


# --- C6: 既定の重みの読み込み ---------------------------------------------


@pytest.mark.parametrize("marker", ["instanttensor", "約 3 分"])
def test_the_procedure_states_the_default_weight_load_format(marker: str) -> None:
    """本文に、既定の重みの読み込み方 (#50) と起動の目安がある (C6)。"""
    assert marker in _read_text(DOC_PATH)


# --- C8: 相対リンクの実在 --------------------------------------------------

_MARKDOWN_LINK: Final[re.Pattern[str]] = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _relative_link_targets(text: str) -> list[str]:
    """本文中の `[text](path)` のうち、外部の URL でも見出しへのアンカーでもないもの。"""
    targets: list[str] = []
    for found in _MARKDOWN_LINK.finditer(text):
        target = found.group(1).strip()
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        targets.append(target)
    return targets


def test_every_relative_link_in_the_procedure_resolves_to_a_real_file() -> None:
    """手順書のリポジトリ内の相対リンクが、すべて実在するファイルを指す (C8)。"""
    text = _read_text(DOC_PATH)
    targets = _relative_link_targets(text)
    assert targets, "手順書にリポジトリ内の相対リンクが 1 つもない"

    missing = [target for target in targets if not (DOC_PATH.parent / target).is_file()]
    assert not missing, f"手順書のリンクの道筋が実在しない: {missing}"
