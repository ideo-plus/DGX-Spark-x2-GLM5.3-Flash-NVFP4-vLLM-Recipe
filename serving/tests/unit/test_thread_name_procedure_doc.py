"""NCCL のスレッドの名前を確かめる手順書 (`docs/vllm-baseline/full-context-procedure.md`) の試験。

文書のタスクなので、振る舞いの試験の代わりに、計画の完了契約 C4・C5 を固定する。確かめること:

1. `--nccl-thread-names` を付けた構成 (`p2-nope-tp2-full-tn`) の生成・読み取り検査のコマンドと、
   生成する TOML の名前 (`tp2-full-tn.toml`) がある (C5)
2. 推論中に `top -H -b -n 1` を読み取りだけで実行する手順と、付ける env (`NCCL_SET_THREAD_NAME=1`)
   が文書にあり、§5.1 の本文にスレッドの使用率 (`%CPU`) を記録する指示がある (C5)
3. 期待 (NCCL の proxy が `NCCL Progress` などの名前で見える) と、名前が違ったときは結論を
   出さないことが書かれている (C5)
4. bench は対象を足さず、§5.1 の本文に、既存の対象名 `p2-nope-tp2-full` での `bench run` と、
   記録に書く構成名 `p2-nope-tp2-full-tn` がある (C4)
5. 相対リンクの道筋が、すべて実在する (C5)

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

DOC_PATH: Final[Path] = REPO_ROOT / "docs" / "vllm-baseline" / "full-context-procedure.md"

THREAD_NAME_CONFIG: Final[str] = "p2-nope-tp2-full-tn"
"""thread-name を付けた構成の名前 (C4、C5)。"""

GENERATION_MARKERS: Final[tuple[str, ...]] = (
    "--nccl-thread-names",
    THREAD_NAME_CONFIG,
    "tp2-full-tn.toml",
)
"""生成・読み取り検査のコマンドの印 (C5)。"""

BENCH_TARGET: Final[str] = "--target p2-nope-tp2-full"
"""対象を足さずに測るときの、既存の bench の対象名 (C4)。"""

_EXISTING_TARGET_RUN: Final[re.Pattern[str]] = re.compile(
    r"\bbench run " + re.escape(BENCH_TARGET) + r"(?![\w-])"
)
"""既存の対象名での `bench run`。`-tn` 付きの対象名 (`p2-nope-tp2-full-tn`) には当たらない (C4)。"""

_THREAD_NAME_SECTION_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^### 5\.1(?!\d).*$", re.MULTILINE
)
"""§5.1 の見出しの行全体 (見出しの行に書かれた構成名を、本文として数えないため)。"""

_ANY_HEADING: Final[re.Pattern[str]] = re.compile(r"^#{1,3} ", re.MULTILINE)
"""§5.1 の本文の終わりを決める、次の見出し。"""


def _read_text(path: Path) -> str:
    """文書の全文 (なければ、そこで落とす)。"""
    if not path.is_file():
        pytest.fail(f"文書がない: {path}")
    return path.read_text(encoding="utf-8")


def _thread_name_section_body(text: str) -> str:
    """§5.1 の本文 (見出しの行を除き、次の見出しの手前まで。見出しがなければ、そこで落とす)。"""
    heading = _THREAD_NAME_SECTION_HEADING.search(text)
    if heading is None:
        pytest.fail("節の見出しが見つからない: ### 5.1")
    rest = text[heading.end() :]
    next_heading = _ANY_HEADING.search(rest)
    return rest if next_heading is None else rest[: next_heading.start()]


# --- C5: thread-name 付きの構成の生成と検査 -------------------------------


@pytest.mark.parametrize("marker", GENERATION_MARKERS)
def test_the_procedure_generates_and_checks_the_thread_name_config(marker: str) -> None:
    """`--nccl-thread-names` を付けた構成の生成・検査のコマンドの印がある (C5)。"""
    assert marker in _read_text(DOC_PATH)


# --- C5: スレッドの名前の読み方 -------------------------------------------


def test_the_procedure_reads_the_thread_names_and_records_them() -> None:
    """推論中に `top -H -b -n 1` を読み取りだけで実行し、名前と `%CPU` を記録すると書いてある。"""
    text = _read_text(DOC_PATH)

    assert "top -H -b -n 1" in text
    assert "NCCL_SET_THREAD_NAME=1" in text
    assert "%CPU" in _thread_name_section_body(text), "§5.1 に、%CPU を記録する指示がない"


def test_the_procedure_names_the_expected_nccl_thread_and_the_no_conclusion_rule() -> None:
    """期待する名前 (`NCCL Progress`) と、名前が違ったときは結論を出さないことが書いてある (C5)。"""
    text = _read_text(DOC_PATH)

    assert "NCCL Progress" in text
    assert "結論" in text


# --- C4: 既存の bench の対象名で測る --------------------------------------


def test_the_procedure_measures_under_the_existing_bench_target() -> None:
    """`-tn` の確認は、対象を足さず既存の対象名で測ると §5.1 に書いてある (C4)。"""
    section = _thread_name_section_body(_read_text(DOC_PATH))

    assert _EXISTING_TARGET_RUN.search(section), "§5.1 に、既存の対象名での bench run がない"
    assert THREAD_NAME_CONFIG in section, "§5.1 の本文に、記録に書く構成名がない"


# --- C5: 相対リンクの実在 --------------------------------------------------

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
    """手順書のリポジトリ内の相対リンクが、すべて実在するファイルを指す (C5)。"""
    text = _read_text(DOC_PATH)
    targets = _relative_link_targets(text)
    assert targets, "手順書にリポジトリ内の相対リンクが 1 つもない"

    missing = [target for target in targets if not (DOC_PATH.parent / target).is_file()]
    assert not missing, f"手順書のリンクの道筋が実在しない: {missing}"
