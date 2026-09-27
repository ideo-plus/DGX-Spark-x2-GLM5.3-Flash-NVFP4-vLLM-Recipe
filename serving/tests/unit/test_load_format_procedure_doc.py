"""重みの読み込み方の既定と比較の手順書 (`docs/vllm-baseline/full-context-procedure.md`) の試験。

文書のタスクなので、振る舞いの試験の代わりに、計画の完了契約 C6 を固定する。確かめること:

1. §1 に、既定の読み込み方 (`instanttensor`) と比較用の生成・読み取り検査のコマンド
   (`--load-format`、`--safetensors-load-strategy`、`--load-format auto`、
   `tp2-full-auto.toml`、`serve check`) と、#50・#37 の実測 (`約 3 分`) がある (C6)
2. §1 に、比べる起動ログの行 (`Loading weights took`、`Model loading took`) がある (C6)
3. §1 に、ページキャッシュをそろえる必要と、捨てる操作に「了承を得てから」と ⚠ がある (C6)
4. §1 が、ページキャッシュを捨てる固定のコマンド (`sudo /usr/local/sbin/spark-drop-caches`。
   `ops/spark-drop-caches/`) を指し、§3 が、起動の前の関門 `memory_free` の断りと、断られたときの
   手順 (⚠ 了承を得てからページキャッシュを捨て、やり直す) を述べる (#84、C6)

#50 より前は比較用の構成の名前が `p2-nope-tp2-full-lf-instanttensor` だったが、既定が
`instanttensor` になったので、明示の `instanttensor` は比較の対照でなくなった。比較用は
`--load-format auto` で作る `tp2-full-auto.toml` に替えた。

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

REQUIRED_MARKERS: Final[tuple[str, ...]] = (
    "--load-format",
    "--safetensors-load-strategy",
    "instanttensor",
    "--load-format auto",
    "tp2-full-auto.toml",
    "serve check",
    "#50",
    "約 3 分",
    "Loading weights took",
    "Model loading took",
    "drop_caches",
    "了承",
    "⚠",
)
"""§1 の本文に要る印 (C6)。"""

_ANY_HEADING: Final[re.Pattern[str]] = re.compile(r"^## ", re.MULTILINE)
"""節の本文の終わりを決める、次の見出し。"""


def _read_text(path: Path) -> str:
    """文書の全文 (なければ、そこで落とす)。"""
    if not path.is_file():
        pytest.fail(f"文書がない: {path}")
    return path.read_text(encoding="utf-8")


def _section_body(text: str, number: int) -> str:
    """`## <number>.` の節の本文 (見出しの行を除き、次の見出しの手前まで。見出しがなければ、
    そこで落とす)。"""
    heading = re.search(rf"^## {number}\.", text, re.MULTILINE)
    if heading is None:
        pytest.fail(f"節の見出しが見つからない: ## {number}.")
    rest = text[heading.end() :]
    next_heading = _ANY_HEADING.search(rest)
    return rest if next_heading is None else rest[: next_heading.start()]


# --- C7: 読み込み方を選ぶ構成の生成と検査 -------------------------------


@pytest.mark.parametrize("marker", REQUIRED_MARKERS)
def test_the_procedure_covers_the_load_format_config(marker: str) -> None:
    """§1 の本文に、読み込み方を選ぶ生成・検査の印がある (C7)。"""
    assert marker in _section_body(_read_text(DOC_PATH), 1)


# --- #84: ページキャッシュを捨てる固定のコマンドと、関門 `memory_free` ------------

SECTION_ONE_FIX_MARKERS: Final[tuple[str, ...]] = (
    "/usr/local/sbin/spark-drop-caches",
    "ops/spark-drop-caches/README.md",
)
"""§1 が指す、ページキャッシュを捨てる固定のコマンドと、その導入の手順。"""

SECTION_THREE_REFUSAL_MARKERS: Final[tuple[str, ...]] = ("spark-drop-caches", "⚠", "了承を得てから")
"""`memory_free` の断りを述べる段落に要る印 (対処の固定のコマンドと、了承の印)。"""


@pytest.mark.parametrize("marker", SECTION_ONE_FIX_MARKERS)
def test_section_one_points_to_the_fixed_drop_caches_command(marker: str) -> None:
    """§1 が、ページキャッシュを捨てる固定のコマンドと、導入の手順 (`ops/`) を指す。"""
    assert marker in _section_body(_read_text(DOC_PATH), 1)


def test_section_three_explains_the_memory_free_refusal_and_the_way_out() -> None:
    """§3 に、`serve start` の関門 `memory_free` を述べた段落があり、その 1 つの段落に、
    断られたときの手順 (⚠ 了承を得てから、固定のコマンドでページキャッシュを捨てる) がある。"""
    paragraphs = re.split(r"\n\s*\n", _section_body(_read_text(DOC_PATH), 3))

    explaining = [paragraph for paragraph in paragraphs if "memory_free" in paragraph]

    assert explaining, "§3 に、関門 `memory_free` を述べた段落がない"
    assert any(
        all(marker in paragraph for marker in SECTION_THREE_REFUSAL_MARKERS)
        for paragraph in explaining
    ), f"`memory_free` を述べた段落に、{SECTION_THREE_REFUSAL_MARKERS} がそろっていない"
