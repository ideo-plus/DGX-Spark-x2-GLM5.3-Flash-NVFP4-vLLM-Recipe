"""重みの読み込み方の既定と比較の手順書 (`docs/vllm-baseline/full-context-procedure.md`) の試験。

文書のタスクなので、振る舞いの試験の代わりに、計画の完了契約 C6 を固定する。確かめること:

1. §1 に、既定の読み込み方 (`instanttensor`) と比較用の生成・読み取り検査のコマンド
   (`--load-format`、`--safetensors-load-strategy`、`--load-format auto`、
   `tp2-full-auto.toml`、`serve check`) と、#50・#37 の実測 (`約 3 分`) がある (C6)
2. §1 に、比べる起動ログの行 (`Loading weights took`、`Model loading took`) がある (C6)
3. §1 に、ページキャッシュをそろえる必要と、捨てる操作に「了承を得てから」と ⚠ がある (C6)

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

_SECTION_ONE_HEADING: Final[re.Pattern[str]] = re.compile(r"^## 1\.", re.MULTILINE)
"""§1 の見出しの行。"""

_ANY_HEADING: Final[re.Pattern[str]] = re.compile(r"^## ", re.MULTILINE)
"""§1 の本文の終わりを決める、次の見出し。"""


def _read_text(path: Path) -> str:
    """文書の全文 (なければ、そこで落とす)。"""
    if not path.is_file():
        pytest.fail(f"文書がない: {path}")
    return path.read_text(encoding="utf-8")


def _section_one_body(text: str) -> str:
    """§1 の本文 (見出しの行を除き、次の見出しの手前まで。見出しがなければ、そこで落とす)。"""
    heading = _SECTION_ONE_HEADING.search(text)
    if heading is None:
        pytest.fail("節の見出しが見つからない: ## 1.")
    rest = text[heading.end() :]
    next_heading = _ANY_HEADING.search(rest)
    return rest if next_heading is None else rest[: next_heading.start()]


# --- C7: 読み込み方を選ぶ構成の生成と検査 -------------------------------


@pytest.mark.parametrize("marker", REQUIRED_MARKERS)
def test_the_procedure_covers_the_load_format_config(marker: str) -> None:
    """§1 の本文に、読み込み方を選ぶ生成・検査の印がある (C7)。"""
    assert marker in _section_one_body(_read_text(DOC_PATH))
