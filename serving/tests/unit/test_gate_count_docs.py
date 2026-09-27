"""`docs/vllm-baseline/` で関門の数を書く手順書の、関門の数と名前の試験 (計画の完了契約 C6・
C6b、#84)。

関門 `memory_free` (起動の前のメモリの空き) を足したので、`serve check` の関門は 9 つになった。
`docs/vllm-baseline/` で関門の数を書く手順書は 7 本 (`k2-derived-weights-procedure.md`、
`full-context-procedure.md`、`patched-tp2-procedure.md`、`initial-benchmark-procedure.md`、
`procedure.md`、`mtp-procedure.md`、`k2-profile-procedure.md`) で、この試験はその 7 本を見る
(`k2-vllm-overlay-procedure.md` など、関門の数を書かない手順書は対象にしない)。確かめること:

1. 対象の 7 本のどの手順書にも、「8 つの関門」「8 関門」「全 8 関門」の記述が残らない
   (改行を挟んだ「8 つの / 関門」も含む)
2. `patched-tp2-procedure.md` の、`serve check` の出力の関門の名前を説明する行 (§6 と、
   やり直しの節) が、数を「9 種」と書き、§6 の行が 9 つの名前を、`disk_space` の次に
   `memory_free` を置いた順に、すべて挙げる

各手順書の、関門 `memory_free` に触れる文と、ページキャッシュを捨てる固定のコマンドの塊は、
それぞれの手順書の試験 (`test_k2_derived_weights_procedure_doc.py`、
`test_load_format_procedure_doc.py`) が固定する。

RED: 文書がなければ、`read_text` で `pytest.fail` する。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from procedure_doc_kit import read_text

SERVING_DIR: Final[Path] = Path(__file__).resolve().parents[2]
"""`serving/` (この試験から見て 2 つ上)。"""

REPO_ROOT: Final[Path] = SERVING_DIR.parent

PROCEDURE_DIR: Final[Path] = REPO_ROOT / "docs" / "vllm-baseline"

PATCHED_TP2_DOC: Final[Path] = PROCEDURE_DIR / "patched-tp2-procedure.md"

IN_SCOPE_DOCS: Final[tuple[str, ...]] = (
    "k2-derived-weights-procedure.md",
    "full-context-procedure.md",
    "patched-tp2-procedure.md",
    "initial-benchmark-procedure.md",
    "procedure.md",
    "mtp-procedure.md",
    "k2-profile-procedure.md",
)
"""`docs/vllm-baseline/` で関門の数を書く手順書だけ (関門の数を書かない手順書は対象にしない)。"""

EIGHT_GATES: Final[re.Pattern[str]] = re.compile(r"(?<![0-9])8\s*(?:つの)?\s*関門")
"""「8 つの関門」「8 関門」(「全 8 関門」を含む)。「つの」と「関門」の間の改行も拾う。"""

GATE_NAMES_IN_ORDER: Final[tuple[str, ...]] = (
    "reachable",
    "own_state",
    "gpu_idle",
    "layout",
    "image_digest",
    "weights_verified",
    "disk_space",
    "memory_free",
    "ports_free",
)
"""`serve check` が並べる関門の名前 (流す順)。`memory_free` は `disk_space` の次にある。"""


def test_no_in_scope_procedure_still_says_there_are_eight_gates() -> None:
    """対象の 7 本のどの手順書にも、関門が 8 つという記述が残らない (関門は 9 つ)。"""
    stale: list[str] = []
    for name in IN_SCOPE_DOCS:
        path = PROCEDURE_DIR / name
        text = read_text(path)
        for found in EIGHT_GATES.finditer(text):
            line_number = text.count("\n", 0, found.start()) + 1
            stale.append(f"{path.name}:{line_number}")

    assert not stale, f"「8 つの関門」の記述が残っている: {stale}"


def _gate_name_lines(text: str) -> list[str]:
    """`serve check` の出力の `gate.N.name` を説明している行。"""
    return [line for line in text.splitlines() if "gate.N.name" in line]


def test_patched_tp2_says_nine_kinds_wherever_it_describes_the_gate_names() -> None:
    """`gate.N.name` を説明する行のそれぞれ (§6 の確認と、やり直しの節の確認) が、数を
    「9 種」と書く。"""
    lines = _gate_name_lines(read_text(PATCHED_TP2_DOC))
    assert len(lines) >= 2, f"`gate.N.name` を説明する行が 2 つ未満: {len(lines)}"

    assert [line for line in lines if "9 種" not in line] == []


def test_patched_tp2_lists_the_gate_names_in_the_order_they_run() -> None:
    """§6 の、関門の名前を挙げる行が、9 つの名前を、`memory_free` を `disk_space` の次に
    した順に、すべて挙げる。"""
    listing = [
        line for line in _gate_name_lines(read_text(PATCHED_TP2_DOC)) if "`reachable`" in line
    ]
    assert len(listing) == 1, f"関門の名前を挙げる行が 1 つでない: {len(listing)}"

    listed = [
        name for name in re.findall(r"`([a-z_]+)`", listing[0]) if name in GATE_NAMES_IN_ORDER
    ]
    assert listed == list(GATE_NAMES_IN_ORDER)
