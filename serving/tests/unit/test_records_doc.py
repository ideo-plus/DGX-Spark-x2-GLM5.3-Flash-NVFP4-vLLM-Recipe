"""記録の書式と、判断の記録の骨組みの完了の状態を、機械で確かめる試験 (tasks.md 6.4)。

tasks.md 6.4 の完了の状態は「一覧の 1 件の項目が、要件 10.1 の項目をすべて持つ。試行の記録の
1 行が、要件 8.2 の項目をすべて持つ」である。設計 (design.md 「Baseline Procedure (文書)」
830-836 行) は、さらに次を定める: ADR は `0001` の見出しの構成に揃えること、`0001` の
「クリーンルーム」の節に参照した資料の一覧 (11.6) と作業者の文脈の切り離し (11.4) を書く
こと、すべての文書が「送った内容と応答の本文、認証の情報、`exl3-tp2` の中身を書かない」
(10.5) という決まりを先頭に持つこと。文書のタスクなので、振る舞いの試験 (RED → GREEN) の
代わりに、この完了の状態を、次の 5 つとして固定する。

1. `not-working.md` の 1 件の雛形が、要件 10.1 の項目 (番号、現象、再現の条件 — 構成の名前・
   イメージのダイジェスト・重みの `repo@revision`、誤りの文面または記録の抜粋、対応しそうな
   上流の issue や変更、回避できたかと方法、関わる段階) と、10.4 (上流に報告するかの判断) を
   すべて持つ
2. `attempts.md` の表の見出しが、要件 8.2 の項目 (日時、段、構成の名前、結果、止まった場所、
   次に進む理由、記録の置き場所) をすべて持つ
3. 判断の記録 4 本 (`0002`〜`0005`) の `##` の見出しが、`0001` の `##` の見出しと、同じ
   集合・同じ順である。**`0001` から読んで比べる** (`0001` にしかない「thinking の切り替え」
   の節は、その ADR の決めたことの中身であって、design.md が定める共通の見出しの構成には
   ない — design.md 830-836 行が挙げる 9 つ (状態、関係する要件、背景、決めたこと、
   採らなかった案、影響と限界、見直す条件、クリーンルーム、実機で確かめたこと) のうち、
   `##` の見出しになるのは後ろの 7 つだけで、そこにも「thinking の切り替え」はない。
   見出しの末尾の `(要件 …)` のような注記は、ADR ごとに関係する要件が違うので、比べる前に
   正規化して落とす)
4. 6 つの文書 (`not-working.md`、`attempts.md`、`0002`〜`0005`) の先頭 (最初の `##` の節、
   または最初の段落) に、「書かないこと」の決まり (10.5: 送った内容と応答の本文、認証の
   情報、`exl3-tp2` の中身) がある
5. 手順書 (`procedure.md`) が張る、リポジトリ内の相対リンクの道筋が、すべて実在する
   (`attempts.md`、`not-working.md` を含む)

RED: この試験は、6 つの文書のどれか 1 つでもなければ、そこで `pytest.fail` して落ちる
(`_read_text` を参照)。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest

SERVING_DIR: Final[Path] = Path(__file__).resolve().parents[2]
"""`serving/` (この試験から見て 2 つ上)。"""

REPO_ROOT: Final[Path] = SERVING_DIR.parent

DOCS_DIR: Final[Path] = REPO_ROOT / "docs" / "vllm-baseline"
DECISIONS_DIR: Final[Path] = REPO_ROOT / "docs" / "decisions"

PROCEDURE_PATH: Final[Path] = DOCS_DIR / "procedure.md"
NOT_WORKING_PATH: Final[Path] = DOCS_DIR / "not-working.md"
ATTEMPTS_PATH: Final[Path] = DOCS_DIR / "attempts.md"

ADR_0001_PATH: Final[Path] = DECISIONS_DIR / "0001-bench-harness-measurement-method.md"
NEW_ADR_PATHS: Final[tuple[Path, ...]] = (
    DECISIONS_DIR / "0002-vllm-baseline-image-and-weights.md",
    DECISIONS_DIR / "0003-vllm-baseline-interconnect.md",
    DECISIONS_DIR / "0004-vllm-baseline-messages-api.md",
    DECISIONS_DIR / "0005-vllm-baseline-exit-condition.md",
)

ALL_RECORD_DOC_PATHS: Final[tuple[Path, ...]] = (
    NOT_WORKING_PATH,
    ATTEMPTS_PATH,
    *NEW_ADR_PATHS,
)
"""「書かないこと」の決まりを持つべき 6 つの文書 (6.4 の完了の状態)。"""


# --- 文書の読み込み (RED: なければ、ここで落ちる) -----------------------


def _read_text(path: Path) -> str:
    """文書の全文 (なければ、そこで落とす)。"""
    if not path.is_file():
        pytest.fail(f"文書がない: {path}")
    return path.read_text(encoding="utf-8")


# --- 1. not-working.md の 1 件の書式 (要件 10.1、10.4) -------------------

_NOT_WORKING_TEMPLATE_HEADING: Final[str] = "### 件 0"

_NOT_WORKING_REQUIRED_MARKERS: Final[tuple[str, ...]] = (
    "番号",
    "現象",
    "構成の名前",
    "イメージのダイジェスト",
    "repo@revision",
    "誤りの文面または記録の抜粋",
    "対応しそうな上流の issue や変更",
    "回避できたかと方法",
    "関わる段階",
    "上流に報告するかの判断",
)
"""要件 10.1 の 6 項目 (現象、再現の条件の 3 つ、誤りの文面、上流の issue、回避、段階) +

10.4 (上流に報告するかの判断) + 番号 (design.md の書式が定める、一覧の中で引くための項目)。
"""


def _template_entry(text: str) -> str:
    """`not-working.md` の、書式を示す雛形の 1 件 (見出しから文書の末尾まで)。"""
    start = text.find(_NOT_WORKING_TEMPLATE_HEADING)
    if start < 0:
        pytest.fail(f"雛形の見出しが見つからない: {_NOT_WORKING_TEMPLATE_HEADING}")
    return text[start:]


def test_the_not_working_template_entry_has_every_field_of_requirement_10_1() -> None:
    """`not-working.md` の雛形の 1 件が、要件 10.1 (+10.4) の項目をすべて持つ。"""
    entry = _template_entry(_read_text(NOT_WORKING_PATH))
    missing = [marker for marker in _NOT_WORKING_REQUIRED_MARKERS if marker not in entry]
    assert not missing, f"雛形に欠けている項目: {missing}"


# --- 2. attempts.md の 1 行の書式 (要件 8.2) ------------------------------

_ATTEMPTS_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "日時",
    "段",
    "構成の名前",
    "結果",
    "止まった場所",
    "次に進む理由",
    "記録の置き場所",
)
"""要件 8.2 の全項目 (design.md 「Baseline Procedure (文書)」の `attempts.md` の 1 行の書式)。"""


def _table_header_cells(text: str) -> list[str]:
    """文書の中の、最初の表の見出し行のセル (飾りの `` ` `` を落とす)。"""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped:
            continue
        cells = [cell.strip().replace("`", "") for cell in stripped.strip("|").split("|")]
        if cells and cells[0]:
            return cells
    pytest.fail("表の見出し行が見つからない")


def test_the_attempts_table_header_has_every_field_of_requirement_8_2() -> None:
    """`attempts.md` の表の見出しが、要件 8.2 の項目をすべて、この順で持つ。"""
    text = _read_text(ATTEMPTS_PATH)
    # 「1 行の書式」の節の表 (説明用) ではなく、「表 (まだ空)」の節の、書き足す表を見る。
    heading = "## 表 (まだ空。試すたびに 1 行足す)"
    start = text.find(heading)
    assert start >= 0, f"見出しが見つからない: {heading}"
    header = _table_header_cells(text[start:])
    assert header == list(_ATTEMPTS_REQUIRED_COLUMNS)


# --- 3. 4 本の ADR の見出しの構成 (`0001` に揃える) -----------------------

_H2_HEADING: Final[re.Pattern[str]] = re.compile(r"^## (.+)$", re.MULTILINE)

_TRAILING_PAREN: Final[re.Pattern[str]] = re.compile(r"\s*\([^()]*\)\s*$")
"""見出しの末尾の `(要件 …)` のような注記 (ADR ごとに関係する要件の番号が違う)。"""

_CONTENT_SPECIFIC_TO_0001: Final[frozenset[str]] = frozenset(
    {"thinking の切り替え (タスク 8.3、8.4)"}
)
"""`0001` にはあるが、design.md 830-836 行が定める共通の見出しの構成には**ない**見出し。

bench-harness の Decision (thinking を切り替えない、という決めたことの中身) であって、
ADR という文書の型の一部ではない。design.md が挙げる 9 つ (状態、関係する要件、背景、
決めたこと、採らなかった案、影響と限界、見直す条件、クリーンルーム、実機で確かめたこと)
にも含まれない。
"""


def _h2_headings(text: str) -> list[str]:
    return [match.group(1).strip() for match in _H2_HEADING.finditer(text)]


def _normalized(heading: str) -> str:
    return _TRAILING_PAREN.sub("", heading).strip()


def _canonical_adr_headings() -> tuple[str, ...]:
    """`0001` の `##` の見出しから、design.md が定める共通の構成を取り出す (正規化ずみ)。"""
    raw = _h2_headings(_read_text(ADR_0001_PATH))
    common = [heading for heading in raw if heading not in _CONTENT_SPECIFIC_TO_0001]
    return tuple(_normalized(heading) for heading in common)


def test_adr_0001_has_the_seven_canonical_headings() -> None:
    """`0001` 自身が、design.md の定める 7 つの見出しを、過不足なく持つ (自己チェック)。"""
    assert _canonical_adr_headings() == (
        "背景",
        "決めたこと",
        "採らなかった案",
        "影響と限界",
        "見直す条件",
        "クリーンルーム",
        "実機で確かめたこと",
    )


@pytest.mark.parametrize("path", NEW_ADR_PATHS, ids=lambda path: path.name)
def test_each_new_adr_matches_the_0001_heading_structure(path: Path) -> None:
    """`0002`〜`0005` の `##` の見出しが、`0001` (正規化ずみ) と同じ集合・同じ順である。"""
    canonical = _canonical_adr_headings()
    actual = tuple(_normalized(heading) for heading in _h2_headings(_read_text(path)))
    assert actual == canonical


# --- 4. 6 つの文書の先頭にある「書かないこと」の決まり (要件 10.5) --------

_WRITE_NOTHING_MARK: Final[str] = "書かないこと"

_FORBIDDEN_CONTENT_MARKERS: Final[tuple[str, ...]] = (
    "送った内容",
    "認証の情報",
    "exl3-tp2",
)
"""要件 10.5: 送った内容と応答の本文、認証の情報、計測者が別に起動していた構成の中身。"""

_LEADING_LINE_LIMIT: Final[int] = 20
"""「先頭」と認める行数の上限 (見出し + 前置きの段落を許す余裕を持たせた値)。"""


@pytest.mark.parametrize("path", ALL_RECORD_DOC_PATHS, ids=lambda path: path.name)
def test_every_record_doc_states_the_write_nothing_rule_up_front(path: Path) -> None:
    """6 つの文書の先頭に、「書かないこと」の決まりと、その 3 項目がある。"""
    text = _read_text(path)
    mark_index = text.find(_WRITE_NOTHING_MARK)
    assert mark_index >= 0, f"「{_WRITE_NOTHING_MARK}」の決まりがない: {path}"

    leading_lines = text[:mark_index].count("\n")
    assert leading_lines < _LEADING_LINE_LIMIT, (
        f"「{_WRITE_NOTHING_MARK}」が文書の先頭にない ({leading_lines} 行目): {path}"
    )

    # 決まりの本文の範囲: マークから、次の `##` の見出しの手前まで (最後まで見出しがなければ末尾)。
    rest = text[mark_index:]
    next_heading = rest.find("\n## ", 1)
    window = rest if next_heading < 0 else rest[:next_heading]

    missing = [marker for marker in _FORBIDDEN_CONTENT_MARKERS if marker not in window]
    assert not missing, f"「書かないこと」の決まりに欠けている項目 {missing}: {path}"


# --- 5. procedure.md が張るリンクの道筋 -----------------------------------

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
    """`procedure.md` のリポジトリ内の相対リンクが、すべて実在するファイルを指す。"""
    text = _read_text(PROCEDURE_PATH)
    targets = _relative_link_targets(text)
    assert targets, "手順書にリポジトリ内の相対リンクが 1 つもない"

    missing = [target for target in targets if not (PROCEDURE_PATH.parent / target).is_file()]
    assert not missing, f"手順書のリンクの道筋が実在しない: {missing}"

    # attempts.md と not-working.md へのリンクが、実際に張られていることも確かめる
    # (procedure.md 自体は 6.3 の担当で、この試験は変えない。6.4 が作るまでは壊れていた)。
    assert "attempts.md" in targets
    assert "not-working.md" in targets
