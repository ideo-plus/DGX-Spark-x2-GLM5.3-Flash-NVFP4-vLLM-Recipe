"""手順書 (`docs/vllm-baseline/procedure.md`) の完了の状態を、機械で確かめる試験 (tasks.md 6.3)。

tasks.md 6.3 の完了の状態は「手順書の段と、構成の名前と、`serve` のコマンドが対応している。
要件 8.1 の順序と、手順書の順序が一致している」である。文書のタスクなので、振る舞いの試験
(RED → GREEN) の代わりに、その完了の状態を、次の 8 つとして固定する。

1. 手順書に現れる `serve <サブコマンド>` が、すべて `cli.build_parser()` の実在のサブコマンド
   である (`serve netcheck <確認>` は、入れ子の側まで見る)。逆に、実在するサブコマンドは、
   すべて手順書のどこかに現れる (手順書が、道具の一部だけを説明したままにならないように)
2. 手順書に現れる構成の名前が、`config.load_configs` で読める構成に実在する。まだ作っていない
   段 1 / 段 3 / 段 4 の名前は、**「この段で作る」と書いた行にだけ**現れる
3. 要件 8.1 の順序 (段 0 → 段 1 → 段 2 → 段 3 → 段 4) の見出しが、その順に現れる
4. research.md の「実機でしか決まらない」項目の番号が、段への割り当ての表にすべてあり、
   割り当てが design.md 「実機での確かめ」の組分けと一致している
5. 計測者に尋ねる場所の一覧が、要件 2.1、2.8、7.9、8.5、8.8、8.9、10.4 を漏らさない
6. 2 つの関門 (試す前に `attempts.md` を見る、通信の確認が通らなければ 2 台での起動に進まない)
   が、文として書かれている
7. **打つ順番**が固定されている。同じコード塊では `serve logs` が `serve stop` より前にあり
   (`serve stop` は記録を回収しないため。申し送り 3.5)、親の決めた大きな順序
   (`push` → `pull-image` → `image-licenses` → 段 0 用の取得 → 段 0 → `netcheck links` →
   `bandwidth` → `sanity` → 重みの取得 → 段 2) の初出が、その順に並ぶ
8. **状態を変えるコマンドを含むコード塊のすべてに、塊ごとに ⚠ の印がある** (総数の数え上げ
   では、1 つ抜けても気付けない。要件 2.1)

この試験は、Spark にも推論サーバーにも触らない (文書と、引数解析器と、構成の読み込みだけ)。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from serving_kit import cli
from serving_kit import config as c

SERVING_DIR: Final[Path] = Path(__file__).resolve().parents[2]
"""`serving/` (この試験から見て 2 つ上)。"""

REPO_ROOT: Final[Path] = SERVING_DIR.parent
PROCEDURE_PATH: Final[Path] = REPO_ROOT / "docs" / "vllm-baseline" / "procedure.md"
CONFIGS_PATH: Final[Path] = SERVING_DIR / "config" / "configs.toml"


# --- 文書の読み込みと、節の切り出し ------------------------------------


def _procedure_text() -> str:
    """手順書の全文 (なければ、そこで落とす)。"""
    if not PROCEDURE_PATH.is_file():
        pytest.fail(f"手順書がない: {PROCEDURE_PATH}")
    return PROCEDURE_PATH.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """`heading` で始まる節の本文 (次の `## ` の見出しまで)。"""
    start = text.find(heading)
    if start < 0:
        pytest.fail(f"節の見出しが見つからない: {heading}")
    rest = text[start + len(heading) :]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


def _table_rows(section: str) -> list[list[str]]:
    """節の中の表の行 (セルごとに分け、飾りの `` ` `` を落とす)。"""
    rows: list[list[str]] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip().replace("`", "") for cell in stripped.strip("|").split("|")]
        rows.append(cells)
    return rows


# --- 1. サブコマンド ----------------------------------------------------

_SERVE_CALL: Final[re.Pattern[str]] = re.compile(r"\bserve +([a-z][a-z-]*)(?![\w-])")
"""手順書の中の `serve <サブコマンド>` (`p1-…` のような、数字を含む構成の名前は拾わない)。"""

_NETCHECK_CALL: Final[re.Pattern[str]] = re.compile(r"\bserve +netcheck +([a-z][a-z-]*)(?![\w-])")


def _subparser_map(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """引数解析器の、サブコマンドの名前から解析器への対応。"""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action._name_parser_map)
    pytest.fail("build_parser() にサブコマンドの解析器がない")


def test_every_serve_command_in_the_procedure_exists() -> None:
    """手順書が打たせるコマンドが、すべて実在する (存在しない引数を書かないための歯止め)。"""
    text = _procedure_text()
    top = _subparser_map(cli.build_parser())
    checks = _subparser_map(top["netcheck"])

    used = {found.group(1) for found in _SERVE_CALL.finditer(text)}
    assert used, "手順書に `serve <サブコマンド>` が 1 つもない"
    assert not sorted(used - set(top)), f"実在しないサブコマンド: {sorted(used - set(top))}"

    used_checks = {found.group(1) for found in _NETCHECK_CALL.finditer(text)}
    assert used_checks, "手順書に `serve netcheck <確認>` が 1 つもない"
    unknown = sorted(used_checks - set(checks))
    assert not unknown, f"実在しない netcheck の確認: {unknown}"


def test_the_procedure_covers_every_serve_command() -> None:
    """実在するサブコマンドが、すべて手順書のどこかに現れる。"""
    text = _procedure_text()
    top = _subparser_map(cli.build_parser())
    checks = _subparser_map(top["netcheck"])

    used = {found.group(1) for found in _SERVE_CALL.finditer(text)}
    missing = sorted(set(top) - used)
    assert not missing, f"手順書に出てこないサブコマンド: {missing}"

    used_checks = {found.group(1) for found in _NETCHECK_CALL.finditer(text)}
    missing_checks = sorted(set(checks) - used_checks)
    assert not missing_checks, f"手順書に出てこない netcheck の確認: {missing_checks}"


# --- 2. 構成の名前 ------------------------------------------------------

_CONFIG_LIKE: Final[re.Pattern[str]] = re.compile(
    r"`((?:probe|p1|netcheck)-[a-z0-9-]+(?:<連番>)?)`"
)
"""構成の名前らしい語 (この 3 つの接頭辞が、`probe-`/`p1-`/`netcheck-` の名前を覆う)。"""

_PLANNED_CONFIGS: Final[dict[str, str]] = {
    "p1-nvfp4-tp2-x<連番>": "段 3",
    "p1-w4a16-tp2": "段 4",
}
"""まだ `configs.toml` にない構成 (design.md 「Baseline Procedure」。必要になった段で作る)。"""

_MADE_HERE: Final[str] = "この段で作る"

_DOCUMENTED_ELSEWHERE: Final[dict[str, str]] = {
    "glm53-tp2-mtp3-marlin": "k2-derived-weights-procedure.md"
}
"""`docs/vllm-baseline/procedure.md` (P1 の基盤の手順書) では扱わず、別の手順書に載せる構成
(issue #103)。`_CONFIG_LIKE` は `probe-`/`p1-`/`netcheck-` の接頭辞しか拾わないので、派生の
重みの構成 (`glm53-tp2-mtp3-marlin`) は、この対応表で「どこで扱うか」を明示する。"""


def test_every_config_name_in_the_procedure_is_real_or_marked() -> None:
    """構成の名前が実在するか、まだ作らないものは「この段で作る」と書いた行にだけある。

    コミット済みの全構成が、この手順書か、`_DOCUMENTED_ELSEWHERE` が指す別の手順書のどちらかに
    現れる (issue #103)。
    """
    text = _procedure_text()
    real = set(c.load_configs(CONFIGS_PATH, REPO_ROOT))

    seen: set[str] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        for found in _CONFIG_LIKE.finditer(line):
            name = found.group(1)
            seen.add(name)
            if name in real:
                continue
            assert name in _PLANNED_CONFIGS, f"{number} 行目: 実在しない構成の名前 {name}"
            assert _MADE_HERE in line, (
                f"{number} 行目: まだない構成 {name} を、"
                f"「{_MADE_HERE}」と書いていない行で使っている: {line.strip()}"
            )

    assert real - set(_DOCUMENTED_ELSEWHERE) <= seen, (
        f"手順書に出てこない構成: {sorted(real - set(_DOCUMENTED_ELSEWHERE) - seen)}"
    )
    assert set(_PLANNED_CONFIGS) <= seen, (
        f"手順書に、作る段を書いていない構成がある: {sorted(set(_PLANNED_CONFIGS) - seen)}"
    )
    for name, doc in _DOCUMENTED_ELSEWHERE.items():
        assert name in real, f"_DOCUMENTED_ELSEWHERE の {name} が、コミット済みの構成にない"
        other_path = REPO_ROOT / "docs" / "vllm-baseline" / doc
        assert name in other_path.read_text(encoding="utf-8"), (
            f"{other_path} に、構成の名前 {name} がない"
        )


# --- 3. 要件 8.1 の順序 -------------------------------------------------

_STAGE_HEADING: Final[re.Pattern[str]] = re.compile(r"^## 段 (\d)\b", re.MULTILINE)


def test_the_stage_headings_follow_the_order_of_requirement_8_1() -> None:
    """段の見出しが、要件 8.1 の順序 (0 → 1 → 2 → 3 → 4) で現れる。"""
    found = [one.group(1) for one in _STAGE_HEADING.finditer(_procedure_text())]
    assert found == ["0", "1", "2", "3", "4"]


# --- 4. 22 項目の割り当て -----------------------------------------------

_ITEMS_HEADING: Final[str] = "## 実機でしか決まらない 22 項目の、段への割り当て"

_ITEM_NUMBER: Final[re.Pattern[str]] = re.compile(r"^\d{1,2}[a-e]?$")

_ITEM_STAGES: Final[dict[str, str]] = {
    # 段 0 (research.md の 1〜4)
    "1": "段 0",
    "2": "段 0",
    "3": "段 0",
    "4": "段 0",
    # 通信の確認 (5〜11、14c、14d)
    "5": "通信の確認",
    "6": "通信の確認",
    "7": "通信の確認",
    "8": "通信の確認",
    "9": "通信の確認",
    "10": "通信の確認",
    "11": "通信の確認",
    "14c": "通信の確認",
    "14d": "通信の確認",
    # 取得 (12、14b、22)
    "12": "取得",
    "14b": "取得",
    "22": "取得",
    # 紙で片が付いた 2 つ (research.md の 13、14)
    "13": "解決済み",
    "14": "解決済み",
    # 段 2 (15〜18)
    "15": "段 2",
    "16": "段 2",
    "17": "段 2",
    "18": "段 2",
    # bench と確かめ (19〜21)
    "19": "bench と確かめ",
    "20": "bench と確かめ",
    "21": "bench と確かめ",
    # あとの比較 (14e)
    "14e": "あとの比較",
}
"""design.md 「実機での確かめ (手順書の段。試験ではない)」の組分け。"""


def test_the_hardware_only_items_are_assigned_to_stages() -> None:
    """22 項目の番号がすべてあり、割り当てが design.md の組分けと一致する。"""
    section = _section(_procedure_text(), _ITEMS_HEADING)
    assigned = {
        row[0]: row[2] for row in _table_rows(section) if row and _ITEM_NUMBER.match(row[0])
    }

    assert assigned == _ITEM_STAGES
    # 1〜22 の番号が、1 つも欠けていない (枝番の 14b〜14e も含めて上で固定している)
    assert {str(number) for number in range(1, 23)} <= set(assigned)


# --- 5. 計測者に尋ねる場所 ----------------------------------------------

_ASK_HEADING: Final[str] = "## 計測者に尋ねる場所 (一覧)"

_ASK_REQUIREMENTS: Final[frozenset[str]] = frozenset(
    {"2.1", "2.8", "7.9", "8.5", "8.8", "8.9", "10.4"}
)
"""状態を変える操作、恒久的な設定、終わったあとの停止、打ち切り、第二の重み、層の分割、上流。"""

_ASK_MARK: Final[str] = "⚠ 計測者に尋ねる"


def test_the_places_to_ask_the_operator_are_listed() -> None:
    """尋ねる場所の一覧が、7 つの要件を漏らさず、本文にも印が付いている。"""
    text = _procedure_text()
    section = _section(text, _ASK_HEADING)
    cited = {
        cell for row in _table_rows(section) for cell in row if re.fullmatch(r"\d+\.\d+", cell)
    }

    assert cited >= _ASK_REQUIREMENTS, f"尋ねる場所が足りない: {sorted(_ASK_REQUIREMENTS - cited)}"
    assert text.count(_ASK_MARK) >= len(_ASK_REQUIREMENTS), (
        f"本文の「{_ASK_MARK}」の印が {text.count(_ASK_MARK)} 件しかない"
    )


# --- 6. 2 つの関門 ------------------------------------------------------


def test_the_procedure_keeps_the_two_gates() -> None:
    """試す前に試行の記録を見る (8.3) と、通信の確認の関門 (4.7) が、文として書かれている。"""
    text = _procedure_text()
    assert "attempts.md" in text
    gate = [
        line
        for line in text.splitlines()
        if "netcheck" in line and "2 台での起動に進まない" in line
    ]
    assert gate, "通信の確認が通らなければ 2 台での起動に進まない、という関門の文がない"


# --- 7. 打つ順番 --------------------------------------------------------


@dataclass(frozen=True)
class _CodeBlock:
    """文書の中の 1 つのコード塊 (``` で囲まれた部分)。"""

    open_line: int
    """開きの ``` の行番号 (1 起点)。"""

    close_line: int
    """閉じの ``` の行番号。"""

    lines: tuple[tuple[int, str], ...]
    """中身 (行番号と、その行)。"""

    @property
    def text(self) -> str:
        return "\n".join(line for _, line in self.lines)


def _code_blocks(text: str) -> list[_CodeBlock]:
    """文書のコード塊を、現れた順に返す。"""
    blocks: list[_CodeBlock] = []
    opened: int | None = None
    body: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("```"):
            if opened is not None:
                body.append((number, line))
            continue
        if opened is None:
            opened, body = number, []
        else:
            blocks.append(_CodeBlock(opened, number, tuple(body)))
            opened, body = None, []
    assert opened is None, f"{opened} 行目のコード塊が閉じていない"
    return blocks


_LOGS_CALL: Final[re.Pattern[str]] = re.compile(r"\bserve +logs(?![\w-])")
_STOP_CALL: Final[re.Pattern[str]] = re.compile(r"\bserve +stop(?![\w-])")


def test_the_log_collection_comes_before_the_stop() -> None:
    """`serve stop` は記録を回収しないので、同じ塊では `serve logs` が先にある。

    申し送り 3.5 (`lifecycle.stop` は「記録は回収しない」)。終わりの節と、段 2 の 2.2 の
    2 か所で、この順序を固定する。
    """
    both = 0
    for block in _code_blocks(_procedure_text()):
        collected = [number for number, line in block.lines if _LOGS_CALL.search(line)]
        stopped = [number for number, line in block.lines if _STOP_CALL.search(line)]
        if not collected or not stopped:
            continue
        both += 1
        assert min(collected) < min(stopped), (
            f"{block.open_line} 行目からの塊で、`serve stop` が `serve logs` より先にある"
        )
    assert both >= 2, f"`serve logs` と `serve stop` を並べた塊が {both} 件しかない"


_ORDER: Final[tuple[tuple[str, str], ...]] = (
    ("配布", r"\bserve +push(?![\w-])"),
    ("イメージの取得", r"\bserve +pull-image(?![\w-])"),
    ("ライセンスの表記", r"\bserve +image-licenses(?![\w-])"),
    ("段 0 用の取得", r"\bserve +fetch +p1-fetch-nvfp4-probe(?![\w-])"),
    ("段 0", r"\bserve +probe +probe-pinned(?![\w-])"),
    ("直結の読み取り", r"\bserve +netcheck +links(?![\w-])"),
    ("帯域", r"\bserve +netcheck +bandwidth(?![\w-])"),
    ("事前の確認", r"\bserve +netcheck +sanity(?![\w-])"),
    ("重みの取得", r"\bserve +fetch +p1-fetch-nvfp4(?![\w-])"),
    ("段 2", r"\bserve +start +p1-nvfp4-tp2(?![\w-])"),
)
"""打つ順番 (design.md 「代替の順序」と、親の決めた大きな順序)。

見るのは**コード塊の中だけ**である (本文では、先の段のコマンドを、関門の説明として先に
名前で挙げることがある)。`p1-fetch-nvfp4` は、`p1-fetch-nvfp4-probe` に食われないように、
うしろの境界で切る。
"""


def test_the_commands_are_typed_in_the_required_order() -> None:
    """手順書のコード塊で、10 のコマンドの初出が、決めた順に並ぶ。"""
    blocks = _code_blocks(_procedure_text())
    seen: list[tuple[str, int]] = []
    for label, pattern in _ORDER:
        found = [
            number for block in blocks for number, line in block.lines if re.search(pattern, line)
        ]
        assert found, f"「{label}」のコマンドが、どのコード塊にもない ({pattern})"
        seen.append((label, min(found)))

    numbers = [number for _, number in seen]
    assert numbers == sorted(numbers), f"打つ順番が、決めた順になっていない: {seen}"


# --- 8. 状態を変える塊の印 ----------------------------------------------

_MUTATING_CALL: Final[re.Pattern[str]] = re.compile(
    r"\bserve +(?:push|pull-image|image-licenses|fetch|verify|start|stop|probe)(?![\w-])"
    r"|\bserve +netcheck +(?:bandwidth|sanity|ab)(?![\w-])"
)
"""Spark の状態を変える 11 のコマンド (`serving/README.md` のサブコマンドの表)。"""


def test_every_mutating_block_is_preceded_by_the_ask_mark() -> None:
    """状態を変えるコマンドを含むコード塊には、**塊ごとに**、直前の本文に ⚠ がある。

    総数の数え上げ (`>= 7`) では、1 つの塊の印が抜けても気付けない (要件 2.1)。見るのは、
    **その塊と同じ節の中で、直前**の範囲である (前のコード塊の終わりと、直前の見出しの、
    うしろにあるほうから、この塊の始まりまで)。節をまたいで、前の節の印を数えない。
    """
    text = _procedure_text()
    lines = text.splitlines()
    blocks = _code_blocks(text)
    headings = [number for number, line in enumerate(lines, start=1) if line.startswith("#")]
    checked = 0
    previous_end = 0
    for block in blocks:
        if _MUTATING_CALL.search(block.text):
            checked += 1
            last_heading = max(
                (number for number in headings if number < block.open_line), default=0
            )
            begin = max(previous_end, last_heading)
            window = "\n".join(lines[begin : block.open_line - 1])
            assert _ASK_MARK in window, (
                f"{block.open_line} 行目からの、状態を変える塊の直前に「{_ASK_MARK}」がない"
                f" (見た範囲: {begin + 1}〜{block.open_line - 1} 行目)"
            )
        previous_end = block.close_line
    assert checked >= 10, f"状態を変える塊が {checked} 件しか見つからない"
