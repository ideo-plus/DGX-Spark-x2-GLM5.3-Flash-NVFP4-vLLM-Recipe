"""手順書 (`docs/vllm-baseline/*.md`) の契約試験が使う、Markdown の走査の助け。

手順書は、実機を操作するコマンドを書いた文書なので、契約試験が「⚠ (了承を得てから) の印」や
「`serve <サブコマンド>` の実在」を、文書のコード塊と見出しから調べる。この module は、その
走査の部分だけを持つ (文書ごとに違う「必要な印」と、期待する内容は、各試験が持つ)。

2 つの手順書の契約試験 (`tests/unit/test_k2_derived_weights_procedure_doc.py` と
`tests/unit/test_k2_vllm_overlay_procedure_doc.py`) が、この module の関数を共有する。⚠ の規則と、
hash の直後に並べ替えを求める規則の所有者は、ここだけである。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest


def read_text(path: Path) -> str:
    """文書の全文 (なければ、そこで落とす)。"""
    if not path.is_file():
        pytest.fail(f"文書がない: {path}")
    return path.read_text(encoding="utf-8")


# --- serve のサブコマンド ---------------------------------------------------

SERVE_CALL: Final[re.Pattern[str]] = re.compile(r"\bserve +([a-z][a-z-]*)(?![\w-])")
"""文書の中の `serve <サブコマンド>` (`p2-…` のような、数字を含む構成の名前は拾わない)。"""


def subparser_map(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """引数解析器の、サブコマンドの名前から解析器への対応。"""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action._name_parser_map)
    pytest.fail("build_parser() にサブコマンドの解析器がない")


# --- 相対リンク -------------------------------------------------------------

_MARKDOWN_LINK: Final[re.Pattern[str]] = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def relative_link_targets(text: str) -> list[str]:
    """本文中の `[text](path)` のうち、外部の URL でも見出しへのアンカーでもないもの。"""
    targets: list[str] = []
    for found in _MARKDOWN_LINK.finditer(text):
        target = found.group(1).strip()
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        targets.append(target)
    return targets


# --- コード塊の走査 ---------------------------------------------------------

_FENCE: Final[re.Pattern[str]] = re.compile(r"^\s*(`{3,}|~{3,})")
"""コード塊の囲い (開きと閉じ)。開きには言語名が続いてよい。"""

_HEADING: Final[re.Pattern[str]] = re.compile(r"^#{1,6}\s")


@dataclass(frozen=True)
class CodeBlock:
    """文書の中の 1 つのコード塊 (囲みで挟まれた部分)。"""

    open_line: int
    """開きの囲みの行番号 (1 起点)。"""

    close_line: int
    """閉じの囲みの行番号。"""

    lines: tuple[tuple[int, str], ...]
    """中身 (行番号と、その行)。"""

    @property
    def text(self) -> str:
        """中身の全文 (注釈の行も含む)。"""
        return "\n".join(line for _, line in self.lines)

    @property
    def commands(self) -> str:
        """中身のうち、注釈の行 (`#` で始まる行) を除いたもの。"""
        return "\n".join(line for _, line in self.lines if not line.lstrip().startswith("#"))


def code_blocks(text: str) -> list[CodeBlock]:
    """文書のコード塊を、現れた順に返す (`` ` `` と `~` の囲みの両方を見る)。"""
    blocks: list[CodeBlock] = []
    fence: str | None = None
    opened = 0
    body: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        found = _FENCE.match(line)
        if fence is None:
            if found is not None:
                fence, opened, body = found.group(1), number, []
            continue
        if found is not None and line.strip() == found.group(1) and _closes(fence, found.group(1)):
            blocks.append(CodeBlock(opened, number, tuple(body)))
            fence = None
        else:
            body.append((number, line))
    assert fence is None, f"{opened} 行目のコード塊が閉じていない"
    return blocks


def _closes(opening: str, closing: str) -> bool:
    """閉じの囲みが、開きの囲みと同じ文字で、同じ長さ以上か。"""
    return closing[0] == opening[0] and len(closing) >= len(opening)


def lines_inside(blocks: list[CodeBlock]) -> set[int]:
    """コード塊の中 (囲みの行も含む) にある行番号。見出しと取り違えないために使う。"""
    inside: set[int] = set()
    for block in blocks:
        inside.update(range(block.open_line, block.close_line + 1))
    return inside


def nearest_heading(lines: list[str], inside: set[int], before: int) -> str:
    """`before` 行目より前で、いちばん近い見出しの行 (コード塊の中の `#` の注釈は見ない)。"""
    for number in range(before - 1, 0, -1):
        if number not in inside and _HEADING.match(lines[number - 1]):
            return lines[number - 1]
    return ""


def preceding_paragraph(lines: list[str], before: int) -> str:
    """`before` 行目の、すぐ前の段落 (空行を飛ばし、空行かコード塊の囲みまでの連なり)。"""
    number = before - 1
    while number >= 1 and not lines[number - 1].strip():
        number -= 1
    collected: list[str] = []
    while number >= 1 and lines[number - 1].strip() and not _FENCE.match(lines[number - 1]):
        collected.append(lines[number - 1])
        number -= 1
    return "\n".join(reversed(collected))


# --- 実機を操作する塊の ⚠ ---------------------------------------------------

ASK_MARKS: Final[tuple[str, ...]] = ("⚠", "了承を得てから")
"""実機を操作する塊の直前に要る印 (⚠ と、計測者に了承を得てから行うこと)。"""

MACHINE_CALLS: Final[dict[str, re.Pattern[str]]] = {
    "ssh": re.compile(r"\bssh +"),
    "docker run": re.compile(r"\bdocker +run\b"),
    "docker stop": re.compile(r"\bdocker +stop\b"),
    "serve push": re.compile(r"\bserve +push(?![\w-])"),
    "serve verify": re.compile(r"\bserve +verify(?![\w-])"),
    "serve start": re.compile(r"\bserve +start(?![\w-])"),
    "serve stop": re.compile(r"\bserve +stop(?![\w-])"),
    "serve autostart set": re.compile(r"\bserve +autostart +set(?![\w-])"),
    "serve autostart clear": re.compile(r"\bserve +autostart +clear(?![\w-])"),
    "vllm-autostart-install": re.compile(r"\bvllm-autostart-install\b"),
    "vllm-autostart-uninstall": re.compile(r"\bvllm-autostart-uninstall\b"),
    "loginctl enable-linger": re.compile(r"\bloginctl +enable-linger\b"),
    "loginctl disable-linger": re.compile(r"\bloginctl +disable-linger\b"),
}
"""実機を操作する呼び出し (Spark に入る、コンテナを起こす・止める、配る、照合する、起動する、
止める、自動起動を指定する、見張りを設置・撤去する、linger を切り替える。issue #88 で
`docker stop`・`autostart set/clear`・見張りの install/uninstall・linger の 4 種を足した)。"""


def ask_mark_problems(text: str) -> tuple[set[str], list[str]]:
    """実機を操作するコード塊が書いている呼び出しの名前と、⚠ の印が足りない塊の説明。

    見るのは、その塊の直前の見出しと、直前の段落 (空行で区切られた、塊のすぐ上の連なり) を
    合わせた範囲である。節をまたいで、前の節の印を数えない。
    """
    lines = text.splitlines()
    blocks = code_blocks(text)
    inside = lines_inside(blocks)

    shown: set[str] = set()
    problems: list[str] = []
    for block in blocks:
        calls = sorted(
            name for name, pattern in MACHINE_CALLS.items() if pattern.search(block.text)
        )
        if not calls:
            continue
        shown.update(calls)
        heading = nearest_heading(lines, inside, block.open_line)
        paragraph = preceding_paragraph(lines, block.open_line)
        window = f"{heading}\n{paragraph}"
        missing = [mark for mark in ASK_MARKS if mark not in window]
        if missing:
            problems.append(
                f"{block.open_line} 行目からの、{calls} を含む塊の直前の見出しと段落に、"
                f"{missing} がない"
            )
    return shown, problems


# --- 命令行と、突き合わせの並べ替え -----------------------------------------

_CONTINUATION: Final[re.Pattern[str]] = re.compile(r"\\\n")
"""シェルの、行を継ぐバックスラッシュと改行。"""


def command_lines(text: str) -> list[str]:
    """文書のコード塊の命令行 (注釈の行を除き、バックスラッシュで継いだ行を 1 行にしたもの)。

    コード塊の外 (段落) の字面は見ない。
    """
    lines: list[str] = []
    for block in code_blocks(text):
        lines.extend(_CONTINUATION.sub(" ", block.commands).splitlines())
    return lines


_HASH_COMMAND: Final[re.Pattern[str]] = re.compile(r"\b(?:sha256sum|shasum\s+-a\s+256)\b")
"""2 台の写しと Mac の写しの、ファイルごとの SHA-256 を出す命令。"""

_SORTED_AFTER_HASH_COMMAND: Final[re.Pattern[str]] = re.compile(r"[^|)'\"]*\|\s*LC_ALL=C\s+sort\b")
"""hash コマンドの直後 (引数のあと、同じ引用符と括弧の中) の `| LC_ALL=C sort`。

`ssh` の引用符や `<( … )` の括弧を越えては探さない。片側の並べ替えを、もう片側のもので
代用させないため。
"""


def checksum_sort_problems(text: str) -> tuple[int, list[str]]:
    """文書のコード塊の命令行にある hash コマンドの数と、直後に並べ替えがないものの説明。

    段落と、コード塊の中の注釈の行 (`#` で始まる行) は、見ない。
    """
    checked = 0
    problems: list[str] = []
    for line in command_lines(text):
        for found in _HASH_COMMAND.finditer(line):
            checked += 1
            if _SORTED_AFTER_HASH_COMMAND.match(line, found.end()) is None:
                problems.append(
                    f"`{found.group(0)}` の直後に `| LC_ALL=C sort` がない: {line.strip()[:100]}"
                )
    return checked, problems
