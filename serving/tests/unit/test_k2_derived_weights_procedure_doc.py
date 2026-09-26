"""K2 の派生の重みを作って使う手順書 (`docs/vllm-baseline/k2-derived-weights-procedure.md`) の試験。

文書のタスクなので、振る舞いの試験の代わりに、手順書の完了の状態を固定する。確かめること:

1. 先頭 20 行以内に「書かないこと」の決まり (要求・応答の本文、認証の情報、`exl3-tp2` の
   中身) がある
2. 派生の重みの構成の生成 (`--weights k2s1` / `tp2-full-k2s1.toml` / `p2-nope-tp2-full-k2s1`)、
   コミットする正解のマニフェスト (`serving/weights/` / `k2s1.manifest.json` / `kind` /
   `derivation`)、変換の道筋 (`CPU` / `readonly`)、2 台の結果の一致の確認 (`cmp` /
   `sha256sum`) の印がある
3. 実機の照合と関門と起動と回収と停止 (`serve verify` / `serve check` / `serve start` /
   `serve smoke` / `serve logs` / `serve stop`) の印と、状態を変える塊の ⚠ の印がある
4. 使う `serve <サブコマンド>` が、すべて `cli.build_parser()` に実在する
5. 相対リンクの道筋が、すべて実在する
6. `ssh`・`docker run`・`serve verify`・`serve start` を含むコード塊の**それぞれに**、直前の
   見出しか直前の段落に、⚠ と「了承を得てから」がある (総数の数え上げでは、1 つの塊の印が
   抜けても気付けない)
7. 変換の `docker run` は、GPU を渡さず (CPU だけ)、元の重みを `readonly` で結び付ける

RED: 文書がなければ、`_read_text` で `pytest.fail` する。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

import pytest

from serving_kit import cli
from serving_kit import config as c

SERVING_DIR: Final[Path] = Path(__file__).resolve().parents[2]
"""`serving/` (この試験から見て 2 つ上)。"""

REPO_ROOT: Final[Path] = SERVING_DIR.parent

DOC_PATH: Final[Path] = REPO_ROOT / "docs" / "vllm-baseline" / "k2-derived-weights-procedure.md"

CONFIGS_PATH: Final[Path] = SERVING_DIR / "config" / "configs.toml"

ORIGIN_CONFIG: Final[str] = "p1-nvfp4-tp2"
"""変換の元の重み (`RedHatAI/GLM-5.3-Flash-NVFP4`) を置き場所に持つ、コミット済みの構成。"""

_WRITE_NOTHING_MARK: Final[str] = "書かないこと"

_FORBIDDEN_CONTENT_MARKERS: Final[tuple[str, ...]] = (
    "送った内容",
    "認証の情報",
    "exl3-tp2",
)

_LEADING_LINE_LIMIT: Final[int] = 20

REQUIRED_MARKERS: Final[tuple[str, ...]] = (
    "--weights k2s1",
    "p2-nope-tp2-full-k2s1",
    "tp2-full-k2s1.toml",
    "k2s1.manifest.json",
    "serving/weights/",
    "cmp",
    "sha256sum",
    "CPU",
    "readonly",
    "serve check",
    "serve verify",
    "serve start",
    "serve smoke",
    "serve logs",
    "serve stop",
    "⚠",
    "kind",
    "derivation",
)
"""手順書に必要な印。"""


def _read_text(path: Path) -> str:
    """文書の全文 (なければ、そこで落とす)。"""
    if not path.is_file():
        pytest.fail(f"文書がない: {path}")
    return path.read_text(encoding="utf-8")


# --- 1. 「書かないこと」の決まり -------------------------------------------


def test_the_procedure_states_the_write_nothing_rule_up_front() -> None:
    """先頭に「書かないこと」の決まりと、その 3 項目がある。"""
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


# --- 2〜3. 必要な印 ---------------------------------------------------------


@pytest.mark.parametrize("marker", REQUIRED_MARKERS)
def test_the_procedure_mentions_the_required_marker(marker: str) -> None:
    """手順書に、生成・変換・一致の確認・照合・関門・起動・回収・停止・⚠ の印がある。"""
    assert marker in _read_text(DOC_PATH)


# --- 4. serve のサブコマンドの実在 ----------------------------------------


_SERVE_CALL: Final[re.Pattern[str]] = re.compile(r"\bserve +([a-z][a-z-]*)(?![\w-])")
"""手順書の中の `serve <サブコマンド>` (`p2-…` のような、数字を含む構成の名前は拾わない)。"""


def _subparser_map(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """引数解析器の、サブコマンドの名前から解析器への対応。"""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action._name_parser_map)
    pytest.fail("build_parser() にサブコマンドの解析器がない")


def test_every_serve_command_in_the_procedure_exists() -> None:
    """手順書が打たせるコマンドが、すべて実在する (存在しない引数を書かないための歯止め)。"""
    text = _read_text(DOC_PATH)
    top = _subparser_map(cli.build_parser())

    used = {found.group(1) for found in _SERVE_CALL.finditer(text)}
    assert used, "手順書に `serve <サブコマンド>` が 1 つもない"
    assert not sorted(used - set(top)), f"実在しないサブコマンド: {sorted(used - set(top))}"


# --- 5. 相対リンクの実在 --------------------------------------------------


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
    """手順書のリポジトリ内の相対リンクが、すべて実在するファイルを指す。"""
    text = _read_text(DOC_PATH)
    targets = _relative_link_targets(text)
    assert targets, "手順書にリポジトリ内の相対リンクが 1 つもない"

    missing = [target for target in targets if not (DOC_PATH.parent / target).is_file()]
    assert not missing, f"手順書のリンクの道筋が実在しない: {missing}"


# --- コード塊の走査 -------------------------------------------------------


_FENCE: Final[re.Pattern[str]] = re.compile(r"^\s*(`{3,}|~{3,})")
"""コード塊の囲い (開きと閉じ)。開きには言語名が続いてよい。"""

_HEADING: Final[re.Pattern[str]] = re.compile(r"^#{1,6}\s")


@dataclass(frozen=True)
class _CodeBlock:
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


def _code_blocks(text: str) -> list[_CodeBlock]:
    """文書のコード塊を、現れた順に返す (`` ` `` と `~` の囲みの両方を見る)。"""
    blocks: list[_CodeBlock] = []
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
            blocks.append(_CodeBlock(opened, number, tuple(body)))
            fence = None
        else:
            body.append((number, line))
    assert fence is None, f"{opened} 行目のコード塊が閉じていない"
    return blocks


def _closes(opening: str, closing: str) -> bool:
    """閉じの囲みが、開きの囲みと同じ文字で、同じ長さ以上か。"""
    return closing[0] == opening[0] and len(closing) >= len(opening)


def _lines_inside(blocks: list[_CodeBlock]) -> set[int]:
    """コード塊の中 (囲みの行も含む) にある行番号。見出しと取り違えないために使う。"""
    inside: set[int] = set()
    for block in blocks:
        inside.update(range(block.open_line, block.close_line + 1))
    return inside


def _nearest_heading(lines: list[str], inside: set[int], before: int) -> str:
    """`before` 行目より前で、いちばん近い見出しの行 (コード塊の中の `#` の注釈は見ない)。"""
    for number in range(before - 1, 0, -1):
        if number not in inside and _HEADING.match(lines[number - 1]):
            return lines[number - 1]
    return ""


def _preceding_paragraph(lines: list[str], before: int) -> str:
    """`before` 行目の、すぐ前の段落 (空行を飛ばし、空行かコード塊の囲みまでの連なり)。"""
    number = before - 1
    while number >= 1 and not lines[number - 1].strip():
        number -= 1
    collected: list[str] = []
    while number >= 1 and lines[number - 1].strip() and not _FENCE.match(lines[number - 1]):
        collected.append(lines[number - 1])
        number -= 1
    return "\n".join(reversed(collected))


# --- 6. 実機を操作する塊の ⚠ ---------------------------------------------

_ASK_MARKS: Final[tuple[str, ...]] = ("⚠", "了承を得てから")

_MACHINE_CALLS: Final[dict[str, re.Pattern[str]]] = {
    "ssh": re.compile(r"\bssh +"),
    "docker run": re.compile(r"\bdocker +run\b"),
    "serve verify": re.compile(r"\bserve +verify(?![\w-])"),
    "serve start": re.compile(r"\bserve +start(?![\w-])"),
    "serve stop": re.compile(r"\bserve +stop(?![\w-])"),
}
"""実機を操作する呼び出し (Spark に入る、コンテナを起こす、照合する、起動する、止める)。"""

_MACHINE_CALLS_THE_PROCEDURE_MUST_SHOW: Final[frozenset[str]] = frozenset(
    {"docker run", "serve verify", "serve start"}
)
"""手順書が、コード塊として書いていなければならない呼び出し (変換、照合、起動)。

`ssh` は、変換を 2 台へ届ける書き方の 1 つなので、なくてもよい (あれば、⚠ の検査の対象)。
"""


def test_every_block_that_touches_the_machines_is_preceded_by_the_ask_mark() -> None:
    """実機を操作するコード塊の**それぞれ**に、直前の見出しか段落に、⚠ と「了承を得てから」がある。

    見るのは、その塊の直前の見出しと、直前の段落 (空行で区切られた、塊のすぐ上の連なり) を
    合わせた範囲である。節をまたいで、前の節の印を数えない。
    """
    text = _read_text(DOC_PATH)
    lines = text.splitlines()
    blocks = _code_blocks(text)
    inside = _lines_inside(blocks)

    shown: set[str] = set()
    problems: list[str] = []
    for block in blocks:
        calls = sorted(
            name for name, pattern in _MACHINE_CALLS.items() if pattern.search(block.text)
        )
        if not calls:
            continue
        shown.update(calls)
        heading = _nearest_heading(lines, inside, block.open_line)
        paragraph = _preceding_paragraph(lines, block.open_line)
        window = f"{heading}\n{paragraph}"
        missing = [mark for mark in _ASK_MARKS if mark not in window]
        if missing:
            problems.append(
                f"{block.open_line} 行目からの、{calls} を含む塊の直前の見出しと段落に、"
                f"{missing} がない"
            )

    absent = sorted(_MACHINE_CALLS_THE_PROCEDURE_MUST_SHOW - shown)
    assert not absent, f"コード塊として書かれていない実機の操作: {absent}"
    assert not problems, "\n".join(problems)


# --- 7. 変換の docker run -------------------------------------------------

_DOCKER_RUN: Final[re.Pattern[str]] = _MACHINE_CALLS["docker run"]

_GPU_OPTIONS: Final[re.Pattern[str]] = re.compile(r"--gpus\b|--runtime[= ]+nvidia\b")
"""コンテナに GPU を渡す指定。変換は CPU だけで行うので、どの `docker run` にも付けない。"""

_BIND_SPEC: Final[re.Pattern[str]] = re.compile(r"(?<![\w-])(?:--mount|--volume|-v)(?:=|\s+)(\S+)")
"""`docker run` の結び付けの指定 (`--mount` と `-v` / `--volume` の、値の 1 語)。"""

_READONLY: Final[re.Pattern[str]] = re.compile(r"[,:](?:readonly|ro)(?:=(?:true|1))?(?![\w=-])")
"""結び付けの指定が、読み取り専用であること (`readonly`、`ro`。`readonly=false` は含めない)。"""

_ENTRYPOINT: Final[re.Pattern[str]] = re.compile(r"(?<![\w-])--entrypoint(?:=|\s+)\S+")
"""`docker run` の、イメージの既定の起動 (ENTRYPOINT) を差し替える指定。"""


def _docker_run_blocks() -> list[_CodeBlock]:
    """`docker run` を含むコード塊 (注釈の行だけに出てくるものは除く)。なければ、そこで落とす。"""
    blocks = [
        block for block in _code_blocks(_read_text(DOC_PATH)) if _DOCKER_RUN.search(block.commands)
    ]
    assert blocks, "`docker run` を含むコード塊が 1 つもない"
    return blocks


def _docker_run_commands(commands: str) -> list[str]:
    """コード塊の中の `docker run` を、1 つずつの区間 (次の `docker run` の手前まで) に分ける。"""
    starts = [found.start() for found in _DOCKER_RUN.finditer(commands)]
    ends = [*starts[1:], len(commands)]
    return [commands[start:end] for start, end in zip(starts, ends, strict=True)]


def _origin_directory_name() -> str:
    """変換の元の重みを Spark の `models/` に置くときのディレクトリ名 (`mount_at` の末尾)。

    コミット済みの構成 (`configs.toml` の `p1-nvfp4-tp2`。`test_config_committed.py` が固定)
    の値から取る。手順書の `docker run` は、この重みの置き場所を読み取り専用で結び付ける。
    """
    weights = c.load_configs(CONFIGS_PATH, REPO_ROOT)[ORIGIN_CONFIG].weights
    assert weights is not None, f"{ORIGIN_CONFIG} に重みの参照がない"
    return PurePosixPath(weights.mount_at).name


def test_the_conversion_docker_run_does_not_hand_over_a_gpu() -> None:
    """変換の `docker run` に、GPU を渡す指定がない (CPU だけで変換する)。"""
    problems: list[str] = []
    for block in _docker_run_blocks():
        for found in _GPU_OPTIONS.finditer(block.commands):
            problems.append(f"{block.open_line} 行目からの塊に GPU の指定がある: {found.group(0)}")
    assert not problems, "\n".join(problems)


def test_the_conversion_docker_run_binds_the_origin_weights_readonly() -> None:
    """変換の `docker run` のそれぞれが、元の重みを、`readonly` で結び付けている。"""
    origin = _origin_directory_name()
    problems: list[str] = []
    for block in _docker_run_blocks():
        for command in _docker_run_commands(block.commands):
            origin_specs = [
                found.group(1) for found in _BIND_SPEC.finditer(command) if origin in found.group(1)
            ]
            if not origin_specs:
                problems.append(
                    f"{block.open_line} 行目からの塊の `docker run` に、"
                    f"元の重み ({origin}) の結び付けがない"
                )
            problems.extend(
                f"{block.open_line} 行目からの塊で、元の重みの結び付けに readonly がない: {spec}"
                for spec in origin_specs
                if not _READONLY.search(spec)
            )
    assert not problems, "\n".join(problems)


def test_the_conversion_docker_run_overrides_the_image_entrypoint() -> None:
    """変換の `docker run` のそれぞれが、イメージの既定の起動を差し替えている。

    このイメージの既定は `vllm serve` である (`configs.toml` の、取得・計測の構成が
    `--entrypoint` で差し替えている理由と同じ)。差し替えないと、コンテナは
    `vllm serve <変換の道具> …` として起動し、変換の道具は動かない。
    """
    problems: list[str] = []
    for block in _docker_run_blocks():
        problems.extend(
            f"{block.open_line} 行目からの塊の `docker run` に、--entrypoint がない"
            for command in _docker_run_commands(block.commands)
            if not _ENTRYPOINT.search(command)
        )
    assert not problems, "\n".join(problems)
