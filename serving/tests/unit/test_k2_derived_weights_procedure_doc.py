"""K2 の派生の重みを作って使う手順書 (`docs/vllm-baseline/k2-derived-weights-procedure.md`) の試験。

文書のタスクなので、振る舞いの試験の代わりに、手順書の完了の状態を固定する。確かめること:

1. 先頭 20 行以内に「書かないこと」の決まり (要求・応答の本文、認証の情報、`exl3-tp2` の
   中身) がある
2. 派生の重みの構成の生成 (`--weights k2s1` / `tp2-full-k2s1.toml` / `p2-nope-tp2-full-k2s1`)、
   コミットする正解のマニフェスト (`serving/weights/` / `k2s1.manifest.json` / `kind` /
   `derivation`)、変換の道筋 (`CPU` / `readonly` / `-m k2_quant` / `payload/k2-quant` /
   `--network none`)、道具の配布 (`serve push`)、マニフェストの取り込みと 2 台の結果の
   確認 (`serve derived-import` / `sha256sum`) の印がある
3. 実機の照合と関門と起動と回収と停止 (`serve verify` / `serve check` / `serve start` /
   `serve smoke` / `serve logs` / `serve stop`) の印と、状態を変える塊の ⚠ の印がある
4. 使う `serve <サブコマンド>` が、すべて `cli.build_parser()` に実在する
5. 相対リンクの道筋が、すべて実在する
6. `ssh`・`docker run`・`serve push`・`serve verify`・`serve start` を含むコード塊の
   **それぞれに**、直前の見出しか直前の段落に、⚠ と「了承を得てから」がある (総数の数え上げ
   では、1 つの塊の印が抜けても気付けない)
7. 変換の `docker run` は、GPU を渡さず (CPU だけ)、元の重みを `readonly` で結び付ける
8. 変換の `docker run` は、コード塊の命令として読める形 (バックスラッシュで継いだ行を連結して
   `shlex` で分けた語の列) で、次を満たす (段落と注釈の行の字面は見ない)
   - `--network none` がある
   - `--entrypoint python3` と `-w /tools/k2-quant` が、`-m k2_quant` より前にある
   - `-m k2_quant` 以降の option 名が、道具の写し (`serving/payload/k2-quant/` の
     `k2_quant/__main__.py`) の `_build_parser` の option の集合に含まれ、必須の option を
     すべて含む
   - `--link` を含まない (段落・注釈の行に書くだけなら、問題にしない)
   - `--source /origin`、`--output /derived`、`--source-repo` と `--source-revision` が
     コミット済みの構成 `p1-nvfp4-tp2` の重みと同じ
   - 道具の写しを `/tools/k2-quant` に `readonly` で、元の重みを `/origin` に `readonly`
     で、変換の結果の置き場所を `/derived` に書き込める形で結び付ける
   - 仮置きの `<変換の道具>` が文書に残っていない
9. コード塊の `serve derived-import` の option 名が、`cli.build_parser()` の
   `derived-import` の解析器に実在し、その必須の option をすべて含む
10. §1 の突き合わせ (2 台の写しの `sha256sum` と、Mac の `shasum -a 256`) は、コード塊の命令として
    読める形で、hash コマンドの**それぞれの**直後に `| LC_ALL=C sort` がある (`*.py` の展開順は
    ロケールで違う。2026-09-26 に、中身が同じでも差分が出た)。段落と注釈の行の字面は見ない
11. §0 の見本の JSON が、道具の写しの既定の正規表現 (`DEFAULT_PATTERN`) と同じ値を書き、`lm_head` と
    MTP の `shared_head.head` を既定の対象にしない (#68)。`modules` は、その正規表現が選ぶ名前
12. §8 (実機でしか分からないこと) が、2026-09-26 に分かったこと (変換の所要時間と 2 台の一致、
    ページキャッシュで `--load-format instanttensor` の起動が落ち、`auto` なら読めること) を、
    既存の項目を残したまま述べる

RED: 文書がなければ、`_read_text` で `pytest.fail` する。道具の写しや `derived-import` が
まだなければ、その理由を示して落とす。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

import pytest

from serving_kit import cli
from serving_kit import config as c
from serving_kit.types import WeightsRef

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
    "sha256sum",
    "CPU",
    "readonly",
    "serve derived-import",
    "-m k2_quant",
    "--preset k2s2a",
    "payload/k2-quant",
    "serve push",
    "--network none",
    "serve check",
    "serve verify",
    "serve start",
    "serve smoke",
    "serve logs",
    "serve stop",
    "⚠",
    "kind",
    "derivation",
    "LC_ALL=C sort",
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
    "serve push": re.compile(r"\bserve +push(?![\w-])"),
    "serve verify": re.compile(r"\bserve +verify(?![\w-])"),
    "serve start": re.compile(r"\bserve +start(?![\w-])"),
    "serve stop": re.compile(r"\bserve +stop(?![\w-])"),
}
"""実機を操作する呼び出し (Spark に入る、コンテナを起こす、配る、照合する、起動する、止める)。"""

_MACHINE_CALLS_THE_PROCEDURE_MUST_SHOW: Final[frozenset[str]] = frozenset(
    {"docker run", "serve push", "serve verify", "serve start"}
)
"""手順書が、コード塊として書いていなければならない呼び出し (変換、道具の配布、照合、起動)。

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


# --- 8. 変換の docker run の全体 (コード塊の命令として、実 CLI と突き合わせる) ------

TOOL_MAIN_PATH: Final[Path] = SERVING_DIR / "payload" / "k2-quant" / "k2_quant" / "__main__.py"
"""変換の道具の写し (`serve push` が Spark へ配る)。実行せず、`ast` で読む。"""

_CONTINUATION: Final[re.Pattern[str]] = re.compile(r"\\\n")
"""シェルの、行を継ぐバックスラッシュと改行。"""

_TOOL_MODULE: Final[tuple[str, str]] = ("-m", "k2_quant")
"""変換の道具の起動 (`python3 -m k2_quant`)。この後ろが、道具の引数。"""

_PLACEHOLDER: Final[str] = "<変換の道具>"
"""道具ができる前に、手順書が仮に置いていた名前。"""


def _command_lines(text: str) -> list[str]:
    """文書のコード塊の命令行 (注釈の行を除き、バックスラッシュで継いだ行を 1 行にしたもの)。

    コード塊の外 (段落) の字面は見ない。
    """
    lines: list[str] = []
    for block in _code_blocks(text):
        lines.extend(_CONTINUATION.sub(" ", block.commands).splitlines())
    return lines


def _split_words(line: str) -> list[str]:
    """命令行を、シェルと同じ規則 (`shlex`) で語に分ける。分けられなければ落とす。"""
    try:
        return shlex.split(line)
    except ValueError as exc:
        pytest.fail(f"命令行を語に分けられない ({exc}): {line}")


def _conversion_commands(text: str) -> list[list[str]]:
    """文書のコード塊の `docker run` の命令を、1 つずつ、語の列にして返す。

    `ssh spark-153d` のような前置きも、語として残る。
    """
    return [_split_words(line) for line in _command_lines(text) if _DOCKER_RUN.search(line)]


def _adjacent(words: list[str], first: str, second: str) -> int | None:
    """`first` の直後に `second` が続く、最初の位置 (なければ None)。"""
    for index in range(len(words) - 1):
        if words[index] == first and words[index + 1] == second:
            return index
    return None


def _values_of(words: list[str], option: str) -> list[str]:
    """`option` の直後の語 (値) を、現れた順にすべて返す。"""
    return [words[index + 1] for index in range(len(words) - 1) if words[index] == option]


def _describe(words: list[str]) -> str:
    """どの命令かを示す短い表記 (`ssh spark-153d docker run` など)。"""
    return "`" + " ".join(words[:4]) + " …`"


def _tool_words(words: list[str]) -> list[str]:
    """`docker run` の命令のうち、`-m k2_quant` より後ろ (道具の引数)。なければ、そこで落とす。"""
    position = _adjacent(words, *_TOOL_MODULE)
    if position is None:
        pytest.fail(f"`-m k2_quant` がない: {_describe(words)}")
    return words[position + len(_TOOL_MODULE) :]


def _doc_conversion_commands() -> list[list[str]]:
    """手順書のコード塊の `docker run` の命令。`-m k2_quant` の命令が 1 つもなければ、落とす。"""
    commands = _conversion_commands(_read_text(DOC_PATH))
    assert any(_adjacent(words, *_TOOL_MODULE) is not None for words in commands), (
        f"手順書のコード塊に、変換の命令 (`docker run … -m k2_quant …`) が 1 つもない: {DOC_PATH}"
    )
    return commands


def _tool_options() -> tuple[set[str], set[str]]:
    """変換の道具の写しの `_build_parser` にある option の集合と、そのうち必須のもの。

    道具は実行せず、`ast` で `add_argument` の第 1 引数 (`--` で始まる文字列) と
    `required=True` を読む。
    """
    if not TOOL_MAIN_PATH.is_file():
        pytest.fail(f"変換の道具の写しがない (serving/payload/k2-quant/ に置く): {TOOL_MAIN_PATH}")
    tree = ast.parse(TOOL_MAIN_PATH.read_text(encoding="utf-8"), filename=str(TOOL_MAIN_PATH))
    builders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_parser"
    ]
    if len(builders) != 1:
        pytest.fail(f"`_build_parser` が 1 つではない ({len(builders)} 個): {TOOL_MAIN_PATH}")

    options: set[str] = set()
    required: set[str] = set()
    for node in ast.walk(builders[0]):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
        ):
            continue
        name = node.args[0]
        if not (
            isinstance(name, ast.Constant)
            and isinstance(name.value, str)
            and name.value.startswith("--")
        ):
            continue
        options.add(name.value)
        if any(
            keyword.arg == "required"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        ):
            required.add(name.value)
    if not options:
        pytest.fail(
            f"`_build_parser` の中に、`--` で始まる option が 1 つも読めない: {TOOL_MAIN_PATH}"
        )
    return options, required


def test_every_conversion_docker_run_cuts_the_network() -> None:
    """変換の `docker run` のそれぞれに、`--network none` がある (Hub から何も取らない保証)。"""
    problems = [
        f"`--network none` がない: {_describe(words)}"
        for words in _doc_conversion_commands()
        if _adjacent(words, "--network", "none") is None
    ]
    assert not problems, "\n".join(problems)


def test_every_conversion_docker_run_starts_the_tool_as_a_module_from_its_directory() -> None:
    """変換の `docker run` は、`--entrypoint python3` と `-w /tools/k2-quant` の後に `-m k2_quant`。

    `-m k2_quant` より後ろは道具の引数になるので、docker の option は、その前に置く。
    """
    problems: list[str] = []
    for words in _doc_conversion_commands():
        module = _adjacent(words, *_TOOL_MODULE)
        if module is None:
            problems.append(f"`-m k2_quant` がない: {_describe(words)}")
        for first, second in (("--entrypoint", "python3"), ("-w", "/tools/k2-quant")):
            position = _adjacent(words, first, second)
            if position is None:
                problems.append(f"`{first} {second}` がない: {_describe(words)}")
            elif module is not None and position > module:
                problems.append(
                    f"`{first} {second}` が `-m k2_quant` の後ろにある: {_describe(words)}"
                )
    assert not problems, "\n".join(problems)


def test_the_conversion_options_match_the_real_tool_cli() -> None:
    """`-m k2_quant` 以降の option 名が、道具の写しの option の集合に含まれ、必須をすべて含む。

    バックスラッシュで継いだ行を連結し、`shlex` で分けた語の列で見る。
    """
    options, required = _tool_options()
    problems: list[str] = []
    for words in _doc_conversion_commands():
        used = {word for word in _tool_words(words) if word.startswith("--")}
        if used - options:
            problems.append(f"道具にない option {sorted(used - options)}: {_describe(words)}")
        if required - used:
            problems.append(
                f"道具の必須の option {sorted(required - used)} がない: {_describe(words)}"
            )
    assert not problems, "\n".join(problems)


def _link_problems(text: str) -> list[str]:
    """文書のコード塊の変換の命令に、`--link` があるものを、1 件ずつ説明する。

    段落と、コード塊の中の注釈の行 (`#` で始まる行) に書かれた `--link` は、見ない。
    """
    return [
        f"変換の命令に `--link` がある: {_describe(words)}"
        for words in _conversion_commands(text)
        if "--link" in words
    ]


def test_the_conversion_commands_do_not_use_link() -> None:
    """変換の命令に `--link` がない (別々の bind mount の間のハードリンクは `EXDEV` で失敗する)。"""
    _doc_conversion_commands()  # 前提: 変換の命令が 1 つもなければ、ここで落とす
    problems = _link_problems(_read_text(DOC_PATH))
    assert not problems, "\n".join(problems)


def _synthetic_document(paragraph: str, block_lines: list[str]) -> str:
    """段落 1 つと、bash のコード塊 1 つだけの文書 (`_link_problems` の試験用)。"""
    fence = "`" * 3
    return "\n".join([paragraph, "", f"{fence}bash", *block_lines, fence, ""])


_SYNTHETIC_CONVERSION: Final[tuple[str, ...]] = (
    "ssh spark-153d docker run --rm --network none \\",
    "  --entrypoint python3 \\",
    "  -w /tools/k2-quant \\",
    "  sha256:0000 \\",
    "  -m k2_quant --source /origin --output /derived",
)
"""変換の命令 (`--link` なし) の見本。行はバックスラッシュで継ぐ。"""


def test_link_in_a_paragraph_or_a_comment_line_is_not_a_problem() -> None:
    """段落の「`--link` は使わない」と、コード塊の注釈行の `--link` は、検査の対象にならない。

    変換の命令は読めている (何も読めずに「問題なし」になっていない)。
    """
    text = _synthetic_document(
        "`--link` は使わない。",
        ["# --link は EXDEV で失敗する", *_SYNTHETIC_CONVERSION],
    )

    assert len(_conversion_commands(text)) == 1
    assert _link_problems(text) == []


def test_link_in_a_conversion_command_line_is_a_problem() -> None:
    """コード塊の変換の命令行に `--link` があれば、`--link` を含むという理由で検出する。

    `--link` は、継いだ最後の行に書かれていても見つかる。
    """
    text = _synthetic_document(
        "`--link` は使わない。",
        [*_SYNTHETIC_CONVERSION[:-1], _SYNTHETIC_CONVERSION[-1] + " \\", "  --link"],
    )

    problems = _link_problems(text)

    assert len(problems) == 1
    assert "--link" in problems[0]


def test_every_conversion_reads_the_origin_and_writes_the_derived_directory() -> None:
    """変換の道具の `--source` が `/origin`、`--output` が `/derived` である。"""
    problems: list[str] = []
    for words in _doc_conversion_commands():
        tool_words = _tool_words(words)
        for option, expected in (("--source", "/origin"), ("--output", "/derived")):
            values = _values_of(tool_words, option)
            if values != [expected]:
                problems.append(
                    f"{option} の値が {expected} 1 つではない ({values}): {_describe(words)}"
                )
    assert not problems, "\n".join(problems)


def _origin_weights() -> WeightsRef:
    """変換の元の重みの参照 (コミット済みの構成 `p1-nvfp4-tp2` の重み)。"""
    weights = c.load_configs(CONFIGS_PATH, REPO_ROOT)[ORIGIN_CONFIG].weights
    assert isinstance(weights, WeightsRef), f"{ORIGIN_CONFIG} の重みが Hub の重みの参照ではない"
    return weights


def test_the_conversion_source_is_the_committed_origin_weights() -> None:
    """`--source-repo` と `--source-revision` が、`p1-nvfp4-tp2` の重みの `repo` と `revision`。"""
    weights = _origin_weights()
    problems: list[str] = []
    for words in _doc_conversion_commands():
        tool_words = _tool_words(words)
        for option, expected in (
            ("--source-repo", weights.repo),
            ("--source-revision", weights.revision),
        ):
            values = _values_of(tool_words, option)
            if values != [expected]:
                problems.append(
                    f"{option} の値が {expected} 1 つではない ({values}): {_describe(words)}"
                )
    assert not problems, "\n".join(problems)


_MOUNT_TARGET: Final[re.Pattern[str]] = re.compile(r"(?:^|,)target=([^,]+)")
_MOUNT_SOURCE: Final[re.Pattern[str]] = re.compile(r"(?:^|,)source=([^,]+)")


def _binds_to(words: list[str], target: str) -> list[str]:
    """`docker run` の結び付けの指定 (`--mount` の値など) のうち、`target=` が `target` のもの。"""
    specs: list[str] = _BIND_SPEC.findall(" ".join(words))
    return [
        spec
        for spec in specs
        if (found := _MOUNT_TARGET.search(spec)) is not None and found.group(1) == target
    ]


def _single_bind(words: list[str], target: str, problems: list[str]) -> str | None:
    """`target` への結び付けがちょうど 1 つならその指定を返す (違えば、問題を足して None)。"""
    specs = _binds_to(words, target)
    if len(specs) != 1:
        problems.append(
            f"target={target} の結び付けが 1 つではない ({len(specs)} 個): {_describe(words)}"
        )
        return None
    return specs[0]


def test_every_conversion_docker_run_binds_the_tool_copy_readonly() -> None:
    """道具の写し (`.../payload/k2-quant`) が、`/tools/k2-quant` に `readonly` で結び付く。"""
    problems: list[str] = []
    for words in _doc_conversion_commands():
        spec = _single_bind(words, "/tools/k2-quant", problems)
        if spec is None:
            continue
        if not _READONLY.search(spec):
            problems.append(f"道具の写しの結び付けに readonly がない: {spec}")
        source = _MOUNT_SOURCE.search(spec)
        if source is None or PurePosixPath(source.group(1)).parts[-2:] != ("payload", "k2-quant"):
            problems.append(
                f"道具の写しの結び付けの source が payload/k2-quant で終わらない: {spec}"
            )
    assert not problems, "\n".join(problems)


def test_every_conversion_docker_run_binds_origin_readonly_and_derived_writable() -> None:
    """`/origin` は読み取り専用、変換の結果の `/derived` は書き込める形で、結び付く。"""
    problems: list[str] = []
    for words in _doc_conversion_commands():
        origin = _single_bind(words, "/origin", problems)
        if origin is not None and not _READONLY.search(origin):
            problems.append(f"target=/origin の結び付けに readonly がない: {origin}")
        derived = _single_bind(words, "/derived", problems)
        if derived is not None and _READONLY.search(derived):
            problems.append(f"target=/derived の結び付けが readonly で、書き込めない: {derived}")
    assert not problems, "\n".join(problems)


def test_the_procedure_leaves_no_placeholder_for_the_conversion_tool() -> None:
    """仮置きの `<変換の道具>` が、手順書のどこにも残っていない (実際の道具の名前で書く)。"""
    _doc_conversion_commands()  # 前提: 変換の命令が 1 つもなければ、ここで落とす
    text = _read_text(DOC_PATH)
    lines = [
        str(number)
        for number, line in enumerate(text.splitlines(), start=1)
        if _PLACEHOLDER in line
    ]
    assert not lines, f"仮置きの {_PLACEHOLDER} が残っている行: {', '.join(lines)}"


# --- 9. serve derived-import ---------------------------------------------

_DERIVED_IMPORT: Final[re.Pattern[str]] = re.compile(r"\bserve +derived-import(?![\w-])")


def test_the_derived_import_options_exist_in_the_real_cli() -> None:
    """コード塊の `serve derived-import` の option 名が、実際の解析器に実在し、必須をすべて含む。

    `derived-import` がまだなければ、その理由を示して落ちる。
    """
    text = _read_text(DOC_PATH)
    commands = [_split_words(line) for line in _command_lines(text) if _DERIVED_IMPORT.search(line)]
    assert commands, f"手順書のコード塊に `serve derived-import` の命令が 1 つもない: {DOC_PATH}"

    parser = _subparser_map(cli.build_parser()).get("derived-import")
    if parser is None:
        pytest.fail("`cli.build_parser()` に `derived-import` のサブコマンドがない")
    options = {name for action in parser._actions for name in action.option_strings}
    required = {
        name for action in parser._actions if action.required for name in action.option_strings
    }

    problems: list[str] = []
    for words in commands:
        position = _adjacent(words, "serve", "derived-import")
        if position is None:
            problems.append(f"`serve derived-import` が語として読めない: {_describe(words)}")
            continue
        used = {word for word in words[position + 2 :] if word.startswith("--")}
        if used - options:
            problems.append(
                f"`derived-import` にない option {sorted(used - options)}: {_describe(words)}"
            )
        if required - used:
            problems.append(f"`derived-import` の必須の option {sorted(required - used)} がない")
    assert not problems, "\n".join(problems)


# --- 10. §1 の突き合わせの並べ替え ---------------------------------------

_HASH_COMMAND: Final[re.Pattern[str]] = re.compile(r"\b(?:sha256sum|shasum\s+-a\s+256)\b")
"""2 台の写しと Mac の写しの、ファイルごとの SHA-256 を出す命令。"""

_SORTED_AFTER_HASH_COMMAND: Final[re.Pattern[str]] = re.compile(r"[^|)'\"]*\|\s*LC_ALL=C\s+sort\b")
"""hash コマンドの直後 (引数のあと、同じ引用符と括弧の中) の `| LC_ALL=C sort`。

`ssh` の引用符や `<( … )` の括弧を越えては探さない。片側の並べ替えを、もう片側のもので
代用させないため。
"""

_MINIMUM_CHECKSUM_COMMANDS: Final[int] = 4
"""突き合わせの hash コマンドの数: 2 台 (`spark-153d`、`spark-5083`) × 2 側 (Spark と Mac)。"""


def _checksum_sort_problems(text: str) -> tuple[int, list[str]]:
    """文書のコード塊の命令行にある hash コマンドの数と、直後に並べ替えがないものの説明。

    段落と、コード塊の中の注釈の行 (`#` で始まる行) は、見ない。
    """
    checked = 0
    problems: list[str] = []
    for line in _command_lines(text):
        for found in _HASH_COMMAND.finditer(line):
            checked += 1
            if _SORTED_AFTER_HASH_COMMAND.match(line, found.end()) is None:
                problems.append(
                    f"`{found.group(0)}` の直後に `| LC_ALL=C sort` がない: {line.strip()[:100]}"
                )
    return checked, problems


def test_the_checksum_comparison_sorts_every_side_by_file_name() -> None:
    """§1 の突き合わせは、2 台 × 2 側の hash コマンドのそれぞれの直後で、並べ替えてから比べる。

    `*.py` の展開順は、ロケールで違う (Spark と Mac で、中身が同じでも差分が出た)。
    片側だけを並べ替えても、もう片側の順は変わらない。
    """
    checked, problems = _checksum_sort_problems(_read_text(DOC_PATH))

    assert checked >= _MINIMUM_CHECKSUM_COMMANDS, (
        f"突き合わせの hash コマンドが {checked} 個しか読めない (2 台 × 2 側 = "
        f"{_MINIMUM_CHECKSUM_COMMANDS} 個が要る): {DOC_PATH}"
    )
    assert not problems, "\n".join(problems)


_SYNTHETIC_CHECKSUM_COMPARISON: Final[tuple[str, ...]] = (
    "diff \\",
    "  <(ssh spark-153d 'cd /a && sha256sum *.py | LC_ALL=C sort -k 2') \\",
    "  <(cd /b && shasum -a 256 *.py | LC_ALL=C sort -k 2)",
    "diff \\",
    "  <(ssh spark-5083 'cd /a && sha256sum *.py | LC_ALL=C sort -k 2') \\",
    "  <(cd /b && shasum -a 256 *.py | LC_ALL=C sort -k 2)",
)
"""2 台 × 2 側の、すべて並べ替えてから比べる突き合わせの見本。行はバックスラッシュで継ぐ。"""


def test_a_checksum_comparison_that_sorts_every_side_has_no_problem() -> None:
    """2 台 × 2 側のすべての hash コマンドの直後に並べ替えがあれば、4 個数えられ、問題がない。"""
    text = _synthetic_document("突き合わせる。", list(_SYNTHETIC_CHECKSUM_COMPARISON))

    checked, problems = _checksum_sort_problems(text)

    assert checked == _MINIMUM_CHECKSUM_COMMANDS
    assert problems == []


@pytest.mark.parametrize(
    ("ssh_side", "label"),
    [
        ("sha256sum *.py", "並べ替えがない"),
        ("sha256sum *.py | sort -k 2", "LC_ALL=C のない sort (ロケールに依る)"),
    ],
)
def test_a_checksum_comparison_that_sorts_only_the_mac_side_is_a_problem(
    ssh_side: str, label: str
) -> None:
    """片側 (ssh 側) が、ロケールに依らない並べ替えをしない命令行は、その命令行を示して落ちる。

    Mac 側 (`shasum -a 256 *.py | LC_ALL=C sort -k 2`) に並べ替えがあっても、代用にならない。
    """
    lines = list(_SYNTHETIC_CHECKSUM_COMPARISON)
    lines[1] = f"  <(ssh spark-153d 'cd /a && {ssh_side}') \\"
    text = _synthetic_document("突き合わせる。", lines)

    checked, problems = _checksum_sort_problems(text)

    assert checked == _MINIMUM_CHECKSUM_COMMANDS, label
    assert len(problems) == 1, label
    assert "sha256sum" in problems[0]
    assert "spark-153d" in problems[0]


def test_checksum_words_in_a_paragraph_or_a_comment_line_are_not_counted() -> None:
    """段落の `sha256sum` / `shasum -a 256` の説明と、コード塊の注釈行は、検査の対象にならない。

    数えられるのは、コード塊の命令行の 4 個だけで、問題はない。
    """
    text = _synthetic_document(
        "2 台の写しの `sha256sum` を、Mac の `shasum -a 256` と突き合わせる。",
        ["# sha256sum は並べ替えてから比べる", *_SYNTHETIC_CHECKSUM_COMPARISON],
    )

    checked, problems = _checksum_sort_problems(text)

    assert checked == _MINIMUM_CHECKSUM_COMMANDS
    assert problems == []


# --- 11. §0 の見本の JSON --------------------------------------------------

TOOL_SELECTION_PATH: Final[Path] = (
    SERVING_DIR / "payload" / "k2-quant" / "k2_quant" / "selection.py"
)
"""変換の道具の写しの、対象の選択 (`DEFAULT_PATTERN`)。実行せず、`ast` で読む。"""


def _tool_default_pattern() -> str:
    """変換の道具の写しの `DEFAULT_PATTERN` (実行せず、`ast` で定数を読む)。"""
    if not TOOL_SELECTION_PATH.is_file():
        pytest.fail(
            f"変換の道具の写しがない (serving/payload/k2-quant/ に置く): {TOOL_SELECTION_PATH}"
        )
    tree = ast.parse(
        TOOL_SELECTION_PATH.read_text(encoding="utf-8"), filename=str(TOOL_SELECTION_PATH)
    )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "DEFAULT_PATTERN"
            and node.value is not None
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str), (
                f"DEFAULT_PATTERN が文字列ではない: {TOOL_SELECTION_PATH}"
            )
            return value
    pytest.fail(f"`DEFAULT_PATTERN` が読めない: {TOOL_SELECTION_PATH}")


def _json_sample(key: str) -> dict[str, object]:
    """手順書のコード塊のうち、JSON として読め、最上位に `key` を持つものの、唯一の 1 つ。"""
    samples: list[dict[str, object]] = []
    for block in _code_blocks(_read_text(DOC_PATH)):
        try:
            parsed = json.loads(block.text)
        except ValueError:
            continue
        if isinstance(parsed, dict) and key in parsed:
            samples.append(parsed)
    assert len(samples) == 1, (
        f"最上位に `{key}` を持つ JSON のコード塊が 1 つではない: {len(samples)}"
    )
    return samples[0]


def _object(value: object, where: str) -> dict[str, object]:
    assert isinstance(value, dict), f"{where} が JSON の object ではない"
    return value


def _pattern_argument(args: object, where: str) -> str:
    """`args` の列の、`--pattern` の直後の値。"""
    assert isinstance(args, list), f"{where} が列ではない"
    assert args.count("--pattern") == 1, f"{where} の `--pattern` が 1 つではない"
    value = args[args.index("--pattern") + 1]
    assert isinstance(value, str), f"{where} の `--pattern` の値が文字列ではない"
    return value


def _tool_manifest_sample() -> dict[str, object]:
    """§0 の、道具が書く manifest の見本の `conversion`。"""
    return _object(_json_sample("conversion")["conversion"], "道具の見本の conversion")


def _derived_conversion_sample() -> dict[str, object]:
    """§0 の、serving が読む派生のマニフェストの見本の `derivation.conversion`。"""
    derivation = _object(_json_sample("derivation")["derivation"], "派生の見本の derivation")
    return _object(derivation["conversion"], "派生の見本の derivation.conversion")


def test_the_manifest_samples_show_the_default_pattern_of_the_tool_copy() -> None:
    """§0 の 2 つの見本の `pattern`・`args` の `--pattern`・`target_pattern` が、道具の写しの
    `DEFAULT_PATTERN` と同じ文字列である。

    手順書は「`--pattern` は渡さなくても、実際に使った既定の値を書きます」と述べている。
    """
    default = _tool_default_pattern()
    tool = _tool_manifest_sample()
    derived = _derived_conversion_sample()

    assert tool["pattern"] == default
    assert _pattern_argument(tool["args"], "道具の見本の args") == default
    assert _pattern_argument(derived["args"], "派生の見本の args") == default
    assert derived["target_pattern"] == default


def test_the_manifest_sample_pattern_does_not_select_lm_head_or_the_mtp_shared_head() -> None:
    """§0 の見本の `pattern` は、dense と共有の専門家を選び、`lm_head` と MTP の
    `shared_head.head` は選ばない (#68)。

    `re.match` は、道具が checkpoint のモジュール名に当てるのと同じ当て方。
    """
    pattern = str(_tool_manifest_sample()["pattern"])

    for selected in (
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.3.mlp.shared_experts.up_proj",
    ):
        assert re.match(pattern, selected), selected
    for not_selected in (
        "lm_head",
        "model.language_model.layers.45.shared_head.head",
    ):
        assert not re.match(pattern, not_selected), not_selected


def test_the_manifest_sample_modules_are_selected_by_the_sample_pattern() -> None:
    """§0 の見本の `modules` の名前が、どれも、見本の `pattern` で選ばれる名前である。

    `pattern` から `lm_head` を外したのに、`modules` の見本が `["lm_head"]` のままだと、
    見本が自分の `pattern` と食い違う。
    """
    tool = _tool_manifest_sample()
    modules = tool["modules"]
    assert isinstance(modules, list)
    assert modules, "§0 の見本の modules が空"

    not_selected = [
        module
        for module in modules
        if not isinstance(module, str) or not re.match(str(tool["pattern"]), module)
    ]
    assert not not_selected, f"見本の pattern で選ばれない modules: {not_selected}"


# --- 12. §8 実機でしか分からないこと -----------------------------------


def _section_8(text: str) -> str:
    """`## 8.` の見出しから、次の `## ` の見出し (なければ末尾) の手前まで (コード塊の中は除く)。"""
    lines = text.splitlines()
    inside = _lines_inside(_code_blocks(text))
    headings = [
        number
        for number, line in enumerate(lines, start=1)
        if number not in inside and line.startswith("## ")
    ]
    begin = next((number for number in headings if lines[number - 1].startswith("## 8.")), None)
    assert begin is not None, f"`## 8.` の見出しがない: {DOC_PATH}"
    end = next((number for number in headings if number > begin), len(lines) + 1)
    return "\n".join(lines[begin - 1 : end - 1])


SECTION_8_NEW_MARKERS: Final[tuple[str, ...]] = (
    "2026-09-26",
    "8.5 分",
    "バイト単位",
    "ページキャッシュ",
    "instanttensor",
    "exceeds device memory budget",
    "--load-format auto",
)
"""§8 に足す、2026-09-26 に分かったことの印 (変換は 2 台とも約 8.5 分でバイト単位で一致、
変換と照合の直後はページキャッシュが埋まり `--load-format instanttensor` の起動が
`buffer_size ... exceeds device memory budget` で落ち、`--load-format auto` なら読める)。"""

SECTION_8_KEPT_MARKERS: Final[tuple[str, ...]] = (
    "numpy",
    "184 GiB",
    "GPU のメモリ",
    "匿名",
)
"""§8 の既存の項目 (追記の対象でないもの) の印。Issue は「足す」なので、置き換えない。"""


@pytest.mark.parametrize("marker", SECTION_8_NEW_MARKERS)
def test_section_8_records_what_was_learned_on_2026_09_26(marker: str) -> None:
    """§8 に、2026-09-26 に分かったこと (変換の時間と一致、`instanttensor` の失敗と回避) がある。"""
    assert marker in _section_8(_read_text(DOC_PATH))


@pytest.mark.parametrize("marker", SECTION_8_KEPT_MARKERS)
def test_section_8_keeps_the_existing_open_questions(marker: str) -> None:
    """§8 の既存の項目 (`numpy`、空き、起動の時間と GPU のメモリ、匿名) を、消していない。"""
    assert marker in _section_8(_read_text(DOC_PATH))
