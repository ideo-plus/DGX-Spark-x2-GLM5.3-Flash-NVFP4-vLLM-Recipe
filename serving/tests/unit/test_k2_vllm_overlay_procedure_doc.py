"""vLLM の直したファイルを重ねる手順書 (`docs/vllm-baseline/k2-vllm-overlay-procedure.md`) の試験。

文書のタスクなので、振る舞いの試験の代わりに、手順書の完了の状態を固定する (ADR 0007 の
第 2a 段、#73。第 2b 段、#79)。確かめること:

1. 先頭 20 行以内に「書かないこと」の決まり (要求・応答の本文、認証の情報、`exl3-tp2` の
   中身) がある
2. 重ねた構成の生成 (`--vllm-overlay k2s2a`、`--weights k2s1b` と `--weights k2s2a`、生成される
   構成の名前とファイル名)、写しとパッチの確認 (`experiments/k2-vllm-overlay`、`k2s2a.json`、
   `serving/payload/vllm-overlay`、`test_vllm_overlay.py`)、配布 (`serve push`)、重ねる先の
   道筋の読み取り確認 (読み取りのコマンド `importlib.util.find_spec(`、期待値の説明
   `vllm.__file__`、`--network none`)、関門 (`test -e`)、起動と回収と停止
   (`serve check` / `serve start` / `serve smoke` / `serve logs` / `serve stop`)、起動のログでの
   FP8 の scheme の確かめ方 (`VLLM_LOGGING_LEVEL`、`Using scheme: CompressedTensorsW8A16Fp8`、
   `HummingFP8ScaledMMLinearKernel`、`GPU KV cache size`)、期待する層と、期待しない層の印がある
3. 使う `serve <サブコマンド>` が、すべて `cli.build_parser()` に実在する
4. 相対リンクの道筋が、すべて実在する
5. `ssh`・`docker run`・`serve push`・`serve verify`・`serve start`・`serve stop` を含むコード塊の
   **それぞれに**、直前の見出しか直前の段落に、⚠ と「了承を得てから」がある。`docker run`・
   `serve push`・`serve start` は、コード塊として書かれている
6. 2 台の写しと Mac の写しの `sha256sum` / `shasum -a 256` の突き合わせは、コード塊の命令として
   読める形で、hash コマンドの**それぞれの**直後に `| LC_ALL=C sort` がある (2 台 × 2 側)
7. 第 2b 段 (`k2s2b`。#79) の節がある。直すこと (`kda.py` の `quant_config`、`model.py` の
   `packed_modules_mapping`)、写しの生成と確認 (`overlay.py refresh k2s2b` / `check k2s2b`)、重ねた
   構成の生成 (`--weights k2s2b`、`--vllm-overlay k2s2b`、構成の名前とファイル名、前提の
   `k2s2b.manifest.json`)、起動のログの期待 (KDA のまとめた層 `in_proj_qkvbfg_a` も
   `CompressedTensorsW8A16Fp8`)、古い重みで重ねたときの失敗の印
   (`Found different quantization schemes`) が、その節にある

RED: 文書がなければ、`read_text` で `pytest.fail` する。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest

from procedure_doc_kit import (
    SERVE_CALL,
    ask_mark_problems,
    checksum_sort_problems,
    code_blocks,
    lines_inside,
    read_text,
    relative_link_targets,
    subparser_map,
)
from serving_kit import cli

SERVING_DIR: Final[Path] = Path(__file__).resolve().parents[2]
"""`serving/` (この試験から見て 2 つ上)。"""

REPO_ROOT: Final[Path] = SERVING_DIR.parent

DOC_PATH: Final[Path] = REPO_ROOT / "docs" / "vllm-baseline" / "k2-vllm-overlay-procedure.md"

_WRITE_NOTHING_MARK: Final[str] = "書かないこと"

_FORBIDDEN_CONTENT_MARKERS: Final[tuple[str, ...]] = (
    "送った内容",
    "認証の情報",
    "exl3-tp2",
)

_LEADING_LINE_LIMIT: Final[int] = 20

REQUIRED_MARKERS: Final[tuple[str, ...]] = (
    # 生成 (段 0: 第 1 段の重みのまま。段 1: 第 2a 段の重み)
    "--vllm-overlay k2s2a",
    "--weights k2s1b",
    "--weights k2s2a",
    "p2-nope-tp2-full-k2s1b-ov-k2s2a",
    "p2-nope-tp2-full-k2s2a-ov-k2s2a",
    "tp2-full-k2s2a-ov-k2s2a.toml",
    "k2s2a.manifest.json",
    "#74",
    # 写しとパッチの確認
    "experiments/k2-vllm-overlay",
    "k2s2a.json",
    "serving/payload/vllm-overlay",
    "test_vllm_overlay.py",
    # 配布・関門・起動・回収・停止
    "serve push",
    "serve check",
    "serve start",
    "serve smoke",
    "serve logs",
    "serve stop",
    "readonly",
    "test -e",
    # 重ねる先の道筋の読み取り確認
    "/usr/local/lib/python3.12/dist-packages/vllm",
    "importlib.util.find_spec(",
    "vllm.__file__",
    "--network none",
    "LC_ALL=C sort",
    # 起動のログでの、FP8 の scheme の確かめ方
    "VLLM_LOGGING_LEVEL",
    "Using scheme: CompressedTensorsW8A16Fp8",
    "HummingFP8ScaledMMLinearKernel",
    "GPU KV cache size",
    "k2s1b",
    # 期待する層と、期待しない層
    "lm_head",
    "fused_qkv_a_proj",
    "q_b_proj",
    "o_proj",
    "f_b_proj",
    "g_b_proj",
    "in_proj_qkvbfg_a",
    "indexer",
    "kv_b_proj",
    # 根拠と、了承
    "ADR 0007",
    "⚠",
    "了承を得てから",
)
"""手順書に必要な印。"""

_MACHINE_CALLS_THE_PROCEDURE_MUST_SHOW: Final[frozenset[str]] = frozenset(
    {"docker run", "serve push", "serve start"}
)
"""手順書が、コード塊として書いていなければならない呼び出し (重ねる先の道筋の読み取り確認、
写しの配布、起動)。`ssh`・`serve verify`・`serve stop` は、あれば ⚠ の検査の対象になる。"""

_MINIMUM_CHECKSUM_COMMANDS: Final[int] = 4
"""突き合わせの hash コマンドの数: 2 台 (`spark-153d`、`spark-5083`) × 2 側 (Spark と Mac)。"""


# --- 1. 「書かないこと」の決まり -------------------------------------------


def test_the_procedure_states_the_write_nothing_rule_up_front() -> None:
    """先頭に「書かないこと」の決まりと、その 3 項目がある。"""
    text = read_text(DOC_PATH)
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


# --- 2. 必要な印 ------------------------------------------------------------


@pytest.mark.parametrize("marker", REQUIRED_MARKERS)
def test_the_procedure_mentions_the_required_marker(marker: str) -> None:
    """手順書に、生成・確認・配布・関門・起動・ログでの確かめ方・層の印と、⚠ の印がある。"""
    assert marker in read_text(DOC_PATH)


# --- 3. serve のサブコマンドの実在 ----------------------------------------


def test_every_serve_command_in_the_procedure_exists() -> None:
    """手順書が打たせるコマンドが、すべて実在する (存在しない引数を書かないための歯止め)。"""
    text = read_text(DOC_PATH)
    top = subparser_map(cli.build_parser())

    used = {found.group(1) for found in SERVE_CALL.finditer(text)}
    assert used, "手順書に `serve <サブコマンド>` が 1 つもない"
    assert not sorted(used - set(top)), f"実在しないサブコマンド: {sorted(used - set(top))}"


# --- 4. 相対リンクの実在 --------------------------------------------------


def test_every_relative_link_in_the_procedure_resolves_to_a_real_file() -> None:
    """手順書のリポジトリ内の相対リンクが、すべて実在するファイルを指す。"""
    text = read_text(DOC_PATH)
    targets = relative_link_targets(text)
    assert targets, "手順書にリポジトリ内の相対リンクが 1 つもない"

    missing = [target for target in targets if not (DOC_PATH.parent / target).is_file()]
    assert not missing, f"手順書のリンクの道筋が実在しない: {missing}"


# --- 5. 実機を操作する塊の ⚠ ---------------------------------------------


def test_every_block_that_touches_the_machines_is_preceded_by_the_ask_mark() -> None:
    """実機を操作するコード塊の**それぞれ**に、直前の見出しか段落に、⚠ と「了承を得てから」がある。

    `docker run`・`serve push`・`serve start` は、コード塊として書かれている (印の検査が、
    塊が 1 つもなくて素通りにならない)。
    """
    shown, problems = ask_mark_problems(read_text(DOC_PATH))

    absent = sorted(_MACHINE_CALLS_THE_PROCEDURE_MUST_SHOW - shown)
    assert not absent, f"コード塊として書かれていない実機の操作: {absent}"
    assert not problems, "\n".join(problems)


# --- 6. 写しの突き合わせの並べ替え ----------------------------------------


def test_the_checksum_comparison_sorts_every_side_by_file_name() -> None:
    """2 台の写しと Mac の写しの突き合わせは、hash コマンドのそれぞれの直後で、並べ替える。

    `*.py` の展開順は、ロケールで違う (Spark と Mac で、中身が同じでも差分が出たことがある)。
    片側だけを並べ替えても、もう片側の順は変わらない。
    """
    checked, problems = checksum_sort_problems(read_text(DOC_PATH))

    assert checked >= _MINIMUM_CHECKSUM_COMMANDS, (
        f"突き合わせの hash コマンドが {checked} 個しか読めない (2 台 × 2 側 = "
        f"{_MINIMUM_CHECKSUM_COMMANDS} 個が要る): {DOC_PATH}"
    )
    assert not problems, "\n".join(problems)


# --- 7. 第 2b 段 (`k2s2b`)。KDA のまとめた層 `in_proj_qkvbfg_a` も FP8 で読む (#79) --------------

_STAGE_2B_HEADING_MARK: Final[str] = "第 2b 段"

STAGE_2B_SECTION_MARKERS: Final[tuple[str, ...]] = (
    # 直すこと
    "quant_config",
    "packed_modules_mapping",
    # 写しの生成と確認 (Mac)
    "overlay.py refresh k2s2b",
    "overlay.py check k2s2b",
    "k2s2b.json",
    # 重ねた構成の生成 (Mac)
    "--vllm-overlay k2s2b",
    "--weights k2s2b",
    "k2s2b.manifest.json",
    "p2-nope-tp2-full-k2s2b-ov-k2s2b",
    "tp2-full-k2s2b-ov-k2s2b.toml",
    # 起動のログでの期待と、古い重みで重ねたときの失敗の印
    "in_proj_qkvbfg_a",
    "CompressedTensorsW8A16Fp8",
    "Found different quantization schemes",
    # 根拠
    "#79",
)
"""第 2b 段の節に必要な印。ほかの節にも出る名前 (`quant_config`、`in_proj_qkvbfg_a` など) は、
節の中にあることを見る (文書のどこかにあるだけでは、第 2b 段の説明にならない)。"""


def _stage_2b_section(text: str) -> str:
    """見出しに「第 2b 段」を含む `## ` の節 (次の `## ` の見出し、なければ末尾の手前まで)。

    コード塊の中の `## ` は見出しと取り違えない。節がなければ、そこで落とす。
    """
    lines = text.splitlines()
    inside = lines_inside(code_blocks(text))
    headings = [
        number
        for number, line in enumerate(lines, start=1)
        if number not in inside and line.startswith("## ")
    ]
    begin = next(
        (number for number in headings if _STAGE_2B_HEADING_MARK in lines[number - 1]), None
    )
    if begin is None:
        pytest.fail(f"見出しに「{_STAGE_2B_HEADING_MARK}」を含む節 (`## `) がない: {DOC_PATH}")
    end = next((number for number in headings if number > begin), len(lines) + 1)
    return "\n".join(lines[begin - 1 : end - 1])


def test_the_procedure_has_a_stage_2b_section_for_the_k2s2b_overlay() -> None:
    """見出しに「第 2b 段」と `k2s2b` を含む節がある。"""
    section = _stage_2b_section(read_text(DOC_PATH))

    assert re.search(r"k2s2b", section.splitlines()[0]), section.splitlines()[0]


@pytest.mark.parametrize("marker", STAGE_2B_SECTION_MARKERS)
def test_the_stage_2b_section_mentions_the_required_marker(marker: str) -> None:
    """第 2b 段の節に、直すこと・写しの生成と確認・構成の生成・期待と失敗の印・根拠がある。"""
    assert marker in _stage_2b_section(read_text(DOC_PATH))


def test_the_stage_2b_section_expects_the_kda_merged_layer_among_the_fp8_hits() -> None:
    """第 2b 段の節は、起動のログの期待として、KDA のまとめた層 `in_proj_qkvbfg_a` を、W8A16 FP8 の
    scheme (`CompressedTensorsW8A16Fp8`) が当たる側に書く。

    第 2a 段では、まとめた層は BF16 のままなので、当たってはいけない層だった。第 2b 段の節の
    `in_proj_qkvbfg_a` を含む段落 (空行で区切られた連なり) のどれかに、W8A16 の印がある。
    """
    section = _stage_2b_section(read_text(DOC_PATH))
    paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n", section) if paragraph.strip()]

    expecting = [
        paragraph
        for paragraph in paragraphs
        if "in_proj_qkvbfg_a" in paragraph and "W8A16" in paragraph
    ]

    assert expecting, "in_proj_qkvbfg_a と W8A16 を同じ段落に書いた期待がない"
