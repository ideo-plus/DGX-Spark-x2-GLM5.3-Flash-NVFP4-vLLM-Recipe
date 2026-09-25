"""K2 の torch プロファイラーの手順書 (`docs/vllm-baseline/k2-profile-procedure.md`) の試験 (C7)。

文書のタスクなので、振る舞いの試験の代わりに、計画の完了契約 C7 を固定する。確かめること:

1. 先頭 20 行以内に「書かないこと」の決まり (要求・応答の本文、認証の情報、`exl3-tp2` の
   中身) がある
2. 構成の生成 (`--torch-profiler` / `tp2-full-prof.toml` / `p2-nope-tp2-full-prof`) と
   `serve check` の印がある
3. 短い負荷の掛け方 (`serve smoke` と `--max-tokens`、`bench run` と `--profile probe` と
   `--target p2-nope-tp2-full`) の印がある
4. `curl -X POST` で `/start_profile` と `/stop_profile` を呼び、`/metrics` の
   `vllm:iteration_tokens_total_count` の増分を `--steps` に渡す、という印がある
5. 回収 (`serve logs` と `logs/torch-profile` と `.pt.trace.json.gz`) と、集計
   (`summarize_trace.py`)、本文が入らないことの確認 (`zgrep`) の印がある
6. 状態を変える塊に ⚠ の印がある
7. 使う `serve <サブコマンド>` が、すべて `cli.build_parser()` に実在する
8. 相対リンクの道筋が、すべて実在する

RED: 文書がなければ、`_read_text` で `pytest.fail` する。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Final

import pytest

from serving_kit import cli

SERVING_DIR: Final[Path] = Path(__file__).resolve().parents[2]
"""`serving/` (この試験から見て 2 つ上)。"""

REPO_ROOT: Final[Path] = SERVING_DIR.parent

DOC_PATH: Final[Path] = REPO_ROOT / "docs" / "vllm-baseline" / "k2-profile-procedure.md"

_WRITE_NOTHING_MARK: Final[str] = "書かないこと"

_FORBIDDEN_CONTENT_MARKERS: Final[tuple[str, ...]] = (
    "送った内容",
    "認証の情報",
    "exl3-tp2",
)

_LEADING_LINE_LIMIT: Final[int] = 20

REQUIRED_MARKERS: Final[tuple[str, ...]] = (
    "--torch-profiler",
    "p2-nope-tp2-full-prof",
    "tp2-full-prof.toml",
    "serve check",
    "serve smoke",
    "--max-tokens",
    "bench run",
    "--profile probe",
    "--target p2-nope-tp2-full",
    "curl -X POST",
    "/start_profile",
    "/stop_profile",
    "vllm:iteration_tokens_total_count",
    "--steps",
    "serve logs",
    "logs/torch-profile",
    ".pt.trace.json.gz",
    "summarize_trace.py",
    "zgrep",
    "⚠",
)
"""手順書に必要な印 (C7)。"""


def _read_text(path: Path) -> str:
    """文書の全文 (なければ、そこで落とす)。"""
    if not path.is_file():
        pytest.fail(f"文書がない: {path}")
    return path.read_text(encoding="utf-8")


# --- 1. 「書かないこと」の決まり -------------------------------------------


def test_the_procedure_states_the_write_nothing_rule_up_front() -> None:
    """先頭に「書かないこと」の決まりと、その 3 項目がある (C7)。"""
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


# --- 2〜6. 必要な印 ---------------------------------------------------------


@pytest.mark.parametrize("marker", REQUIRED_MARKERS)
def test_the_procedure_mentions_the_required_marker(marker: str) -> None:
    """手順書に、生成・負荷・計測・回収・集計・本文の確認・⚠ の印がある (C7)。"""
    assert marker in _read_text(DOC_PATH)


# --- 7. serve のサブコマンドの実在 ----------------------------------------


_SERVE_CALL: Final[re.Pattern[str]] = re.compile(r"\bserve +([a-z][a-z-]*)(?![\w-])")
"""手順書の中の `serve <サブコマンド>` (`p2-…` のような、数字を含む構成の名前は拾わない)。"""


def _subparser_map(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """引数解析器の、サブコマンドの名前から解析器への対応。"""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action._name_parser_map)
    pytest.fail("build_parser() にサブコマンドの解析器がない")


def test_every_serve_command_in_the_procedure_exists() -> None:
    """手順書が打たせるコマンドが、すべて実在する (存在しない引数を書かないための歯止め。C7)。"""
    text = _read_text(DOC_PATH)
    top = _subparser_map(cli.build_parser())

    used = {found.group(1) for found in _SERVE_CALL.finditer(text)}
    assert used, "手順書に `serve <サブコマンド>` が 1 つもない"
    assert not sorted(used - set(top)), f"実在しないサブコマンド: {sorted(used - set(top))}"


# --- 8. 相対リンクの実在 --------------------------------------------------


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
    """手順書のリポジトリ内の相対リンクが、すべて実在するファイルを指す (C7)。"""
    text = _read_text(DOC_PATH)
    targets = _relative_link_targets(text)
    assert targets, "手順書にリポジトリ内の相対リンクが 1 つもない"

    missing = [target for target in targets if not (DOC_PATH.parent / target).is_file()]
    assert not missing, f"手順書のリンクの道筋が実在しない: {missing}"
