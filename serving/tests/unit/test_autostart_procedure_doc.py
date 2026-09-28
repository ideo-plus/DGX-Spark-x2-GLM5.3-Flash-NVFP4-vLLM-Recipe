"""自動起動の手順書 (`docs/vllm-baseline/autostart-procedure.md`) の試験 (計画の完了契約 C10)。

文書のタスクなので、振る舞いの試験の代わりに、計画の完了契約 C10 を固定する
(`test_k2_vllm_overlay_procedure_doc.py` を型にする)。確かめること:

1. 先頭 20 行以内に「書かないこと」の決まりがある
2. 設置 (`serve push`、`visudo -cf`、`loginctl enable-linger`、`vllm-autostart-install`)、
   指定 (`serve autostart set`)、確認 (`serve autostart status`、
   `systemctl --user is-active`)、短い実機試験 (`docker stop` で戻る確認、戻るまでの時間の
   読み方)、外し方 (`serve autostart clear`、`vllm-autostart-uninstall`、
   `loginctl disable-linger`) の印がある
3. 使う `serve <サブコマンド>` が、すべて `cli.build_parser()` に実在する
4. 相対リンクの道筋が、すべて実在する
5. 実機を操作するコード塊のそれぞれに、直前の見出しか段落に、⚠ と「了承を得てから」がある

RED: 文書がなければ、`read_text` で `pytest.fail` する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from procedure_doc_kit import (
    SERVE_CALL,
    ask_mark_problems,
    read_text,
    relative_link_targets,
    subparser_map,
)
from serving_kit import cli

SERVING_DIR: Final[Path] = Path(__file__).resolve().parents[2]
"""`serving/` (この試験から見て 2 つ上)。"""

REPO_ROOT: Final[Path] = SERVING_DIR.parent

DOC_PATH: Final[Path] = REPO_ROOT / "docs" / "vllm-baseline" / "autostart-procedure.md"

_WRITE_NOTHING_MARK: Final[str] = "書かないこと"

_FORBIDDEN_CONTENT_MARKERS: Final[tuple[str, ...]] = (
    "送った内容",
    "認証の情報",
    "docker logs",
)

_LEADING_LINE_LIMIT: Final[int] = 20

REQUIRED_MARKERS: Final[tuple[str, ...]] = (
    # 設置
    "serve push",
    "visudo -cf",
    "loginctl enable-linger",
    "vllm-autostart-install",
    # 指定
    "serve autostart set",
    # 確認
    "serve autostart status",
    "systemctl --user is-active",
    # 短い実機試験 (起動の確認、コンテナを止めて自動で戻ることの確認、戻るまでの時間の読み方)
    "docker stop",
    "ready_at",
    "started_at",
    # 外し方
    "serve autostart clear",
    "vllm-autostart-uninstall",
    "loginctl disable-linger",
    # 了承
    "⚠",
    "了承を得てから",
)
"""手順書に必要な印 (C10)。"""

_MACHINE_CALLS_THE_PROCEDURE_MUST_SHOW: Final[frozenset[str]] = frozenset(
    {
        "serve push",
        "serve autostart set",
        "serve autostart clear",
        "vllm-autostart-install",
        "vllm-autostart-uninstall",
        "loginctl enable-linger",
        "loginctl disable-linger",
        "docker stop",
    }
)
"""手順書が、コード塊として書いていなければならない、実機を操作する呼び出し。"""


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


# --- 2. 必要な印 -------------------------------------------------------------


@pytest.mark.parametrize("marker", REQUIRED_MARKERS)
def test_the_procedure_mentions_the_required_marker(marker: str) -> None:
    """手順書に、設置・指定・確認・短い実機試験・外し方・⚠ の印がある (C10)。"""
    assert marker in read_text(DOC_PATH)


# --- 3. serve のサブコマンドの実在 ------------------------------------------


def test_every_serve_command_in_the_procedure_exists() -> None:
    """手順書が打たせるコマンドが、すべて実在する。"""
    text = read_text(DOC_PATH)
    top = subparser_map(cli.build_parser())

    used = {found.group(1) for found in SERVE_CALL.finditer(text)}
    assert used, "手順書に `serve <サブコマンド>` が 1 つもない"
    assert not sorted(used - set(top)), f"実在しないサブコマンド: {sorted(used - set(top))}"


# --- 4. 相対リンクの実在 ------------------------------------------------------


def test_every_relative_link_in_the_procedure_resolves_to_a_real_file() -> None:
    """手順書のリポジトリ内の相対リンクが、すべて実在するファイルを指す。"""
    text = read_text(DOC_PATH)
    targets = relative_link_targets(text)
    assert targets, "手順書にリポジトリ内の相対リンクが 1 つもない"

    missing = [target for target in targets if not (DOC_PATH.parent / target).is_file()]
    assert not missing, f"手順書のリンクの道筋が実在しない: {missing}"


# --- 5. 実機を操作する塊の ⚠ -------------------------------------------------


def test_every_block_that_touches_the_machines_is_preceded_by_the_ask_mark() -> None:
    """実機を操作するコード塊の**それぞれ**に、直前の見出しか段落に、
    ⚠ と「了承を得てから」がある。
    """
    shown, problems = ask_mark_problems(read_text(DOC_PATH))

    absent = sorted(_MACHINE_CALLS_THE_PROCEDURE_MUST_SHOW - shown)
    assert not absent, f"コード塊として書かれていない実機の操作: {absent}"
    assert not problems, "\n".join(problems)
