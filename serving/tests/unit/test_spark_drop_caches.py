"""`ops/spark-drop-caches/` の試験 (Issue #84。`test_spark_power_caps.py` と同じ形)。

`spark-drop-caches` は、両台の Spark で、ページキャッシュを捨てる (`sync` のあとに
`/proc/sys/vm/drop_caches` へ `3` を書く) root 所有の POSIX sh のスクリプトである。GB10 は CPU と
GPU がメモリを共有するので、ページキャッシュが埋まると CUDA から見える空きが小さくなり、
`--load-format instanttensor` の起動が落ちる。その前に、計測者が了承したうえで、固定の
1 コマンドとして流す。

書き先の道筋は、試験用に、コマンドライン引数 `--target` だけで差し替えられる。環境変数では
差し替えられない。sudoers は、引数なしのこのコマンドだけを許すので、`--target` は sudo 経由では
使えない (root で任意のファイルに書けない)。実機の `/proc/sys/vm/drop_caches` には、この試験は
触れない (一時ファイルに書かせる)。

確かめること:

- 引数なしの実行は、対象に `3` を書いて 0 で終わる (`--target` で一時ファイルを指す)
- 未知の引数と位置引数は、対象に書かずに 2 で終わる。環境変数では書き先を差し替えられない
- `--target` を付けないときの書き先の既定は、`/proc/sys/vm/drop_caches`
- sudoers は、引数なしの固定のコマンド 1 つだけを NOPASSWD で許す (ワイルドカードなし)。
  `visudo -cf` があれば、それでも検査する (なければ skip)
- README は、⚠ と「了承を得てから」を、`sudo` を使う塊のそれぞれに付け、導入の手順 (`visudo -cf`、
  設置先、`/etc/sudoers.d/`) と、関門 `memory_free` との関係と、未確認のことを書く

実機の確かめ (sudoers の `""` の形が Spark の `visudo -cf` を通ること、`/usr/local/sbin` の実在、
`drop_caches` への書き込みで実際に `MemFree` が増えること) は、この試験の対象外である。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest

from procedure_doc_kit import (
    ASK_MARKS,
    code_blocks,
    command_lines,
    lines_inside,
    nearest_heading,
    preceding_paragraph,
    read_text,
)

SERVING_DIR: Final[Path] = Path(__file__).resolve().parents[2]
REPO_ROOT: Final[Path] = SERVING_DIR.parent
OPS_DIR: Final[Path] = REPO_ROOT / "ops" / "spark-drop-caches"
SCRIPT_PATH: Final[Path] = OPS_DIR / "spark-drop-caches"
SUDOERS_PATH: Final[Path] = OPS_DIR / "spark-drop-caches.sudoers"
README_PATH: Final[Path] = OPS_DIR / "README.md"

INSTALLED_PATH: Final[str] = "/usr/local/sbin/spark-drop-caches"
"""スクリプトを置く場所 (sudoers が許す、ただ 1 つのコマンド)。"""

DROP_CACHES_PATH: Final[str] = "/proc/sys/vm/drop_caches"
"""書き先の既定 (実機の、ページキャッシュを捨てる入り口)。"""


def _run(
    args: Sequence[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """`spark-drop-caches` を `sh` で流す (`env` は、呼び出し元の環境変数に足す)。"""
    return subprocess.run(
        ["sh", str(SCRIPT_PATH), *args],
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )


# --- スクリプト -------------------------------------------------------------------


def test_it_writes_three_to_the_target_and_exits_zero(tmp_path: Path) -> None:
    """引数なしの実行 (書き先だけを、試験用の `--target` で一時ファイルにする) は、対象に
    `3` を書いて 0 で終わる。"""
    target = tmp_path / "drop_caches"

    result = _run(["--target", str(target)])

    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8") == "3\n"


@pytest.mark.parametrize(
    "template",
    [
        ["--bogus", "--target", "{target}"],
        ["--target", "{target}", "--bogus"],
        ["--target", "{target}", "extra-positional"],
    ],
    ids=["unknown-option-first", "unknown-option-last", "positional"],
)
def test_an_unknown_argument_exits_two_without_writing(tmp_path: Path, template: list[str]) -> None:
    """未知の引数と位置引数は、使い方を出して 2 で終わり、対象に書かない (書き先は、間違って
    実機の入り口にならないよう、一時ファイルを指したまま流す)。"""
    target = tmp_path / "drop_caches"

    result = _run([argument.format(target=target) for argument in template])

    assert result.returncode == 2, result.stderr
    assert not target.exists()


def test_the_target_cannot_be_replaced_through_the_environment(tmp_path: Path) -> None:
    """環境変数では、書き先を差し替えられない。`TARGET` だけを与えて、`--target` で別のファイルを
    指すと、`--target` の道筋にだけ書き、環境変数の道筋には書かない。"""
    from_environment = tmp_path / "from-environment"
    from_argument = tmp_path / "from-argument"

    result = _run(["--target", str(from_argument)], env={"TARGET": str(from_environment)})

    assert result.returncode == 0, result.stderr
    assert from_argument.read_text(encoding="utf-8") == "3\n"
    assert not from_environment.exists()


def test_the_default_target_is_the_drop_caches_entry() -> None:
    """`--target` を付けないときの書き先は `/proc/sys/vm/drop_caches` である (実機の入り口には
    触れられないので、スクリプトの中の既定の道筋で確かめる)。"""
    assert DROP_CACHES_PATH in read_text(SCRIPT_PATH)


# --- sudoers -----------------------------------------------------------------------


def _sudoers_commands() -> list[str]:
    """sudoers の、コメントと空行を除いた行。"""
    return [
        line.strip()
        for line in read_text(SUDOERS_PATH).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_the_sudoers_file_allows_exactly_the_fixed_command_without_arguments() -> None:
    """sudoers は、`j5ik2o` に、root として、引数なしの固定のコマンド 1 つだけを NOPASSWD で
    許す (末尾の `""` は、引数なしだけを許す形)。ワイルドカードは、コメントも含めて書かない。"""
    commands = _sudoers_commands()

    assert commands == [f'j5ik2o ALL=(root) NOPASSWD: {INSTALLED_PATH} ""'], commands
    assert "*" not in read_text(SUDOERS_PATH)


def test_the_sudoers_file_passes_visudo_when_available() -> None:
    """`visudo` があれば、`visudo -cf` で sudoers の断片を検査する (なければ skip)。"""
    visudo = shutil.which("visudo")
    if visudo is None:
        pytest.skip("visudo がない")
    result = subprocess.run(
        [visudo, "-cf", str(SUDOERS_PATH)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --- README ------------------------------------------------------------------------

README_MARKERS: Final[tuple[str, ...]] = (
    INSTALLED_PATH,
    "visudo -cf",
    "/etc/sudoers.d/spark-drop-caches",
    "memory_free",
    "instanttensor",
    "未確認",
)
"""README に要る印 (固定のコマンドの置き場所、導入の手順、関門との関係、対策の根拠、未確認の
こと)。"""


@pytest.mark.parametrize("marker", README_MARKERS)
def test_the_readme_covers_the_installation_and_the_reason(marker: str) -> None:
    assert marker in read_text(README_PATH)


def test_every_sudo_block_in_the_readme_is_preceded_by_the_ask_mark() -> None:
    """`sudo` を使うコード塊の**それぞれ**に、直前の見出しか直前の段落に、⚠ と「了承を得てから」
    がある (root の操作は、了承を得てから行う)。"""
    text = read_text(README_PATH)
    lines = text.splitlines()
    blocks = code_blocks(text)
    inside = lines_inside(blocks)

    sudo_blocks = [block for block in blocks if "sudo" in block.commands]
    assert sudo_blocks, "README に `sudo` を使うコード塊が 1 つもない"

    problems: list[str] = []
    for block in sudo_blocks:
        window = (
            f"{nearest_heading(lines, inside, block.open_line)}\n"
            f"{preceding_paragraph(lines, block.open_line)}"
        )
        missing = [mark for mark in ASK_MARKS if mark not in window]
        if missing:
            problems.append(f"{block.open_line} 行目からの塊の直前に、{missing} がない")
    assert not problems, "\n".join(problems)


def test_the_readme_shows_the_usage_from_the_mac_with_the_fixed_command() -> None:
    """README の使い方は、Mac から両台に、`ssh` で固定のコマンド (引数なし) を流す形で書く。"""
    commands = "\n".join(block.commands for block in code_blocks(read_text(README_PATH)))

    for host in ("spark-153d", "spark-5083"):
        assert re.search(rf"ssh .*{host} +'sudo {re.escape(INSTALLED_PATH)}'", commands), host


def test_the_diagnostic_docker_run_does_not_use_sudo() -> None:
    """「原因の確かめ方」の `docker run` の読み取りは、root を求めない (issue #84)。この一式の
    sudoers は `spark-drop-caches` 単体だけを NOPASSWD で許すので、別のコマンドに `sudo` を
    足すと、対話なし ssh (`BatchMode=yes`) の前提が崩れる。"""
    docker_run_blocks = [
        block for block in code_blocks(read_text(README_PATH)) if "docker run" in block.commands
    ]
    assert docker_run_blocks, "README に `docker run` を使うコード塊が 1 つもない"

    problems = [
        f"{block.open_line} 行目からの塊に sudo がある"
        for block in docker_run_blocks
        if "sudo" in block.commands
    ]
    assert not problems, "\n".join(problems)


_RSYNC_DESTINATION: Final[re.Pattern[str]] = re.compile(
    r"rsync\b.*\s([\w.-]+):(/\S+/)\s*$", re.MULTILINE
)
"""rsync の宛先 (ホスト名と、その下のディレクトリ)。"""

_INSTALL_SOURCE: Final[re.Pattern[str]] = re.compile(r"sudo install\b.*?\s(/\S+/)[^/\s]+\s+/\S+")
"""`sudo install` が読む、写し元のディレクトリ (ファイル名の手前まで)。"""


def test_the_install_source_matches_the_rsync_destination() -> None:
    """初回の設置で、`sudo install` が読む道筋は、直前の rsync が両台へ写した宛先と一致する
    (issue #84。どちらかだけを書き換えると、install が存在しない道筋を指す)。"""
    commands = "\n".join(block.commands for block in code_blocks(read_text(README_PATH)))

    rsync_destinations = {directory for _host, directory in _RSYNC_DESTINATION.findall(commands)}
    assert rsync_destinations, "README に rsync の宛先を書いた行が 1 つもない"
    assert len(rsync_destinations) == 1, f"rsync の宛先が揃っていない: {rsync_destinations}"

    install_sources = set(_INSTALL_SOURCE.findall(commands))
    assert install_sources, "README に sudo install の写し元を書いた行が 1 つもない"

    assert install_sources == rsync_destinations, (
        f"sudo install の写し元 {install_sources} が rsync の宛先 {rsync_destinations} と"
        " 一致しない"
    )


def _sudoers_fragment_destinations(text: str) -> dict[str, str]:
    """rsync の宛先 (`_RSYNC_DESTINATION`) から、台ごとに、移した先の sudoers 断片の道筋を
    組み立てる (書き直した道筋の文字列を、別の場所に重ねて書かない)。"""
    commands = "\n".join(block.commands for block in code_blocks(text))
    return {
        host: f"{directory}{SUDOERS_PATH.name}"
        for host, directory in _RSYNC_DESTINATION.findall(commands)
    }


def _syntax_check_problems(text: str) -> list[str]:
    """台ごとに、sudoers の断片を検査する行が、同じ台でその断片を `/etc/sudoers.d/` へ置く行
    より前に出現するかを確かめ、問題の一覧を返す (空なら問題なし)。

    検査の行は、host がトークンとして現れ、`visudo -cf` と、その台の移した先の道筋
    (`_sudoers_fragment_destinations` で組み立てる) を含む行に絞る。Mac での事前検査の行
    (`README.md` の段 1。host を含まない) は数えない。台ごとに見ないと、片方の台の検査の行が
    欠けても、もう片方の台の行だけで判定が通ってしまう (issue #84。U6)。
    """
    lines = command_lines(text)
    destinations = _sudoers_fragment_destinations(text)

    problems: list[str] = []
    for host, destination in destinations.items():
        check_indices = [
            index
            for index, line in enumerate(lines)
            if host in line.split() and "visudo -cf" in line and destination in line
        ]
        install_indices = [
            index
            for index, line in enumerate(lines)
            if host in line.split()
            and "sudo install" in line
            and "/etc/sudoers.d/spark-drop-caches" in line
        ]

        if not check_indices:
            problems.append(f"{host}: sudoers の断片を検査する行がない")
            continue
        if not install_indices:
            problems.append(f"{host}: sudoers の断片を /etc/sudoers.d/ へ置く行がない")
            continue
        if min(check_indices) >= min(install_indices):
            problems.append(
                f"{host}: 検査の行 (index {check_indices}) が、設置の行 "
                f"(index {install_indices}) より前にない"
            )
    return problems


def test_the_syntax_check_comes_before_the_sudoers_fragment_is_installed() -> None:
    """初回の設置で、台ごとに、`visudo -cf` で sudoers の断片を検査する行は、同じ台でその断片を
    `/etc/sudoers.d/` へ置く行より前に出現する (issue #84。U5・U6。前回の修正 (U-README-2) が
    この検査の段を落とし、両台で検査せずに断片を置く形になった退行があったため、順序が崩れて
    検査なしに断片が置かれることを検出する)。"""
    problems = _syntax_check_problems(read_text(README_PATH))
    assert not problems, "\n".join(problems)
