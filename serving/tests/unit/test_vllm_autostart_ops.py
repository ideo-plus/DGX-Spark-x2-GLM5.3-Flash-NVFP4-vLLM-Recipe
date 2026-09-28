"""`ops/vllm-autostart/` の unit・install・uninstall・sudoers と、`serving/payload/vllm-autostart/`
への配布の同一性の試験 (計画の完了契約 C6)。

`ops/spark-power-caps/` の試験 (`test_spark_power_caps.py`) と同じ考え方: 実機には触らず、
コミットしたファイルの中身と、`sh -n`・`visudo -cf` (あれば) の構文だけを確かめる。

要求シナリオ (gherkin) の対象外の契約なので、計画の完了契約表の「成立する振る舞い」列 (unit
の 4 つの値、`remote_root` と `nodes.toml` の一致、install/uninstall が `sh -n` を通る、
sudoers が 2 つの固定形だけ、payload の 4 ファイルが ops とバイト同一) を P、
「拒否すべき誤実装」列 (sudoers に `*` や他コマンド、payload と ops の不一致) を N として、
1 つずつの観点に分けてテストに落とす (`test_spark_power_caps.py` と同じ粒度)。
"""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
SERVING_DIR: Final[Path] = REPO_ROOT / "serving"
OPS_DIR: Final[Path] = REPO_ROOT / "ops" / "vllm-autostart"
PAYLOAD_DIR: Final[Path] = SERVING_DIR / "payload" / "vllm-autostart"
NODES_PATH: Final[Path] = SERVING_DIR / "config" / "nodes.toml"

SCRIPT_NAME: Final[str] = "vllm_autostart.py"
UNIT_NAME: Final[str] = "vllm-autostart.service"
INSTALL_NAME: Final[str] = "vllm-autostart-install"
UNINSTALL_NAME: Final[str] = "vllm-autostart-uninstall"
SUDOERS_NAME: Final[str] = "vllm-autostart.sudoers"

OPS_UNIT: Final[Path] = OPS_DIR / UNIT_NAME
OPS_INSTALL: Final[Path] = OPS_DIR / INSTALL_NAME
OPS_UNINSTALL: Final[Path] = OPS_DIR / UNINSTALL_NAME
OPS_SUDOERS: Final[Path] = OPS_DIR / SUDOERS_NAME

PAYLOAD_FILES: Final[tuple[str, ...]] = (SCRIPT_NAME, UNIT_NAME, INSTALL_NAME, UNINSTALL_NAME)


def _read(path: Path) -> str:
    assert path.is_file(), f"ファイルがない: {path}"
    return path.read_text(encoding="utf-8")


# --- unit ファイル (成立する振る舞い: Type/ExecStart/Restart/WantedBy) -------


def test_unit_is_a_simple_restart_on_failure_user_service() -> None:
    """unit は `Type=simple`、`Restart=on-failure`、`WantedBy=default.target` を持つ
    (ユーザー単位の systemd で、落ちても自動で立ち上がる)。
    """
    text = _read(OPS_UNIT)
    assert "Type=simple" in text
    assert "Restart=on-failure" in text
    assert "WantedBy=default.target" in text


def test_unit_exec_start_matches_the_configured_remote_root() -> None:
    """`ExecStart` は `%h/vllm-baseline/payload/vllm-autostart/vllm_autostart.py` を、
    `--remote-root %h/vllm-baseline` を付けて起こし、この道筋は `nodes.toml` の
    両台の `remote_root` (ホームディレクトリ以下の相対部分) と一致する。
    """
    text = _read(OPS_UNIT)
    exec_start_lines = [line for line in text.splitlines() if line.startswith("ExecStart=")]
    assert exec_start_lines, "ExecStart がない"
    exec_start = exec_start_lines[0]
    assert "%h/vllm-baseline/payload/vllm-autostart/vllm_autostart.py" in exec_start
    assert "--remote-root" in exec_start
    assert "%h/vllm-baseline" in exec_start

    nodes = tomllib.loads(_read(NODES_PATH))
    roots = {node["remote_root"] for node in nodes["nodes"].values()}
    assert roots == {"/home/j5ik2o/vllm-baseline"}, roots


# --- install / uninstall (成立する振る舞い: sh -n を通る) --------------------


def test_install_and_uninstall_scripts_are_syntactically_valid_posix_sh() -> None:
    """install/uninstall は POSIX sh の構文として正しい (`sh -n` で検査、実行はしない)。"""
    for path in (OPS_INSTALL, OPS_UNINSTALL):
        result = subprocess.run(
            ["sh", "-n", str(path)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, f"{path}: {result.stderr}"


def test_install_and_uninstall_use_a_fixed_path_and_set_eu() -> None:
    """install/uninstall は `set -eu` を持ち、`ops/spark-power-caps` の流儀
    (固定した `PATH`) に倣う。
    """
    for path in (OPS_INSTALL, OPS_UNINSTALL):
        text = _read(path)
        assert "set -eu" in text


def test_install_enables_the_user_unit_and_uninstall_disables_it() -> None:
    """install は `systemctl --user enable --now`、uninstall は `disable --now` を含む。"""
    install_text = _read(OPS_INSTALL)
    uninstall_text = _read(OPS_UNINSTALL)
    assert "systemctl --user" in install_text
    assert "enable" in install_text and "--now" in install_text
    assert UNIT_NAME in install_text
    assert "systemctl --user" in uninstall_text
    assert "disable" in uninstall_text and "--now" in uninstall_text


# --- sudoers (成立する振る舞い: 2 つの固定形だけ。拒否すべき誤実装: * や他コマンド) -------


def test_the_sudoers_file_allows_exactly_the_linger_enable_and_disable_forms() -> None:
    """sudoers は、`loginctl enable-linger j5ik2o` と `disable-linger j5ik2o` の
    2 つの固定形だけを NOPASSWD で許す (`test_spark_power_caps.py` と同じ照合の単位: 許可の
    行そのもの)。
    """
    text = _read(OPS_SUDOERS)
    commands = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert len(commands) == 1, f"非コメントの行は 1 行だけ: {commands}"
    line = commands[0]

    assert "NOPASSWD:" in line, line
    principal, _, rest = line.partition("NOPASSWD:")
    assert principal.strip() == "j5ik2o ALL=(root)", principal

    allowed = [item.strip() for item in rest.split(",")]
    assert allowed == [
        "/usr/bin/loginctl enable-linger j5ik2o",
        "/usr/bin/loginctl disable-linger j5ik2o",
    ], allowed


def test_the_sudoers_file_has_no_wildcard_or_other_commands() -> None:
    """sudoers 全体 (コメントを含む) に `*` がなく、`loginctl` 以外のコマンドを許していない。"""
    text = _read(OPS_SUDOERS)
    assert "*" not in text
    commands = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert len(commands) == 1
    _principal, _, rest = commands[0].partition("NOPASSWD:")
    for item in rest.split(","):
        assert item.strip().startswith("/usr/bin/loginctl "), item


def test_the_sudoers_file_passes_visudo_when_available() -> None:
    """`visudo` があれば `visudo -cf` で検査する
    (なければ skip。`test_spark_power_caps.py` と同じ)。
    """
    visudo = shutil.which("visudo")
    if visudo is None:
        pytest.skip("visudo がない (CI の runner では実行する)")
    result = subprocess.run(
        [visudo, "-cf", str(OPS_SUDOERS)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# --- payload との同一性 (成立する振る舞い/拒否すべき誤実装: バイト同一かどうか) -----------


def test_the_payload_copy_has_the_same_four_files_as_ops() -> None:
    """`serving/payload/vllm-autostart/` は、`ops/vllm-autostart/` の 4 つのファイルを、
    過不足なく持つ (`serve push` は `serving/payload/` の中身しか配らない)。
    """
    assert PAYLOAD_DIR.is_dir(), f"ディレクトリがない: {PAYLOAD_DIR}"
    for name in PAYLOAD_FILES:
        assert (PAYLOAD_DIR / name).is_file(), f"payload に無い: {name}"


def test_the_payload_copy_is_byte_identical_to_ops() -> None:
    """写しの各ファイルは、`ops/vllm-autostart/` のファイルと 1 バイトも違わない。"""
    different = []
    for name in PAYLOAD_FILES:
        ops_bytes = (OPS_DIR / name).read_bytes()
        payload_bytes = (PAYLOAD_DIR / name).read_bytes()
        if ops_bytes != payload_bytes:
            different.append(name)
    assert not different, f"写しが ops と違う: {different}"
