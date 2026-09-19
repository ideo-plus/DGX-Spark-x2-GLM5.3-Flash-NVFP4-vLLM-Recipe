"""パッケージが取り込めて、CLI の --help と --version が動くことを確かめる、最小の疎通試験。"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_package_exposes_version_string() -> None:
    import bench_harness

    assert isinstance(bench_harness.__version__, str)
    assert bench_harness.__version__ != ""


def test_cli_help_exits_zero_and_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    from bench_harness.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower()


def test_cli_version_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    import bench_harness
    from bench_harness.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert bench_harness.__version__ in captured.out


def test_cli_help_via_subprocess_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "bench_harness.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
