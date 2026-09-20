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


def test_the_duplicated_constants_and_seed_helpers_cannot_drift_apart() -> None:
    """依存の向きの都合で 2 か所以上に書いてある決まりが、同じ値のままであること。

    片方だけ変えると、印は付くのに集計からは外れない、生成される入力が変わるのに
    生成器の版が上がらない、といった形で、結果が黙って歪む (全体の検証の指摘)。
    """
    from bench_harness.analysis import summarize
    from bench_harness.corpus import conversation, needle, synth, tools
    from bench_harness.suites import base

    assert base.MIN_SPEED_OUTPUT_TOKENS == summarize.MIN_SPEED_OUTPUT_TOKENS

    parts = ("purpose", 7, "条件/鍵", 0)
    encoders = [
        base._encode_part,
        synth._encode_part,
        tools._encode_part,
        needle._encode_part,
        conversation._encode_part,
    ]
    for part in parts:
        assert len({encoder(part) for encoder in encoders}) == 1, part

    seeds = {
        tools._derive_seed(11, 3, "p"),
        needle._derive_seed(11, 3, "p"),
        conversation._derive_seed(11, 3, "p"),
    }
    assert len(seeds) == 1
