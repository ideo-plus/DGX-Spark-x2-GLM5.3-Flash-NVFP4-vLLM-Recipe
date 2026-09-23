"""`serve watch --thermal-threshold` の端から端までの試験 (熱と CPU の観察の追加)。

`--thermal-threshold` の CLI からの配線 (`cli._add_watch` の `celsius` 型 →
`cli._cmd_watch` → `watch.watch(thermal_threshold_c=)`) を、`cli.main` を通して確かめる。
`tests/e2e/test_probe_watch.py` は書き込み対象外のファイルなので、そこに試験を足さずに、
この新規ファイルに置く (`e2e_kit.py` の下ごしらえは共有する)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import e2e_kit as k
import pytest

from fake_runner import Reply, Rule
from fake_vllm import FakeVllm, MetricsSample
from serving_kit import cli
from serving_kit import watch as w


@dataclass
class WatchClock:
    """試験用の時計 (`test_probe_watch.py` の `WatchClock` と同じ考え方)。壁の時計と単調な
    時計を、同じ歩幅で進める。実際には眠らない。"""

    mono: float = 0.0
    wall: datetime = field(default_factory=lambda: datetime(2026, 9, 22, 3, 0, 0, tzinfo=UTC))
    slept: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.mono

    def now(self) -> datetime:
        return self.wall

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.mono += seconds
        self.wall += timedelta(seconds=seconds)


def _watch_repo(base: Path, name: str, port: int) -> k.Repo:
    return k.make_repo(base / name, port=port)


def test_watch_thermal_threshold_from_the_cli_produces_a_thermal_event(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """`--thermal-threshold` を指定すると、その値以上の熱区域の観察で `thermal` の出来事が
    `cli.main` 経由で出る。熱の出来事では、記録の回収 (`docker` の呼び出し) をしない。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, waiting_requests=0))
    repo = _watch_repo(tmp_path, "thermal-low", k.port_of(fake_vllm))
    clock = WatchClock()
    zone0 = ("cat", f"{w.THERMAL_ZONE_DIR}thermal_zone0/temp")
    script = (Rule(prefix=zone0, replies=(Reply(stdout="86000\n"),)),)

    result = k.invoke(
        [
            "watch",
            k.SERVE_CONFIG,
            "--duration",
            "2",
            "--interval",
            "1",
            "--thermal-threshold",
            "85",
        ],
        repo,
        script=script,
        default=Reply(exit_code=1),
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
    )

    assert result.code == cli.EXIT_FAILED, result.err
    pairs = k.kv(result.out)
    assert pairs["events"] == "1"
    assert pairs["event.1.finding"] == "thermal"
    assert not any(call.argv and call.argv[0] == "docker" for call in result.runner.calls)


def test_watch_default_thermal_threshold_is_ninety(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """`--thermal-threshold` を付けなければ、既定 (90℃) が使われるので、86℃ では出ない。
    `result.json` の `thermal_threshold_c` も、既定の値がそのまま出ることを確かめる。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, waiting_requests=0))
    repo = _watch_repo(tmp_path, "thermal-default", k.port_of(fake_vllm))
    clock = WatchClock()
    zone0 = ("cat", f"{w.THERMAL_ZONE_DIR}thermal_zone0/temp")
    script = (Rule(prefix=zone0, replies=(Reply(stdout="86000\n"),)),)

    result = k.invoke(
        ["watch", k.SERVE_CONFIG, "--duration", "2", "--interval", "1"],
        repo,
        script=script,
        default=Reply(exit_code=1),
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
    )

    assert result.code == cli.EXIT_OK, result.err
    pairs = k.kv(result.out)
    assert pairs["events"] == "0"
    result_path = Path(pairs["samples_path"]).parent / "result.json"
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written["thermal_threshold_c"] == 90.0


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf", "abc"])
def test_watch_rejects_a_degenerate_thermal_threshold_at_the_parser(value: str) -> None:
    """退化した `--thermal-threshold` は、`argparse` の型検査で、Spark に触る前に断る。"""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["watch", "x", "--thermal-threshold", value])
