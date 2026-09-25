"""連続の負荷の間の見張りの試験 (tasks.md 4.5)。

確かめること (design.md 「確認 › watch」、tasks.md 4.5 の完了の状態):

- **固まった**: GPU の使用率が読める (しきい値以上) とき、生成のトークンの数が窓のあいだ
  増えず、処理中の要求があると、時刻つきで `WatchOutcome.events` に出る (requirements 7.7)
- **応答しない**: `/health` の確認が続けて失敗しても、しきい値 (既定 3) に届くまでは出ず、
  届いた回に出る
- 処理中の要求がないときは、トークンの数が増えなくても「固まった」にならない
- GPU の使用率が (一度も) 読めないときは、生成のトークンの数だけで判定し、そのことが
  `WatchOutcome.detail` に出る
- 出来事が起きたら、2 台の記録が回収される (`FakeRunner` の記録)。続いているあいだは 1 件
- 出来事が起きたあとも、決めた時間まで観察が続く
- **読み取りだけ**: `FakeRunner` の記録に `mutating=True`、`docker run`/`stop`/`rm`、`push`
  が 1 つもない
- `samples.jsonl` に、観察のたびに 1 行が足される (中断でも、そこまでが残り、要約が書かれる)
- 片方の台に入れない回があっても、見張りは続く

実物の ssh、docker、推論サーバーには、どの段でもつながない (`FakeRunner` と、実際に HTTP で
叩く `fake_vllm.FakeVllm` を相手にする)。試験は、実際に眠らない (`WatchClock` が、眠りを記録
するだけで、実際の時間を進めない)。
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import HttpUrl

from fake_runner import FakeRunner, Reply, Rule
from fake_vllm import FakeVllm, Fault, MetricsSample
from serving_kit import watch as w
from serving_kit.config import ConfigError
from serving_kit.logs import var_dir
from serving_kit.plan import OWNER_FILTER, build_plans
from serving_kit.types import (
    ApprovedPlan,
    CommandResult,
    ConfigDef,
    ImageRef,
    NodeDef,
    NodeRole,
    Setting,
)
from watch_script import (
    cpufreq_candidates,
    discovery_calls,
    discovery_calls_for_node,
    discovery_rule,
    hwmon_input_candidates,
    hwmon_name_candidates,
    label_candidates,
    thermal_candidates,
    unreadable_lines,
)

# --- 見本の値 ---------------------------------------------------------------

DIGEST = "b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
IMAGE_REF = f"vllm/vllm-openai@sha256:{DIGEST}"
REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"
CONFIG_NAME = "watch-test"
SERVED_MODEL = "glm-5-3-flash"

SOURCE = HttpUrl("https://docs.vllm.ai/en/latest/cli/serve/")
QUOTE = "vllm serve [model_tag] [options]"

STARTED_WALL = datetime(2026, 9, 22, 3, 0, 0, tzinfo=UTC)
"""試験の壁の時計の出発点 (`WatchClock` の既定)。"""

CONTAINER_IDS: dict[NodeRole, str] = {"head": "0123456789ab", "worker": "cdef01234567"}

HEAD = NodeDef(
    role="head", ssh_host="spark-153d", lan_addr=IPv4Address("127.0.0.1"), remote_root=REMOTE_ROOT
)
WORKER = NodeDef(
    role="worker", ssh_host="spark-5083", lan_addr=IPv4Address("127.0.0.1"), remote_root=REMOTE_ROOT
)
NODES: dict[NodeRole, NodeDef] = {"head": HEAD, "worker": WORKER}

IMAGE = ImageRef(
    ref=IMAGE_REF,
    seen_as="vllm/vllm-openai:glm53-flash-arm64-cu130",
    size_bytes=9666567584,
    source=HttpUrl("https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags"),
    quote="glm53-flash-arm64-cu130 / linux/arm64 / 9666567584 B",
)

OWN_CONTAINERS_ARGV = (
    "docker",
    "ps",
    "-a",
    "--filter",
    f"label={OWNER_FILTER}",
    "--format",
    "json",
)
"""`guards.list_own_containers` が流す、ただ 1 つのコンテナの一覧の読み取り。"""


def _setting(
    flag: str | None = None, value: str | None = None, *, is_port: bool = False
) -> Setting:
    """根拠の付いた設定を 1 つ作る (根拠の中身は、この試験では問わない)。"""
    return Setting(
        flag=flag, value=value, why="試験のための設定", is_port=is_port, source=SOURCE, quote=QUOTE
    )


def config_for(port: int) -> ConfigDef:
    """見張る対象の、最小の `serve` の構成。"""
    return ConfigDef(
        name=CONFIG_NAME,
        kind="serve",
        description="watch の試験用の構成",
        nodes=("head", "worker"),
        image=IMAGE,
        weights=None,
        docker={},
        args={"port": _setting("--port", str(port), is_port=True)},
        env={},
        ready_timeout_s=1800,
        served_model_name=SERVED_MODEL,
    )


def port_of(fake: FakeVllm) -> int:
    """偽の推論サーバーが待ち受けている番号。"""
    return int(fake.base_url.rsplit(":", 1)[1])


@dataclass
class WatchClock:
    """試験用の時計。眠りは記録するだけで、実際には眠らない。壁の時計と単調な時計は、同じ
    経過だけ進む (`watch.watch` が、両方を同じ歩幅で進める前提を確かめられるようにする)。

    `interrupt_after` を指定すると、その回数だけ眠ったところで `KeyboardInterrupt` を投げる
    (中断でも、そこまでの要約が書かれることを確かめる試験で使う)。
    """

    mono: float = 0.0
    wall: datetime = field(default_factory=lambda: STARTED_WALL)
    slept: list[float] = field(default_factory=list)
    interrupt_after: int | None = None

    def monotonic(self) -> float:
        return self.mono

    def now(self) -> datetime:
        return self.wall

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        if self.interrupt_after is not None and len(self.slept) >= self.interrupt_after:
            raise KeyboardInterrupt
        self.mono += seconds
        self.wall += timedelta(seconds=seconds)


@dataclass
class _CostlyRunner:
    """`FakeRunner` を包み、`run` が呼ばれるたびに、渡した時計を進める (1 回の観察に、実際に
    経過がかかることを作る試験専用。眠り (`sleep`) が、間隔から、その経過を引いた値を受ける
    ことを確かめるのに使う)。
    """

    inner: FakeRunner
    clock: WatchClock
    cost_s: float

    def approve(self, plan: ApprovedPlan) -> None:
        self.inner.approve(plan)

    def run(
        self, node: NodeDef, argv: Sequence[str], *, timeout_s: float, mutating: bool
    ) -> CommandResult:
        result = self.inner.run(node, argv, timeout_s=timeout_s, mutating=mutating)
        self.clock.mono += self.cost_s
        return result

    def push(
        self, node: NodeDef, local_dir: Path, remote_subdir: str, *, delete: bool
    ) -> CommandResult:
        return self.inner.push(node, local_dir, remote_subdir, delete=delete)

    def pull(self, node: NodeDef, remote_path: str, local_dir: Path) -> CommandResult:
        return self.inner.pull(node, remote_path, local_dir)


class _InterruptingTransport(httpx.BaseTransport):
    """`fail_at` 回目の要求で `KeyboardInterrupt` を投げ、それより前は実物の `HTTPTransport` に
    渡す (HTTP の読み取りの最中に中断が来ても、そこまでの要約が書かれることを確かめる試験用)。
    """

    def __init__(self, *, fail_at: int) -> None:
        self._inner = httpx.HTTPTransport()
        self._count = 0
        self._fail_at = fail_at

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._count += 1
        if self._count >= self._fail_at:
            raise KeyboardInterrupt
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


def _ps_line(role: NodeRole, container_name: str, labels: dict[str, str]) -> str:
    """`docker ps -a --filter label=… --format json` の 1 行 (実機から採った見本ではない)。"""
    row = {
        "ID": CONTAINER_IDS[role],
        "Names": container_name,
        "State": "running",
        "Image": IMAGE_REF,
        "Labels": ",".join(f"{key}={value}" for key, value in sorted(labels.items())),
    }
    return json.dumps(row) + "\n"


# --- 固まった --------------------------------------------------------------


def test_stalled_event_has_a_timestamp_when_gpu_utilization_is_available(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    fake_vllm.set_metrics(MetricsSample(running_requests=2, generation_tokens_total=100))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="65, 1781, 28.90, 97\n"),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=5.0,
        interval_s=1.0,
        stall_window_s=3.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.gpu_utilization_available is True
    assert [event.finding for event in outcome.events] == ["stalled"]
    assert outcome.events[0].at_utc == STARTED_WALL + timedelta(seconds=3)
    assert outcome.events[0].context  # 前後の観察が入っている
    assert outcome.sample_count == 6


def test_no_stalled_event_when_no_requests_are_running(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=50))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="65, 1781, 28.90, 97\n"),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=5.0,
        interval_s=1.0,
        stall_window_s=2.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.events == ()


def test_gpu_unavailable_falls_back_to_token_only_judgement(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    fake_vllm.set_metrics(MetricsSample(running_requests=1, generation_tokens_total=50))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    # GB10 が対応していないときに出る文字列 (int に変換できない) を、4 列とも常に返す
    not_supported = "[Not Supported], [Not Supported], [Not Supported], [Not Supported]\n"
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout=not_supported),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=3.0,
        interval_s=1.0,
        stall_window_s=2.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.gpu_utilization_available is False
    assert "生成のトークンの数だけ" in outcome.detail
    assert [event.finding for event in outcome.events] == ["stalled"]


# --- GPU の 4 列の読み取り ----------------------------------------------------


def test_gpu_columns_are_recorded_from_one_nvidia_smi_query(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """温度・SM クロック・電力・使用率の 4 列が、1 回の `nvidia-smi` の問い合わせから読める。
    台ごとに、観察 1 回あたり `nvidia-smi` はちょうど 1 回だけ流れる。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="65, 1781, 28.90, 97\n"),))],
        default=Reply(exit_code=1),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    sample = json.loads(outcome.samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert sample["gpu_temperature_c"]["head"] == 65
    assert sample["gpu_sm_clock_mhz"]["head"] == 1781
    assert sample["gpu_power_w"]["head"] == 28.9
    assert sample["gpu_utilization_pct"]["head"] == 97

    per_node: dict[str, int] = {"head": 0, "worker": 0}
    for call in runner.calls:
        if call.argv and call.argv[0] == "nvidia-smi":
            per_node[call.node] += 1
    assert per_node == {"head": outcome.sample_count, "worker": outcome.sample_count}


def test_gpu_columns_that_cannot_be_read_are_empty_one_by_one(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """読めない列だけが空になり、列を 1 つしか持たない行 (旧い形の応答) は 4 列とも空になる。
    1 台が旧形式で 1 度も読めなくても、見張りは止まらず観察を続ける。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[
            Rule(
                prefix=w.GPU_UTIL_ARGV,
                node="head",
                replies=(Reply(stdout="[N/A], 1781, [Not Supported], 97\n"),),
            ),
            Rule(prefix=w.GPU_UTIL_ARGV, node="worker", replies=(Reply(stdout="97 %\n"),)),
        ],
        default=Reply(exit_code=1),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    sample = json.loads(outcome.samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert sample["gpu_temperature_c"]["head"] is None
    assert sample["gpu_sm_clock_mhz"]["head"] == 1781
    assert sample["gpu_power_w"]["head"] is None
    assert sample["gpu_utilization_pct"]["head"] == 97
    for column in ("gpu_temperature_c", "gpu_sm_clock_mhz", "gpu_power_w", "gpu_utilization_pct"):
        assert sample[column]["worker"] is None
    # worker は旧形式しか返さないので、一度も読めないまま終わる (決めごとの 3: 台ごとに
    # 一度でも読めたかどうかで判定するので、worker が読めない限り全体は False のまま)
    assert outcome.gpu_utilization_available is False
    assert outcome.sample_count == 2


# --- 応答しない --------------------------------------------------------------


def test_no_unresponsive_event_after_only_two_consecutive_failures(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    fake_vllm.set_health_fault(Fault(status=None))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(var_root=tmp_path, default=Reply())

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.sample_count == 2
    assert outcome.events == ()


def test_unresponsive_event_appears_on_the_third_consecutive_failure(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    fake_vllm.set_health_fault(Fault(status=None))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(var_root=tmp_path, default=Reply())

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=2.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.sample_count == 3
    assert [event.finding for event in outcome.events] == ["unresponsive"]
    assert outcome.events[0].at_utc == STARTED_WALL + timedelta(seconds=2)


# --- 回収と継続 --------------------------------------------------------------


def test_logs_from_both_nodes_are_collected_once_when_an_event_fires(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    fake_vllm.set_health_fault(Fault(status=None))
    config = config_for(port_of(fake_vllm))
    plans = {plan.node: plan for plan in build_plans(config, NODES, STARTED_WALL)}
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[
            Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="60, 1700, 25.00, 10\n"),)),
            Rule(
                prefix=OWN_CONTAINERS_ARGV,
                node="head",
                replies=(
                    Reply(
                        stdout=_ps_line("head", plans["head"].container_name, plans["head"].labels)
                    ),
                ),
            ),
            Rule(
                prefix=OWN_CONTAINERS_ARGV,
                node="worker",
                replies=(
                    Reply(
                        stdout=_ps_line(
                            "worker", plans["worker"].container_name, plans["worker"].labels
                        )
                    ),
                ),
            ),
            Rule(
                prefix=("docker", "logs", "--timestamps", CONTAINER_IDS["head"]),
                node="head",
                replies=(Reply(stdout="head log\n"),),
            ),
            Rule(
                prefix=("docker", "logs", "--timestamps", CONTAINER_IDS["worker"]),
                node="worker",
                replies=(Reply(stdout="worker log\n"),),
            ),
        ],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=4.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    # 応答しないままなので、続いているあいだ (5 回の観察) は 1 件だけ
    assert [event.finding for event in outcome.events] == ["unresponsive"]
    assert "記録は" in outcome.events[0].detail

    collected = [p for p in tmp_path.iterdir() if p != outcome.samples_path.parent]
    assert len(collected) == 1
    assert (collected[0] / "head" / "container.stdout.log").read_text(
        encoding="utf-8"
    ) == "head log\n"
    assert (collected[0] / "worker" / "container.stdout.log").read_text(
        encoding="utf-8"
    ) == "worker log\n"

    # 一覧と記録の読み取りは、出来事が起きた 1 回ぶんだけ (続いていても増えない)
    ps_calls = [call for call in runner.calls if call.argv[:2] == ("docker", "ps")]
    assert len(ps_calls) == 2  # head と worker で 1 回ずつ


def test_watch_keeps_observing_until_the_duration_after_an_event(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    fake_vllm.set_health_fault(Fault(status=None))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(var_root=tmp_path, default=Reply())

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=5.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    # 3 回目で「応答しない」が出たあとも、決めた時間 (6 回ぶん) まで観察が続く
    assert outcome.sample_count == 6
    assert [event.finding for event in outcome.events] == ["unresponsive"]


def test_continues_when_one_node_cannot_be_reached_for_a_gpu_reading(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=10))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[
            Rule(
                prefix=w.GPU_UTIL_ARGV,
                node="head",
                replies=(Reply(stdout="60, 1700, 25.00, 10\n"),),
            ),
            Rule(
                prefix=w.GPU_UTIL_ARGV,
                node="worker",
                replies=(
                    Reply(exit_code=255, stderr="ssh: no route to host"),
                    Reply(stdout="60, 1700, 25.00, 10\n"),
                ),
            ),
        ],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    lines = outcome.samples_path.read_text(encoding="utf-8").splitlines()
    first, second = (json.loads(line) for line in lines)
    assert first["gpu_utilization_pct"]["worker"] is None
    assert second["gpu_utilization_pct"]["worker"] == 10
    assert outcome.gpu_utilization_available is True  # 少なくとも 1 回は読めた


# --- 読み取りだけであること、記録の積み方 -------------------------------------


def test_watch_never_issues_a_mutating_or_stop_start_push_call(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """読み取りだけであること (熱区域・hwmon・CPU の発見と観察を含めても)。流れる
    `argv[0]` は `cat` と `nvidia-smi` だけで、`push`/`pull` は 1 つも出ない。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=1, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="60, 1700, 25.00, 10\n"),))],
        default=Reply(),
    )

    w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=3.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert runner.pushes == ()
    assert runner.pulls == ()
    assert {call.argv[0] for call in runner.calls if call.argv} <= {"cat", "nvidia-smi"}
    for call in runner.calls:
        assert call.mutating is False
        if call.argv and call.argv[0] == "docker":
            assert call.argv[1] not in {"run", "stop", "rm"}


def test_samples_file_gets_one_line_per_observation(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    fake_vllm.set_metrics(MetricsSample(running_requests=1, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="60, 1700, 25.00, 10\n"),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=3.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.sample_count == 4
    lines = outcome.samples_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    for line in lines:
        payload = json.loads(line)
        assert "vllm:" not in json.dumps(payload)  # 生の応答の本文を書いていない


def test_partial_summary_is_written_before_the_interrupt_is_raised(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    fake_vllm.set_metrics(MetricsSample(running_requests=1, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock(interrupt_after=2)
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="60, 1700, 25.00, 10\n"),))],
        default=Reply(),
    )

    with pytest.raises(KeyboardInterrupt):
        w.watch(
            runner,
            config,
            NODES,
            var_root=tmp_path,
            duration_s=100.0,
            interval_s=1.0,
            sleep=clock.sleep,
            clock=clock.monotonic,
            now=clock.now,
            report=io.StringIO(),
        )

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    result = json.loads((run_dirs[0] / w.RESULT_FILE_NAME).read_text(encoding="utf-8"))
    assert result["sample_count"] == 2
    lines = (run_dirs[0] / w.SAMPLES_FILE_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


# --- レビューの指摘 1: GPU のしきい値の項が、誤検知を防いでいること -------------------


def test_no_stalled_event_when_gpu_is_readable_but_below_threshold(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """処理中の要求があり、トークンが増えなくても、GPU の使用率が読めて、しきい値
    (既定 90%) を下回っていれば「固まった」にしない (GPU のしきい値の項を丸ごと外す変異が
    あれば、この試験は落ちる)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=2, generation_tokens_total=100))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="60, 1700, 25.00, 50\n"),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=5.0,
        interval_s=1.0,
        stall_window_s=3.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.gpu_utilization_available is True
    assert outcome.events == ()


# --- レビューの指摘 2: 連続の失敗の数が、成功で数え直されること -----------------------


def test_unresponsive_only_fires_once_after_a_success_resets_the_streak(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """`失敗,失敗,成功,失敗,失敗,失敗` の系列では、合計の失敗 (5 回) ではなく、直近の
    連続 (3 回) で判定するので、最後の回にだけ「応答しない」が出る (連続の数を成功で 0 に
    戻さない変異があれば、この試験は落ちる)。"""
    fake_vllm.set_health_fault_sequence(
        [
            Fault(status=None),
            Fault(status=None),
            Fault(),
            Fault(status=None),
            Fault(status=None),
            Fault(status=None),
        ]
    )
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(var_root=tmp_path, default=Reply())

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=5.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.sample_count == 6
    assert [event.finding for event in outcome.events] == ["unresponsive"]
    assert outcome.events[0].at_utc == STARTED_WALL + timedelta(seconds=5)


# --- レビューの指摘 3: GPU の判定は、窓の全体を見ること -------------------------------


def test_no_stalled_event_when_gpu_dips_once_within_the_window(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """トークンの数が増えない、同じ窓の中で、GPU の使用率が 1 度でもしきい値を割れば、
    窓の最後がしきい値以上に戻っていても「固まった」にしない (各回の値だけで見る変異が
    あれば、この試験は落ちる: 95 → 95 → 10 → 95 の最後の回だけを見ると、誤って出てしまう)。
    """
    fake_vllm.set_metrics(MetricsSample(running_requests=1, generation_tokens_total=100))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    gpu_sequence = (
        Reply(stdout="70, 1781, 30.00, 95\n"),
        Reply(stdout="70, 1781, 30.00, 95\n"),
        Reply(stdout="60, 1700, 25.00, 10\n"),
        Reply(stdout="70, 1781, 30.00, 95\n"),
    )
    runner = FakeRunner(
        var_root=tmp_path,
        script=[
            Rule(prefix=w.GPU_UTIL_ARGV, node="head", replies=gpu_sequence),
            Rule(prefix=w.GPU_UTIL_ARGV, node="worker", replies=gpu_sequence),
        ],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=3.0,
        interval_s=1.0,
        stall_window_s=2.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.sample_count == 4
    assert outcome.events == ()


def test_stalled_event_still_fires_when_gpu_stays_above_the_threshold_throughout(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """95 → 95 → 95 → 95 のように、窓のあいだ、ずっとしきい値以上なら「固まった」が出る
    (指摘 3 の直し方が、通るべき場合まで塞いでいないことを確かめる)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=1, generation_tokens_total=100))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="70, 1781, 30.00, 95\n"),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=3.0,
        interval_s=1.0,
        stall_window_s=2.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert [event.finding for event in outcome.events] == ["stalled"]


def test_stall_window_is_recounted_when_tokens_increase_mid_window(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """窓の途中でトークンの数が増えたら、窓を、増えた時刻から測り直す (増えた時刻を基準に
    し直さない変異があれば、この試験は落ちる)。"""
    fake_vllm.set_metrics_sequence(
        [
            MetricsSample(running_requests=1, generation_tokens_total=100),
            MetricsSample(running_requests=1, generation_tokens_total=150),
            MetricsSample(running_requests=1, generation_tokens_total=150),
            MetricsSample(running_requests=1, generation_tokens_total=150),
        ]
    )
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="70, 1781, 30.00, 95\n"),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=3.0,
        interval_s=1.0,
        stall_window_s=2.5,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    # トークンは t=1 で増えたので、窓はそこから測り直す。duration が尽きる t=3 までの
    # 経過は 2 秒で、stall_window_s (2.5 秒) に届かない
    assert outcome.events == ()


def test_stall_window_continues_through_an_unreadable_metrics_round(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """`/metrics` が 1 回だけ読めなくても (前後の値が同じなら)、窓を測り直さない。GPU の
    履歴も、その回ぶんを含めて窓に積む。"""
    fake_vllm.set_metrics_sequence(
        [
            MetricsSample(running_requests=1, generation_tokens_total=100),
            MetricsSample(running_requests=1, generation_tokens_total=100),
            MetricsSample(running_requests=1, generation_tokens_total=100),
        ]
    )
    fake_vllm.set_metrics_fault_sequence([Fault(), Fault(status=503), Fault()])
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="70, 1781, 30.00, 95\n"),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=2.0,
        interval_s=1.0,
        stall_window_s=2.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert [event.finding for event in outcome.events] == ["stalled"]


def test_stall_window_resets_and_the_restart_is_noted_when_the_counter_decreases(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """生成のトークンの数が減ったら (起こし直されたとみられる)、窓を数え直し、GPU の履歴も
    取り直す。起こし直したこと自体は要約に残る。"""
    fake_vllm.set_metrics_sequence(
        [
            MetricsSample(running_requests=1, generation_tokens_total=500),
            MetricsSample(running_requests=1, generation_tokens_total=10),
            MetricsSample(running_requests=1, generation_tokens_total=10),
        ]
    )
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="70, 1781, 30.00, 95\n"),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=2.0,
        interval_s=1.0,
        stall_window_s=2.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    # 起こし直しで窓が t=1 から測り直され、duration (t=2) までに 2 秒に届かない
    assert outcome.events == ()
    assert "起こし直された" in outcome.detail


# --- レビューの指摘 5: 中断が、眠り以外の最中に来ても、要約が書かれること -------------------


def test_interrupt_during_an_http_read_still_writes_a_partial_summary(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    fake_vllm.set_metrics(MetricsSample(running_requests=1, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="60, 1700, 25.00, 10\n"),))],
        default=Reply(),
    )
    # 1 回目の観察 (/health, /metrics で 2 回) は通し、2 回目の /health (3 回目の要求) で
    # 中断を投げる
    transport = _InterruptingTransport(fail_at=3)
    client = httpx.Client(transport=transport, trust_env=False)
    try:
        with pytest.raises(KeyboardInterrupt):
            w.watch(
                runner,
                config,
                NODES,
                var_root=tmp_path,
                duration_s=100.0,
                interval_s=1.0,
                sleep=clock.sleep,
                clock=clock.monotonic,
                now=clock.now,
                report=io.StringIO(),
                client=client,
            )
    finally:
        client.close()

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    result = json.loads((run_dirs[0] / w.RESULT_FILE_NAME).read_text(encoding="utf-8"))
    assert result["sample_count"] == 1


def test_interrupt_during_a_gpu_reading_still_writes_a_partial_summary(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    fake_vllm.set_metrics(MetricsSample(running_requests=1, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[
            Rule(
                prefix=w.GPU_UTIL_ARGV,
                node="head",
                replies=(Reply(stdout="60, 1700, 25.00, 10\n"),),
            ),
            Rule(
                prefix=w.GPU_UTIL_ARGV,
                node="worker",
                replies=(Reply(stdout="60, 1700, 25.00, 10\n"), Reply(raises=KeyboardInterrupt())),
            ),
        ],
        default=Reply(),
    )

    with pytest.raises(KeyboardInterrupt):
        w.watch(
            runner,
            config,
            NODES,
            var_root=tmp_path,
            duration_s=100.0,
            interval_s=1.0,
            sleep=clock.sleep,
            clock=clock.monotonic,
            now=clock.now,
            report=io.StringIO(),
        )

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    result = json.loads((run_dirs[0] / w.RESULT_FILE_NAME).read_text(encoding="utf-8"))
    assert result["sample_count"] == 1


def test_interrupt_during_log_collection_still_writes_a_partial_summary(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    fake_vllm.set_health_fault(Fault(status=None))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[
            Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="60, 1700, 25.00, 10\n"),)),
            Rule(prefix=OWN_CONTAINERS_ARGV, replies=(Reply(raises=KeyboardInterrupt()),)),
        ],
        default=Reply(),
    )

    with pytest.raises(KeyboardInterrupt):
        w.watch(
            runner,
            config,
            NODES,
            var_root=tmp_path,
            duration_s=100.0,
            interval_s=1.0,
            sleep=clock.sleep,
            clock=clock.monotonic,
            now=clock.now,
            report=io.StringIO(),
        )

    # 中断が回収の最中に来たので、確定していた出来事はなく (append の前)、観察の 3 件だけが残る
    run_dir = var_dir(tmp_path, STARTED_WALL, w.COMMAND_NAME, CONFIG_NAME)
    result = json.loads((run_dir / w.RESULT_FILE_NAME).read_text(encoding="utf-8"))
    assert result["sample_count"] == 3
    assert result["events"] == []


# --- レビューの指摘 6: 触る前に断る、壊れた応答でも落ちない --------------------------------


def test_kind_must_be_serve_and_nothing_is_touched_first(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    config = config_for(port_of(fake_vllm)).model_copy(update={"kind": "probe"})
    runner = FakeRunner(var_root=tmp_path, default=Reply())

    with pytest.raises(ConfigError):
        w.watch(runner, config, NODES, var_root=tmp_path, duration_s=1.0, interval_s=1.0)

    assert runner.calls == ()


def test_port_must_be_decidable_and_nothing_is_touched_first(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    config = config_for(port_of(fake_vllm)).model_copy(update={"args": {}})
    runner = FakeRunner(var_root=tmp_path, default=Reply())

    with pytest.raises(ConfigError):
        w.watch(runner, config, NODES, var_root=tmp_path, duration_s=1.0, interval_s=1.0)

    assert runner.calls == ()


def test_duration_must_be_positive_and_nothing_is_touched_first(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    config = config_for(port_of(fake_vllm))
    runner = FakeRunner(var_root=tmp_path, default=Reply())

    with pytest.raises(ValueError):
        w.watch(runner, config, NODES, var_root=tmp_path, duration_s=0.0, interval_s=1.0)

    assert runner.calls == ()


def test_corrupted_metrics_body_does_not_crash(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    fake_vllm.set_metrics_fault(Fault(body="not a prometheus body at all"))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="60, 1700, 25.00, 10\n"),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    sample = json.loads(outcome.samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert sample["health_ok"] is True
    assert sample["generation_tokens_total"] is None
    assert sample["running_requests"] is None
    assert sample["waiting_requests"] is None


def test_metrics_failure_alongside_a_healthy_check_does_not_crash(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    fake_vllm.set_metrics_fault(Fault(status=503))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="60, 1700, 25.00, 10\n"),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    sample = json.loads(outcome.samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert sample["health_ok"] is True
    assert sample["generation_tokens_total"] is None
    assert outcome.events == ()


def test_unresponsive_fires_from_the_start_when_the_server_never_answers(
    tmp_path: Path, fake_vllm_factory: Callable[[], FakeVllm]
) -> None:
    fake = fake_vllm_factory()
    port = port_of(fake)
    fake.stop()  # 二度とつながらない状態にする
    config = config_for(port)
    clock = WatchClock()
    runner = FakeRunner(var_root=tmp_path, default=Reply())

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=5.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.sample_count == 6
    assert [event.finding for event in outcome.events] == ["unresponsive"]
    assert outcome.events[0].at_utc == STARTED_WALL + timedelta(seconds=2)


# --- レビューの指摘 8: 前後の観察が、起きた回を含む直近 3 回であること ------------------


def test_event_context_is_the_triggering_sample_and_two_before_it(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    fake_vllm.set_health_fault(Fault(status=None))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(var_root=tmp_path, default=Reply())

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=2.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert len(outcome.events) == 1
    context = outcome.events[0].context
    assert len(context) == 3
    assert context[-1].taken_at_utc == outcome.events[0].at_utc
    timestamps = [sample.taken_at_utc for sample in context]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == STARTED_WALL
    assert timestamps[-1] == STARTED_WALL + timedelta(seconds=2)


# --- 再レビューの指摘 1・2: 退化した数の引数を、触る前に断ること -----------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"interval_s": 0.0},
        {"interval_s": -1.0},
        {"interval_s": float("nan")},
        {"interval_s": float("inf")},
        {"duration_s": 0.0},
        {"duration_s": float("nan")},
        {"duration_s": float("inf")},
        {"stall_window_s": 0.0},
        {"stall_window_s": -5.0},
        {"stall_window_s": float("nan")},
        {"stall_window_s": float("inf")},
        {"unresponsive_threshold": 0},
        {"unresponsive_threshold": -1},
        {"unresponsive_threshold": float("nan")},
        {"unresponsive_threshold": float("inf")},
        {"unresponsive_threshold": 1.5},
        {"unresponsive_threshold": True},
        {"stall_gpu_threshold_pct": -1},
        {"stall_gpu_threshold_pct": 101},
        {"health_timeout_s": 0.0},
        {"health_timeout_s": float("nan")},
        {"gpu_timeout_s": 0.0},
        {"gpu_timeout_s": float("nan")},
        {"log_timeout_s": 0.0},
        {"log_timeout_s": float("nan")},
        {"interval_s": 10.0, "duration_s": 1.0},  # interval_s が duration_s を超える
        {"thermal_threshold_c": 0.0},
        {"thermal_threshold_c": -1.0},
        {"thermal_threshold_c": float("nan")},
        {"thermal_threshold_c": float("inf")},
    ],
    ids=lambda v: ",".join(f"{k}={val}" for k, val in v.items()),
)
def test_degenerate_numeric_arguments_are_rejected_before_touching_anything(
    tmp_path: Path, fake_vllm: FakeVllm, overrides: dict[str, float | int]
) -> None:
    """退化した数の引数 (0、負、NaN、無限大、しきい値の範囲外、`interval_s` と `duration_s`
    の逆転) は、どれも、Spark にも推論サーバーにも触る前に `ValueError` で断る (指摘 1・2)。
    `interval_s <= 0` を断らないと、眠らずに回り続け、`stall_window_s <= 0` /
    `unresponsive_threshold <= 0` を断らないと、健全な状態でもすぐに出来事を誤って出す。
    `gpu_timeout_s` と `interval_s` の大小関係は、意図して断らない (決めごとの 7)。
    """
    config = config_for(port_of(fake_vllm))
    runner = FakeRunner(var_root=tmp_path, default=Reply())
    clock = WatchClock()
    # 断らずに素通りしてしまう変異が、実際の時計で長々と眠り続けないよう、ここでも
    # 差し替えた時計を渡す (退化した値が、断られずに素通りしても、試験が長引かない)
    kwargs: dict[str, Any] = {
        "duration_s": 5.0,
        "interval_s": 1.0,
        "sleep": clock.sleep,
        "clock": clock.monotonic,
        "now": clock.now,
        "report": io.StringIO(),
        **overrides,
    }

    with pytest.raises(ValueError):
        w.watch(runner, config, NODES, var_root=tmp_path, **kwargs)

    assert runner.calls == ()
    assert fake_vllm.requests == ()


def test_sleep_receives_the_interval_minus_the_time_spent_observing(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """眠りは、間隔をそのまま渡すのではなく、間隔から、観察にかかった経過を引いた値を渡す
    (負にはならない。指摘 1・2 の「あわせて」)。

    発見 (熱区域・hwmon) は、すべて `exit_code=1` にして空にする (発見の所要は `start_mono`
    より前に消費されるので、間隔の計算に混ざらない)。空の発見のもとでは、観察 1 回あたりの
    `run` は、台ごとに `nvidia-smi` と `/proc/stat` の 2 回、2 台で 4 回になる。
    """
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    base_runner = FakeRunner(var_root=tmp_path, default=Reply(exit_code=1))
    # 観察 1 回 (2 台ぶん、nvidia-smi と /proc/stat で 4 回の run) に、0.1 秒ずつ経過が積まれる
    runner = _CostlyRunner(inner=base_runner, clock=clock, cost_s=0.1)

    w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=2.5,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    # 1 回の観察に 0.4 秒 (4 回ぶん) かかるので、眠りは 1.0 - 0.4 = 0.6 秒になる
    assert clock.slept
    for value in clock.slept:
        assert value > 0
    assert clock.slept[0] == pytest.approx(0.6)


def test_sleep_is_skipped_without_a_negative_value_when_observing_overruns_the_interval(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """観察が間隔より長くかかっても、眠りに負の値を渡さない (眠りそのものを呼ばない)。

    発見をすべて空にした上で (`test_sleep_receives_the_interval_minus_the_time_spent_observing`
    と同じ考え方)、観察 1 回の所要 (4 回の run ぶん) が間隔 (1.0 秒) を超えるようにする。
    """
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    base_runner = FakeRunner(var_root=tmp_path, default=Reply(exit_code=1))
    # 観察 1 回 (4 回の run) に 1.2 秒 (0.3 秒 × 4) かかり、間隔 (1.0 秒) を超える
    runner = _CostlyRunner(inner=base_runner, clock=clock, cost_s=0.3)

    w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=3.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert clock.slept == []


# --- 再レビューの指摘 3: GPU の窓の履歴を、トークンが増えたときに数え直すこと -----------


def test_stalled_event_fires_in_the_new_window_even_after_a_gpu_dip_in_the_old_one(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """窓の途中で GPU が 1 度だけ低くても (古い窓は、そのために固まったと言わない)、
    トークンが増えて窓が数え直されたあとは、その新しい窓だけを見て、GPU が高いまま
    トークンが止まれば「固まった」が正しく出る (`window_gpu_status` を、増えた回に
    数え直さない変異があれば、古い窓の低い値が残り続けて、この試験は落ちる)。
    """
    fake_vllm.set_metrics_sequence(
        [
            MetricsSample(running_requests=1, generation_tokens_total=100),  # t=0 窓 1 開始
            MetricsSample(running_requests=1, generation_tokens_total=100),  # t=1
            MetricsSample(running_requests=1, generation_tokens_total=100),  # t=2
            MetricsSample(running_requests=1, generation_tokens_total=150),  # t=3 窓 2 開始
            MetricsSample(running_requests=1, generation_tokens_total=150),  # t=4
            MetricsSample(running_requests=1, generation_tokens_total=150),  # t=5 ここで出る
        ]
    )
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    # 窓 1 (t=0..2) の途中 (t=1) で 1 度だけ低い値、窓 2 (t=3..5) はずっと高い値
    gpu_sequence = (
        Reply(stdout="70, 1781, 30.00, 95\n"),
        Reply(stdout="60, 1700, 25.00, 10\n"),
        Reply(stdout="70, 1781, 30.00, 95\n"),
        Reply(stdout="70, 1781, 30.00, 95\n"),
        Reply(stdout="70, 1781, 30.00, 95\n"),
        Reply(stdout="70, 1781, 30.00, 95\n"),
    )
    runner = FakeRunner(
        var_root=tmp_path,
        script=[
            Rule(prefix=w.GPU_UTIL_ARGV, node="head", replies=gpu_sequence),
            Rule(prefix=w.GPU_UTIL_ARGV, node="worker", replies=gpu_sequence),
        ],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=5.0,
        interval_s=1.0,
        stall_window_s=2.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.sample_count == 6
    assert [event.finding for event in outcome.events] == ["stalled"]
    assert outcome.events[0].at_utc == STARTED_WALL + timedelta(seconds=5)


# --- 再レビューの指摘 4: 「処理中の要求があった」も、窓の全体を見ること ------------------


def test_no_stalled_event_when_requests_drop_to_zero_once_within_the_window(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """トークンの数が変わらない同じ窓の中で、処理中の要求が 1 度でも 0 になれば (要求が
    なくなっただけで、正常)、窓の最後で要求が戻っていても「固まった」にしない (判定の回
    だけで見る変異があれば、この試験は落ちる)。
    """
    fake_vllm.set_metrics_sequence(
        [
            MetricsSample(running_requests=1, generation_tokens_total=100),
            MetricsSample(running_requests=0, generation_tokens_total=100),
            MetricsSample(running_requests=1, generation_tokens_total=100),
            MetricsSample(running_requests=1, generation_tokens_total=100),
            MetricsSample(running_requests=1, generation_tokens_total=100),
            MetricsSample(running_requests=1, generation_tokens_total=100),
        ]
    )
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="70, 1781, 30.00, 95\n"),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=5.0,
        interval_s=1.0,
        stall_window_s=2.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.events == ()


def test_stalled_event_fires_in_the_new_window_after_a_zero_request_dip_in_the_old_one(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """処理中の要求が 1 度 0 になった窓のあとで、トークンが増えて窓が数え直されれば、その
    新しい窓で要求がずっと 1 以上のまま止まれば「固まった」が正しく出る (要求の履歴を、
    増えた回に数え直さない変異があれば、この試験は落ちる)。
    """
    fake_vllm.set_metrics_sequence(
        [
            MetricsSample(running_requests=1, generation_tokens_total=100),  # t=0 窓 1 開始
            MetricsSample(running_requests=0, generation_tokens_total=100),  # t=1 要求 0
            MetricsSample(running_requests=1, generation_tokens_total=100),  # t=2
            MetricsSample(running_requests=1, generation_tokens_total=150),  # t=3 窓 2 開始
            MetricsSample(running_requests=1, generation_tokens_total=150),  # t=4
            MetricsSample(running_requests=1, generation_tokens_total=150),  # t=5 ここで出る
        ]
    )
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="70, 1781, 30.00, 95\n"),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=5.0,
        interval_s=1.0,
        stall_window_s=2.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.sample_count == 6
    assert [event.finding for event in outcome.events] == ["stalled"]
    assert outcome.events[0].at_utc == STARTED_WALL + timedelta(seconds=5)


def test_no_stalled_event_when_requests_are_never_readable_within_the_window(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """窓の中で、処理中の要求を 1 度も読めなければ (`/metrics` がずっと読めない)、確かめ
    られていないとして「固まった」にしない (GPU と違い、要求は主たる条件なので、読めない
    ときに素通りさせない)。"""
    fake_vllm.set_metrics_fault(Fault(status=503))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="70, 1781, 30.00, 95\n"),))],
        default=Reply(),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=3.0,
        interval_s=1.0,
        stall_window_s=2.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.events == ()


# --- 熱区域と hwmon の発見 ----------------------------------------------------


def test_discovery_runs_once_at_start_and_stops_at_the_limit(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """熱区域と hwmon のパスは、見張りの開始時に 1 回だけ、種類ごとに 1 回の `cat` に
    まとめて番号順に見つける。名前が `mlx5`/`nvme`/`acpitz` でない hwmon は採用しない。
    発見した一覧は `result.json` に出て、観察を重ねても発見をやり直さない。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    zone0 = f"{w.THERMAL_ZONE_DIR}thermal_zone0/temp"
    zone4 = f"{w.THERMAL_ZONE_DIR}thermal_zone4/temp"
    hwmon2_name = f"{w.HWMON_DIR}hwmon2/name"
    hwmon5_name = f"{w.HWMON_DIR}hwmon5/name"
    hwmon2_input = f"{w.HWMON_DIR}hwmon2/temp1_input"
    hwmon2_label = f"{w.HWMON_DIR}hwmon2/temp1_label"
    script = [
        discovery_rule(thermal_candidates(), {zone0: "82000\n", zone4: "70000\n"}, node="head"),
        discovery_rule(
            hwmon_name_candidates(), {hwmon2_name: "mlx5\n", hwmon5_name: "gpu\n"}, node="head"
        ),
        discovery_rule(hwmon_input_candidates(("hwmon2",)), {hwmon2_input: "47000\n"}, node="head"),
        discovery_rule(label_candidates(("hwmon2/temp1",)), {hwmon2_label: "asic\n"}, node="head"),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=2.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.sample_count == 3
    result = json.loads(
        (outcome.samples_path.parent / w.RESULT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert result["thermal_zones_found"]["head"] == [0, 4]
    assert result["hwmon_chip_names"]["head"] == {"hwmon2": "mlx5"}
    assert result["hwmon_sensors_found"]["head"] == {"hwmon2/temp1": "asic"}

    shown = [" ".join(call.argv) for call in runner.calls]
    assert not any("thermal_zone32" in one for one in shown)
    assert not any("hwmon32" in one for one in shown)
    assert not any("temp33_input" in one for one in shown)
    assert not any("hwmon5/temp1_input" in one for one in shown)

    # 観察は 3 回あるので、最初の `nvidia-smi` で打ち切る `discovery_calls_for_node` ではなく、
    # 全呼び出しから数える (2 回目以降の観察の中で発見をやり直す実装も検出するため)。
    name_batches = [
        call
        for call in runner.calls
        if call.node == "head" and call.argv == ("cat", *hwmon_name_candidates())
    ]
    assert len(name_batches) == 1, "発見は開始時の 1 回だけのはずが、観察のたびに繰り返している"


def test_discovery_reads_up_to_thirty_two_numbers_per_kind(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """発見は、熱区域・hwmon の名前・cpufreq のコアの番号を、それぞれ 0 から 31 まで
    (上限 32) だけ、種類ごとに 1 回の `cat` にまとめて試す。台本を敷かない (既定の `Reply()`
    は exit 0 なので、番号があれば必ず「存在する」ことになる)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(var_root=tmp_path, default=Reply())

    w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    head_cats = [
        call.argv
        for call in runner.calls
        if call.node == "head" and call.argv and call.argv[0] == "cat"
    ]
    assert ("cat", *thermal_candidates()) in head_cats
    assert ("cat", *hwmon_name_candidates()) in head_cats
    assert ("cat", *cpufreq_candidates()) in head_cats

    shown = [" ".join(call.argv) for call in runner.calls]
    assert not any("thermal_zone32" in one for one in shown)
    assert not any("hwmon32" in one for one in shown)
    assert not any("cpu32" in one for one in shown)


def test_discovery_treats_a_nonzero_exit_as_absent_without_recording_a_note(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """発見で exit 1 (存在しない) は、これまでどおり黙って続け、`detail` に書かない。
    採用する chip がないので、台ごとの発見は熱区域・hwmon の名前・cpufreq のコアの 3 回で
    終わる (input と label は流さない)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(var_root=tmp_path, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.thermal_zones_found["worker"] == ()
    assert "発見" not in outcome.detail
    assert len(discovery_calls_for_node(runner, "worker")) == 3


def test_discovery_stops_for_a_node_it_cannot_reach_and_records_it(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """発見で台に届かなければ (`RemoteError`)、その台の発見を打ち切り、役割つきで `detail`
    に記録する。見張り自体は続き、決めた時間ぶん観察する。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[
            Rule(
                prefix=("cat", *thermal_candidates()),
                node="worker",
                replies=(Reply(exit_code=255),),
            )
        ],
        default=Reply(exit_code=1),
    )

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=2.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert "worker" in outcome.detail
    assert "発見" in outcome.detail
    assert outcome.thermal_zones_found["worker"] == ()
    assert len(discovery_calls_for_node(runner, "worker")) == 1
    assert outcome.sample_count == 3


def test_discovery_round_trips_are_fixed_per_node(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """開始時の発見の往復 (`runner.run`) の回数を、偽の実行役で数えて固定する。

    head は熱区域が 2 つ・採用 chip が 1 つ・温度センサーが 1 つ・cpufreq のコアが 2 つ
    なので、熱区域・hwmon の名前・温度センサー・ラベル・cpufreq の 5 回。worker は何も
    見つからないので、熱区域・hwmon の名前・cpufreq の 3 回。合わせて 8 回で、すべて `cat`
    である。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    zone0 = f"{w.THERMAL_ZONE_DIR}thermal_zone0/temp"
    zone4 = f"{w.THERMAL_ZONE_DIR}thermal_zone4/temp"
    hwmon2_name = f"{w.HWMON_DIR}hwmon2/name"
    hwmon5_name = f"{w.HWMON_DIR}hwmon5/name"
    hwmon2_input = f"{w.HWMON_DIR}hwmon2/temp1_input"
    hwmon2_label = f"{w.HWMON_DIR}hwmon2/temp1_label"
    core0 = f"{w.CPU_DIR}cpu0/cpufreq/scaling_max_freq"
    core5 = f"{w.CPU_DIR}cpu5/cpufreq/scaling_max_freq"
    script = [
        discovery_rule(thermal_candidates(), {zone0: "82000\n", zone4: "70000\n"}, node="head"),
        discovery_rule(
            hwmon_name_candidates(), {hwmon2_name: "mlx5\n", hwmon5_name: "gpu\n"}, node="head"
        ),
        discovery_rule(hwmon_input_candidates(("hwmon2",)), {hwmon2_input: "47000\n"}, node="head"),
        discovery_rule(label_candidates(("hwmon2/temp1",)), {hwmon2_label: "asic\n"}, node="head"),
        discovery_rule(cpufreq_candidates(), {core0: "2808000\n", core5: "3000000\n"}, node="head"),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    calls = discovery_calls(runner)
    assert len(calls) == 8
    assert all(call.argv[0] == "cat" for call in calls)
    assert len(discovery_calls_for_node(runner, "head")) == 5
    assert len(discovery_calls_for_node(runner, "worker")) == 3

    result = json.loads(
        (outcome.samples_path.parent / w.RESULT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert result["thermal_zones_found"] == {"head": [0, 4], "worker": []}
    assert result["hwmon_chip_names"] == {"head": {"hwmon2": "mlx5"}, "worker": {}}
    assert result["hwmon_sensors_found"] == {"head": {"hwmon2/temp1": "asic"}, "worker": {}}
    assert result["cpufreq_cores_found"] == {"head": [0, 5], "worker": []}


def test_interrupt_during_discovery_still_writes_a_partial_summary(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """発見の最中に `KeyboardInterrupt` が来ても、`result.json` を書いてから中断を伝える。
    発見が終わっていないので、見つけた一覧は空で、観察は 0 件である。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[
            Rule(
                prefix=("cat", *thermal_candidates()),
                node="head",
                replies=(Reply(raises=KeyboardInterrupt()),),
            )
        ],
        default=Reply(exit_code=1),
    )

    with pytest.raises(KeyboardInterrupt):
        w.watch(
            runner,
            config,
            NODES,
            var_root=tmp_path,
            duration_s=100.0,
            interval_s=1.0,
            sleep=clock.sleep,
            clock=clock.monotonic,
            now=clock.now,
            report=io.StringIO(),
        )

    run_dir = var_dir(tmp_path, STARTED_WALL, w.COMMAND_NAME, CONFIG_NAME)
    result = json.loads((run_dir / w.RESULT_FILE_NAME).read_text(encoding="utf-8"))
    assert result["sample_count"] == 0
    assert result["thermal_zones_found"] == {}


def test_discovery_ignores_stderr_lines_that_do_not_name_a_candidate(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """発見の `cat` が exit 1 のとき、stderr の各行のうち、候補の道筋を名指す行だけを
    「存在しない」の根拠にする。ssh の警告の行と、別の種類 (hwmon) の道筋の行は無視し、
    区域 1 の行が区域 10・11 を「存在しない」にしない。区域 1 の行だけが効くので、区域 1 を
    欠く 31 個が見つかる。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    candidates = thermal_candidates()
    stderr = (
        "Warning: Permanently added 'spark-153d' (ED25519) to the list of known hosts.\n"
        f"cat: {w.THERMAL_ZONE_DIR}thermal_zone1/temp: No such file or directory\n"
        f"cat: {w.HWMON_DIR}hwmon0/name: No such file or directory\n"
    )
    script = [
        Rule(
            prefix=("cat", *candidates),
            node="head",
            replies=(Reply(exit_code=1, stderr=stderr),),
        )
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    found = outcome.thermal_zones_found["head"]
    assert found == tuple(number for number in range(32) if number != 1)
    assert 0 in found
    assert 10 in found
    assert 11 in found


def test_discovery_finds_nothing_when_cat_fails_without_per_path_errors(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """発見の `cat` が exit 1 で、stderr に道筋ごとの誤りの行が 1 つもないときは、その種類を
    「何も見つからない」にする (誤検出しない側に倒す)。`detail` に注記は足さない。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    script = [
        Rule(
            prefix=("cat", *thermal_candidates()),
            node="head",
            replies=(Reply(exit_code=1),),
        )
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.thermal_zones_found["head"] == ()
    assert "発見" not in outcome.detail


def test_discovery_treats_exit_zero_as_all_present_despite_stderr_noise(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """発見の `cat` が exit 0 なら、stderr に ssh の警告があっても候補すべてが存在する。
    cpufreq のコア 0〜31 が全部見つかる。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    script = [
        Rule(
            prefix=("cat", *cpufreq_candidates()),
            node="head",
            replies=(
                Reply(
                    exit_code=0,
                    stderr=(
                        "Warning: Permanently added 'spark-153d' (ED25519) "
                        "to the list of known hosts.\n"
                    ),
                ),
            ),
        )
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.cpufreq_cores_found["head"] == tuple(range(32))


def test_hwmon_names_that_do_not_align_with_the_present_paths_adopt_no_chip(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """hwmon の `name` で、存在する道筋の数と stdout の行数が合わなければ、その種類の中身は
    読めなかったものとして、採用する chip を 1 つも作らない (input と label を流さない)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    names = hwmon_name_candidates()
    present_names = (f"{w.HWMON_DIR}hwmon30/name", f"{w.HWMON_DIR}hwmon31/name")
    absent = [path for path in names if path not in present_names]
    script = [
        Rule(
            prefix=("cat", *names),
            node="head",
            replies=(
                # 存在する道筋は 2 つだが、stdout は 1 行しかない (対応が取れない)
                Reply(exit_code=1, stdout="mlx5\n", stderr=unreadable_lines(absent)),
            ),
        )
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    result = json.loads(
        (outcome.samples_path.parent / w.RESULT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert result["hwmon_chip_names"]["head"] == {}
    assert result["hwmon_sensors_found"]["head"] == {}
    assert not any(
        "_input" in " ".join(call.argv) for call in discovery_calls_for_node(runner, "head")
    )


def test_hwmon_labels_are_matched_to_the_sensors_that_have_them(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """採用する chip が複数あり、label のあるセンサーとないセンサーが混じるとき、label は
    存在した label の道筋の順に対応づけられ、label のないセンサーだけが `None` になる
    (label の行を、存在した道筋ではなく全センサーの順に対応づけない)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    hwmon1_name = f"{w.HWMON_DIR}hwmon1/name"
    hwmon2_name = f"{w.HWMON_DIR}hwmon2/name"
    hwmon3_name = f"{w.HWMON_DIR}hwmon3/name"
    hwmon2_temp1_input = f"{w.HWMON_DIR}hwmon2/temp1_input"
    hwmon2_temp2_input = f"{w.HWMON_DIR}hwmon2/temp2_input"
    hwmon3_temp1_input = f"{w.HWMON_DIR}hwmon3/temp1_input"
    script = [
        # hwmon1 は不採用 (name が mlx5/nvme/acpitz のどれでもない)。採用は hwmon2 と hwmon3
        discovery_rule(
            hwmon_name_candidates(),
            {hwmon1_name: "gpu\n", hwmon2_name: "mlx5\n", hwmon3_name: "nvme\n"},
            node="head",
        ),
        discovery_rule(
            hwmon_input_candidates(("hwmon2", "hwmon3")),
            {
                hwmon2_temp1_input: "47000\n",
                hwmon2_temp2_input: "48000\n",
                hwmon3_temp1_input: "50000\n",
            },
            node="head",
        ),
        # label は hwmon2/temp1 と hwmon3/temp1 にだけある (hwmon2/temp2 にはない)
        discovery_rule(
            label_candidates(("hwmon2/temp1", "hwmon2/temp2", "hwmon3/temp1")),
            {
                f"{w.HWMON_DIR}hwmon2/temp1_label": "asic\n",
                f"{w.HWMON_DIR}hwmon3/temp1_label": "nvme-label\n",
            },
            node="head",
        ),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    result = json.loads(
        (outcome.samples_path.parent / w.RESULT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert result["hwmon_chip_names"]["head"] == {"hwmon2": "mlx5", "hwmon3": "nvme"}
    assert result["hwmon_sensors_found"]["head"] == {
        "hwmon2/temp1": "asic",
        "hwmon2/temp2": None,
        "hwmon3/temp1": "nvme-label",
    }


def test_hwmon_labels_that_do_not_align_leave_all_sensors_without_a_label(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """label の stdout の行数が、存在した label の道筋の数と合わなければ、対応が取れないので
    全センサーの label を `None` にし、温度センサーの発見自体は残す (決めごとの 9)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    hwmon2_name = f"{w.HWMON_DIR}hwmon2/name"
    hwmon2_temp1_input = f"{w.HWMON_DIR}hwmon2/temp1_input"
    hwmon2_temp2_input = f"{w.HWMON_DIR}hwmon2/temp2_input"
    labels = label_candidates(("hwmon2/temp1", "hwmon2/temp2"))
    script = [
        discovery_rule(hwmon_name_candidates(), {hwmon2_name: "mlx5\n"}, node="head"),
        discovery_rule(
            hwmon_input_candidates(("hwmon2",)),
            {hwmon2_temp1_input: "47000\n", hwmon2_temp2_input: "48000\n"},
            node="head",
        ),
        # 存在する label の道筋は 2 つだが、stdout は 1 行しかない (対応が取れない)
        Rule(
            prefix=("cat", *labels),
            node="head",
            replies=(Reply(exit_code=0, stdout="asic\n"),),
        ),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    result = json.loads(
        (outcome.samples_path.parent / w.RESULT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert result["hwmon_sensors_found"]["head"] == {
        "hwmon2/temp1": None,
        "hwmon2/temp2": None,
    }


def test_an_adopted_chip_without_any_temperature_sensor_keeps_the_chip_and_reads_no_labels(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """採用した chip に温度センサー (`temp<M>_input`) が 1 つもなければ、chip の一覧は残し、
    センサーは空にする。ラベルの候補がないので、ラベルの `cat` は流さない (道筋のない
    `cat` は許されず、流すと見張り全体が落ちる)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    hwmon2_name = f"{w.HWMON_DIR}hwmon2/name"
    script = [
        discovery_rule(hwmon_name_candidates(), {hwmon2_name: "mlx5\n"}, node="head"),
        # 温度センサーの `cat` は台本を敷かない (既定の exit 1、`cat:` の行なし = 何も存在しない)
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    result = json.loads(
        (outcome.samples_path.parent / w.RESULT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert result["hwmon_chip_names"]["head"] == {"hwmon2": "mlx5"}
    assert result["hwmon_sensors_found"]["head"] == {}
    head_discovery = discovery_calls_for_node(runner, "head")
    assert ("cat", *hwmon_input_candidates(("hwmon2",))) in [call.argv for call in head_discovery]
    assert not any("_label" in " ".join(call.argv) for call in head_discovery)


# --- 熱区域・hwmon・CPU の 1 回の観察 --------------------------------------------


def test_thermal_zones_and_hwmon_are_recorded_per_sensor(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """発見した熱区域と hwmon の値が、区域番号・センサーの識別子つきで `samples.jsonl` に
    出る (最高値だけにまとめない)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    zone0 = f"{w.THERMAL_ZONE_DIR}thermal_zone0/temp"
    zone4 = f"{w.THERMAL_ZONE_DIR}thermal_zone4/temp"
    hwmon2_name = f"{w.HWMON_DIR}hwmon2/name"
    hwmon2_input = f"{w.HWMON_DIR}hwmon2/temp1_input"
    hwmon2_label = f"{w.HWMON_DIR}hwmon2/temp1_label"
    script = [
        discovery_rule(thermal_candidates(), {zone0: "82000\n", zone4: "70000\n"}, node="head"),
        discovery_rule(hwmon_name_candidates(), {hwmon2_name: "mlx5\n"}, node="head"),
        discovery_rule(hwmon_input_candidates(("hwmon2",)), {hwmon2_input: "47000\n"}, node="head"),
        discovery_rule(label_candidates(("hwmon2/temp1",)), {hwmon2_label: "asic\n"}, node="head"),
        Rule(prefix=("cat", zone0, zone4), node="head", replies=(Reply(stdout="82000\n70000\n"),)),
        Rule(prefix=("cat", hwmon2_input), node="head", replies=(Reply(stdout="47000\n"),)),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    first = json.loads(outcome.samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert first["thermal_zones_c"]["head"] == {"0": 82.0, "4": 70.0}
    assert first["hwmon_temps_c"]["head"] == {"hwmon2/temp1": 47.0}


def test_hwmon_sensor_without_a_label_is_recorded_with_an_empty_label(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """`temp<M>_label` が読めない (存在しない) hwmon センサーは、`hwmon_sensors_found` で
    そのセンサーのラベルだけを空にし、温度の値 (`hwmon_temps_c`) はそのまま残る。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    hwmon2_name = f"{w.HWMON_DIR}hwmon2/name"
    hwmon2_input = f"{w.HWMON_DIR}hwmon2/temp1_input"
    script = [
        discovery_rule(hwmon_name_candidates(), {hwmon2_name: "mlx5\n"}, node="head"),
        discovery_rule(hwmon_input_candidates(("hwmon2",)), {hwmon2_input: "47000\n"}, node="head"),
        # ラベルは存在しない (発見の規則に当たらず、既定の exit 1 になる)
        Rule(prefix=("cat", hwmon2_input), node="head", replies=(Reply(stdout="47000\n"),)),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    result = json.loads(
        (outcome.samples_path.parent / w.RESULT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert result["hwmon_sensors_found"]["head"] == {"hwmon2/temp1": None}
    first = json.loads(outcome.samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert first["hwmon_temps_c"]["head"] == {"hwmon2/temp1": 47.0}


def test_an_unreadable_item_is_empty_without_stopping_the_watch(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """行数が、発見した区域の数と合わない回は、熱区域の項目だけを空にし、ほかの項目 (CPU)
    と見張りは続く。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    zone0 = f"{w.THERMAL_ZONE_DIR}thermal_zone0/temp"
    zone2 = f"{w.THERMAL_ZONE_DIR}thermal_zone2/temp"
    zone4 = f"{w.THERMAL_ZONE_DIR}thermal_zone4/temp"
    proc_stat = ("cat", w.PROC_STAT_PATH)
    script = [
        discovery_rule(
            thermal_candidates(),
            {zone0: "82000\n", zone2: "75000\n", zone4: "70000\n"},
            node="head",
        ),
        # 発見した区域は [0, 2, 4] の 3 つだが、観察の返事は 2 行しかない (行数の不一致)
        Rule(
            prefix=("cat", zone0, zone2, zone4),
            node="head",
            replies=(Reply(stdout="82000\n70000\n"),),
        ),
        Rule(
            prefix=proc_stat,
            replies=(
                Reply(stdout="cpu0 100 0 100 800 0 0 0 0 0 0\n"),
                Reply(stdout="cpu0 200 0 200 800 0 0 0 0 0 0\n"),
            ),
        ),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    lines = [
        json.loads(line) for line in outcome.samples_path.read_text(encoding="utf-8").splitlines()
    ]
    assert outcome.sample_count == 2
    second = lines[1]
    assert second["thermal_zones_c"]["head"] == {"0": None, "2": None, "4": None}
    assert second["cpu_utilization_pct"]["head"] == {"0": 100.0}


def test_a_line_that_fails_to_parse_only_empties_that_key(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """行数は発見した区域の数と合うが、1 行だけ整数に変えられないときは、そのキーだけを
    空にし、ほかのキーの値は残す (行数が合わず全キーを空にする場合とは違う経路)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    zone0 = f"{w.THERMAL_ZONE_DIR}thermal_zone0/temp"
    zone4 = f"{w.THERMAL_ZONE_DIR}thermal_zone4/temp"
    script = [
        discovery_rule(thermal_candidates(), {zone0: "82000\n", zone4: "70000\n"}, node="head"),
        # 観察は 2 行 (zone0 は数、zone4 は壊れた行)
        Rule(prefix=("cat", zone0, zone4), node="head", replies=(Reply(stdout="82000\nabc\n"),)),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    first = json.loads(outcome.samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert first["thermal_zones_c"]["head"] == {"0": 82.0, "4": None}


def test_cpu_utilization_is_the_difference_from_the_previous_reading(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """コアごとの使用率は、直前に読めた観察との差から出る。初回は空になる。合計行
    (`cpu `) はコアとして数えず、差のないコアも含めない。合計行の値が観察のあいだで
    増えても (2 回目)、コアごとの計算には影響しない。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    proc_stat = ("cat", w.PROC_STAT_PATH)
    reading1 = (
        "cpu  300 0 300 1600 0 0 0 0 0 0\n"
        "cpu0 100 0 100 800 0 0 0 0 0 0\n"
        "cpu1 200 0 200 800 0 0 0 0 0 0\n"
    )
    reading2 = (
        "cpu  400 0 400 1600 0 0 0 0 0 0\n"
        "cpu0 200 0 200 800 0 0 0 0 0 0\n"
        "cpu1 200 0 200 800 0 0 0 0 0 0\n"
    )
    script = [Rule(prefix=proc_stat, replies=(Reply(stdout=reading1), Reply(stdout=reading2)))]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    lines = [
        json.loads(line) for line in outcome.samples_path.read_text(encoding="utf-8").splitlines()
    ]
    assert outcome.sample_count == 2
    assert lines[0]["cpu_utilization_pct"]["head"] == {}
    assert lines[1]["cpu_utilization_pct"]["head"] == {"0": 100.0}


# --- thermal の出来事 -----------------------------------------------------------


def test_thermal_event_fires_once_per_rising_edge_without_collecting_logs(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """熱区域が閾値以上になった観察の立ち上がりで `thermal` が 1 件になり、続く間は増えない。
    条件が外れてまた起きれば、別の 1 件になる。熱の出来事では、2 台の記録の回収
    (`docker` の呼び出し) をしない (推論サーバーは壊れていないため)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    zone0 = f"{w.THERMAL_ZONE_DIR}thermal_zone0/temp"
    # node="head" に絞る: NODES は head と worker の 2 台なので、絞らないと、両台の発見・
    # 観察の読みが同じ規則の replies を奪い合い、意図した順で値が出ない (worker は発見で
    # 区域が見つからないまま default(exit_code=1) になり、zone0 の値を持たない)
    script = [
        discovery_rule(thermal_candidates(), {zone0: "85000\n"}, node="head"),
        Rule(
            prefix=("cat", zone0),
            node="head",
            replies=(
                Reply(stdout="89000\n"),
                Reply(stdout="90500\n"),
                Reply(stdout="91000\n"),
                Reply(stdout="88000\n"),
                Reply(stdout="90000\n"),
            ),
        ),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=4.0,
        interval_s=1.0,
        thermal_threshold_c=90.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.sample_count == 5
    assert [event.finding for event in outcome.events] == ["thermal", "thermal"]
    assert outcome.events[0].at_utc == STARTED_WALL + timedelta(seconds=1)
    assert outcome.events[1].at_utc == STARTED_WALL + timedelta(seconds=4)
    for event in outcome.events:
        assert "記録は" not in event.detail
        assert "回収" not in event.detail
    for call in runner.calls:
        assert not (call.argv and call.argv[0] == "docker")


def test_no_thermal_event_below_the_threshold(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """熱区域の値が、既定のしきい値 (90℃) をわずかに下回れば `thermal` は出ない。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    zone0 = f"{w.THERMAL_ZONE_DIR}thermal_zone0/temp"
    script = [
        discovery_rule(thermal_candidates(), {zone0: "89999\n"}, node="head"),
        Rule(prefix=("cat", zone0), node="head", replies=(Reply(stdout="89999\n"),)),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.events == ()


# --- result.json の要約 ----------------------------------------------------------


def test_summary_has_per_node_maxima_and_the_sm_clock_range(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """`result.json` に、台ごと・項目ごとの最大値 (GPU 温度、SM の範囲、電力、熱区域、
    hwmon) が出る。1 度も読めなかった台の項目は空になる。熱区域と hwmon には、途中の回
    (最後ではない回) で最大になる値の列を与え、読めない回を挟んでも、その最大値がそのまま
    `result.json` に出ること (最後の値で上書きしないこと) を確かめる。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    zone0 = f"{w.THERMAL_ZONE_DIR}thermal_zone0/temp"
    hwmon2_name = f"{w.HWMON_DIR}hwmon2/name"
    hwmon2_input = f"{w.HWMON_DIR}hwmon2/temp1_input"
    # node="head" に絞る (絞らないと、head と worker が同じ規則の replies を奪い合う)
    script = [
        Rule(
            prefix=w.GPU_UTIL_ARGV,
            node="head",
            replies=(
                Reply(stdout="65, 1781, 28.90, 10\n"),
                Reply(stdout="70, 1774, 30.10, 10\n"),
                Reply(stdout="68, 1781, 29.00, 10\n"),
            ),
        ),
        Rule(
            prefix=w.GPU_UTIL_ARGV,
            node="worker",
            replies=(Reply(stdout="[N/A], [N/A], [N/A], [N/A]\n"),),
        ),
        discovery_rule(thermal_candidates(), {zone0: "80000\n"}, node="head"),
        discovery_rule(hwmon_name_candidates(), {hwmon2_name: "mlx5\n"}, node="head"),
        discovery_rule(hwmon_input_candidates(("hwmon2",)), {hwmon2_input: "47000\n"}, node="head"),
        Rule(
            prefix=("cat", zone0),
            node="head",
            replies=(
                Reply(stdout="80000\n"),  # 観察 1
                Reply(stdout="88000\n"),  # 観察 2 (途中の回で最大)
                Reply(exit_code=1),  # 観察 3 (読めない)
                Reply(stdout="84000\n"),  # 観察 4
            ),
        ),
        Rule(
            prefix=("cat", hwmon2_input),
            node="head",
            replies=(
                Reply(stdout="47000\n"),  # 観察 1
                Reply(stdout="52000\n"),  # 観察 2 (途中の回で最大)
                Reply(exit_code=1),  # 観察 3 (読めない)
                Reply(stdout="49000\n"),  # 観察 4
            ),
        ),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=3.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.sample_count == 4
    result = json.loads(
        (outcome.samples_path.parent / w.RESULT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert result["gpu_temperature_max_c"]["head"] == 70
    assert result["gpu_sm_clock_range_mhz"]["head"] == [1774, 1781]
    assert result["gpu_power_max_w"]["head"] == 30.1
    assert result["gpu_temperature_max_c"]["worker"] is None
    assert result["thermal_zone_max_c"]["head"] == {"0": 88.0}
    assert result["hwmon_temp_max_c"]["head"] == {"hwmon2/temp1": 52.0}


def test_summary_counts_the_samples_over_the_threshold_and_their_time_span(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """`result.json` に、熱区域が閾値以上だった観察の数と、その時刻の範囲が出る。観察の数は
    出来事の件数 (立ち上がりの回数) とは別に数える。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    zone0 = f"{w.THERMAL_ZONE_DIR}thermal_zone0/temp"
    # node="head" に絞る (理由は test_thermal_event_fires_once_per_rising_edge... と同じ:
    # 絞らないと、head と worker が同じ規則の replies を奪い合ってしまう)
    script = [
        discovery_rule(thermal_candidates(), {zone0: "80000\n"}, node="head"),
        Rule(
            prefix=("cat", zone0),
            node="head",
            replies=(
                Reply(stdout="85000\n"),
                Reply(stdout="86000\n"),
                Reply(stdout="91000\n"),
                Reply(stdout="92000\n"),
                Reply(stdout="87000\n"),
            ),
        ),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=4.0,
        interval_s=1.0,
        thermal_threshold_c=90.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    result = json.loads(
        (outcome.samples_path.parent / w.RESULT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert result["thermal_threshold_c"] == 90.0
    assert result["thermal_over_threshold_samples"] == 2
    # 内部の直列化の書式 (Z か +00:00 か) に依存しないよう、値を datetime に戻して比べる
    span = [datetime.fromisoformat(value) for value in result["thermal_over_threshold_span"]]
    assert span == [
        STARTED_WALL + timedelta(seconds=2),
        STARTED_WALL + timedelta(seconds=3),
    ]
    assert [event.finding for event in outcome.events] == ["thermal"]


# --- 既存の判定が変わらないこと、読み取りだけであること --------------------------------


def test_thermal_condition_does_not_change_the_unresponsive_judgement(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """熱区域がずっと閾値以上でも、`unresponsive` の判定 (続けての失敗の回数と、出来事の
    時刻) は変わらない。既存試験
    (`test_unresponsive_event_appears_on_the_third_consecutive_failure`) と同じ時刻で
    `unresponsive` が出ることを、`thermal` が同時に起きていても固定する。"""
    fake_vllm.set_health_fault(Fault(status=None))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    zone0 = f"{w.THERMAL_ZONE_DIR}thermal_zone0/temp"
    script = [
        discovery_rule(thermal_candidates(), {zone0: "95000\n"}),
        Rule(prefix=("cat", zone0), replies=(Reply(stdout="95000\n"),)),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=2.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    assert outcome.sample_count == 3
    assert [event.finding for event in outcome.events] == ["thermal", "unresponsive"]
    unresponsive = next(event for event in outcome.events if event.finding == "unresponsive")
    assert unresponsive.at_utc == STARTED_WALL + timedelta(seconds=2)


# --- cpufreq のコアの発見と、周波数の上限の観察 ------------------------------------


def test_cpufreq_cores_are_discovered_once_and_read_per_observation(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """cpufreq のコアは、見張りの開始時に 1 回だけ発見し、観察ごとに 1 回の `cat` で読む。

    存在確認が通ったコアだけが `cpufreq_cores_found` に出る (上限 32 の手前まで)。発見した
    一覧のぶんの値が、`samples.jsonl` の各行の `cpu_scaling_max_freq_khz` に、台ごと・
    コア番号つきの kHz の整数で出る。
    """
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    core0 = f"{w.CPU_DIR}cpu0/cpufreq/scaling_max_freq"
    core5 = f"{w.CPU_DIR}cpu5/cpufreq/scaling_max_freq"
    script = [
        discovery_rule(cpufreq_candidates(), {core0: "2808000\n", core5: "3000000\n"}, node="head"),
        # 観察は 1 回の `cat` に 2 つの道筋を並べる (2 回とも同じ値)
        Rule(
            prefix=("cat", core0, core5),
            node="head",
            replies=(Reply(stdout="2808000\n3000000\n"),),
        ),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    result = json.loads(
        (outcome.samples_path.parent / w.RESULT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert result["cpufreq_cores_found"]["head"] == [0, 5]

    lines = [
        json.loads(line) for line in outcome.samples_path.read_text(encoding="utf-8").splitlines()
    ]
    assert outcome.sample_count == 2
    assert lines[0]["cpu_scaling_max_freq_khz"]["head"] == {"0": 2808000, "5": 3000000}
    assert lines[1]["cpu_scaling_max_freq_khz"]["head"] == {"0": 2808000, "5": 3000000}

    shown = [" ".join(call.argv) for call in runner.calls]
    assert not any("cpu32" in one for one in shown), "コアの上限 (32) を超えて試している"
    # 観察 1 回あたり、台ごとに cpufreq の `cat` は 1 回 (複数の道筋を 1 つに並べる)
    observation_cats = [
        call for call in runner.calls if call.node == "head" and call.argv == ("cat", core0, core5)
    ]
    assert len(observation_cats) == 2, (
        "観察 1 回あたり、台ごとの cpufreq の `cat` は 1 回 (コアごとに別々に流していない)"
    )
    # 発見のまとめ読みの呼び出しが、全呼び出しの中で 1 回だけであることを確かめる (観察は 2 回
    # あるので、最初の `nvidia-smi` で打ち切る `discovery_calls_for_node` ではなく、全体から
    # 数える。観察のたびに発見をやり直す実装は、観察の回数ぶん増えて検出できる)。
    discovery_batches = [
        call
        for call in runner.calls
        if call.node == "head" and call.argv == ("cat", *cpufreq_candidates())
    ]
    assert len(discovery_batches) == 1, "発見は開始時の 1 回だけで、観察のたびには繰り返さない"


def test_an_unreadable_cpufreq_round_empties_every_key_and_the_watch_continues(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """行数が発見したコアの数と合わない回は、その項目の全キーを空にし、CPU の使用率
    (`cpu_utilization_pct`) と見張りは続く。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    core0 = f"{w.CPU_DIR}cpu0/cpufreq/scaling_max_freq"
    core5 = f"{w.CPU_DIR}cpu5/cpufreq/scaling_max_freq"
    # 観察 1 (2 行。どちらも数) → 観察 2 (1 行だけ。行数の不一致)
    script = [
        discovery_rule(cpufreq_candidates(), {core0: "2808000\n", core5: "3000000\n"}, node="head"),
        Rule(
            prefix=("cat", core0, core5),
            node="head",
            replies=(Reply(stdout="2808000\n3000000\n"), Reply(stdout="2808000\n")),
        ),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    lines = [
        json.loads(line) for line in outcome.samples_path.read_text(encoding="utf-8").splitlines()
    ]
    assert outcome.sample_count == 2
    assert lines[1]["cpu_scaling_max_freq_khz"]["head"] == {"0": None, "5": None}
    assert lines[1]["cpu_utilization_pct"]["head"] == {}


def test_a_single_unparseable_cpufreq_line_empties_only_that_key(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """行数は合うが、1 行だけ整数に変えられないときは、そのキーだけを空にし、ほかのキーの
    値は残す (行数が合わず全キーを空にする場合とは違う経路)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    core0 = f"{w.CPU_DIR}cpu0/cpufreq/scaling_max_freq"
    core5 = f"{w.CPU_DIR}cpu5/cpufreq/scaling_max_freq"
    script = [
        discovery_rule(cpufreq_candidates(), {core0: "2808000\n", core5: "3000000\n"}, node="head"),
        Rule(
            prefix=("cat", core0, core5),
            node="head",
            replies=(Reply(stdout="2808000\nabc\n"),),
        ),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    first = json.loads(outcome.samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert first["cpu_scaling_max_freq_khz"]["head"] == {"0": 2808000, "5": None}


def test_summary_records_the_cluster_maxima_across_all_samples(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """`result.json` に、台ごとの X925 と A725 の上限の最大値が出る。

    観察の途中で最大になった値が、最後の観察の値で上書きされないこと (全観察の最大) と、
    発見のない台に空でないキーを出さないことを確かめる。X925 のコア (5) の上限が途中で
    3900000 に上がっても、最大値は 3900000 のまま残る。
    """
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    core0 = f"{w.CPU_DIR}cpu0/cpufreq/scaling_max_freq"
    core5 = f"{w.CPU_DIR}cpu5/cpufreq/scaling_max_freq"
    # 観察 1: X925 (コア 5) が 3000000、A725 (コア 0) が 2808000
    # 観察 2: X925 が 3900000 (上限が外れた) — これが最大として残る
    # 観察 3: 1 行だけ (行数の不一致 → 全キー空) — 最大値は 3900000 のまま
    # 観察 4: X925 が 3000000 に戻る
    script = [
        discovery_rule(cpufreq_candidates(), {core0: "2808000\n", core5: "3000000\n"}, node="head"),
        Rule(
            prefix=("cat", core0, core5),
            node="head",
            replies=(
                Reply(stdout="2808000\n3000000\n"),
                Reply(stdout="2808000\n3900000\n"),
                Reply(stdout="2808000\n"),
                Reply(stdout="2808000\n3000000\n"),
            ),
        ),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=4.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    result = json.loads(
        (outcome.samples_path.parent / w.RESULT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert result["cpu_cluster_max_freq_khz"]["head"] == {"x925": 3900000, "a725": 2808000}

    # 発見のない台 (worker は 1 つも発見できない台本) は、キーを持たない
    assert "worker" not in result["cpu_cluster_max_freq_khz"]
    assert result["cpufreq_cores_found"]["worker"] == []


def test_a_cluster_that_was_never_read_has_no_key_in_the_summary(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """X925 のコアだけが発見した台は、`a725` のキーを持たない (空の値を作らない)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    core5 = f"{w.CPU_DIR}cpu5/cpufreq/scaling_max_freq"
    script = [
        discovery_rule(cpufreq_candidates(), {core5: "3000000\n"}, node="head"),
        Rule(prefix=("cat", core5), node="head", replies=(Reply(stdout="3000000\n"),)),
    ]
    runner = FakeRunner(var_root=tmp_path, script=script, default=Reply(exit_code=1))

    outcome = w.watch(
        runner,
        config,
        NODES,
        var_root=tmp_path,
        duration_s=1.0,
        interval_s=1.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
        report=io.StringIO(),
    )

    result = json.loads(
        (outcome.samples_path.parent / w.RESULT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert result["cpu_cluster_max_freq_khz"]["head"] == {"x925": 3000000}
    assert result["cpufreq_cores_found"]["head"] == [5]
