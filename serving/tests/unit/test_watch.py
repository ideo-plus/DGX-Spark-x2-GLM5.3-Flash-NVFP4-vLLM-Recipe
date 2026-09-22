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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="97 %\n"),))],
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="97 %\n"),))],
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
    # GB10 が対応していないときに出る文字列 (int に変換できない) を、両台とも常に返す
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="[Not Supported]\n"),))],
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
            Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="10 %\n"),)),
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
            Rule(prefix=w.GPU_UTIL_ARGV, node="head", replies=(Reply(stdout="10 %\n"),)),
            Rule(
                prefix=w.GPU_UTIL_ARGV,
                node="worker",
                replies=(
                    Reply(exit_code=255, stderr="ssh: no route to host"),
                    Reply(stdout="10 %\n"),
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
    fake_vllm.set_metrics(MetricsSample(running_requests=1, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="10 %\n"),))],
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="10 %\n"),))],
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="10 %\n"),))],
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="50 %\n"),))],
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
        Reply(stdout="95 %\n"),
        Reply(stdout="95 %\n"),
        Reply(stdout="10 %\n"),
        Reply(stdout="95 %\n"),
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="95 %\n"),))],
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="95 %\n"),))],
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="95 %\n"),))],
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="95 %\n"),))],
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="10 %\n"),))],
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
            Rule(prefix=w.GPU_UTIL_ARGV, node="head", replies=(Reply(stdout="10 %\n"),)),
            Rule(
                prefix=w.GPU_UTIL_ARGV,
                node="worker",
                replies=(Reply(stdout="10 %\n"), Reply(raises=KeyboardInterrupt())),
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
            Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="10 %\n"),)),
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="10 %\n"),))],
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="10 %\n"),))],
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
    (負にはならない。指摘 1・2 の「あわせて」)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    base_runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="10 %\n"),))],
        default=Reply(),
    )
    # 1 回の観察 (head と worker、2 回の nvidia-smi の呼び出し) に、0.3 秒ずつ経過が積まれる
    runner = _CostlyRunner(inner=base_runner, clock=clock, cost_s=0.3)

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

    # 1 回の観察に 0.6 秒 (2 台ぶん) かかるので、眠りは 1.0 - 0.6 = 0.4 秒になる
    assert clock.slept
    for value in clock.slept:
        assert value > 0
    assert clock.slept[0] == pytest.approx(0.4)


def test_sleep_is_skipped_without_a_negative_value_when_observing_overruns_the_interval(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """観察が間隔より長くかかっても、眠りに負の値を渡さない (眠りそのものを呼ばない)。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, generation_tokens_total=1))
    config = config_for(port_of(fake_vllm))
    clock = WatchClock()
    base_runner = FakeRunner(
        var_root=tmp_path,
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="10 %\n"),))],
        default=Reply(),
    )
    # 1 回の観察に 1.2 秒 (2 台ぶん、0.6 秒ずつ) かかり、間隔 (1.0 秒) を超える
    runner = _CostlyRunner(inner=base_runner, clock=clock, cost_s=0.6)

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
        Reply(stdout="95 %\n"),
        Reply(stdout="10 %\n"),
        Reply(stdout="95 %\n"),
        Reply(stdout="95 %\n"),
        Reply(stdout="95 %\n"),
        Reply(stdout="95 %\n"),
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="95 %\n"),))],
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="95 %\n"),))],
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
        script=[Rule(prefix=w.GPU_UTIL_ARGV, replies=(Reply(stdout="95 %\n"),))],
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
