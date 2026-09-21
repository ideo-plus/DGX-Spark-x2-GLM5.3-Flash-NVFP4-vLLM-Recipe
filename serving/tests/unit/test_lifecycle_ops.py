"""状態の確認、停止、短い要求での確かめの試験 (tasks.md 3.5)。

確かめること (design.md 「運転 › lifecycle」の `status` / `stop` / `smoke`、tasks.md 3.5 の
完了の状態):

- **停止**: head → worker の順に `docker stop -t 90`、そのあと 2 台で `docker rm`
  (requirements 1.7)。2 度目の停止は、了承も求めず、状態を変える呼び出しも出さずに
  `already_stopped` で終わる (requirements 1.8)。了承しないと、`stop` も `rm` も出ない
  (requirements 2.1)。止める対象は、自分の一覧に出た名前だけである (requirements 2.3、2.4)
- **停止のあとの GPU**: プロセスが 0 件になるまで待ち、空かなければ、名前とメモリの量を示して
  `gpu_not_released` で終わる。**残っているプロセスを止めには行かない** (requirements 1.7、2.3)
- **状態の確認**: 2 台のコンテナの状態、構成の名前、種類、イメージのダイジェスト、GPU の
  プロセス、直結のリンク、受け付けの可否、名乗るモデルの名前、入力の長さの上限、処理中と
  待ちの要求の数を示し、**状態を変える呼び出しを 1 つも出さない** (requirements 1.6、6.5)。
  何も動いていないときも、片方の台に入れないときも、落ちずに示す
- **短い要求**: 英語と日本語を 1 つずつ送り、**応答の本文が、どのファイルにも書かれない**
  (requirements 6.5、10.5)。置き換え文字 (U+FFFD) を含む応答には、印が立つ

実物の ssh、rsync、docker、推論サーバーには、どの段でもつながない (`FakeRunner` と、
`fake_vllm` の偽の推論サーバーを相手にする)。試験は、実際に眠らない (待つ間隔と時計は、
引数で差し替える)。
"""

from __future__ import annotations

import io
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl

from fake_runner import FakeRunner, Reply, Rule
from fake_vllm import FakeVllm, Fault, MessagesReply, MetricsSample
from serving_kit import guards as g
from serving_kit import lifecycle as lc
from serving_kit.config import ConfigError
from serving_kit.plan import (
    LABEL_CONFIG,
    LABEL_CONFIG_SHA256,
    LABEL_IMAGE,
    LABEL_KIND,
    LABEL_OWNER,
    LABEL_ROLE,
    LABEL_STARTED_AT,
    OWNER,
    OWNER_FILTER,
)
from serving_kit.remote import RemoteError
from serving_kit.types import (
    ConfigDef,
    GpuApp,
    ImageRef,
    NodeDef,
    NodeRole,
    NodeStatus,
    PlannedRun,
    ServiceStatus,
    Setting,
    WeightsRef,
)

# --- 見本の値 -------------------------------------------------------------

ROLES: tuple[NodeRole, ...] = ("head", "worker")

DIGEST = "b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
IMAGE_REF = f"vllm/vllm-openai@sha256:{DIGEST}"
CONFIG_SHA256 = "c0ffee" + "0" * 58
REPO = "RedHatAI/GLM-5.3-Flash-NVFP4"
REVISION = "18d55bfd" + "0" * 32
SLUG = "RedHatAI__GLM-5.3-Flash-NVFP4"
MOUNT_AT = "/models/nvfp4"
REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"
CONFIG_NAME = "p1-nvfp4-tp2"
SERVED_MODEL = "glm-5-3-flash"
MASTER_PORT = "29501"
UNUSED_PORT = 8123
"""HTTP に進まない試験が使う、偽の推論サーバーのない番号。"""

STARTED_AT = datetime(2026, 9, 22, 3, 0, 0, tzinfo=UTC)
STARTED_AT_LABEL = "2026-09-22T03:00:00Z"

SOURCE = HttpUrl("https://docs.vllm.ai/en/latest/cli/serve/")
QUOTE = "vllm serve [model_tag] [options]"

FABRIC_IFNAME = "enp1s0f0np0"
"""直結の側の、つながっているインターフェース (見本 `tests/fixtures/spark/*/ip-br-link.txt`)。"""

CONTAINER_NAMES: Mapping[NodeRole, str] = {
    "head": f"vb-{CONFIG_NAME}-head",
    "worker": f"vb-{CONFIG_NAME}-worker",
}
CONTAINER_IDS: Mapping[NodeRole, str] = {"head": "0123456789ab", "worker": "cdef01234567"}


def _setting(
    flag: str | None = None,
    value: str | None = None,
    *,
    only_on: NodeRole | None = None,
    is_port: bool = False,
) -> Setting:
    """根拠の付いた設定を 1 つ作る (根拠の中身は、この試験では問わない)。"""
    return Setting(
        flag=flag,
        value=value,
        why="試験のための設定",
        only_on=only_on,
        is_port=is_port,
        source=SOURCE,
        quote=QUOTE,
    )


IMAGE = ImageRef(
    ref=IMAGE_REF,
    seen_as="vllm/vllm-openai:glm53-flash-arm64-cu130",
    size_bytes=9666567584,
    source=HttpUrl("https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags"),
    quote="glm53-flash-arm64-cu130 / linux/arm64 / 9666567584 B",
)

WEIGHTS = WeightsRef(
    repo=REPO,
    revision=REVISION,
    manifest=f"{SLUG}.manifest.json",
    mount_at=MOUNT_AT,
    source=HttpUrl("https://huggingface.co/api/models/RedHatAI/GLM-5.3-Flash-NVFP4"),
    quote="List the content of a repository tree, with pagination support.",
)

HEAD = NodeDef(
    role="head",
    ssh_host="spark-153d",
    lan_addr=IPv4Address("127.0.0.1"),
    remote_root=REMOTE_ROOT,
)
WORKER = NodeDef(
    role="worker",
    ssh_host="spark-5083",
    lan_addr=IPv4Address("127.0.0.1"),
    remote_root=REMOTE_ROOT,
)
NODES: Mapping[NodeRole, NodeDef] = {"head": HEAD, "worker": WORKER}

FABRIC_NODES: Mapping[NodeRole, NodeDef] = {
    "head": HEAD.model_copy(update={"fabric_ifname": FABRIC_IFNAME}),
    "worker": WORKER.model_copy(update={"fabric_ifname": FABRIC_IFNAME}),
}
"""直結の側のインターフェースの名前まで入ったノードの定義 (`fabric_link_up` を読む)。"""


def serve_config(port: int, *, served_model_name: str = SERVED_MODEL) -> ConfigDef:
    """推論サーバーの構成 (design.md Data Models の `p1-nvfp4-tp2` を縮めたもの)。"""
    mount = "type=bind,source={remote_root}/models/" + SLUG + ",target={weights.mount_at},readonly"
    return ConfigDef(
        name=CONFIG_NAME,
        kind="serve",
        description="P1 の第一の構成 (試験用に縮めたもの)",
        nodes=ROLES,
        image=IMAGE,
        weights=WEIGHTS,
        docker={"models": _setting("--mount", mount)},
        args={
            "model-path": _setting(value="{weights.mount_at}"),
            "served-model-name": _setting("--served-model-name", served_model_name),
            "host": _setting("--host", "{head.lan_addr}"),
            "port": _setting("--port", str(port), is_port=True),
            "master-port": _setting("--master-port", MASTER_PORT, is_port=True),
        },
        env={},
        ready_timeout_s=1800,
        served_model_name=served_model_name,
    )


def portless_config() -> ConfigDef:
    """`--port` を持たない構成 (HTTP の宛先を決められない場合の見本)。"""
    base = serve_config(UNUSED_PORT)
    args = {key: value for key, value in base.args.items() if key != "port"}
    return base.model_copy(update={"args": args})


def configs_of(config: ConfigDef) -> Mapping[str, ConfigDef]:
    return {config.name: config}


def port_of(fake: FakeVllm) -> int:
    """偽の推論サーバーが待ち受けている番号。"""
    return int(fake.base_url.rsplit(":", 1)[1])


# --- 台本の部品 ------------------------------------------------------------

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

LINK_UP = (
    f"{FABRIC_IFNAME}      UP             02:00:00:00:00:02 <BROADCAST,MULTICAST,UP,LOWER_UP> \n"
)
LINK_DOWN = (
    f"{FABRIC_IFNAME}      DOWN           02:00:00:00:00:03 <NO-CARRIER,BROADCAST,MULTICAST,UP> \n"
)
"""`ip -br link show dev <名前>` の 1 行 (見本 `tests/fixtures/spark/*/ip-br-link.txt` の形)。"""

GPU_LINE = "12345, vllm::EngineCore, 92160 MiB\n"
"""`nvidia-smi --query-compute-apps=pid,process_name,used_memory` の 1 行。"""


def own_row(
    role: NodeRole,
    *,
    state: str = "running",
    kind: str = "serve",
    name: str | None = None,
    config_name: str = CONFIG_NAME,
    image: str = IMAGE_REF,
    owner: str = OWNER,
) -> str:
    """自分のラベルで絞った一覧 (`docker ps -a --format json`) の 1 行。

    **これは実機から採った見本ではない** (tasks.md 1.6 の「ここで採れないもの」)。項目の
    名前は `test_guards.py` / `test_lifecycle_start.py` の見本に合わせた。
    """
    labels = {
        LABEL_OWNER: owner,
        LABEL_CONFIG: config_name,
        LABEL_KIND: kind,
        LABEL_ROLE: role,
        LABEL_IMAGE: image,
        LABEL_STARTED_AT: STARTED_AT_LABEL,
        LABEL_CONFIG_SHA256: CONFIG_SHA256,
    }
    return (
        json.dumps(
            {
                "ID": CONTAINER_IDS[role],
                "Names": CONTAINER_NAMES[role] if name is None else name,
                "State": state,
                "Image": image,
                "Labels": ",".join(f"{key}={value}" for key, value in sorted(labels.items())),
            }
        )
        + "\n"
    )


def listings(**rows: str) -> Mapping[NodeRole, tuple[Reply, ...]]:
    """2 台ぶんの一覧の返事 (書かなかった台は、空の一覧)。"""
    return {role: (Reply(stdout=rows.get(role, "")),) for role in ROLES}


def running_listings() -> Mapping[NodeRole, tuple[Reply, ...]]:
    return listings(head=own_row("head"), worker=own_row("worker"))


@dataclass
class SpyConfirmer:
    """計画を見せて、決めたとおりに答える了承の口 (`guards.Confirmer` の形)。"""

    approves: bool = True
    shown: list[str] = field(default_factory=list)

    def confirm(self, plan_text: str) -> bool:
        self.shown.append(plan_text)
        return self.approves


@dataclass
class FakeClock:
    """試験用の時計。眠りは記録するだけで、実際には眠らない (時刻は、眠ったぶんだけ進む)。"""

    now: float = 0.0
    slept: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@dataclass
class OpsScript:
    """`serve status` / `serve stop` の台本 (既定: 何を読んでも空の成功)。"""

    listings: Mapping[NodeRole, tuple[Reply, ...]]
    gpu: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    links: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    states: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    stops: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    removes: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    extra: Sequence[Rule] = ()
    """いちばん先に見る規則 (1 つの呼び出しだけに、中断などを仕込むため)。"""

    def rules(self) -> tuple[Rule, ...]:
        rules: list[Rule] = list(self.extra)
        for role in ROLES:
            rules.append(Rule(prefix=OWN_CONTAINERS_ARGV, node=role, replies=self.listings[role]))
            for prefix, table in (
                (("nvidia-smi",), self.gpu),
                (("ip",), self.links),
                (("docker", "container", "inspect"), self.states),
                (("docker", "stop"), self.stops),
                (("docker", "rm"), self.removes),
            ):
                if table is not None:
                    rules.append(Rule(prefix=prefix, node=role, replies=table[role]))
        rules.extend(
            (
                Rule(prefix=("nvidia-smi",), replies=(Reply(stdout=""),)),
                Rule(prefix=("ip",), replies=(Reply(stdout=LINK_UP),)),
                Rule(
                    prefix=("docker", "container", "inspect"),
                    replies=(Reply(stdout="exited 1\n"),),
                ),
                Rule(prefix=("docker", "stop"), replies=(Reply(),)),
                Rule(prefix=("docker", "rm"), replies=(Reply(),)),
            )
        )
        return tuple(rules)


def runner_of(tmp_path: Path, script: OpsScript, *, default: Reply | None = None) -> FakeRunner:
    return FakeRunner(var_root=tmp_path / "var", script=script.rules(), default=default)


@pytest.fixture(autouse=True)
def no_real_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """実物の ssh / rsync / docker を呼ばないことと、実際に眠らないことを固定する。"""

    def explode(argv: Sequence[str], **kwargs: Any) -> None:
        raise AssertionError(f"実物のコマンドを呼んだ: {list(argv)}")

    def no_sleep(seconds: float) -> None:
        raise AssertionError(f"試験が実際に眠った: {seconds} 秒")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(time, "sleep", no_sleep)


# --- 呼び出しの読み取り ----------------------------------------------------


def steps(runner: FakeRunner) -> tuple[str, ...]:
    """記録された呼び出しを、短い名前の列にする (順序を見るため)。"""
    names: list[str] = []
    for call in runner.calls:
        argv = call.argv
        if call.kind != "run":
            names.append(call.kind)
        elif argv[0] != "docker":
            names.append(argv[0])
        elif argv[1] == "container":
            names.append("container inspect")
        else:
            names.append(argv[1])
    return tuple(names)


def mutating_argvs(runner: FakeRunner) -> tuple[tuple[str, ...], ...]:
    return tuple(call.argv for call in runner.calls if call.mutating)


def argv_of(runner: FakeRunner, *prefix: str) -> tuple[tuple[str, ...], ...]:
    return tuple(argv for argv in runner.argvs if argv[: len(prefix)] == prefix)


def container_targets(runner: FakeRunner) -> tuple[str, ...]:
    """コンテナを対象にした操作の、対象の語だけを取り出す (末尾の語)。"""
    watched = (("docker", "stop"), ("docker", "rm"), ("docker", "logs"))
    targets = [argv[-1] for argv in runner.argvs if argv[:2] in watched]
    targets.extend(argv[-1] for argv in argv_of(runner, "docker", "container", "inspect"))
    return tuple(targets)


def node_status(service: ServiceStatus, role: NodeRole) -> NodeStatus:
    """`ServiceStatus.nodes` から、1 台ぶんを取る。"""
    return next(item for item in service.nodes if item.node == role)


# --- 停止: 順序と、2 度目 --------------------------------------------------


def test_stop_stops_head_then_worker_and_then_removes(tmp_path: Path) -> None:
    """head → worker の順に `docker stop -t 90`、そのあと 2 台で `docker rm` (1.7)。"""
    runner = runner_of(tmp_path, OpsScript(listings=running_listings()))
    confirmer = SpyConfirmer()

    clock = FakeClock()
    outcome = lc.stop(
        runner,
        NODES,
        confirmer=confirmer,
        report=io.StringIO(),
        sleep=clock.sleep,
        clock=clock.monotonic,
    )

    assert outcome.status == "stopped"
    assert mutating_argvs(runner) == (
        ("docker", "stop", "-t", str(g.STOP_TIMEOUT_S), CONTAINER_NAMES["head"]),
        ("docker", "stop", "-t", str(g.STOP_TIMEOUT_S), CONTAINER_NAMES["worker"]),
        ("docker", "rm", CONTAINER_NAMES["head"]),
        ("docker", "rm", CONTAINER_NAMES["worker"]),
    )
    assert [call.node for call in runner.calls if call.mutating] == [
        "head",
        "worker",
        "head",
        "worker",
    ]
    assert len(confirmer.shown) == 1


def test_stop_uses_the_shared_stop_and_remove_argv(tmp_path: Path) -> None:
    """止める列は、必ず `guards.stop_argv` / `remove_argv` で作る (了承との完全な一致)。"""
    runner = runner_of(tmp_path, OpsScript(listings=running_listings()))

    lc.stop(runner, NODES, confirmer=SpyConfirmer(), report=io.StringIO())

    assert argv_of(runner, "docker", "stop") == tuple(
        g.stop_argv(CONTAINER_NAMES[role]) for role in ROLES
    )
    assert argv_of(runner, "docker", "rm") == tuple(
        g.remove_argv(CONTAINER_NAMES[role]) for role in ROLES
    )


def test_second_stop_is_already_stopped_without_changing_anything(tmp_path: Path) -> None:
    """2 度目の停止は、了承も求めず、状態を変える呼び出しも出さずに正常に終わる (1.8)。"""
    runner = runner_of(tmp_path, OpsScript(listings=listings()))
    confirmer = SpyConfirmer()

    outcome = lc.stop(runner, NODES, confirmer=confirmer, report=io.StringIO())

    assert outcome.status == "already_stopped"
    assert mutating_argvs(runner) == ()
    assert confirmer.shown == []
    assert steps(runner) == ("ps", "ps")


def test_stop_without_approval_runs_no_stop_and_no_remove(tmp_path: Path) -> None:
    """了承しなければ、`stop` も `rm` も 1 つも出ない (requirements 2.1)。"""
    runner = runner_of(tmp_path, OpsScript(listings=running_listings()))

    with pytest.raises(g.ApprovalError):
        lc.stop(runner, NODES, confirmer=SpyConfirmer(approves=False), report=io.StringIO())

    assert mutating_argvs(runner) == ()
    assert runner.plan is None


def test_stop_shows_the_plan_with_the_kind_and_the_logs_hint(tmp_path: Path) -> None:
    """了承の前に、どの台のどのコンテナを止めるか (種類も) と、記録の回収を促す文を見せる。"""
    rows = listings(head=own_row("head", kind="fetch"), worker=own_row("worker", kind="fetch"))
    runner = runner_of(tmp_path, OpsScript(listings=rows))
    confirmer = SpyConfirmer()
    report = io.StringIO()

    outcome = lc.stop(runner, NODES, confirmer=confirmer, report=report)

    assert outcome.status == "stopped"
    shown = report.getvalue() + "".join(confirmer.shown)
    assert CONTAINER_NAMES["head"] in shown
    assert "fetch" in shown
    assert "serve logs" in shown


# --- 停止: 対象は、自分の一覧の名前だけ ------------------------------------


def test_stop_targets_only_the_names_from_the_own_listing(tmp_path: Path) -> None:
    """ラベルのないコンテナにも、一覧にない名前にも、どの操作も向けない (2.3、2.4)。"""
    runner = runner_of(tmp_path, OpsScript(listings=running_listings()))

    lc.stop(runner, NODES, confirmer=SpyConfirmer(), report=io.StringIO())

    assert set(container_targets(runner)) == set(CONTAINER_NAMES.values())
    assert argv_of(runner, "docker", "ps") != ()
    for argv in runner.argvs:
        assert "-f" not in argv and "--force" not in argv


def test_stop_does_not_touch_a_node_whose_listing_is_refused(tmp_path: Path) -> None:
    """所有のラベルのない行が紛れた台では、何も止めない (一覧が信じられないため)。"""
    foreign = own_row("worker", owner="someone-else")
    rows = listings(head=own_row("head"), worker=foreign)
    runner = runner_of(tmp_path, OpsScript(listings=rows))

    with pytest.raises(lc.LifecycleError) as caught:
        lc.stop(runner, NODES, confirmer=SpyConfirmer(), report=io.StringIO())

    assert "worker" in str(caught.value)
    # head のぶんは片付け、worker には 1 つも向けない
    assert set(container_targets(runner)) == {CONTAINER_NAMES["head"]}
    assert [call.node for call in runner.calls if call.mutating] == ["head", "head"]


def test_stop_keeps_the_reachable_node_when_the_other_cannot_be_read(tmp_path: Path) -> None:
    """片方に入れないときは、読めた台だけを止め、読めなかった台を結果の文に出す (2.3、1.7)。"""
    rows: Mapping[NodeRole, tuple[Reply, ...]] = {
        "head": (Reply(stdout=own_row("head")),),
        "worker": (Reply(exit_code=255, stderr="ssh: connect timed out"),),
    }
    runner = runner_of(tmp_path, OpsScript(listings=rows))

    with pytest.raises(lc.LifecycleError) as caught:
        lc.stop(runner, NODES, confirmer=SpyConfirmer(), report=io.StringIO())

    message = str(caught.value)
    assert "worker" in message and "serve stop" in message
    assert set(container_targets(runner)) == {CONTAINER_NAMES["head"]}


def test_stop_asks_for_nothing_when_no_node_could_be_read(tmp_path: Path) -> None:
    """どの台の一覧も読めないときは、了承も求めず、何にも向けずに断る (requirements 2.3)。"""
    rows: Mapping[NodeRole, tuple[Reply, ...]] = {
        role: (Reply(exit_code=255, stderr="ssh: connect timed out"),) for role in ROLES
    }
    runner = runner_of(tmp_path, OpsScript(listings=rows))
    confirmer = SpyConfirmer()

    with pytest.raises(lc.LifecycleError) as caught:
        lc.stop(runner, NODES, confirmer=confirmer, report=io.StringIO())

    assert "head" in str(caught.value) and "worker" in str(caught.value)
    assert confirmer.shown == []
    assert mutating_argvs(runner) == ()
    assert runner.plan is None


def test_stop_does_not_collect_the_logs(tmp_path: Path) -> None:
    """記録の回収は `serve logs` の仕事なので、`stop` は読まない・写さない。"""
    runner = runner_of(tmp_path, OpsScript(listings=running_listings()))

    lc.stop(runner, NODES, confirmer=SpyConfirmer(), report=io.StringIO())

    assert argv_of(runner, "docker", "logs") == ()
    assert runner.pulls == ()
    assert runner.pushes == ()


# --- 停止: 終了したコンテナ、失敗、中断 ------------------------------------


def test_stop_uses_stop_then_remove_for_an_exited_container(tmp_path: Path) -> None:
    """終了したコンテナが残っているだけでも、列は `stop` → `rm` にする (道を 1 つにする)。"""
    rows = listings(head=own_row("head", state="exited"), worker=own_row("worker", state="exited"))
    runner = runner_of(tmp_path, OpsScript(listings=rows))

    outcome = lc.stop(runner, NODES, confirmer=SpyConfirmer(), report=io.StringIO())

    assert outcome.status == "stopped"
    assert [argv[1] for argv in mutating_argvs(runner)] == ["stop", "stop", "rm", "rm"]


def test_stop_raises_when_a_container_cannot_be_removed(tmp_path: Path) -> None:
    """止められなかった・消せなかったことは、名前を示して、実行しての失敗にする。"""
    removes: Mapping[NodeRole, tuple[Reply, ...]] = {
        "head": (Reply(exit_code=1, stderr="Error response from daemon: removal in progress"),),
        "worker": (Reply(),),
    }
    runner = runner_of(tmp_path, OpsScript(listings=running_listings(), removes=removes))

    with pytest.raises(lc.LifecycleError) as caught:
        lc.stop(runner, NODES, confirmer=SpyConfirmer(), report=io.StringIO())

    assert CONTAINER_NAMES["head"] in str(caught.value)
    # 失敗しても、残りの片付けには進む
    assert [argv[1] for argv in mutating_argvs(runner)] == ["stop", "stop", "rm", "rm"]


def test_stop_finishes_the_cleanup_after_an_interrupt(tmp_path: Path) -> None:
    """中断が来ても、台のすべてで `stop` → `rm` を試みてから、中断を伝える (3.4 の決まり)。"""
    interrupting = Rule(
        prefix=("docker", "stop"),
        node="head",
        replies=(Reply(raises=KeyboardInterrupt()),),
    )
    runner = runner_of(tmp_path, OpsScript(listings=running_listings(), extra=(interrupting,)))

    with pytest.raises(KeyboardInterrupt):
        lc.stop(runner, NODES, confirmer=SpyConfirmer(), report=io.StringIO())

    assert [argv[1] for argv in mutating_argvs(runner)] == ["stop", "stop", "rm", "rm"]
    assert argv_of(runner, "nvidia-smi") == ()


# --- 停止: GPU のプロセスを待つ --------------------------------------------


def test_stop_waits_until_the_gpu_processes_are_gone(tmp_path: Path) -> None:
    """止めたあと、GPU のプロセスが 0 件になるまで待つ (requirements 1.7)。"""
    gpu = {role: (Reply(stdout=GPU_LINE), Reply(stdout="")) for role in ROLES}
    runner = runner_of(tmp_path, OpsScript(listings=running_listings(), gpu=gpu))
    clock = FakeClock()

    outcome = lc.stop(
        runner,
        NODES,
        confirmer=SpyConfirmer(),
        report=io.StringIO(),
        sleep=clock.sleep,
        clock=clock.monotonic,
    )

    assert outcome.status == "stopped"
    assert outcome.remaining_gpu_apps == ()
    assert clock.slept == [lc.GPU_POLL_INTERVAL_S]
    assert len(argv_of(runner, "nvidia-smi")) == 4


def test_stop_reports_gpu_not_released_without_killing_anything(tmp_path: Path) -> None:
    """空かなければ、名前とメモリの量を示して `gpu_not_released`。止めには行かない。"""
    gpu = {role: (Reply(stdout=GPU_LINE),) for role in ROLES}
    runner = runner_of(tmp_path, OpsScript(listings=running_listings(), gpu=gpu))
    clock = FakeClock()

    outcome = lc.stop(
        runner,
        NODES,
        confirmer=SpyConfirmer(),
        report=io.StringIO(),
        sleep=clock.sleep,
        clock=clock.monotonic,
    )

    assert outcome.status == "gpu_not_released"
    assert (
        outcome.remaining_gpu_apps
        == (GpuApp(pid=12345, process_name="vllm::EngineCore", used_memory_mib=92160),) * 2
    )
    assert "vllm::EngineCore" in outcome.detail and "92160" in outcome.detail
    # 待った合計が、決めた上限に収まる。止める呼び出しは、片付けの 4 つだけ
    assert sum(clock.slept) <= lc.GPU_RELEASE_TIMEOUT_S
    assert len(mutating_argvs(runner)) == 4


def test_stop_says_when_the_gpu_processes_cannot_be_read(tmp_path: Path) -> None:
    """`nvidia-smi` が読めないときは、空いたとは言わず、確かめられなかったことを示す。"""
    gpu = {
        role: (Reply(exit_code=9, stderr="NVML: Driver/library version mismatch"),)
        for role in ROLES
    }
    runner = runner_of(tmp_path, OpsScript(listings=running_listings(), gpu=gpu))

    outcome = lc.stop(runner, NODES, confirmer=SpyConfirmer(), report=io.StringIO())

    assert outcome.status == "stopped"
    assert "確かめられなかった" in outcome.detail


# --- 状態の確認 ------------------------------------------------------------


def test_status_reports_both_nodes_and_the_service(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """2 台の状態と、受け付けの可否・名乗る名前・入力の長さの上限・要求の数を示す (1.6、6.5)。"""
    fake_vllm.set_max_model_len(163840)
    fake_vllm.set_metrics(MetricsSample(running_requests=2, waiting_requests=3))
    config = serve_config(port_of(fake_vllm))
    gpu = {role: (Reply(stdout=GPU_LINE),) for role in ROLES}
    links: Mapping[NodeRole, tuple[Reply, ...]] = {
        "head": (Reply(stdout=LINK_UP),),
        "worker": (Reply(stdout=LINK_DOWN),),
    }
    runner = runner_of(tmp_path, OpsScript(listings=running_listings(), gpu=gpu, links=links))

    service = lc.status(runner, configs_of(config), FABRIC_NODES, report=io.StringIO())

    head = node_status(service, "head")
    assert head.container_state == "running"
    assert head.config_name == CONFIG_NAME
    assert head.kind == "serve"
    assert head.image_digest == IMAGE_REF
    assert head.config_sha256 == CONFIG_SHA256
    assert head.started_at == STARTED_AT
    assert head.gpu_apps == (
        GpuApp(pid=12345, process_name="vllm::EngineCore", used_memory_mib=92160),
    )
    assert head.fabric_link_up is True
    assert node_status(service, "worker").fabric_link_up is False
    assert service.health_ok is True
    assert service.served_model == SERVED_MODEL
    assert service.max_model_len == 163840
    assert service.running_requests == 2
    assert service.waiting_requests == 3


def test_status_makes_no_mutating_calls(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """状態の確認は、状態を変える呼び出しを 1 つも出さない (design の cli の表)。"""
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, OpsScript(listings=running_listings()))

    lc.status(runner, configs_of(config), NODES, report=io.StringIO())

    assert mutating_argvs(runner) == ()
    assert runner.plan is None
    assert set(container_targets(runner)) <= set(CONTAINER_NAMES.values()) | set(
        CONTAINER_IDS.values()
    )


def test_status_when_nothing_is_running(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """何も動いていないときは、2 台とも `absent` で、HTTP にも行かずに示す。"""
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, OpsScript(listings=listings()))

    service = lc.status(runner, configs_of(config), NODES, report=io.StringIO())

    assert [item.container_state for item in service.nodes] == ["absent", "absent"]
    assert service.health_ok is None
    assert service.served_model is None
    assert service.max_model_len is None
    assert fake_vllm.requests == ()


def test_status_reads_the_exit_code_of_an_exited_container(tmp_path: Path) -> None:
    """終了したコンテナの終了コードは、一覧から来た識別子への `inspect` で読む。"""
    rows = listings(head=own_row("head", state="exited"), worker=own_row("worker", state="exited"))
    states = {role: (Reply(stdout="exited 3\n"),) for role in ROLES}
    runner = runner_of(tmp_path, OpsScript(listings=rows, states=states))

    service = lc.status(runner, configs_of(serve_config(UNUSED_PORT)), NODES, report=io.StringIO())

    assert [item.container_state for item in service.nodes] == ["exited", "exited"]
    assert [item.exit_code for item in service.nodes] == [3, 3]
    assert argv_of(runner, "docker", "container", "inspect")[0][-1] == CONTAINER_IDS["head"]


def test_status_when_one_node_cannot_be_reached(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """片方に入れなくても落ちずに、その台の項目を「読めなかった」として示す。"""
    config = serve_config(port_of(fake_vllm))
    rows: Mapping[NodeRole, tuple[Reply, ...]] = {
        "head": (Reply(stdout=own_row("head")),),
        "worker": (Reply(exit_code=255, stderr="ssh: connect timed out"),),
    }
    runner = runner_of(tmp_path, OpsScript(listings=rows), default=Reply(exit_code=255))
    report = io.StringIO()

    service = lc.status(runner, configs_of(config), NODES, report=report)

    assert node_status(service, "head").container_state == "running"
    worker = node_status(service, "worker")
    assert worker.container_state == "absent"
    assert worker.config_name is None
    assert worker.gpu_apps == ()
    assert worker.fabric_link_up is None
    assert "worker" in report.getvalue()
    assert service.health_ok is True


def test_status_when_the_gpu_processes_cannot_be_read(tmp_path: Path) -> None:
    """`nvidia-smi` が読めなくても落ちずに、その台の GPU のプロセスを空で示す。"""
    gpu = {role: (Reply(exit_code=9, stderr="NVML error"),) for role in ROLES}
    runner = runner_of(tmp_path, OpsScript(listings=running_listings(), gpu=gpu))
    report = io.StringIO()

    service = lc.status(runner, configs_of(serve_config(UNUSED_PORT)), NODES, report=report)

    assert [item.gpu_apps for item in service.nodes] == [(), ()]
    assert "nvidia-smi" in report.getvalue() or "GPU" in report.getvalue()


def test_status_when_the_server_does_not_answer(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """推論サーバーが読めないときは、断らずに、HTTP の項目を空で返す。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, OpsScript(listings=running_listings()))

    service = lc.status(runner, configs_of(config), NODES, report=io.StringIO())

    assert node_status(service, "head").container_state == "running"
    assert service.health_ok is None
    assert service.served_model is None
    assert service.running_requests is None


def test_status_when_the_config_of_the_running_container_is_unknown(tmp_path: Path) -> None:
    """ラベルの構成の名前が手元の定義にないときは、HTTP に行かずに、警告だけを出す。"""
    rows = listings(
        head=own_row("head", config_name="gone-from-the-file"),
        worker=own_row("worker", config_name="gone-from-the-file"),
    )
    runner = runner_of(tmp_path, OpsScript(listings=rows))
    report = io.StringIO()

    service = lc.status(runner, configs_of(serve_config(UNUSED_PORT)), NODES, report=report)

    assert node_status(service, "head").config_name == "gone-from-the-file"
    assert service.health_ok is None
    assert "gone-from-the-file" in report.getvalue()


def test_status_when_the_http_port_cannot_be_decided(tmp_path: Path) -> None:
    """HTTP の宛先を決められない構成でも、断らずに、台の状態だけを返す。"""
    runner = runner_of(tmp_path, OpsScript(listings=running_listings()))
    report = io.StringIO()

    service = lc.status(runner, configs_of(portless_config()), NODES, report=report)

    assert node_status(service, "head").container_state == "running"
    assert service.health_ok is None
    assert "--port" in report.getvalue()


def test_status_does_not_go_to_http_for_a_fetch_container(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """取得のコンテナが動いているだけのときは、推論サーバーの口に問い合わせない。"""
    rows = listings(head=own_row("head", kind="fetch"), worker=own_row("worker", kind="fetch"))
    runner = runner_of(tmp_path, OpsScript(listings=rows))

    service = lc.status(
        runner, configs_of(serve_config(port_of(fake_vllm))), NODES, report=io.StringIO()
    )

    assert node_status(service, "head").kind == "fetch"
    assert fake_vllm.requests == ()


def test_status_leaves_the_fabric_link_empty_without_an_interface_name(tmp_path: Path) -> None:
    """ノードの定義に直結のインターフェースがなければ、`ip` を読みに行かない。"""
    runner = runner_of(tmp_path, OpsScript(listings=running_listings()))

    service = lc.status(runner, configs_of(serve_config(UNUSED_PORT)), NODES, report=io.StringIO())

    assert [item.fabric_link_up for item in service.nodes] == [None, None]
    assert argv_of(runner, "ip") == ()


def test_status_reads_the_fabric_link_with_a_read_only_ip_command(tmp_path: Path) -> None:
    """直結のリンクは、`ip -br link show dev <名前>` の読み取りの形で読む。"""
    runner = runner_of(tmp_path, OpsScript(listings=running_listings()))

    lc.status(runner, configs_of(serve_config(UNUSED_PORT)), FABRIC_NODES, report=io.StringIO())

    assert argv_of(runner, "ip") == (("ip", "-br", "link", "show", "dev", FABRIC_IFNAME),) * 2


# --- 短い要求での確かめ ----------------------------------------------------

EN_REPLY = "The capital of France is Paris."
JA_REPLY = "日本の首都は東京です。"


def smoke_config(fake: FakeVllm) -> ConfigDef:
    return serve_config(port_of(fake))


def test_smoke_sends_one_english_and_one_japanese_request(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """英語と日本語を 1 つずつ、Anthropic の Messages API の形で送る (requirements 6.5)。"""
    fake_vllm.set_messages_sequence((MessagesReply(text=EN_REPLY), MessagesReply(text=JA_REPLY)))

    outcome = lc.smoke(smoke_config(fake_vllm), NODES, report=io.StringIO())

    sent = fake_vllm.requests_for("/v1/messages")
    assert len(sent) == 2
    bodies = [request.body for request in sent if request.body is not None]
    assert len(bodies) == 2, "要求の本文が JSON として届いていない"
    assert [body["model"] for body in bodies] == [SERVED_MODEL, SERVED_MODEL]
    assert [body["messages"][0]["content"] for body in bodies] == [
        lc.SMOKE_PROMPTS["en"],
        lc.SMOKE_PROMPTS["ja"],
    ]
    assert all(body["max_tokens"] == lc.SMOKE_MAX_TOKENS for body in bodies)
    assert [reply.lang for reply in outcome.replies] == ["en", "ja"]
    assert [reply.http_status for reply in outcome.replies] == [200, 200]
    assert [reply.stop_reason for reply in outcome.replies] == ["end_turn", "end_turn"]
    assert [reply.input_tokens for reply in outcome.replies] == [10, 10]


def test_smoke_sends_no_credentials(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """待ち受けに認証がないので、認証のヘッダを付けない (Security Considerations)。"""
    lc.smoke(smoke_config(fake_vllm), NODES, report=io.StringIO())

    for request in fake_vllm.requests_for("/v1/messages"):
        assert "authorization" not in request.headers
        assert "x-api-key" not in request.headers


def test_smoke_shows_the_body_but_writes_it_to_no_file(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """応答の本文は画面に出すが、どのファイルにも書かない (requirements 10.5)。"""
    fake_vllm.set_messages_sequence((MessagesReply(text=EN_REPLY), MessagesReply(text=JA_REPLY)))
    report = io.StringIO()
    var_root = tmp_path / "var"
    var_root.mkdir()

    outcome = lc.smoke(smoke_config(fake_vllm), NODES, report=report)
    # 5.1 が記録として書くとしたら、この形になる (本文が入らないことを型で固定する)
    (var_root / "smoke.json").write_text(outcome.model_dump_json(indent=2), encoding="utf-8")

    assert EN_REPLY in report.getvalue() and JA_REPLY in report.getvalue()
    assert EN_REPLY not in outcome.detail and JA_REPLY not in outcome.detail
    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert written, "書いたファイルが 1 つもないと、この試験は何も確かめていない"
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert EN_REPLY not in text
        assert JA_REPLY not in text


def test_smoke_flags_the_replacement_character(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """置き換え文字 (U+FFFD) を含む応答には、印が立つ (日本語の文字化けの見分け)。"""
    fake_vllm.set_messages_sequence(
        (MessagesReply(text=EN_REPLY), MessagesReply(text="日本の首�都は�"))
    )

    outcome = lc.smoke(smoke_config(fake_vllm), NODES, report=io.StringIO())

    assert [reply.replacement_char for reply in outcome.replies] == [False, True]
    assert "置き換え文字" in outcome.detail


def test_smoke_survives_a_failing_request(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """500 でも落ちずに、HTTP の状態を結果に入れる (人が画面で判断する)。"""
    fake_vllm.set_messages_fault_sequence((Fault(status=500), Fault()))

    outcome = lc.smoke(smoke_config(fake_vllm), NODES, report=io.StringIO())

    assert [reply.http_status for reply in outcome.replies] == [500, 200]
    assert outcome.replies[0].stop_reason is None


def test_smoke_survives_a_server_that_does_not_answer(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """つながらない・応答しないときも、落ちずに結果に入れる。"""
    fake_vllm.set_messages_fault(Fault(status=None))

    outcome = lc.smoke(smoke_config(fake_vllm), NODES, report=io.StringIO(), timeout_s=2.0)

    assert [reply.http_status for reply in outcome.replies] == [None, None]


def test_smoke_survives_a_reply_without_a_text_block(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """thinking のブロックだけの応答では、本文が空であることを結果に出す。"""
    fake_vllm.set_messages_fault(
        Fault(
            body=json.dumps(
                {
                    "id": "msg_fake",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "考えている"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 7, "output_tokens": 0},
                }
            )
        )
    )

    outcome = lc.smoke(smoke_config(fake_vllm), NODES, report=io.StringIO())

    assert [reply.http_status for reply in outcome.replies] == [200, 200]
    assert [reply.output_tokens for reply in outcome.replies] == [0, 0]
    assert "空" in outcome.detail


def test_smoke_survives_a_broken_body(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """JSON として読めない応答でも、落ちずに結果に入れる。"""
    fake_vllm.set_messages_fault(Fault(body="{not json"))

    outcome = lc.smoke(smoke_config(fake_vllm), NODES, report=io.StringIO())

    assert [reply.http_status for reply in outcome.replies] == [200, 200]
    assert [reply.stop_reason for reply in outcome.replies] == [None, None]


def test_smoke_refuses_a_config_that_is_not_a_server(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """`serve smoke` が受けるのは、推論サーバーの構成だけである。"""
    base = smoke_config(fake_vllm)
    fetch = base.model_copy(update={"kind": "fetch", "served_model_name": None})

    with pytest.raises(ConfigError):
        lc.smoke(fetch, NODES, report=io.StringIO())

    assert fake_vllm.requests == ()


# --- 止める計画の口 (guards への追加) --------------------------------------


def test_stop_plan_only_holds_stop_and_remove_for_listed_names(tmp_path: Path) -> None:
    """`serve stop` の計画は、一覧の行の名前の `stop` / `rm` しか持たない (2.3)。"""
    runner = runner_of(tmp_path, OpsScript(listings=running_listings()))

    lc.stop(runner, NODES, confirmer=SpyConfirmer(), report=io.StringIO())

    plan = runner.plan
    assert plan is not None
    assert plan.rollback == ()
    assert set(plan.own_container_names) == set(CONTAINER_NAMES.values())
    commands = [item for item in plan.forward if isinstance(item, PlannedRun)]
    assert len(commands) == len(plan.forward), "止める計画に、配布などの別の呼び出しが入っている"
    assert [command.argv[1] for command in commands] == ["stop", "stop", "rm", "rm"]
    for command in commands:
        assert command.argv[-1] in CONTAINER_NAMES.values()


def test_stop_refuses_when_there_is_no_node_definition(tmp_path: Path) -> None:
    """見る台の定義が 1 つもなければ、Spark に触る前に断る。"""
    runner = runner_of(tmp_path, OpsScript(listings=running_listings()))

    with pytest.raises(ValueError):
        lc.stop(runner, {}, confirmer=SpyConfirmer(), report=io.StringIO())

    assert runner.calls == ()


def test_remote_error_while_waiting_for_the_gpu_is_not_a_crash(tmp_path: Path) -> None:
    """GPU の読み取りが届かなくても、停止の結果を返す (片付けは終わっている)。"""
    gpu = {
        role: (Reply(raises=RemoteError("切れた", node=role, ssh_host="x", argv=())),)
        for role in ROLES
    }
    runner = runner_of(tmp_path, OpsScript(listings=running_listings(), gpu=gpu))

    outcome = lc.stop(runner, NODES, confirmer=SpyConfirmer(), report=io.StringIO())

    assert outcome.status == "stopped"


def test_stop_reads_the_listing_once_per_node(tmp_path: Path) -> None:
    """一覧は、台ごとに 1 度だけ読む (計画を、その 1 回の読み取りから作る)。"""
    runner = runner_of(tmp_path, OpsScript(listings=running_listings()))

    lc.stop(runner, NODES, confirmer=SpyConfirmer(), report=io.StringIO())

    assert len(argv_of(runner, "docker", "ps")) == 2


def test_callable_defaults_are_replaceable(tmp_path: Path) -> None:
    """時計と眠りは差し替えられる (試験は、実際に眠らない)。"""
    assert isinstance(lc.GPU_RELEASE_TIMEOUT_S, float)
    assert isinstance(lc.GPU_POLL_INTERVAL_S, float)
    assert lc.GPU_RELEASE_TIMEOUT_S == 60.0
    used: list[float] = []

    def sleeper(seconds: float) -> None:
        used.append(seconds)

    gpu = {role: (Reply(stdout=GPU_LINE), Reply(stdout="")) for role in ROLES}
    runner = runner_of(tmp_path, OpsScript(listings=running_listings(), gpu=gpu))
    clock = FakeClock()

    lc.stop(
        runner,
        NODES,
        confirmer=SpyConfirmer(),
        report=io.StringIO(),
        sleep=sleeper,
        clock=clock.monotonic,
    )

    assert used == [lc.GPU_POLL_INTERVAL_S]


def test_status_and_stop_do_not_import_the_upper_layers() -> None:
    """依存の向き: `lifecycle` は、`probe` / `netcheck` / `watch` / `thinking` / `cli` を
    読み込まない。"""
    source = Path(lc.__file__).read_text(encoding="utf-8")
    for name in ("probe", "netcheck", "watch", "thinking", "cli"):
        assert f"from serving_kit.{name}" not in source
        assert f"import serving_kit.{name}" not in source
