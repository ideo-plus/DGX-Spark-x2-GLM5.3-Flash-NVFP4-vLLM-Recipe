"""通信の確認のジョブの試験 (帯域、事前の確認、A/B。tasks.md 4.4)。

確かめること (design.md 「確認 › netcheck」、requirements 4.3、4.4、4.5、4.6、tasks.md 4.4 の
完了の状態):

- **不合格**: ふつうのネットワークに落ちた記録 (`Using network Socket`、`NET/Socket` の
  チャンネル) に対して、帯域の計測が不合格になる (requirements 4.4)
- **合格**: 高速の直結の経路の記録で合格になり、結果に、大きさごとの `busbw` と、比べる相手
  (NVIDIA の公表の実測としきい値) と、**道具が違うこと**が入る (requirements 4.3、4.6)
- NCCL の記録が読めない (回収できない) ときは、合格にしない (「確かめられなかった」と言う)
- コンテナが 0 以外で終わる、終わらずに時間切れになる → 不合格で、その旨が結果に出る
- **事前の確認**: 4 段がすべて通る / 3 段目で止まる (どこまで通ったかが結果に出る。4.6)
- **A/B**: 範囲が重なると「採用できない」、重ならない (B の最小 > A の最大) と「採用できる」
  (requirements 4.5)
- A/B の 6 回が、交互の順 (A1、B1、A2、B2、A3、B3) で流れ、コンテナの名前に回の番号、ラベルに
  腕と回が付き、**足した環境変数は B の回にだけ**ある
- A/B の**了承は 1 回**で、見せた計画に、6 回ぶん (12 個) の `docker run` と、その巻き戻しが
  すべてある。了承しないと、`docker run` が 1 つも出ない (requirements 2.1)
- どの結果でも、コンテナが残らない (記録の回収 → `stop` → `rm`)。途中の回が失敗すると、残りの
  回が流れない
- コンテナを対象にする操作が、自分の一覧の ID か、了承済みの計画の名前だけを対象にしている
  (requirements 2.3、2.4)
- 了承のあとの、状態を変える呼び出しが、見せた計画の列と完全に一致する
  (`FakeRunner` に `default` を渡さないので、台本にない呼び出しは誤りになる)
- `kind` が `job` でない構成、1 台の構成は、Spark に触る前に断る
- **回ごとに、NCCL の記録のファイルの名前を分ける** (`nccl-<回の札>.%h.%p.log` に上書きする)。
  その回の札のファイルがない台は「確かめられなかった」で、**古い回のファイルは読まない**
- 合否の連言を、1 つずつ固定する: 経路が `Socket` / `network=IB` でも `via NET/Socket` の
  チャンネルがある / `NET/IB : No device found.` がある / **片方の台だけが落ちた** →
  どれも `failed` (requirements 4.4、research.md §e-4 の「判定は連言で行う」)
- rank 0 の JSON の `world_size` が台の数と合わない、`busbw` が 0 以下 → `failed`
- 記録が取れない構成 (`NCCL_DEBUG` が INFO 未満、`NCCL_DEBUG_FILE` がない・結び付けの外) は、
  Spark に触る前に断る
- 片方の台が 0 以外で終わったら、**ほかの台を待たずに**打ち切る (rendezvous の相手は固まる)
- 比べる相手の値と、出典と、原文が、research.md の記述と一致している
- **A/B の回ごと・台ごとの経路の観察が `AbReport` から読める**。どれかの回で、どれかの台の
  記録が読めなければ、値を比較に使わずに止まる。採否は速さだけで決めるが、経路が高速でない
  回があれば、まとめと注意を `detail` に書く
- 回の札は、前後を固定した一致で絞る (`nccl-<札>.` で始まるファイルだけ)。`baseline-01` が
  `baseline-10` のファイルを読まない。札に、起こす時刻のマイクロ秒まで入る
- `logs/` の結び付けが読み取り専用 (`readonly` / `ro`) の構成を、Spark に触る前に断る
- 1 台の記録に `Using network IB` と `Using network Socket` の両方があれば、合格にしない
  (`observe_nccl` の `network` は、最初の行だけを見るので、この module が本文で補う)

実物の ssh、rsync、docker には、どの段でもつながない (`FakeRunner` を相手にする)。試験は、
実際に眠らない (待つ間隔と時計は、引数で差し替える)。
"""

from __future__ import annotations

import io
import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any, TextIO

import pytest
from pydantic import HttpUrl

from fake_runner import FakeRunner, RecordedCall, Reply, Rule
from serving_kit import netcheck as nc
from serving_kit.config import ConfigError
from serving_kit.guards import ApprovalError, remove_argv, stop_argv
from serving_kit.plan import LABEL_RUN, OWNER_FILTER, build_plans
from serving_kit.types import (
    ConfigDef,
    ContainerPlan,
    ImageRef,
    NodeDef,
    NodeRole,
    PlannedRun,
    Setting,
)

# --- 見本の値 -------------------------------------------------------------

DIGEST = "b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
IMAGE_REF = f"vllm/vllm-openai@sha256:{DIGEST}"
REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"
BANDWIDTH_CONFIG = "netcheck-bandwidth"
SANITY_CONFIG = "netcheck-sanity"
RDZV_PORT = 29502
"""torchrun の rendezvous のポート (推論サーバーの `--master-port` と別の値)。"""

STARTED_AT = datetime(2026, 9, 22, 6, 0, 0, tzinfo=UTC)

SOURCE = HttpUrl("https://docs.vllm.ai/en/latest/usage/troubleshooting/")
QUOTE = "torchrun --nnodes 2 --nproc-per-node=1 --rdzv_backend=static"

HEAD_ID = "aaaa11112222"
WORKER_ID = "bbbb33334444"
CONTAINER_IDS: Mapping[NodeRole, str] = {"head": HEAD_ID, "worker": WORKER_ID}

MIB = 1 << 20
GIB = 1 << 30

NCCL_IB_LOG = (
    "spark-153d:7:7 [0] NCCL INFO NCCL version 2.30.7+cuda13.0\n"
    "spark-153d:7:19 [0] NCCL INFO NET/IB : Using [0]rocep1s0f0:1/RoCE"
    " [1]roceP2p1s0f0:1/RoCE; OOB enp1s0f0np0:192.168.100.1<0>\n"
    "spark-153d:7:19 [0] NCCL INFO TOPO/NET : Made vNic 0\n"
    "spark-153d:7:19 [0] NCCL INFO Using network IB\n"
    "spark-153d:7:19 [0] NCCL INFO 8 coll channels, 0 collnet channels, 8 p2p channels\n"
    "spark-153d:7:19 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0\n"
)
"""高速の直結の経路が使われた記録 (research.md §e-4 の探す文字列を含む)。"""

NCCL_SOCKET_LOG = (
    "spark-153d:7:7 [0] NCCL INFO NCCL version 2.30.7+cuda13.0\n"
    "spark-153d:7:19 [0] NCCL INFO NET/IB : No device found.\n"
    "spark-153d:7:19 [0] NCCL INFO Using network Socket\n"
    "spark-153d:7:19 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Socket/0\n"
)
"""ふつうのネットワークの経路に落ちた記録 (research.md §e-4 の判定の連言)。"""

NCCL_IB_NO_DEVICE_LOG = (
    "spark-153d:7:7 [0] NCCL INFO NCCL version 2.30.7+cuda13.0\n"
    "spark-153d:7:19 [0] NCCL INFO NET/IB : No device found.\n"
    "spark-153d:7:19 [0] NCCL INFO Using network IB\n"
)
"""経路は IB と言いながら、IB のデバイスが見つかっていない記録 (research.md §e-4 の連言)。"""

NCCL_IB_THEN_SOCKET_LOG = (
    "spark-153d:7:7 [0] NCCL INFO NCCL version 2.30.7+cuda13.0\n"
    "spark-153d:7:19 [0] NCCL INFO Using network IB\n"
    "spark-153d:7:19 [0] NCCL INFO 8 coll channels, 0 collnet channels, 8 p2p channels\n"
    "spark-153d:7:31 [0] NCCL INFO Using network Socket\n"
)
"""1 つの記録に、経路の行が 2 度ある記録 (2 度目はふつうのネットワークの経路)。

`observe_nccl` の `network` は最初の行だけを見るので、この module が本文で補う (指摘 4)。
"""

NCCL_IB_WITH_SOCKET_CHANNEL_LOG = NCCL_IB_LOG + (
    "spark-153d:7:19 [0] NCCL INFO Channel 01/0 : 1[0] -> 0[0] [receive] via NET/Socket/0\n"
)
"""経路は IB でも、ふつうのネットワークの経路のチャンネルが 1 つある記録。"""

ADDED_ENV: Mapping[str, str] = {"NCCL_IB_QPS_PER_CONNECTION": "4"}
"""A/B で足す環境変数の見本 (research.md §e-7 の候補の 1 つ)。"""

STAMP = "20260922T060000-000000Z"
"""`STARTED_AT` の UTC の札 (`%Y%m%dT%H%M%S-%fZ`)。回の札の先頭に付く。

**マイクロ秒まで入る** (指摘 2 の直し。同じ秒に 2 度流しても、札が分かれる)。
"""

LOGS_TARGET = "/logs"
"""NCCL の記録を書く場所 (構成の `--mount` の `target`)。"""


def round_tag(label: str, repeat: int) -> str:
    """回の札 (起こす時刻 + 腕 (1 回だけの確認は `sanity`) + 回の番号。番号は 2 桁にそろえる)。"""
    return f"{STAMP}-{label}-{repeat:02d}"


def nccl_env(tag: str) -> dict[str, str]:
    """この道具が、回ごとに上書きする NCCL の記録のファイルの名前 (指摘 1 の直し)。"""
    return {"NCCL_DEBUG_FILE": f"{LOGS_TARGET}/nccl-{tag}.%h.%p.log"}


BANDWIDTH_TAG = round_tag(nc.BASELINE_ARM, 1)
SANITY_TAG = round_tag("sanity", 1)


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

HEAD = NodeDef(
    role="head",
    ssh_host="spark-153d",
    lan_addr=IPv4Address("10.0.1.60"),
    remote_root=REMOTE_ROOT,
    fabric_addr=IPv4Address("192.168.100.1"),
    fabric_ifname="enp1s0f0np0",
    fabric_measured="docs/results/2026-09-22-netcheck-links.md",
)
WORKER = NodeDef(
    role="worker",
    ssh_host="spark-5083",
    lan_addr=IPv4Address("10.0.1.61"),
    remote_root=REMOTE_ROOT,
    fabric_addr=IPv4Address("192.168.100.2"),
    fabric_ifname="enp1s0f0np0",
    fabric_measured="docs/results/2026-09-22-netcheck-links.md",
)
NODES: Mapping[NodeRole, NodeDef] = {"head": HEAD, "worker": WORKER}


def job_config(
    *,
    name: str = BANDWIDTH_CONFIG,
    script: str = "/payload/allreduce_bench.py",
    kind: str = "job",
    nodes: tuple[NodeRole, ...] = ("head", "worker"),
    ready_timeout_s: int = 600,
) -> ConfigDef:
    """2 台の通信の確認のジョブの構成 (design.md 「netcheck」の `kind = "job"`)。

    コンテナの中で動かすプログラムは docker の設定 (`--entrypoint torchrun`) で表し、
    rendezvous のポートは `is_port` の設定として明示し、NCCL の記録は結び付けた `logs/` の
    下に書かせる (実際の値は 6.2 が書く。ここでは試験のために縮めたものを作る)。
    """
    payload_mount = "type=bind,source={remote_root}/payload,target=/payload,readonly"
    logs_mount = "type=bind,source={remote_root}/logs,target=/logs"
    return ConfigDef(
        name=name,
        kind=kind,  # type: ignore[arg-type]  # `kind` が job でない構成も作って断らせる
        description="2 台の通信の確認 (試験用に縮めたもの)",
        nodes=nodes,
        image=IMAGE,
        docker={
            "gpus": _setting("--gpus", "all"),
            "network": _setting("--network", "host"),
            "ipc": _setting("--ipc", "host"),
            "entrypoint": _setting("--entrypoint", "torchrun"),
            "mount-payload": _setting("--mount", payload_mount),
            "mount-logs": _setting("--mount", logs_mount),
        },
        args={
            "nnodes": _setting("--nnodes", "2"),
            "nproc-per-node": _setting("--nproc-per-node", "1"),
            "rdzv-backend": _setting("--rdzv_backend", "static"),
            "master-addr": _setting("--master-addr", "{head.fabric_addr}"),
            "master-port": _setting("--master-port", str(RDZV_PORT), is_port=True),
            "node-rank": _setting("--node-rank", "{node.rank}"),
            "script": _setting(value=script),
        },
        env={
            "nccl-debug": _setting("NCCL_DEBUG", "INFO"),
            "nccl-debug-subsys": _setting("NCCL_DEBUG_SUBSYS", "INIT,BOOTSTRAP,ENV,NET,GRAPH"),
            "nccl-debug-file": _setting("NCCL_DEBUG_FILE", "/logs/nccl_%h_%p.log"),
            "nccl-socket-ifname": _setting("NCCL_SOCKET_IFNAME", "={node.fabric_ifname}"),
        },
        ready_timeout_s=ready_timeout_s,
        # 名乗るモデルの名前を持てるのは、`kind` が serve と probe の構成だけ (types の決まり)
        served_model_name="glm-5-3-flash" if kind in ("serve", "probe") else None,
    )


def plans_of(config: ConfigDef, *, tag: str = BANDWIDTH_TAG) -> tuple[ContainerPlan, ...]:
    """1 回ぶんの計画 (この道具が、NCCL の記録のファイルの名前を、回ごとに上書きする)。"""
    return build_plans(config, NODES, STARTED_AT, extra_env=nccl_env(tag))


def ab_order(repeats: int = 3) -> tuple[tuple[str, int], ...]:
    """A/B の回の順 (A1、B1、A2、B2、A3、B3)。"""
    return tuple(
        (arm, repeat)
        for repeat in range(1, repeats + 1)
        for arm in (nc.BASELINE_ARM, nc.CANDIDATE_ARM)
    )


def ab_rounds(
    config: ConfigDef, extra_env: Mapping[str, str], *, repeats: int = 3
) -> tuple[tuple[ContainerPlan, ...], ...]:
    """A/B の回ごとの計画 (交互の順 A1、B1、A2、B2、A3、B3)。

    足す環境変数は B の回だけで、NCCL の記録のファイルの名前の上書きは、**どちらの腕にも**
    入る (指摘 1 の直し。腕 A の通信のふるまいは変わらない)。
    """
    rounds: list[tuple[ContainerPlan, ...]] = []
    for arm, repeat in ab_order(repeats):
        added = {} if arm == nc.BASELINE_ARM else dict(extra_env)
        rounds.append(
            build_plans(
                config,
                NODES,
                STARTED_AT,
                arm=arm,
                repeat_index=repeat,
                extra_env={**added, **nccl_env(round_tag(arm, repeat))},
            )
        )
    return tuple(rounds)


# --- コンテナの出力の見本 ---------------------------------------------------


def full_sizes() -> dict[int, float]:
    """design の決まりの列 (1 MiB から 1 GiB まで 4 倍ずつの 6 つ)。"""
    return {MIB << (2 * index): 100.0 + index for index in range(6)}


def bench_json(busbw: Mapping[int, float] | None = None, *, world_size: int = 2) -> str:
    """`allreduce_bench.py` が rank 0 の標準出力に出す、1 行の JSON (4.2 の出力の形)。"""
    values = {MIB: 20.0, GIB: 176.2} if busbw is None else busbw
    samples = [
        {
            "size_bytes": size,
            "time_s": {"n": 20, "mean": 0.001, "min": 0.0009, "max": 0.0012, "stdev": 0.0001},
            "algbw_gbps": value,
            "busbw_gbps": value,
        }
        for size, value in sorted(values.items())
    ]
    report = {
        "world_size": world_size,
        "dtype": "float32",
        "warmup_iters": 5,
        "measure_iters": 20,
        "torch_version": "2.9.0+cu130",
        "nccl_version": "2.30.7",
        "samples": samples,
    }
    return json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def bench_stdout(busbw: Mapping[int, float] | None = None, *, world_size: int = 2) -> str:
    """rank 0 の標準出力 (JSON の 1 行のほかに、関係のない行が混ざっている)。"""
    return (
        "W0922 06:00:01 torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS to 1\n"
        '{"note": "これは結果の JSON ではない"}\n' + bench_json(busbw, world_size=world_size) + "\n"
    )


def sanity_stdout(stages: int) -> str:
    """事前の確認の、`stages` 段目まで通った出力 (成功の文面は `payload` の本文のもの)。"""
    return "".join(f"{stage.marker}\n" for stage in nc.SANITY_STAGES[:stages])


# --- 台本 -----------------------------------------------------------------

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


def ps_row(plan: ContainerPlan, *, state: str) -> str:
    """この計画のコンテナの、自分のラベルで絞った一覧の 1 行。"""
    return (
        json.dumps(
            {
                "ID": CONTAINER_IDS[plan.node],
                "Names": plan.container_name,
                "State": state,
                "Image": IMAGE_REF,
                "Labels": ",".join(f"{key}={value}" for key, value in sorted(plan.labels.items())),
            }
        )
        + "\n"
    )


@dataclass(frozen=True)
class Round:
    """1 回ぶんの台本 (2 台の計画と、その回のコンテナの出力と終わり方)。"""

    plans: Sequence[ContainerPlan]
    tag: str = BANDWIDTH_TAG
    """この回の札 (回収する NCCL の記録のファイルの名前に入る)。"""

    stdout: Mapping[NodeRole, str] = field(default_factory=dict)
    states: Mapping[NodeRole, tuple[Reply, ...]] = field(default_factory=dict)
    """`docker container inspect` の返事 (既定は 1 回で `exited 0`)。"""

    nccl: Mapping[NodeRole, str] | None = None
    """この回の札のファイルとして回収できる NCCL の記録 (`None` なら、この回のものがない)。"""

    stale: Mapping[NodeRole, Mapping[str, str]] = field(default_factory=dict)
    """`logs/` に残っている、**ほかの回**のファイル (名前 → 中身)。Spark の `logs/` を消す道は
    ないので、前の回や前の呼び出しのファイルが積もっている状態を作る。"""


@dataclass
class JobScript:
    """通信の確認のジョブの台本 (既定: 関門が通り、2 台が起きて終わり、片付けもできる)。"""

    rounds: Sequence[Round]
    extra: Sequence[Rule] = ()
    """いちばん先に見る規則 (1 つの呼び出しだけに、中断などを仕込むため)。"""

    gpu_apps: str = ""
    avail: str = "Avail\n999999999999\n"
    listening: str = ""
    layout_ok: bool = True

    def rules(self) -> tuple[Rule, ...]:
        rules: list[Rule] = list(self.extra)
        # 1 回も流さない台本 (触る前に断る試験) では、台ごとの規則を作らない (返事のない規則を
        # 残すと、うっかり Spark に触る道に入ったときの誤りが読みにくくなる)
        for role in ("head", "worker") if self.rounds else ():
            listings = [""]  # 関門の 1 回目は、自分のコンテナが 1 つもない
            states: list[Reply] = []
            outputs: list[Reply] = []
            pulls: list[Reply] = []
            for item in self.rounds:
                plan = next(plan for plan in item.plans if plan.node == role)
                # 起こした直後は running、終わったあとの 2 回 (回収、片付け) は exited
                listings.append(ps_row(plan, state="running"))
                listings.extend([ps_row(plan, state="exited")] * 2)
                states.extend(item.states.get(role, (Reply(stdout="exited 0\n"),)))
                outputs.append(Reply(stdout=item.stdout.get(role, "")))
                writes: list[tuple[str, str]] = []
                # 台ごとに書かない (その台の、その回の記録がない) 形も作れるようにする
                text = None if item.nccl is None else item.nccl.get(role)
                if text is not None:
                    writes.append((f"logs/nccl-{item.tag}.{role}.7.log", text))
                writes.extend(
                    (f"logs/{name}", text) for name, text in item.stale.get(role, {}).items()
                )
                pulls.append(Reply(writes=tuple(writes)) if writes else Reply(exit_code=23))
            rules.extend(
                (
                    Rule(
                        prefix=OWN_CONTAINERS_ARGV,
                        node=role,
                        replies=tuple(Reply(stdout=text) for text in listings),
                    ),
                    Rule(
                        prefix=("docker", "container", "inspect"), node=role, replies=tuple(states)
                    ),
                    # 記録の全量の回収 (`--timestamps` 付き) と、こちらが読む出力を分ける
                    Rule(
                        prefix=("docker", "logs", "--timestamps"),
                        node=role,
                        replies=(Reply(stdout="2026-09-22T06:00:01Z (記録の全量)\n"),),
                    ),
                    Rule(prefix=("docker", "logs"), node=role, replies=tuple(outputs)),
                    Rule(kind="pull", node=role, remote_prefix="logs", replies=tuple(pulls)),
                )
            )
        rules.extend(
            (
                Rule(prefix=("uname",), replies=(Reply(stdout="spark-153d\n"),)),
                Rule(prefix=("nvidia-smi",), replies=(Reply(stdout=self.gpu_apps),)),
                Rule(prefix=("test", "-d"), replies=(Reply(exit_code=0 if self.layout_ok else 1),)),
                Rule(
                    prefix=("docker", "image", "inspect"),
                    replies=(Reply(stdout=json.dumps([IMAGE_REF]) + "\n"),),
                ),
                Rule(prefix=("df",), replies=(Reply(stdout=self.avail),)),
                Rule(prefix=("ss",), replies=(Reply(stdout=self.listening),)),
                Rule(prefix=("docker", "run"), replies=(Reply(stdout=f"{HEAD_ID}\n"),)),
                Rule(prefix=("docker", "stop"), replies=(Reply(),)),
                Rule(prefix=("docker", "rm"), replies=(Reply(),)),
                # 起動の記録 (`state/`) は、ジョブでは置かないので、回収では欠ける
                Rule(kind="pull", remote_prefix="state", replies=(Reply(exit_code=23),)),
            )
        )
        return tuple(rules)


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
    on_sleep: Callable[[], None] | None = None

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()


def var_root(tmp_path: Path) -> Path:
    return tmp_path / "var"


def runner_of(tmp_path: Path, script: JobScript, *, default: Reply | None = None) -> FakeRunner:
    return FakeRunner(var_root=var_root(tmp_path), script=script.rules(), default=default)


def run_bandwidth(
    runner: FakeRunner,
    config: ConfigDef,
    tmp_path: Path,
    *,
    confirmer: SpyConfirmer | None = None,
    clock: FakeClock | None = None,
    report: TextIO | None = None,
    **extra: Any,
) -> nc.BandwidthOutcome:
    """既定の下ごしらえで `netcheck.run_bandwidth` を流す助け。"""
    used = FakeClock() if clock is None else clock
    return nc.run_bandwidth(
        runner,
        config,
        NODES,
        STARTED_AT,
        confirmer=SpyConfirmer() if confirmer is None else confirmer,
        var_root=var_root(tmp_path),
        poll_interval_s=10.0,
        sleep=used.sleep,
        clock=used.monotonic,
        report=io.StringIO() if report is None else report,
        **extra,
    )


def run_sanity(
    runner: FakeRunner,
    config: ConfigDef,
    tmp_path: Path,
    *,
    confirmer: SpyConfirmer | None = None,
    clock: FakeClock | None = None,
    report: TextIO | None = None,
    **extra: Any,
) -> nc.SanityOutcome:
    """既定の下ごしらえで `netcheck.run_sanity` を流す助け。"""
    used = FakeClock() if clock is None else clock
    return nc.run_sanity(
        runner,
        config,
        NODES,
        STARTED_AT,
        confirmer=SpyConfirmer() if confirmer is None else confirmer,
        var_root=var_root(tmp_path),
        poll_interval_s=10.0,
        sleep=used.sleep,
        clock=used.monotonic,
        report=io.StringIO() if report is None else report,
        **extra,
    )


def run_ab(
    runner: FakeRunner,
    config: ConfigDef,
    tmp_path: Path,
    *,
    extra_env: Mapping[str, str] | None = None,
    confirmer: SpyConfirmer | None = None,
    clock: FakeClock | None = None,
    report: TextIO | None = None,
    **extra: Any,
) -> nc.AbReport:
    """既定の下ごしらえで `netcheck.run_ab` を流す助け。"""
    used = FakeClock() if clock is None else clock
    return nc.run_ab(
        runner,
        config,
        NODES,
        STARTED_AT,
        extra_env=ADDED_ENV if extra_env is None else extra_env,
        confirmer=SpyConfirmer() if confirmer is None else confirmer,
        var_root=var_root(tmp_path),
        poll_interval_s=10.0,
        sleep=used.sleep,
        clock=used.monotonic,
        report=io.StringIO() if report is None else report,
        **extra,
    )


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
        if call.kind != "run":
            names.append(call.kind)
            continue
        argv = call.argv
        if argv[0] != "docker":
            names.append(argv[0])
        elif argv[1] in ("container", "image"):
            names.append(f"{argv[1]} inspect")
        elif argv[1] == "logs":
            names.append("logs all" if "--timestamps" in argv else "logs")
        else:
            names.append(argv[1])
    return tuple(names)


def mutating_calls(runner: FakeRunner) -> tuple[RecordedCall, ...]:
    return tuple(call for call in runner.calls if call.mutating)


def argv_of(runner: FakeRunner, *prefix: str) -> tuple[tuple[str, ...], ...]:
    return tuple(argv for argv in runner.argvs if argv[: len(prefix)] == prefix)


def started_names(runner: FakeRunner) -> tuple[str, ...]:
    """流れた `docker run` の `--name` の値 (起こされた順)。"""
    names: list[str] = []
    for argv in argv_of(runner, "docker", "run"):
        names.extend(value for flag, value in zip(argv, argv[1:], strict=False) if flag == "--name")
    return tuple(names)


def cleanup_order(runner: FakeRunner) -> list[str]:
    """片付けの道すじだけを取り出す (回収 → 停止 → 削除)。"""
    watched = ("logs all", "pull", "stop", "rm")
    return [name for name in steps(runner) if name in watched]


CLEANUP_STEPS = [
    "logs all",
    "pull",
    "pull",
    "logs all",
    "pull",
    "pull",
    "stop",
    "stop",
    "rm",
    "rm",
]
"""2 台ぶんの片付けの道すじ。

台ごとに、記録の全量 (`docker logs --timestamps`) → 通信の記録 → 起動の記録を回収し、
そのあと 2 台を止めて消す (3.4 の片付けの決まり)。
"""


def container_targets(runner: FakeRunner) -> tuple[str, ...]:
    """コンテナを対象にする呼び出しの、対象の文字列だけ。"""
    targets: list[str] = []
    for argv in runner.argvs:
        heads = (("docker", "logs"), ("docker", "stop"), ("docker", "rm"))
        if argv[:2] in heads or argv[:3] == ("docker", "container", "inspect"):
            targets.append(argv[-1])
    return tuple(targets)


# --- 帯域の計測: 合否 (requirements 4.4) -----------------------------------


def test_socket_fallback_is_not_passed(tmp_path: Path) -> None:
    """ふつうのネットワークに落ちた記録に対して、不合格になる (tasks.md 4.4 の完了の状態)。"""
    config = job_config()
    plans = plans_of(config)
    script = JobScript(
        rounds=[
            Round(
                plans=plans,
                stdout={"head": bench_stdout(), "worker": ""},
                nccl={"head": NCCL_SOCKET_LOG, "worker": NCCL_SOCKET_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert "Socket" in outcome.detail
    assert outcome.run is not None
    assert outcome.run.nccl is not None
    assert outcome.run.nccl.network == "Socket"  # `BandwidthRun.nccl` は rank 0 (head) のもの
    assert outcome.run.nccl.socket_channel_seen is True
    # 台ごとの観察を、結果に残す (7.4 が、使われた経路と束ねを、台ごとに書く)
    assert {role: found.network for role, found in outcome.nccl.items() if found is not None} == {
        "head": "Socket",
        "worker": "Socket",
    }


def test_the_fast_path_passes_and_reports_busbw_and_the_reference(tmp_path: Path) -> None:
    """高速の経路の記録で合格になり、大きさごとの `busbw`、比べる相手、道具の違いが入る。"""
    config = job_config()
    script = JobScript(
        rounds=[
            Round(
                plans=plans_of(config),
                stdout={"head": bench_stdout(), "worker": ""},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "passed"
    assert outcome.run is not None
    assert [(sample.size_bytes, sample.busbw_gbps) for sample in outcome.run.samples] == [
        (MIB, 20.0),
        (GIB, 176.2),
    ]
    assert outcome.run.nccl is not None
    assert outcome.run.nccl.network == "IB"
    assert outcome.run.nccl.merged_nic is True
    # 比べる相手 (NVIDIA の公表の実測と、2 つのしきい値) と、道具が違うこと (4.3、4.6)
    assert outcome.reference.measured.gbps == 189.85
    assert outcome.reference.sync_lower_bound.gbps == 184.0
    assert outcome.reference.nccl_threshold.gbps == 175.0
    assert "ib_write_bw" in outcome.comparison
    assert "allreduce_bench.py" in outcome.comparison
    assert "189.85" in outcome.comparison
    assert "184" in outcome.comparison
    assert "175" in outcome.comparison


def test_an_unreadable_nccl_log_is_not_passed(tmp_path: Path) -> None:
    """NCCL の記録を回収できなければ、合格にしない (「確かめられなかった」と言う)。"""
    config = job_config()
    script = JobScript(
        rounds=[Round(plans=plans_of(config), stdout={"head": bench_stdout(), "worker": ""})]
    )
    runner = runner_of(tmp_path, script)

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert "確かめられなかった" in outcome.detail
    assert outcome.run is not None
    assert outcome.run.nccl is None
    assert outcome.nccl == {"head": None, "worker": None}
    # 測った値そのものは読めている (経路だけが確かめられない)
    assert outcome.run.samples


def test_a_nonzero_exit_is_not_passed(tmp_path: Path) -> None:
    """コンテナが 0 以外で終わったら、不合格で、その終了コードが結果に出る。"""
    config = job_config()
    script = JobScript(
        rounds=[
            Round(
                plans=plans_of(config),
                stdout={"head": "", "worker": ""},
                states={"worker": (Reply(stdout="exited 1\n"),)},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert [(end.node, end.exit_code) for end in outcome.ends] == [("head", 0), ("worker", 1)]
    assert "worker" in outcome.detail


def test_a_job_that_never_ends_times_out(tmp_path: Path) -> None:
    """終わらないコンテナは、待ちの上限で打ち切り、不合格になる (コンテナは残らない)。"""
    config = job_config(ready_timeout_s=30)
    plans = plans_of(config)
    script = JobScript(
        rounds=[
            Round(
                plans=plans,
                stdout={"head": "", "worker": ""},
                states={
                    "head": (Reply(stdout="running 0\n"),),
                    "worker": (Reply(stdout="running 0\n"),),
                },
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)
    clock = FakeClock()

    outcome = run_bandwidth(runner, config, tmp_path, clock=clock)

    assert outcome.status == "failed"
    assert "時間切れ" in outcome.detail
    assert clock.slept  # 眠ったのは、差し替えた時計の上だけ
    assert argv_of(runner, "docker", "stop") == tuple(
        stop_argv(plan.container_name) for plan in plans
    )


def test_the_json_line_is_found_among_other_lines(tmp_path: Path) -> None:
    """rank 0 の標準出力に、警告やほかの JSON が混ざっていても、結果の 1 行を見つける。"""
    config = job_config()
    script = JobScript(
        rounds=[
            Round(
                plans=plans_of(config),
                stdout={"head": bench_stdout({GIB: 180.0}), "worker": "rank 1 に JSON はない\n"},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "passed"
    assert outcome.run is not None
    assert [sample.busbw_gbps for sample in outcome.run.samples] == [180.0]


def test_a_missing_json_line_is_not_passed(tmp_path: Path) -> None:
    """rank 0 の標準出力に結果の JSON がなければ、合格にしない。"""
    config = job_config()
    script = JobScript(
        rounds=[
            Round(
                plans=plans_of(config),
                stdout={"head": "何も出なかった\n", "worker": ""},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert outcome.run is not None
    assert outcome.run.samples == ()


# --- 合否の連言 (requirements 4.4、research.md §e-4) -------------------------


BOTH_IB: Mapping[NodeRole, str] = {"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG}
"""2 台とも、高速の直結の経路が使われた記録 (`one_round` の既定)。"""


def one_round(
    config: ConfigDef,
    *,
    nccl: Mapping[NodeRole, str] | None = BOTH_IB,
    stale: Mapping[NodeRole, Mapping[str, str]] | None = None,
    stdout: Mapping[NodeRole, str] | None = None,
    states: Mapping[NodeRole, tuple[Reply, ...]] | None = None,
    busbw: Mapping[int, float] | None = None,
    world_size: int = 2,
) -> JobScript:
    """帯域の計測 1 回ぶんの、ふつうの台本 (変えたいところだけを渡す)。"""
    return JobScript(
        rounds=[
            Round(
                plans=plans_of(config),
                stdout={"head": bench_stdout(busbw, world_size=world_size), "worker": ""}
                if stdout is None
                else stdout,
                states={} if states is None else states,
                nccl=nccl,
                stale={} if stale is None else stale,
            )
        ]
    )


def test_a_socket_channel_alone_is_not_passed(tmp_path: Path) -> None:
    """経路が IB でも、ふつうのネットワークの経路のチャンネルがあれば、合格にしない (指摘 6)。"""
    config = job_config()
    runner = runner_of(
        tmp_path,
        one_round(
            config,
            nccl={
                "head": NCCL_IB_WITH_SOCKET_CHANNEL_LOG,
                "worker": NCCL_IB_WITH_SOCKET_CHANNEL_LOG,
            },
        ),
    )

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert "NET/Socket" in outcome.detail
    head = outcome.nccl["head"]
    assert head is not None
    assert head.network == "IB"  # 1 つ目の連言は満たしている
    assert head.socket_channel_seen is True


def test_only_the_worker_falling_back_is_not_passed(tmp_path: Path) -> None:
    """片方の台だけがふつうのネットワークに落ちても、合格にしない (指摘 2)。"""
    config = job_config()
    runner = runner_of(
        tmp_path, one_round(config, nccl={"head": NCCL_IB_LOG, "worker": NCCL_SOCKET_LOG})
    )

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert "worker" in outcome.detail
    head = outcome.nccl["head"]
    worker = outcome.nccl["worker"]
    assert head is not None and worker is not None
    assert (head.network, worker.network) == ("IB", "Socket")


def test_ib_no_device_is_not_passed(tmp_path: Path) -> None:
    """`NET/IB : No device found.` が出ている台があれば、合格にしない (指摘 3)。"""
    config = job_config()
    runner = runner_of(
        tmp_path,
        one_round(config, nccl={"head": NCCL_IB_NO_DEVICE_LOG, "worker": NCCL_IB_NO_DEVICE_LOG}),
    )

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert "No device found" in outcome.detail
    head = outcome.nccl["head"]
    assert head is not None
    assert (head.network, head.ib_no_device) == ("IB", True)


def test_a_world_size_mismatch_is_not_passed(tmp_path: Path) -> None:
    """rank 0 の JSON の `world_size` が台の数と合わなければ、合格にしない (指摘 4)。"""
    config = job_config()
    runner = runner_of(tmp_path, one_round(config, world_size=1))

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert "world_size" in outcome.detail


def test_a_zero_busbw_is_read_as_unreadable(tmp_path: Path) -> None:
    """`busbw` が 0 以下の項目は読めないものとして落とし、1 つも残らなければ合格にしない。"""
    config = job_config()
    runner = runner_of(tmp_path, one_round(config, busbw={MIB: 0.0, GIB: 0.0}))

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert outcome.run is not None
    assert outcome.run.samples == ()
    assert "busbw" in outcome.detail


def test_a_zero_busbw_sample_is_dropped_but_the_others_are_kept(tmp_path: Path) -> None:
    """0 以下の項目だけを落とす (残りが読めても、読めない項目があれば合格にしない)。"""
    config = job_config()
    runner = runner_of(tmp_path, one_round(config, busbw={MIB: 0.0, GIB: 176.2}))

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert outcome.run is not None
    assert [sample.size_bytes for sample in outcome.run.samples] == [GIB]


def test_the_default_size_list_gets_no_note_but_a_short_one_does(tmp_path: Path) -> None:
    """design の決まりの列 (6 つ) なら注記なし、縮めた列なら注記を付ける (合否には数えない)。"""
    config = job_config()
    runner = runner_of(tmp_path, one_round(config, busbw=full_sizes()))
    outcome = run_bandwidth(runner, config, tmp_path)
    assert outcome.status == "passed"
    assert "design の決まりの列" not in outcome.detail

    runner = runner_of(tmp_path, one_round(config, busbw={GIB: 176.2}))
    short = run_bandwidth(runner, config, tmp_path)
    assert short.status == "passed"
    assert "design の決まりの列" in short.detail


def test_two_network_lines_are_not_passed(tmp_path: Path) -> None:
    """1 台の記録に `Using network IB` と `Using network Socket` の両方があれば、合格にしない。

    `observe_nccl` の `network` は、最初の `Using network …` の行だけを見る (`observe.py` は
    凍結) ので、この module が本文を見て補う (指摘 4)。
    """
    config = job_config()
    runner = runner_of(
        tmp_path,
        one_round(config, nccl={"head": NCCL_IB_THEN_SOCKET_LOG, "worker": NCCL_IB_LOG}),
    )

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert "Using network Socket" in outcome.detail
    head = outcome.nccl["head"]
    assert head is not None
    # observe の読み取りそのものは変えない (最初の行のまま)。合否だけを、この module が補う
    assert head.network == "IB"
    assert head.socket_channel_seen is False
    assert head.ib_no_device is False


# --- 回ごとの NCCL の記録 (指摘 1) ------------------------------------------


def _arm_of(argv: Sequence[str]) -> tuple[str, int]:
    """`docker run` の列のラベル `vllm-baseline.run=<腕>-<回>` から、腕と回を読む。"""
    for flag, value in zip(argv, argv[1:], strict=False):
        if flag == "--label" and value.startswith(f"{LABEL_RUN}="):
            arm, _, repeat = value.split("=", 1)[1].rpartition("-")
            return arm, int(repeat)
    raise AssertionError(f"run のラベルがない: {list(argv)}")


def test_the_nccl_log_file_name_is_overridden_per_round(tmp_path: Path) -> None:
    """どの回でも、この道具が `NCCL_DEBUG_FILE` を、回の札を含む名前に上書きする。"""
    config = job_config()
    script, plan_rounds = ab_script(config, SEPARATED)
    runner = runner_of(tmp_path, script)

    run_ab(runner, config, tmp_path)

    names: list[str] = []
    for argv in argv_of(runner, "docker", "run"):
        values = [
            value.split("=", 1)[1]
            for flag, value in zip(argv, argv[1:], strict=False)
            if flag == "-e" and value.startswith("NCCL_DEBUG_FILE=")
        ]
        # 構成の値と、この道具の上書きが、この順に並ぶ (docker は後ろの -e を採る)
        assert values == [
            "/logs/nccl_%h_%p.log",
            f"{LOGS_TARGET}/nccl-{round_tag(*_arm_of(argv))}.%h.%p.log",
        ]
        names.append(values[-1])
    assert len(names) == 12  # 12 個のコンテナのすべてに入る (腕 A にも)
    assert len(set(names)) == 6  # 回ごとに違う (2 台は同じ名前でよい。%h で分かれる)
    assert names[0] == f"{LOGS_TARGET}/nccl-{round_tag(nc.BASELINE_ARM, 1)}.%h.%p.log"
    # 見せた計画の列と、実際に流す列が、一致していること (上書きを足しても壊れない)
    expected: list[tuple[str, ...]] = []
    for plans in plan_rounds:
        expected.extend(plan.argv for plan in plans)
    assert list(argv_of(runner, "docker", "run")) == expected


def test_a_missing_round_log_is_not_passed(tmp_path: Path) -> None:
    """その回の札のファイルがない台は「確かめられなかった」で、合格にしない (指摘 1)。"""
    config = job_config()
    stale = {"nccl-20260101T000000Z-baseline-1.head.7.log": NCCL_IB_LOG}
    runner = runner_of(
        tmp_path, one_round(config, nccl=None, stale={"head": stale, "worker": stale})
    )

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert "確かめられなかった" in outcome.detail
    assert outcome.nccl == {"head": None, "worker": None}


def test_only_the_files_named_for_this_round_are_read(tmp_path: Path) -> None:
    """`nccl-<その回の札>.` で始まる名前のファイルだけを読む (指摘 2)。

    絞り込みが部分一致だと、次の 3 つのどれでも、古い IB の記録で、この回の失敗が隠れる:

    - `…-baseline-10` (10 回目。回の番号を 2 桁にそろえると、`-01` と `-10` は別になる)
    - `nccl-<この回の札>-old.….log` (札のあとに続きがある)
    - `copy-of-nccl-<この回の札>.….log` (`nccl-` の前に続きがある)
    """
    config = job_config()
    stale = {
        f"nccl-{round_tag(nc.BASELINE_ARM, 10)}.head.7.log": NCCL_IB_LOG,
        f"nccl-{BANDWIDTH_TAG}-old.head.7.log": NCCL_IB_LOG,
        f"copy-of-nccl-{BANDWIDTH_TAG}.head.7.log": NCCL_IB_LOG,
    }
    runner = runner_of(
        tmp_path, one_round(config, nccl=None, stale={"head": stale, "worker": stale})
    )

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert "確かめられなかった" in outcome.detail
    assert outcome.nccl == {"head": None, "worker": None}


def test_the_tag_includes_the_microseconds_of_started_at(tmp_path: Path) -> None:
    """同じ秒に 2 度流しても、札が分かれる (`started_at` のマイクロ秒まで入る。指摘 2)。"""
    config = job_config()
    names: list[str] = []
    for microsecond in (0, 1):
        runner = runner_of(tmp_path, one_round(config))
        nc.run_bandwidth(
            runner,
            config,
            NODES,
            STARTED_AT.replace(microsecond=microsecond),
            confirmer=SpyConfirmer(),
            var_root=var_root(tmp_path),
            poll_interval_s=10.0,
            sleep=FakeClock().sleep,
            clock=FakeClock().monotonic,
            report=io.StringIO(),
        )
        argv = argv_of(runner, "docker", "run")[0]
        # 構成の値のあとに、この道具の上書きが並ぶ (docker は後ろの -e を採る)
        names.append(
            [
                value
                for flag, value in zip(argv, argv[1:], strict=False)
                if flag == "-e" and value.startswith("NCCL_DEBUG_FILE=")
            ][-1]
        )
    assert names[0] != names[1]
    assert "-000000Z-" in names[0] and "-000001Z-" in names[1]


def test_a_stale_ib_log_does_not_hide_this_round_falling_back(tmp_path: Path) -> None:
    """古い回の IB の記録が `logs/` に残っていても、今回の Socket の記録で判定する (指摘 1)。"""
    config = job_config()
    stale = {"nccl-20260101T000000Z-baseline-1.head.7.log": NCCL_IB_LOG}
    runner = runner_of(
        tmp_path,
        one_round(
            config,
            nccl={"head": NCCL_SOCKET_LOG, "worker": NCCL_SOCKET_LOG},
            stale={"head": stale, "worker": stale},
        ),
    )

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    head = outcome.nccl["head"]
    assert head is not None
    assert head.network == "Socket"


def test_a_config_whose_logs_mount_is_readonly_is_refused(tmp_path: Path) -> None:
    """`logs/` の結び付けが読み取り専用の構成は、Spark に触る前に断る (指摘 3)。

    起こしてから「記録を回収できなかった」で失敗すると、了承とコンテナの起動が無駄になる。
    """
    for flag in ("readonly", "ro"):
        docker = dict(job_config().docker)
        docker["mount-logs"] = _setting(
            "--mount", f"type=bind,source={{remote_root}}/logs,target={LOGS_TARGET},{flag}"
        )
        config = job_config().model_copy(update={"docker": docker})
        runner = runner_of(tmp_path, JobScript(rounds=[]))
        with pytest.raises(ConfigError, match="読み取り専用"):
            run_bandwidth(runner, config, tmp_path)
        assert runner.calls == (), flag


def test_a_config_that_cannot_keep_the_nccl_log_is_refused(tmp_path: Path) -> None:
    """記録が取れない構成は、Spark に触る前に断る (合否を出せないため。指摘 1)。"""
    broken: dict[str, dict[str, Setting]] = {
        "NCCL_DEBUG がない": {
            "nccl-debug-file": _setting("NCCL_DEBUG_FILE", f"{LOGS_TARGET}/nccl.%h.%p.log")
        },
        "NCCL_DEBUG が INFO 未満": {
            "nccl-debug": _setting("NCCL_DEBUG", "WARN"),
            "nccl-debug-file": _setting("NCCL_DEBUG_FILE", f"{LOGS_TARGET}/nccl.%h.%p.log"),
        },
        "NCCL_DEBUG_FILE がない": {"nccl-debug": _setting("NCCL_DEBUG", "INFO")},
        "NCCL_DEBUG_FILE が相対の道筋": {
            "nccl-debug": _setting("NCCL_DEBUG", "INFO"),
            "nccl-debug-file": _setting("NCCL_DEBUG_FILE", "nccl.%h.%p.log"),
        },
        "NCCL_DEBUG_FILE が結び付けの外": {
            "nccl-debug": _setting("NCCL_DEBUG", "INFO"),
            "nccl-debug-file": _setting("NCCL_DEBUG_FILE", "/tmp/nccl.%h.%p.log"),
        },
    }
    for label, env in broken.items():
        config = job_config().model_copy(update={"env": env})
        runner = runner_of(tmp_path, JobScript(rounds=[]))
        with pytest.raises(ConfigError, match="NCCL"):
            run_bandwidth(runner, config, tmp_path)
        assert runner.calls == (), label


def test_ab_refuses_the_nccl_env_in_extra_env(tmp_path: Path) -> None:
    """`NCCL_DEBUG_FILE` / `NCCL_DEBUG` は、この道具が使うので、A/B の足す設定に書けない。"""
    config = job_config()
    for name in ("NCCL_DEBUG_FILE", "NCCL_DEBUG"):
        runner = runner_of(tmp_path, JobScript(rounds=[]))
        with pytest.raises(ValueError, match=name):
            run_ab(runner, config, tmp_path, extra_env={name: "INFO"})
        assert runner.calls == ()


# --- 待ちの打ち切り (指摘 8) -------------------------------------------------


def test_a_bad_exit_does_not_wait_for_the_other_node(tmp_path: Path) -> None:
    """片方が 0 以外で終わったら、ほかの台の終了を待たずに、片付けに進む (指摘 8)。"""
    config = job_config(ready_timeout_s=1800)
    plans = plans_of(config)
    script = JobScript(
        rounds=[
            Round(
                plans=plans,
                stdout={"head": "", "worker": ""},
                states={
                    # head は動き続け (rendezvous の相手が死ぬと固まる)、worker は 1 で終わる
                    "head": (Reply(stdout="running 0\n"),),
                    "worker": (Reply(stdout="exited 1\n"),),
                },
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)
    clock = FakeClock()

    outcome = run_bandwidth(runner, config, tmp_path, clock=clock)

    assert outcome.status == "failed"
    assert clock.slept == []  # 1 周目で打ち切る (1,800 秒を空待ちしない)
    assert "時間切れ" not in outcome.detail
    assert [(end.node, end.exit_code) for end in outcome.ends] == [("worker", 1)]
    assert argv_of(runner, "docker", "stop") == tuple(
        stop_argv(plan.container_name) for plan in plans
    )


# --- 比べる相手の出典 (指摘 7) ----------------------------------------------


def test_the_reference_points_match_research_md() -> None:
    """比べる相手の値と、出典と、原文が、research.md の記述と一致している (指摘 7)。"""
    reference = nc.NVIDIA_REFERENCE
    # 189.85 Gbps は ib_write_bw の実測 (research.md §e-1 の原文)
    assert reference.measured.gbps == 189.85
    assert reference.measured.tool == "ib_write_bw"
    assert reference.measured.quote == "Total throughput = 92.57 + 97.28 = 189.85 Gbps"
    # research.md の References が挙げる URL まで書く (research.md にある URL だけ)
    assert "docs.nvidia.com/dgx/dgx-spark/spark-clustering.html" in reference.measured.source
    assert "github.com/NVIDIA/dgx-spark-playbooks" in reference.measured.source
    # 184 Gbit/s は NVIDIA Sync の文書の原文 (URL も research.md の References にある)
    assert reference.sync_lower_bound.gbps == 184.0
    assert "184 Gbit/s" in reference.sync_lower_bound.quote
    assert "docs.nvidia.com/sync" in reference.sync_lower_bound.source
    # 175 Gbps (= 21.875 GB/s) は、別の出どころ (クラスタ設定スクリプトの定数)
    assert reference.nccl_threshold.gbps == 175.0
    assert "docs.nvidia.com/sync" not in reference.nccl_threshold.source
    assert "21.875" in reference.nccl_threshold.note
    assert "Avg bus bandwidth" in reference.nccl_threshold.quote
    # しきい値は、合否に入れない (道具が違うので、比べて示すだけ)
    assert "ib_write_bw" in reference.tool_note
    assert "allreduce_bench.py" in reference.tool_note


def test_the_thresholds_are_not_part_of_the_verdict(tmp_path: Path) -> None:
    """しきい値を下回っても、経路が高速なら合格にする (合否は経路と 4 段だけ。design の合否)。"""
    config = job_config()
    runner = runner_of(tmp_path, one_round(config, busbw=dict.fromkeys(full_sizes(), 1.0)))

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "passed"
    assert "175" in outcome.comparison


# --- 事前の確認 (requirements 4.6) ------------------------------------------


def test_all_four_sanity_stages_pass(tmp_path: Path) -> None:
    """4 段 (PyTorch の NCCL、GLOO、vLLM の NCCL、CUDA グラフの中の NCCL) がすべて通る。"""
    config = job_config(name=SANITY_CONFIG, script="/payload/vllm_sanity_check.py")
    output = sanity_stdout(4)
    script = JobScript(
        rounds=[
            Round(
                plans=plans_of(config, tag=SANITY_TAG),
                tag=SANITY_TAG,
                stdout={"head": output, "worker": output},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    outcome = run_sanity(runner, config, tmp_path)

    assert outcome.status == "passed"
    assert outcome.passed_stages == 4
    assert [stage.passed for stage in outcome.stages] == [True, True, True, True]
    assert all(stage.nodes_seen == ("head", "worker") for stage in outcome.stages)


def test_sanity_stopping_at_the_third_stage(tmp_path: Path) -> None:
    """3 段目で止まると、どこまで通ったかと、どこで止まったかが結果に出る。"""
    config = job_config(name=SANITY_CONFIG, script="/payload/vllm_sanity_check.py")
    output = sanity_stdout(2)
    script = JobScript(
        rounds=[
            Round(
                plans=plans_of(config, tag=SANITY_TAG),
                tag=SANITY_TAG,
                stdout={"head": output, "worker": output},
                states={
                    "head": (Reply(stdout="exited 1\n"),),
                    "worker": (Reply(stdout="exited 1\n"),),
                },
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    outcome = run_sanity(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert outcome.passed_stages == 2
    assert [stage.passed for stage in outcome.stages] == [True, True, False, False]
    assert outcome.stages[2].index == 3
    assert outcome.stages[2].marker == "vLLM NCCL is successful!"
    assert "3" in outcome.detail
    assert "1" in outcome.detail  # 0 以外の終了コード


def test_sanity_needs_the_marker_on_both_nodes(tmp_path: Path) -> None:
    """成功の文面は、各 rank が出すので、片方の台にしか出ていなければ通ったとしない。"""
    config = job_config(name=SANITY_CONFIG, script="/payload/vllm_sanity_check.py")
    script = JobScript(
        rounds=[
            Round(
                plans=plans_of(config, tag=SANITY_TAG),
                tag=SANITY_TAG,
                stdout={"head": sanity_stdout(4), "worker": sanity_stdout(3)},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    outcome = run_sanity(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert outcome.passed_stages == 3
    assert outcome.stages[3].nodes_seen == ("head",)


def test_sanity_uses_the_upstream_success_lines(tmp_path: Path) -> None:
    """成功の文面は、上流の原文 (`payload/vllm_sanity_check.py` が `print` する文字列)。"""
    markers = [stage.marker for stage in nc.SANITY_STAGES]
    payload_path = Path(__file__).resolve().parents[2] / "payload" / "vllm_sanity_check.py"
    payload = payload_path.read_text(encoding="utf-8")
    assert markers == [
        "PyTorch NCCL is successful!",
        "PyTorch GLOO is successful!",
        "vLLM NCCL is successful!",
        "vLLM NCCL with cuda graph is successful!",
    ]
    for marker in markers:
        assert f'print("{marker}")' in payload


# --- A/B (requirements 4.5) -------------------------------------------------


def ab_script(
    config: ConfigDef,
    busbw: Sequence[Mapping[int, float]],
    *,
    extra_env: Mapping[str, str] | None = None,
    repeats: int = 3,
) -> tuple[JobScript, tuple[tuple[ContainerPlan, ...], ...]]:
    """A/B の 6 回ぶん (`repeats` を変えれば、その回数ぶん) の台本。"""
    plan_rounds = ab_rounds(config, ADDED_ENV if extra_env is None else extra_env, repeats=repeats)
    script = JobScript(
        rounds=[
            Round(
                plans=plans,
                tag=round_tag(arm, repeat),
                stdout={"head": bench_stdout(busbw[index]), "worker": ""},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
            for index, (plans, (arm, repeat)) in enumerate(
                zip(plan_rounds, ab_order(repeats), strict=True)
            )
        ]
    )
    return script, plan_rounds


OVERLAPPING = [{GIB: 170.0}, {GIB: 171.0}, {GIB: 172.0}, {GIB: 169.0}, {GIB: 168.0}, {GIB: 173.0}]
"""範囲が重なる 6 回 (A: 170、172、168 / B: 171、169、173)。"""

SEPARATED = [{GIB: 150.0}, {GIB: 180.0}, {GIB: 152.0}, {GIB: 181.0}, {GIB: 151.0}, {GIB: 179.0}]
"""範囲が重ならない 6 回 (A の最大 152 < B の最小 179)。"""


def test_ab_with_overlapping_ranges_cannot_be_adopted(tmp_path: Path) -> None:
    """範囲が重なれば「採用できない」(requirements 4.5)。"""
    config = job_config()
    script, _ = ab_script(config, OVERLAPPING)
    runner = runner_of(tmp_path, script)

    report = run_ab(runner, config, tmp_path)

    assert report.status == "compared"
    assert report.outcome is not None
    assert report.outcome.adopt is False
    assert report.outcome.compared_size_bytes == GIB
    assert "採用できない" in report.outcome.detail


def test_ab_with_separated_ranges_can_be_adopted(tmp_path: Path) -> None:
    """足した側の最小が、最小の設定の側の最大を上回れば「採用できる」。"""
    config = job_config()
    script, _ = ab_script(config, SEPARATED)
    runner = runner_of(tmp_path, script)

    report = run_ab(runner, config, tmp_path)

    assert report.status == "compared"
    assert report.outcome is not None
    assert report.outcome.adopt is True
    assert report.outcome.added_env == dict(ADDED_ENV)
    assert len(report.outcome.baseline_runs) == 3
    assert len(report.outcome.candidate_runs) == 3
    assert "採用できる" in report.outcome.detail


def test_ab_runs_six_alternating_rounds_with_names_and_labels(tmp_path: Path) -> None:
    """6 回が交互の順で流れ、名前に回の番号、ラベルに腕と回が付く。"""
    config = job_config()
    script, plan_rounds = ab_script(config, SEPARATED)
    runner = runner_of(tmp_path, script)

    report = run_ab(runner, config, tmp_path)

    assert [(run.arm, run.repeat_index) for run in report.runs] == [
        (nc.BASELINE_ARM, 1),
        (nc.CANDIDATE_ARM, 1),
        (nc.BASELINE_ARM, 2),
        (nc.CANDIDATE_ARM, 2),
        (nc.BASELINE_ARM, 3),
        (nc.CANDIDATE_ARM, 3),
    ]
    assert started_names(runner) == tuple(
        plan.container_name for plans in plan_rounds for plan in plans
    )
    assert started_names(runner)[:2] == (
        f"vb-{BANDWIDTH_CONFIG}-head-r1",
        f"vb-{BANDWIDTH_CONFIG}-worker-r1",
    )
    labels = [plans[0].labels[LABEL_RUN] for plans in plan_rounds]
    assert labels == [
        f"{nc.BASELINE_ARM}-1",
        f"{nc.CANDIDATE_ARM}-1",
        f"{nc.BASELINE_ARM}-2",
        f"{nc.CANDIDATE_ARM}-2",
        f"{nc.BASELINE_ARM}-3",
        f"{nc.CANDIDATE_ARM}-3",
    ]
    # 回ごとに、別の置き場所に回収する
    assert len({run.log_dir for run in report.runs}) == 6


def test_only_the_candidate_arm_gets_the_added_env(tmp_path: Path) -> None:
    """足した環境変数は、B の回にだけ渡り、全ノードに同じ値が渡る。"""
    config = job_config()
    script, _ = ab_script(config, SEPARATED)
    runner = runner_of(tmp_path, script)

    run_ab(runner, config, tmp_path)

    added = "NCCL_IB_QPS_PER_CONNECTION=4"
    runs = argv_of(runner, "docker", "run")
    assert len(runs) == 12
    with_added = [index for index, argv in enumerate(runs) if added in argv]
    # A1、B1、A2、B2、A3、B3 の順に 2 台ずつなので、B の回は 2〜3、6〜7、10〜11 番目
    assert with_added == [2, 3, 6, 7, 10, 11]


def test_the_added_env_comes_after_the_config_env(tmp_path: Path) -> None:
    """構成にある名前を足したときは、あとに並ぶ (docker は、後ろの `-e` を採る)。"""
    config = job_config()
    override = {"NCCL_SOCKET_IFNAME": "=enP7s7"}
    script, _ = ab_script(config, SEPARATED, extra_env=override)
    runner = runner_of(tmp_path, script)

    run_ab(runner, config, tmp_path, extra_env=override)

    candidate = argv_of(runner, "docker", "run")[2]
    values = [value for flag, value in zip(candidate, candidate[1:], strict=False) if flag == "-e"]
    ifname = [value for value in values if value.startswith("NCCL_SOCKET_IFNAME=")]
    assert ifname == ["NCCL_SOCKET_IFNAME==enp1s0f0np0", "NCCL_SOCKET_IFNAME==enP7s7"]


def test_ab_asks_for_approval_once_with_every_round_in_the_plan(tmp_path: Path) -> None:
    """了承は 1 回で、見せた計画に 6 回ぶんの `docker run` と巻き戻しがすべてある。"""
    config = job_config()
    script, plan_rounds = ab_script(config, SEPARATED)
    runner = runner_of(tmp_path, script)
    confirmer = SpyConfirmer()

    run_ab(runner, config, tmp_path, confirmer=confirmer)

    assert len(confirmer.shown) == 1
    shown = confirmer.shown[0]
    assert "前に進むコマンド (12 件)" in shown
    assert "巻き戻すコマンド (24 件" in shown
    for plans in plan_rounds:
        for plan in plans:
            assert f"--name {plan.container_name}" in shown
    approved = runner.plan
    assert approved is not None
    assert len(approved.forward) == 12
    assert len(approved.rollback) == 24


def test_ab_without_approval_starts_nothing(tmp_path: Path) -> None:
    """了承しないと、`docker run` が 1 つも出ない (requirements 2.1)。"""
    config = job_config()
    script, _ = ab_script(config, SEPARATED)
    runner = runner_of(tmp_path, script)

    with pytest.raises(ApprovalError):
        run_ab(runner, config, tmp_path, confirmer=SpyConfirmer(approves=False))

    assert mutating_calls(runner) == ()
    assert argv_of(runner, "docker", "run") == ()


def test_a_failing_round_stops_the_rest(tmp_path: Path) -> None:
    """途中の回が失敗したら、その回を片付けて、残りの回を流さずに、そこまでを返す。"""
    config = job_config()
    plan_rounds = ab_rounds(config, ADDED_ENV)
    script = JobScript(
        rounds=[
            Round(
                plans=plan_rounds[0],
                stdout={"head": bench_stdout({GIB: 150.0}), "worker": ""},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            ),
            Round(
                plans=plan_rounds[1],
                stdout={"head": "結果の JSON が出なかった\n", "worker": ""},
                states={"head": (Reply(stdout="exited 1\n"),)},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            ),
        ]
    )
    runner = runner_of(tmp_path, script)

    report = run_ab(runner, config, tmp_path)

    assert report.status == "stopped"
    assert report.outcome is None
    assert len(report.runs) == 2
    assert started_names(runner) == tuple(
        plan.container_name for plans in plan_rounds[:2] for plan in plans
    )
    # 流れた 2 回ぶんのコンテナは、どちらも止めて消してある
    assert len(argv_of(runner, "docker", "stop")) == 4
    assert len(argv_of(runner, "docker", "rm")) == 4


def ab_script_with_routes(
    config: ConfigDef,
    busbw: Sequence[Mapping[int, float]],
    *,
    baseline_nccl: Mapping[NodeRole, str],
    candidate_nccl: Mapping[NodeRole, str],
    rounds: int | None = None,
) -> tuple[JobScript, tuple[tuple[ContainerPlan, ...], ...]]:
    """腕ごとに、別の NCCL の記録を返す A/B の台本 (`rounds` で、流れる回を減らせる)。"""
    plan_rounds = ab_rounds(config, ADDED_ENV)
    order = ab_order()
    used = len(plan_rounds) if rounds is None else rounds
    script = JobScript(
        rounds=[
            Round(
                plans=plans,
                tag=round_tag(arm, repeat),
                stdout={"head": bench_stdout(busbw[index]), "worker": ""},
                nccl=baseline_nccl if arm == nc.BASELINE_ARM else candidate_nccl,
            )
            for index, (plans, (arm, repeat)) in enumerate(zip(plan_rounds, order, strict=True))
            if index < used
        ]
    )
    return script, plan_rounds


def test_ab_keeps_the_route_reading_per_round_and_node(tmp_path: Path) -> None:
    """A/B の回ごと・台ごとの経路の観察が `AbReport` から読め、注意が `detail` に出る (指摘 1)。

    採否は**速さだけ**で決める (要件 4.5) ので、B が速ければ `adopt` は真になる。ただし、
    B の worker がふつうのネットワークの経路に落ちていることは、`routes` と `detail` から
    分かる (7.4 が、判断の記録に書けるようにする)。
    """
    config = job_config()
    script, _ = ab_script_with_routes(
        config,
        SEPARATED,
        baseline_nccl=BOTH_IB,
        candidate_nccl={"head": NCCL_IB_LOG, "worker": NCCL_SOCKET_LOG},
    )
    runner = runner_of(tmp_path, script)

    report = run_ab(runner, config, tmp_path)

    assert report.status == "compared"
    assert report.outcome is not None
    assert report.outcome.adopt is True  # 速さの決まりのとおり (経路は採否に使わない)
    # 回ごと・台ごとの観察が、`runs` と同じ順で読める
    assert len(report.routes) == len(report.runs) == 6
    assert [(route.arm, route.repeat_index) for route in report.routes] == [
        (run.arm, run.repeat_index) for run in report.runs
    ]
    candidate = [route for route in report.routes if route.arm == nc.CANDIDATE_ARM]
    assert len(candidate) == 3
    for route in candidate:
        worker = route.observed["worker"]
        head = route.observed["head"]
        assert worker is not None and head is not None
        assert (head.network, worker.network) == ("IB", "Socket")
    # 腕ごと・台ごとのまとめと、注意が `detail` に出る
    assert "worker=Socket" in report.detail
    assert "注意" in report.detail
    assert report.outcome.detail.count("注意") == 1


def test_ab_notes_a_second_network_line_in_the_summary(tmp_path: Path) -> None:
    """経路の行が 2 度ある記録でも、A/B のまとめと注意に出る (指摘 4 が、A/B にも効く)。"""
    config = job_config()
    script, _ = ab_script_with_routes(
        config,
        SEPARATED,
        baseline_nccl=BOTH_IB,
        candidate_nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_THEN_SOCKET_LOG},
    )
    runner = runner_of(tmp_path, script)

    report = run_ab(runner, config, tmp_path)

    assert report.status == "compared"
    assert report.outcome is not None
    assert report.outcome.adopt is True
    worker = report.routes[1].observed["worker"]
    assert worker is not None
    assert worker.network == "IB"  # observe の読み取りは、最初の行のまま
    assert report.routes[1].socket_network_seen["worker"] is True
    assert "注意" in report.detail


def test_ab_with_all_ib_has_no_caution(tmp_path: Path) -> None:
    """6 回とも記録がそろい、全台 IB なら、注意は出ない (まとめだけが出る)。"""
    config = job_config()
    script, _ = ab_script_with_routes(
        config, SEPARATED, baseline_nccl=BOTH_IB, candidate_nccl=BOTH_IB
    )
    runner = runner_of(tmp_path, script)

    report = run_ab(runner, config, tmp_path)

    assert report.status == "compared"
    assert report.outcome is not None
    assert report.outcome.adopt is True
    assert "注意" not in report.detail
    assert "注意" not in report.outcome.detail
    assert "head=IB" in report.detail and "worker=IB" in report.detail


def test_ab_stops_when_a_round_has_no_route_record(tmp_path: Path) -> None:
    """その回の札の記録が、どれかの台で読めなければ、値を比較に使わずに止まる (指摘 1)。"""
    config = job_config()
    plan_rounds = ab_rounds(config, ADDED_ENV)
    script = JobScript(
        rounds=[
            Round(
                plans=plan_rounds[0],
                tag=round_tag(nc.BASELINE_ARM, 1),
                stdout={"head": bench_stdout({GIB: 150.0}), "worker": ""},
                nccl={"head": NCCL_IB_LOG},  # worker のぶんがない
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    report = run_ab(runner, config, tmp_path)

    assert report.status == "stopped"
    assert report.outcome is None  # 経路の証拠のない値で、採否を出さない
    assert "確かめられなかった" in report.detail
    assert len(report.runs) == 1
    assert started_names(runner) == tuple(plan.container_name for plan in plan_rounds[0])


def test_a_world_size_mismatch_stops_the_ab(tmp_path: Path) -> None:
    """A/B の回でも、`world_size` が合わなければ、その回を失敗として扱い、残りを流さない。"""
    config = job_config()
    plan_rounds = ab_rounds(config, ADDED_ENV)
    script = JobScript(
        rounds=[
            Round(
                plans=plan_rounds[0],
                tag=round_tag(nc.BASELINE_ARM, 1),
                stdout={"head": bench_stdout({GIB: 150.0}, world_size=1), "worker": ""},
                nccl=BOTH_IB,
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    report = run_ab(runner, config, tmp_path)

    assert report.status == "stopped"
    assert "world_size" in report.detail
    assert len(report.runs) == 1
    assert started_names(runner) == tuple(plan.container_name for plan in plan_rounds[0])


def test_an_interrupt_in_a_round_cleans_up_and_stops_the_rest(tmp_path: Path) -> None:
    """中断は、その回を片付けてから伝える (残りの回は流さない)。"""
    config = job_config()
    plan_rounds = ab_rounds(config, ADDED_ENV)
    script = JobScript(
        rounds=[
            Round(
                plans=plan_rounds[0],
                stdout={"head": bench_stdout(), "worker": ""},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ],
        # 1 回目の、終了を待つ読み取りの最中に Ctrl-C が来る
        extra=(
            Rule(
                prefix=("docker", "container", "inspect"),
                node="head",
                replies=(Reply(raises=KeyboardInterrupt()),),
            ),
        ),
    )
    runner = runner_of(tmp_path, script)

    with pytest.raises(KeyboardInterrupt):
        run_ab(runner, config, tmp_path)

    # 1 回目だけが起きて、その 2 台とも止めて消してある。残りの回は流れていない
    assert started_names(runner) == tuple(plan.container_name for plan in plan_rounds[0])
    assert argv_of(runner, "docker", "stop") == tuple(
        stop_argv(plan.container_name) for plan in plan_rounds[0]
    )
    assert argv_of(runner, "docker", "rm") == tuple(
        remove_argv(plan.container_name) for plan in plan_rounds[0]
    )
    # 中断のあとは、記録を回収せずに片付けに進む (3.4 の決まり)
    assert runner.pulls == ()


def test_ab_repeats_can_be_changed_but_not_below_two(tmp_path: Path) -> None:
    """腕ごとの回数は変えられるが、1 回ずつは断る (範囲が重なるかどうかを見られない)。"""
    config = job_config()
    script, plan_rounds = ab_script(config, SEPARATED[:4], repeats=2)
    runner = runner_of(tmp_path, script)

    report = run_ab(runner, config, tmp_path, repeats=2)

    assert report.status == "compared"
    assert report.outcome is not None
    assert len(report.outcome.baseline_runs) == 2
    assert len(report.outcome.candidate_runs) == 2
    assert started_names(runner) == tuple(
        plan.container_name for plans in plan_rounds for plan in plans
    )

    runner = runner_of(tmp_path, JobScript(rounds=[]))
    with pytest.raises(ValueError, match="2 回以上"):
        run_ab(runner, config, tmp_path, repeats=1)
    assert runner.calls == ()


def test_ab_mutating_calls_match_the_shown_plan(tmp_path: Path) -> None:
    """A/B でも、了承のあとの、状態を変える呼び出しが、見せた計画の列と完全に一致する。"""
    config = job_config()
    script, plan_rounds = ab_script(config, SEPARATED)
    runner = runner_of(tmp_path, script)

    run_ab(runner, config, tmp_path)

    expected: list[tuple[str, ...]] = []
    for plans in plan_rounds:
        expected.extend(plan.argv for plan in plans)
        expected.extend(stop_argv(plan.container_name) for plan in plans)
        expected.extend(remove_argv(plan.container_name) for plan in plans)
    assert [call.argv for call in mutating_calls(runner)] == expected
    approved = runner.plan
    assert approved is not None
    planned = {
        command.argv
        for command in (*approved.forward, *approved.rollback)
        if isinstance(command, PlannedRun)
    }
    assert {call.argv for call in mutating_calls(runner)} <= planned


def test_ab_container_operations_target_only_listed_ids_or_planned_names(tmp_path: Path) -> None:
    """A/B でも、コンテナを対象にする操作は、一覧の ID か、計画の名前だけを対象にする。"""
    config = job_config()
    script, plan_rounds = ab_script(config, SEPARATED)
    runner = runner_of(tmp_path, script)

    run_ab(runner, config, tmp_path)

    allowed = {
        *CONTAINER_IDS.values(),
        *(plan.container_name for plans in plan_rounds for plan in plans),
    }
    assert set(container_targets(runner)) <= allowed


# --- 安全の決まり -----------------------------------------------------------


def test_gates_refusing_starts_nothing(tmp_path: Path) -> None:
    """関門が断ったら、了承も求めず、状態を変える呼び出しが 1 つも出ない。"""
    config = job_config()
    script = JobScript(
        rounds=[Round(plans=plans_of(config))],
        gpu_apps="1234, python3, 12000 MiB\n",
    )
    runner = runner_of(tmp_path, script)
    confirmer = SpyConfirmer()

    outcome = run_bandwidth(runner, config, tmp_path, confirmer=confirmer)

    assert outcome.status == "refused"
    assert "gpu_idle" in outcome.detail
    assert confirmer.shown == []
    assert mutating_calls(runner) == ()


def test_bandwidth_without_approval_starts_nothing(tmp_path: Path) -> None:
    """了承しないと、`docker run` が 1 つも出ない (requirements 2.1)。"""
    config = job_config()
    script = JobScript(rounds=[Round(plans=plans_of(config))])
    runner = runner_of(tmp_path, script)

    with pytest.raises(ApprovalError):
        run_bandwidth(runner, config, tmp_path, confirmer=SpyConfirmer(approves=False))

    assert mutating_calls(runner) == ()


def test_every_result_leaves_no_container(tmp_path: Path) -> None:
    """合格でも、記録を回収してから、必ず止めて消す (3.4 の片付けの決まり)。"""
    config = job_config()
    plans = plans_of(config)
    script = JobScript(
        rounds=[
            Round(
                plans=plans,
                stdout={"head": bench_stdout(), "worker": ""},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "passed"
    assert cleanup_order(runner) == CLEANUP_STEPS
    assert argv_of(runner, "docker", "stop") == tuple(
        stop_argv(plan.container_name) for plan in plans
    )
    assert argv_of(runner, "docker", "rm") == tuple(
        remove_argv(plan.container_name) for plan in plans
    )


def test_container_operations_target_only_listed_ids_or_planned_names(tmp_path: Path) -> None:
    """コンテナを対象にする操作は、自分の一覧の ID か、了承済みの計画の名前だけを対象にする。"""
    config = job_config()
    plans = plans_of(config)
    script = JobScript(
        rounds=[
            Round(
                plans=plans,
                stdout={"head": bench_stdout(), "worker": ""},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    run_bandwidth(runner, config, tmp_path)

    allowed = {*CONTAINER_IDS.values(), *(plan.container_name for plan in plans)}
    assert set(container_targets(runner)) <= allowed
    assert container_targets(runner)  # 何も対象にしていない、という空振りではない


def test_the_mutating_calls_match_the_shown_plan(tmp_path: Path) -> None:
    """了承のあとの、状態を変える呼び出しが、見せた計画の列と完全に一致する。"""
    config = job_config()
    plans = plans_of(config)
    script = JobScript(
        rounds=[
            Round(
                plans=plans,
                stdout={"head": bench_stdout(), "worker": ""},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    run_bandwidth(runner, config, tmp_path)

    expected = [
        *(plan.argv for plan in plans),
        *(stop_argv(plan.container_name) for plan in plans),
        *(remove_argv(plan.container_name) for plan in plans),
    ]
    assert [call.argv for call in mutating_calls(runner)] == expected


def test_a_cleanup_failure_on_a_passing_run_is_an_error(tmp_path: Path) -> None:
    """合格でも、片付けが終わらなければ `NetcheckError` にする (コンテナが残りうるため)。"""
    config = job_config()
    plans = plans_of(config)
    script = JobScript(
        rounds=[
            Round(
                plans=plans,
                stdout={"head": bench_stdout(), "worker": ""},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ],
        extra=(
            Rule(
                prefix=("docker", "rm"),
                node="worker",
                replies=(Reply(exit_code=1, stderr="No such container"),),
            ),
        ),
    )
    runner = runner_of(tmp_path, script)

    with pytest.raises(nc.NetcheckError, match="serve stop"):
        run_bandwidth(runner, config, tmp_path)


def test_a_cleanup_failure_on_a_failing_run_is_reported_in_the_detail(tmp_path: Path) -> None:
    """不合格のときは、片付けの問題を `detail` に書いて、失敗の理由とともに返す。"""
    config = job_config()
    script = JobScript(
        rounds=[
            Round(
                plans=plans_of(config),
                stdout={"head": bench_stdout(), "worker": ""},
                nccl={"head": NCCL_SOCKET_LOG, "worker": NCCL_SOCKET_LOG},
            )
        ],
        extra=(
            Rule(
                prefix=("docker", "stop"),
                node="head",
                replies=(Reply(exit_code=1, stderr="permission denied"),),
            ),
        ),
    )
    runner = runner_of(tmp_path, script)

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert "片付けが終わらなかった" in outcome.detail
    assert "Socket" in outcome.detail


def test_a_name_conflict_on_docker_run_stops_nothing(tmp_path: Path) -> None:
    """名前の衝突で `docker run` が失敗したら、その名前に `stop` / `rm` を向けない (3.1)。"""
    config = job_config()
    plans = plans_of(config)
    conflict = (
        'docker: Error response from daemon: Conflict. The container name "/'
        f'{plans[0].container_name}" is already in use by container "0f0f0f".'
    )
    script = JobScript(
        rounds=[Round(plans=plans)],
        extra=(
            # 自分のラベルで絞った一覧は、つねに空 (衝突した相手は、自分のものではない)
            Rule(prefix=OWN_CONTAINERS_ARGV, replies=(Reply(stdout=""),)),
            Rule(
                prefix=("docker", "run"),
                node="head",
                replies=(Reply(exit_code=125, stderr=conflict),),
            ),
        ),
    )
    runner = runner_of(tmp_path, script)

    outcome = run_bandwidth(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert "衝突" in outcome.detail
    # head で失敗したので worker は起こさず、どちらの名前にも stop / rm を向けない
    assert started_names(runner) == (plans[0].container_name,)
    assert argv_of(runner, "docker", "stop") == ()
    assert argv_of(runner, "docker", "rm") == ()


def test_sanity_also_needs_the_fast_path(tmp_path: Path) -> None:
    """事前の確認は、4 段が通っても、経路が高速でなければ合格にしない (design の合否)。"""
    config = job_config(name=SANITY_CONFIG, script="/payload/vllm_sanity_check.py")
    output = sanity_stdout(4)
    script = JobScript(
        rounds=[
            Round(
                plans=plans_of(config, tag=SANITY_TAG),
                tag=SANITY_TAG,
                stdout={"head": output, "worker": output},
                nccl={"head": NCCL_SOCKET_LOG, "worker": NCCL_SOCKET_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    outcome = run_sanity(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert outcome.passed_stages == 4
    assert "Socket" in outcome.detail
    head = outcome.nccl["head"]
    assert head is not None
    assert head.ib_no_device is True


def test_no_launch_record_is_pushed(tmp_path: Path) -> None:
    """ジョブは、起動の記録を配らない (配布の呼び出しが 1 つも出ない)。"""
    config = job_config()
    script = JobScript(
        rounds=[
            Round(
                plans=plans_of(config),
                stdout={"head": bench_stdout(), "worker": ""},
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
            )
        ]
    )
    runner = runner_of(tmp_path, script)

    run_bandwidth(runner, config, tmp_path)

    assert runner.pushes == ()


# --- 構成の検査 (Spark に触る前) --------------------------------------------


def test_a_config_that_is_not_a_job_is_refused_before_touching_spark(tmp_path: Path) -> None:
    """`kind` が `job` でない構成は、Spark に触る前に断る。"""
    config = job_config(kind="serve")
    runner = runner_of(tmp_path, JobScript(rounds=[]))

    with pytest.raises(ConfigError, match="job"):
        run_bandwidth(runner, config, tmp_path)

    assert runner.calls == ()


def test_a_one_node_job_is_refused_before_touching_spark(tmp_path: Path) -> None:
    """1 台の構成は、2 台の間の通信を確かめられないので、触る前に断る。"""
    config = job_config(nodes=("head",))
    runner = runner_of(tmp_path, JobScript(rounds=[]))

    with pytest.raises(ConfigError, match="2 台"):
        run_bandwidth(runner, config, tmp_path)

    assert runner.calls == ()


def test_ab_without_added_env_is_refused(tmp_path: Path) -> None:
    """足す環境変数がなければ、A/B の比較にならないので断る。"""
    config = job_config()
    runner = runner_of(tmp_path, JobScript(rounds=[]))

    with pytest.raises(ValueError, match="環境変数"):
        run_ab(runner, config, tmp_path, extra_env={})

    assert runner.calls == ()
