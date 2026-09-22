"""構成から、コンテナの起動の引数の列を組み立てる部品の試験 (tasks.md 2.1)。

確かめること (design.md 「組み立てと読み取り › plan」、tasks.md 2.1 の完了の状態):

- 見本の構成から作った 2 台ぶんの引数の列が、ここに固定した列と一致する
- 2 台の差が、順位、API を持たない指定 (`--headless`)、自分のアドレス、置き換えの印
  (と、役割そのものを表す名前とラベル) だけである
- イメージの参照のあとの引数が、TOML に書いた順に並び、フラグを持たない設定が位置の
  引数になる
- 重みの参照を持つ構成にラベルが 8 つ、持たない `inspect` の構成に 7 つ付く
  (A/B の回は、腕と回のラベルが足される)
- 説明と理由と根拠と `ready_timeout_s` だけを変えても `config-sha256` のラベルが
  変わらず、フラグか値を 1 つ変えると変わる (requirements 3.10)
- Mac の側に置いたトークンの変数が、どの引数にも現れない (requirements 2.6)
- イメージをビルドする列、前面で動かす列、`--rm` を含む列を作る経路がない
  (requirements 8.6。5 つの `kind` のすべてで、組み立てた列を検査する)
- JSON の値を持つ `--hf-overrides` が、壊れずに 1 つの引数として並ぶ
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from ipaddress import IPv4Address
from typing import Any

import pytest
from pydantic import HttpUrl

from serving_kit import plan as p
from serving_kit.config import ConfigError
from serving_kit.types import (
    ConfigDef,
    ConfigKind,
    ContainerPlan,
    ImageRef,
    NodeDef,
    NodeRole,
    Setting,
    WeightsRef,
)

# --- 見本の組み立て -----------------------------------------------------

DIGEST = "b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
IMAGE_REF = f"vllm/vllm-openai@sha256:{DIGEST}"
REPO = "RedHatAI/GLM-5.3-Flash-NVFP4"
REVISION = "18d55bfd" + "0" * 32
MOUNT_AT = "/models/nvfp4"
REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"
MEASURED = "docs/results/2026-09-21-netcheck-links.md"

STARTED_AT = datetime(2026, 9, 21, 3, 0, 0, tzinfo=UTC)
STARTED_LABEL = "2026-09-21T03:00:00Z"

SOURCE = HttpUrl("https://docs.vllm.ai/en/latest/cli/serve/")
QUOTE = "vllm serve [model_tag] [options]"


def _setting(
    flag: str | None = None,
    value: str | None = None,
    *,
    why: str = "試験のための設定",
    only_on: NodeRole | None = None,
    is_port: bool = False,
) -> Setting:
    """根拠の付いた設定を 1 つ作る (根拠の中身は、この試験では問わない)。"""
    return Setting(
        flag=flag,
        value=value,
        why=why,
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
    manifest="RedHatAI__GLM-5.3-Flash-NVFP4.manifest.json",
    mount_at=MOUNT_AT,
    source=HttpUrl("https://huggingface.co/api/models/RedHatAI/GLM-5.3-Flash-NVFP4"),
    quote="List the content of a repository tree, with pagination support.",
)

HEAD = NodeDef(
    role="head",
    ssh_host="spark-153d",
    lan_addr=IPv4Address("10.0.1.60"),
    remote_root=REMOTE_ROOT,
    fabric_addr=IPv4Address("192.168.100.1"),
    fabric_ifname="enp1s0f0np0",
    fabric_measured=MEASURED,
)
WORKER = NodeDef(
    role="worker",
    ssh_host="spark-5083",
    lan_addr=IPv4Address("10.0.1.61"),
    remote_root=REMOTE_ROOT,
    fabric_addr=IPv4Address("192.168.100.2"),
    fabric_ifname="enp1s0f0np0",
    fabric_measured=MEASURED,
)
NODES: Mapping[NodeRole, NodeDef] = {"head": HEAD, "worker": WORKER}

_BARE = {"fabric_addr": None, "fabric_ifname": None, "fabric_measured": None}
BARE_NODES: Mapping[NodeRole, NodeDef] = {
    "head": HEAD.model_copy(update=_BARE),
    "worker": WORKER.model_copy(update=_BARE),
}
"""直結の値を実測する前のノードの定義 (`fetch`、`inspect`、`probe` は、これでも組める)。"""


def _serve_docker() -> dict[str, Setting]:
    return {
        "gpus": _setting("--gpus", "all"),
        "ipc": _setting("--ipc", "host"),
        "ulimit-memlock": _setting("--ulimit", "memlock=-1"),
        "network": _setting("--network", "host"),
        "mount-weights": _setting(
            "--mount",
            f"type=bind,source={{remote_root}}/models/nvfp4,target={MOUNT_AT},readonly",
        ),
        "mount-cache": _setting(
            "--mount", "type=bind,source={remote_root}/cache,target=/root/.cache"
        ),
    }


def _serve_args() -> dict[str, Setting]:
    return {
        "model-path": _setting(value="{weights.mount_at}"),
        "served-model-name": _setting("--served-model-name", "glm-5-3-flash"),
        "host": _setting("--host", "{head.lan_addr}"),
        "port": _setting("--port", "8000", is_port=True),
        "tensor-parallel-size": _setting("--tensor-parallel-size", "2"),
        "nnodes": _setting("--nnodes", "2"),
        "node-rank": _setting("--node-rank", "{node.rank}"),
        "headless": _setting("--headless", only_on="worker"),
        "master-addr": _setting("--master-addr", "{head.fabric_addr}"),
        "master-port": _setting("--master-port", "29501", is_port=True),
    }


def _serve_env() -> dict[str, Setting]:
    return {
        "vllm-host-ip": _setting("VLLM_HOST_IP", "{node.fabric_addr}"),
        "nccl-socket-ifname": _setting("NCCL_SOCKET_IFNAME", "{node.fabric_ifname}"),
    }


def serve_config() -> ConfigDef:
    """design.md Data Models の `p1-nvfp4-tp2` に沿った、2 台の推論サーバーの構成。"""
    return ConfigDef(
        name="p1-nvfp4-tp2",
        kind="serve",
        description="P1 の第一の構成。上流の公式のイメージ + NVFP4 + 2 台 TP=2",
        nodes=("head", "worker"),
        image=IMAGE,
        weights=WEIGHTS,
        docker=_serve_docker(),
        args=_serve_args(),
        env=_serve_env(),
        ready_timeout_s=1800,
        served_model_name="glm-5-3-flash",
    )


def probe_config() -> ConfigDef:
    """1 台の縮小の確認の構成 (直結の値を実測する前に流すので、直結の印を使わない)。"""
    return ConfigDef(
        name="probe-pinned",
        kind="probe",
        description="段 0。中身のない重みで、1 台だけ起こす",
        nodes=("head",),
        image=IMAGE,
        weights=WEIGHTS,
        docker={
            "gpus": _setting("--gpus", "all"),
            "mount-probe": _setting(
                "--mount", "type=bind,source={remote_root}/probe/nvfp4,target=/probe,readonly"
            ),
        },
        args={
            "model-path": _setting(value="/probe"),
            "served-model-name": _setting("--served-model-name", "glm-5-3-flash"),
            "load-format": _setting("--load-format", "dummy"),
            "hf-overrides": _setting("--hf-overrides", '{"num_hidden_layers": 4}'),
        },
        env={"logging-level": _setting("VLLM_LOGGING_LEVEL", "DEBUG")},
        ready_timeout_s=600,
        served_model_name="glm-5-3-flash",
    )


def job_config() -> ConfigDef:
    """通信の確認のジョブの構成 (2 台。A/B の回で使う)。"""
    return ConfigDef(
        name="netcheck-bandwidth",
        kind="job",
        description="2 台の all-reduce の帯域を測る",
        nodes=("head", "worker"),
        image=IMAGE,
        weights=None,
        docker={
            "gpus": _setting("--gpus", "all"),
            "network": _setting("--network", "host"),
            "entrypoint": _setting("--entrypoint", "torchrun"),
            "mount-payload": _setting(
                "--mount", "type=bind,source={remote_root}/payload,target=/payload,readonly"
            ),
        },
        args={
            "nnodes": _setting("--nnodes", "2"),
            "node-rank": _setting("--node-rank", "{node.rank}"),
            "rdzv-endpoint": _setting("--rdzv_endpoint", "{head.fabric_addr}:29502"),
            "script": _setting(value="/payload/allreduce_bench.py"),
        },
        env={"nccl-socket-ifname": _setting("NCCL_SOCKET_IFNAME", "{node.fabric_ifname}")},
        ready_timeout_s=900,
        served_model_name=None,
    )


def fetch_config() -> ConfigDef:
    """重みの取得の構成 (2 台。直結の値を実測する前に流す)。"""
    return ConfigDef(
        name="p1-fetch-nvfp4",
        kind="fetch",
        description="第一の候補の重みを 2 台に取得する",
        nodes=("head", "worker"),
        image=IMAGE,
        weights=WEIGHTS,
        docker={
            "entrypoint": _setting("--entrypoint", "hf"),
            "mount-models": _setting(
                "--mount", "type=bind,source={remote_root}/models,target=/models"
            ),
        },
        args={
            "download": _setting(value="download"),
            "repo": _setting(value=REPO),
            "revision": _setting("--revision", REVISION),
            "local-dir": _setting("--local-dir", "{weights.mount_at}"),
            "max-workers": _setting("--max-workers", "8"),
        },
        env={"telemetry": _setting("HF_HUB_DISABLE_TELEMETRY", "1")},
        ready_timeout_s=36000,
        served_model_name=None,
    )


def inspect_config() -> ConfigDef:
    """イメージの中のライセンスの表記の読み取りの構成 (重みの参照を持たない)。"""
    return ConfigDef(
        name="p1-image-licenses",
        kind="inspect",
        description="イメージの中のライセンスの表記を読む",
        nodes=("head",),
        image=IMAGE,
        weights=None,
        docker={"entrypoint": _setting("--entrypoint", "cat")},
        args={"license": _setting(value="/usr/share/doc/vllm/LICENSE")},
        env={},
        ready_timeout_s=120,
        served_model_name=None,
    )


ALL_KINDS: Mapping[ConfigKind, ConfigDef] = {
    "serve": serve_config(),
    "probe": probe_config(),
    "job": job_config(),
    "fetch": fetch_config(),
    "inspect": inspect_config(),
}


# --- 試験の側で組み直す、直列化と取り出し --------------------------------


def _frame(text: str) -> str:
    """長さを前に付けて、区切りの曖昧さをなくす (仕様を、試験の側にも書く)。"""
    return f"{len(text.encode('utf-8'))}:{text}"


def _without_labels(argv: Sequence[str]) -> list[str]:
    """`--label` の引数 (フラグとその値) を除く。"""
    kept: list[str] = []
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        if item == "--label":
            skip = True
            continue
        kept.append(item)
    return kept


def _expected_sha(plans: Sequence[ContainerPlan]) -> str:
    """2 台ぶんの、ラベルを除いた引数の列から、期待する sha256 を作る。"""
    frames: list[str] = []
    for item in plans:
        stripped = _without_labels(item.argv)
        frames.append(_frame(item.node))
        frames.append(_frame(str(len(stripped))))
        frames.extend(_frame(arg) for arg in stripped)
    return hashlib.sha256("".join(frames).encode("utf-8")).hexdigest()


def _sha_label(item: ContainerPlan) -> str:
    return item.labels["vllm-baseline.config-sha256"]


def _build(
    config: ConfigDef, nodes: Mapping[NodeRole, NodeDef] = NODES
) -> tuple[ContainerPlan, ...]:
    return p.build_plans(config, nodes, STARTED_AT)


# --- 固定した引数の列 ---------------------------------------------------


def test_serve_plans_match_the_fixed_argv() -> None:
    """見本の構成から作った 2 台ぶんの列が、ここに固定した列と一致する。"""
    head, worker = _build(serve_config())
    sha = _expected_sha((head, worker))

    assert head.node == "head"
    assert head.container_name == "vb-p1-nvfp4-tp2-head"
    assert head.argv == (
        "docker",
        "run",
        "-d",
        "--pull",
        "never",
        "--name",
        "vb-p1-nvfp4-tp2-head",
        "--label",
        "vllm-baseline.config=p1-nvfp4-tp2",
        "--label",
        f"vllm-baseline.config-sha256={sha}",
        "--label",
        f"vllm-baseline.image={IMAGE_REF}",
        "--label",
        "vllm-baseline.kind=serve",
        "--label",
        "vllm-baseline.owner=serving-kit",
        "--label",
        "vllm-baseline.role=head",
        "--label",
        f"vllm-baseline.started-at={STARTED_LABEL}",
        "--label",
        f"vllm-baseline.weights={REPO}@{REVISION}",
        "--restart",
        "no",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--ulimit",
        "memlock=-1",
        "--network",
        "host",
        "--mount",
        f"type=bind,source={REMOTE_ROOT}/models/nvfp4,target={MOUNT_AT},readonly",
        "--mount",
        f"type=bind,source={REMOTE_ROOT}/cache,target=/root/.cache",
        "-e",
        "VLLM_HOST_IP=192.168.100.1",
        "-e",
        "NCCL_SOCKET_IFNAME=enp1s0f0np0",
        IMAGE_REF,
        MOUNT_AT,
        "--served-model-name",
        "glm-5-3-flash",
        "--host",
        "10.0.1.60",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "2",
        "--nnodes",
        "2",
        "--node-rank",
        "0",
        "--master-addr",
        "192.168.100.1",
        "--master-port",
        "29501",
    )

    assert worker.node == "worker"
    assert worker.container_name == "vb-p1-nvfp4-tp2-worker"
    assert worker.argv == (
        "docker",
        "run",
        "-d",
        "--pull",
        "never",
        "--name",
        "vb-p1-nvfp4-tp2-worker",
        "--label",
        "vllm-baseline.config=p1-nvfp4-tp2",
        "--label",
        f"vllm-baseline.config-sha256={sha}",
        "--label",
        f"vllm-baseline.image={IMAGE_REF}",
        "--label",
        "vllm-baseline.kind=serve",
        "--label",
        "vllm-baseline.owner=serving-kit",
        "--label",
        "vllm-baseline.role=worker",
        "--label",
        f"vllm-baseline.started-at={STARTED_LABEL}",
        "--label",
        f"vllm-baseline.weights={REPO}@{REVISION}",
        "--restart",
        "no",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--ulimit",
        "memlock=-1",
        "--network",
        "host",
        "--mount",
        f"type=bind,source={REMOTE_ROOT}/models/nvfp4,target={MOUNT_AT},readonly",
        "--mount",
        f"type=bind,source={REMOTE_ROOT}/cache,target=/root/.cache",
        "-e",
        "VLLM_HOST_IP=192.168.100.2",
        "-e",
        "NCCL_SOCKET_IFNAME=enp1s0f0np0",
        IMAGE_REF,
        MOUNT_AT,
        "--served-model-name",
        "glm-5-3-flash",
        "--host",
        "10.0.1.60",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "2",
        "--nnodes",
        "2",
        "--node-rank",
        "1",
        "--headless",
        "--master-addr",
        "192.168.100.1",
        "--master-port",
        "29501",
    )


def test_only_the_rank_headless_own_address_and_names_differ() -> None:
    """2 台の差が、順位、`--headless`、自分のアドレスと、役割そのものだけである。

    head の列に、この 5 か所の差を当てると、worker の列とちょうど一致する。
    """
    head, worker = _build(serve_config())
    expected = list(head.argv)
    expected[expected.index("vb-p1-nvfp4-tp2-head")] = "vb-p1-nvfp4-tp2-worker"
    expected[expected.index("vllm-baseline.role=head")] = "vllm-baseline.role=worker"
    expected[expected.index("VLLM_HOST_IP=192.168.100.1")] = "VLLM_HOST_IP=192.168.100.2"
    expected[expected.index("--node-rank") + 1] = "1"
    expected.insert(expected.index("--master-addr"), "--headless")
    assert tuple(expected) == worker.argv


def test_args_keep_the_written_order_and_flagless_settings_are_positional() -> None:
    """イメージの参照のあとの引数が、書いた順に並び、フラグのない設定が位置の引数になる。"""
    head, _ = _build(serve_config())
    after_image = head.argv[head.argv.index(IMAGE_REF) + 1 :]
    assert after_image[0] == MOUNT_AT, "フラグのない設定が、値だけの位置の引数として先頭に来る"
    flags = [item for item in after_image if item.startswith("--")]
    assert flags == [
        "--served-model-name",
        "--host",
        "--port",
        "--tensor-parallel-size",
        "--nnodes",
        "--node-rank",
        "--master-addr",
        "--master-port",
    ]


def test_docker_settings_come_before_the_image_reference() -> None:
    """docker の引数 → イメージの参照 → `args` の順に並ぶ。"""
    head, _ = _build(serve_config())
    image_at = head.argv.index(IMAGE_REF)
    assert head.argv.index("--gpus") < image_at
    assert head.argv.index("-e") < image_at
    assert head.argv.index("--served-model-name") > image_at


# --- ラベル -------------------------------------------------------------


def test_labels_are_eight_with_weights_and_seven_without() -> None:
    """重みの参照を持つ構成に 8 つ、持たない `inspect` の構成に 7 つ付く。"""
    head, worker = _build(serve_config())
    common = {
        "vllm-baseline.owner": "serving-kit",
        "vllm-baseline.config": "p1-nvfp4-tp2",
        "vllm-baseline.kind": "serve",
        "vllm-baseline.image": IMAGE_REF,
        "vllm-baseline.started-at": STARTED_LABEL,
        "vllm-baseline.weights": f"{REPO}@{REVISION}",
    }
    assert len(head.labels) == 8
    assert head.labels["vllm-baseline.role"] == "head"
    assert worker.labels["vllm-baseline.role"] == "worker"
    for key, value in common.items():
        assert head.labels[key] == value
        assert worker.labels[key] == value

    (only,) = _build(inspect_config(), BARE_NODES)
    assert len(only.labels) == 7
    assert "vllm-baseline.weights" not in only.labels
    assert only.labels["vllm-baseline.kind"] == "inspect"


def test_labels_appear_in_the_argv_as_label_pairs() -> None:
    """ラベルは、すべて `--label <鍵>=<値>` として列に並ぶ。"""
    head, _ = _build(serve_config())
    pairs = {
        head.argv[index + 1]
        for index, item in enumerate(head.argv)
        if item == "--label" and index + 1 < len(head.argv)
    }
    assert pairs == {f"{key}={value}" for key, value in head.labels.items()}


def test_started_at_must_carry_a_timezone() -> None:
    """素の日時は受けない (記録の時刻は UTC)。"""
    with pytest.raises(ConfigError, match="started_at"):
        p.build_plans(serve_config(), NODES, datetime(2026, 9, 21, 3, 0, 0))


# --- config-sha256 ------------------------------------------------------


def test_config_sha256_is_the_same_on_both_nodes() -> None:
    """2 台のどちらのコンテナにも、同じ値が付く。"""
    head, worker = _build(serve_config())
    assert _sha_label(head) == _sha_label(worker) == _expected_sha((head, worker))


def test_config_sha256_ignores_description_why_evidence_and_timeout() -> None:
    """説明、理由、根拠、`ready_timeout_s` を変えても、sha256 は変わらない。"""
    before = _build(serve_config())
    args = _serve_args()
    args["port"] = Setting(
        flag="--port",
        value="8000",
        why="理由と根拠だけを書き換える",
        is_port=True,
        measured=MEASURED,
    )
    after = _build(
        serve_config().model_copy(
            update={
                "description": "説明を書き換える",
                "ready_timeout_s": 3600,
                "args": args,
            }
        )
    )
    assert _sha_label(after[0]) == _sha_label(before[0])


def _changed_args(key: str, setting: Setting) -> dict[str, Any]:
    args = _serve_args()
    args[key] = setting
    return {"args": args}


@pytest.mark.parametrize(
    "update",
    [
        pytest.param(_changed_args("port", _setting("--port", "8001", is_port=True)), id="value"),
        pytest.param(_changed_args("nnodes", _setting("--num-nodes", "2")), id="flag"),
        pytest.param(
            {"image": IMAGE.model_copy(update={"ref": f"vllm/vllm-openai@sha256:{'a' * 64}"})},
            id="image",
        ),
        pytest.param(
            {"weights": WEIGHTS.model_copy(update={"mount_at": "/models/other"})}, id="weights"
        ),
    ],
)
def test_config_sha256_changes_with_the_container_contents(update: dict[str, Any]) -> None:
    """フラグ、値、イメージ、重みを変えると、sha256 が変わる。"""
    before = _build(serve_config())
    after = _build(serve_config().model_copy(update=update))
    assert _sha_label(after[0]) != _sha_label(before[0])


def test_config_sha256_changes_when_a_node_value_changes() -> None:
    """ノードの値 (直結のアドレス) を変えると、sha256 が変わる。"""
    before = _build(serve_config())
    moved: Mapping[NodeRole, NodeDef] = {
        "head": HEAD,
        "worker": WORKER.model_copy(update={"fabric_addr": IPv4Address("192.168.100.9")}),
    }
    after = _build(serve_config(), moved)
    assert _sha_label(after[0]) != _sha_label(before[0])


def _sha_of_args(args: dict[str, Setting]) -> str:
    """`args` だけを差し替えた 1 台の構成の `config-sha256`。"""
    config = inspect_config().model_copy(update={"args": args})
    return _sha_label(_build(config, BARE_NODES)[0])


def test_config_sha256_does_not_confuse_one_argument_with_two() -> None:
    """`["a b"]` と `["a", "b"]` が、同じ sha256 にならない (区切りの曖昧さがない)。"""
    joined = _sha_of_args({"one": _setting(value="a b")})
    split = _sha_of_args({"one": _setting(value="a"), "two": _setting(value="b")})
    assert joined != split


def test_config_sha256_does_not_confuse_where_the_split_falls() -> None:
    """語の数が同じでも、切れ目が違えば sha256 が違う (長さを前に付けているため)。"""
    early = _sha_of_args({"one": _setting(value="a"), "two": _setting(value="bc")})
    late = _sha_of_args({"one": _setting(value="ab"), "two": _setting(value="c")})
    assert early != late


def test_config_sha256_excludes_the_label_arguments() -> None:
    """起こした時刻だけが違う 2 回は、同じ sha256 になる (ラベルを除いて取る)。"""
    early = p.build_plans(serve_config(), NODES, STARTED_AT)
    later = p.build_plans(serve_config(), NODES, datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC))
    assert _sha_label(early[0]) == _sha_label(later[0])
    assert (
        early[0].labels["vllm-baseline.started-at"] != later[0].labels["vllm-baseline.started-at"]
    )


# --- 置き換えの印 -------------------------------------------------------


def test_every_marker_is_filled() -> None:
    """7 つの印が、それぞれの値で埋まり、印そのものは列に残らない。"""
    head, worker = _build(serve_config())
    for item in (head, worker):
        assert not [arg for arg in item.argv if "{" in arg], item.argv
    assert "VLLM_HOST_IP=192.168.100.1" in head.argv
    assert "VLLM_HOST_IP=192.168.100.2" in worker.argv
    assert "NCCL_SOCKET_IFNAME=enp1s0f0np0" in head.argv
    assert f"type=bind,source={REMOTE_ROOT}/cache,target=/root/.cache" in head.argv
    assert MOUNT_AT in head.argv
    assert "10.0.1.60" in head.argv and "10.0.1.60" in worker.argv
    assert "192.168.100.1" in worker.argv, "{head.fabric_addr} は、2 台とも head の値になる"
    assert head.argv[head.argv.index("--node-rank") + 1] == "0"
    assert worker.argv[worker.argv.index("--node-rank") + 1] == "1"


def test_json_values_survive_the_replacement() -> None:
    """JSON の波かっこを壊さずに、1 つの引数として並べる。"""
    (only,) = _build(probe_config(), BARE_NODES)
    assert '{"num_hidden_layers": 4}' in only.argv
    assert only.argv[only.argv.index("--hf-overrides") + 1] == '{"num_hidden_layers": 4}'


def test_markers_in_the_flag_are_filled_too() -> None:
    """印は、値だけでなくフラグの側に書いても埋まる。"""
    args = dict(probe_config().args)
    args["iface"] = _setting("--iface-{node.fabric_ifname}", "1")
    config = probe_config().model_copy(update={"args": args})
    (only,) = _build(config, NODES)
    assert "--iface-enp1s0f0np0" in only.argv
    assert only.argv[only.argv.index("--iface-enp1s0f0np0") + 1] == "1"


def test_a_marker_without_a_value_in_the_flag_is_refused() -> None:
    """フラグの側の印も、埋める値がなければ、項目の名前 (`.flag`) を示して断る。"""
    args = dict(probe_config().args)
    args["iface"] = _setting("--iface-{node.fabric_ifname}", "1")
    config = probe_config().model_copy(update={"args": args})
    with pytest.raises(ConfigError) as caught:
        p.build_plans(config, BARE_NODES, STARTED_AT)
    assert "configs.probe-pinned.args.iface.flag" in str(caught.value)
    assert "{node.fabric_ifname}" in str(caught.value)


def test_a_marker_without_a_value_is_refused() -> None:
    """埋める値のない印は、黙って空の文字列にせず、項目の名前を示して断る。"""
    with pytest.raises(ConfigError) as caught:
        p.build_plans(serve_config(), BARE_NODES, STARTED_AT)
    message = str(caught.value)
    assert "{node.fabric_addr}" in message
    assert "configs.p1-nvfp4-tp2.env.vllm-host-ip" in message
    assert "fabric_addr" in message


def test_the_weights_marker_without_weights_is_refused() -> None:
    """重みを持たない構成で `{weights.mount_at}` を書いたら断る。"""
    broken = inspect_config().model_copy(
        update={"args": {"path": _setting(value="{weights.mount_at}")}}
    )
    with pytest.raises(ConfigError, match=r"\{weights\.mount_at\}"):
        p.build_plans(broken, BARE_NODES, STARTED_AT)


def test_a_missing_node_definition_is_refused() -> None:
    """構成が使う役割のノードの定義がなければ断る。"""
    with pytest.raises(ConfigError, match="nodes.worker"):
        p.build_plans(serve_config(), {"head": HEAD}, STARTED_AT)


def test_only_on_must_name_a_role_that_the_config_uses() -> None:
    """使わない役割を指す `only_on` は、黙って落とさずに断る。"""
    args = dict(probe_config().args)
    args["late"] = _setting("--late", "1", only_on="worker")
    broken = probe_config().model_copy(update={"args": args})
    with pytest.raises(ConfigError, match="only_on"):
        p.build_plans(broken, BARE_NODES, STARTED_AT)


# --- 環境変数 -----------------------------------------------------------


def test_the_host_environment_is_not_inherited(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mac の側に置いたトークンの変数が、どの引数にも現れない (requirements 2.6)。"""
    monkeypatch.setenv("HF_TOKEN", "hf_thisMustNeverLeaveTheMac")
    monkeypatch.setenv("VLLM_HOST_IP", "10.9.9.9")
    for config in ALL_KINDS.values():
        for item in _build(config):
            joined = "\n".join((*item.argv, *item.labels.values()))
            assert "hf_thisMustNeverLeaveTheMac" not in joined
            assert "10.9.9.9" not in joined
            assert "HF_TOKEN" not in joined


def test_env_is_passed_as_name_and_value_together() -> None:
    """`-e NAME` の値なしの形は使わない (docker がホストの環境から取るため)。"""
    head, _ = _build(serve_config())
    for index, item in enumerate(head.argv):
        if item == "-e":
            assert "=" in head.argv[index + 1], head.argv[index + 1]

    broken = serve_config().model_copy(update={"env": {"inherit": _setting("VLLM_HOST_IP", None)}})
    with pytest.raises(ConfigError, match="env.inherit"):
        p.build_plans(broken, NODES, STARTED_AT)


@pytest.mark.parametrize(
    "name", ["HF_TOKEN", "VLLM_API_KEY", "MY_SECRET", "DB_PASSWORD", "AWS_CREDENTIALS"]
)
def test_secret_looking_env_names_are_refused(name: str) -> None:
    """秘密らしい名前の環境変数を断る (requirements 2.6)。"""
    broken = serve_config().model_copy(update={"env": {"leak": _setting(name, "x")}})
    with pytest.raises(ConfigError) as caught:
        p.build_plans(broken, NODES, STARTED_AT)
    assert name in str(caught.value)


def test_extra_env_of_an_ab_arm_is_also_checked() -> None:
    """A/B の腕が足す環境変数も、同じ決まりで断る。"""
    with pytest.raises(ConfigError, match="HF_TOKEN"):
        p.build_plans(
            job_config(),
            NODES,
            STARTED_AT,
            arm="candidate",
            repeat_index=1,
            extra_env={"HF_TOKEN": "x"},
        )


# --- A/B の回 -----------------------------------------------------------


def test_ab_run_adds_the_repeat_to_the_name_the_labels_and_the_env() -> None:
    """腕と回の番号が、コンテナの名前とラベルと、足す環境変数に表れる。"""
    head, worker = p.build_plans(
        job_config(),
        NODES,
        STARTED_AT,
        arm="ifname-fabric",
        repeat_index=2,
        extra_env={"NCCL_IB_MERGE_NICS": "1"},
    )
    assert head.container_name == "vb-netcheck-bandwidth-head-r2"
    assert worker.container_name == "vb-netcheck-bandwidth-worker-r2"
    assert head.labels["vllm-baseline.run"] == "ifname-fabric-2"
    assert len(head.labels) == 8, "重みを持たない 7 つ + 腕と回で 8 つ"
    assert "NCCL_IB_MERGE_NICS=1" in head.argv
    assert head.argv.index("NCCL_IB_MERGE_NICS=1") < head.argv.index(IMAGE_REF)


def test_ab_arms_differ_in_the_config_sha256() -> None:
    """腕ごとに足す環境変数が違えば、sha256 も違う (中身に効く変更)。"""
    baseline = p.build_plans(job_config(), NODES, STARTED_AT, arm="baseline", repeat_index=1)
    candidate = p.build_plans(
        job_config(),
        NODES,
        STARTED_AT,
        arm="candidate",
        repeat_index=1,
        extra_env={"NCCL_IB_MERGE_NICS": "1"},
    )
    assert _sha_label(baseline[0]) != _sha_label(candidate[0])


def test_the_arm_and_the_repeat_go_together() -> None:
    """腕だけ、回だけの指定は断る。"""
    with pytest.raises(ConfigError, match="repeat_index"):
        p.build_plans(job_config(), NODES, STARTED_AT, arm="baseline")
    with pytest.raises(ConfigError, match="arm"):
        p.build_plans(job_config(), NODES, STARTED_AT, repeat_index=1)


# --- 1 つの起動の仕組み (requirements 8.6、2.3) --------------------------


FOREGROUND = frozenset({"-i", "-t", "-it", "-ti", "-a", "--attach", "--interactive", "--tty"})


@pytest.mark.parametrize("kind", sorted(ALL_KINDS))
def test_every_kind_is_started_detached_without_build_or_rm(kind: ConfigKind) -> None:
    """5 つの `kind` のどれも、`docker run -d` で起こし、前面も自動の削除も作らない。"""
    for item in _build(ALL_KINDS[kind], NODES):
        assert item.argv[:2] == ("docker", "run"), item.argv[:2]
        assert "build" not in item.argv
        assert "-d" in item.argv
        assert item.argv[item.argv.index("--pull") + 1] == "never"
        assert item.argv[item.argv.index("--restart") + 1] == "no"
        assert "--rm" not in item.argv
        assert not (FOREGROUND & set(item.argv)), item.argv
        assert item.labels["vllm-baseline.owner"] == "serving-kit"
        assert item.container_name.startswith("vb-")


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--name", "mine"),
        ("--label", "vllm-baseline.owner=someone-else"),
        ("--label-file", "/tmp/labels"),
        ("-d", None),
        ("--pull", "always"),
        ("--restart", "no"),
        ("--rm", None),
        ("-it", None),
    ],
)
def test_flags_that_the_tool_adds_are_refused_in_the_config(flag: str, value: str | None) -> None:
    """道具が必ず付ける引数と、前面で動かす指定を、構成に書いてあれば断る。"""
    docker = dict(inspect_config().docker)
    docker["extra"] = _setting(flag, value)
    broken = inspect_config().model_copy(update={"docker": docker})
    with pytest.raises(ConfigError) as caught:
        p.build_plans(broken, BARE_NODES, STARTED_AT)
    assert flag in str(caught.value)


def _refused_docker(setting: Setting) -> str:
    """`docker` の節に 1 つ足した構成を組み立てて、断りの文を返す。"""
    docker = dict(inspect_config().docker)
    docker["extra"] = setting
    broken = inspect_config().model_copy(update={"docker": docker})
    with pytest.raises(ConfigError) as caught:
        p.build_plans(broken, BARE_NODES, STARTED_AT)
    return str(caught.value)


@pytest.mark.parametrize(
    ("setting", "shown"),
    [
        # 短い形 (指摘 1: `-l` が抜けていると、道具のラベルを後勝ちで畳める)
        pytest.param(_setting("-l", "vllm-baseline.owner=someone-else"), "-l", id="short-label"),
        pytest.param(_setting("-d"), "-d", id="short-detach"),
        pytest.param(_setting("-i"), "-i", id="short-interactive"),
        pytest.param(_setting("-t"), "-t", id="short-tty"),
        pytest.param(_setting("-a", "stdout"), "-a", id="short-attach"),
        # 環境変数のフラグ (指摘 2: `env` の節を通らないので、秘密の検査が効かない)
        pytest.param(_setting("-e", "HF_TOKEN=leak"), "-e", id="short-env"),
        pytest.param(_setting("--env", "HF_TOKEN=leak"), "--env", id="long-env"),
        pytest.param(_setting("-e", None), "-e", id="env-without-value"),
        pytest.param(_setting("--env-file", "/tmp/secrets.env"), "--env-file", id="env-file"),
        # `=` つなぎ
        pytest.param(_setting("--env=HF_TOKEN=leak"), "--env", id="inline-env"),
        pytest.param(_setting("--label=vllm-baseline.owner=x"), "--label", id="inline-label"),
        pytest.param(_setting("--env-file=/tmp/secrets.env"), "--env-file", id="inline-env-file"),
        pytest.param(_setting("--detach=true"), "--detach", id="inline-detach"),
        # 位置の引数として書く (`flag` を書かない)
        pytest.param(_setting(value="-l"), "-l", id="positional-short-label"),
        pytest.param(_setting(value="-e"), "-e", id="positional-short-env"),
        pytest.param(_setting(value="--env-file=/tmp/secrets.env"), "--env-file", id="positional"),
        pytest.param(_setting(value="--rm"), "--rm", id="positional-rm"),
        # 前後の空白
        pytest.param(_setting(" --label", "a=b"), "--label", id="space-before-label"),
        pytest.param(_setting("-l ", "a=b"), "-l", id="space-after-short-label"),
        pytest.param(_setting(" -e", "HF_TOKEN=leak"), "-e", id="space-before-short-env"),
        # 短い形の組み合わせ (1 文字ずつ見る)
        pytest.param(_setting("-itd"), "-itd", id="combined-itd"),
        pytest.param(_setting("-id"), "-id", id="combined-id"),
        pytest.param(_setting("-td"), "-td", id="combined-td"),
        pytest.param(_setting("-ia", "stdout"), "-ia", id="combined-ia"),
        pytest.param(_setting("-le", "a=b"), "-le", id="combined-le"),
        # 値を詰めて書く短い形 (先頭の文字で断る)
        pytest.param(_setting("-eHF_TOKEN=leak"), "-e", id="tight-env"),
        pytest.param(_setting("-lvllm-baseline.owner=x"), "-l", id="tight-label"),
    ],
)
def test_the_docker_section_cannot_reach_the_tool_owned_flags(setting: Setting, shown: str) -> None:
    """道具のフラグと環境変数のフラグは、どの書き方でも `docker` の節から書けない。

    要件 2.3 (自分のもの以外を触らない) と 2.6 (認証の情報を Spark に置かない) は、
    ラベルと `env` の節を通ることで保たれるので、その抜け道をすべて塞ぐ。
    """
    message = _refused_docker(setting)
    assert shown in message
    assert "configs.p1-image-licenses.docker.extra" in message


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        pytest.param(_setting("--gpus", "all"), ("--gpus", "all"), id="gpus"),
        pytest.param(_setting("--network", "host"), ("--network", "host"), id="network"),
        pytest.param(_setting("--ipc", "host"), ("--ipc", "host"), id="ipc-is-not-short-i"),
        pytest.param(_setting("--ulimit", "memlock=-1"), ("--ulimit", "memlock=-1"), id="ulimit"),
        pytest.param(_setting("--shm-size", "16g"), ("--shm-size", "16g"), id="shm-size"),
        pytest.param(
            _setting("--entrypoint", "torchrun"),
            ("--entrypoint", "torchrun"),
            id="entrypoint-is-not-short-e",
        ),
        pytest.param(
            _setting("--mount", "type=bind,source={remote_root}/x,target=/x,readonly"),
            ("--mount", f"type=bind,source={REMOTE_ROOT}/x,target=/x,readonly"),
            id="mount",
        ),
        pytest.param(
            _setting("--device", "/dev/infiniband"),
            ("--device", "/dev/infiniband"),
            id="device-is-not-short-d",
        ),
        pytest.param(_setting("--cap-add", "IPC_LOCK"), ("--cap-add", "IPC_LOCK"), id="cap-add"),
        pytest.param(_setting("--workdir", "/x"), ("--workdir", "/x"), id="workdir"),
        pytest.param(_setting("--tmpfs", "/tmp"), ("--tmpfs", "/tmp"), id="tmpfs-is-not-short-t"),
    ],
)
def test_ordinary_docker_settings_are_not_refused(
    setting: Setting, expected: tuple[str, str]
) -> None:
    """正当な docker の設定を、抜け道の検査で誤って断らない。

    `docker` の節は、この 1 つだけにする (見本の `--entrypoint cat` と同じフラグを
    確かめる回があるので、どちらを見たかが曖昧にならないようにする)。
    """
    config = inspect_config().model_copy(update={"docker": {"only": setting}})
    (only,) = _build(config, BARE_NODES)
    at = only.argv.index(expected[0])
    assert only.argv[at : at + 2] == expected


def test_container_names_are_refused_when_the_config_name_is_unusable() -> None:
    """コンテナの名前にできない構成の名前は、断る。"""
    broken = inspect_config().model_copy(update={"name": "p1_image licenses"})
    with pytest.raises(ConfigError, match="コンテナの名前"):
        p.build_plans(broken, BARE_NODES, STARTED_AT)


# --- 道具が必ず付ける引数の根拠 -----------------------------------------


def test_the_tool_added_flags_carry_evidence() -> None:
    """道具が必ず付ける 5 つにも、構成と同じ形の根拠がある (design.md 「plan」)。"""
    assert set(p.TOOL_SETTINGS) == {"detach", "pull", "name", "label", "restart"}
    for key, setting in p.TOOL_SETTINGS.items():
        assert setting.why.strip(), key
        assert setting.source is not None, key
        assert setting.quote is not None and setting.quote.strip(), key
        assert str(setting.source).startswith("https://docs.docker.com/"), key


# --- 依存の向きと、入出力のなさ -----------------------------------------


ALLOWED_IMPORT_ROOTS = frozenset(
    {"__future__", "collections", "datetime", "hashlib", "re", "typing", "pydantic", "serving_kit"}
)
FORBIDDEN_IMPORTS = frozenset({"os", "subprocess", "socket", "pathlib", "time", "random", "httpx"})


def test_plan_module_reads_nothing_from_the_outside() -> None:
    """`plan` は、`types` と `config` だけを読み込み、入出力の道具を持たない。"""
    tree = ast.parse(inspect.getsource(p))
    roots: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "相対の読み込みがある"
            assert node.module is not None
            roots.add(node.module.split(".")[0])
            modules.add(node.module)

    assert roots <= ALLOWED_IMPORT_ROOTS, f"許していない読み込み: {roots - ALLOWED_IMPORT_ROOTS}"
    assert not (roots & FORBIDDEN_IMPORTS)
    assert {module for module in modules if module.startswith("serving_kit")} <= {
        "serving_kit.types",
        "serving_kit.config",
    }
