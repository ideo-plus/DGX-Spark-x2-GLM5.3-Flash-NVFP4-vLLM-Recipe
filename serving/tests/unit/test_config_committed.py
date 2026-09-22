"""コミットした構成の定義とノードの定義を、そのまま固定する試験 (tasks.md 6.2)。

確かめること (tasks.md 6.2 の完了の状態、requirements 3.1、3.2、3.6、6.2、6.7):

- `serving/config/configs.toml` と `serving/config/nodes.toml` が、`config.load_configs` /
  `config.load_nodes` でそのまま読み込める (根拠の検査を含む、drift の見張り)
- 第一の構成 `p1-nvfp4-tp2` から、試験用の直結の値を与えて組み立てた 2 台ぶんの引数の列が、
  この試験に固定した列と一致する。2 台の差が、名前と役割のラベル、自分のアドレス、順位、
  API を持たない指定 (`--headless`) だけである
- 縮小の確認、取得、読み取り、通信の確認の構成が `select_config` で選べる
- 第一の構成は、**直結の値がないという理由だけ**で選べない (requirements 4.7)
- どの構成にも、投機的デコードの指定がない (requirements 6.7)

この試験は、Spark に 1 度も触らない (読み込みと、純粋な組み立てだけを流す)。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from ipaddress import IPv4Address
from pathlib import Path

import pytest

from serving_kit import config as c
from serving_kit import plan as p
from serving_kit.types import ConfigDef, NodeDef, NodeRole

# --- コミットしたファイルの場所 ----------------------------------------

SERVING_DIR = Path(__file__).resolve().parents[2]
"""`serving/` (この試験から見て 2 つ上)。"""

REPO_ROOT = SERVING_DIR.parent
"""リポジトリの最上位 (`serving/weights/` の実在の検査が、ここを起点に見る)。"""

CONFIGS_PATH = SERVING_DIR / "config" / "configs.toml"
NODES_PATH = SERVING_DIR / "config" / "nodes.toml"

# --- 固定した値 --------------------------------------------------------

IMAGE_REF = (
    "vllm/vllm-openai@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
)
WEIGHTS_LABEL = "RedHatAI/GLM-5.3-Flash-NVFP4@18d55bfd5a2194887738da73753975c9d3842f46"
REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"
MOUNT_AT = "/models/glm-5-3-flash-nvfp4"

STARTED_AT = datetime(2026, 9, 22, 0, 0, 0, tzinfo=UTC)
STARTED_LABEL = "2026-09-22T00:00:00Z"

CONFIG_SHA256 = "daf24e95a3e04e6902408abe2d2c97bd8082fad68ae783c27dc855eef931d7ab"
"""`p1-nvfp4-tp2` の `config-sha256`。

構成の値 (フラグ、値、イメージ、重み、ノードの値) を 1 つでも変えると変わるので、この
1 行が、意図しない変更の見張りになる (design.md 「plan」)。`description` / `why` / 根拠 /
`ready_timeout_s` を直しても変わらない。
"""

EXPECTED_NAMES = frozenset(
    {
        "probe-pinned",
        "p1-nvfp4-tp2",
        "p1-fetch-nvfp4",
        "p1-fetch-nvfp4-probe",
        "p1-image-licenses",
        "p1-image-hf-version",
        "netcheck-bandwidth",
        "netcheck-sanity",
    }
)

FABRIC_HEAD_ADDR = "192.168.100.1"
FABRIC_WORKER_ADDR = "192.168.100.2"
FABRIC_IFNAME = "enp1s0f0np0"
FABRIC_MEASURED = "docs/results/2026-09-21-netcheck-links.md"
"""試験用の直結の値。**実測ではない** (7.3 が実測で `nodes.toml` を埋めるまで、コミットした
ノードの定義の直結の値は空である)。`fabric_measured` は、`config.load_nodes` が実在を求める
ので、この試験では `nodes.toml` を読んだあとに差し替えて使い、ファイルには書かない。
"""


# --- 読み込み ----------------------------------------------------------


def _load_configs() -> dict[str, ConfigDef]:
    return c.load_configs(CONFIGS_PATH, REPO_ROOT)


def _load_nodes() -> dict[NodeRole, NodeDef]:
    return c.load_nodes(NODES_PATH, REPO_ROOT)


def _with_fabric(nodes: Mapping[NodeRole, NodeDef]) -> dict[NodeRole, NodeDef]:
    """試験用の直結の値を与えたノードの定義 (`fabric_measured` は検査を通る道筋にする)。"""
    addrs = {"head": FABRIC_HEAD_ADDR, "worker": FABRIC_WORKER_ADDR}
    return {
        role: node.model_copy(
            update={
                "fabric_addr": IPv4Address(addrs[role]),
                "fabric_ifname": FABRIC_IFNAME,
                "fabric_measured": FABRIC_MEASURED,
            }
        )
        for role, node in nodes.items()
    }


def test_committed_files_load_as_they_are() -> None:
    """コミットした 2 つのファイルが、そのまま読み込める (drift の見張り)。"""
    configs = _load_configs()
    nodes = _load_nodes()

    assert set(configs) == set(EXPECTED_NAMES)
    assert set(nodes) == {"head", "worker"}
    assert nodes["head"].ssh_host == "spark-153d"
    assert str(nodes["head"].lan_addr) == "10.0.1.60"
    assert nodes["head"].remote_root == REMOTE_ROOT
    assert nodes["worker"].ssh_host == "spark-5083"
    assert str(nodes["worker"].lan_addr) == "10.0.1.61"
    assert nodes["worker"].remote_root == REMOTE_ROOT


def test_committed_nodes_leave_the_fabric_values_empty() -> None:
    """直結の値は、7.3 の実測まで空のままである (見込みの値を書かない)。"""
    for node in _load_nodes().values():
        assert node.fabric_addr is None
        assert node.fabric_ifname is None
        assert node.fabric_measured is None


# --- 固定した引数の列 --------------------------------------------------


def _label_argv(role: str) -> tuple[str, ...]:
    """ラベルの列 (鍵の順。重みを持つ構成なので 8 つ)。"""
    labels = {
        "vllm-baseline.config": "p1-nvfp4-tp2",
        "vllm-baseline.config-sha256": CONFIG_SHA256,
        "vllm-baseline.image": IMAGE_REF,
        "vllm-baseline.kind": "serve",
        "vllm-baseline.owner": "serving-kit",
        "vllm-baseline.role": role,
        "vllm-baseline.started-at": STARTED_LABEL,
        "vllm-baseline.weights": WEIGHTS_LABEL,
    }
    argv: list[str] = []
    for key in sorted(labels):
        argv.extend(("--label", f"{key}={labels[key]}"))
    return tuple(argv)


_DOCKER_ARGV: tuple[str, ...] = (
    "--restart",
    "no",
    "--gpus",
    "all",
    "--ipc",
    "host",
    "--shm-size",
    "16g",
    "--ulimit",
    "memlock=-1",
    # `--ulimit stack=67108864` は入れていない。64 MiB という値の唯一の出どころが NVIDIA の
    # NGC の手引きのコマンドで、requirements 11.2 が写さないと定めている形だからである
    # (親の判断、2026-09-22)。`memlock=-1` のほうは、NCCL の文書が理由まで述べている
    "--network",
    "host",
    "--mount",
    f"type=bind,source={REMOTE_ROOT}/models/glm-5-3-flash-nvfp4,target={MOUNT_AT},readonly",
    "--mount",
    f"type=bind,source={REMOTE_ROOT}/cache,target=/root/.cache",
    "--mount",
    f"type=bind,source={REMOTE_ROOT}/logs,target=/logs",
)

_TAIL_ARGV: tuple[str, ...] = (
    "--master-addr",
    FABRIC_HEAD_ADDR,
    "--master-port",
    "29501",
    "--max-model-len",
    "163840",
    "--max-num-seqs",
    "16",
    "--max-num-batched-tokens",
    "2048",
    "--gpu-memory-utilization",
    "0.90",
    "--language-model-only",
    "--no-enable-flashinfer-autotune",
    "--tool-call-parser",
    "glm47",
    "--reasoning-parser",
    "glm47",
    "--enable-auto-tool-choice",
    "--enable-prompt-tokens-details",
    "--shutdown-timeout",
    "60",
)

_HEAD_ARGV: tuple[str, ...] = (
    "docker",
    "run",
    "-d",
    "--pull",
    "never",
    "--name",
    "vb-p1-nvfp4-tp2-head",
    *_label_argv("head"),
    *_DOCKER_ARGV,
    "-e",
    f"VLLM_HOST_IP={FABRIC_HEAD_ADDR}",
    "-e",
    # 先頭の = は、NCCL の「名前の完全一致」の印である (構成の値のとおり)
    f"NCCL_SOCKET_IFNAME=={FABRIC_IFNAME}",
    "-e",
    f"GLOO_SOCKET_IFNAME={FABRIC_IFNAME}",
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
    *_TAIL_ARGV,
)

_WORKER_ARGV: tuple[str, ...] = (
    "docker",
    "run",
    "-d",
    "--pull",
    "never",
    "--name",
    "vb-p1-nvfp4-tp2-worker",
    *_label_argv("worker"),
    *_DOCKER_ARGV,
    "-e",
    f"VLLM_HOST_IP={FABRIC_WORKER_ADDR}",
    "-e",
    # 先頭の = は、NCCL の「名前の完全一致」の印である (構成の値のとおり)
    f"NCCL_SOCKET_IFNAME=={FABRIC_IFNAME}",
    "-e",
    f"GLOO_SOCKET_IFNAME={FABRIC_IFNAME}",
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
    *_TAIL_ARGV,
)


def test_committed_first_config_builds_the_fixed_argv() -> None:
    """コミットした第一の構成から、2 台ぶんの固定した引数の列ができる。"""
    nodes = _with_fabric(_load_nodes())
    config = c.select_config(_load_configs(), "p1-nvfp4-tp2", nodes)
    plans = p.build_plans(config, nodes, STARTED_AT)

    assert tuple(plan.node for plan in plans) == ("head", "worker")
    assert plans[0].argv == _HEAD_ARGV
    assert plans[1].argv == _WORKER_ARGV
    # 2 台のラベルの `config-sha256` は同じ (同じ構成の 1 つのサーバーである)
    assert (
        plans[0].labels[p.LABEL_CONFIG_SHA256]
        == plans[1].labels[p.LABEL_CONFIG_SHA256]
        == CONFIG_SHA256
    )


def test_the_two_nodes_differ_only_where_they_must() -> None:
    """2 台の差が、役割そのものを表すものだけである (ほかの設定は 2 台で同じ)。"""
    nodes = _with_fabric(_load_nodes())
    config = c.select_config(_load_configs(), "p1-nvfp4-tp2", nodes)
    head, worker = p.build_plans(config, nodes, STARTED_AT)

    only_head = set(head.argv) - set(worker.argv)
    only_worker = set(worker.argv) - set(head.argv)
    assert only_head == {
        "vb-p1-nvfp4-tp2-head",
        f"{p.LABEL_ROLE}=head",
        f"VLLM_HOST_IP={FABRIC_HEAD_ADDR}",
        "0",
    }
    assert only_worker == {
        "vb-p1-nvfp4-tp2-worker",
        f"{p.LABEL_ROLE}=worker",
        f"VLLM_HOST_IP={FABRIC_WORKER_ADDR}",
        "1",
        "--headless",
    }


# --- 選べる構成、選べない構成 ------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["probe-pinned", "p1-fetch-nvfp4", "p1-fetch-nvfp4-probe", "p1-image-licenses"],
)
def test_configs_selectable_before_the_fabric_is_measured(name: str) -> None:
    """縮小の確認、取得、読み取りの構成は、直結の値が空のままでも選べる。"""
    configs = _load_configs()
    nodes = _load_nodes()
    config = c.select_config(configs, name, nodes)
    assert config.name == name
    # 組み立てまで通る (直結の印を使っていないことの確かめ)
    plans = p.build_plans(config, nodes, STARTED_AT)
    assert len(plans) == len(config.nodes)


@pytest.mark.parametrize("name", ["netcheck-bandwidth", "netcheck-sanity"])
def test_job_configs_selectable_once_the_fabric_is_measured(name: str) -> None:
    """通信の確認のジョブの構成は、直結の値を埋めれば選べて、組み立てまで通る。"""
    nodes = _with_fabric(_load_nodes())
    config = c.select_config(_load_configs(), name, nodes)
    assert config.kind == "job"
    assert config.nodes == ("head", "worker")
    plans = p.build_plans(config, nodes, STARTED_AT)
    assert len(plans) == 2


_FABRIC_FIELDS = ("fabric_addr", "fabric_ifname", "fabric_measured")


@pytest.mark.parametrize("name", ["p1-nvfp4-tp2", "netcheck-bandwidth", "netcheck-sanity"])
def test_two_node_configs_are_refused_only_for_the_missing_fabric(name: str) -> None:
    """2 台の構成が選べない理由が、直結の値がないことだけである (ほかの誤りがない)。"""
    configs = _load_configs()
    nodes = _load_nodes()
    with pytest.raises(c.ConfigError) as caught:
        c.select_config(configs, name, nodes)
    lines = str(caught.value).splitlines()[1:]
    expected = [f"nodes.{role}.{field}" for role in ("head", "worker") for field in _FABRIC_FIELDS]
    assert [line.split(":", 1)[0] for line in lines] == expected


# --- 投機的デコードがない ----------------------------------------------


_SPECULATIVE = ("--speculative-config", "--spec-method", "--spec-model", "--spec-tokens")


def test_no_config_asks_for_speculative_decoding() -> None:
    """どの構成にも、投機的デコードの指定がない (requirements 6.7、design の Decision 5)。"""
    for name, config in _load_configs().items():
        for section in ("docker", "args", "env"):
            settings: Mapping[str, object] = getattr(config, section)
            for key, setting in settings.items():
                flag_text = getattr(setting, "flag", None) or ""
                value_text = getattr(setting, "value", None) or ""
                written = f"{flag_text} {value_text}"
                for flag in _SPECULATIVE:
                    assert flag not in written, f"{name}.{section}.{key} に {flag} がある"


def test_every_setting_carries_provenance() -> None:
    """設定の 1 つ 1 つが、出典と原文の組か、実測の記録を持つ (requirements 3.6)。

    `types.Provenance` が型で断っているので、読み込めた時点で満たされているが、**出典が
    http(s) の URL で、原文が空でないこと**を、ここでも固定する (根拠の欄を、あとから空の
    文字列で埋められないようにする)。
    """
    for name, config in _load_configs().items():
        for section in ("docker", "args", "env"):
            settings: Mapping[str, object] = getattr(config, section)
            for key, setting in settings.items():
                where = f"{name}.{section}.{key}"
                measured = getattr(setting, "measured", None)
                if measured is not None:
                    assert measured.startswith("docs/"), where
                    continue
                source = getattr(setting, "source", None)
                quote = getattr(setting, "quote", None)
                assert source is not None, where
                assert str(source).startswith("https://"), where
                assert quote is not None and quote.strip(), where


def test_the_probe_shrinks_the_model_without_touching_the_attention_shape() -> None:
    """縮小の確認が、`config.json` の層の並びから作った 4 層の上書きを渡す。

    アテンションの形 (`qk_rope_head_dim`、`kv_lora_rank`、`index_kpool` など) を上書きせず、
    線形アテンション、sparse MLA、MoE の層を、それぞれ 1 つ以上含むことを固定する。
    """
    config = _load_configs()["probe-pinned"]
    overrides = next(
        setting for setting in config.args.values() if setting.flag == "--hf-overrides"
    )
    assert overrides.value is not None
    text = overrides.value
    # 層の数と、層ごとの並びが、そろって 4 になっている
    assert '"num_hidden_layers":4' in text
    assert text.count('"linear_attention"') == 3
    assert text.count('"deepseek_sparse_attention"') == 1
    assert text.count('"dense"') == 3
    assert text.count('"sparse"') == 1
    # アテンションの形を変える鍵が、1 つも入っていない
    for key in (
        "qk_rope_head_dim",
        "qk_nope_head_dim",
        "kv_lora_rank",
        "index_kpool",
        "index_topk",
        "index_head_dim",
        "num_attention_heads",
        "hidden_size",
    ):
        assert key not in text, key
    assert re.search(r'"--load-format"|dummy', str(config.args)) is not None
