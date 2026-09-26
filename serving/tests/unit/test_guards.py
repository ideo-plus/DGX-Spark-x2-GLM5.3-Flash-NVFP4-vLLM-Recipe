"""起動の前の関門と、了承の仕組みの試験 (tasks.md 2.3)。

確かめること (design.md 「関門 › guards」、tasks.md 2.3 の完了の状態):

- 関門を、design の順に、読み取りだけで流す。状態を変える呼び出しが 1 つも出ない
  (requirements 2.1)
- GPU を使っているプロセスがあるときに、名前とメモリの量を示して断る (requirements 2.2)
- 重みの取得が動いている間の起動と、終了した自分のコンテナが残っているときの起動が
  断られる。すでに動いているものと同じ構成なら、関門の前に「すでに動いている」と分かる
- 置き場所がないときに、配布を促して断る。空きが足りないときに、要る量と空いている量を
  示す (requirements 2.5)
- ダイジェストが `RepoDigests` の 2 番目にあっても通る (requirements 3.2、3.3)
- 縮小の確認の構成では、重みの本体がなくても、設定とトークナイザが照合済みなら通り、
  要る空きが数十 MiB として計算される (requirements 3.5)
- 使うポートがふさがっているときに、ポートの番号を示して断る
- 偽の実行役に記録された、コンテナを対象にする操作のすべてが、ラベルで絞った一覧から来た
  識別子か、了承済みの計画が自分で起こす名前だけを対象にしている (requirements 2.3、2.4)
- 了承: `yes` 以外の入力、端末でない、`--yes` のそれぞれ (requirements 2.1)
- 巻き戻すコマンドに、ほかの名前を入れられない

読み取りの試験の台本は、2 台の Spark から採った実物の見本 (`tests/fixtures/spark/`) を
使う。実物の ssh、rsync、docker は、どの段でも呼ばない。
"""

from __future__ import annotations

import ast
import inspect
import io
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl

from fake_runner import FakeRunner, Reply, Rule
from serving_kit import guards as g
from serving_kit import types as kit_types
from serving_kit.plan import (
    LABEL_CONFIG_SHA256,
    LABEL_IMAGE,
    LABEL_OWNER,
    OWNER,
    OWNER_FILTER,
    build_plans,
)
from serving_kit.remote import CallGuard
from serving_kit.types import (
    ApprovedPlan,
    ConfigDef,
    ContainerPlan,
    GateResult,
    GpuApp,
    ImageRef,
    ManifestFile,
    NodeDef,
    NodeRole,
    PlannedPush,
    PlannedRun,
    Setting,
    VerificationRecord,
    WeightsManifest,
    WeightsRef,
)

# --- 見本の値 -----------------------------------------------------------

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "spark"
ROLES: tuple[NodeRole, ...] = ("head", "worker")

DIGEST = "b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
IMAGE_REF = f"vllm/vllm-openai@sha256:{DIGEST}"
OTHER_REF = "vllm/vllm-openai@sha256:" + "a1" * 32
REPO = "RedHatAI/GLM-5.3-Flash-NVFP4"
REVISION = "18d55bfd" + "0" * 32
SLUG = "RedHatAI__GLM-5.3-Flash-NVFP4"
MOUNT_AT = "/models/nvfp4"
REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"
MEASURED = "docs/results/2026-09-21-netcheck-links.md"

STARTED_AT = datetime(2026, 9, 21, 3, 0, 0, tzinfo=UTC)
VERIFIED_AT = datetime(2026, 9, 21, 2, 0, 0, tzinfo=UTC)

SOURCE = HttpUrl("https://docs.vllm.ai/en/latest/cli/serve/")
QUOTE = "vllm serve [model_tag] [options]"


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

WEIGHTS_DIR = f"{REMOTE_ROOT}/models/{SLUG}"
PROBE_DIR = f"{REMOTE_ROOT}/probe/{SLUG}"
CACHE_DIR = f"{REMOTE_ROOT}/cache"

# 照合の結果の記録は、範囲ごとに別のファイルに置く (`serve verify` と
# `serve fetch --probe-files` が、互いの記録を上書きしないため)
ALL_RECORD = f"{REMOTE_ROOT}/state/{SLUG}.verified.json"
PROBE_RECORD = f"{REMOTE_ROOT}/state/{SLUG}.probe.verified.json"

# マニフェスト: 大きな safetensors 2 つと、縮小の確認が使う設定とトークナイザ
MANIFEST_FILES = (
    ManifestFile(path="config.json", size=4_096, sha256="11" * 32),
    ManifestFile(path="model-00001-of-00002.safetensors", size=98_000_000_000, sha256="22" * 32),
    ManifestFile(path="model-00002-of-00002.safetensors", size=98_000_000_000, sha256="33" * 32),
    ManifestFile(path="tokenizer.json", size=36_000_000, sha256="44" * 32),
)
MANIFEST = WeightsManifest(
    repo=REPO,
    revision=REVISION,
    generated_at=datetime(2026, 9, 21, 1, 0, 0, tzinfo=UTC),
    total_bytes=sum(entry.size for entry in MANIFEST_FILES),
    files=MANIFEST_FILES,
)
PROBE_BYTES = sum(entry.size for entry in MANIFEST.probe_files)


def serve_config(*, port: str = "8000") -> ConfigDef:
    """2 台の推論サーバーの構成 (design.md Data Models の `p1-nvfp4-tp2` に沿う)。"""
    return ConfigDef(
        name="p1-nvfp4-tp2",
        kind="serve",
        description="P1 の第一の構成",
        nodes=("head", "worker"),
        image=IMAGE,
        weights=WEIGHTS,
        docker={
            "gpus": _setting("--gpus", "all"),
            "network": _setting("--network", "host"),
            "mount-weights": _setting(
                "--mount",
                f"type=bind,source={{remote_root}}/models/{SLUG},target={MOUNT_AT},readonly",
            ),
            "mount-cache": _setting(
                "--mount", "type=bind,source={remote_root}/cache,target=/root/.cache"
            ),
        },
        args={
            "model-path": _setting(value="{weights.mount_at}"),
            "served-model-name": _setting("--served-model-name", "glm-5-3-flash"),
            "host": _setting("--host", "{head.lan_addr}"),
            "port": _setting("--port", port, is_port=True),
            "master-port": _setting("--master-port", "29501", is_port=True),
            "node-rank": _setting("--node-rank", "{node.rank}"),
            "headless": _setting("--headless", only_on="worker"),
        },
        env={"vllm-host-ip": _setting("VLLM_HOST_IP", "{node.fabric_addr}")},
        ready_timeout_s=1800,
        served_model_name="glm-5-3-flash",
    )


def probe_config() -> ConfigDef:
    """1 台の縮小の確認の構成 (重みの本体は取らず、設定とトークナイザだけを見せる)。"""
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
                "--mount",
                f"type=bind,source={{remote_root}}/probe/{SLUG},target=/probe,readonly",
            ),
        },
        args={
            "model-path": _setting(value="/probe"),
            "served-model-name": _setting("--served-model-name", "glm-5-3-flash"),
            "port": _setting("--port", "8000", is_port=True),
            "load-format": _setting("--load-format", "dummy"),
        },
        env={},
        ready_timeout_s=600,
        served_model_name="glm-5-3-flash",
    )


def fetch_config() -> ConfigDef:
    """重みの取得の構成 (動いている取得を見つける試験に使う)。"""
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
        args={"download": _setting(value="download"), "repo": _setting(value=REPO)},
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


def plans_of(config: ConfigDef) -> tuple[ContainerPlan, ...]:
    """構成から、コンテナの計画を組み立てる (`plan` の実物を通す)。"""
    return build_plans(config, NODES, STARTED_AT)


# --- 実物の見本と、台本 --------------------------------------------------


def sample(role: NodeRole, name: str) -> str:
    """2 台の Spark から採った、読み取りの実物の出力 (tasks.md 1.6)。"""
    return (FIXTURES / role / name).read_text()


def ps_line(
    plan: ContainerPlan,
    *,
    state: str = "running",
    container_id: str = "0123456789ab",
    labels: Mapping[str, str] | None = None,
    labels_text: str | None = None,
) -> str:
    """`docker ps --format json` の 1 行 (自分のコンテナがあるときの見本)。

    **これは実機から採った見本ではない**。自分のコンテナがあるときの出力は、まだ採れて
    いない (tasks.md 1.6 の「ここで採れないもの」。7.1 で採って足す)。項目の名前は、
    Docker の公式の文書の `docker ps --format json` の項目に沿って書いた。`Labels` が
    `鍵=値,鍵=値` の 1 つの文字列であることに注意する (`labels_text` で、その文字列を
    そのまま書ける)。
    """
    used = dict(plan.labels if labels is None else labels)
    joined = (
        ",".join(f"{key}={value}" for key, value in sorted(used.items()))
        if labels_text is None
        else labels_text
    )
    return json.dumps(
        {
            "Command": '"/opt/vllm/entrypoint.sh"',
            "CreatedAt": "2026-09-21 03:00:00 +0000 UTC",
            "ID": container_id,
            "Image": plan.labels[LABEL_IMAGE],
            "Labels": joined,
            "LocalVolumes": "0",
            "Mounts": WEIGHTS_DIR,
            "Names": plan.container_name,
            "Networks": "host",
            "Ports": "",
            "RunningFor": "2 minutes ago",
            "Size": "0B",
            "State": state,
            "Status": "Up 2 minutes",
        }
    )


def record_json(
    role: NodeRole,
    *,
    scope: str = "all",
    revision: str = REVISION,
    file_count: int | None = None,
    total_bytes: int | None = None,
    mismatched: Sequence[str] = (),
) -> str:
    """照合の結果の記録の中身 (範囲ごとに、別のファイルに置く)。"""
    files = MANIFEST.probe_files if scope == "probe_files" else MANIFEST.files
    record = VerificationRecord(
        repo=REPO,
        revision=revision,
        scope="probe_files" if scope == "probe_files" else "all",
        node=role,
        verified_at=VERIFIED_AT,
        file_count=len(files) if file_count is None else file_count,
        total_bytes=sum(entry.size for entry in files) if total_bytes is None else total_bytes,
        mismatched=tuple(mismatched),
    )
    return record.model_dump_json()


Records = dict[NodeRole, dict[str, str]]
"""台ごとの、照合の記録 (道筋 → 中身)。"""


def _fixture_map(name: str) -> dict[NodeRole, str]:
    return {role: sample(role, name) for role in ROLES}


def _all_records() -> dict[NodeRole, dict[str, str]]:
    """既定: 2 台とも、全体の照合の記録だけがある (縮小の確認用の記録はない)。"""
    return {role: {ALL_RECORD: record_json(role)} for role in ROLES}


def _matches_path(wanted: str) -> Callable[[tuple[str, ...]], bool]:
    """`cat <道筋>` の、道筋まで見る台本の述語 (範囲ごとに別の記録を返すため)。"""
    return lambda argv: argv[1:] == (wanted,)


@dataclass
class Script:
    """偽の実行役の台本 (既定は、2 台とも空いている状態の実物の見本)。"""

    ps: dict[NodeRole, str] = field(default_factory=lambda: _fixture_map("docker-ps-own.txt"))
    gpu: dict[NodeRole, str] = field(
        default_factory=lambda: _fixture_map("nvidia-smi-compute-apps.txt")
    )
    df: dict[NodeRole, str] = field(default_factory=lambda: _fixture_map("df-avail.txt"))
    ss: dict[NodeRole, str] = field(default_factory=lambda: _fixture_map("ss-ltnH.txt"))
    digests: tuple[str, ...] = (OTHER_REF, IMAGE_REF)
    records: dict[NodeRole, dict[str, str]] = field(default_factory=_all_records)
    """台ごとの、照合の記録 (道筋 → 中身)。書いていない道筋は、ファイルがない。"""
    missing_dirs: tuple[str, ...] = ()
    unreachable: tuple[NodeRole, ...] = ()

    def rules(self) -> tuple[Rule, ...]:
        rules: list[Rule] = []
        for role in ROLES:
            uname = (
                Reply(exit_code=255, stderr="ssh: connect to host port 22: No route to host")
                if role in self.unreachable
                else Reply(stdout=sample(role, "uname-n.txt"))
            )
            rules.extend(
                (
                    Rule(prefix=("uname",), node=role, replies=(uname,)),
                    Rule(
                        prefix=("docker", "ps"),
                        node=role,
                        replies=(Reply(stdout=self.ps[role]),),
                    ),
                    Rule(
                        prefix=("nvidia-smi",),
                        node=role,
                        replies=(Reply(stdout=self.gpu[role]),),
                    ),
                    Rule(prefix=("df",), node=role, replies=(Reply(stdout=self.df[role]),)),
                    Rule(prefix=("ss",), node=role, replies=(Reply(stdout=self.ss[role]),)),
                )
            )
            # 道筋まで見て返す (範囲ごとに別のファイル)。書いていない道筋は、ない
            rules.extend(
                Rule(
                    prefix=("cat",),
                    node=role,
                    when=_matches_path(path),
                    replies=(Reply(stdout=text),),
                )
                for path, text in self.records[role].items()
            )
            rules.append(
                Rule(
                    prefix=("cat",),
                    node=role,
                    replies=(Reply(exit_code=1, stderr="cat: No such file or directory"),),
                )
            )
        rules.append(
            Rule(
                prefix=("docker", "image", "inspect"),
                replies=(Reply(stdout=json.dumps(list(self.digests)) + "\n"),),
            )
        )
        rules.append(
            Rule(
                prefix=("test", "-e"),
                when=lambda argv: len(argv) > 2 and argv[2] in self.missing_dirs,
                replies=(Reply(exit_code=1),),
            )
        )
        rules.append(Rule(prefix=("test", "-e"), replies=(Reply(),)))
        return tuple(rules)


def runner_of(tmp_path: Path, script: Script | None = None) -> FakeRunner:
    """台本どおりに答える偽の実行役 (台本にない呼び出しは、誤りになる)。"""
    return FakeRunner(var_root=tmp_path, script=(script or Script()).rules())


@pytest.fixture(autouse=True)
def no_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """関門が、うっかり実物の ssh、rsync、docker を呼ばないことを固定する。"""

    def explode(argv: Sequence[str], **kwargs: Any) -> None:
        raise AssertionError(f"関門が実物のコマンドを呼んだ: {list(argv)}")

    monkeypatch.setattr(subprocess, "run", explode)


# --- 結果の取り出し ------------------------------------------------------


def refused(results: Sequence[GateResult]) -> tuple[GateResult, ...]:
    return tuple(result for result in results if not result.passed)


def detail_of(results: Sequence[GateResult], gate: str, node: NodeRole = "head") -> str:
    for result in results:
        if result.gate == gate and result.node == node:
            return result.detail
    raise AssertionError(f"{node} の関門 {gate} の結果がない: {[r.gate for r in results]}")


def passed_of(results: Sequence[GateResult], gate: str, node: NodeRole = "head") -> bool:
    for result in results:
        if result.gate == gate and result.node == node:
            return result.passed
    raise AssertionError(f"{node} の関門 {gate} の結果がない")


def gates_for(
    tmp_path: Path, config: ConfigDef, script: Script | None = None
) -> tuple[FakeRunner, tuple[GateResult, ...]]:
    """関門を全部流して、偽の実行役と結果を返す。"""
    runner = runner_of(tmp_path, script)
    results = g.run_gates(runner, config, NODES, plans_of(config), manifest=MANIFEST)
    return runner, results


# --- 関門の全体 ---------------------------------------------------------


def test_every_gate_passes_with_the_real_samples(tmp_path: Path) -> None:
    _, results = gates_for(tmp_path, serve_config())
    assert refused(results) == (), [(r.node, r.gate, r.detail) for r in refused(results)]


def test_the_gates_run_in_the_designed_order_on_every_node(tmp_path: Path) -> None:
    _, results = gates_for(tmp_path, serve_config())
    order = (
        "reachable",
        "own_state",
        "gpu_idle",
        "layout",
        "image_digest",
        "weights_verified",
        "disk_space",
        "ports_free",
    )
    assert order == g.GATE_ORDER
    assert tuple((r.node, r.gate) for r in results) == tuple(
        (role, gate) for role in ("head", "worker") for gate in order
    )


def test_no_gate_changes_the_state(tmp_path: Path) -> None:
    runner, _ = gates_for(tmp_path, serve_config())
    assert runner.pushes == ()
    assert runner.pulls == ()
    assert [call for call in runner.runs if call.mutating] == []
    for argv in runner.argvs:
        assert argv[0] != "mkdir"
        assert argv[:2] not in (
            ("docker", "run"),
            ("docker", "stop"),
            ("docker", "rm"),
            ("docker", "pull"),
        )


def test_the_container_list_is_always_filtered_by_our_label(tmp_path: Path) -> None:
    runner, _ = gates_for(tmp_path, serve_config())
    listings = [argv for argv in runner.argvs if argv[:2] == ("docker", "ps")]
    assert len(listings) == 2
    for argv in listings:
        assert f"label={OWNER_FILTER}" in argv


def test_an_unreachable_node_skips_the_rest_of_its_gates(tmp_path: Path) -> None:
    _, results = gates_for(tmp_path, serve_config(), Script(unreachable=("worker",)))
    worker = [result for result in results if result.node == "worker"]
    assert [result.gate for result in worker] == ["reachable"]
    assert not worker[0].passed
    assert [result.passed for result in results if result.node == "head"] == [True] * 8


# --- 自分のコンテナと、GPU ----------------------------------------------


def test_gpu_processes_are_refused_with_name_and_memory(tmp_path: Path) -> None:
    gpu = dict(_fixture_map("nvidia-smi-compute-apps.txt"))
    gpu["worker"] = "1234, /usr/bin/python3, 4096 MiB\n"
    _, results = gates_for(tmp_path, serve_config(), Script(gpu=gpu))
    detail = detail_of(results, "gpu_idle", "worker")
    assert "/usr/bin/python3" in detail
    assert "4096" in detail
    assert not passed_of(results, "gpu_idle", "worker")
    assert passed_of(results, "gpu_idle", "head")


def test_the_memory_column_may_be_not_available(tmp_path: Path) -> None:
    # GB10 のユニファイドメモリで、量が読めないことがある (落ちずに、名前だけを示す)
    gpu = dict(_fixture_map("nvidia-smi-compute-apps.txt"))
    gpu["head"] = "4321, /usr/bin/python3, [N/A]\n"
    _, results = gates_for(tmp_path, serve_config(), Script(gpu=gpu))
    detail = detail_of(results, "gpu_idle", "head")
    assert "/usr/bin/python3" in detail
    assert not passed_of(results, "gpu_idle", "head")
    assert g.parse_gpu_apps(gpu["head"])[0].used_memory_mib is None


def test_a_running_fetch_container_is_refused_and_tells_to_wait(tmp_path: Path) -> None:
    fetching = plans_of(fetch_config())[0]
    ps = dict(_fixture_map("docker-ps-own.txt"))
    ps["head"] = ps_line(fetching) + "\n"
    _, results = gates_for(tmp_path, serve_config(), Script(ps=ps))
    detail = detail_of(results, "own_state", "head")
    assert not passed_of(results, "own_state", "head")
    assert fetching.container_name in detail
    assert "取得" in detail and "待" in detail


def test_a_running_own_container_is_refused_and_tells_to_stop(tmp_path: Path) -> None:
    other = plans_of(probe_config())[0]
    ps = dict(_fixture_map("docker-ps-own.txt"))
    ps["head"] = ps_line(other) + "\n"
    _, results = gates_for(tmp_path, serve_config(), Script(ps=ps))
    detail = detail_of(results, "own_state", "head")
    assert not passed_of(results, "own_state", "head")
    assert "serve stop" in detail


def test_an_exited_own_container_is_refused_and_tells_to_collect_logs(tmp_path: Path) -> None:
    dead = plans_of(serve_config())[0]
    ps = dict(_fixture_map("docker-ps-own.txt"))
    ps["head"] = ps_line(dead, state="exited") + "\n"
    _, results = gates_for(tmp_path, serve_config(), Script(ps=ps))
    detail = detail_of(results, "own_state", "head")
    assert not passed_of(results, "own_state", "head")
    assert "serve logs" in detail and "serve stop" in detail


def test_the_container_list_is_read_as_one_json_per_line() -> None:
    plans = plans_of(serve_config())
    text = "\n".join(
        ps_line(plan, container_id=f"{index}" * 12) for index, plan in enumerate(plans)
    )
    containers = g.parse_own_containers(text + "\n")
    assert [c.name for c in containers] == [plan.container_name for plan in plans]
    assert containers[0].labels[LABEL_CONFIG_SHA256] == plans[0].labels[LABEL_CONFIG_SHA256]
    assert containers[0].state == "running"


# --- 置き場所、イメージ、重み、空き、ポート ------------------------------


def test_a_missing_layout_dir_is_refused_and_asks_for_push(tmp_path: Path) -> None:
    _, results = gates_for(tmp_path, serve_config(), Script(missing_dirs=(WEIGHTS_DIR,)))
    detail = detail_of(results, "layout", "worker")
    assert not passed_of(results, "layout", "worker")
    assert WEIGHTS_DIR in detail
    assert "serve push" in detail


def test_the_layout_gate_looks_at_the_mounts_of_the_built_plan(tmp_path: Path) -> None:
    runner, _ = gates_for(tmp_path, serve_config())
    checked = [argv[2] for argv in runner.argvs if argv[:2] == ("test", "-e")]
    assert sorted(set(checked)) == sorted({WEIGHTS_DIR, CACHE_DIR})


# --- ファイルの bind mount (vLLM の直したファイルの重ね。ADR 0007 の第 2a 段、#73) ---

OVERLAY_FILE = "vllm/models/glm5next/common/model.py"
OVERLAY_SOURCE = f"{REMOTE_ROOT}/payload/vllm-overlay/k2s2a/{OVERLAY_FILE}"
"""Spark の側の、直したファイルの写し (`serve push` が `payload/` の下へ配る)。"""

OVERLAY_MOUNT = (
    "type=bind,source={remote_root}/payload/vllm-overlay/k2s2a/"
    f"{OVERLAY_FILE},target=/usr/local/lib/python3.12/dist-packages/{OVERLAY_FILE},readonly"
)


def overlay_config() -> ConfigDef:
    """置き場所 (ディレクトリ) の bind mount に、ファイルの bind mount を 1 つ足した構成。"""
    base = serve_config()
    return base.model_copy(
        update={
            "docker": {
                **base.docker,
                "mount-vllm-overlay-models-glm5next-common-model-py": _setting(
                    "--mount", OVERLAY_MOUNT
                ),
            }
        }
    )


def test_a_file_bind_mount_is_looked_at_with_test_e_and_passes(tmp_path: Path) -> None:
    """ファイルの bind mount は、`test -e` で見て、あれば通る (`test -d` では通らない)。"""
    # Given: ファイルの bind mount を持つ計画と、`test -e` に終了 0 を返す台本
    plan = plans_of(overlay_config())[0]
    runner = runner_of(tmp_path)

    # When: 置き場所の関門を流す
    result = g.gate_layout(runner, HEAD, plan)

    # Then: そのファイルの道筋を `test -e` で見て、通る
    assert ("test", "-e", OVERLAY_SOURCE) in runner.argvs
    assert result.passed is True, result.detail


def test_a_file_bind_mount_whose_source_is_missing_is_refused_and_asks_for_push(
    tmp_path: Path,
) -> None:
    """元のファイルがなければ落ち、その道筋と `serve push` を示す。"""
    # Given: ファイルの bind mount を持つ計画と、そのファイルにだけ終了 1 を返す台本
    plan = plans_of(overlay_config())[0]
    runner = runner_of(tmp_path, Script(missing_dirs=(OVERLAY_SOURCE,)))

    # When: 置き場所の関門を流す
    result = g.gate_layout(runner, HEAD, plan)

    # Then: 落ちて、道筋と配り方を示す
    assert result.passed is False
    assert OVERLAY_SOURCE in result.detail
    assert "serve push" in result.detail


def test_the_layout_gate_asks_only_test_e_for_every_mount_source(tmp_path: Path) -> None:
    """ディレクトリもファイルも、`test -e` だけで見る (`test -d` は流さない)。"""
    # Given: ディレクトリ 2 つとファイル 1 つの bind mount を持つ計画
    plan = plans_of(overlay_config())[0]
    runner = runner_of(tmp_path)

    # When: 置き場所の関門を流す
    g.gate_layout(runner, HEAD, plan)

    # Then: 3 つの元の道筋が `test -e` で見られ、`test -d` は 1 度も流れない
    checked = [argv[2] for argv in runner.argvs if argv[:2] == ("test", "-e")]
    assert sorted(set(checked)) == sorted({WEIGHTS_DIR, CACHE_DIR, OVERLAY_SOURCE})
    assert [argv for argv in runner.argvs if argv[:2] == ("test", "-d")] == []


def test_a_digest_in_the_second_place_of_repo_digests_passes(tmp_path: Path) -> None:
    _, results = gates_for(tmp_path, serve_config(), Script(digests=(OTHER_REF, IMAGE_REF)))
    assert passed_of(results, "image_digest", "head")


def test_a_digest_that_is_not_in_the_list_is_refused(tmp_path: Path) -> None:
    _, results = gates_for(tmp_path, serve_config(), Script(digests=(OTHER_REF,)))
    detail = detail_of(results, "image_digest", "head")
    assert not passed_of(results, "image_digest", "head")
    assert IMAGE_REF in detail and OTHER_REF in detail


@pytest.mark.parametrize(
    ("reply", "passed"),
    [
        (Reply(stdout=json.dumps(f"sha256:{DIGEST}")), True),
        (Reply(stdout=json.dumps("sha256:" + "a1" * 32)), False),
        (Reply(stdout="null"), False),
        (Reply(stdout=json.dumps([f"sha256:{DIGEST}"])), False),
        (Reply(stdout="invalid json"), False),
        (Reply(exit_code=1, stderr="No such image"), False),
        (Reply(exit_code=255, stderr="Connection failed"), False),
    ],
)
def test_local_image_identity_is_checked_before_launch(
    tmp_path: Path, reply: Reply, passed: bool
) -> None:
    config = serve_config()
    image = config.image.model_copy(update={"ref": f"sha256:{DIGEST}"})
    config = config.model_copy(update={"image": image})
    argv = ("docker", "image", "inspect", "--format", "{{json .Id}}", image.ref)
    runner = FakeRunner(var_root=tmp_path, script=[Rule(prefix=argv, replies=(reply,))])
    result = g.gate_image_digest(runner, HEAD, config)
    assert result.passed is passed
    assert runner.argvs == (argv,)
    assert image.ref in plans_of(config)[0].argv


def test_a_missing_verification_record_is_refused(tmp_path: Path) -> None:
    records: Records = {"head": {ALL_RECORD: record_json("head")}, "worker": {}}
    _, results = gates_for(tmp_path, serve_config(), Script(records=records))
    detail = detail_of(results, "weights_verified", "worker")
    assert not passed_of(results, "weights_verified", "worker")
    assert ALL_RECORD in detail


def test_a_verification_record_of_another_revision_is_refused(tmp_path: Path) -> None:
    other = "0" * 40
    records: Records = {
        "head": {ALL_RECORD: record_json("head", revision=other)},
        "worker": {ALL_RECORD: record_json("worker")},
    }
    _, results = gates_for(tmp_path, serve_config(), Script(records=records))
    detail = detail_of(results, "weights_verified", "head")
    assert not passed_of(results, "weights_verified", "head")
    assert other in detail and REVISION in detail


def test_a_verification_record_with_a_mismatched_file_is_refused(tmp_path: Path) -> None:
    broken = record_json("head", mismatched=("model-00002-of-00002.safetensors",))
    records: Records = {
        "head": {ALL_RECORD: broken},
        "worker": {ALL_RECORD: record_json("worker")},
    }
    _, results = gates_for(tmp_path, serve_config(), Script(records=records))
    detail = detail_of(results, "weights_verified", "head")
    assert not passed_of(results, "weights_verified", "head")
    assert "model-00002-of-00002.safetensors" in detail


def test_a_verification_record_with_a_different_file_count_is_refused(tmp_path: Path) -> None:
    records: Records = {
        "head": {ALL_RECORD: record_json("head", file_count=3)},
        "worker": {ALL_RECORD: record_json("worker")},
    }
    _, results = gates_for(tmp_path, serve_config(), Script(records=records))
    detail = detail_of(results, "weights_verified", "head")
    assert not passed_of(results, "weights_verified", "head")
    assert "3" in detail and str(len(MANIFEST.files)) in detail


def test_a_config_without_weights_passes_the_weights_gate(tmp_path: Path) -> None:
    config = inspect_config()
    empty: Records = {"head": {}, "worker": {}}
    runner = runner_of(tmp_path, Script(records=empty))
    results = g.run_gates(runner, config, NODES, plans_of(config), manifest=None)
    assert passed_of(results, "weights_verified", "head")
    assert [argv for argv in runner.argvs if argv[0] == "cat"] == []


# --- 照合の記録は、範囲ごとに別のファイル --------------------------------


def test_the_two_scopes_are_written_to_different_paths() -> None:
    # `serve verify` と `serve fetch --probe-files` が、互いの記録を上書きしないため
    assert g.verification_record_path(HEAD, REPO, "all") == ALL_RECORD
    assert g.verification_record_path(HEAD, REPO, "probe_files") == PROBE_RECORD
    assert ALL_RECORD != PROBE_RECORD


def test_the_record_paths_pass_the_rules_of_the_remote_runner(tmp_path: Path) -> None:
    # `cat` で読めるのは `<remote_root>/` の下だけ。使える文字も絞られている
    guard = CallGuard(var_root=tmp_path)
    for path in (ALL_RECORD, PROBE_RECORD):
        assert path.startswith(f"{REMOTE_ROOT}/state/")
        guard.check_run(HEAD, ("cat", path), mutating=False)


def test_a_probe_config_is_refused_when_only_the_full_record_exists(tmp_path: Path) -> None:
    # 重みの本体を見た記録は、`probe/<slug>/` に設定とトークナイザがある証拠にならない
    config = probe_config()
    records: Records = {"head": {ALL_RECORD: record_json("head")}, "worker": {}}
    _, results = gates_for(tmp_path, config, Script(records=records))
    detail = detail_of(results, "weights_verified", "head")
    assert not passed_of(results, "weights_verified", "head")
    assert PROBE_RECORD in detail


def test_a_serve_config_is_refused_when_only_the_probe_record_exists(tmp_path: Path) -> None:
    probe_record = record_json("head", scope="probe_files")
    records: Records = {
        "head": {PROBE_RECORD: probe_record},
        "worker": {PROBE_RECORD: record_json("worker", scope="probe_files")},
    }
    _, results = gates_for(tmp_path, serve_config(), Script(records=records))
    detail = detail_of(results, "weights_verified", "head")
    assert not passed_of(results, "weights_verified", "head")
    assert ALL_RECORD in detail


def test_both_records_can_live_side_by_side(tmp_path: Path) -> None:
    records: Records = {
        role: {
            ALL_RECORD: record_json(role),
            PROBE_RECORD: record_json(role, scope="probe_files"),
        }
        for role in ROLES
    }
    _, served = gates_for(tmp_path, serve_config(), Script(records=records))
    assert refused(served) == (), [(r.node, r.gate, r.detail) for r in refused(served)]
    _, probed = gates_for(tmp_path, probe_config(), Script(records=records))
    assert refused(probed) == (), [(r.node, r.gate, r.detail) for r in refused(probed)]


def test_a_record_whose_scope_disagrees_with_its_path_is_refused(tmp_path: Path) -> None:
    # 縮小の確認用の道筋に、全体の照合の記録が置かれていたら断る (断る側に寄せる)
    config = probe_config()
    records: Records = {"head": {PROBE_RECORD: record_json("head")}, "worker": {}}
    _, results = gates_for(tmp_path, config, Script(records=records))
    detail = detail_of(results, "weights_verified", "head")
    assert not passed_of(results, "weights_verified", "head")
    assert "probe_files" in detail and "all" in detail


def test_not_enough_disk_is_refused_with_the_required_and_the_available(tmp_path: Path) -> None:
    df = dict(_fixture_map("df-avail.txt"))
    df["worker"] = "        Avail\n1000000000\n"
    _, results = gates_for(tmp_path, serve_config(), Script(df=df))
    detail = detail_of(results, "disk_space", "worker")
    assert not passed_of(results, "disk_space", "worker")
    needed = MANIFEST.total_bytes + MANIFEST.total_bytes // 10
    assert f"{needed:,}" in detail
    assert f"{1000000000:,}" in detail


def test_the_required_space_is_the_manifest_plus_ten_percent(tmp_path: Path) -> None:
    assert g.required_free_bytes(serve_config(), MANIFEST) == MANIFEST.total_bytes
    assert g.required_free_bytes(inspect_config(), None) == IMAGE.size_bytes


def test_a_busy_port_is_refused_with_the_port_number(tmp_path: Path) -> None:
    # 見本の `ss -ltnH` は、8080 を待ち受けている (この道具のものではない)
    _, results = gates_for(tmp_path, serve_config(port="8080"))
    detail = detail_of(results, "ports_free", "head")
    assert not passed_of(results, "ports_free", "head")
    assert "8080" in detail


def test_the_ports_come_from_the_settings_of_the_config() -> None:
    assert g.config_ports(serve_config(), "head") == (8000, 29501)
    assert g.config_ports(inspect_config(), "head") == ()
    assert 8080 in g.parse_listening_ports(sample("head", "ss-ltnH.txt"))
    assert 53 in g.parse_listening_ports(sample("head", "ss-ltnH.txt"))  # 127.0.0.53%lo:53
    assert 33122 in g.parse_listening_ports(sample("head", "ss-ltnH.txt"))  # [fd00:db8::1]:33122
    assert 8000 not in g.parse_listening_ports(sample("head", "ss-ltnH.txt"))


# --- 縮小の確認の構成 ----------------------------------------------------


def _probe_records() -> dict[NodeRole, dict[str, str]]:
    """縮小の確認用の記録だけがある台 (重みの本体は、まだ取っていない)。"""
    return {"head": {PROBE_RECORD: record_json("head", scope="probe_files")}, "worker": {}}


def test_a_probe_config_passes_with_only_the_config_and_tokenizer_verified(tmp_path: Path) -> None:
    config = probe_config()
    runner = runner_of(tmp_path, Script(records=_probe_records()))
    results = g.run_gates(runner, config, NODES, plans_of(config), manifest=MANIFEST)
    assert refused(results) == (), [(r.gate, r.detail) for r in refused(results)]
    assert [result.node for result in results] == ["head"] * 8


def test_the_probe_config_needs_only_tens_of_mib(tmp_path: Path) -> None:
    config = probe_config()
    required = g.required_free_bytes(config, MANIFEST)
    assert required == PROBE_BYTES
    assert 10 * 1024**2 < required < 100 * 1024**2
    _, results = gates_for(tmp_path, config, Script(records=_probe_records()))
    assert "MiB" in detail_of(results, "disk_space", "head")


def test_the_probe_layout_gate_looks_at_the_probe_directory(tmp_path: Path) -> None:
    config = probe_config()
    runner, _ = gates_for(tmp_path, config, Script(records=_probe_records()))
    assert [argv[2] for argv in runner.argvs if argv[:2] == ("test", "-e")] == [PROBE_DIR]


# --- 派生の重み (手元で変換した重み) -------------------------------------

DERIVED_NAME = "k2s1"
DERIVED_DIR = f"{REMOTE_ROOT}/models/{DERIVED_NAME}"
DERIVED_ALL_RECORD = f"{REMOTE_ROOT}/state/{DERIVED_NAME}.derived.verified.json"
DERIVED_PROBE_RECORD = f"{REMOTE_ROOT}/state/{DERIVED_NAME}.derived.probe.verified.json"


def derivation(*, origin_revision: str = REVISION, commit: str = "a" * 40) -> kit_types.Derivation:
    """派生の同一性 (元の重み、変換の条件)。構成、マニフェスト、記録が同じものを持つ。"""
    return kit_types.Derivation(
        name=DERIVED_NAME,
        origin=kit_types.WeightsOrigin(repo=REPO, revision=origin_revision),
        conversion=kit_types.ConversionSpec(
            tool="experiments/k2-quant/convert.py",
            commit=commit,
            args=("--dtype", "fp8"),
            target_pattern=r"^model\.layers\.\d+\.self_attn\..*$",
        ),
    )


def derived_weights(
    *, origin_revision: str = REVISION, commit: str = "a" * 40
) -> kit_types.DerivedWeightsRef:
    return kit_types.DerivedWeightsRef(
        kind="derived",
        name=DERIVED_NAME,
        origin=kit_types.OriginWeightsRef(
            repo=REPO, revision=origin_revision, manifest=f"{SLUG}.manifest.json"
        ),
        conversion=derivation(commit=commit).conversion,
        manifest=f"{DERIVED_NAME}.manifest.json",
        mount_at=f"/models/{DERIVED_NAME}",
    )


def derived_serve_config(*, origin_revision: str = REVISION) -> ConfigDef:
    """派生の重みを、読み取り専用で結び付ける推論サーバーの構成 (`serve_config` に沿う)。"""
    docker = dict(serve_config().docker)
    docker["mount-weights"] = _setting(
        "--mount",
        f"type=bind,source={{remote_root}}/models/{DERIVED_NAME},"
        f"target=/models/{DERIVED_NAME},readonly",
    )
    return serve_config().model_copy(
        update={
            "name": f"p2-nope-tp2-full-{DERIVED_NAME}",
            "weights": derived_weights(origin_revision=origin_revision),
            "docker": docker,
        }
    )


def derived_manifest(
    *, derived: kit_types.Derivation | None = None
) -> kit_types.DerivedWeightsManifest:
    return kit_types.DerivedWeightsManifest(
        kind="derived",
        derivation=derivation() if derived is None else derived,
        generated_at=datetime(2026, 9, 21, 1, 0, 0, tzinfo=UTC),
        total_bytes=MANIFEST.total_bytes,
        files=MANIFEST_FILES,
    )


def derived_record_json(
    role: NodeRole,
    *,
    derived: kit_types.Derivation | None = None,
    manifest: kit_types.DerivedWeightsManifest | None = None,
    file_count: int | None = None,
    mismatched: Sequence[str] = (),
) -> str:
    """派生の照合の結果の記録の中身 (`record_json` の派生版)。

    記録が結び付くマニフェストは、既定では `derived_manifest()` (関門に渡す既定のマニフェスト)。
    """
    bound = derived_manifest() if manifest is None else manifest
    record = kit_types.DerivedVerificationRecord(
        kind="derived",
        derivation=derivation() if derived is None else derived,
        manifest_sha256=bound.content_sha256,
        scope="all",
        node=role,
        verified_at=VERIFIED_AT,
        file_count=len(MANIFEST.files) if file_count is None else file_count,
        total_bytes=MANIFEST.total_bytes,
        mismatched=tuple(mismatched),
    )
    return record.model_dump_json()


def _derived_records(
    *, derived: kit_types.Derivation | None = None
) -> dict[NodeRole, dict[str, str]]:
    """2 台とも、派生の全体の照合の記録がある (Hub の記録も、別の道筋にある)。"""
    return {
        role: {
            DERIVED_ALL_RECORD: derived_record_json(role, derived=derived),
            ALL_RECORD: record_json(role),
        }
        for role in ROLES
    }


def derived_gates_for(
    tmp_path: Path,
    config: ConfigDef,
    *,
    manifest: kit_types.DerivedWeightsManifest | WeightsManifest | None,
    script: Script | None = None,
) -> tuple[FakeRunner, tuple[GateResult, ...]]:
    """派生の構成で、関門を全部流して、偽の実行役と結果を返す。"""
    runner = runner_of(tmp_path, script or Script(records=_derived_records()))
    results = g.run_gates(runner, config, NODES, plans_of(config), manifest=manifest)
    return runner, results


def cat_paths(runner: FakeRunner) -> list[str]:
    """`cat` に渡った道筋の全部 (照合の記録を、どこから読んだか)。"""
    return [argv[1] for argv in runner.argvs if argv[0] == "cat"]


def test_a_derived_config_passes_the_weights_gate_and_reads_only_the_derived_record(
    tmp_path: Path,
) -> None:
    runner, results = derived_gates_for(
        tmp_path, derived_serve_config(), manifest=derived_manifest()
    )

    assert passed_of(results, "weights_verified", "head")
    assert passed_of(results, "weights_verified", "worker")
    assert set(cat_paths(runner)) == {DERIVED_ALL_RECORD}


def test_a_derived_config_is_refused_without_the_hub_record_being_read(tmp_path: Path) -> None:
    # Hub の記録 (`<slug>.verified.json`) だけがあっても、派生の記録の代わりにしない
    hub_only: Records = {role: {ALL_RECORD: record_json(role)} for role in ROLES}
    runner, results = derived_gates_for(
        tmp_path,
        derived_serve_config(),
        manifest=derived_manifest(),
        script=Script(records=hub_only),
    )

    assert not passed_of(results, "weights_verified", "head")
    assert DERIVED_ALL_RECORD in detail_of(results, "weights_verified", "head")
    assert ALL_RECORD not in cat_paths(runner)


def test_a_derived_config_without_a_manifest_is_refused_before_any_record_is_read(
    tmp_path: Path,
) -> None:
    runner, results = derived_gates_for(tmp_path, derived_serve_config(), manifest=None)

    assert not passed_of(results, "weights_verified", "head")
    assert cat_paths(runner) == []


def test_a_manifest_of_another_origin_revision_is_refused_before_any_record_is_read(
    tmp_path: Path,
) -> None:
    other = derivation(origin_revision="c" * 40)
    config = derived_serve_config()
    manifest = derived_manifest(derived=other)
    runner, results = derived_gates_for(tmp_path, config, manifest=manifest)

    detail = detail_of(results, "weights_verified", "head")
    assert not passed_of(results, "weights_verified", "head")
    assert config.weights is not None
    assert config.weights.identity in detail
    assert manifest.identity in detail
    assert cat_paths(runner) == []


def test_a_manifest_converted_with_another_tool_commit_is_refused_before_any_record_is_read(
    tmp_path: Path,
) -> None:
    # 元の重みが同じでも、変換の条件が違うマニフェストは、この構成の重みではない
    manifest = derived_manifest(derived=derivation(commit="b" * 40))
    runner, results = derived_gates_for(tmp_path, derived_serve_config(), manifest=manifest)

    assert not passed_of(results, "weights_verified", "head")
    assert cat_paths(runner) == []


def test_a_hub_manifest_is_refused_for_a_derived_config(tmp_path: Path) -> None:
    runner, results = derived_gates_for(tmp_path, derived_serve_config(), manifest=MANIFEST)

    assert not passed_of(results, "weights_verified", "head")
    assert "派生のマニフェストでない" in detail_of(results, "weights_verified", "head")
    assert cat_paths(runner) == []


def test_a_derived_manifest_is_refused_for_a_hub_config(tmp_path: Path) -> None:
    """Hub の構成に派生のマニフェストを渡しても、記録を読む前に断る。"""
    config = serve_config()
    runner = runner_of(tmp_path, Script(records=_derived_records()))

    results = g.run_gates(runner, config, NODES, plans_of(config), manifest=derived_manifest())

    assert not passed_of(results, "weights_verified", "head")
    assert "Hub のマニフェストでない" in detail_of(results, "weights_verified", "head")
    assert cat_paths(runner) == []


def test_a_missing_derived_record_is_refused(tmp_path: Path) -> None:
    records: Records = {
        "head": {DERIVED_ALL_RECORD: derived_record_json("head")},
        "worker": {},
    }
    _, results = derived_gates_for(
        tmp_path,
        derived_serve_config(),
        manifest=derived_manifest(),
        script=Script(records=records),
    )

    assert passed_of(results, "weights_verified", "head")
    assert not passed_of(results, "weights_verified", "worker")
    assert DERIVED_ALL_RECORD in detail_of(results, "weights_verified", "worker")


@pytest.mark.parametrize("difference", ["origin-revision", "conversion-commit"])
def test_a_derived_record_of_another_derivation_is_refused(tmp_path: Path, difference: str) -> None:
    recorded = (
        derivation(origin_revision="0" * 40)
        if difference == "origin-revision"
        else derivation(commit="b" * 40)
    )
    records: Records = {
        "head": {DERIVED_ALL_RECORD: derived_record_json("head", derived=recorded)},
        "worker": {DERIVED_ALL_RECORD: derived_record_json("worker")},
    }
    _, results = derived_gates_for(
        tmp_path,
        derived_serve_config(),
        manifest=derived_manifest(),
        script=Script(records=records),
    )

    assert not passed_of(results, "weights_verified", "head")
    assert passed_of(results, "weights_verified", "worker")
    assert DERIVED_ALL_RECORD in detail_of(results, "weights_verified", "head")


def test_a_derived_record_difference_is_labelled_as_the_record(tmp_path: Path) -> None:
    """記録と比べる文では、記録の値の見出しが「記録」になる (構成の値と取り違えない)。"""
    recorded = derivation(commit="b" * 40)
    records: Records = {
        "head": {DERIVED_ALL_RECORD: derived_record_json("head", derived=recorded)},
        "worker": {DERIVED_ALL_RECORD: derived_record_json("worker")},
    }
    _, results = derived_gates_for(
        tmp_path,
        derived_serve_config(),
        manifest=derived_manifest(),
        script=Script(records=records),
    )

    assert not passed_of(results, "weights_verified", "head")
    assert passed_of(results, "weights_verified", "worker")
    detail = detail_of(results, "weights_verified", "head")
    assert f"変換の道具のコミット (記録: {'b' * 40}、マニフェスト: {'a' * 40})" in detail


def test_a_derived_record_with_a_mismatched_file_is_refused(tmp_path: Path) -> None:
    broken = derived_record_json("head", mismatched=("model-00002-of-00002.safetensors",))
    records: Records = {
        "head": {DERIVED_ALL_RECORD: broken},
        "worker": {DERIVED_ALL_RECORD: derived_record_json("worker")},
    }
    _, results = derived_gates_for(
        tmp_path,
        derived_serve_config(),
        manifest=derived_manifest(),
        script=Script(records=records),
    )

    assert not passed_of(results, "weights_verified", "head")
    assert "model-00002-of-00002.safetensors" in detail_of(results, "weights_verified", "head")


def test_a_derived_record_with_a_different_file_count_is_refused(tmp_path: Path) -> None:
    records: Records = {
        "head": {DERIVED_ALL_RECORD: derived_record_json("head", file_count=3)},
        "worker": {DERIVED_ALL_RECORD: derived_record_json("worker")},
    }
    _, results = derived_gates_for(
        tmp_path,
        derived_serve_config(),
        manifest=derived_manifest(),
        script=Script(records=records),
    )

    detail = detail_of(results, "weights_verified", "head")
    assert not passed_of(results, "weights_verified", "head")
    assert "3" in detail and str(len(MANIFEST.files)) in detail


def swapped_derived_manifest() -> kit_types.DerivedWeightsManifest:
    """`derived_manifest()` と、1 ファイルの sha256 だけが違う。

    件数・大きさ・合計・`derivation` は同じである。
    """
    first = MANIFEST_FILES[0]
    files = (first.model_copy(update={"sha256": "f" * 64}), *MANIFEST_FILES[1:])
    return derived_manifest().model_copy(update={"files": files})


def test_a_derived_record_is_refused_when_the_committed_manifest_was_swapped(
    tmp_path: Path,
) -> None:
    """記録が結び付くマニフェストと、いま渡されたマニフェストの中身が違えば断る。

    同じ `derivation`・ファイルの数・大きさの合計でも、1 ファイルの sha256 が違うマニフェストを
    正解にすると、その前のマニフェストで作った記録は、この正解の照合にならない。
    """
    swapped = swapped_derived_manifest()
    assert swapped.derivation == derived_manifest().derivation
    assert swapped.total_bytes == derived_manifest().total_bytes
    assert len(swapped.files) == len(derived_manifest().files)

    runner, results = derived_gates_for(tmp_path, derived_serve_config(), manifest=swapped)

    for role in ROLES:
        assert not passed_of(results, "weights_verified", role)
        detail = detail_of(results, "weights_verified", role)
        assert "マニフェストが違う" in detail
        assert derived_manifest().content_sha256 in detail
        assert swapped.content_sha256 in detail
    assert set(cat_paths(runner)) == {DERIVED_ALL_RECORD}


def test_a_derived_record_is_accepted_when_it_is_bound_to_the_committed_manifest(
    tmp_path: Path,
) -> None:
    """記録が、いま渡されたマニフェストに結び付いていれば、差し替えた側でも通る。

    記録を作り直せば (照合し直せば)、差し替えたマニフェストを正解にできる。
    """
    swapped = swapped_derived_manifest()
    records: Records = {
        role: {DERIVED_ALL_RECORD: derived_record_json(role, manifest=swapped)} for role in ROLES
    }

    _, results = derived_gates_for(
        tmp_path,
        derived_serve_config(),
        manifest=swapped,
        script=Script(records=records),
    )

    assert passed_of(results, "weights_verified", "head")
    assert passed_of(results, "weights_verified", "worker")


def test_a_derived_record_without_the_manifest_sha256_is_refused(tmp_path: Path) -> None:
    """マニフェストへの結び付けのない記録 (`manifest_sha256` がない) は、読めない記録として断る。"""
    unbound = json.loads(derived_record_json("head"))
    del unbound["manifest_sha256"]
    records: Records = {
        "head": {DERIVED_ALL_RECORD: json.dumps(unbound)},
        "worker": {DERIVED_ALL_RECORD: derived_record_json("worker")},
    }

    _, results = derived_gates_for(
        tmp_path,
        derived_serve_config(),
        manifest=derived_manifest(),
        script=Script(records=records),
    )

    assert not passed_of(results, "weights_verified", "head")
    assert "重みの照合の記録を読めない" in detail_of(results, "weights_verified", "head")
    assert passed_of(results, "weights_verified", "worker")


def test_the_derived_record_paths_are_separate_from_the_hub_record_paths(tmp_path: Path) -> None:
    weights = derived_weights()
    assert g.weights_record_path(HEAD, weights, "all") == DERIVED_ALL_RECORD
    assert g.weights_record_path(HEAD, weights, "probe_files") == DERIVED_PROBE_RECORD
    assert len({DERIVED_ALL_RECORD, DERIVED_PROBE_RECORD, ALL_RECORD, PROBE_RECORD}) == 4
    # `cat` で読めるのは `<remote_root>/` の下だけ。使える文字も絞られている
    guard = CallGuard(var_root=tmp_path)
    for path in (DERIVED_ALL_RECORD, DERIVED_PROBE_RECORD):
        guard.check_run(HEAD, ("cat", path), mutating=False)


def test_the_hub_record_path_is_unchanged(tmp_path: Path) -> None:
    # Hub の重みは、いまと同じ道筋 (`<slug>.verified.json`) で読み書きする
    assert g.weights_record_path(HEAD, WEIGHTS, "all") == ALL_RECORD
    assert g.weights_record_path(HEAD, WEIGHTS, "probe_files") == PROBE_RECORD


def test_the_required_space_of_a_derived_config_is_the_derived_manifest_total() -> None:
    manifest = derived_manifest()
    assert g.required_free_bytes(derived_serve_config(), manifest) == manifest.total_bytes


# --- すでに動いているかの判定 --------------------------------------------


def test_already_running_when_the_name_image_and_sha256_match_on_every_node() -> None:
    plans = plans_of(serve_config())
    containers = {
        plan.node: (
            g.OwnContainer(
                id=f"{index}" * 12,
                name=plan.container_name,
                state="running",
                image=plan.labels[LABEL_IMAGE],
                labels=dict(plan.labels),
            ),
        )
        for index, plan in enumerate(plans)
    }
    match = g.match_running(plans, containers)
    assert match.already_running is True
    assert match.differences == ()


def test_a_different_config_sha256_is_not_already_running_and_shows_the_difference() -> None:
    plans = plans_of(serve_config())
    stale = {**plans[0].labels, LABEL_CONFIG_SHA256: "0" * 64}
    containers: dict[NodeRole, tuple[g.OwnContainer, ...]] = {
        "head": (
            g.OwnContainer(
                id="aaaaaaaaaaaa",
                name=plans[0].container_name,
                state="running",
                image=plans[0].labels[LABEL_IMAGE],
                labels=stale,
            ),
        ),
        "worker": (
            g.OwnContainer(
                id="bbbbbbbbbbbb",
                name=plans[1].container_name,
                state="running",
                image=plans[1].labels[LABEL_IMAGE],
                labels=dict(plans[1].labels),
            ),
        ),
    }
    match = g.match_running(plans, containers)
    assert match.already_running is False
    assert any(LABEL_CONFIG_SHA256 in difference for difference in match.differences)


def test_an_absent_container_is_not_a_difference() -> None:
    plans = plans_of(serve_config())
    match = g.match_running(plans, {"head": (), "worker": ()})
    assert match.already_running is False
    assert match.differences == ()


# --- 了承を得る計画 ------------------------------------------------------


def test_the_rollback_only_targets_the_containers_the_plan_starts() -> None:
    plans = plans_of(serve_config())
    approved = g.build_approved_plan(plans)
    names = tuple(plan.container_name for plan in plans)
    assert approved.own_container_names == names
    assert tuple(command.argv for command in approved.rollback) == (
        ("docker", "stop", "-t", str(g.STOP_TIMEOUT_S), names[0]),
        ("docker", "stop", "-t", str(g.STOP_TIMEOUT_S), names[1]),
        ("docker", "rm", names[0]),
        ("docker", "rm", names[1]),
    )
    started = tuple(command.argv for command in approved.forward if isinstance(command, PlannedRun))
    assert started == tuple(plan.argv for plan in plans)


def test_the_forward_commands_can_carry_the_launch_record(tmp_path: Path) -> None:
    plans = plans_of(serve_config())
    push = PlannedPush(
        node="head", local_dir=tmp_path, remote_subdir="state", purpose="起動の記録を置く"
    )
    approved = g.build_approved_plan(plans, extra_forward=(push,))
    assert approved.forward[0] == push
    assert len(approved.forward) == len(plans) + 1


def test_a_foreign_container_name_cannot_be_smuggled_into_the_plan() -> None:
    plans = plans_of(serve_config())
    foreign = PlannedRun(
        node="head", argv=("docker", "stop", "exl3-tp2"), container="exl3-tp2", purpose="よその停止"
    )
    with pytest.raises(ValueError, match="exl3-tp2"):
        g.build_approved_plan(plans, extra_forward=(foreign,))


def test_the_plan_text_shows_the_commands_and_the_target_machines() -> None:
    plans = plans_of(serve_config())
    text = g.format_plan(g.build_approved_plan(plans), NODES)
    assert HEAD.ssh_host in text and WORKER.ssh_host in text
    for plan in plans:
        assert plan.container_name in text
    assert "docker stop" in text and "docker rm" in text


# --- 止めるだけの計画 (`serve stop`。tasks.md 3.5 で足した) ---------------


def own_containers(
    *plans: ContainerPlan, state: str = "running"
) -> dict[NodeRole, tuple[g.OwnContainer, ...]]:
    """一覧の行を通して作った、自分のコンテナ (名前を文字列で直に渡す道を通らない)。"""
    return {plan.node: g.parse_own_containers(ps_line(plan, state=state)) for plan in plans}


def foreign_row(name: str, *, owner: str = OWNER) -> tuple[g.OwnContainer, ...]:
    """一覧の 1 行を、名前と所有のラベルだけ決めて作る (計画の口の歯止めを試すため)。"""
    return g.parse_own_containers(
        json.dumps(
            {
                "ID": "0123456789ab",
                "Names": name,
                "State": "running",
                "Image": IMAGE_REF,
                "Labels": f"{LABEL_OWNER}={owner}",
            }
        )
    )


def forward_runs(plan: ApprovedPlan) -> tuple[PlannedRun, ...]:
    """計画の、前に進むコマンドのうち、遠隔のコマンドだけ (配布は入らない)。"""
    return tuple(command for command in plan.forward if isinstance(command, PlannedRun))


def test_the_stop_plan_holds_stop_then_remove_for_the_listed_containers() -> None:
    """`serve stop` の計画は、一覧の行の名前の `stop` → `rm` だけを持つ (requirements 1.7)。"""
    plans = plans_of(serve_config())
    names = tuple(plan.container_name for plan in plans)

    approved = g.build_stop_plan(own_containers(*plans))

    assert approved.own_container_names == names
    assert approved.rollback == ()
    assert len(forward_runs(approved)) == len(approved.forward)
    assert tuple(command.argv for command in forward_runs(approved)) == (
        g.stop_argv(names[0]),
        g.stop_argv(names[1]),
        g.remove_argv(names[0]),
        g.remove_argv(names[1]),
    )


def test_the_stop_plan_puts_the_head_before_the_worker() -> None:
    """止める順序は head → worker (渡した辞書の順に引きずられない)。"""
    plans = plans_of(serve_config())
    found = own_containers(*plans)
    reversed_order: dict[NodeRole, tuple[g.OwnContainer, ...]] = {
        "worker": found["worker"],
        "head": found["head"],
    }

    approved = g.build_stop_plan(reversed_order)

    assert [command.node for command in approved.forward] == ["head", "worker", "head", "worker"]


def test_the_stop_plan_works_for_an_exited_container() -> None:
    """終了した自分のコンテナも、同じ `stop` → `rm` の列で片付ける。"""
    plans = plans_of(serve_config())

    approved = g.build_stop_plan(own_containers(*plans, state="exited"))

    assert [command.argv[1] for command in forward_runs(approved)] == ["stop", "stop", "rm", "rm"]


def test_the_stop_plan_refuses_a_row_that_is_not_ours() -> None:
    """所有のラベルのない行は、止める計画に入れられない (requirements 2.3)。"""
    with pytest.raises(ValueError, match=LABEL_OWNER):
        g.build_stop_plan({"head": foreign_row("exl3-tp2", owner="someone-else")})


def test_the_stop_plan_refuses_a_name_that_could_be_read_as_a_flag() -> None:
    """`docker stop` の対象と読めない名前は断る (フラグに化けさせない)。"""
    with pytest.raises(ValueError):
        g.build_stop_plan({"head": foreign_row("--force")})


def test_the_stop_plan_refuses_an_empty_listing() -> None:
    """止める対象が 1 つもなければ、空の計画を作らない (呼ぶ側が、先に判定する)。"""
    with pytest.raises(ValueError):
        g.build_stop_plan({"head": (), "worker": ()})


def test_the_stop_plan_passes_the_check_of_the_remote_runner(tmp_path: Path) -> None:
    """止める列が、そのまま `remote` の了承の検査を通る (完全な一致で見るため)。"""
    plans = plans_of(serve_config())
    approved = g.build_stop_plan(own_containers(*plans))
    guard = CallGuard(var_root=tmp_path, plan=approved)

    for command in forward_runs(approved):
        guard.check_run(NODES[command.node], command.argv, mutating=True)


# --- コンテナを対象にする操作の不変条件 ----------------------------------

_CONTAINER_SUBCOMMANDS = frozenset(
    {"stop", "rm", "logs", "top", "kill", "start", "restart", "pause", "unpause", "wait", "port"}
)
"""コンテナを対象に取る docker のサブコマンド (許可の一覧の外のものも並べて見張る)。"""

_VALUE_FLAGS = frozenset({"-t", "--time", "--tail", "--since", "--until", "-s", "--signal"})
"""値を取るフラグ (`docker stop -t 90 <名前>` の 90 を、対象と読み違えないため)。"""


def container_targets(argv: Sequence[str]) -> tuple[str, ...]:
    """docker の呼び出しから、コンテナを対象にしている語だけを取り出す。"""
    if tuple(argv[:1]) != ("docker",) or len(argv) < 2:
        return ()
    rest = list(argv[1:])
    if rest[0] == "container" and len(rest) > 1:
        if rest[1] != "inspect":
            return ()
        rest = rest[2:]
    elif rest[0] == "run":
        return tuple(value for flag, value in zip(rest, rest[1:], strict=False) if flag == "--name")
    elif rest[0] in _CONTAINER_SUBCOMMANDS:
        rest = rest[1:]
    else:
        return ()
    targets: list[str] = []
    skip = False
    for item in rest:
        if skip:
            skip = False
            continue
        if item in _VALUE_FLAGS:
            skip = True
            continue
        if item.startswith("-"):
            continue
        targets.append(item)
    return tuple(targets)


def test_the_target_reader_finds_the_targets_it_must_watch() -> None:
    # この見張りそのものが空振りでないこと
    assert container_targets(("docker", "stop", "-t", "90", "vb-x-head")) == ("vb-x-head",)
    assert container_targets(("docker", "rm", "vb-x-head")) == ("vb-x-head",)
    assert container_targets(("docker", "logs", "--tail", "80", "abc123")) == ("abc123",)
    assert container_targets(("docker", "container", "inspect", "abc123")) == ("abc123",)
    assert container_targets(("docker", "run", "-d", "--name", "vb-x-head", IMAGE_REF)) == (
        "vb-x-head",
    )
    # イメージを対象にする呼び出しと、一覧は、コンテナの対象ではない
    image_inspect = ("docker", "image", "inspect", "--format", "{{json .X}}", IMAGE_REF)
    assert container_targets(image_inspect) == ()
    assert container_targets(("docker", "ps", "-a", "--filter", "label=x")) == ()


def test_every_docker_call_targets_only_our_containers(tmp_path: Path) -> None:
    config = serve_config()
    plans = plans_of(config)
    ps = dict(_fixture_map("docker-ps-own.txt"))
    ps["head"] = ps_line(plans_of(fetch_config())[0], container_id="feedfacefeed") + "\n"
    runner = runner_of(tmp_path, Script(ps=ps))
    containers = {role: g.list_own_containers(runner, NODES[role]) for role in ROLES}
    results = g.run_gates(runner, config, NODES, plans, manifest=MANIFEST, containers=containers)
    assert refused(results) != ()  # 取得が動いているので、関門は断る
    approved = g.build_approved_plan(plans)

    # 「一覧が返したもの = 自分のもの」を前提にすると循環するので、試験の側でも、
    # 1 件ずつ、所有のラベルを見てから、触ってよい識別子として数える
    from_listing: set[str] = set()
    for found in containers.values():
        for container in found:
            assert container.labels.get(LABEL_OWNER) == OWNER, "所有のラベルのない行が混ざった"
            from_listing.add(container.id)
    from_plan = {plan.container_name for plan in plans}
    checked = 0
    planned = tuple(
        command.argv
        for command in (*approved.forward, *approved.rollback)
        if isinstance(command, PlannedRun)
    )
    for argv in (*runner.argvs, *planned):
        for target in container_targets(argv):
            checked += 1
            assert target in from_listing | from_plan, (
                f"ラベルで絞った一覧にも、了承済みの計画にもない対象: {target} ({argv})"
            )
    assert checked >= len(plans) * 3  # run + stop + rm のぶん
    assert "feedfacefeed" in from_listing


def test_no_gate_touches_a_container_at_all(tmp_path: Path) -> None:
    # 関門は、ラベルのないコンテナにも、自分のコンテナにも、どの操作も向けない
    ps = dict(_fixture_map("docker-ps-own.txt"))
    ps["head"] = ps_line(plans_of(serve_config())[0], state="exited") + "\n"
    runner, _ = gates_for(tmp_path, serve_config(), Script(ps=ps))
    for argv in runner.argvs:
        assert container_targets(argv) == (), f"関門がコンテナを対象にした: {argv}"
        assert argv[:2] != ("docker", "inspect")


# --- 了承 ---------------------------------------------------------------


def test_the_terminal_confirmer_passes_on_yes() -> None:
    err = io.StringIO()
    confirmer = g.TerminalConfirmer(stdin=io.StringIO("yes\n"), stderr=err, isatty=lambda: True)
    assert confirmer.confirm("計画の全体") is True
    assert "計画の全体" in err.getvalue()


def test_the_terminal_confirmer_refuses_any_other_input() -> None:
    for answer in ("y\n", "no\n", "YES\n", "\n", ""):
        confirmer = g.TerminalConfirmer(
            stdin=io.StringIO(answer), stderr=io.StringIO(), isatty=lambda: True
        )
        assert confirmer.confirm("計画の全体") is False, answer


@pytest.mark.parametrize(
    ("answer", "approved"),
    [
        ("yes\n", True),
        ("yes\r\n", True),
        ("yes", True),  # 改行なしで終わった入力
        ("yes \n", False),
        (" yes\n", False),
        ("\tyes\n", False),
        ("yes.\n", False),
    ],
)
def test_the_terminal_confirmer_compares_the_input_exactly(answer: str, approved: bool) -> None:
    # 落とすのは、末尾の改行だけ。前後の空白を落とすと、打ち間違いが了承になる
    confirmer = g.TerminalConfirmer(
        stdin=io.StringIO(answer), stderr=io.StringIO(), isatty=lambda: True
    )
    assert confirmer.confirm("計画の全体") is approved


def test_the_terminal_confirmer_refuses_when_there_is_no_terminal() -> None:
    err = io.StringIO()
    confirmer = g.TerminalConfirmer(stdin=io.StringIO("yes\n"), stderr=err, isatty=lambda: False)
    with pytest.raises(g.ApprovalError, match="--yes"):
        confirmer.confirm("計画の全体")


def test_the_assume_yes_confirmer_shows_the_plan_and_passes() -> None:
    err = io.StringIO()
    confirmer = g.AssumeYesConfirmer(stderr=err)
    assert confirmer.confirm("計画の全体") is True
    assert "計画の全体" in err.getvalue()


def test_the_confirmer_is_chosen_by_the_yes_flag() -> None:
    assert isinstance(g.make_confirmer(assume_yes=True, stderr=io.StringIO()), g.AssumeYesConfirmer)
    assert isinstance(
        g.make_confirmer(assume_yes=False, stdin=io.StringIO(), stderr=io.StringIO()),
        g.TerminalConfirmer,
    )


def test_the_plan_reaches_the_runner_only_after_the_approval(tmp_path: Path) -> None:
    plans = plans_of(serve_config())
    approved = g.build_approved_plan(plans)

    refusing = runner_of(tmp_path)
    with pytest.raises(g.ApprovalError):
        g.request_approval(
            g.TerminalConfirmer(
                stdin=io.StringIO("no\n"), stderr=io.StringIO(), isatty=lambda: True
            ),
            refusing,
            approved,
            NODES,
        )
    assert refusing.plan is None
    assert refusing.calls == ()

    accepting = runner_of(tmp_path)
    g.request_approval(g.AssumeYesConfirmer(stderr=io.StringIO()), accepting, approved, NODES)
    assert accepting.plan is approved


# --- 壊れた出力を、黙って通さない ----------------------------------------


def _direct_gates(runner: FakeRunner) -> dict[str, Callable[[], GateResult]]:
    """個々の関門を、`run_gates` を通さずに直に呼ぶ口 (3.1 の `serve pull-image` の使い方)。"""
    config = serve_config()
    plan = plans_of(config)[0]
    return {
        "reachable": lambda: g.gate_reachable(runner, HEAD),
        "gpu_idle": lambda: g.gate_gpu_idle(runner, HEAD),
        "layout": lambda: g.gate_layout(runner, HEAD, plan),
        "image_digest": lambda: g.gate_image_digest(runner, HEAD, config),
        "weights_verified": lambda: g.gate_weights_verified(runner, HEAD, config, MANIFEST),
        "disk_space": lambda: g.gate_disk_space(runner, HEAD, MANIFEST.total_bytes),
        "ports_free": lambda: g.gate_ports_free(runner, HEAD, config),
    }


DIRECT_GATES = (
    "reachable",
    "gpu_idle",
    "layout",
    "image_digest",
    "weights_verified",
    "disk_space",
    "ports_free",
)
PARSING_GATES = ("gpu_idle", "image_digest", "weights_verified", "disk_space", "ports_free")
"""出力を読む関門 (`layout` は `test -e` の終了コードだけ、`reachable` は名前を見ない)。"""


@pytest.mark.parametrize("gate", DIRECT_GATES)
def test_a_gate_called_directly_refuses_instead_of_raising(tmp_path: Path, gate: str) -> None:
    # 3.1 は、`run_gates` を通さずに `gate_reachable` と `gate_disk_space` を直に呼ぶ
    runner = FakeRunner(var_root=tmp_path, default=Reply(exit_code=255, stderr="No route to host"))
    result = _direct_gates(runner)[gate]()
    assert result.passed is False
    assert result.node == "head"
    assert result.detail


@pytest.mark.parametrize("gate", PARSING_GATES)
def test_a_gate_called_directly_refuses_an_output_it_cannot_read(tmp_path: Path, gate: str) -> None:
    runner = FakeRunner(var_root=tmp_path, default=Reply(stdout="??? 読めない出力 ???\n"))
    result = _direct_gates(runner)[gate]()
    assert result.passed is False


@pytest.mark.parametrize(
    "output",
    ["{ これは JSON ではない\n", '"文字列だけの行"\n', "{}\n", '{"ID": "abc"}\n'],
)
def test_a_line_of_the_container_list_that_cannot_be_read_is_an_error(
    tmp_path: Path, output: str
) -> None:
    # 読めない行を捨てて進むと、「自分のコンテナは 1 つもない」と読めて、関門が素通しになる
    ps = dict(_fixture_map("docker-ps-own.txt"))
    ps["head"] = output
    with pytest.raises(ValueError):
        g.list_own_containers(runner_of(tmp_path, Script(ps=ps)), HEAD)
    _, results = gates_for(tmp_path, serve_config(), Script(ps=ps))
    assert not passed_of(results, "own_state", "head")


def test_a_broken_line_next_to_a_good_one_is_an_error(tmp_path: Path) -> None:
    ps = dict(_fixture_map("docker-ps-own.txt"))
    ps["head"] = ps_line(plans_of(serve_config())[0]) + "\n{ これは JSON ではない\n"
    with pytest.raises(ValueError):
        g.list_own_containers(runner_of(tmp_path, Script(ps=ps)), HEAD)


@pytest.mark.parametrize("output", ["No devices were found\n", "1234, python3\n"])
def test_a_line_from_nvidia_smi_without_three_columns_refuses_the_gate(
    tmp_path: Path, output: str
) -> None:
    gpu = dict(_fixture_map("nvidia-smi-compute-apps.txt"))
    gpu["head"] = output
    _, results = gates_for(tmp_path, serve_config(), Script(gpu=gpu))
    assert not passed_of(results, "gpu_idle", "head")


@pytest.mark.parametrize("output", ["        Avail\n", "        Avail\nたくさん\n", ""])
def test_an_unreadable_disk_output_refuses_the_gate(tmp_path: Path, output: str) -> None:
    df = dict(_fixture_map("df-avail.txt"))
    df["head"] = output
    _, results = gates_for(tmp_path, serve_config(), Script(df=df))
    assert not passed_of(results, "disk_space", "head")


def test_an_unreadable_listening_list_refuses_the_gate(tmp_path: Path) -> None:
    ss = dict(_fixture_map("ss-ltnH.txt"))
    ss["head"] = "State Recv-Q Send-Q Local-Address:Port Peer\n"
    _, results = gates_for(tmp_path, serve_config(), Script(ss=ss))
    assert not passed_of(results, "ports_free", "head")


def test_the_port_comparison_is_exact(tmp_path: Path) -> None:
    # 前方一致や部分一致で読むと、80800 が 8080 に、8080 が 80 に当たってしまう
    ss = dict(_fixture_map("ss-ltnH.txt"))
    ss["head"] = "LISTEN 0      4096                      0.0.0.0:80800 0.0.0.0:*\n"
    ss["worker"] = ss["head"]
    _, wide = gates_for(tmp_path, serve_config(port="8080"), Script(ss=ss))
    assert passed_of(wide, "ports_free", "head")
    _, narrow = gates_for(tmp_path, serve_config(port="80"))
    assert passed_of(narrow, "ports_free", "head")


# --- 一覧の行が、本当に自分のものか --------------------------------------


def test_a_row_without_our_owner_label_is_an_error(tmp_path: Path) -> None:
    # 絞り込みが効いていない、という異常なので、行を捨てずに断る (名前も ID も見せない)
    plan = plans_of(serve_config())[0]
    without = {key: value for key, value in plan.labels.items() if key != LABEL_OWNER}
    ps = dict(_fixture_map("docker-ps-own.txt"))
    ps["head"] = ps_line(plan, labels=without, container_id="deadbeefdead") + "\n"
    with pytest.raises(ValueError, match="所有"):
        g.list_own_containers(runner_of(tmp_path, Script(ps=ps)), HEAD)
    _, results = gates_for(tmp_path, serve_config(), Script(ps=ps))
    detail = detail_of(results, "own_state", "head")
    assert not passed_of(results, "own_state", "head")
    assert "deadbeefdead" not in detail
    assert plan.container_name not in detail


def test_a_row_owned_by_someone_else_is_an_error(tmp_path: Path) -> None:
    plan = plans_of(serve_config())[0]
    foreign = {**plan.labels, LABEL_OWNER: "someone-else"}
    ps = dict(_fixture_map("docker-ps-own.txt"))
    ps["head"] = ps_line(plan, labels=foreign) + "\n"
    with pytest.raises(ValueError, match="所有"):
        g.list_own_containers(runner_of(tmp_path, Script(ps=ps)), HEAD)


def test_match_running_does_not_accept_a_row_owned_by_someone_else() -> None:
    plans = plans_of(serve_config())
    containers: dict[NodeRole, tuple[g.OwnContainer, ...]] = {
        plan.node: (
            g.OwnContainer(
                id=f"{index}" * 12,
                name=plan.container_name,
                state="running",
                image=plan.labels[LABEL_IMAGE],
                labels={**plan.labels, LABEL_OWNER: "someone-else"},
            ),
        )
        for index, plan in enumerate(plans)
    }
    match = g.match_running(plans, containers)
    assert match.already_running is False
    assert any(LABEL_OWNER in difference for difference in match.differences)


def test_a_label_key_that_appears_twice_is_an_error(tmp_path: Path) -> None:
    # `Labels` は `鍵=値,鍵=値` の 1 つの文字列なので、値に `,鍵=値` を書けば鍵を足せる
    plan = plans_of(serve_config())[0]
    injected = (
        ",".join(f"{key}={value}" for key, value in sorted(plan.labels.items()))
        + f",{LABEL_CONFIG_SHA256}={'0' * 64}"
    )
    ps = dict(_fixture_map("docker-ps-own.txt"))
    ps["head"] = ps_line(plan, labels_text=injected) + "\n"
    with pytest.raises(ValueError, match="2 度"):
        g.list_own_containers(runner_of(tmp_path, Script(ps=ps)), HEAD)


# --- 前に進むコマンドの、対象の読み取り ----------------------------------


def _forward(argv: tuple[str, ...]) -> PlannedRun:
    return PlannedRun(node="head", argv=argv, purpose="試験のための前に進むコマンド")


@pytest.mark.parametrize(
    "argv",
    [
        ("docker", "stop", "-t", "90", "12345"),
        ("docker", "rm", "909090909090"),
        ("docker", "stop", "--time=90", "exl3-tp2"),
        ("docker", "rm", "-f", "exl3-tp2"),
    ],
)
def test_a_foreign_target_cannot_be_smuggled_into_the_forward_commands(
    argv: tuple[str, ...],
) -> None:
    # 数字だけの名前や短い ID も、値を取るフラグの値と読み違えずに、対象として見る
    plans = plans_of(serve_config())
    with pytest.raises(ValueError, match="この計画が起こさないコンテナ"):
        g.build_approved_plan(plans, extra_forward=(_forward(argv),))


@pytest.mark.parametrize(
    "argv",
    [
        ("docker", "stop", "--signal-file", "x", "vb-p1-nvfp4-tp2-head"),
        ("docker", "rm", "--filter", "label=x", "vb-p1-nvfp4-tp2-head"),
    ],
)
def test_an_unknown_flag_in_a_stop_or_remove_is_refused(argv: tuple[str, ...]) -> None:
    # 読み方の分からないフラグがあると、どれが対象かを読み違える
    plans = plans_of(serve_config())
    with pytest.raises(ValueError, match="フラグ"):
        g.build_approved_plan(plans, extra_forward=(_forward(argv),))


@pytest.mark.parametrize(
    "argv",
    [
        ("docker", "run", "-d", "--name", "x", IMAGE_REF),
        ("docker", "container", "run", "-d", "--name", "x", IMAGE_REF),
        ("docker", "create", "--name", "x", IMAGE_REF),
        ("docker", "container", "create", "--name", "x", IMAGE_REF),
    ],
)
def test_a_container_cannot_be_started_from_the_forward_commands(argv: tuple[str, ...]) -> None:
    # コンテナを起こす道は、計画 (ContainerPlan) の 1 つだけ (`_check_plans` を必ず通す)
    plans = plans_of(serve_config())
    with pytest.raises(ValueError, match="コンテナを起こす"):
        g.build_approved_plan(plans, extra_forward=(_forward(argv),))


def test_the_forward_commands_still_carry_a_pull_and_a_mkdir(tmp_path: Path) -> None:
    plans = plans_of(serve_config())
    pull = _forward(("docker", "pull", IMAGE_REF))
    make = _forward(("mkdir", "-p", f"{REMOTE_ROOT}/state"))
    push = PlannedPush(
        node="head", local_dir=tmp_path, remote_subdir="state", purpose="起動の記録を置く"
    )
    approved = g.build_approved_plan(plans, extra_forward=(pull, make, push))
    assert approved.forward[:3] == (pull, make, push)


def test_the_two_word_form_is_refused_by_the_remote_runner(tmp_path: Path) -> None:
    # 層の受け持ち: 2 語の形 (`docker container stop`) は、guards の検査の対象ではない。
    # `remote.CallGuard` が、許可の一覧にないサブコマンドとして断る
    plans = plans_of(serve_config())
    sneaky = _forward(("docker", "container", "stop", plans[0].container_name))
    approved = g.build_approved_plan(plans, extra_forward=(sneaky,))
    runner = runner_of(tmp_path)
    runner.approve(approved)
    with pytest.raises(RuntimeError, match="許可の一覧"):
        runner.run(HEAD, sneaky.argv, timeout_s=5.0, mutating=True)


def test_a_programming_error_inside_a_gate_is_not_turned_into_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 断りに変えるのは、届かなかったこと (RemoteError) と、読めなかったこと (ValueError)
    # だけ。実装の誤りを「断り」に見せない
    def explode(text: str) -> tuple[GpuApp, ...]:
        raise TypeError("実装の誤り")

    monkeypatch.setattr(g, "parse_gpu_apps", explode)
    runner = runner_of(tmp_path)
    with pytest.raises(TypeError):
        g.gate_gpu_idle(runner, HEAD)
    config = serve_config()
    with pytest.raises(TypeError):
        g.run_gates(runner, config, NODES, plans_of(config), manifest=MANIFEST)


def test_stopping_our_own_container_is_allowed_in_the_forward_commands() -> None:
    plans = plans_of(serve_config())
    name = plans[0].container_name
    approved = g.build_approved_plan(plans, extra_forward=(_forward(g.stop_argv(name)),))
    first = approved.forward[0]
    assert isinstance(first, PlannedRun)
    assert first.argv == ("docker", "stop", "-t", str(g.STOP_TIMEOUT_S), name)


# --- 計画の引数の列そのものを確かめる ------------------------------------


def test_a_plan_that_is_not_a_docker_run_is_refused() -> None:
    plan = plans_of(serve_config())[0]
    broken = plan.model_copy(update={"argv": ("docker", "stop", plan.container_name)})
    with pytest.raises(ValueError, match="docker run"):
        g.build_approved_plan((broken,))


def test_a_plan_whose_argv_starts_another_name_is_refused() -> None:
    plan = plans_of(serve_config())[0]
    broken = plan.model_copy(update={"container_name": "vb-someone-else-head"})
    with pytest.raises(ValueError, match="--name"):
        g.build_approved_plan((broken,))


def test_a_plan_without_our_owner_label_in_its_argv_is_refused() -> None:
    plan = plans_of(serve_config())[0]
    kept: list[str] = []
    skip = False
    for index, word in enumerate(plan.argv):
        if skip:
            skip = False
            continue
        if word == "--label" and plan.argv[index + 1] == f"{LABEL_OWNER}={OWNER}":
            skip = True
            continue
        kept.append(word)
    broken = plan.model_copy(update={"argv": tuple(kept)})
    with pytest.raises(ValueError, match="所有"):
        g.build_approved_plan((broken,))


# --- 依存の向き ---------------------------------------------------------

ALLOWED_SERVING_MODULES = {"types", "config", "remote", "plan", "observe"}


def test_guards_imports_only_the_modules_the_design_allows() -> None:
    tree = ast.parse(Path(inspect.getfile(g)).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("serving_kit"):
            if node.module == "serving_kit":
                imported.update(alias.name for alias in node.names)
            else:
                imported.add((node.module or "").split(".")[-1])
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("serving_kit."):
                    imported.add(alias.name.split(".")[-1])
    assert imported <= ALLOWED_SERVING_MODULES, imported


# --- 巻き戻しの相手を、一覧で確かめる (tasks.md 3.1 のレビューでの補強) ---


def test_the_rollback_commands_come_from_our_listing(tmp_path: Path) -> None:
    # 2 台とも、自分の一覧にその名前がある → 了承済みの計画の巻き戻しと、完全に同じ列が返る
    plans = plans_of(serve_config())
    ps = {plan.node: ps_line(plan, state="exited") + "\n" for plan in plans}
    runner = runner_of(tmp_path, Script(ps=ps))
    commands, unchecked = g.rollback_commands(runner, NODES, plans)
    assert unchecked == ()
    assert commands == g.build_approved_plan(plans).rollback


def test_no_rollback_command_for_a_name_that_is_not_in_our_listing(tmp_path: Path) -> None:
    # 名前の衝突 (その名前は、よそのコンテナのもの) と、コンテナができなかった場合。
    # `docker stop <名前>` は、誰のものでも止めてしまうので、一覧にない名前には行かない
    plans = plans_of(serve_config())
    ps: dict[NodeRole, str] = {"head": ps_line(plans[0], state="exited") + "\n", "worker": ""}
    runner = runner_of(tmp_path, Script(ps=ps))
    commands, unchecked = g.rollback_commands(runner, NODES, plans)
    assert unchecked == ()
    assert [command.node for command in commands] == ["head", "head"]
    assert plans[1].container_name not in " ".join(" ".join(command.argv) for command in commands)


def test_a_listing_that_cannot_be_read_stops_the_rollback_with_a_reason(tmp_path: Path) -> None:
    # worker に入れない (一覧を読めない) ので、その台では、名前に向けて止めも消しもしない
    plans = plans_of(serve_config())
    unreachable = Rule(
        prefix=("docker", "ps"),
        node="worker",
        replies=(Reply(exit_code=255, stderr="ssh: connect to host port 22: No route to host"),),
    )
    runner = FakeRunner(
        var_root=tmp_path, script=(unreachable, *Script(ps={"head": "", "worker": ""}).rules())
    )
    commands, unchecked = g.rollback_commands(runner, NODES, plans)
    assert commands == ()
    assert len(unchecked) == 1
    assert "worker" in unchecked[0] and "serve stop" in unchecked[0]


def test_the_reason_does_not_show_anything_about_foreign_containers(tmp_path: Path) -> None:
    # 所有のラベルのない行が紛れた一覧は、使わずに断る。理由に、その行の名前も識別子も出さない
    plans = plans_of(serve_config())
    foreign = json.dumps(
        {
            "ID": "deadbeefdead",
            "Names": "exl3-tp2",
            "State": "running",
            "Image": OTHER_REF,
            "Labels": "com.example.owner=someone-else",
        }
    )
    runner = runner_of(tmp_path, Script(ps={"head": foreign + "\n", "worker": ""}))
    commands, unchecked = g.rollback_commands(runner, NODES, plans)
    assert commands == ()
    assert len(unchecked) == 1
    assert "exl3-tp2" not in unchecked[0] and "deadbeefdead" not in unchecked[0]


def test_deciding_the_rollback_does_not_change_the_state(tmp_path: Path) -> None:
    plans = plans_of(serve_config())
    ps = {plan.node: ps_line(plan, state="exited") + "\n" for plan in plans}
    runner = runner_of(tmp_path, Script(ps=ps))
    g.rollback_commands(runner, NODES, plans)
    assert [call for call in runner.calls if call.mutating] == []
    for argv in runner.argvs:
        assert argv[:2] == ("docker", "ps"), f"一覧の読み取り以外を出した: {argv}"
        assert f"label={OWNER_FILTER}" in argv


def test_the_rollback_commands_work_for_an_ab_run_name(tmp_path: Path) -> None:
    # A/B の回は、名前に `-r<回>` が付く (design.md 「netcheck」)。一覧の突き合わせも、
    # その名前で行う (4.4 が、同じ関数で片付けを決める)
    config = serve_config()
    plans = build_plans(config, NODES, STARTED_AT, arm="baseline", repeat_index=2)
    assert [plan.container_name for plan in plans] == [
        "vb-p1-nvfp4-tp2-head-r2",
        "vb-p1-nvfp4-tp2-worker-r2",
    ]
    ps = {plan.node: ps_line(plan, state="exited") + "\n" for plan in plans}
    runner = runner_of(tmp_path, Script(ps=ps))
    commands, unchecked = g.rollback_commands(runner, NODES, plans)
    assert unchecked == ()
    assert commands == g.build_approved_plan(plans).rollback
    assert all(command.argv[-1].endswith("-r2") for command in commands)
