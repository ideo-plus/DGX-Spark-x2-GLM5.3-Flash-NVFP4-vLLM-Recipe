"""共有の型の試験 (task 1.2)。

確かめること:

- すべての公開の型が JSON と Python の値を往復できる (見本を 1 つずつ持つ)
- すべての公開の型が、書いたあとに変えられず、知らない項目を断る
- 根拠 (`Provenance`) が、「出典と原文の組」か「実測の記録の場所」の、どちらか
  一方だけを受ける (requirements 3.6、design.md 「types / config」の検査 1)
- 設計が名指しする形の決まり (イメージのダイジェスト、40 桁の版、名乗るモデルの
  名前、コンテナの名前、マニフェストの並びと合計) が、型の検証で断られる
- 巻き戻しの計画が、その計画が起こす名前のコンテナ以外を対象にできない
  (design.md 「remote」)
- 結果の型に、送った内容と応答の本文を入れる項目がない (requirements 10.5)
- `types.py` が、ほかの `serving_kit` の module を読み込まない (依存の向き)
"""

from __future__ import annotations

import ast
import inspect
import re
from datetime import UTC, datetime
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import BaseModel, HttpUrl, ValidationError

from serving_kit import types as t

# --- 例の組み立て -------------------------------------------------------

AT = datetime(2026, 9, 21, 3, 0, 0, tzinfo=UTC)
DIGEST = "b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
REVISION = "18d55bfd" + "0" * 32
CONFIG_SHA = "c" * 64
SOURCE = HttpUrl("https://docs.vllm.ai/en/latest/cli/serve/")
QUOTE = "vllm serve [model_tag] [options]"

PROVENANCE = t.Provenance(source=SOURCE, quote=QUOTE)
MEASURED = t.Provenance(measured="docs/results/2026-09-21-netcheck-links.md")

SETTING_FLAG = t.Setting(
    flag="--tensor-parallel-size",
    value="2",
    why="2 台を 1 つの推論サーバーとして使う",
    source=SOURCE,
    quote=QUOTE,
)
SETTING_POSITIONAL = t.Setting(
    value="{weights.mount_at}",
    why="vllm serve の第一の位置の引数は、モデルの場所",
    source=SOURCE,
    quote=QUOTE,
)
SETTING_PORT = t.Setting(
    flag="--port",
    value="8000",
    why="既存の対象サーバー (8001) と重ならない番号",
    is_port=True,
    source=SOURCE,
    quote=QUOTE,
)
SETTING_WORKER = t.Setting(
    flag="--node-rank",
    value="1",
    why="2 台のうちの 2 台目",
    only_on="worker",
    source=SOURCE,
    quote=QUOTE,
)

IMAGE = t.ImageRef(
    ref=f"vllm/vllm-openai@sha256:{DIGEST}",
    seen_as="vllm/vllm-openai:glm53-flash-arm64-cu130",
    size_bytes=9666567584,
    source=HttpUrl("https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags"),
    quote="glm53-flash-arm64-cu130 / linux/arm64 / 9666567584 B",
)
WEIGHTS = t.WeightsRef(
    repo="RedHatAI/GLM-5.3-Flash-NVFP4",
    revision=REVISION,
    manifest="RedHatAI__GLM-5.3-Flash-NVFP4.manifest.json",
    mount_at="/models/nvfp4",
    source=HttpUrl("https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4"),
    quote="license: mit",
)

CONFIG = t.ConfigDef(
    name="p1-nvfp4-tp2",
    kind="serve",
    description="P1 の第一の構成。上流の公式のイメージ + NVFP4 + 2 台 TP=2",
    nodes=("head", "worker"),
    image=IMAGE,
    weights=WEIGHTS,
    docker={"ipc-host": SETTING_FLAG},
    args={"model-path": SETTING_POSITIONAL, "port": SETTING_PORT, "node-rank": SETTING_WORKER},
    env={
        "vllm-host-ip": t.Setting(
            flag="VLLM_HOST_IP",
            value="{node.fabric_addr}",
            why="2 台の間の通信に、直結の側のアドレスを使わせる",
            source=SOURCE,
            quote=QUOTE,
        )
    },
    ready_timeout_s=1800,
    served_model_name="glm-5-3-flash",
)

NODE_HEAD = t.NodeDef(
    role="head",
    ssh_host="spark-153d",
    lan_addr=IPv4Address("10.0.1.60"),
    remote_root="/home/j5ik2o/vllm-baseline",
)
NODE_WORKER = t.NodeDef(
    role="worker",
    ssh_host="spark-5083",
    lan_addr=IPv4Address("10.0.1.61"),
    remote_root="/home/j5ik2o/vllm-baseline",
    fabric_addr=IPv4Address("192.168.100.2"),
    fabric_ifname="enp1s0f0np1",
    fabric_measured="docs/results/2026-09-21-netcheck-links.md",
)

HEAD_CONTAINER = "vb-p1-nvfp4-tp2-head"
CONTAINER_PLAN = t.ContainerPlan(
    node="head",
    container_name=HEAD_CONTAINER,
    labels={"vllm-baseline.owner": "serving-kit", "vllm-baseline.config": "p1-nvfp4-tp2"},
    argv=("docker", "run", "-d", "--pull", "never", "--name", HEAD_CONTAINER),
)
PLANNED_RUN = t.PlannedRun(
    node="head",
    argv=CONTAINER_PLAN.argv,
    container=HEAD_CONTAINER,
    purpose="head の推論サーバーを起こす",
)
PLANNED_PUSH = t.PlannedPush(
    node="head",
    local_dir=Path("serving/var/state"),
    remote_subdir="state",
    delete=False,
    purpose="起動の記録を置く",
)
ROLLBACK_STOP = t.PlannedRun(
    node="head",
    argv=("docker", "stop", "-t", "90", HEAD_CONTAINER),
    container=HEAD_CONTAINER,
    purpose="head のコンテナを止める",
)
ROLLBACK_RM = t.PlannedRun(
    node="head",
    argv=("docker", "rm", HEAD_CONTAINER),
    container=HEAD_CONTAINER,
    purpose="head のコンテナを消す",
)
APPROVED_PLAN = t.ApprovedPlan(
    forward=(PLANNED_PUSH, PLANNED_RUN),
    rollback=(ROLLBACK_STOP, ROLLBACK_RM),
    own_container_names=(HEAD_CONTAINER,),
)

COMMAND_RESULT = t.CommandResult(
    argv=("uname", "-n"),
    exit_code=0,
    stdout="spark-153d\n",
    stderr="",
    duration_s=0.12,
)
GATE_REFUSED = t.GateResult(
    gate="gate_gpu_idle",
    node="head",
    passed=False,
    detail="GPU を使っているプロセスがある: python3 (pid 12345、1024 MiB)",
)

LAUNCH_OBS = t.LaunchObservation(
    vllm_version="0.11.1",
    attention_backend="FLASHINFER_MLA",
    attention_candidates=("FLASHINFER_MLA", "TRITON_MLA"),
    moe_backend="triton",
    kv_cache_tokens=163840,
    kv_cache_gib=12.5,
    max_concurrency_note="Maximum concurrency for 163840 tokens per request: 1.00x",
    model_loading_gib=92.0,
    model_loading_s=612.5,
    engine_init_s=740.0,
    speculative_config_seen=False,
)
NCCL_OBS = t.NcclObservation(
    nccl_version="2.28.3",
    network="IB",
    ib_no_device=False,
    ib_devices_line="NCCL INFO NET/IB : Using [0]rocep1s0f0:1/RoCE",
    merged_nic=True,
    coll_channels=8,
    gdrdma_seen=False,
    socket_channel_seen=False,
)

MANIFEST_FILES = (
    t.ManifestFile(path="config.json", size=1234, sha256="a" * 64),
    t.ManifestFile(path="model-00001-of-00002.safetensors", size=5678, sha256="b" * 64),
    t.ManifestFile(path="tokenizer.json", size=90, sha256="c" * 64),
)
MANIFEST = t.WeightsManifest(
    repo="RedHatAI/GLM-5.3-Flash-NVFP4",
    revision=REVISION,
    generated_at=AT,
    total_bytes=1234 + 5678 + 90,
    files=MANIFEST_FILES,
)
VERIFIED = t.VerificationRecord(
    repo="RedHatAI/GLM-5.3-Flash-NVFP4",
    revision=REVISION,
    scope="all",
    node="head",
    verified_at=AT,
    file_count=3,
    total_bytes=1234 + 5678 + 90,
)

# --- 派生の重み (手元で変換した重み) --------------------------------------

ORIGIN_REPO = "RedHatAI/GLM-5.3-Flash-NVFP4"
ORIGIN_MANIFEST = "RedHatAI__GLM-5.3-Flash-NVFP4.manifest.json"
DERIVED_NAME = "k2s1"
DERIVED_MANIFEST_NAME = f"{DERIVED_NAME}.manifest.json"
DERIVED_MOUNT_AT = f"/models/{DERIVED_NAME}"
DERIVED_TOOL = "experiments/k2-quant/convert.py"
DERIVED_COMMIT = "a" * 40
DERIVED_ARGS = ("--dtype", "fp8", "--note", "日本語")
DERIVED_TARGET = r"^model\.layers\.\d+\.self_attn\..*$"
DERIVED_IDENTITY = f"derived:{DERIVED_NAME}:{ORIGIN_REPO}@{REVISION}"

CONVERSION = t.ConversionSpec(
    tool=DERIVED_TOOL,
    commit=DERIVED_COMMIT,
    args=DERIVED_ARGS,
    target_pattern=DERIVED_TARGET,
)
WEIGHTS_ORIGIN = t.WeightsOrigin(repo=ORIGIN_REPO, revision=REVISION)
ORIGIN_WEIGHTS = t.OriginWeightsRef(repo=ORIGIN_REPO, revision=REVISION, manifest=ORIGIN_MANIFEST)
DERIVATION = t.Derivation(name=DERIVED_NAME, origin=WEIGHTS_ORIGIN, conversion=CONVERSION)
DERIVED_WEIGHTS = t.DerivedWeightsRef(
    kind="derived",
    name=DERIVED_NAME,
    origin=ORIGIN_WEIGHTS,
    conversion=CONVERSION,
    manifest=DERIVED_MANIFEST_NAME,
    mount_at=DERIVED_MOUNT_AT,
)
MANIFEST_FILES_MODEL = t.ManifestFiles(total_bytes=1234 + 5678 + 90, files=MANIFEST_FILES)
DERIVED_MANIFEST = t.DerivedWeightsManifest(
    kind="derived",
    derivation=DERIVATION,
    generated_at=AT,
    total_bytes=1234 + 5678 + 90,
    files=MANIFEST_FILES,
)
DERIVED_VERIFIED = t.DerivedVerificationRecord(
    kind="derived",
    derivation=DERIVATION,
    manifest_sha256="d" * 64,
    scope="all",
    node="head",
    verified_at=AT,
    file_count=3,
    total_bytes=1234 + 5678 + 90,
)

GPU_APP = t.GpuApp(pid=12345, process_name="python3", used_memory_mib=1024)
NODE_STATUS = t.NodeStatus(
    node="head",
    container_state="running",
    config_name="p1-nvfp4-tp2",
    kind="serve",
    image_digest=f"sha256:{DIGEST}",
    config_sha256=CONFIG_SHA,
    started_at=AT,
    gpu_apps=(GPU_APP,),
    fabric_link_up=True,
)
SERVICE_STATUS = t.ServiceStatus(
    nodes=(NODE_STATUS,),
    health_ok=True,
    served_model="glm-5-3-flash",
    max_model_len=163840,
    running_requests=0,
    waiting_requests=0,
)
LAUNCH_RECORD = t.LaunchRecord(
    config_name="p1-nvfp4-tp2",
    image_digest=f"sha256:{DIGEST}",
    weights=f"RedHatAI/GLM-5.3-Flash-NVFP4@{REVISION}",
    started_at=AT,
    plans=(CONTAINER_PLAN,),
    config_sha256=CONFIG_SHA,
    repo_commit="0966d11",
    repo_dirty=False,
)
START_OUTCOME = t.StartOutcome(
    status="ready",
    detail="受け付けを始めた",
    gates=(GATE_REFUSED.model_copy(update={"passed": True, "detail": "空いている"}),),
    service=SERVICE_STATUS,
    observation=LAUNCH_OBS,
)
STOP_OUTCOME = t.StopOutcome(status="stopped", detail="2 台とも止めて消した")
SMOKE_REPLY = t.SmokeReply(
    lang="ja",
    http_status=200,
    stop_reason="end_turn",
    input_tokens=20,
    output_tokens=64,
    replacement_char=False,
)
SMOKE_OUTCOME = t.SmokeOutcome(replies=(SMOKE_REPLY,))
PROBE_OUTCOME = t.ProbeOutcome(
    status="failed",
    config_name="probe-pinned",
    observation=LAUNCH_OBS.model_copy(
        update={"known_failure": t.KnownFailure.PE_DIM_ASSERT, "failure_excerpt": ("pe_dim",)}
    ),
    detail="pe_dim must be 64 で終了した",
)

INTERFACE = t.InterfaceLink(
    name="enp1s0f0np0",
    state="UP",
    mtu=9000,
    speed_mbps=200000,
    addrs=("192.168.100.1/24",),
    roce_device="rocep1s0f0",
)
LINK_REPORT = t.LinkReport(
    node="head",
    interfaces=(INTERFACE,),
    cable_count=1,
    tools_missing=("ibv_devinfo",),
    detail="ibv_devinfo は入っていない (入れずに記録する)",
)
BANDWIDTH_SAMPLE = t.BandwidthSample(size_bytes=1073741824, algbw_gbps=180.2, busbw_gbps=180.2)
BANDWIDTH_RUN = t.BandwidthRun(
    arm="baseline",
    repeat_index=1,
    samples=(BANDWIDTH_SAMPLE,),
    nccl=NCCL_OBS,
    log_dir=Path("serving/var/20260921T030000Z-netcheck-bandwidth-p1-job"),
)
AB_OUTCOME = t.AbOutcome(
    added_env={"NCCL_SOCKET_IFNAME": "enp1s0f0np0"},
    compared_size_bytes=1073741824,
    baseline_runs=(BANDWIDTH_RUN,),
    candidate_runs=(BANDWIDTH_RUN.model_copy(update={"arm": "candidate"}),),
    adopt=False,
    detail="範囲が重なるので採用できない",
)

WATCH_SAMPLE = t.WatchSample(
    taken_at_utc=AT,
    health_ok=True,
    generation_tokens_total=12345.0,
    running_requests=2,
    waiting_requests=0,
    gpu_utilization_pct={"head": 98, "worker": None},
)
WATCH_EVENT = t.WatchEvent(
    finding="stalled",
    at_utc=AT,
    context=(WATCH_SAMPLE,),
    detail="処理中の要求があるのに、生成のトークンの数が 5 分増えない",
)
WATCH_OUTCOME = t.WatchOutcome(
    config_name="p1-nvfp4-tp2",
    started_at=AT,
    finished_at=AT,
    samples_path=Path("serving/var/20260921T030000Z-watch-p1-nvfp4-tp2/samples.jsonl"),
    sample_count=240,
    events=(WATCH_EVENT,),
    gpu_utilization_available=True,
)

THINKING_TRIAL = t.ThinkingTrial(
    variant="output_config_low",
    trial_index=1,
    http_status=200,
    thinking_chars=120,
    input_tokens=30,
    output_tokens=64,
    stop_reason="end_turn",
)
THINKING_OUTCOME = t.ThinkingOutcome(
    model="glm-5-3-flash",
    trials=(THINKING_TRIAL,),
    effective=None,
    detail="3 回の範囲が重なるので、効いたとは言えない",
)

EXAMPLES: dict[str, BaseModel] = {
    "Provenance": PROVENANCE,
    "Setting": SETTING_FLAG,
    "ImageRef": IMAGE,
    "WeightsRef": WEIGHTS,
    "ConfigDef": CONFIG,
    "NodeDef": NODE_WORKER,
    "CommandResult": COMMAND_RESULT,
    "PlannedRun": PLANNED_RUN,
    "PlannedPush": PLANNED_PUSH,
    "ApprovedPlan": APPROVED_PLAN,
    "ContainerPlan": CONTAINER_PLAN,
    "LaunchObservation": LAUNCH_OBS,
    "NcclObservation": NCCL_OBS,
    "GateResult": GATE_REFUSED,
    "ManifestFile": MANIFEST_FILES[0],
    "WeightsManifest": MANIFEST,
    "VerificationRecord": VERIFIED,
    "WeightsOrigin": WEIGHTS_ORIGIN,
    "OriginWeightsRef": ORIGIN_WEIGHTS,
    "ConversionSpec": CONVERSION,
    "Derivation": DERIVATION,
    "DerivedWeightsRef": DERIVED_WEIGHTS,
    "ManifestFiles": MANIFEST_FILES_MODEL,
    "DerivedWeightsManifest": DERIVED_MANIFEST,
    "DerivedVerificationRecord": DERIVED_VERIFIED,
    "GpuApp": GPU_APP,
    "NodeStatus": NODE_STATUS,
    "ServiceStatus": SERVICE_STATUS,
    "LaunchRecord": LAUNCH_RECORD,
    "StartOutcome": START_OUTCOME,
    "StopOutcome": STOP_OUTCOME,
    "SmokeReply": SMOKE_REPLY,
    "SmokeOutcome": SMOKE_OUTCOME,
    "ProbeOutcome": PROBE_OUTCOME,
    "InterfaceLink": INTERFACE,
    "LinkReport": LINK_REPORT,
    "BandwidthSample": BANDWIDTH_SAMPLE,
    "BandwidthRun": BANDWIDTH_RUN,
    "AbOutcome": AB_OUTCOME,
    "WatchSample": WATCH_SAMPLE,
    "WatchEvent": WATCH_EVENT,
    "WatchOutcome": WATCH_OUTCOME,
    "ThinkingTrial": THINKING_TRIAL,
    "ThinkingOutcome": THINKING_OUTCOME,
}


def _public_models() -> dict[str, type[BaseModel]]:
    """`types.py` が定義する、公開の pydantic の型をすべて集める。"""
    found: dict[str, type[BaseModel]] = {}
    for name, obj in vars(t).items():
        if name.startswith("_") or not isinstance(obj, type):
            continue
        if issubclass(obj, BaseModel) and obj.__module__ == t.__name__:
            found[name] = obj
    return found


# --- 見本と、書いたあとに変えないこと -----------------------------------


def test_every_public_model_has_an_example() -> None:
    """設計が挙げる型を漏らさないように、公開の型と見本の顔ぶれを一致させる。"""
    assert set(_public_models()) == set(EXAMPLES)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_model_round_trips_through_json(name: str) -> None:
    model = EXAMPLES[name]
    assert type(model).model_validate_json(model.model_dump_json()) == model


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_model_round_trips_through_python_objects(name: str) -> None:
    model = EXAMPLES[name]
    assert type(model).model_validate(model.model_dump()) == model


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_model_is_frozen(name: str) -> None:
    """書いたあとに項目を変えられない (design.md 「types / config」)。"""
    model = EXAMPLES[name]
    field = next(iter(type(model).model_fields))
    with pytest.raises(ValidationError):
        setattr(model, field, None)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_model_forbids_unknown_fields(name: str) -> None:
    """知らない項目は、綴りの誤りとして断る (design.md 「types / config」)。"""
    model = EXAMPLES[name]
    payload: dict[str, Any] = model.model_dump()
    payload["no_such_field"] = 1
    with pytest.raises(ValidationError):
        type(model).model_validate(payload)


# --- 3.6 根拠 -----------------------------------------------------------


def test_provenance_accepts_a_source_with_its_quote() -> None:
    provenance = t.Provenance(source=SOURCE, quote=QUOTE)

    assert provenance.measured is None


def test_provenance_accepts_a_measured_record() -> None:
    provenance = t.Provenance(measured="docs/results/2026-09-21-netcheck-links.md")

    assert provenance.source is None


def test_provenance_rejects_both_kinds_of_evidence() -> None:
    """出典と実測の両方は断る (完了の状態: 根拠が両方ある)。"""
    with pytest.raises(ValidationError, match="根拠"):
        t.Provenance(source=SOURCE, quote=QUOTE, measured="docs/results/x.md")


def test_provenance_rejects_no_evidence_at_all() -> None:
    """どちらもない設定は断る (完了の状態: どちらもない)。"""
    with pytest.raises(ValidationError, match="根拠"):
        t.Provenance()


def test_provenance_rejects_a_source_without_a_quote() -> None:
    """出典だけで原文がないものは断る (完了の状態: 出典だけで原文がない)。"""
    with pytest.raises(ValidationError, match="quote"):
        t.Provenance(source=SOURCE)


def test_provenance_rejects_a_quote_without_a_source() -> None:
    with pytest.raises(ValidationError, match="source"):
        t.Provenance(quote=QUOTE)


def test_provenance_rejects_an_empty_quote() -> None:
    with pytest.raises(ValidationError):
        t.Provenance(source=SOURCE, quote="")


@pytest.mark.parametrize(
    "bad",
    ["/Users/j5ik2o/docs/results/x.md", "results/x.md", "docs/../secrets/x.md", "docs/"],
)
def test_provenance_rejects_a_measured_path_outside_docs(bad: str) -> None:
    """`measured` は、リポジトリの `docs/` の下の相対のパスだけ (design.md types / config)。"""
    with pytest.raises(ValidationError, match="measured"):
        t.Provenance(measured=bad)


def test_setting_inherits_the_evidence_rule() -> None:
    """設定は根拠を必ず持つ (requirements 3.7 の、読み込みの時点の断りの土台)。"""
    with pytest.raises(ValidationError, match="根拠"):
        t.Setting(flag="--port", value="8000", why="理由")


# --- 設定の形 -----------------------------------------------------------


def test_setting_without_a_flag_is_a_positional_argument() -> None:
    """`flag` を書かないと、位置の引数になる (design.md Data Models)。"""
    assert SETTING_POSITIONAL.flag is None
    assert SETTING_POSITIONAL.value == "{weights.mount_at}"


def test_setting_without_a_flag_and_without_a_value_is_rejected() -> None:
    with pytest.raises(ValidationError, match="flag"):
        t.Setting(why="理由", source=SOURCE, quote=QUOTE)


def test_setting_with_a_flag_and_no_value_is_a_bare_flag() -> None:
    bare = t.Setting(flag="--headless", why="worker は API を持たない", source=SOURCE, quote=QUOTE)

    assert bare.value is None


def test_setting_requires_a_reason() -> None:
    with pytest.raises(ValidationError, match="why"):
        t.Setting(flag="--port", value="8000", why="", source=SOURCE, quote=QUOTE)


@pytest.mark.parametrize("bad", ["", "http", "0", "65536", "8000 8001"])
def test_a_port_setting_must_carry_a_port_number(bad: str) -> None:
    """`is_port` の値は、`gate_ports_free` が見る待ち受けのポートの番号になる。"""
    with pytest.raises(ValidationError, match="is_port"):
        t.Setting(flag="--port", value=bad, why="理由", is_port=True, source=SOURCE, quote=QUOTE)


def test_a_port_setting_needs_a_value() -> None:
    with pytest.raises(ValidationError, match="is_port"):
        t.Setting(flag="--port", why="理由", is_port=True, source=SOURCE, quote=QUOTE)


# --- 3.2 イメージ、3.4 重み ----------------------------------------------


def test_image_ref_accepts_a_digest_reference() -> None:
    assert IMAGE.ref.endswith(DIGEST)


def test_image_ref_accepts_a_complete_local_image_id() -> None:
    image = t.ImageRef(
        ref=f"sha256:{DIGEST}", seen_as="local-build", size_bytes=1, source=SOURCE, quote=QUOTE
    )
    assert image.ref == f"sha256:{DIGEST}"


@pytest.mark.parametrize(
    "bad",
    [
        "vllm/vllm-openai:glm53-flash-arm64-cu130",
        "vllm/vllm-openai@sha256:b0501f99",
        f"vllm/vllm-openai@md5:{DIGEST}",
        f"vllm/vllm-openai@sha256:{DIGEST.upper()}",
        f"@sha256:{DIGEST}",
        "sha256:b0501f99",
        DIGEST,
        f"sha256:{DIGEST.upper()}",
    ],
)
def test_image_ref_rejects_anything_but_a_sha256_digest(bad: str) -> None:
    """タグだけの参照は断る (design.md types / config の検査 2)。"""
    with pytest.raises(ValidationError, match="ref"):
        t.ImageRef(ref=bad, seen_as="tag", size_bytes=1, source=SOURCE, quote=QUOTE)


@pytest.mark.parametrize("bad", ["18d55bfd", "0" * 39, "0" * 41, "g" * 40])
def test_weights_ref_requires_a_forty_digit_revision(bad: str) -> None:
    """版は 40 桁の 16 進 (design.md types / config の検査 3)。"""
    with pytest.raises(ValidationError, match="revision"):
        t.WeightsRef(
            repo="RedHatAI/GLM-5.3-Flash-NVFP4",
            revision=bad,
            manifest="x.manifest.json",
            mount_at="/models/nvfp4",
            source=SOURCE,
            quote=QUOTE,
        )


# --- 派生の重み ---------------------------------------------------------


def test_derived_weights_ref_derivation_and_identity() -> None:
    """構成に書く派生の参照が、元の参照と変換の条件から同一性を導ける。"""
    assert DERIVED_WEIGHTS.derivation == DERIVATION
    assert DERIVED_WEIGHTS.derivation.origin == WEIGHTS_ORIGIN
    assert DERIVED_WEIGHTS.identity == DERIVED_IDENTITY


def test_hub_weights_ref_manifest_and_record_identity_are_repo_at_revision() -> None:
    """Hub の重みの同一性は、いまと同じ `repo@revision` の文字列である。"""
    assert WEIGHTS.identity == f"{ORIGIN_REPO}@{REVISION}"
    assert MANIFEST.identity == f"{ORIGIN_REPO}@{REVISION}"
    assert VERIFIED.identity == f"{ORIGIN_REPO}@{REVISION}"


def test_derived_manifest_and_record_expose_the_same_derivation() -> None:
    """変換の結果のマニフェストと照合の記録が、構成と同じ派生の同一性を持つ。"""
    assert DERIVED_MANIFEST.derivation == DERIVATION
    assert DERIVED_MANIFEST.identity == DERIVED_IDENTITY
    assert DERIVED_VERIFIED.derivation == DERIVATION
    assert DERIVED_VERIFIED.identity == DERIVED_IDENTITY


def test_hub_weights_manifest_json_keys_are_unchanged() -> None:
    """Hub のマニフェストの外形 (JSON の鍵) を変えない。"""
    assert set(MANIFEST.model_dump()) == {
        "repo",
        "revision",
        "generated_at",
        "total_bytes",
        "files",
    }


def test_hub_weights_ref_and_record_json_keys_are_unchanged() -> None:
    """Hub の重みの参照と照合の記録にも、派生のための項目 (`kind` など) を足さない。"""
    assert set(WEIGHTS.model_dump()) == {
        "repo",
        "revision",
        "manifest",
        "mount_at",
        "source",
        "quote",
        "measured",
    }
    assert set(VERIFIED.model_dump()) == {
        "repo",
        "revision",
        "scope",
        "node",
        "verified_at",
        "file_count",
        "total_bytes",
        "mismatched",
    }


def test_derived_manifest_keeps_the_same_file_checks_as_the_hub_manifest() -> None:
    """派生のマニフェストも、並び、合計、probe_files の決まりを同じように受ける。"""
    assert [entry.path for entry in DERIVED_MANIFEST.probe_files] == [
        "config.json",
        "tokenizer.json",
    ]
    payload = DERIVED_MANIFEST.model_dump()
    payload["total_bytes"] = 1
    with pytest.raises(ValidationError, match="total_bytes"):
        t.DerivedWeightsManifest.model_validate(payload)


@pytest.mark.parametrize(
    "bad",
    [
        "RedHatAI__GLM-5.3-Flash-NVFP4",
        "k2/s1",
        "-k2",
        "",
        "k2 s1",
        "K2.S1",
        "k2s1_",
    ],
)
def test_derived_name_is_a_bare_alphanumeric_hyphen_name(bad: str) -> None:
    """派生の名前は、英数字とハイフンだけにする (Hub の slug と衝突する名前を作らない)。"""
    payload = DERIVED_WEIGHTS.model_dump()
    payload["name"] = bad
    with pytest.raises(ValidationError, match="name"):
        t.DerivedWeightsRef.model_validate(payload)


@pytest.mark.parametrize("bad", ["k2s1", "0" * 39, "0" * 41, "g" * 40, ""])
def test_derived_conversion_commit_is_forty_hex_digits(bad: str) -> None:
    """変換に使った道具のコミットは、40 桁の 16 進にする。"""
    payload = DERIVED_WEIGHTS.model_dump()
    payload["conversion"]["commit"] = bad
    with pytest.raises(ValidationError, match="commit"):
        t.DerivedWeightsRef.model_validate(payload)


@pytest.mark.parametrize("bad", ["18d55bfd", "0" * 39, "0" * 41])
def test_derived_origin_revision_is_forty_hex_digits(bad: str) -> None:
    """元の重みの版も、40 桁の 16 進にする。"""
    payload = DERIVED_WEIGHTS.model_dump()
    payload["origin"]["revision"] = bad
    with pytest.raises(ValidationError, match="revision"):
        t.DerivedWeightsRef.model_validate(payload)


@pytest.mark.parametrize("bad", ["/abs/convert.py", "../convert.py", "a/../b.py", "a/..", ""])
def test_derived_conversion_tool_is_a_repo_relative_path(bad: str) -> None:
    """変換の道具の道筋は、先頭 `/` も `..` も受けない。"""
    payload = DERIVED_WEIGHTS.model_dump()
    payload["conversion"]["tool"] = bad
    with pytest.raises(ValidationError, match="tool"):
        t.DerivedWeightsRef.model_validate(payload)


def test_derived_target_pattern_is_compiled_at_the_boundary() -> None:
    """対象の正規表現は、読んだ時点でコンパイルできるものだけを通す。"""
    payload = DERIVED_WEIGHTS.model_dump()
    payload["conversion"]["target_pattern"] = "("
    with pytest.raises(ValidationError, match="target_pattern"):
        t.DerivedWeightsRef.model_validate(payload)


@pytest.mark.parametrize("bad_kind", [None, "hub", "", "Derived"])
def test_derived_ref_kind_must_be_the_explicit_derived_literal(bad_kind: str | None) -> None:
    """派生の参照は、`kind = "derived"` を明示したものだけを受ける。"""
    payload = DERIVED_WEIGHTS.model_dump()
    if bad_kind is None:
        payload.pop("kind")
    else:
        payload["kind"] = bad_kind
    with pytest.raises(ValidationError, match="kind"):
        t.DerivedWeightsRef.model_validate(payload)


def test_derived_manifest_kind_must_be_the_explicit_derived_literal() -> None:
    payload = DERIVED_MANIFEST.model_dump()
    payload["kind"] = None
    with pytest.raises(ValidationError, match="kind"):
        t.DerivedWeightsManifest.model_validate(payload)


def test_manifest_content_sha256_is_a_sha256_hex_digest() -> None:
    """マニフェストの中身の SHA-256 は、64 桁の小文字の 16 進である。"""
    assert re.fullmatch(r"[0-9a-f]{64}", DERIVED_MANIFEST.content_sha256)


def test_equal_manifests_have_the_same_content_sha256() -> None:
    """同じ中身のマニフェストは、JSON を往復しても同じ SHA-256 になる。"""
    restored = t.DerivedWeightsManifest.model_validate_json(DERIVED_MANIFEST.model_dump_json())

    assert restored == DERIVED_MANIFEST
    assert restored.content_sha256 == DERIVED_MANIFEST.content_sha256


def test_manifest_content_sha256_changes_when_one_file_hash_changes() -> None:
    """1 ファイルの sha256 が違うだけのマニフェストは、別の SHA-256 になる。

    件数・大きさ・`derivation` が同じでも、差し替えたマニフェストを見分けられること。
    """
    first = DERIVED_MANIFEST.files[0]
    swapped_files = (
        first.model_copy(update={"sha256": "f" * 64}),
        *DERIVED_MANIFEST.files[1:],
    )
    swapped = DERIVED_MANIFEST.model_copy(update={"files": swapped_files})

    assert swapped.total_bytes == DERIVED_MANIFEST.total_bytes
    assert swapped.derivation == DERIVED_MANIFEST.derivation
    assert swapped.content_sha256 != DERIVED_MANIFEST.content_sha256


def test_derived_verification_record_requires_the_manifest_sha256() -> None:
    """`manifest_sha256` のない記録 (結び付けのない古い形) は、型として断る。"""
    payload = DERIVED_VERIFIED.model_dump()
    del payload["manifest_sha256"]

    with pytest.raises(ValidationError, match="manifest_sha256"):
        t.DerivedVerificationRecord.model_validate(payload)


@pytest.mark.parametrize("bad", ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64])
def test_derived_verification_record_manifest_sha256_is_sixty_four_lowercase_hex(bad: str) -> None:
    payload = DERIVED_VERIFIED.model_dump()
    payload["manifest_sha256"] = bad

    with pytest.raises(ValidationError, match="manifest_sha256"):
        t.DerivedVerificationRecord.model_validate(payload)


def test_derived_weights_are_refused_for_a_probe_config_with_the_reason() -> None:
    """派生の重みは、`kind = "probe"` の構成に結び付けられない。

    理由 (`serve fetch --probe-files` の道がなく、縮小の確認の置き場所を作れない) を、
    型の誤りの文で示す。
    """
    payload = CONFIG.model_dump()
    payload["kind"] = "probe"
    payload["weights"] = DERIVED_WEIGHTS.model_dump()

    with pytest.raises(ValidationError) as caught:
        t.ConfigDef.model_validate(payload)

    message = str(caught.value)
    assert "派生の重みを使えない" in message
    assert "--probe-files" in message


def test_derived_weights_are_accepted_for_a_serve_config() -> None:
    """派生の重みは、`kind = "serve"` の構成には結び付けられる。"""
    payload = CONFIG.model_dump()
    payload["weights"] = DERIVED_WEIGHTS.model_dump()

    config = t.ConfigDef.model_validate(payload)

    assert config.kind == "serve"
    assert config.weights == DERIVED_WEIGHTS


def test_hub_weights_are_still_accepted_for_a_probe_config() -> None:
    """Hub の重みは、`kind = "probe"` の構成に、いまと同じように結び付けられる。"""
    payload = CONFIG.model_dump()
    payload["kind"] = "probe"

    config = t.ConfigDef.model_validate(payload)

    assert config.kind == "probe"
    assert config.weights == WEIGHTS


# --- 3.1 構成 -----------------------------------------------------------


def test_config_kind_has_exactly_the_five_kinds() -> None:
    assert set(get_args(t.ConfigKind)) == {"serve", "probe", "job", "fetch", "inspect"}


def test_config_args_keep_the_order_they_were_written_in() -> None:
    """`args` は、TOML に書いた順に並ぶ (design.md types / config)。"""
    order = ["model-path", "port", "node-rank"]

    assert list(CONFIG.args) == order
    assert list(t.ConfigDef.model_validate_json(CONFIG.model_dump_json()).args) == order


def test_config_def_defaults_allow_speculative_to_false() -> None:
    """投機的デコードの許可は、書かなければ偽である (C2、design 6.7)。"""
    assert CONFIG.allow_speculative is False
    assert CONFIG.model_dump()["allow_speculative"] is False


def test_config_def_accepts_allow_speculative_true() -> None:
    """`allow_speculative = true` の構成は、型として読める (C2)。"""
    payload = CONFIG.model_dump()
    payload["allow_speculative"] = True

    assert t.ConfigDef.model_validate(payload).allow_speculative is True


@pytest.mark.parametrize("bad", ["true", 1, "yes"])
def test_config_def_refuses_a_non_boolean_allow_speculative(bad: object) -> None:
    """真偽値以外は、型として断る (C2。文字列の "true" を真と読み替えない)。"""
    payload = CONFIG.model_dump()
    payload["allow_speculative"] = bad

    with pytest.raises(ValidationError, match="allow_speculative"):
        t.ConfigDef.model_validate(payload)


@pytest.mark.parametrize("kind", ["serve", "probe"])
def test_serve_and_probe_require_a_served_model_name(kind: str) -> None:
    """名乗るモデルの名前は、`serve` と `probe` で必須 (検査 4、requirements 6.2)。"""
    payload = CONFIG.model_dump()
    payload["kind"] = kind
    payload["served_model_name"] = None
    with pytest.raises(ValidationError, match="served_model_name"):
        t.ConfigDef.model_validate(payload)


@pytest.mark.parametrize("kind", ["job", "fetch", "inspect"])
def test_other_kinds_must_not_carry_a_served_model_name(kind: str) -> None:
    payload = CONFIG.model_dump()
    payload["kind"] = kind
    with pytest.raises(ValidationError, match="served_model_name"):
        t.ConfigDef.model_validate(payload)


@pytest.mark.parametrize("bad", ["RedHatAI/GLM-5.3-Flash-NVFP4", "glm 5 3", "a,b", ""])
def test_served_model_name_must_be_one_name_without_a_slash(bad: str) -> None:
    payload = CONFIG.model_dump()
    payload["served_model_name"] = bad
    with pytest.raises(ValidationError, match="served_model_name"):
        t.ConfigDef.model_validate(payload)


def test_config_nodes_must_be_non_empty_and_without_duplicates() -> None:
    payload = CONFIG.model_dump()
    payload["nodes"] = []
    with pytest.raises(ValidationError, match="nodes"):
        t.ConfigDef.model_validate(payload)
    payload["nodes"] = ["head", "head"]
    with pytest.raises(ValidationError, match="nodes"):
        t.ConfigDef.model_validate(payload)


def test_config_without_weights_and_env_loads() -> None:
    """`job` と `inspect` は重みを持たず、`env` を書かない構成もある (design.md Data Models)。"""
    config = t.ConfigDef(
        name="p1-image-licenses",
        kind="inspect",
        description="イメージの中のライセンスの表記を読む",
        nodes=("head",),
        image=IMAGE,
        docker={"entrypoint": SETTING_FLAG},
        args={"license-path": SETTING_POSITIONAL},
        ready_timeout_s=120,
    )

    assert config.weights is None
    assert config.env == {}
    assert config.served_model_name is None


# --- ノードの定義 -------------------------------------------------------


def test_node_def_keeps_the_fabric_values_empty_until_they_are_measured() -> None:
    """直結の値は、実測まで空のまま読み込める (design.md Data Models、検査 7)。"""
    assert NODE_HEAD.fabric_addr is None
    assert NODE_HEAD.fabric_ifname is None
    assert NODE_HEAD.fabric_measured is None


@pytest.mark.parametrize("bad", ["home/j5ik2o/vllm-baseline", "~/vllm-baseline", "/home/j5ik2o/"])
def test_remote_root_must_be_an_absolute_path_without_a_trailing_slash(bad: str) -> None:
    with pytest.raises(ValidationError, match="remote_root"):
        t.NodeDef(
            role="head", ssh_host="spark-153d", lan_addr=IPv4Address("10.0.1.60"), remote_root=bad
        )


def test_fabric_measured_must_point_under_docs() -> None:
    with pytest.raises(ValidationError, match="fabric_measured"):
        t.NodeDef(
            role="worker",
            ssh_host="spark-5083",
            lan_addr=IPv4Address("10.0.1.61"),
            remote_root="/home/j5ik2o/vllm-baseline",
            fabric_addr=IPv4Address("192.168.100.2"),
            fabric_ifname="enp1s0f0np1",
            fabric_measured="/tmp/netcheck.md",
        )


# --- 了承を得た計画 -----------------------------------------------------


def test_rollback_may_only_target_the_containers_the_plan_starts() -> None:
    """巻き戻しに、ほかの名前を入れられない (design.md remote)。"""
    with pytest.raises(ValidationError, match="own_container_names"):
        t.ApprovedPlan(
            forward=(PLANNED_RUN,),
            rollback=(
                t.PlannedRun(
                    node="head",
                    argv=("docker", "stop", "exl3-tp2"),
                    container="exl3-tp2",
                    purpose="よその構成を止める",
                ),
            ),
            own_container_names=(HEAD_CONTAINER,),
        )


def test_rollback_commands_must_name_their_container() -> None:
    with pytest.raises(ValidationError, match="container"):
        t.ApprovedPlan(
            forward=(PLANNED_RUN,),
            rollback=(
                t.PlannedRun(
                    node="head", argv=("docker", "stop", HEAD_CONTAINER), purpose="止める"
                ),
            ),
            own_container_names=(HEAD_CONTAINER,),
        )


def test_rollback_commands_must_be_docker_stop_or_rm() -> None:
    with pytest.raises(ValidationError, match="rollback"):
        t.ApprovedPlan(
            forward=(PLANNED_RUN,),
            rollback=(
                t.PlannedRun(
                    node="head",
                    argv=("docker", "logs", HEAD_CONTAINER),
                    container=HEAD_CONTAINER,
                    purpose="記録を読む",
                ),
            ),
            own_container_names=(HEAD_CONTAINER,),
        )


def test_a_plan_without_containers_needs_no_rollback() -> None:
    """`serve push` の計画は、置き場所を作るだけで、巻き戻す名前を持たない。"""
    plan = t.ApprovedPlan(
        forward=(
            t.PlannedRun(
                node="head",
                argv=("mkdir", "-p", "/home/j5ik2o/vllm-baseline/payload"),
                purpose="置き場所を作る",
            ),
            PLANNED_PUSH,
        ),
    )

    assert plan.rollback == ()
    assert plan.own_container_names == ()


def test_a_plan_needs_at_least_one_forward_command() -> None:
    with pytest.raises(ValidationError, match="forward"):
        t.ApprovedPlan(forward=())


# --- 組み立てたコンテナの計画 -------------------------------------------


@pytest.mark.parametrize("bad", ["vb_p1_head", "vb/p1/head", "", "-vb-head", "vb-p1-head!"])
def test_container_name_is_alphanumeric_and_hyphens(bad: str) -> None:
    with pytest.raises(ValidationError, match="container_name"):
        t.ContainerPlan(node="head", container_name=bad, labels={}, argv=("docker", "run"))


def test_container_plan_needs_an_argv() -> None:
    with pytest.raises(ValidationError, match="argv"):
        t.ContainerPlan(node="head", container_name=HEAD_CONTAINER, labels={}, argv=())


# --- 記録からの読み取り -------------------------------------------------


def test_launch_observation_defaults_to_nothing_read() -> None:
    """読めなかった項目は、断らずに空のまま返す (design.md observe)。"""
    empty = t.LaunchObservation()

    assert empty.vllm_version is None
    assert empty.attention_backend is None
    assert empty.attention_candidates == ()
    assert empty.kv_cache_tokens is None
    assert empty.known_failure is None
    assert empty.failure_excerpt == ()
    assert empty.speculative_config_seen is False


def test_failure_excerpt_is_capped_at_forty_lines() -> None:
    with pytest.raises(ValidationError, match="failure_excerpt"):
        t.LaunchObservation(failure_excerpt=tuple(f"line {i}" for i in range(41)))


def test_known_failure_has_exactly_the_seven_kinds() -> None:
    assert [member.value for member in t.KnownFailure] == [
        "pe_dim_assert",
        "no_attention_backend",
        "no_kernel_image",
        "kpool_block_size",
        "startup_memory_check",
        "deep_gemm_missing",
        "unclassified",
    ]


def test_nccl_observation_defaults_to_nothing_read() -> None:
    empty = t.NcclObservation()

    assert empty.network is None
    assert empty.ib_no_device is False
    assert empty.merged_nic is False
    assert empty.socket_channel_seen is False


def test_nccl_network_is_either_ib_or_socket() -> None:
    with pytest.raises(ValidationError, match="network"):
        t.NcclObservation.model_validate({"network": "RoCE"})


# --- マニフェストと照合 -------------------------------------------------


def test_manifest_files_must_be_sorted_by_path() -> None:
    payload = MANIFEST.model_dump()
    payload["files"] = list(reversed(payload["files"]))
    with pytest.raises(ValidationError, match="files"):
        t.WeightsManifest.model_validate(payload)


def test_manifest_rejects_a_duplicated_path() -> None:
    payload = MANIFEST.model_dump()
    payload["files"] = [payload["files"][0], payload["files"][0]]
    payload["total_bytes"] = 1234 * 2
    with pytest.raises(ValidationError, match="files"):
        t.WeightsManifest.model_validate(payload)


def test_manifest_total_must_match_the_sum_of_the_sizes() -> None:
    payload = MANIFEST.model_dump()
    payload["total_bytes"] = 1
    with pytest.raises(ValidationError, match="total_bytes"):
        t.WeightsManifest.model_validate(payload)


@pytest.mark.parametrize("excluded", ["README.md", ".gitattributes"])
def test_manifest_never_lists_the_model_card_or_gitattributes(excluded: str) -> None:
    """モデルカードと `.gitattributes` は載せない (design.md Data Models、8.8)。"""
    payload = MANIFEST.model_dump()
    payload["files"] = [
        {"path": excluded, "size": 10, "sha256": "d" * 64},
        *payload["files"],
    ]
    payload["total_bytes"] = 1234 + 5678 + 90 + 10
    with pytest.raises(ValidationError, match="files"):
        t.WeightsManifest.model_validate(payload)


def test_manifest_probe_files_leave_the_safetensors_out() -> None:
    """縮小の確認は、設定とトークナイザだけを見る (design.md guards)。"""
    assert [entry.path for entry in MANIFEST.probe_files] == ["config.json", "tokenizer.json"]


def test_verification_record_is_ok_only_without_mismatches() -> None:
    assert VERIFIED.ok is True
    assert VERIFIED.model_copy(update={"mismatched": ("config.json",)}).ok is False


def test_verification_scope_is_all_or_probe_files() -> None:
    assert set(get_args(t.VerificationScope)) == {"all", "probe_files"}


# --- 状態と結果 ---------------------------------------------------------


def test_container_state_has_the_three_states() -> None:
    assert set(get_args(t.ContainerState)) == {"absent", "running", "exited"}


def test_node_status_carries_the_labels_the_already_running_check_compares() -> None:
    """`already_running` は、構成の名前、ダイジェスト、`config-sha256` で判定する。"""
    assert NODE_STATUS.config_name == "p1-nvfp4-tp2"
    assert NODE_STATUS.image_digest == f"sha256:{DIGEST}"
    assert NODE_STATUS.config_sha256 == CONFIG_SHA


def test_service_status_needs_at_least_one_node() -> None:
    with pytest.raises(ValidationError, match="nodes"):
        t.ServiceStatus(nodes=())


def test_start_status_has_the_four_results() -> None:
    assert set(get_args(t.StartStatus)) == {"ready", "already_running", "refused", "failed"}


def test_stop_status_has_the_three_results() -> None:
    assert set(get_args(t.StopStatus)) == {"stopped", "already_stopped", "gpu_not_released"}


def test_probe_status_has_the_three_results() -> None:
    assert set(get_args(t.ProbeStatus)) == {"ready", "failed", "inconclusive"}


def test_watch_finding_has_the_two_findings() -> None:
    assert set(get_args(t.WatchFinding)) == {"unresponsive", "stalled"}


def test_thinking_variant_has_the_five_ways() -> None:
    assert set(get_args(t.ThinkingVariant)) == {
        "none",
        "output_config_low",
        "chat_template_low",
        "output_config_medium",
        "clear_thinking",
    }


def test_watch_outcome_rejects_an_inverted_range() -> None:
    payload = WATCH_OUTCOME.model_dump()
    payload["finished_at"] = datetime(2026, 9, 20, 3, 0, 0, tzinfo=UTC)
    with pytest.raises(ValidationError, match="finished_at"):
        t.WatchOutcome.model_validate(payload)


def test_smoke_outcome_needs_at_least_one_reply() -> None:
    with pytest.raises(ValidationError, match="replies"):
        t.SmokeOutcome(replies=())


def test_config_schema_version_starts_at_one() -> None:
    assert t.CONFIG_SCHEMA_VERSION == 1


# --- 10.5 本文を残さない ------------------------------------------------

BODY_CARRYING_TOKENS = frozenset(
    {"text", "body", "content", "contents", "prompt", "response", "messages", "system"}
)
BODY_CARRYING_NAMES = frozenset({"input", "output", "blocks", "thinking"})
RESULT_MODELS = ("SmokeOutcome", "ProbeOutcome", "ThinkingOutcome", "WatchOutcome", "LaunchRecord")


class _BodyCarrier(BaseModel):
    """対照の型。スキーマの歩き方が空振りでないことを確かめるためだけに置く。"""

    prompt: str
    blocks: list[str]


def _collect_property_names(schema: Any, props: set[str]) -> None:
    """スキーマを再帰で歩き、項目の名前を集める。"""
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                props.update(str(name) for name in value)
            _collect_property_names(value, props)
    elif isinstance(schema, list):
        for item in schema:
            _collect_property_names(item, props)


@pytest.mark.parametrize("name", RESULT_MODELS)
def test_result_models_have_no_place_for_a_request_or_response_body(name: str) -> None:
    props: set[str] = set()
    _collect_property_names(_public_models()[name].model_json_schema(), props)

    assert props, "項目を 1 つも集められていない (歩き方の誤り)"
    assert props & BODY_CARRYING_NAMES == set()
    offending = {item for item in props if set(item.split("_")) & BODY_CARRYING_TOKENS}
    assert offending == set(), f"{name} に本文らしい項目がある: {sorted(offending)}"


def test_the_schema_walk_does_flag_a_body_carrying_model() -> None:
    """対照: 本文を持つ型は、同じ検査に引っかかる (検査が空振りでないこと)。"""
    props: set[str] = set()
    _collect_property_names(_BodyCarrier.model_json_schema(), props)

    assert props & BODY_CARRYING_NAMES != set()
    assert {item for item in props if set(item.split("_")) & BODY_CARRYING_TOKENS} != set()


def test_the_reply_summary_itself_carries_no_body() -> None:
    """`ProbeOutcome.reply` が指す `SmokeReply` は、長さと数と状態だけを持つ (10.5)。"""
    assert set(t.SmokeReply.model_fields) == {
        "lang",
        "http_status",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "replacement_char",
    }


# --- 依存の向き ---------------------------------------------------------

ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "__future__",
        "datetime",
        "enum",
        "hashlib",
        "ipaddress",
        "json",
        "pathlib",
        "re",
        "typing",
        "pydantic",
    }
)
"""標準ライブラリと pydantic だけ。

`hashlib` と `json` は、マニフェストの正規のバイト列と、その SHA-256 (`ManifestFiles` の
`canonical_bytes` と `content_sha256`) を、この部品が持つために足した。
"""


def test_types_module_imports_nothing_from_serving_kit() -> None:
    """この部品は、ほかの `serving_kit` の部品を読み込まない (tasks.md 1.2)。"""
    tree = ast.parse(inspect.getsource(t))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "相対の読み込みがある"
            assert node.module is not None
            roots.add(node.module.split(".")[0])

    assert "serving_kit" not in roots
    assert roots <= ALLOWED_IMPORT_ROOTS, f"許していない読み込み: {roots - ALLOWED_IMPORT_ROOTS}"
