"""起動と、その待ちと、失敗のときの片付けの試験 (tasks.md 3.4)。

確かめること (design.md 「運転 › lifecycle」「System Flows › 起動」、tasks.md 3.4 の完了の
状態):

- **起動できる場合**: 呼び出しの順序が「一覧 → 関門 → (了承) → 起動の記録の配布 →
  `docker run` ×2 → 待ち」で、`ready` になる (requirements 1.3、1.4)
- **時間切れの場合**: `/health` が 200 にならないまま上限に達し、記録の末尾 → 回収 → 停止 →
  削除の順に片付けて `failed` になる。上限は、その回だけ上書きできる (requirements 1.5)
- **片方のコンテナが終了する場合**: 時間切れを待たずに `failed` になり、失敗の種類が読まれる
- 名乗る名前が構成と違う、投機的デコードの指標が出ている、指標が読めないまま上限
  (requirements 6.2、6.7)
- すでに動いていて 2 台とも一致 → `already_running` で、状態を変える呼び出しが 1 つも出ない。
  名前が同じで中身が違う → 違う項目を示して `refused` (requirements 1.8、3.10)
- 関門で断られる → `refused` で、了承も出ない。了承しない → 状態を変える呼び出しが出ない
  (requirements 2.1)
- 名前の衝突 → 巻き戻しが、よその名前に出ない。もう片方の台の自分のコンテナは片付く
  (requirements 2.3)
- 中断 → 末尾 → 回収 → 停止 → 削除のあと、中断が伝わる。**片付けの最中の中断でも、`rm` まで
  試みて、中断が伝わる** (終了コード 130)
- 偽の実行役に記録された、コンテナを対象にする操作のすべてが、自分の一覧の ID か、了承済みの
  計画の名前だけを対象にしている (requirements 2.3、2.4)
- 了承のあとの、状態を変える呼び出しが、見せた計画の列と完全に一致する

実物の ssh、rsync、docker、推論サーバーには、どの段でもつながない (`FakeRunner` と、
`fake_vllm` の偽の推論サーバーを相手にする)。試験は、実際に眠らない (待つ間隔と時計は、
引数で差し替える)。
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
from fake_vllm import FakeVllm, Fault, MetricsSample
from serving_kit import lifecycle as lc
from serving_kit import types as kt
from serving_kit.config import ConfigError
from serving_kit.guards import ApprovalError, build_approved_plan, remove_argv, stop_argv
from serving_kit.logs import DEFAULT_TAIL_LINES
from serving_kit.plan import (
    LABEL_CONFIG_SHA256,
    LABEL_IMAGE,
    LABEL_OWNER,
    LABEL_STARTED_AT,
    OWNER,
    OWNER_FILTER,
    build_plans,
)
from serving_kit.types import (
    ConfigDef,
    ContainerPlan,
    ImageRef,
    KnownFailure,
    LaunchRecord,
    ManifestFile,
    NodeDef,
    NodeRole,
    Setting,
    StartOutcome,
    VerificationRecord,
    WeightsManifest,
    WeightsRef,
)

# --- 見本の値 -------------------------------------------------------------

ROLES: tuple[NodeRole, ...] = ("head", "worker")

DIGEST = "b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
IMAGE_REF = f"vllm/vllm-openai@sha256:{DIGEST}"
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
VERIFIED_AT = datetime(2026, 9, 22, 1, 30, 0, tzinfo=UTC)
GENERATED_AT = datetime(2026, 9, 22, 1, 0, 0, tzinfo=UTC)

REPO_COMMIT = "23ee8da" + "0" * 33
SOURCE = HttpUrl("https://docs.vllm.ai/en/latest/cli/serve/")
QUOTE = "vllm serve [model_tag] [options]"

CONTAINER_IDS: Mapping[NodeRole, str] = {"head": "0123456789ab", "worker": "cdef01234567"}

FILE_SIZES: Mapping[str, int] = {"config.json": 100, "model-00001-of-00001.safetensors": 4096}

MANIFEST = WeightsManifest(
    repo=REPO,
    revision=REVISION,
    generated_at=GENERATED_AT,
    total_bytes=sum(FILE_SIZES.values()),
    files=tuple(
        ManifestFile(path=path, size=size, sha256=f"{index:064x}")
        for index, (path, size) in enumerate(sorted(FILE_SIZES.items()))
    ),
)


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


def serve_config(
    port: int,
    *,
    ready_timeout_s: int = 1800,
    served_model_name: str = SERVED_MODEL,
) -> ConfigDef:
    """推論サーバーの構成 (design.md Data Models の `p1-nvfp4-tp2` を縮めたもの)。

    `--port` と `--master-port` の両方に待ち受けの印を付ける (HTTP のポートが、印だけでは
    選べないことを、試験で固定するため)。
    """
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
            "node-rank": _setting("--node-rank", "{node.rank}"),
        },
        env={},
        ready_timeout_s=ready_timeout_s,
        served_model_name=served_model_name,
    )


def plans_of(config: ConfigDef) -> tuple[ContainerPlan, ...]:
    return build_plans(config, NODES, STARTED_AT)


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

READY_LOG = (
    "INFO Using FLASH_ATTN_MLA attention backend out of potential backends: ['FLASH_ATTN_MLA']\n"
    "INFO GPU KV cache size: 1,234,567 tokens,"
    " Maximum concurrency for 163,840 tokens per request: 7.54x\n"
    "INFO init engine (profile, create kv cache, warmup model) took 42.00 seconds\n"
)
"""起動できたときの記録の見本 (`observe.observe_launch` が読む行を含む)。"""

PE_DIM_LOG = (
    "INFO Using FLASH_ATTN_MLA attention backend out of potential backends: ['FLASH_ATTN_MLA']\n"
    "ERROR AssertionError: pe_dim must be 64\n"
    "ERROR Engine core initialization failed.\n"
)
"""知っている失敗 (`pe_dim must be 64`) を含む記録の見本。"""

UNKNOWN_LOG = "ERROR something we have never seen before\n"
"""知っている失敗のどれでもない記録の見本 (`UNCLASSIFIED` になる)。"""

OBSERVE_LINES = str(lc.OBSERVE_TAIL_LINES)
"""分類に使う末尾の行数 (台本で、長い読み取りだけを選び分けるために使う)。"""

TRACEBACK_LINE = '  File "/usr/lib/python3/vllm/v1/engine/core.py", line {}, in run_engine_core'
"""vLLM の失敗のときに重なる traceback の 1 行 (関係のない行の見本)。"""


def ps_row(name: str, container_id: str, state: str, labels: Mapping[str, str]) -> str:
    """`docker ps -a --filter label=… --format json` の 1 行。

    **これは実機から採った見本ではない** (tasks.md 1.6 の「ここで採れないもの」)。項目の
    名前は `test_guards.py` / `test_weights_fetch.py` の見本に合わせた。
    """
    return (
        json.dumps(
            {
                "ID": container_id,
                "Names": name,
                "State": state,
                "Image": labels.get(LABEL_IMAGE, IMAGE_REF),
                "Labels": ",".join(f"{key}={value}" for key, value in sorted(labels.items())),
            }
        )
        + "\n"
    )


def ps_line(
    plan: ContainerPlan, *, state: str = "running", labels: Mapping[str, str] | None = None
) -> str:
    """この計画のコンテナの、一覧の 1 行 (`labels` で、ラベルを差し替えられる)。"""
    return ps_row(
        plan.container_name,
        CONTAINER_IDS[plan.node],
        state,
        plan.labels if labels is None else {**plan.labels, **labels},
    )


def verified_record(role: NodeRole) -> str:
    """`gate_weights_verified` が読む、照合の結果の記録。"""
    return VerificationRecord(
        repo=REPO,
        revision=REVISION,
        scope="all",
        node=role,
        verified_at=VERIFIED_AT,
        file_count=len(MANIFEST.files),
        total_bytes=MANIFEST.total_bytes,
    ).model_dump_json()


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


@dataclass
class StartScript:
    """`serve start` の台本 (既定: 関門が通り、2 台とも起き、片付けもできる)。"""

    plans: Sequence[ContainerPlan]
    listings: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    states: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    runs: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    stops: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    tails: Mapping[NodeRole, str] | None = None
    long_tails: Mapping[NodeRole, str] | None = None
    """`--tail <OBSERVE_TAIL_LINES>` の読み取りだけが返す記録 (実物の docker と同じく、長く
    読んだときだけ、手前の行まで返ることを表す)。書かなければ、`tails` と同じものが返る。"""

    logs: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    """`docker logs` の返事そのもの (`tails` より優先。読めない場合を作る)。"""

    extra: Sequence[Rule] = ()
    """いちばん先に見る規則 (1 つの呼び出しだけに、中断などを仕込むため)。"""

    gpu_apps: str = ""
    avail: str = "Avail\n999999999999\n"
    listening: str = ""
    digests: str = json.dumps([IMAGE_REF]) + "\n"
    layout_ok: bool = True

    def rules(self) -> tuple[Rule, ...]:
        rules: list[Rule] = list(self.extra)
        for plan in self.plans:
            role = plan.node
            listings = (
                (Reply(stdout=""), Reply(stdout=ps_line(plan)))
                if self.listings is None
                else self.listings[role]
            )
            states = (Reply(stdout="running 0\n"),) if self.states is None else self.states[role]
            tail = READY_LOG if self.tails is None else self.tails[role]
            logs = (Reply(stdout=tail),) if self.logs is None else self.logs[role]
            rules.append(Rule(prefix=OWN_CONTAINERS_ARGV, node=role, replies=listings))
            rules.append(Rule(prefix=("docker", "container", "inspect"), node=role, replies=states))
            if self.long_tails is not None:
                # 長く読んだときだけ、手前の行まで返る (実物の `docker logs --tail` と同じ)
                rules.append(
                    Rule(
                        prefix=("docker", "logs", "--timestamps", "--tail", OBSERVE_LINES),
                        node=role,
                        replies=(Reply(stdout=self.long_tails[role]),),
                    )
                )
            rules.append(Rule(prefix=("docker", "logs", "--timestamps"), node=role, replies=logs))
            rules.append(
                Rule(prefix=("cat",), node=role, replies=(Reply(stdout=verified_record(role)),))
            )
            if self.runs is not None:
                rules.append(Rule(prefix=("docker", "run"), node=role, replies=self.runs[role]))
            if self.stops is not None:
                rules.append(Rule(prefix=("docker", "stop"), node=role, replies=self.stops[role]))
        rules.extend(
            (
                Rule(prefix=("uname",), replies=(Reply(stdout="spark-153d\n"),)),
                Rule(prefix=("nvidia-smi",), replies=(Reply(stdout=self.gpu_apps),)),
                Rule(prefix=("test", "-e"), replies=(Reply(exit_code=0 if self.layout_ok else 1),)),
                Rule(prefix=("docker", "image", "inspect"), replies=(Reply(stdout=self.digests),)),
                Rule(prefix=("df",), replies=(Reply(stdout=self.avail),)),
                Rule(prefix=("ss",), replies=(Reply(stdout=self.listening),)),
                Rule(prefix=("docker", "run"), replies=(Reply(stdout="0123456789ab\n"),)),
                Rule(prefix=("docker", "stop"), replies=(Reply(),)),
                Rule(prefix=("docker", "rm"), replies=(Reply(),)),
                Rule(kind="push", replies=(Reply(),)),
                Rule(kind="pull", replies=(Reply(),)),
            )
        )
        return tuple(rules)


def var_root(tmp_path: Path) -> Path:
    return tmp_path / "var"


def record_dir(tmp_path: Path) -> Path:
    return tmp_path / "var" / "records"


def runner_of(tmp_path: Path, script: StartScript, *, default: Reply | None = None) -> FakeRunner:
    return FakeRunner(var_root=var_root(tmp_path), script=script.rules(), default=default)


def run_start(
    runner: FakeRunner,
    config: ConfigDef,
    tmp_path: Path,
    *,
    confirmer: SpyConfirmer | None = None,
    clock: FakeClock | None = None,
    report: TextIO | None = None,
    **extra: Any,
) -> StartOutcome:
    """既定の下ごしらえで `lifecycle.start` を流す助け。"""
    used_clock = FakeClock() if clock is None else clock
    return lc.start(
        runner,
        config,
        NODES,
        STARTED_AT,
        confirmer=SpyConfirmer() if confirmer is None else confirmer,
        var_root=var_root(tmp_path),
        record_dir=record_dir(tmp_path),
        repo_commit=REPO_COMMIT,
        repo_dirty=False,
        manifest=MANIFEST,
        poll_interval_s=10.0,
        sleep=used_clock.sleep,
        clock=used_clock.monotonic,
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
            names.append("logs tail" if "--tail" in argv else "logs")
        else:
            names.append(argv[1])
    return tuple(names)


def mutating_calls(runner: FakeRunner) -> tuple[RecordedCall, ...]:
    return tuple(call for call in runner.calls if call.mutating)


def argv_of(runner: FakeRunner, *prefix: str) -> tuple[tuple[str, ...], ...]:
    return tuple(argv for argv in runner.argvs if argv[: len(prefix)] == prefix)


def cleanup_order(runner: FakeRunner) -> list[str]:
    """片付けの道すじだけを取り出す (末尾 → 回収 → 停止 → 削除)。"""
    watched = ("logs tail", "logs", "pull", "stop", "rm")
    return [name for name in steps(runner) if name in watched]


GATE_STEPS = ("uname", "nvidia-smi", "test", "image inspect", "cat", "df", "ss")
"""1 台ぶんの関門の読み取り (`guards.GATE_ORDER` の順。`own_state` は、読んだ一覧を使う)。"""

CLEANUP_STEPS = [
    "logs tail",
    "logs tail",
    "logs tail",
    "logs tail",
    "logs",
    "pull",
    "pull",
    "logs",
    "pull",
    "pull",
    "stop",
    "stop",
    "rm",
    "rm",
]
"""2 台ぶんの片付けの道すじ。

「**見せる**末尾 (80 行) ×2 → **分類に使う**末尾 (2,000 行) ×2 → 回収 (記録・通信の記録・
起動の記録) → 停止 → 削除」。末尾を 2 通り読むのは、知っている失敗の行が、見せる 80 行より
手前にあっても分類できるようにするためである (親の判断、2026-09-22)。
"""

CLEANUP_STEPS_WITHOUT_CLASSIFY = CLEANUP_STEPS[2:]
"""分類の読み取りをしない経路 (中断、実行しての失敗) の片付けの道すじ。

分類の結果を使わない経路では、長い末尾を読まない (片付けを遅らせるだけ)。
"""


def tail_line_counts(runner: FakeRunner) -> list[str]:
    """`docker logs --tail <N>` の N を、流れた順に並べる。"""
    return [
        argv[argv.index("--tail") + 1]
        for argv in runner.argvs
        if argv[:2] == ("docker", "logs") and "--tail" in argv
    ]


def traceback_lines(count: int) -> str:
    """関係のない traceback だけが並ぶ記録 (見せる末尾に入る部分の見本)。"""
    return "".join(f"{TRACEBACK_LINE.format(index)}\n" for index in range(count))


def deep_log(marker: str, *, above: int = 500) -> str:
    """知っている失敗の行が、末尾から `above` 行**手前**にある記録の見本。

    vLLM の起動の失敗では、API のサーバー・EngineCore・worker の traceback が重なって数百行に
    なりうるので、肝心の行が、見せる 80 行に入らないことがある。
    """
    lines = [f"INFO booting up {index}" for index in range(20)]
    lines.append(f"ERROR AssertionError: {marker}")
    return "".join(f"{line}\n" for line in lines) + traceback_lines(above)


# --- 起動できる場合 --------------------------------------------------------


def test_ready_runs_the_steps_in_the_designed_order(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """一覧 → 関門 → 了承 → 起動の記録の配布 → `docker run` ×2 → 待ち、の順に進む。"""
    fake_vllm.set_health_fault_sequence((Fault(status=503), Fault(status=503), Fault()))
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))
    confirmer = SpyConfirmer()

    outcome = run_start(runner, config, tmp_path, confirmer=confirmer)

    assert outcome.status == "ready"
    assert len(confirmer.shown) == 1
    assert steps(runner)[:20] == (
        "ps",
        "ps",
        *GATE_STEPS,
        *GATE_STEPS,
        "push",
        "push",
        "run",
        "run",
    )
    # 待ちの間は、起こした識別子を一覧から取り、毎回、2 台のコンテナの状態を見る
    assert steps(runner)[20:24] == ("ps", "ps", "container inspect", "container inspect")


def test_ready_reads_health_models_and_metrics(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """受け付けの開始は、応答の確認・名乗る名前・指標の 3 つで判定する。"""
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))

    outcome = run_start(runner, config, tmp_path)

    assert outcome.status == "ready"
    assert fake_vllm.call_count("/health") == 1
    assert fake_vllm.call_count("/v1/models") == 1
    assert fake_vllm.call_count("/metrics") == 1
    assert outcome.service is not None
    assert outcome.service.health_ok is True
    assert outcome.service.served_model == SERVED_MODEL
    assert outcome.service.max_model_len == 4096
    assert [node.container_state for node in outcome.service.nodes] == ["running", "running"]


def test_ready_sends_http_only_to_the_head_lan_address(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """HTTP は、head の LAN のアドレスと、構成の `--port` にだけ送る。"""
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))

    run_start(runner, config, tmp_path)

    # `fake_vllm` は、ヘッダの鍵を小文字にして記録する
    hosts = {call.headers.get("host") for call in fake_vllm.requests}
    assert hosts == {f"{HEAD.lan_addr}:{port_of(fake_vllm)}"}
    assert {call.path for call in fake_vllm.requests} == {
        "/health",
        "/v1/models",
        "/metrics",
        "/version",
    }


def test_ready_reads_the_observation_from_the_launch_log(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """成功のときも、起動の記録から、選ばれた部品と KV の大きさを読む (requirements 6.3)。"""
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))

    outcome = run_start(runner, config, tmp_path)

    assert outcome.observation is not None
    assert outcome.observation.attention_backend == "FLASH_ATTN_MLA"
    assert outcome.observation.kv_cache_tokens == 1234567
    assert outcome.observation.engine_init_s == 42.0
    assert outcome.observation.known_failure is None
    assert outcome.observation.vllm_version == "0.0.0-fake"


def test_ready_does_not_collect_the_whole_record_nor_clean_up(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """成功のときは、記録の全量を回収せず、コンテナも止めない (design の System Flows)。"""
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))

    run_start(runner, config, tmp_path)

    assert runner.pulls == ()
    assert argv_of(runner, "docker", "stop") == ()
    assert argv_of(runner, "docker", "rm") == ()


def test_launch_record_is_pushed_to_state_on_both_nodes(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """起動の記録を Mac で作り、2 台の `state/` に `--delete` なしで置く (requirements 3.10)。"""
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))

    run_start(runner, config, tmp_path)

    assert [call.node for call in runner.pushes] == ["head", "worker"]
    for call in runner.pushes:
        assert call.remote == "state"
        assert call.delete is False
        assert call.local_dir is not None
        written = call.local_dir / f"{CONFIG_NAME}.launch.json"
        record = LaunchRecord.model_validate_json(written.read_text(encoding="utf-8"))
        assert record.config_name == CONFIG_NAME
        assert record.image_digest == IMAGE_REF
        assert record.weights == f"{REPO}@{REVISION}"
        assert record.repo_commit == REPO_COMMIT
        assert record.repo_dirty is False
        assert record.started_at == STARTED_AT
        assert record.config_sha256 == plans_of(config)[0].labels[LABEL_CONFIG_SHA256]
        assert len(record.plans) == len(ROLES)


# --- 派生の重み (手元で変換した重み) の構成 ----------------------------------

DERIVED_NAME = "k2s1"
DERIVED_CONFIG_NAME = "p2-nope-tp2-full-k2s1"
DERIVED_MOUNT_AT = f"/models/{DERIVED_NAME}"
DERIVED_RECORD_PATH = f"{REMOTE_ROOT}/state/{DERIVED_NAME}.derived.verified.json"
"""派生の重みの照合の記録の置き場所 (Hub の `state/<slug>.verified.json` とは別の名前)。"""


def derived_serve_config(port: int) -> tuple[ConfigDef, kt.DerivedWeightsManifest]:
    """`serve_config` の重みを派生の重みに差し替えた構成と、それと一致する派生のマニフェスト。"""
    conversion = kt.ConversionSpec(
        tool="experiments/k2-quant/convert.py",
        commit="a" * 40,
        args=("--dtype", "fp8"),
        target_pattern=r"^model\.layers\.\d+\.self_attn\..*$",
    )
    weights = kt.DerivedWeightsRef(
        kind="derived",
        name=DERIVED_NAME,
        origin=kt.OriginWeightsRef(repo=REPO, revision=REVISION, manifest=WEIGHTS.manifest),
        conversion=conversion,
        manifest=f"{DERIVED_NAME}.manifest.json",
        mount_at=DERIVED_MOUNT_AT,
    )
    mount = (
        f"type=bind,source={{remote_root}}/models/{DERIVED_NAME}"
        ",target={weights.mount_at},readonly"
    )
    config = serve_config(port).model_copy(
        update={
            "name": DERIVED_CONFIG_NAME,
            "weights": weights,
            "docker": {"models": _setting("--mount", mount)},
        }
    )
    manifest = kt.DerivedWeightsManifest(
        kind="derived",
        derivation=weights.derivation,
        generated_at=GENERATED_AT,
        total_bytes=MANIFEST.total_bytes,
        files=MANIFEST.files,
    )
    return config, manifest


def derived_verified_record(role: NodeRole, manifest: kt.DerivedWeightsManifest) -> str:
    """派生の重みの `gate_weights_verified` が読む、照合の結果の記録。"""
    return kt.DerivedVerificationRecord(
        kind="derived",
        derivation=manifest.derivation,
        manifest_sha256=manifest.content_sha256,
        scope="all",
        node=role,
        verified_at=VERIFIED_AT,
        file_count=len(manifest.files),
        total_bytes=manifest.total_bytes,
    ).model_dump_json()


def run_derived_start(
    runner: FakeRunner, config: ConfigDef, manifest: kt.DerivedWeightsManifest, tmp_path: Path
) -> StartOutcome:
    """派生の重みの構成を、その派生のマニフェストを渡して `lifecycle.start` に流す助け。"""
    used_clock = FakeClock()
    return lc.start(
        runner,
        config,
        NODES,
        STARTED_AT,
        confirmer=SpyConfirmer(),
        var_root=var_root(tmp_path),
        record_dir=record_dir(tmp_path),
        repo_commit=REPO_COMMIT,
        repo_dirty=False,
        manifest=manifest,
        poll_interval_s=10.0,
        sleep=used_clock.sleep,
        clock=used_clock.monotonic,
        report=io.StringIO(),
    )


def test_a_derived_weights_start_records_the_derived_identity(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """派生の重みの構成を起動すると、起動の記録の `weights` が派生の同一性になる。"""
    config, manifest = derived_serve_config(port_of(fake_vllm))
    script = StartScript(
        plans=plans_of(config),
        extra=tuple(
            Rule(
                prefix=("cat", DERIVED_RECORD_PATH),
                node=role,
                replies=(Reply(stdout=derived_verified_record(role, manifest)),),
            )
            for role in ROLES
        ),
    )
    runner = runner_of(tmp_path, script)

    outcome = run_derived_start(runner, config, manifest, tmp_path)

    assert outcome.status == "ready", outcome.detail
    assert [call.node for call in runner.pushes] == ["head", "worker"]
    for call in runner.pushes:
        assert call.local_dir is not None
        written = call.local_dir / f"{DERIVED_CONFIG_NAME}.launch.json"
        record = LaunchRecord.model_validate_json(written.read_text(encoding="utf-8"))
        assert record.weights == f"derived:{DERIVED_NAME}:{REPO}@{REVISION}"


def test_launch_record_source_is_emptied_before_the_push(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """配る元には、今回の 1 ファイルだけが残る (よそのファイルを Spark に配らない)。"""
    stale = record_dir(tmp_path) / "launch" / "head" / "よその記録.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))

    run_start(runner, config, tmp_path)

    assert not stale.exists()
    head_push = runner.pushes[0]
    assert head_push.local_dir is not None
    assert sorted(path.name for path in head_push.local_dir.iterdir()) == [
        f"{CONFIG_NAME}.launch.json"
    ]


def test_record_dir_must_be_absolute(tmp_path: Path) -> None:
    """配る元を空にする操作があるので、相対の道筋は、Spark に触る前に断る。"""
    config = serve_config(UNUSED_PORT)
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))

    with pytest.raises(ValueError, match="絶対の道筋"):
        lc.start(
            runner,
            config,
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            var_root=var_root(tmp_path),
            record_dir=Path("records"),
            repo_commit=REPO_COMMIT,
            repo_dirty=False,
            manifest=MANIFEST,
        )
    assert runner.calls == ()


def test_record_source_symlink_is_refused(tmp_path: Path) -> None:
    """配る元がシンボリックリンクなら、よそのファイルを配りうるので断る。"""
    elsewhere = tmp_path / "よそ"
    elsewhere.mkdir()
    link = record_dir(tmp_path) / "launch" / "head"
    link.parent.mkdir(parents=True)
    link.symlink_to(elsewhere, target_is_directory=True)
    config = serve_config(UNUSED_PORT)
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))

    with pytest.raises(lc.LifecycleError, match="シンボリックリンク"):
        run_start(runner, config, tmp_path)
    assert runner.pushes == ()
    assert argv_of(runner, "docker", "run") == ()


def test_mutating_calls_match_the_shown_plan(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """了承のあとの、状態を変える呼び出しが、見せた計画の前に進む列と完全に一致する。"""
    config = serve_config(port_of(fake_vllm))
    # `default=Reply()` で、計画にない呼び出しを隠さない (台本の穴で通ってしまわないように)
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)), default=Reply())

    run_start(runner, config, tmp_path)

    plan = runner.plan
    assert plan is not None
    shown = [
        (command.node, getattr(command, "argv", ()), getattr(command, "remote_subdir", None))
        for command in plan.forward
    ]
    happened = [(call.node, call.argv, call.remote) for call in mutating_calls(runner)]
    assert happened == shown


def test_container_operations_target_only_listed_ids_or_planned_names(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """コンテナを対象にする操作は、自分の一覧の ID か、了承済みの計画の名前だけを向く。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    plans = plans_of(config)
    script = StartScript(
        plans=plans,
        states={"head": (Reply(stdout="exited 1\n"),), "worker": (Reply(stdout="running 0\n"),)},
    )
    runner = runner_of(tmp_path, script)

    outcome = run_start(runner, config, tmp_path)

    assert outcome.status == "failed"
    allowed = {*CONTAINER_IDS.values(), *(plan.container_name for plan in plans)}
    targeting = ("logs", "stop", "rm", "container")
    for argv in runner.argvs:
        if argv[0] == "docker" and argv[1] in targeting:
            # どの形でも、対象は末尾の 1 語である (`docker container inspect --format … <ID>`、
            # `docker logs --timestamps [--tail N] <ID>`、`docker stop -t 90 <名前>`、
            # `docker rm <名前>`)
            assert argv[-1] in allowed, argv


# --- 時間切れと、片方のコンテナの終了 ---------------------------------------


def test_timeout_shows_tails_collects_then_stops_and_removes(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """`/health` が 200 にならないまま上限に達したら、末尾 → 回収 → 停止 → 削除で片付ける。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))
    clock = FakeClock()

    outcome = run_start(runner, config, tmp_path, clock=clock, timeout_s=30.0)

    assert outcome.status == "failed"
    assert "30 秒" in outcome.detail
    assert set(outcome.log_tails) == set(ROLES)
    assert clock.slept == [10.0, 10.0, 10.0]
    assert cleanup_order(runner) == CLEANUP_STEPS


def test_timeout_can_be_overridden_for_this_run_only(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """待ちの上限は、その回だけ上書きできる (構成の値は変えない)。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm), ready_timeout_s=1800)
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))
    clock = FakeClock()

    outcome = run_start(runner, config, tmp_path, clock=clock, timeout_s=20.0)

    assert outcome.status == "failed"
    assert clock.slept == [10.0, 10.0]
    assert config.ready_timeout_s == 1800


def test_collected_records_are_written_under_var_root(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """回収した記録は、`serving/var/` の下の、日時とコマンドと構成の名前の場所に入る。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))

    run_start(runner, config, tmp_path, timeout_s=10.0)

    collected = var_root(tmp_path) / f"20260922T030000Z-start-{CONFIG_NAME}"
    assert (collected / "head" / "container.stdout.log").read_text(encoding="utf-8") == READY_LOG
    assert (collected / "collect.json").exists()


def test_stop_uses_the_planned_rollback_commands(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """止める列は、必ず `guards.stop_argv` / `remove_argv` で作る (計画と完全に一致する)。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    plans = plans_of(config)
    runner = runner_of(tmp_path, StartScript(plans=plans))

    run_start(runner, config, tmp_path, timeout_s=10.0)

    assert argv_of(runner, "docker", "stop") == tuple(
        stop_argv(plan.container_name) for plan in plans
    )
    assert argv_of(runner, "docker", "rm") == tuple(
        remove_argv(plan.container_name) for plan in plans
    )


def test_exited_container_fails_before_the_timeout(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """待っている間にコンテナが終了したら、時間切れを待たずに失敗にする。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    script = StartScript(
        plans=plans_of(config),
        states={
            "head": (Reply(stdout="running 0\n"),),
            "worker": (Reply(stdout="running 0\n"), Reply(stdout="exited 1\n")),
        },
        tails={"head": READY_LOG, "worker": PE_DIM_LOG},
    )
    runner = runner_of(tmp_path, script)
    clock = FakeClock()

    outcome = run_start(runner, config, tmp_path, clock=clock, timeout_s=1800.0)

    assert outcome.status == "failed"
    assert clock.slept == [10.0]
    assert "worker" in outcome.detail
    assert outcome.observation is not None
    assert outcome.observation.known_failure is KnownFailure.PE_DIM_ASSERT
    assert any("pe_dim must be 64" in line for line in outcome.observation.failure_excerpt)
    assert cleanup_order(runner) == CLEANUP_STEPS


def test_unknown_exit_is_unclassified(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """知っている失敗のどれでもない終了は、`UNCLASSIFIED` にする (この module が決める)。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    script = StartScript(
        plans=plans_of(config),
        states={"head": (Reply(stdout="exited 3\n"),), "worker": (Reply(stdout="running 0\n"),)},
        tails={"head": UNKNOWN_LOG, "worker": UNKNOWN_LOG},
    )
    runner = runner_of(tmp_path, script)

    outcome = run_start(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert outcome.observation is not None
    assert outcome.observation.known_failure is KnownFailure.UNCLASSIFIED


def test_known_failure_above_the_shown_tail_is_still_classified(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """知っている失敗の行が、見せる 80 行より手前にあっても、種類が読める。

    見せる末尾 (requirements 1.5 の 80 行) と、分類に使う末尾 (`OBSERVE_TAIL_LINES`) を分ける。
    段 0 と段 2 の判定の要 (requirements 5.3、8.1) なので、読み落としを減らす。
    """
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    script = StartScript(
        plans=plans_of(config),
        states={
            "head": (Reply(stdout="running 0\n"), Reply(stdout="exited 1\n")),
            "worker": (Reply(stdout="running 0\n"),),
        },
        # 見せる 80 行には、関係のない traceback の行だけが入る
        tails={"head": traceback_lines(DEFAULT_TAIL_LINES), "worker": UNKNOWN_LOG},
        # 長く読んだときだけ、肝心の行 (末尾から 500 行手前) まで返る
        long_tails={"head": deep_log("pe_dim must be 64"), "worker": UNKNOWN_LOG},
    )
    runner = runner_of(tmp_path, script)

    outcome = run_start(runner, config, tmp_path, timeout_s=1800.0)

    assert outcome.status == "failed"
    assert outcome.observation is not None
    assert outcome.observation.known_failure is KnownFailure.PE_DIM_ASSERT
    assert any("pe_dim must be 64" in line for line in outcome.observation.failure_excerpt)
    # 見せる末尾は 80 行のままで、肝心の行は、そこに入っていない
    shown = outcome.log_tails["head"]
    assert len(shown.splitlines()) == DEFAULT_TAIL_LINES
    assert "pe_dim must be 64" not in shown
    # 末尾を 2 通り読む (見せるための 80 行と、分類のための長い末尾)
    assert tail_line_counts(runner) == [
        str(DEFAULT_TAIL_LINES),
        str(DEFAULT_TAIL_LINES),
        OBSERVE_LINES,
        OBSERVE_LINES,
    ]


def test_known_failure_only_on_the_worker_is_classified(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """worker の記録にだけ失敗が出ていても、種類が読める (head の記録は、きれいなまま)。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    script = StartScript(
        plans=plans_of(config),
        states={
            "head": (Reply(stdout="running 0\n"),),
            "worker": (Reply(stdout="running 0\n"), Reply(stdout="exited 1\n")),
        },
        tails={"head": READY_LOG, "worker": traceback_lines(DEFAULT_TAIL_LINES)},
        long_tails={"head": READY_LOG, "worker": deep_log("pe_dim must be 64")},
    )
    runner = runner_of(tmp_path, script)

    outcome = run_start(runner, config, tmp_path, timeout_s=1800.0)

    assert outcome.status == "failed"
    assert outcome.observation is not None
    assert outcome.observation.known_failure is KnownFailure.PE_DIM_ASSERT
    assert "pe_dim must be 64" not in outcome.log_tails["worker"]


def test_two_known_failures_prefer_the_earlier_enum_member(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """2 台の両方に失敗が出たら、`KnownFailure` の優先の順序で先のものを採る。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    script = StartScript(
        plans=plans_of(config),
        states={"head": (Reply(stdout="exited 1\n"),), "worker": (Reply(stdout="exited 1\n"),)},
        tails={
            # head は、優先の順序で後ろの種類 (no_kernel_image)
            "head": deep_log("no kernel image is available for execution on the device", above=10),
            # worker は、先の種類 (pe_dim_assert)
            "worker": deep_log("pe_dim must be 64", above=10),
        },
    )
    runner = runner_of(tmp_path, script)

    outcome = run_start(runner, config, tmp_path, timeout_s=1800.0)

    assert outcome.observation is not None
    assert outcome.observation.known_failure is KnownFailure.PE_DIM_ASSERT


def test_the_same_known_failure_on_both_nodes_prefers_head(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """同じ種類が 2 台に出たら、構成の `nodes` の順で先の台 (head) の読み取りを採る。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))

    def marked(backend: str) -> str:
        return (
            f"INFO Using {backend} attention backend out of potential backends: ['{backend}']\n"
            + deep_log("pe_dim must be 64", above=10)
        )

    script = StartScript(
        plans=plans_of(config),
        states={"head": (Reply(stdout="exited 1\n"),), "worker": (Reply(stdout="exited 1\n"),)},
        tails={"head": marked("HEAD_BACKEND"), "worker": marked("WORKER_BACKEND")},
    )
    runner = runner_of(tmp_path, script)

    outcome = run_start(runner, config, tmp_path, timeout_s=1800.0)

    assert outcome.observation is not None
    assert outcome.observation.known_failure is KnownFailure.PE_DIM_ASSERT
    assert outcome.observation.attention_backend == "HEAD_BACKEND"


def test_unreadable_records_do_not_stop_the_cleanup(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """分類のための読み取りが失敗しても、片付けは、これまでどおり進む。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    plans = plans_of(config)
    broken = (Reply(exit_code=255, stderr="切れた"),)
    script = StartScript(
        plans=plans,
        states={"head": (Reply(stdout="exited 1\n"),), "worker": (Reply(stdout="running 0\n"),)},
        logs={"head": broken, "worker": broken},
    )
    runner = runner_of(tmp_path, script)

    outcome = run_start(runner, config, tmp_path, timeout_s=1800.0)

    assert outcome.status == "failed"
    assert argv_of(runner, "docker", "stop") == tuple(
        stop_argv(plan.container_name) for plan in plans
    )
    assert argv_of(runner, "docker", "rm") == tuple(
        remove_argv(plan.container_name) for plan in plans
    )
    # 読めなかったので、知っている失敗はない。終了した台があるので UNCLASSIFIED になる
    assert outcome.observation is not None
    assert outcome.observation.known_failure is KnownFailure.UNCLASSIFIED
    assert "読めない" in outcome.log_tails["head"] or "届かなかった" in outcome.log_tails["head"]


def test_timeout_without_exit_is_not_classified(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """コンテナが終了していない時間切れには、失敗の種類を付けない。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    script = StartScript(plans=plans_of(config), tails={"head": UNKNOWN_LOG, "worker": UNKNOWN_LOG})
    runner = runner_of(tmp_path, script)

    outcome = run_start(runner, config, tmp_path, timeout_s=10.0)

    assert outcome.status == "failed"
    assert outcome.observation is not None
    assert outcome.observation.known_failure is None


# --- 待っても直らない食い違い ----------------------------------------------


def test_wrong_served_model_name_fails_without_waiting(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """名乗るモデルの名前が構成と違うのは、待っても直らないので、すぐ失敗にする。"""
    fake_vllm.set_model("glm-5-3-flash-other")
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))
    clock = FakeClock()

    outcome = run_start(runner, config, tmp_path, clock=clock, timeout_s=1800.0)

    assert outcome.status == "failed"
    assert clock.slept == []
    assert "glm-5-3-flash-other" in outcome.detail
    assert SERVED_MODEL in outcome.detail
    assert cleanup_order(runner) == CLEANUP_STEPS


def test_speculative_decoding_metrics_fail_without_waiting(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """投機的デコードの指標が出ていたら、待たずに失敗にする (requirements 6.7)。"""
    fake_vllm.set_metrics(MetricsSample(spec_decode=True))
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))
    clock = FakeClock()

    outcome = run_start(runner, config, tmp_path, clock=clock, timeout_s=1800.0)

    assert outcome.status == "failed"
    assert clock.slept == []
    assert "vllm:spec_decode_" in outcome.detail


def test_allowed_speculative_metrics_reach_ready(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """投機を許す構成では、投機の指標が出ているのを正常として受け付ける (C3)。"""
    fake_vllm.set_metrics(MetricsSample(spec_decode=True))
    config = serve_config(port_of(fake_vllm)).model_copy(update={"allow_speculative": True})
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))
    clock = FakeClock()

    outcome = run_start(runner, config, tmp_path, clock=clock, timeout_s=1800.0)

    assert outcome.status == "ready"
    assert outcome.service is not None
    assert outcome.service.health_ok is True


def test_allowed_config_without_speculative_metrics_fails_without_waiting(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """投機を許す構成で投機の指標が出ていなければ、待たずに失敗にする (C3)。"""
    config = serve_config(port_of(fake_vllm)).model_copy(update={"allow_speculative": True})
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))
    clock = FakeClock()

    outcome = run_start(runner, config, tmp_path, clock=clock, timeout_s=1800.0)

    assert outcome.status == "failed"
    assert clock.slept == []
    assert "allow_speculative" in outcome.detail


def test_allowed_config_with_unreadable_metrics_waits_until_the_timeout(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """投機を許す構成でも、指標が読めないだけなら上限まで待つ (C3)。"""
    fake_vllm.set_metrics_fault(Fault(body="これは Prometheus の形ではない\n"))
    config = serve_config(port_of(fake_vllm)).model_copy(update={"allow_speculative": True})
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))
    clock = FakeClock()

    outcome = run_start(runner, config, tmp_path, clock=clock, timeout_s=20.0)

    assert outcome.status == "failed"
    assert clock.slept == [10.0, 10.0]
    assert fake_vllm.call_count("/metrics") == 3


def test_a_models_payload_that_is_not_a_list_says_so(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """`/v1/models` の `data` が配列でないときは、「空」ではなく「配列でない」と言う。"""
    fake_vllm.set_models_fault(Fault(body=json.dumps({"data": {"id": SERVED_MODEL}})))
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))

    outcome = run_start(runner, config, tmp_path, timeout_s=10.0)

    assert outcome.status == "failed"
    assert "配列でない" in outcome.detail


def test_unreadable_metrics_wait_until_the_timeout(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """指標が読めないだけなら、上限まで待つ (起動の途中でありうる)。"""
    fake_vllm.set_metrics_fault(Fault(body="これは Prometheus の形ではない\n"))
    config = serve_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))
    clock = FakeClock()

    outcome = run_start(runner, config, tmp_path, clock=clock, timeout_s=20.0)

    assert outcome.status == "failed"
    assert clock.slept == [10.0, 10.0]
    assert fake_vllm.call_count("/metrics") == 3


# --- すでに動いている / 中身が違う ------------------------------------------


def test_already_running_makes_no_mutating_call(tmp_path: Path) -> None:
    """2 台とも一致して動いていれば、関門も了承も通さずに、いまの状態を示して終わる。"""
    config = serve_config(UNUSED_PORT)
    plans = plans_of(config)
    script = StartScript(
        plans=plans, listings={plan.node: (Reply(stdout=ps_line(plan)),) for plan in plans}
    )
    runner = runner_of(tmp_path, script)
    confirmer = SpyConfirmer()

    outcome = run_start(runner, config, tmp_path, confirmer=confirmer)

    assert outcome.status == "already_running"
    assert mutating_calls(runner) == ()
    assert confirmer.shown == []
    assert steps(runner) == ("ps", "ps")
    assert outcome.service is not None
    assert [node.container_state for node in outcome.service.nodes] == ["running", "running"]
    assert [node.started_at for node in outcome.service.nodes] == [STARTED_AT, STARTED_AT]


def test_same_name_but_different_config_sha256_is_refused(tmp_path: Path) -> None:
    """名前が同じで中身が違うときは、違う項目を示して断る (終了コード 1)。"""
    config = serve_config(UNUSED_PORT)
    plans = plans_of(config)
    script = StartScript(
        plans=plans,
        listings={
            plan.node: (Reply(stdout=ps_line(plan, labels={LABEL_CONFIG_SHA256: "f" * 64})),)
            for plan in plans
        },
    )
    runner = runner_of(tmp_path, script)
    confirmer = SpyConfirmer()

    outcome = run_start(runner, config, tmp_path, confirmer=confirmer)

    assert outcome.status == "refused"
    assert LABEL_CONFIG_SHA256 in outcome.detail
    assert "serve stop" in outcome.detail
    assert mutating_calls(runner) == ()
    assert confirmer.shown == []


# --- 関門と了承 -------------------------------------------------------------


def test_refused_gate_skips_the_approval(tmp_path: Path) -> None:
    """関門が 1 つでも断れば、了承を尋ねず、状態を変える呼び出しを 1 つも出さない。"""
    config = serve_config(UNUSED_PORT)
    script = StartScript(plans=plans_of(config), gpu_apps="4242, python3, 12345 MiB\n")
    runner = runner_of(tmp_path, script)
    confirmer = SpyConfirmer()

    outcome = run_start(runner, config, tmp_path, confirmer=confirmer)

    assert outcome.status == "refused"
    assert "python3" in outcome.detail
    assert confirmer.shown == []
    assert mutating_calls(runner) == ()
    assert [gate.gate for gate in outcome.gates if not gate.passed] == ["gpu_idle", "gpu_idle"]


def test_declined_approval_makes_no_mutating_call(tmp_path: Path) -> None:
    """了承しなければ、状態を変える呼び出しが 1 つも出ない (requirements 2.1)。"""
    config = serve_config(UNUSED_PORT)
    runner = runner_of(tmp_path, StartScript(plans=plans_of(config)))

    with pytest.raises(ApprovalError):
        run_start(runner, config, tmp_path, confirmer=SpyConfirmer(approves=False))
    assert mutating_calls(runner) == ()


def test_non_serve_config_is_refused(tmp_path: Path) -> None:
    """`serve start` に使えるのは、推論サーバーの構成だけである。"""
    config = serve_config(UNUSED_PORT).model_copy(update={"kind": "job", "served_model_name": None})
    runner = runner_of(tmp_path, StartScript(plans=()))

    with pytest.raises(ConfigError, match="kind"):
        run_start(runner, config, tmp_path)
    assert runner.calls == ()


# --- 名前の衝突 -------------------------------------------------------------


def test_name_conflict_does_not_roll_back_the_foreign_name(tmp_path: Path) -> None:
    """名前の衝突では、よその名前を止めない。もう片方の台の自分のコンテナは片付ける。"""
    config = serve_config(UNUSED_PORT)
    head_plan, worker_plan = plans_of(config)
    conflict = (
        "docker: Error response from daemon: Conflict. The container name"
        f' "/{worker_plan.container_name}" is already in use.\n'
    )
    script = StartScript(
        plans=(head_plan, worker_plan),
        runs={
            "head": (Reply(stdout="0123456789ab\n"),),
            "worker": (Reply(exit_code=125, stderr=conflict),),
        },
        # worker の一覧には、衝突した名前の行がない (その名前は、よそのコンテナのもの)
        listings={
            "head": (Reply(stdout=""), Reply(stdout=ps_line(head_plan))),
            "worker": (Reply(stdout=""),),
        },
    )
    runner = runner_of(tmp_path, script)

    with pytest.raises(lc.LifecycleError, match=worker_plan.container_name):
        run_start(runner, config, tmp_path)

    assert argv_of(runner, "docker", "stop") == (stop_argv(head_plan.container_name),)
    assert argv_of(runner, "docker", "rm") == (remove_argv(head_plan.container_name),)


# --- 中断 -------------------------------------------------------------------


def test_interrupt_while_waiting_cleans_up_then_propagates(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """待ちの途中の中断でも、末尾 → 回収 → 停止 → 削除のあと、中断をそのまま伝える。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    plans = plans_of(config)
    runner = runner_of(tmp_path, StartScript(plans=plans))

    def interrupt() -> None:
        raise KeyboardInterrupt

    report = io.StringIO()
    with pytest.raises(KeyboardInterrupt):
        run_start(runner, config, tmp_path, clock=FakeClock(on_sleep=interrupt), report=report)

    # 中断の経路では、分類用の長い末尾を読まない (片付けを遅らせるだけ)
    assert cleanup_order(runner) == CLEANUP_STEPS_WITHOUT_CLASSIFY
    assert tail_line_counts(runner) == [str(DEFAULT_TAIL_LINES), str(DEFAULT_TAIL_LINES)]
    assert argv_of(runner, "docker", "rm") == tuple(
        remove_argv(plan.container_name) for plan in plans
    )
    # 回収できた記録の置き場所を、計測者に知らせる
    assert f"20260922T030000Z-start-{CONFIG_NAME}" in report.getvalue()


def test_interrupt_during_cleanup_still_stops_and_removes_both(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """片付けの最中に中断が来ても、2 台とも `stop` → `rm` の両方を試みる。

    `docker rm` (強制なし) は、動いているコンテナを消せない (`-f` は、了承済みの計画にも、
    `remote` の許可の一覧の決まりにもない)。止めずに `rm` だけ流すと、コンテナが残るので、
    中断のあとでも、`stop` を飛ばさない。
    """
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    plans = plans_of(config)
    script = StartScript(
        plans=plans, stops={"head": (Reply(raises=KeyboardInterrupt()),), "worker": (Reply(),)}
    )
    runner = runner_of(tmp_path, script)

    with pytest.raises(KeyboardInterrupt):
        run_start(runner, config, tmp_path, timeout_s=10.0)

    assert argv_of(runner, "docker", "stop") == tuple(
        stop_argv(plan.container_name) for plan in plans
    )
    assert argv_of(runner, "docker", "rm") == tuple(
        remove_argv(plan.container_name) for plan in plans
    )


def test_interrupt_while_reading_the_shown_tail_still_cleans_up(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """見せる末尾の読み取りで中断が来ても、残りの読み取りを飛ばして、2 台を片付ける。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    plans = plans_of(config)
    script = StartScript(
        plans=plans,
        extra=(
            Rule(
                prefix=("docker", "logs", "--timestamps", "--tail", str(DEFAULT_TAIL_LINES)),
                node="head",
                replies=(Reply(raises=KeyboardInterrupt()),),
            ),
        ),
    )
    runner = runner_of(tmp_path, script)

    with pytest.raises(KeyboardInterrupt):
        run_start(runner, config, tmp_path, timeout_s=10.0)

    # 分類の末尾と、記録の回収は、飛ばして片付けに急ぐ
    assert tail_line_counts(runner) == [str(DEFAULT_TAIL_LINES)]
    assert runner.pulls == ()
    assert argv_of(runner, "docker", "stop") == tuple(
        stop_argv(plan.container_name) for plan in plans
    )
    assert argv_of(runner, "docker", "rm") == tuple(
        remove_argv(plan.container_name) for plan in plans
    )


def test_interrupt_while_reading_the_classification_tail_still_cleans_up(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """分類の末尾の読み取りで中断が来ても、記録の回収を飛ばして、2 台を片付ける。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    plans = plans_of(config)
    script = StartScript(
        plans=plans,
        extra=(
            Rule(
                prefix=("docker", "logs", "--timestamps", "--tail", OBSERVE_LINES),
                node="head",
                replies=(Reply(raises=KeyboardInterrupt()),),
            ),
        ),
    )
    runner = runner_of(tmp_path, script)

    with pytest.raises(KeyboardInterrupt):
        run_start(runner, config, tmp_path, timeout_s=10.0)

    assert tail_line_counts(runner) == [
        str(DEFAULT_TAIL_LINES),
        str(DEFAULT_TAIL_LINES),
        OBSERVE_LINES,
    ]
    assert runner.pulls == ()
    assert argv_of(runner, "docker", "stop") == tuple(
        stop_argv(plan.container_name) for plan in plans
    )
    assert argv_of(runner, "docker", "rm") == tuple(
        remove_argv(plan.container_name) for plan in plans
    )


def test_interrupt_while_collecting_the_records_still_cleans_up(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """記録の回収 (`pull`) で中断が来ても、2 台を片付ける。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    plans = plans_of(config)
    script = StartScript(
        plans=plans,
        extra=(Rule(kind="pull", node="head", replies=(Reply(raises=KeyboardInterrupt()),)),),
    )
    runner = runner_of(tmp_path, script)
    report = io.StringIO()

    with pytest.raises(KeyboardInterrupt):
        run_start(runner, config, tmp_path, timeout_s=10.0, report=report)

    assert len(runner.pulls) == 1
    assert argv_of(runner, "docker", "stop") == tuple(
        stop_argv(plan.container_name) for plan in plans
    )
    assert argv_of(runner, "docker", "rm") == tuple(
        remove_argv(plan.container_name) for plan in plans
    )
    assert f"20260922T030000Z-start-{CONFIG_NAME}" in report.getvalue()


def test_interrupt_while_listing_for_the_rollback_does_not_stop_by_name(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """片付けの一覧の読み取りで中断が来たら、名前だけで止めない (requirements 2.3)。

    一覧が読めていないので、その名前が自分のコンテナかどうかが分からない。`serve stop` を
    促して、中断を伝える。
    """
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    plans = plans_of(config)
    head_plan, worker_plan = plans
    # head の一覧は、片付けの直前の 6 回目で中断する (1: 関門の前、2: 起こしたあと、
    # 3: 見せる末尾、4: 分類の末尾、5: 記録の回収、6: 片付け)
    script = StartScript(
        plans=plans,
        listings={
            "head": (
                Reply(stdout=""),
                *(Reply(stdout=ps_line(head_plan)) for _ in range(4)),
                Reply(raises=KeyboardInterrupt()),
            ),
            "worker": (Reply(stdout=""), Reply(stdout=ps_line(worker_plan))),
        },
    )
    runner = runner_of(tmp_path, script)
    report = io.StringIO()

    with pytest.raises(KeyboardInterrupt):
        run_start(runner, config, tmp_path, timeout_s=10.0, report=report)

    assert argv_of(runner, "docker", "stop") == ()
    assert argv_of(runner, "docker", "rm") == ()
    assert "serve stop" in report.getvalue()


def test_interrupt_during_cleanup_after_a_remote_error_is_still_an_interrupt(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """元の誤りが何であれ、片付けの最中の中断は、中断として伝える (終了コード 130)。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    plans = plans_of(config)
    script = StartScript(
        plans=plans,
        states={"head": (Reply(exit_code=255, stderr="切れた"),), "worker": (Reply(),)},
        stops={"head": (Reply(raises=KeyboardInterrupt()),), "worker": (Reply(),)},
    )
    runner = runner_of(tmp_path, script)

    with pytest.raises(KeyboardInterrupt):
        run_start(runner, config, tmp_path)

    assert argv_of(runner, "docker", "rm") == tuple(
        remove_argv(plan.container_name) for plan in plans
    )


def test_remote_error_while_waiting_becomes_a_lifecycle_error(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """了承のあとに読み取りが届かなかったことは、片付けてから、実行しての失敗にする。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    plans = plans_of(config)
    script = StartScript(
        plans=plans, states={"head": (Reply(exit_code=255, stderr="切れた"),), "worker": (Reply(),)}
    )
    runner = runner_of(tmp_path, script)

    with pytest.raises(lc.LifecycleError, match="届かなかった"):
        run_start(runner, config, tmp_path)

    assert argv_of(runner, "docker", "rm") == tuple(
        remove_argv(plan.container_name) for plan in plans
    )


def test_unchecked_node_asks_for_serve_stop(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """片付けで確かめられなかった台があれば、`serve stop` を促す文を誤りの文に入れる。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = serve_config(port_of(fake_vllm))
    plans = plans_of(config)
    script = StartScript(
        plans=plans,
        listings={
            "head": (
                Reply(stdout=""),
                Reply(stdout=ps_line(plans[0])),
                Reply(stdout=ps_line(plans[0])),
                Reply(stdout=ps_line(plans[0])),
                Reply(exit_code=255, stderr="切れた"),
            ),
            "worker": (Reply(stdout=""), Reply(stdout=ps_line(plans[1]))),
        },
    )
    runner = runner_of(tmp_path, script)

    outcome = run_start(runner, config, tmp_path, timeout_s=10.0)

    assert outcome.status == "failed"
    assert "serve stop" in outcome.detail


# --- 構成から読む値 ---------------------------------------------------------


def test_http_port_comes_from_the_port_flag() -> None:
    """HTTP のポートは、`args` の `--port` の設定から取る (`--master-port` ではない)。"""
    assert lc.http_port(serve_config(UNUSED_PORT)) == UNUSED_PORT


def test_http_port_is_refused_when_the_config_has_no_port_flag() -> None:
    """`--port` の設定がなければ、Spark に触る前に断る。"""
    config = serve_config(UNUSED_PORT)
    without = config.model_copy(
        update={"args": {key: value for key, value in config.args.items() if key != "port"}}
    )

    with pytest.raises(ConfigError, match="--port"):
        lc.http_port(without)


def test_http_client_does_not_trust_the_environment() -> None:
    """Mac の側のプロキシの設定を拾わない (`trust_env=False`)。"""
    client = lc.new_client()
    try:
        assert client.trust_env is False
    finally:
        client.close()


def test_label_time_reads_the_label_that_plan_writes() -> None:
    """ラベルの時刻の読み方が、`plan` の書き方と合っている。"""
    plan = plans_of(serve_config(UNUSED_PORT))[0]

    assert lc.label_time(plan.labels[LABEL_STARTED_AT]) == STARTED_AT
    assert lc.label_time("こわれた時刻") is None
    assert lc.label_time(None) is None


def test_the_started_containers_carry_the_owner_label() -> None:
    """起こす計画には、所有のラベルが付いている (あとから自分のものとして選べる)。"""
    plan = plans_of(serve_config(UNUSED_PORT))[0]

    assert plan.labels[LABEL_OWNER] == OWNER


# --- 4.1 (probe) が使い回す、公開の口 ---------------------------------------

PROBE_HELPERS = (
    "Controls",
    "Readiness",
    "RunFailed",
    "Target",
    "Waited",
    "WrapUp",
    "check_ready",
    "clean_up",
    "launch_record",
    "observation",
    "push_launch_records",
    "record_pushes",
    "start_all",
    "started_targets",
    "wait_ready",
    "wrap_up",
)
"""4.1 (`probe`) が、下線つきの名前を 1 つも触らずに使える口 (レビューの指摘 3)。"""


def test_the_helpers_for_the_probe_task_are_public() -> None:
    """4.1 は `lifecycle.py` を直せないので、使い回す口を下線なしで公開する。"""
    assert set(PROBE_HELPERS) <= set(lc.__all__)
    for name in PROBE_HELPERS:
        assert getattr(lc, name, None) is not None, name


def test_controls_can_be_built_with_defaults() -> None:
    """`Controls` は、時計・眠り・知らせの先・間隔の既定の値つきで組み立てられる。"""
    client = lc.new_client()
    try:
        controls = lc.Controls(client=client, timeout_s=60.0)
        assert controls.poll_interval_s == lc.READY_POLL_INTERVAL_S
        assert controls.tail_lines == DEFAULT_TAIL_LINES
        assert controls.start_timeout_s == lc.START_TIMEOUT_S
        assert controls.clock() > 0.0
        assert controls.report is not None
    finally:
        client.close()


def probe_config(port: int) -> ConfigDef:
    """1 台の縮小の確認の構成 (4.1 が使う形。ここでは、公開の口の組み立てだけを見る)。"""
    return serve_config(port).model_copy(update={"kind": "probe", "nodes": ("head",)})


@pytest.mark.parametrize("relative", [Path(""), Path("records"), Path("./var/x")])
def test_the_public_record_helpers_refuse_a_relative_record_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: Path
) -> None:
    """公開の口 (4.1 が使う) でも、`record_dir` は絶対の道筋だけを受ける。

    配る元は、配る前に空にする (消す操作)。相対の道筋だと、作業ディレクトリに対して効く。
    """
    monkeypatch.chdir(tmp_path)
    victim = tmp_path / relative / "launch" / "head" / "keep-me.txt"
    config = probe_config(8000)
    plans = build_plans(config, NODES, STARTED_AT)
    runner = runner_of(tmp_path, StartScript(plans=plans))
    record = lc.launch_record(config, plans, STARTED_AT, repo_commit=REPO_COMMIT, repo_dirty=False)

    with pytest.raises(ValueError, match="絶対"):
        lc.record_pushes(config, relative)
    with pytest.raises(ValueError, match="絶対"):
        lc.push_launch_records(runner, config, NODES, record, relative)

    assert runner.calls == ()
    assert not victim.parent.exists()


def test_the_public_helpers_compose_a_one_node_run(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """4.1 の道すじを、公開の口だけで組み立てられる (1 台、成功でも必ず片付ける)。

    「計画を受ける → 起動の記録を置く → 起こす → 待つ → 観察する → 片付ける」を、下線つきの
    名前を 1 つも触らずに書けること。`wrap_up` / `clean_up` は、成功のあとにも呼べること。
    """
    config = probe_config(port_of(fake_vllm))
    plans = build_plans(config, NODES, STARTED_AT)
    # 関門の前の一覧の読み取り (`start` がするもの) は、この組み立てには出てこない
    script = StartScript(plans=plans, listings={"head": (Reply(stdout=ps_line(plans[0])),)})
    runner = runner_of(tmp_path, script)
    approved = build_approved_plan(
        plans, extra_forward=lc.record_pushes(config, record_dir(tmp_path))
    )
    runner.approve(approved)
    clock = FakeClock()
    client = lc.new_client()
    try:
        controls = lc.Controls(
            client=client,
            timeout_s=60.0,
            poll_interval_s=10.0,
            sleep=clock.sleep,
            clock=clock.monotonic,
            report=io.StringIO(),
        )
        record = lc.launch_record(
            config, plans, STARTED_AT, repo_commit=REPO_COMMIT, repo_dirty=False
        )
        lc.push_launch_records(runner, config, NODES, record, record_dir(tmp_path))
        lc.start_all(runner, config, NODES, plans, controls)
        targets = lc.started_targets(runner, config, NODES, plans, controls)
        assert [target.plan.node for target in targets] == ["head"]
        waited = lc.wait_ready(
            runner,
            targets,
            controls,
            base_url=f"http://{HEAD.lan_addr}:{port_of(fake_vllm)}",
            served_model_name=SERVED_MODEL,
            speculative_allowed=False,
        )
        assert waited.failure is None
        assert waited.readiness.ready is True
        done = lc.wrap_up(
            runner,
            config,
            NODES,
            plans,
            controls,
            var_root=var_root(tmp_path),
            started_at=STARTED_AT,
            classify=True,
        )
    finally:
        client.close()

    assert done.interrupted is None
    assert done.problems == ()
    found = lc.observation(config, done.classified, ())
    assert found is not None
    assert found.attention_backend == "FLASH_ATTN_MLA"
    # 成功のあとでも、必ず止めて消す (4.1 は、どの結果でもコンテナを残さない)
    assert argv_of(runner, "docker", "stop") == (stop_argv(plans[0].container_name),)
    assert argv_of(runner, "docker", "rm") == (remove_argv(plans[0].container_name),)
