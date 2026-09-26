"""1 台の縮小の確認の試験 (tasks.md 4.1)。

確かめること (design.md 「確認 › probe」、requirements 5.1〜5.5、8.7、10.5、tasks.md 4.1 の
完了の状態):

- **失敗した**: アテンションの形の assert (`pe_dim must be 64`) で終了する記録を返す偽物で、
  `failed` と、失敗の種類 (`pe_dim_assert`) と、前後の行の抜き出しが返り、コンテナが残らない
  (記録の回収 → `stop` → `rm` が出る。requirements 5.3)
- **起動した**: 短い要求が**ちょうど 1 つ**送られ、応答の本文が、どのファイルにも、結果の型
  にも、画面にも出ない (requirements 5.4、10.5)。成功でも、コンテナが残らない
- **判定できない**: 終了せずに時間切れ、関門で断られた、起こす途中で届かなかった、待っても
  直らない食い違い (requirements 5.5)
- 知っている失敗でない終了は `failed` の `unclassified` (8.7 の材料は `known_failure`)
- 関門で断られる、了承しない → 状態を変える呼び出しが 1 つも出ない (requirements 2.1)
- 2 台の構成、`kind` が `probe` でない構成 → Spark に触る前に断る
- 待ちの途中、短い要求の最中、片付けの最中の中断 → コンテナが残らず、中断が伝わる
- 片付けが終わらなかったら、起動していても `ProbeError` (コンテナが残ったかもしれないことが、
  終了コードで分かる)
- 偽の実行役に記録された、コンテナを対象にする操作のすべてが、自分の一覧の ID か、了承済みの
  計画の名前だけを対象にしている (requirements 2.3、2.4)
- 了承のあとの、状態を変える呼び出しが、見せた計画の列と完全に一致する
- `probe.py` が、`lifecycle` の下線つきの名前を 1 つも参照していない (ソースを `ast` で歩く)

実物の ssh、rsync、docker、推論サーバーには、どの段でもつながない (`FakeRunner` と、
`fake_vllm` の偽の推論サーバーを相手にする)。試験は、実際に眠らない (待つ間隔と時計は、
引数で差し替える)。
"""

from __future__ import annotations

import ast
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

import httpx
import pytest
from pydantic import HttpUrl

from fake_runner import FakeRunner, RecordedCall, Reply, Rule
from fake_vllm import FakeVllm, Fault, MessagesReply, MetricsSample
from serving_kit import lifecycle as lc
from serving_kit import probe as pr
from serving_kit.config import ConfigError
from serving_kit.guards import ApprovalError, remove_argv, stop_argv
from serving_kit.plan import LABEL_IMAGE, OWNER_FILTER, build_plans
from serving_kit.types import (
    ConfigDef,
    ContainerPlan,
    ImageRef,
    KnownFailure,
    ManifestFile,
    NodeDef,
    NodeRole,
    ProbeOutcome,
    Setting,
    VerificationRecord,
    WeightsManifest,
    WeightsRef,
)

# --- 見本の値 -------------------------------------------------------------

DIGEST = "b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
IMAGE_REF = f"vllm/vllm-openai@sha256:{DIGEST}"
REPO = "RedHatAI/GLM-5.3-Flash-NVFP4"
REVISION = "18d55bfd" + "0" * 32
SLUG = "RedHatAI__GLM-5.3-Flash-NVFP4"
MOUNT_AT = "/models/nvfp4"
REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"
CONFIG_NAME = "probe-pinned"
SERVED_MODEL = "glm-5-3-flash"
UNUSED_PORT = 8123
"""HTTP に進まない試験が使う、偽の推論サーバーのない番号。"""

HF_OVERRIDES = '{"num_hidden_layers": 4, "first_k_dense_replace": 1}'
"""層の数を減らす指定 (JSON の値。波かっこを含むので、置き換えの印として扱われない)。"""

STARTED_AT = datetime(2026, 9, 22, 6, 0, 0, tzinfo=UTC)
VERIFIED_AT = datetime(2026, 9, 22, 1, 30, 0, tzinfo=UTC)
GENERATED_AT = datetime(2026, 9, 22, 1, 0, 0, tzinfo=UTC)

REPO_COMMIT = "5dd44fc" + "0" * 33
SOURCE = HttpUrl("https://docs.vllm.ai/en/latest/cli/serve/")
QUOTE = "vllm serve [model_tag] [options]"

CONTAINER_ID = "0123456789ab"

BODY_MARKER = "ZZ-PROBE-BODY-MARKER-ZZ"
"""応答の本文にだけ現れる目印 (どのファイルにも、結果にも、画面にも出ないことを見る)。"""

PROBE_RECORD_PATH = f"{REMOTE_ROOT}/state/{SLUG}.probe.verified.json"
"""縮小の確認の構成が読む、照合の結果の記録 (範囲は `probe_files`)。"""

FILE_SIZES: Mapping[str, int] = {
    "config.json": 4_096,
    "tokenizer.json": 36_000_000,
    "model-00001-of-00001.safetensors": 98_000_000_000,
}

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


def probe_config(
    port: int,
    *,
    ready_timeout_s: int = 600,
    nodes: tuple[NodeRole, ...] = ("head",),
    kind: str = "probe",
) -> ConfigDef:
    """1 台の縮小の確認の構成 (design.md 「probe」: `--load-format dummy`、`--hf-overrides`)。

    重みの本体は結び付けず、設定とトークナイザだけを置いた `probe/<slug>/` を見せる。
    """
    mount = f"type=bind,source={{remote_root}}/probe/{SLUG},target=/probe,readonly"
    return ConfigDef(
        name=CONFIG_NAME,
        kind=kind,  # type: ignore[arg-type]  # `kind` が probe でない構成も作って断らせる
        description="段 0。中身のない重みで、1 台だけ起こす (試験用に縮めたもの)",
        nodes=nodes,
        image=IMAGE,
        weights=WEIGHTS,
        docker={
            "gpus": _setting("--gpus", "all"),
            "mount-probe": _setting("--mount", mount),
        },
        args={
            "model-path": _setting(value="/probe"),
            "served-model-name": _setting("--served-model-name", SERVED_MODEL),
            "host": _setting("--host", "{head.lan_addr}"),
            "port": _setting("--port", str(port), is_port=True),
            "load-format": _setting("--load-format", "dummy"),
            "hf-overrides": _setting("--hf-overrides", HF_OVERRIDES),
            "max-model-len": _setting("--max-model-len", "2048"),
            "max-num-seqs": _setting("--max-num-seqs", "1"),
        },
        env={"log-level": _setting("VLLM_LOGGING_LEVEL", "DEBUG")},
        ready_timeout_s=ready_timeout_s,
        served_model_name=SERVED_MODEL,
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

PE_DIM_MARK = "pe_dim must be 64"
"""アテンションの形の assert (design.md 「observe」の `pe_dim_assert`)。"""

UNKNOWN_MARK = "something we have never seen before"
"""知っている失敗のどれでもない誤りの行 (`unclassified` になる)。"""

OBSERVE_LINES = str(lc.OBSERVE_TAIL_LINES)
"""分類に使う末尾の行数 (台本で、長い読み取りだけを選び分けるために使う)。"""

TRACEBACK_LINE = '  File "/usr/lib/python3/vllm/v1/engine/core.py", line {}, in run_engine_core'
"""vLLM の失敗のときに重なる traceback の 1 行 (関係のない行の見本)。"""


def traceback_lines(count: int) -> str:
    """関係のない traceback だけが並ぶ記録 (見せる末尾に入る部分の見本)。"""
    return "".join(f"{TRACEBACK_LINE.format(index)}\n" for index in range(count))


def deep_log(marker: str, *, above: int = 500) -> str:
    """肝心の行が、末尾から `above` 行**手前**にある記録の見本。

    見せる末尾 (80 行) には入らないので、**分類に使う長い末尾を読んだときだけ**、失敗の
    種類が読める (tasks.md の Implementation Notes 3.4 の決めごとの 5)。
    """
    lines = [f"INFO booting up {index}" for index in range(20)]
    lines.append(f"ERROR AssertionError: {marker}")
    return "".join(f"{line}\n" for line in lines) + traceback_lines(above)


def ps_row(name: str, container_id: str, state: str, labels: Mapping[str, str]) -> str:
    """`docker ps -a --filter label=… --format json` の 1 行。"""
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


def ps_line(plan: ContainerPlan, *, state: str = "running") -> str:
    """この計画のコンテナの、一覧の 1 行。"""
    return ps_row(plan.container_name, CONTAINER_ID, state, plan.labels)


def verified_record(role: NodeRole) -> str:
    """`gate_weights_verified` が読む、照合の結果の記録 (範囲は `probe_files`)。"""
    files = MANIFEST.probe_files
    return VerificationRecord(
        repo=REPO,
        revision=REVISION,
        scope="probe_files",
        node=role,
        verified_at=VERIFIED_AT,
        file_count=len(files),
        total_bytes=sum(entry.size for entry in files),
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
class ProbeScript:
    """`serve probe` の台本 (既定: 関門が通り、1 台が起き、片付けもできる)。"""

    plans: Sequence[ContainerPlan]
    listings: tuple[Reply, ...] | None = None
    states: tuple[Reply, ...] | None = None
    runs: tuple[Reply, ...] | None = None
    stops: tuple[Reply, ...] | None = None
    tails: str = READY_LOG
    long_tails: str | None = None
    """`--tail <OBSERVE_TAIL_LINES>` の読み取りだけが返す記録 (実物の docker と同じく、長く
    読んだときだけ、手前の行まで返ることを表す)。書かなければ、`tails` と同じものが返る。"""

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
                else self.listings
            )
            states = (Reply(stdout="running 0\n"),) if self.states is None else self.states
            rules.append(Rule(prefix=OWN_CONTAINERS_ARGV, node=role, replies=listings))
            rules.append(Rule(prefix=("docker", "container", "inspect"), node=role, replies=states))
            if self.long_tails is not None:
                # 長く読んだときだけ、手前の行まで返る (実物の `docker logs --tail` と同じ)
                rules.append(
                    Rule(
                        prefix=("docker", "logs", "--timestamps", "--tail", OBSERVE_LINES),
                        node=role,
                        replies=(Reply(stdout=self.long_tails),),
                    )
                )
            rules.append(
                Rule(
                    prefix=("docker", "logs", "--timestamps"),
                    node=role,
                    replies=(Reply(stdout=self.tails),),
                )
            )
            rules.append(
                Rule(prefix=("cat",), node=role, replies=(Reply(stdout=verified_record(role)),))
            )
            if self.runs is not None:
                rules.append(Rule(prefix=("docker", "run"), node=role, replies=self.runs))
            if self.stops is not None:
                rules.append(Rule(prefix=("docker", "stop"), node=role, replies=self.stops))
        rules.extend(
            (
                Rule(prefix=("uname",), replies=(Reply(stdout="spark-153d\n"),)),
                Rule(prefix=("nvidia-smi",), replies=(Reply(stdout=self.gpu_apps),)),
                Rule(prefix=("test", "-e"), replies=(Reply(exit_code=0 if self.layout_ok else 1),)),
                Rule(prefix=("docker", "image", "inspect"), replies=(Reply(stdout=self.digests),)),
                Rule(prefix=("df",), replies=(Reply(stdout=self.avail),)),
                Rule(prefix=("ss",), replies=(Reply(stdout=self.listening),)),
                Rule(prefix=("docker", "run"), replies=(Reply(stdout=f"{CONTAINER_ID}\n"),)),
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


def runner_of(tmp_path: Path, script: ProbeScript, *, default: Reply | None = None) -> FakeRunner:
    return FakeRunner(var_root=var_root(tmp_path), script=script.rules(), default=default)


def run_probe(
    runner: FakeRunner,
    config: ConfigDef,
    tmp_path: Path,
    *,
    confirmer: SpyConfirmer | None = None,
    clock: FakeClock | None = None,
    report: TextIO | None = None,
    **extra: Any,
) -> ProbeOutcome:
    """既定の下ごしらえで `probe.run_probe` を流す助け。"""
    used_clock = FakeClock() if clock is None else clock
    return pr.run_probe(
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


CLEANUP_STEPS = ["logs tail", "logs tail", "logs", "pull", "pull", "stop", "rm"]
"""1 台ぶんの片付けの道すじ。

「**見せる**末尾 (80 行) → **分類に使う**末尾 (2,000 行) → 回収 (記録・通信の記録・起動の
記録) → 停止 → 削除」。縮小の確認は、**どの結果でも**この道を通る (requirements 5.4)。
"""

CLEANUP_STEPS_WITHOUT_CLASSIFY = CLEANUP_STEPS[1:]
"""分類の読み取りをしない経路 (中断、起こす途中の失敗) の片付けの道すじ。"""


def files_under(root: Path) -> list[Path]:
    """`root` の下のファイルを、すべて並べる。"""
    return [path for path in root.rglob("*") if path.is_file()]


def assert_marker_is_nowhere(tmp_path: Path, outcome: ProbeOutcome | None, shown: str) -> None:
    """応答の本文が、どのファイルにも、結果の型にも、画面にも出ていないこと (10.5)。"""
    for path in files_under(tmp_path):
        assert BODY_MARKER not in path.read_bytes().decode("utf-8", "replace"), path
    if outcome is not None:
        assert BODY_MARKER not in outcome.model_dump_json()
        assert BODY_MARKER not in outcome.detail
    assert BODY_MARKER not in shown


# --- 失敗した (requirements 5.3、8.7) --------------------------------------


def test_a_pe_dim_assert_is_reported_as_failed_with_its_kind(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """アテンションの形の assert で終了する偽物で、`failed` と失敗の種類と抜き出しが返る。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = probe_config(port_of(fake_vllm))
    script = ProbeScript(
        plans=plans_of(config),
        states=(Reply(stdout="exited 1\n"),),
        # 肝心の行は、見せる 80 行には入らない (分類に使う長い末尾でだけ読める)
        tails=traceback_lines(120),
        long_tails=deep_log(PE_DIM_MARK),
    )
    runner = runner_of(tmp_path, script)

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert outcome.config_name == CONFIG_NAME
    assert outcome.observation is not None
    assert outcome.observation.known_failure == KnownFailure.PE_DIM_ASSERT
    assert any(PE_DIM_MARK in line for line in outcome.observation.failure_excerpt)
    assert outcome.reply is None


def test_a_failed_probe_leaves_no_container(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """失敗しても、記録を回収してから、必ず止めて消す (requirements 5.4)。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = probe_config(port_of(fake_vllm))
    plans = plans_of(config)
    script = ProbeScript(
        plans=plans, states=(Reply(stdout="exited 1\n"),), long_tails=deep_log(PE_DIM_MARK)
    )
    runner = runner_of(tmp_path, script)

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert cleanup_order(runner) == CLEANUP_STEPS
    assert argv_of(runner, "docker", "stop") == (stop_argv(plans[0].container_name),)
    assert argv_of(runner, "docker", "rm") == (remove_argv(plans[0].container_name),)


def test_an_unknown_exit_is_failed_as_unclassified(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """知っている失敗でない終了も「失敗した」で、種類は `unclassified` になる。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = probe_config(port_of(fake_vllm))
    script = ProbeScript(
        plans=plans_of(config),
        states=(Reply(stdout="exited 137\n"),),
        long_tails=deep_log(UNKNOWN_MARK),
    )
    runner = runner_of(tmp_path, script)

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert outcome.observation is not None
    assert outcome.observation.known_failure == KnownFailure.UNCLASSIFIED


# --- 起動した (requirements 5.4、10.5) -------------------------------------


def test_a_ready_probe_sends_exactly_one_request(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """起動したら、短い要求を**ちょうど 1 つ**送り、返り切ったかだけを記録する。"""
    fake_vllm.set_messages_reply(MessagesReply(text=BODY_MARKER, stop_reason="end_turn"))
    config = probe_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "ready", outcome.detail
    assert fake_vllm.call_count("/v1/messages") == 1
    assert outcome.reply is not None
    assert outcome.reply.http_status == 200
    assert outcome.reply.stop_reason == "end_turn"
    assert outcome.reply.replacement_char is False


def test_a_ready_probe_writes_the_reply_body_nowhere(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """応答の本文が、どのファイルにも、結果の型にも、画面にも出ない (requirements 10.5)。"""
    fake_vllm.set_messages_reply(MessagesReply(text=BODY_MARKER))
    config = probe_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))
    report = io.StringIO()

    outcome = run_probe(runner, config, tmp_path, report=report)

    assert outcome.status == "ready", outcome.detail
    assert files_under(tmp_path), "回収した記録が 1 つもないと、この試験は何も見ていない"
    assert_marker_is_nowhere(tmp_path, outcome, report.getvalue())


def test_a_ready_probe_leaves_no_container(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """成功でも、記録を回収してから、必ず止めて消す (requirements 5.4)。"""
    config = probe_config(port_of(fake_vllm))
    plans = plans_of(config)
    runner = runner_of(tmp_path, ProbeScript(plans=plans))

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "ready", outcome.detail
    assert cleanup_order(runner) == CLEANUP_STEPS
    assert argv_of(runner, "docker", "stop") == (stop_argv(plans[0].container_name),)
    assert argv_of(runner, "docker", "rm") == (remove_argv(plans[0].container_name),)


def test_a_ready_probe_reads_the_observation(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """起動したときも、記録から選ばれた部品を読む (requirements 5.2)。"""
    config = probe_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "ready", outcome.detail
    assert outcome.observation is not None
    assert outcome.observation.attention_backend == "FLASH_ATTN_MLA"
    assert outcome.observation.known_failure is None


def test_a_probe_that_allows_speculation_reaches_ready(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """投機を許す probe の構成では、投機の指標が出ているのを正常として受け付ける (C3)。"""
    fake_vllm.set_metrics(MetricsSample(spec_decode=True))
    config = probe_config(port_of(fake_vllm)).model_copy(update={"allow_speculative": True})
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "ready", outcome.detail


def test_a_probe_that_does_not_allow_speculation_is_inconclusive(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """投機を許さない probe の構成では、投機の指標が出ていたら受け付けない (C3)。"""
    fake_vllm.set_metrics(MetricsSample(spec_decode=True))
    config = probe_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "inconclusive", outcome.detail


def test_a_failing_short_request_still_counts_as_ready(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """短い要求が 200 を返さなくても、受け付けは始まっていたので `ready` のままにする。

    見るのは HTTP の状態と、返り切ったか (終わりの理由) だけで、断りには使わない。
    """
    fake_vllm.set_messages_fault(Fault(status=500))
    config = probe_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "ready", outcome.detail
    assert outcome.reply is not None
    assert outcome.reply.http_status == 500
    assert outcome.reply.stop_reason is None


# --- 判定できない (requirements 5.5) ---------------------------------------


def test_a_timeout_without_exit_is_inconclusive(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """終了せずに時間切れになったら「判定できない」で、コンテナは残らない。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = probe_config(port_of(fake_vllm), ready_timeout_s=60)
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "inconclusive"
    assert outcome.reply is None
    assert cleanup_order(runner) == CLEANUP_STEPS
    assert argv_of(runner, "docker", "rm") != ()


def test_the_wait_limit_can_be_overridden_for_this_run(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """待ちの上限は、その回だけ上書きできる (design.md 「cli」の `--timeout`)。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = probe_config(port_of(fake_vllm), ready_timeout_s=100_000)
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))
    clock = FakeClock()

    outcome = run_probe(runner, config, tmp_path, clock=clock, timeout_s=30.0)

    assert outcome.status == "inconclusive"
    assert sum(clock.slept) <= 30.0


def test_a_wrong_served_model_name_is_inconclusive(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """待っても直らない食い違いは、時間切れを待たずに「判定できない」にする。"""
    fake_vllm.set_model("someone-elses-model")
    config = probe_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))
    clock = FakeClock()

    outcome = run_probe(runner, config, tmp_path, clock=clock)

    assert outcome.status == "inconclusive"
    assert clock.slept == []
    assert "someone-elses-model" in outcome.detail
    assert cleanup_order(runner) == CLEANUP_STEPS


def test_a_failed_docker_run_is_inconclusive_and_cleans_up(tmp_path: Path) -> None:
    """起こす途中で届かなかったら「判定できない」で、片付けまで進む。"""
    config = probe_config(UNUSED_PORT)
    script = ProbeScript(
        plans=plans_of(config), runs=(Reply(exit_code=125, stderr="no such image"),)
    )
    runner = runner_of(tmp_path, script)

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "inconclusive"
    assert "no such image" in outcome.detail
    # 分類の結果を使わない経路なので、長い末尾は読まない
    assert cleanup_order(runner) == CLEANUP_STEPS_WITHOUT_CLASSIFY


def test_a_refused_gate_is_inconclusive_without_any_mutating_call(tmp_path: Path) -> None:
    """関門で断られたら、了承も、状態を変える呼び出しも、1 つも出さない (requirements 2.1)。"""
    config = probe_config(UNUSED_PORT)
    script = ProbeScript(plans=plans_of(config), gpu_apps="1234, python3, 8000 MiB\n")
    runner = runner_of(tmp_path, script)
    confirmer = SpyConfirmer()

    outcome = run_probe(runner, config, tmp_path, confirmer=confirmer)

    assert outcome.status == "inconclusive"
    assert "python3" in outcome.detail
    assert confirmer.shown == []
    assert mutating_calls(runner) == ()


def test_declining_the_approval_makes_no_mutating_call(tmp_path: Path) -> None:
    """了承しなければ、状態を変える呼び出しが 1 つも出ない (requirements 2.1)。"""
    config = probe_config(UNUSED_PORT)
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))

    with pytest.raises(ApprovalError):
        run_probe(runner, config, tmp_path, confirmer=SpyConfirmer(approves=False))

    assert mutating_calls(runner) == ()
    assert runner.pushes == ()
    assert argv_of(runner, "docker", "run") == ()


# --- Spark に触る前に断る --------------------------------------------------


def test_a_two_node_config_is_refused_before_touching_spark(tmp_path: Path) -> None:
    """縮小の確認は 1 台だけで行う (2 台以上の構成は、Spark に触る前に断る)。"""
    config = probe_config(UNUSED_PORT, nodes=("head", "worker"))
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))

    with pytest.raises(ConfigError, match="1 台"):
        run_probe(runner, config, tmp_path)

    assert runner.calls == ()


def test_a_non_probe_config_is_refused_before_touching_spark(tmp_path: Path) -> None:
    """`kind` が `probe` でない構成は、Spark に触る前に断る。"""
    config = probe_config(UNUSED_PORT, kind="serve")
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))

    with pytest.raises(ConfigError, match="probe"):
        run_probe(runner, config, tmp_path)

    assert runner.calls == ()


def test_a_relative_record_dir_is_refused_before_touching_spark(tmp_path: Path) -> None:
    """`record_dir` は絶対の道筋だけを受ける (配る元を、配る前に空にするため)。"""
    config = probe_config(UNUSED_PORT)
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))

    with pytest.raises(ValueError, match="絶対"):
        pr.run_probe(
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
            report=io.StringIO(),
        )

    assert runner.calls == ()


# --- 中断 (requirements 1.5、2.3) ------------------------------------------


def test_an_interrupt_while_waiting_still_removes_the_container(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """待ちの途中の中断でも、片付けまで進んでから、中断が伝わる。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = probe_config(port_of(fake_vllm))
    plans = plans_of(config)
    runner = runner_of(tmp_path, ProbeScript(plans=plans))

    def interrupt() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_probe(runner, config, tmp_path, clock=FakeClock(on_sleep=interrupt))

    assert cleanup_order(runner) == CLEANUP_STEPS_WITHOUT_CLASSIFY
    assert argv_of(runner, "docker", "rm") == (remove_argv(plans[0].container_name),)


def test_an_interrupt_during_the_short_request_still_removes_the_container(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """短い要求の最中の中断でも、コンテナが残らず、中断が伝わる。"""
    config = probe_config(port_of(fake_vllm))
    plans = plans_of(config)
    runner = runner_of(tmp_path, ProbeScript(plans=plans))
    client = httpx.Client(transport=InterruptingTransport())
    try:
        with pytest.raises(KeyboardInterrupt):
            run_probe(runner, config, tmp_path, smoke_client=client)
    finally:
        client.close()

    assert argv_of(runner, "docker", "stop") == (stop_argv(plans[0].container_name),)
    assert argv_of(runner, "docker", "rm") == (remove_argv(plans[0].container_name),)


def test_an_interrupt_during_the_cleanup_still_removes_the_container(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """片付けの最中に中断が来ても、`rm` まで試みてから、中断が伝わる。"""
    config = probe_config(port_of(fake_vllm))
    plans = plans_of(config)
    script = ProbeScript(plans=plans, stops=(Reply(raises=KeyboardInterrupt()),))
    runner = runner_of(tmp_path, script)

    with pytest.raises(KeyboardInterrupt):
        run_probe(runner, config, tmp_path)

    assert argv_of(runner, "docker", "stop") == (stop_argv(plans[0].container_name),)
    assert argv_of(runner, "docker", "rm") == (remove_argv(plans[0].container_name),)


class InterruptingTransport(httpx.BaseTransport):
    """短い要求の最中の中断を作る (実物のサーバーにはつながない)。"""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise KeyboardInterrupt


# --- 片付けが終わらなかったとき --------------------------------------------


def test_a_cleanup_problem_after_a_ready_probe_becomes_an_error(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """止められなかったら、起動していても誤りにする (コンテナが残ったかもしれない)。"""
    config = probe_config(port_of(fake_vllm))
    plans = plans_of(config)
    script = ProbeScript(plans=plans, stops=(Reply(exit_code=1, stderr="permission denied"),))
    runner = runner_of(tmp_path, script)

    with pytest.raises(pr.ProbeError) as caught:
        run_probe(runner, config, tmp_path)

    assert plans[0].container_name in str(caught.value)
    assert "permission denied" in str(caught.value)


def test_a_cleanup_problem_after_a_failure_keeps_the_kind_of_failure(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """「失敗した」は、片付けが終わらなくても結果を返す (終了コードは、どちらも 2)。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = probe_config(port_of(fake_vllm))
    script = ProbeScript(
        plans=plans_of(config),
        states=(Reply(stdout="exited 1\n"),),
        stops=(Reply(exit_code=1, stderr="permission denied"),),
        long_tails=deep_log(PE_DIM_MARK),
    )
    runner = runner_of(tmp_path, script)

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "failed"
    assert outcome.observation is not None
    assert outcome.observation.known_failure == KnownFailure.PE_DIM_ASSERT
    assert "permission denied" in outcome.detail


# --- 安全の決まり (requirements 2.3、2.4) ----------------------------------


def test_container_operations_target_only_listed_ids_or_planned_names(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """コンテナを対象にする操作は、自分の一覧の ID か、了承済みの計画の名前だけを向く。"""
    fake_vllm.set_health_fault(Fault(status=503))
    config = probe_config(port_of(fake_vllm))
    plans = plans_of(config)
    script = ProbeScript(plans=plans, states=(Reply(stdout="exited 1\n"),))
    runner = runner_of(tmp_path, script)

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "failed"
    allowed = {CONTAINER_ID, *(plan.container_name for plan in plans)}
    targeting = ("logs", "stop", "rm", "container")
    for argv in runner.argvs:
        if argv[0] == "docker" and argv[1] in targeting:
            # どの形でも、対象は末尾の 1 語である
            assert argv[-1] in allowed, argv


def test_mutating_calls_match_the_shown_plan(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """了承のあとの、状態を変える呼び出しが、見せた計画の列と完全に一致する。

    縮小の確認は、成功でも片付ける (requirements 5.4) ので、巻き戻しの列まで含めて一致する。
    """
    config = probe_config(port_of(fake_vllm))
    # `default=Reply()` で、計画にない呼び出しを隠さない (台本の穴で通ってしまわないように)
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)), default=Reply())

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "ready", outcome.detail
    plan = runner.plan
    assert plan is not None
    shown = [
        (command.node, getattr(command, "argv", ()), getattr(command, "remote_subdir", None))
        for command in (*plan.forward, *plan.rollback)
    ]
    happened = [(call.node, call.argv, call.remote) for call in mutating_calls(runner)]
    assert happened == shown


def test_the_probe_reads_the_record_of_the_probe_files_only(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """重みの関門は、`probe_files` の範囲の記録だけを読む (tasks.md の Notes 2.3)。"""
    config = probe_config(port_of(fake_vllm))
    runner = runner_of(tmp_path, ProbeScript(plans=plans_of(config)))

    outcome = run_probe(runner, config, tmp_path)

    assert outcome.status == "ready", outcome.detail
    assert argv_of(runner, "cat") == (("cat", PROBE_RECORD_PATH),)


# --- 依存の向きと、公開の口だけで書くこと ----------------------------------


def _lifecycle_aliases(tree: ast.Module) -> set[str]:
    """`probe.py` の中で、`lifecycle` を指している名前を集める。"""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "serving_kit":
            found.update(
                alias.asname or alias.name for alias in node.names if alias.name == "lifecycle"
            )
        if isinstance(node, ast.Import):
            found.update(
                alias.asname or alias.name.rsplit(".", 1)[-1]
                for alias in node.names
                if alias.name == "serving_kit.lifecycle"
            )
    return found


def probe_tree() -> ast.Module:
    return ast.parse(Path(pr.__file__).read_text(encoding="utf-8"))


def test_probe_touches_no_private_name_of_lifecycle() -> None:
    """`probe` は、`lifecycle` の公開の口だけで書く (`lifecycle.py` を直せないため)。"""
    tree = probe_tree()
    aliases = _lifecycle_aliases(tree)
    assert aliases, "probe.py が lifecycle を module として読み込んでいない"
    touched: list[str] = []
    private: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "serving_kit.lifecycle":
            private.extend(
                f"from serving_kit.lifecycle import {alias.name}"
                for alias in node.names
                if alias.name.startswith("_")
            )
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id not in aliases:
                continue
            touched.append(node.attr)
            if node.attr.startswith("_"):
                private.append(f"{node.value.id}.{node.attr}")
    assert private == []
    assert touched, "probe.py が lifecycle の口を 1 つも使っていない"


def test_probe_imports_only_the_allowed_modules() -> None:
    """依存の向き: `probe` は `netcheck` / `watch` / `thinking` / `cli` を読み込まない。"""
    forbidden = {"netcheck", "watch", "thinking", "cli"}
    imported: set[str] = set()
    for node in ast.walk(probe_tree()):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("serving_kit"):
            tail = (node.module or "").split(".")[1:]
            imported.update(tail)
            imported.update(alias.name for alias in node.names)
        if isinstance(node, ast.Import):
            imported.update(
                alias.name.split(".")[1]
                for alias in node.names
                if alias.name.startswith("serving_kit.")
            )
    assert imported & forbidden == set()
