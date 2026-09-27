"""安全の決まりの試験 (5.3) が、`test_safety.py` と `test_probe_watch.py` の両方から使う下ごしらえ。

`test_start_stop.py` (5.2) の作り方 (`_header` / `_setting`、`Repo`、`invoke()`) を見本にして、
5 つの `kind` すべての構成 (`serve`、`probe`、`fetch`、`inspect`、`job`) を持つリポジトリを
組み立てられるようにした。`test_start_stop.py` は書き換えない (このファイルは、その隣に置く
新規のモジュールである)。

置き場所の決まりにより、ここに置くのは組み立てのための道具だけで、決まりを確かめる assert は
`test_safety.py` / `test_probe_watch.py` の側に書く。
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from fake_runner import FakeRunner, Reply, Rule
from fake_vllm import FakeVllm
from meminfo_sample import meminfo_rule
from serving_kit import cli
from serving_kit import config as config_mod
from serving_kit import guards as g
from serving_kit import plan as plan_mod
from serving_kit import weights as weights_mod
from serving_kit.plan import LABEL_IMAGE, OWNER_FILTER
from serving_kit.types import (
    ConfigDef,
    ContainerPlan,
    ManifestFile,
    NodeDef,
    NodeRole,
    VerificationRecord,
    WeightsManifest,
)

# --- 見本の値 (fixture データであり、ほかの試験ファイルの値と揃えても問題ない) --------------

DIGEST = "b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
IMAGE_REF = f"vllm/vllm-openai@sha256:{DIGEST}"
REPO = "RedHatAI/GLM-5.3-Flash-NVFP4"
REVISION = "18d55bfd" + "0" * 32
SLUG = "RedHatAI__GLM-5.3-Flash-NVFP4"
MANIFEST_NAME = f"{SLUG}.manifest.json"
REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"
SOURCE = "https://docs.vllm.ai/en/latest/cli/serve/"
QUOTE = "vllm serve [model_tag] [options]"
MEASURED = "docs/results/e2e-safety-netcheck-links.md"
FABRIC_IFNAME = "enp1s0f0np0"
SERVED_MODEL = "glm-5-3-flash"

ROLES: tuple[NodeRole, ...] = ("head", "worker")

SERVE_CONFIG = "e2e-serve"
PROBE_CONFIG = "e2e-probe"
FETCH_CONFIG = "e2e-fetch"
INSPECT_CONFIG = "e2e-inspect"
JOB_CONFIG = "e2e-job"

REPO_COMMIT = "e2efeed" + "0" * 33
UNUSED_PORT = 8123
"""HTTP に進まない試験 (関門の断り、了承の拒否) が使う、偽の推論サーバーのない番号。"""

RDZV_PORT = 29502
"""torchrun の rendezvous のポート (推論サーバーの --port / --master-port とは別の値)。"""

CONTAINER_IDS: Mapping[NodeRole, str] = {"head": "0123456789ab", "worker": "cdef01234567"}

PLAN_BUILD_TIME = datetime(2026, 1, 1, tzinfo=UTC)
VERIFIED_AT = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)

FILE_SIZES: Mapping[str, int] = {
    "config.json": 4_096,
    "tokenizer.json": 36_000_000,
    "model-00001-of-00001.safetensors": 98_000_000_000,
}

MANIFEST = WeightsManifest(
    repo=REPO,
    revision=REVISION,
    generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    total_bytes=sum(FILE_SIZES.values()),
    files=tuple(
        ManifestFile(path=path, size=size, sha256=f"{index:064x}")
        for index, (path, size) in enumerate(sorted(FILE_SIZES.items()))
    ),
)

OWN_CONTAINERS_ARGV: tuple[str, ...] = (
    "docker",
    "ps",
    "-a",
    "--filter",
    f"label={OWNER_FILTER}",
    "--format",
    "json",
)
"""`guards.list_own_containers` が流す、ただ 1 つのコンテナの一覧の読み取り。"""

DF_AVAIL = "Avail\n999999999999999\n"


# --- 構成の TOML を組み立てる (test_cli.py の _header / _setting の作り方を見て、独立に書いた) --


def _toml_value(text: str) -> str:
    """TOML の文字列にする (`"` を含む値は、リテラル文字列で書く)。"""
    if '"' in text and "'" not in text:
        return f"'{text}'"
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _setting(
    table: str,
    *,
    flag: str | None = None,
    value: str | None = None,
    why: str = "端から端までの安全の試験のための、最小の設定",
    only_on: str | None = None,
    is_port: bool = False,
    measured: bool = False,
) -> str:
    lines = [f"[{table}]"]
    if flag is not None:
        lines.append(f"flag = {_toml_value(flag)}")
    if value is not None:
        lines.append(f"value = {_toml_value(value)}")
    lines.append(f'why = "{why}"')
    if only_on is not None:
        lines.append(f'only_on = "{only_on}"')
    if is_port:
        lines.append("is_port = true")
    if measured:
        lines.append(f'measured = "{MEASURED}"')
    else:
        lines.append(f'source = "{SOURCE}"')
        lines.append(f'quote = "{QUOTE}"')
    return "\n".join(lines) + "\n\n"


def _header(
    name: str,
    kind: str,
    nodes: Sequence[str],
    *,
    served: str | None = None,
    weights: bool = True,
    mount_at: str = f"/models/{SLUG}",
) -> str:
    lines = [
        f"[configs.{name}]",
        f'kind = "{kind}"',
        f'description = "安全の試験用の構成 ({kind})"',
        "nodes = [" + ", ".join(f'"{role}"' for role in nodes) + "]",
        "ready_timeout_s = 1800",
    ]
    if served is not None:
        lines.append(f'served_model_name = "{served}"')
    text = "\n".join(lines) + "\n\n"
    text += (
        "\n".join(
            [
                f"[configs.{name}.image]",
                f'ref = "{IMAGE_REF}"',
                'seen_as = "vllm/vllm-openai:e2e-safety"',
                "size_bytes = 9666567584",
                'source = "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags"',
                'quote = "glm53-flash-arm64-cu130 / linux/arm64 / 9666567584 B"',
            ]
        )
        + "\n\n"
    )
    if weights:
        text += (
            "\n".join(
                [
                    f"[configs.{name}.weights]",
                    f'repo = "{REPO}"',
                    f'revision = "{REVISION}"',
                    f'manifest = "{MANIFEST_NAME}"',
                    f'mount_at = "{mount_at}"',
                    f'source = "https://huggingface.co/api/models/{REPO}"',
                    'quote = "license: mit"',
                ]
            )
            + "\n\n"
        )
    return text


def serve_config_toml(port: int) -> str:
    name = SERVE_CONFIG
    text = _header(name, "serve", ROLES, served=SERVED_MODEL)
    text += _setting(
        f"configs.{name}.docker.models",
        flag="--mount",
        value="type=bind,source={remote_root}/models/" + SLUG + ",target={weights.mount_at}"
        ",readonly",
    )
    text += _setting(
        f"configs.{name}.docker.cache",
        flag="--mount",
        value="type=bind,source={remote_root}/cache,target=/root/.cache",
    )
    text += _setting(f"configs.{name}.args.model-path", value="{weights.mount_at}")
    text += _setting(
        f"configs.{name}.args.served-model-name", flag="--served-model-name", value=SERVED_MODEL
    )
    text += _setting(f"configs.{name}.args.host", flag="--host", value="{head.lan_addr}")
    text += _setting(f"configs.{name}.args.port", flag="--port", value=str(port), is_port=True)
    text += _setting(
        f"configs.{name}.args.master-port", flag="--master-port", value="29501", is_port=True
    )
    text += _setting(
        f"configs.{name}.env.host-ip",
        flag="VLLM_HOST_IP",
        value="{node.fabric_addr}",
        measured=True,
    )
    return text


def probe_config_toml(port: int) -> str:
    name = PROBE_CONFIG
    text = _header(name, "probe", ("head",), served=SERVED_MODEL, mount_at=f"/probe/{SLUG}")
    text += _setting(
        f"configs.{name}.docker.probe",
        flag="--mount",
        value="type=bind,source={remote_root}/probe,target=/probe,readonly",
    )
    text += _setting(f"configs.{name}.args.model-path", value="{weights.mount_at}")
    text += _setting(f"configs.{name}.args.port", flag="--port", value=str(port), is_port=True)
    text += _setting(f"configs.{name}.args.load-format", flag="--load-format", value="dummy")
    return text


def fetch_config_toml() -> str:
    name = FETCH_CONFIG
    text = _header(name, "fetch", ROLES)
    text += _setting(f"configs.{name}.docker.entrypoint", flag="--entrypoint", value="hf")
    text += _setting(
        f"configs.{name}.docker.models",
        flag="--mount",
        value="type=bind,source={remote_root}/models,target=/models",
    )
    text += _setting(f"configs.{name}.args.download", value="download")
    text += _setting(f"configs.{name}.args.repo", value=REPO)
    text += _setting(f"configs.{name}.args.revision", flag="--revision", value=REVISION)
    text += _setting(
        f"configs.{name}.args.local-dir", flag="--local-dir", value="{weights.mount_at}"
    )
    text += _setting(f"configs.{name}.args.max-workers", flag="--max-workers", value="8")
    text += _setting(f"configs.{name}.args.exclude", flag="--exclude", value="README.md")
    text += _setting(f"configs.{name}.env.telemetry", flag="HF_HUB_DISABLE_TELEMETRY", value="1")
    return text


def inspect_config_toml() -> str:
    name = INSPECT_CONFIG
    text = _header(name, "inspect", ("head",), weights=False)
    text += _setting(f"configs.{name}.docker.entrypoint", flag="--entrypoint", value="cat")
    text += _setting(f"configs.{name}.args.license", value="/usr/share/doc/vllm/LICENSE")
    return text


def job_config_toml() -> str:
    name = JOB_CONFIG
    text = _header(name, "job", ROLES, weights=False)
    text += _setting(f"configs.{name}.docker.network", flag="--network", value="host")
    text += _setting(
        f"configs.{name}.docker.logs",
        flag="--mount",
        value="type=bind,source={remote_root}/logs,target=/logs",
    )
    text += _setting(f"configs.{name}.args.script", value="/payload/allreduce_bench.py")
    text += _setting(
        f"configs.{name}.args.master-port",
        flag="--master-port",
        value=str(RDZV_PORT),
        is_port=True,
    )
    text += _setting(f"configs.{name}.env.nccl-debug", flag="NCCL_DEBUG", value="INFO")
    text += _setting(
        f"configs.{name}.env.nccl-debug-subsys", flag="NCCL_DEBUG_SUBSYS", value="INIT,NET"
    )
    text += _setting(
        f"configs.{name}.env.nccl-debug-file", flag="NCCL_DEBUG_FILE", value="/logs/nccl.log"
    )
    return text


def nodes_toml() -> str:
    return f"""\
[nodes.head]
ssh_host = "e2e-head"
lan_addr = "127.0.0.1"
remote_root = "{REMOTE_ROOT}"
fabric_addr = "192.168.100.1"
fabric_ifname = "{FABRIC_IFNAME}"
fabric_measured = "{MEASURED}"

[nodes.worker]
ssh_host = "e2e-worker"
lan_addr = "127.0.0.1"
remote_root = "{REMOTE_ROOT}"
fabric_addr = "192.168.100.2"
fabric_ifname = "{FABRIC_IFNAME}"
fabric_measured = "{MEASURED}"
"""


# --- リポジトリ --------------------------------------------------------------


@dataclass(frozen=True)
class Repo:
    """試験用のリポジトリ (`serving/config/`、`serving/weights/`、`serving/payload/` を持つ)。"""

    root: Path

    @property
    def serving(self) -> Path:
        return self.root / "serving"

    @property
    def configs(self) -> Path:
        return self.serving / "config" / "configs.toml"

    @property
    def nodes(self) -> Path:
        return self.serving / "config" / "nodes.toml"

    @property
    def var_root(self) -> Path:
        return self.serving / "var"

    @property
    def payload(self) -> Path:
        return self.serving / "payload"

    @property
    def manifest(self) -> Path:
        return self.serving / "weights" / MANIFEST_NAME


def make_repo(tmp_path: Path, *, port: int) -> Repo:
    """5 つの `kind` すべての構成を持つ、最小のリポジトリを作る。"""
    root = tmp_path / "repo"
    (root / "docs" / "results").mkdir(parents=True, exist_ok=True)
    (root / MEASURED).write_text(
        "端から端までの安全の試験用の、実測の記録の見本\n", encoding="utf-8"
    )
    (root / "serving" / "config").mkdir(parents=True, exist_ok=True)
    (root / "serving" / "weights").mkdir(parents=True, exist_ok=True)
    (root / "serving" / "payload").mkdir(parents=True, exist_ok=True)
    (root / "serving" / "payload" / "allreduce_bench.py").write_text(
        "# 安全の試験のための見本\n", encoding="utf-8"
    )
    repo = Repo(root=root)
    repo.manifest.write_bytes(weights_mod.to_json_bytes(MANIFEST))
    text = "schema_version = 1\n\n"
    text += serve_config_toml(port)
    text += probe_config_toml(port)
    text += fetch_config_toml()
    text += inspect_config_toml()
    text += job_config_toml()
    repo.configs.write_text(text, encoding="utf-8")
    repo.nodes.write_text(nodes_toml(), encoding="utf-8")
    return repo


def load_config_and_nodes(repo: Repo, name: str) -> tuple[ConfigDef, dict[NodeRole, NodeDef]]:
    """`cli` が内部で読むのと同じ道 (台本を作るために、先に一度読む)。"""
    nodes = config_mod.load_nodes(repo.nodes, repo.root)
    configs = config_mod.load_configs(repo.configs, repo.root)
    config = config_mod.select_config(configs, name, nodes)
    return config, nodes


def container_plans(
    config: ConfigDef, nodes: Mapping[NodeRole, NodeDef], *, started_at: datetime = PLAN_BUILD_TIME
) -> dict[NodeRole, ContainerPlan]:
    plans = plan_mod.build_plans(config, nodes, started_at)
    return {plan.node: plan for plan in plans}


# --- 台本の部品 --------------------------------------------------------------


def ps_row(plan: ContainerPlan, *, container_id: str, state: str = "running") -> str:
    """`docker ps -a --filter label=… --format json` の 1 行 (自分のコンテナ)。"""
    return (
        json.dumps(
            {
                "ID": container_id,
                "Names": plan.container_name,
                "State": state,
                "Image": plan.labels.get(LABEL_IMAGE, IMAGE_REF),
                "Labels": ",".join(f"{key}={value}" for key, value in sorted(plan.labels.items())),
            }
        )
        + "\n"
    )


def record_json(role: NodeRole, scope: str, *, repo: str = REPO, revision: str = REVISION) -> str:
    """`gate_weights_verified` が読む、照合の結果の記録。"""
    files = MANIFEST.probe_files if scope == "probe_files" else MANIFEST.files
    return VerificationRecord(
        repo=repo,
        revision=revision,
        scope=scope,  # type: ignore[arg-type]
        node=role,
        verified_at=VERIFIED_AT,
        file_count=len(files),
        total_bytes=sum(entry.size for entry in files),
    ).model_dump_json()


def gate_rules(
    roles: Sequence[NodeRole] = ROLES,
    *,
    listing: Mapping[NodeRole, str] | None = None,
    layout_ok: bool = True,
    digests: Sequence[str] = (IMAGE_REF,),
    avail: str = DF_AVAIL,
    listening: str = "",
) -> tuple[Rule, ...]:
    """9 つの関門のうち、読み取りを要るものを、すべて通す台本 (`test_cli.py` の `gate_rules`
    と同じ考え方: 重みを持つ構成では、`cat` の 2 つの規則で、`.probe.verified.json` かどうかで
    読み分ける。`/proc/meminfo` の規則は、それらより前に置く)。
    """
    rows = {} if listing is None else listing
    rules: list[Rule] = []
    for role in roles:
        rules.append(
            Rule(prefix=OWN_CONTAINERS_ARGV, node=role, replies=(Reply(stdout=rows.get(role, "")),))
        )
        rules.append(meminfo_rule(node=role))
        rules.append(
            Rule(
                prefix=("cat",),
                node=role,
                when=lambda argv: argv[-1].endswith(".probe.verified.json"),
                replies=(Reply(stdout=record_json(role, "probe_files")),),
            )
        )
        rules.append(
            Rule(prefix=("cat",), node=role, replies=(Reply(stdout=record_json(role, "all")),))
        )
    rules.extend(
        (
            Rule(prefix=("uname",), replies=(Reply(stdout="e2e-host\n"),)),
            Rule(prefix=("nvidia-smi",), replies=(Reply(stdout=""),)),
            Rule(prefix=("test", "-e"), replies=(Reply(exit_code=0 if layout_ok else 1),)),
            Rule(
                prefix=("docker", "image", "inspect"),
                replies=(Reply(stdout=json.dumps(list(digests)) + "\n"),),
            ),
            Rule(prefix=("df",), replies=(Reply(stdout=avail),)),
            Rule(prefix=("ss",), replies=(Reply(stdout=listening),)),
        )
    )
    return tuple(rules)


# --- `cli.main` の 1 回の呼び出し --------------------------------------------


@dataclass
class Invocation:
    code: int
    out: str
    err: str
    runner: FakeRunner


def never_sleep(seconds: float) -> None:
    """既定は、実際に眠らないはずという前提を固定する (決めごとの検証)。"""
    raise AssertionError(f"このシナリオは眠らないはずなのに、{seconds} 秒眠ろうとした")


def invoke(
    argv: Sequence[str],
    repo: Repo,
    *,
    script: Sequence[Rule] = (),
    default: Reply | None = None,
    confirmer: g.Confirmer | None = None,
    stdin: str = "",
    client_factory: Callable[[float], httpx.Client] | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    now: Callable[[], datetime] | None = None,
    expect_runner: bool = True,
) -> Invocation:
    """`cli.main` を、共通の下ごしらえ (構成、ノード、記録の置き場所) を足して 1 回呼ぶ。

    差し込むのは `cli.main` 自身の口だけ (`runner_factory`、`client_factory`、`confirmer`、
    `stdin`/`stdout`/`stderr`、`sleep`、`clock`、`now`)。`serving_kit` の部品を `monkeypatch` で
    置き換えることはしない。
    """
    made: list[FakeRunner] = []

    def runner_factory(var_root: Path) -> FakeRunner:
        runner = FakeRunner(var_root=var_root, script=tuple(script), default=default)
        made.append(runner)
        return runner

    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(
        [
            *argv,
            "--configs",
            str(repo.configs),
            "--nodes",
            str(repo.nodes),
            "--var-root",
            str(repo.var_root),
        ],
        runner_factory=runner_factory,
        client_factory=client_factory,
        confirmer=confirmer,
        stdin=io.StringIO(stdin),
        stdout=out,
        stderr=err,
        repo_root=repo.root,
        repo_facts=(REPO_COMMIT, False),
        now=(lambda: datetime.now(UTC)) if now is None else now,
        sleep=never_sleep if sleep is None else sleep,
        clock=clock,
    )
    if expect_runner:
        assert made, "実行役が 1 つも作られていない (runner_factory が呼ばれなかった)"
        runner = made[0]
    else:
        assert not made, "Spark に触らないはずのコマンドが、実行役を作った"
        runner = FakeRunner(var_root=repo.var_root)
    return Invocation(code=code, out=out.getvalue(), err=err.getvalue(), runner=runner)


def port_of(fake: FakeVllm) -> int:
    """偽の推論サーバーが待ち受けている番号 (`fake.base_url` の末尾)。"""
    return int(fake.base_url.rsplit(":", 1)[1])


def kv(out: str) -> dict[str, str]:
    """標準出力の `key=value` の行を読む (表の行は飛ばす)。"""
    pairs: dict[str, str] = {}
    for line in out.splitlines():
        if not line or line.startswith("|"):
            continue
        assert "=" in line, f"決まった形でない標準出力の行: {line!r}"
        key, value = line.split("=", 1)
        pairs[key] = value
    return pairs


def mutating_calls(runner: FakeRunner) -> tuple[str, ...]:
    """状態を変えた呼び出しの種類だけを、呼ばれた順に並べる (`kind` か `argv[:2]`)。"""
    names: list[str] = []
    for call in runner.calls:
        if not call.mutating:
            continue
        names.append(call.kind if call.kind != "run" else " ".join(call.argv[:2]))
    return tuple(names)


def files_under(root: Path) -> list[Path]:
    """`root` の下のファイルを、すべて並べる (存在しなければ空)。"""
    if not root.is_dir():
        return []
    return [path for path in root.rglob("*") if path.is_file()]
