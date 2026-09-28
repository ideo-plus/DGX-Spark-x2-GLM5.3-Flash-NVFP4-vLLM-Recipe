"""`serve autostart set|clear|status` の試験 (計画の完了契約 C7、C8)。

`serving_kit.autostart` と `cli.py` の配線を、`FakeRunner` で確かめる (`test_cli.py:396-415`
の `make_repo`/`run` と同じ考え方だが、`test_cli.py` を肥大させないため、自動起動の指定に
要る最小の構成だけを持つ、この試験専用のリポジトリを新規に組み立てる)。

要求シナリオ (gherkin) の対象外の契約なので、計画の完了契約表の「成立する振る舞い」列を P、
「拒否すべき誤実装」列を N として、1 対 1 でテストに落とす。

- C7-P → test_set_pushes_the_designation_to_both_nodes_when_launch_records_match
- C7-N → test_set_pushes_nothing_when_a_launch_record_is_missing_or_mismatched
- C8-P → test_status_only_reads_both_files_from_both_nodes
- C8-P → test_clear_pushes_a_null_config_designation_to_both_nodes
- C8-N → test_status_never_makes_a_mutating_call
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fake_runner import FakeRunner, Reply, Rule
from serving_kit import cli, guards
from serving_kit import config as config_mod
from serving_kit import plan as plan_mod
from serving_kit import weights as weights_mod
from serving_kit.types import (
    ContainerPlan,
    ConversionSpec,
    Derivation,
    DerivedWeightsManifest,
    LaunchRecord,
    ManifestFile,
    NodeRole,
    WeightsManifest,
    WeightsOrigin,
)

IMAGE_REF = "vllm/vllm-openai@sha256:" + "b0" * 32
SOURCE = "https://docs.vllm.ai/en/latest/cli/serve/"
QUOTE = "vllm serve [model_tag] [options]"
REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"
CONFIG_NAME = "auto-cfg"
SERVED_MODEL = "auto-model"
REPO_COMMIT = "d" * 40
NOW = datetime(2026, 9, 27, 8, 0, 0, tzinfo=UTC)

WEIGHTED_CONFIG_NAME = "auto-cfg-weighted"
WEIGHTS_REPO = "RedHatAI/GLM-5.3-Flash-NVFP4"
WEIGHTS_REVISION = "18d55bfd" + "0" * 32
WEIGHTS_MANIFEST_FILENAME = "auto-cfg-weighted.manifest.json"
WEIGHTS_MOUNT_AT = "/models/auto-cfg-weighted"
WEIGHTS_FILE_SIZES = {"config.json": 4096, "model-00001-of-00001.safetensors": 98_000_000_000}
WEIGHTS_MANIFEST = WeightsManifest(
    repo=WEIGHTS_REPO,
    revision=WEIGHTS_REVISION,
    generated_at=NOW,
    total_bytes=sum(WEIGHTS_FILE_SIZES.values()),
    files=tuple(
        ManifestFile(path=path, size=size, sha256=f"{index:064x}")
        for index, (path, size) in enumerate(sorted(WEIGHTS_FILE_SIZES.items()))
    ),
)

NON_SERVE_CONFIG_NAME = "auto-cfg-inspect"

DERIVED_CONFIG_NAME = "auto-cfg-derived"
DERIVED_WEIGHTS_NAME = "k2s1"
DERIVED_MANIFEST_FILENAME = "auto-cfg-derived.manifest.json"
DERIVED_MOUNT_AT = f"/models/{DERIVED_WEIGHTS_NAME}"
DERIVED_TOOL = "experiments/k2-quant/convert.py"
DERIVED_COMMIT = "c" * 40
DERIVED_TARGET_PATTERN = r"^model\.layers\.\d+\.self_attn\..*$"
DERIVED_WEIGHTS_MANIFEST = DerivedWeightsManifest(
    kind="derived",
    derivation=Derivation(
        name=DERIVED_WEIGHTS_NAME,
        origin=WeightsOrigin(repo=WEIGHTS_REPO, revision=WEIGHTS_REVISION),
        conversion=ConversionSpec(
            tool=DERIVED_TOOL,
            commit=DERIVED_COMMIT,
            args=("--dtype", "fp8"),
            target_pattern=DERIVED_TARGET_PATTERN,
        ),
    ),
    generated_at=NOW,
    total_bytes=WEIGHTS_MANIFEST.total_bytes,
    files=WEIGHTS_MANIFEST.files,
)


CONFIGS_TOML = f"""\
schema_version = 1

[configs.{CONFIG_NAME}]
kind = "serve"
description = "自動起動の指定の試験用の構成"
nodes = ["head", "worker"]
ready_timeout_s = 1800
served_model_name = "{SERVED_MODEL}"

[configs.{CONFIG_NAME}.image]
ref = "{IMAGE_REF}"
seen_as = "vllm/vllm-openai:試験"
size_bytes = 1024
source = "{SOURCE}"
quote = "見本"

[configs.{CONFIG_NAME}.docker]

[configs.{CONFIG_NAME}.args.port]
flag = "--port"
value = "8000"
is_port = true
why = "待ち受けのポート"
source = "{SOURCE}"
quote = "{QUOTE}"

[configs.{WEIGHTED_CONFIG_NAME}]
kind = "serve"
description = "自動起動の指定の試験用の構成 (重みを持つ)"
nodes = ["head", "worker"]
ready_timeout_s = 1800
served_model_name = "{SERVED_MODEL}-w"

[configs.{WEIGHTED_CONFIG_NAME}.image]
ref = "{IMAGE_REF}"
seen_as = "vllm/vllm-openai:試験"
size_bytes = 1024
source = "{SOURCE}"
quote = "見本"

[configs.{WEIGHTED_CONFIG_NAME}.weights]
repo = "{WEIGHTS_REPO}"
revision = "{WEIGHTS_REVISION}"
manifest = "{WEIGHTS_MANIFEST_FILENAME}"
mount_at = "{WEIGHTS_MOUNT_AT}"
source = "https://huggingface.co/{WEIGHTS_REPO}"
quote = "license: mit"

[configs.{WEIGHTED_CONFIG_NAME}.docker]

[configs.{WEIGHTED_CONFIG_NAME}.args.port]
flag = "--port"
value = "8000"
is_port = true
why = "待ち受けのポート"
source = "{SOURCE}"
quote = "{QUOTE}"

[configs.{DERIVED_CONFIG_NAME}]
kind = "serve"
description = "自動起動の指定の試験用の構成 (派生の重みを持つ)"
nodes = ["head", "worker"]
ready_timeout_s = 1800
served_model_name = "{SERVED_MODEL}-d"

[configs.{DERIVED_CONFIG_NAME}.image]
ref = "{IMAGE_REF}"
seen_as = "vllm/vllm-openai:試験"
size_bytes = 1024
source = "{SOURCE}"
quote = "見本"

[configs.{DERIVED_CONFIG_NAME}.weights]
kind = "derived"
name = "{DERIVED_WEIGHTS_NAME}"
manifest = "{DERIVED_MANIFEST_FILENAME}"
mount_at = "{DERIVED_MOUNT_AT}"

[configs.{DERIVED_CONFIG_NAME}.weights.origin]
repo = "{WEIGHTS_REPO}"
revision = "{WEIGHTS_REVISION}"
manifest = "{WEIGHTS_MANIFEST_FILENAME}"

[configs.{DERIVED_CONFIG_NAME}.weights.conversion]
tool = "{DERIVED_TOOL}"
commit = "{DERIVED_COMMIT}"
args = ["--dtype", "fp8"]
target_pattern = '{DERIVED_TARGET_PATTERN}'

[configs.{DERIVED_CONFIG_NAME}.docker]

[configs.{DERIVED_CONFIG_NAME}.args.port]
flag = "--port"
value = "8000"
is_port = true
why = "待ち受けのポート"
source = "{SOURCE}"
quote = "{QUOTE}"

[configs.{NON_SERVE_CONFIG_NAME}]
kind = "inspect"
description = "自動起動の指定の試験用の構成 (kind が serve でない)"
nodes = ["head"]
ready_timeout_s = 1800

[configs.{NON_SERVE_CONFIG_NAME}.image]
ref = "{IMAGE_REF}"
seen_as = "vllm/vllm-openai:試験"
size_bytes = 1024
source = "{SOURCE}"
quote = "見本"

[configs.{NON_SERVE_CONFIG_NAME}.docker]

[configs.{NON_SERVE_CONFIG_NAME}.args.port]
flag = "--port"
value = "8000"
is_port = true
why = "待ち受けのポート"
source = "{SOURCE}"
quote = "{QUOTE}"
"""

MEASURED = "docs/results/autostart-test-netcheck-links.md"

NODES_TOML = f"""\
[nodes.head]
ssh_host = "spark-153d"
lan_addr = "127.0.0.1"
remote_root = "{REMOTE_ROOT}"
fabric_addr = "192.168.100.1"
fabric_ifname = "enp1s0f0np0"
fabric_measured = "{MEASURED}"

[nodes.worker]
ssh_host = "spark-5083"
lan_addr = "127.0.0.1"
remote_root = "{REMOTE_ROOT}"
fabric_addr = "192.168.100.2"
fabric_ifname = "enp1s0f0np0"
fabric_measured = "{MEASURED}"
"""


@dataclass(frozen=True)
class Repo:
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


def make_repo(tmp_path: Path) -> Repo:
    root = tmp_path / "repo"
    (root / "serving" / "config").mkdir(parents=True, exist_ok=True)
    (root / "serving" / "weights").mkdir(parents=True, exist_ok=True)
    (root / MEASURED).parent.mkdir(parents=True, exist_ok=True)
    (root / MEASURED).write_text("直結のインターフェースの実測の要約 (試験用)\n", encoding="utf-8")
    repo = Repo(root=root)
    repo.configs.write_text(CONFIGS_TOML, encoding="utf-8")
    repo.nodes.write_text(NODES_TOML, encoding="utf-8")
    weights_mod.write_manifest(
        WEIGHTS_MANIFEST, root / "serving" / "weights" / WEIGHTS_MANIFEST_FILENAME
    )
    weights_mod.write_manifest(
        DERIVED_WEIGHTS_MANIFEST, root / "serving" / "weights" / DERIVED_MANIFEST_FILENAME
    )
    return repo


def _built_plans(repo: Repo, *, config_name: str = CONFIG_NAME) -> dict[NodeRole, ContainerPlan]:
    """この試験の構成から、実際に組み立てたコンテナの計画 (Mac 側の正解の sha を持つ)。"""
    nodes = config_mod.load_nodes(repo.nodes, repo.root)
    configs = config_mod.load_configs(repo.configs, repo.root)
    config = config_mod.select_config(configs, config_name, nodes)
    plans = plan_mod.build_plans(config, nodes, NOW)
    return {plan.node: plan for plan in plans}


def _launch_record_text(
    plans: Sequence[ContainerPlan],
    *,
    config: str = CONFIG_NAME,
    config_sha256: str | None = None,
) -> str:
    sha = plans[0].labels[plan_mod.LABEL_CONFIG_SHA256] if config_sha256 is None else config_sha256
    record = LaunchRecord(
        config_name=config,
        image_digest=IMAGE_REF,
        weights=None,
        started_at=NOW,
        plans=tuple(plans),
        config_sha256=sha,
        repo_commit=REPO_COMMIT,
        repo_dirty=False,
        gates=(),
    )
    return record.model_dump_json()


@dataclass
class Spy:
    script: tuple[Rule, ...] = ()
    default: Reply | None = Reply()
    made: list[FakeRunner] = field(default_factory=list)

    def __call__(self, var_root: Path) -> FakeRunner:
        runner = FakeRunner(var_root=var_root, script=self.script, default=self.default)
        self.made.append(runner)
        return runner

    @property
    def runner(self) -> FakeRunner:
        assert self.made, "実行役が 1 つも作られていない"
        return self.made[-1]


@dataclass
class Run:
    code: int
    out: str
    err: str
    spy: Spy


def run(argv: Sequence[str], repo: Repo, *, spy: Spy | None = None) -> Run:
    used = Spy() if spy is None else spy
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
        runner_factory=used,
        stdin=io.StringIO("yes\n"),
        stdout=out,
        stderr=err,
        repo_root=repo.root,
        repo_facts=(REPO_COMMIT, False),
        now=lambda: NOW,
        sleep=lambda seconds: (_ for _ in ()).throw(AssertionError(f"眠った: {seconds}")),
    )
    return Run(code=code, out=out.getvalue(), err=err.getvalue(), spy=used)


def kv(out: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line in out.splitlines():
        if not line or line.startswith("|"):
            continue
        assert "=" in line, f"決まった形でない標準出力の行: {line!r}"
        key, value = line.split("=", 1)
        pairs[key] = value
    return pairs


def _launch_record_rule(
    plans: dict[NodeRole, ContainerPlan],
    *,
    config: str = CONFIG_NAME,
    config_sha256: str | None = None,
) -> tuple[Rule, ...]:
    return tuple(
        Rule(
            prefix=("cat",),
            node=role,
            when=lambda argv: argv[-1].endswith(".launch.json"),
            replies=(
                Reply(
                    stdout=_launch_record_text([plan], config=config, config_sha256=config_sha256)
                ),
            ),
        )
        for role, plan in plans.items()
    )


# --- C7: serve autostart set ------------------------------------------------


def test_set_pushes_the_designation_to_both_nodes_when_launch_records_match(tmp_path: Path) -> None:
    """[C7-P] 両台の `launch.json` の `config_sha256` が、Mac の計画と一致するときだけ、
    両台の `state/` へ指定を配る (`payload/` には配らず、`docker` は流さない)。
    """
    repo = make_repo(tmp_path)
    plans = _built_plans(repo)
    spy = Spy(script=_launch_record_rule(plans))

    result = run(["autostart", "set", CONFIG_NAME, "--yes"], repo, spy=spy)

    assert result.code == cli.EXIT_OK, result.err
    runner = result.spy.runner
    assert [call.kind for call in runner.calls] == ["run", "run", "push", "push"]
    assert all(call.argv[0] == "cat" for call in runner.runs)
    assert not any(call.argv[0] == "docker" for call in runner.calls if call.kind == "run")
    assert {call.remote for call in runner.pushes} == {"state"}
    assert all(not call.delete for call in runner.pushes)
    assert {call.node for call in runner.pushes} == {"head", "worker"}


def test_set_pushes_nothing_when_a_launch_record_is_missing_or_mismatched(tmp_path: Path) -> None:
    """[C7-N] launch.json の `config_sha256` が Mac の計画と違えば、
    一致を確かめられないので、両台とも 1 度も配らず、状態を変えない (終了コード 1)。
    """
    repo = make_repo(tmp_path)
    plans = _built_plans(repo)
    spy = Spy(script=_launch_record_rule(plans, config_sha256="bb" + "0" * 62))

    result = run(["autostart", "set", CONFIG_NAME, "--yes"], repo, spy=spy)

    assert result.code == cli.EXIT_PRECONDITION
    assert result.spy.runner.pushes == ()


def test_set_includes_the_weights_verification_expectation_for_a_weighted_config(
    tmp_path: Path,
) -> None:
    """[C7-P] 重みを持つ構成では、指定に `weights_record` (照合の記録の道筋、`scope`、
    `file_count`、`total_bytes`、Hub の `repo`/`revision`) を埋め込む
    (`autostart.build_designation` が組み立てる、D5 の `weights_verified` が読む期待値)。
    """
    repo = make_repo(tmp_path)
    nodes = config_mod.load_nodes(repo.nodes, repo.root)
    configs = config_mod.load_configs(repo.configs, repo.root)
    config = config_mod.select_config(configs, WEIGHTED_CONFIG_NAME, nodes)
    plans = _built_plans(repo, config_name=WEIGHTED_CONFIG_NAME)
    spy = Spy(script=_launch_record_rule(plans, config=WEIGHTED_CONFIG_NAME))

    result = run(["autostart", "set", WEIGHTED_CONFIG_NAME, "--yes"], repo, spy=spy)

    assert result.code == cli.EXIT_OK, result.err
    runner = result.spy.runner
    head_push = next(call for call in runner.pushes if call.node == "head")
    assert head_push.local_dir is not None
    written = json.loads((head_push.local_dir / "autostart.json").read_text(encoding="utf-8"))
    assert config.weights is not None
    expected_path = guards.weights_record_path(nodes["head"], config.weights, "all")
    assert written["weights_record"] == {
        "path": expected_path,
        "scope": "all",
        "file_count": len(WEIGHTS_MANIFEST.files),
        "total_bytes": WEIGHTS_MANIFEST.total_bytes,
        "fields": {"repo": WEIGHTS_REPO, "revision": WEIGHTS_REVISION},
    }


def test_set_includes_the_manifest_sha256_expectation_for_a_derived_weights_config(
    tmp_path: Path,
) -> None:
    """[問題10・E-T1] 派生の重み (`kind="derived"`) を持つ構成でも、指定に `weights_record` を
    埋め込む。`fields` は Hub の `repo`/`revision` ではなく `manifest_sha256`
    (変換の結果のマニフェストの `content_sha256`) になる (`autostart._weights_expectation` の
    派生の分岐。この確認をする試験がなかった)。
    """
    repo = make_repo(tmp_path)
    nodes = config_mod.load_nodes(repo.nodes, repo.root)
    configs = config_mod.load_configs(repo.configs, repo.root)
    config = config_mod.select_config(configs, DERIVED_CONFIG_NAME, nodes)
    plans = _built_plans(repo, config_name=DERIVED_CONFIG_NAME)
    spy = Spy(script=_launch_record_rule(plans, config=DERIVED_CONFIG_NAME))

    result = run(["autostart", "set", DERIVED_CONFIG_NAME, "--yes"], repo, spy=spy)

    assert result.code == cli.EXIT_OK, result.err
    runner = result.spy.runner
    head_push = next(call for call in runner.pushes if call.node == "head")
    assert head_push.local_dir is not None
    written = json.loads((head_push.local_dir / "autostart.json").read_text(encoding="utf-8"))
    assert config.weights is not None
    expected_path = guards.weights_record_path(nodes["head"], config.weights, "all")
    assert written["weights_record"] == {
        "path": expected_path,
        "scope": "all",
        "file_count": len(DERIVED_WEIGHTS_MANIFEST.files),
        "total_bytes": DERIVED_WEIGHTS_MANIFEST.total_bytes,
        "fields": {"manifest_sha256": DERIVED_WEIGHTS_MANIFEST.content_sha256},
    }


def test_set_refuses_a_non_serve_kind_config_before_touching_any_node(tmp_path: Path) -> None:
    """[companion 2026-09-28] `set_autostart` の `kind != "serve"` の確認 (要件21: serve 以外の
    構成を自動起動の対象にしない) を運動させる試験がなかったため追加する。修正単位 G
    (問題15) で、この確認を `build_designation` から `set_autostart` の 1 か所に絞ったが、
    その 1 か所自体を確かめる試験は既存に無かった。`ctx.selected()` は kind で構成を
    絞り込まないので、この確認が要件21を守る唯一の関所である。
    """
    repo = make_repo(tmp_path)
    spy = Spy()

    result = run(["autostart", "set", NON_SERVE_CONFIG_NAME, "--yes"], repo, spy=spy)

    assert result.code == cli.EXIT_PRECONDITION, result.err
    assert "kind" in result.err
    assert "serve" in result.err
    assert result.spy.runner.calls == ()
    assert "status=designated" not in result.out


def test_set_fails_and_shows_nothing_designated_when_a_push_to_worker_fails(tmp_path: Path) -> None:
    """[問題18] worker への配布 (`push`) が 0 以外の終了コードで終わると、`push_designations` が
    `AutostartError` を投げ、`cli._EXIT_BY_ERROR` の写しで終了コード `EXIT_FAILED` になる
    (この経路の試験がなかった)。1 度も `status=designated` を出力しない
    (head への配布はすでに済んでいても、`_cmd_autostart_set` の `ctx.show.say` にはたどり着かない)。
    """
    repo = make_repo(tmp_path)
    plans = _built_plans(repo)
    script = _launch_record_rule(plans) + (
        Rule(
            kind="push",
            node="worker",
            replies=(Reply(exit_code=23, stderr="配布できない (試験)"),),
        ),
    )
    spy = Spy(script=script)

    result = run(["autostart", "set", CONFIG_NAME, "--yes"], repo, spy=spy)

    assert result.code == cli.EXIT_FAILED, result.err
    assert "status=designated" not in result.out


def test_clear_fails_and_shows_nothing_cleared_when_a_push_to_worker_fails(tmp_path: Path) -> None:
    """[問題18] `clear` でも、worker への配布が 0 以外の終了コードで終わると、同じく
    `AutostartError` から `EXIT_FAILED` になり、`status=cleared` を出力しない。
    """
    repo = make_repo(tmp_path)
    script = (
        Rule(
            kind="push",
            node="worker",
            replies=(Reply(exit_code=23, stderr="配布できない (試験)"),),
        ),
    )
    spy = Spy(script=script)

    result = run(["autostart", "clear", "--yes"], repo, spy=spy)

    assert result.code == cli.EXIT_FAILED, result.err
    assert "status=cleared" not in result.out


# --- C8: serve autostart status / clear -------------------------------------


_DESIGNATION_JSON = (
    '{"schema_version":1,"role":"head","config":"'
    + CONFIG_NAME
    + '","config_sha256":"'
    + "aa" * 32
    + '","ready_timeout_s":1800,"ports":[8000],"weights_record":null,'
    '"designated_at":"2026-09-27T08:00:00Z","repo_commit":"' + REPO_COMMIT + '","repo_dirty":false}'
)

_STATUS_JSON = (
    '{"schema_version":1,"role":"head","config":"'
    + CONFIG_NAME
    + '","container_name":"vb-'
    + CONFIG_NAME
    + '-head","state":"running","reason":"","updated_at":"2026-09-27T08:00:30Z",'
    '"started_at":"2026-09-27T08:00:00Z","ready_at":"2026-09-27T08:00:20Z",'
    '"container_state":"running","container_exit_code":null,"last_health_status":200,'
    '"last_probe_status":200,"consecutive_start_failures":0}'
)


def _status_read_rules() -> tuple[Rule, ...]:
    return (
        Rule(
            prefix=("cat",),
            when=lambda argv: argv[-1].endswith("/autostart.json"),
            replies=(Reply(stdout=_DESIGNATION_JSON),),
        ),
        Rule(
            prefix=("cat",),
            when=lambda argv: argv[-1].endswith("/autostart.status.json"),
            replies=(Reply(stdout=_STATUS_JSON),),
        ),
    )


def test_status_only_reads_both_files_from_both_nodes(tmp_path: Path) -> None:
    """[C8-P] `status` は、両台の `autostart.json` と `autostart.status.json` を `cat` するだけで、
    状態を変える呼び出しを 1 つも出さない。
    """
    repo = make_repo(tmp_path)
    spy = Spy(script=_status_read_rules())

    result = run(["autostart", "status"], repo, spy=spy)

    assert result.code == cli.EXIT_OK, result.err
    runner = result.spy.runner
    assert len(runner.calls) == 4, runner.calls
    assert all(call.kind == "run" for call in runner.calls)
    assert all(call.argv[0] == "cat" for call in runner.calls)
    assert all(not call.mutating for call in runner.calls)
    assert runner.pushes == ()
    assert {call.node for call in runner.calls} == {"head", "worker"}


def test_status_never_makes_a_mutating_call(tmp_path: Path) -> None:
    """[C8-N] `status` は `mutating` の呼び出しを 1 つも出さない (読むだけで、指定を変えない。
    読めない台があっても同じ)。両台とも読めないときは、終了コード `EXIT_PRECONDITION` で、
    `status=partial`・`unreadable=head,worker` を出す (問題17: この終了コードと出力を確かめる
    試験がなかった)。
    """
    repo = make_repo(tmp_path)
    spy = Spy(default=Reply(exit_code=1, stderr="ファイルがない"))

    result = run(["autostart", "status"], repo, spy=spy)

    assert [call for call in result.spy.runner.calls if call.mutating] == []
    assert result.spy.runner.pushes == ()
    assert result.code == cli.EXIT_PRECONDITION
    pairs = kv(result.out)
    assert pairs["status"] == "partial"
    assert pairs["unreadable"] == "head,worker"


def test_clear_pushes_a_null_config_designation_to_both_nodes(tmp_path: Path) -> None:
    """[C8-P] `clear` は、`config: null` の指定を両台に配るだけで、`launch.json` は読まない。"""
    repo = make_repo(tmp_path)
    spy = Spy()

    result = run(["autostart", "clear", "--yes"], repo, spy=spy)

    assert result.code == cli.EXIT_OK, result.err
    runner = result.spy.runner
    assert [call.kind for call in runner.calls] == ["push", "push"]
    assert {call.remote for call in runner.pushes} == {"state"}
    assert {call.node for call in runner.pushes} == {"head", "worker"}
    assert all(not call.delete for call in runner.pushes)
