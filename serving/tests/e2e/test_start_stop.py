"""起動と停止の、端から端までの試験 (tasks.md 5.2)。

確かめること (requirements 1.3、1.4、1.5、1.6、1.7、1.8、2.1。design.md「運転 › lifecycle」
「System Flows › 起動」「Error Handling」、tasks.md 5.2 の完了の状態):

1. **起動 → 状態 → もう一度の起動 → 停止 → もう一度の停止** が、期待した終了コード
   (0 → 0 → 0 (`already_running`: 名前、ダイジェスト、`config-sha256` まで一致) → 0 →
   0 (`already_stopped`)) になる
2. **応答の確認が 200 を返さないまま、worker のコンテナが終了する**と、時間切れを待たずに
   失敗し、記録の末尾が stderr に出て、呼び出しの順序が「記録の回収 (`docker logs`) →
   停止 (`docker stop`) → 削除 (`docker rm`)」で、head も worker も片付き、終了コードが 2
3. **了承しないと、状態を変える呼び出しが 1 つも出ない**。(a) 端末の了承で `no` と答えると 1
   で、`docker run` などが 0 件。(b) 端末でなく `--yes` もないと 1 で、0 件。`start` と `stop`
   の両方について

すべて **`cli.main` だけを入口にして**流す (`serve start <構成>` → `status` → `start` →
`stop` → `stop` のように、コマンドの列として)。`monkeypatch` で `serving_kit` の部品を
差し替えることはしない (端から端まで)。差し込むのは `cli.main` 自身の口
(`runner_factory`、`client_factory`、`confirmer`、`stdin`/`stdout`/`stderr`、`sleep`) だけ
である。3 の (a) は、`guards.TerminalConfirmer` 自身が持つ試験用の差し込み口 (`isatty=`) を
使う (これも `serving_kit` の公開の部品であり、モンキーパッチではない)。

相手にするのは、偽の実行役 `tests/fake_runner.py` (`FakeRunner`、`Rule`、`Reply`) と、
偽の推論サーバー `tests/fake_vllm.py` (`fake_vllm` フィクスチャ、`tests/conftest.py` にある。
`tests/` 直下のものなので、e2e からもそのまま使える)。実物の ssh、rsync、docker、推論
サーバーには、どの段でもつながない。台本の作り方 (関門を通す読み取りの台本、`docker run`
の返事、`docker ps` の一覧に自分の行を出す順、`docker container inspect` の状態、
`docker logs` の返事) は `serving/tests/unit/test_lifecycle_start.py` を見本にした。

構成の TOML は、`test_cli.py` の作り方 (`_header` / `_setting`) を見て、最小限にした
(重みを持たない `serve` 構成。`docker` の節は許可の一覧にある `--mount` を 1 つだけ持ち、
source は `{remote_root}/…`、`--port` は 1 つ)。重みを持たないので、`gate_weights_verified`
は関門を素通りし (`config.weights is None`)、台本を単純にできる。
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from fake_runner import FakeRunner, Reply, Rule
from fake_vllm import FakeVllm, Fault
from meminfo_sample import meminfo_rule
from serving_kit import cli
from serving_kit import config as config_mod
from serving_kit import guards as g
from serving_kit import plan as plan_mod
from serving_kit.plan import LABEL_IMAGE, OWNER_FILTER
from serving_kit.types import ConfigDef, ContainerPlan, NodeDef, NodeRole

# --- 見本の値 ---------------------------------------------------------------

ROLES: tuple[NodeRole, ...] = ("head", "worker")

CONFIG_NAME = "e2e-serve"
SERVED_MODEL = "glm-5-3-flash"
"""`FakeVllm` の既定のモデル名と合わせる (`fake_vllm` フィクスチャは差し替えない)。"""

IMAGE_REF = (
    "vllm/vllm-openai@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
)
REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"
SOURCE = "https://docs.vllm.ai/en/latest/cli/serve/"
QUOTE = "vllm serve [model_tag] [options]"
MEASURED = "docs/results/e2e-netcheck-links.md"
FABRIC_IFNAME = "enp1s0f0np0"

REPO_COMMIT = "e2efeed" + "0" * 33
UNUSED_PORT = 8123
"""HTTP に進まない試験 (了承の拒否) が使う、偽の推論サーバーのない番号。"""

CONTAINER_IDS: Mapping[NodeRole, str] = {"head": "0123456789ab", "worker": "cdef01234567"}

PLAN_BUILD_TIME = datetime(2026, 1, 1, tzinfo=UTC)
"""`plan.build_plans` に渡すだけの日時 (ラベルの `started-at` にしか効かない。
`config-sha256` はこれを材料にしないので、どの回に渡しても値は変わらない)。"""

OWN_CONTAINERS_ARGV: tuple[str, ...] = (
    "docker",
    "ps",
    "-a",
    "--filter",
    f"label={OWNER_FILTER}",
    "--format",
    "json",
)
"""`guards.list_own_containers` が流す、ただ 1 つのコンテナの一覧の読み取り
(`test_lifecycle_start.py` の `OWN_CONTAINERS_ARGV` と同じ形)。"""


# --- 試験用のリポジトリ (構成、ノード) ---------------------------------------


def _toml_str(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _setting_block(
    table: str, *, flag: str | None = None, value: str | None = None, is_port: bool = False
) -> str:
    """根拠 (`source` / `quote`) の付いた設定を 1 つ、TOML の 1 節として書く。"""
    lines = [f"[{table}]"]
    if flag is not None:
        lines.append(f"flag = {_toml_str(flag)}")
    if value is not None:
        lines.append(f"value = {_toml_str(value)}")
    lines.append('why = "端から端までの試験のための、最小の設定"')
    if is_port:
        lines.append("is_port = true")
    lines.append(f'source = "{SOURCE}"')
    lines.append(f'quote = "{QUOTE}"')
    return "\n".join(lines) + "\n\n"


def _configs_toml(port: int) -> str:
    """重みを持たない `serve` 構成を 1 つだけ持つ `configs.toml`。

    `docker` の節は、許可の一覧 (design.md「types / config」の検査 8) にある `--mount` を
    1 つだけ持ち、source は `{remote_root}/…` にする (config.py の検査 9)。`--port` は
    ちょうど 1 つ (Implementation Notes 3.4: HTTP のポートは `--port` の設定の値で決める)。
    """
    text = "schema_version = 1\n\n"
    text += "\n".join(
        [
            f"[configs.{CONFIG_NAME}]",
            'kind = "serve"',
            'description = "端から端までの試験用の、最小の推論サーバーの構成"',
            'nodes = ["head", "worker"]',
            "ready_timeout_s = 1800",
            f'served_model_name = "{SERVED_MODEL}"',
        ]
    )
    text += "\n\n"
    text += "\n".join(
        [
            f"[configs.{CONFIG_NAME}.image]",
            f'ref = "{IMAGE_REF}"',
            'seen_as = "vllm/vllm-openai:e2e-test"',
            "size_bytes = 1024",
            f'source = "{SOURCE}"',
            f'quote = "{QUOTE}"',
        ]
    )
    text += "\n\n"
    text += _setting_block(
        f"configs.{CONFIG_NAME}.docker.cache",
        flag="--mount",
        value="type=bind,source={remote_root}/cache,target=/root/.cache",
    )
    text += _setting_block(
        f"configs.{CONFIG_NAME}.args.served-model-name",
        flag="--served-model-name",
        value=SERVED_MODEL,
    )
    text += _setting_block(
        f"configs.{CONFIG_NAME}.args.host", flag="--host", value="{head.lan_addr}"
    )
    text += _setting_block(
        f"configs.{CONFIG_NAME}.args.port", flag="--port", value=str(port), is_port=True
    )
    return text


def _nodes_toml() -> str:
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


@dataclass(frozen=True)
class Repo:
    """試験用のリポジトリ (`serving/config/` だけを持つ)。"""

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


def make_repo(tmp_path: Path, *, port: int) -> Repo:
    """構成とノードの定義だけを持つ、最小のリポジトリを作る。"""
    root = tmp_path / "repo"
    (root / "docs" / "results").mkdir(parents=True, exist_ok=True)
    (root / MEASURED).write_text("端から端までの試験用の、実測の記録の見本\n", encoding="utf-8")
    (root / "serving" / "config").mkdir(parents=True, exist_ok=True)
    repo = Repo(root=root)
    repo.configs.write_text(_configs_toml(port), encoding="utf-8")
    repo.nodes.write_text(_nodes_toml(), encoding="utf-8")
    return repo


def load_config_and_nodes(repo: Repo) -> tuple[ConfigDef, dict[NodeRole, NodeDef]]:
    """`cli` が内部で読むのと同じ道 (`config.load_configs` / `load_nodes` / `select_config`)。

    台本を作るために、コンテナの計画 (ラベル、名前) を先に知る必要があるので、
    ここで一度読む。実際に流す `cli.main` は、これとは独立に、自分でもう一度読み直す。
    """
    nodes = config_mod.load_nodes(repo.nodes, repo.root)
    configs = config_mod.load_configs(repo.configs, repo.root)
    config = config_mod.select_config(configs, CONFIG_NAME, nodes)
    return config, nodes


def container_plans(
    config: ConfigDef, nodes: Mapping[NodeRole, NodeDef]
) -> dict[NodeRole, ContainerPlan]:
    plans = plan_mod.build_plans(config, nodes, PLAN_BUILD_TIME)
    return {plan.node: plan for plan in plans}


def port_of(fake: FakeVllm) -> int:
    return int(fake.base_url.rsplit(":", 1)[1])


# --- 台本の部品 --------------------------------------------------------------


def own_row(plan: ContainerPlan, *, container_id: str, state: str = "running") -> str:
    """`docker ps -a --filter label=… --format json` の 1 行 (自分のコンテナ)。

    実機の見本ではない (tasks.md 1.6 の「ここで採れないもの」)。項目の名前は、
    `test_lifecycle_start.py` / `test_cli.py` の見本に合わせた。
    """
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


def passing_gate_rules(role: NodeRole) -> tuple[Rule, ...]:
    """1 台ぶんの、9 つの関門のうち、読み取りを要る 7 つを、すべて通す台本。

    重みを持たない構成なので `gate_weights_verified` は呼び出しなしで通り、
    `gate_own_state` は `list_own_containers` の 1 回の読み取りを使い回す
    (Implementation Notes 2.3)。残り 7 つが、ここで読み取りを要る関門である。
    """
    return (
        Rule(prefix=("uname",), node=role, replies=(Reply(stdout="e2e-host\n"),)),
        Rule(prefix=("nvidia-smi",), node=role, replies=(Reply(stdout=""),)),
        Rule(prefix=("test", "-e"), node=role, replies=(Reply(exit_code=0),)),
        Rule(
            prefix=("docker", "image", "inspect"),
            node=role,
            replies=(Reply(stdout=json.dumps([IMAGE_REF]) + "\n"),),
        ),
        Rule(prefix=("df",), node=role, replies=(Reply(stdout="Avail\n999999999999999\n"),)),
        meminfo_rule(node=role),
        Rule(prefix=("ss",), node=role, replies=(Reply(stdout=""),)),
    )


def listing_rule(role: NodeRole, plan: ContainerPlan, *, container_id: str) -> Rule:
    """一覧の読み取り: 1 回目は空 (まだ起こしていない)、2 回目からは自分の行 (起こしたあと)。

    台本が尽きると最後の返事を繰り返す (`fake_runner.py` の決まり) ので、起こしたあとに
    何度読んでも、同じ行が返る。"""
    return Rule(
        prefix=OWN_CONTAINERS_ARGV,
        node=role,
        replies=(Reply(stdout=""), Reply(stdout=own_row(plan, container_id=container_id))),
    )


def running_listing_rule(role: NodeRole, plan: ContainerPlan, *, container_id: str) -> Rule:
    """すでに動いている前提の一覧 (`status`、2 度目の `start`、`stop` が使う)。"""
    return Rule(
        prefix=OWN_CONTAINERS_ARGV,
        node=role,
        replies=(Reply(stdout=own_row(plan, container_id=container_id)),),
    )


def empty_listing_rule(role: NodeRole) -> Rule:
    """自分のコンテナが 1 つもない一覧 (2 度目の `stop` が使う)。"""
    return Rule(prefix=OWN_CONTAINERS_ARGV, node=role, replies=(Reply(stdout=""),))


def start_ready_rules(plans: Mapping[NodeRole, ContainerPlan]) -> tuple[Rule, ...]:
    """起動が `ready` になるまでの、ただ 1 つの台本 (一覧 → 関門 → 記録の配布 → `docker run`)。

    `/health` は既定 (`Fault()`) で 200 を返すので、`wait_ready` は 1 回目で `ready` になり、
    `docker container inspect` は 1 度も呼ばれない (`started_targets` が読む一覧の 2 回目の
    返事だけで、起こしたコンテナの識別子が分かる)。`ready` のあとは、head の記録の末尾を
    `OBSERVE_TAIL_LINES` で読む (`_ready_outcome`)。
    """
    rules: list[Rule] = []
    for role in ROLES:
        plan = plans[role]
        rules.append(listing_rule(role, plan, container_id=CONTAINER_IDS[role]))
        rules.extend(passing_gate_rules(role))
        rules.append(Rule(kind="push", node=role, replies=(Reply(),)))
        rules.append(
            Rule(
                prefix=("docker", "run"),
                node=role,
                replies=(Reply(stdout=f"{CONTAINER_IDS[role]}\n"),),
            )
        )
    rules.append(
        Rule(prefix=("docker", "logs"), node="head", replies=(Reply(stdout="INFO ready\n"),))
    )
    return tuple(rules)


def status_rules(plans: Mapping[NodeRole, ContainerPlan]) -> tuple[Rule, ...]:
    """`serve status` の台本: 動いている一覧、GPU、直結のリンク (読み取りだけ)。"""
    rules: list[Rule] = []
    for role in ROLES:
        plan = plans[role]
        rules.append(running_listing_rule(role, plan, container_id=CONTAINER_IDS[role]))
        rules.append(Rule(prefix=("nvidia-smi",), node=role, replies=(Reply(stdout=""),)))
        rules.append(
            Rule(
                prefix=("ip", "-br", "link"),
                node=role,
                replies=(Reply(stdout=f"{FABRIC_IFNAME}          UP             <UP,LOWER_UP>\n"),),
            )
        )
    return tuple(rules)


def already_running_rules(plans: Mapping[NodeRole, ContainerPlan]) -> tuple[Rule, ...]:
    """もう一度の `start` の台本: 一覧を 1 度読むだけで `already_running` になる。"""
    return tuple(
        running_listing_rule(role, plans[role], container_id=CONTAINER_IDS[role]) for role in ROLES
    )


def stop_rules(plans: Mapping[NodeRole, ContainerPlan]) -> tuple[Rule, ...]:
    """`serve stop` (動いている場合) の台本: 一覧 → `stop` → `rm` → GPU の確かめ。"""
    rules: list[Rule] = []
    for role in ROLES:
        rules.append(running_listing_rule(role, plans[role], container_id=CONTAINER_IDS[role]))
        rules.append(Rule(prefix=("docker", "stop"), node=role, replies=(Reply(),)))
        rules.append(Rule(prefix=("docker", "rm"), node=role, replies=(Reply(),)))
        rules.append(Rule(prefix=("nvidia-smi",), node=role, replies=(Reply(stdout=""),)))
    return tuple(rules)


def stopped_rules() -> tuple[Rule, ...]:
    """もう一度の `stop` の台本: 2 台とも、自分のコンテナが 1 つもない。"""
    return tuple(empty_listing_rule(role) for role in ROLES)


def start_approval_rules(plans: Mapping[NodeRole, ContainerPlan]) -> tuple[Rule, ...]:
    """`start` が了承の直前まで届く台本 (一覧は空、関門はすべて通る。`docker run` は書かない)。"""
    rules: list[Rule] = []
    for role in ROLES:
        rules.append(empty_listing_rule(role))
        rules.extend(passing_gate_rules(role))
    return tuple(rules)


def stop_approval_rules(plans: Mapping[NodeRole, ContainerPlan]) -> tuple[Rule, ...]:
    """`stop` が了承の直前まで届く台本 (対象がある一覧だけ。`docker stop` / `rm` は書かない)。"""
    return tuple(
        running_listing_rule(role, plans[role], container_id=CONTAINER_IDS[role]) for role in ROLES
    )


# --- `cli.main` の 1 回の呼び出し --------------------------------------------


@dataclass
class Invocation:
    code: int
    out: str
    err: str
    runner: FakeRunner


def _never_sleep(seconds: float) -> None:
    """このシナリオは実際には眠らないはず、という前提を固定する (決めごとの検証)。"""
    raise AssertionError(f"このシナリオは眠らないはずなのに、{seconds} 秒眠ろうとした")


def invoke(
    argv: Sequence[str],
    repo: Repo,
    *,
    script: Sequence[Rule],
    default: Reply | None = None,
    confirmer: g.Confirmer | None = None,
    stdin: str = "",
    client_factory: Callable[[float], httpx.Client] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Invocation:
    """`cli.main` を、共通の下ごしらえ (構成、ノード、記録の置き場所) を足して 1 回呼ぶ。

    差し込むのは `cli.main` 自身の口だけ (`runner_factory`、`client_factory`、`confirmer`、
    `stdin`/`stdout`/`stderr`、`sleep`)。`serving_kit` の部品を `monkeypatch` で置き換える
    ことはしない。
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
        now=lambda: datetime.now(UTC),
        sleep=_never_sleep if sleep is None else sleep,
    )
    assert made, "実行役が 1 つも作られていない (runner_factory が呼ばれなかった)"
    return Invocation(code=code, out=out.getvalue(), err=err.getvalue(), runner=made[0])


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


# --- 1: 起動 → 状態 → もう一度の起動 → 停止 → もう一度の停止 ------------------


def test_start_status_start_stop_stop(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """起動 → 状態 → もう一度の起動 → 停止 → もう一度の停止が、期待した終了コードになる。

    各段で、標準出力の `key=value` (`status=`、`detail=`) も確かめる。状態の変化 (自分の
    コンテナがある/ない) は、各段ごとに新しく作る `FakeRunner` の台本の違いで表す
    (実物では、この状態は Spark の上の docker が持つ)。
    """
    repo = make_repo(tmp_path, port=port_of(fake_vllm))
    config, nodes = load_config_and_nodes(repo)
    plans = container_plans(config, nodes)

    # 1. 起動 (まだ何も動いていない) → ready で 0
    first_start = invoke(["start", CONFIG_NAME, "--yes"], repo, script=start_ready_rules(plans))
    assert first_start.code == cli.EXIT_OK
    first_out = kv(first_start.out)
    assert first_out["status"] == "ready"
    assert first_out["detail"]

    # 2. 状態 (動いている) → 0、読めなかった台がない
    status = invoke(["status"], repo, script=status_rules(plans))
    assert status.code == cli.EXIT_OK
    status_out = kv(status.out)
    assert status_out["unreadable"] == ""
    assert status_out["node.head.container_state"] == "running"
    assert status_out["node.worker.container_state"] == "running"

    # 3. もう一度の起動 (名前、ダイジェスト、config-sha256 まで一致) → already_running で 0、
    #    状態を変える呼び出しは 1 つも出ない
    second_start = invoke(
        ["start", CONFIG_NAME, "--yes"], repo, script=already_running_rules(plans)
    )
    assert second_start.code == cli.EXIT_OK
    second_out = kv(second_start.out)
    assert second_out["status"] == "already_running"
    assert mutating_calls(second_start.runner) == ()
    for word in ("名前", "ダイジェスト", "config-sha256"):
        assert word in second_out["detail"], (
            f"{word!r} が detail に出ていない: {second_out['detail']!r}"
        )

    # 4. 停止 (動いている) → stopped で 0。head → worker の順に stop、そのあと rm
    first_stop = invoke(["stop", "--yes"], repo, script=stop_rules(plans))
    assert first_stop.code == cli.EXIT_OK
    stop_out = kv(first_stop.out)
    assert stop_out["status"] == "stopped"
    assert mutating_calls(first_stop.runner) == (
        "docker stop",
        "docker stop",
        "docker rm",
        "docker rm",
    )
    stop_call_nodes = [call.node for call in first_stop.runner.calls if call.mutating]
    assert stop_call_nodes == ["head", "worker", "head", "worker"]

    # 5. もう一度の停止 (何も動いていない) → already_stopped で 0、了承も求めない
    second_stop = invoke(["stop", "--yes"], repo, script=stopped_rules())
    assert second_stop.code == cli.EXIT_OK
    assert kv(second_stop.out)["status"] == "already_stopped"
    assert mutating_calls(second_stop.runner) == ()


# --- 2: worker が応答の確認の前に終了する ------------------------------------


def test_start_fails_fast_when_worker_exits_before_healthy(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """`/health` が 200 を返さないまま worker が終了すると、時間切れを待たずに失敗する。

    `--timeout 1h` を渡しても、失敗の検出はコンテナの状態の読み取りだけで決まり (`wait_ready`
    は、待ちの 1 周目に、応答の確認のあとで両台の状態を見る)、この試験は一度も眠らない
    (`sleep=` に、眠ろうとしたら落ちる関数を渡してあるので、実際に眠れば test は落ちる)。
    実測の壁時計でも、これを裏づける。
    """
    fake_vllm.set_health_fault(Fault(status=None))  # 応答しない (時間切れを待たせないため)
    repo = make_repo(tmp_path, port=port_of(fake_vllm))
    config, nodes = load_config_and_nodes(repo)
    plans = container_plans(config, nodes)

    head_marker = "ZZ-HEAD-TAIL-MARKER-ZZ"
    worker_marker = "ZZ-WORKER-TAIL-MARKER-ZZ"

    rules: list[Rule] = []
    for role in ROLES:
        plan = plans[role]
        rules.append(listing_rule(role, plan, container_id=CONTAINER_IDS[role]))
        rules.extend(passing_gate_rules(role))
        rules.append(Rule(kind="push", node=role, replies=(Reply(),)))
        rules.append(
            Rule(
                prefix=("docker", "run"),
                node=role,
                replies=(Reply(stdout=f"{CONTAINER_IDS[role]}\n"),),
            )
        )
        rules.append(Rule(kind="pull", node=role, replies=(Reply(),)))
        rules.append(Rule(prefix=("docker", "stop"), node=role, replies=(Reply(),)))
        rules.append(Rule(prefix=("docker", "rm"), node=role, replies=(Reply(),)))
    # 待ちの 1 周目で、head は動いたまま、worker は終了している (時間切れを待たない)
    rules.append(
        Rule(
            prefix=("docker", "container", "inspect"),
            node="head",
            replies=(Reply(stdout="running 0\n"),),
        )
    )
    rules.append(
        Rule(
            prefix=("docker", "container", "inspect"),
            node="worker",
            replies=(Reply(stdout="exited 1\n"),),
        )
    )
    rules.append(
        Rule(prefix=("docker", "logs"), node="head", replies=(Reply(stdout=f"{head_marker}\n"),))
    )
    rules.append(
        Rule(
            prefix=("docker", "logs"),
            node="worker",
            replies=(Reply(stdout=f"{worker_marker}\n"),),
        )
    )

    started = time.monotonic()
    result = invoke(["start", CONFIG_NAME, "--timeout", "1h", "--yes"], repo, script=tuple(rules))
    elapsed = time.monotonic() - started

    assert result.code == cli.EXIT_FAILED
    assert elapsed < 5.0, f"時間切れを待った形跡がある ({elapsed:.1f} 秒かかった)"

    # 記録の末尾が stderr に出ている (2 台とも)
    assert head_marker in result.err
    assert worker_marker in result.err
    assert head_marker not in result.out
    assert worker_marker not in result.out

    # 正しい経路の detail であることを、文面で固定する (レビューの指摘: `wait_ready` の
    # 終了の検出を無効にしても、`sleep=_never_sleep` の `AssertionError` が `_launch` の
    # `except BaseException` を通って `wrap_up` に落ち、終了コード・末尾・片付けの順序が
    # 偶然そろってしまう。正しい経路の文言と、壊れた経路 (予期しない例外) の文言を、
    # 積極的に区別する)
    assert "終了した" in result.err
    assert "時間切れを待たずに" in result.err
    assert "予期しない失敗" not in result.err
    assert "眠ろうとした" not in result.err

    # 待ちのループが 1 周目で止まっている (2 周目に入っていない) ことを、呼び出しの回数で
    # 直接固定する。`_UNFINISHED_STATES` から `exited` を除く決まりが壊れると、1 周目では
    # 終了を検出できず、2 周目 (`controls.sleep()` のあと) の `docker container inspect` が
    # 増える (このテストの `sleep=` は眠ろうとした時点で落ちるので、増えることはなく、
    # そもそも `main` が予期しない例外で終わる。この assert は、その手前の事実を固定する)
    inspect_calls_by_node: dict[str, int] = {}
    for call in result.runner.calls:
        if call.kind == "run" and call.argv[:3] == ("docker", "container", "inspect"):
            inspect_calls_by_node[call.node] = inspect_calls_by_node.get(call.node, 0) + 1
    assert inspect_calls_by_node == {"head": 1, "worker": 1}, (
        f"待ちのループが 2 周目に入っている: {inspect_calls_by_node}"
    )

    # 呼び出しの順序: 記録の回収 (docker logs) → 停止 (docker stop) → 削除 (docker rm)
    ordered = [
        (call.argv[0], call.argv[1])
        for call in result.runner.calls
        if call.kind == "run"
        and call.argv[:2] in (("docker", "logs"), ("docker", "stop"), ("docker", "rm"))
    ]
    last_logs = max(i for i, (_, sub) in enumerate(ordered) if sub == "logs")
    first_stop = min(i for i, (_, sub) in enumerate(ordered) if sub == "stop")
    first_rm = min(i for i, (_, sub) in enumerate(ordered) if sub == "rm")
    assert last_logs < first_stop < first_rm, f"片付けの順序が違う: {ordered}"

    # head も worker も片付く
    stopped_nodes = {
        call.node for call in result.runner.calls if call.argv[:2] == ("docker", "stop")
    }
    removed_nodes = {call.node for call in result.runner.calls if call.argv[:2] == ("docker", "rm")}
    assert stopped_nodes == {"head", "worker"}
    assert removed_nodes == {"head", "worker"}


# --- 3: 了承しないと、状態を変える呼び出しが 1 つも出ない ---------------------


@pytest.mark.parametrize("command", ["start", "stop"])
@pytest.mark.parametrize("scenario", ["terminal_no", "non_terminal"])
def test_no_mutating_call_without_approval(tmp_path: Path, command: str, scenario: str) -> None:
    """了承しなければ、`start` / `stop` のどちらも、状態を変える呼び出しを 1 つも出さない。

    (a) `terminal_no`: 端末で `yes` 以外 (`no`) と答える → `guards.TerminalConfirmer` を、
        `isatty=lambda: True` で組み立てて、`cli.main(confirmer=...)` に直に渡す
        (`TerminalConfirmer` 自身の、試験のための差し込み口。モンキーパッチではない)
    (b) `non_terminal`: 端末でなく `--yes` もない → `confirmer` を渡さず、既定の
        `TerminalConfirmer` に任せる (`stdin` が `io.StringIO` なので `isatty()` は偽になる)

    どちらも、終了コードは 1 (`ApprovalError` → `EXIT_PRECONDITION`)。`start` は関門をすべて
    通す台本 (`docker run` は書かない)、`stop` は対象のある一覧の台本 (`docker stop` / `rm`
    は書かない) を使うので、了承のところまで届いたのに、状態を変える呼び出しが出なければ、
    それは了承がなかったからだと言える。
    """
    repo = make_repo(tmp_path, port=UNUSED_PORT)  # HTTP には届かないので、実物の port でよい
    config, nodes = load_config_and_nodes(repo)
    plans = container_plans(config, nodes)

    if command == "start":
        argv = ["start", CONFIG_NAME]
        script = start_approval_rules(plans)
    else:
        argv = ["stop"]
        script = stop_approval_rules(plans)

    confirmer: g.Confirmer | None
    if scenario == "terminal_no":
        confirmer = g.TerminalConfirmer(
            stdin=io.StringIO("no\n"), stderr=io.StringIO(), isatty=lambda: True
        )
    else:
        confirmer = None

    result = invoke(argv, repo, script=script, confirmer=confirmer)

    assert result.code == cli.EXIT_PRECONDITION
    assert mutating_calls(result.runner) == ()
    assert [call for call in result.runner.calls if call.kind in ("push", "pull")] == []
