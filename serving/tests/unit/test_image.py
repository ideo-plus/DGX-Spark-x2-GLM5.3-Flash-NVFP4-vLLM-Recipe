"""イメージの取得と、ライセンスの表記の読み取りの試験 (tasks.md 3.1)。

確かめること (design.md 「イメージと重み › image」、tasks.md 3.1 の完了の状態):

- **了承しないと、取得の呼び出し (`docker pull`) が 1 つも出ない** (requirements 2.1)
- 了承すると、2 台で、**ダイジェストでの** `docker pull` が出て、そのあとに照合が流れる
  (requirements 3.2)
- 取得のあとにダイジェストが合わなければ、食い違い (構成の値と、台が返した `RepoDigests`) を
  示して失敗する (requirements 3.3)
- 関門 (入れない、空きが足りない) で断られたら、了承も取得も出ない (requirements 2.5)
- ライセンスの表記の読み取りが、**切り離して起こす呼び出し** (`docker run -d`、名前とラベル
  つき) として記録され、`--rm` も、前面の指定も含まない (requirements 3.8)
- 読み取りのコンテナが、終了を待たれ、出力が読まれ、**必ず消される**。成功、`cat` の失敗、
  時間切れ、起動の失敗、中断のそれぞれで、呼び出しの順序が「起こす → 待つ → 読む → 止める →
  消す」で、コンテナが残らない
- 名前の衝突のときは、巻き戻しが出ない (requirements 2.3。自分のものでないコンテナを、名前で
  止めない)
- 偽の実行役に記録された、コンテナを対象にする操作のすべてが、ラベルで絞った一覧の識別子か、
  了承済みの計画の名前だけを対象にしている (requirements 2.3、2.4)
- `docker build`、`docker rmi`、タグでの `docker pull` の列が、どの経路でも出ない (8.6)

読み取りの試験の台本は、2 台の Spark から採った実物の見本 (`tests/fixtures/spark/`) を使う。
自分のコンテナがあるときの `docker ps --format json` の行だけは、実機からまだ採れていない
(tasks.md 1.6 の「ここで採れないもの」。7.1 で採って見本を足す)。実物の ssh、rsync、docker は、
どの段でも呼ばない。試験は、実際に眠らない (待つ間隔と時計は、引数で差し替える)。
"""

from __future__ import annotations

import ast
import inspect as inspect_module
import io
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl

from fake_runner import FakeRunner, RecordedCall, Reply, Rule
from meminfo_sample import meminfo_rule
from serving_kit import image as im
from serving_kit.config import ConfigError
from serving_kit.guards import ApprovalError
from serving_kit.plan import LABEL_IMAGE, LABEL_OWNER, OWNER, OWNER_FILTER, build_plans
from serving_kit.types import (
    ConfigDef,
    ContainerPlan,
    GateResult,
    ImageRef,
    NodeDef,
    NodeRole,
    Setting,
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

STARTED_AT = datetime(2026, 9, 21, 3, 0, 0, tzinfo=UTC)

SOURCE = HttpUrl("https://docs.vllm.ai/en/latest/cli/serve/")
QUOTE = "vllm serve [model_tag] [options]"

LICENSE_PATH = "/usr/share/doc/vllm/LICENSE"
NOTICE_PATH = "/NGC-DL-CONTAINER-LICENSE"
LICENSE_TEXT = "Apache License\nVersion 2.0, January 2004\n"

CONTAINER_ID = "0123456789ab"


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
)
WORKER = NodeDef(
    role="worker",
    ssh_host="spark-5083",
    lan_addr=IPv4Address("10.0.1.61"),
    remote_root=REMOTE_ROOT,
)
NODES: Mapping[NodeRole, NodeDef] = {"head": HEAD, "worker": WORKER}


def serve_config() -> ConfigDef:
    """2 台の推論サーバーの構成 (`serve pull-image` の対象にする)。"""
    return ConfigDef(
        name="p1-nvfp4-tp2",
        kind="serve",
        description="P1 の第一の構成",
        nodes=("head", "worker"),
        image=IMAGE,
        weights=WEIGHTS,
        docker={"gpus": _setting("--gpus", "all")},
        args={"model-path": _setting(value="{weights.mount_at}")},
        env={},
        ready_timeout_s=1800,
        served_model_name="glm-5-3-flash",
    )


def inspect_config(*, ready_timeout_s: int = 120) -> ConfigDef:
    """ライセンスの表記の読み取りの構成 (design.md Data Models の `p1-image-licenses`)。

    動かすプログラムは docker の設定 (`--entrypoint cat`)、読むファイルの道筋は、`args` の
    位置の引数である。重みの参照も `--mount` も持たず、`nodes` は 1 つ。
    """
    return ConfigDef(
        name="p1-image-licenses",
        kind="inspect",
        description="イメージの中のライセンスの表記を読む",
        nodes=("head",),
        image=IMAGE,
        weights=None,
        docker={"entrypoint": _setting("--entrypoint", "cat")},
        args={
            "license": _setting(value=LICENSE_PATH),
            "notice": _setting(value=NOTICE_PATH),
        },
        env={},
        ready_timeout_s=ready_timeout_s,
        served_model_name=None,
    )


INSPECT_PLAN: ContainerPlan = build_plans(inspect_config(), NODES, STARTED_AT)[0]


def sample(role: NodeRole, name: str) -> str:
    """2 台の Spark から採った、読み取りの実物の出力 (tasks.md 1.6)。"""
    return (FIXTURES / role / name).read_text()


def ps_line(plan: ContainerPlan, *, state: str = "exited", container_id: str = CONTAINER_ID) -> str:
    """`docker ps -a --format json` の 1 行 (自分のコンテナがあるときの見本)。

    **これは実機から採った見本ではない** (tasks.md 1.6 の「ここで採れないもの」)。項目の
    名前は `test_guards.py` / `test_logs.py` の見本に合わせた。
    """
    labels = ",".join(f"{key}={value}" for key, value in sorted(plan.labels.items()))
    return json.dumps(
        {
            "ID": container_id,
            "Names": plan.container_name,
            "State": state,
            "Image": plan.labels[LABEL_IMAGE],
            "Labels": labels,
        }
    )


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


# --- 了承の口と、時計 ----------------------------------------------------


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
    """試験用の時計。呼ばれるたびに進み、眠りは記録するだけで、実際には眠らない。"""

    step: float = 30.0
    now: float = 0.0
    slept: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        value = self.now
        self.now += self.step
        return value

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


# --- 台本 ----------------------------------------------------------------


@dataclass
class PullScript:
    """`serve pull-image` の台本 (既定: 2 台とも入れて、空いていて、取得も照合も通る)。"""

    unreachable: tuple[NodeRole, ...] = ()
    avail: Mapping[NodeRole, str] | None = None
    df_exit_code: int = 0
    pull: Reply = Reply()
    digests: tuple[str, ...] = (OTHER_REF, IMAGE_REF)

    def rules(self) -> tuple[Rule, ...]:
        rules: list[Rule] = []
        for role in ROLES:
            uname = (
                Reply(exit_code=255, stderr="ssh: connect to host port 22: No route to host")
                if role in self.unreachable
                else Reply(stdout=sample(role, "uname-n.txt"))
            )
            avail = sample(role, "df-avail.txt") if self.avail is None else self.avail[role]
            df = (
                Reply(stdout=avail)
                if self.df_exit_code == 0
                else Reply(
                    exit_code=self.df_exit_code,
                    stderr=f"df: {REMOTE_ROOT}: No such file or directory",
                )
            )
            rules.append(Rule(prefix=("uname",), node=role, replies=(uname,)))
            rules.append(Rule(prefix=("df",), node=role, replies=(df,)))
        rules.append(Rule(prefix=("docker", "pull"), replies=(self.pull,)))
        rules.append(
            Rule(
                prefix=("docker", "image", "inspect"),
                replies=(Reply(stdout=json.dumps(list(self.digests)) + "\n"),),
            )
        )
        return tuple(rules)


@dataclass
class LicenseScript:
    """`serve image-licenses` の台本 (既定: 関門が通り、コンテナが 0 で終わり、記録が読める)。"""

    listings: tuple[Reply, ...] | None = None
    """`docker ps` の返事の列 (既定は「関門のときは無い → 起こしたあとはある」)。"""

    run: Reply = Reply(stdout=f"{CONTAINER_ID}\n")
    states: tuple[Reply, ...] = (Reply(stdout="exited 0\n"),)
    logs: Reply = Reply(stdout=LICENSE_TEXT)
    stop: Reply = Reply()
    remove: Reply = Reply()
    digests: tuple[str, ...] = (IMAGE_REF,)

    def rules(self) -> tuple[Rule, ...]:
        listings = (
            (Reply(stdout=""), Reply(stdout=ps_line(INSPECT_PLAN) + "\n"))
            if self.listings is None
            else self.listings
        )
        return (
            Rule(prefix=("uname",), replies=(Reply(stdout=sample("head", "uname-n.txt")),)),
            Rule(prefix=OWN_CONTAINERS_ARGV, replies=listings),
            Rule(
                prefix=("nvidia-smi",),
                replies=(Reply(stdout=sample("head", "nvidia-smi-compute-apps.txt")),),
            ),
            Rule(prefix=("df",), replies=(Reply(stdout=sample("head", "df-avail.txt")),)),
            meminfo_rule(),
            Rule(prefix=("ss",), replies=(Reply(stdout=sample("head", "ss-ltnH.txt")),)),
            Rule(
                prefix=("docker", "image", "inspect"),
                replies=(Reply(stdout=json.dumps(list(self.digests)) + "\n"),),
            ),
            Rule(prefix=("docker", "run"), replies=(self.run,)),
            Rule(prefix=("docker", "container", "inspect"), replies=self.states),
            Rule(prefix=("docker", "logs"), replies=(self.logs,)),
            Rule(prefix=("docker", "stop"), replies=(self.stop,)),
            Rule(prefix=("docker", "rm"), replies=(self.remove,)),
        )


def pull_runner(tmp_path: Path, script: PullScript | None = None) -> FakeRunner:
    return FakeRunner(var_root=tmp_path, script=(script or PullScript()).rules())


def license_runner(tmp_path: Path, script: LicenseScript | None = None) -> FakeRunner:
    return FakeRunner(var_root=tmp_path, script=(script or LicenseScript()).rules())


@pytest.fixture(autouse=True)
def no_real_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """実物の ssh / rsync / docker を呼ばないことと、実際に眠らないことを固定する。"""

    def explode(argv: Sequence[str], **kwargs: Any) -> None:
        raise AssertionError(f"実物のコマンドを呼んだ: {list(argv)}")

    def no_sleep(seconds: float) -> None:
        raise AssertionError(f"試験が実際に眠った: {seconds} 秒")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(time, "sleep", no_sleep)


# --- 呼び出しの読み取り --------------------------------------------------


def steps(runner: FakeRunner) -> tuple[str, ...]:
    """記録された呼び出しを、短い名前の列にする (順序を見るため)。"""
    names: list[str] = []
    for call in runner.runs:
        argv = call.argv
        if argv[0] != "docker":
            names.append(argv[0])
        elif argv[1] == "container":
            names.append("container inspect")
        elif argv[1] == "image":
            names.append("image inspect")
        elif argv[1] == "ps":
            names.append("ps")
        else:
            names.append(argv[1])
    return tuple(names)


def after_the_run(runner: FakeRunner) -> tuple[str, ...]:
    """`docker run` から後ろの呼び出しの列 (「起こす → 待つ → 読む → 止める → 消す」)。"""
    named = steps(runner)
    assert "run" in named, f"コンテナを起こす呼び出しがない: {named}"
    return named[named.index("run") :]


def mutating_calls(runner: FakeRunner) -> tuple[RecordedCall, ...]:
    return tuple(call for call in runner.calls if call.mutating)


# --- イメージの取得 ------------------------------------------------------


def test_the_gates_run_before_the_approval_and_the_pull(tmp_path: Path) -> None:
    runner = pull_runner(tmp_path)
    confirmer = SpyConfirmer()
    outcome = im.pull_image(runner, serve_config(), NODES, confirmer=confirmer)

    assert outcome.status == "pulled"
    # 関門 (入れるか、空き) → 取得 → 照合 の順に流れる
    assert steps(runner) == (
        "uname",
        "df",
        "uname",
        "df",
        "pull",
        "pull",
        "image inspect",
        "image inspect",
    )
    assert len(confirmer.shown) == 1
    # 了承は、関門のあと、取得の前に 1 度だけ求める
    assert "docker pull" in confirmer.shown[0]


def test_nothing_is_pulled_when_the_operator_refuses(tmp_path: Path) -> None:
    runner = pull_runner(tmp_path)
    confirmer = SpyConfirmer(approves=False)
    with pytest.raises(ApprovalError):
        im.pull_image(runner, serve_config(), NODES, confirmer=confirmer)

    assert [argv for argv in runner.argvs if argv[:2] == ("docker", "pull")] == []
    assert mutating_calls(runner) == ()
    assert runner.plan is None, "了承されていないのに、計画が実行役に渡った"


def test_local_image_id_is_refused_before_any_remote_call(tmp_path: Path) -> None:
    config = serve_config()
    config = config.model_copy(
        update={"image": config.image.model_copy(update={"ref": f"sha256:{DIGEST}"})}
    )
    runner = FakeRunner(var_root=tmp_path)
    confirmer = SpyConfirmer()
    with pytest.raises(ValueError, match="ローカルイメージ ID"):
        im.pull_image(runner, config, NODES, confirmer=confirmer)
    assert not runner.argvs
    assert not confirmer.shown


def test_the_pull_uses_the_digest_reference_on_both_nodes(tmp_path: Path) -> None:
    runner = pull_runner(tmp_path)
    im.pull_image(runner, serve_config(), NODES, confirmer=SpyConfirmer())

    pulls = [call for call in runner.runs if call.argv[:2] == ("docker", "pull")]
    assert [call.node for call in pulls] == ["head", "worker"]
    for call in pulls:
        assert call.argv == ("docker", "pull", IMAGE_REF)
        assert call.mutating is True


def test_a_digest_that_does_not_match_after_the_pull_fails(tmp_path: Path) -> None:
    script = PullScript(digests=(OTHER_REF,))
    runner = pull_runner(tmp_path, script)
    with pytest.raises(im.ImageError) as caught:
        im.pull_image(runner, serve_config(), NODES, confirmer=SpyConfirmer())

    message = str(caught.value)
    assert IMAGE_REF in message, "構成のダイジェストが示されていない"
    assert OTHER_REF in message, "台が返した RepoDigests が示されていない"
    # 黙って取り直さない
    assert steps(runner).count("pull") == 2


def test_a_missing_image_after_the_pull_fails(tmp_path: Path) -> None:
    # `docker image inspect` が失敗する (取得できていない) ときも、断って止まる
    runner = FakeRunner(
        var_root=tmp_path,
        script=(
            *PullScript().rules()[:-1],
            Rule(
                prefix=("docker", "image", "inspect"),
                replies=(Reply(exit_code=1, stderr="Error: No such image"),),
            ),
        ),
    )
    with pytest.raises(im.ImageError, match="ダイジェスト"):
        im.pull_image(runner, serve_config(), NODES, confirmer=SpyConfirmer())


def test_an_unreachable_node_refuses_without_asking_for_the_approval(tmp_path: Path) -> None:
    runner = pull_runner(tmp_path, PullScript(unreachable=("worker",)))
    confirmer = SpyConfirmer()
    outcome = im.pull_image(runner, serve_config(), NODES, confirmer=confirmer)

    assert outcome.status == "refused"
    assert confirmer.shown == []
    assert steps(runner).count("pull") == 0
    assert mutating_calls(runner) == ()
    refused = [gate for gate in outcome.gates if not gate.passed]
    assert [(gate.node, gate.gate) for gate in refused] == [("worker", "reachable")]
    # 入れない台の、残りの関門は流さない
    assert steps(runner) == ("uname", "df", "uname")


def test_not_enough_disk_refuses_with_the_required_and_the_available(tmp_path: Path) -> None:
    small: Mapping[NodeRole, str] = {
        "head": "        Avail\n2705780625408\n",
        "worker": "        Avail\n1024\n",
    }
    runner = pull_runner(tmp_path, PullScript(avail=small))
    confirmer = SpyConfirmer()
    outcome = im.pull_image(runner, serve_config(), NODES, confirmer=confirmer)

    assert outcome.status == "refused"
    assert confirmer.shown == []
    assert steps(runner).count("pull") == 0
    detail = next(gate.detail for gate in outcome.gates if not gate.passed)
    assert "要る量" in detail and "空いている量" in detail
    assert "1,024 バイト" in detail


def test_a_missing_remote_root_tells_to_push(tmp_path: Path) -> None:
    runner = pull_runner(tmp_path, PullScript(df_exit_code=1))
    outcome = im.pull_image(runner, serve_config(), NODES, confirmer=SpyConfirmer())

    assert outcome.status == "refused"
    detail = next(gate.detail for gate in outcome.gates if not gate.passed)
    assert "serve push" in detail
    assert steps(runner).count("pull") == 0


def test_a_pull_that_cannot_be_reached_is_an_error(tmp_path: Path) -> None:
    # 了承のあとに届かなかったことは、実行しての失敗 (終了コード 2) として包む。
    # どの台の、どの段かが分かる文にして、5.1 が迷わないようにする
    script = PullScript(pull=Reply(exit_code=255, stderr="ssh: connect to host port 22"))
    runner = pull_runner(tmp_path, script)
    with pytest.raises(im.ImageError, match="届かなかった") as caught:
        im.pull_image(runner, serve_config(), NODES, confirmer=SpyConfirmer())

    message = str(caught.value)
    assert "head" in message and IMAGE_REF in message
    assert "docker pull" in message or "取得" in message


def test_a_failed_pull_is_an_error(tmp_path: Path) -> None:
    script = PullScript(pull=Reply(exit_code=1, stderr="Error response from daemon: not found"))
    runner = pull_runner(tmp_path, script)
    with pytest.raises(im.ImageError, match="取得できなかった"):
        im.pull_image(runner, serve_config(), NODES, confirmer=SpyConfirmer())

    # 1 台目で失敗したら、2 台目には進まない (照合もしない)
    assert steps(runner) == ("uname", "df", "uname", "df", "pull")


def test_the_refused_outcome_carries_every_gate_that_was_run(tmp_path: Path) -> None:
    runner = pull_runner(tmp_path, PullScript(unreachable=("worker",)))
    outcome = im.pull_image(runner, serve_config(), NODES, confirmer=SpyConfirmer())
    assert [(gate.node, gate.gate) for gate in outcome.gates] == [
        ("head", "reachable"),
        ("head", "disk_space"),
        ("worker", "reachable"),
    ]


# --- ライセンスの表記の読み取り ------------------------------------------


def test_the_reading_container_is_started_detached_with_a_name_and_labels(tmp_path: Path) -> None:
    runner = license_runner(tmp_path)
    im.read_image_licenses(
        runner,
        inspect_config(),
        NODES,
        STARTED_AT,
        confirmer=SpyConfirmer(),
        sleep=FakeClock().sleep,
    )

    argv = next(call.argv for call in runner.runs if call.argv[:2] == ("docker", "run"))
    assert "-d" in argv, "切り離して起こしていない"
    assert "--name" in argv and INSPECT_PLAN.container_name in argv
    assert f"{LABEL_OWNER}={OWNER}" in argv, "所有のラベルがない"
    # 終了時に自動で消す指定と、前面で動かす指定は、どの経路でも作らない
    for forbidden in ("--rm", "-i", "-t", "-it", "-a", "--attach", "--interactive", "--tty"):
        assert forbidden not in argv, f"例外の経路の指定が混ざった: {forbidden}"
    # 読むファイルの道筋は、構成の位置の引数をそのまま使う
    assert argv[-2:] == (LICENSE_PATH, NOTICE_PATH)
    assert argv[-3] == IMAGE_REF


def test_the_reading_starts_waits_reads_stops_and_removes(tmp_path: Path) -> None:
    runner = license_runner(tmp_path)
    clock = FakeClock()
    outcome = im.read_image_licenses(
        runner,
        inspect_config(),
        NODES,
        STARTED_AT,
        confirmer=SpyConfirmer(),
        sleep=clock.sleep,
        clock=clock.monotonic,
    )

    assert outcome.status == "read"
    # 2 つめの `ps` は、巻き戻しの直前の確かめ (一覧にその名前があるときだけ、止めて消す)
    assert after_the_run(runner) == ("run", "ps", "container inspect", "logs", "ps", "stop", "rm")
    assert clock.slept == [], "すぐ終わったのに眠った"
    reading = outcome.readings[0]
    assert reading.node == "head"
    assert reading.container_name == INSPECT_PLAN.container_name
    assert reading.exit_code == 0
    assert reading.stdout == LICENSE_TEXT
    assert LICENSE_TEXT.strip() in outcome.text


def test_a_failing_cat_keeps_the_output_and_the_exit_code(tmp_path: Path) -> None:
    script = LicenseScript(
        states=(Reply(stdout="exited 1\n"),),
        logs=Reply(
            stdout=LICENSE_TEXT,
            stderr=f"cat: {NOTICE_PATH}: No such file or directory\n",
        ),
    )
    runner = license_runner(tmp_path, script)
    outcome = im.read_image_licenses(
        runner,
        inspect_config(),
        NODES,
        STARTED_AT,
        confirmer=SpyConfirmer(),
        sleep=FakeClock().sleep,
    )

    reading = outcome.readings[0]
    assert reading.exit_code == 1
    assert LICENSE_PATH not in reading.stderr
    assert "No such file" in reading.stderr
    assert reading.stdout == LICENSE_TEXT
    # 読めたぶんと、読めなかったぶんの両方が、表示する文字列に出る
    assert "No such file" in outcome.text
    assert after_the_run(runner) == ("run", "ps", "container inspect", "logs", "ps", "stop", "rm")


def test_a_timeout_reads_stops_removes_and_fails(tmp_path: Path) -> None:
    script = LicenseScript(states=(Reply(stdout="running 0\n"),))
    runner = license_runner(tmp_path, script)
    clock = FakeClock(step=30.0)
    with pytest.raises(im.ImageError, match="終わらなかった") as caught:
        im.read_image_licenses(
            runner,
            inspect_config(ready_timeout_s=120),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=clock.sleep,
            clock=clock.monotonic,
        )

    # 読めた分も、誤りの文に入れる (計測者が、どこまで読めたかを見られるように)
    assert LICENSE_TEXT.strip() in str(caught.value)
    named = after_the_run(runner)
    assert named[0] == "run"
    assert named[-4:] == ("logs", "ps", "stop", "rm"), f"読んでから止めて消していない: {named}"
    assert named.count("container inspect") >= 2, "1 度も待たずに時間切れにした"
    # 待ちの間隔は引数から来る。試験は、実際には眠らない
    assert clock.slept and set(clock.slept) == {im.POLL_INTERVAL_S}


def test_a_failed_start_without_our_container_does_not_roll_back(tmp_path: Path) -> None:
    """**標準エラーの文面に頼らない**ことを固定する。

    `docker run` が、衝突とは違う文面で失敗し、自分のラベルで絞った一覧に、その名前の行が
    ない。`docker stop <名前>` / `docker rm <名前>` は、その名前のコンテナが誰のものでも
    止めて消すので、一覧で確かめられないかぎり、名前に向けて流さない (requirements 2.3)。
    """
    script = LicenseScript(
        run=Reply(exit_code=125, stderr="docker: Error response from daemon: something else\n"),
        listings=(Reply(stdout=""),),
    )
    runner = license_runner(tmp_path, script)
    with pytest.raises(im.ImageError, match="起こせなかった"):
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
        )

    named = steps(runner)
    assert "stop" not in named and "rm" not in named, (
        f"一覧にない名前に、止めると消すを向けた: {named}"
    )


def test_a_failed_start_still_removes_the_container(tmp_path: Path) -> None:
    script = LicenseScript(
        run=Reply(exit_code=125, stderr="docker: Error response from daemon: no such file\n"),
        states=(Reply(stdout="created 0\n"), Reply(stdout="exited 127\n")),
    )
    runner = license_runner(tmp_path, script)
    clock = FakeClock()
    with pytest.raises(im.ImageError, match="起こせなかった"):
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=clock.sleep,
            clock=clock.monotonic,
        )

    named = after_the_run(runner)
    assert named[-2:] == ("stop", "rm"), f"起動に失敗したのに片付けていない: {named}"


def test_a_name_conflict_does_not_roll_back(tmp_path: Path) -> None:
    conflict = (
        "docker: Error response from daemon: Conflict. The container name"
        f' "/{INSPECT_PLAN.container_name}" is already in use by container'
        ' "abc123def456". You have to remove (or rename) that container to be able to'
        " reuse that name.\n"
    )
    # 衝突した名前のコンテナは、この道具のものではないので、自分の一覧には出ない
    script = LicenseScript(run=Reply(exit_code=125, stderr=conflict), listings=(Reply(stdout=""),))
    runner = license_runner(tmp_path, script)
    with pytest.raises(im.ImageError, match="衝突") as caught:
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
        )

    assert INSPECT_PLAN.container_name in str(caught.value)
    named = steps(runner)
    assert "stop" not in named and "rm" not in named, (
        f"名前の衝突なのに、自分のものでないコンテナを止めた: {named}"
    )
    # 起こしたあとに出るのは、一覧の読み取りだけ (状態を変える呼び出しは、取得も停止もない)
    assert set(after_the_run(runner)) == {"run", "ps"}


def test_an_interruption_still_removes_the_container(tmp_path: Path) -> None:
    script = LicenseScript(logs=Reply(raises=KeyboardInterrupt()))
    runner = license_runner(tmp_path, script)
    with pytest.raises(KeyboardInterrupt):
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
        )

    assert after_the_run(runner)[-2:] == ("stop", "rm"), "中断でコンテナが残った"


def test_an_unreachable_docker_run_still_removes_our_container(tmp_path: Path) -> None:
    """`docker run` そのものが届かなくても (ssh の時間切れ、切断)、片付けの道を通る。

    遠隔の `docker run -d` は、ssh が切れたあとも完了しうるので、コンテナが残りうる。
    一覧にその名前があれば、止めて消す。
    """
    script = LicenseScript(
        run=Reply(exit_code=255, stderr="ssh: connect to host port 22: No route to host"),
        listings=(Reply(stdout=""), Reply(stdout=ps_line(INSPECT_PLAN) + "\n")),
    )
    runner = license_runner(tmp_path, script)
    with pytest.raises(im.ImageError, match="届かなかった"):
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
        )

    assert after_the_run(runner)[-2:] == ("stop", "rm"), "届かなかったときに、コンテナが残った"


def test_an_unreachable_docker_run_does_not_touch_a_name_we_cannot_claim(tmp_path: Path) -> None:
    # 一覧にその名前がなければ、コンテナはできていない (または、よそのもの) ので、行かない
    script = LicenseScript(
        run=Reply(exit_code=255, stderr="ssh: connect to host port 22: No route to host"),
        listings=(Reply(stdout=""),),
    )
    runner = license_runner(tmp_path, script)
    with pytest.raises(im.ImageError, match="届かなかった"):
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
        )

    named = steps(runner)
    assert "stop" not in named and "rm" not in named


def test_an_unreachable_run_with_an_unreadable_listing_tells_to_use_serve_stop(
    tmp_path: Path,
) -> None:
    # 一覧を読めないと、片付けてよいかを確かめられない。残っているかもしれないことを言う
    script = LicenseScript(
        run=Reply(exit_code=255, stderr="ssh: connect to host port 22: No route to host"),
        listings=(Reply(stdout=""), Reply(exit_code=255, stderr="ssh: connect to host")),
    )
    runner = license_runner(tmp_path, script)
    with pytest.raises(im.ImageError) as caught:
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
        )

    message = str(caught.value)
    assert "serve stop" in message, f"片付けの案内がない: {message}"
    assert INSPECT_PLAN.container_name in message
    named = steps(runner)
    assert "stop" not in named and "rm" not in named


def test_a_listing_that_cannot_be_read_after_the_start_is_an_error(tmp_path: Path) -> None:
    # 何も読めていないのに、正常に終わらない (5.1 が終了コード 0 に写さない)
    script = LicenseScript(
        listings=(
            Reply(stdout=""),
            Reply(exit_code=255, stderr="ssh: connect to host"),
            Reply(stdout=ps_line(INSPECT_PLAN) + "\n"),
        )
    )
    runner = license_runner(tmp_path, script)
    with pytest.raises(im.ImageError, match="見つけられなかった|読めなかった"):
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
        )

    assert after_the_run(runner)[-2:] == ("stop", "rm"), "片付けを通っていない"


def test_a_container_that_is_not_in_the_listing_is_an_error(tmp_path: Path) -> None:
    script = LicenseScript(
        listings=(
            Reply(stdout=""),
            Reply(stdout=""),
            Reply(stdout=ps_line(INSPECT_PLAN) + "\n"),
        )
    )
    runner = license_runner(tmp_path, script)
    with pytest.raises(im.ImageError, match="見つけられなかった"):
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
        )

    assert after_the_run(runner)[-2:] == ("stop", "rm"), "片付けを通っていない"


def test_an_unreadable_container_output_is_an_error(tmp_path: Path) -> None:
    # `docker logs` が 0 以外で終わる (コンテナの `cat` の失敗とは別物)
    script = LicenseScript(logs=Reply(exit_code=1, stderr="Error: No such container"))
    runner = license_runner(tmp_path, script)
    with pytest.raises(im.ImageError, match="読めなかった"):
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
        )

    assert after_the_run(runner)[-2:] == ("stop", "rm"), "片付けを通っていない"


def test_an_interrupted_stop_still_removes_the_container(tmp_path: Path) -> None:
    # 片付けの途中で中断が来ても、消すところまでは試みる (コンテナを残さないことを優先)
    script = LicenseScript(stop=Reply(raises=KeyboardInterrupt()))
    runner = license_runner(tmp_path, script)
    with pytest.raises(KeyboardInterrupt):
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
        )

    assert "rm" in steps(runner), "止める最中の中断で、消すのを試みなかった"


def test_a_second_interruption_during_the_cleanup_is_passed_on(tmp_path: Path) -> None:
    script = LicenseScript(
        stop=Reply(raises=KeyboardInterrupt()), remove=Reply(raises=KeyboardInterrupt())
    )
    runner = license_runner(tmp_path, script)
    with pytest.raises(KeyboardInterrupt):
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
        )

    named = steps(runner)
    assert named.count("stop") == 1 and named.count("rm") == 1


def test_the_cleanup_problems_are_reported_when_we_are_interrupted(tmp_path: Path) -> None:
    # 中断の経路でも、片付けの失敗を捨てない (出力の先は、差し替えられる)
    script = LicenseScript(
        logs=Reply(raises=KeyboardInterrupt()),
        remove=Reply(exit_code=1, stderr="Error: No such container"),
    )
    runner = license_runner(tmp_path, script)
    reported = io.StringIO()
    with pytest.raises(KeyboardInterrupt):
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
            report=reported,
        )

    shown = reported.getvalue()
    assert INSPECT_PLAN.container_name in shown, f"片付けの失敗を捨てた: {shown!r}"
    assert shown.count("\n") == 1, f"1 行で知らせる: {shown!r}"


def test_a_cleanup_that_fails_is_an_error(tmp_path: Path) -> None:
    script = LicenseScript(remove=Reply(exit_code=1, stderr="Error: No such container"))
    runner = license_runner(tmp_path, script)
    with pytest.raises(im.ImageError, match="片付け") as caught:
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
        )
    assert INSPECT_PLAN.container_name in str(caught.value)


def test_nothing_is_started_when_the_operator_refuses(tmp_path: Path) -> None:
    runner = license_runner(tmp_path)
    with pytest.raises(ApprovalError):
        im.read_image_licenses(
            runner,
            inspect_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(approves=False),
            sleep=FakeClock().sleep,
        )

    assert "run" not in steps(runner)
    assert mutating_calls(runner) == ()


def test_a_refused_gate_stops_before_the_approval(tmp_path: Path) -> None:
    # 自分のコンテナが動いている (同時に動かすのは、1 つの構成ぶんだけ)
    running = ps_line(INSPECT_PLAN, state="running", container_id="feedfacefeed")
    script = LicenseScript(listings=(Reply(stdout=running + "\n"),))
    runner = license_runner(tmp_path, script)
    confirmer = SpyConfirmer()
    outcome = im.read_image_licenses(
        runner,
        inspect_config(),
        NODES,
        STARTED_AT,
        confirmer=confirmer,
        sleep=FakeClock().sleep,
    )

    assert outcome.status == "refused"
    assert outcome.readings == ()
    assert confirmer.shown == []
    assert "run" not in steps(runner)
    assert mutating_calls(runner) == ()
    assert any(gate.gate == "own_state" and not gate.passed for gate in outcome.gates)


def test_the_licenses_command_needs_an_inspect_config(tmp_path: Path) -> None:
    runner = license_runner(tmp_path)
    with pytest.raises(ConfigError, match="inspect"):
        im.read_image_licenses(
            runner,
            serve_config(),
            NODES,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            sleep=FakeClock().sleep,
        )
    assert runner.calls == ()


def test_the_paths_come_from_the_config_only() -> None:
    # 読むファイルの候補を、この module に書かない (値を書くのは 6.2 の仕事)
    text = Path(inspect_module.getfile(im)).read_text(encoding="utf-8")
    for candidate in ("LICENSE", "NOTICE", "NGC-DL-CONTAINER-LICENSE"):
        assert f"/{candidate}" not in text, (
            f"読むファイルの道筋が module に書かれている: {candidate}"
        )


# --- コンテナを対象にする操作の不変条件 ----------------------------------

_CONTAINER_SUBCOMMANDS = frozenset(
    {"stop", "rm", "logs", "top", "kill", "start", "restart", "pause", "unpause", "wait", "port"}
)
"""コンテナを対象に取る docker のサブコマンド (許可の一覧の外のものも並べて見張る)。"""

_VALUE_FLAGS = frozenset(
    {"-t", "--time", "--tail", "--since", "--until", "-s", "--signal", "--format"}
)
"""値を取るフラグ (`docker stop -t 90 <名前>` の 90 を、対象と読み違えないため)。

`test_guards.py` の同じ読み手に `--format` を足してある (`docker container inspect` の書式を、
対象と読み違えないため。`docker rm -f` は値を取らないので、足さない)。
"""


def container_targets(argv: Sequence[str]) -> tuple[str, ...]:
    """docker の呼び出しから、コンテナを対象にしている語だけを取り出す
    (`test_guards.py` と同じ読み方)。"""
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
    assert container_targets(("docker", "logs", CONTAINER_ID)) == (CONTAINER_ID,)
    assert container_targets(("docker", "container", "inspect", CONTAINER_ID)) == (CONTAINER_ID,)
    formatted = ("docker", "container", "inspect", "--format", "{{.State.Status}}", CONTAINER_ID)
    assert container_targets(formatted) == (CONTAINER_ID,)
    assert container_targets(("docker", "rm", "-f", "vb-x-head")) == ("vb-x-head",)
    assert container_targets(("docker", "run", "-d", "--name", "vb-x-head", IMAGE_REF)) == (
        "vb-x-head",
    )
    # イメージを対象にする呼び出しと、取得と、一覧は、コンテナの対象ではない
    image_inspect = ("docker", "image", "inspect", "--format", "{{json .X}}", IMAGE_REF)
    assert container_targets(image_inspect) == ()
    assert container_targets(("docker", "pull", IMAGE_REF)) == ()
    assert container_targets(("docker", "ps", "-a", "--filter", "label=x")) == ()


def test_every_docker_call_targets_only_our_containers(tmp_path: Path) -> None:
    runner = license_runner(tmp_path)
    im.read_image_licenses(
        runner,
        inspect_config(),
        NODES,
        STARTED_AT,
        confirmer=SpyConfirmer(),
        sleep=FakeClock().sleep,
    )

    # 「一覧が返したもの = 自分のもの」を前提にすると循環するので、試験の側でも、台本に
    # 書いた行の所有のラベルを見てから、触ってよい識別子として数える
    row = json.loads(ps_line(INSPECT_PLAN))
    assert f"{LABEL_OWNER}={OWNER}" in row["Labels"]
    from_listing = {row["ID"]}
    from_plan = {INSPECT_PLAN.container_name}
    checked = 0
    for argv in runner.argvs:
        for target in container_targets(argv):
            checked += 1
            assert target in from_listing | from_plan, (
                f"ラベルで絞った一覧にも、了承済みの計画にもない対象: {target} ({argv})"
            )
    assert checked >= 4, "見張るべき対象が 1 つも無い (run, container inspect, logs, stop, rm)"


def test_the_container_list_is_always_filtered_by_our_label(tmp_path: Path) -> None:
    runner = license_runner(tmp_path)
    im.read_image_licenses(
        runner,
        inspect_config(),
        NODES,
        STARTED_AT,
        confirmer=SpyConfirmer(),
        sleep=FakeClock().sleep,
    )
    listings = [argv for argv in runner.argvs if argv[:2] == ("docker", "ps")]
    assert listings, "コンテナの一覧を 1 度も取っていない"
    for argv in listings:
        assert f"label={OWNER_FILTER}" in argv, f"絞らない一覧を取った: {argv}"


def test_no_build_no_rmi_and_no_tag_pull_anywhere(tmp_path: Path) -> None:
    pull = pull_runner(tmp_path)
    im.pull_image(pull, serve_config(), NODES, confirmer=SpyConfirmer())
    licenses = license_runner(tmp_path)
    im.read_image_licenses(
        licenses,
        inspect_config(),
        NODES,
        STARTED_AT,
        confirmer=SpyConfirmer(),
        sleep=FakeClock().sleep,
    )

    for argv in (*pull.argvs, *licenses.argvs):
        assert argv[:2] != ("docker", "build")
        assert argv[:2] != ("docker", "rmi")
        if argv[:2] == ("docker", "pull"):
            assert "@sha256:" in argv[-1], f"タグでの取得が出た: {argv}"


# --- 依存の向き ---------------------------------------------------------

ALLOWED_SERVING_MODULES = {"types", "config", "remote", "plan", "observe", "guards", "logs"}


def test_image_imports_only_the_modules_the_design_allows() -> None:
    tree = ast.parse(Path(inspect_module.getfile(im)).read_text(encoding="utf-8"))
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


def test_the_gate_results_are_the_shared_type(tmp_path: Path) -> None:
    runner = pull_runner(tmp_path)
    outcome = im.pull_image(runner, serve_config(), NODES, confirmer=SpyConfirmer())
    assert all(isinstance(gate, GateResult) for gate in outcome.gates)
