"""試験用の偽の実行役そのものの試験 (task 1.4)。

偽物 (`tests/fake_runner.py`) は、あとのタスク (2.3、2.4、3.x、4.x、5.x) が書き換えずに
使うので、ここで振る舞いを固定する。

確かめること:

- 呼ばれた (ノード、引数の列、`mutating`、`push` / `pull` の引数) と順序を記録する
- 台本は、引数の列の前方一致と述語で結果を選べ、同じコマンドの 1 回目と 2 回目で
  違う結果を返せる。`push` と `pull` の規則は、ノードと道筋で選び分けられる
- 台本に合わない呼び出しは、既定では誤りにする (`default` を渡すと、空の成功を返す)
- 偽物も、実物と同じ、許可の一覧と、了承の検査を通す (これが緩いと、5.3 の安全の試験が
  意味を失う)
- 失敗の意味が実物と揃っている (`raises` で例外、終了コード 255 は `RemoteError`)
- `pull` は、実物と同じく宛先のディレクトリを作り、台本の指定したファイルを置く
- 偽物が `RemoteRunner` を満たすことを、mypy で固定する
- 実物の ssh と rsync を、どの段でも呼ばない
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any

import pytest

from fake_runner import FakeRunner, Reply, Rule
from serving_kit.remote import RemoteError, RemoteRunner
from serving_kit.types import ApprovedPlan, NodeDef, PlannedPush, PlannedRun

REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"

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

CONTAINER = "vb-probe-pinned-head"
IMAGE = "vllm/vllm-openai@sha256:" + "b0" * 32
RUN_ARGV = ("docker", "run", "-d", "--name", CONTAINER, IMAGE)
STOP_ARGV = ("docker", "stop", CONTAINER)


@pytest.fixture(autouse=True)
def no_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """偽物が、うっかり実物の ssh や rsync を呼ばないことを、どの試験でも固定する。"""

    def explode(argv: Sequence[str], **kwargs: Any) -> None:
        raise AssertionError(f"偽の実行役が実物のコマンドを呼んだ: {list(argv)}")

    monkeypatch.setattr(subprocess, "run", explode)


@pytest.fixture
def var_root(tmp_path: Path) -> Path:
    root = tmp_path / "var"
    root.mkdir()
    return root


def approved(*, forward: Sequence[Any], rollback: Sequence[PlannedRun] = ()) -> ApprovedPlan:
    return ApprovedPlan(
        forward=tuple(forward),
        rollback=tuple(rollback),
        own_container_names=(CONTAINER,),
    )


# --- 記録 ---------------------------------------------------------------


def test_the_fake_records_the_calls_in_order(var_root: Path) -> None:
    """ノード、引数の列、`mutating` と、呼ばれた順序を記録する。"""
    runner = FakeRunner(var_root=var_root, default=Reply())

    runner.run(HEAD, ["uname", "-n"], timeout_s=5.0, mutating=False)
    runner.run(WORKER, ["df", "-B1", "--output=avail", REMOTE_ROOT], timeout_s=5.0, mutating=False)

    assert [(call.node, call.argv) for call in runner.calls] == [
        ("head", ("uname", "-n")),
        ("worker", ("df", "-B1", "--output=avail", REMOTE_ROOT)),
    ]
    assert [call.kind for call in runner.calls] == ["run", "run"]
    assert [call.mutating for call in runner.calls] == [False, False]
    assert runner.calls[0].timeout_s == 5.0


def test_the_fake_records_the_transfers(var_root: Path, tmp_path: Path) -> None:
    """`push` と `pull` の引数も記録する。"""
    local = tmp_path / "payload"
    local.mkdir()
    plan = approved(
        forward=[PlannedPush(node="head", local_dir=local, remote_subdir="payload", delete=True)]
    )
    runner = FakeRunner(var_root=var_root, plan=plan, default=Reply())

    runner.push(HEAD, local, "payload", delete=True)
    runner.pull(HEAD, "logs/nccl.log", var_root / "logs")

    push, pull = runner.pushes[0], runner.pulls[0]
    assert (push.node, push.local_dir, push.remote, push.delete) == ("head", local, "payload", True)
    assert push.mutating is True
    assert (pull.node, pull.remote, pull.local_dir) == ("head", "logs/nccl.log", var_root / "logs")
    assert pull.mutating is False


def test_the_fake_exposes_the_run_argvs(var_root: Path) -> None:
    """流れた引数の列だけを、順に取り出せる (安全の試験が全体を見るため)。"""
    runner = FakeRunner(var_root=var_root, default=Reply())

    runner.run(HEAD, ["uname", "-n"], timeout_s=5.0, mutating=False)
    runner.pull(HEAD, "logs/nccl.log", var_root / "logs")

    assert runner.argvs == (("uname", "-n"),)


# --- 台本 ---------------------------------------------------------------


def test_the_script_selects_a_reply_by_prefix(var_root: Path) -> None:
    """台本は、引数の列の前方一致で結果を選べる。"""
    runner = FakeRunner(
        var_root=var_root,
        script=[
            Rule(("docker", "version"), (Reply(stdout="28.0.1"),)),
            Rule(("docker", "ps"), (Reply(stdout="[]"),)),
        ],
    )

    assert runner.run(HEAD, ["docker", "version"], timeout_s=5.0, mutating=False).stdout == "28.0.1"
    assert runner.run(HEAD, ["docker", "ps", "-a"], timeout_s=5.0, mutating=False).stdout == "[]"


def test_the_script_selects_a_reply_by_predicate(var_root: Path) -> None:
    """台本は、述語でも結果を選べる。"""
    runner = FakeRunner(
        var_root=var_root,
        script=[
            Rule(when=lambda argv: argv[0] == "test" and argv[-1].endswith("/models")),
            Rule(when=lambda argv: argv[0] == "test", replies=(Reply(exit_code=1),)),
        ],
    )

    assert (
        runner.run(HEAD, ["test", "-d", f"{REMOTE_ROOT}/models"], timeout_s=5.0, mutating=False)
    ).exit_code == 0
    assert (
        runner.run(HEAD, ["test", "-d", f"{REMOTE_ROOT}/probe"], timeout_s=5.0, mutating=False)
    ).exit_code == 1


def test_the_script_can_answer_differently_on_the_second_call(var_root: Path) -> None:
    """同じコマンドの 1 回目と 2 回目で、違う結果を返せる (尽きたら最後を繰り返す)。"""
    runner = FakeRunner(
        var_root=var_root,
        script=[
            Rule(
                ("docker", "container", "inspect"),
                (Reply(stdout="running"), Reply(stdout="exited")),
            )
        ],
    )
    argv = ["docker", "container", "inspect", CONTAINER]

    assert runner.run(HEAD, argv, timeout_s=5.0, mutating=False).stdout == "running"
    assert runner.run(HEAD, argv, timeout_s=5.0, mutating=False).stdout == "exited"
    assert runner.run(HEAD, argv, timeout_s=5.0, mutating=False).stdout == "exited"


def test_the_script_can_be_scoped_to_one_node(var_root: Path) -> None:
    """台本は、ノードごとに違う結果を返せる (2 台の差を作るため)。"""
    runner = FakeRunner(
        var_root=var_root,
        script=[
            Rule(("uname", "-n"), (Reply(stdout="spark-153d"),), node="head"),
            Rule(("uname", "-n"), (Reply(stdout="spark-5083"),), node="worker"),
        ],
    )

    assert runner.run(HEAD, ["uname", "-n"], timeout_s=5.0, mutating=False).stdout == "spark-153d"
    assert runner.run(WORKER, ["uname", "-n"], timeout_s=5.0, mutating=False).stdout == "spark-5083"


def test_the_result_carries_the_argv_that_was_asked_for(var_root: Path) -> None:
    """結果の `argv` は、呼ばれた引数の列である。"""
    runner = FakeRunner(var_root=var_root, default=Reply())

    result = runner.run(HEAD, ["uname", "-n"], timeout_s=5.0, mutating=False)

    assert result.argv == ("uname", "-n")


def test_a_call_the_script_does_not_cover_is_an_error(var_root: Path) -> None:
    """台本に合わない呼び出しは、既定では誤りにする (見落ちを黙って通さない)。"""
    runner = FakeRunner(var_root=var_root, script=[Rule(("uname", "-n"))])

    with pytest.raises(AssertionError, match="台本"):
        runner.run(HEAD, ["df", "-B1", REMOTE_ROOT], timeout_s=5.0, mutating=False)


def test_a_default_reply_covers_the_calls_the_script_does_not(var_root: Path) -> None:
    """`default` を渡すと、台本に合わない呼び出しは、その結果になる。"""
    runner = FakeRunner(var_root=var_root, default=Reply(stdout=""))

    assert runner.run(HEAD, ["df", "-B1"], timeout_s=5.0, mutating=False).exit_code == 0


# --- 許可の一覧と、了承の検査 (実物と同じもの) --------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["sudo", "-i"],
        ["apt", "install", "rsync"],
        ["pip", "install", "vllm"],
        ["systemctl", "restart", "docker"],
        ["/usr/bin/docker", "ps"],
        ["docker", "exec", CONTAINER, "sh"],
        ["docker", "build", "."],
        ["docker", "system", "prune"],
        ["docker", "-H", "tcp://10.0.1.61:2375", "ps"],
        ["ip", "link", "set", "dev", "enp1s0f0np0", "mtu", "9000"],
        ["ip", "route", "del", "default"],
        ["ethtool", "-s", "enp1s0f0np0", "speed", "100"],
        ["cat", "/etc/shadow"],
        ["cat", f"{REMOTE_ROOT}/../x"],
        [],
    ],
)
def test_the_fake_refuses_what_the_real_runner_refuses(var_root: Path, argv: list[str]) -> None:
    """偽物も、許可の一覧の検査を通す (偽物だけが緩いと、安全の試験が意味を失う)。"""
    runner = FakeRunner(var_root=var_root, default=Reply())

    with pytest.raises(RuntimeError):
        runner.run(HEAD, argv, timeout_s=5.0, mutating=False)

    assert runner.calls == ()


@pytest.mark.parametrize(
    "argv",
    [list(RUN_ARGV), list(STOP_ARGV), ["docker", "pull", IMAGE], ["mkdir", "-p", REMOTE_ROOT]],
)
def test_the_fake_refuses_an_unapproved_state_change(var_root: Path, argv: list[str]) -> None:
    """了承を得た計画がなければ、偽物も状態を変える呼び出しを断る。"""
    runner = FakeRunner(var_root=var_root, default=Reply())

    with pytest.raises(RuntimeError, match="了承"):
        runner.run(HEAD, argv, timeout_s=5.0, mutating=False)

    assert runner.calls == ()


def test_the_fake_records_an_approved_state_change_as_mutating(var_root: Path) -> None:
    """了承した計画に含まれていれば流れ、状態を変える呼び出しとして記録される。"""
    plan = approved(forward=[PlannedRun(node="head", argv=RUN_ARGV, container=CONTAINER)])
    runner = FakeRunner(var_root=var_root, plan=plan, default=Reply())

    runner.run(HEAD, list(RUN_ARGV), timeout_s=5.0, mutating=True)

    assert runner.calls[0].mutating is True


def test_the_fake_lets_the_rollback_run_after_a_failure(var_root: Path) -> None:
    """偽物でも、巻き戻しは、前に進むコマンドが失敗したあとで実行できる。"""
    plan = approved(
        forward=[PlannedRun(node="head", argv=RUN_ARGV, container=CONTAINER)],
        rollback=[PlannedRun(node="head", argv=STOP_ARGV, container=CONTAINER)],
    )
    runner = FakeRunner(
        var_root=var_root,
        plan=plan,
        script=[
            Rule(("docker", "run"), (Reply(exit_code=125, stderr="no kernel image"),)),
            Rule(("docker", "stop")),
        ],
    )

    assert runner.run(HEAD, list(RUN_ARGV), timeout_s=5.0, mutating=True).exit_code == 125
    assert runner.run(HEAD, list(STOP_ARGV), timeout_s=5.0, mutating=True).exit_code == 0


def test_the_fake_can_be_approved_once_after_it_was_built(var_root: Path) -> None:
    """了承は、あとから 1 度だけ渡せる (実物と同じ)。"""
    runner = FakeRunner(var_root=var_root, default=Reply())
    plan = approved(forward=[PlannedRun(node="head", argv=RUN_ARGV, container=CONTAINER)])

    runner.approve(plan)
    runner.run(HEAD, list(RUN_ARGV), timeout_s=5.0, mutating=True)

    with pytest.raises(RuntimeError):
        runner.approve(plan)


@pytest.mark.parametrize("remote_subdir", ["models", "cache", "logs", "probe", "..", "/etc"])
def test_the_fake_refuses_a_push_outside_the_allowed_destinations(
    var_root: Path, tmp_path: Path, remote_subdir: str
) -> None:
    """偽物も、配布の宛先の決まりを通す。"""
    local = tmp_path / "payload"
    local.mkdir()
    plan = approved(
        forward=[
            PlannedPush(node="head", local_dir=local, remote_subdir=remote_subdir, delete=False)
        ]
    )
    runner = FakeRunner(var_root=var_root, plan=plan, default=Reply())

    with pytest.raises(RuntimeError, match="配布"):
        runner.push(HEAD, local, remote_subdir, delete=False)

    assert runner.calls == ()


def test_the_fake_refuses_an_unapproved_push(var_root: Path, tmp_path: Path) -> None:
    local = tmp_path / "payload"
    local.mkdir()
    runner = FakeRunner(var_root=var_root, default=Reply())

    with pytest.raises(RuntimeError, match="了承"):
        runner.push(HEAD, local, "payload", delete=False)


def test_the_fake_refuses_a_pull_outside_the_record_root(var_root: Path, tmp_path: Path) -> None:
    """偽物も、回収の宛先と元の決まりを通す。"""
    runner = FakeRunner(var_root=var_root, default=Reply())

    with pytest.raises(RuntimeError, match="回収"):
        runner.pull(HEAD, "logs/nccl.log", tmp_path / "elsewhere")
    with pytest.raises(RuntimeError, match="回収"):
        runner.pull(HEAD, "/etc/passwd", var_root / "logs")

    assert runner.calls == ()


@pytest.mark.parametrize("remote_path", ["logs/a b.log", "logs/$(id)", "~/logs"])
def test_the_fake_refuses_a_pull_with_shell_characters(var_root: Path, remote_path: str) -> None:
    """偽物も、遠隔の道筋の文字の絞り込みを通す。"""
    runner = FakeRunner(var_root=var_root, default=Reply())

    with pytest.raises(RuntimeError, match="使えない文字"):
        runner.pull(HEAD, remote_path, var_root / "logs")


# --- 失敗の意味を、実物と揃える -----------------------------------------


def test_the_fake_satisfies_the_remote_runner_protocol(var_root: Path) -> None:
    """偽物が `RemoteRunner` を満たすことを、mypy で固定する。"""
    fake: RemoteRunner = FakeRunner(var_root=var_root, default=Reply())

    assert fake.run(HEAD, ["uname", "-n"], timeout_s=5.0, mutating=False).exit_code == 0


def test_a_reply_can_raise_instead_of_returning(var_root: Path) -> None:
    """台本は、結果を返すかわりに例外を投げられる (時間切れの経路を作るため)。"""
    boom = RemoteError(
        "5.0 秒のうちに終わらなかった", node="head", ssh_host="spark-153d", argv=("uname", "-n")
    )
    runner = FakeRunner(var_root=var_root, script=[Rule(("uname", "-n"), (Reply(raises=boom),))])

    with pytest.raises(RemoteError) as caught:
        runner.run(HEAD, ["uname", "-n"], timeout_s=5.0, mutating=False)

    assert caught.value is boom
    assert runner.calls[0].argv == ("uname", "-n")


def test_the_fake_turns_exit_code_255_into_a_remote_error(var_root: Path) -> None:
    """終了コード 255 は、実物と同じく接続の失敗なので `RemoteError` にする。"""
    runner = FakeRunner(
        var_root=var_root,
        script=[Rule(("uname", "-n"), (Reply(exit_code=255, stderr="connect: timed out"),))],
    )

    with pytest.raises(RemoteError) as caught:
        runner.run(HEAD, ["uname", "-n"], timeout_s=5.0, mutating=False)

    assert caught.value.node == "head"
    assert caught.value.ssh_host == "spark-153d"


def test_the_fake_turns_a_failed_transfer_255_into_a_remote_error(
    var_root: Path, tmp_path: Path
) -> None:
    """配布と回収でも、255 は `RemoteError` になる。"""
    runner = FakeRunner(
        var_root=var_root, script=[Rule(kind="pull", replies=(Reply(exit_code=255),))]
    )

    with pytest.raises(RemoteError):
        runner.pull(HEAD, "logs/nccl.log", var_root / "logs")


# --- 記録の回収を、台本で作る (タスク 2.4 のため) ------------------------


def test_the_fake_pull_creates_the_destination_like_the_real_runner(var_root: Path) -> None:
    """`pull` は、実物と同じく、宛先のディレクトリを作る。"""
    runner = FakeRunner(var_root=var_root, default=Reply())
    into = var_root / "logs" / "2026-09-21-start"

    runner.pull(HEAD, "logs/nccl.log", into)

    assert into.is_dir()


def test_a_pull_reply_can_write_the_records_it_brings_back(var_root: Path) -> None:
    """回収の返事は、宛先に置くファイル (名前と中身) を台本で指定できる。"""
    runner = FakeRunner(
        var_root=var_root,
        script=[
            Rule(
                kind="pull",
                remote_prefix="logs/",
                replies=(Reply(writes=(("nccl.log", "NCCL INFO Using network IB\n"),)),),
            )
        ],
    )
    into = var_root / "logs" / "head"

    runner.pull(HEAD, "logs/nccl.log", into)

    assert (into / "nccl.log").read_text(encoding="utf-8") == "NCCL INFO Using network IB\n"


def test_the_transfer_rules_can_be_selected_by_node_and_path(
    var_root: Path, tmp_path: Path
) -> None:
    """`push` と `pull` の規則は、ノードと道筋で選び分けられる。

    タスク 2.4 の「head の記録はある、worker の記録はない」を、台本で作れること。
    """
    runner = FakeRunner(
        var_root=var_root,
        script=[
            Rule(
                kind="pull",
                node="head",
                remote_prefix="logs/",
                replies=(Reply(writes=(("a.log", "head\n"),)),),
            ),
            Rule(kind="pull", node="worker", remote_prefix="logs/", replies=(Reply(exit_code=23),)),
            Rule(kind="pull", remote_prefix="state/", replies=(Reply(stdout="state"),)),
        ],
    )

    head_logs = var_root / "logs" / "head"
    assert runner.pull(HEAD, "logs/a.log", head_logs).exit_code == 0
    assert (head_logs / "a.log").read_text(encoding="utf-8") == "head\n"
    assert runner.pull(WORKER, "logs/a.log", var_root / "logs" / "worker").exit_code == 23
    assert runner.pull(HEAD, "state/p1.launch.json", var_root / "state").stdout == "state"


def test_a_push_rule_can_be_selected_by_path(var_root: Path, tmp_path: Path) -> None:
    """配布の規則も、宛先の道筋で選び分けられる。"""
    local = tmp_path / "payload"
    local.mkdir()
    plan = approved(
        forward=[
            PlannedPush(node="head", local_dir=local, remote_subdir="payload", delete=True),
            PlannedPush(node="head", local_dir=local, remote_subdir="state", delete=False),
        ]
    )
    runner = FakeRunner(
        var_root=var_root,
        plan=plan,
        script=[
            Rule(kind="push", remote_prefix="payload", replies=(Reply(stdout="配った"),)),
            Rule(kind="push", remote_prefix="state", replies=(Reply(exit_code=1),)),
        ],
    )

    assert runner.push(HEAD, local, "payload", delete=True).stdout == "配った"
    assert runner.push(HEAD, local, "state", delete=False).exit_code == 1
