"""記録の回収と、置き場所の決まりの試験 (tasks.md 2.4)。

確かめること (design.md 「記録の置き場所」「remote」のコンテナを対象にする操作の不変条件、
tasks.md 2.4 の完了の状態):

- 偽の実行役で、2 台ぶんの記録 (自分のコンテナ、通信、起動) が、決まった場所にできる
  (requirements 1.9)
- **`docker logs` の対象は、`guards.list_own_containers` (自分のラベルで絞った一覧) の中で、
  名前が `ContainerPlan.container_name` と一致する行の ID だけである** (requirements 2.3、
  2.4)。この module は、了承済みの計画を持たずに呼ばれることがある (`serve logs`、起動の
  失敗のときの末尾の表示) ので、design の不変条件のうち「了承済みの計画が自分で起こす名前」
  は使えない。名前が一致する行が一覧に無い・一覧が読めないときは、`docker logs` を 1 つも
  出さない (同じ名前の、ラベルの無い別のコンテナを、うっかり読まない)
- `docker ps` は、つねに自分のラベルの絞り込みつきで呼ばれる
- 回収は、状態を変える呼び出しを 1 つも出さない (`mutating=True` が無い)
- 片方の台が入れない・自分のコンテナが無い・記録がまだ無いときに、もう片方の台のぶんは写り、
  写せなかったことが `collect.json` から分かる
- 置き場所の名前が、日時 (UTC)・コマンド・構成の名前から決まり、`..` や `/` を含む名前は断る
- 末尾だけを読む口が、2 台ぶんを返す (起動の失敗のときに使う)

実物の ssh、rsync、docker は、どの段でも呼ばない。自分のコンテナがあるときの
`docker ps --format json` の行は、**実機からまだ採れていない** (tasks.md 1.6 の「ここで採れ
ないもの」。7.1 で採って見本を足す予定)。ここでは、`test_guards.py` の `ps_line` に合わせた
形の合成の見本を使う。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from ipaddress import IPv4Address
from pathlib import Path

import pytest

from fake_runner import FakeRunner, Reply, Rule
from serving_kit import logs as lg
from serving_kit.plan import LABEL_IMAGE, LABEL_OWNER, OWNER, OWNER_FILTER
from serving_kit.types import ContainerPlan, NodeDef, NodeRole

# --- 見本の値 -----------------------------------------------------------

REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"
CONFIG_NAME = "p1-nvfp4-tp2"
STARTED_AT = datetime(2026, 9, 21, 3, 0, 0, tzinfo=UTC)

HEAD = NodeDef(
    role="head", ssh_host="spark-153d", lan_addr=IPv4Address("10.0.1.60"), remote_root=REMOTE_ROOT
)
WORKER = NodeDef(
    role="worker",
    ssh_host="spark-5083",
    lan_addr=IPv4Address("10.0.1.61"),
    remote_root=REMOTE_ROOT,
)
NODES: Mapping[NodeRole, NodeDef] = {"head": HEAD, "worker": WORKER}

_OWN_LABELS = {LABEL_OWNER: OWNER, LABEL_IMAGE: "vllm/vllm-openai@sha256:" + "b0" * 32}

HEAD_PLAN = ContainerPlan(
    node="head",
    container_name="vb-p1-nvfp4-tp2-head",
    labels=_OWN_LABELS,
    argv=("docker", "run", "-d", "--name", "vb-p1-nvfp4-tp2-head"),
)
WORKER_PLAN = ContainerPlan(
    node="worker",
    container_name="vb-p1-nvfp4-tp2-worker",
    labels=_OWN_LABELS,
    argv=("docker", "run", "-d", "--name", "vb-p1-nvfp4-tp2-worker"),
)
PLANS = (HEAD_PLAN, WORKER_PLAN)

LAUNCH_RECORD_PATH = lg.launch_record_remote_path(CONFIG_NAME)

HEAD_CONTAINER_ID = "aa11aa11aa11"
WORKER_CONTAINER_ID = "bb22bb22bb22"

# `guards.list_own_containers` が流す、ただ 1 つの読み取り (module の docstring と同じ形)
OWN_CONTAINERS_ARGV = (
    "docker",
    "ps",
    "-a",
    "--filter",
    f"label={OWNER_FILTER}",
    "--format",
    "json",
)


def _own_container_line(plan: ContainerPlan, *, container_id: str, state: str = "running") -> str:
    """`docker ps -a --filter label=... --format json` の 1 行 (自分のコンテナがある
    ときの合成の見本。実機の出力ではない。test_guards.py の `ps_line` に項目の名前を合わせた)。
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


def _own_containers_rule(node: NodeRole, reply: Reply) -> Rule:
    return Rule(node=node, kind="run", prefix=OWN_CONTAINERS_ARGV, replies=(reply,))


def _found_rule(plan: ContainerPlan, container_id: str) -> Rule:
    """自分のコンテナの一覧に、この計画の名前の行が 1 つだけある台本。"""
    line = _own_container_line(plan, container_id=container_id)
    return _own_containers_rule(plan.node, Reply(stdout=line + "\n"))


def _container_log_rule_by_id(
    node: NodeRole, container_id: str, *, stdout: str = "", stderr: str = ""
) -> Rule:
    return Rule(
        node=node,
        kind="run",
        prefix=("docker", "logs", "--timestamps", container_id),
        replies=(Reply(stdout=stdout, stderr=stderr),),
    )


def _tail_rule_by_id(
    node: NodeRole, container_id: str, *, lines: int, stdout: str = "", stderr: str = ""
) -> Rule:
    return Rule(
        node=node,
        kind="run",
        prefix=("docker", "logs", "--timestamps", "--tail", str(lines), container_id),
        replies=(Reply(stdout=stdout, stderr=stderr),),
    )


def _comm_log_rule(plan: ContainerPlan, *, reply: Reply) -> Rule:
    return Rule(
        node=plan.node, kind="pull", remote_prefix=lg.COMM_LOG_REMOTE_SUBDIR, replies=(reply,)
    )


def _launch_record_rule(plan: ContainerPlan, *, reply: Reply) -> Rule:
    return Rule(node=plan.node, kind="pull", remote_prefix=LAUNCH_RECORD_PATH, replies=(reply,))


def _full_script(plan: ContainerPlan, *, node_out: str, container_id: str) -> tuple[Rule, ...]:
    """1 台ぶんの、3 つとも写せる台本 (自分のコンテナの一覧に見つかる場合)。"""
    return (
        _found_rule(plan, container_id),
        _container_log_rule_by_id(
            plan.node, container_id, stdout=f"{node_out} out\n", stderr=f"{node_out} err\n"
        ),
        _comm_log_rule(
            plan, reply=Reply(writes=((f"logs/nccl.{node_out}.log", f"{node_out} nccl\n"),))
        ),
        _launch_record_rule(
            plan,
            reply=Reply(writes=((f"{CONFIG_NAME}.launch.json", f'{{"node": "{node_out}"}}'),)),
        ),
    )


def _calls_targeting_docker_logs(runner: FakeRunner) -> Sequence[tuple[str, ...]]:
    return tuple(call.argv for call in runner.runs if call.argv[:2] == ("docker", "logs"))


# --- var_dir (入出力のない関数) -------------------------------------------


def test_var_dir_builds_the_designed_name(tmp_path: Path) -> None:
    destination = lg.var_dir(tmp_path, STARTED_AT, "logs", CONFIG_NAME)
    assert destination == tmp_path / "20260921T030000Z-logs-p1-nvfp4-tp2"


def test_var_dir_refuses_a_command_name_with_a_slash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="コマンドの名前"):
        lg.var_dir(tmp_path, STARTED_AT, "lo/gs", CONFIG_NAME)


def test_var_dir_refuses_a_config_name_that_escapes_with_dot_dot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="構成の名前"):
        lg.var_dir(tmp_path, STARTED_AT, "logs", "../escape")


def test_var_dir_refuses_an_empty_config_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="構成の名前"):
        lg.var_dir(tmp_path, STARTED_AT, "logs", "")


def test_var_dir_refuses_a_naive_datetime(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="started_at"):
        lg.var_dir(tmp_path, datetime(2026, 9, 21, 3, 0, 0), "logs", CONFIG_NAME)


def test_launch_record_remote_path_is_relative_to_state() -> None:
    assert lg.launch_record_remote_path(CONFIG_NAME) == f"state/{CONFIG_NAME}.launch.json"


# --- collect_logs: 写せる場合 ----------------------------------------------


def test_collect_logs_writes_container_comm_and_launch_records_for_both_nodes(
    tmp_path: Path,
) -> None:
    script = _full_script(
        HEAD_PLAN, node_out="head", container_id=HEAD_CONTAINER_ID
    ) + _full_script(WORKER_PLAN, node_out="worker", container_id=WORKER_CONTAINER_ID)
    runner = FakeRunner(var_root=tmp_path, script=script)

    destination = lg.collect_logs(
        runner,
        NODES,
        PLANS,
        var_root=tmp_path,
        started_at=STARTED_AT,
        command="logs",
        config_name=CONFIG_NAME,
    )

    assert destination == tmp_path / "20260921T030000Z-logs-p1-nvfp4-tp2"
    assert (destination / "head" / "container.stdout.log").read_text() == "head out\n"
    assert (destination / "head" / "container.stderr.log").read_text() == "head err\n"
    assert (destination / "head" / "logs" / "nccl.head.log").read_text() == "head nccl\n"
    assert (destination / "head" / f"{CONFIG_NAME}.launch.json").read_text() == '{"node": "head"}'
    assert (destination / "worker" / "container.stdout.log").read_text() == "worker out\n"
    assert (destination / "worker" / "container.stderr.log").read_text() == "worker err\n"
    assert (destination / "worker" / "logs" / "nccl.worker.log").read_text() == "worker nccl\n"
    assert (
        destination / "worker" / f"{CONFIG_NAME}.launch.json"
    ).read_text() == '{"node": "worker"}'

    missing = json.loads((destination / lg.MISSING_FILE_NAME).read_text())
    assert missing["missing"] == []
    assert missing["command"] == "logs"
    assert missing["config_name"] == CONFIG_NAME

    # (i) 不変条件 (requirements 2.3、2.4): docker logs の対象は、docker ps (ラベルの絞り込み
    # つき) の台本が返した ID だけ。名前に向けた docker logs は 1 つも無い
    names = {plan.container_name for plan in PLANS}
    ids = {HEAD_CONTAINER_ID, WORKER_CONTAINER_ID}
    logs_calls = _calls_targeting_docker_logs(runner)
    assert logs_calls, "docker logs が 1 つも呼ばれていない"
    for argv in logs_calls:
        assert argv[-1] in ids
        assert argv[-1] not in names
    # (iv) docker ps は、つねにラベルの絞り込みつき
    ps_calls = [call for call in runner.runs if call.argv[:2] == ("docker", "ps")]
    assert ps_calls, "docker ps が 1 つも呼ばれていない"
    for call in ps_calls:
        assert f"label={OWNER_FILTER}" in call.argv
    # 回収は、状態を変える呼び出しを 1 つも出さない
    assert runner.calls
    assert all(call.mutating is False for call in runner.calls)


def test_collect_logs_creates_the_destination_even_without_plans(tmp_path: Path) -> None:
    runner = FakeRunner(var_root=tmp_path, script=())
    destination = lg.collect_logs(
        runner,
        NODES,
        (),
        var_root=tmp_path,
        started_at=STARTED_AT,
        command="logs",
        config_name=CONFIG_NAME,
    )
    assert destination.is_dir()
    assert json.loads((destination / lg.MISSING_FILE_NAME).read_text())["missing"] == []
    assert runner.calls == ()


# --- collect_logs: 台の失敗 -------------------------------------------------


def test_collect_logs_keeps_the_other_node_when_one_node_is_unreachable(tmp_path: Path) -> None:
    unreachable = Reply(exit_code=255, stderr="ssh: connect to host spark-5083: timed out\n")
    # worker 向けの読み取りは、すべて「入れない」で失敗する台本にする
    # (docker ps 自体が届かないので、docker logs の対象を決められない)
    script = (
        *_full_script(HEAD_PLAN, node_out="head", container_id=HEAD_CONTAINER_ID),
        _own_containers_rule("worker", unreachable),
        _comm_log_rule(WORKER_PLAN, reply=unreachable),
        _launch_record_rule(WORKER_PLAN, reply=unreachable),
    )
    runner = FakeRunner(var_root=tmp_path, script=script)

    destination = lg.collect_logs(
        runner,
        NODES,
        PLANS,
        var_root=tmp_path,
        started_at=STARTED_AT,
        command="logs",
        config_name=CONFIG_NAME,
    )

    assert (destination / "head" / "container.stdout.log").read_text() == "head out\n"
    assert not (destination / "worker" / "container.stdout.log").exists()
    assert not (destination / "worker" / "logs").exists()
    assert not (destination / "worker" / f"{CONFIG_NAME}.launch.json").exists()
    # worker 向けの docker logs は 1 つも出ない (対象を決められないため)
    logs_calls = _calls_targeting_docker_logs(runner)
    assert all(argv[3] != WORKER_PLAN.container_name for argv in logs_calls)

    missing = json.loads((destination / lg.MISSING_FILE_NAME).read_text())["missing"]
    items = {(entry["node"], entry["item"]) for entry in missing}
    assert items == {
        ("worker", lg.ITEM_CONTAINER_LOG),
        ("worker", lg.ITEM_COMM_LOG),
        ("worker", lg.ITEM_LAUNCH_RECORD),
    }
    for entry in missing:
        assert "読み取りが届かなかった" in entry["detail"]


def test_collect_logs_records_a_missing_comm_log_when_it_is_not_there_yet(tmp_path: Path) -> None:
    not_there_yet = Reply(
        exit_code=23,
        stderr='rsync: change_dir "logs" failed: No such file or directory (2)\n',
    )

    def rules_for(plan: ContainerPlan, container_id: str) -> tuple[Rule, ...]:
        return (
            _found_rule(plan, container_id),
            _container_log_rule_by_id(plan.node, container_id, stdout=f"{plan.node} out\n"),
            _comm_log_rule(plan, reply=not_there_yet),
            _launch_record_rule(plan, reply=Reply(writes=((f"{CONFIG_NAME}.launch.json", "{}"),))),
        )

    script = rules_for(HEAD_PLAN, HEAD_CONTAINER_ID) + rules_for(WORKER_PLAN, WORKER_CONTAINER_ID)
    runner = FakeRunner(var_root=tmp_path, script=script)

    destination = lg.collect_logs(
        runner,
        NODES,
        PLANS,
        var_root=tmp_path,
        started_at=STARTED_AT,
        command="logs",
        config_name=CONFIG_NAME,
    )

    missing = json.loads((destination / lg.MISSING_FILE_NAME).read_text())["missing"]
    items = {(entry["node"], entry["item"]) for entry in missing}
    assert items == {("head", lg.ITEM_COMM_LOG), ("worker", lg.ITEM_COMM_LOG)}
    # 写せたものは、そのまま残る
    assert (destination / "head" / "container.stdout.log").read_text() == "head out\n"
    assert (destination / "head" / f"{CONFIG_NAME}.launch.json").read_text() == "{}"
    assert not (destination / "head" / "logs").exists()


def test_collect_logs_does_not_read_a_container_that_is_not_in_the_owned_list(
    tmp_path: Path,
) -> None:
    """(ii) 一覧に名前が一致する行が無い台では、docker logs を 1 つも出さない。

    docker の `--filter` が、ラベルの無いコンテナを一覧から外すので、同じ名前のラベルの無い
    コンテナが実際にあっても、一覧には映らない (ここでは、その状態を、空の一覧として表す)。
    """
    script = (
        _own_containers_rule("head", Reply(stdout="")),
        _comm_log_rule(HEAD_PLAN, reply=Reply(writes=(("logs/nccl.head.log", "head nccl\n"),))),
        _launch_record_rule(HEAD_PLAN, reply=Reply(writes=((f"{CONFIG_NAME}.launch.json", "{}"),))),
    )
    runner = FakeRunner(var_root=tmp_path, script=script)

    destination = lg.collect_logs(
        runner,
        NODES,
        (HEAD_PLAN,),
        var_root=tmp_path,
        started_at=STARTED_AT,
        command="logs",
        config_name=CONFIG_NAME,
    )

    assert _calls_targeting_docker_logs(runner) == ()
    assert not (destination / "head" / "container.stdout.log").exists()
    missing = json.loads((destination / lg.MISSING_FILE_NAME).read_text())["missing"]
    items = {(entry["node"], entry["item"]) for entry in missing}
    assert ("head", lg.ITEM_CONTAINER_LOG) in items
    # 通信の記録と起動の記録は、一覧と関係なく写せる
    assert (destination / "head" / "logs" / "nccl.head.log").read_text() == "head nccl\n"


def test_collect_logs_does_not_read_when_the_owned_list_has_a_foreign_row(tmp_path: Path) -> None:
    """(iii) 一覧に、所有のラベルの無い行が紛れている台では、docker logs を 1 つも出さない
    (`guards.list_own_containers` が、その一覧そのものを断るため)。
    """
    foreign_line = json.dumps(
        {
            "ID": "deadbeef0000",
            "Names": "some-other-container",
            "State": "running",
            "Image": "nginx:latest",
            "Labels": "",
        }
    )
    own_line = _own_container_line(HEAD_PLAN, container_id=HEAD_CONTAINER_ID)
    script = (
        _own_containers_rule("head", Reply(stdout=f"{own_line}\n{foreign_line}\n")),
        _comm_log_rule(HEAD_PLAN, reply=Reply(writes=(("logs/nccl.head.log", "head nccl\n"),))),
        _launch_record_rule(HEAD_PLAN, reply=Reply(writes=((f"{CONFIG_NAME}.launch.json", "{}"),))),
    )
    runner = FakeRunner(var_root=tmp_path, script=script)

    destination = lg.collect_logs(
        runner,
        NODES,
        (HEAD_PLAN,),
        var_root=tmp_path,
        started_at=STARTED_AT,
        command="logs",
        config_name=CONFIG_NAME,
    )

    assert _calls_targeting_docker_logs(runner) == ()
    missing = json.loads((destination / lg.MISSING_FILE_NAME).read_text())["missing"]
    items = {(entry["node"], entry["item"]) for entry in missing}
    assert ("head", lg.ITEM_CONTAINER_LOG) in items
    # 誤りの文に、よそのコンテナの名前や ID が出ない (guards.list_own_containers の決まり)
    detail = next(e["detail"] for e in missing if e["item"] == lg.ITEM_CONTAINER_LOG)
    assert "deadbeef0000" not in detail
    assert "some-other-container" not in detail


def test_collect_logs_refuses_an_unknown_node(tmp_path: Path) -> None:
    runner = FakeRunner(var_root=tmp_path, script=())
    stray = ContainerPlan(
        node="worker",
        container_name="vb-other-worker",
        labels={},
        argv=("docker", "run", "-d", "--name", "vb-other-worker"),
    )
    with pytest.raises(ValueError, match="ノードの定義がない"):
        lg.collect_logs(
            runner,
            {"head": HEAD},
            (stray,),
            var_root=tmp_path,
            started_at=STARTED_AT,
            command="logs",
            config_name=CONFIG_NAME,
        )


# --- tail_logs -------------------------------------------------------------


def test_tail_logs_returns_both_nodes_with_the_default_tail_count(tmp_path: Path) -> None:
    script = (
        _found_rule(HEAD_PLAN, HEAD_CONTAINER_ID),
        _tail_rule_by_id(
            "head",
            HEAD_CONTAINER_ID,
            lines=lg.DEFAULT_TAIL_LINES,
            stdout="head tail out\n",
            stderr="head tail err\n",
        ),
        _found_rule(WORKER_PLAN, WORKER_CONTAINER_ID),
        _tail_rule_by_id(
            "worker", WORKER_CONTAINER_ID, lines=lg.DEFAULT_TAIL_LINES, stdout="worker tail out\n"
        ),
    )
    runner = FakeRunner(var_root=tmp_path, script=script)

    tails = lg.tail_logs(runner, NODES, PLANS)

    assert tails == {
        "head": "head tail out\nhead tail err\n",
        "worker": "worker tail out\n",
    }
    # 末尾は、どのファイルにも書かない
    assert runner.pulls == ()
    # 名前ではなく ID に向いている
    for argv in _calls_targeting_docker_logs(runner):
        assert argv[-1] in {HEAD_CONTAINER_ID, WORKER_CONTAINER_ID}
        assert argv[-1] not in {HEAD_PLAN.container_name, WORKER_PLAN.container_name}


def test_tail_logs_uses_the_given_line_count(tmp_path: Path) -> None:
    script = (
        _found_rule(HEAD_PLAN, HEAD_CONTAINER_ID),
        _tail_rule_by_id("head", HEAD_CONTAINER_ID, lines=20, stdout="20 lines\n"),
    )
    runner = FakeRunner(var_root=tmp_path, script=script)

    tails = lg.tail_logs(runner, NODES, (HEAD_PLAN,), lines=20)

    assert tails == {"head": "20 lines\n"}


def test_tail_logs_reports_an_unreachable_node_without_raising(tmp_path: Path) -> None:
    unreachable = Reply(exit_code=255, stderr="ssh: connect to host spark-5083: timed out\n")
    script = (
        _found_rule(HEAD_PLAN, HEAD_CONTAINER_ID),
        _tail_rule_by_id(
            "head", HEAD_CONTAINER_ID, lines=lg.DEFAULT_TAIL_LINES, stdout="head ok\n"
        ),
        _own_containers_rule("worker", unreachable),
    )
    runner = FakeRunner(var_root=tmp_path, script=script)

    tails = lg.tail_logs(runner, NODES, PLANS)

    assert tails["head"] == "head ok\n"
    assert "読み取りが届かなかった" in tails["worker"]


def test_tail_logs_reports_a_container_that_is_not_in_the_owned_list(tmp_path: Path) -> None:
    script = (_own_containers_rule("head", Reply(stdout="")),)
    runner = FakeRunner(var_root=tmp_path, script=script)

    tails = lg.tail_logs(runner, NODES, (HEAD_PLAN,))

    assert _calls_targeting_docker_logs(runner) == ()
    assert "自分のコンテナがない" in tails["head"]
