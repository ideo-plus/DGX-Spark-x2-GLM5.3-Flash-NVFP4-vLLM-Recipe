"""遠隔の実行の部品の試験 (task 1.4)。

確かめること (tasks.md 1.4 の完了の状態、design.md 「remote」):

- 空白と引用符を含む引数が、`shlex.join` で 1 つの引数のまま遠隔に渡る (2.3)
- 許可の一覧にない `argv[0]` と、許可の一覧にない docker のサブコマンドが断られる
  (requirements 2.7、8.6)。docker の大域のオプションで、検査をすり抜けられない
- 了承した計画にない、状態を変える呼び出しが `RuntimeError` になる (requirements 2.1)
- 了承した計画の巻き戻しのコマンドは、前に進むコマンドが途中で失敗したあとでも、
  断られずに実行できる (requirements 1.5)
- `push` の宛先は `payload/` と `state/` だけ、`pull` の宛先は Mac の記録の置き場所の
  下だけ (requirements 1.1、1.2、8.6)
- rsync の遠隔の道筋 (`push` の宛先、`pull` の元、`remote_root`、`ssh_host`) は、遠隔の
  シェルに素で渡るので、使える文字を絞る。Mac の側の道筋は絞らず、1 つの引数のまま渡る
- `ip` と `ethtool` は読み取りの形だけ (requirements 2.7)、`cat` は `remote_root` の下と
  インターフェースの読み取りの場所だけ (requirements 2.4、2.6)
- 時間切れと接続の失敗は `RemoteError`、遠隔の 0 以外の終了は `CommandResult`
- `remote` は `types` だけを読み込む (依存の向き)

実物の ssh と rsync は、どの試験でも呼ばない。`subprocess.run` を差し替えて、渡った
引数のリストを検査する。
"""

from __future__ import annotations

import ast
import inspect
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any

import pytest

from serving_kit import remote
from serving_kit.remote import RemoteError, SshRunner
from serving_kit.types import ApprovedPlan, NodeDef, PlannedPush, PlannedRun

# --- 見本のノード -------------------------------------------------------

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
RM_ARGV = ("docker", "rm", CONTAINER)

# 空白、引用符、シェルの記号を含む道筋 (引用の事故が起きれば、戻したときに壊れる)
TRICKY = "a b'c\"d;rm -rf / $(whoami) \\t"

ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "collections",
    "dataclasses",
    "os",
    "pathlib",
    "re",
    "shlex",
    "subprocess",
    "time",
    "typing",
    "serving_kit",
}


def approved(*, forward: Sequence[Any], rollback: Sequence[PlannedRun] = ()) -> ApprovedPlan:
    """試験用の、了承を得た計画を作る。"""
    return ApprovedPlan(
        forward=tuple(forward),
        rollback=tuple(rollback),
        own_container_names=(CONTAINER,),
    )


# --- subprocess.run の差し替え ------------------------------------------


@dataclass
class SentCall:
    """`subprocess.run` に渡ったもの。"""

    argv: list[str]
    kwargs: dict[str, Any]


@dataclass
class Recorder:
    """`subprocess.run` の代わり。渡った引数のリストを覚え、台本どおりに返す。

    `outcomes` は、1 回ごとの結末で、整数なら終了コード、例外ならそれを投げる。
    尽きたら、最後の結末を繰り返す。
    """

    outcomes: list[int | BaseException] = field(default_factory=lambda: [0])
    stdout: str = ""
    stderr: str = ""
    calls: list[SentCall] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(SentCall(argv=list(argv), kwargs=dict(kwargs)))
        index = min(len(self.calls) - 1, len(self.outcomes) - 1)
        outcome = self.outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return subprocess.CompletedProcess(
            args=list(argv), returncode=outcome, stdout=self.stdout, stderr=self.stderr
        )

    @property
    def last(self) -> list[str]:
        assert self.calls, "subprocess.run が 1 度も呼ばれていない"
        return self.calls[-1].argv


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """`subprocess.run` を差し替える (実物の ssh と rsync を呼ばない)。"""
    recorder = Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    return recorder


@pytest.fixture
def var_root(tmp_path: Path) -> Path:
    """Mac の記録の置き場所 (`serving/var/`) の代わり。"""
    root = tmp_path / "var"
    root.mkdir()
    return root


@pytest.fixture
def runner(var_root: Path) -> SshRunner:
    """了承を得た計画を持たない実行役 (読み取りだけができる)。"""
    return SshRunner(var_root=var_root)


# --- 引用の事故 ---------------------------------------------------------


def test_the_remote_command_is_one_shell_string_built_with_shlex(
    runner: SshRunner, sent: Recorder
) -> None:
    """空白と引用符を含む引数が、`shlex.split` で戻すと 1 つの引数のままである。"""
    argv = ["cat", f"{REMOTE_ROOT}/logs/{TRICKY}.log"]
    runner.run(HEAD, argv, timeout_s=10.0, mutating=False)

    assert shlex.split(sent.last[-1]) == argv


def test_the_ssh_call_is_a_list_with_the_non_interactive_options(
    runner: SshRunner, sent: Recorder
) -> None:
    """ssh は引数のリストで呼び、非対話の指定と接続の時間切れを付ける。"""
    runner.run(HEAD, ["uname", "-n"], timeout_s=10.0, mutating=False)

    assert sent.last == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "--",
        "spark-153d",
        "uname -n",
    ]
    assert sent.calls[-1].kwargs.get("shell") in (None, False), "shell=True を使っている"


def test_the_destination_is_the_ssh_host_of_the_node(runner: SshRunner, sent: Recorder) -> None:
    """ssh の宛先は `NodeDef.ssh_host` である (LAN のアドレスではない)。"""
    runner.run(WORKER, ["uname", "-n"], timeout_s=10.0, mutating=False)

    assert "spark-5083" in sent.last
    assert "10.0.1.61" not in sent.last


def test_the_result_carries_the_remote_argv_and_the_output(
    runner: SshRunner, sent: Recorder
) -> None:
    """結果は、遠隔で流した引数の列と、出力と、所要を持つ。"""
    sent.stdout = "spark-153d\n"
    result = runner.run(HEAD, ["uname", "-n"], timeout_s=10.0, mutating=False)

    assert result.argv == ("uname", "-n")
    assert result.exit_code == 0
    assert result.stdout == "spark-153d\n"
    assert result.duration_s >= 0.0


# --- 許可の一覧 (argv[0]) ------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["docker", "version"],
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory"],
        ["df", "-B1", "--output=avail", REMOTE_ROOT],
        ["sha256sum", f"{REMOTE_ROOT}/models/a.safetensors"],
        ["ss", "-ltnH"],
        ["ip", "-br", "addr"],
        ["ethtool", "enp1s0f0np0"],
        ["ibv_devinfo"],
        ["ibdev2netdev"],
        ["cat", f"{REMOTE_ROOT}/logs/nccl.log"],
        ["test", "-d", f"{REMOTE_ROOT}/models"],
        ["uname", "-n"],
    ],
)
def test_the_allowed_commands_run(runner: SshRunner, sent: Recorder, argv: list[str]) -> None:
    """design.md 「remote」の `argv[0]` の許可の一覧にあるものは、13 個すべて流せる。

    `mkdir` だけは状態を変えるので、了承の試験の側で確かめる。
    """
    runner.run(HEAD, argv, timeout_s=10.0, mutating=False)

    assert sent.calls, f"{argv} が流れていない"


@pytest.mark.parametrize("command", ["sudo", "apt", "apt-get", "pip", "systemctl", "rm", "sh"])
def test_a_command_outside_the_allowlist_is_refused(
    runner: SshRunner, sent: Recorder, command: str
) -> None:
    """一覧にない `argv[0]` は、流さずに断る (requirements 2.7)。"""
    with pytest.raises(RuntimeError, match="許可の一覧"):
        runner.run(HEAD, [command, "-x"], timeout_s=10.0, mutating=False)

    assert sent.calls == [], "断ったのに ssh が呼ばれている"


@pytest.mark.parametrize("command", ["/usr/bin/sudo", "/usr/bin/docker", "./docker", "bin/docker"])
def test_a_command_with_a_path_is_refused(runner: SshRunner, sent: Recorder, command: str) -> None:
    """道筋つきの `argv[0]` は断る (一覧の名前と突き合わせられない)。"""
    with pytest.raises(RuntimeError):
        runner.run(HEAD, [command, "ps"], timeout_s=10.0, mutating=False)

    assert sent.calls == []


def test_an_empty_argv_is_refused(runner: SshRunner, sent: Recorder) -> None:
    """引数の列が空の呼び出しは断る。"""
    with pytest.raises(RuntimeError):
        runner.run(HEAD, [], timeout_s=10.0, mutating=False)

    assert sent.calls == []


# --- 許可の一覧 (docker のサブコマンド) ---------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["docker", "ps", "-a", "--filter", "label=vllm-baseline.owner=serving-kit"],
        ["docker", "logs", "--timestamps", CONTAINER],
        ["docker", "version", "--format", "{{json .}}"],
        ["docker", "image", "inspect", IMAGE],
        ["docker", "container", "inspect", CONTAINER],
    ],
)
def test_the_allowed_docker_subcommands_run(
    runner: SshRunner, sent: Recorder, argv: list[str]
) -> None:
    """読み取りの docker のサブコマンドは、了承の計画なしで流せる。"""
    runner.run(HEAD, argv, timeout_s=10.0, mutating=False)

    assert sent.calls, f"{argv} が流れていない"


@pytest.mark.parametrize(
    "argv",
    [
        ["docker", "exec", CONTAINER, "sh"],
        ["docker", "build", "."],
        ["docker", "rmi", IMAGE],
        ["docker", "system", "prune", "-f"],
        ["docker", "volume", "ls"],
        ["docker", "network", "ls"],
        ["docker", "cp", f"{CONTAINER}:/etc/passwd", "."],
        ["docker", "commit", CONTAINER],
        ["docker", "image", "ls"],
        ["docker", "container", "ls"],
        ["docker", "image"],
        ["docker"],
    ],
)
def test_a_docker_subcommand_outside_the_allowlist_is_refused(
    runner: SshRunner, sent: Recorder, argv: list[str]
) -> None:
    """一覧にない docker のサブコマンドは断る (requirements 2.3、8.6)。"""
    with pytest.raises(RuntimeError, match="docker"):
        runner.run(HEAD, argv, timeout_s=10.0, mutating=False)

    assert sent.calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ["docker", "-H", "tcp://10.0.1.61:2375", "run", "-d", IMAGE],
        ["docker", "--context", "worker", "run", "-d", IMAGE],
        ["docker", "--host", "tcp://10.0.1.61:2375", "ps"],
        ["docker", "-D", "exec", CONTAINER, "sh"],
    ],
)
def test_docker_global_options_cannot_slip_past_the_subcommand_check(
    runner: SshRunner, sent: Recorder, argv: list[str]
) -> None:
    """`docker` の大域のオプションで、サブコマンドの検査をすり抜けられない。

    大域のオプションだと名指しして断ることまでを固定する (この検査を外すと、
    「許可の一覧にないサブコマンド」という別の文で断られるので、ここが落ちる)。
    """
    with pytest.raises(RuntimeError, match="大域のオプション"):
        runner.run(HEAD, argv, timeout_s=10.0, mutating=False)

    assert sent.calls == []


# --- 読み取りの形だけに絞るコマンド (ip、ethtool、cat) -------------------

IF = "enp1s0f0np0"


@pytest.mark.parametrize(
    "argv",
    [
        ["ip", "-br", "link"],
        ["ip", "-br", "addr"],
        ["ip", "link"],
        ["ip", "addr", "show"],
        ["ip", "address", "show"],
        ["ip", "-br", "link", "show", IF],
        ["ip", "-j", "addr", "show", "dev", IF],
        ["ip", "-json", "-d", "link"],
        ["ip", "-details", "-s", "link"],
        ["ip", "-4", "addr"],
        ["ip", "-6", "addr"],
        ["ip", "-o", "addr", "show"],
        ["ip", "-brief", "link"],
    ],
)
def test_the_read_only_forms_of_ip_run(runner: SshRunner, sent: Recorder, argv: list[str]) -> None:
    """design が使う `ip` の形 (リンクとアドレスの読み取り) は流せる。"""
    runner.run(HEAD, argv, timeout_s=10.0, mutating=False)

    assert sent.calls, f"{argv} が流れていない"


@pytest.mark.parametrize(
    "argv",
    [
        ["ip", "link", "set", "dev", IF, "mtu", "9000"],
        ["ip", "addr", "add", "192.168.100.1/24", "dev", IF],
        ["ip", "addr", "flush", "dev", IF],
        ["ip", "link", "del", IF],
        ["ip", "route", "del", "default"],
        ["ip", "netns", "exec", "x", "sh"],
        ["ip", "-batch", "f"],
        ["ip", "-force", "link"],
        ["ip", "-netns", "x", "link"],
        ["ip", "link", "show", "-x"],
        ["ip"],
    ],
)
def test_ip_outside_the_read_only_forms_is_refused(
    runner: SshRunner, sent: Recorder, argv: list[str]
) -> None:
    """`ip` で、ネットワークの設定を変える形は断る (requirements 2.7)。"""
    with pytest.raises(RuntimeError, match="ip"):
        runner.run(HEAD, argv, timeout_s=10.0, mutating=False)

    assert sent.calls == []


@pytest.mark.parametrize("argv", [["ethtool", IF], ["ethtool", "-i", IF], ["ethtool", "-S", IF]])
def test_the_read_only_forms_of_ethtool_run(
    runner: SshRunner, sent: Recorder, argv: list[str]
) -> None:
    """design が使う `ethtool` の形 (速さ、ドライバ、統計の読み取り) は流せる。"""
    runner.run(HEAD, argv, timeout_s=10.0, mutating=False)

    assert sent.calls, f"{argv} が流れていない"


@pytest.mark.parametrize(
    "argv",
    [
        ["ethtool", "-s", IF, "speed", "100"],
        ["ethtool", "-G", IF, "rx", "1"],
        ["ethtool", "-K", IF, "tso", "off"],
        ["ethtool", "--help"],
        ["ethtool", "-i"],
        ["ethtool", "-i", IF, "extra"],
        ["ethtool"],
    ],
)
def test_ethtool_outside_the_read_only_forms_is_refused(
    runner: SshRunner, sent: Recorder, argv: list[str]
) -> None:
    """`ethtool` で、インターフェースの設定を変える形は断る (requirements 2.7)。"""
    with pytest.raises(RuntimeError, match="ethtool"):
        runner.run(HEAD, argv, timeout_s=10.0, mutating=False)

    assert sent.calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ["cat", f"{REMOTE_ROOT}/logs/nccl.log"],
        ["cat", f"{REMOTE_ROOT}/state/p1.launch.json", f"{REMOTE_ROOT}/logs/a.log"],
        ["cat", f"/sys/class/net/{IF}/mtu"],
        ["cat", "/sys/class/infiniband/rocep1s0f0/ports/1/state"],
    ],
)
def test_cat_reads_only_the_allowed_places(
    runner: SshRunner, sent: Recorder, argv: list[str]
) -> None:
    """`cat` は、Spark の置き場所の下と、インターフェースの読み取りの場所だけを読める。"""
    runner.run(HEAD, argv, timeout_s=10.0, mutating=False)

    assert sent.calls, f"{argv} が流れていない"


@pytest.mark.parametrize(
    "argv",
    [
        ["cat", "/etc/shadow"],
        ["cat", "~/.ssh/id_ed25519"],
        ["cat", "/proc/1/environ"],
        ["cat", f"{REMOTE_ROOT}/../x"],
        ["cat", "/var/lib/docker/containers/x/x-json.log"],
        ["cat", "-n", f"{REMOTE_ROOT}/logs/a.log"],
        ["cat", f"{REMOTE_ROOT}/logs/a.log", "/etc/passwd"],
        ["cat"],
    ],
)
def test_cat_outside_the_allowed_places_is_refused(
    runner: SshRunner, sent: Recorder, argv: list[str]
) -> None:
    """よその場所、認証の情報、別の構成の記録を読む `cat` は断る (requirements 2.4、2.6)。"""
    with pytest.raises(RuntimeError, match="cat"):
        runner.run(HEAD, argv, timeout_s=10.0, mutating=False)

    assert sent.calls == []


# --- 了承を得た計画 -----------------------------------------------------


def test_a_state_changing_call_without_a_plan_is_an_error(
    runner: SshRunner, sent: Recorder
) -> None:
    """了承を得た計画がなければ、状態を変える呼び出しは流さずに誤りにする (2.1)。"""
    with pytest.raises(RuntimeError, match="了承"):
        runner.run(HEAD, list(RUN_ARGV), timeout_s=10.0, mutating=True)

    assert sent.calls == []


@pytest.mark.parametrize(
    "argv",
    [
        list(RUN_ARGV),
        list(STOP_ARGV),
        list(RM_ARGV),
        ["docker", "pull", IMAGE],
        ["mkdir", "-p", f"{REMOTE_ROOT}/payload"],
    ],
)
def test_a_state_changing_call_is_gated_even_when_declared_read_only(
    runner: SshRunner, sent: Recorder, argv: list[str]
) -> None:
    """`mutating=False` と申告されても、状態を変えるコマンドは読み取りとして通さない。"""
    with pytest.raises(RuntimeError, match="了承"):
        runner.run(HEAD, argv, timeout_s=10.0, mutating=False)

    assert sent.calls == []


def test_a_read_only_call_needs_no_plan(runner: SshRunner, sent: Recorder) -> None:
    """読み取りの呼び出しは、了承を得た計画がなくても流せる。"""
    runner.run(HEAD, ["df", "-B1", "--output=avail", REMOTE_ROOT], timeout_s=10.0, mutating=False)

    assert sent.calls


def test_a_planned_forward_command_runs(var_root: Path, sent: Recorder) -> None:
    """計画の、前に進むコマンドは流せる。"""
    plan = approved(forward=[PlannedRun(node="head", argv=RUN_ARGV, container=CONTAINER)])
    runner = SshRunner(var_root=var_root, plan=plan)

    result = runner.run(HEAD, list(RUN_ARGV), timeout_s=10.0, mutating=True)

    assert result.exit_code == 0
    assert shlex.split(sent.last[-1]) == list(RUN_ARGV)


def test_a_planned_mkdir_of_the_remote_directories_runs(var_root: Path, sent: Recorder) -> None:
    """`serve push` が置き場所を作る `mkdir -p` も、計画に入っていれば流せる。"""
    argv = ("mkdir", "-p", f"{REMOTE_ROOT}/payload", f"{REMOTE_ROOT}/models")
    plan = approved(forward=[PlannedRun(node="head", argv=argv, purpose="置き場所を作る")])
    runner = SshRunner(var_root=var_root, plan=plan)

    assert runner.run(HEAD, list(argv), timeout_s=10.0, mutating=True).exit_code == 0
    assert shlex.split(sent.last[-1]) == list(argv)


def test_the_plan_membership_is_an_exact_match_on_node_and_argv(
    var_root: Path, sent: Recorder
) -> None:
    """「含まれている」は、ノードと引数の列の完全な一致で見る (前方一致にしない)。"""
    plan = approved(forward=[PlannedRun(node="head", argv=STOP_ARGV, container=CONTAINER)])
    runner = SshRunner(var_root=var_root, plan=plan)

    with pytest.raises(RuntimeError, match="了承"):
        runner.run(HEAD, [*STOP_ARGV, "--time", "30"], timeout_s=10.0, mutating=True)
    with pytest.raises(RuntimeError, match="了承"):
        runner.run(HEAD, list(STOP_ARGV[:2]), timeout_s=10.0, mutating=True)
    with pytest.raises(RuntimeError, match="了承"):
        runner.run(WORKER, list(STOP_ARGV), timeout_s=10.0, mutating=True)

    assert sent.calls == []


def test_the_rollback_runs_after_the_forward_command_failed(var_root: Path, sent: Recorder) -> None:
    """巻き戻しは、前に進むコマンドが失敗したあとでも、断られずに実行できる (1.5)。"""
    plan = approved(
        forward=[PlannedRun(node="head", argv=RUN_ARGV, container=CONTAINER)],
        rollback=[
            PlannedRun(node="head", argv=STOP_ARGV, container=CONTAINER),
            PlannedRun(node="head", argv=RM_ARGV, container=CONTAINER),
        ],
    )
    runner = SshRunner(var_root=var_root, plan=plan)
    sent.outcomes = [125, 0, 0]

    failed = runner.run(HEAD, list(RUN_ARGV), timeout_s=10.0, mutating=True)
    assert failed.exit_code == 125

    assert runner.run(HEAD, list(STOP_ARGV), timeout_s=10.0, mutating=True).exit_code == 0
    assert runner.run(HEAD, list(RM_ARGV), timeout_s=10.0, mutating=True).exit_code == 0
    assert len(sent.calls) == 3


def test_the_rollback_runs_after_the_forward_command_timed_out(
    var_root: Path, sent: Recorder
) -> None:
    """時間切れで計画が無効にならない (巻き戻しは、そのあとでも実行できる)。"""
    plan = approved(
        forward=[PlannedRun(node="head", argv=RUN_ARGV, container=CONTAINER)],
        rollback=[PlannedRun(node="head", argv=STOP_ARGV, container=CONTAINER)],
    )
    runner = SshRunner(var_root=var_root, plan=plan)
    sent.outcomes = [subprocess.TimeoutExpired(cmd="ssh", timeout=10.0), 0]

    with pytest.raises(RemoteError):
        runner.run(HEAD, list(RUN_ARGV), timeout_s=10.0, mutating=True)

    assert runner.run(HEAD, list(STOP_ARGV), timeout_s=10.0, mutating=True).exit_code == 0


def test_a_plan_can_be_approved_once_after_the_runner_was_built(
    var_root: Path, sent: Recorder
) -> None:
    """了承は、読み取りのあとに来るので、あとから 1 度だけ渡せる。"""
    runner = SshRunner(var_root=var_root)
    plan = approved(forward=[PlannedRun(node="head", argv=RUN_ARGV, container=CONTAINER)])

    runner.approve(plan)
    assert runner.run(HEAD, list(RUN_ARGV), timeout_s=10.0, mutating=True).exit_code == 0

    with pytest.raises(RuntimeError):
        runner.approve(plan)


# --- 配布 (push) --------------------------------------------------------


def planned_push(local_dir: Path, remote_subdir: str, *, delete: bool) -> ApprovedPlan:
    return approved(
        forward=[
            PlannedPush(
                node="head", local_dir=local_dir, remote_subdir=remote_subdir, delete=delete
            )
        ]
    )


@pytest.mark.parametrize("remote_subdir", ["payload", "state", "payload/scripts", "state/probe"])
def test_push_allows_only_the_payload_and_state_directories(
    tmp_path: Path, var_root: Path, sent: Recorder, remote_subdir: str
) -> None:
    """配布の宛先は、`payload/` と `state/` の下だけ。"""
    local = tmp_path / "payload"
    local.mkdir()
    runner = SshRunner(var_root=var_root, plan=planned_push(local, remote_subdir, delete=False))

    runner.push(HEAD, local, remote_subdir, delete=False)

    assert sent.last[0] == "rsync"
    assert sent.last[-1] == f"spark-153d:{REMOTE_ROOT}/{remote_subdir}/"
    assert sent.last[-2] == f"{local}/"
    assert "--delete" not in sent.last
    assert sent.calls[-1].kwargs.get("shell") in (None, False)


def test_push_leaves_python_bytecode_caches_behind(
    tmp_path: Path, var_root: Path, sent: Recorder
) -> None:
    """`__pycache__` は配らない (7.1 の実機で、試験が作った `payload/__pycache__/` まで
    Spark に届いた。Spark で動かすのは `.py` だけで、Mac の Python のバイトコードは要らない)。"""
    local = tmp_path / "payload"
    local.mkdir()
    runner = SshRunner(var_root=var_root, plan=planned_push(local, "payload", delete=False))

    runner.push(HEAD, local, "payload", delete=False)

    assert sent.last[0] == "rsync"
    assert "--exclude=__pycache__" in sent.last
    assert sent.last.index("--exclude=__pycache__") < sent.last.index("-e")


@pytest.mark.parametrize(
    "remote_subdir",
    [
        "models",
        "models/glm",
        "probe",
        "cache",
        "logs",
        "payload/../models",
        "..",
        "/etc",
        f"{REMOTE_ROOT}/payload",
        "",
        ".",
        "./",
        "././",
    ],
)
def test_push_refuses_every_other_destination(
    tmp_path: Path, var_root: Path, sent: Recorder, remote_subdir: str
) -> None:
    """重み、確認用、キャッシュ、記録、`..`、絶対の道筋、`.` は、配布の宛先にできない。"""
    local = tmp_path / "payload"
    local.mkdir()
    runner = SshRunner(var_root=var_root, plan=planned_push(local, "payload", delete=False))

    with pytest.raises(RuntimeError, match="配布"):
        runner.push(HEAD, local, remote_subdir, delete=False)

    assert sent.calls == []


def test_push_passes_delete_when_asked(tmp_path: Path, var_root: Path, sent: Recorder) -> None:
    """`delete=True` のときだけ `--delete` を付ける。"""
    local = tmp_path / "payload"
    local.mkdir()
    runner = SshRunner(var_root=var_root, plan=planned_push(local, "payload", delete=True))

    runner.push(HEAD, local, "payload", delete=True)

    assert "--delete" in sent.last


def test_push_without_a_plan_is_an_error(tmp_path: Path, var_root: Path, sent: Recorder) -> None:
    """配布は、つねに状態を変える呼び出しとして扱う (了承がなければ誤り)。"""
    local = tmp_path / "payload"
    local.mkdir()
    runner = SshRunner(var_root=var_root)

    with pytest.raises(RuntimeError, match="了承"):
        runner.push(HEAD, local, "payload", delete=True)

    assert sent.calls == []


def test_push_membership_is_an_exact_match(tmp_path: Path, var_root: Path, sent: Recorder) -> None:
    """了承した配布と、ノード、元、宛先、`--delete` の有無が、すべて一致するときだけ通る。"""
    local = tmp_path / "payload"
    local.mkdir()
    runner = SshRunner(var_root=var_root, plan=planned_push(local, "payload", delete=True))

    with pytest.raises(RuntimeError, match="了承"):
        runner.push(HEAD, local, "payload", delete=False)
    with pytest.raises(RuntimeError, match="了承"):
        runner.push(WORKER, local, "payload", delete=True)
    with pytest.raises(RuntimeError, match="了承"):
        runner.push(HEAD, local, "state", delete=True)

    assert sent.calls == []


SHELL_TRAPS = [
    "payload/$(id > /tmp/pwned)",
    "payload/a b.log",
    "payload/x;id",
    "payload/`id`",
    "payload/~root",
    "payload/a\nb",
    "payload/'x'",
    'payload/x"y',
    "payload/*",
    "payload/x|y",
    "payload/x&y",
]
"""rsync の遠隔の道筋は、遠隔のシェルを通るので、これらが 1 つでも通ると事故になる。"""


@pytest.mark.parametrize("remote_subdir", SHELL_TRAPS)
def test_push_refuses_a_remote_subdir_with_shell_characters(
    tmp_path: Path, var_root: Path, sent: Recorder, remote_subdir: str
) -> None:
    """配布の宛先は、シェルの記号や空白を含められない (引用の事故を構造でなくす)。"""
    local = tmp_path / "payload"
    local.mkdir()
    runner = SshRunner(var_root=var_root, plan=planned_push(local, remote_subdir, delete=False))

    with pytest.raises(RuntimeError, match="使えない文字"):
        runner.push(HEAD, local, remote_subdir, delete=False)

    assert sent.calls == []


def test_push_keeps_a_mac_path_with_spaces_as_one_argument(
    tmp_path: Path, var_root: Path, sent: Recorder
) -> None:
    """Mac の側の道筋は、遠隔のシェルを通らないので絞らない (1 つの引数のまま渡る)。"""
    local = tmp_path / "a b'c"
    local.mkdir()
    runner = SshRunner(var_root=var_root, plan=planned_push(local, "payload", delete=False))

    runner.push(HEAD, local, "payload", delete=False)

    assert sent.last.count(f"{local}/") == 1
    assert sent.last[-2] == f"{local}/"


def test_push_uses_the_non_interactive_ssh(tmp_path: Path, var_root: Path, sent: Recorder) -> None:
    """rsync が使う ssh にも、非対話の指定と接続の時間切れを付ける。"""
    local = tmp_path / "payload"
    local.mkdir()
    runner = SshRunner(var_root=var_root, plan=planned_push(local, "payload", delete=False))

    runner.push(HEAD, local, "payload", delete=False)

    assert "-e" in sent.last
    transport = sent.last[sent.last.index("-e") + 1]
    assert "BatchMode=yes" in transport
    assert "ConnectTimeout=5" in transport


# --- 回収 (pull) --------------------------------------------------------


def test_pull_writes_under_the_mac_record_root(var_root: Path, sent: Recorder) -> None:
    """回収の宛先は、Mac の記録の置き場所の下だけ。"""
    runner = SshRunner(var_root=var_root)
    into = var_root / "logs" / "2026-09-21-start"

    result = runner.pull(HEAD, "logs/nccl.log", into)

    assert result.exit_code == 0
    assert sent.last[0] == "rsync"
    assert sent.last[-2] == f"spark-153d:{REMOTE_ROOT}/logs/nccl.log"
    assert sent.last[-1] == f"{into.resolve()}/"


def test_pull_accepts_an_absolute_source_under_the_remote_root(
    var_root: Path, sent: Recorder
) -> None:
    """元を絶対の道筋で渡しても、`remote_root` の下なら通る。"""
    runner = SshRunner(var_root=var_root)

    runner.pull(HEAD, f"{REMOTE_ROOT}/state/p1.launch.json", var_root / "state")

    assert sent.last[-2] == f"spark-153d:{REMOTE_ROOT}/state/p1.launch.json"


@pytest.mark.parametrize(
    "remote_path",
    ["/etc/passwd", "../../etc/passwd", "logs/../../../etc/passwd", "", "/home/j5ik2o"],
)
def test_pull_refuses_a_source_outside_the_remote_root(
    var_root: Path, sent: Recorder, remote_path: str
) -> None:
    """回収の元は、Spark の置き場所の下だけ。"""
    runner = SshRunner(var_root=var_root)

    with pytest.raises(RuntimeError, match="回収"):
        runner.pull(HEAD, remote_path, var_root / "logs")

    assert sent.calls == []


def test_pull_refuses_a_destination_outside_the_mac_record_root(
    tmp_path: Path, var_root: Path, sent: Recorder
) -> None:
    """記録の置き場所の外に書く回収は断る (`..` での遡りも断る)。"""
    runner = SshRunner(var_root=var_root)

    with pytest.raises(RuntimeError, match="回収"):
        runner.pull(HEAD, "logs/nccl.log", tmp_path / "elsewhere")
    with pytest.raises(RuntimeError, match="回収"):
        runner.pull(HEAD, "logs/nccl.log", var_root / ".." / "elsewhere")

    assert sent.calls == []


def test_pull_needs_no_plan(var_root: Path, sent: Recorder) -> None:
    """回収は Spark の状態を変えないので、了承を得た計画は要らない。"""
    runner = SshRunner(var_root=var_root)

    runner.pull(HEAD, "logs/nccl.log", var_root / "logs")

    assert sent.calls


@pytest.mark.parametrize(
    "remote_path",
    [
        "logs/$(id > /tmp/pwned)",
        "logs/a b.log",
        "logs/x;id",
        "logs/`id`",
        "logs/a\nb",
        "logs/'x'",
        'logs/x"y',
        "logs/*",
        "logs/x|y",
    ],
)
def test_pull_refuses_a_source_with_shell_characters(
    var_root: Path, sent: Recorder, remote_path: str
) -> None:
    """回収の元も、シェルの記号や空白を含められない (rsync の遠隔の道筋はシェルを通る)。"""
    runner = SshRunner(var_root=var_root)

    with pytest.raises(RuntimeError, match="使えない文字"):
        runner.pull(HEAD, remote_path, var_root / "logs")

    assert sent.calls == []


@pytest.mark.parametrize("remote_path", ["~", "~/logs", "~root/logs", "logs/~/x"])
def test_pull_refuses_a_home_relative_source(
    var_root: Path, sent: Recorder, remote_path: str
) -> None:
    """`~` は、遠隔のシェルが展開してしまうので断る。"""
    runner = SshRunner(var_root=var_root)

    with pytest.raises(RuntimeError, match="使えない文字"):
        runner.pull(HEAD, remote_path, var_root / "logs")

    assert sent.calls == []


def test_pull_keeps_a_mac_destination_with_spaces_as_one_argument(
    var_root: Path, sent: Recorder
) -> None:
    """回収の宛先 (Mac の側) は、空白を含んでも 1 つの引数のまま渡る。"""
    runner = SshRunner(var_root=var_root)
    into = var_root / "a b'c"

    runner.pull(HEAD, "logs/nccl.log", into)

    assert sent.last[-1] == f"{into.resolve()}/"
    assert into.is_dir(), "回収の宛先が作られていない"


# --- ノードの値の絞り込み -----------------------------------------------


def unsafe_node(*, ssh_host: str = "spark-153d", remote_root: str = REMOTE_ROOT) -> NodeDef:
    return NodeDef(
        role="head",
        ssh_host=ssh_host,
        lan_addr=IPv4Address("10.0.1.60"),
        remote_root=remote_root,
    )


@pytest.mark.parametrize(
    "node",
    [
        unsafe_node(ssh_host="spark-153d;id"),
        unsafe_node(ssh_host="spark 153d"),
        unsafe_node(ssh_host="$(id)"),
        unsafe_node(ssh_host="-oProxyCommand=id"),
        unsafe_node(ssh_host="j5ik2o@spark-153d"),
        unsafe_node(remote_root="/home/j5ik2o/vllm baseline"),
        unsafe_node(remote_root="/home/j5ik2o/$(id)"),
        unsafe_node(remote_root="/home/j5ik2o/vllm;id"),
    ],
)
def test_a_node_with_an_unsafe_ssh_host_or_remote_root_is_refused(
    var_root: Path, sent: Recorder, tmp_path: Path, node: NodeDef
) -> None:
    """ノードの `ssh_host` と `remote_root` も、使える文字を絞る (どの口でも断る)。"""
    local = tmp_path / "payload"
    local.mkdir()
    runner = SshRunner(var_root=var_root, plan=planned_push(local, "payload", delete=False))

    with pytest.raises(RuntimeError, match="使えない文字"):
        runner.run(node, ["uname", "-n"], timeout_s=10.0, mutating=False)
    with pytest.raises(RuntimeError, match="使えない文字"):
        runner.push(node, local, "payload", delete=False)
    with pytest.raises(RuntimeError, match="使えない文字"):
        runner.pull(node, "logs/nccl.log", var_root / "logs")

    assert sent.calls == []


# --- 誤りの扱い ---------------------------------------------------------


def test_a_timeout_becomes_a_remote_error(runner: SshRunner, sent: Recorder) -> None:
    """時間切れは `RemoteError` (どのノードの、どのコマンドかを持つ)。"""
    sent.outcomes = [subprocess.TimeoutExpired(cmd="ssh", timeout=3.0)]

    with pytest.raises(RemoteError) as caught:
        runner.run(HEAD, ["uname", "-n"], timeout_s=3.0, mutating=False)

    assert caught.value.node == "head"
    assert caught.value.ssh_host == "spark-153d"
    assert caught.value.argv == ("uname", "-n")


def test_a_connection_failure_becomes_a_remote_error(runner: SshRunner, sent: Recorder) -> None:
    """ssh の終了コード 255 (接続の失敗) は `RemoteError`。"""
    sent.outcomes = [255]
    sent.stderr = "ssh: connect to host spark-153d port 22: Operation timed out"

    with pytest.raises(RemoteError) as caught:
        runner.run(HEAD, ["uname", "-n"], timeout_s=3.0, mutating=False)

    assert caught.value.node == "head"
    assert "spark-153d" in str(caught.value)


def test_a_non_zero_remote_exit_is_not_an_error(runner: SshRunner, sent: Recorder) -> None:
    """遠隔のコマンドが 0 以外で終わったことは、誤りではなく終了コードで返す。"""
    sent.outcomes = [1]

    argv = ["test", "-d", f"{REMOTE_ROOT}/models"]
    result = runner.run(HEAD, argv, timeout_s=3.0, mutating=False)

    assert result.exit_code == 1


def test_a_transfer_failure_becomes_a_remote_error(
    tmp_path: Path, var_root: Path, sent: Recorder
) -> None:
    """配布の接続の失敗も `RemoteError`。"""
    local = tmp_path / "payload"
    local.mkdir()
    runner = SshRunner(var_root=var_root, plan=planned_push(local, "payload", delete=False))
    sent.outcomes = [255]

    with pytest.raises(RemoteError):
        runner.push(HEAD, local, "payload", delete=False)


# --- 依存の向き ---------------------------------------------------------


def test_the_remote_module_imports_only_the_types_module() -> None:
    """`remote` は `types` (と、必要なら `config`) だけを import する。"""
    tree = ast.parse(inspect.getsource(remote))
    roots: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "相対の読み込みがある"
            assert node.module is not None
            roots.add(node.module.split(".")[0])
            modules.add(node.module)

    assert roots <= ALLOWED_IMPORT_ROOTS, f"許していない読み込み: {roots - ALLOWED_IMPORT_ROOTS}"
    assert {name for name in modules if name.startswith("serving_kit")} <= {
        "serving_kit.types",
        "serving_kit.config",
    }
