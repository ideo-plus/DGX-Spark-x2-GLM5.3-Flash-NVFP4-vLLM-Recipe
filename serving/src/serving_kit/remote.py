"""Spark に触る、ただ 1 つの場所 (design.md 「遠隔の実行 › remote」)。

`ssh` と `rsync` を `subprocess` に引数のリストで渡して流す。`shell=True` は使わず、
遠隔のシェルに渡る文字列は、必ず `shlex.join` で作る (引用の事故を防ぐ)。

守る決まり:

- **許可の一覧**: `argv[0]` は決まった 13 個だけ。`sudo`、`apt`、`pip`、`systemctl` は
  一覧にないので呼べない (requirements 2.7)。`docker` は、サブコマンドも一覧で絞る
  (`exec`、`build`、`rmi`、`system` などは呼べない。requirements 2.3、8.6)
- **了承**: 状態を変える呼び出しは、了承を得た計画 (`ApprovedPlan`) に含まれていなければ、
  実行せずに `RuntimeError` にする (requirements 2.1)。計画は、前に進むコマンドと、それを
  巻き戻すコマンドの組なので、巻き戻しは、前に進むコマンドが途中で失敗したあとでも実行
  できる (requirements 1.5。失敗で計画が無効にならない)
- **配布の宛先**: `push` は、つねに状態を変える呼び出しとして扱う。宛先は `remote_root` の
  下の `payload/` と `state/` だけで、`models/`、`probe/`、`cache/`、`logs/` には配れない
  (`--delete` つきの rsync が、取得した重みを消す事故を、構造でなくす)
- **回収の宛先**: `pull` の宛先は、Mac の記録の置き場所 (`serving/var/`) の下だけ。元は
  `remote_root` の下だけ
- 時間切れと接続の失敗 (ssh の終了コード 255) は `RemoteError`。遠隔のコマンドが 0 以外で
  終わったことは、誤りではなく `CommandResult.exit_code` で返す

依存の向きにより、この module が読み込む `serving_kit` は `types` だけである。

設計の文からの、意図した 6 つの違い (どれも、より安全な側に寄せたもの):

1. `--` を、宛先のあと (`ssh … <ssh_host> -- <コマンド>`) ではなく、宛先の前に置く。
   OpenSSH の SYNOPSIS は `ssh [options] destination [command …]` で、宛先のあとの語は
   すべて遠隔のコマンドになる。macOS の getopt は引数を並べ替えないので、宛先のあとの
   `--` は遠隔のシェルに渡ってしまう。宛先の前に置けば、どの getopt でも、ちょうど
   「ここまでが指定」の意味になる
2. `mutating=False` と申告されても、状態を変えるコマンド (`docker run` / `stop` / `rm` /
   `pull`、`mkdir`) は、読み取りとして通さない。呼ぶ側の申告だけに頼ると、申告の誤りが
   「了承なしに状態を変える経路」になるため (requirements 2.1 の「構造でなくす」)。申告は
   厳しい側にしか効かない (`mutating=True` の読み取りは、了承の検査に掛かる)
3. `approve` を足した。了承は、読み取りの関門 (`guards`) のあとに来るので、実行役を作った
   あとに 1 度だけ計画を渡せる必要がある。2 度目の了承は断る
4. **rsync の遠隔の道筋に、使える文字を絞る**。`ssh` のコマンドは `shlex.join` で 1 つの
   文字列にできるが、rsync の `<host>:<path>` は、rsync 自身が遠隔のシェルに素で渡すので、
   こちらで引用できない (この Mac の rsync は openrsync で、`-s` / `--protect-args` がない)。
   そこで、`push` の `remote_subdir`、`pull` の `remote_path`、`node.remote_root` を
   `[A-Za-z0-9._/-]` だけに、`node.ssh_host` を `[A-Za-z0-9._-]` だけ (`-` で始めない) に
   絞り、ほかの文字を含むものは実行せずに断る。空白、`$(…)`、`;`、`` ` ``、`~`、改行が、
   構造として入り込めなくなる。Mac の側の道筋 (`local_dir`、`var_root`) は、遠隔のシェルを
   通らず、引数 1 つのまま rsync に渡るので、絞らない
5. **`ip` と `ethtool` を、読み取りの形だけに絞る** (design は `argv[0]` の一覧だけを言う)。
   `ip` は `ip [限られたオプション] (link|addr|address) [show [dev] <名前>]` だけ、
   `ethtool` は `ethtool <名前>`、`-i <名前>`、`-S <名前>` だけを通す。`ip link set`、
   `ip addr add`、`ip route`、`ethtool -s` などは、申告や了承にかかわらず断る
   (requirements 2.7 の「ネットワークの恒久的な設定を変えない」を、構造で守る)
6. **`cat` の読み取り先を絞る**。引数はすべて道筋で (オプションを通さない)、
   `<remote_root>/` の下 (`..` を含まない)、`/sys/class/net/` の下、
   `/sys/class/infiniband/` の下のどれかだけ。認証の情報、`/proc`、ほかのコンテナの記録に
   届かなくする (requirements 2.4、2.6)
"""

from __future__ import annotations

import re
import shlex
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Final, Protocol

from serving_kit.types import (
    ApprovedPlan,
    CommandResult,
    NodeDef,
    NodeRole,
    PlannedPush,
    PlannedRun,
)

__all__ = [
    "CONNECTION_FAILED_EXIT_CODE",
    "CallGuard",
    "RemoteError",
    "RemoteRunner",
    "SshRunner",
]


# --- ssh と rsync の呼び方 ----------------------------------------------

SSH_OPTIONS: Final[tuple[str, ...]] = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=5")
"""非対話の指定と、接続の時間切れ (design.md 「remote」、CLAUDE.md の入り方)。"""

_SSH_TRANSPORT: Final[str] = " ".join(("ssh", *SSH_OPTIONS))
"""rsync の `-e` に渡す ssh の呼び方 (rsync が空白で切り分けるので、空白を含む語は書かない)。"""

CONNECTION_FAILED_EXIT_CODE: Final[int] = 255
"""ssh が、つながらなかったときに返す終了コード。rsync も、同じ値で伝えてくる。"""

_DEFAULT_TRANSFER_TIMEOUT_S: Final[float] = 900.0
"""`push` と `pull` の時間切れ。Protocol に指定の口がないので、実行役を作るときに決める。"""


# --- 許可の一覧 ---------------------------------------------------------

_ALLOWED_COMMANDS: Final[frozenset[str]] = frozenset(
    {
        "docker",
        "nvidia-smi",
        "df",
        "sha256sum",
        "ss",
        "ip",
        "ethtool",
        "ibv_devinfo",
        "ibdev2netdev",
        "cat",
        "mkdir",
        "test",
        "uname",
    }
)
"""`argv[0]` に書けるもの (design.md 「remote」)。一覧を広げるのは、設計の変更として扱う。"""

_ALLOWED_DOCKER_SUBCOMMANDS: Final[frozenset[str]] = frozenset(
    {"run", "ps", "stop", "rm", "logs", "pull", "version"}
)
"""1 語の docker のサブコマンドの許可の一覧。"""

_ALLOWED_DOCKER_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset(
    {("image", "inspect"), ("container", "inspect")}
)
"""2 語の docker のサブコマンドの許可の一覧 (`image ls` などは、ここにないので断る)。"""

_MUTATING_DOCKER_SUBCOMMANDS: Final[frozenset[str]] = frozenset({"run", "stop", "rm", "pull"})
"""状態を変える docker のサブコマンド (申告に関わらず、了承の検査に掛ける)。"""

_MUTATING_COMMANDS: Final[frozenset[str]] = frozenset({"mkdir"})
"""docker 以外で、状態を変えるコマンド (`serve push` が置き場所を作る `mkdir -p`)。"""

_PUSH_SUBDIRS: Final[frozenset[str]] = frozenset({"payload", "state"})
"""配布の宛先にできる、`remote_root` の下の置き場所 (design.md 「remote」)。"""

_IP_OPTIONS: Final[frozenset[str]] = frozenset(
    {"-br", "-brief", "-j", "-json", "-d", "-details", "-s", "-4", "-6", "-o"}
)
"""`ip` に渡せるオプション。`-batch`、`-force`、`-netns` は、読み取りでないので入れない。"""

_IP_OBJECTS: Final[frozenset[str]] = frozenset({"link", "addr", "address"})
"""`ip` で見られる対象。`route`、`netns` などは見ない。"""

_ETHTOOL_OPTIONS: Final[frozenset[str]] = frozenset({"-i", "-S"})
"""`ethtool` に渡せるオプション (ドライバの情報と、統計。どちらも読み取り)。"""

_CAT_PREFIXES: Final[tuple[str, ...]] = ("/sys/class/net/", "/sys/class/infiniband/")
"""`remote_root` の下のほかに、`cat` で読める場所 (インターフェースの読み取り)。"""


# --- 遠隔のシェルに渡る道筋の、文字の絞り込み ---------------------------

_SAFE_PATH_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._/-]+")
"""rsync の遠隔の道筋に書ける文字 (module の docstring の 4)。"""

_SAFE_HOST_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._][A-Za-z0-9._-]*")
"""ssh の宛先に書ける文字 (`-` で始まると、ssh のオプションに見える)。"""


def _check_shell_safe_path(value: str, label: str) -> None:
    """遠隔のシェルに素で渡る道筋を、使える文字だけに絞る。"""
    if not _SAFE_PATH_RE.fullmatch(value):
        raise RuntimeError(
            f"{label}に使えない文字がある"
            " (rsync の遠隔の道筋は遠隔のシェルを通り、こちらで引用できないので、"
            f"英数字と . _ - / だけにする): {value!r}"
        )


def _check_node(node: NodeDef) -> None:
    """ノードの `ssh_host` と `remote_root` を、使える文字だけに絞る。"""
    if not _SAFE_HOST_RE.fullmatch(node.ssh_host):
        raise RuntimeError(
            "ssh の宛先 (ssh_host) に使えない文字がある"
            " (英数字と . _ - だけで、- で始めない。利用者の名前は ~/.ssh/config に書く):"
            f" {node.ssh_host!r}"
        )
    _check_shell_safe_path(node.remote_root, "Spark の置き場所 (remote_root) ")


# --- 誤り ---------------------------------------------------------------


class RemoteError(Exception):
    """遠隔の実行そのものが届かなかったこと (時間切れ、接続の失敗)。

    どのノードの、どのコマンドかを持つ。遠隔のコマンドが 0 以外で終わったことは、ここでは
    なく `CommandResult.exit_code` で返す (design.md 「remote」)。
    """

    def __init__(self, reason: str, *, node: NodeRole, ssh_host: str, argv: Sequence[str]) -> None:
        self.node: NodeRole = node
        self.ssh_host = ssh_host
        self.argv: tuple[str, ...] = tuple(argv)
        super().__init__(f"{node} ({ssh_host}) で{reason}: {shlex.join(self.argv)}")


# --- 実行役の口 ---------------------------------------------------------


class RemoteRunner(Protocol):
    """Spark に触る口 (design.md 「remote」)。

    実装は `SshRunner` の 1 つだけで、試験では `tests/fake_runner.py` の `FakeRunner` に
    差し替える。`approve` は、読み取りの関門のあとに了承が来るための追加である。
    """

    def approve(self, plan: ApprovedPlan) -> None: ...

    def run(
        self, node: NodeDef, argv: Sequence[str], *, timeout_s: float, mutating: bool
    ) -> CommandResult: ...

    def push(
        self, node: NodeDef, local_dir: Path, remote_subdir: str, *, delete: bool
    ) -> CommandResult: ...

    def pull(self, node: NodeDef, remote_path: str, local_dir: Path) -> CommandResult: ...


# --- 許可の一覧と、宛先の決まりの検査 -----------------------------------


def _check_allowlist(node: NodeDef, argv: tuple[str, ...]) -> None:
    """`argv[0]` を許可の一覧で絞り、絞りの要るコマンドは、形まで見る。"""
    if not argv:
        raise RuntimeError("遠隔で流すコマンドが空である")
    command = argv[0]
    if "/" in command:
        raise RuntimeError(
            "遠隔で流すコマンドは、道筋を付けずに名前で書く"
            f" (許可の一覧と突き合わせられない): {command}"
        )
    if command not in _ALLOWED_COMMANDS:
        allowed = ", ".join(sorted(_ALLOWED_COMMANDS))
        raise RuntimeError(f"許可の一覧にないコマンドは流せない: {command} (流せるのは {allowed})")
    if command == "docker":
        _check_docker_subcommand(argv[1:])
    elif command == "ip":
        _check_ip(argv[1:])
    elif command == "ethtool":
        _check_ethtool(argv[1:])
    elif command == "cat":
        _check_cat(node, argv[1:])


def _check_docker_subcommand(rest: tuple[str, ...]) -> None:
    """docker のサブコマンドを絞る (大域のオプションで、すり抜けられないようにする)。"""
    allowed = ", ".join(
        sorted({*_ALLOWED_DOCKER_SUBCOMMANDS, *(" ".join(pair) for pair in _ALLOWED_DOCKER_PAIRS)})
    )
    if not rest:
        raise RuntimeError(f"docker のサブコマンドがない (流せるのは {allowed})")
    if rest[0].startswith("-"):
        raise RuntimeError(
            "docker の大域のオプションは使えない"
            f" (サブコマンドの検査をすり抜けるため): docker {rest[0]}"
        )
    if len(rest) >= 2 and (rest[0], rest[1]) in _ALLOWED_DOCKER_PAIRS:
        return
    if rest[0] in _ALLOWED_DOCKER_SUBCOMMANDS:
        return
    raise RuntimeError(
        f"許可の一覧にない docker のサブコマンドは流せない: {' '.join(rest[:2])}"
        f" (流せるのは {allowed})"
    )


def _check_ip(rest: tuple[str, ...]) -> None:
    """`ip` を、リンクとアドレスの読み取りの形だけに絞る (requirements 2.7)。

    通す形は `ip [限られたオプション…] (link|addr|address) [show [dev] <名前>]` だけである。
    `set`、`add`、`del`、`flush`、`route`、`netns`、`exec` は、`mutating` の申告や了承の
    有無にかかわらず断る (この道具は、ネットワークの設定を変えない)。
    """
    index = 0
    while index < len(rest) and rest[index].startswith("-"):
        if rest[index] not in _IP_OPTIONS:
            allowed = ", ".join(sorted(_IP_OPTIONS))
            raise RuntimeError(f"ip に渡せないオプション: {rest[index]} (渡せるのは {allowed})")
        index += 1
    objects = ", ".join(sorted(_IP_OBJECTS))
    if index >= len(rest) or rest[index] not in _IP_OBJECTS:
        raise RuntimeError(
            f"ip で見られるのは {objects} だけ (ネットワークの設定は変えない): {' '.join(rest)}"
        )
    tail = rest[index + 1 :]
    read_only = (
        not tail
        or tail == ("show",)
        or (len(tail) == 2 and tail[0] == "show" and not tail[1].startswith("-"))
        or (len(tail) == 3 and tail[:2] == ("show", "dev") and not tail[2].startswith("-"))
    )
    if not read_only:
        raise RuntimeError(
            "ip は 'ip [オプション] (link|addr|address) [show [dev] <名前>]' の形だけ"
            f" (ネットワークの設定は変えない): {' '.join(rest)}"
        )


def _check_ethtool(rest: tuple[str, ...]) -> None:
    """`ethtool` を、読み取りの 3 つの形だけに絞る (requirements 2.7)。"""
    name_given = len(rest) == 1 and not rest[0].startswith("-")
    option_given = len(rest) == 2 and rest[0] in _ETHTOOL_OPTIONS and not rest[1].startswith("-")
    if not (name_given or option_given):
        allowed = ", ".join(sorted(_ETHTOOL_OPTIONS))
        raise RuntimeError(
            f"ethtool は 'ethtool <名前>' と、{allowed} を付けた形だけ"
            f" (インターフェースの設定は変えない): {' '.join(rest)}"
        )


def _check_cat(node: NodeDef, rest: tuple[str, ...]) -> None:
    """`cat` の読み取り先を絞る (requirements 2.4、2.6)。

    引数はすべて道筋で (オプションを通さない)、`remote_root` の下か、インターフェースの
    読み取りの場所だけを読める。認証の情報、`/proc`、ほかのコンテナの記録には届かない。
    """
    if not rest:
        raise RuntimeError("cat には、読むファイルの道筋が要る")
    inside_root = f"{node.remote_root}/"
    for value in rest:
        if value.startswith("-"):
            raise RuntimeError(f"cat にオプションは渡せない (道筋だけを書く): {value}")
        allowed_place = value.startswith(inside_root) or value.startswith(_CAT_PREFIXES)
        if not allowed_place or ".." in PurePosixPath(value).parts:
            places = ", ".join((inside_root, *_CAT_PREFIXES))
            raise RuntimeError(
                f"cat で読めるのは {places} の下だけ"
                f" (よその場所、認証の情報、別の構成の記録を読まない): {value}"
            )


def _changes_state(argv: tuple[str, ...]) -> bool:
    """引数の列そのものから見て、Spark の状態を変えるコマンドかどうか。

    呼ぶ側の申告 (`mutating`) だけに頼らないための判定である (module の docstring の 2)。
    """
    command = argv[0]
    if command in _MUTATING_COMMANDS:
        return True
    return command == "docker" and len(argv) > 1 and argv[1] in _MUTATING_DOCKER_SUBCOMMANDS


def _push_destination(node: NodeDef, remote_subdir: str) -> str:
    """配布の宛先を組み立てる (`payload/` と `state/` の下だけ)。

    `""`、`"."`、`"./"` は、`PurePosixPath` の `parts` が空になるので、最初の要素を見る前に
    断る (最初の要素を直に見ると `IndexError` になる)。
    """
    path = PurePosixPath(remote_subdir)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or path.parts[0] not in _PUSH_SUBDIRS
    ):
        allowed = ", ".join(f"{name}/" for name in sorted(_PUSH_SUBDIRS))
        raise RuntimeError(
            f"配布の宛先にできるのは、Spark の置き場所の下の {allowed} だけ"
            " (重み、確認用、キャッシュ、記録の置き場所には配らない):"
            f" {remote_subdir!r}"
        )
    _check_shell_safe_path(remote_subdir, "配布の宛先の道筋")
    return f"{node.remote_root}/{path}"


def _pull_source(node: NodeDef, remote_path: str) -> str:
    """回収の元を組み立てる (`remote_root` の下だけ。絶対の道筋でも書ける)。"""
    _check_shell_safe_path(remote_path, "回収の元の道筋")
    root = PurePosixPath(node.remote_root)
    path = PurePosixPath(remote_path)
    outside = ".." in path.parts
    if not outside and path.is_absolute():
        outside = not path.is_relative_to(root)
    if outside:
        raise RuntimeError(
            f"回収の元は、Spark の置き場所 ({node.remote_root}) の下だけにする: {remote_path!r}"
        )
    return str(path if path.is_absolute() else root / path)


def _pull_target(var_root: Path, local_dir: Path) -> Path:
    """回収の宛先を確かめる (Mac の記録の置き場所の下だけ。`..` での遡りを断る)。"""
    if ".." in local_dir.parts:
        raise RuntimeError(f"回収の宛先に、.. での遡りは書けない: {local_dir}")
    resolved = local_dir.resolve()
    if not resolved.is_relative_to(var_root):
        raise RuntimeError(
            f"回収の宛先は、Mac の記録の置き場所 ({var_root}) の下だけにする: {local_dir}"
        )
    return resolved


class CallGuard:
    """許可の一覧と、了承の検査。

    実物の `SshRunner` と、試験の `FakeRunner` が、この 1 つを共有する (偽物だけが緩いと、
    安全の試験が意味を失うため、検査を二重に持たない)。
    """

    def __init__(self, *, var_root: Path, plan: ApprovedPlan | None = None) -> None:
        self._var_root = var_root.resolve()
        self._plan = plan

    @property
    def plan(self) -> ApprovedPlan | None:
        """いま受け取っている、了承を得た計画。"""
        return self._plan

    def approve(self, plan: ApprovedPlan) -> None:
        """了承を得た計画を、あとから 1 度だけ渡す。"""
        if self._plan is not None:
            raise RuntimeError("了承を得た計画は、1 度しか渡せない (入れ替えを構造でなくす)")
        self._plan = plan

    def check_run(
        self, node: NodeDef, argv: Sequence[str], *, mutating: bool
    ) -> tuple[tuple[str, ...], bool]:
        """コマンドを検査し、引数の列と、状態を変える呼び出しかどうかを返す。"""
        frozen = tuple(argv)
        _check_node(node)
        _check_allowlist(node, frozen)
        effective = mutating or _changes_state(frozen)
        if effective:
            self._require_planned_run(node.role, frozen)
        return frozen, effective

    def check_push(
        self, node: NodeDef, local_dir: Path, remote_subdir: str, *, delete: bool
    ) -> str:
        """配布を検査し、Spark の側の宛先の道筋を返す。"""
        _check_node(node)
        destination = _push_destination(node, remote_subdir)
        self._require_planned_push(node.role, local_dir, remote_subdir, delete=delete)
        return destination

    def check_pull(self, node: NodeDef, remote_path: str, local_dir: Path) -> tuple[str, Path]:
        """回収を検査し、Spark の側の元と、Mac の側の宛先を返す。"""
        _check_node(node)
        return _pull_source(node, remote_path), _pull_target(self._var_root, local_dir)

    def _require_planned_run(self, role: NodeRole, argv: tuple[str, ...]) -> None:
        """ノードと引数の列が、計画に完全に一致するものがあるかを見る。

        前に進むコマンドも、巻き戻すコマンドも通す。前に進むコマンドの失敗で計画が無効に
        ならないので、片付けが、了承のない呼び出しとして断られることがない
        (requirements 1.5)。前方一致や、部分の一致では通さない。
        """
        plan = self._plan
        if plan is not None:
            for command in (*plan.forward, *plan.rollback):
                if (
                    isinstance(command, PlannedRun)
                    and command.node == role
                    and command.argv == argv
                ):
                    return
        raise RuntimeError(
            f"了承を得た計画にない、状態を変える呼び出しは実行しない ({role}: {shlex.join(argv)})"
        )

    def _require_planned_push(
        self, role: NodeRole, local_dir: Path, remote_subdir: str, *, delete: bool
    ) -> None:
        """ノード、元、宛先、`--delete` の有無が、計画に完全に一致するかを見る。"""
        plan = self._plan
        if plan is not None:
            for command in plan.forward:
                if (
                    isinstance(command, PlannedPush)
                    and command.node == role
                    and command.local_dir == local_dir
                    and command.remote_subdir == remote_subdir
                    and command.delete == delete
                ):
                    return
        raise RuntimeError(
            "了承を得た計画にない、状態を変える呼び出しは実行しない"
            f" ({role}: {local_dir} -> {remote_subdir}, delete={delete})"
        )


# --- ただ 1 つの実装 ----------------------------------------------------


class SshRunner:
    """システムの `ssh` と `rsync` で、2 台の Spark に触る実行役。

    `var_root` は、回収の宛先にできる Mac の記録の置き場所 (`serving/var/`) の根である。
    """

    def __init__(
        self,
        *,
        var_root: Path,
        plan: ApprovedPlan | None = None,
        transfer_timeout_s: float = _DEFAULT_TRANSFER_TIMEOUT_S,
    ) -> None:
        self._guard = CallGuard(var_root=var_root, plan=plan)
        self._transfer_timeout_s = transfer_timeout_s

    def approve(self, plan: ApprovedPlan) -> None:
        """了承を得た計画を、あとから 1 度だけ渡す。"""
        self._guard.approve(plan)

    def run(
        self, node: NodeDef, argv: Sequence[str], *, timeout_s: float, mutating: bool
    ) -> CommandResult:
        """1 台で 1 つのコマンドを流す。遠隔に渡る文字列は `shlex.join` で作る。"""
        frozen, _ = self._guard.check_run(node, argv, mutating=mutating)
        sent = ["ssh", *SSH_OPTIONS, "--", node.ssh_host, shlex.join(frozen)]
        return self._execute(node, sent, frozen, timeout_s=timeout_s)

    def push(
        self, node: NodeDef, local_dir: Path, remote_subdir: str, *, delete: bool
    ) -> CommandResult:
        """Mac の側のディレクトリの中身を、Spark の決まった宛先に配る。"""
        destination = self._guard.check_push(node, local_dir, remote_subdir, delete=delete)
        sent = [
            "rsync",
            "-a",
            *(("--delete",) if delete else ()),
            "-e",
            _SSH_TRANSPORT,
            f"{local_dir}/",
            f"{node.ssh_host}:{destination}/",
        ]
        return self._execute(node, sent, tuple(sent), timeout_s=self._transfer_timeout_s)

    def pull(self, node: NodeDef, remote_path: str, local_dir: Path) -> CommandResult:
        """Spark の置き場所の下のものを、Mac の記録の置き場所の下に写す。"""
        source, target = self._guard.check_pull(node, remote_path, local_dir)
        target.mkdir(parents=True, exist_ok=True)
        sent = [
            "rsync",
            "-a",
            "-e",
            _SSH_TRANSPORT,
            f"{node.ssh_host}:{source}",
            f"{target}/",
        ]
        return self._execute(node, sent, tuple(sent), timeout_s=self._transfer_timeout_s)

    def _execute(
        self, node: NodeDef, sent: Sequence[str], argv: tuple[str, ...], *, timeout_s: float
    ) -> CommandResult:
        """引数のリストで流し、時間切れと接続の失敗だけを `RemoteError` にする。"""
        started = time.monotonic()
        try:
            completed = subprocess.run(
                sent,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteError(
                f"{timeout_s} 秒のうちに終わらなかった",
                node=node.role,
                ssh_host=node.ssh_host,
                argv=argv,
            ) from exc
        except OSError as exc:
            raise RemoteError(
                f"コマンドを起こせなかった ({exc})",
                node=node.role,
                ssh_host=node.ssh_host,
                argv=argv,
            ) from exc
        duration_s = time.monotonic() - started
        if completed.returncode == CONNECTION_FAILED_EXIT_CODE:
            raise RemoteError(
                f"つながらなかった ({completed.stderr.strip()})",
                node=node.role,
                ssh_host=node.ssh_host,
                argv=argv,
            )
        return CommandResult(
            argv=argv,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_s=duration_s,
        )
