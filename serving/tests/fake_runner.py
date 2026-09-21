"""`RemoteRunner` の偽物 (task 1.4)。

Spark に触らずに、呼ばれた (ノード、引数の列、`mutating`、`push` / `pull` の引数) と
その順序を記録し、台本どおりの `CommandResult` を返す。実物の `ssh`、`rsync`、`docker` は、
どの段でも呼ばない。

呼び名について: テストダブルの分類では、これは Fake (中身を簡略にした実装) ではなく、
**Stub (台本どおりに返す) + Spy (呼び出しを記録して、試験があとから検査する)** である。期待を
先に仕込んで自分で検証する Mock でもない。`fake_` という名前は、P0 の `bench/tests/fake_server.py`
と、design.md の File Structure Plan に合わせている。

**あとのタスク (2.3、2.4、3.x、4.x、5.x) は、この偽物を書き換えずに、台本だけで使う。**
足りない台本の形が出てきたら、並行の作業を止めて、この偽物の変更を 1 つの作業として先に
行うこと (`types.py` と同じ扱い)。

台本の決まり:

- 規則 (`Rule`) は、引数の列の**前方一致** (`prefix`) と**述語** (`when`) で選べる。両方
  書いたときは、両方を満たす呼び出しにだけ当たる。どちらも書かない規則は、何にでも当たる
- 上から順に見て、最初に当たった規則の返事 (`replies`) を使う。返事は 1 回ずつ順に消費し、
  尽きたら最後の返事を繰り返す。これで、同じコマンドの 1 回目と 2 回目で違う結果を返せる
  (「起こした直後は running、次は exited」など)
- `node` を書いた規則は、その役割の呼び出しにだけ当たる (2 台の差を作るため)
- `kind` は、`run` (既定)、`push`、`pull` のどれか。`push` と `pull` の規則は、`node` と
  `remote_prefix` (`remote_subdir` / `remote_path` の前方一致) で選び分けられる。これで、
  「head の記録はある、worker の記録はない」を台本で作れる
- `pull` の返事は、`writes` で、宛先に置くファイル (名前と中身) を指定できる。`pull` は、
  実物と同じく、宛先のディレクトリを作る
- 失敗の意味は実物と揃える。`Reply(raises=…)` は例外を投げ、`Reply(exit_code=255)` は
  `RemoteError` になる
- **台本に合わない呼び出しは、既定では `AssertionError` にする**。試験が思っていない呼び出しを
  黙って通さないためである。気にしない試験は、`default=Reply()` を渡すと、空の成功が返る

安全の検査 (`argv[0]` と docker のサブコマンドの許可の一覧、`ip` / `ethtool` / `cat` の形、
了承を得た計画、配布と回収の宛先と、遠隔の道筋の文字) は、`remote.CallGuard` を共有して
行う。偽物だけが緩いと、タスク 5.3 の安全の試験が意味を失うので、検査は二重に持たない。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from serving_kit.remote import CONNECTION_FAILED_EXIT_CODE, CallGuard, RemoteError
from serving_kit.types import ApprovedPlan, CommandResult, NodeDef, NodeRole

__all__ = ["CallKind", "FakeRunner", "RecordedCall", "Reply", "Rule"]

CallKind = Literal["run", "push", "pull"]
"""記録と台本の、呼び出しの種類。"""


@dataclass(frozen=True)
class Reply:
    """1 回ぶんの返事。`CommandResult` の `argv` は、呼ばれた引数の列で埋める。

    失敗の意味は、実物と揃える。

    - `raises` を書くと、結果を返すかわりに、その例外を投げる (時間切れの経路を作れる)
    - `exit_code` が 255 のときは、実物と同じく接続の失敗として `RemoteError` になる
      (偽物だけが `CommandResult(exit_code=255)` を作れる、という食い違いをなくす)
    - `writes` は、`pull` の宛先に置くファイル (相対の名前と中身) の組。回収した記録が
      その場所にできることを、試験から確かめられる
    """

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    raises: BaseException | None = None
    writes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Rule:
    """台本の 1 つの規則。

    `prefix` と `when` は `run` の引数の列に、`remote_prefix` は `push` の `remote_subdir` と
    `pull` の `remote_path` に掛かる。`node` は、どの種類にも掛かる。
    """

    prefix: tuple[str, ...] | None = None
    replies: tuple[Reply, ...] = (Reply(),)
    when: Callable[[tuple[str, ...]], bool] | None = None
    node: NodeRole | None = None
    kind: CallKind = "run"
    remote_prefix: str | None = None

    def matches(
        self, kind: CallKind, node: NodeRole, argv: tuple[str, ...], remote: str | None
    ) -> bool:
        """この規則が、1 つの呼び出しに当たるかどうか。"""
        if self.kind != kind or (self.node is not None and self.node != node):
            return False
        if self.remote_prefix is not None and not (
            remote is not None and remote.startswith(self.remote_prefix)
        ):
            return False
        if self.prefix is not None and argv[: len(self.prefix)] != self.prefix:
            return False
        return self.when is None or self.when(argv)


@dataclass(frozen=True)
class RecordedCall:
    """記録した 1 つの呼び出し。

    `mutating` は、実際に了承の検査に掛けたかどうかである (呼ぶ側の申告ではなく、
    `remote` が引数の列から判断したあとの値)。
    """

    kind: CallKind
    node: NodeRole
    argv: tuple[str, ...] = ()
    mutating: bool = False
    timeout_s: float | None = None
    local_dir: Path | None = None
    remote: str | None = None
    """`push` の `remote_subdir`、`pull` の `remote_path`。"""
    delete: bool = False


class FakeRunner:
    """`RemoteRunner` の偽物。`remote.CallGuard` と同じ検査を通す。

    `var_root` は、回収の宛先にできる根 (試験では `tmp_path` の下)。`script` は台本、
    `default` は台本に合わない呼び出しの返事 (渡さなければ誤りにする)。
    """

    def __init__(
        self,
        *,
        var_root: Path,
        plan: ApprovedPlan | None = None,
        script: Sequence[Rule] = (),
        default: Reply | None = None,
    ) -> None:
        self.script = tuple(script)
        self.default = default
        self._guard = CallGuard(var_root=var_root, plan=plan)
        self._calls: list[RecordedCall] = []
        self._used: dict[int, int] = {}

    # --- 記録の読み出し -------------------------------------------------

    @property
    def plan(self) -> ApprovedPlan | None:
        """いま受け取っている、了承を得た計画。"""
        return self._guard.plan

    @property
    def calls(self) -> tuple[RecordedCall, ...]:
        """呼ばれたものの全部 (呼ばれた順)。"""
        return tuple(self._calls)

    @property
    def runs(self) -> tuple[RecordedCall, ...]:
        """コマンドの呼び出しだけ。"""
        return tuple(call for call in self._calls if call.kind == "run")

    @property
    def pushes(self) -> tuple[RecordedCall, ...]:
        """配布だけ。"""
        return tuple(call for call in self._calls if call.kind == "push")

    @property
    def pulls(self) -> tuple[RecordedCall, ...]:
        """回収だけ。"""
        return tuple(call for call in self._calls if call.kind == "pull")

    @property
    def argvs(self) -> tuple[tuple[str, ...], ...]:
        """流れた引数の列だけ (安全の試験が、全体を見るため)。"""
        return tuple(call.argv for call in self.runs)

    # --- 実行役の口 -----------------------------------------------------

    def approve(self, plan: ApprovedPlan) -> None:
        """了承を得た計画を、あとから 1 度だけ渡す (実物と同じ)。"""
        self._guard.approve(plan)

    def run(
        self, node: NodeDef, argv: Sequence[str], *, timeout_s: float, mutating: bool
    ) -> CommandResult:
        frozen, effective = self._guard.check_run(node, argv, mutating=mutating)
        self._calls.append(
            RecordedCall(
                kind="run",
                node=node.role,
                argv=frozen,
                mutating=effective,
                timeout_s=timeout_s,
            )
        )
        return self._answer("run", node, frozen, None, None)

    def push(
        self, node: NodeDef, local_dir: Path, remote_subdir: str, *, delete: bool
    ) -> CommandResult:
        self._guard.check_push(node, local_dir, remote_subdir, delete=delete)
        self._calls.append(
            RecordedCall(
                kind="push",
                node=node.role,
                mutating=True,
                local_dir=local_dir,
                remote=remote_subdir,
                delete=delete,
            )
        )
        return self._answer("push", node, (), remote_subdir, None)

    def pull(self, node: NodeDef, remote_path: str, local_dir: Path) -> CommandResult:
        _, target = self._guard.check_pull(node, remote_path, local_dir)
        self._calls.append(
            RecordedCall(
                kind="pull",
                node=node.role,
                local_dir=local_dir,
                remote=remote_path,
            )
        )
        # 実物と同じく、宛先のディレクトリを作ってから写す
        target.mkdir(parents=True, exist_ok=True)
        return self._answer("pull", node, (), remote_path, target)

    # --- 台本 -----------------------------------------------------------

    def _answer(
        self,
        kind: CallKind,
        node: NodeDef,
        argv: tuple[str, ...],
        remote: str | None,
        target: Path | None,
    ) -> CommandResult:
        """台本から、この呼び出しの返事を選び、実物と同じ失敗の意味にして返す。"""
        reply = self._reply(kind, node.role, argv, remote)
        if reply.raises is not None:
            raise reply.raises
        if target is not None:
            _write_files(target, reply.writes)
        if reply.exit_code == CONNECTION_FAILED_EXIT_CODE:
            raise RemoteError(
                f"つながらなかった ({reply.stderr.strip()})",
                node=node.role,
                ssh_host=node.ssh_host,
                argv=argv or ("rsync", kind, remote or ""),
            )
        return CommandResult(
            argv=argv,
            exit_code=reply.exit_code,
            stdout=reply.stdout,
            stderr=reply.stderr,
            duration_s=reply.duration_s,
        )

    def _reply(
        self, kind: CallKind, node: NodeRole, argv: tuple[str, ...], remote: str | None
    ) -> Reply:
        """上から順に見て、最初に当たった規則の、この回の返事を取る。"""
        for index, rule in enumerate(self.script):
            if not rule.matches(kind, node, argv, remote):
                continue
            used = self._used.get(index, 0)
            self._used[index] = used + 1
            return rule.replies[min(used, len(rule.replies) - 1)]
        if self.default is not None:
            return self.default
        shown = " ".join(argv) if argv else f"{kind} {remote}"
        raise AssertionError(f"台本にない呼び出し ({node}, {kind}): {shown}")


def _write_files(target: Path, writes: tuple[tuple[str, str], ...]) -> None:
    """回収した記録として、宛先にファイルを置く。"""
    for name, text in writes:
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts, (
            f"回収の宛先の外に書く台本: {name}"
        )
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
