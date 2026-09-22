"""イメージの取得と照合、イメージの中のライセンスの表記の読み取り
(design.md 「イメージと重み › image」)。

受け持つのは、2 つのコマンドの中身である。

- `serve pull-image <構成>`: 関門 (`gate_reachable`、`gate_disk_space`) → 了承 → 2 台で
  `docker pull <ダイジェストでの参照>` → `gate_image_digest` で照合
- `serve image-licenses <構成>`: `kind = "inspect"` の構成から `plan.build_plans` で計画を
  作り、関門 → 了承 → `-d` で起こし → 終了を待ち → `docker logs` で出力を読み → 止めて消す

守る決まり:

- **1 つの起動の仕組み** (design.md 「Architecture Integration」): 読み取りのコンテナも、
  ほかの 4 つの `kind` と同じ道を通る。`plan.build_plans` が組み立てた引数の列を、名前と
  ラベルを付けて `-d` で起こし、終わりを待ち、出力を読んでから消す。数秒で終わる `inspect`
  も例外にしない。前面で動かす経路、終了時に自動で消す指定 (`--rm`)、構成を持たずに起こす
  経路は、どれも作らない
- **ダイジェストでの取得だけ** (requirements 3.2): 取得の参照は、つねに `ImageRef.ref`
  (`<名前>@sha256:<64 桁>`) である。`types.ImageRef` が形を固定しているので、タグでの取得の
  経路は、この module にも構成にもない。取得のあとに照合し、合わなければ、食い違いを示して
  止まる (**黙って取り直さない**)
- **了承** (requirements 2.1): 状態を変える呼び出し (`docker pull`、`docker run`、
  `docker stop` / `rm`) は、すべて `guards.build_approved_plan` が作った計画に入れ、
  `guards.request_approval` で了承を得てから流す。了承されなければ、`ApprovalError` に
  なるので、取得の呼び出しも、コンテナを起こす呼び出しも、1 つも出ない
- **コンテナを対象にする操作の不変条件** (requirements 2.3、2.4): 対象にできるのは、(a)
  `guards.list_own_containers` (自分のラベルで絞った、ただ 1 つの入口) が返した行の識別子か、
  (b) 了承済みの計画が自分で起こす名前だけである。終わりの確認 (`docker container inspect`)
  と出力の読み取り (`docker logs`) は (a) の識別子に向ける。`docker run` が返す識別子は
  使わない。止めると消す ((b) の名前に向ける列) は、**流す直前に、その名前の行が自分の一覧に
  あることを `guards.rollback_commands` で確かめてから**流す
- **片付けてから終わる** (design.md 「Error Handling」): 成功でも、失敗でも、時間切れでも、
  中断でも、読み取りのコンテナを残さない。**ただし、止めてよい相手だけを止める**。名前の
  衝突 (その名前が、よそのコンテナのもの) と、コンテナができなかった場合は、自分の一覧に
  その名前が無いので、`docker stop` も `docker rm` も 1 つも出さずに、名前を示して止まる

依存の向きにより、この module が読み込む `serving_kit` は `types`、`config`、`remote`、
`plan`、`guards` だけである (`weights`、`lifecycle` 以降は読み込まない)。

design の文からの、意図した決めごと (design が明示しない細部を、ここで決めて残す):

1. **取得の前に `gate_image_digest` を流さない**。design の `serve pull-image` の流れが、
   照合を取得の**あと**に置いているためである。`docker pull` は、すでにある層を取り直さない
   ので、2 台のうち片方にだけイメージがある場合も、同じ 1 つの道で済む (取得を飛ばす分岐を
   作らない。分岐を作ると、飛ばした台の照合が「取得の結果」ではなくなる)
2. **待ちの間隔は 1 秒** (`POLL_INTERVAL_S`)。design の「10 秒ごとに見る」は、推論サーバーの
   受け付けの開始の判定 (`lifecycle`) の値である。ライセンスの表記の読み取りは数秒で終わる
   ので、ここは短くする。上限は、構成の `ready_timeout_s`。間隔と時計は引数で受け取るので、
   試験は実際に眠らない
3. **終わりの確認は `docker container inspect`** の `State.Status` と `State.ExitCode` で
   行う。`docker ps` の `Status` の文言 (`Exited (0) 3 seconds ago`) を読まないのは、その形に
   公式の定義がないためである。識別子は、必ず `guards.list_own_containers` から取る
4. **`docker logs` に `--timestamps` を付けない**。`logs.collect_logs` は記録として写すので
   時刻を付けるが、ここはライセンスの表記そのものを見せるので、行を加工しない
5. **名前の衝突は、標準エラーの文面で「判定しない」**。`docker stop <名前>` と
   `docker rm <名前>` は、その名前のコンテナが**誰のものでも**止めて消すので、文面
   (`Conflict.` と `is already in use`) を見分けの根拠にすると、文面が実物と違ったときに、
   自分のものでないコンテナを止めてしまう (requirements 2.3 に反する)。そこで、**止めて消す
   相手は、つねに `guards.rollback_commands` が、自分のラベルで絞った一覧で決める** (流す
   直前に読み、その名前の行があるときだけ流す。名前は 1 台の中で一意なので、自分の一覧に
   あれば、それは自分のコンテナである)。`run_gates` の `own_state` が「自分のコンテナが
   残っていない」ことを確かめた直後に起こすので、`docker run` のあとに自分の一覧にその名前が
   現れるのは、「コンテナはできたが、起動に失敗した」場合だけで、それは止めてよい。
   標準エラーの文面 (`_NAME_CONFLICT_MARKS`) は、**誤りの文を親切にするためだけ**に見る
   (合えば「名前が衝突した」と言い、合わなければ「起こせなかった」と言う。終了コードは
   どちらも 2)。実機の文面と終了コードは未確認なので、7.1 で確かめて、文言を直す
6. **結果の型は、この module に置く**。`types.py` は、ほかのタスクが使っているので変更しない
   (tasks.md の Implementation Notes 1.2: image の結果の型は、凍結の対象の外)
7. **終了コードの振り分け** (design.md 「Error Handling」): 関門の不通過は `status =
   "refused"` の結果 (呼ぶ側が理由を並べて終了コード 1)、了承されなかったことは
   `guards.ApprovalError` (終了コード 1)、構成の種類の取り違えは `config.ConfigError`
   (終了コード 1)。取得の失敗、照合の不一致、時間切れ、名前の衝突、片付けの失敗は
   `ImageError` (**実行して失敗した**ので、終了コード 2)。**了承のあとに読み取りが届かな
   かったこと (`remote.RemoteError`) も、`ImageError` に包む** (どの台の、どの段かを文に
   入れる)。了承の**前**の「入れない」は、関門が `GateResult` に変えるので、終了コード 1 に
   なる。これで、5.1 は `RemoteError` を見分けずに、この 2 つの型だけを写せばよい
8. **何も読めなかったことを、正常な結果にしない**。起こしたあとに (a) 自分のコンテナの一覧を
   読めない、(b) 一覧にその名前がない、(c) `docker logs` が 0 以外で終わる、の 3 つは、
   `LicensesOutcome` では返さず `ImageError` にする (終了コード 2)。返してしまうと、5.1 が
   終了コード 0 に写し、「読めた」と誤って伝わるためである。**コンテナの中の `cat` が 0 以外
   で終わったこと** (読むファイルが無かった) は、読み取りの失敗ではないので、これまでどおり
   結果として返す (出力と終了コードを、計測者が読んで `LICENSES.md` に書く)。どの経路でも、
   誤りを投げる前に、片付けを必ず通す
9. **片付けの途中で中断が来ても、消すところまでは試みる** (コンテナを残さないことを優先。
   2 度目の中断は、そのまま伝える)。中断の経路で片付けが終わらなかったことは、捨てずに
   1 行で知らせる (出し先は `report` 引数。既定は `sys.stderr`)
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, TextIO

from serving_kit.config import ConfigError
from serving_kit.guards import (
    READ_TIMEOUT_S,
    STOP_TIMEOUT_S,
    Confirmer,
    OwnContainer,
    build_approved_plan,
    gate_disk_space,
    gate_image_digest,
    gate_reachable,
    list_own_containers,
    request_approval,
    required_free_bytes,
    rollback_commands,
    run_gates,
)
from serving_kit.plan import build_plans
from serving_kit.remote import RemoteError, RemoteRunner
from serving_kit.types import (
    CommandResult,
    ConfigDef,
    ContainerPlan,
    GateResult,
    NodeDef,
    NodeRole,
    PlannedRun,
)

__all__ = [
    "CLEANUP_TIMEOUT_S",
    "POLL_INTERVAL_S",
    "PULL_TIMEOUT_S",
    "START_TIMEOUT_S",
    "ImageError",
    "LicenseReading",
    "LicensesOutcome",
    "PullImageOutcome",
    "pull_image",
    "read_image_licenses",
]


# --- 決まった値 -----------------------------------------------------------

PULL_TIMEOUT_S: Final[float] = 3600.0
"""1 台ぶんの `docker pull` の時間切れ。

イメージは 10 GiB ほどあり、回線と Docker Hub の速さで大きく変わるので、長めに取る。
"""

START_TIMEOUT_S: Final[float] = 120.0
"""`docker run -d` の時間切れ。切り離して起こすので、すぐ返る。"""

POLL_INTERVAL_S: Final[float] = 1.0
"""終わりを待つ間隔 (module の docstring の決めごとの 2)。"""

CLEANUP_TIMEOUT_S: Final[float] = STOP_TIMEOUT_S + 30.0
"""片付け (`docker stop -t 90` と `docker rm`) の時間切れ。停止の猶予より長くする。"""

_KIND_INSPECT: Final[str] = "inspect"
"""ライセンスの表記の読み取りの構成の種類 (design.md Data Models の `p1-image-licenses`)。"""

_STATE_FORMAT: Final[str] = "{{.State.Status}} {{.State.ExitCode}}"
"""`docker container inspect` に渡す書式 (状態と終了コードを、1 行で読む)。"""

_STATE_ABSENT: Final[str] = "absent"
"""コンテナがもう無い (`docker container inspect` が 0 以外で終わった)。"""

_STATE_UNKNOWN: Final[str] = "unknown"
"""状態を読めなかった (読み取りが届かない、出力が空)。"""

_UNFINISHED_STATES: Final[frozenset[str]] = frozenset(
    {"created", "running", "restarting", "paused", "removing"}
)
"""まだ終わっていないコンテナの状態。ここにない状態 (`exited`、`dead`、読めなかったとき) は、
待つのをやめて、出力を読み、片付けに進む。"""

_NAME_CONFLICT_MARKS: Final[tuple[str, ...]] = ("conflict.", "is already in use")
"""名前の衝突らしい、標準エラーの印 (module の docstring の決めごとの 5)。

**誤りの文を親切にするためだけに使う**。片付けに行くかどうかは、この印では決めない。
"""

_CLEANUP_LABELS: Final[Mapping[str, str]] = {"stop": "止められなかった", "rm": "消せなかった"}
"""片付けの失敗を言うときの、docker のサブコマンドごとの言い方。"""

_REMOVE_SUBCOMMAND: Final[str] = "rm"
"""片付けのうち、中断が来たあとでも試みるもの (コンテナを残さないことを優先する)。"""


class ImageError(Exception):
    """イメージの取得と読み取りが、**実行して失敗した**こと。

    取得の失敗、取得のあとの照合の不一致、読み取りの時間切れ、コンテナの名前の衝突、
    片付けの失敗が、ここに乗る。design.md 「Error Handling」の終了コード 2 である
    (関門の不通過と、了承されなかったことは、終了コード 1 なので、ここには乗せない)。
    """


# --- 結果の型 -------------------------------------------------------------


@dataclass(frozen=True)
class PullImageOutcome:
    """`serve pull-image` の結末 (module の docstring の決めごとの 6)。

    `refused` は、関門が断ったこと (呼ぶ側は `gates` の理由を並べて、終了コード 1)。
    取得や照合が失敗したときは、この型では返らず `ImageError` になる。
    """

    status: Literal["pulled", "refused"]
    config_name: str
    reference: str
    gates: tuple[GateResult, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class LicenseReading:
    """1 台ぶんの、読み取りのコンテナの出力。

    `cat` が 0 以外で終わった場合 (読むファイルが無い) も、これを返す。どのファイルがあって、
    どれが無かったかを、計測者が読めるように、標準出力と標準エラーの両方と、終了コードを
    持つ。読むファイルの道筋は、構成の位置の引数から来るので、この型にも module にも書かない。
    """

    node: NodeRole
    container_name: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    detail: str = ""

    @property
    def text(self) -> str:
        """計測者に見せる文字列 (読めたものと、読めなかったものの両方)。"""
        code = "不明" if self.exit_code is None else str(self.exit_code)
        lines = [f"[{self.node}] {self.container_name} (終了コード {code})"]
        if self.detail:
            lines.append(f"  {self.detail}")
        lines.extend(body.rstrip("\n") for body in (self.stdout, self.stderr) if body.strip())
        return "\n".join(lines)


@dataclass(frozen=True)
class LicensesOutcome:
    """`serve image-licenses` の結末。

    `refused` は、関門が断ったこと (終了コード 1)。読み取りが始まったあとの失敗は、
    `ImageError` になる (どの場合も、コンテナは残さない)。
    """

    status: Literal["read", "refused"]
    config_name: str
    gates: tuple[GateResult, ...] = ()
    readings: tuple[LicenseReading, ...] = ()
    detail: str = ""

    @property
    def text(self) -> str:
        """計測者に見せる文字列 (台ごとの出力を、続けて並べる)。"""
        if not self.readings:
            return self.detail
        return "\n\n".join(reading.text for reading in self.readings)


@dataclass(frozen=True)
class _Controls:
    """1 台ぶんの読み取りの、差し替えられる口と時間切れ。

    時計と眠りと知らせの出し先を引数から受けることで、試験は、実際に眠らず、画面にも
    書かずに済む。
    """

    sleep: Callable[[float], None]
    clock: Callable[[], float]
    report: TextIO
    poll_interval_s: float
    timeout_s: float
    start_timeout_s: float
    read_timeout_s: float


# --- 小さな助け -----------------------------------------------------------


def _node_of(nodes: Mapping[NodeRole, NodeDef], config: ConfigDef, role: NodeRole) -> NodeDef:
    """構成が使う役割のノードの定義を取る (なければ、触る前に断る)。"""
    node = nodes.get(role)
    if node is None:
        raise ValueError(f"nodes.{role} の定義がない (構成 '{config.name}' が使う役割)")
    return node


def _shown_failure(result: CommandResult) -> str:
    """失敗した呼び出しの、見せる文 (標準エラーがなければ標準出力)。"""
    return result.stderr.strip() or result.stdout.strip() or "(出力なし)"


def _refused(gates: Sequence[GateResult]) -> tuple[GateResult, ...]:
    return tuple(gate for gate in gates if not gate.passed)


def _refusal_detail(refused: Sequence[GateResult]) -> str:
    """断った関門を、台と関門の名前つきで並べる。"""
    return "関門が断った: " + " / ".join(
        f"{gate.node or '-'} の {gate.gate}: {gate.detail}" for gate in refused
    )


# --- イメージの取得 -------------------------------------------------------


def _pull_argv(reference: str) -> tuple[str, ...]:
    """イメージを取得する引数の列 (**つねにダイジェストでの参照**)。

    呼び出し元の `pull_image` がローカル ID を断った後に使う。
    タグでの取得の列を作る口は、この module にない。
    """
    return ("docker", "pull", reference)


def pull_image(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    *,
    confirmer: Confirmer,
    pull_timeout_s: float = PULL_TIMEOUT_S,
    read_timeout_s: float = READ_TIMEOUT_S,
) -> PullImageOutcome:
    """構成のイメージを、ダイジェストで 2 台に取得して、照合する (`serve pull-image`)。

    進む順は design.md 「image」のとおり: 関門 (入れるか、空きがあるか) → 了承 → 2 台で
    `docker pull` → `gate_image_digest` で照合。関門が 1 つでも断れば、了承を尋ねず、取得の
    呼び出しを 1 つも出さずに返る (requirements 2.1、2.5)。

    引数:
        runner: 遠隔の実行役。
        config: イメージを取得する構成 (`config.select_config` を通ったもの)。
        nodes: 役割ごとのノードの定義。
        confirmer: 計画を見せて、了承を得る口。
        pull_timeout_s: 1 台ぶんの `docker pull` の時間切れ。
        read_timeout_s: 関門と照合の読み取りの時間切れ。

    返り値:
        取得できたか、関門が断ったか (`gates` に、流した関門の結果を、流した順に持つ)。

    例外:
        ImageError: 取得が失敗した、**了承のあとに読み取りが届かなかった** (ssh の時間切れ、
            切断)、取得のあとの照合が合わない (どれも、実行しての失敗で終了コード 2)。
        guards.ApprovalError: 計測者が了承しなかった (終了コード 1)。
        ValueError: 構成が使う役割のノードの定義がないとき。

    **`RemoteError` は、この関数からは出ない**。了承の前 (関門) の「入れない」は
    `gate_reachable` / `gate_disk_space` が `GateResult` (`passed=False`) に変えるので
    `status = "refused"` になり、5.1 は終了コード 1 に写す。了承のあと (`docker pull`) に
    届かなかったことは、ここで `ImageError` に包むので、5.1 は終了コード 2 に写す
    (照合の段の `RemoteError` も、`gate_image_digest` が `passed=False` にしたうえで、
    食い違いの `ImageError` になる)。
    """
    reference = config.image.ref
    if reference.startswith("sha256:"):
        raise ValueError("ローカルイメージ ID は pull できない。対象ノードでビルド結果を確認する")
    # 取得するのはイメージなので、要る空きはイメージの大きさである (重みのマニフェストは
    # 渡さない。`required_free_bytes` は、マニフェストがなければイメージの大きさを返す)
    required_bytes = required_free_bytes(config, None)
    gates: list[GateResult] = []
    for role in config.nodes:
        node = _node_of(nodes, config, role)
        reachable = gate_reachable(runner, node, timeout_s=read_timeout_s)
        gates.append(reachable)
        if not reachable.passed:
            continue  # 入れない台では、何も読めない (残りの関門を飛ばす)
        gates.append(gate_disk_space(runner, node, required_bytes, timeout_s=read_timeout_s))
    refused = _refused(gates)
    if refused:
        return PullImageOutcome(
            status="refused",
            config_name=config.name,
            reference=reference,
            gates=tuple(gates),
            detail=_refusal_detail(refused),
        )

    argv = _pull_argv(reference)
    # コンテナを起こさないので、計画は `extra_forward` の取得だけである。巻き戻しは要らない
    # (取得したイメージを消さない。`docker rmi` は、`remote` の許可の一覧にない)
    approved = build_approved_plan(
        (),
        extra_forward=tuple(
            PlannedRun(node=role, argv=argv, purpose=f"{role} でイメージを取得する ({reference})")
            for role in config.nodes
        ),
    )
    request_approval(confirmer, runner, approved, nodes)

    for role in config.nodes:
        node = _node_of(nodes, config, role)
        try:
            result = runner.run(node, argv, timeout_s=pull_timeout_s, mutating=True)
        except RemoteError as exc:
            # 了承のあとに届かなかったことは、実行しての失敗 (終了コード 2) として包む。
            # どの台の、どの段かを言うので、5.1 は `RemoteError` を見分けなくてよい
            raise ImageError(
                f"{role} ({node.ssh_host}) で、イメージの取得 (docker pull) の途中に、"
                f"読み取りが届かなかった ({reference}): {exc}"
            ) from exc
        if result.exit_code != 0:
            raise ImageError(
                f"{role} ({node.ssh_host}) で、イメージを取得できなかった"
                f" ({reference}): {_shown_failure(result)}"
            )

    verified = tuple(
        gate_image_digest(runner, _node_of(nodes, config, role), config, timeout_s=read_timeout_s)
        for role in config.nodes
    )
    gates.extend(verified)
    mismatched = _refused(verified)
    if mismatched:
        raise ImageError(
            "取得したあとも、手元のイメージが構成のダイジェストと合わない"
            " (黙って取り直さない): "
            + " / ".join(f"{gate.node}: {gate.detail}" for gate in mismatched)
        )
    return PullImageOutcome(
        status="pulled",
        config_name=config.name,
        reference=reference,
        gates=tuple(gates),
        detail=f"{len(config.nodes)} 台で {reference} を取得し、ダイジェストを照合した",
    )


# --- ライセンスの表記の読み取り -------------------------------------------


def _own_container(
    runner: RemoteRunner, node: NodeDef, container_name: str, *, timeout_s: float
) -> tuple[OwnContainer | None, str]:
    """触ってよい識別子を、自分のラベルで絞った一覧から決める (不変条件の (a))。

    一覧が読めない (ssh で入れない、所有のラベルのない行が紛れている) ときも、名前の合う行が
    ないときも、例外を出さずに理由の文を返す (呼ぶ側は、読むのをやめて片付けに進む)。
    """
    try:
        containers = list_own_containers(runner, node, timeout_s=timeout_s)
    except RemoteError as exc:
        return None, f"読み取りが届かなかった: {exc}"
    except ValueError as exc:
        return None, f"自分のコンテナの一覧を読めなかった: {exc}"
    found = next((container for container in containers if container.name == container_name), None)
    if found is None:
        return None, f"自分のコンテナがない ({container_name})"
    return found, ""


def _container_state(
    runner: RemoteRunner, node: NodeDef, container_id: str, *, timeout_s: float
) -> tuple[str, int | None]:
    """コンテナの状態と終了コードを読む (対象は、一覧から来た識別子だけ)。"""
    argv = ("docker", "container", "inspect", "--format", _STATE_FORMAT, container_id)
    try:
        result = runner.run(node, argv, timeout_s=timeout_s, mutating=False)
    except RemoteError:
        return _STATE_UNKNOWN, None
    if result.exit_code != 0:
        return _STATE_ABSENT, None
    fields = result.stdout.split()
    if not fields:
        return _STATE_UNKNOWN, None
    code = fields[1] if len(fields) > 1 else ""
    return fields[0], int(code) if code.lstrip("-").isdecimal() else None


def _wait_for_exit(
    runner: RemoteRunner, node: NodeDef, container_id: str, controls: _Controls
) -> tuple[str, int | None, bool]:
    """コンテナが終わるまで待つ。

    待ちの上限は、構成の `ready_timeout_s` (`controls.timeout_s`)。時計と眠りは引数から来る
    ので、試験は実際に眠らない。

    返り値:
        最後に見た状態、終了コード (読めなければ空)、時間切れかどうか。
    """
    deadline = controls.clock() + controls.timeout_s
    while True:
        status, exit_code = _container_state(
            runner, node, container_id, timeout_s=controls.read_timeout_s
        )
        if status not in _UNFINISHED_STATES:
            return status, exit_code, False
        if controls.clock() >= deadline:
            # まだ動いているので、終了コードは無い
            return status, None, True
        controls.sleep(controls.poll_interval_s)


def _read_output(
    runner: RemoteRunner, node: NodeDef, container_id: str, *, timeout_s: float
) -> tuple[str, str, str]:
    """コンテナの出力を読む (行を加工しない。対象は、一覧から来た識別子だけ)。

    返り値:
        標準出力、標準エラー、読めなかったときの理由。
    """
    argv = ("docker", "logs", container_id)
    try:
        result = runner.run(node, argv, timeout_s=timeout_s, mutating=False)
    except RemoteError as exc:
        return "", "", f"コンテナの出力を読めなかった: {exc}"
    if result.exit_code != 0:
        return "", "", f"コンテナの出力を読めなかった: {_shown_failure(result)}"
    return result.stdout, result.stderr, ""


def _roll_back(
    runner: RemoteRunner, node: NodeDef, plan: ContainerPlan
) -> tuple[tuple[str, ...], KeyboardInterrupt | None]:
    """了承済みの計画の巻き戻し (止める → 消す) を、流してよい相手にだけ流す。

    成功でも、失敗でも、時間切れでも、中断でも、必ずここを通る。**相手を決めるのは
    `guards.rollback_commands`** で、流す直前に、自分のラベルで絞った一覧に、その計画の名前の
    行があることを確かめる (`docker stop <名前>` は、その名前のコンテナが誰のものでも止めて
    しまうので、名前だけを頼りにしない。requirements 2.3)。引数の列も、その関数が
    `guards.stop_argv` / `guards.remove_argv` で作るので、了承を得た計画の巻き戻しと完全に
    一致する (`remote.CallGuard` は、完全な一致で見る)。

    **片付けの途中で中断 (Ctrl-C) が来ても、消すところまでは試みる** (コンテナを残さないことを
    優先する)。2 度目の中断は、そのまま伝える。1 度目の中断は、呼ぶ側に返して、呼ぶ側が、
    片付けの結果を知らせてから伝える。

    返り値:
        片付けられなかったこと (流した結果の失敗と、一覧を読めずに流せなかったこと) の一覧と、
        途中で受けた中断。一覧が空なら、この台にコンテナは残っていない。
    """
    commands, unchecked = rollback_commands(
        runner, {node.role: node}, (plan,), timeout_s=READ_TIMEOUT_S
    )
    problems: list[str] = list(unchecked)
    interrupted: KeyboardInterrupt | None = None
    for command in commands:
        what = _CLEANUP_LABELS[command.argv[1]]
        if interrupted is not None and command.argv[1] != _REMOVE_SUBCOMMAND:
            continue  # 中断のあとは、消すことだけを試みる
        try:
            result = runner.run(node, command.argv, timeout_s=CLEANUP_TIMEOUT_S, mutating=True)
        except KeyboardInterrupt as exc:
            if interrupted is not None:
                raise  # 2 度目の中断は、そのまま伝える
            interrupted = exc
            problems.append(f"{what} (中断された。消すところまでは試みる)")
            continue
        except RemoteError as exc:
            problems.append(f"{what} (読み取りが届かなかった): {exc}")
            continue
        if result.exit_code != 0:
            problems.append(f"{what}: {_shown_failure(result)}")
    return tuple(problems), interrupted


def _report(stream: TextIO, node: NodeDef, plan: ContainerPlan, problems: Sequence[str]) -> None:
    """片付けが終わらなかったことを、1 行で知らせる (中断の経路でも、捨てない)。"""
    if not problems:
        return
    stream.write(
        f"警告: {node.role} ({node.ssh_host}) の {plan.container_name} の片付けが、"
        f"すべては終わらなかった: {' / '.join(problems)}\n"
    )
    stream.flush()


def _start_failure_detail(
    node: NodeDef, plan: ContainerPlan, started: CommandResult, problems: Sequence[str]
) -> str:
    """`docker run` が失敗したときの、見せる文。

    標準エラーが名前の衝突らしければ、そう言う (`docker` の文面は版で変わりうるので、
    **文の親切さのためだけに見る**)。片付けに行ったかどうかは、この見分けでは決まらず、
    `guards.rollback_commands` が、自分のラベルで絞った一覧で決めている。
    """
    lowered = started.stderr.lower()
    if all(mark in lowered for mark in _NAME_CONFLICT_MARKS):
        reason = (
            f"コンテナの名前が衝突した ({plan.container_name})。その名前のコンテナは、"
            "自分のラベルで絞った一覧にないので、止めも消しもしない"
        )
    else:
        reason = f"読み取りのコンテナを起こせなかった ({plan.container_name})"
    return (
        f"{node.role} ({node.ssh_host}) で、{reason}:"
        f" {_shown_failure(started)}{_also_failed(problems)}"
    )


def _also_failed(problems: Sequence[str]) -> str:
    """片付けの失敗を、ほかの誤りの文に添える。"""
    return "" if not problems else " 片付けも失敗した: " + " / ".join(problems)


def _read_so_far(reading: LicenseReading) -> str:
    """誤りの文に、読めた分を添える (どこまで読めたかを、計測者が見られるように)。"""
    if not (reading.stdout.strip() or reading.stderr.strip()):
        return ""
    return f"\n読めた分:\n{reading.text}"


def _watch_and_read(
    runner: RemoteRunner, node: NodeDef, plan: ContainerPlan, controls: _Controls
) -> tuple[LicenseReading, str | None]:
    """起こしたコンテナの終わりを待ち、出力を読む (止めて消すのは、呼ぶ側)。

    **何も読めなかったことを、正常な結果として返さない** (module の docstring の決めごとの
    8)。コンテナを見つけられない (一覧が読めない、一覧にその名前がない) ことと、
    `docker logs` が 0 以外で終わることは、読み取りの失敗として、理由の文を返す。
    コンテナの中の `cat` が 0 以外で終わったこと (読むファイルが無かった) は、読み取りの
    失敗ではないので、結果として返す (出力と終了コードを、計測者が読む)。

    返り値:
        読み取りの結果と、読み取りの失敗の理由 (成功なら空)。
    """
    container, reason = _own_container(
        runner, node, plan.container_name, timeout_s=controls.read_timeout_s
    )
    if container is None:
        return (
            LicenseReading(node=node.role, container_name=plan.container_name, detail=reason),
            f"読み取りのコンテナを見つけられなかった ({plan.container_name}): {reason}",
        )
    status, exit_code, timed_out = _wait_for_exit(runner, node, container.id, controls)
    stdout, stderr, problem = _read_output(
        runner, node, container.id, timeout_s=controls.read_timeout_s
    )
    reading = LicenseReading(
        node=node.role,
        container_name=plan.container_name,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        detail=problem or f"コンテナの状態: {status}",
    )
    if timed_out:
        failure = (
            f"読み取りのコンテナが、{controls.timeout_s:.0f} 秒のうちに終わらなかった"
            f" ({plan.container_name})。出力を読んでから、止めて消した"
        )
    elif problem:
        failure = f"{problem} ({plan.container_name})"
    else:
        failure = ""
    return reading, failure or None


def _read_one(
    runner: RemoteRunner, node: NodeDef, plan: ContainerPlan, controls: _Controls
) -> LicenseReading:
    """1 台ぶんの「起こす → 待つ → 読む → 止める → 消す」。

    起こすのが失敗しても、**起こす呼び出しそのものが届かなくても** (ssh の時間切れ、切断)、
    中断されても、同じ 1 つの片付けの道を通る。遠隔の `docker run -d` は、ssh が切れたあとも
    完了しうるので、`docker run` の呼び出しも `try` の中に置く。

    **止めて消す相手は、つねに `guards.rollback_commands` が、自分のラベルで絞った一覧で
    決める**ので、コンテナができていなければ、`docker stop` も `docker rm` も出ない。名前の
    衝突 (その名前が、よそのコンテナのもの) のときも同じである。`docker run` の標準エラーの
    文面は、誤りの文を親切にするためだけに見る。
    """
    started: CommandResult | None = None
    try:
        started = runner.run(node, plan.argv, timeout_s=controls.start_timeout_s, mutating=True)
        reading, failure = _watch_and_read(runner, node, plan, controls)
    except RemoteError as exc:
        # 了承のあとに届かなかったことは、実行しての失敗 (終了コード 2) として包む
        problems, _ = _roll_back(runner, node, plan)
        step = "起こす" if started is None else "読み取る"
        raise ImageError(
            f"{node.role} ({node.ssh_host}) で、読み取りのコンテナを{step}途中に、"
            f"読み取りが届かなかった ({plan.container_name}): {exc}{_also_failed(problems)}"
        ) from exc
    except BaseException:
        # 中断 (Ctrl-C) と、思わぬ誤りのときも、コンテナを残さない
        problems, _ = _roll_back(runner, node, plan)
        _report(controls.report, node, plan, problems)
        raise
    problems, interrupted = _roll_back(runner, node, plan)
    if interrupted is not None:
        _report(controls.report, node, plan, problems)
        raise interrupted

    if started.exit_code != 0:
        raise ImageError(_start_failure_detail(node, plan, started, problems))
    if failure is not None:
        raise ImageError(
            f"{node.role} ({node.ssh_host}) で、{failure}"
            f"{_also_failed(problems)}{_read_so_far(reading)}"
        )
    if problems:
        raise ImageError(
            f"読み取りのコンテナを片付けられなかった ({plan.container_name}):"
            f" {' / '.join(problems)}"
        )
    return reading


def read_image_licenses(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    started_at: datetime,
    *,
    confirmer: Confirmer,
    poll_interval_s: float = POLL_INTERVAL_S,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    report: TextIO | None = None,
    start_timeout_s: float = START_TIMEOUT_S,
    read_timeout_s: float = READ_TIMEOUT_S,
) -> LicensesOutcome:
    """イメージの中の、ライセンスの表記を読んで返す (`serve image-licenses`)。

    `kind = "inspect"` の構成を、ほかの種類と同じ仕組みで起こす: `plan.build_plans` で計画を
    作り (名前とラベルが付き、`-d` で切り離す)、関門を流し、了承を得て、起こし、終了を待ち、
    `docker logs` で出力を読み、**必ず**止めて消す。読むファイルの道筋は、構成の位置の引数
    から来る (この module は、候補のファイルの名前を持たない)。

    **何も読めなかったときは、`status = "read"` で返らない** (module の docstring の決めごとの
    8)。コンテナを見つけられない、出力を読めない、時間切れは、どれも `ImageError` である。

    引数:
        runner: 遠隔の実行役。
        config: `kind = "inspect"` の構成。
        nodes: 役割ごとのノードの定義。
        started_at: 起こす時刻 (ラベルに書く。時差の付いた日時)。
        confirmer: 計画を見せて、了承を得る口。
        poll_interval_s: 終わりを待つ間隔。
        sleep: 眠る口 (試験は、実際に眠らないものを渡す)。既定は `time.sleep`。
        clock: 時計 (単調増加の秒)。既定は `time.monotonic`。
        report: 中断の経路で、片付けが終わらなかったことを知らせる先。既定は `sys.stderr`
            (標準出力は、読み取った表記を見せるのに使うので、混ぜない)。
        start_timeout_s: `docker run -d` の時間切れ。
        read_timeout_s: 関門と、状態と出力の読み取りの時間切れ。

    返り値:
        読み取りの結果か、関門が断ったこと。`text` が、そのまま見せられる文字列になる。

    例外:
        config.ConfigError: `kind` が `inspect` でない構成を渡したとき (終了コード 1)。
        guards.ApprovalError: 計測者が了承しなかった (終了コード 1)。
        ImageError: 起動の失敗、名前の衝突、コンテナを見つけられない、出力を読めない、
            時間切れ、片付けの失敗、**了承のあとに読み取りが届かなかったこと** (どれも、
            実行しての失敗で終了コード 2)。
        KeyboardInterrupt: 中断 (終了コード 130)。コンテナは、片付けてから伝える。
        ValueError: 構成が使う役割のノードの定義がないとき。

    **`RemoteError` は、この関数からは出ない**。了承の前 (関門) の「入れない」は
    `run_gates` が `GateResult` (`passed=False`) に変えるので `status = "refused"` になり、
    5.1 は終了コード 1 に写す。了承のあと (`docker run`) に届かなかったことは、ここで
    `ImageError` に包むので、5.1 は終了コード 2 に写す。
    """
    if config.kind != _KIND_INSPECT:
        raise ConfigError(
            f"ライセンスの表記の読み取りに使えるのは、kind = '{_KIND_INSPECT}' の構成だけである"
            f" (構成 '{config.name}' の kind は '{config.kind}')"
        )
    plans = build_plans(config, nodes, started_at)
    gates = run_gates(runner, config, nodes, plans, timeout_s=read_timeout_s)
    refused = _refused(gates)
    if refused:
        return LicensesOutcome(
            status="refused",
            config_name=config.name,
            gates=gates,
            detail=_refusal_detail(refused),
        )

    approved = build_approved_plan(plans)
    request_approval(confirmer, runner, approved, nodes)
    controls = _Controls(
        sleep=time.sleep if sleep is None else sleep,
        clock=time.monotonic if clock is None else clock,
        report=sys.stderr if report is None else report,
        poll_interval_s=poll_interval_s,
        timeout_s=float(config.ready_timeout_s),
        start_timeout_s=start_timeout_s,
        read_timeout_s=read_timeout_s,
    )
    readings = tuple(
        _read_one(runner, _node_of(nodes, config, plan.node), plan, controls) for plan in plans
    )
    return LicensesOutcome(
        status="read",
        config_name=config.name,
        gates=gates,
        readings=readings,
        detail=f"{len(readings)} 台で読み取り、コンテナを止めて消した",
    )
