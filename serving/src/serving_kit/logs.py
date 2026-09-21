"""記録の回収と、`serving/var/` の下の置き場所の決まり (design.md 「記録の置き場所」)。

**写すもの** (tasks.md 2.4、requirements 1.9、7.8、10.5):

1. 自分のコンテナの記録 (`docker logs --timestamps <識別子>` の全量。標準出力と標準エラーの
   両方)
2. 通信の記録 (Spark の `<remote_root>/logs/` の下。まだ何もないこともある)
3. 起動の記録 (`<remote_root>/state/<構成>.launch.json`)

2 台 (構成が使うノードのすべて) から、Mac の `var_root` の下の、日時 (UTC)・コマンド・構成の
名前が付いた場所に写す。置き場所の決定 (`var_dir`) は、入出力のない純粋な関数にする (時刻は
引数で受け、`datetime.now` はここでは呼ばない)。

守る決まり:

- **コンテナを対象にする操作の不変条件** (requirements 2.3、2.4): `docker logs` を向ける相手は、
  `guards.list_own_containers` (自分のラベルで絞り、受け取った行の所有のラベルも確かめる、
  ただ 1 つの入口) が返した行のうち、**名前が `ContainerPlan.container_name` と一致する行の
  ID** だけである。この module は、`serve logs` や、起動の失敗のときの末尾の表示のように、
  **了承済みの計画を持たずに** 呼ばれることがあるので、design の不変条件のうち (a) (ラベルで
  絞った一覧から来た識別子) だけを使う ((b) の「了承済みの計画が自分で起こす名前」は、了承の
  前に計画が確定している場合の話で、ここには当たらない。その名前の、ラベルの無い別の
  コンテナが、たまたまあっても読んでしまわないようにする)。一覧にその名前の行が無い・
  一覧が読めない (所有のラベルのない行が紛れている、ssh で入れない) ときは、`docker logs` を
  1 つも出さずに、欠けたものとして扱う。名前を文字列で自由に受け取って `docker logs` に渡す
  公開の口は作らない
- **状態を変えない**: `collect_logs` と `tail_logs` は、`docker ps` / `docker logs` (読み取り) と
  `RemoteRunner.pull` (Mac に写すだけで、Spark の状態を変えない) しか呼ばない。
  `mutating=True` の呼び出しも、`docker stop` / `rm` も出さない。了承を得た計画は要らない
- **片方の台が失敗しても続ける**: ssh で入れない、コンテナがもう無い、通信の記録がまだ無い、
  起動の記録がまだ無い、のどれで失敗しても、例外を外に出さず、もう片方の台のぶんは写す。
  写せなかったものは、置き場所の中の `collect.json` (`MISSING_FILE_NAME`) に、いつでも書く
  (何も欠けていなければ、`missing` が空の配列になる)。回収の結果の型は専用に持たず (design の
  口が `Path` を返すだけなので)、呼ぶ側は、この `collect.json` を読めば何が写せたかが分かる
  (design.md「記録の置き場所」の文には出てこないが、tasks.md 2.4 の完了の状態が要求する
  「写せなかったことが分かる」を満たすための、この module の決めごと)
- **加工しない** (requirements 10.5): `CommandResult.stdout` / `.stderr` は、そのままファイルに
  書く。送った内容や応答の本文を、ここで読んだり、抜き出して別の場所に写したりしない

依存の向きにより、この module が読み込む `serving_kit` は `types`、`remote`、`guards` だけで
ある (design.md の依存の向き `… → guards → image, weights, logs → …` のとおり。`lifecycle`
以降は読み込まない)。

design の文からの、意図した決めごと (design が明示しない細部を、ここで決めて残す):

1. **置き場所の名前の時刻の形は `%Y%m%dT%H%M%SZ`** (コロンを含まない。ファイル名として安全な
   形にする。`bench/src/bench_harness/store/rawstore.py` の `new_run_id` と同じ形にした。
   ラベルの時刻 (`plan._STARTED_AT_FORMAT`) はコロンを含むので、ここでは使わない)
2. **コマンドの名前と構成の名前は、英数字とハイフンだけ、先頭は英数字にする**
   (`plan._CONFIG_NAME_RE` と同じ絞り方。`/` や `..` を含む名前は、置き場所が `var_root` の
   外に出る前に、ここで断る)
3. **末尾だけを読む口の既定の行数は 80** (design.md の Requirements Traceability、要件 1.5 の
   行: `docker logs --tail 80`)
4. **標準出力と標準エラーは別のファイルに書く** (`container.stdout.log` /
   `container.stderr.log`)。どちらのファイルも、送られてきた文字列をそのまま書く。1 つに
   連ねて見出しを挟むと、それも「加工」になるため
5. **通信の記録は `<remote_root>/logs/` をまるごと `pull` する**。`remote._pull_source` は
   `PurePosixPath` で道筋を作るため、末尾の `/` が必ず失われ、`rsync` は常に元の名前を
   1 段のディレクトリとして残す。したがって回収の結果は `<役割>/logs/…` に 1 段ネストする
   (中のファイルの名前は、通信の記録を書く側 (task 4.4) が決める。この module は中身を読まない)
6. **`docker logs` の対象は、名前ではなく ID で決める**。`guards.list_own_containers` を
   台ごとに 1 回呼び、返ってきた行のうち、名前が `ContainerPlan.container_name` と一致する
   ものだけを対象にする (一致する行がない・一覧そのものが読めないときは、`docker logs` を
   出さない)。**名前をそのまま渡さない理由**: この module は、了承済みの計画を持たずに呼ばれる
   (`serve logs`、失敗のときの末尾の表示)。そのとき、同じ名前の、ラベルの無い別のコンテナが
   たまたまあると (design が「名前の衝突」として扱う状況)、名前だけを頼りにしては、よその
   コンテナの記録を読んでしまう (requirements 2.4 違反)。一覧に絞り込んでから ID で引くことで、
   その経路を構造でなくす
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from serving_kit.guards import OwnContainer, list_own_containers
from serving_kit.remote import RemoteError, RemoteRunner
from serving_kit.types import ContainerPlan, NodeDef, NodeRole

__all__ = [
    "COMM_LOG_REMOTE_SUBDIR",
    "DEFAULT_TAIL_LINES",
    "ITEM_COMM_LOG",
    "ITEM_CONTAINER_LOG",
    "ITEM_LAUNCH_RECORD",
    "LOG_READ_TIMEOUT_S",
    "MISSING_FILE_NAME",
    "TAIL_TIMEOUT_S",
    "collect_logs",
    "launch_record_remote_path",
    "tail_logs",
    "var_dir",
]


# --- 決まった値 -----------------------------------------------------------

_TIMESTAMP_FORMAT: Final[str] = "%Y%m%dT%H%M%SZ"
"""置き場所の名前に使う UTC の時刻の形 (module の docstring の決めごとの 1)。"""

_COMPONENT_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")
"""コマンドの名前・構成の名前に使える文字 (module の docstring の決めごとの 2)。"""

DEFAULT_TAIL_LINES: Final[int] = 80
"""末尾だけを読む口の既定の行数 (design.md Requirements Traceability、要件 1.5 の行)。"""

LOG_READ_TIMEOUT_S: Final[float] = 300.0
"""`docker logs` の全量を読む時間切れ。起動の記録は長くなりうるので、guards の
`READ_TIMEOUT_S` (30 秒) より長くする。"""

TAIL_TIMEOUT_S: Final[float] = 30.0
"""`--tail` で絞った読み取りの時間切れ (guards の `READ_TIMEOUT_S` と同じ、すぐ返る読み取り)。"""

COMM_LOG_REMOTE_SUBDIR: Final[str] = "logs"
"""通信の記録の置き場所 (design.md File Structure Plan の Spark の上の `logs/`。
`remote_root` からの相対の道筋)。"""

MISSING_FILE_NAME: Final[str] = "collect.json"
"""写せなかったものの一覧を書くファイルの名前 (module の docstring の決めごと)。"""

ITEM_CONTAINER_LOG: Final[str] = "container_log"
ITEM_COMM_LOG: Final[str] = "comm_log"
ITEM_LAUNCH_RECORD: Final[str] = "launch_record"
"""`collect.json` の `missing` の 1 件が持つ `item` の値。"""


@dataclass(frozen=True)
class _Missing:
    """写せなかった 1 件。"""

    node: NodeRole
    item: str
    detail: str


# --- 置き場所の決定 (入出力のない関数) -------------------------------------


def _check_component(value: str, label: str) -> None:
    """置き場所の名前の一部にする文字列を、安全な文字だけに絞る。"""
    if not _COMPONENT_RE.fullmatch(value):
        raise ValueError(
            f"{label}は、英数字とハイフンだけ、先頭は英数字にする"
            f" (置き場所の名前の一部になるので、/ や .. を含められない): {value!r}"
        )


def var_dir(var_root: Path, started_at: datetime, command: str, config_name: str) -> Path:
    """記録の置き場所を決める (design.md「記録の置き場所」)。

    `<var_root>/<UTC の日時>-<コマンド>-<構成>/` を返すだけの、入出力のない純粋な関数である
    (ディレクトリを作らない。作るのは `collect_logs`)。`command` と `config_name` は、
    英数字とハイフンだけに絞る (`/` や `..` を含む名前は、`var_root` の外を指しうるので断る)。

    引数:
        var_root: 記録の置き場所の根 (`serving/var/`)。
        started_at: 時差の付いた日時。素の日時 (naive) は、UTC か不明かが決まらないので断る。
        command: この記録を作った `serve` のサブコマンドの名前 (例: `logs`、`start`)。
        config_name: 選んだ構成の名前。

    例外:
        ValueError: `command` / `config_name` に使えない文字があるとき、`started_at` が
            時差を持たないとき。
    """
    _check_component(command, "コマンドの名前")
    _check_component(config_name, "構成の名前")
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise ValueError(
            "started_at は、時差の付いた日時にする (素の日時は、UTC か不明かが決まらない)"
        )
    stamp = started_at.astimezone(UTC).strftime(_TIMESTAMP_FORMAT)
    return var_root / f"{stamp}-{command}-{config_name}"


def launch_record_remote_path(config_name: str) -> str:
    """起動の記録の、`remote_root` からの相対の道筋 (design.md 「lifecycle」の State
    Management: `state/<構成>.launch.json`)。"""
    _check_component(config_name, "構成の名前")
    return f"state/{config_name}.launch.json"


# --- 1 台ぶんの回収 ---------------------------------------------------------


def _resolve_own_container(
    runner: RemoteRunner, node: NodeDef, container_name: str, *, timeout_s: float
) -> tuple[OwnContainer | None, str | None]:
    """`docker logs` を向ける相手を、自分のラベルで絞った一覧から ID で決める (module の
    docstring の不変条件。design の (a) 「ラベルで絞った一覧から来た識別子」だけを使う)。

    `guards.list_own_containers` (自分のラベルで絞り、受け取った行の所有のラベルも確かめる、
    ただ 1 つの入口) を呼び、名前が `container_name` と一致する行を探す。一覧が読めない
    (ssh で入れない、所有のラベルのない行が紛れている) ときも、一致する行がないときも、
    `docker logs` を向けずに、理由の文字列を返す (呼ぶ側が `missing` / 末尾の文言に使う)。

    返り値:
        `(見つかった行, None)` か `(None, 理由の文)` のどちらか一方。
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
    return found, None


def _collect_container_log(
    runner: RemoteRunner,
    node: NodeDef,
    plan: ContainerPlan,
    node_dir: Path,
    missing: list[_Missing],
    *,
    timeout_s: float,
) -> None:
    """`docker logs --timestamps <ID>` の全量を、2 つのファイルに書く。

    対象は、自分のラベルで絞った一覧の中で、名前が `plan.container_name` と一致する行の ID
    だけ (module の docstring の不変条件。`_resolve_own_container` が決める)。標準出力と
    標準エラーは、別のファイルにそのまま書く (加工しない)。
    """
    container, reason = _resolve_own_container(
        runner, node, plan.container_name, timeout_s=timeout_s
    )
    if container is None:
        missing.append(_Missing(node.role, ITEM_CONTAINER_LOG, reason or "コンテナが見つからない"))
        return
    argv = ("docker", "logs", "--timestamps", container.id)
    try:
        result = runner.run(node, argv, timeout_s=timeout_s, mutating=False)
    except RemoteError as exc:
        missing.append(_Missing(node.role, ITEM_CONTAINER_LOG, f"読み取りが届かなかった: {exc}"))
        return
    if result.exit_code != 0:
        missing.append(
            _Missing(
                node.role,
                ITEM_CONTAINER_LOG,
                f"コンテナの記録を読めなかった ({plan.container_name}):"
                f" {result.stderr.strip() or result.stdout.strip()}",
            )
        )
        return
    (node_dir / "container.stdout.log").write_text(result.stdout, encoding="utf-8")
    (node_dir / "container.stderr.log").write_text(result.stderr, encoding="utf-8")


def _collect_comm_log(
    runner: RemoteRunner, node: NodeDef, node_dir: Path, missing: list[_Missing]
) -> None:
    """`<remote_root>/logs/` をまるごと `node_dir/logs/` に写す (まだ無ければ欠けたと記録する)。"""
    try:
        result = runner.pull(node, COMM_LOG_REMOTE_SUBDIR, node_dir)
    except RemoteError as exc:
        missing.append(_Missing(node.role, ITEM_COMM_LOG, f"読み取りが届かなかった: {exc}"))
        return
    if result.exit_code != 0:
        missing.append(
            _Missing(
                node.role,
                ITEM_COMM_LOG,
                f"通信の記録がまだない ({node.remote_root}/{COMM_LOG_REMOTE_SUBDIR}):"
                f" {result.stderr.strip() or result.stdout.strip()}",
            )
        )


def _collect_launch_record(
    runner: RemoteRunner, node: NodeDef, config_name: str, node_dir: Path, missing: list[_Missing]
) -> None:
    """`<remote_root>/state/<構成>.launch.json` を `node_dir/` に写す。"""
    remote_path = launch_record_remote_path(config_name)
    try:
        result = runner.pull(node, remote_path, node_dir)
    except RemoteError as exc:
        missing.append(_Missing(node.role, ITEM_LAUNCH_RECORD, f"読み取りが届かなかった: {exc}"))
        return
    if result.exit_code != 0:
        missing.append(
            _Missing(
                node.role,
                ITEM_LAUNCH_RECORD,
                f"起動の記録がまだない ({node.remote_root}/{remote_path}):"
                f" {result.stderr.strip() or result.stdout.strip()}",
            )
        )


def _write_missing(
    destination: Path,
    missing: Sequence[_Missing],
    *,
    command: str,
    config_name: str,
    started_at: datetime,
) -> None:
    """写せなかったものの一覧を `collect.json` に書く (何もなければ、空の配列を書く)。"""
    payload = {
        "command": command,
        "config_name": config_name,
        "started_at": started_at.astimezone(UTC).isoformat(),
        "missing": [
            {"node": item.node, "item": item.item, "detail": item.detail} for item in missing
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (destination / MISSING_FILE_NAME).write_text(text, encoding="utf-8")


# --- 公開の口 ---------------------------------------------------------------


def collect_logs(
    runner: RemoteRunner,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    *,
    var_root: Path,
    started_at: datetime,
    command: str,
    config_name: str,
    timeout_s: float = LOG_READ_TIMEOUT_S,
) -> Path:
    """`plans` が指す 2 台ぶんの記録を集め、`var_dir` が決めた場所に写す (requirements 1.9)。

    写すもの (module の docstring): 自分のコンテナの記録、通信の記録、起動の記録。
    `<var_root>/<UTC>-<command>-<config_name>/<role>/` の下に、`role` ごとに書く。

    片方の台で失敗しても (ssh で入れない、コンテナがもう無い、記録がまだ無い)、例外を外に
    出さず、もう片方の台のぶんは写す。写せなかったものは、置き場所の直下の `collect.json`
    (`MISSING_FILE_NAME`) に、いつでも書く (欠けが無ければ `missing` は空の配列)。

    `docker logs` の対象は、台ごとに `guards.list_own_containers` で取った、自分のラベルで
    絞った一覧の中で、名前が `ContainerPlan.container_name` と一致する行の ID だけである
    (module の docstring の不変条件)。一致する行がない・一覧が読めない台では、`docker logs`
    を出さずに、`ITEM_CONTAINER_LOG` として `missing` に書く。

    引数:
        runner: 遠隔の実行役。`docker ps` / `docker logs` の読み取りと `pull` しか呼ばない。
            了承を得た計画は要らない (状態を変えないため)。
        nodes: 役割ごとのノードの定義 (`plans` が使う役割の分を持つこと)。
        plans: 記録を集める対象のコンテナの計画 (`plan.build_plans` が組み立てたもの)。
        var_root: 記録の置き場所の根 (`SshRunner` を作ったときと同じ値にする。でなければ、
            `remote.CallGuard` に宛先の外として断られる)。
        started_at: 置き場所の名前に使う、時差の付いた日時。
        command: この回収を行った `serve` のサブコマンドの名前。
        config_name: 選んだ構成の名前。
        timeout_s: 1 台ぶんの `docker logs` の時間切れ。

    返り値:
        記録を書いた置き場所 (`var_dir` の値。`plans` が空でも作る)。

    例外:
        ValueError: `plans` が使う役割のノードの定義が `nodes` にないとき、`command` /
            `config_name` に使えない文字があるとき。
    """
    destination = var_dir(var_root, started_at, command, config_name)
    destination.mkdir(parents=True, exist_ok=True)
    missing: list[_Missing] = []
    for plan in plans:
        node = nodes.get(plan.node)
        if node is None:
            raise ValueError(f"構成 '{config_name}' の {plan.node} のノードの定義がない")
        node_dir = destination / plan.node
        node_dir.mkdir(parents=True, exist_ok=True)
        _collect_container_log(runner, node, plan, node_dir, missing, timeout_s=timeout_s)
        _collect_comm_log(runner, node, node_dir, missing)
        _collect_launch_record(runner, node, config_name, node_dir, missing)
    _write_missing(
        destination, missing, command=command, config_name=config_name, started_at=started_at
    )
    return destination


def tail_logs(
    runner: RemoteRunner,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    *,
    lines: int = DEFAULT_TAIL_LINES,
    timeout_s: float = TAIL_TIMEOUT_S,
) -> dict[NodeRole, str]:
    """記録の末尾だけを読む (起動の失敗のときに使う。design.md System Flows の起動)。

    台ごとに、まず `guards.list_own_containers` で自分のラベルで絞った一覧を取り、名前が
    `plan.container_name` と一致する行の ID に、`docker logs --timestamps --tail <lines> <ID>`
    を向ける (module の docstring の不変条件)。標準出力と標準エラーを連ねて返す (末尾は画面に
    見せるためだけで、ファイルには書かない)。一覧が読めない・一致する行がない・読み取りが
    届かないときも、例外を出さず、その旨の文字列を返す (呼ぶ側は、つねに `plans` にある役割
    ぶんの結果を受け取れる)。

    引数:
        runner: 遠隔の実行役。`docker ps` / `docker logs` の読み取りしか呼ばない。
        nodes: 役割ごとのノードの定義。
        plans: 記録を読む対象のコンテナの計画。
        lines: `--tail` に渡す行数。
        timeout_s: 1 台ぶんの時間切れ (`docker ps` と `docker logs` の両方に使う)。

    返り値:
        役割ごとの、記録の末尾 (`StartOutcome.log_tails` に渡せる形)。

    例外:
        ValueError: `plans` が使う役割のノードの定義が `nodes` にないとき。
    """
    tails: dict[NodeRole, str] = {}
    for plan in plans:
        node = nodes.get(plan.node)
        if node is None:
            raise ValueError(f"{plan.node} のノードの定義がない")
        container, reason = _resolve_own_container(
            runner, node, plan.container_name, timeout_s=timeout_s
        )
        if container is None:
            tails[plan.node] = f"(コンテナの記録を読めない: {reason})"
            continue
        argv = ("docker", "logs", "--timestamps", "--tail", str(lines), container.id)
        try:
            result = runner.run(node, argv, timeout_s=timeout_s, mutating=False)
        except RemoteError as exc:
            tails[plan.node] = f"(読み取りが届かなかった: {exc})"
            continue
        if result.exit_code != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            tails[plan.node] = f"(コンテナの記録を読めなかった: {detail})"
            continue
        tails[plan.node] = result.stdout + result.stderr
    return tails
