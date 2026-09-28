"""Mac 側の自動起動の指定・解除・状態確認 (issue #88 P6。`serve autostart set|clear|status`)。

Spark 上の見張り (`ops/vllm-autostart/vllm_autostart.py`) が読む `state/autostart.json` を
配り、`state/autostart.status.json` を読んで示す。この module は、コンテナを起こす計画
(`lifecycle.start`) には触れない (`lifecycle` を import しない)。指定は、すでに了承された
`serve start` の固定 `argv` (`launch.json`) が、Mac の計画と一致することを確かめてから
配るだけである (要件 21: 任意のコマンドを流せる口を作らない)。

依存の向きは `types → config → remote → plan, guards, logs → autostart` (`lifecycle` は
含まない)。
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from serving_kit import guards, logs
from serving_kit import plan as plan_mod
from serving_kit.config import ConfigError
from serving_kit.remote import RemoteRunner
from serving_kit.types import (
    AnyWeightsManifest,
    AutostartDesignation,
    AutostartStatus,
    AutostartWeightsExpectation,
    ConfigDef,
    DerivedWeightsRef,
    LaunchRecord,
    NodeDef,
    NodeRole,
    PlannedPush,
)

__all__ = [
    "AutostartError",
    "AutostartOutcome",
    "AutostartShown",
    "NodeAutostartView",
    "clear_autostart",
    "read_autostart",
    "read_launch_record",
    "set_autostart",
]

DESIGNATION_FILE = "autostart.json"
"""`state/` の下の、指定のファイルの名前。"""

STATUS_FILE = "autostart.status.json"
"""`state/` の下の、見張りが書く状態のファイルの名前。"""

RECORD_SUBDIR = "state"
"""指定を配る宛先 (`remote._PUSH_SUBDIRS` にある置き場所)。"""

_ROLE_ORDER: tuple[NodeRole, ...] = ("head", "worker")

_READ_TIMEOUT_S: float = 30.0

_RECORD_DIRNAME = "autostart"
"""指定を配る元を作る、この module だけが使う場所 (決めごとは `lifecycle.py` と同じ)。"""


class AutostartError(Exception):
    """自動起動の指定・解除を実行して失敗した (配れなかった。終了コード 2)。"""


@dataclass(frozen=True)
class AutostartOutcome:
    """`set`/`clear` の結末。

    `refused` は、一致を確かめられなかった・`kind` が `serve` でないなど、Spark に触る前に
    分かる前提の不足 (終了コード 1)。`designated`/`cleared` は、両台に配り終えたことを表す。
    """

    status: Literal["designated", "cleared", "refused"]
    detail: str
    roles: tuple[NodeRole, ...] = ()


@dataclass(frozen=True)
class NodeAutostartView:
    """1 台ぶんの、指定と状態の読み取り結果。"""

    role: NodeRole
    designation: AutostartDesignation | None
    status: AutostartStatus | None

    @property
    def readable(self) -> bool:
        """指定・状態のどちらも読めたかどうか。"""
        return self.designation is not None and self.status is not None


@dataclass(frozen=True)
class AutostartShown:
    """`status` が示す、両台ぶんの読み取り結果。"""

    nodes: tuple[NodeAutostartView, ...]

    @property
    def all_readable(self) -> bool:
        return all(node.readable for node in self.nodes)


# --- 起動の記録を読む --------------------------------------------------------


def read_launch_record(
    runner: RemoteRunner, node: NodeDef, config_name: str
) -> LaunchRecord | None:
    """`state/<構成>.launch.json` を `cat` で読む。読めない・壊れていれば `None`。"""
    path = logs.launch_record_remote_path(config_name)
    result = runner.run(
        node, ("cat", f"{node.remote_root}/{path}"), timeout_s=_READ_TIMEOUT_S, mutating=False
    )
    if result.exit_code != 0:
        return None
    try:
        return LaunchRecord.model_validate_json(result.stdout)
    except ValueError:
        return None


# --- 指定の組み立て ----------------------------------------------------------


def _weights_expectation(
    config: ConfigDef, node: NodeDef, manifest: AnyWeightsManifest | None
) -> AutostartWeightsExpectation | None:
    """重みを持つ構成では、見張りが照合の記録と突き合わせる期待値を組む
    (D5 の `weights_verified`)。
    """
    if config.weights is None:
        return None
    if manifest is None:
        raise ConfigError(
            f"構成 '{config.name}' は重みを使うのに、マニフェストが渡されていない"
            " (自動起動の指定には、照合の期待値が要る)"
        )
    path = guards.weights_record_path(node, config.weights, "all")
    if isinstance(config.weights, DerivedWeightsRef):
        fields = {"manifest_sha256": manifest.content_sha256}
    else:
        fields = {"repo": config.weights.repo, "revision": config.weights.revision}
    return AutostartWeightsExpectation(
        path=path,
        scope="all",
        file_count=len(manifest.files),
        total_bytes=manifest.total_bytes,
        fields=fields,
    )


def build_designation(
    config: ConfigDef,
    node: NodeDef,
    config_sha256: str,
    manifest: AnyWeightsManifest | None,
    now: datetime,
    *,
    repo_commit: str,
    repo_dirty: bool,
) -> AutostartDesignation:
    """1 台ぶんの自動起動の指定を組む (`serve autostart set` が使う)。

    `kind` が `serve` かどうかの確認は、呼び出し元の `set_autostart` が Spark に触る前に
    済ませている (この関数では重ねない。重複確認をなくすため)。
    """
    return AutostartDesignation(
        role=node.role,
        config=config.name,
        config_sha256=config_sha256,
        ready_timeout_s=config.ready_timeout_s,
        ports=guards.config_ports(config, node.role),
        weights_record=_weights_expectation(config, node, manifest),
        designated_at=now,
        repo_commit=repo_commit,
        repo_dirty=repo_dirty,
    )


def _cleared_designation(
    role: NodeRole, now: datetime, *, repo_commit: str, repo_dirty: bool
) -> AutostartDesignation:
    """`serve autostart clear` が配る、`config` が `None` の指定。"""
    return AutostartDesignation(
        role=role,
        config=None,
        config_sha256=None,
        ready_timeout_s=None,
        ports=(),
        weights_record=None,
        designated_at=now,
        repo_commit=repo_commit,
        repo_dirty=repo_dirty,
    )


# --- 配布 --------------------------------------------------------------------


def _designation_bytes(designation: AutostartDesignation) -> bytes:
    """指定を、同じ入力なら同じバイト列になる形で書き出す
    (`lifecycle._record_bytes` と同じ流儀)。
    """
    text = json.dumps(
        designation.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True
    )
    return (text + "\n").encode("utf-8")


def _check_record_dir(record_dir: Path) -> None:
    if not record_dir.is_absolute():
        raise ValueError(
            f"record_dir は、絶対の道筋で渡す (配る元を配る前に空にするため): '{record_dir}'"
        )


def _record_source_dir(record_dir: Path, role: NodeRole) -> Path:
    """指定を配る元 (`lifecycle._record_source_dir` と同じ考え方。この module だけが使う場所)。"""
    return record_dir / _RECORD_DIRNAME / role


def designation_pushes(
    designations: Mapping[NodeRole, AutostartDesignation], record_dir: Path
) -> tuple[PlannedPush, ...]:
    """指定を、配る元 (`<record_dir>/autostart/<役割>/`) に書き、配布の計画にする。"""
    _check_record_dir(record_dir)
    pushes: list[PlannedPush] = []
    for role, designation in designations.items():
        local_dir = _record_source_dir(record_dir, role)
        if local_dir.is_symlink():
            raise AutostartError(
                f"自動起動の指定を配る元 ({local_dir}) がシンボリックリンクになっている"
                " (この場所は、この道具だけが使う。リンクを外してから、やり直す)"
            )
        shutil.rmtree(local_dir, ignore_errors=True)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / DESIGNATION_FILE).write_bytes(_designation_bytes(designation))
        pushes.append(
            PlannedPush(
                node=role,
                local_dir=local_dir,
                remote_subdir=RECORD_SUBDIR,
                delete=False,
                purpose=f"{role} に、自動起動の指定を置く",
            )
        )
    return tuple(pushes)


def push_designations(
    runner: RemoteRunner, nodes: Mapping[NodeRole, NodeDef], pushes: Sequence[PlannedPush]
) -> None:
    """了承済みの配布の計画を、そのまま流す (0 以外で終わったら `AutostartError`)。"""
    for step in pushes:
        node = nodes[step.node]
        result = runner.push(node, step.local_dir, step.remote_subdir, delete=step.delete)
        if result.exit_code != 0:
            raise AutostartError(
                f"{step.node} ({node.ssh_host}) に、自動起動の指定を置けなかった"
                f" (終了コード {result.exit_code}): {result.stderr.strip()}"
            )


# --- serve autostart set ------------------------------------------------------


def set_autostart(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    manifest: AnyWeightsManifest | None,
    record_dir: Path,
    now: datetime,
    *,
    repo_commit: str,
    repo_dirty: bool,
    confirmer: guards.Confirmer,
) -> AutostartOutcome:
    """両台の `launch.json` を読み、Mac の計画と `config_sha256` が一致するときだけ配る。

    一致を確かめられない (読めない・違う) ときは、1 度も配らず `refused` で返す
    (要件 21: 了承済みの構成の固定 argv だけを使う。一致を確かめずには配らない)。
    """
    if config.kind != "serve":
        raise ConfigError(
            f"構成 '{config.name}' は kind が '{config.kind}' なので、自動起動を指定できない"
            " (serve の構成だけを指定できる)"
        )
    plans = plan_mod.build_plans(config, nodes, now)
    plans_by_role = {plan.node: plan for plan in plans}

    problems: list[str] = []
    for role in config.nodes:
        node = nodes[role]
        record = read_launch_record(runner, node, config.name)
        if record is None:
            problems.append(f"{role}: 起動の記録 (launch.json) を読めない")
            continue
        expected_sha = plans_by_role[role].labels[plan_mod.LABEL_CONFIG_SHA256]
        if record.config_sha256 != expected_sha:
            problems.append(f"{role}: config_sha256 が Mac の計画と違う")
    if problems:
        return AutostartOutcome(status="refused", detail="; ".join(problems))

    designations = {
        role: build_designation(
            config,
            nodes[role],
            plans_by_role[role].labels[plan_mod.LABEL_CONFIG_SHA256],
            manifest,
            now,
            repo_commit=repo_commit,
            repo_dirty=repo_dirty,
        )
        for role in config.nodes
    }
    pushes = designation_pushes(designations, record_dir)
    approved = guards.build_approved_plan((), extra_forward=pushes)
    guards.request_approval(confirmer, runner, approved, nodes)
    push_steps = [step for step in approved.forward if isinstance(step, PlannedPush)]
    push_designations(runner, nodes, push_steps)

    return AutostartOutcome(
        status="designated",
        detail=f"構成 '{config.name}' の自動起動を、{len(designations)} 台に指定した",
        roles=tuple(designations),
    )


# --- serve autostart clear ----------------------------------------------------


def clear_autostart(
    runner: RemoteRunner,
    nodes: Mapping[NodeRole, NodeDef],
    record_dir: Path,
    now: datetime,
    *,
    repo_commit: str,
    repo_dirty: bool,
    confirmer: guards.Confirmer,
) -> AutostartOutcome:
    """`config: null` の指定を両台に配る (見張りは、以後 `docker run` を流さない)。"""
    roles = tuple(role for role in _ROLE_ORDER if role in nodes)
    if not roles:
        raise ValueError("nodes に head も worker もない (配る先がない)")
    designations = {
        role: _cleared_designation(role, now, repo_commit=repo_commit, repo_dirty=repo_dirty)
        for role in roles
    }
    pushes = designation_pushes(designations, record_dir)
    approved = guards.build_approved_plan((), extra_forward=pushes)
    guards.request_approval(confirmer, runner, approved, nodes)
    push_steps = [step for step in approved.forward if isinstance(step, PlannedPush)]
    push_designations(runner, nodes, push_steps)

    return AutostartOutcome(
        status="cleared", detail=f"{len(designations)} 台の自動起動の指定を解除した", roles=roles
    )


# --- serve autostart status ---------------------------------------------------


def _read_json(runner: RemoteRunner, node: NodeDef, filename: str) -> dict[str, object] | None:
    """`state/<filename>` を `cat` で読み、JSON の object として返す (読めなければ `None`)。"""
    result = runner.run(
        node,
        ("cat", f"{node.remote_root}/{RECORD_SUBDIR}/{filename}"),
        timeout_s=_READ_TIMEOUT_S,
        mutating=False,
    )
    if result.exit_code != 0:
        return None
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _validate_designation(raw: dict[str, object] | None) -> AutostartDesignation | None:
    """読めた指定の JSON を、型で確かめる (綴りの違う鍵や、形の壊れは `None` にする)。"""
    if raw is None:
        return None
    try:
        return AutostartDesignation.model_validate(raw)
    except ValueError:
        return None


def _validate_status(raw: dict[str, object] | None) -> AutostartStatus | None:
    """読めた状態の JSON を、型で確かめる (壊れていれば `None`)。"""
    if raw is None:
        return None
    try:
        return AutostartStatus.model_validate(raw)
    except ValueError:
        return None


def read_autostart(runner: RemoteRunner, nodes: Mapping[NodeRole, NodeDef]) -> AutostartShown:
    """両台の指定と状態を、`cat` で読むだけ (状態を変えない)。"""
    views: list[NodeAutostartView] = []
    for role in _ROLE_ORDER:
        node = nodes.get(role)
        if node is None:
            continue
        designation = _validate_designation(_read_json(runner, node, DESIGNATION_FILE))
        status = _validate_status(_read_json(runner, node, STATUS_FILE))
        views.append(NodeAutostartView(role=role, designation=designation, status=status))
    return AutostartShown(nodes=tuple(views))
