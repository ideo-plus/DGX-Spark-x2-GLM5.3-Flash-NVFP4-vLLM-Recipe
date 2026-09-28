#!/usr/bin/env python3
"""Spark の 1 台で常駐する、推論サーバーの自動起動・ヘルスチェック・立ち上げ直しの見張り
(issue #88 P6)。

`~/.config/systemd/user/vllm-autostart.service` から起こされる。この module は標準ライブラリ
だけに依存し、`serving_kit` を import しない (Spark には `serving_kit` を配らないため。
`serving/tests/unit/test_payload.py` の `allreduce_bench.py` と同じ流儀)。

設計の根拠は `ops/vllm-autostart/README.md` にまとめてある (issue #88 やること 1 の答え)。
ここではその要点だけを記す:

- 起動は `<remote_root>/state/<構成>.launch.json` に置かれた、Mac がすでに了承した
  `docker run` の引数の列 (`plans[].argv`) を、1 語も変えずに再生するだけである
  (「引数を加工しない」が要件 21 の核心)。Spark の上に構成ファイルは要らない
- 自動で起こす対象は `<remote_root>/state/autostart.json` (`serve autostart set` が配る) で
  指定する。指定が無い・`config` が `null`・`launch.json` の `config_sha256` が指定と違う
  ときは、`docker run` を 1 度も流さない
- 2 台は互いに ssh できないので、ノードごとに独立して動く。両台とも、自ノードの
  `docker run` の引数に含まれる `--host`/`--port` (常に head の LAN アドレス) を使って、
  head の `/health` を見る
- head は、ready のあと `/health` が 180 秒連続で落ちる、または推論プローブが 3 回連続で
  失敗したら、自コンテナを止めて起こし直す。worker は、自コンテナが終了しても、head が
  まだ `/health` に 200 を返す間は待ち (multi-node TP は 2 台そろって起こし直す必要がある
  ため)、head が 200 でなくなるか、600 秒待っても head が 200 のままなら、起こし直す
- 起動の失敗 (ready 前の終了、`ready_timeout_s` の超過) が連続 3 回続いたら `halted` にし、
  以後 `docker run` を流さない。`serve autostart set` で `designated_at` が変わると、
  この見張りを再起動しなくても、次の周から自動で再開する
- 記録するのは状態・時刻・終了の理由・HTTP の状態コードだけで、要求・応答の本文や
  `docker logs` は保存しない (要件 23)
- `--load-format instanttensor` の argv だけ、`docker run` の前に、root 不要の固定 argv で
  重みのページキャッシュを捨ててから `/proc/meminfo` の `MemFree` を確かめる (D5 の関門
  `memory_free`。下限 8 GiB、`guards.INSTANTTENSOR_MEMFREE_FLOOR_BYTES` と同じ値)。足りなければ
  起こさず、理由を状態ファイルに書く。`sudo`/`spark-drop-caches` は自動で呼ばない
  (issue #88 コメント1・更新 2026-09-28)
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["Env", "Watcher", "main"]

# --- ラベル (serving_kit.plan と同じ値。この script は import しないので、そのまま書く) -----

LABEL_OWNER: str = "vllm-baseline.owner"
OWNER: str = "serving-kit"
LABEL_CONFIG: str = "vllm-baseline.config"

# --- ファイル名 --------------------------------------------------------------

DESIGNATION_FILENAME: str = "autostart.json"
STATUS_FILENAME: str = "autostart.status.json"
STATE_SUBDIR: str = "state"

# --- 定数 (根拠は ops/vllm-autostart/README.md) -------------------------------

POLL_INTERVAL_S: float = 30.0
HEALTH_TIMEOUT_S: float = 10.0
UNHEALTHY_AFTER_S: float = 180.0
PROBE_INTERVAL_S: float = 60.0
PROBE_TIMEOUT_S: float = 60.0
PROBE_FAILURES_TO_RESTART: int = 3
START_FAILURES_TO_HALT: int = 3
WORKER_WAIT_FOR_HEAD_S: float = 600.0
STOP_TIMEOUT_S: int = 90

_DEFAULT_DOCKER: str = "/usr/bin/docker"
_DEFAULT_NVIDIA_SMI: str = "/usr/bin/nvidia-smi"
_DEFAULT_SS: str = "/usr/bin/ss"
_DEFAULT_FIND: str = "/usr/bin/find"

_ROLE_HEAD = "head"
_ROLE_WORKER = "worker"

# --- memory_free (起こす前の確認。issue #88 コメント1・更新 2026-09-28) ------------------
#
# `serving_kit.guards` (`guards.py:246-262`) と同じ値をここに写す。この script は
# `serving_kit` を import しないため (module docstring 参照)。

MEMINFO_PATH: str = "/proc/meminfo"
INSTANTTENSOR_MEMFREE_FLOOR_BYTES: int = 8 * 1024**3
MODELS_SUBDIR: str = "models"
_LOAD_FORMAT_FLAG: str = "--load-format"
_LOAD_FORMAT_INSTANTTENSOR: str = "instanttensor"


# --- Env: 実行環境の差し替え口 ------------------------------------------------


@dataclass
class Env:
    """見張りが外の世界に触れる、唯一の口 (試験は、この型を偽物に差し替える)。"""

    run: Callable[[Sequence[str]], tuple[int, str, str]]
    """`argv` を流し、`(終了コード, 標準出力, 標準エラー)` を返す。"""

    http: Callable[[str, str, bytes | None, float], int | None]
    """HTTP を 1 回呼ぶ。`method`、`url`、本文 (無ければ `None`)、`timeout_s`。
    到達しなければ (時間切れ・接続の失敗) `None` を返す。応答の本文は読み捨てる。"""

    clock: Callable[[], float]
    """単調に増える時計 (秒)。"""

    sleep: Callable[[float], None]

    now_iso: Callable[[], str]
    """状態ファイルに書く、UTC の ISO 8601 の時刻。"""

    exists: Callable[[str], bool]
    """道筋が存在するか (`--mount` の元の確認)。"""


def _real_run(
    docker: str, nvidia_smi: str, ss: str, find: str
) -> Callable[[Sequence[str]], tuple[int, str, str]]:
    """論理名 (`docker`/`nvidia-smi`/`ss`/`find`) を、差し替え口の実在の道筋に写して実行する。"""
    mapping = {"docker": docker, "nvidia-smi": nvidia_smi, "ss": ss, "find": find}

    def run(argv: Sequence[str]) -> tuple[int, str, str]:
        frozen = list(argv)
        if frozen and frozen[0] in mapping:
            frozen[0] = mapping[frozen[0]]
        completed = subprocess.run(frozen, capture_output=True, text=True, check=False)
        return completed.returncode, completed.stdout, completed.stderr

    return run


def _real_http(method: str, url: str, body: bytes | None, timeout_s: float) -> int | None:
    request = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response.read()
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def real_env(*, docker: str, nvidia_smi: str, ss: str, find: str) -> Env:
    """本番の `Env` (実物の `subprocess`・`urllib`・壁の時計)。"""
    return Env(
        run=_real_run(docker, nvidia_smi, ss, find),
        http=_real_http,
        clock=time.monotonic,
        sleep=time.sleep,
        now_iso=_now_iso,
        exists=os.path.exists,
    )


# --- argv の読み取りの助け (guards.py の同名の関数と同じ規則を、標準ライブラリだけで写す) -----


def _mount_sources(argv: Sequence[str]) -> list[str]:
    """`--mount` (`--mount X` と `--mount=X` の両方) から、元の道筋を集める。"""
    sources: list[str] = []
    pending = False
    for item in argv:
        if pending:
            pending = False
            source = _mount_source(item)
            if source is not None:
                sources.append(source)
            continue
        if item == "--mount":
            pending = True
        elif item.startswith("--mount="):
            source = _mount_source(item[len("--mount=") :])
            if source is not None:
                sources.append(source)
    seen: dict[str, None] = dict.fromkeys(sources)
    return list(seen)


def _mount_source(value: str) -> str | None:
    """`type=bind,source=…,target=…` (`src=` も可) から、元の道筋を取る。"""
    for part in value.split(","):
        key, _, found = part.partition("=")
        if key.strip() in ("source", "src"):
            return found.strip()
    return None


def _flag_value(argv: Sequence[str], flag: str) -> str | None:
    """`flag` の直後の値 (最後に現れたものを使う)。"""
    value: str | None = None
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            value = argv[index + 1]
    return value


def _name_matches(argv: Sequence[str], expected_name: str) -> bool:
    return _flag_value(argv, "--name") == expected_name


def _uses_instanttensor(argv: Sequence[str]) -> bool:
    """argv に `--load-format instanttensor` があるか (`guards.uses_instanttensor` と同じ規則)。"""
    return _flag_value(argv, _LOAD_FORMAT_FLAG) == _LOAD_FORMAT_INSTANTTENSOR


def _drop_page_cache_argv(models_dir: str) -> tuple[str, ...]:
    """重みのファイルのページキャッシュを捨てる、root 不要の固定 argv (issue #88 更新 2026-09-28。
    `docs/vllm-baseline/k2-derived-weights-procedure.md` の手順と同じ形)。
    """
    return (
        "find",
        models_dir,
        "-type",
        "f",
        "-exec",
        "dd",
        "if={}",
        "iflag=nocache",
        "count=0",
        "status=none",
        ";",
    )


def _mem_free_bytes(text: str) -> int | None:
    """`/proc/meminfo` の本文から `MemFree` (バイト) を読む。行が無い・整数でなければ `None`
    (`guards.parse_meminfo` と同じ読み取りの規則。見張りは `MemFree` だけ使う)。
    """
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if not sep or key.strip() != "MemFree":
            continue
        digits = rest.strip().split(" ", 1)[0]
        if digits.isdecimal():
            return int(digits) * 1024
    return None


# --- 段階と、周ごとの解決済みの入力 (issue #88 修正計画 修正単位 A) -----------------


class _Phase(enum.Enum):
    """見張りが自コンテナについて持つ、唯一の段階 (旧: `_fresh`・`_confirmed_ready`・
    `_halted` の 3 つの真偽値の組み合わせで段階を推測していたのを、1 つの値に置き換える)。

    内部の名前・値は契約ではない (testing-lite: 内部構造を契約化しない)。観測できるのは
    `state/autostart.status.json` の `state` (`idle`・`refused`・`starting`・`running`・
    `released`・`halted` の 6 つ、変更なし) だけである。
    """

    NEW = "new"
    """新しい指定。まだ自コンテナを 1 度も観測していない (見張りの起動直後も同じ扱い)。"""
    INVOKE = "invoke"
    """起こす。次の周で「止める → 消す → 前提検査 → 起こす」を試みる
    (新しい指定・断り・失敗のいずれの後も、この段階に入る)。"""
    BOOTING = "booting"
    """起動中。`docker run` は流れたが、まだ `/health` で ready を確認していない。"""
    READY = "ready"
    """ready。`/health` が 200 を返した、運転中の定常状態。"""
    WAITING = "waiting"
    """待ち (worker だけ)。ready の後に自コンテナが終了したが、head がまだ `/health` に
    200 を返す間、起こし直しを待つ。"""
    HALTED = "halted"
    """起動の失敗が 3 回連続した。指定 (`designated_at`) が変わるまで、何も観測しない。"""
    RELEASED = "released"
    """ready だったコンテナが見当たらない。`serve stop` などによる意図した停止として扱い、
    起こし直さない。"""


@dataclass(frozen=True)
class _Resolved:
    """1 周ぶんの、解決済みの入力 (修正単位 A)。周の最初に 1 度だけ組み立て、以後は
    再計算しない (フォールバック・デフォルト引数の禁止: `ready_timeout_s` はここで確定し、
    `or 1800` のような既定値は持たない)。
    """

    role: str
    config: str
    container_name: str
    argv: tuple[str, ...]
    host: str | None
    port: str | None
    served_model: str | None
    ready_timeout_s: int
    image_digest: str
    ports: tuple[int, ...]
    weights_record: dict[str, Any] | None


_KNOWN_STOPPED_STATES: frozenset[str] = frozenset(
    {"created", "restarting", "removing", "paused", "exited", "dead", "absent"}
)
"""Docker のコンテナの状態のうち、「止まっている」として扱うもの (問題6: `created`・`dead` も
「無い」とは扱わない)。"""


def _classify_container_state(raw_state: str) -> str:
    """`docker ps`/`inspect` が返した状態の文字列を、`running`・`stopped`・`unknown` の
    3 つに分ける。`running`・既知の「止まっている」状態のどちらでもなければ `unknown`
    (問題6: running・exited 以外の状態を「無い」とは扱わない。未知の状態は保守的に扱う)。
    """
    if raw_state == "running":
        return "running"
    if raw_state in _KNOWN_STOPPED_STATES:
        return "stopped"
    return "unknown"


def _parse_listening_ports(text: str) -> frozenset[int]:
    """`ss -ltnH` の出力から、待ち受けているポート番号を読む
    (`serving_kit.guards.parse_listening_ports` と同じ判定の規則)。読めない行があれば、
    その行番号を持つ `ValueError` を送出する (呼び出し側は、行番号だけを理由に使い、
    標準エラーや行の中身は理由に書かない)。
    """
    ports: set[int] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        port_text = fields[3].rsplit(":", 1)[-1] if len(fields) >= 4 else ""
        if fields[0] != "LISTEN" or not port_text.isdecimal():
            raise ValueError(str(line_number))
        ports.add(int(port_text))
    return frozenset(ports)


# --- Watcher -------------------------------------------------------------


class Watcher:
    """1 つの構成の、1 台ぶんの見張り。`step()` が観測・判定・行動の 1 周を進める。

    内部の状態 (このクラスの属性) は、この script だけの決めごとであり、公開の契約ではない
    (`serving/tests/unit/test_vllm_autostart.py` が観測するのは、`Env` に記録された
    呼び出しと、`state/autostart.status.json` の中身だけ)。
    """

    def __init__(self, env: Env, remote_root: str) -> None:
        self._env = env
        self._root = Path(remote_root)

        # 「新しい指定」を、まだ 1 度も見ていない (見張りの起動直後も、この扱いにする)
        self._last_designated_at: object = object()
        self._phase: _Phase = _Phase.NEW
        self._starting_since: float | None = None
        self._consecutive_start_failures = 0
        self._unhealthy_since: float | None = None
        self._consecutive_probe_failures = 0
        self._last_probe_at: float | None = None
        self._worker_wait_since: float | None = None

        self._role: str | None = None
        self._config: str | None = None
        self._container_name: str | None = None
        self._state = "idle"
        self._reason = ""
        self._container_state: str | None = None
        self._container_exit_code: int | None = None
        self._started_at: str | None = None
        self._ready_at: str | None = None
        self._last_health_status: int | None = None
        self._last_probe_status: int | None = None

    # --- 道筋 ---------------------------------------------------------

    def _designation_path(self) -> Path:
        return self._root / STATE_SUBDIR / DESIGNATION_FILENAME

    def _status_path(self) -> Path:
        return self._root / STATE_SUBDIR / STATUS_FILENAME

    def _launch_path(self, config: str) -> Path:
        return self._root / STATE_SUBDIR / f"{config}.launch.json"

    # --- 状態ファイル ---------------------------------------------------

    def _write_status(self) -> None:
        data: dict[str, Any] = {
            "schema_version": 1,
            "role": self._role,
            "config": self._config,
            "container_name": self._container_name,
            "state": self._state,
            "reason": self._reason,
            "updated_at": self._env.now_iso(),
            "started_at": self._started_at,
            "ready_at": self._ready_at,
            "container_state": self._container_state,
            "container_exit_code": self._container_exit_code,
            "last_health_status": self._last_health_status,
            "last_probe_status": self._last_probe_status,
            "consecutive_start_failures": self._consecutive_start_failures,
        }
        path = self._status_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)

    def _refuse(self, reason: str) -> None:
        self._state = "refused"
        self._reason = reason
        self._container_state = None
        self._container_exit_code = None
        self._write_status()

    # --- 自ラベルの一覧 ---------------------------------------------------

    def _own_rows(self) -> tuple[int, list[dict[str, Any]]]:
        """自ラベルの一覧。戻り値は `(docker ps の終了コード, 行の一覧)`。終了コードが
        0 以外なら、行の一覧は空にする (問題6: 呼び出し側が終了コードを見て、読めなかった
        ことを判定の状態を変えずに扱う)。`docker ps` 自体が 0 で終わったのに出力の JSON が
        壊れている場合の扱いは変えない (`json.loads` の例外がそのまま伝わる。範囲外)。
        """
        code, out, _err = self._env.run(
            ("docker", "ps", "-a", "--filter", f"label={LABEL_OWNER}={OWNER}", "--format", "json")
        )
        if code != 0:
            return code, []
        rows: list[dict[str, Any]] = []
        for line in out.splitlines():
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
        return code, rows

    def _inspect(self, container_id: str) -> tuple[int, str, int | None]:
        """コンテナ 1 つの詳しい状態。戻り値は `(inspect の終了コード, 状態, コンテナの
        終了コード)`。`docker container inspect` 自体が 0 以外で終わったら、状態は空文字列
        にする (問題6: 呼び出し側が終了コードを見て、`absent` とは扱わない)。
        """
        code, out, _err = self._env.run(
            (
                "docker",
                "container",
                "inspect",
                "--format",
                "{{.State.Status}} {{.State.ExitCode}}",
                container_id,
            )
        )
        if code != 0:
            return code, "", None
        fields = out.split()
        if not fields:
            return code, "", None
        exit_text = fields[1] if len(fields) > 1 else ""
        exit_code = int(exit_text) if exit_text.lstrip("-").isdecimal() else None
        return code, fields[0], exit_code

    def _stop_and_remove(self, container_name: str, rows: Sequence[dict[str, Any]]) -> None:
        """一覧に、その名前の自ラベルの行があるときだけ、止めて消す (guards.py と同じ不変条件)。
        一覧はこの周にすでに読んだものを使う (修正単位 A: 読み直しをしない)。
        """
        if not any(row.get("Names") == container_name for row in rows):
            return
        self._env.run(("docker", "stop", "-t", str(STOP_TIMEOUT_S), container_name))
        self._env.run(("docker", "rm", container_name))

    # --- 前提検査 (D5) ---------------------------------------------------

    def _gpu_busy_failure(self) -> str | None:
        """GPU の使用状況を確かめる (D5 の `gpu_idle`)。`nvidia-smi` が 0 以外で終わったら、
        空いているとは読まず、理由を返す (修正単位 C・問題7)。
        """
        code, out, _err = self._env.run(
            (
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            )
        )
        if code != 0:
            return f"GPU の使用状況を読めなかった (nvidia-smi の終了コード {code})"
        names = [line.strip() for line in out.splitlines() if line.strip()]
        if names:
            return f"GPU を使っているプロセスがある: {', '.join(names)}"
        return None

    def _image_digest_ok(self, image_digest: str) -> bool:
        if image_digest.startswith("sha256:"):
            code, out, _err = self._env.run(
                ("docker", "image", "inspect", "--format", "{{json .Id}}", image_digest)
            )
            if code != 0:
                return False
            try:
                return str(json.loads(out.strip())) == image_digest
            except (json.JSONDecodeError, ValueError):
                return False
        code, out, _err = self._env.run(
            ("docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image_digest)
        )
        if code != 0:
            return False
        try:
            digests = json.loads(out.strip()) or []
        except json.JSONDecodeError:
            return False
        return image_digest in digests

    def _ports_free_failure(self, ports: Sequence[int]) -> str | None:
        """指定の `ports` が空いているかを確かめる (D5 の `ports_free`)。`ss` が 0 以外で
        終わった、または読めない行があれば、空いているとは読まず、理由を返す
        (修正単位 C・問題7。行の判定の規則は `guards.parse_listening_ports` と同じ)。
        """
        if not ports:
            return None
        code, out, _err = self._env.run(("ss", "-ltnH"))
        if code != 0:
            return f"待ち受けの一覧を読めなかった (ss の終了コード {code})"
        try:
            listening = _parse_listening_ports(out)
        except ValueError as exc:
            return f"待ち受けの一覧の {exc.args[0]} 行目を読めない (ss)"
        busy = [port for port in ports if port in listening]
        if busy:
            return f"待ち受けのポートが空いていない: {', '.join(str(p) for p in busy)}"
        return None

    def _weights_verified(self, weights_record: dict[str, Any] | None, role: str) -> str | None:
        """`weights_record` の期待値と、実際の照合の記録を突き合わせる。合えば `None`、
        合わなければ理由の文を返す (D5 の `weights_verified`。指定になければ検査しない)。
        """
        if weights_record is None:
            return None
        path = Path(weights_record["path"])
        if not path.is_file():
            return f"重みの照合の記録がない: {path}"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return f"重みの照合の記録を読めない: {path}"
        if record.get("node") != role:
            return f"重みの照合の記録の node がこの役割と違う: {record.get('node')}"
        if record.get("scope") != weights_record.get("scope"):
            return "重みの照合の記録の scope が指定と違う"
        if record.get("file_count") != weights_record.get("file_count"):
            return "重みの照合の記録の file_count が指定と違う"
        if record.get("total_bytes") != weights_record.get("total_bytes"):
            return "重みの照合の記録の total_bytes が指定と違う"
        if record.get("mismatched"):
            return f"重みの照合で合わなかったファイルがある: {record.get('mismatched')}"
        for key, expected in weights_record.get("fields", {}).items():
            if record.get(key) != expected:
                return f"重みの照合の記録の {key} が指定と違う"
        return None

    def _memory_free_failure(self, argv: Sequence[str]) -> str | None:
        """instanttensor の起動の前だけ、ページキャッシュを捨ててから `MemFree` の下限を確かめる
        (D5 の関門 `memory_free`。issue #88 コメント1・更新 2026-09-28)。instanttensor でない
        argv では、何もせず `None` を返す (auto 構成では下限を見る目的が無い)。`sudo` や
        `spark-drop-caches` は、ここでは呼ばない (root の操作を自動で流さない)。
        """
        if not _uses_instanttensor(argv):
            return None
        models_dir = str(self._root / MODELS_SUBDIR)
        self._env.run(_drop_page_cache_argv(models_dir))
        code, out, _err = self._env.run(("cat", MEMINFO_PATH))
        if code != 0:
            return f"メモリの空きを読めなかった ({MEMINFO_PATH})"
        free = _mem_free_bytes(out)
        if free is None:
            return f"メモリの空きを読めなかった ({MEMINFO_PATH})"
        if free < INSTANTTENSOR_MEMFREE_FLOOR_BYTES:
            return (
                f"MemFree が下限に足りない (MemFree: {free:,} B、"
                f"下限: {INSTANTTENSOR_MEMFREE_FLOOR_BYTES:,} B)。"
                "ページキャッシュを捨てても足りないので起こさない"
            )
        return None

    def _precheck_failure(
        self,
        *,
        argv: Sequence[str],
        image_digest: str,
        ports: Sequence[int],
        weights_record: dict[str, Any] | None,
        role: str,
    ) -> str | None:
        """D5 の前提検査 (own_state は呼び出し側ですでに読んでいるので、ここでは残りの 4 つ +
        weights_verified)。すべて通れば `None`、通らなければ理由の文を返す。
        """
        gpu_reason = self._gpu_busy_failure()
        if gpu_reason is not None:
            return gpu_reason
        if not self._image_digest_ok(image_digest):
            return f"手元のイメージが、構成のダイジェストと合わない: {image_digest}"
        ports_reason = self._ports_free_failure(ports)
        if ports_reason is not None:
            return ports_reason
        weights_reason = self._weights_verified(weights_record, role)
        if weights_reason is not None:
            return weights_reason
        memory_reason = self._memory_free_failure(argv)
        if memory_reason is not None:
            return memory_reason
        return None

    # --- head の URL とプローブ -------------------------------------------

    def _health_url(self, host: str | None, port: str | None) -> str | None:
        if host is None or port is None:
            return None
        return f"http://{host}:{port}/health"

    def _get_head_health(self, host: str | None, port: str | None) -> int | None:
        url = self._health_url(host, port)
        if url is None:
            return None
        return self._env.http("GET", url, None, HEALTH_TIMEOUT_S)

    def _probe_due(self) -> bool:
        if self._last_probe_at is None:
            return True
        return self._env.clock() - self._last_probe_at >= PROBE_INTERVAL_S

    def _do_probe(self, host: str | None, port: str | None, served_model: str | None) -> int | None:
        if host is None or port is None:
            return None
        url = f"http://{host}:{port}/v1/chat/completions"
        body = json.dumps(
            {
                "model": served_model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }
        ).encode("utf-8")
        self._last_probe_at = self._env.clock()
        return self._env.http("POST", url, body, PROBE_TIMEOUT_S)

    # --- 1 周 -----------------------------------------------------------

    def step(self) -> None:
        designation_path = self._designation_path()
        if not designation_path.is_file():
            self._role = None
            self._config = None
            self._container_name = None
            self._state = "idle"
            self._reason = "指定 (autostart.json) がない"
            self._container_state = None
            self._container_exit_code = None
            self._write_status()
            return

        designation = json.loads(designation_path.read_text(encoding="utf-8"))
        role = str(designation["role"])
        config = designation.get("config")
        self._role = role
        self._config = config

        designated_at = designation.get("designated_at")
        if designated_at != self._last_designated_at:
            self._last_designated_at = designated_at
            self._phase = _Phase.NEW
            self._starting_since = None
            self._consecutive_start_failures = 0
            self._unhealthy_since = None
            self._consecutive_probe_failures = 0
            self._last_probe_at = None
            self._worker_wait_since = None

        if config is None:
            self._container_name = None
            self._state = "idle"
            self._reason = "指定された構成がない (config が null)"
            self._container_state = None
            self._container_exit_code = None
            self._write_status()
            return

        launch_path = self._launch_path(config)
        if not launch_path.is_file():
            self._refuse(f"構成 '{config}' の起動の記録 (launch.json) がない")
            return
        record = json.loads(launch_path.read_text(encoding="utf-8"))
        if record.get("config_sha256") != designation.get("config_sha256"):
            self._refuse("指定の config_sha256 と launch.json の config_sha256 が違う")
            return

        plan = next((p for p in record.get("plans", []) if p.get("node") == role), None)
        if plan is None:
            self._refuse(f"launch.json に役割 '{role}' の計画がない")
            return

        argv = tuple(plan["argv"])
        container_name = f"vb-{config}-{role}"
        self._container_name = container_name
        labels = plan.get("labels", {})
        shape_ok = (
            len(argv) >= 2
            and argv[0] == "docker"
            and argv[1] == "run"
            and labels.get(LABEL_OWNER) == OWNER
            and labels.get(LABEL_CONFIG) == config
            and _name_matches(argv, container_name)
        )
        if not shape_ok:
            self._refuse(
                "argv の形が指定と一致しない"
                " (docker run であること、所有と構成のラベル、--name のいずれかが違う)"
            )
            return

        missing_mounts = [source for source in _mount_sources(argv) if not self._env.exists(source)]
        if missing_mounts:
            self._refuse(f"--mount の元が存在しない: {', '.join(missing_mounts)}")
            return

        raw_ready_timeout = designation.get("ready_timeout_s")
        if (
            not isinstance(raw_ready_timeout, int)
            or isinstance(raw_ready_timeout, bool)
            or raw_ready_timeout < 1
        ):
            self._refuse(
                f"ready_timeout_s は 1 以上の整数でなければならない: {raw_ready_timeout!r}"
            )
            return

        resolved = _Resolved(
            role=role,
            config=config,
            container_name=container_name,
            argv=argv,
            host=_flag_value(argv, "--host"),
            port=_flag_value(argv, "--port"),
            served_model=_flag_value(argv, "--served-model-name"),
            ready_timeout_s=raw_ready_timeout,
            image_digest=str(record.get("image_digest", "")),
            ports=tuple(int(port) for port in designation.get("ports", [])),
            weights_record=designation.get("weights_record"),
        )

        if self._phase is _Phase.HALTED:
            # designated_at が変わっていなければ、何も観測せずに待ち続ける
            self._write_status()
            return

        # own_state: 自ラベルの一覧から、自分のコンテナの状態を読む (問題6: 終了コードを見る)
        own_code, rows = self._own_rows()
        if own_code != 0:
            self._reason = f"自ラベルの一覧を読めなかった (docker ps の終了コード {own_code})"
            self._write_status()
            return

        own_row = next((row for row in rows if row.get("Names") == container_name), None)
        container_exit_code: int | None
        if own_row is None:
            raw_state = "absent"
            container_exit_code = None
        else:
            raw_state = str(own_row.get("State"))
            if raw_state == "exited":
                inspect_code, inspected_state, container_exit_code = self._inspect(
                    str(own_row["ID"])
                )
                if inspect_code != 0:
                    self._reason = (
                        "コンテナの状態を読めなかった"
                        f" (docker container inspect の終了コード {inspect_code})"
                    )
                    self._write_status()
                    return
                raw_state = inspected_state
            else:
                container_exit_code = None

        classified = _classify_container_state(raw_state)
        if classified == "unknown":
            self._reason = f"知らないコンテナの状態: {raw_state}"
            self._write_status()
            return

        self._container_state = raw_state if own_row is not None else "absent"
        self._container_exit_code = container_exit_code

        if self._phase is _Phase.NEW:
            if classified == "running":
                self._adopt_running()
                return
            self._invoke_now(resolved, rows)
            return

        if self._phase is _Phase.INVOKE:
            self._invoke_now(resolved, rows)
            return

        if self._phase is _Phase.BOOTING:
            if classified == "running":
                self._monitor_starting(resolved, rows)
            else:
                # 起こした自分のコンテナが、ready になる前に落ちている。起動の失敗として
                # 数える (C4 と同じ判定。前提検査の断りの持ち越し (INVOKE) とは区別する)
                halted = self._count_start_failure()
                if not halted:
                    self._invoke_now(resolved, rows)
            return

        if self._phase is _Phase.READY:
            if classified == "running":
                self._monitor_ready(resolved, rows)
            else:
                self._handle_ready_crash(resolved, rows, own_row is not None)
            return

        if self._phase is _Phase.WAITING:
            if classified == "running":
                self._phase = _Phase.BOOTING
                self._monitor_starting(resolved, rows)
            else:
                self._monitor_waiting(resolved, rows)
            return

        # RELEASED
        if classified == "running":
            self._phase = _Phase.BOOTING
            self._monitor_starting(resolved, rows)
            return
        self._state = "released"
        self._write_status()

    # --- 分岐の中身 -------------------------------------------------------

    def _adopt_running(self) -> None:
        """見張りの起動直後 (または新しい指定の直後) に、すでに動いているコンテナを採用する
        (計画の留意点: 変えない振る舞い)。
        """
        self._phase = _Phase.READY
        self._starting_since = None
        self._consecutive_start_failures = 0
        self._unhealthy_since = None
        self._consecutive_probe_failures = 0
        self._ready_at = self._env.now_iso()
        self._state = "running"
        self._reason = "すでに動いているコンテナを採用した"
        self._write_status()

    def _invoke_now(self, resolved: _Resolved, rows: Sequence[dict[str, Any]]) -> None:
        """起こす手順: 止める → 消す → 前提検査 → 起こす、の 1 か所だけ (修正単位 A)。
        一覧はこの周にすでに読んだものを使う (読み直しをしない)。前提検査より先に自分の
        コンテナを止めて消す (問題1: 自分の GPU 使用やポートの占有が、前提検査を
        いつまでも通らせない自己ブロックにならないようにする)。
        """
        self._phase = _Phase.INVOKE
        self._stop_and_remove(resolved.container_name, rows)
        precheck_reason = self._precheck_failure(
            argv=resolved.argv,
            image_digest=resolved.image_digest,
            ports=resolved.ports,
            weights_record=resolved.weights_record,
            role=resolved.role,
        )
        if precheck_reason is not None:
            self._refuse(f"前提検査を通らない: {precheck_reason}")
            return
        code, _out, _err = self._env.run(resolved.argv)
        if code != 0:
            halted = self._count_start_failure()
            if not halted:
                self._state = "starting"
                self._reason = f"docker run が失敗した (終了コード {code})。次の周で起こし直す"
                self._write_status()
            return
        self._phase = _Phase.BOOTING
        self._starting_since = None
        self._state = "starting"
        self._reason = f"構成 '{resolved.config}' を起こした"
        self._started_at = self._env.now_iso()
        self._ready_at = None
        self._write_status()

    def _count_start_failure(self) -> bool:
        """1 つの起動の失敗を数える。3 回連続で halted にする。halted にしたかどうかを返す
        (halted でなければ、呼び出し側が `state`/`reason` を決めて `_write_status` する)。
        """
        self._consecutive_start_failures += 1
        if self._consecutive_start_failures >= START_FAILURES_TO_HALT:
            self._phase = _Phase.HALTED
            self._state = "halted"
            self._reason = (
                f"起動の失敗が {self._consecutive_start_failures} 回連続した"
                " (serve autostart set で指定をやり直すまで、起こさない)"
            )
            self._write_status()
            return True
        return False

    def _monitor_starting(self, resolved: _Resolved, rows: Sequence[dict[str, Any]]) -> None:
        """`docker run` のあと、まだ ready を確認していない間の判定 (D4)。起動中の起点が
        無ければ、この周を起点にする (問題8: `or 0.0` を使わない)。
        """
        if self._starting_since is None:
            self._starting_since = self._env.clock()
        status = self._get_head_health(resolved.host, resolved.port)
        self._last_health_status = status
        if status == 200:
            self._phase = _Phase.READY
            self._starting_since = None
            self._consecutive_start_failures = 0
            self._ready_at = self._env.now_iso()
            self._state = "running"
            self._reason = ""
            self._write_status()
            return
        elapsed = self._env.clock() - self._starting_since
        if elapsed > resolved.ready_timeout_s:
            halted = self._count_start_failure()
            if not halted:
                self._invoke_now(resolved, rows)
            return
        self._state = "starting"
        self._reason = "起動を待っている (ready_timeout_s の範囲内)"
        self._write_status()

    def _monitor_ready(self, resolved: _Resolved, rows: Sequence[dict[str, Any]]) -> None:
        """ready のあとの、head の健康観察と、必要なら立ち上げ直し (D3・修正単位 B)。
        head は自分の `/health` を見る。worker も、running のまま head の `/health` が
        `UNHEALTHY_AFTER_S` 続けて 200 でなければ、自分を起こし直す (worker の早期の
        return を削除)。プローブは head だけに残す。
        """
        status = self._get_head_health(resolved.host, resolved.port)
        self._last_health_status = status

        if status == 200:
            self._unhealthy_since = None
        else:
            if self._unhealthy_since is None:
                self._unhealthy_since = self._env.clock()
            if self._env.clock() - self._unhealthy_since >= UNHEALTHY_AFTER_S:
                self._restart_from_ready(resolved, rows)
                return

        if resolved.role == _ROLE_HEAD and self._probe_due():
            probe_status = self._do_probe(resolved.host, resolved.port, resolved.served_model)
            self._last_probe_status = probe_status
            if probe_status == 200:
                self._consecutive_probe_failures = 0
            else:
                self._consecutive_probe_failures += 1
                if self._consecutive_probe_failures >= PROBE_FAILURES_TO_RESTART:
                    self._restart_from_ready(resolved, rows)
                    return

        self._state = "running"
        self._reason = ""
        self._write_status()

    def _restart_from_ready(self, resolved: _Resolved, rows: Sequence[dict[str, Any]]) -> None:
        self._consecutive_start_failures = 0
        self._unhealthy_since = None
        self._consecutive_probe_failures = 0
        self._invoke_now(resolved, rows)

    def _handle_ready_crash(
        self, resolved: _Resolved, rows: Sequence[dict[str, Any]], own_row_present: bool
    ) -> None:
        """ready だったコンテナが、running でなくなったとき (問題3・問題5・問題9)。

        コンテナが無ければ (`docker ps` の一覧に自分の名前の行が無い)、`serve stop` などに
        よる意図した停止として扱い、`released` にして起こし直さない。行はあるが `exited` など
        (運転中のクラッシュ) なら、起動の失敗としては数えず (D4)、worker だけ head の
        `/health` を見て起こし直しを待つ (multi-node TP は 2 台そろって起こし直す必要がある
        ため。修正単位 A の「待ち」)。
        """
        if not own_row_present:
            self._phase = _Phase.RELEASED
            self._state = "released"
            self._reason = "ready のコンテナが見当たらない (serve stop による停止として扱う)"
            self._starting_since = None
            self._unhealthy_since = None
            self._consecutive_probe_failures = 0
            self._worker_wait_since = None
            self._write_status()
            return

        self._consecutive_start_failures = 0
        self._unhealthy_since = None
        self._consecutive_probe_failures = 0

        if resolved.role == _ROLE_WORKER:
            if self._worker_wait_since is None:
                self._worker_wait_since = self._env.clock()
            head_status = self._get_head_health(resolved.host, resolved.port)
            self._last_health_status = head_status
            waited = self._env.clock() - self._worker_wait_since
            if head_status == 200 and waited < WORKER_WAIT_FOR_HEAD_S:
                self._phase = _Phase.WAITING
                self._state = "starting"
                self._reason = "head がまだ応答するので、起こし直しを待っている"
                self._write_status()
                return
            self._worker_wait_since = None

        self._invoke_now(resolved, rows)

    def _monitor_waiting(self, resolved: _Resolved, rows: Sequence[dict[str, Any]]) -> None:
        """worker の「待ち」の間、周ごとに head の `/health` と待ちの時間を確かめる。"""
        if self._worker_wait_since is None:
            self._worker_wait_since = self._env.clock()
        head_status = self._get_head_health(resolved.host, resolved.port)
        self._last_health_status = head_status
        waited = self._env.clock() - self._worker_wait_since
        if head_status == 200 and waited < WORKER_WAIT_FOR_HEAD_S:
            self._state = "starting"
            self._reason = "head がまだ応答するので、起こし直しを待っている"
            self._write_status()
            return
        self._worker_wait_since = None
        self._invoke_now(resolved, rows)


# --- CLI -----------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm_autostart",
        description="Spark の 1 台で、指定された構成の推論サーバーを自動起動・見張りする",
    )
    parser.add_argument(
        "--remote-root",
        required=True,
        metavar="<道筋>",
        help="この台の vllm-baseline の根 (絶対のパス)",
    )
    parser.add_argument(
        "--docker", default=_DEFAULT_DOCKER, metavar="<道筋>", help="docker の実行ファイル"
    )
    parser.add_argument(
        "--nvidia-smi",
        default=_DEFAULT_NVIDIA_SMI,
        metavar="<道筋>",
        help="nvidia-smi の実行ファイル",
    )
    parser.add_argument("--ss", default=_DEFAULT_SS, metavar="<道筋>", help="ss の実行ファイル")
    parser.add_argument(
        "--find",
        default=_DEFAULT_FIND,
        metavar="<道筋>",
        help="find の実行ファイル (instanttensor の起動の前のページキャッシュ捨てに使う)",
    )
    parser.add_argument(
        "--once", action="store_true", help="1 周だけ流して終わる (試験・手動確認用)"
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=POLL_INTERVAL_S,
        metavar="<秒>",
        help=f"周期の間隔 (既定 {POLL_INTERVAL_S})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    env = real_env(docker=args.docker, nvidia_smi=args.nvidia_smi, ss=args.ss, find=args.find)
    watcher = Watcher(env, args.remote_root)

    stopped = {"value": False}

    def _handle_sigterm(_signum: int, _frame: object) -> None:
        stopped["value"] = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    watcher.step()
    if args.once:
        return 0
    while not stopped["value"]:
        env.sleep(args.poll_interval_s)
        if stopped["value"]:
            break
        watcher.step()
    return 0


if __name__ == "__main__":
    sys.exit(main())
