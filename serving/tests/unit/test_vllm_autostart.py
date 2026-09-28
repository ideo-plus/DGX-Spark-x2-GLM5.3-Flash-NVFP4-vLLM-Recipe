"""Spark 上の見張り `ops/vllm-autostart/vllm_autostart.py` の試験
(計画の完了契約 C1〜C5、C11、C12、C13、C15、C16)。

見張りは Spark 上で単独で動く標準ライブラリだけの script で、`serving_kit` を import しない
(`test_payload.py` の `allreduce_bench.py` と同じ確認をここでも行う)。ssh を挟まないので、
`docker`/`nvidia-smi`/`ss` の呼び出しと `/health`・推論プローブの HTTP だけを、この試験の
`FakeEnv` で差し替える。ファイル (`state/autostart.json`、`state/<構成>.launch.json`、
`state/autostart.status.json`) は、`tmp_path` の下に実際に読み書きする (見張りはローカルの
`remote_root` を直に読むので、遠隔の実行役を経由しない)。

この試験が固定する、見張りの公開の形 (実装ガイドラインに基づく。実装はこの形に従うこと):

- `Env` (dataclass): `run(argv) -> (exit_code, stdout, stderr)`、
  `http(method, url, body, timeout_s) -> int | None` (到達しなければ `None`)、
  `clock() -> float`、`sleep(seconds) -> None`、`now_iso() -> str`、`exists(path) -> bool`
- `Watcher(env, remote_root)`: `step()` が 1 周を進める。内部の分類・判定の関数名や構造は
  契約にしない (testing-lite: 内部構造を契約化しない)。観測できるのは、`env` に記録された
  呼び出しと、`remote_root/state/autostart.status.json` の中身だけである
- 状態ファイルの鍵は固定 (C5): `schema_version`、`role`、`config`、`container_name`、`state`、
  `reason`、`updated_at`、`started_at`、`ready_at`、`container_state`、`container_exit_code`、
  `last_health_status`、`last_probe_status`、`consecutive_start_failures`

対応する要求シナリオ (計画の gherkin) と、この試験の対応:

- SCN-C1-P1 → test_matching_argv_that_passes_every_precheck_is_run_verbatim
- SCN-C1-N1 → test_argv_with_wrong_owner_label_or_wrong_verb_is_refused
- SCN-C1-N2 → test_config_sha256_mismatch_between_designation_and_launch_record_is_refused
- SCN-C1-P2 → test_layout_precheck_reads_both_mount_source_spellings

C2〜C5、C11、C12、C13、C15、C16 は要求シナリオ (gherkin) の対象外なので (計画「要求シナリオ
（条件付き）」: 対象外)、計画の完了契約表の「成立する振る舞い」列を P、「拒否すべき誤実装」列を
N として、1 対 1 でテストに落とす:

- C11-P/C12-P → test_instanttensor_argv_drops_the_page_cache_then_checks_memfree_before_run
- C11-N → test_low_memfree_refuses_without_calling_sudo_or_drop_caches
- C11/C12 境界 → test_non_instanttensor_argv_neither_drops_cache_nor_reads_meminfo
- C13-P → test_launch_record_with_gates_is_read_and_the_argv_is_run
- C15 → test_memfree_floor_and_meminfo_path_match_guards
- C16 は `_load_watcher_module` (この module の読み込みの助け) が固定する。専用のテストは
  持たず、この試験一式を実行したあとに `ops/vllm-autostart/__pycache__` が無いことで確かめる
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from meminfo_sample import EXHAUSTED_MEMINFO, HEALTHY_MEMINFO
from serving_kit import guards
from serving_kit.plan import (
    LABEL_CONFIG,
    LABEL_CONFIG_SHA256,
    LABEL_IMAGE,
    LABEL_KIND,
    LABEL_OWNER,
    LABEL_ROLE,
    LABEL_STARTED_AT,
    OWNER,
)
from serving_kit.types import (
    ContainerPlan,
    ConversionSpec,
    Derivation,
    DerivedVerificationRecord,
    DerivedWeightsManifest,
    GateResult,
    LaunchRecord,
    ManifestFile,
    NodeRole,
    WeightsOrigin,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WATCHER_PATH = REPO_ROOT / "ops" / "vllm-autostart" / "vllm_autostart.py"

CONFIG = "cfg"
CONFIG_SHA256 = "aa" + "0" * 62
OTHER_SHA256 = "bb" + "0" * 62
IMAGE_REF = "img"
REPO_COMMIT = "c" * 40
HEAD_HOST = "10.0.1.60"
PORT = "8000"
STARTED_AT = datetime(2026, 9, 27, 0, 0, 0, tzinfo=UTC)
DESIGNATED_AT = "2026-09-27T08:00:00Z"
LATER_DESIGNATED_AT = "2026-09-27T09:00:00Z"


def _load_watcher_module() -> Any:
    """`ops/vllm-autostart/vllm_autostart.py` を、パッケージを介さずに import する。

    `test_payload.py._load_module` と同じ手法 (Spark に配る script は `serving_kit` の
    パッケージではないので、ファイルの道筋から直接読み込む)。

    `ops/**` は `.gitignore` で無視されていないので (issue #88 の書き込み範囲外)、この読み込みで
    `__pycache__` を書くと、実行するたびに `ops/vllm-autostart/` へ汚れが残る (C16)。
    `sys.dont_write_bytecode` を読み込みの間だけ立てて、それを避ける。
    """
    if not WATCHER_PATH.is_file():
        pytest.fail(f"見張りの script がない: {WATCHER_PATH}")
    spec = importlib.util.spec_from_file_location("vllm_autostart", WATCHER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `sys.modules` に先に登録する。登録しないと、script が `from __future__ import
    # annotations` と `dataclasses.dataclass` を組み合わせて使ったとき (書式の注釈の解決に
    # module を探すため)、`sys.modules` に見つからずに落ちる
    sys.modules[spec.name] = module
    saved_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = saved_dont_write_bytecode
    return module


@pytest.fixture(scope="module")
def watcher_module() -> Any:
    return _load_watcher_module()


# --- 見本の argv とラベル ---------------------------------------------------


def _labels(
    role: NodeRole, *, config: str = CONFIG, config_sha256: str = CONFIG_SHA256
) -> dict[str, str]:
    return {
        LABEL_OWNER: OWNER,
        LABEL_CONFIG: config,
        LABEL_CONFIG_SHA256: config_sha256,
        LABEL_KIND: "serve",
        LABEL_ROLE: role,
        LABEL_IMAGE: IMAGE_REF,
        LABEL_STARTED_AT: "2026-09-27T00:00:00Z",
    }


def _label_argv(labels: dict[str, str]) -> list[str]:
    argv: list[str] = []
    for key in sorted(labels):
        argv.extend(("--label", f"{key}={labels[key]}"))
    return argv


def _valid_argv(
    role: NodeRole,
    *,
    config: str = CONFIG,
    config_sha256: str = CONFIG_SHA256,
    mounts: Sequence[str] = (),
    container_name: str | None = None,
    labels: dict[str, str] | None = None,
    load_format: str | None = None,
) -> tuple[str, ...]:
    """指定・argv の形の検査を通る、見本の `docker run` の引数の列 (SCN-C1-P1 の形)。

    `load_format` を渡すと、末尾に `--load-format <値>` を足す (C11/C12 の
    `--load-format instanttensor` の argv を作るため)。
    """
    name = container_name if container_name is not None else f"vb-{config}-{role}"
    used_labels = (
        labels if labels is not None else _labels(role, config=config, config_sha256=config_sha256)
    )
    argv: list[str] = ["docker", "run", "-d", "--pull", "never", "--name", name]
    argv.extend(_label_argv(used_labels))
    for mount in mounts:
        argv.extend(("--mount", mount))
    argv.extend(["--restart", "no", IMAGE_REF, "--host", HEAD_HOST, "--port", PORT])
    if load_format is not None:
        argv.extend(["--load-format", load_format])
    return tuple(argv)


_DEFAULT_GATES: tuple[GateResult, ...] = (
    GateResult(gate="memory_free", node="head", passed=True, detail="試験の見本"),
)
"""`gates` の既定値 (見張りは読まない項目なので、空でない見本の値で「読み飛ばせる」ことを試す)。"""


def _launch_record_text(
    plans: Sequence[ContainerPlan],
    *,
    config: str = CONFIG,
    config_sha256: str = CONFIG_SHA256,
    gates: Sequence[GateResult] = _DEFAULT_GATES,
) -> str:
    record = LaunchRecord(
        config_name=config,
        image_digest=IMAGE_REF,
        weights=None,
        started_at=STARTED_AT,
        plans=tuple(plans),
        config_sha256=config_sha256,
        repo_commit=REPO_COMMIT,
        repo_dirty=False,
        gates=tuple(gates),
    )
    return record.model_dump_json()


def _designation(
    *,
    role: NodeRole = "head",
    config: str | None = CONFIG,
    config_sha256: str = CONFIG_SHA256,
    ready_timeout_s: int = 1800,
    ports: Sequence[int] = (8000,),
    designated_at: str = DESIGNATED_AT,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": role,
        "config": config,
        "config_sha256": config_sha256,
        "ready_timeout_s": ready_timeout_s,
        "ports": list(ports),
        "weights_record": None,
        "designated_at": designated_at,
        "repo_commit": REPO_COMMIT,
        "repo_dirty": False,
    }


# --- ローカルの `remote_root` の下ごしらえ ----------------------------------


def _write_json(path: Path, data: dict[str, Any] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = data if isinstance(data, str) else json.dumps(data)
    path.write_text(text, encoding="utf-8")


def make_remote_root(
    tmp_path: Path,
    *,
    designation: dict[str, Any] | None,
    launch_record_text: str | None,
    config: str = CONFIG,
) -> Path:
    root = tmp_path / "vllm-baseline"
    if designation is not None:
        _write_json(root / "state" / "autostart.json", designation)
    if launch_record_text is not None:
        _write_json(root / "state" / f"{config}.launch.json", launch_record_text)
    return root


def read_status(remote_root: Path) -> dict[str, Any]:
    path = remote_root / "state" / "autostart.status.json"
    assert path.is_file(), f"状態ファイルがない: {path}"
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


# --- 偽の Env ----------------------------------------------------------------


@dataclass(frozen=True)
class Reply:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class EnvRule:
    """`argv` の前方一致 (`prefix`) と述語 (`when`) で選ぶ、1 つの規則。"""

    prefix: tuple[str, ...]
    when: Callable[[tuple[str, ...]], bool] | None = None
    replies: tuple[Reply, ...] = (Reply(),)


def _run_argv_name(argv: tuple[str, ...]) -> str | None:
    """`docker run` の argv から `--name` の値を取る。"""
    for index, item in enumerate(argv):
        if item == "--name" and index + 1 < len(argv):
            return argv[index + 1]
    return None


@dataclass
class FakeContainer:
    """`FakeEnv` が持つ、自コンテナ 1 つの状態 (issue #88 修正計画 修正単位 A0)。

    `docker ps`・`inspect`・`stop`・`rm`・`run`・`nvidia-smi`・`ss` を、Docker の
    一般的な振る舞いに合わせて、この状態から動かす。実機で未確認の仮定であることは
    修正計画の留意点に記す (止まっているコンテナへの stop が 0、動いているコンテナへの
    rm が失敗、名前の衝突で run が失敗、開始に失敗したコンテナが created で残る)。
    """

    name: str
    role: str = "head"
    image: str = ""
    labels_text: str = ""
    container_id: str = "abc123"
    state: str = "absent"
    """`absent`・`running`・`exited`・`created`・`dead` のいずれか。"""
    exit_code: int | None = None

    ps_fails: bool = False
    inspect_fails: bool = False
    run_fails: bool = False
    """`True` なら `run` が 0 以外 (127) で終わり、`created` を残す。"""

    def handle_ps(self) -> tuple[int, str, str]:
        if self.ps_fails:
            return 1, "", "docker: エラー (見本)"
        if self.state == "absent":
            return 0, "", ""
        row = {
            "ID": self.container_id,
            "Names": self.name,
            "State": self.state,
            "Image": self.image,
            "Labels": self.labels_text,
        }
        return 0, json.dumps(row) + "\n", ""

    def handle_inspect(self, container_id: str) -> tuple[int, str, str] | None:
        if container_id != self.container_id:
            return None
        if self.inspect_fails:
            return 1, "", "docker: エラー (見本)"
        exit_text = "" if self.exit_code is None else str(self.exit_code)
        return 0, f"{self.state} {exit_text}".rstrip(), ""

    def handle_stop(self, name: str) -> tuple[int, str, str] | None:
        if name != self.name:
            return None
        if self.state == "absent":
            return 1, "", "docker: そのコンテナはない (見本)"
        if self.state == "running":
            self.state = "exited"
            self.exit_code = 0
        return 0, "", ""

    def handle_rm(self, name: str) -> tuple[int, str, str] | None:
        if name != self.name:
            return None
        if self.state == "absent":
            return 1, "", "docker: そのコンテナはない (見本)"
        if self.state == "running":
            return 1, "", "docker: 動いているコンテナは消せない (見本)"
        self.state = "absent"
        self.exit_code = None
        return 0, "", ""

    def handle_run(self, argv: tuple[str, ...]) -> tuple[int, str, str] | None:
        name = _run_argv_name(argv)
        if name != self.name:
            return None
        if self.state != "absent":
            return 125, "", f"docker: 名前が衝突している: {name} (見本)"
        if self.run_fails:
            self.state = "created"
            self.exit_code = None
            return 127, "", "docker: 開始に失敗した (見本)"
        self.state = "running"
        self.exit_code = None
        return 0, "", ""

    def handle_nvidia_smi(self) -> tuple[int, str, str]:
        if self.state == "running":
            return 0, "1234, python3, 2048", ""
        return 0, "", ""

    def handle_ss(self) -> tuple[int, str, str]:
        if self.state != "running":
            return 0, "", ""
        if self.role == "head":
            return (
                0,
                "LISTEN 0 128 0.0.0.0:8000 0.0.0.0:*\nLISTEN 0 128 0.0.0.0:29501 0.0.0.0:*\n",
                "",
            )
        return 0, "", ""


@dataclass
class FakeEnv:
    """見張りの `Env` の偽物。呼ばれた引数の列と HTTP の呼び出しを記録する。

    `container` を渡すと、`docker ps`・`inspect`・`stop`・`rm`・`run`・`nvidia-smi`・`ss` は
    その状態から答える (`script`・`default` は、それ以外の呼び出し (`docker image inspect`・
    `cat /proc/meminfo` など) にだけ使う)。`container` が無ければ、既存の `script`・`default`
    の台本だけで答える (既存の試験はそのまま使える)。
    """

    script: tuple[EnvRule, ...] = ()
    default: Reply | None = Reply()
    container: FakeContainer | None = None
    health: Callable[[], int | None] = field(default=lambda: 200)
    probe: Callable[[], int | None] = field(default=lambda: 200)
    exists_fn: Callable[[str], bool] = field(default=lambda path: True)
    clock_fn: Callable[[], float] = field(default=lambda: 0.0)

    calls: list[tuple[str, ...]] = field(default_factory=list)
    http_calls: list[tuple[str, str]] = field(default_factory=list)
    _used: dict[int, int] = field(default_factory=dict)

    def _dispatch_container(self, frozen: tuple[str, ...]) -> tuple[int, str, str] | None:
        container = self.container
        if container is None:
            return None
        if frozen[:2] == ("docker", "ps"):
            return container.handle_ps()
        if frozen[:3] == ("docker", "container", "inspect"):
            return container.handle_inspect(frozen[-1])
        if frozen[:2] == ("docker", "stop"):
            return container.handle_stop(frozen[-1])
        if frozen[:2] == ("docker", "rm"):
            return container.handle_rm(frozen[-1])
        if frozen[:2] == ("docker", "run"):
            return container.handle_run(frozen)
        if frozen[:1] == ("nvidia-smi",):
            return container.handle_nvidia_smi()
        if frozen[:1] == ("ss",):
            return container.handle_ss()
        return None

    def run(self, argv: Sequence[str]) -> tuple[int, str, str]:
        frozen = tuple(argv)
        self.calls.append(frozen)
        container_reply = self._dispatch_container(frozen)
        if container_reply is not None:
            return container_reply
        for index, rule in enumerate(self.script):
            if frozen[: len(rule.prefix)] != rule.prefix:
                continue
            if rule.when is not None and not rule.when(frozen):
                continue
            used = self._used.get(index, 0)
            self._used[index] = used + 1
            reply = rule.replies[min(used, len(rule.replies) - 1)]
            return reply.exit_code, reply.stdout, reply.stderr
        if self.default is not None:
            return self.default.exit_code, self.default.stdout, self.default.stderr
        raise AssertionError(f"台本にない呼び出し: {frozen}")

    def http(self, method: str, url: str, body: bytes | None, timeout_s: float) -> int | None:
        self.http_calls.append((method, url))
        if method == "GET":
            return self.health()
        if method == "POST":
            return self.probe()
        raise AssertionError(f"想定しない http の呼び出し: {method} {url}")

    def clock(self) -> float:
        return self.clock_fn()

    def sleep(self, seconds: float) -> None:
        raise AssertionError(f"試験のなかで実際に眠ろうとした: {seconds} 秒")

    def now_iso(self) -> str:
        return "2026-09-27T00:00:00Z"

    def exists(self, path: str) -> bool:
        return self.exists_fn(path)

    @property
    def docker_calls(self) -> tuple[tuple[str, ...], ...]:
        return tuple(call for call in self.calls if call and call[0] == "docker")

    @property
    def docker_run_calls(self) -> tuple[tuple[str, ...], ...]:
        return tuple(call for call in self.docker_calls if len(call) > 1 and call[1] == "run")


def _own_state_rule(*, listing: str = "") -> EnvRule:
    return EnvRule(prefix=("docker", "ps"), replies=(Reply(stdout=listing),))


def _passthrough_rules() -> tuple[EnvRule, ...]:
    """own_state・gpu_idle・image_digest・ports_free を、すべて通す最小の台本。"""
    return (
        _own_state_rule(),
        EnvRule(prefix=("nvidia-smi",), replies=(Reply(stdout=""),)),
        EnvRule(
            prefix=("docker", "image", "inspect"),
            replies=(Reply(stdout=json.dumps([IMAGE_REF])),),
        ),
        EnvRule(prefix=("ss",), replies=(Reply(stdout=""),)),
    )


_DEFAULT_REPLY: Reply = Reply()
"""既定の空の成功の返事 (`make_env` の既定引数に、呼び出しを書かないための、名前付きの値)。"""


def _image_digest_rule() -> EnvRule:
    """`docker image inspect` を通す規則 (`FakeContainer` と組み合わせて使う)。"""
    return EnvRule(
        prefix=("docker", "image", "inspect"),
        replies=(Reply(stdout=json.dumps([IMAGE_REF])),),
    )


def make_env(
    *,
    script: Sequence[EnvRule] = (),
    default: Reply | None = _DEFAULT_REPLY,
    container: FakeContainer | None = None,
    health: Callable[[], int | None] | None = None,
    probe: Callable[[], int | None] | None = None,
    exists_fn: Callable[[str], bool] | None = None,
    clock_fn: Callable[[], float] | None = None,
) -> FakeEnv:
    kwargs: dict[str, Any] = {"script": tuple(script), "default": default, "container": container}
    if health is not None:
        kwargs["health"] = health
    if probe is not None:
        kwargs["probe"] = probe
    if exists_fn is not None:
        kwargs["exists_fn"] = exists_fn
    if clock_fn is not None:
        kwargs["clock_fn"] = clock_fn
    return FakeEnv(**kwargs)


def make_container_env(
    *,
    role: NodeRole = "head",
    container_name: str = "vb-cfg-head",
    state: str = "absent",
    labels: dict[str, str] | None = None,
    health: Callable[[], int | None] | None = None,
    probe: Callable[[], int | None] | None = None,
    clock_fn: Callable[[], float] | None = None,
    exists_fn: Callable[[str], bool] | None = None,
) -> FakeEnv:
    """状態を持つ自コンテナ (`FakeContainer`) を使う `FakeEnv` (修正単位 A0)。
    `docker image inspect` だけ、通す規則を足す (`gpu_idle`・`ports_free` は
    `FakeContainer` の状態から答える)。
    """
    used_labels = labels if labels is not None else _labels(role)
    container = FakeContainer(
        name=container_name,
        role=role,
        image=IMAGE_REF,
        labels_text=",".join(f"{k}={v}" for k, v in sorted(used_labels.items())),
        state=state,
    )
    return make_env(
        script=(_image_digest_rule(),),
        container=container,
        health=health,
        probe=probe,
        clock_fn=clock_fn,
        exists_fn=exists_fn,
    )


def make_watcher(module: Any, env: FakeEnv, remote_root: Path) -> Any:
    return module.Watcher(env, str(remote_root))


def container_of(env: FakeEnv) -> FakeContainer:
    """`env.container` (`make_container_env` で必ず設定済み) を、`None` を除いた型で返す。"""
    assert env.container is not None
    return env.container


# --- 見張りは serving_kit を import しない -----------------------------------


def test_the_watcher_script_does_not_import_serving_kit() -> None:
    """Spark には `serving_kit` を配らないので、見張りは import していない
    (`test_payload.py` と同じ確認)。
    """
    if not WATCHER_PATH.is_file():
        pytest.fail(f"見張りの script がない: {WATCHER_PATH}")
    import ast

    source = WATCHER_PATH.read_text(encoding="utf-8")
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert "serving_kit" not in names


# --- C1: 指定・argv の形・sha・前提検査を通ったときだけ docker run を流す ----


def test_matching_argv_that_passes_every_precheck_is_run_verbatim(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[SCN-C1-P1] 指定と一致し、argv の形の検査と前提検査を通る argv だけを、
    そのまま `docker run` として、1 語も変えずに流す。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(script=_passthrough_rules())
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == (argv,), env.docker_run_calls
    assert len(env.docker_run_calls) == 1


def test_argv_with_wrong_owner_label_or_wrong_verb_is_refused(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[SCN-C1-N1] 所有のラベルがない・名前が違う argv も、`docker run` でない (`create`) argv も、
    1 度も docker に流さず、状態を `refused` にする。
    """
    wrong_name_and_labels = ("docker", "run", "-d", "--name", "vb-other-head", IMAGE_REF)
    wrong_name_plan = ContainerPlan(
        node="head",
        container_name="vb-cfg-head",
        labels=_labels("head"),
        argv=wrong_name_and_labels,
    )
    root_a = make_remote_root(
        tmp_path / "a",
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([wrong_name_plan]),
    )
    env_a = make_env(script=_passthrough_rules())
    make_watcher(watcher_module, env_a, root_a).step()

    not_run_argv = (
        "docker",
        "create",
        "--name",
        "vb-cfg-head",
        "--label",
        f"{LABEL_OWNER}={OWNER}",
        IMAGE_REF,
    )
    not_run_plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=not_run_argv
    )
    root_b = make_remote_root(
        tmp_path / "b",
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([not_run_plan]),
    )
    env_b = make_env(script=_passthrough_rules())
    make_watcher(watcher_module, env_b, root_b).step()

    assert env_a.docker_run_calls == ()
    assert "create" not in {call[1] for call in env_a.docker_calls if len(call) > 1}
    assert env_b.docker_run_calls == ()
    assert not any(call[:2] == ("docker", "create") for call in env_b.docker_calls)

    assert read_status(root_a)["state"] == "refused"
    assert read_status(root_b)["state"] == "refused"


def test_config_sha256_mismatch_between_designation_and_launch_record_is_refused(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[SCN-C1-N2] 指定の `config_sha256` と `launch.json` の `config_sha256` が違えば、
    `docker run` を流さず、`refused` にして理由に sha の不一致を書く。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head", config_sha256=CONFIG_SHA256),
        launch_record_text=_launch_record_text([plan], config_sha256=OTHER_SHA256),
    )
    env = make_env(script=_passthrough_rules())
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == ()
    status = read_status(remote_root)
    assert status["state"] == "refused"
    assert "sha" in status["reason"].lower() or "sha256" in status["reason"]


def test_layout_precheck_reads_both_mount_source_spellings(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[SCN-C1-P2] `--mount` の元は `source=` と `src=` の両方の書き方から取り、
    存在しない元があれば、それを名指しして `refused` にする (存在する元は名指ししない)。
    """
    argv = _valid_argv(
        "head",
        mounts=("type=bind,source=/a,target=/x", "type=bind,src=/b,target=/y"),
    )
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(
        script=_passthrough_rules(),
        exists_fn=lambda path: path == "/a",
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == ()
    status = read_status(remote_root)
    assert status["state"] == "refused"
    assert "/b" in status["reason"]
    assert "/a" not in status["reason"]


def test_a_busy_gpu_blocks_the_run(watcher_module: Any, tmp_path: Path) -> None:
    """[C1 D5] `nvidia-smi` が GPU を使っているプロセスを 1 件でも示せば、
    `docker run` を流さず `refused` にする (前提検査 `gpu_idle`)。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(
        script=(
            _own_state_rule(),
            EnvRule(prefix=("nvidia-smi",), replies=(Reply(stdout="1234, python3, 2048"),)),
            EnvRule(
                prefix=("docker", "image", "inspect"),
                replies=(Reply(stdout=json.dumps([IMAGE_REF])),),
            ),
            EnvRule(prefix=("ss",), replies=(Reply(stdout=""),)),
        )
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == ()
    assert read_status(remote_root)["state"] == "refused"


def test_an_image_digest_mismatch_blocks_the_run(watcher_module: Any, tmp_path: Path) -> None:
    """[C1 D5] 手元のイメージのダイジェストが構成と合わなければ、
    `docker run` を流さず `refused` にする (前提検査 `image_digest`)。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(
        script=(
            _own_state_rule(),
            EnvRule(prefix=("nvidia-smi",), replies=(Reply(stdout=""),)),
            EnvRule(
                prefix=("docker", "image", "inspect"),
                replies=(Reply(stdout=json.dumps(["some-other-image"])),),
            ),
            EnvRule(prefix=("ss",), replies=(Reply(stdout=""),)),
        )
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == ()
    assert read_status(remote_root)["state"] == "refused"


def test_a_busy_port_blocks_the_run(watcher_module: Any, tmp_path: Path) -> None:
    """[C1 D5] 指定の `ports` のどれかがすでに待ち受けられていれば、
    `docker run` を流さず `refused` にする (前提検査 `ports_free`)。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head", ports=(8000,)),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(
        script=(
            _own_state_rule(),
            EnvRule(prefix=("nvidia-smi",), replies=(Reply(stdout=""),)),
            EnvRule(
                prefix=("docker", "image", "inspect"),
                replies=(Reply(stdout=json.dumps([IMAGE_REF])),),
            ),
            EnvRule(
                prefix=("ss",),
                replies=(Reply(stdout="LISTEN 0 128 0.0.0.0:8000 0.0.0.0:*\n"),),
            ),
        )
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == ()
    assert read_status(remote_root)["state"] == "refused"


def test_a_weights_verification_mismatch_blocks_the_run(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[C1 D5] 指定に埋め込んだ重みの照合の期待値と、実際の照合の記録が合わなければ、
    `docker run` を流さず `refused` にする (前提検査 `weights_verified`)。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    verified_path = tmp_path / "vllm-baseline" / "state" / "example.verified.json"
    _write_json(
        verified_path,
        {
            "repo": "org/model",
            "revision": "a" * 40,
            "scope": "all",
            "node": "head",
            "verified_at": "2026-09-27T00:00:00Z",
            "file_count": 3,
            "total_bytes": 100,
            # 実際の照合で、合わなかったファイルが残っている
            "mismatched": ["config.json"],
        },
    )
    designation = _designation(role="head")
    designation["weights_record"] = {
        "path": str(verified_path),
        "scope": "all",
        "file_count": 3,
        "total_bytes": 100,
        "fields": {"repo": "org/model", "revision": "a" * 40},
    }
    remote_root = make_remote_root(
        tmp_path,
        designation=designation,
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(script=_passthrough_rules())
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == ()
    assert read_status(remote_root)["state"] == "refused"


# --- C11/C12: instanttensor の起動の前のページキャッシュ捨てと memory_free ---


def _find_page_cache_argv(remote_root: Path) -> tuple[str, ...]:
    """`--load-format instanttensor` のときに流す、ページキャッシュ捨ての固定 argv
    (issue #88 更新 (2026-09-28) が名指しした形。root は要らない)。
    """
    return (
        "find",
        str(remote_root / "models"),
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


def test_instanttensor_argv_drops_the_page_cache_then_checks_memfree_before_run(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[C11-P/C12-P] `--load-format instanttensor` の argv では、`docker run` の前に、
    固定 argv でページキャッシュを捨ててから `MemFree` を確かめ、下限 (8 GiB) を満たせば
    起こす。呼び出しの順序は `find` (キャッシュ捨て) → `cat /proc/meminfo` → `docker run`。
    """
    argv = _valid_argv("head", load_format="instanttensor")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(
        script=(
            *_passthrough_rules(),
            EnvRule(prefix=("cat", "/proc/meminfo"), replies=(Reply(stdout=HEALTHY_MEMINFO),)),
        )
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == (argv,), env.docker_run_calls
    assert len(env.docker_run_calls) == 1

    expected_find = _find_page_cache_argv(remote_root)
    find_indices = [index for index, call in enumerate(env.calls) if call == expected_find]
    meminfo_indices = [
        index for index, call in enumerate(env.calls) if call == ("cat", "/proc/meminfo")
    ]
    run_indices = [index for index, call in enumerate(env.calls) if call[:2] == ("docker", "run")]
    assert len(find_indices) == 1, env.calls
    assert len(meminfo_indices) == 1, env.calls
    assert len(run_indices) == 1, env.calls
    assert find_indices[0] < meminfo_indices[0] < run_indices[0], env.calls
    assert all(call[0] != "sudo" for call in env.calls)


def test_low_memfree_refuses_without_calling_sudo_or_drop_caches(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[C11-N] `MemFree` が下限 (8 GiB) に足りなければ、`docker run` を流さず `refused` に
    し、理由に `MemFree` を書く。足りないときも `sudo` と `spark-drop-caches` は、
    一度も呼ばない (root の操作を見張りから自動で流さない)。
    """
    argv = _valid_argv("head", load_format="instanttensor")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(
        script=(
            *_passthrough_rules(),
            EnvRule(prefix=("cat", "/proc/meminfo"), replies=(Reply(stdout=EXHAUSTED_MEMINFO),)),
        )
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == ()
    status = read_status(remote_root)
    assert status["state"] == "refused"
    assert "MemFree" in status["reason"], status["reason"]
    assert all(call[0] != "sudo" for call in env.calls)
    assert not any("spark-drop-caches" in word for call in env.calls for word in call)


def test_non_instanttensor_argv_neither_drops_cache_nor_reads_meminfo(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[C11/C12 境界] `--load-format instanttensor` を持たない argv では、ページキャッシュ
    捨ても `MemFree` の確認も行わず、そのまま起こす (auto 構成では目的が無い処理をしない)。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(script=_passthrough_rules())
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == (argv,)
    assert not any(call[0] == "find" for call in env.calls)
    assert not any(call == ("cat", "/proc/meminfo") for call in env.calls)


# --- C13: gates を持つ launch.json を読み飛ばせる ----------------------------


def test_launch_record_with_gates_is_read_and_the_argv_is_run(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[C13-P] `launch.json` が `gates` (issue #84 の起動の直前に通った関門の結果) を
    持っていても、見張りはそれを読み飛ばして、argv をそのまま起こす。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text(
            [plan],
            gates=(
                GateResult(gate="memory_free", node="head", passed=True, detail="試験の見本"),
                GateResult(gate="ports_free", node="worker", passed=True, detail="試験の見本"),
            ),
        ),
    )
    env = make_env(script=_passthrough_rules())
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == (argv,)


# --- C15: 見張りの定数が guards と一致する -----------------------------------


def test_memfree_floor_and_meminfo_path_match_guards(watcher_module: Any) -> None:
    """[C15] 見張りの `MemFree` の下限と `/proc/meminfo` の道筋は、
    `serving_kit.guards` の値と独自にずれず、一致する。
    """
    assert (
        watcher_module.INSTANTTENSOR_MEMFREE_FLOOR_BYTES == guards.INSTANTTENSOR_MEMFREE_FLOOR_BYTES
    )
    assert watcher_module.MEMINFO_PATH == guards.MEMINFO_PATH


# --- C2: absent/exited からの起動と、運転中の absent の扱い -----------------


def test_boot_then_crash_then_release_observed_on_one_instance(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[C2-P] 同一インスタンスで、次の一続きの遷移を観測する:
    (1) 見張りの起動直後、`absent` から起こす
    (2) `/health` 200 で ready になった後に `exited` になったら、`docker rm` してから
        起こし直す。起動の失敗としては数えない (D4・問題9: ready の前の終了 (C4) と区別する)
    (3) 再び ready になった後に `absent` になったら、`released` として起こさない
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    running_row = json.dumps(
        {
            "ID": "abc123",
            "Names": "vb-cfg-head",
            "State": "running",
            "Image": IMAGE_REF,
            "Labels": ",".join(f"{k}={v}" for k, v in sorted(_labels("head").items())),
        }
    )
    exited_row = json.dumps(
        {
            "ID": "abc123",
            "Names": "vb-cfg-head",
            "State": "exited",
            "Image": IMAGE_REF,
            "Labels": ",".join(f"{k}={v}" for k, v in sorted(_labels("head").items())),
        }
    )

    # (1) 起動直後は absent → 起こす
    listing_absent = ""
    env = make_env(script=(_own_state_rule(listing=listing_absent), *_passthrough_rules()[1:]))
    watcher = make_watcher(watcher_module, env, remote_root)
    watcher.step()
    assert len(env.docker_run_calls) == 1
    assert read_status(remote_root)["container_state"] in ("absent", None)

    # `/health` 200 で ready を確認する
    env.script = (_own_state_rule(listing=running_row + "\n"), *_passthrough_rules()[1:])
    env.health = lambda: 200
    watcher.step()
    assert read_status(remote_root)["state"] == "running"

    # (2) ready の後に exited になったら rm してから起こし直す (起動の失敗に数えない)
    env.script = (
        _own_state_rule(listing=exited_row + "\n"),
        EnvRule(
            prefix=("docker", "container", "inspect"),
            replies=(Reply(stdout="exited 1"),),
        ),
        *_passthrough_rules()[1:],
    )
    watcher.step()
    assert ("docker", "rm", "vb-cfg-head") in env.docker_calls
    assert len(env.docker_run_calls) == 2
    assert read_status(remote_root)["consecutive_start_failures"] == 0

    # 再び `/health` 200 で ready を確認する
    env.script = (_own_state_rule(listing=running_row + "\n"), *_passthrough_rules()[1:])
    watcher.step()
    assert read_status(remote_root)["state"] == "running"

    # (3) ready の後に absent になったら released で、起こさない
    env.script = (_own_state_rule(listing=""), *_passthrough_rules()[1:])
    run_calls_before = len(env.docker_run_calls)
    watcher.step()
    assert len(env.docker_run_calls) == run_calls_before
    assert read_status(remote_root)["state"] == "released"


def test_rm_targets_only_the_named_container_from_the_own_labeled_listing(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[C2-N] `docker rm` の対象は、自ラベルの一覧に載っている、自分の構成の名前だけである。

    一覧に無関係な (自ラベルの) 別名のコンテナが混じっていても、rm はその名前を対象にしない。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    other_row = json.dumps(
        {
            "ID": "other000",
            "Names": "vb-other-head",
            "State": "exited",
            "Image": IMAGE_REF,
            "Labels": f"{LABEL_OWNER}={OWNER}",
        }
    )
    own_row = json.dumps(
        {
            "ID": "abc123",
            "Names": "vb-cfg-head",
            "State": "exited",
            "Image": IMAGE_REF,
            "Labels": ",".join(f"{k}={v}" for k, v in sorted(_labels("head").items())),
        }
    )
    env = make_env(
        script=(
            _own_state_rule(listing=other_row + "\n" + own_row + "\n"),
            EnvRule(prefix=("docker", "container", "inspect"), replies=(Reply(stdout="exited 1"),)),
            *_passthrough_rules()[1:],
        )
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    rm_targets = [call[-1] for call in env.docker_calls if call[:2] == ("docker", "rm")]
    assert rm_targets == ["vb-cfg-head"], rm_targets


# --- C3: head/worker の立ち上げ直し ------------------------------------------
#
# C3-P (head の立ち上げ直し・worker の待ち) は、状態を持つ偽物 (`FakeContainer`) を使う
# A-T1・A-T2・A-T4・A-T5 が確かめる (修正単位 A0: running の間も GPU とポートを空きと返す
# 台本の偽物では、前提検査と停止の順序の誤りを検出できないため、台本の試験から移した)。


def test_health_failures_during_the_ready_timeout_window_are_not_counted_as_unhealthy(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[C3-N] `ready_timeout_s` の範囲内 (起動中) の `/health` の失敗は、
    ready 後の 180 秒の判定に数えない (起動中の失敗を、落ちたと誤って数えない)。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head", ready_timeout_s=1800),
        launch_record_text=_launch_record_text([plan]),
    )
    clock_box = {"t": 0.0}
    env = make_env(
        script=(_own_state_rule(listing=""), *_passthrough_rules()[1:]), health=lambda: None
    )
    env.clock_fn = lambda: clock_box["t"]
    watcher = make_watcher(watcher_module, env, remote_root)

    # 見張りの起動直後は absent → docker run を流す (起動そのものは、この 1 回だけ)
    watcher.step()
    assert len(env.docker_run_calls) == 1

    # docker run のあとは、コンテナ自体は running になる (アプリはまだ health に応えない)。
    # health は落ちたまま 300 秒続く (180 秒を超えるが、ready_timeout_s=1800 の中)
    running_row = json.dumps(
        {
            "ID": "abc123",
            "Names": "vb-cfg-head",
            "State": "running",
            "Image": IMAGE_REF,
            "Labels": ",".join(f"{k}={v}" for k, v in sorted(_labels("head").items())),
        }
    )
    env.script = (_own_state_rule(listing=running_row + "\n"), *_passthrough_rules()[1:])
    for elapsed in (30, 90, 200, 300):
        clock_box["t"] = float(elapsed)
        watcher.step()

    assert len(env.docker_run_calls) == 1, "起動中の health の失敗で、立ち上げ直してはいけない"
    assert not any(call[:2] == ("docker", "stop") for call in env.docker_calls)


# --- C4: 連続の起動の失敗と halted --------------------------------------------


def test_three_consecutive_start_failures_halt_then_new_designation_resumes(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[C4-P] 同一インスタンスで、起動の失敗 (ready 前の終了) が連続 3 回で `halted` になり、
    以後 `docker run` を流さない。`designated_at` が変わると、また起こす。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head", ready_timeout_s=1800),
        launch_record_text=_launch_record_text([plan]),
    )
    clock_box = {"t": 0.0}
    env = make_env(
        script=(_own_state_rule(listing=""), *_passthrough_rules()[1:]), health=lambda: None
    )
    env.clock_fn = lambda: clock_box["t"]
    watcher = make_watcher(watcher_module, env, remote_root)

    exited_row = json.dumps(
        {
            "ID": "abc123",
            "Names": "vb-cfg-head",
            "State": "exited",
            "Image": IMAGE_REF,
            "Labels": ",".join(f"{k}={v}" for k, v in sorted(_labels("head").items())),
        }
    )

    # 1 回目: absent → 起こす
    watcher.step()
    assert len(env.docker_run_calls) == 1

    # 3 回、ready 前に exited する (連続 3 回の起動の失敗)
    for _ in range(3):
        env.script = (
            _own_state_rule(listing=exited_row + "\n"),
            EnvRule(prefix=("docker", "container", "inspect"), replies=(Reply(stdout="exited 1"),)),
            *_passthrough_rules()[1:],
        )
        clock_box["t"] += 30.0
        watcher.step()

    assert read_status(remote_root)["state"] == "halted"
    run_count_at_halt = len(env.docker_run_calls)

    # halted のあとの周でも run が増えない
    clock_box["t"] += 30.0
    watcher.step()
    assert len(env.docker_run_calls) == run_count_at_halt

    # 新しい指定 (designated_at が変わる) → また起こす
    _write_json(
        remote_root / "state" / "autostart.json",
        _designation(role="head", designated_at=LATER_DESIGNATED_AT),
    )
    env.script = (_own_state_rule(listing=""), *_passthrough_rules()[1:])
    watcher.step()
    assert len(env.docker_run_calls) == run_count_at_halt + 1
    assert read_status(remote_root)["state"] != "halted"


def test_halted_state_persists_across_many_cycles_without_a_new_designation(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[C4-N] `halted` は、指定を変えないかぎり、何周進めても解けない
    (無限に起こし続ける実装と、見張りの再起動でしか解けない実装の、どちらも拒否する)。

    ここでは、同一インスタンスのまま何周も進めて `halted` が保たれることを確かめる (見張りの
    プロセスを再起動しなくても、指定さえ変われば解けるという設計を裏づける)。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head", ready_timeout_s=1800),
        launch_record_text=_launch_record_text([plan]),
    )
    clock_box = {"t": 0.0}
    env = make_env(
        script=(_own_state_rule(listing=""), *_passthrough_rules()[1:]), health=lambda: None
    )
    env.clock_fn = lambda: clock_box["t"]
    watcher = make_watcher(watcher_module, env, remote_root)

    exited_row = json.dumps(
        {
            "ID": "abc123",
            "Names": "vb-cfg-head",
            "State": "exited",
            "Image": IMAGE_REF,
            "Labels": ",".join(f"{k}={v}" for k, v in sorted(_labels("head").items())),
        }
    )
    watcher.step()
    for _ in range(3):
        env.script = (
            _own_state_rule(listing=exited_row + "\n"),
            EnvRule(prefix=("docker", "container", "inspect"), replies=(Reply(stdout="exited 1"),)),
            *_passthrough_rules()[1:],
        )
        clock_box["t"] += 30.0
        watcher.step()
    assert read_status(remote_root)["state"] == "halted"
    run_count_at_halt = len(env.docker_run_calls)

    for _ in range(10):
        clock_box["t"] += 30.0
        watcher.step()

    assert len(env.docker_run_calls) == run_count_at_halt
    assert read_status(remote_root)["state"] == "halted"


# --- C5: 状態ファイルの鍵集合と、本文の非保持 --------------------------------

_EXPECTED_STATUS_KEYS = frozenset(
    {
        "schema_version",
        "role",
        "config",
        "container_name",
        "state",
        "reason",
        "updated_at",
        "started_at",
        "ready_at",
        "container_state",
        "container_exit_code",
        "last_health_status",
        "last_probe_status",
        "consecutive_start_failures",
    }
)


def test_status_file_key_set_is_fixed(watcher_module: Any, tmp_path: Path) -> None:
    """[C5-P] 状態ファイルの鍵は、固定の 14 個の集合と一致する (問題16: この試験が確かめて
    いるのは鍵の集合だけである。`Env.http` は状態コード [`int | None`] しか返さない形なので、
    応答の本文はそもそも見張りに渡らない。以前ここにあった「目印が出ない」という確認は、
    どの応答にも目印を実際に入れていない空振りの確認だったため取り除いた)。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(script=_passthrough_rules())
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    status = read_status(remote_root)
    assert set(status) == _EXPECTED_STATUS_KEYS, set(status) ^ _EXPECTED_STATUS_KEYS


def test_watcher_never_calls_docker_logs(watcher_module: Any, tmp_path: Path) -> None:
    """[C5-N] 見張りは、異常終了したコンテナの `docker logs` を、1 度も呼ばない
    (本文の非保持と両立しないので、保存もしないし読みもしない)。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    exited_row = json.dumps(
        {
            "ID": "abc123",
            "Names": "vb-cfg-head",
            "State": "exited",
            "Image": IMAGE_REF,
            "Labels": ",".join(f"{k}={v}" for k, v in sorted(_labels("head").items())),
        }
    )
    env = make_env(
        script=(
            _own_state_rule(listing=exited_row + "\n"),
            EnvRule(prefix=("docker", "container", "inspect"), replies=(Reply(stdout="exited 1"),)),
            *_passthrough_rules()[1:],
        )
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert not any(call[:2] == ("docker", "logs") for call in env.docker_calls)


# --- 修正計画 (issue #88 P6 の裁定・修正) の A-T*・B-T1・C-T*・D-T1・E-T2・E-T3 ------------
#
# `FakeContainer` (状態を持つ偽物。修正単位 A0) を使い、段階の一元化・起こす手順の統合
# (修正単位 A)・worker の head 死亡検知 (修正単位 B)・前提検査の読み取りの失敗 (修正単位 C)・
# 指定の解除後の serve stop (修正単位 D)・派生の重みの照合 (修正単位 E) を確かめる。


def test_a_t1_head_restart_after_180s_unhealthy_uses_stop_rm_precheck_run_order(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[A-T1 / C3-P] head が ready のあと `/health` が 180 秒連続で落ちたら、
    `docker stop -t 90` → `docker rm` → 前提検査 (`nvidia-smi`) → `docker run` が
    この順に 1 回ずつ流れる (問題1: 前提検査を止める・消すより先に行わない)。
    150 秒の周では、まだ止めない。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    # `ready_timeout_s` は大きく取り、立ち上げ直したあとの新しい起動中の窓が、この試験の
    # 短い時間の中で起動の失敗の判定 (別の仕組み) と混ざらないようにする
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head", ready_timeout_s=1800),
        launch_record_text=_launch_record_text([plan]),
    )
    clock_box = {"t": 0.0}
    env = make_container_env(
        role="head", container_name="vb-cfg-head", state="running", health=lambda: 200
    )
    env.clock_fn = lambda: clock_box["t"]
    watcher = make_watcher(watcher_module, env, remote_root)

    clock_box["t"] = 70.0
    watcher.step()
    assert read_status(remote_root)["state"] == "running"

    env.health = lambda: None
    clock_box["t"] = 100.0
    watcher.step()
    assert env.docker_run_calls == ()

    clock_box["t"] = 100.0 + 150.0
    watcher.step()
    assert not any(c[:2] == ("docker", "stop") for c in env.docker_calls)
    assert env.docker_run_calls == ()

    clock_box["t"] = 100.0 + 180.0
    watcher.step()

    stop_calls = [c for c in env.docker_calls if c[:2] == ("docker", "stop")]
    rm_calls = [c for c in env.docker_calls if c[:2] == ("docker", "rm")]
    nvidia_calls = [c for c in env.calls if c[:1] == ("nvidia-smi",)]
    run_calls = env.docker_run_calls
    assert stop_calls == [("docker", "stop", "-t", "90", "vb-cfg-head")], stop_calls
    assert rm_calls == [("docker", "rm", "vb-cfg-head")], rm_calls
    assert len(nvidia_calls) == 1
    # すでに動いていたコンテナを採用しただけの最初の周では docker run を流していないので、
    # ここでの 1 回は、立ち上げ直しの 1 回だけである
    assert len(run_calls) == 1
    stop_index = env.calls.index(stop_calls[0])
    rm_index = env.calls.index(rm_calls[0])
    nvidia_index = env.calls.index(nvidia_calls[0])
    run_index = env.calls.index(run_calls[0])
    assert stop_index < rm_index < nvidia_index < run_index, env.calls
    assert read_status(remote_root)["state"] == "starting"


def test_a_t2_head_restart_after_three_probe_failures_uses_stop_rm_precheck_run_order(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[A-T2 / C3-P] head は、推論プローブが 3 回連続で失敗したら、
    `docker stop` → `docker rm` → 前提検査 → `docker run` の順に 1 回ずつ流す。
    2 回目の失敗の周までは流さない。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head", ready_timeout_s=1800),
        launch_record_text=_launch_record_text([plan]),
    )
    clock_box = {"t": 0.0}
    env = make_container_env(
        role="head",
        container_name="vb-cfg-head",
        state="running",
        health=lambda: 200,
        probe=lambda: 200,
    )
    env.clock_fn = lambda: clock_box["t"]
    watcher = make_watcher(watcher_module, env, remote_root)

    clock_box["t"] = 70.0
    watcher.step()
    assert read_status(remote_root)["state"] == "running"

    env.probe = lambda: None
    for step_index in range(1, 4):
        clock_box["t"] = 70.0 + step_index * 60.0
        watcher.step()
        if step_index < 3:
            assert env.docker_run_calls == ()

    stop_calls = [c for c in env.docker_calls if c[:2] == ("docker", "stop")]
    rm_calls = [c for c in env.docker_calls if c[:2] == ("docker", "rm")]
    nvidia_calls = [c for c in env.calls if c[:1] == ("nvidia-smi",)]
    run_calls = env.docker_run_calls
    assert len(stop_calls) == 1 and len(rm_calls) == 1 and len(nvidia_calls) == 1
    # すでに動いていたコンテナを採用しただけの最初の周では docker run を流していないので、
    # ここでの 1 回は、3 回連続の失敗のあとの立ち上げ直しの 1 回だけである
    assert len(run_calls) == 1
    assert (
        env.calls.index(stop_calls[0])
        < env.calls.index(rm_calls[0])
        < env.calls.index(nvidia_calls[0])
        < env.calls.index(run_calls[0])
    ), env.calls


def test_a_t3_ready_timeout_exceeded_repeatedly_retries_then_halts(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[A-T3] `ready_timeout_s` を超えるたびに `docker stop` → `rm` → 前提検査 → `run` が
    流れる。3 回目で `halted` になり、回数は 3、`docker run` は計 3 回だけ流れる。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head", ready_timeout_s=60),
        launch_record_text=_launch_record_text([plan]),
    )
    clock_box = {"t": 0.0}
    env = make_container_env(
        role="head", container_name="vb-cfg-head", state="absent", health=lambda: None
    )
    env.clock_fn = lambda: clock_box["t"]
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()  # absent → 起こす (1 回目の run)
    assert len(env.docker_run_calls) == 1

    t = 0.0
    for _ in range(2):
        t += 1.0
        clock_box["t"] = t
        watcher.step()  # running を観測、起動中の起点をこの周にする (elapsed=0)
        t += 61.0
        clock_box["t"] = t
        watcher.step()  # elapsed=61 > 60 → 起動の失敗として数え、起こし直す

    assert len(env.docker_run_calls) == 3
    assert read_status(remote_root)["state"] != "halted"

    t += 1.0
    clock_box["t"] = t
    watcher.step()
    t += 61.0
    clock_box["t"] = t
    watcher.step()  # 3 回目の失敗 → halted

    assert read_status(remote_root)["state"] == "halted"
    assert read_status(remote_root)["consecutive_start_failures"] == 3
    assert len(env.docker_run_calls) == 3


def _ready_then_exited_worker(
    watcher_module: Any, tmp_path: Path
) -> tuple[FakeEnv, Path, Any, dict[str, float]]:
    """A-T4・A-T5 の共通の段取り: worker の動いているコンテナを採用して ready にし (時計 70)、
    自コンテナが `exited` になった最初の周 (時計 100。待ちの起点) まで進める。head の
    `/health` は 200 のまま。この周では、`docker run` も起動の失敗の数えもまだ無い。
    """
    argv = _valid_argv("worker")
    plan = ContainerPlan(
        node="worker", container_name="vb-cfg-worker", labels=_labels("worker"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="worker", ready_timeout_s=1800),
        launch_record_text=_launch_record_text([plan]),
    )
    clock_box = {"t": 0.0}
    env = make_container_env(
        role="worker", container_name="vb-cfg-worker", state="running", health=lambda: 200
    )
    env.clock_fn = lambda: clock_box["t"]
    watcher = make_watcher(watcher_module, env, remote_root)

    clock_box["t"] = 70.0
    watcher.step()  # 採用 → ready
    assert read_status(remote_root)["state"] == "running"

    container_of(env).state = "exited"
    container_of(env).exit_code = 1
    clock_box["t"] = 100.0
    watcher.step()  # ready の後の異常終了を最初に観測した周 (待ちの起点)
    assert env.docker_run_calls == ()
    assert not any(c[:2] == ("docker", "rm") for c in env.docker_calls)
    assert read_status(remote_root)["consecutive_start_failures"] == 0
    return env, remote_root, watcher, clock_box


def _assert_worker_still_waiting(env: FakeEnv, remote_root: Path, at: float) -> None:
    assert env.docker_run_calls == (), f"時計 {at} の周で起こしてはいけない"
    assert not any(c[:2] == ("docker", "rm") for c in env.docker_calls), at
    assert read_status(remote_root)["consecutive_start_failures"] == 0, at


def _assert_worker_restarted_once(env: FakeEnv, remote_root: Path) -> None:
    rm_calls = [c for c in env.docker_calls if c[:2] == ("docker", "rm")]
    run_calls = env.docker_run_calls
    assert rm_calls == [("docker", "rm", "vb-cfg-worker")], rm_calls
    assert len(run_calls) == 1
    assert env.calls.index(rm_calls[0]) < env.calls.index(run_calls[0]), env.calls
    assert read_status(remote_root)["consecutive_start_failures"] == 0


def test_a_t4_worker_waits_every_cycle_while_head_is_healthy_until_600_seconds(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[A-T4 / C3-P] worker は、ready の後に自コンテナが `exited` になっても、head の
    `/health` が 200 で待ちが 600 秒 (`WORKER_WAIT_FOR_HEAD_S`) 未満の間は、どの周でも
    `docker run` を流さず、起動の失敗にも数えない (問題3)。health は 200 のまま時間だけを
    進め、待ちがちょうど 600 秒になった周で、`docker rm` → `docker run` を 1 回だけ流す。
    """
    env, remote_root, watcher, clock_box = _ready_then_exited_worker(watcher_module, tmp_path)

    for at in range(130, 700, 30):  # 130, 160, ..., 670 (待ちは 30〜570 秒)
        clock_box["t"] = float(at)
        watcher.step()
        _assert_worker_still_waiting(env, remote_root, clock_box["t"])

    clock_box["t"] = 700.0  # 待ちがちょうど 600 秒
    watcher.step()
    _assert_worker_restarted_once(env, remote_root)


def test_a_t5_worker_waits_while_head_is_healthy_then_restarts_once_head_stops_answering(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[A-T5 / C3-P] worker は、ready の後に自コンテナが `exited` になっても、head の
    `/health` が 200 の間は待つ (時計 130・160 の周)。待ちが 600 秒よりずっと短い
    (90 秒) うちに health が 200 でなくなった周 (時計 190) で、`docker rm` → `docker run` を
    1 回だけ流し、起動の失敗には数えない (問題3: health の変化と周の進行を分けて確かめる)。
    """
    env, remote_root, watcher, clock_box = _ready_then_exited_worker(watcher_module, tmp_path)

    for at in (130.0, 160.0):
        clock_box["t"] = at
        watcher.step()
        _assert_worker_still_waiting(env, remote_root, at)

    env.health = lambda: None
    clock_box["t"] = 190.0
    watcher.step()
    _assert_worker_restarted_once(env, remote_root)


def test_a_t6_memfree_refusal_persists_without_release_then_runs_once_healthy(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[A-T6] instanttensor の argv で `MemFree` が足りず断られた周は、`released` にならず、
    起動の失敗にも数えない。次の周も断られたままなら同様。`MemFree` が足りるようになった
    周で、指定をやり直さなくても `docker run` を 1 回流す。
    """
    argv = _valid_argv("head", load_format="instanttensor")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_container_env(role="head", container_name="vb-cfg-head", state="absent")
    env.script = (
        *env.script,
        EnvRule(
            prefix=("cat", "/proc/meminfo"),
            replies=(
                Reply(stdout=EXHAUSTED_MEMINFO),
                Reply(stdout=EXHAUSTED_MEMINFO),
                Reply(stdout=HEALTHY_MEMINFO),
            ),
        ),
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()
    status = read_status(remote_root)
    assert status["state"] == "refused"
    assert "MemFree" in status["reason"]
    assert status["consecutive_start_failures"] == 0

    watcher.step()
    status = read_status(remote_root)
    assert status["state"] == "refused"
    assert status["consecutive_start_failures"] == 0

    watcher.step()
    assert len(env.docker_run_calls) == 1
    assert read_status(remote_root)["state"] == "starting"


def test_a_t7_stop_and_remove_persists_across_refused_retries_after_ready_crash(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[A-T7] ready の後に `exited` を観測すると、まず止めて消す。前提検査が断り続けても、
    次の周は「無い」ままで、止める・消すを繰り返さない。`MemFree` が足りるようになった
    周で `docker run` を 1 回流す。
    """
    argv = _valid_argv("head", load_format="instanttensor")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_container_env(
        role="head", container_name="vb-cfg-head", state="running", health=lambda: 200
    )
    env.script = (
        *env.script,
        EnvRule(
            prefix=("cat", "/proc/meminfo"),
            replies=(
                Reply(stdout=EXHAUSTED_MEMINFO),
                Reply(stdout=EXHAUSTED_MEMINFO),
                Reply(stdout=HEALTHY_MEMINFO),
            ),
        ),
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()  # 採用 → ready
    assert read_status(remote_root)["state"] == "running"

    container_of(env).state = "exited"
    container_of(env).exit_code = 1
    watcher.step()  # ready の後の異常終了 → stop < rm < 検査で refused
    assert ("docker", "stop", "-t", "90", "vb-cfg-head") in env.docker_calls
    assert ("docker", "rm", "vb-cfg-head") in env.docker_calls
    status = read_status(remote_root)
    assert status["state"] == "refused"
    assert status["consecutive_start_failures"] == 0
    stop_count_after_first = sum(1 for c in env.docker_calls if c[:2] == ("docker", "stop"))
    rm_count_after_first = sum(1 for c in env.docker_calls if c[:2] == ("docker", "rm"))

    watcher.step()  # まだ無いまま、断りをやり直す (止める・消すは増えない)
    status = read_status(remote_root)
    assert status["state"] == "refused"
    assert status["consecutive_start_failures"] == 0
    assert sum(1 for c in env.docker_calls if c[:2] == ("docker", "stop")) == stop_count_after_first
    assert sum(1 for c in env.docker_calls if c[:2] == ("docker", "rm")) == rm_count_after_first

    watcher.step()  # MemFree が足りる → 起こす
    assert len(env.docker_run_calls) == 1


def test_a_t8_exited_with_missing_mount_source_persists_without_counting_until_fixed(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[A-T8] `--mount` の元が無い間は、自コンテナが `exited` でも `docker rm`/`run` を
    流さず、起動の失敗にも数えない。元が現れた周で、止める → 消す → 検査 → 起こす。
    """
    argv = _valid_argv("head", mounts=("type=bind,source=/missing,target=/x",))
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    exists = {"ok": False}
    env = make_container_env(
        role="head",
        container_name="vb-cfg-head",
        state="exited",
        exists_fn=lambda path: exists["ok"],
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()
    assert env.docker_run_calls == ()
    status = read_status(remote_root)
    assert status["state"] == "refused"
    assert "/missing" in status["reason"]
    assert status["consecutive_start_failures"] == 0

    watcher.step()
    assert env.docker_run_calls == ()
    assert not any(c[:2] == ("docker", "rm") for c in env.docker_calls)
    assert read_status(remote_root)["consecutive_start_failures"] == 0

    exists["ok"] = True
    watcher.step()
    assert ("docker", "rm", "vb-cfg-head") in env.docker_calls
    assert len(env.docker_run_calls) == 1


def test_a_t9_docker_ps_failure_writes_reason_without_calling_run_then_recovers(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[A-T9 / SCN-A-N1] `docker ps` が終了コード 1 で失敗した周は、`docker run` を流さず、
    `reason` に読めなかったことを書く。次の周に読めたら `docker run` を 1 回流す。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_container_env(role="head", container_name="vb-cfg-head", state="absent")
    container_of(env).ps_fails = True
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()
    assert env.docker_run_calls == ()
    status = read_status(remote_root)
    assert "docker ps" in status["reason"]

    container_of(env).ps_fails = False
    watcher.step()
    assert len(env.docker_run_calls) == 1


def test_a_t10_docker_ps_failure_after_ready_leaves_state_running_then_recovers(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[A-T10] ready の後に `docker ps` が失敗した周は、`docker stop`/`rm`/`run` を流さず、
    `state` は `running` のまま変えない。次の周に読めたら rm < run が流れ、回数は 0。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_container_env(
        role="head", container_name="vb-cfg-head", state="running", health=lambda: 200
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()
    assert read_status(remote_root)["state"] == "running"

    container_of(env).ps_fails = True
    container_of(env).state = "exited"
    watcher.step()
    assert env.docker_run_calls == ()
    assert not any(c[:2] == ("docker", "stop") for c in env.docker_calls)
    assert not any(c[:2] == ("docker", "rm") for c in env.docker_calls)
    assert read_status(remote_root)["state"] == "running"

    container_of(env).ps_fails = False
    watcher.step()
    assert ("docker", "rm", "vb-cfg-head") in env.docker_calls
    assert len(env.docker_run_calls) == 1
    assert read_status(remote_root)["consecutive_start_failures"] == 0


def test_a_t11_inspect_failure_leaves_state_unchanged_then_recovers(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[A-T11] `docker container inspect` が失敗した周は、`absent` と扱わず、判定の状態を
    変えない。次の周に読めたら rm < run が流れ、回数は 0。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_container_env(
        role="head", container_name="vb-cfg-head", state="running", health=lambda: 200
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()
    assert read_status(remote_root)["state"] == "running"

    container_of(env).state = "exited"
    container_of(env).exit_code = 1
    container_of(env).inspect_fails = True
    watcher.step()
    assert env.docker_run_calls == ()
    assert not any(c[:2] == ("docker", "rm") for c in env.docker_calls)
    assert read_status(remote_root)["state"] == "running"

    container_of(env).inspect_fails = False
    watcher.step()
    assert ("docker", "rm", "vb-cfg-head") in env.docker_calls
    assert len(env.docker_run_calls) == 1
    assert read_status(remote_root)["consecutive_start_failures"] == 0


def test_a_t12_run_failure_leaves_created_then_halts_after_three_attempts(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[A-T12] `docker run` が 0 以外で終わり `created` を残す場合、それぞれ起動の失敗として
    数える。3 回目で `halted` になり、以後 `docker run` が増えない。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head", ready_timeout_s=1800),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_container_env(role="head", container_name="vb-cfg-head", state="absent")
    container_of(env).run_fails = True
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()
    assert len(env.docker_run_calls) == 1
    status = read_status(remote_root)
    assert status["state"] == "starting"
    assert status["consecutive_start_failures"] == 1
    assert container_of(env).state == "created"

    watcher.step()
    assert ("docker", "rm", "vb-cfg-head") in env.docker_calls
    assert len(env.docker_run_calls) == 2
    assert read_status(remote_root)["consecutive_start_failures"] == 2

    watcher.step()
    assert len(env.docker_run_calls) == 3
    assert read_status(remote_root)["state"] == "halted"

    run_count_at_halt = len(env.docker_run_calls)
    watcher.step()
    assert len(env.docker_run_calls) == run_count_at_halt


def test_a_t13_running_observed_after_released_starts_a_fresh_boot_window(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[A-T13] `released` の後に running が (指定をやり直さずに) 再び現れたら、起動中の
    起点をその周にする (`0.0` を起点にしない)。まだ超えていない周は失敗に数えず、
    `stop`/`run` も流さない。`ready_timeout_s` を超えた周で、初めて起こし直す。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head", ready_timeout_s=60),
        launch_record_text=_launch_record_text([plan]),
    )
    clock_box = {"t": 0.0}
    env = make_container_env(
        role="head", container_name="vb-cfg-head", state="running", health=lambda: 200
    )
    env.clock_fn = lambda: clock_box["t"]
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()  # 採用 → ready
    assert read_status(remote_root)["state"] == "running"

    container_of(env).state = "absent"  # serve stop などによる意図した停止を装う
    clock_box["t"] = 100.0
    watcher.step()
    assert read_status(remote_root)["state"] == "released"

    clock_box["t"] = 5_000.0
    watcher.step()
    assert read_status(remote_root)["state"] == "released"
    assert env.docker_run_calls == ()

    # running が外から再び現れる。時計は 10,000
    container_of(env).state = "running"
    env.health = lambda: None
    clock_box["t"] = 10_000.0
    watcher.step()
    status = read_status(remote_root)
    assert status["consecutive_start_failures"] == 0
    assert env.docker_run_calls == ()
    assert not any(c[:2] == ("docker", "stop") for c in env.docker_calls)

    clock_box["t"] = 10_000.0 + 60.0  # ちょうど ready_timeout_s の境界。まだ超えていない
    watcher.step()
    assert read_status(remote_root)["consecutive_start_failures"] == 0
    assert env.docker_run_calls == ()

    clock_box["t"] = 10_000.0 + 90.0  # 60 秒を超える
    watcher.step()
    status = read_status(remote_root)
    assert status["consecutive_start_failures"] == 1
    assert any(c[:2] == ("docker", "stop") for c in env.docker_calls)
    assert len(env.docker_run_calls) == 1


@pytest.mark.parametrize("bad_value", [None, 0, True, "60", -1])
def test_a_t15_invalid_ready_timeout_s_is_refused_without_running(
    watcher_module: Any, tmp_path: Path, bad_value: Any
) -> None:
    """[A-T15 / SCN-A-N2] `ready_timeout_s` が (bool を含め) 正の整数でなければ、
    `docker run` を流さず `refused` にし、理由に `ready_timeout_s` を書く。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    designation = _designation(role="head")
    designation["ready_timeout_s"] = bad_value
    remote_root = make_remote_root(
        tmp_path, designation=designation, launch_record_text=_launch_record_text([plan])
    )
    env = make_container_env(role="head", container_name="vb-cfg-head", state="absent")
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == ()
    status = read_status(remote_root)
    assert status["state"] == "refused"
    assert "ready_timeout_s" in status["reason"]


def test_b_t1_worker_restarts_itself_after_head_health_stays_down_for_180s(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[B-T1] worker は、自コンテナが running のままでも、head の `/health` が
    `UNHEALTHY_AFTER_S` (180 秒) 連続で 200 でなければ、自分を起こし直す
    (修正単位 B: worker の早期の return を削除)。
    """
    argv = _valid_argv("worker")
    plan = ContainerPlan(
        node="worker", container_name="vb-cfg-worker", labels=_labels("worker"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="worker", ready_timeout_s=1800),
        launch_record_text=_launch_record_text([plan]),
    )
    clock_box = {"t": 0.0}
    env = make_container_env(
        role="worker", container_name="vb-cfg-worker", state="running", health=lambda: 200
    )
    env.clock_fn = lambda: clock_box["t"]
    watcher = make_watcher(watcher_module, env, remote_root)

    clock_box["t"] = 70.0
    watcher.step()
    assert read_status(remote_root)["state"] == "running"

    env.health = lambda: None
    clock_box["t"] = 100.0
    watcher.step()
    assert env.docker_run_calls == ()

    clock_box["t"] = 100.0 + 180.0
    watcher.step()

    assert any(c[:2] == ("docker", "stop") for c in env.docker_calls)
    assert len(env.docker_run_calls) == 1
    assert read_status(remote_root)["consecutive_start_failures"] == 0


def test_b_t1_negative_worker_does_not_restart_while_head_stays_healthy(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[B-T1 の反例] head の `/health` が 200 のままなら、400 秒進めても worker は
    running のまま起こさない。
    """
    argv = _valid_argv("worker")
    plan = ContainerPlan(
        node="worker", container_name="vb-cfg-worker", labels=_labels("worker"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="worker", ready_timeout_s=1800),
        launch_record_text=_launch_record_text([plan]),
    )
    clock_box = {"t": 0.0}
    env = make_container_env(
        role="worker", container_name="vb-cfg-worker", state="running", health=lambda: 200
    )
    env.clock_fn = lambda: clock_box["t"]
    watcher = make_watcher(watcher_module, env, remote_root)

    clock_box["t"] = 70.0
    watcher.step()

    clock_box["t"] = 70.0 + 400.0
    watcher.step()

    assert env.docker_run_calls == ()
    assert not any(c[:2] == ("docker", "stop") for c in env.docker_calls)
    assert read_status(remote_root)["state"] == "running"


def test_c_t1_nvidia_smi_failure_is_refused_with_reason(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[C-T1 / SCN-C-N3] `nvidia-smi` が 0 以外で終わったら、GPU が空いているとは読まず、
    `docker run` を流さず `refused` にする。標準エラーの中身は理由に書かない。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head"),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(
        script=(
            _own_state_rule(),
            EnvRule(prefix=("nvidia-smi",), replies=(Reply(exit_code=9, stderr="oops-detail"),)),
            _image_digest_rule(),
            EnvRule(prefix=("ss",), replies=(Reply(stdout=""),)),
        )
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == ()
    status = read_status(remote_root)
    assert status["state"] == "refused"
    assert "GPU" in status["reason"]
    assert "oops-detail" not in status["reason"]


def test_c_t2_ss_nonzero_exit_is_refused_and_not_read_as_free(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[C-T2] `ss` が 0 以外で終わったら、ポートが空いているとは読まず、`refused` にする。
    標準エラーの中身は理由に書かない。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head", ports=(8000,)),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(
        script=(
            _own_state_rule(),
            EnvRule(prefix=("nvidia-smi",), replies=(Reply(stdout=""),)),
            _image_digest_rule(),
            EnvRule(prefix=("ss",), replies=(Reply(exit_code=1, stderr="oops-detail"),)),
        )
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == ()
    status = read_status(remote_root)
    assert status["state"] == "refused"
    assert "ss" in status["reason"]
    assert "oops-detail" not in status["reason"]


def test_c_t3_ss_unreadable_line_is_refused(watcher_module: Any, tmp_path: Path) -> None:
    """[C-T3 / SCN-C-N1] `ss` の出力に読めない行があれば、空いているとは読まず、
    `refused` にする。
    """
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head", ports=(8000,)),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(
        script=(
            _own_state_rule(),
            EnvRule(prefix=("nvidia-smi",), replies=(Reply(stdout=""),)),
            _image_digest_rule(),
            EnvRule(prefix=("ss",), replies=(Reply(stdout="garbage\n"),)),
        )
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == ()
    status = read_status(remote_root)
    assert status["state"] == "refused"
    assert "行" in status["reason"]


def test_scn_c_p1_readable_ss_line_with_own_port_free_still_runs(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[SCN-C-P1] 読める `ss` の行があっても、指定の port と重なっていなければ流す。"""
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    remote_root = make_remote_root(
        tmp_path,
        designation=_designation(role="head", ports=(8000,)),
        launch_record_text=_launch_record_text([plan]),
    )
    env = make_env(
        script=(
            _own_state_rule(),
            EnvRule(prefix=("nvidia-smi",), replies=(Reply(stdout=""),)),
            _image_digest_rule(),
            EnvRule(prefix=("ss",), replies=(Reply(stdout="LISTEN 0 128 0.0.0.0:22 0.0.0.0:*\n"),)),
        )
    )
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert len(env.docker_run_calls) == 1


def test_d_t1_cleared_designation_never_touches_docker_across_many_cycles(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[D-T1] 指定の解除 (`config: null`) を読んだ後は、それ以降どの周でも
    `docker` に触れない。head・worker の両方で確かめる。
    """
    for role in ("head", "worker"):
        designation = _designation(role=role, config=None)
        remote_root = make_remote_root(
            tmp_path / role, designation=designation, launch_record_text=None
        )
        env = make_env()
        watcher = make_watcher(watcher_module, env, remote_root)

        for _ in range(5):
            watcher.step()

        assert env.calls == []
        status = read_status(remote_root)
        assert status["state"] == "idle"
        assert status["config"] is None


def _derived_manifest_for_watcher() -> DerivedWeightsManifest:
    return DerivedWeightsManifest(
        kind="derived",
        derivation=Derivation(
            name="k2s1",
            origin=WeightsOrigin(repo="org/model", revision="a" * 40),
            conversion=ConversionSpec(
                tool="experiments/k2-quant/convert.py",
                commit="b" * 40,
                args=("--dtype", "fp8"),
                target_pattern=r"^model\.layers\.\d+\.self_attn\..*$",
            ),
        ),
        generated_at=STARTED_AT,
        total_bytes=100,
        files=(ManifestFile(path="config.json", size=100, sha256="c" * 64),),
    )


def _designation_with_derived_weights_record(
    manifest: DerivedWeightsManifest, *, manifest_sha256: str
) -> dict[str, Any]:
    designation = _designation(role="head")
    designation["weights_record"] = {
        "path": "",  # 呼び出し側が差し替える
        "scope": "all",
        "file_count": len(manifest.files),
        "total_bytes": manifest.total_bytes,
        "fields": {"manifest_sha256": manifest_sha256},
    }
    return designation


def test_e_t2_matching_derived_verification_record_runs(
    watcher_module: Any, tmp_path: Path
) -> None:
    """[E-T2 / SCN-E-P1] 派生の重みの照合の記録の `manifest_sha256` が指定と一致すれば、
    `docker run` を 1 回流す。
    """
    manifest = _derived_manifest_for_watcher()
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    verified_path = tmp_path / "vllm-baseline" / "state" / "k2s1.derived.verified.json"
    record = DerivedVerificationRecord(
        kind="derived",
        derivation=manifest.derivation,
        manifest_sha256=manifest.content_sha256,
        scope="all",
        node="head",
        verified_at=STARTED_AT,
        file_count=len(manifest.files),
        total_bytes=manifest.total_bytes,
    )
    _write_json(verified_path, json.dumps(record.model_dump(mode="json")))
    designation = _designation_with_derived_weights_record(
        manifest, manifest_sha256=manifest.content_sha256
    )
    designation["weights_record"]["path"] = str(verified_path)
    remote_root = make_remote_root(
        tmp_path, designation=designation, launch_record_text=_launch_record_text([plan])
    )
    env = make_env(script=_passthrough_rules())
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert len(env.docker_run_calls) == 1


def test_e_t3_mismatching_manifest_sha256_is_refused(watcher_module: Any, tmp_path: Path) -> None:
    """[E-T3 / SCN-E-N1] 記録の `manifest_sha256` が指定と違えば、`docker run` を流さず
    `refused` にする。
    """
    manifest = _derived_manifest_for_watcher()
    argv = _valid_argv("head")
    plan = ContainerPlan(
        node="head", container_name="vb-cfg-head", labels=_labels("head"), argv=argv
    )
    verified_path = tmp_path / "vllm-baseline" / "state" / "k2s1.derived.verified.json"
    correct_sha = manifest.content_sha256
    wrong_sha = ("0" if correct_sha[0] != "0" else "1") + correct_sha[1:]
    record = DerivedVerificationRecord(
        kind="derived",
        derivation=manifest.derivation,
        manifest_sha256=wrong_sha,
        scope="all",
        node="head",
        verified_at=STARTED_AT,
        file_count=len(manifest.files),
        total_bytes=manifest.total_bytes,
    )
    _write_json(verified_path, json.dumps(record.model_dump(mode="json")))
    designation = _designation_with_derived_weights_record(manifest, manifest_sha256=correct_sha)
    designation["weights_record"]["path"] = str(verified_path)
    remote_root = make_remote_root(
        tmp_path, designation=designation, launch_record_text=_launch_record_text([plan])
    )
    env = make_env(script=_passthrough_rules())
    watcher = make_watcher(watcher_module, env, remote_root)

    watcher.step()

    assert env.docker_run_calls == ()
    status = read_status(remote_root)
    assert status["state"] == "refused"
    assert "manifest_sha256" in status["reason"]
