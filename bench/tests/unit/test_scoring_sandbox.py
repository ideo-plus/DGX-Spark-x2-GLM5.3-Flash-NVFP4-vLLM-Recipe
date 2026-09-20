"""隔離の実行の試験 (task 6.6: scoring/sandbox、5.4)。コンテナは使わない。

外部のコマンドの層 (`which`、`run_command`、`start_process`) を差し替えて、
起動の引数、時間切れと中断の後始末、終了コードの読み分け、出力の上限、
使えるかどうかの判定を確かめる。実物のコンテナを使う確認は
`tests/integration/test_sandbox_container.py`。
"""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from bench_harness.scoring import code as code_module
from bench_harness.scoring import sandbox as sandbox_module
from bench_harness.scoring.sandbox import (
    CONTAINER_NAME_PREFIX,
    FORBIDDEN_RUN_ARGS,
    STDERR_TAIL_BYTES,
    CommandOutput,
    ContainerSandbox,
    SandboxError,
    build_run_argv,
)
from bench_harness.types import SandboxSettings, SandboxUnavailable

PINNED_ID = "sha256:" + "a" * 64
IMAGE_NAME = "bench-sandbox:py3.13-numpy2.5.3"
NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def make_settings(**overrides: Any) -> SandboxSettings:
    """試験で使う設定。`model_validate` で作る (凍結の型の決まり、note 1.2)。"""
    data: dict[str, Any] = {
        "runtime": "docker",
        "image": IMAGE_NAME,
        "image_digest": PINNED_ID,
        "timeout_s": 20.0,
        "memory_mb": 512,
        "cpus": 1.0,
        "pids_limit": 64,
    }
    data.update(overrides)
    return SandboxSettings.model_validate(data)


# --- 偽の外部のコマンドの層 --------------------------------------------------


class FakeStream:
    """標準入力と標準エラーの代わり。書いた中身と、読んだ大きさを覚える。"""

    def __init__(self, chunk: bytes = b"", repeat: int = 1) -> None:
        self.written = bytearray()
        self.read_sizes: list[int] = []
        self.close_count = 0
        self._chunk = chunk
        self._left = repeat

    def read(self, size: int = -1, /) -> bytes:
        self.read_sizes.append(size)
        if self._left <= 0:
            return b""
        self._left -= 1
        return self._chunk

    def write(self, data: bytes, /) -> int:
        self.written.extend(data)
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.close_count += 1


class FakeProcess:
    """`docker run` のプロセスの代わり。"""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stderr_chunk: bytes = b"",
        stderr_repeat: int = 1,
        timeout_first: bool = False,
        wait_error: BaseException | None = None,
    ) -> None:
        self.stdin = FakeStream()
        self.stderr = FakeStream(stderr_chunk, stderr_repeat)
        self.kill_count = 0
        self.wait_timeouts: list[float | None] = []
        self._returncode = returncode
        self._timeout_first = timeout_first
        self._wait_error = wait_error

    def kill(self) -> None:
        self.kill_count += 1

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self._wait_error is not None:
            error, self._wait_error = self._wait_error, None
            raise error
        if self._timeout_first:
            self._timeout_first = False
            raise subprocess.TimeoutExpired(cmd="docker", timeout=timeout or 0.0)
        return self._returncode


class FakeRuntime:
    """`which` と、2 種類のコマンドの口をまとめた差し替え先。"""

    def __init__(
        self,
        *,
        installed: Sequence[str] = ("docker",),
        version_code: int = 0,
        inspect_code: int = 0,
        inspect_id: str = PINNED_ID,
        processes: Sequence[FakeProcess] = (),
    ) -> None:
        self.installed = tuple(installed)
        self.version_code = version_code
        self.inspect_code = inspect_code
        self.inspect_id = inspect_id
        self.commands: list[list[str]] = []
        self.started: list[list[str]] = []
        self.processes = list(processes)
        self.which_calls: list[str] = []

    def which(self, name: str) -> str | None:
        self.which_calls.append(name)
        return f"/usr/local/bin/{name}" if name in self.installed else None

    def run_command(self, argv: Sequence[str], timeout_s: float) -> CommandOutput:
        assert timeout_s > 0.0
        self.commands.append(list(argv))
        if argv[1:2] == ["version"]:
            return CommandOutput(returncode=self.version_code, stdout="29.7.2\n", stderr="")
        if argv[1:3] == ["image", "inspect"]:
            return CommandOutput(
                returncode=self.inspect_code,
                stdout=f"{self.inspect_id}\n" if self.inspect_code == 0 else "",
                stderr="" if self.inspect_code == 0 else "Error: No such image",
            )
        return CommandOutput(returncode=0, stdout="", stderr="")

    def start_process(self, argv: Sequence[str]) -> FakeProcess:
        self.started.append(list(argv))
        if not self.processes:
            return FakeProcess()
        return self.processes.pop(0)

    def kill_calls(self) -> list[list[str]]:
        return [argv for argv in self.commands if argv[1:2] == ["kill"]]


def make_sandbox(runtime: FakeRuntime, settings: SandboxSettings | None = None) -> ContainerSandbox:
    return ContainerSandbox(
        settings if settings is not None else make_settings(),
        which=runtime.which,
        run_command=runtime.run_command,
        start_process=runtime.start_process,
    )


# --- 1. 起動の引数 -----------------------------------------------------------


def test_argv_fixes_the_isolation_flags() -> None:
    """設計の「起動の引数を固定する」を、そのまま確かめる。"""
    argv = build_run_argv("/usr/local/bin/docker", PINNED_ID, "bench-sbx-1-1-abcd", make_settings())
    assert argv[0] == "/usr/local/bin/docker"
    assert argv[1] == "run"
    pairs = list(zip(argv, argv[1:], strict=False))
    assert ("--network", "none") in pairs
    assert ("--name", "bench-sbx-1-1-abcd") in pairs
    assert ("--tmpfs", "/tmp:rw,size=64m") in pairs
    assert ("--cap-drop", "ALL") in pairs
    assert ("--security-opt", "no-new-privileges") in pairs
    assert ("--memory", "512m") in pairs
    assert ("--memory-swap", "512m") in pairs
    assert ("--cpus", "1") in pairs
    assert ("--pids-limit", "64") in pairs
    assert ("--user", "65534:65534") in pairs
    assert ("--pull", "never") in pairs
    assert "--rm" in argv
    assert "-i" in argv
    assert "--read-only" in argv
    assert argv[-4:] == [PINNED_ID, "python", "-I", "-"]


def test_argv_has_no_flag_that_would_open_the_box() -> None:
    """マウント、環境変数、特権、デバイスの渡しは、いっさい付けない。"""
    argv = build_run_argv("/usr/local/bin/docker", PINNED_ID, "bench-sbx-1-1-abcd", make_settings())
    assert FORBIDDEN_RUN_ARGS.isdisjoint(argv)
    forbidden_prefixes = (
        "-v",
        "--volume",
        "--mount",
        "-e",
        "--env",
        "--privileged",
        "--device",
        "--cap-add",
        "--pid=",
        "--ipc=",
        "--userns",
    )
    offenders = [arg for arg in argv if arg.startswith(forbidden_prefixes)]
    assert offenders == []
    assert "host" not in argv


def test_forbidden_args_cover_the_ways_out_of_the_box() -> None:
    """外に穴を開ける引数を、一覧として固定する (足し忘れを防ぐ)。"""
    must_be_listed = {
        "-v",
        "--volume",
        "--volumes-from",
        "--mount",
        "-e",
        "--env",
        "--env-file",
        "--privileged",
        "--device",
        "--cap-add",
        "--userns",
        "--pid",
        "--ipc",
        "--net",
        "-p",
        "--publish",
        "-P",
        "--publish-all",
        "--gpus",
        "--sysctl",
        "--entrypoint",
        "--cgroupns",
    }
    assert must_be_listed <= FORBIDDEN_RUN_ARGS


def test_the_only_security_opt_is_no_new_privileges() -> None:
    """`--security-opt` に渡す値を 1 つだけにする (seccomp を緩めない)。"""
    argv = build_run_argv("/usr/local/bin/docker", PINNED_ID, "bench-sbx-1-1-abcd", make_settings())
    values = [argv[index + 1] for index, arg in enumerate(argv) if arg == "--security-opt"]
    assert values == ["no-new-privileges"]


def test_argv_rejects_an_image_or_name_that_looks_like_a_flag() -> None:
    """引数の注入を断る (fail closed)。"""
    with pytest.raises(SandboxError):
        build_run_argv("/usr/local/bin/docker", "--privileged", "bench-sbx-1", make_settings())
    with pytest.raises(SandboxError):
        build_run_argv("/usr/local/bin/docker", PINNED_ID, "--rm", make_settings())


def test_memory_and_cpu_come_from_the_settings() -> None:
    settings = make_settings(memory_mb=256, cpus=0.5, pids_limit=16)
    argv = build_run_argv("/usr/local/bin/docker", PINNED_ID, "bench-sbx-1", settings)
    pairs = list(zip(argv, argv[1:], strict=False))
    assert ("--memory", "256m") in pairs
    assert ("--memory-swap", "256m") in pairs
    assert ("--cpus", "0.5") in pairs
    assert ("--pids-limit", "16") in pairs


# --- 2. 実行 -----------------------------------------------------------------


def test_runs_by_image_id_when_pinned() -> None:
    """ダイジェストが決まっているときは、名前ではなく識別子で動かす。

    名前を付け替えただけの別のイメージに、黙って入れ替わらないようにする。
    """
    runtime = FakeRuntime()
    result = make_sandbox(runtime).run_python("print(1)\n", 5.0)
    assert result.passed is True
    argv = runtime.started[0]
    assert PINNED_ID in argv
    assert IMAGE_NAME not in argv


def test_runs_by_the_inspected_image_id_when_not_pinned() -> None:
    """固定していなくても、名前ではなく調べた識別子で動かして、それを記録する。

    設計の「識別子を計測ランに記録する」に合わせる。名前のままだと、計測ランに
    付け替え可能な名前が残ってしまう。
    """
    runtime = FakeRuntime(inspect_id=PINNED_ID)
    sandbox = make_sandbox(runtime, make_settings(image_digest=None))
    sandbox.run_python("print(1)\n", 5.0)
    assert PINNED_ID in runtime.started[0]
    assert IMAGE_NAME not in runtime.started[0]
    assert sandbox.image_ref() == PINNED_ID


def test_image_ref_is_the_inspected_id_when_pinned() -> None:
    runtime = FakeRuntime()
    sandbox = make_sandbox(runtime)
    assert sandbox.available() is True
    assert sandbox.image_ref() == PINNED_ID


def test_image_ref_is_none_when_unavailable() -> None:
    assert make_sandbox(FakeRuntime(installed=())).image_ref() is None


@pytest.mark.parametrize("inspect_id", ["", "not-an-id", "sha256:zz", "<no value>"])
def test_unexpected_inspect_output_is_unavailable(inspect_id: str) -> None:
    """`image inspect` が識別子らしくない値を返したら、使えないものにする。"""
    runtime = FakeRuntime(inspect_id=inspect_id)
    status = make_sandbox(runtime, make_settings(image_digest=None)).available()
    assert isinstance(status, SandboxUnavailable)
    assert runtime.started == []


def test_bare_hex_image_id_is_accepted_and_matched() -> None:
    """podman の `{{.Id}}` は `sha256:` を付けない。同じ識別子として扱う。"""
    runtime = FakeRuntime(inspect_id="a" * 64)
    sandbox = make_sandbox(runtime)
    assert sandbox.available() is True
    assert sandbox.image_ref() == PINNED_ID
    sandbox.run_python("print(1)\n", 5.0)
    assert PINNED_ID in runtime.started[0]


def test_default_process_starter_discards_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """標準出力は捨てる。パイプにすると、大量に書くコードで詰まって止まる。"""
    captured: dict[str, Any] = {}

    def fake_popen(argv: Sequence[str], **kwargs: Any) -> FakeProcess:
        captured["argv"] = list(argv)
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    sandbox_module._default_start_process(["/usr/local/bin/docker", "run"])
    assert captured["stdout"] is subprocess.DEVNULL
    assert captured["stderr"] is subprocess.PIPE
    assert captured["stdin"] is subprocess.PIPE


def test_stdin_receives_the_whole_program_as_utf8() -> None:
    process = FakeProcess()
    runtime = FakeRuntime(processes=[process])
    source = "print('日本語')\n" + "x = 1\n" * 1000
    make_sandbox(runtime).run_python(source, 5.0)
    assert bytes(process.stdin.written) == source.encode("utf-8")
    assert process.stdin.close_count >= 1


def test_container_names_are_unique_and_valid() -> None:
    runtime = FakeRuntime()
    sandbox = make_sandbox(runtime)
    sandbox.run_python("print(1)\n", 5.0)
    sandbox.run_python("print(2)\n", 5.0)
    names = [argv[argv.index("--name") + 1] for argv in runtime.started]
    assert len(set(names)) == 2
    for name in names:
        assert name.startswith(CONTAINER_NAME_PREFIX)
        assert NAME_RE.match(name)


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("inf"), float("nan")])
def test_bad_timeout_is_rejected(timeout_s: float) -> None:
    """外から来る値は、型だけでなく範囲も確かめる (note 2.7)。"""
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="timeout"):
        make_sandbox(runtime).run_python("print(1)\n", timeout_s)
    assert runtime.started == []


# --- 3. 時間切れ、中断、例外 --------------------------------------------------


def test_timeout_kills_the_container_by_name() -> None:
    """`docker run` のプロセスを殺してもコンテナは残るので、名前で kill する。"""
    process = FakeProcess(timeout_first=True, returncode=137)
    runtime = FakeRuntime(processes=[process])
    result = make_sandbox(runtime).run_python("while True: pass\n", 2.0)

    assert result.timed_out is True
    assert result.passed is False
    name = runtime.started[0][runtime.started[0].index("--name") + 1]
    kills = runtime.kill_calls()
    assert len(kills) == 1
    assert kills[0][1:] == ["kill", name]
    assert process.wait_timeouts[0] == 2.0


def test_keyboard_interrupt_kills_the_container_and_reraises() -> None:
    process = FakeProcess(wait_error=KeyboardInterrupt())
    runtime = FakeRuntime(processes=[process])
    with pytest.raises(KeyboardInterrupt):
        make_sandbox(runtime).run_python("print(1)\n", 5.0)
    assert len(runtime.kill_calls()) == 1


def test_any_exception_kills_the_container_and_reraises() -> None:
    process = FakeProcess(wait_error=RuntimeError("boom"))
    runtime = FakeRuntime(processes=[process])
    with pytest.raises(RuntimeError, match="boom"):
        make_sandbox(runtime).run_python("print(1)\n", 5.0)
    assert len(runtime.kill_calls()) == 1


def test_kill_failure_does_not_hide_the_timeout() -> None:
    """後始末の失敗を、元の結果の上に投げない。"""

    class Exploding(FakeRuntime):
        def run_command(self, argv: Sequence[str], timeout_s: float) -> CommandOutput:
            if argv[1:2] == ["kill"]:
                self.commands.append(list(argv))
                raise OSError("kill failed")
            return super().run_command(argv, timeout_s)

    runtime = Exploding(processes=[FakeProcess(timeout_first=True, returncode=137)])
    result = make_sandbox(runtime).run_python("while True: pass\n", 2.0)
    assert result.timed_out is True
    assert len(runtime.kill_calls()) == 1


# --- 4. 終了コードの読み分け --------------------------------------------------


def test_zero_exit_code_passes() -> None:
    runtime = FakeRuntime(processes=[FakeProcess(returncode=0)])
    result = make_sandbox(runtime).run_python("print(1)\n", 5.0)
    assert result.passed is True
    assert result.timed_out is False
    assert result.exit_code == 0


def test_nonzero_exit_code_fails() -> None:
    runtime = FakeRuntime(processes=[FakeProcess(returncode=1, stderr_chunk=b"AssertionError\n")])
    result = make_sandbox(runtime).run_python("assert False\n", 5.0)
    assert result.passed is False
    assert result.timed_out is False
    assert result.exit_code == 1
    assert "AssertionError" in result.stderr_tail


def test_oom_exit_code_is_a_failed_answer_not_an_infrastructure_error() -> None:
    """137 (SIGKILL) は、メモリの上限に当たったモデルのコードの結果である。"""
    runtime = FakeRuntime(processes=[FakeProcess(returncode=137)])
    result = make_sandbox(runtime).run_python("x = bytearray(3 * 1024**3)\n", 5.0)
    assert result.passed is False
    assert result.exit_code == 137
    assert result.timed_out is False


@pytest.mark.parametrize(
    ("exit_code", "stderr"),
    [
        (125, b"docker: Error response from daemon: OCI runtime create failed\n"),
        (125, b"Error: unable to start container\n"),
        (126, b'docker: Error response from daemon: exec: "python": cannot be invoked\n'),
        (127, b"docker: Error: executable file not found in $PATH\n"),
    ],
)
def test_cli_error_with_the_runtime_marker_is_an_infrastructure_error(
    exit_code: int, stderr: bytes
) -> None:
    """実行環境自身の失敗だけを、採点できなかったほうに回す (6.7 は NOT_SCORED)。"""
    runtime = FakeRuntime(processes=[FakeProcess(returncode=exit_code, stderr_chunk=stderr)])
    with pytest.raises(SandboxError):
        make_sandbox(runtime).run_python("print(1)\n", 5.0)


@pytest.mark.parametrize("exit_code", [125, 126, 127])
def test_same_exit_code_without_the_marker_is_an_ordinary_failure(exit_code: int) -> None:
    """モデルのコードが自分で `sys.exit(127)` を呼んだ場合は、不正解として数える。

    実行環境の失敗には、必ず実行環境自身の印 (`docker: ...` など) が付く。
    印がなければモデルの側の終了なので、分母から外さない (正確さを水増ししない)。
    """
    runtime = FakeRuntime(processes=[FakeProcess(returncode=exit_code)])
    result = make_sandbox(runtime).run_python(f"import sys\nsys.exit({exit_code})\n", 5.0)
    assert result.passed is False
    assert result.timed_out is False
    assert result.exit_code == exit_code


# --- 5. 出力の上限 -----------------------------------------------------------


def test_stderr_tail_is_bounded_for_a_program_that_prints_50mb() -> None:
    """50 MB を吐くプログラムでも、覚えておくのは末尾だけ。"""
    process = FakeProcess(returncode=1, stderr_chunk=b"E" * 1_000_000, stderr_repeat=50)
    runtime = FakeRuntime(processes=[process])
    result = make_sandbox(runtime).run_python("print(1)\n", 5.0)
    assert len(result.stderr_tail.encode("utf-8")) <= STDERR_TAIL_BYTES
    # 読み取りは、必ず大きさを指定して少しずつ行う (read() を丸呑みしない)
    assert process.stderr.read_sizes
    assert all(0 < size <= 1024 * 1024 for size in process.stderr.read_sizes)


def test_stderr_tail_keeps_the_end_not_the_beginning() -> None:
    process = FakeProcess(returncode=1, stderr_chunk=b"A" * 4096 + b"ZZZ-LAST", stderr_repeat=1)
    runtime = FakeRuntime(processes=[process])
    result = make_sandbox(runtime).run_python("print(1)\n", 5.0)
    assert result.stderr_tail.endswith("ZZZ-LAST")


# --- 6. 使えるかどうかの判定 --------------------------------------------------


def test_available_is_true_and_cached() -> None:
    runtime = FakeRuntime()
    sandbox = make_sandbox(runtime)
    assert sandbox.available() is True
    assert sandbox.available() is True
    assert sandbox.available() is True
    versions = [argv for argv in runtime.commands if argv[1:2] == ["version"]]
    assert len(versions) == 1


def test_missing_runtime_is_unavailable_with_a_reason() -> None:
    runtime = FakeRuntime(installed=())
    status = make_sandbox(runtime).available()
    assert isinstance(status, SandboxUnavailable)
    assert status.reason


def test_auto_looks_for_podman_then_docker() -> None:
    runtime = FakeRuntime(installed=("docker",))
    sandbox = make_sandbox(runtime, make_settings(runtime="auto"))
    assert sandbox.available() is True
    assert runtime.which_calls == ["podman", "docker"]
    assert runtime.commands[0][0] == "/usr/local/bin/docker"


def test_explicit_runtime_does_not_fall_back_to_another_one() -> None:
    runtime = FakeRuntime(installed=("docker",))
    sandbox = make_sandbox(runtime, make_settings(runtime="podman"))
    status = sandbox.available()
    assert isinstance(status, SandboxUnavailable)
    assert runtime.which_calls == ["podman"]


def test_daemon_that_does_not_answer_is_unavailable() -> None:
    runtime = FakeRuntime(version_code=1)
    status = make_sandbox(runtime).available()
    assert isinstance(status, SandboxUnavailable)
    assert runtime.started == []


def test_missing_image_is_unavailable_and_nothing_is_pulled() -> None:
    runtime = FakeRuntime(inspect_code=1)
    status = make_sandbox(runtime).available()
    assert isinstance(status, SandboxUnavailable)
    assert "Dockerfile" in status.reason
    assert all(argv[1:2] != ["pull"] for argv in runtime.commands)
    assert runtime.started == []


def test_image_id_mismatch_is_unavailable_and_nothing_runs() -> None:
    """手元のイメージの識別子が合わなければ、使えないものとして扱う。"""
    runtime = FakeRuntime(inspect_id="sha256:" + "b" * 64)
    sandbox = make_sandbox(runtime)
    status = sandbox.available()
    assert isinstance(status, SandboxUnavailable)
    assert "image_digest" in status.reason
    with pytest.raises(SandboxError):
        sandbox.run_python("print(1)\n", 5.0)
    assert runtime.started == []


def test_image_id_comparison_is_exact() -> None:
    """前方一致では通さない (短い識別子で紛れないようにする)。"""
    runtime = FakeRuntime(inspect_id=PINNED_ID + "0")
    assert isinstance(make_sandbox(runtime).available(), SandboxUnavailable)


@pytest.mark.parametrize(
    "digest",
    ["", "sha256:zz", "f" * 64, "sha256:" + "A" * 64, "sha256:" + "a" * 63],
)
def test_malformed_image_digest_is_unavailable(digest: str) -> None:
    """設定の値の形が想定外なら、黙って無視せずに使えないものにする (note 2.7)。"""
    runtime = FakeRuntime()
    status = make_sandbox(runtime, make_settings(image_digest=digest)).available()
    assert isinstance(status, SandboxUnavailable)
    assert runtime.started == []


@pytest.mark.parametrize("image", ["", "   ", "--privileged", "-v/:/host"])
def test_malformed_image_name_is_unavailable(image: str) -> None:
    runtime = FakeRuntime()
    status = make_sandbox(runtime, make_settings(image=image, image_digest=None)).available()
    assert isinstance(status, SandboxUnavailable)
    assert runtime.started == []


def test_config_cannot_name_an_unknown_runtime() -> None:
    """設定の段階で、知らない実行環境の名前は弾かれる。"""
    with pytest.raises(ValueError, match="runtime"):
        make_settings(runtime="not-a-runtime")


# --- 7. 隔離なしでは動かさない ------------------------------------------------


def test_run_python_refuses_when_the_runtime_is_unavailable() -> None:
    """実行環境がないときに、手元でコードを動かす逃げ道はない。"""
    runtime = FakeRuntime(installed=())
    sandbox = make_sandbox(runtime)
    with pytest.raises(SandboxError):
        sandbox.run_python("print('host')\n", 5.0)
    assert runtime.started == []
    assert runtime.commands == []


def _module_tree(module: object) -> ast.Module:
    path = Path(str(module.__file__))  # type: ignore[attr-defined]
    return ast.parse(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("module", [sandbox_module, code_module], ids=["sandbox", "code"])
def test_scoring_modules_never_execute_untrusted_code_in_process(module: object) -> None:
    """`exec` / `eval` などで、モデルのコードを手元で動かす道がないこと。"""
    tree = _module_tree(module)
    banned_calls = {"exec", "eval", "compile", "__import__"}
    banned_attrs = {
        "system",
        "popen",
        "execv",
        "execl",
        "execvp",
        "spawnv",
        "fork",
        "call",
        "check_call",
        "check_output",
        "getoutput",
        "getstatusoutput",
    }
    plain_calls: set[str] = set()
    attr_calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                plain_calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                attr_calls.add(node.func.attr)
    assert banned_calls.isdisjoint(plain_calls)
    assert banned_attrs.isdisjoint(attr_calls)


def test_sandbox_module_touches_subprocess_only_through_the_two_defaults() -> None:
    """`subprocess` の使い方を、既定の 2 つの口に閉じ込める。"""
    tree = _module_tree(sandbox_module)
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "subprocess"
    }
    assert used <= {"run", "Popen", "PIPE", "DEVNULL", "TimeoutExpired"}


def test_the_python_command_appears_only_in_the_argv_builder() -> None:
    """`python` を起動する文字列が、コンテナの引数の組み立て以外に現れない。"""
    tree = _module_tree(sandbox_module)
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_run_argv"
    )
    inside = {id(node) for node in ast.walk(builder)}
    outside = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in inside
    ]
    assert "python" not in outside
    assert "-I" not in outside


def test_code_module_does_not_import_the_forbidden_layers() -> None:
    """採点は corpus / client / suites / runner / analysis を読み込まない。"""
    for module in (sandbox_module, code_module):
        tree = _module_tree(module)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        forbidden = {
            "bench_harness.corpus",
            "bench_harness.client",
            "bench_harness.suites",
            "bench_harness.runner",
            "bench_harness.analysis",
            "bench_harness.config",
            "bench_harness.store",
        }
        assert not any(name.startswith(tuple(forbidden)) for name in imported)
