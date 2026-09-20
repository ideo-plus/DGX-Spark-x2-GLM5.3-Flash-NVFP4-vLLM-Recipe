"""実物のコンテナを使う、隔離の確認 (task 6.6: scoring/sandbox、5.4)。

実行環境 (podman / docker) と、`bench/config/profiles.toml` の `quick` に
書いてあるイメージが手元にないときは、理由を添えて飛ばす。したがって、
コンテナのない機械でも、この試験の入ったまとまりは緑のままになる。

イメージの作り方は `bench/sandbox/Dockerfile` の先頭に書いてある。

すべてのプログラムは、確かめたいことを**標準エラー**に書く。採点の側は
標準出力を捨てる (巨大な出力で手元のメモリを使い切らないため) ので、
標準出力に書いても試験からは見えない。
"""

from __future__ import annotations

import subprocess
import time
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from bench_harness.scoring.code import (
    build_checked_program,
    build_program,
    extract_code,
    score_checked,
    score_code,
)
from bench_harness.scoring.sandbox import CONTAINER_NAME_PREFIX, ContainerSandbox, SandboxError
from bench_harness.types import (
    CodeProblem,
    QualityOutcome,
    QualityVerdict,
    SandboxResult,
    SandboxSettings,
    SandboxUnavailable,
)

PROFILES_TOML = Path(__file__).resolve().parents[2] / "config" / "profiles.toml"
LEFTOVER_POLL_S = 15.0
"""`--rm` の後始末が終わるのを待つ上限 (コンテナが残っていないことの確認)。"""


def _quick_sandbox_settings() -> SandboxSettings:
    """計測で実際に使う設定 (`quick`) を、そのまま読む。"""
    data: dict[str, Any] = tomllib.loads(PROFILES_TOML.read_text(encoding="utf-8"))
    return SandboxSettings.model_validate(data["profiles"]["quick"]["sandbox"])


def _with(settings: SandboxSettings, **overrides: Any) -> SandboxSettings:
    """凍結の型を、検証し直して差し替える (note 1.2)。"""
    data = settings.model_dump()
    data.update(overrides)
    return SandboxSettings.model_validate(data)


@pytest.fixture(scope="module")
def settings() -> SandboxSettings:
    return _quick_sandbox_settings()


@pytest.fixture(scope="module")
def working_runtime(settings: SandboxSettings) -> None:
    """実行環境・守護プロセス・イメージがそろっていなければ、理由を添えて飛ばす。

    設定を差し替えて「使えないこと」を確かめる試験も、**実行環境そのものは
    要る** (イメージがない、ダイジェストが合わない、を実行環境に尋ねるため)。
    その種の試験も、この fixture を通すこと。通さないと、コンテナのない機械で
    「飛ばす」ではなく「落ちる」になる。
    """
    status = ContainerSandbox(settings).available()
    if isinstance(status, SandboxUnavailable):
        pytest.skip(f"隔離の実行環境が使えない: {status.reason}")


@pytest.fixture(scope="module")
def sandbox(settings: SandboxSettings, working_runtime: None) -> ContainerSandbox:
    return ContainerSandbox(settings)


def _leftover_containers(settings: SandboxSettings) -> list[str]:
    """`bench-sbx-` の名前で残っているコンテナの一覧 (実行環境が無ければ空)。"""
    runtime = "docker" if settings.runtime in ("auto", "docker") else settings.runtime
    try:
        completed = subprocess.run(
            [
                runtime,
                "ps",
                "-a",
                "--filter",
                f"name={CONTAINER_NAME_PREFIX}",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - 実行環境がない機械
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _assert_no_leftover_containers(settings: SandboxSettings) -> None:
    """コンテナが 1 つも残っていないこと (`--rm` の後始末を待つ)。"""
    deadline = time.monotonic() + LEFTOVER_POLL_S
    leftovers = _leftover_containers(settings)
    while leftovers and time.monotonic() < deadline:
        time.sleep(0.2)
        leftovers = _leftover_containers(settings)
    assert leftovers == [], f"コンテナが残っている: {leftovers}"


@pytest.fixture(autouse=True, scope="module")
def _no_leftovers_at_the_end(settings: SandboxSettings) -> Iterator[None]:
    yield
    _assert_no_leftover_containers(settings)


# --- 手で書いた課題 ----------------------------------------------------------

CLOSE_ELEMENTS = CodeProblem(
    task_id="HumanEval/0",
    prompt=(
        "from typing import List\n"
        "\n"
        "\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """ Check if any two numbers are closer than the given threshold."""\n'
    ),
    entry_point="has_close_elements",
    test=(
        "import numpy as np\n"
        "\n"
        "\n"
        "def check(candidate):\n"
        "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3)\n"
        "    assert not candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05)\n"
        "    assert np.allclose([1.0], [1.0])\n"
    ),
    canonical_solution=None,
)

CORRECT_ANSWER = (
    "解答です。\n"
    "\n"
    "```python\n"
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    "    for i, a in enumerate(numbers):\n"
    "        for j, b in enumerate(numbers):\n"
    "            if i != j and abs(a - b) < threshold:\n"
    "                return True\n"
    "    return False\n"
    "```\n"
)
"""注釈に `List` を使う、会話の形の正直な正解。`from typing import List` は
指示の側にしかないので、指示を前に置かないと `NameError` で落ちる (note 6.4)。"""

CORRECT_BODY = (
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    "    for i, a in enumerate(numbers):\n"
    "        for j, b in enumerate(numbers):\n"
    "            if i != j and abs(a - b) < threshold:\n"
    "                return True\n"
    "    return False\n"
)

WRONG_BODY = (
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    return False\n"
)

WRONG_ANSWER = "```python\n" + WRONG_BODY + "```\n"


# --- 1. 採点の流れ (取り出し → 組み立て → 実行 → 採点) ----------------------


def _score_answer(sandbox: ContainerSandbox, answer: str) -> tuple[SandboxResult, QualityVerdict]:
    """6.7 が使う道すじ: 取り出し → 印つきの組み立て → 実行 → 採点。"""
    code = extract_code(answer, CLOSE_ELEMENTS.entry_point)
    assert code is not None
    program = build_checked_program(CLOSE_ELEMENTS, code)
    result = sandbox.run_python(program.source, 20.0)
    return result, score_checked(result, program)


def test_honest_correct_answer_is_scored_correct(sandbox: ContainerSandbox) -> None:
    result, verdict = _score_answer(sandbox, CORRECT_ANSWER)
    assert result.passed is True, result.stderr_tail
    assert verdict.outcome is QualityOutcome.CORRECT


def test_honest_correct_answer_also_passes_the_plain_path(sandbox: ContainerSandbox) -> None:
    """印なしの古い道すじ (`build_program` + `score_code`) も、動いたままにする。"""
    code = extract_code(CORRECT_ANSWER, CLOSE_ELEMENTS.entry_point)
    assert code is not None
    result = sandbox.run_python(build_program(CLOSE_ELEMENTS, code), 20.0)
    assert result.passed is True, result.stderr_tail
    assert score_code(result).outcome is QualityOutcome.CORRECT


def test_wrong_answer_is_scored_incorrect(sandbox: ContainerSandbox) -> None:
    result, verdict = _score_answer(sandbox, WRONG_ANSWER)
    assert result.passed is False
    assert result.timed_out is False
    assert result.exit_code == 1
    assert verdict.outcome is QualityOutcome.INCORRECT


@pytest.mark.parametrize(
    ("shape", "escape"),
    [
        ("sys_exit", "import sys\nsys.exit(0)\n"),
        ("os_exit", "import os\nos._exit(0)\n"),
        ("raise_system_exit", "raise SystemExit(0)\n"),
    ],
    ids=["sys_exit", "os_exit", "raise_system_exit"],
)
def test_exiting_before_the_check_is_not_correct(
    sandbox: ContainerSandbox, shape: str, escape: str
) -> None:
    """検査に届かずに終了コード 0 で終わる応答を、正解にしない。"""
    answer = "```python\n" + WRONG_BODY + escape + "```\n"
    result, verdict = _score_answer(sandbox, answer)
    assert result.exit_code == 0
    assert verdict.outcome is QualityOutcome.INCORRECT
    assert "exited_before_check" in verdict.detail


def test_candidate_cannot_forge_the_sentinel(sandbox: ContainerSandbox) -> None:
    """印を知らないまま、それらしい行を書いても正解にならない。"""
    answer = (
        "```python\n"
        + WRONG_BODY
        + 'import os\nos.write(2, b"\\n__bench_ok_" + b"0" * 32 + b"__\\n")\n'
        + "import sys\nsys.exit(0)\n"
        + "```\n"
    )
    _, verdict = _score_answer(sandbox, answer)
    assert verdict.outcome is QualityOutcome.INCORRECT


def test_candidate_that_replaces_stderr_still_reports_the_sentinel(
    sandbox: ContainerSandbox,
) -> None:
    """`sys.stderr` を差し替えられても、記述子 2 に直接書くので届く。"""
    answer = "```python\n" + CORRECT_BODY + "import io, sys\nsys.stderr = io.StringIO()\n" + "```\n"
    result, verdict = _score_answer(sandbox, answer)
    assert result.passed is True, result.stderr_tail
    assert verdict.outcome is QualityOutcome.CORRECT


def test_noisy_stderr_before_the_sentinel_does_not_hide_it(sandbox: ContainerSandbox) -> None:
    """検査の前に大量の標準エラーを吐いても、末尾の印は残る。"""
    answer = (
        "```python\n"
        + CORRECT_BODY
        + "import sys\nsys.stderr.write('noise\\n' * 200000)\nsys.stderr.write('partial')\n"
        + "```\n"
    )
    result, verdict = _score_answer(sandbox, answer)
    assert result.passed is True, result.stderr_tail[-200:]
    assert verdict.outcome is QualityOutcome.CORRECT


def test_large_stdout_does_not_block(sandbox: ContainerSandbox) -> None:
    """標準出力に 1 MB 書くコードでも、詰まらずに終わる (標準出力は捨てる)。"""
    program = "import sys\nsys.stdout.write('o' * (1024 * 1024))\nsys.stdout.flush()\n"
    result = sandbox.run_python(program, 30.0)
    assert result.timed_out is False
    assert result.passed is True, result.stderr_tail


def test_candidate_exit_127_is_an_ordinary_failure(sandbox: ContainerSandbox) -> None:
    """モデルのコードが `sys.exit(127)` を呼んでも、設備の失敗として除外しない。"""
    result = sandbox.run_python("import sys\nsys.exit(127)\n", 20.0)
    assert result.passed is False
    assert result.exit_code == 127
    assert score_code(result).outcome is QualityOutcome.INCORRECT


def test_numpy_is_importable(sandbox: ContainerSandbox) -> None:
    """HumanEval+ の検査のプログラムの 163/164 が numpy を使う (note 6.5)。"""
    result = sandbox.run_python(
        "import numpy as np\nassert np.allclose([1.0, 2.0], [1.0, 2.0])\n", 20.0
    )
    assert result.passed is True, result.stderr_tail


# --- 2. 終わらないコード ------------------------------------------------------


def test_non_terminating_code_times_out_and_leaves_no_container(
    sandbox: ContainerSandbox, settings: SandboxSettings
) -> None:
    """時間切れのあと、コンテナが 1 つも残らないこと (note 6.1)。"""
    result = sandbox.run_python("while True:\n    pass\n", 2.5)
    assert result.timed_out is True
    assert result.passed is False
    _assert_no_leftover_containers(settings)


# --- 3. 隔離 -----------------------------------------------------------------


def test_network_is_unreachable(sandbox: ContainerSandbox) -> None:
    result = sandbox.run_python(
        'import socket\nsocket.create_connection(("1.1.1.1", 53), timeout=3)\n', 20.0
    )
    assert result.passed is False
    assert result.timed_out is False


def test_host_files_are_not_visible(sandbox: ContainerSandbox) -> None:
    program = (
        "import os, sys\n"
        'sys.stderr.write("host_mnt=%r Volumes=%r Users=%r\\n" % ('
        '    os.path.isdir("/host_mnt"), os.path.isdir("/Volumes"), os.path.isdir("/Users")))\n'
        'open("/Users/j5ik2o/.zshrc")\n'
    )
    result = sandbox.run_python(program, 20.0)
    assert result.passed is False
    assert "host_mnt=False" in result.stderr_tail
    assert "Volumes=False" in result.stderr_tail
    assert "Users=False" in result.stderr_tail


def test_writing_outside_tmp_fails(sandbox: ContainerSandbox) -> None:
    result = sandbox.run_python('open("/scratch.txt", "w").write("x")\n', 20.0)
    assert result.passed is False
    assert "Read-only file system" in result.stderr_tail


def test_tmp_stays_writable(sandbox: ContainerSandbox) -> None:
    """検査のプログラムが一時ファイルを使えるように、`/tmp` だけは書ける。"""
    result = sandbox.run_python(
        'open("/tmp/x.txt", "w").write("x")\nassert open("/tmp/x.txt").read() == "x"\n', 20.0
    )
    assert result.passed is True, result.stderr_tail


def test_process_count_is_bounded(sandbox: ContainerSandbox) -> None:
    """`--pids-limit` で、プロセスを増やし続けるコードが止まる。"""
    program = (
        "import subprocess, sys\n"
        "procs = []\n"
        "failures = 0\n"
        "for _ in range(200):\n"
        "    try:\n"
        '        procs.append(subprocess.Popen(["sleep", "5"]))\n'
        "    except Exception:\n"
        "        failures += 1\n"
        "for p in procs:\n"
        "    p.kill()\n"
        "for p in procs:\n"
        "    p.wait()\n"
        'sys.stderr.write("spawned=%d failures=%d\\n" % (len(procs), failures))\n'
        "sys.exit(0 if failures > 0 and len(procs) < 200 else 1)\n"
    )
    result = sandbox.run_python(program, 30.0)
    assert result.passed is True, result.stderr_tail
    assert "failures=" in result.stderr_tail


def test_memory_hog_is_a_failed_answer_not_an_infrastructure_error(
    sandbox: ContainerSandbox,
) -> None:
    """2 GB を確保しようとするコードは、殺されて「不正解」になる (137)。"""
    program = "buf = []\nfor _ in range(2000):\n    buf.append(bytearray(1024 * 1024))\n"
    result = sandbox.run_python(program, 30.0)
    assert result.passed is False
    assert result.timed_out is False
    verdict = score_code(result)
    assert verdict.outcome is QualityOutcome.INCORRECT


def test_stdin_is_the_program_so_input_cannot_hang(sandbox: ContainerSandbox) -> None:
    """プログラムそのものを標準入力で渡すので、`input()` は EOF になる。"""
    result = sandbox.run_python("input()\n", 20.0)
    assert result.passed is False
    assert result.timed_out is False
    assert "EOFError" in result.stderr_tail


def test_stderr_tail_is_bounded_for_a_noisy_program(sandbox: ContainerSandbox) -> None:
    """50 MB を標準エラーに吐くプログラムでも、手元に残すのは末尾だけ。"""
    program = (
        "import sys\n"
        "chunk = b'E' * (1024 * 1024)\n"
        "for _ in range(50):\n"
        "    sys.stderr.buffer.write(chunk)\n"
        "sys.stderr.buffer.write(b'ZZZ-LAST')\n"
        "sys.stderr.buffer.flush()\n"
    )
    result = sandbox.run_python(program, 60.0)
    assert result.passed is True, result.stderr_tail[-200:]
    assert len(result.stderr_tail.encode("utf-8")) <= 2048
    assert result.stderr_tail.endswith("ZZZ-LAST")


# --- 4. 使えないときは、何も動かさない ---------------------------------------


def test_runtime_that_is_not_installed_is_unavailable(settings: SandboxSettings) -> None:
    """この Mac に podman は入れていない (note 6.1)。"""
    box = ContainerSandbox(_with(settings, runtime="podman"))
    status = box.available()
    if status is True:  # pragma: no cover - podman が入っている機械
        pytest.skip("podman が入っているので、存在しない実行環境の確認ができない")
    assert isinstance(status, SandboxUnavailable)
    assert status.reason
    with pytest.raises(SandboxError):
        box.run_python("print(1)\n", 5.0)


def test_image_that_does_not_exist_is_unavailable(
    settings: SandboxSettings, working_runtime: None
) -> None:
    box = ContainerSandbox(
        _with(settings, image="bench-sandbox-does-not-exist:v0", image_digest=None)
    )
    status = box.available()
    assert isinstance(status, SandboxUnavailable)
    assert "Dockerfile" in status.reason


def test_image_digest_mismatch_is_unavailable_and_nothing_runs(
    settings: SandboxSettings, working_runtime: None
) -> None:
    """ダイジェストが合わないイメージでは、1 行も動かさない。"""
    box = ContainerSandbox(_with(settings, image_digest="sha256:" + "0" * 64))
    status = box.available()
    assert isinstance(status, SandboxUnavailable)
    assert "image_digest" in status.reason
    before = _leftover_containers(settings)
    with pytest.raises(SandboxError):
        box.run_python('open("/tmp/should-never-exist", "w")\n', 5.0)
    assert _leftover_containers(settings) == before
