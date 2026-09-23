"""`ops/spark-power-caps/` と、`serving/payload/spark-power-caps.default` の試験。

`spark-power-caps-apply` は、実機の sysfs と `nvidia-smi`/`cpupower` の道筋を、コマンド
ライン引数 (`--sysfs`、`--nvidia-smi`、`--cpupower`) で差し替えられるので、偽の sysfs と、
呼び出しを記録する偽のコマンドで、実機なしに試す。差し替え口は引数だけで、環境変数
(`SPARK_POWER_CAPS_*`) や `PATH` では差し替えられないことも確かめる。実行は `sh` で
subprocess により行う (bats などは足さない)。

確かめること (計画の完了契約 C5〜C10、および issue #10 の修正計画):

- `apply` は、値の検査に 1 つでも外れると、`nvidia-smi -lgc` も `cpupower` も呼ばずに、
  理由を標準エラーに出して 1 で終わる。単位付きの `3000MHz`、空の要素を含む CPU の集合、
  先頭が 0 の値、桁数が大きすぎる値は受けない
- 検査を通る値は、`nvidia-smi --query-gpu=clocks.max.sm …`、`nvidia-smi -lgc <範囲>`、
  `cpupower -c <集合> frequency-set -u <上限>MHz` をすべて呼び、0 で終わる (`nvidia-smi` と
  `cpupower` は別々の記録に残るため、両者の前後関係は確かめない)
- 環境変数や `PATH` でコマンド・sysfs を注入しても、引数で渡した差し替え口だけが使われる
- `reset` は、値の環境変数が壊れていても、`nvidia-smi -rgc` と、sysfs から決めた X925 への
  `cpupower frequency-set -u <cpuinfo_max_freq>kHz` で既定に戻す
- 未知の引数は、使い方を出して 2 で終わる
- unit ファイル (`spark-power-caps.service`) は、`Type=oneshot`、`RemainAfterExit=yes`、
  `EnvironmentFile=/etc/default/spark-power-caps` (`-` なし)、`ExecStart`/`ExecStop`、
  `WantedBy=multi-user.target`、GPU ドライバ準備を待つ `After=`、動的ローダーに効く代表的な
  環境変数を外す `UnsetEnvironment=` を持つ
- `spark-power-caps.default` は、`EnvironmentFile` の書式で読めて、3 つの値と根拠の記録を
  持つ。その値のまま `apply` が通る
- sudoers は、固定形の `install` (写し先を root だけが読めるモード) と `systemctl restart`
  の 2 つだけを NOPASSWD で許す。`visudo -cf` があれば、それでも検査する (なければ skip)

試験用の sysfs の値は、X925 (5-9、15-19) を `cpuinfo_max_freq = 3900000`、A725 を
`2808000`、`cpuinfo_min_freq = 1000000` にした見本である。実機の見本は未採取で、実機の
確かめはこの試験の対象外である。
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

SERVING_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = SERVING_DIR.parent
OPS_DIR = REPO_ROOT / "ops" / "spark-power-caps"
APPLY_PATH = OPS_DIR / "spark-power-caps-apply"
UNIT_PATH = OPS_DIR / "spark-power-caps.service"
SUDOERS_PATH = OPS_DIR / "spark-power-caps.sudoers"
DEFAULT_PATH = SERVING_DIR / "payload" / "spark-power-caps.default"
NODES_PATH = SERVING_DIR / "config" / "nodes.toml"

"""既知の事実から組み立てた、見本の sysfs と値の値 (実機の見本は未採取)。"""
X925_MAX_FREQ = 3900000
A725_MAX_FREQ = 2808000
MIN_FREQ = 1000000
X925_CORES = frozenset({5, 6, 7, 8, 9, 15, 16, 17, 18, 19})
CORES = tuple(range(20))


def _make_sysfs(root: Path, *, x925_max: int = X925_MAX_FREQ) -> Path:
    """試験用の sysfs (`cpu<N>/cpufreq/{cpuinfo_min_freq,cpuinfo_max_freq}`) を作る。"""
    for core in CORES:
        cpufreq = root / f"cpu{core}" / "cpufreq"
        cpufreq.mkdir(parents=True)
        (cpufreq / "cpuinfo_min_freq").write_text(f"{MIN_FREQ}\n", encoding="utf-8")
        (cpufreq / "cpuinfo_max_freq").write_text(
            f"{x925_max if core in X925_CORES else A725_MAX_FREQ}\n", encoding="utf-8"
        )
    return root


def _make_bin(root: Path) -> tuple[Path, Path]:
    """呼び出しを記録する偽の `nvidia-smi` と `cpupower` を作る。

    `nvidia-smi` は、`--query-gpu=clocks.max.sm` を含む呼び出しに、`sm_max` ファイルの値を
    返す。それ以外の呼び出し (`-lgc`、`-rgc`) は記録するだけ。`cpupower` は、すべての
    呼び出しを記録する。記録は、呼び出しの引数の列を 1 行にして、追記する。
    """
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$SMI_LOG"\n'
        'case "$*" in *"--query-gpu=clocks.max.sm"*) cat "$SMI_MAX";; esac\n',
        encoding="utf-8",
    )
    cpupower = bin_dir / "cpupower"
    cpupower.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CPUPOWER_LOG"\n', encoding="utf-8")
    nvidia_smi.chmod(0o755)
    cpupower.chmod(0o755)
    return nvidia_smi, cpupower


def _run_apply(
    sysfs: Path,
    bin_dir: Path,
    *,
    mode: str,
    values: dict[str, str],
    sm_max: str = "2160",
    cwd: Path | None = None,
    omit: frozenset[str] = frozenset(),
) -> subprocess.CompletedProcess[str]:
    """`spark-power-caps-apply` を、差し替え口 (コマンドライン引数) と値 (環境変数) を
    渡して `sh` で流す。`sm_max` は、偽の `nvidia-smi` が `--query-gpu=clocks.max.sm` に
    答える値そのもの (値の辞書や環境変数には含めない。呼び出し元の値と受け渡しを一致させる)。
    `cwd` は作業ディレクトリ (既定は呼び出し元のカレントディレクトリのまま)。`omit` は、本番と
    同じ引数なしの条件を試すために省く差し替え口の名前の集合 (`"--sysfs"`・`"--nvidia-smi"`・
    `"--cpupower"`)。省いた差し替え口は、スクリプトの既定の絶対パスがそのまま使われる。
    """
    work = bin_dir.parent
    env = {
        **os.environ,
        "SMI_LOG": str(work / "nvidia-smi.log"),
        "SMI_MAX": str(work / "sm_max"),
        "CPUPOWER_LOG": str(work / "cpupower.log"),
        **values,
    }
    (work / "sm_max").write_text(f"{sm_max}\n", encoding="utf-8")
    argv = ["sh", str(APPLY_PATH)]
    if "--sysfs" not in omit:
        argv += ["--sysfs", str(sysfs)]
    if "--nvidia-smi" not in omit:
        argv += ["--nvidia-smi", str(bin_dir / "nvidia-smi")]
    if "--cpupower" not in omit:
        argv += ["--cpupower", str(bin_dir / "cpupower")]
    argv.append(mode)
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _smi_calls(log: Path) -> list[list[str]]:
    """偽の `nvidia-smi` が受けた引数の列を読む (記録がないときは空)。"""
    if not log.is_file():
        return []
    return [line.split() for line in log.read_text(encoding="utf-8").splitlines()]


def _cpupower_calls(log: Path) -> list[list[str]]:
    if not log.is_file():
        return []
    return [line.split() for line in log.read_text(encoding="utf-8").splitlines()]


DEFAULT_VALUES = {
    "SPARK_GPU_CLOCK_RANGE_MHZ": "300,1800",
    "SPARK_X925_CPUS": "5-9,15-19",
    "SPARK_X925_MAX_MHZ": "3000",
}


# --- apply: 検査に外れた値は、何も変えずに断る ------------------------------------------


def _reject_case(
    sysfs: Path,
    bin_dir: Path,
    values: dict[str, str],
    *,
    sm_max: str = "2160",
) -> subprocess.CompletedProcess[str]:
    """検査に外れた値は、GPU の上限も CPU の上限も変えずに断る。

    SM の最大の検査は `nvidia-smi --query-gpu=clocks.max.sm …` の読み取りを伴うため
    (計画 C5 の契約: `-lgc`/`cpupower` を呼ばないことだけを断りの条件にする)、
    `nvidia-smi` の呼び出し記録の増減ではなく、`-lgc`/`-rgc` を含む呼び出しの不在で判定する。
    """
    work = bin_dir.parent
    result = _run_apply(sysfs, bin_dir, mode="apply", values=values, sm_max=sm_max)
    assert result.returncode == 1, f"終了コード 1 のはず: {result.returncode} ({result.stderr})"
    assert result.stderr.strip(), "理由を標準エラーに出していない"
    smi = _smi_calls(work / "nvidia-smi.log")
    assert not [call for call in smi if call[:1] in (["-lgc"], ["-rgc"])], (
        "断ったのに GPU の上限を変えた"
    )
    assert _cpupower_calls(work / "cpupower.log") == [], "断ったのに cpupower を呼んだ"
    return result


@pytest.fixture
def fake_env(tmp_path: Path) -> tuple[Path, Path]:
    """見本の sysfs と、記録する偽のコマンドの置き場所 (`bin_dir`) を返す。"""
    sysfs = _make_sysfs(tmp_path / "sysfs")
    nvidia_smi, _cpupower = _make_bin(tmp_path)
    return sysfs, nvidia_smi.parent


def test_apply_rejects_a_gpu_range_that_is_not_two_integers(fake_env: Any) -> None:
    sysfs, bin_dir = fake_env
    _reject_case(sysfs, bin_dir, {**DEFAULT_VALUES, "SPARK_GPU_CLOCK_RANGE_MHZ": "300-1800"})


def test_apply_rejects_a_gpu_range_with_no_comma(fake_env: Any) -> None:
    """カンマが無い単一の値 (例: "1800") は、<最小>,<最大> の 2 つの整数ではないので断る。

    `nvidia-smi -lgc` は、単一の値だとその周波数へのクロック固定という別の意味になるため、
    範囲の検査をすり抜けてそのまま渡ってしまうと、要求と異なる操作になる。
    """
    sysfs, bin_dir = fake_env
    _reject_case(sysfs, bin_dir, {**DEFAULT_VALUES, "SPARK_GPU_CLOCK_RANGE_MHZ": "1800"})


def test_apply_rejects_an_inverted_gpu_range(fake_env: Any) -> None:
    sysfs, bin_dir = fake_env
    _reject_case(sysfs, bin_dir, {**DEFAULT_VALUES, "SPARK_GPU_CLOCK_RANGE_MHZ": "1800,300"})


def test_apply_rejects_a_gpu_max_above_the_sm_maximum(fake_env: Any) -> None:
    """GPU の最大が、`nvidia-smi` が答える SM の最大 (ここでは 1700) を超えたら断る。"""
    sysfs, bin_dir = fake_env
    _reject_case(
        sysfs, bin_dir, {**DEFAULT_VALUES, "SPARK_GPU_CLOCK_RANGE_MHZ": "300,1800"}, sm_max="1700"
    )


def test_apply_rejects_a_cpu_set_outside_the_x925_cores(fake_env: Any) -> None:
    """X925 以外のコア (0) を含む CPU の集合は断る。"""
    sysfs, bin_dir = fake_env
    _reject_case(sysfs, bin_dir, {**DEFAULT_VALUES, "SPARK_X925_CPUS": "5-9,15-19,0"})


def test_apply_rejects_a_cpu_set_with_a_semicolon(fake_env: Any) -> None:
    """`cpupower -c` の書式 (カンマ区切り) 以外の書式 (`;`) は断る。"""
    sysfs, bin_dir = fake_env
    _reject_case(sysfs, bin_dir, {**DEFAULT_VALUES, "SPARK_X925_CPUS": "5-9;15-19"})


def test_apply_rejects_an_empty_cpu_set(fake_env: Any) -> None:
    sysfs, bin_dir = fake_env
    _reject_case(sysfs, bin_dir, {**DEFAULT_VALUES, "SPARK_X925_CPUS": ""})


def test_apply_rejects_a_cpu_cap_above_the_max_freq(fake_env: Any) -> None:
    """X925 の上限が、`cpuinfo_max_freq` (3900000 kHz = 3900 MHz) を超えたら断る。"""
    sysfs, bin_dir = fake_env
    _reject_case(sysfs, bin_dir, {**DEFAULT_VALUES, "SPARK_X925_MAX_MHZ": "4000"})


def test_apply_rejects_a_cpu_cap_with_a_unit(fake_env: Any) -> None:
    """単位付きの `3000MHz` は、整数の MHz ではないので断る。"""
    sysfs, bin_dir = fake_env
    _reject_case(sysfs, bin_dir, {**DEFAULT_VALUES, "SPARK_X925_MAX_MHZ": "3000MHz"})


def test_apply_rejects_a_cpu_cap_below_the_min_freq(fake_env: Any) -> None:
    """X925 の上限が、`cpuinfo_min_freq` (1000000 kHz = 1000 MHz) を下回ったら断る。"""
    sysfs, bin_dir = fake_env
    _reject_case(sysfs, bin_dir, {**DEFAULT_VALUES, "SPARK_X925_MAX_MHZ": "500"})


@pytest.mark.parametrize(
    "cpus",
    [",5-9", "5-9,,15-19", "5-9,15-19,", ","],
    ids=["leading_comma", "middle_comma", "trailing_comma", "comma_only"],
)
def test_apply_rejects_a_cpu_set_with_an_empty_element(fake_env: Any, cpus: str) -> None:
    """先頭・途中・末尾に空の要素がある、またはカンマだけの値は、`cpupower -c` の書式でない
    ので断る。空の要素は、分けたあとの要素ごとの検査だけでは見つけられない (途中の空の要素は
    `""` になって検査を素通りし、末尾のカンマは要素そのものを作らない)。"""
    sysfs, bin_dir = fake_env
    _reject_case(sysfs, bin_dir, {**DEFAULT_VALUES, "SPARK_X925_CPUS": cpus})


@pytest.mark.parametrize(
    "mhz",
    ["04000", "18446744073710552"],
    ids=["leading_zero", "overflowing_digit_count"],
)
def test_apply_rejects_a_cpu_cap_that_breaks_the_arithmetic_expansion(
    fake_env: Any, mhz: str
) -> None:
    """先頭が 0 の値は、続く算術展開 (`$((...))`) で 8 進数として解釈され、検査した値と
    cpupower にそのまま渡る元の文字列がずれる。桁数が大きい値は算術展開で桁があふれる。
    どちらも、検査した値と反映する値の一致を保てないので断る。"""
    sysfs, bin_dir = fake_env
    _reject_case(sysfs, bin_dir, {**DEFAULT_VALUES, "SPARK_X925_MAX_MHZ": mhz})


def test_apply_rejects_a_cpu_set_with_a_glob_character(fake_env: Any, tmp_path: Path) -> None:
    """`set -- $cpus` は引用符なしの展開なので、単語分割だけでなくパス名展開 (グロブ) も
    働く。`*`・`?`・`[...]` を許したまま分けると、実行時のカレントディレクトリの中身が
    意図せず要素に混ざりうる (実際に確かめた: カレントディレクトリに数字だけの名前がある
    と、そのまま要素として展開される)。分ける前の文字クラスの検査で、値だけを理由に断る
    ことを、グロブがマッチする名前を用意した作業ディレクトリで確かめる。"""
    sysfs, bin_dir = fake_env
    work = bin_dir.parent
    cwd = tmp_path / "cwd"
    (cwd / "5").mkdir(parents=True)
    (cwd / "9").mkdir()

    result = _run_apply(
        sysfs,
        bin_dir,
        mode="apply",
        values={**DEFAULT_VALUES, "SPARK_X925_CPUS": "5-9,*"},
        cwd=cwd,
    )

    assert result.returncode == 1, f"終了コード 1 のはず: {result.returncode} ({result.stderr})"
    assert result.stderr.strip(), "理由を標準エラーに出していない"
    assert _cpupower_calls(work / "cpupower.log") == [], "断ったのに cpupower を呼んだ"


# --- apply: 検査を通ると反映する --------------------------------------------------------


def test_apply_sets_both_caps_when_the_values_pass(fake_env: Any) -> None:
    sysfs, bin_dir = fake_env
    work = bin_dir.parent
    result = _run_apply(sysfs, bin_dir, mode="apply", values=DEFAULT_VALUES)

    assert result.returncode == 0, result.stderr
    smi = _smi_calls(work / "nvidia-smi.log")
    cpupower = _cpupower_calls(work / "cpupower.log")
    # SM の最大の読み取りと、GPU・CPU の上限の反映が、それぞれの記録にある
    assert any("--query-gpu=clocks.max.sm" in " ".join(call) for call in smi)
    assert ["-lgc", "300,1800"] in smi, smi
    assert [
        "-c",
        "5-9,15-19",
        "frequency-set",
        "-u",
        "3000MHz",
    ] in cpupower, cpupower


def test_apply_accepts_a_gpu_max_equal_to_the_sm_maximum(fake_env: Any) -> None:
    """GPU の最大が SM の最大と等しい境界 (`<=`) は、断らずに反映する
    (比較を `-lt` と取り違えると、この境界で誤って断ってしまう)。"""
    sysfs, bin_dir = fake_env
    work = bin_dir.parent
    result = _run_apply(
        sysfs,
        bin_dir,
        mode="apply",
        values={**DEFAULT_VALUES, "SPARK_GPU_CLOCK_RANGE_MHZ": "300,2160"},
        sm_max="2160",
    )

    assert result.returncode == 0, result.stderr
    smi = _smi_calls(work / "nvidia-smi.log")
    assert ["-lgc", "300,2160"] in smi, smi


def test_apply_accepts_a_cpu_cap_equal_to_the_max_freq(fake_env: Any) -> None:
    """CPU の上限が `cpuinfo_max_freq` (3900 MHz) と等しい境界 (`<=`) は、断らずに反映する
    (比較を `-lt` と取り違えると、この境界で誤って断ってしまう)。"""
    sysfs, bin_dir = fake_env
    work = bin_dir.parent
    result = _run_apply(
        sysfs, bin_dir, mode="apply", values={**DEFAULT_VALUES, "SPARK_X925_MAX_MHZ": "3900"}
    )

    assert result.returncode == 0, result.stderr
    cpupower = _cpupower_calls(work / "cpupower.log")
    assert [
        "-c",
        "5-9,15-19",
        "frequency-set",
        "-u",
        "3900MHz",
    ] in cpupower, cpupower


def test_apply_accepts_a_cpu_cap_equal_to_the_min_freq(fake_env: Any) -> None:
    """CPU の上限が `cpuinfo_min_freq` (1000 MHz) と等しい境界 (`>=`) は、断らずに反映する
    (比較を `-gt` と取り違えると、この境界で誤って断ってしまう)。"""
    sysfs, bin_dir = fake_env
    work = bin_dir.parent
    result = _run_apply(
        sysfs, bin_dir, mode="apply", values={**DEFAULT_VALUES, "SPARK_X925_MAX_MHZ": "1000"}
    )

    assert result.returncode == 0, result.stderr
    cpupower = _cpupower_calls(work / "cpupower.log")
    assert [
        "-c",
        "5-9,15-19",
        "frequency-set",
        "-u",
        "1000MHz",
    ] in cpupower, cpupower


def test_reset_restores_the_defaults_without_the_env_values(fake_env: Any) -> None:
    """`reset` は、値の環境変数が壊れていても動く。GPU は `-rgc` で、X925 は sysfs の
    `cpuinfo_max_freq` で既定に戻す。"""
    sysfs, bin_dir = fake_env
    work = bin_dir.parent
    broken = {
        "SPARK_GPU_CLOCK_RANGE_MHZ": "not-a-range",
        "SPARK_X925_CPUS": "not-a-set",
        "SPARK_X925_MAX_MHZ": "not-a-number",
    }
    result = _run_apply(sysfs, bin_dir, mode="reset", values=broken)

    assert result.returncode == 0, result.stderr
    smi = _smi_calls(work / "nvidia-smi.log")
    cpupower = _cpupower_calls(work / "cpupower.log")
    assert ["-rgc"] in smi, smi
    assert [
        "-c",
        "5,6,7,8,9,15,16,17,18,19",
        "frequency-set",
        "-u",
        "3900000kHz",
    ] in cpupower, cpupower


def _make_injected_bin(root: Path, marker: Path) -> Path:
    """環境変数や `PATH` で注入する、呼ばれたら印を残して失敗するだけの偽のコマンドを作る。

    差し替え口が引数だけであることを確かめるため、`nvidia-smi`・`cpupower` に加えて、
    スクリプトが内部で使う補助コマンド (`cat`・`head`・`tr`・`sort`) も用意する。
    """
    bin_dir = root / "injected-bin"
    bin_dir.mkdir(parents=True)
    for name in ("nvidia-smi", "cpupower", "cat", "head", "tr", "sort"):
        script = bin_dir / name
        script.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 1\n', encoding="utf-8")
        script.chmod(0o755)
    return bin_dir


def test_apply_ignores_env_injected_commands_sysfs_and_path(fake_env: Any, tmp_path: Path) -> None:
    """環境変数 (`SPARK_POWER_CAPS_*`) と `PATH` で注入したコマンド・sysfs は使われない。
    引数で渡した差し替え口だけが使われる (SCN-FU-A-N1)。"""
    sysfs, bin_dir = fake_env
    work = bin_dir.parent
    marker = tmp_path / "marker"
    injected_bin = _make_injected_bin(tmp_path, marker)
    missing_sysfs = tmp_path / "no-such-sysfs"

    result = _run_apply(
        sysfs,
        bin_dir,
        mode="apply",
        values={
            "SPARK_POWER_CAPS_NVIDIA_SMI": str(injected_bin / "nvidia-smi"),
            "SPARK_POWER_CAPS_CPUPOWER": str(injected_bin / "cpupower"),
            "SPARK_POWER_CAPS_CPU_SYSFS": str(missing_sysfs),
            "PATH": f"{injected_bin}{os.pathsep}{os.environ.get('PATH', '')}",
            **DEFAULT_VALUES,
        },
    )

    assert result.returncode == 0, result.stderr
    assert not marker.exists(), "環境変数や PATH で注入したコマンドが呼ばれた"
    smi = _smi_calls(work / "nvidia-smi.log")
    assert ["-lgc", "300,1800"] in smi, smi


def test_reset_ignores_env_injected_commands_sysfs_and_path(fake_env: Any, tmp_path: Path) -> None:
    """`reset` でも、環境変数と `PATH` の注入が働かない (SCN-FU-A-N1 の reset 版)。"""
    sysfs, bin_dir = fake_env
    work = bin_dir.parent
    marker = tmp_path / "marker"
    injected_bin = _make_injected_bin(tmp_path, marker)
    missing_sysfs = tmp_path / "no-such-sysfs"

    result = _run_apply(
        sysfs,
        bin_dir,
        mode="reset",
        values={
            "SPARK_POWER_CAPS_NVIDIA_SMI": str(injected_bin / "nvidia-smi"),
            "SPARK_POWER_CAPS_CPUPOWER": str(injected_bin / "cpupower"),
            "SPARK_POWER_CAPS_CPU_SYSFS": str(missing_sysfs),
            "PATH": f"{injected_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        },
    )

    assert result.returncode == 0, result.stderr
    assert not marker.exists(), "環境変数や PATH で注入したコマンドが呼ばれた"
    smi = _smi_calls(work / "nvidia-smi.log")
    assert ["-rgc"] in smi, smi


# --- 引数を省いた、本番と同じ条件での注入の拒否 (unit は差し替え口の引数を渡さない) --------

_REAL_NVIDIA_SMI_EXISTS = Path("/usr/bin/nvidia-smi").exists()
_REAL_CPUPOWER_EXISTS = Path("/usr/bin/cpupower").exists()
_DEFAULT_COMMAND_PATHS = {
    "--nvidia-smi": "/usr/bin/nvidia-smi",
    "--cpupower": "/usr/bin/cpupower",
}


@pytest.mark.parametrize(
    ("omit_flag", "env_key", "skip_if_real_exists"),
    [
        pytest.param(
            "--nvidia-smi", "SPARK_POWER_CAPS_NVIDIA_SMI", _REAL_NVIDIA_SMI_EXISTS, id="nvidia_smi"
        ),
        pytest.param(
            "--cpupower", "SPARK_POWER_CAPS_CPUPOWER", _REAL_CPUPOWER_EXISTS, id="cpupower"
        ),
    ],
)
@pytest.mark.parametrize("mode", ["apply", "reset"], ids=["apply", "reset"])
def test_the_default_command_is_used_when_its_argument_is_omitted(
    fake_env: Any,
    tmp_path: Path,
    omit_flag: str,
    env_key: str,
    skip_if_real_exists: bool,
    mode: str,
) -> None:
    """unit と同じ、差し替え口の引数を渡さない条件では、既定の絶対パス (`/usr/bin/nvidia-smi`
    または `/usr/bin/cpupower`) だけが使われる。`EnvironmentFile` 経由で書ける
    `SPARK_POWER_CAPS_*` を注入しても、コマンドは差し替わらない (SCN-FU-I-P1)。

    既定の絶対パスが実際に試みられたことを、標準エラーの文言 (存在しないコマンドの実行に
    よる「そのファイルがない」旨のメッセージ) でも確かめる。省いた側の本物のコマンドが
    実機にあると、それを実際に呼んでしまうおそれがあるため、その場合は skip する
    (計画内の調査。ローカル・CI のどちらでも `/usr/bin/nvidia-smi`・`/usr/bin/cpupower` は
    確認できておらず、この skip 条件がその不確実性を吸収する)。
    """
    if skip_if_real_exists:
        pytest.skip(f"実機の本物のコマンドがある環境なので、{omit_flag} を省く条件は skip する")
    sysfs, bin_dir = fake_env
    marker = tmp_path / "marker"
    injected_bin = _make_injected_bin(tmp_path, marker)
    injected_name = omit_flag.removeprefix("--")

    result = _run_apply(
        sysfs,
        bin_dir,
        mode=mode,
        values={**DEFAULT_VALUES, env_key: str(injected_bin / injected_name)},
        omit=frozenset({omit_flag}),
    )

    assert not marker.exists(), (
        f"{env_key} で注入したコマンドが呼ばれた (returncode={result.returncode}, "
        f"stderr={result.stderr!r})"
    )
    assert _DEFAULT_COMMAND_PATHS[omit_flag] in result.stderr, (
        f"既定の絶対パスを試みた形跡が標準エラーにない: {result.stderr!r}"
    )


def _real_default_sysfs_has_no_cpufreq() -> bool:
    """このホストの本物の既定 sysfs (`/sys/devices/system/cpu`) に、`cpufreq` を持つコアが
    1 つもないかどうか。ないなら `find_x925` は必ず `cpuinfo_max_freq を 1 つも読めなかった`
    で断り、失敗の理由に既定の絶対パスがそのまま出ることを保証できる (macOS や、cpufreq を
    公開しない Linux の仮想マシンで真になる)。ある場合は、実機のコア数・グルーピング次第で
    どの検査で断るか (または断らずに実際の値で通るか) が変わるため、この保証はできない。"""
    return not glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq")


def test_apply_ignores_the_env_sysfs_when_its_argument_is_omitted(
    fake_env: Any, tmp_path: Path
) -> None:
    """`--sysfs` を省くと、`SPARK_POWER_CAPS_CPU_SYSFS` に注入した sysfs から決まる引数
    (実機の値と重ならない見本 `x925_max=9900000`) が `cpupower` に渡らない (SCN-FU-I-P2)。
    `--sysfs` を省く条件は読み取りだけなので、本物のコマンドを呼ぶおそれがなく skip しない。

    このホストに本物の cpufreq がなければ (`_real_default_sysfs_has_no_cpufreq`)、失敗の
    理由に既定の絶対パスがそのまま出ることも確かめる。本物の cpufreq があるホスト (実機や、
    それに近い CI の runner) では、コア数やグルーピング次第でどの検査が先に断るか (あるいは
    断らずに実際の値で通るか) が変わり、失敗の文言に既定の絶対パスが出るとは限らないため、
    この追加の確認はしない (注入した値が使われないことは、どちらの場合も変わらず確かめる)。
    """
    sysfs, bin_dir = fake_env
    work = bin_dir.parent
    injected_sysfs = _make_sysfs(tmp_path / "injected-sysfs", x925_max=9_900_000)

    result = _run_apply(
        sysfs,
        bin_dir,
        mode="apply",
        values={
            **DEFAULT_VALUES,
            "SPARK_X925_MAX_MHZ": "9900",
            "SPARK_POWER_CAPS_CPU_SYSFS": str(injected_sysfs),
        },
        omit=frozenset({"--sysfs"}),
    )

    cpupower = _cpupower_calls(work / "cpupower.log")
    assert not any("9900MHz" in " ".join(call) for call in cpupower), (
        f"注入した sysfs から決まる引数が cpupower に渡った (returncode={result.returncode}): "
        f"{cpupower}"
    )
    if _real_default_sysfs_has_no_cpufreq():
        assert "/sys/devices/system/cpu" in result.stderr, (
            "cpufreq のない環境のはずが、既定の絶対パスを試みた形跡が標準エラーにない: "
            f"{result.stderr!r}"
        )


def test_reset_ignores_the_env_sysfs_when_its_argument_is_omitted(
    fake_env: Any, tmp_path: Path
) -> None:
    """`reset` でも、`--sysfs` を省くと注入した sysfs の値が使われない (SCN-FU-I-P2 の
    reset 版)。cpufreq のないホストでの追加確認は apply 版と同じ (docstring 参照)。"""
    sysfs, bin_dir = fake_env
    work = bin_dir.parent
    injected_sysfs = _make_sysfs(tmp_path / "injected-sysfs", x925_max=9_900_000)

    result = _run_apply(
        sysfs,
        bin_dir,
        mode="reset",
        values={"SPARK_POWER_CAPS_CPU_SYSFS": str(injected_sysfs)},
        omit=frozenset({"--sysfs"}),
    )

    cpupower = _cpupower_calls(work / "cpupower.log")
    assert not any("9900000kHz" in " ".join(call) for call in cpupower), (
        f"注入した sysfs から決まる引数が cpupower に渡った (returncode={result.returncode}): "
        f"{cpupower}"
    )
    if _real_default_sysfs_has_no_cpufreq():
        assert "/sys/devices/system/cpu" in result.stderr, (
            "cpufreq のない環境のはずが、既定の絶対パスを試みた形跡が標準エラーにない: "
            f"{result.stderr!r}"
        )


def test_an_unknown_mode_prints_a_usage_and_exits_two(fake_env: Any) -> None:
    sysfs, bin_dir = fake_env
    result = _run_apply(sysfs, bin_dir, mode="apply2", values=DEFAULT_VALUES)

    assert result.returncode == 2, result.stderr
    assert result.stderr.strip(), "使い方を出していない"


# --- unit ファイル -------------------------------------------------------------------


def test_the_unit_file_has_the_oneshot_shape() -> None:
    """unit は、oneshot の形と、GPU ドライバの準備を待つ依存関係を持つ。

    `EnvironmentFile` に `-` (値がなくても静かに成功) を付けないことと、`Wants=` で
    停止中のサービスを起こさないことも固定する。
    """
    text = UNIT_PATH.read_text(encoding="utf-8")
    service = text.split("[Service]")[1]
    install = text.split("[Install]")[1]
    unit = text.split("[Unit]")[1].split("[Service]")[0]
    assert "Type=oneshot" in service
    assert "RemainAfterExit=yes" in service
    assert "EnvironmentFile=/etc/default/spark-power-caps" in service
    assert "EnvironmentFile=-/etc/default/spark-power-caps" not in service
    assert "ExecStart=/usr/local/sbin/spark-power-caps-apply apply" in service
    assert "ExecStop=/usr/local/sbin/spark-power-caps-apply reset" in service
    assert "WantedBy=multi-user.target" in install
    unit_lines = [
        line.strip()
        for line in unit.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any(line.startswith("Wants=") for line in unit_lines), (
        "設定行に Wants= がある (コメントの字面ではなく、実際の設定を検査する)"
    )
    assert any(line.startswith("After=") for line in unit_lines)
    # 根拠のコメントが付いていること ([Unit] 節にコメントがある)
    comments = [line for line in unit.splitlines() if line.strip().startswith("#")]
    assert comments, "依存関係の根拠を書いたコメントがない"


def test_the_unit_file_unsets_dynamic_loader_variables() -> None:
    """`[Service]` の、コメントでない `UnsetEnvironment=` の行が、動的ローダーに効く代表的な
    変数を含む。`EnvironmentFile` で渡された値が、これらの名前で root のプロセスに影響しない
    ようにする。"""
    text = UNIT_PATH.read_text(encoding="utf-8")
    service = text.split("[Service]")[1].split("[Install]")[0]
    service_lines = [
        line.strip()
        for line in service.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    unset_lines = [line for line in service_lines if line.startswith("UnsetEnvironment=")]
    assert unset_lines, "UnsetEnvironment= の設定行がない"
    names: set[str] = set()
    for line in unset_lines:
        names.update(line.split("=", 1)[1].split())
    expected = {"LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "GLIBC_TUNABLES", "PATH"}
    assert expected <= names, names


# --- default ファイル ----------------------------------------------------------------


def test_the_default_file_has_the_adopted_values_and_their_records() -> None:
    """`default` は、EnvironmentFile の書式 (`KEY=VALUE`) で読めて、採用した値と、根拠の
    記録の名前を持つ。戻し方は変数にせずコメントで書く。"""
    text = DEFAULT_PATH.read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "=" in stripped, f"EnvironmentFile の書式でない行: {stripped}"
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    assert values["SPARK_GPU_CLOCK_RANGE_MHZ"] == "300,1800"
    assert values["SPARK_X925_CPUS"] == "5-9,15-19"
    assert values["SPARK_X925_MAX_MHZ"] == "3000"
    assert "2026-09-23-decode-gpu-clock-cap.md" in text
    assert "2026-09-23-thermal-source.md" in text
    assert "-rgc" in text
    assert "cpuinfo_max_freq" in text


def test_the_default_file_values_pass_the_apply_check(tmp_path: Path) -> None:
    """`default` に書いた値は、そのまま `apply` を通る (見本の sysfs で)。"""
    text = DEFAULT_PATH.read_text(encoding="utf-8")
    values = {
        line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#") and "=" in line
    }
    sysfs = _make_sysfs(tmp_path / "sysfs")
    nvidia_smi, _cpupower = _make_bin(tmp_path / "bin")
    result = _run_apply(sysfs, nvidia_smi.parent, mode="apply", values=values)
    assert result.returncode == 0, result.stderr


# --- sudoers ------------------------------------------------------------------------


def test_the_sudoers_file_allows_exactly_the_two_fixed_forms() -> None:
    """sudoers は、固定の `install` (写し先を root だけが読めるモード) と
    `systemctl restart` の 2 つだけを NOPASSWD で許す。

    写し元は、`nodes.toml` の両台の `remote_root` + `/payload/spark-power-caps.default` と
    一致する。コメントではなく、許可の行そのもの (`NOPASSWD:` の前後) を照合の単位にする
    (コメントに固定形を書いてあるだけで、許可の行自体が違う場合を見逃さないため)。
    """
    text = SUDOERS_PATH.read_text(encoding="utf-8")
    commands = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert len(commands) == 1, f"非コメントの行は 1 行だけ: {commands}"
    line = commands[0]

    assert "NOPASSWD:" in line, line
    principal, _, rest = line.partition("NOPASSWD:")
    assert principal.strip() == "j5ik2o ALL=(root)", principal

    nodes = tomllib.loads(NODES_PATH.read_text(encoding="utf-8"))
    roots = sorted({node["remote_root"] for node in nodes["nodes"].values()})
    expected = [
        f"/usr/bin/install -m 0600 -o root -g root {root}/payload/spark-power-caps.default"
        " /etc/default/spark-power-caps"
        for root in roots
    ]
    expected.append("/usr/bin/systemctl restart spark-power-caps.service")
    allowed = [item.strip() for item in rest.split(",")]
    assert allowed == expected, allowed

    # ワイルドカードは書かない (コメントも含めて、ファイル全体で確かめる)
    assert "*" not in text


def test_the_sudoers_file_passes_visudo_when_available() -> None:
    """`visudo` があれば、`visudo -cf` で sudoers の断片を検査する (なければ skip)。"""
    visudo = shutil.which("visudo")
    if visudo is None:
        pytest.skip("visudo がない (CI の runner では実行する)")
    result = subprocess.run(
        [visudo, "-cf", str(SUDOERS_PATH)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
