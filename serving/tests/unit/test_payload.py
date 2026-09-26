"""`serving/payload/` の 2 つのスクリプトの試験 (task 4.2)。

`serving/payload/` は `serving_kit` のパッケージではない (Spark にだけ配るので、
`pyproject.toml` のビルド対象に入れていない)。`importlib.util.spec_from_file_location` で、
ファイルの道筋から直接読み込む。

確かめること:

- `allreduce_bench.py` の計算の部分 (大きさの列、`algbw`/`busbw`、統計、JSON の組み立て) が、
  `torch` なし・GPU なしで、既知の入力から既知の値を返す (design.md 「netcheck」の決まり)
- `algbw`/`busbw` の項目名が、`serving_kit.types.BandwidthSample` / `BandwidthRun` に
  写せる形になっている (task 4.4 が写せる形にする、という完了の状態)
- `allreduce_bench.py` を import しただけでは `torch` を import しない・標準出力に何も
  出さないこと (モジュール直下に `torch` の import がないこと、`sys.modules` の両方で確かめる)
- `vllm_sanity_check.py` の先頭のコメントに、出典の URL、40 桁の commit、ライセンス、
  sha256 があり、区切りの行より下の本文の sha256 が、そのコメントの値と一致すること
  (あとで本文が直されたら落ちる)。本文が Python として構文が正しいこと (`ast.parse`。
  実行はしない)
- 2 つのスクリプトが `serving_kit` を import していないこと。秘密らしい文字列、
  IP アドレスの直書きがないこと
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from serving_kit.types import BandwidthRun, BandwidthSample

PAYLOAD_DIR = Path(__file__).resolve().parents[2] / "payload"
K2_QUANT_TOOL_DIR = Path(__file__).resolve().parents[3] / "experiments" / "k2-quant" / "k2_quant"
K2_QUANT_COPY_DIR = PAYLOAD_DIR / "k2-quant" / "k2_quant"
ALLREDUCE_BENCH_PATH = PAYLOAD_DIR / "allreduce_bench.py"
VLLM_SANITY_CHECK_PATH = PAYLOAD_DIR / "vllm_sanity_check.py"

_MIB = 1 << 20
_GIB = 1 << 30

_UPSTREAM_SEPARATOR = (
    "# ---8<--- 区切り: この行より下は、上流の文書のコードブロックをそのまま置いたもの "
    "(変更なし) ---8<---\n"
)


def _load_module(name: str, path: Path) -> Any:
    """モジュールを、パッケージを介さずに import して返す (`Any` として扱う)。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- allreduce_bench.py: import そのものの安全 ------------------------------------------------


def test_allreduce_bench_has_no_module_level_torch_import() -> None:
    """`torch` の import 文が、モジュール直下 (関数の外) に 1 つもないこと。

    `torch` の import は `main()` の中か、GPU を使う関数の中でだけ行う決まり
    (Mac には `torch` が入っていないので、モジュール直下に置くと import そのものが
    失敗する)。この確かめは、`torch` が入っている環境でも通る (関数の中は見ない)。
    """
    tree = ast.parse(ALLREDUCE_BENCH_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
            assert "torch" not in names
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "torch"


def test_importing_allreduce_bench_does_not_import_torch_or_print(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """import しただけでは `torch` が `sys.modules` に入らず、標準出力にも何も出ない。"""
    assert "torch" not in sys.modules

    module = _load_module("allreduce_bench_import_check", ALLREDUCE_BENCH_PATH)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert "torch" not in sys.modules
    # 計算の部分の関数が、ちゃんと定義されていることも、ついでに確かめる。
    assert callable(module.run_benchmark)
    assert callable(module.default_message_sizes_bytes)


@pytest.fixture(scope="module")
def allreduce_bench() -> Any:
    """`allreduce_bench.py` を 1 度だけ読み込み、以降の試験で使い回す。"""
    return _load_module("allreduce_bench", ALLREDUCE_BENCH_PATH)


# --- 大きさの列 ---------------------------------------------------------------------------------


def test_default_message_sizes_are_the_six_expected_values(allreduce_bench: Any) -> None:
    """1 MiB から 1 GiB まで、4 倍ずつの 6 つになる (design.md 「netcheck」の決まり)。"""
    sizes = allreduce_bench.default_message_sizes_bytes()
    assert sizes == (
        1 * _MIB,
        4 * _MIB,
        16 * _MIB,
        64 * _MIB,
        256 * _MIB,
        1 * _GIB,
    )


def test_default_message_sizes_can_be_overridden(allreduce_bench: Any) -> None:
    """A/B の短い回のために、上限や倍率をコマンドの引数で上書きできる。"""
    sizes = allreduce_bench.default_message_sizes_bytes(
        min_bytes=1 * _MIB, max_bytes=4 * _MIB, factor=2
    )
    assert sizes == (1 * _MIB, 2 * _MIB, 4 * _MIB)


def test_message_sizes_reject_invalid_bounds(allreduce_bench: Any) -> None:
    with pytest.raises(ValueError):
        allreduce_bench.default_message_sizes_bytes(min_bytes=0)
    with pytest.raises(ValueError):
        allreduce_bench.default_message_sizes_bytes(max_bytes=0)
    with pytest.raises(ValueError):
        allreduce_bench.default_message_sizes_bytes(min_bytes=100, max_bytes=10)
    with pytest.raises(ValueError):
        allreduce_bench.default_message_sizes_bytes(factor=1)


# --- algbw / busbw -------------------------------------------------------------------------------


def test_algbw_for_one_gib_in_one_second(allreduce_bench: Any) -> None:
    """1 GiB を 1 秒で送ると、algbw = 1.073741824 GB/s = 8.589934592 Gbps になる。"""
    algbw_gbps = allreduce_bench.compute_algbw_gbps(1 * _GIB, 1.0)
    assert algbw_gbps == pytest.approx(8.589934592, rel=1e-12)


def test_busbw_equals_algbw_for_world_size_two(allreduce_bench: Any) -> None:
    """2 台のとき、係数 2(n-1)/n は 1 になるので、busbw と algbw が等しい。"""
    algbw_gbps = allreduce_bench.compute_algbw_gbps(1 * _GIB, 1.0)
    busbw_gbps = allreduce_bench.compute_busbw_gbps(algbw_gbps, world_size=2)
    assert busbw_gbps == pytest.approx(algbw_gbps, rel=1e-12)


def test_busbw_for_world_size_four_is_one_point_five_times(allreduce_bench: Any) -> None:
    """4 台のとき、係数 2(n-1)/n = 2*3/4 = 1.5 になる。"""
    algbw_gbps = allreduce_bench.compute_algbw_gbps(1 * _GIB, 1.0)
    busbw_gbps = allreduce_bench.compute_busbw_gbps(algbw_gbps, world_size=4)
    assert busbw_gbps == pytest.approx(algbw_gbps * 1.5, rel=1e-12)


def test_busbw_for_world_size_one_is_zero(allreduce_bench: Any) -> None:
    """1 台のとき、集団通信が要らないので、係数は 0 になる。"""
    algbw_gbps = allreduce_bench.compute_algbw_gbps(1 * _GIB, 1.0)
    busbw_gbps = allreduce_bench.compute_busbw_gbps(algbw_gbps, world_size=1)
    assert busbw_gbps == 0.0


def test_algbw_rejects_zero_or_negative_time(allreduce_bench: Any) -> None:
    """0 や負の time_s は、計測できなかったことを示すので断る。"""
    with pytest.raises(ValueError):
        allreduce_bench.compute_algbw_gbps(1 * _MIB, 0.0)
    with pytest.raises(ValueError):
        allreduce_bench.compute_algbw_gbps(1 * _MIB, -1.0)


def test_algbw_rejects_non_positive_size(allreduce_bench: Any) -> None:
    with pytest.raises(ValueError):
        allreduce_bench.compute_algbw_gbps(0, 1.0)


def test_busbw_rejects_non_positive_world_size(allreduce_bench: Any) -> None:
    with pytest.raises(ValueError):
        allreduce_bench.compute_busbw_gbps(1.0, world_size=0)
    with pytest.raises(ValueError):
        allreduce_bench.compute_busbw_gbps(1.0, world_size=-2)


# --- 時間の統計 -----------------------------------------------------------------------------------


def test_summarize_times_computes_basic_stats(allreduce_bench: Any) -> None:
    stats = allreduce_bench.summarize_times_s([1.0, 2.0, 3.0])
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
    assert stats["stdev"] == pytest.approx(1.0)


def test_summarize_times_stdev_is_none_for_single_sample(allreduce_bench: Any) -> None:
    stats = allreduce_bench.summarize_times_s([1.0])
    assert stats["n"] == 1
    assert stats["stdev"] is None


def test_summarize_times_rejects_empty_or_non_positive(allreduce_bench: Any) -> None:
    """時間が 0 や負のとき (計測できなかったことを示す) と、空の列を断る。"""
    with pytest.raises(ValueError):
        allreduce_bench.summarize_times_s([])
    with pytest.raises(ValueError):
        allreduce_bench.summarize_times_s([1.0, 0.0, 2.0])
    with pytest.raises(ValueError):
        allreduce_bench.summarize_times_s([1.0, -0.5])


# --- 要素数 (バッファの確保に使う、torch なしで試せる部分) -------------------------------------


def test_element_count_for_default_sizes_divides_evenly(allreduce_bench: Any) -> None:
    for size_bytes in allreduce_bench.default_message_sizes_bytes():
        count = allreduce_bench.element_count_for_size(size_bytes)
        assert count * allreduce_bench._DTYPE_ELEMENT_SIZE_BYTES == size_bytes


def test_element_count_rejects_size_not_divisible(allreduce_bench: Any) -> None:
    with pytest.raises(ValueError):
        allreduce_bench.element_count_for_size(10, element_size_bytes=3)


# --- BandwidthSample / BandwidthRun に写せる形 ----------------------------------------------------


def test_build_bandwidth_sample_maps_onto_types_bandwidth_sample(allreduce_bench: Any) -> None:
    """`size_bytes` / `algbw_gbps` / `busbw_gbps` が、`types.BandwidthSample` にそのまま写る。"""
    sample = allreduce_bench.build_bandwidth_sample(1 * _GIB, [1.0] * 20, world_size=2)

    assert sample["size_bytes"] == 1 * _GIB
    assert sample["algbw_gbps"] == pytest.approx(8.589934592, rel=1e-12)
    assert sample["busbw_gbps"] == pytest.approx(8.589934592, rel=1e-12)
    assert sample["time_s"]["n"] == 20

    # task 4.4 が、この辞書から `BandwidthSample` を組み立てられることを確かめる。
    typed_sample = BandwidthSample(
        size_bytes=sample["size_bytes"],
        algbw_gbps=sample["algbw_gbps"],
        busbw_gbps=sample["busbw_gbps"],
    )
    assert typed_sample.size_bytes == 1 * _GIB


def test_build_bandwidth_report_shape_and_json_is_single_line(allreduce_bench: Any) -> None:
    samples = [
        allreduce_bench.build_bandwidth_sample(size_bytes, [0.01] * 20, world_size=2)
        for size_bytes in allreduce_bench.default_message_sizes_bytes()
    ]
    report = allreduce_bench.build_bandwidth_report(
        samples=samples,
        world_size=2,
        dtype="float32",
        warmup_iters=5,
        measure_iters=20,
        torch_version="2.9.1+cu130",
        nccl_version="2.30.7",
    )

    for key in (
        "world_size",
        "dtype",
        "warmup_iters",
        "measure_iters",
        "torch_version",
        "nccl_version",
        "samples",
    ):
        assert key in report
    assert len(report["samples"]) == 6

    rendered = allreduce_bench.render_report_json(report)
    assert "\n" not in rendered

    loaded = json.loads(rendered)
    assert loaded["world_size"] == 2
    assert loaded["torch_version"] == "2.9.1+cu130"

    typed_samples = tuple(
        BandwidthSample(
            size_bytes=item["size_bytes"],
            algbw_gbps=item["algbw_gbps"],
            busbw_gbps=item["busbw_gbps"],
        )
        for item in loaded["samples"]
    )
    run = BandwidthRun(arm="baseline", repeat_index=1, samples=typed_samples)
    assert len(run.samples) == 6


def test_build_bandwidth_report_includes_skipped_sizes_when_present(
    allreduce_bench: Any,
) -> None:
    """GPU のメモリが足りず確保できなかった大きさは、`skipped_sizes` に記録できる。"""
    report = allreduce_bench.build_bandwidth_report(
        samples=(),
        world_size=2,
        dtype="float32",
        warmup_iters=5,
        measure_iters=20,
        torch_version="2.9.1+cu130",
        nccl_version="2.30.7",
        skipped_sizes=({"size_bytes": 1 * _GIB, "reason": "oom"},),
    )
    assert report["skipped_sizes"] == [{"size_bytes": 1 * _GIB, "reason": "oom"}]


# --- コマンドラインの引数 (既定値は design の決まり、上書きできる) -------------------------------


def test_cli_defaults_match_design_decisions(allreduce_bench: Any) -> None:
    args = allreduce_bench._parse_args([])
    assert args.min_bytes == 1 * _MIB
    assert args.max_bytes == 1 * _GIB
    assert args.factor == 4
    assert args.warmup_iters == 5
    assert args.measure_iters == 20


def test_cli_args_can_be_overridden_for_short_ab_runs(allreduce_bench: Any) -> None:
    args = allreduce_bench._parse_args(
        ["--max-bytes", "4194304", "--warmup-iters", "1", "--measure-iters", "3"]
    )
    assert args.max_bytes == 4194304
    assert args.warmup_iters == 1
    assert args.measure_iters == 3


# --- vllm_sanity_check.py -------------------------------------------------------------------------


def _vllm_sanity_check_source() -> str:
    return VLLM_SANITY_CHECK_PATH.read_text(encoding="utf-8")


def _vllm_sanity_check_header_and_body() -> tuple[str, str]:
    source = _vllm_sanity_check_source()
    assert _UPSTREAM_SEPARATOR in source
    header, body = source.split(_UPSTREAM_SEPARATOR, maxsplit=1)
    return header, body


def test_vllm_sanity_check_header_has_source_commit_and_license() -> None:
    header, _body = _vllm_sanity_check_header_and_body()
    assert "385dce36bcee42309924a5ece951a96db3dce7f2" in header
    assert "https://github.com/vllm-project/vllm/blob/" in header
    assert "Apache-2.0" in header
    assert "SHA-256" in header


def test_vllm_sanity_check_body_sha256_matches_the_header() -> None:
    """区切りの行より下の本文の sha256 が、コメントに書かれた値と一致する。

    あとで誰かが本文を直したら、この試験が落ちる。
    """
    header, body = _vllm_sanity_check_header_and_body()

    match = re.search(r"SHA-256:\s*\n#\s+([0-9a-f]{64})", header)
    assert match is not None, "先頭のコメントに、64 桁の 16 進の sha256 が見つからない"
    declared_sha256 = match.group(1)

    actual_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert actual_sha256 == declared_sha256


def test_vllm_sanity_check_body_is_syntactically_valid_python() -> None:
    """本文が Python として構文が正しいこと (`ast.parse`。実行はしない)。"""
    _header, body = _vllm_sanity_check_header_and_body()
    ast.parse(body)  # SyntaxError を投げなければ良い


def test_vllm_sanity_check_whole_file_is_syntactically_valid_python() -> None:
    ast.parse(_vllm_sanity_check_source())


def test_vllm_sanity_check_body_matches_the_four_stage_check() -> None:
    """4 段 (PyTorch NCCL、GLOO、vLLM NCCL、CUDA グラフの中の vLLM NCCL) がそろっている。"""
    _header, body = _vllm_sanity_check_header_and_body()
    assert "PyTorch NCCL is successful!" in body
    assert "PyTorch GLOO is successful!" in body
    assert "vLLM NCCL is successful!" in body
    assert "vLLM NCCL with cuda graph is successful!" in body


# --- どちらのスクリプトにも共通の決まり ----------------------------------------------------------

_IPV4_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
_SECRET_LIKE_RE = re.compile(r"(?i)\b(api[_-]?key|secret|password|token)\b")

_PAYLOAD_PATHS = [ALLREDUCE_BENCH_PATH, VLLM_SANITY_CHECK_PATH]
_PAYLOAD_IDS = ["allreduce_bench", "vllm_sanity_check"]


def _imported_top_level_module_names(source: str) -> set[str]:
    """`import` 文 (どこにあっても) が読み込む、先頭の module 名の集まりを返す。"""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", _PAYLOAD_PATHS, ids=_PAYLOAD_IDS)
def test_payload_scripts_do_not_import_serving_kit(path: Path) -> None:
    """Spark には `serving_kit` を配らないので、2 つのスクリプトは import していない。

    `import` 文だけを見る (`ast` で確かめる)。docstring や、出典のコメントの中で
    `serving_kit.types.BandwidthSample` のように項目名を説明する文字列は、import ではない
    ので、これには当たらない。
    """
    imported = _imported_top_level_module_names(path.read_text(encoding="utf-8"))
    assert "serving_kit" not in imported


@pytest.mark.parametrize("path", _PAYLOAD_PATHS, ids=_PAYLOAD_IDS)
def test_payload_scripts_have_no_ip_literals_or_secret_like_names(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    assert _IPV4_RE.search(source) is None
    assert _SECRET_LIKE_RE.search(source) is None


def test_the_upstream_script_carries_its_own_guards_against_linters_and_formatters() -> None:
    """上流の原文は、ファイルの中の印で、検査と整形から守る (設定に頼らない)。

    `pyproject.toml` の除外の設定 (`extend-exclude`、`force-exclude`) は、ruff を `serving/` の
    外から呼ぶと効かない (設定は、作業ディレクトリから探される)。印は、区切りの行より上
    (この道具が書いた部分) に置く。下の本文 (上流の原文) には、何も足さない。
    """
    source = VLLM_SANITY_CHECK_PATH.read_text(encoding="utf-8")
    header, body = source.split(_UPSTREAM_SEPARATOR, maxsplit=1)
    header_lines = header.splitlines()

    assert header_lines[0] == "# ruff: noqa"
    assert header_lines[1] == "# fmt: off"
    # 整形を、途中で戻さない (本文の側にも、ヘッダの側にも、`# fmt: on` の行がない)
    assert re.search(r"^[ \t]*#[ \t]*fmt:[ \t]*on[ \t]*$", source, flags=re.MULTILINE) is None
    assert "ruff: noqa" not in body
    assert "fmt: off" not in body


def test_the_ruff_config_also_excludes_the_upstream_script() -> None:
    """設定の側の除外 (ディレクトリを歩くときの、二重の歯止め) も、この 1 ファイルだけ。"""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    ruff = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["ruff"]

    assert ruff.get("force-exclude") is True
    assert ruff.get("extend-exclude") == ["payload/vllm_sanity_check.py"]


def _python_sources(directory: Path) -> dict[str, bytes]:
    """ディレクトリ直下の `*.py` を、名前からバイト列への対応にする (`__pycache__` は見ない)。"""
    assert directory.is_dir(), f"ディレクトリがない: {directory}"
    return {path.name: path.read_bytes() for path in sorted(directory.glob("*.py"))}


def test_the_k2_quant_payload_copy_has_the_same_files_as_the_tool() -> None:
    """Spark に配る変換の道具の写しは、道具と同じ名前の `*.py` を、過不足なく持つ。

    `serve push` は `serving/payload/` の中身を配る。道具の本体は `experiments/k2-quant/` に
    あるので、配るための写しを `serving/payload/k2-quant/` に置く。
    """
    assert set(_python_sources(K2_QUANT_COPY_DIR)) == set(_python_sources(K2_QUANT_TOOL_DIR))


def test_the_k2_quant_payload_copy_is_byte_identical_to_the_tool() -> None:
    """写しの各ファイルは、道具のファイルと 1 バイトも違わない。

    コミットした道具の内容と、Spark で動かす写しが同じであることを、この試験で固定する。
    """
    tool = _python_sources(K2_QUANT_TOOL_DIR)
    copy = _python_sources(K2_QUANT_COPY_DIR)

    different = sorted(name for name in tool if name in copy and tool[name] != copy[name])

    assert not different, f"写しが道具と違う: {different}"


def _run_ruff(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    # 書き換えない形 (`--check`、`check`) でだけ呼ぶ。キャッシュは、作業ディレクトリに作らせない
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args, "--no-cache"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("where", ["serving", "repo-root", "elsewhere"])
@pytest.mark.parametrize("isolated", [False, True], ids=["found-config", "no-config"])
@pytest.mark.parametrize("command", [["format", "--check"], ["check"]], ids=["format", "check"])
def test_ruff_leaves_the_upstream_script_alone_from_any_directory(
    tmp_path: Path, where: str, isolated: bool, command: list[str]
) -> None:
    """設定の値だけでなく、実際に ruff を、名指しで呼んで確かめる。

    作業ディレクトリ (`serving/`、リポジトリの直下、関係のない場所) にも、設定の有無
    (`--isolated` は、どの設定も読まない) にも依らずに、整形も検査もされないこと。
    """
    serving = Path(__file__).resolve().parents[2]
    cwd = {"serving": serving, "repo-root": serving.parent, "elsewhere": tmp_path}[where]
    before = VLLM_SANITY_CHECK_PATH.read_bytes()

    done = _run_ruff(
        [*command, *(["--isolated"] if isolated else []), str(VLLM_SANITY_CHECK_PATH)], cwd
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert "would be reformatted" not in done.stdout + done.stderr
    assert VLLM_SANITY_CHECK_PATH.read_bytes() == before


def test_ruff_still_checks_our_own_payload_script_from_the_repo_root() -> None:
    """守るのは、上流の 1 ファイルだけ。自前の `allreduce_bench.py` は、検査の対象のまま。"""
    serving = Path(__file__).resolve().parents[2]
    own = serving / "payload" / "allreduce_bench.py"

    done = _run_ruff(["check", "--show-files", str(own)], serving.parent)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "allreduce_bench.py" in done.stdout
    assert "ruff: noqa" not in own.read_text(encoding="utf-8")
    assert "fmt: off" not in own.read_text(encoding="utf-8")
