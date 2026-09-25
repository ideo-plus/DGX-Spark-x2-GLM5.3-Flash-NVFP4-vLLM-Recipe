"""torch プロファイラーの trace を集計する道具 (`summarize_trace.py`) の試験 (C5、C6)。

確かめること:

- カーネル名 → 区分の対応 (`CATEGORY_RULES`) が 1 か所にあり、当たらない名前は `other` に
  入る (C6)
- GPU の展開されたイベント (`ph == "X"` かつ `cat` が GPU のもの) だけを数え、`cpu_op` /
  `cuda_runtime` / `M` / `f` と、`dur` のないイベントは数えない。最上位が配列の形と
  `{"traceEvents": …}` の形で同じ結果になる (C5、SCN-C5-N1)
- 区分ごとの合計、`wall`、`busy` (重なりは 1 回だけ)、`gap = wall − busy`、`overlap` を
  正しく出す (C5、SCN-C5-P1、SCN-C5-P2)
- 名前ごとに回数と合計を集約し、合計の降順に並べる (C5)
- gzip の trace を、拡張子ではなく中身で見分けて読む (C5、SCN-C5-P3)
- `--steps` が必須で 1 以上、GPU のイベントが 0 件なら終了 1 (C5)
- 出力に、区分の名前、上位のカーネル、区分ごとの合計・1 ステップあたりの平均・割合、
  `other` の上位が出る。上位の一覧の行は名前・区分・回数・合計を持ち、`--top` が全体の
  一覧と `other` の一覧をそれぞれ切る (C5)
- 道具は、ネットワークも `subprocess` も読み込まない (C5)

試験は、合成の trace を `tmp_path` に書き、道具の関数と `main` を直接呼ぶ。実機にも
`serving/var/` にも触らない。この試験は、道具と同じディレクトリに置くので、`pytest` が
このディレクトリを読み込みの道に足し、`import summarize_trace` で読める。
"""

from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path
from typing import Any

import pytest
import summarize_trace

TOOL_PATH = Path(__file__).resolve().parent / "summarize_trace.py"
"""集計の道具のソース (import の検査に使う)。"""

ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "__future__",
        "argparse",
        "collections",
        "dataclasses",
        "gzip",
        "json",
        "pathlib",
        "re",
        "typing",
    }
)
FORBIDDEN_IMPORTS = frozenset({"os", "subprocess", "socket", "urllib", "httpx", "http"})


def _event(
    name: str, ts: float, dur: float, *, cat: str = "kernel", ph: str = "X"
) -> dict[str, Any]:
    """1 つの trace イベント (Chrome の trace の形)。"""
    return {"ph": ph, "cat": cat, "name": name, "ts": ts, "dur": dur}


def _trace(*events: dict[str, Any]) -> dict[str, Any]:
    """`traceEvents` を持つ trace。"""
    return {"traceEvents": list(events)}


def _write_plain(path: Path, trace: Any) -> Path:
    """trace を平文の JSON で書く。"""
    path.write_text(json.dumps(trace), encoding="utf-8")
    return path


# --- C6: 名前 → 区分の対応 -------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("nvjet_bf16_gemm", "other_gemm", id="other-gemm"),
        pytest.param("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "nccl", id="nccl"),
        pytest.param("paged_attention_v1", "attention", id="attention"),
        pytest.param("fused_moe_kernel", "moe_gemm", id="moe-gemm"),
        pytest.param("moe_gemm_kernel", "moe_gemm", id="moe-word-then-gemm-word"),
        pytest.param("cutlass_moe_kernel", "moe_gemm", id="gemm-word-then-moe-word"),
        pytest.param("grouped_gemm_kernel", "moe_gemm", id="grouped-gemm"),
    ],
)
def test_classify_puts_known_kernels_in_their_category(name: str, expected: str) -> None:
    """代表的なカーネルが、それぞれの区分になる。

    MoE の語と GEMM の語を両方含む名前は、語の並びの順によらず `moe_gemm` になる (C6)。
    """
    assert summarize_trace.classify(name) == expected


def test_classify_sends_unknown_names_to_other() -> None:
    """どの規則にも当たらない名前は `other` になる (C6)。"""
    assert summarize_trace.classify("rms_norm_kernel_xyz") == summarize_trace.OTHER


def test_classify_keeps_moe_only_names_out_of_moe_gemm() -> None:
    """MoE の語だけで GEMM の語を含まない名前は、`moe_gemm` にも `other_gemm` にも入らない (C6)。"""
    assert summarize_trace.classify("moe_align_block_size_kernel") == summarize_trace.OTHER


def test_classify_follows_the_order_of_the_rules() -> None:
    """2 つの規則に当たる名前は、表の先にある規則の区分になる (C6)。"""
    assert summarize_trace.classify("nccl_attention_kernel") == "nccl"


def _assigned_name(node: ast.stmt) -> str | None:
    """代入の文の左辺が、単純な名前 (`NAME = …`) ならその名前。"""
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def test_the_category_rules_live_in_one_table() -> None:
    """名前 → 区分の表 (`CATEGORY_RULES`) が、モジュールの直下に 1 つだけある (C6)。"""
    assert TOOL_PATH.is_file(), f"道具がない: {TOOL_PATH}"
    tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if _assigned_name(node) == "CATEGORY_RULES"]

    assert len(definitions) == 1, "名前 → 区分の表が 1 か所にない"


# --- C5: GPU イベントの取り出し -------------------------------------------


def test_gpu_events_keeps_only_gpu_expanded_events() -> None:
    """`ph == "X"` かつ GPU の `cat` のイベントだけを取り、CPU のイベントは数えない (SCN-C5-N1)。"""
    trace = _trace(
        _event("k1", 0, 10),
        _event("aten::mm", 0, 100, cat="cpu_op"),
        _event("cudaLaunchKernel", 0, 100, cat="cuda_runtime"),
        {"ph": "M", "name": "process_name", "args": {"name": "x"}},
        _event("ac2g", 0, 100, cat="ac2g", ph="f"),
        {"ph": "X", "cat": "kernel", "name": "no_dur", "ts": 0},
    )

    events = summarize_trace.gpu_events(trace)

    assert [event.name for event in events] == ["k1"]
    assert events[0].start_us == 0
    assert events[0].dur_us == 10


def test_summarize_is_the_same_for_the_dict_and_array_shapes() -> None:
    """最上位が配列の trace と `{"traceEvents": …}` の trace で、同じ集計になる (SCN-C5-N1)。"""
    events = [_event("k1", 0, 10)]

    from_dict = summarize_trace.summarize(summarize_trace.gpu_events(_trace(*events)), steps=1)
    from_list = summarize_trace.summarize(summarize_trace.gpu_events(list(events)), steps=1)

    assert from_dict == from_list


# --- C5: 区分ごとの時間 ---------------------------------------------------


def test_summarize_counts_a_kernel_in_other_gemm() -> None:
    """`nvjet_bf16_gemm` が `other_gemm` に 40 µs として入る (SCN-C5-P1)。"""
    events = summarize_trace.gpu_events(_trace(_event("nvjet_bf16_gemm", 100, 40)))

    summary = summarize_trace.summarize(events, steps=1)

    assert summary.category_us["other_gemm"] == 40
    assert summary.wall_us == 40
    assert summary.busy_us == 40
    assert summary.gap_us == 0


def test_summarize_measures_busy_overlap_and_gap() -> None:
    """重なるカーネルで、busy は和集合、overlap は重なり、gap は wall−busy (SCN-C5-P2)。"""
    events = summarize_trace.gpu_events(_trace(_event("k1", 0, 10), _event("k2", 5, 10)))

    summary = summarize_trace.summarize(events, steps=1)

    assert summary.wall_us == 15
    assert summary.busy_us == 15
    assert summary.overlap_us == 5
    assert summary.gap_us == 0


def test_summarize_counts_the_idle_gap() -> None:
    """離れた 2 つのカーネルの間は、`gap` (GPU の空き) になる。"""
    events = summarize_trace.gpu_events(_trace(_event("k1", 0, 10), _event("k2", 30, 10)))

    summary = summarize_trace.summarize(events, steps=1)

    assert summary.wall_us == 40
    assert summary.busy_us == 20
    assert summary.gap_us == 20


def test_summarize_aggregates_kernels_by_name_and_sorts_by_total() -> None:
    """同じ名前のカーネルを回数と合計にまとめ、合計の降順に並べる (C5)。"""
    events = summarize_trace.gpu_events(
        _trace(
            _event("rms_norm", 0, 5),
            _event("nvjet_bf16_gemm", 5, 30),
            _event("rms_norm", 35, 7),
        )
    )

    summary = summarize_trace.summarize(events, steps=1)

    assert [kernel.name for kernel in summary.kernels] == ["nvjet_bf16_gemm", "rms_norm"]
    norm = next(kernel for kernel in summary.kernels if kernel.name == "rms_norm")
    assert norm.count == 2
    assert norm.total_us == 12


# --- C5: 入力の読み込み ---------------------------------------------------


def test_read_trace_reads_a_plain_json_trace(tmp_path: Path) -> None:
    """平文の JSON の trace を読める (C5)。"""
    path = _write_plain(tmp_path / "trace.json", _trace(_event("k1", 0, 10)))

    events = summarize_trace.read_trace(path)

    assert [event.name for event in events] == ["k1"]


def test_read_trace_reads_gzip_by_content_not_extension(tmp_path: Path) -> None:
    """gzip の trace を、拡張子がなくても中身で見分けて、平文と同じに読む (SCN-C5-P3)。"""
    body = json.dumps(_trace(_event("k1", 0, 10))).encode("utf-8")
    plain = _write_plain(tmp_path / "trace.json", _trace(_event("k1", 0, 10)))
    gzipped = tmp_path / "trace-blob"
    gzipped.write_bytes(gzip.compress(body))

    plain_summary = summarize_trace.summarize(summarize_trace.read_trace(plain), steps=1)
    gzip_summary = summarize_trace.summarize(summarize_trace.read_trace(gzipped), steps=1)

    assert gzip_summary == plain_summary


# --- C5: CLI の入口と出力 -------------------------------------------------


def test_main_requires_a_positive_steps(tmp_path: Path) -> None:
    """`--steps` は必須で、1 以上だけを受ける (C5)。"""
    path = _write_plain(tmp_path / "trace.json", _trace(_event("k1", 0, 10)))

    with pytest.raises(SystemExit):
        summarize_trace.main([str(path)])

    with pytest.raises(SystemExit):
        summarize_trace.main(["--steps", "0", str(path)])


def test_main_exits_one_when_the_trace_has_no_gpu_events(tmp_path: Path) -> None:
    """GPU のイベントが 0 件の trace (AsyncLLM の CPU trace) では、終了 1 にする (C5)。"""
    path = _write_plain(tmp_path / "cpu.json", _trace(_event("aten::mm", 0, 10, cat="cpu_op")))

    assert summarize_trace.main(["--steps", "1", str(path)]) == 1


def _section(text: str, heading: str) -> str:
    """`heading` で始まる節の本文 (次の `## ` の見出しまで)。"""
    start = text.index(heading)
    rest = text[start:]
    end = rest.find("\n## ", 1)
    return rest if end < 0 else rest[:end]


def _table_rows(section: str) -> list[list[str]]:
    """表の行 (セルごとに分ける)。"""
    rows: list[list[str]] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        rows.append([cell.strip() for cell in stripped.strip("|").split("|")])
    return rows


def _category_row(out: str, label: str) -> list[str]:
    """区分の表から、区分の列が `label` と完全に一致する行を返す (部分一致では選ばない)。"""
    for row in _table_rows(_section(out, "## 区分ごとの時間")):
        if row and row[0] == label:
            return row
    pytest.fail(f"区分の行がない: {label}")


def _numbers(row: list[str]) -> list[float]:
    """行の、区分の名前より後のセルを数にする。"""
    return [float(cell) for cell in row[1:]]


def test_main_prints_categories_averages_ratios_and_top_kernels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """区分ごとの合計・1 ステップの平均・割合と、上位のカーネルと `other` の上位を出す (C5)。

    合成の trace は、NCCL 1,000・その他 GEMM 5,000・アテンション本体 1,000・その他 2,000 ms を、
    間に 1,000 ms の隙間を挟んで並べたもの。`--steps 5` なので、wall は 10,000 ms、busy は
    9,000 ms、隙間は 1,000 ms になる。割合は wall を分母にする (busy を分母にすると
    その他 GEMM は 55.556 になり、この試験で区別できる)。
    """
    path = _write_plain(
        tmp_path / "trace.json",
        _trace(
            _event("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 0, 1_000_000),
            _event("nvjet_bf16_gemm", 1_000_000, 5_000_000),
            _event("paged_attention_v1", 6_000_000, 1_000_000),
            _event("mystery_kernel_xyz", 8_000_000, 2_000_000),
        ),
    )

    result = summarize_trace.main(["--steps", "5", "--top", "20", str(path)])

    assert result == 0
    out = capsys.readouterr().out

    # 区分ごとの行 (合計・1 ステップの平均・割合)。wall の 10,000 ms を分母にする
    assert _numbers(_category_row(out, summarize_trace.CATEGORY_LABELS["nccl"])) == pytest.approx(
        [1000.0, 200.0, 10.0]
    )
    assert _numbers(
        _category_row(out, summarize_trace.CATEGORY_LABELS["other_gemm"])
    ) == pytest.approx([5000.0, 1000.0, 50.0])
    assert _numbers(
        _category_row(out, summarize_trace.CATEGORY_LABELS["attention"])
    ) == pytest.approx([1000.0, 200.0, 10.0])
    assert _numbers(
        _category_row(out, summarize_trace.CATEGORY_LABELS[summarize_trace.OTHER])
    ) == pytest.approx([2000.0, 400.0, 20.0])
    assert _numbers(
        _category_row(out, summarize_trace.CATEGORY_LABELS[summarize_trace.GAP])
    ) == pytest.approx([1000.0, 200.0, 10.0])

    # 全体の上位のカーネルの一覧に、名前が出る
    top = _section(out, "## 上位のカーネル")
    assert "nvjet_bf16_gemm" in top
    assert "mystery_kernel_xyz" in top

    # 「その他」の上位の節には、未分類のカーネルだけが入る
    other_section = _section(
        out, f"## {summarize_trace.CATEGORY_LABELS[summarize_trace.OTHER]}の上位"
    )
    other_names = [row[0] for row in _table_rows(other_section) if row and row[0] != "名前"]
    assert other_names == ["mystery_kernel_xyz"]
    assert "nvjet_bf16_gemm" not in other_section


def test_main_prints_top_kernel_rows_and_cuts_each_list_by_top(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """上位の一覧の行 (名前・区分・回数・合計) を出し、`--top` で 2 つの一覧をそれぞれ切る (C5)。

    合成の trace は、その他 GEMM の `nvjet_bf16_gemm` を 2 回 (3,000 + 2,000 ms)、未分類の
    2 つのカーネルを 1,000 ms と 500 ms で並べたもの。`--top 1` なので、全体の上位は
    `nvjet_bf16_gemm` だけになる。それでも「その他」の上位には、全体の切り詰めとは別に、
    未分類のうち合計が最大のカーネルが残る (切り詰めの後に「その他」を選ぶと、空になる)。
    """
    path = _write_plain(
        tmp_path / "trace.json",
        _trace(
            _event("nvjet_bf16_gemm", 0, 3_000_000),
            _event("nvjet_bf16_gemm", 3_000_000, 2_000_000),
            _event("mystery_kernel_a", 5_000_000, 1_000_000),
            _event("mystery_kernel_b", 6_000_000, 500_000),
        ),
    )

    result = summarize_trace.main(["--steps", "1", "--top", "1", str(path)])

    assert result == 0
    out = capsys.readouterr().out

    top_rows = [row for row in _table_rows(_section(out, "## 上位のカーネル")) if row[0] != "名前"]
    assert top_rows == [
        ["nvjet_bf16_gemm", summarize_trace.CATEGORY_LABELS["other_gemm"], "2", "5000.000"]
    ]

    other_section = _section(
        out, f"## {summarize_trace.CATEGORY_LABELS[summarize_trace.OTHER]}の上位"
    )
    other_rows = [row for row in _table_rows(other_section) if row[0] != "名前"]
    assert other_rows == [["mystery_kernel_a", "1", "1000.000"]]


# --- C5: 依存の向き -------------------------------------------------------


def test_tool_imports_no_network_or_subprocess() -> None:
    """道具は、ネットワークも `subprocess` も読み込まない (C5)。"""
    assert TOOL_PATH.is_file(), f"道具がない: {TOOL_PATH}"
    tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            roots.add(node.module.split(".")[0])

    assert roots <= ALLOWED_IMPORT_ROOTS, f"許していない読み込み: {roots - ALLOWED_IMPORT_ROOTS}"
    assert not (roots & FORBIDDEN_IMPORTS)
