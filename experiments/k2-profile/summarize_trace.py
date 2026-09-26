"""torch プロファイラーの trace から、1 ステップの GPU の時間の内訳を集計する (#42)。

入力は、vLLM の torch プロファイラーが書いた Chrome の trace 形式の JSON である
(`{"traceEvents": [...]}` と、最上位が配列の形の両方を受ける)。ファイルの中身の先頭 2 バイトが
`1f 8b` なら (拡張子にはよらず)、gzip として展開して読む。そうでなければ平文の JSON として読む。
ネットワークも `subprocess` も使わない。Mac に回収したファイルだけを読む。

GPU の時間は、`ph == "X"` かつ `cat` が GPU のもの (`kernel` / `gpu_memcpy` /
`gpu_memset`) のイベントだけから数える。名前に `run.py` のような CPU のイベントや、
`cudaLaunchKernel` のような実行のイベントは数えない。

区分けは次の 8 つである。

- MoE の専門家の GEMM (NVFP4 の grouped GEMM。`moe_gemm`)
- MoE の周辺 (expand・finalize・並べ替え・活性化・topk・fp4 への変換。`moe_aux`)
- BF16 の重みの GEMV と GEMM (cuBLAS の `gemvx` と `gemvNSP`、`cutlass_80_wmma`、
  `nvjet`。`weight_gemm`)
- アテンションの本体 (`attention`)
- mHC (`mhc`)
- NCCL の all-reduce など (`nccl`)
- その他のカーネル (`other`。どの規則にも当たらない名前)
- カーネルのない隙間 (`gap`。カーネルではなく、`wall − busy` で導く)

名前から区分への対応は `CATEGORY_RULES` の 1 か所にまとめる。順序つきで、先に当たった
規則が勝つ。当たらない名前は `other` に入れ、出力では `other` だけの上位一覧も並べる。

trace には 1 ステップの区切りの印が入らない (vLLM の `record_function_or_nullcontext` は
`VLLM_CUSTOM_SCOPES_FOR_PROFILING` が無いと `nullcontext` になる) ので、ステップ数は
`--steps` で受け取る。ステップ数は、`/start_profile` の前と、負荷の終了後・`/stop_profile`
の前に `GET /metrics` の `vllm:iteration_tokens_total_count` を読み、その増分から得る
(head が OOM で止まると停止の後は読めないので、停止の前の値を使う。停止の後にも読めるなら
同じ値であることを確かめる)。

使い方:

    python summarize_trace.py --steps N [--top N] TRACE [TRACE ...]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

GPU_CATEGORIES: Final[frozenset[str]] = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
"""GPU の時間として数えるイベントの `cat` (C5)。"""

OTHER: Final[str] = "other"
"""どの規則にも当たらない名前の区分 (C6)。"""

GAP: Final[str] = "gap"
"""カーネルのない隙間 (カーネル名からは導かない区分。C5)。"""

CATEGORY_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("nccl", re.compile(r"nccl", re.IGNORECASE)),
    ("moe_gemm", re.compile(r"GroupProblemShape", re.IGNORECASE)),
    (
        "moe_aux",
        re.compile(
            r"expandInputRows|finalizeMoeRouting|fusedBuildExpertMaps|computeStrides"
            r"|doActivation|single_group_topk|cvt_fp16_to_fp4|vllm::moe::",
            re.IGNORECASE,
        ),
    ),
    ("mhc", re.compile(r"mhc_", re.IGNORECASE)),
    (
        "attention",
        re.compile(
            r"attention|attn|fmha|flash_fwd|mla|paged_kv|paged|gated_delta|delta_rule"
            r"|recurrent|xqa|kda|causal_conv1d|_kpool_|persistent_topk|fwht",
            re.IGNORECASE,
        ),
    ),
    ("weight_gemm", re.compile(r"gemvx|gemvNSP|cutlass_80_wmma|nvjet", re.IGNORECASE)),
)
"""カーネル名 → 区分の対応 (この 1 か所だけ。C6)。

断片が固有な規則 (moe_gemm・moe_aux・mhc) を広い規則 (attention) より先に置き、MoE 周辺の
名前を先に取り切ってから weight_gemm を最後に置く。当たらない名前は「その他」の上位一覧に
並ぶので、新しい名前が現れたらここを見直して再集計する。
"""

CATEGORY_LABELS: Final[Mapping[str, str]] = {
    "moe_gemm": "MoE 専門家 GEMM",
    "moe_aux": "MoE 周辺",
    "weight_gemm": "BF16 重み GEMV/GEMM",
    "attention": "アテンション本体",
    "mhc": "mHC",
    "nccl": "NCCL",
    OTHER: "その他",
    GAP: "カーネルのない隙間",
}
"""出力に使う区分の名前 (C5)。"""


@dataclass(frozen=True)
class GpuEvent:
    """1 つの GPU のイベント (マイクロ秒)。"""

    name: str
    start_us: float
    dur_us: float


@dataclass(frozen=True)
class KernelTotal:
    """カーネル名ごとの回数と合計の時間 (マイクロ秒)。"""

    name: str
    category: str
    count: int
    total_us: float


@dataclass(frozen=True)
class Summary:
    """trace 1 つぶんの集計 (時間はマイクロ秒)。"""

    steps: int
    wall_us: float
    busy_us: float
    gap_us: float
    overlap_us: float
    category_us: Mapping[str, float]
    kernels: tuple[KernelTotal, ...]
    """合計の降順 (同点は名前の昇順)。"""


def classify(name: str) -> str:
    """カーネル名を区分に写す (`CATEGORY_RULES` の先頭一致。当たらない名前は `other`。C6)。"""
    for category, pattern in CATEGORY_RULES:
        if pattern.search(name):
            return category
    return OTHER


def _trace_events(trace: object) -> list[object]:
    """`{"traceEvents": [...]}` と、最上位が配列の形を、イベントの並びに正規化する。"""
    if isinstance(trace, dict):
        raw = trace.get("traceEvents")
        return list(raw) if isinstance(raw, list) else []
    return list(trace) if isinstance(trace, list) else []


def _gpu_event(item: object) -> GpuEvent | None:
    """1 つのイベントが GPU の展開されたイベントなら `GpuEvent` にする。そうでなければ空。"""
    if not isinstance(item, dict):
        return None
    if item.get("ph") != "X" or item.get("cat") not in GPU_CATEGORIES:
        return None
    name = item.get("name")
    start = item.get("ts")
    duration = item.get("dur")
    if not isinstance(name, str):
        return None
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        return None
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        return None
    return GpuEvent(name=name, start_us=float(start), dur_us=float(duration))


def gpu_events(trace: object) -> list[GpuEvent]:
    """trace から GPU のイベントだけを取り出す (C5)。"""
    events: list[GpuEvent] = []
    for item in _trace_events(trace):
        event = _gpu_event(item)
        if event is not None:
            events.append(event)
    return events


def read_trace(path: Path) -> list[GpuEvent]:
    """trace のファイルを読み、GPU のイベントを返す (gzip は中身で見分ける。C5)。"""
    body = path.read_bytes()
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    parsed: object = json.loads(body.decode("utf-8"))
    return gpu_events(parsed)


def busy_time_us(events: Sequence[GpuEvent]) -> float:
    """イベントの区間の和集合の長さ (重なりは 1 回だけ数える。C5)。"""
    if not events:
        return 0.0
    intervals = sorted((event.start_us, event.start_us + event.dur_us) for event in events)
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    total += current_end - current_start
    return total


def summarize(events: Sequence[GpuEvent], steps: int) -> Summary:
    """GPU のイベントを、区分ごとの時間と上位のカーネルに集計する (C5)。"""
    if not events:
        return Summary(
            steps=steps,
            wall_us=0.0,
            busy_us=0.0,
            gap_us=0.0,
            overlap_us=0.0,
            category_us={},
            kernels=(),
        )

    wall_us = max(event.start_us + event.dur_us for event in events) - min(
        event.start_us for event in events
    )
    busy_us = busy_time_us(events)
    gap_us = wall_us - busy_us
    overlap_us = max(0.0, sum(event.dur_us for event in events) - busy_us)

    category_us: dict[str, float] = {
        category: 0.0 for category in CATEGORY_LABELS if category != GAP
    }
    counts: defaultdict[str, int] = defaultdict(int)
    totals_us: defaultdict[str, float] = defaultdict(float)
    categories: dict[str, str] = {}
    for event in events:
        category = classify(event.name)
        category_us[category] += event.dur_us
        categories.setdefault(event.name, category)
        counts[event.name] += 1
        totals_us[event.name] += event.dur_us

    kernels = tuple(
        sorted(
            (
                KernelTotal(
                    name=name,
                    category=categories[name],
                    count=counts[name],
                    total_us=totals_us[name],
                )
                for name in totals_us
            ),
            key=lambda kernel: (-kernel.total_us, kernel.name),
        )
    )
    return Summary(
        steps=steps,
        wall_us=wall_us,
        busy_us=busy_us,
        gap_us=gap_us,
        overlap_us=overlap_us,
        category_us=category_us,
        kernels=kernels,
    )


def _milliseconds(us: float) -> str:
    """マイクロ秒を、ミリ秒の小数 3 桁にする (C5)。"""
    return f"{us / 1000.0:.3f}"


def _percent(part_us: float, whole_us: float) -> str:
    """`whole` に対する `part` の割合を、小数 3 桁にする (C5)。"""
    if whole_us <= 0:
        return "0.000"
    return f"{part_us / whole_us * 100.0:.3f}"


def format_summary(summary: Summary, *, top: int, source: str) -> str:
    """集計を、区分ごとの合計・1 ステップの平均・割合と、上位のカーネルの一覧にする (C5)。"""
    lines: list[str] = [
        f"# {source}",
        f"steps: {summary.steps}",
        f"wall: {_milliseconds(summary.wall_us)} ms",
        f"busy: {_milliseconds(summary.busy_us)} ms",
        f"gap: {_milliseconds(summary.gap_us)} ms",
    ]
    if summary.overlap_us > 0:
        lines.append(
            f"overlap: {_milliseconds(summary.overlap_us)} ms"
            " (ストリーム間の重なり。割合の合計が 100 を超える原因)"
        )
    lines.append("")
    lines.append("## 区分ごとの時間")
    lines.append("| 区分 | 合計 (ms) | 1 ステップ (ms) | 割合 (%) |")
    for category, label in CATEGORY_LABELS.items():
        if category == GAP:
            continue
        total_us = summary.category_us.get(category, 0.0)
        lines.append(
            f"| {label} | {_milliseconds(total_us)} |"
            f" {_milliseconds(total_us / summary.steps)} |"
            f" {_percent(total_us, summary.wall_us)} |"
        )
    lines.append(
        f"| {CATEGORY_LABELS[GAP]} | {_milliseconds(summary.gap_us)} |"
        f" {_milliseconds(summary.gap_us / summary.steps)} |"
        f" {_percent(summary.gap_us, summary.wall_us)} |"
    )
    lines.append("")
    lines.append(f"## 上位のカーネル (上位 {top} 件)")
    lines.append("| 名前 | 区分 | 回数 | 合計 (ms) |")
    for kernel in summary.kernels[:top]:
        lines.append(
            f"| {kernel.name} | {CATEGORY_LABELS[kernel.category]} |"
            f" {kernel.count} | {_milliseconds(kernel.total_us)} |"
        )
    lines.append("")
    other_kernels = [kernel for kernel in summary.kernels if kernel.category == OTHER][:top]
    lines.append(f"## {CATEGORY_LABELS[OTHER]}の上位 (上位 {top} 件)")
    lines.append("| 名前 | 回数 | 合計 (ms) |")
    for kernel in other_kernels:
        lines.append(f"| {kernel.name} | {kernel.count} | {_milliseconds(kernel.total_us)} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """trace ごとに集計して標準出力に出す。GPU のイベントが 0 件の trace があれば終了 1 (C5)。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="計測の窓のステップ数 (1 以上。`/metrics` の増分から得る)",
    )
    parser.add_argument("--top", type=int, default=20, help="上位に並べる件数 (既定 20)")
    parser.add_argument("traces", nargs="+", type=Path, help="回収した trace のファイル")
    args = parser.parse_args(argv)

    if args.steps < 1:
        parser.error("--steps は 1 以上にする")
    if args.top < 1:
        parser.error("--top は 1 以上にする")

    exit_code = 0
    for trace_path in args.traces:
        events = read_trace(trace_path)
        if not events:
            print(f"GPU のイベントがない: {trace_path}")
            exit_code = 1
            continue
        summary = summarize(events, args.steps)
        print(format_summary(summary, top=args.top, source=str(trace_path)))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
