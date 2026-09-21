"""2 台の間の all-reduce の帯域を計測する、自前のベンチマーク (design.md 「netcheck」)。

ライセンス: このリポジトリ (Apache-2.0。PLAN.md) の一部として、自分たちで書いたコードである。
nccl-tests のコードは 1 行も参照・複製していない (requirements 11.2、11.3)。

## 動かし方 (Spark の上、固定した vLLM のイメージのコンテナの中で)

vLLM のトラブルシュートの文書が示す `torchrun` の形をそのまま使う (`static` の rendezvous を
使う理由も、同じ文書に書かれている: `c10d` は複数ノードで DNS の解決に失敗することがあるため)。

    torchrun --nnodes 2 \\
        --nproc-per-node 1 \\
        --rdzv_backend static \\
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \\
        --node-rank $NODE_RANK \\
        allreduce_bench.py [--min-bytes N] [--max-bytes N] [--factor N]
        [--warmup-iters N] [--measure-iters N]

大きさの上限と、温め・計測の回数は、既定の値がこの下の「決まり」のとおりで、A/B の短い回の
ためにコマンドの引数で上書きできる。環境変数は読まない (NCCL の設定は、コンテナの環境変数
として、構成 (`serve netcheck bandwidth` が組み立てる docker の引数) から渡る。この
スクリプト自身は `os.environ` を読まない)。

## 決まり (design.md 「netcheck」より)

- 集団通信は all-reduce
- メッセージの大きさは 1 MiB から 1 GiB まで 4 倍ずつ (1、4、16、64、256 MiB、1 GiB の 6 つ)
- 大きさごとに、温めを 5 回、計測を 20 回
- `algbw = S / t`、`busbw = algbw × 2(n−1)/n` (n = world size)。n=2 では係数が 1 になるので、
  `busbw` と `algbw` は同じ値になる
- 単位は 10 進 (1 GB = 1e9 B)。この module は `algbw`/`busbw` をまず 10 進の GB/s として
  計算し、`serving_kit.types.BandwidthSample` の項目名 (`algbw_gbps`、`busbw_gbps`) に
  合わせて、8 を掛けて 10 進の Gbps (ビット/秒) に直した値を JSON に出す
- 出力は 1 行の JSON。rank 0 だけが、標準出力に出す。それ以外の行 (進捗、警告) は標準エラー
  に出し、標準出力に混ぜない

## `busbw` の定義の出典

nccl-tests の `doc/PERFORMANCE.md` (BSD ライセンス) — https://github.com/NVIDIA/nccl-tests
(このプロジェクトの `research.md` の §e-5 が、2026-09-21 に一次資料として引用したものを、
そのまま出典として使う。要件 11.3 によりこのスクリプトの実装のために nccl-tests のソース
そのものは開いていない)。原文の抜粋 (research.md §e-5 からの孫引き):

    "algbw = S/t"
    "In all cases, we need n-1 additions and n assignments for each element. […]
     we need 2(n-1) data transfers (x number of elements) to perform an allReduce
     operation."
    "B = S/t * (2*(n-1)/n) = algbw * (2*(n-1)/n)"

この定義 (足し算の式) だけを、自分たちの言葉で `compute_algbw_gbps` / `compute_busbw_gbps`
として実装した。nccl-tests のソースコード (C/C++ の実装、コマンドライン引数の処理、出力の
書式など) は、参照も複製もしていない。

## JSON の出力の形

    {
      "world_size": 2,
      "dtype": "float32",
      "warmup_iters": 5,
      "measure_iters": 20,
      "torch_version": "...",
      "nccl_version": "...",
      "samples": [
        {"size_bytes": 1048576, "time_s": {"n": 20, "mean": ..., "min": ..., "max": ...,
         "stdev": ...}, "algbw_gbps": ..., "busbw_gbps": ...},
        ...
      ]
    }

`samples` の各要素の `size_bytes` / `algbw_gbps` / `busbw_gbps` は、
`serving_kit.types.BandwidthSample` と同じ項目名を持つ (task 4.4 が、この JSON から
`BandwidthSample` / `BandwidthRun` を組み立てられる形にするため)。`time_s` は、この
スクリプト独自の補足情報 (`BandwidthSample` にはない) で、4.4 は無視してよい。

## 計算の部分と、GPU を使う部分の分け方

大きさの列を作る、`algbw`/`busbw` を計算する、単位を変える、結果の JSON を組み立てる
--- という**計算だけの部分**は、`torch` を import しなくても呼べる、純粋な関数に分けている
(このファイルの前半)。`torch` の import は `_import_torch()` の中だけで行い、それを呼ぶのは
`run_benchmark()` (`main()` の中でだけ呼ばれる) だけである。これにより、`torch` の入って
いない Mac でも、このファイルを import して、計算の部分だけを単体で試験できる
(`serving/tests/unit/test_payload.py`)。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final

# --- 決まりの既定値 (design.md 「netcheck」) -------------------------------------------------

_MEBIBYTE: Final[int] = 1 << 20
_GIBIBYTE: Final[int] = 1 << 30

DEFAULT_MIN_BYTES: Final[int] = _MEBIBYTE
"""メッセージの大きさの下限 (1 MiB)。"""

DEFAULT_MAX_BYTES: Final[int] = _GIBIBYTE
"""メッセージの大きさの上限 (1 GiB)。"""

DEFAULT_SIZE_FACTOR: Final[int] = 4
"""大きさを増やす倍率 (4 倍ずつ)。"""

DEFAULT_WARMUP_ITERS: Final[int] = 5
"""大きさごとの、温めの回数。"""

DEFAULT_MEASURE_ITERS: Final[int] = 20
"""大きさごとの、計測の回数。"""

_DECIMAL_GIGA: Final[float] = 1_000_000_000.0
"""10 進の giga (1 GB = 1e9 B)。GiB (2^30) と混同しないこと。"""

_BITS_PER_BYTE: Final[float] = 8.0

_DTYPE_NAME: Final[str] = "float32"
_DTYPE_ELEMENT_SIZE_BYTES: Final[int] = 4


# --- 計算だけの、純粋な関数 (torch を import しない) ------------------------------------------


def default_message_sizes_bytes(
    *,
    min_bytes: int = DEFAULT_MIN_BYTES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    factor: int = DEFAULT_SIZE_FACTOR,
) -> tuple[int, ...]:
    """メッセージの大きさの列を作る (design.md の決まり: 1 MiB から 1 GiB まで 4 倍ずつ)。

    既定の引数では `(1 MiB, 4 MiB, 16 MiB, 64 MiB, 256 MiB, 1 GiB)` の 6 つになる。
    """
    if min_bytes <= 0:
        raise ValueError(f"min_bytes は正でなければならない: {min_bytes}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes は正でなければならない: {max_bytes}")
    if max_bytes < min_bytes:
        raise ValueError(f"max_bytes ({max_bytes}) が min_bytes ({min_bytes}) を下回っている")
    if factor <= 1:
        raise ValueError(f"factor は 2 以上でなければならない: {factor}")

    sizes: list[int] = []
    size = min_bytes
    while size <= max_bytes:
        sizes.append(size)
        size *= factor
    return tuple(sizes)


def element_count_for_size(
    size_bytes: int, *, element_size_bytes: int = _DTYPE_ELEMENT_SIZE_BYTES
) -> int:
    """メッセージの大きさ (バイト) から、確保するテンソルの要素数を決める。

    `element_size_bytes` で割り切れない大きさは、GPU の上のバッファの大きさが
    `size_bytes` と食い違ってしまうので、断る (design.md の 6 つの大きさは、いずれも
    4 バイトの `float32` で割り切れる)。
    """
    if size_bytes <= 0:
        raise ValueError(f"size_bytes は正でなければならない: {size_bytes}")
    if element_size_bytes <= 0:
        raise ValueError(f"element_size_bytes は正でなければならない: {element_size_bytes}")
    if size_bytes % element_size_bytes != 0:
        raise ValueError(
            f"size_bytes ({size_bytes}) が要素の大きさ ({element_size_bytes}) で割り切れない"
        )
    return size_bytes // element_size_bytes


def compute_algbw_gbps(size_bytes: int, time_s: float) -> float:
    """`algbw = S / t` を、10 進の Gbps (ビット/秒) で返す。

    まず 10 進の GB/s (`S` はバイト、`1 GB = 1e9 B`) を計算し、8 を掛けてビット/秒に
    直す (`serving_kit.types.BandwidthSample.algbw_gbps` の単位に合わせる)。

    例: 1 GiB (1,073,741,824 B) を 1 秒で送ると、`algbw` = 1.073741824 GB/s =
    8.589934592 Gbps になる。
    """
    if size_bytes <= 0:
        raise ValueError(f"size_bytes は正でなければならない: {size_bytes}")
    if time_s <= 0:
        raise ValueError(
            f"time_s は正でなければならない (0 や負は、計測できなかったことを示す): {time_s}"
        )
    gigabytes_per_second = size_bytes / time_s / _DECIMAL_GIGA
    return gigabytes_per_second * _BITS_PER_BYTE


def compute_busbw_gbps(algbw_gbps: float, world_size: int) -> float:
    """`busbw = algbw × 2(n−1)/n` を返す (NCCL の公式の Performance の文書の定義)。

    `world_size` (n) が 1 のとき、係数は 0 になる (集団通信が要らないため)。n=2 のとき、
    係数は 1 になるので、`busbw` と `algbw` は同じ値になる。
    """
    if world_size <= 0:
        raise ValueError(f"world_size は正でなければならない: {world_size}")
    if algbw_gbps < 0:
        raise ValueError(f"algbw_gbps は負にならない: {algbw_gbps}")
    coefficient = 2.0 * (world_size - 1) / world_size
    return algbw_gbps * coefficient


def summarize_times_s(times_s: Sequence[float]) -> dict[str, float | int | None]:
    """計測した経過時間 (秒) の列から、`n` / `mean` / `min` / `max` / `stdev` を作る。

    `times_s` が空、または 0 以下の値を 1 つでも含む場合は断る (0 や負は、計測できなかった
    ことを示すので、平均に混ぜない)。`n` が 1 のときは、標本標準偏差が定義できないので
    `stdev` は `None` にする。
    """
    if not times_s:
        raise ValueError("times_s は空にできない")
    for value in times_s:
        if value <= 0:
            raise ValueError(
                f"time_s は正でなければならない (0 や負は、計測できなかったことを示す): {value}"
            )

    count = len(times_s)
    return {
        "n": count,
        "mean": statistics.fmean(times_s),
        "min": min(times_s),
        "max": max(times_s),
        "stdev": statistics.stdev(times_s) if count >= 2 else None,
    }


def build_bandwidth_sample(
    size_bytes: int, times_s: Sequence[float], world_size: int
) -> dict[str, Any]:
    """1 つの大きさぶんの結果を、`BandwidthSample` に写せる形の辞書として組み立てる。"""
    stats = summarize_times_s(times_s)
    mean_time_s = stats["mean"]
    assert isinstance(mean_time_s, float)  # noqa: S101 -- summarize_times_s が保証する

    algbw_gbps = compute_algbw_gbps(size_bytes, mean_time_s)
    busbw_gbps = compute_busbw_gbps(algbw_gbps, world_size)
    return {
        "size_bytes": size_bytes,
        "time_s": stats,
        "algbw_gbps": algbw_gbps,
        "busbw_gbps": busbw_gbps,
    }


def build_bandwidth_report(
    *,
    samples: Sequence[Mapping[str, Any]],
    world_size: int,
    dtype: str,
    warmup_iters: int,
    measure_iters: int,
    torch_version: str,
    nccl_version: str,
    skipped_sizes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """`samples` と実行時の情報から、1 行の JSON にする直前の辞書を組み立てる。"""
    report: dict[str, Any] = {
        "world_size": world_size,
        "dtype": dtype,
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "torch_version": torch_version,
        "nccl_version": nccl_version,
        "samples": [dict(sample) for sample in samples],
    }
    if skipped_sizes:
        report["skipped_sizes"] = [dict(item) for item in skipped_sizes]
    return report


def render_report_json(report: Mapping[str, Any]) -> str:
    """結果を、改行を含まない 1 行の JSON の文字列にする。"""
    return json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# --- コマンドラインの引数 (環境変数は読まない) ------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="allreduce_bench.py",
        description="2 台の間の all-reduce の帯域を計測する (torchrun で動かす)。",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=DEFAULT_MIN_BYTES,
        help=f"メッセージの大きさの下限 (バイト、既定 {DEFAULT_MIN_BYTES} = 1 MiB)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"メッセージの大きさの上限 (バイト、既定 {DEFAULT_MAX_BYTES} = 1 GiB)。"
        "A/B の短い回のときに小さくできる",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=DEFAULT_SIZE_FACTOR,
        help=f"大きさを増やす倍率 (既定 {DEFAULT_SIZE_FACTOR})",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=DEFAULT_WARMUP_ITERS,
        help=f"大きさごとの温めの回数 (既定 {DEFAULT_WARMUP_ITERS})",
    )
    parser.add_argument(
        "--measure-iters",
        type=int,
        default=DEFAULT_MEASURE_ITERS,
        help=f"大きさごとの計測の回数 (既定 {DEFAULT_MEASURE_ITERS})。"
        "A/B の短い回のときに小さくできる",
    )
    return parser


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    return _build_arg_parser().parse_args(argv)


# --- GPU を使う部分 (torch の import は、ここから先の関数の中でだけ行う) -----------------------


def _import_torch() -> tuple[Any, Any]:
    """`torch` と `torch.distributed` を、呼ばれたときだけ import する。

    Mac には `torch` が入っていないので、この関数を呼ばない限り (= `run_benchmark` を
    呼ばない限り)、このファイルを import しても `ModuleNotFoundError` にならない。
    """
    import torch
    import torch.distributed as dist

    return torch, dist


def _nccl_version_string(torch_module: Any) -> str:
    """`torch.cuda.nccl.version()` の結果を、`"2.30.7"` のような文字列にする。

    版が読めない場合 (CUDA の入っていない torch など) は `"unknown"` を返す。
    """
    try:
        version_tuple = torch_module.cuda.nccl.version()
    except Exception:  # noqa: BLE001 -- 版が読めないことは、計測の失敗にしない
        return "unknown"
    return ".".join(str(part) for part in version_tuple)


def _measure_one_size(
    torch_module: Any,
    dist_module: Any,
    *,
    size_bytes: int,
    warmup_iters: int,
    measure_iters: int,
    device: Any,
) -> list[float] | None:
    """1 つの大きさについて、温めと計測を行い、全 rank の最大を取った経過時間の列を返す。

    バッファは、計測のループの外で 1 度だけ確保する (design.md の決まり)。GPU のメモリが
    足りずに確保できない場合は、警告を標準エラーに出して `None` を返す (呼び出し元は、
    その大きさを `skipped_sizes` に記録して、次の大きさに進む)。
    """
    element_count = element_count_for_size(size_bytes, element_size_bytes=_DTYPE_ELEMENT_SIZE_BYTES)

    try:
        buffer = torch_module.empty(element_count, dtype=torch_module.float32, device=device)
    except torch_module.cuda.OutOfMemoryError as exc:
        print(
            f"[allreduce_bench] size_bytes={size_bytes} のバッファの確保に失敗 "
            f"(GPU のメモリ不足): {exc}",
            file=sys.stderr,
        )
        return None

    try:
        buffer.fill_(1.0)

        for _ in range(warmup_iters):
            dist_module.all_reduce(buffer, op=dist_module.ReduceOp.SUM)
        torch_module.cuda.synchronize()

        local_times_s: list[float] = []
        for _ in range(measure_iters):
            torch_module.cuda.synchronize()
            started_at = time.perf_counter()
            dist_module.all_reduce(buffer, op=dist_module.ReduceOp.SUM)
            torch_module.cuda.synchronize()
            local_times_s.append(time.perf_counter() - started_at)

        # 時間は、全 rank の最大を取る (いちばん遅い rank が、集団通信の時間)。
        times_tensor = torch_module.tensor(local_times_s, dtype=torch_module.float64, device=device)
        dist_module.all_reduce(times_tensor, op=dist_module.ReduceOp.MAX)
        result: list[float] = times_tensor.tolist()
        return result
    finally:
        del buffer
        torch_module.cuda.empty_cache()


def run_benchmark(argv: Sequence[str] | None = None) -> int:
    """`torchrun` の下で 1 プロセスとして動き、計測して、rank 0 だけが JSON を 1 行出す。"""
    args = _parse_args(argv)
    torch, dist = _import_torch()

    dist.init_process_group(backend="nccl")
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device_count = torch.cuda.device_count()
        local_rank = rank % device_count if device_count else 0
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        sizes = default_message_sizes_bytes(
            min_bytes=args.min_bytes, max_bytes=args.max_bytes, factor=args.factor
        )

        samples: list[dict[str, Any]] = []
        skipped_sizes: list[dict[str, Any]] = []
        for size_bytes in sizes:
            print(f"[allreduce_bench] size_bytes={size_bytes} を計測中", file=sys.stderr)
            times_s = _measure_one_size(
                torch,
                dist,
                size_bytes=size_bytes,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                device=device,
            )
            if times_s is None:
                skipped_sizes.append({"size_bytes": size_bytes, "reason": "oom"})
                continue
            samples.append(build_bandwidth_sample(size_bytes, times_s, world_size))

        report = build_bandwidth_report(
            samples=samples,
            world_size=world_size,
            dtype=_DTYPE_NAME,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            torch_version=str(torch.__version__),
            nccl_version=_nccl_version_string(torch),
            skipped_sizes=skipped_sizes,
        )

        if rank == 0:
            # rank 0 だけが、標準出力に 1 行の JSON を出す。それ以外はすべて標準エラー。
            print(render_report_json(report))
    finally:
        dist.destroy_process_group()

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_benchmark(argv)


if __name__ == "__main__":
    raise SystemExit(main())
