"""起動の記録と、通信の記録から、事実を読み取る部品の試験 (tasks.md 2.2)。

**見本の記録は、実機の出力ではない。** research.md が引く、vLLM と NCCL の上流のソースと
issue の原文の文面から組み立てた見本である (research.md §a-3、§d-1、§d-2、§d-6、§d-7、
§d-9、§e-4)。実機の記録は、まだ採れていない (7.2 で採ったら、見本を足す)。ログの行頭には、
vLLM のログの接頭辞 (`(EngineCore_DP0 pid=123) INFO 09-21 12:00:00 [file.py:123]`) や、
NCCL の `<hostname>:<pid>:<tid> [<cudaDev>]` の形を添えて、探す文字列が行の途中に現れる
場合でも読めることを確かめる。

確かめること (design.md 「組み立てと読み取り › observe」、tasks.md 2.2 の完了の状態):

- 探す文字列を含む見本の記録から、期待した値が読める
- どの探す文字列も含まない記録は、すべて空 (`None` / `()` / `False`) で返る (断らない)
- 知っている失敗の種類のそれぞれについて、その文面を含む記録から、正しい種類と、
  前後の行の抜き出しが返る
- 失敗の文面が 2 種類ある記録では、`types.KnownFailure` の並び順で先に来るものを返す
  (`observe` module の docstring に書く決め方)
- 同じ項目が 2 度現れたとき (head と worker、再試行) は、最初に現れたものを返す
  (`observe` module の docstring に書く決め方)
- `observe` は `types` だけを読み込み、ファイルも時刻もネットワークも読まない (入出力のない
  純粋な関数)
- `observe.py` のすべての正規表現の定数が、「錨の文字列のあとに、一致しない 20 万文字の
  塊が続く 1 行」を与えても、二次関数的な後戻り (ReDoS) を起こさず、1 秒以内に返る
  (task 2.2 のレビューの差し戻しで足した回帰試験)
- 量指定子を持たない探索 (`_SPECULATIVE_CONFIG_MARKER`、`_KNOWN_FAILURE_PATTERNS`、
  `_NETWORK_RE`、`_IB_NO_DEVICE_RE`) は、`_MAX_LINE_LENGTH` (4,096 文字) より後ろに印が
  あっても読み落とさない (task 2.2 の 3 回目のレビューの差し戻しで足した回帰試験)。量指定子
  を持つ探索は、同じ長さでも 1 秒以内に返る (線形の時間の回帰)
- `observe.py` のすべての正規表現の定数が、切り詰めない側 (組 (a)) と切り詰める側 (組 (b))
  のどちらか一方に、もれなく、重複なく分類されている (分類の網羅のメタ試験)
"""

from __future__ import annotations

import ast
import inspect
import re
import time
from collections.abc import Callable

import pytest

from serving_kit import observe as o
from serving_kit.types import KnownFailure

# --- 見本の記録の組み立て ------------------------------------------------


def _log(*lines: str) -> str:
    return "\n".join(lines)


_IRRELEVANT_LAUNCH_LOG = _log(
    "(EngineCore_DP0 pid=123) INFO 09-21 12:00:00 [core.py:50] Starting vLLM engine",
    "(EngineCore_DP0 pid=123) INFO 09-21 12:00:01 [core.py:60] Loading tokenizer",
)

_IRRELEVANT_NCCL_LOG = _log(
    "spark-153d:123:124 [0] NCCL INFO Bootstrap : Using enp1s0f0np0:192.168.100.1<0>",
    "spark-153d:123:124 [0] NCCL INFO NCCL_SOCKET_IFNAME set to enp1s0f0np0",
)


# --- アテンションと MoE のバックエンド (research.md §d-1、§d-2) ---------


def test_attention_backend_and_candidates_are_read() -> None:
    """`Using %s attention backend out of potential backends: %s.` から読む。

    出どころ: research.md §d-2 の観察の表、`vllm/platforms/cuda.py` L536。
    """
    log_text = _log(
        _IRRELEVANT_LAUNCH_LOG,
        "(EngineCore_DP0 pid=123) INFO 09-21 12:00:02 [cuda.py:536] Using"
        " FLASHINFER_MLA_SPARSE_SM120 attention backend out of potential backends:"
        " ['TRITON_MLA', 'FLASHINFER_MLA_SPARSE_SM120'].",
    )
    observation = o.observe_launch(log_text)
    assert observation.attention_backend == "FLASHINFER_MLA_SPARSE_SM120"
    assert observation.attention_candidates == ("TRITON_MLA", "FLASHINFER_MLA_SPARSE_SM120")


def test_moe_backend_is_read() -> None:
    """`Using '{backend}' NvFp4 MoE backend out of potential backends: {...}.` から読む。

    出どころ: research.md §d-2 の観察の表、
    `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` L228-233。
    """
    log_text = _log(
        "(EngineCore_DP0 pid=123) INFO 09-21 12:00:03 [nvfp4.py:230] Using"
        " 'flashinfer_b12x' NvFp4 MoE backend out of potential backends:"
        " ['marlin', 'flashinfer_b12x'].",
    )
    observation = o.observe_launch(log_text)
    assert observation.moe_backend == "flashinfer_b12x"


# --- KV キャッシュ、ロードと起動の所要 (research.md §d-2) ----------------


def test_kv_cache_tokens_and_max_concurrency_note_are_read() -> None:
    """`GPU KV cache size: N tokens, Maximum concurrency for M tokens per request: X.XXx`。

    出どころ: research.md §d-2 の観察の表、`vllm/v1/core/kv_cache_utils.py` L2405-2406。
    """
    log_text = _log(
        "(EngineCore_DP0 pid=123) INFO 09-21 12:00:10 [kv_cache_utils.py:2405] GPU KV"
        " cache size: 2,300,000 tokens, Maximum concurrency for 163840 tokens per"
        " request: 14.04x",
    )
    observation = o.observe_launch(log_text)
    assert observation.kv_cache_tokens == 2_300_000
    assert (
        observation.max_concurrency_note
        == "Maximum concurrency for 163840 tokens per request: 14.04x"
    )


def test_kv_cache_gib_is_read_from_available_kv_cache_memory() -> None:
    """`Available KV cache memory: N GiB` から読む。

    出どころ: research.md §d-2 の観察の表、`vllm/v1/worker/gpu_worker.py` L645-648。
    """
    log_text = _log(
        "(EngineCore_DP0 pid=123) INFO 09-21 12:00:09 [gpu_worker.py:645] Available"
        " KV cache memory: 16.23 GiB",
    )
    observation = o.observe_launch(log_text)
    assert observation.kv_cache_gib == 16.23


def test_model_loading_time_is_read() -> None:
    """`Model loading took %s GiB memory and %.6f seconds` から読む。

    出どころ: research.md §d-2 の観察の表 (重みのロード時間)。
    """
    log_text = _log(
        "(EngineCore_DP0 pid=123) INFO 09-21 12:05:00 [default_loader.py:296] Model"
        " loading took 92.100000 GiB memory and 612.500000 seconds",
    )
    observation = o.observe_launch(log_text)
    assert observation.model_loading_gib == 92.1
    assert observation.model_loading_s == 612.5


def test_engine_init_seconds_is_read() -> None:
    """`init engine (profile, create kv cache, warmup model) took …` から読む。

    出どころ: research.md §d-2 の観察の表 (起動全体)。末尾の単位の書式は research.md でも
    省略 (`…`) されているので、`observe.py` は接頭ぎだけを確かな探す文字列にし、続く数値を
    そのまま読む (module の docstring に書く CONCERNS)。
    """
    log_text = _log(
        "(EngineCore_DP0 pid=123) INFO 09-21 12:12:20 [core.py:170] init engine"
        " (profile, create kv cache, warmup model) took 740.00 seconds",
    )
    observation = o.observe_launch(log_text)
    assert observation.engine_init_s == 740.0


def test_vllm_version_is_not_extracted_from_log_text() -> None:
    """`vllm_version` は、確かな探す文字列が research.md にも design.md にもないので、

    つねに `None` を返す (推測で文字列を作らない。CONCERNS)。`GET /version` は
    HTTP の応答で、`observe` はログの文字列だけを読む純粋な関数なので、ここでは扱わない。
    """
    log_text = _log(
        "(EngineCore_DP0 pid=123) INFO 09-21 12:00:00 [api_server.py:1] vLLM API"
        " server version 0.11.1",
    )
    observation = o.observe_launch(log_text)
    assert observation.vllm_version is None


# --- 投機的デコード (research.md §d-7) -----------------------------------


def test_speculative_config_seen_true_when_dumped() -> None:
    """起動時の設定のダンプに `SpeculativeConfig(...)` が出れば真になる。

    出どころ: research.md §d-7 (投機的デコードを確実に切る)。
    """
    log_text = _log(
        "(EngineCore_DP0 pid=123) DEBUG 09-21 12:00:00 [config.py:100] vllm_config:"
        " VllmConfig(..., speculative_config=SpeculativeConfig(method='mtp',"
        " num_speculative_tokens=1), ...)",
    )
    observation = o.observe_launch(log_text)
    assert observation.speculative_config_seen is True


def test_speculative_config_seen_false_when_absent() -> None:
    observation = o.observe_launch(_IRRELEVANT_LAUNCH_LOG)
    assert observation.speculative_config_seen is False


# --- 知っている失敗 (design.md 「観察」の表) ------------------------------


def test_pe_dim_assert_is_known_failure_with_excerpt() -> None:
    """`pe_dim must be 64` を含む記録が `pe_dim_assert` になる (issue #57578)。"""
    log_text = _log(
        "(EngineCore_DP0 pid=123) INFO 09-21 12:00:20 [worker.py:1] loading weights",
        "(EngineCore_DP0 pid=123) ERROR 09-21 12:00:21 [worker.py:2] RuntimeError:"
        " Worker failed with error 'concat_and_cache_mla,"
        " csrc/libtorch_stable/cache_kernels.cu:939, pe_dim must be 64 for"
        " fp8_ds_mla'",
        "(EngineCore_DP0 pid=123) ERROR 09-21 12:00:21 [worker.py:3] Traceback"
        " (most recent call last):",
    )
    observation = o.observe_launch(log_text)
    assert observation.known_failure is KnownFailure.PE_DIM_ASSERT
    assert any("pe_dim must be 64" in line for line in observation.failure_excerpt)
    assert len(observation.failure_excerpt) <= 40


def test_no_attention_backend_is_known_failure() -> None:
    """`No valid attention backend found` を含む記録が `no_attention_backend` になる。"""
    log_text = _log(
        "(EngineCore_DP0 pid=123) ERROR 09-21 12:00:21 [worker.py:2] ValueError:"
        " No valid attention backend found for cuda with"
        " AttentionSelectorConfig(head_size=512, dtype=torch.bfloat16,"
        " kv_cache_dtype=bfloat16, use_mla=True, use_sparse=True, ...)",
    )
    observation = o.observe_launch(log_text)
    assert observation.known_failure is KnownFailure.NO_ATTENTION_BACKEND


def test_no_kernel_image_is_known_failure() -> None:
    """`no kernel image is available for execution on the device` (research.md §a-3)。"""
    log_text = _log(
        "(EngineCore_DP0 pid=123) ERROR 09-21 12:00:21 [worker.py:2] RuntimeError:"
        " CUDA error: no kernel image is available for execution on the device",
    )
    observation = o.observe_launch(log_text)
    assert observation.known_failure is KnownFailure.NO_KERNEL_IMAGE


def test_kpool_block_size_is_known_failure() -> None:
    """`kpool indexer requires cache block_size` (research.md §d-2 の観察の表)。"""
    log_text = _log(
        "(EngineCore_DP0 pid=123) ERROR 09-21 12:00:21 [attention.py:1] ValueError:"
        " Glm5NextIndexerCache: kpool indexer requires cache block_size to be a"
        " multiple of index_kpool * 32 (128), got 64",
    )
    observation = o.observe_launch(log_text)
    assert observation.known_failure is KnownFailure.KPOOL_BLOCK_SIZE


def test_startup_memory_check_is_known_failure() -> None:
    """`is less than desired GPU memory utilization` (research.md §d-6)。"""
    log_text = _log(
        "(EngineCore_DP0 pid=123) ERROR 09-21 12:00:21 [gpu_worker.py:545]"
        " ValueError: Free memory on device (10.00/121.70 GiB) on startup is less"
        " than desired GPU memory utilization (0.92, 112.00 GiB).",
        "(EngineCore_DP0 pid=123) ERROR 09-21 12:00:21 [gpu_worker.py:546] Decrease"
        " GPU memory utilization or reduce GPU memory used by other processes.",
    )
    observation = o.observe_launch(log_text)
    assert observation.known_failure is KnownFailure.STARTUP_MEMORY_CHECK


def test_deep_gemm_missing_is_known_failure() -> None:
    """`requires DeepGEMM to be installed` (research.md §d-9)。"""
    log_text = _log(
        "(EngineCore_DP0 pid=123) ERROR 09-21 12:00:21 [sparse_indexer.py:1]"
        " RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM to be"
        " installed.",
    )
    observation = o.observe_launch(log_text)
    assert observation.known_failure is KnownFailure.DEEP_GEMM_MISSING


def test_no_known_failure_string_returns_none() -> None:
    """どの探す文字列も含まない記録は、`known_failure` が `None` になる (断らない)。"""
    observation = o.observe_launch(_IRRELEVANT_LAUNCH_LOG)
    assert observation.known_failure is None
    assert observation.failure_excerpt == ()


def test_two_known_failure_strings_prefer_the_earlier_enum_member() -> None:
    """失敗の文面が 2 種類ある記録では、`KnownFailure` の並び順で先のものを返す。

    ここでは `pe_dim_assert` (並びの先頭) と `no_attention_backend` を両方含む記録を渡し、
    `pe_dim_assert` が返ることを固定する (tasks.md 2.2 の完了の状態、observe module の
    docstring に書く決め方)。
    """
    log_text = _log(
        "(EngineCore_DP0 pid=123) ERROR 09-21 12:00:21 [worker.py:2] RuntimeError:"
        " Worker failed with error 'concat_and_cache_mla,"
        " csrc/libtorch_stable/cache_kernels.cu:939, pe_dim must be 64 for"
        " fp8_ds_mla'",
        "(EngineCore_DP0 pid=123) ERROR 09-21 12:00:22 [worker.py:2] ValueError:"
        " No valid attention backend found for cuda with ...",
    )
    observation = o.observe_launch(log_text)
    assert observation.known_failure is KnownFailure.PE_DIM_ASSERT


def test_failure_excerpt_is_capped_at_forty_lines() -> None:
    """前後の行の抜き出しは、`types.LaunchObservation.failure_excerpt` の上限 (40) を超えない。

    大きな記録 (前後 100 行ずつ) を渡しても、返る行数が 40 を超えないことを固定する。
    """
    before = [
        f"(EngineCore_DP0 pid=123) INFO 09-21 12:00:{i:02d} [x.py:{i}] line {i}" for i in range(100)
    ]
    after = [
        f"(EngineCore_DP0 pid=123) INFO 09-21 12:01:{i:02d} [x.py:{i}] line {i}" for i in range(100)
    ]
    failure = (
        "(EngineCore_DP0 pid=123) ERROR 09-21 12:00:59 [worker.py:2] RuntimeError:"
        " pe_dim must be 64 for fp8_ds_mla"
    )
    log_text = _log(*before, failure, *after)
    observation = o.observe_launch(log_text)
    assert observation.known_failure is KnownFailure.PE_DIM_ASSERT
    assert len(observation.failure_excerpt) <= 40
    assert any("pe_dim must be 64" in line for line in observation.failure_excerpt)


def test_empty_and_irrelevant_launch_logs_return_all_empty() -> None:
    """空の記録と、探す文字列を 1 つも含まない記録は、すべて空で返る。"""
    for log_text in ("", _IRRELEVANT_LAUNCH_LOG):
        observation = o.observe_launch(log_text)
        assert observation.vllm_version is None
        assert observation.attention_backend is None
        assert observation.attention_candidates == ()
        assert observation.moe_backend is None
        assert observation.kv_cache_tokens is None
        assert observation.kv_cache_gib is None
        assert observation.max_concurrency_note is None
        assert observation.model_loading_gib is None
        assert observation.model_loading_s is None
        assert observation.engine_init_s is None
        assert observation.speculative_config_seen is False
        assert observation.known_failure is None
        assert observation.failure_excerpt == ()


def test_duplicate_attention_backend_lines_use_the_first_occurrence() -> None:
    """同じ項目が 2 度現れたとき (head と worker、再試行) は、最初のものを返す。

    ここでは、1 回目 (head) と 2 回目 (worker、再試行) で異なるバックエンドが読める
    記録を渡し、1 回目の値が返ることを固定する (observe module の docstring の決め方)。
    """
    log_text = _log(
        "(EngineCore_DP0 pid=123) INFO 09-21 12:00:02 [cuda.py:536] Using"
        " FLASHINFER_MLA_SPARSE_SM120 attention backend out of potential backends:"
        " ['FLASHINFER_MLA_SPARSE_SM120'].",
        "(EngineCore_DP1 pid=456) INFO 09-21 12:00:03 [cuda.py:536] Using TRITON_MLA"
        " attention backend out of potential backends: ['TRITON_MLA'].",
    )
    observation = o.observe_launch(log_text)
    assert observation.attention_backend == "FLASHINFER_MLA_SPARSE_SM120"


# --- NCCL の記録 (research.md §e-4) --------------------------------------


def test_nccl_version_is_read() -> None:
    """`NCCL version [0-9.]+[^ ]*\\+cuda[0-9.]+` (`src/init.cc` の `VERSION_STRING`)。"""
    log_text = _log("spark-153d:123:124 [0] NCCL INFO NCCL version 2.30.7+cuda13.0")
    observation = o.observe_nccl(log_text)
    assert observation.nccl_version == "2.30.7+cuda13.0"


def test_network_ib_is_read() -> None:
    """`NCCL INFO Using network (IB|Socket)` (`src/init.cc:557`)。"""
    log_text = _log("spark-153d:123:124 [0] NCCL INFO Using network IB")
    observation = o.observe_nccl(log_text)
    assert observation.network == "IB"


def test_network_socket_is_read() -> None:
    log_text = _log("spark-153d:123:124 [0] NCCL INFO Using network Socket")
    observation = o.observe_nccl(log_text)
    assert observation.network == "Socket"


def test_ib_no_device_and_socket_fallback_are_read() -> None:
    """`NET/IB : No device found.` と `Using network Socket` の連言で、Socket 落ちを見る。

    出どころ: research.md §e-4 (「IB から Socket に落ちた」の判定)。
    """
    log_text = _log(
        "spark-153d:123:124 [0] NCCL INFO NET/IB : No device found.",
        "spark-153d:123:124 [0] NCCL INFO Using network Socket",
    )
    observation = o.observe_nccl(log_text)
    assert observation.ib_no_device is True
    assert observation.network == "Socket"


def test_ib_devices_line_is_read() -> None:
    """`NET/IB : Using` から、選ばれた IB デバイスの行を読む。"""
    log_text = _log(
        "spark-153d:123:124 [0] NCCL INFO NET/IB : Using [0]rocep1s0f0:1/RoCE; OOB"
        " enp1s0f0np0:192.168.100.1<0>",
    )
    observation = o.observe_nccl(log_text)
    assert observation.ib_devices_line == (
        "NET/IB : Using [0]rocep1s0f0:1/RoCE; OOB enp1s0f0np0:192.168.100.1<0>"
    )


def test_merged_nic_true_via_vnic_line() -> None:
    """`TOPO/NET : Made vNic` は、ndevs==1 では出ないので、直接の証拠になる。"""
    log_text = _log("spark-153d:123:124 [0] NCCL INFO TOPO/NET : Made vNic 0")
    observation = o.observe_nccl(log_text)
    assert observation.merged_nic is True


def test_merged_nic_true_via_ndevs_at_least_two() -> None:
    log_text = _log(
        "spark-153d:123:124 [0] NCCL INFO NET/IB : Made virtual device [0]"
        " name=mlx5_0+mlx5_1 speed=200000 ndevs=2",
    )
    observation = o.observe_nccl(log_text)
    assert observation.merged_nic is True


def test_merged_nic_false_when_only_a_single_device_is_made() -> None:
    """`Made virtual device` は `ndevs=1` の素のデバイスにも毎回出るので、束ねではない。"""
    log_text = _log(
        "spark-153d:123:124 [0] NCCL INFO NET/IB : Made virtual device [0]"
        " name=mlx5_0 speed=200000 ndevs=1",
    )
    observation = o.observe_nccl(log_text)
    assert observation.merged_nic is False


def test_coll_channels_is_read() -> None:
    """`%d coll channels, %d collnet channels` から、coll channels の数を読む。"""
    log_text = _log(
        "spark-153d:123:124 [0] NCCL INFO 4 coll channels, 4 collnet channels, 2"
        " nvls channels, 4 p2p channels",
    )
    observation = o.observe_nccl(log_text)
    assert observation.coll_channels == 4


def test_gdrdma_seen_is_read() -> None:
    """`via NET/IB/[0-9]+/GDRDMA` (DGX Spark では出ないのが正常)。"""
    log_text = _log(
        "spark-153d:123:124 [0] NCCL INFO Channel 00/04 : 0 [send] via NET/IB/0/GDRDMA",
    )
    observation = o.observe_nccl(log_text)
    assert observation.gdrdma_seen is True


def test_socket_channel_seen_is_read() -> None:
    """`[send|receive] via NET/Socket/[0-9]+`。"""
    log_text = _log(
        "spark-153d:123:124 [0] NCCL INFO Channel 00/04 : 0 [receive] via NET/Socket/0",
    )
    observation = o.observe_nccl(log_text)
    assert observation.socket_channel_seen is True


def test_empty_and_irrelevant_nccl_logs_return_all_empty() -> None:
    for log_text in ("", _IRRELEVANT_NCCL_LOG):
        observation = o.observe_nccl(log_text)
        assert observation.nccl_version is None
        assert observation.network is None
        assert observation.ib_no_device is False
        assert observation.ib_devices_line is None
        assert observation.merged_nic is False
        assert observation.coll_channels is None
        assert observation.gdrdma_seen is False
        assert observation.socket_channel_seen is False


def test_duplicate_network_lines_use_the_first_occurrence() -> None:
    """head と worker で経路が違って読めても、最初 (head) の値を返す (決め方の固定)。"""
    log_text = _log(
        "spark-153d:123:124 [0] NCCL INFO Using network IB",
        "spark-5083:456:457 [0] NCCL INFO Using network Socket",
    )
    observation = o.observe_nccl(log_text)
    assert observation.network == "IB"


# --- 正規表現の後戻り (ReDoS) の回帰試験 ---------------------------------
#
# task 2.2 のレビューの差し戻し (指摘 1): `_NCCL_VERSION_RE`
# (`NCCL version ([0-9.]+[^ ]*\+cuda[0-9.]+)`) は、文字集合が重なる 2 つの無制限の量指定子
# (`[0-9.]+` と `[^ ]*`) が隣り合い、そのあとに一致しないことのあるリテラル (`\+cuda`) が
# 続くので、二次関数的な後戻りが起きた (実測: 正規表現だけで 80,000 文字に 1.39 秒、
# `observe_nccl()` の全体で 160,000 文字に 5.3 秒)。
#
# ここでは、`observe.py` のすべての正規表現の定数について、「錨の文字列のあとに、一致しない
# 20 万文字の塊 (数字とピリオド、空白なしの英字、その定数の文字集合に合う文字) が続く 1 行」
# を与えても、1 秒以内に返ることを固定する。

_STRESS_FILLER_LENGTH = 200_000
_STRESS_TIME_LIMIT_S = 1.0


def _assert_returns_within_time_limit(call: Callable[[], object]) -> None:
    """`call` が `_STRESS_TIME_LIMIT_S` 秒以内に返ることを確かめる。"""
    started = time.perf_counter()
    call()
    elapsed = time.perf_counter() - started
    assert elapsed < _STRESS_TIME_LIMIT_S, (
        f"{elapsed:.3f} 秒かかった (上限 {_STRESS_TIME_LIMIT_S} 秒。後戻りを見直すこと)"
    )


def test_attention_backend_re_is_linear_time() -> None:
    """`\\S+` は、直後の必須の空白 (文字集合の外) で区切られるので、後戻りが線形になる。"""
    log_text = "Using " + ("x" * _STRESS_FILLER_LENGTH)
    _assert_returns_within_time_limit(lambda: o.observe_launch(log_text))


def test_moe_backend_re_is_linear_time() -> None:
    """`[^']+` は、直後の必須の `'` (文字集合の外) で区切られるので、後戻りが線形になる。"""
    log_text = "Using '" + ("x" * _STRESS_FILLER_LENGTH)
    _assert_returns_within_time_limit(lambda: o.observe_launch(log_text))


def test_kv_cache_size_re_is_linear_time() -> None:
    log_text = "GPU KV cache size: " + ("1," * (_STRESS_FILLER_LENGTH // 2))
    _assert_returns_within_time_limit(lambda: o.observe_launch(log_text))


def test_kv_cache_memory_re_is_linear_time() -> None:
    """塊は句読点のない数字だけにする (`.` を混ぜると、一致したときに `float()` が壊れる

    多小数点の文字列になり、この試験の目的 (後戻りの速さ) とは別の誤りになるため)。
    """
    log_text = "Available KV cache memory: " + ("1" * _STRESS_FILLER_LENGTH)
    _assert_returns_within_time_limit(lambda: o.observe_launch(log_text))


def test_model_loading_re_is_linear_time() -> None:
    log_text = "Model loading took " + ("1" * _STRESS_FILLER_LENGTH)
    _assert_returns_within_time_limit(lambda: o.observe_launch(log_text))


def test_engine_init_re_is_linear_time() -> None:
    """末尾の量指定子には何も続かないので、必ず一致し、後戻りは起こらない。

    塊は句読点のない数字だけにする (`.` を混ぜると、一致した捕捉が多小数点の文字列になり、
    `observe_launch` 自身の `float()` が `ValueError` で落ちる。後戻りの速さとは別の誤り
    になるので避ける)。
    """
    log_text = "init engine (profile, create kv cache, warmup model) took " + (
        "1" * _STRESS_FILLER_LENGTH
    )
    _assert_returns_within_time_limit(lambda: o.observe_launch(log_text))


def test_known_failure_patterns_are_linear_time() -> None:
    """6 つの知っている失敗の探す文字列は、いずれも量指定子のないリテラルだけである。"""
    log_text = "x" * _STRESS_FILLER_LENGTH
    _assert_returns_within_time_limit(lambda: o.observe_launch(log_text))


def test_nccl_version_re_is_linear_time() -> None:
    """指摘 1 そのもの。`+cuda` を含まない数字とピリオドの塊で、後戻りを起こす。

    直す前の `_NCCL_VERSION_RE` (`[0-9.]+[^ ]*\\+cuda[0-9.]+`) では、この試験が落ちる
    (RED_PHASE_OUTPUT に、直す前の `observe.py` に当てた出力を、そのまま書く)。
    """
    log_text = "NCCL version " + ("1." * (_STRESS_FILLER_LENGTH // 2))
    _assert_returns_within_time_limit(lambda: o.observe_nccl(log_text))


def test_network_re_is_linear_time() -> None:
    """`(IB|Socket)` はどちらも量指定子を持たないリテラルの選択なので、後戻りが起きない。"""
    log_text = "NCCL INFO Using network " + ("x" * _STRESS_FILLER_LENGTH)
    _assert_returns_within_time_limit(lambda: o.observe_nccl(log_text))


def test_ib_no_device_re_is_linear_time() -> None:
    log_text = "NET/IB : No device foun" + ("x" * _STRESS_FILLER_LENGTH)
    _assert_returns_within_time_limit(lambda: o.observe_nccl(log_text))


def test_ib_using_re_is_linear_time() -> None:
    """`.*$` は、単独の量指定子のあとに 0 幅の行末の錨が続くだけなので、後戻りが起きない。"""
    log_text = "NET/IB : Using" + ("x" * _STRESS_FILLER_LENGTH)
    _assert_returns_within_time_limit(lambda: o.observe_nccl(log_text))


def test_merged_nic_virtual_device_re_is_linear_time() -> None:
    """`[^ ]+` は必須の空白で区切られ、続く `.* ndevs=` は単独の量指定子なので線形になる。"""
    log_text = "NET/IB : Made virtual device [0] name=" + ("x" * _STRESS_FILLER_LENGTH)
    _assert_returns_within_time_limit(lambda: o.observe_nccl(log_text))


def test_merged_nic_vnic_re_is_linear_time() -> None:
    log_text = "TOPO/NET : Made vNi" + ("x" * _STRESS_FILLER_LENGTH)
    _assert_returns_within_time_limit(lambda: o.observe_nccl(log_text))


def test_coll_channels_re_is_linear_time() -> None:
    log_text = "1234567890" * (_STRESS_FILLER_LENGTH // 10)
    _assert_returns_within_time_limit(lambda: o.observe_nccl(log_text))


def test_socket_channel_re_is_linear_time() -> None:
    """末尾の量指定子には何も続かないので、後戻りが起こりようがない。"""
    log_text = "[send] via NET/Socket/" + ("1" * _STRESS_FILLER_LENGTH)
    _assert_returns_within_time_limit(lambda: o.observe_nccl(log_text))


def test_gdrdma_re_is_linear_time() -> None:
    log_text = "via NET/IB/" + ("1" * _STRESS_FILLER_LENGTH)
    _assert_returns_within_time_limit(lambda: o.observe_nccl(log_text))


# --- 構造の上限 (1 行を _MAX_LINE_LENGTH で切り詰める) ---------------------
#
# task 2.2 の 2 回目のレビューの差し戻し。`observe.py` は、記録を行に分け、1 行ごとに
# `_MAX_LINE_LENGTH` (4,096 文字) で切り詰めてから正規表現を当てる。これで、個々の
# 正規表現にどんな後戻りの性質が残っていても、1 行あたりの手間が定数で抑えられる。


def test_content_beyond_the_line_length_limit_is_not_read() -> None:
    """上限より後ろにある文字列は、読まれない (決まりとして固定する)。"""
    padding = "x" * o._MAX_LINE_LENGTH
    log_text = "Model loading took " + padding + " 1.0 GiB memory and 2.0 seconds"
    observation = o.observe_launch(log_text)
    assert observation.model_loading_gib is None
    assert observation.model_loading_s is None


def test_content_just_before_the_line_length_limit_is_still_read() -> None:
    """切り詰めの上限より短い、ふつうの行は、これまでどおり読める。"""
    padding = "x" * (o._MAX_LINE_LENGTH - 60)
    log_text = f"{padding} Available KV cache memory: 16.23 GiB"
    assert len(log_text) <= o._MAX_LINE_LENGTH
    observation = o.observe_launch(log_text)
    assert observation.kv_cache_gib == 16.23


def test_failure_excerpt_lines_are_truncated_to_the_line_length_limit() -> None:
    """`failure_excerpt` の行も、`_MAX_LINE_LENGTH` を超えない。"""
    long_line = "pe_dim must be 64" + ("y" * (o._MAX_LINE_LENGTH * 2))
    observation = o.observe_launch(long_line)
    assert observation.known_failure is KnownFailure.PE_DIM_ASSERT
    assert observation.failure_excerpt
    assert all(len(line) <= o._MAX_LINE_LENGTH for line in observation.failure_excerpt)


# --- 正規表現の後戻り (ReDoS): 錨が繰り返される形 --------------------------
#
# task 2.2 の 2 回目のレビューの差し戻し (指摘)。`_MERGED_NIC_VIRTUAL_DEVICE_RE` に、
# 指摘 1 とは別の形の後戻りが残っていた。錨の文字列「NET/IB : Made virtual device [0]
# name=x 」を 1 行の中で反復し、`ndevs=` を一度も出さない入力で、`re.search` が錨の
# 出現ごとに `.*` の全域の後戻りをやり直すため、二次関数的になる (実測: 5,000 回で
# 0.487 秒、10,000 回で 1.973 秒、20,000 回で 7.927 秒、50,000 回で 52.8 秒)。この形は、
# 錨が 1 回だけの `*_is_linear_time` (上) では見つからない。
#
# ここでは、それぞれの正規表現の定数を、2 つの新しい形で追加で叩く:
#   (ii)  錨の文字列そのものを 5 万回連ねて、末尾のリテラルを一度も出さない 1 行
#   (iii) (ii) に準じた行を 1,000 行並べた記録 (行数に対して線形であることを確かめる)
#
# `_KNOWN_FAILURE_PATTERNS` (6 件) は、量指定子を持たないリテラルだけなので、「錨」と
# 「末尾のリテラル」の区別がなく、この形の後戻りの対象にならない (`test_known_failure_
# patterns_are_linear_time` (上) がすでに固定している)。ここには含めない。

_ANCHOR_REPEAT_COUNT = 50_000
"""(ii) 錨の文字列を連ねる回数 (レビューの実測と同じ桁)。"""

_MULTILINE_LINE_COUNT = 1_000
"""(iii) の行数。"""

_MULTILINE_ANCHOR_REPEAT_COUNT = 200
"""(iii) の 1 行あたりの錨の反復回数。

(ii) と同じ 5 万回を 1,000 行ぶん作ると、正規表現を掛ける前の文字列そのものの組み立てだけ
で、後戻りの速さの試験とは無関係に、実務的でない時間とメモリ (数 GB) を使う。切り詰めの
上限 (`_MAX_LINE_LENGTH` = 4,096 文字) を確実に超える行を 1,000 行並べれば、「構造の上限が
行数に対して線形であること」を確かめる目的には十分なので、1 行あたりの反復はここまで
減らす (200 回でも、どの錨の長さでも 4,096 文字を上回る)。
"""

_ANCHOR_STRESS_CASES: tuple[tuple[str, str, str], ...] = (
    # (識別子, 錨の文字列, "launch" か "nccl" か)
    ("attention_backend", "Using ", "launch"),
    ("moe_backend", "Using '", "launch"),
    ("kv_cache_size", "GPU KV cache size: ", "launch"),
    ("kv_cache_memory", "Available KV cache memory: ", "launch"),
    ("model_loading", "Model loading took ", "launch"),
    ("engine_init", "init engine (profile, create kv cache, warmup model) took ", "launch"),
    ("nccl_version", "NCCL version ", "nccl"),
    ("network", "NCCL INFO Using network ", "nccl"),
    ("ib_no_device", "NET/IB : No device foun", "nccl"),
    ("ib_using", "NET/IB : Using", "nccl"),
    ("merged_nic_virtual_device", "NET/IB : Made virtual device [0] name=x ", "nccl"),
    ("merged_nic_vnic", "TOPO/NET : Made vNic ", "nccl"),
    ("coll_channels", "1234567890", "nccl"),
    ("socket_channel", "[send] via NET/Socket/", "nccl"),
    ("gdrdma", "via NET/IB/", "nccl"),
)
_ANCHOR_STRESS_IDS = [case[0] for case in _ANCHOR_STRESS_CASES]


def _observe(kind: str, log_text: str) -> object:
    return o.observe_launch(log_text) if kind == "launch" else o.observe_nccl(log_text)


@pytest.mark.parametrize("name, anchor, kind", _ANCHOR_STRESS_CASES, ids=_ANCHOR_STRESS_IDS)
def test_anchor_repeated_in_one_line_is_linear_time(name: str, anchor: str, kind: str) -> None:
    """(ii) 錨の文字列そのものを 5 万回連ねて、末尾のリテラルを一度も出さない 1 行。"""
    log_text = anchor * _ANCHOR_REPEAT_COUNT
    _assert_returns_within_time_limit(lambda: _observe(kind, log_text))


@pytest.mark.parametrize("name, anchor, kind", _ANCHOR_STRESS_CASES, ids=_ANCHOR_STRESS_IDS)
def test_anchor_repeated_across_many_lines_is_linear_time(
    name: str, anchor: str, kind: str
) -> None:
    """(iii) (ii) に準じた行を 1,000 行並べた記録。"""
    line = anchor * _MULTILINE_ANCHOR_REPEAT_COUNT
    log_text = "\n".join([line] * _MULTILINE_LINE_COUNT)
    _assert_returns_within_time_limit(lambda: _observe(kind, log_text))


# --- 依存の向きと、入出力のなさ -----------------------------------------

ALLOWED_IMPORT_ROOTS = frozenset({"__future__", "collections", "re", "typing", "serving_kit"})
FORBIDDEN_IMPORTS = frozenset(
    {"os", "subprocess", "socket", "pathlib", "time", "random", "httpx", "datetime"}
)


def test_observe_module_reads_nothing_from_the_outside() -> None:
    """`observe` は `types` だけを読み込み、ファイルも時刻もネットワークも読まない。"""
    tree = ast.parse(inspect.getsource(o))
    roots: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "相対の読み込みがある"
            assert node.module is not None
            roots.add(node.module.split(".")[0])
            modules.add(node.module)

    assert roots <= ALLOWED_IMPORT_ROOTS, f"許していない読み込み: {roots - ALLOWED_IMPORT_ROOTS}"
    assert not (roots & FORBIDDEN_IMPORTS)
    assert {module for module in modules if module.startswith("serving_kit")} <= {
        "serving_kit.types",
    }


# --- 読み落としの回帰 (task 2.2 の 3 回目のレビューの差し戻し) -----------
#
# `_truncated_lines` が、量指定子を持たない探索 (`_SPECULATIVE_CONFIG_MARKER`、
# `_KNOWN_FAILURE_PATTERNS` の 6 件、`_NETWORK_RE`、`_IB_NO_DEVICE_RE`) にまで及んでいて、
# 長い 1 行 (vLLM は全設定を 1 行でダンプする。数キロバイトになりうる) の `_MAX_LINE_LENGTH`
# (4,096 文字) より後ろにある印を読み落としていた。実測: 全長 6,109 文字の設定のダンプの
# 行で、`speculative_config_seen` が `False` (正しくは `True`)。ここでは、これらの探索が、
# `_MAX_LINE_LENGTH` より後ろでも読めることを固定する。

_PAST_TRUNCATION_PADDING = "z" * (o._MAX_LINE_LENGTH + 2000)
"""`_MAX_LINE_LENGTH` を確実に超える詰め物 (詰め物だけで 4,096 + 2,000 文字)。"""


def test_speculative_config_marker_past_the_line_length_limit_is_still_read() -> None:
    """`SpeculativeConfig(` が `_MAX_LINE_LENGTH` より後ろにあっても読める (設定の 1 行ダンプ)。

    出どころ: research.md §d-7。全長 6,000 文字を超える 1 行で確かめる。
    """
    log_text = (
        "(EngineCore_DP0 pid=123) DEBUG 09-21 12:00:00 [config.py:100] vllm_config:"
        " VllmConfig(model=" + _PAST_TRUNCATION_PADDING + ", speculative_config="
        "SpeculativeConfig(method='mtp', num_speculative_tokens=1), ...)"
    )
    assert len(log_text) > 6000
    observation = o.observe_launch(log_text)
    assert observation.speculative_config_seen is True


_KNOWN_FAILURE_PHRASES: tuple[tuple[KnownFailure, str], ...] = (
    (KnownFailure.PE_DIM_ASSERT, "pe_dim must be 64"),
    (KnownFailure.NO_ATTENTION_BACKEND, "No valid attention backend found"),
    (
        KnownFailure.NO_KERNEL_IMAGE,
        "no kernel image is available for execution on the device",
    ),
    (KnownFailure.KPOOL_BLOCK_SIZE, "kpool indexer requires cache block_size"),
    (
        KnownFailure.STARTUP_MEMORY_CHECK,
        "is less than desired GPU memory utilization",
    ),
    (KnownFailure.DEEP_GEMM_MISSING, "requires DeepGEMM to be installed"),
)
_KNOWN_FAILURE_PHRASE_IDS = [failure.value for failure, _ in _KNOWN_FAILURE_PHRASES]


@pytest.mark.parametrize("failure, phrase", _KNOWN_FAILURE_PHRASES, ids=_KNOWN_FAILURE_PHRASE_IDS)
def test_known_failure_phrase_past_the_line_length_limit_is_still_read(
    failure: KnownFailure, phrase: str
) -> None:
    """6 件の失敗の文面それぞれが、`_MAX_LINE_LENGTH` より後ろにあっても、種類が読める。"""
    log_text = _PAST_TRUNCATION_PADDING + " " + phrase
    assert len(log_text) > o._MAX_LINE_LENGTH
    observation = o.observe_launch(log_text)
    assert observation.known_failure is failure


@pytest.mark.parametrize("failure, phrase", _KNOWN_FAILURE_PHRASES, ids=_KNOWN_FAILURE_PHRASE_IDS)
def test_known_failure_phrase_past_the_line_length_limit_appears_in_excerpt(
    failure: KnownFailure, phrase: str
) -> None:
    """抜き出しに、上限より後ろにある失敗の文面が入り、どの行も窓の上限を超えない。

    窓の決まり: 一致した行だけ、一致の位置の前 `_EXCERPT_WINDOW_BEFORE` 文字から
    `_MAX_LINE_LENGTH` 文字を取るので、抜き出しのどの行も、この 2 つの和を超えない。
    """
    log_text = _PAST_TRUNCATION_PADDING + " " + phrase
    observation = o.observe_launch(log_text)
    assert observation.known_failure is failure
    assert any(phrase in line for line in observation.failure_excerpt)
    window_limit = o._EXCERPT_WINDOW_BEFORE + o._MAX_LINE_LENGTH
    assert all(len(line) <= window_limit for line in observation.failure_excerpt)


def test_network_socket_past_the_line_length_limit_is_still_read() -> None:
    """`NCCL INFO Using network Socket` が `_MAX_LINE_LENGTH` より後ろにあっても読める。"""
    log_text = _PAST_TRUNCATION_PADDING + " NCCL INFO Using network Socket"
    assert len(log_text) > o._MAX_LINE_LENGTH
    observation = o.observe_nccl(log_text)
    assert observation.network == "Socket"


def test_ib_no_device_past_the_line_length_limit_is_still_read() -> None:
    """`NET/IB : No device found.` が `_MAX_LINE_LENGTH` より後ろにあっても読める。"""
    log_text = _PAST_TRUNCATION_PADDING + " NCCL INFO NET/IB : No device found."
    assert len(log_text) > o._MAX_LINE_LENGTH
    observation = o.observe_nccl(log_text)
    assert observation.ib_no_device is True


# --- 線形の時間の回帰 (1,250 万文字の 1 行) -------------------------------
#
# 量指定子を持たない探索を、切り詰めずに掛けても、既存の時間の上限の助け (1 秒。
# `_STRESS_TIME_LIMIT_S`) の中で返り、かつ値が読めることを固定する。

_LARGE_LINE_LENGTH = 12_500_000
"""8: 線形の時間の回帰に使う、1 行の長さ (1,250 万文字)。"""


def _call_within_time_limit(call: Callable[[], object]) -> object:
    """`call` が `_STRESS_TIME_LIMIT_S` 秒以内に返ることを確かめ、返り値を返す。

    `_assert_returns_within_time_limit` (上) と同じ時間の上限を使うが、返り値も
    確かめたいので、別に持つ。
    """
    started = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - started
    assert elapsed < _STRESS_TIME_LIMIT_S, (
        f"{elapsed:.3f} 秒かかった (上限 {_STRESS_TIME_LIMIT_S} 秒。後戻りを見直すこと)"
    )
    return result


def test_pe_dim_assert_is_read_within_the_time_limit_on_a_large_line() -> None:
    log_text = ("x" * _LARGE_LINE_LENGTH) + "pe_dim must be 64"
    observation = _call_within_time_limit(lambda: o.observe_launch(log_text))
    assert isinstance(observation, type(o.observe_launch("")))
    assert observation.known_failure is KnownFailure.PE_DIM_ASSERT


def test_speculative_config_marker_is_read_within_the_time_limit_on_a_large_line() -> None:
    log_text = ("x" * _LARGE_LINE_LENGTH) + "SpeculativeConfig("
    observation = _call_within_time_limit(lambda: o.observe_launch(log_text))
    assert isinstance(observation, type(o.observe_launch("")))
    assert observation.speculative_config_seen is True


def test_network_ib_is_read_within_the_time_limit_on_a_large_line() -> None:
    log_text = ("x" * _LARGE_LINE_LENGTH) + "NCCL INFO Using network IB"
    observation = _call_within_time_limit(lambda: o.observe_nccl(log_text))
    assert isinstance(observation, type(o.observe_nccl("")))
    assert observation.network == "IB"


def test_ib_no_device_is_read_within_the_time_limit_on_a_large_line() -> None:
    log_text = ("x" * _LARGE_LINE_LENGTH) + "NET/IB : No device found."
    observation = _call_within_time_limit(lambda: o.observe_nccl(log_text))
    assert isinstance(observation, type(o.observe_nccl("")))
    assert observation.ib_no_device is True


# --- 分類の網羅のメタ試験 (同じ種類の差し戻しを止める仕組み) ----------------
#
# module の docstring の「構造の上限」の 2 つの組: (a) 量指定子を持たない探索は切り詰めない
# 行に、(b) 量指定子を持つ探索は切り詰めた行に当てる。ここでは、その分類が module のすべての
# 正規表現の定数を、もれなく、重複なく覆っていることを固定する。新しい探す定数を、どちらにも
# 分類せずに足したら、この試験群が落ちる。

_UNTRUNCATED_PATTERNS: tuple[re.Pattern[str], ...] = (
    o._NETWORK_RE,
    o._IB_NO_DEVICE_RE,
    *(pattern for _, pattern in o._KNOWN_FAILURE_PATTERNS),
)
"""組 (a): 量指定子を持たず、切り詰めていない行 (`raw_lines`) に当てる探索。"""

_TRUNCATED_PATTERNS: tuple[re.Pattern[str], ...] = (
    o._ATTENTION_BACKEND_RE,
    o._MOE_BACKEND_RE,
    o._KV_CACHE_SIZE_RE,
    o._KV_CACHE_MEMORY_RE,
    o._MODEL_LOADING_RE,
    o._ENGINE_INIT_RE,
    o._NCCL_VERSION_RE,
    o._IB_USING_RE,
    o._MERGED_NIC_VIRTUAL_DEVICE_RE,
    o._MERGED_NIC_VNIC_RE,
    o._COLL_CHANNELS_RE,
    o._SOCKET_CHANNEL_RE,
    o._GDRDMA_RE,
)
"""組 (b): 量指定子を持ち、切り詰めた行 (`lines`) に当てる探索。"""


def _all_module_level_regex_patterns(module: object) -> list[re.Pattern[str]]:
    """`vars(module)` を歩いて `re.Pattern` をすべて集める。

    `_KNOWN_FAILURE_PATTERNS` のような、`(ラベル, パターン)` の組を持つタプルの中も見る。
    """
    patterns: list[re.Pattern[str]] = []
    for value in vars(module).values():
        if isinstance(value, re.Pattern):
            patterns.append(value)
        elif isinstance(value, tuple):
            for item in value:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], re.Pattern):
                    patterns.append(item[1])
    return patterns


def test_every_regex_pattern_is_classified_as_truncated_or_untruncated() -> None:
    """(9-a) module のすべての正規表現の定数が、2 つの表の和集合と完全に一致する。"""
    all_patterns = _all_module_level_regex_patterns(o)
    classified = list(_UNTRUNCATED_PATTERNS) + list(_TRUNCATED_PATTERNS)

    assert len(all_patterns) == len(set(id(p) for p in all_patterns)), (
        "vars(observe) から見つけた正規表現の定数に重複がある (集め方を見直すこと)"
    )
    assert len(classified) == len(set(id(p) for p in classified)), (
        "分類の表 (_UNTRUNCATED_PATTERNS / _TRUNCATED_PATTERNS) に重複がある"
    )
    assert {id(p) for p in all_patterns} == {id(p) for p in classified}, (
        "module の正規表現の定数と、分類の表の和集合が一致しない"
        " (新しい定数を、分類せずに足していないか確かめること)"
    )


def test_untruncated_patterns_have_no_quantifiers() -> None:
    """(9-b) 切り詰めない側 (組 (a)) は、量指定子を持たない。

    エスケープされた文字 (`\\.` など) と、文字集合 (`[...]`) を取り除いた残りに、
    量指定子 (`*`、`+`、`?`、`{`) が現れないことを確かめる。読み落としのなさを優先する
    探索は、後戻りの危険が構造的にないことが、この分類の前提になる。
    """
    for pattern in _UNTRUNCATED_PATTERNS:
        source = pattern.pattern
        without_escapes: list[str] = []
        i = 0
        while i < len(source):
            if source[i] == "\\" and i + 1 < len(source):
                i += 2
                continue
            without_escapes.append(source[i])
            i += 1
        residual = re.sub(r"\[[^\]]*\]", "", "".join(without_escapes))
        for quantifier in ("*", "+", "?", "{"):
            assert quantifier not in residual, (
                f"{pattern.pattern!r} に量指定子 {quantifier!r} が残っている"
                " (切り詰めない側に分類できない)"
            )


_ANCHOR_CASE_PATTERNS: dict[str, re.Pattern[str]] = {
    "attention_backend": o._ATTENTION_BACKEND_RE,
    "moe_backend": o._MOE_BACKEND_RE,
    "kv_cache_size": o._KV_CACHE_SIZE_RE,
    "kv_cache_memory": o._KV_CACHE_MEMORY_RE,
    "model_loading": o._MODEL_LOADING_RE,
    "engine_init": o._ENGINE_INIT_RE,
    "nccl_version": o._NCCL_VERSION_RE,
    "network": o._NETWORK_RE,
    "ib_no_device": o._IB_NO_DEVICE_RE,
    "ib_using": o._IB_USING_RE,
    "merged_nic_virtual_device": o._MERGED_NIC_VIRTUAL_DEVICE_RE,
    "merged_nic_vnic": o._MERGED_NIC_VNIC_RE,
    "coll_channels": o._COLL_CHANNELS_RE,
    "socket_channel": o._SOCKET_CHANNEL_RE,
    "gdrdma": o._GDRDMA_RE,
}
"""`_ANCHOR_STRESS_CASES` の識別子から、対応する `observe.py` の正規表現の定数への対応。"""


def test_anchor_stress_cases_cover_every_truncated_side_pattern() -> None:
    """(9-c) `_ANCHOR_STRESS_CASES` (錨の反復の試験) が、切り詰める側の定数を、すべて覆う。"""
    case_names = {name for name, _, _ in _ANCHOR_STRESS_CASES}
    assert case_names <= set(_ANCHOR_CASE_PATTERNS), (
        f"{case_names - set(_ANCHOR_CASE_PATTERNS)} の対応が _ANCHOR_CASE_PATTERNS にない"
        " (試験の対応表を足すこと)"
    )
    covered_ids = {id(_ANCHOR_CASE_PATTERNS[name]) for name in case_names}
    truncated_ids = {id(p) for p in _TRUNCATED_PATTERNS}
    assert truncated_ids <= covered_ids, (
        f"切り詰める側の定数のうち、錨の反復の試験が覆っていないものがある: "
        f"{[p.pattern for p in _TRUNCATED_PATTERNS if id(p) not in covered_ids]}"
    )
