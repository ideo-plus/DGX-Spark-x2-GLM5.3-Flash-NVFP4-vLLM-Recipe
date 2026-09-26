"""確認済みの自前イメージから TP=2 の構成を生成する。

変種は 3 つ: 初回確認用の `smoke`、計測用の `full`、`full` にモデル付属の MTP を足した
`full-mtp` (`--spec-tokens N`)。`full-mtp` の構成の名前は `p2-nope-tp2-mtp<N>` になる。

`--nccl-thread-names` を付けると、`NCCL_SET_THREAD_NAME=1` の env を根拠つきで足し、構成の
名前を `-tn` 付きにする (どの `--variant` にも付けられる)。

`--torch-profiler` を付けると、`--profiler-config` (torch プロファイラー、出力先
`/logs/torch-profile`、スタックの記録と `key_averages` の表の書き出しは切る) を根拠つきで
args に足し、構成の名前を `-prof` 付きにする (どの `--variant` にも付けられる)。

`--weights <名前>` を付けると、`serving/weights/<名前>.manifest.json` (手元で変換した重みの、
コミットした派生のマニフェスト) から重みの参照 (元の重みの参照と変換の条件) を組み、
`weights` の節と `mount-weights` を派生の重み (`{remote_root}/models/<名前>`、読み取り専用)
に差し替える。構成の名前は基の変種名の直後に `-<名前>` を付ける (どの `--variant` にも付け
られる)。マニフェストが無い、`kind` が `derived` でない、`derivation.name` が `<名前>` と違う、
元の重みが `p1-nvfp4-tp2` の重みと違うときは、生成の前に断る。

`--vllm-overlay <名前>` を付けると、`experiments/k2-vllm-overlay/<名前>.json` (固定した vLLM の
直したファイルの定義) に並ぶファイルごとに、`serving/payload/vllm-overlay/<名前>/<道筋>` の写し
(`serve push` で `{remote_root}/payload/` へ配る) を、イメージの中の vLLM の同じファイルに重ねる
読み取り専用の bind mount を、根拠つきで docker の節の末尾に足す (ADR 0007 の第 2a 段、#73)。
構成の名前は `--weights` の接尾辞の直後に `-ov-<名前>` を付ける (どの `--variant` にも付け
られる)。定義が無い、`name` が `<名前>` と違う、`vllm_commit` が固定の commit と違う、道筋が
`vllm/` で始まる相対の道筋でない、写しが無い、mount の鍵が重なるときは、生成の前に断る。
写しが「固定のソース + パッチ」と一致することは、生成器ではなく試験 (`test_vllm_overlay.py`)
が固定する。

`full` と `full-mtp` は、重みの読み込み方を `--load-format instanttensor` にするのを既定とし、
根拠つきで args に足す (構成の名前は変えない。`#50`)。`--load-format auto` を付けると既定を
外し、出力は `#50` より前の既定と同じになる。`smoke` には既定を付けない。

`--load-format` (`instanttensor` / `fastsafetensors` / `runai_streamer`) と
`--safetensors-load-strategy` (`eager` / `prefetch`) を明示すると、重みの読み込み方を根拠つきで
args に足し、構成の名前を `-lf-<値>` / `-sls-<値>` 付きにする。明示した値は既定を置き換える
(`runai_streamer` は `_` を `-` に替えて `-lf-runai-streamer`)。`--load-format` はどの
`--variant` にも付けられる。

`--safetensors-load-strategy` が効くのは `--load-format` を付けない読み方だけなので、既定で
`instanttensor` を持つ `full` と `full-mtp` で strategy が効く構成は、`--load-format auto` と
組にしたものだけになる。`--load-format` を付けずに指定すると、生成の前に断る (明示の
`--load-format` と重ねた構成は作れるが、strategy は効かない)。既定を持たない `smoke` には、
単独で付けられる。
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_PATH = ROOT / "serving/config/configs.toml"
WEIGHTS_DIR = ROOT / "serving/weights"
"""コミットした重みのマニフェストの置き場 (`--weights <名前>` が読む)。"""
OVERLAY_DIR = ROOT / "experiments/k2-vllm-overlay"
"""固定した vLLM の直したファイルの定義 (`<名前>.json`) の置き場 (`--vllm-overlay` が読む)。"""
OVERLAY_PAYLOAD_DIR = ROOT / "serving/payload/vllm-overlay"
"""直したファイルの写しの置き場 (`<名前>/<道筋>`。`serve push` が配る)。"""
OVERLAY_PAYLOAD_SUBDIR = "payload/vllm-overlay"
"""Spark の側の、写しの置き場 (`{remote_root}` の下)。"""
VLLM_COMMIT = "0961bbae2894d574be790d219651824eb199318e"
"""イメージの vLLM の固定の commit (重ねる定義の `vllm_commit` と一致しなければ断る)。"""
IMAGE_SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"
"""イメージの中の vLLM の導入先 (`uv pip install --system` が置く場所)。"""
OVERLAY_SUFFIX = "-ov-"
OVERLAY_MOUNT_KEY_PREFIX = "mount-vllm-overlay-"
ARGS_COMMENT = "# --- vllm serve の引数 (この順で並ぶ)"
"""p1 の節の、docker の設定の末尾を示すコメント (重ねる mount は、この直前に入る)。"""
OVERLAY_MOUNT_SOURCE = (
    "https://github.com/vllm-project/vllm/blob/"
    "0961bbae2894d574be790d219651824eb199318e/docker/Dockerfile"
)
OVERLAY_MOUNT_QUOTE = (
    "ARG PYTHON_VERSION=3.12 … uv pip install --system dist/*.whl --verbose … "
    '"/usr/local/lib/python${PYTHON_VERSION}/dist-packages/nvpl/lib"'
)
"""固定の commit の `docker/Dockerfile` の 26 行目、1062 行目、971 行目の原文。"""
OVERLAY_MOUNT_WHY = (
    "ADR 0007 の第 2a 段 (#73)。固定した vLLM (0961bbae) のこのファイルに "
    "experiments/k2-vllm-overlay のパッチを当てた写し (serving/payload/vllm-overlay/<名前>/…。"
    "serve push で配る) を、イメージを作り直さずに読み取り専用の bind mount で重ねる。"
    "重ねる先は、イメージの中の vLLM の導入先 (uv pip install --system が置く dist-packages) の"
    "同じ道筋"
)
EXPECTED_IMAGE_ID = "sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90"
IMAGE_SEEN_AS = "vllm-nope:0961bbae-fi070"
IMAGE_MEASURED = "docs/results/2026-09-22-nope-build.md"
SOURCE_NAME = "p1-nvfp4-tp2"
TARGET_NAME = "p2-nope-tp2-smoke"
DERIVED_MOUNT_WHY = (
    "手元で変換した重みを、書き換えない形で見せる (固定したマニフェストと、ファイルごとの "
    "SHA-256 を守る。requirements 3.4)"
)
"""派生の重みを読み取り専用で結び付ける理由。p1 の `mount-weights` の理由を写す。"""
FULL_TARGET_NAME = "p2-nope-tp2-full"
NCCL_SOURCE_NAME = "netcheck-bandwidth"
FULL_MTP_VARIANT_NAME = "full-mtp"
MTP_TARGET_NAME_FORMAT = "p2-nope-tp2-mtp{n}"
MTP_SOURCE = (
    "https://github.com/vllm-project/vllm/blob/"
    "0961bbae2894d574be790d219651824eb199318e/vllm/config/speculative.py"
)
MTP_QUOTE = (
    "The name of the speculative method to use. … The number of speculative tokens, if provided. "
    "It will default to the number in the draft model config if present, otherwise, it is required."
)
THREAD_NAMES_SUFFIX = "-tn"
THREAD_NAMES_ENV_KEY = "nccl-set-thread-name"
THREAD_NAMES_SOURCE = "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html"
THREAD_NAMES_QUOTE = (
    "Give more meaningful names to NCCL CPU threads to ease debugging and analysis."
)
THREAD_NAMES_WHY = (
    "#17 で推定した、Worker_TP の中で張り付くメイン以外のスレッドが NCCL の proxy かを、"
    "top -H のスレッドの名前 (NCCL Progress など) で確かめる。この確認のときだけ付ける"
)
PROFILER_SUFFIX = "-prof"
PROFILER_ARG_KEY = "profiler-config"
PROFILER_FLAG = "--profiler-config"
PROFILER_DIR = "/logs/torch-profile"
PROFILER_VALUE = (
    '{"profiler":"torch","torch_profiler_dir":"/logs/torch-profile",'
    '"torch_profiler_with_stack":false,"torch_profiler_dump_cuda_time_total":false}'
)
PROFILER_SOURCE = (
    "https://github.com/vllm-project/vllm/blob/"
    "0961bbae2894d574be790d219651824eb199318e/vllm/config/profiler.py"
)
PROFILER_QUOTE = (
    "Which profiler to use. … Directory to save torch profiler traces. "
    "Both AsyncLLM's CPU traces and worker's traces (CPU & GPU) will be saved under this "
    "directory. Note that it must be an absolute path. … "
    "If `True`, enables stack tracing in the torch profiler. Enabled by default as it is useful "
    "for debugging. Can be disabled via --profiler-config.torch_profiler_with_stack=false CLI "
    "flag. … If `True`, dumps total CUDA time in torch profiler traces. Enabled by default."
)
PROFILER_WHY = (
    "#42 (ADR 0006 の K2 の判断)。1 ステップの GPU の時間の内訳を torch プロファイラーで測る。"
    "記録は /start_profile と /stop_profile の間だけ。trace は mount-logs で結び付けた /logs の"
    "下に書き、serve logs で回収する。この計測のときだけ付ける。#48 で、2026-09-26 の初回の"
    "計測では /stop_profile の書き出しで head の VLLM::Worker_TP が oom-killer に止められた"
    "(head は起動時の available が約 1 GB)。スタックの記録 (torch_profiler_with_stack=false) と"
    "key_averages の表の書き出し (torch_profiler_dump_cuda_time_total=false) を切って、書き出し"
    "のメモリを減らす"
)
LOAD_FORMAT_SUFFIX = "-lf-"
LOAD_FORMAT_ARG_KEY = "load-format"
LOAD_FORMAT_FLAG = "--load-format"
LOAD_CONFIG_SOURCE = (
    "https://github.com/vllm-project/vllm/blob/"
    "0961bbae2894d574be790d219651824eb199318e/vllm/config/load.py"
)
FASTSAFETENSORS_DOC_SOURCE = (
    "https://github.com/vllm-project/vllm/blob/"
    "0961bbae2894d574be790d219651824eb199318e/docs/models/extensions/fastsafetensor.md"
)
LOAD_FORMAT_PROVENANCE: Mapping[str, tuple[str, str]] = {
    "instanttensor": (
        LOAD_CONFIG_SOURCE,
        '"instanttensor" will load the Safetensors weights on CUDA devices using InstantTensor, '
        "which enables distributed loading with pipelined prefetching and fast direct I/O.",
    ),
    "fastsafetensors": (
        FASTSAFETENSORS_DOC_SOURCE,
        "Using fastsafetensors library enables loading model weights to GPU memory by leveraging "
        "GPU direct storage. … To enable this feature, use the `--load-format fastsafetensors` "
        "command-line argument",
    ),
    "runai_streamer": (
        LOAD_CONFIG_SOURCE,
        '"runai_streamer" will load the Safetensors weights using Run:ai Model Streamer.',
    ),
}
# opt-out の値は、vLLM 固定 commit (0961bbae) の `LoadConfig.load_format` の既定値 "auto" に
# 合わせる (「既定の読み方 = vLLM の既定に任せる」と読みが一致する)。
LOAD_FORMAT_OPT_OUT = "auto"
DEFAULT_LOAD_FORMAT_VALUE = "instanttensor"
# #50 の既定の根拠。source と quote は #46 の `LOAD_FORMAT_PROVENANCE["instanttensor"]` を使う。
DEFAULT_LOAD_FORMAT_WHY = (
    "#50 (#37 の実測、2026-09-26)。--load-format instanttensor で、重みの読み込みが "
    "363 秒 (head) から 32 秒に、起動全体が約 11 分から約 3 分になった。同じ構成の probe で、"
    "decode の tok/s (code/en 14.06、code/ja 13.99、prose/en 14.07、prose/ja 13.88) は"
    "既定の読み方 (14.13、14.01、14.12、13.98) と同じ範囲、toolcall は 5/5 で正解・不正解が"
    "投機なしの回と同じ、needle 8k は正解、壊れの印は 0 件。計測のたびに起動するので、"
    "full と full-mtp の既定にする。外すときは --load-format auto (出力は #50 より前の既定と同じ)"
)
LOAD_STRATEGY_SUFFIX = "-sls-"
LOAD_STRATEGY_ARG_KEY = "safetensors-load-strategy"
LOAD_STRATEGY_FLAG = "--safetensors-load-strategy"
LOAD_STRATEGY_PROVENANCE: Mapping[str, tuple[str, str]] = {
    "eager": (
        LOAD_CONFIG_SOURCE,
        '"eager": The entire file is read into CPU memory upfront before loading. This is '
        "recommended for models on network filesystems (e.g., Lustre, NFS) as it avoids "
        "inefficient random reads, significantly speeding up model initialization. However, it "
        "uses more CPU RAM.",
    ),
    "prefetch": (
        LOAD_CONFIG_SOURCE,
        '"prefetch": Checkpoint files are read into the OS page cache before workers load them, '
        "speeding up the model loading phase. Useful on network or high-latency storage.",
    ),
}


class DerivedWeights(NamedTuple):
    """`--weights` で選ぶ、手元で変換した重み (コミットした派生マニフェストから組む)。

    `origin_*` は元の重み (p1 の重みと一致することを生成の前に確かめる)。`tool`・`commit`・
    `args`・`target_pattern` は変換の条件で、そのまま構成の TOML に写す。`manifest` は
    変換の結果のマニフェストの名前である。
    """

    name: str
    origin_repo: str
    origin_revision: str
    origin_manifest: str
    manifest: str
    tool: str
    commit: str
    args: tuple[str, ...]
    target_pattern: str


class VllmOverlay(NamedTuple):
    """`--vllm-overlay` で選ぶ、固定した vLLM の直したファイルの重ね。

    `files` は `vllm/` で始まる相対の道筋で、定義 (`<名前>.json`) の並び (mount の並び) である。
    """

    name: str
    files: tuple[str, ...]


class LoadFormat(NamedTuple):
    """args に書く `--load-format` の値と、その理由。"""

    value: str
    why: str


DEFAULT_LOAD_FORMAT = LoadFormat(DEFAULT_LOAD_FORMAT_VALUE, DEFAULT_LOAD_FORMAT_WHY)


class Variant(NamedTuple):
    """生成する構成の違いをまとめたもの。

    `arg_overrides` は `(args の鍵, 旧値, 新値, why)` の並びで、`render` が
    `[configs.{SOURCE_NAME}.args.{鍵}]` の `value` と `why` をこの順に置き換える。
    空なら置き換えず、p1 の値と根拠をそのまま残す。

    `spec_tokens` は、モデル付属の MTP (投機的デコード) を有効にする下書きのトークン数である。
    `None` なら投機の指定を足さない (smoke と full は `None`)。

    `nccl_thread_names` が真なら、`render` は env に `NCCL_SET_THREAD_NAME=1` を根拠つきで
    足し、名前は `-tn` 付きになる (`with_thread_names` が作る)。

    `torch_profiler` が真なら、`render` は args に `--profiler-config` を根拠つきで足し、
    名前は `-prof` 付きになる (`with_torch_profiler` が作る)。

    `load_format` は、重みの読み込み方の値と理由 (`None` なら足さない)。`full` / `full-mtp` は
    `DEFAULT_LOAD_FORMAT` を持ち、`with_load_format` が置き換え、`without_load_format` が外す。
    明示 `--load-format` の名前は `-lf-<値>` 付きになる。

    `safetensors_load_strategy` は、safetensors の読み方の値 (`None` なら足さない)。
    `with_load_strategy` が作り、名前は `-sls-<値>` 付きになる。

    `weights` は、手元で変換した重みを使うときの指定 (`None` なら p1 の重みのまま)。
    `with_weights` が作り、名前は `-<名前>` 付きになる。

    `vllm_overlay` は、固定した vLLM の直したファイルを重ねるときの指定 (`None` なら重ねない)。
    `with_vllm_overlay` が作り、名前は `-ov-<名前>` 付きになる。`render` は、ファイルごとの
    読み取り専用の bind mount を docker の節の末尾に足す。
    """

    name: str
    description: str
    env_comment: str
    nccl_debug_why: str
    arg_overrides: tuple[tuple[str, str, str, str], ...]
    spec_tokens: int | None = None
    nccl_thread_names: bool = False
    torch_profiler: bool = False
    load_format: LoadFormat | None = None
    safetensors_load_strategy: str | None = None
    weights: DerivedWeights | None = None
    vllm_overlay: VllmOverlay | None = None


SMOKE = Variant(
    name=TARGET_NAME,
    description=(
        "NoPE 修正イメージ (vllm-nope:0961bbae-fi070) で、"
        "2 台 TP=2 の初回の起動と短い応答だけを確かめる。"
        "短い文脈 (4096) と同時実行 1。実重みでの起動は未確認"
    ),
    env_comment=(
        "# --- 環境変数 (p1-nvfp4-tp2 の 3 つ + 初回の起動だけに付ける NCCL の記録の 3 つ) -----"
    ),
    nccl_debug_why=(
        "初回の起動だけ、NCCL の記録を採る (手順書 2.1)。"
        "経路の確定の行 (Using network …) は INFO の段でしか出ない。"
        "2 台とも IB であることを、この記録から確かめる"
    ),
    arg_overrides=(
        (
            "max-model-len",
            "163840",
            "4096",
            (
                "初回は起動と短い応答だけを確かめる運用上の選択。"
                "文脈長の上限を 4096 に絞り、初回に必要な KV 容量の条件を抑える。"
                "長文脈は次の段階で確かめる"
            ),
        ),
        (
            "max-num-seqs",
            "16",
            "1",
            (
                "初回は同時に 1 本しか送らない運用上の選択。"
                "線形アテンションの状態は同時の本数に比例するので、起動と応答の確認に要らないぶんを取らない。"
                "同時実行は次の段階で確かめる"
            ),
        ),
    ),
)

FULL = Variant(
    name=FULL_TARGET_NAME,
    description=(
        "NoPE 修正イメージ (vllm-nope:0961bbae-fi070) で、"
        "2 台 TP=2 の prefill・並列・長文脈の計測を行う。"
        "文脈長 163840 と同時実行 16 は p1-nvfp4-tp2 のまま。実機では未確認"
    ),
    env_comment=(
        "# --- 環境変数 (p1-nvfp4-tp2 の 3 つ + IB の確認のために"
        "起動のたびに付ける NCCL の記録の 3 つ) -----"
    ),
    nccl_debug_why=(
        "起動のたびに NCCL の記録を採る。"
        "経路の確定の行 (Using network …) は INFO の段でしか出ない。"
        "2 台とも IB であることを、この記録から毎回確かめる"
    ),
    arg_overrides=(),
    load_format=DEFAULT_LOAD_FORMAT,
)

VARIANTS: Mapping[str, Variant] = {"smoke": SMOKE, "full": FULL}


def mtp_variant(spec_tokens: int) -> Variant:
    """`--variant full-mtp --spec-tokens N` の構成を作る。

    `full` と同じ構成 (文脈長 163840、同時実行 16) に、モデル付属の MTP を下書き
    `spec_tokens` トークンで有効にする。`spec_tokens` は 1 以上。
    """
    if spec_tokens < 1:
        raise ValueError("--spec-tokens は 1 以上にする")
    return Variant(
        name=MTP_TARGET_NAME_FORMAT.format(n=spec_tokens),
        description=(
            "NoPE 修正イメージ (vllm-nope:0961bbae-fi070) で、"
            "2 台 TP=2 の full と同じ構成に、モデル付属の MTP を"
            f"下書き {spec_tokens} トークンで有効にする。実機では未確認"
        ),
        env_comment=FULL.env_comment,
        nccl_debug_why=FULL.nccl_debug_why,
        arg_overrides=(),
        load_format=DEFAULT_LOAD_FORMAT,
        spec_tokens=spec_tokens,
    )


def with_thread_names(variant: Variant) -> Variant:
    """`--nccl-thread-names` の構成 (基の変種 + NCCL のスレッドの名前)。"""
    return variant._replace(
        name=f"{variant.name}{THREAD_NAMES_SUFFIX}",
        description=(
            f"{variant.description}。NCCL の CPU スレッドに名前を付け (NCCL_SET_THREAD_NAME=1)、"
            "top -H で張り付くスレッドが NCCL の proxy かを確かめる (#17)"
        ),
        nccl_thread_names=True,
    )


def with_torch_profiler(variant: Variant) -> Variant:
    """`--torch-profiler` の構成 (基の変種 + torch プロファイラーの設定)。"""
    return variant._replace(
        name=f"{variant.name}{PROFILER_SUFFIX}",
        description=(
            f"{variant.description}。torch プロファイラーを有効にし (--profiler-config)、"
            f"/start_profile と /stop_profile の間の GPU の記録を {PROFILER_DIR} に書く (#42)"
        ),
        torch_profiler=True,
    )


def with_load_format(variant: Variant, load_format: str) -> Variant:
    """`--load-format <値>` の構成 (基の変種 + 重みの読み込み方)。許さない値は ValueError。"""
    if load_format not in LOAD_FORMAT_PROVENANCE:
        raise ValueError(f"--load-format に書けるのは {', '.join(LOAD_FORMAT_PROVENANCE)} だけ")
    return variant._replace(
        name=f"{variant.name}{LOAD_FORMAT_SUFFIX}{load_format.replace('_', '-')}",
        description=(
            f"{variant.description}。重みの読み込みを --load-format {load_format} にし、"
            "起動の時間を比べる (#46)"
        ),
        load_format=LoadFormat(load_format, _load_option_why(LOAD_FORMAT_FLAG, load_format)),
    )


def without_load_format(variant: Variant) -> Variant:
    """`--load-format auto` の構成 (既定の読み込み方を外し、vLLM の既定に任せる)。

    args に `--load-format` を書かず、名前と説明も変えない。#50 より前の出力とバイト単位で同じ。
    """
    return variant._replace(load_format=None)


def with_load_strategy(variant: Variant, strategy: str) -> Variant:
    """`--safetensors-load-strategy <値>` の構成 (基の変種 + safetensors の読み方)。

    許さない値は `ValueError` で断る。
    """
    if strategy not in LOAD_STRATEGY_PROVENANCE:
        raise ValueError(
            f"--safetensors-load-strategy に書けるのは {', '.join(LOAD_STRATEGY_PROVENANCE)} だけ"
        )
    return variant._replace(
        name=f"{variant.name}{LOAD_STRATEGY_SUFFIX}{strategy}",
        description=(
            f"{variant.description}。safetensors の読み込みを --safetensors-load-strategy "
            f"{strategy} にし、起動の時間を比べる (#46)"
        ),
        safetensors_load_strategy=strategy,
    )


def load_derived_manifest(path: Path) -> dict[str, Any]:
    """コミットした派生のマニフェストを読み、契約の形 (存在と型) を確かめる。

    形が違えば `ValueError`。読み分けは `kind` だけで行い、項目の有無からは推測しない。
    """
    if not path.is_file():
        raise ValueError(f"派生のマニフェストがない: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"派生のマニフェストを読めない: {path} ({exc})") from exc
    if not isinstance(data, dict) or data.get("kind") != "derived":
        raise ValueError(f'派生のマニフェストでない (kind = "derived" が要る): {path}')
    derivation = data.get("derivation")
    if not isinstance(derivation, dict):
        raise ValueError(f"派生のマニフェストに derivation がない: {path}")
    origin = derivation.get("origin")
    conversion = derivation.get("conversion")
    if (
        not isinstance(derivation.get("name"), str)
        or not isinstance(origin, dict)
        or not isinstance(conversion, dict)
    ):
        raise ValueError(f"派生のマニフェストの derivation の形が正しくない: {path}")
    args = conversion.get("args")
    if (
        not isinstance(origin.get("repo"), str)
        or not isinstance(origin.get("revision"), str)
        or not isinstance(conversion.get("tool"), str)
        or not isinstance(conversion.get("commit"), str)
        or not isinstance(conversion.get("target_pattern"), str)
        or not isinstance(args, list)
        or any(not isinstance(item, str) for item in args)
    ):
        raise ValueError(f"派生のマニフェストの項目の形が正しくない: {path}")
    return data


def derived_weights(
    name: str, data: Mapping[str, Any], p1_weights: Mapping[str, Any]
) -> DerivedWeights:
    """`--weights <名前>` の指定を、コミット済みの派生マニフェストから組む。

    名前が違う、元の参照が p1 の重みと違うときは `ValueError` (生成の前に断る)。
    """
    derivation: Mapping[str, Any] = data["derivation"]
    origin: Mapping[str, Any] = derivation["origin"]
    conversion: Mapping[str, Any] = derivation["conversion"]
    if derivation["name"] != name:
        raise ValueError(
            f"派生のマニフェストの名前 ({derivation['name']}) が、--weights の値 ({name}) と違う"
        )
    if origin["repo"] != p1_weights["repo"] or origin["revision"] != p1_weights["revision"]:
        raise ValueError(
            f"派生のマニフェストの元の重み ({origin['repo']}@{origin['revision']}) が、"
            f"p1 の重み ({p1_weights['repo']}@{p1_weights['revision']}) と違う"
        )
    return DerivedWeights(
        name=name,
        origin_repo=str(origin["repo"]),
        origin_revision=str(origin["revision"]),
        origin_manifest=str(p1_weights["manifest"]),
        manifest=f"{name}.manifest.json",
        tool=str(conversion["tool"]),
        commit=str(conversion["commit"]),
        args=tuple(str(item) for item in conversion["args"]),
        target_pattern=str(conversion["target_pattern"]),
    )


def with_weights(variant: Variant, derived: DerivedWeights) -> Variant:
    """`--weights <名前>` の構成 (基の変種 + 手元で変換した重み)。

    名前の接尾辞は、基の変種名の直後 (`p2-nope-tp2-full-k2s1`) に付ける。重みの違いは、
    道具のフラグ (`-tn` / `-prof` / `-lf-…`) より本質的な差なので先に置く。
    """
    return variant._replace(
        name=f"{variant.name}-{derived.name}",
        description=(
            f"{variant.description}。重みは、手元で変換した派生の重み {derived.name}"
            f" (元: {derived.origin_repo}@{derived.origin_revision}、"
            f"変換 {derived.tool}@{derived.commit[:8]})。実機では未確認"
        ),
        weights=derived,
    )


def load_overlay_definition(path: Path) -> dict[str, Any]:
    """重ねる定義 (`<名前>.json`) を読む。

    無い・読めない・JSON の object でなければ `ValueError`。
    """
    if not path.is_file():
        raise ValueError(f"重ねる定義がない: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"重ねる定義を読めない: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"重ねる定義が JSON の object でない: {path}")
    return data


def overlay_mount_key(path: str) -> str:
    """直したファイルの道筋から決まる、docker の設定の鍵。"""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", path.removeprefix("vllm/")).strip("-")
    return f"{OVERLAY_MOUNT_KEY_PREFIX}{slug}"


def overlay_mount_value(name: str, path: str) -> str:
    """直したファイル 1 つを重ねる、読み取り専用の bind mount の値 (`{remote_root}` は埋める前)。"""
    return (
        f"type=bind,source={{remote_root}}/{OVERLAY_PAYLOAD_SUBDIR}/{name}/{path},"
        f"target={IMAGE_SITE_PACKAGES}/{path},readonly"
    )


def _is_vllm_relative_path(path: str) -> bool:
    parts = path.split("/")
    return len(parts) >= 2 and parts[0] == "vllm" and ".." not in parts and "" not in parts


def vllm_overlay(name: str, data: Mapping[str, Any]) -> VllmOverlay:
    """`--vllm-overlay <名前>` の指定を、重ねる定義から組む。

    名前・commit が違う、道筋が `vllm/` で始まる相対の道筋でない、写しが無い、mount の鍵が
    重なるときは `ValueError` (生成の前に断る)。
    """
    if data.get("name") != name:
        raise ValueError(
            f"重ねる定義の名前 ({data.get('name')!r}) が、--vllm-overlay の値 ({name}) と違う"
        )
    if data.get("vllm_commit") != VLLM_COMMIT:
        raise ValueError(
            f"重ねる定義の vllm_commit ({data.get('vllm_commit')!r}) が、"
            f"イメージの vLLM の固定の commit ({VLLM_COMMIT}) と違う"
        )
    entries = data.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("重ねる定義の files が空でない配列でない")
    files: list[str] = []
    keys: dict[str, str] = {}
    for entry in entries:
        path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(path, str) or not _is_vllm_relative_path(path):
            raise ValueError(f"重ねる定義の道筋が vllm/ で始まる相対の道筋でない: {path!r}")
        copy = OVERLAY_PAYLOAD_DIR / name / path
        if not copy.is_file():
            raise ValueError(f"重ねる写しがない: {copy}")
        key = overlay_mount_key(path)
        if key in keys:
            raise ValueError(f"重ねる道筋 {keys[key]} と {path} の mount の鍵 ({key}) が重なる")
        keys[key] = path
        files.append(path)
    return VllmOverlay(name=name, files=tuple(files))


def with_vllm_overlay(variant: Variant, overlay: VllmOverlay) -> Variant:
    """`--vllm-overlay <名前>` の構成 (基の変種 + 固定した vLLM の直したファイルの重ね)。

    名前の接尾辞は、`--weights` の接尾辞の直後に付ける。使う vLLM の違いは、重みと同じく
    道具のフラグ (`-tn` / `-prof` / `-lf-…`) より本質的な差なので先に置く。
    """
    return variant._replace(
        name=f"{variant.name}{OVERLAY_SUFFIX}{overlay.name}",
        description=(
            f"{variant.description}。固定した vLLM の直したファイル"
            f" ({', '.join(overlay.files)}) を、"
            f"読み取り専用の bind mount で重ねる (vllm-overlay {overlay.name}。"
            "ADR 0007 の第 2a 段、#73)"
        ),
        vllm_overlay=overlay,
    )


def _load_option_why(flag: str, value: str) -> str:
    """読み込み方の引数 (`--load-format` / `--safetensors-load-strategy`) を足す理由。

    #37 の調査で分かった読み込みの律速を含む。
    """
    return (
        "#46 (#37 の調査)。起動の約 7 分を占める重みの読み込みは、ディスク (dd で 1.1〜11.8 GB/s) "
        f"ではなく vLLM の既定の mmap の読み方 (約 220 MB/s) が律速。{flag} {value} に"
        "替えて起動の時間を比べるときだけ付ける"
    )


def _spec_value(spec_tokens: int) -> str:
    """`--speculative-config` に渡す JSON の値 (vLLM の公式の文書とソースの形)。"""
    return f'{{"method":"mtp","num_speculative_tokens":{spec_tokens}}}'


def _mtp_why(spec_tokens: int) -> str:
    """MTP の設定の理由 (未修正の #58454 の条件を含む)。"""
    return (
        "ADR 0006 の K1。モデル付属の MTP (num_nextn_predict_layers = 1) を"
        f"下書き {spec_tokens} トークンで使う。n_predict = 1 なので、どの N も vLLM の"
        "割り切れる検査を通る。N >= 2 で文脈が 2048 を超えると kpool が壊れる報告"
        " (vLLM #58454、未修正) があるので、手順書の確認を必ず行う"
    )


def load_image(inspect_json: Path) -> dict[str, Any]:
    """inspect JSON を読み、確認済みのイメージ情報を検証する。"""
    image: Any = json.loads(inspect_json.read_text(encoding="utf-8"))
    if isinstance(image, list) and len(image) == 1:
        image = image[0]
    if not isinstance(image, dict):
        raise ValueError("inspect JSON の形式が不正")
    if image.get("Id") != EXPECTED_IMAGE_ID:
        raise ValueError("確認済みの完全なイメージ ID が必要")
    if image.get("Architecture") != "arm64" or image.get("Os") != "linux":
        raise ValueError("linux/arm64 のイメージが必要")
    size = image.get("Size")
    if type(size) is not int or size <= 0:
        raise ValueError("イメージの大きさが不正")
    return image


def _derived_weights_section(name: str, derived: DerivedWeights) -> str:
    """派生の重みの節 (`kind` / `name` / `manifest` / `mount_at` と、元の参照と変換の条件)。"""
    return (
        f"[configs.{name}.weights]\n"
        'kind = "derived"\n'
        f"name = {json.dumps(derived.name)}\n"
        f"manifest = {json.dumps(derived.manifest)}\n"
        f"mount_at = {json.dumps(f'/models/{derived.name}')}\n"
        "\n"
        f"[configs.{name}.weights.origin]\n"
        f"repo = {json.dumps(derived.origin_repo)}\n"
        f"revision = {json.dumps(derived.origin_revision)}\n"
        f"manifest = {json.dumps(derived.origin_manifest)}\n"
        "\n"
        f"[configs.{name}.weights.conversion]\n"
        f"tool = {json.dumps(derived.tool)}\n"
        f"commit = {json.dumps(derived.commit)}\n"
        f"args = {json.dumps(list(derived.args), ensure_ascii=False)}\n"
        f"target_pattern = {json.dumps(derived.target_pattern, ensure_ascii=False)}\n"
    )


def _replace_weights_section(block: str, name: str, derived: DerivedWeights) -> str:
    """p1 の weights の節を、派生の重みの節に置き換える。

    `[configs.p1-nvfp4-tp2.weights]` から、次の `# --- docker の設定` の直前までを差し替える
    (p1 の節のコメントも、いっしょに落とす)。
    """
    marker = "# --- docker の設定"
    start = block.index(f"[configs.{SOURCE_NAME}.weights]")
    end = block.index(marker, start)
    return block[:start] + _derived_weights_section(name, derived) + "\n" + block[end:]


def _replace_in_table(block: str, header: str, old_value: str, new_value: str, new_why: str) -> str:
    start = block.index(header)
    end = block.find("\n[", start + len(header))
    if end == -1:
        end = len(block)
    table = block[start:end]
    old_line = f'value = "{old_value}"'
    if old_line not in table:
        raise ValueError(f"{header} の value が見つからない")
    table = table.replace(old_line, f'value = "{new_value}"', 1)
    why_start = table.index("why = ")
    why_end = table.index("\n", why_start)
    table = table[:why_start] + f"why = {json.dumps(new_why, ensure_ascii=False)}" + table[why_end:]
    return block[:start] + table + block[end:]


def _insert_arg_table(
    block: str,
    variant: Variant,
    *,
    key: str,
    flag: str,
    value: str,
    why: str,
    source: str,
    quote: str,
) -> str:
    """根拠つきの args のテーブルを 1 つ、`args` の末尾 (env のコメントの直前) に足す。"""
    table = (
        f"[configs.{variant.name}.args.{key}]\n"
        f'flag = "{flag}"\n'
        f"value = '{value}'\n"
        f"why = {json.dumps(why, ensure_ascii=False)}\n"
        f"source = {json.dumps(source)}\n"
        f"quote = {json.dumps(quote, ensure_ascii=False)}\n"
    )
    return block.replace(variant.env_comment, table + "\n" + variant.env_comment, 1)


def _insert_docker_table(
    block: str,
    variant: Variant,
    *,
    key: str,
    flag: str,
    value: str,
    why: str,
    source: str,
    quote: str,
) -> str:
    """根拠つきの docker のテーブルを 1 つ、docker の節の末尾 (args のコメントの直前) に足す。"""
    table = (
        f"[configs.{variant.name}.docker.{key}]\n"
        f"flag = {json.dumps(flag)}\n"
        f"value = {json.dumps(value)}\n"
        f"why = {json.dumps(why, ensure_ascii=False)}\n"
        f"source = {json.dumps(source)}\n"
        f"quote = {json.dumps(quote, ensure_ascii=False)}\n"
    )
    at = block.index(ARGS_COMMENT)
    return block[:at] + table + "\n" + block[at:]


def render(configs_text: str, image: Mapping[str, Any], variant: Variant = SMOKE) -> str:
    """configs.toml の p1 構成から `variant` に応じた構成の TOML を返す。"""
    start = configs_text.index(f"[configs.{SOURCE_NAME}]")
    end = configs_text.index("# ====", start)
    block = configs_text[start:end].rstrip() + "\n"

    image_start = block.index(f"[configs.{SOURCE_NAME}.image]")
    image_end = block.index(f"[configs.{SOURCE_NAME}.weights]")
    image_section = (
        f"[configs.{SOURCE_NAME}.image]\n"
        f"ref = {json.dumps(EXPECTED_IMAGE_ID)}\n"
        f"seen_as = {json.dumps(IMAGE_SEEN_AS)}\n"
        f"size_bytes = {image['Size']}\n"
        f"measured = {json.dumps(IMAGE_MEASURED)}\n\n"
    )
    block = block[:image_start] + image_section + block[image_end:]
    for key, old_value, new_value, why in variant.arg_overrides:
        block = _replace_in_table(
            block,
            f"[configs.{SOURCE_NAME}.args.{key}]",
            old_value,
            new_value,
            why,
        )
    if variant.weights is not None:
        derived = variant.weights
        source_config = tomllib.loads(configs_text)["configs"][SOURCE_NAME]
        block = _replace_weights_section(block, variant.name, derived)
        block = _replace_in_table(
            block,
            f"[configs.{SOURCE_NAME}.docker.mount-weights]",
            source_config["docker"]["mount-weights"]["value"],
            "type=bind,source={remote_root}/models/"
            f"{derived.name},target=/models/{derived.name},readonly",
            DERIVED_MOUNT_WHY,
        )
    block = block.replace(f"configs.{SOURCE_NAME}", f"configs.{variant.name}")
    block = re.sub(
        r"^description = .*$",
        f"description = {json.dumps(variant.description, ensure_ascii=False)}",
        block,
        count=1,
        flags=re.MULTILINE,
    )
    block = block.replace(
        "# --- 環境変数 (最初の 3 つだけ。A/B で足すものは docs/results/ の実測を根拠にする) -----",
        variant.env_comment,
    )
    if variant.vllm_overlay is not None:
        overlay = variant.vllm_overlay
        for path in overlay.files:
            block = _insert_docker_table(
                block,
                variant,
                key=overlay_mount_key(path),
                flag="--mount",
                value=overlay_mount_value(overlay.name, path),
                why=OVERLAY_MOUNT_WHY,
                source=OVERLAY_MOUNT_SOURCE,
                quote=OVERLAY_MOUNT_QUOTE,
            )
    if variant.spec_tokens is not None:
        # (a) 投機を許す項目は、最上位テーブルの `served_model_name` の直後にだけ足す
        block = re.sub(
            r"^served_model_name = .*$",
            lambda match: match.group(0) + "\nallow_speculative = true",
            block,
            count=1,
            flags=re.MULTILINE,
        )
        block = _insert_arg_table(
            block,
            variant,
            key="speculative-config",
            flag="--speculative-config",
            value=_spec_value(variant.spec_tokens),
            why=_mtp_why(variant.spec_tokens),
            source=MTP_SOURCE,
            quote=MTP_QUOTE,
        )
    if variant.torch_profiler:
        block = _insert_arg_table(
            block,
            variant,
            key=PROFILER_ARG_KEY,
            flag=PROFILER_FLAG,
            value=PROFILER_VALUE,
            why=PROFILER_WHY,
            source=PROFILER_SOURCE,
            quote=PROFILER_QUOTE,
        )
    if variant.load_format is not None:
        load_format = variant.load_format
        source, quote = LOAD_FORMAT_PROVENANCE[load_format.value]
        block = _insert_arg_table(
            block,
            variant,
            key=LOAD_FORMAT_ARG_KEY,
            flag=LOAD_FORMAT_FLAG,
            value=load_format.value,
            why=load_format.why,
            source=source,
            quote=quote,
        )
    if variant.safetensors_load_strategy is not None:
        source, quote = LOAD_STRATEGY_PROVENANCE[variant.safetensors_load_strategy]
        block = _insert_arg_table(
            block,
            variant,
            key=LOAD_STRATEGY_ARG_KEY,
            flag=LOAD_STRATEGY_FLAG,
            value=variant.safetensors_load_strategy,
            why=_load_option_why(LOAD_STRATEGY_FLAG, variant.safetensors_load_strategy),
            source=source,
            quote=quote,
        )
    data = tomllib.loads(configs_text)
    source_env = data["configs"][NCCL_SOURCE_NAME]["env"]
    additions = (
        (
            "nccl-debug",
            "NCCL_DEBUG",
            "INFO",
            variant.nccl_debug_why,
        ),
        (
            "nccl-debug-subsys",
            "NCCL_DEBUG_SUBSYS",
            "INIT,NET",
            (
                "この起動で読みたいのは、経路の確定とデバイスの初期化だけなので、"
                "netcheck-bandwidth の INIT,BOOTSTRAP,ENV,NET,GRAPH より狭い INIT,NET にする"
            ),
        ),
        (
            "nccl-debug-file",
            "NCCL_DEBUG_FILE",
            "/logs/nccl.%h.%p.log",
            (
                "2 台ぶんの記録を、mount-logs で結び付けた /logs に書き、serve logs で回収する。"
                "標準出力の起動の記録と混ぜない"
            ),
        ),
    )
    for key, flag, value, why in additions:
        origin = source_env[key]
        block += (
            f"\n[configs.{variant.name}.env.{key}]\n"
            f"flag = {json.dumps(flag)}\n"
            f"value = {json.dumps(value)}\n"
            f"why = {json.dumps(why, ensure_ascii=False)}\n"
            f"source = {json.dumps(origin['source'])}\n"
            f"quote = {json.dumps(origin['quote'], ensure_ascii=False)}\n"
        )
    if variant.nccl_thread_names:
        block += (
            "\n# --- スレッドの名前 (--nccl-thread-names のときだけ) -----\n"
            f"\n[configs.{variant.name}.env.{THREAD_NAMES_ENV_KEY}]\n"
            'flag = "NCCL_SET_THREAD_NAME"\n'
            'value = "1"\n'
            f"why = {json.dumps(THREAD_NAMES_WHY, ensure_ascii=False)}\n"
            f"source = {json.dumps(THREAD_NAMES_SOURCE)}\n"
            f"quote = {json.dumps(THREAD_NAMES_QUOTE)}\n"
        )
    result = "schema_version = 1\n\n" + block
    tomllib.loads(result)
    return result


def generate(inspect_json: Path, output: Path, variant: Variant = SMOKE) -> None:
    """inspect JSON から構成を生成し、既存の出力を保護して保存する。"""
    result = render(CONFIGS_PATH.read_text(encoding="utf-8"), load_image(inspect_json), variant)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(result)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=(*VARIANTS, FULL_MTP_VARIANT_NAME),
        default="smoke",
        help=(
            "生成する構成 (smoke: 初回確認用 4096/1、full: 計測用 163840/16、"
            "full-mtp: full と同じ構成 + MTP (`--spec-tokens N`))。"
            "full と full-mtp は --load-format instanttensor が既定 (#50)"
        ),
    )
    parser.add_argument(
        "--spec-tokens",
        type=int,
        default=None,
        help="full-mtp の、モデル付属の MTP の下書きのトークン数 (1 以上)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "使う、手元で変換した重みの名前 (serving/weights/<名前>.manifest.json を読む)。"
            "構成の名前に -<名前> を付け、重みの結び付けを {remote_root}/models/<名前> に"
            "差し替える (読み取り専用)"
        ),
    )
    parser.add_argument(
        "--vllm-overlay",
        default=None,
        help=(
            "重ねる、固定した vLLM の直したファイルの定義の名前"
            " (experiments/k2-vllm-overlay/<名前>.json を読む)。ファイルごとに、"
            "{remote_root}/payload/vllm-overlay/<名前>/<道筋> を、イメージの中の vLLM の"
            "同じファイルに"
            "読み取り専用で重ねる mount を足し、構成の名前に -ov-<名前> を付ける"
        ),
    )
    parser.add_argument(
        "--nccl-thread-names",
        action="store_true",
        help=(
            "NCCL のスレッドに名前を付ける env を足し、構成の名前に -tn を付ける"
            " (どの --variant にも付けられる)"
        ),
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help=(
            "torch プロファイラーの --profiler-config を足し、構成の名前に -prof を付ける"
            " (どの --variant にも付けられる)"
        ),
    )
    parser.add_argument(
        "--load-format",
        choices=(*LOAD_FORMAT_PROVENANCE, LOAD_FORMAT_OPT_OUT),
        default=None,
        help=(
            "重みの読み込み方 (--load-format)。full と full-mtp は instanttensor が既定 (#50)。"
            " 3 値を明示すると既定を置き換え、構成の名前に -lf-<値> を付ける"
            " (runai_streamer は -lf-runai-streamer)。auto は既定を外し、args に書かない"
            " (smoke では何も変わらない)"
        ),
    )
    parser.add_argument(
        "--safetensors-load-strategy",
        choices=tuple(LOAD_STRATEGY_PROVENANCE),
        default=None,
        help=(
            "safetensors の読み方 (--safetensors-load-strategy) を足し、構成の名前に -sls-<値> を"
            "付ける。効くのは --load-format を付けない読み方だけなので、full と full-mtp では"
            " --load-format auto と組で使う (付けないと断る。smoke は単独で付けられる)"
        ),
    )
    parser.add_argument("inspect_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    if args.variant == FULL_MTP_VARIANT_NAME:
        if args.spec_tokens is None:
            parser.error("--variant full-mtp には --spec-tokens が要る")
        try:
            variant = mtp_variant(args.spec_tokens)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if args.spec_tokens is not None:
            parser.error("--spec-tokens は --variant full-mtp のときだけ使える")
        variant = VARIANTS[args.variant]
    if args.weights is not None:
        manifest_path = WEIGHTS_DIR / f"{args.weights}.manifest.json"
        try:
            data = load_derived_manifest(manifest_path)
            p1_weights = tomllib.loads(CONFIGS_PATH.read_text(encoding="utf-8"))["configs"][
                SOURCE_NAME
            ]["weights"]
            derived = derived_weights(args.weights, data, p1_weights)
        except ValueError as exc:
            parser.error(str(exc))
        variant = with_weights(variant, derived)
    if args.vllm_overlay is not None:
        try:
            overlay = vllm_overlay(
                args.vllm_overlay,
                load_overlay_definition(OVERLAY_DIR / f"{args.vllm_overlay}.json"),
            )
        except ValueError as exc:
            parser.error(str(exc))
        variant = with_vllm_overlay(variant, overlay)
    if args.nccl_thread_names:
        variant = with_thread_names(variant)
    if args.torch_profiler:
        variant = with_torch_profiler(variant)
    if args.load_format == LOAD_FORMAT_OPT_OUT:
        variant = without_load_format(variant)
    elif args.load_format is not None:
        variant = with_load_format(variant, args.load_format)
    if args.safetensors_load_strategy is not None:
        # strategy が効くのは --load-format を付けない読み方だけ (手順書 §1)。既定の
        # instanttensor を黙って外さず、strategy が効かない構成も作らないよう、生成の前に断る。
        if args.load_format is None and variant.load_format is not None:
            parser.error(
                "--safetensors-load-strategy が効くのは --load-format を付けない読み方だけ。"
                f"--variant {args.variant} は既定で --load-format {DEFAULT_LOAD_FORMAT_VALUE} を"
                f"持つので、--load-format {LOAD_FORMAT_OPT_OUT} と組で使う"
            )
        variant = with_load_strategy(variant, args.safetensors_load_strategy)
    generate(args.inspect_json, args.output, variant)


if __name__ == "__main__":
    main()
