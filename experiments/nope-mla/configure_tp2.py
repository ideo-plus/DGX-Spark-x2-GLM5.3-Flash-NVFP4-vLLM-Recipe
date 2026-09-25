"""確認済みの自前イメージから TP=2 の構成を生成する。

変種は 3 つ: 初回確認用の `smoke`、計測用の `full`、`full` にモデル付属の MTP を足した
`full-mtp` (`--spec-tokens N`)。`full-mtp` の構成の名前は `p2-nope-tp2-mtp<N>` になる。

`--nccl-thread-names` を付けると、`NCCL_SET_THREAD_NAME=1` の env を根拠つきで足し、構成の
名前を `-tn` 付きにする (どの `--variant` にも付けられる)。

`--torch-profiler` を付けると、`--profiler-config` (torch プロファイラー、出力先
`/logs/torch-profile`) を根拠つきで args に足し、構成の名前を `-prof` 付きにする
(どの `--variant` にも付けられる)。

`--load-format` (`instanttensor` / `fastsafetensors` / `runai_streamer`) と
`--safetensors-load-strategy` (`eager` / `prefetch`) を付けると、重みの読み込み方を根拠つきで
args に足し、構成の名前を `-lf-<値>` / `-sls-<値>` 付きにする。`runai_streamer` は `_` を `-`
に替えて `-lf-runai-streamer` になる。どちらもどの `--variant` にも付けられる (付けないときの
出力は変わらない)。
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
EXPECTED_IMAGE_ID = "sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90"
IMAGE_SEEN_AS = "vllm-nope:0961bbae-fi070"
IMAGE_MEASURED = "docs/results/2026-09-22-nope-build.md"
SOURCE_NAME = "p1-nvfp4-tp2"
TARGET_NAME = "p2-nope-tp2-smoke"
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
PROFILER_VALUE = '{"profiler":"torch","torch_profiler_dir":"/logs/torch-profile"}'
PROFILER_SOURCE = (
    "https://github.com/vllm-project/vllm/blob/"
    "0961bbae2894d574be790d219651824eb199318e/vllm/config/profiler.py"
)
PROFILER_QUOTE = (
    "Which profiler to use. … Directory to save torch profiler traces. "
    "Both AsyncLLM's CPU traces and worker's traces (CPU & GPU) will be saved under this "
    "directory. Note that it must be an absolute path."
)
PROFILER_WHY = (
    "#42 (ADR 0006 の K2 の判断)。1 ステップの GPU の時間の内訳を torch プロファイラーで測る。"
    "記録は /start_profile と /stop_profile の間だけ。trace は mount-logs で結び付けた /logs の"
    "下に書き、serve logs で回収する。この計測のときだけ付ける"
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

    `load_format` / `safetensors_load_strategy` は、重みの読み込み方の値 (`None` なら足さない)。
    `with_load_format` / `with_load_strategy` が作る。名前は `-lf-<値>` / `-sls-<値>` 付きになる。
    """

    name: str
    description: str
    env_comment: str
    nccl_debug_why: str
    arg_overrides: tuple[tuple[str, str, str, str], ...]
    spec_tokens: int | None = None
    nccl_thread_names: bool = False
    torch_profiler: bool = False
    load_format: str | None = None
    safetensors_load_strategy: str | None = None


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
        load_format=load_format,
    )


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
        source, quote = LOAD_FORMAT_PROVENANCE[variant.load_format]
        block = _insert_arg_table(
            block,
            variant,
            key=LOAD_FORMAT_ARG_KEY,
            flag=LOAD_FORMAT_FLAG,
            value=variant.load_format,
            why=_load_option_why(LOAD_FORMAT_FLAG, variant.load_format),
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
            "full-mtp: full と同じ構成 + MTP (`--spec-tokens N`))"
        ),
    )
    parser.add_argument(
        "--spec-tokens",
        type=int,
        default=None,
        help="full-mtp の、モデル付属の MTP の下書きのトークン数 (1 以上)",
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
        choices=tuple(LOAD_FORMAT_PROVENANCE),
        default=None,
        help=(
            "重みの読み込み方 (--load-format) を足し、構成の名前に -lf-<値> を付ける"
            " (どの --variant にも付けられる。runai_streamer は -lf-runai-streamer)"
        ),
    )
    parser.add_argument(
        "--safetensors-load-strategy",
        choices=tuple(LOAD_STRATEGY_PROVENANCE),
        default=None,
        help=(
            "safetensors の読み方 (--safetensors-load-strategy) を足し、構成の名前に -sls-<値> を"
            "付ける (どの --variant にも付けられる)"
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
    if args.nccl_thread_names:
        variant = with_thread_names(variant)
    if args.torch_profiler:
        variant = with_torch_profiler(variant)
    if args.load_format is not None:
        variant = with_load_format(variant, args.load_format)
    if args.safetensors_load_strategy is not None:
        variant = with_load_strategy(variant, args.safetensors_load_strategy)
    generate(args.inspect_json, args.output, variant)


if __name__ == "__main__":
    main()
