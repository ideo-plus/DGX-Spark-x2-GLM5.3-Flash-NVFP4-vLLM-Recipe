"""自前イメージの TP=2 の構成の生成器 (`experiments/nope-mla/configure_tp2.py`) の試験。

確かめること:

- inspect の JSON の `Id` が、2026-09-22 のビルドの記録 (`docs/results/2026-09-22-nope-build.md`)
  が記した完全なイメージ ID と一致し、`Os == "linux"` かつ `Architecture == "arm64"` で、
  `Size` が `bool` でない正の整数であることを要求する。違えば `ValueError` で、出力を作らない
- 出力先が既存なら上書きしない (既存の内容は 1 バイトも変わらない)
- 生成した TOML は、実物の `serving/config/configs.toml` を読み取るだけで作り、
  実物の `load_configs` / `load_nodes` / `select_config` / `build_plans` で読める
- 基の `p1-nvfp4-tp2` との差は、説明、イメージの節、`--max-model-len` と `--max-num-seqs`
  の値と理由、初回の NCCL の記録の 3 変数 (`env` への追加) だけである
- `full` と `full-mtp` の args には、既定で `--load-format instanttensor` が根拠つきで入り、
  構成の名前は変わらない (#50)。`smoke` には入らない
- `--load-format auto` を付けると既定を外し、出力は変更前とバイト単位で同じになる。
  外した出力の SHA-256 は固定値で、既定つきの出力はその固定値から導出できる
- 生成の前後で、実物の `configs.toml` のバイト列が同じである (生成器は読むだけで書かない)
- 生成器は、ネットワークも `subprocess` も読み込まない

試験は、`tmp_path` に inspect の JSON の見本を作り、実物の `configs.toml` と `nodes.toml` を
読み取りで使う。`serving/var/` (`.gitignore` 対象) には依存せず、実機・ネットワークには
一切つながない。生成器の module は、パッケージでない `experiments/nope-mla/` の下にあるので
`importlib` で読む。
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

import pytest

from serving_kit.config import load_configs, load_nodes, select_config
from serving_kit.plan import LABEL_CONFIG, LABEL_CONFIG_SHA256, build_plans
from serving_kit.types import ContainerPlan

SERVING_DIR = Path(__file__).resolve().parents[2]
"""`serving/` (この試験から見て 2 つ上)。"""

REPO_ROOT = SERVING_DIR.parent
CONFIGS_PATH = SERVING_DIR / "config" / "configs.toml"
NODES_PATH = SERVING_DIR / "config" / "nodes.toml"
GENERATOR_PATH = REPO_ROOT / "experiments" / "nope-mla" / "configure_tp2.py"
STARTED_AT = datetime(2026, 9, 23, 0, 0, 0, tzinfo=UTC)

EXPECTED_IMAGE_ID = "sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90"
IMAGE_TAG = "vllm-nope:0961bbae-fi070"
IMAGE_SIZE = 23438274807
P1_IMAGE_REF = (
    "vllm/vllm-openai@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
)
NCCL_ENV = (
    ("-e", "NCCL_DEBUG=INFO"),
    ("-e", "NCCL_DEBUG_SUBSYS=INIT,NET"),
    ("-e", "NCCL_DEBUG_FILE=/logs/nccl.%h.%p.log"),
)
TARGET_NAME = "p2-nope-tp2-smoke"
FULL_TARGET_NAME = "p2-nope-tp2-full"
WEIGHTS_LABEL = "RedHatAI/GLM-5.3-Flash-NVFP4@18d55bfd5a2194887738da73753975c9d3842f46"

# 引数なし (`--variant` 省略) の `render` の出力の SHA-256。計算のしかた:
# 変更前の生成器 (full の追加の前) で、`render(configs.toml の本文, load_image(image-inspect.json))`
# のバイト列 (UTF-8) に hashlib.sha256 を掛けたもの。既定の出力が替わっていないことの固定に使う
SMOKE_RENDER_SHA256 = "d9d5313864d9b8ae5d90f441cc100f2a9af18a9924f93178ac1c8dde3a80539d"

# `--variant full` の既定 (`--load-format instanttensor`。#50) の `render` の出力の SHA-256。
# 計算のしかた: まず既定を外した形 (opt-out) の出力を作る。これは #50 より前の生成器で
# `render(本文, image, FULL)` のバイト列 (UTF-8) に hashlib.sha256 を掛けたものと一致する。
# その出力の `FULL.env_comment` の直前に、`default_load_format_table(FULL)` が組む
# `[configs.p2-nope-tp2-full.args.load-format]` のテーブル 1 つ (value instanttensor、why は
# #50・#37 の実測、source/quote は #46 で入れた vLLM の `LoadConfig` の docstring) を
# `str.replace(env_comment, table + "\n" + env_comment, 1)` で挿入したバイト列に掛けたもの。
# 根拠は #50 と #37 の実測 (重みの読み込み 363 秒 (head) → 32 秒、起動 約 11 分 → 約 3 分)。
FULL_RENDER_SHA256 = "6db1815b7d64cdb2876686f34441dc9721b4119392ed2c50440d19b1169e092a"

# `--load-format auto` (`without_load_format(FULL)`) の出力の SHA-256。計算のしかた:
# #50 より前の生成器で `render(本文, image, FULL)` のバイト列 (UTF-8) に hashlib.sha256 を
# 掛けたもの。既定を外したときの出力が、変更前とバイト単位で同じであることの固定に使う (C2)。
FULL_AUTO_RENDER_SHA256 = "194cdd43aa1c5e5d2c92b66cb9cc0466d5aeef870e7eb0e3a1e74a2faf6c4f8a"

MTP_SPEC_TOKENS: tuple[int, ...] = (1, 2, 3, 5)
"""MTP の下書きトークン数 (Issue「すること 4」、C7)。"""

THREAD_NAME_ENV_KEY = "nccl-set-thread-name"
"""`--nccl-thread-names` で足す env テーブルの鍵 (C1)。"""

THREAD_NAME_ENV: tuple[str, str] = ("-e", "NCCL_SET_THREAD_NAME=1")
"""`--nccl-thread-names` で足す `-e` の組 (C1)。"""

NCCL_ENV_DOC = "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html"
"""NCCL の公式文書の環境変数の説明。`NCCL_SET_THREAD_NAME` の根拠に使う (C2)。"""

NCCL_QUOTE_FRAGMENT = "NCCL CPU threads"
"""公式文書の原文 (`NCCL_SET_THREAD_NAME` の説明) の一部 (C2)。"""

THREAD_NAME_BASE_KINDS: tuple[str, ...] = ("smoke", "full", "mtp1", "mtp2")
"""`--nccl-thread-names` を付けられる基の変種 (C1、C3)。"""

# `--variant full-mtp --spec-tokens N` の既定 (`--load-format instanttensor`) の `render` の
# SHA-256。計算のしかたは FULL_RENDER_SHA256 と同じで、opt-out の
# `render(本文, image, without_load_format(mtp_variant(N)))` の `mtp_variant(N).env_comment` の
# 直前に `default_load_format_table(mtp_variant(N))` を挿入したバイト列 (UTF-8) に
# hashlib.sha256 を掛けたもの。根拠は #50 と #37 の実測である (C3)。
MTP1_RENDER_SHA256 = "98de07c744a7e74059507348b010508e1daec62befcf855309a94241b6e9dad0"
MTP2_RENDER_SHA256 = "fc4e88ee0844bb1d3dd02d625c5fa2cec62e9cdb7257aeb6ede96abf5fdee4d6"

# `--load-format auto` (`without_load_format(mtp_variant(N))`) の出力の SHA-256。計算のしかたは
# SMOKE と同じで、生成器に thread-name を足す前の `render(本文, image, mtp_variant(N))` の
# バイト列 (UTF-8) に hashlib.sha256 を掛けたもの。既定を外した MTP の出力が、変更前と
# バイト単位で同じであることの固定に使う (C2)。
MTP1_AUTO_RENDER_SHA256 = "645a4f9596ab1d53b7fea821807b641d9508525b6a9a4c0bea24430ff88bf64c"
MTP2_AUTO_RENDER_SHA256 = "6ab08944640e3997398204da40d47f0f2ae608035ec987eda52b43eeff6a27a3"

PROFILER_SUFFIX = "-prof"
"""`--torch-profiler` で付ける、構成の名前の接尾辞 (C1、C3)。"""

PROFILER_ARG_KEY = "profiler-config"
"""`--profiler-config` の args テーブルの鍵 (C1)。"""

PROFILER_ARG_FLAG = "--profiler-config"
"""足す args のフラグ (C1)。"""

PROFILER_DIR = "/logs/torch-profile"
"""torch プロファイラーの出力先 (コンテナの `/logs/` の下。C1)。"""

# `--profiler-config` の値 (#48)。`torch_profiler_with_stack=false` と
# `torch_profiler_dump_cuda_time_total=false` を足し、trace の書き出しのメモリを減らす。根拠は
# vLLM の `0961bbae` の `vllm/config/profiler.py` の docstring (`torch_profiler_with_stack` は
# 既定 True で "stack tracing" を有効にし、`torch_profiler_dump_cuda_time_total` は既定 True で
# "dumps total CUDA time in torch profiler traces")。`wrapper.py` の `_stop` は
# `torch_profiler_dump_cuda_time_total` を写した `dump_device_time_total` が真のときだけ
# `key_averages()` の表 (`profiler_out_<rank>.txt`) を書く。2026-09-26 の初回計測では、head の
# `VLLM::Worker_TP` がこの書き出しで oom-killer に止められた。JSON なので `false` は小文字。
PROFILER_VALUE = (
    '{"profiler":"torch","torch_profiler_dir":"/logs/torch-profile",'
    '"torch_profiler_with_stack":false,"torch_profiler_dump_cuda_time_total":false}'
)
"""足す args の値 (#48)。"""

PROFILER_ARG_PAIR: tuple[str, str] = (PROFILER_ARG_FLAG, PROFILER_VALUE)
"""足す `--profiler-config` の 2 語 (C1、C3)。"""

PROFILER_SOURCE = (
    "https://github.com/vllm-project/vllm/blob/"
    "0961bbae2894d574be790d219651824eb199318e/vllm/config/profiler.py"
)
"""根拠の出典 (vLLM のソースの docstring。C2)。"""

PROFILER_QUOTE_FRAGMENTS: tuple[str, ...] = (
    "torch profiler traces",
    "absolute path",
    "stack tracing",
    "dumps total CUDA time",
)
"""根拠の原文に含まれる語 (C2、#48)。あとの 2 つは 2 つの `false` の根拠。"""

# `--variant full --nccl-thread-names` の既定 (`--load-format instanttensor`) の `render` の
# SHA-256。計算のしかたは FULL_RENDER_SHA256 と同じで、opt-out の
# `render(本文, image, without_load_format(with_thread_names(FULL)))` の
# `with_thread_names(FULL).env_comment` の直前に
# `default_load_format_table(with_thread_names(FULL))` を挿入したバイト列 (UTF-8) に
# hashlib.sha256 を掛けたもの (C3)。
FULL_TN_RENDER_SHA256 = "ce9fc27cbeaaaa5ddc0459f81b70f01983d8ec2cd51adaeddd3a39df0b08b48c"

# `--load-format auto` (`without_load_format(with_thread_names(FULL))`) の出力の SHA-256。
# 計算のしかたは SMOKE と同じで、生成器に `--torch-profiler` を足す前の
# `render(本文, image, with_thread_names(FULL))` のバイト列 (UTF-8) に hashlib.sha256 を
# 掛けたもの。既定を外したときの出力が、変更前とバイト単位で同じであることの固定に使う (C2)。
FULL_TN_AUTO_RENDER_SHA256 = "fcec31120db71411ec0525377658e42efa62268e74112aa895dc102c1df6b0d3"

PROFILER_BASE_KINDS: tuple[str, ...] = THREAD_NAME_BASE_KINDS
"""`--torch-profiler` を付けられる基の変種 (C1、C3)。"""

LOAD_FORMAT_SUFFIX = "-lf-"
"""`--load-format` で付ける、構成の名前の接尾辞 (C1)。"""

LOAD_FORMAT_ARG_KEY = "load-format"
"""`--load-format` の args テーブルの鍵 (C1)。"""

LOAD_FORMAT_ARG_FLAG = "--load-format"
"""足す args のフラグ (C1)。"""

LOAD_FORMAT_VALUES: tuple[str, ...] = ("instanttensor", "fastsafetensors", "runai_streamer")
"""`--load-format` に許す値 (C1、C3)。"""

LOAD_STRATEGY_SUFFIX = "-sls-"
"""`--safetensors-load-strategy` で付ける、構成の名前の接尾辞 (C2)。"""

LOAD_STRATEGY_ARG_KEY = "safetensors-load-strategy"
"""`--safetensors-load-strategy` の args テーブルの鍵 (C2)。"""

LOAD_STRATEGY_ARG_FLAG = "--safetensors-load-strategy"
"""足す args のフラグ (C2)。"""

LOAD_STRATEGY_VALUES: tuple[str, ...] = ("eager", "prefetch")
"""`--safetensors-load-strategy` に許す値 (C2、C3)。"""

LOAD_CONFIG_SOURCE = (
    "https://github.com/vllm-project/vllm/blob/"
    "0961bbae2894d574be790d219651824eb199318e/vllm/config/load.py"
)
"""`load_format` と `safetensors_load_strategy` の根拠の出典 (C3)。"""

FASTSAFETENSORS_DOC_SOURCE = (
    "https://github.com/vllm-project/vllm/blob/"
    "0961bbae2894d574be790d219651824eb199318e/docs/models/extensions/fastsafetensor.md"
)
"""`fastsafetensors` だけの根拠の出典 (docstring に無いため。C3)。"""

LOAD_FORMAT_QUOTE_FRAGMENTS: Mapping[str, str] = {
    "instanttensor": "InstantTensor",
    "fastsafetensors": "GPU direct storage",
    "runai_streamer": "Run:ai Model Streamer",
}
"""`--load-format` の根拠の原文に含まれる語 (C3)。"""

LOAD_STRATEGY_QUOTE_FRAGMENTS: Mapping[str, str] = {
    "eager": "CPU memory upfront",
    "prefetch": "OS page cache",
}
"""`--safetensors-load-strategy` の根拠の原文に含まれる語 (C3)。"""

LOAD_FORMAT_OPT_OUT = "auto"
"""既定 (`full` / `full-mtp` の `instanttensor`) を外す `--load-format auto` (C2)。"""

LOAD_FORMAT_OPT_OUT_SUFFIX = "-lf-auto"
"""`--load-format auto` が名前や本文に現れてはならない接尾辞 (C2)。"""

DEFAULT_LOAD_FORMAT_VALUE = "instanttensor"
"""`full` / `full-mtp` の既定の `--load-format` の値 (#50。C1)。"""

DEFAULT_LOAD_FORMAT_SOURCE = LOAD_CONFIG_SOURCE
"""既定の `--load-format` の根拠の出典 (#46 で入れた vLLM の `LoadConfig` の docstring。C1)。"""

DEFAULT_LOAD_FORMAT_QUOTE = (
    '"instanttensor" will load the Safetensors weights on CUDA devices using InstantTensor, '
    "which enables distributed loading with pipelined prefetching and fast direct I/O."
)
"""既定の `--load-format` の根拠の原文 (#46 で入れた vLLM の `LoadConfig` の docstring。C1)。"""

DEFAULT_LOAD_FORMAT_WHY = (
    "#50 (#37 の実測、2026-09-26)。--load-format instanttensor で、重みの読み込みが "
    "363 秒 (head) から 32 秒に、起動全体が約 11 分から約 3 分になった。同じ構成の probe で、"
    "decode の tok/s (code/en 14.06、code/ja 13.99、prose/en 14.07、prose/ja 13.88) は"
    "既定の読み方 (14.13、14.01、14.12、13.98) と同じ範囲、toolcall は 5/5 で正解・不正解が"
    "投機なしの回と同じ、needle 8k は正解、壊れの印は 0 件。計測のたびに起動するので、"
    "full と full-mtp の既定にする。外すときは --load-format auto (出力は #50 より前の既定と同じ)"
)
"""既定の `--load-format` を args に足す理由 (#50 と #37 の実測。C1、C3)。"""


def load_format_suffix(value: str) -> str:
    """`--load-format <値>` の名前の接尾辞 (`_` はコンテナ名に使えないので `-` に替える。C1)。"""
    return f"{LOAD_FORMAT_SUFFIX}{value.replace('_', '-')}"


def load_strategy_suffix(value: str) -> str:
    """`--safetensors-load-strategy <値>` の名前の接尾辞 (C2)。"""
    return f"{LOAD_STRATEGY_SUFFIX}{value}"


def default_load_format_table(variant: Any) -> str:
    """既定の `--load-format instanttensor` の args テーブルを組む (C3 の導出。実装に依らない)。

    書式は生成器の `_insert_arg_table` と同じである。why は #50 と #37 の実測、source と
    quote は #46 で入れた vLLM の `LoadConfig` の docstring を使う。
    """
    return (
        f"[configs.{variant.name}.args.{LOAD_FORMAT_ARG_KEY}]\n"
        f'flag = "{LOAD_FORMAT_ARG_FLAG}"\n'
        f"value = '{DEFAULT_LOAD_FORMAT_VALUE}'\n"
        f"why = {json.dumps(DEFAULT_LOAD_FORMAT_WHY, ensure_ascii=False)}\n"
        f"source = {json.dumps(DEFAULT_LOAD_FORMAT_SOURCE)}\n"
        f"quote = {json.dumps(DEFAULT_LOAD_FORMAT_QUOTE, ensure_ascii=False)}\n"
    )


def derive_default_load_format(generator: Any, text: str, image: Any, variant: Any) -> str:
    """既定つきの出力を、opt-out の出力 + テーブル 1 つから導出する (C3 の導出)。

    `variant` は既定を持つ変種 (`FULL` / `mtp_variant(N)` / その `with_thread_names` など)。
    `without_load_format(variant)` の出力の `variant.env_comment` の直前に、`variant` の
    テーブルを 1 つ挿入する。文字列の置換で作るので、実装の内部には依らない。
    """
    opt_out = generator.render(text, image, generator.without_load_format(variant))
    derived = opt_out.replace(
        variant.env_comment, default_load_format_table(variant) + "\n" + variant.env_comment, 1
    )
    return str(derived)


def mtp_target_name(n: int) -> str:
    """MTP の構成の名前 (C5、C7 で同じ名前を使う)。"""
    return f"p2-nope-tp2-mtp{n}"


def mtp_spec_value(n: int) -> str:
    """`--speculative-config` の JSON の値 (C5)。"""
    return f'{{"method":"mtp","num_speculative_tokens":{n}}}'


def _base_variant(generator: Any, kind: str) -> Any:
    """thread-name を付ける前の、基の変種 (`smoke` / `full` / `mtp1` / `mtp2`)。"""
    if kind == "smoke":
        return generator.SMOKE
    if kind == "full":
        return generator.FULL
    return generator.mtp_variant(int(kind.removeprefix("mtp")))


def _variant_for_extra(generator: Any, extra: tuple[str, ...]) -> Any:
    """`main` の `--variant` の指定 (`extra`) に対応する、既定を持つ変種を返す。"""
    if "--variant" not in extra:
        return generator.SMOKE
    kind = extra[extra.index("--variant") + 1]
    if kind == "full":
        return generator.FULL
    return generator.mtp_variant(int(extra[extra.index("--spec-tokens") + 1]))


def _load_generator() -> Any:
    """生成器の module を、パッケージでない道筋から読む (`experiments/nope-mla/`)。"""
    loader = SourceFileLoader("configure_tp2", str(GENERATOR_PATH))
    spec = importlib.util.spec_from_file_location("configure_tp2", GENERATOR_PATH, loader=loader)
    assert spec is not None, "生成器の spec が取れる"
    assert spec.loader is not None, "生成器の loader が取れる"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def generator() -> Any:
    return _load_generator()


def _inspect_item(**overrides: Any) -> dict[str, Any]:
    """実物の `docker image inspect` の形の見本 (使う鍵だけ)。"""
    item: dict[str, Any] = {
        "Id": EXPECTED_IMAGE_ID,
        "RepoTags": [IMAGE_TAG],
        "Architecture": "arm64",
        "Os": "linux",
        "Size": IMAGE_SIZE,
    }
    item.update(overrides)
    return item


def _write_inspect_json(tmp_path: Path, item: Mapping[str, Any], *, wrap: bool = True) -> Path:
    """見本の inspect JSON を一時ディレクトリに書く (実物と同じ、配列 1 要素の形)。"""
    path = tmp_path / "image-inspect.json"
    body: Any = [dict(item)] if wrap else dict(item)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _generate(generator: Any, tmp_path: Path, *, item: Mapping[str, Any] | None = None) -> Path:
    """正しい入力で生成し、出力の道筋を返す。"""
    source = _write_inspect_json(tmp_path, item if item is not None else _inspect_item())
    output = tmp_path / "tp2.toml"
    generator.generate(source, output)
    return output


def _generate_full(generator: Any, tmp_path: Path) -> Path:
    """`--variant full` に当たる呼び出しで生成し、出力の道筋を返す。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / "tp2-full.toml"
    generator.generate(source, output, generator.FULL)
    return output


def _render_default_input(generator: Any, tmp_path: Path) -> tuple[str, Any]:
    """実物の `configs.toml` と、見本の inspect の項目を用意して返す。"""
    text = CONFIGS_PATH.read_text(encoding="utf-8")
    image = generator.load_image(_write_inspect_json(tmp_path, _inspect_item()))
    return text, image


def _insert_decoy_table(text: str) -> str:
    """p1 節の `env.vllm-host-ip` の直前に、同じ字面の `value = "16"` の囮テーブルを差し込む。"""
    decoy = (
        "\n[configs.p1-nvfp4-tp2.args.decoy]\n"
        'flag = "--decoy"\n'
        'value = "16"\n'
        'why = "同じ字面の value を持つ別のテーブル"\n'
        'source = "https://example.invalid/"\n'
        'quote = "decoy"\n\n'
    )
    start = text.index("[configs.p1-nvfp4-tp2]")
    end = text.index("# ====", start)
    block = text[start:end]
    anchor = block.index("[configs.p1-nvfp4-tp2.env.vllm-host-ip]")
    block_with_decoy = block[:anchor] + decoy + block[anchor:]
    return text[:start] + block_with_decoy + text[end:]


# --- 入力の検査 (不正な入力では、出力を作らない) -------------------------


@pytest.mark.parametrize(
    "image_id",
    [
        pytest.param("sha256:9df45888", id="shortened-id"),
        pytest.param(IMAGE_TAG, id="tag"),
        pytest.param("sha256:" + "a" * 64, id="another-digest"),
    ],
)
def test_refuses_a_wrong_image_id(generator: Any, tmp_path: Path, image_id: str) -> None:
    """構成の ID が、記録の完全なイメージ ID でなければ断る。"""
    source = _write_inspect_json(tmp_path, _inspect_item(Id=image_id))
    output = tmp_path / "tp2.toml"

    with pytest.raises(ValueError, match="イメージ ID"):
        generator.generate(source, output)

    assert not output.exists()


@pytest.mark.parametrize(
    "platform",
    [
        pytest.param({"Architecture": "amd64"}, id="amd64"),
        pytest.param({"Os": "windows"}, id="windows"),
    ],
)
def test_refuses_a_wrong_platform(generator: Any, tmp_path: Path, platform: dict[str, str]) -> None:
    """linux/arm64 以外の平台は断る。"""
    source = _write_inspect_json(tmp_path, _inspect_item(**platform))
    output = tmp_path / "tp2.toml"

    with pytest.raises(ValueError, match="linux/arm64"):
        generator.generate(source, output)

    assert not output.exists()


@pytest.mark.parametrize(
    "size",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param("23438274807", id="string"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
    ],
)
def test_refuses_a_bad_size(generator: Any, tmp_path: Path, size: Any) -> None:
    """大きさは、`bool` を含めない正の整数だけを受ける。"""
    source = _write_inspect_json(tmp_path, _inspect_item(Size=size))
    output = tmp_path / "tp2.toml"

    with pytest.raises(ValueError, match="大きさ"):
        generator.generate(source, output)

    assert not output.exists()


def test_accepts_a_bare_mapping_without_the_array_wrapper(generator: Any, tmp_path: Path) -> None:
    """inspect の JSON は、配列で包まれていない辞書も受ける。"""
    source = _write_inspect_json(tmp_path, _inspect_item(), wrap=False)
    output = tmp_path / "tp2.toml"

    generator.generate(source, output)

    assert output.is_file()


@pytest.mark.parametrize("body", [[], [{}, {}], None, "image", 42, [None]])
def test_refuses_an_invalid_json_shape(generator: Any, tmp_path: Path, body: Any) -> None:
    """辞書または辞書 1 件の配列以外から構成を作らない。"""
    source = tmp_path / "image-inspect.json"
    source.write_text(json.dumps(body), encoding="utf-8")
    output = tmp_path / "tp2.toml"

    with pytest.raises(ValueError, match="形式が不正"):
        generator.generate(source, output)

    assert not output.exists()


# --- 出力先が既存なら、上書きしない -------------------------------------


def test_refuses_an_existing_output_and_keeps_it(generator: Any, tmp_path: Path) -> None:
    """既存の出力先には書かず、内容を 1 バイトも変えない。"""
    output = tmp_path / "tp2.toml"
    output.write_text("keep\n", encoding="utf-8")
    source = _write_inspect_json(tmp_path, _inspect_item())

    with pytest.raises(FileExistsError):
        generator.generate(source, output)

    assert output.read_text(encoding="utf-8") == "keep\n"


# --- 生成した TOML の読み込みと、基との差分 ------------------------------


def test_generated_config_loads_and_differs_from_p1_only_where_intended(
    generator: Any, tmp_path: Path
) -> None:
    """生成した TOML は `load_configs` で読める。基の `p1-nvfp4-tp2` との差は意図した場所だけ。

    差分の根拠は、計画の C3 である。
    """
    output = _generate(generator, tmp_path)
    generated_configs = load_configs(output, REPO_ROOT)

    assert set(generated_configs) == {TARGET_NAME}
    p2 = generated_configs[TARGET_NAME]
    p1 = load_configs(CONFIGS_PATH, REPO_ROOT)["p1-nvfp4-tp2"]

    assert p2.image.ref == EXPECTED_IMAGE_ID
    assert p2.image.seen_as == IMAGE_TAG
    assert p2.image.size_bytes == IMAGE_SIZE
    assert p2.image.measured == "docs/results/2026-09-22-nope-build.md"
    assert p2.description != p1.description
    assert "実重みでの起動は未確認" in p2.description
    assert p2.args["max-model-len"].value == "4096"
    assert p2.args["max-num-seqs"].value == "1"
    # C3: 差分はこの 2 つの value と why だけ。why は更新され、flag/source/quote は p1 のまま
    for key in ("max-model-len", "max-num-seqs"):
        assert p2.args[key].why != p1.args[key].why
        assert p2.args[key].flag == p1.args[key].flag
        assert p2.args[key].source == p1.args[key].source
        assert p2.args[key].quote == p1.args[key].quote

    # 差分は、説明、イメージの節、2 つの値 (と理由)、env の追加だけである
    assert p2.kind == p1.kind
    assert p2.nodes == p1.nodes
    assert p2.ready_timeout_s == p1.ready_timeout_s
    assert p2.served_model_name == p1.served_model_name
    assert p2.weights == p1.weights
    assert p2.docker == p1.docker
    other_args = {
        key: setting
        for key, setting in p2.args.items()
        if key not in ("max-model-len", "max-num-seqs")
    }
    assert other_args == {
        key: setting
        for key, setting in p1.args.items()
        if key not in ("max-model-len", "max-num-seqs")
    }
    kept_env = {key: setting for key, setting in p2.env.items() if key in p1.env}
    assert kept_env == p1.env

    # 追加した 3 つの根拠は、`netcheck-bandwidth` の同名の設定と同じである
    source_configs = load_configs(CONFIGS_PATH, REPO_ROOT)
    for key in ("nccl-debug", "nccl-debug-subsys", "nccl-debug-file"):
        assert key not in p1.env
        added = p2.env[key]
        origin = source_configs["netcheck-bandwidth"].env[key]
        assert added.source == origin.source
        assert added.quote == origin.quote

    # 基の節が残らず、次の構成の見出しのコメントも混ざらない
    text = output.read_text(encoding="utf-8")
    assert "[configs.p1-nvfp4-tp2" not in text
    assert "p1-fetch-nvfp4" not in text


def test_configs_toml_is_unchanged_by_generation(generator: Any, tmp_path: Path) -> None:
    """生成は、実物の `configs.toml` を読むだけで、書き換えない。"""
    before = CONFIGS_PATH.read_bytes()
    _generate(generator, tmp_path)

    assert CONFIGS_PATH.read_bytes() == before


# --- 起動計画 (2 台) -----------------------------------------------------


def _plans_for_generated_config(generator: Any, tmp_path: Path) -> tuple[ContainerPlan, ...]:
    """生成した TOML を、実物の `nodes.toml` で選び、2 台ぶんの計画を組み立てる。"""
    output = _generate(generator, tmp_path)
    configs = load_configs(output, REPO_ROOT)
    nodes = load_nodes(NODES_PATH, REPO_ROOT)
    config = select_config(configs, TARGET_NAME, nodes)
    return build_plans(config, nodes, STARTED_AT)


def _plans_for_full_config(generator: Any, tmp_path: Path) -> tuple[ContainerPlan, ...]:
    """full の生成 TOML を、実物の `nodes.toml` で選び、2 台ぶんの計画を組み立てる。"""
    output = _generate_full(generator, tmp_path)
    configs = load_configs(output, REPO_ROOT)
    nodes = load_nodes(NODES_PATH, REPO_ROOT)
    config = select_config(configs, FULL_TARGET_NAME, nodes)
    return build_plans(config, nodes, STARTED_AT)


def _find_pair(argv: tuple[str, ...], flag: str) -> tuple[str, str]:
    """`flag` と、その直後の値の組を返す (env 以外のフラグ付きの引数)。"""
    index = argv.index(flag)
    return argv[index], argv[index + 1]


def _insert_before_the_load_format(argv: list[str], pair: tuple[str, ...]) -> None:
    """`pair` を、`--load-format` の直前 (まだ無ければ末尾) に挿入する (C4 の挿入順の導出)。"""
    if LOAD_FORMAT_ARG_FLAG in argv:
        at = argv.index(LOAD_FORMAT_ARG_FLAG)
        argv[at:at] = list(pair)
    else:
        argv.extend(pair)


def test_plans_for_both_nodes(generator: Any, tmp_path: Path) -> None:
    """2 台の計画が、head → worker の順にでき、役割・名前・短い文脈・IB・NCCL の記録を持つ。

    差分の根拠は、計画の C4 である。
    """
    head, worker = _plans_for_generated_config(generator, tmp_path)

    assert [plan.node for plan in (head, worker)] == ["head", "worker"]
    assert head.container_name == f"vb-{TARGET_NAME}-head"
    assert worker.container_name == f"vb-{TARGET_NAME}-worker"

    for plan in (head, worker):
        argv = plan.argv
        assert EXPECTED_IMAGE_ID in argv
        image_at = argv.index(EXPECTED_IMAGE_ID)
        # docker の設定と環境変数は、イメージの参照より前に並ぶ
        assert argv.index("--device") < image_at
        assert argv[argv.index("--device") + 1] == "/dev/infiniband"
        for flag, value in NCCL_ENV:
            assert any(argv[index : index + 2] == (flag, value) for index in range(len(argv) - 1))
            assert argv.index(value) < image_at
        # 短い文脈と同時実行 1
        assert _find_pair(argv, "--max-model-len")[1] == "4096"
        assert _find_pair(argv, "--max-num-seqs")[1] == "1"
        # TP=2、2 台、順位、宛先
        assert _find_pair(argv, "--tensor-parallel-size")[1] == "2"
        assert _find_pair(argv, "--nnodes")[1] == "2"
        assert _find_pair(argv, "--master-addr")[1] == "192.168.100.10"
        # 重みのラベル
        assert plan.labels["vllm-baseline.weights"] == WEIGHTS_LABEL

    # 順位と、API を持たない指定
    assert _find_pair(head.argv, "--node-rank")[1] == "0"
    assert "--headless" not in head.argv
    assert _find_pair(worker.argv, "--node-rank")[1] == "1"
    assert "--headless" in worker.argv


def test_p2_argv_matches_p1_argv_with_the_intended_substitutions(
    generator: Any, tmp_path: Path
) -> None:
    """基の `p1-nvfp4-tp2` の 2 台の列に、意図した置換を当てると、`p2` の列と一致する。

    意図した置換: コンテナの名前、構成のラベル、イメージのラベルと参照、
    `--max-model-len` と `--max-num-seqs` の直後の値、`GLOO_SOCKET_IFNAME` の直後への
    初回の NCCL の記録の 3 つの `-e` の組の挿入。`config-sha256` のラベルは、コンテナの
    中身が変わるので両者で違うのが正しいため、比較から除く。
    """
    p1_configs = load_configs(CONFIGS_PATH, REPO_ROOT)
    nodes = load_nodes(NODES_PATH, REPO_ROOT)
    p1_plans = build_plans(select_config(p1_configs, "p1-nvfp4-tp2", nodes), nodes, STARTED_AT)
    p2_plans = _plans_for_generated_config(generator, tmp_path)

    for p1_plan, p2_plan in zip(p1_plans, p2_plans, strict=True):
        expected = list(p1_plan.argv)
        expected[expected.index(f"vb-p1-nvfp4-tp2-{p1_plan.node}")] = p2_plan.container_name
        expected[expected.index("vllm-baseline.config=p1-nvfp4-tp2")] = (
            f"vllm-baseline.config={TARGET_NAME}"
        )
        expected[expected.index(f"vllm-baseline.image={P1_IMAGE_REF}")] = (
            f"vllm-baseline.image={EXPECTED_IMAGE_ID}"
        )
        expected[expected.index(P1_IMAGE_REF)] = EXPECTED_IMAGE_ID
        expected[expected.index("--max-model-len") + 1] = "4096"
        expected[expected.index("--max-num-seqs") + 1] = "1"
        gloo_at = expected.index("GLOO_SOCKET_IFNAME=enp1s0f0np0")
        inserted: list[str] = []
        for flag, value in NCCL_ENV:
            inserted.extend((flag, value))
        expected[gloo_at + 1 : gloo_at + 1] = inserted

        kept = [arg for arg in expected if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        actual = [arg for arg in p2_plan.argv if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        assert kept == actual

        # 中身が変わったので、`config-sha256` は変わる (同じでは断る)
        assert p1_plan.labels[LABEL_CONFIG_SHA256] != p2_plan.labels[LABEL_CONFIG_SHA256]


# --- 構造化入力の負例 (同じ字面の value を、別のテーブルで替えない) -------


def test_render_leaves_the_same_value_line_in_another_table_alone(
    generator: Any, tmp_path: Path
) -> None:
    """同じ字面の `value = "16"` が別のテーブルにあっても、`max-num-seqs` だけが替わる。"""
    decoy = (
        "\n[configs.p1-nvfp4-tp2.args.decoy]\n"
        'flag = "--decoy"\n'
        'value = "16"\n'
        'why = "同じ字面の value を持つ別のテーブル"\n'
        'source = "https://example.invalid/"\n'
        'quote = "decoy"\n\n'
    )
    text = CONFIGS_PATH.read_text(encoding="utf-8")
    start = text.index("[configs.p1-nvfp4-tp2]")
    end = text.index("# ====", start)
    block = text[start:end]
    anchor = block.index("[configs.p1-nvfp4-tp2.env.vllm-host-ip]")

    # 囮のテーブルを、`env.vllm-host-ip` の直前に挿入する (節の中で、args と env の境に)
    block_with_decoy = block[:anchor] + decoy + block[anchor:]
    text_with_decoy = text[:start] + block_with_decoy + text[end:]

    image = generator.load_image(_write_inspect_json(tmp_path, _inspect_item()))
    rendered = generator.render(text_with_decoy, image)

    data = tomllib.loads(rendered)
    generated = data["configs"][TARGET_NAME]
    assert generated["args"]["decoy"]["value"] == "16"
    assert generated["args"]["max-num-seqs"]["value"] == "1"
    assert generated["args"]["max-model-len"]["value"] == "4096"
    # 囮の挿入で、別のテーブルの値が替わらない
    assert generated["docker"]["shm-size"]["value"] == "16g"
    assert generated["args"]["max-num-batched-tokens"]["value"] == "2048"
    # 区切り行 (`# ====`) より後の、次の構成の見出しのコメントは混ざらない
    assert "p1-fetch-nvfp4" not in rendered


# --- 生成器の依存の向き --------------------------------------------------

ALLOWED_IMPORT_ROOTS = frozenset(
    {"__future__", "argparse", "collections", "json", "pathlib", "re", "tomllib", "typing"}
)
FORBIDDEN_IMPORTS = frozenset({"os", "subprocess", "socket", "urllib", "httpx", "http"})


def test_generator_imports_no_network_or_subprocess() -> None:
    """生成器は、ネットワークも `subprocess` も読み込まない (`docker` を呼びに行かない)。

    `generator` の fixture を使わないのは、module の読み込みが要らないためである
    (`ast` は、ソースの文字列だけを見る)。
    """
    assert GENERATOR_PATH.is_file(), f"生成器がない: {GENERATOR_PATH}"
    source = GENERATOR_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            roots.add(node.module.split(".")[0])

    assert roots <= ALLOWED_IMPORT_ROOTS, f"許していない読み込み: {roots - ALLOWED_IMPORT_ROOTS}"
    assert not (roots & FORBIDDEN_IMPORTS)


# --- CLI の入口 ----------------------------------------------------------


def test_main_takes_the_two_paths(generator: Any, tmp_path: Path) -> None:
    """`main` は、inspect の JSON と出力先の 2 つの道筋を受けて生成する。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / "tp2.toml"

    generator.main([str(source), str(output)])

    assert output.is_file()
    loaded = load_configs(output, REPO_ROOT)
    assert set(loaded) == {TARGET_NAME}


# --- 引数なし (既定の variant) の出力の不変 -------------------------------


def test_default_variant_output_is_unchanged(generator: Any, tmp_path: Path) -> None:
    """`--variant` を省いた呼び出しのすべての形が、同じ smoke の出力になる。

    - `render(text, image)` と `render(text, image, generator.SMOKE)` が一致する
    - `render(text, image)` の SHA-256 が、変更前の生成器で計算した定数と一致する
    - `main([source, output])` (位置引数 2 つ) と `generate(source, output2)` の
      出力のバイト列が一致する
    - `main` の 2 台の計画の `config-sha256` が、実機で使った記録
      (HAND_OFF.md の構成ハッシュ) と一致する
    """
    text, image = _render_default_input(generator, tmp_path)
    default = generator.render(text, image)
    explicit = generator.render(text, image, generator.SMOKE)

    assert default == explicit
    assert hashlib.sha256(default.encode("utf-8")).hexdigest() == SMOKE_RENDER_SHA256

    source = _write_inspect_json(tmp_path, _inspect_item())
    output_main = tmp_path / "tp2-via-main.toml"
    output_generate = tmp_path / "tp2-via-generate.toml"
    generator.main([str(source), str(output_main)])
    generator.generate(source, output_generate)

    assert output_main.read_bytes() == output_generate.read_bytes()

    # 実機で使った smoke 構成との argv 一致の見張り (HAND_OFF.md の構成ハッシュ)
    configs = load_configs(output_main, REPO_ROOT)
    nodes = load_nodes(NODES_PATH, REPO_ROOT)
    plans = build_plans(select_config(configs, TARGET_NAME, nodes), nodes, STARTED_AT)
    assert plans[0].labels[LABEL_CONFIG_SHA256] == (
        "9bd65ac3e94c8690db3861bee46c7c22fe4985d317d01995c9ee5096072d61cd"
    )
    assert plans[1].labels[LABEL_CONFIG_SHA256] == (
        "9bd65ac3e94c8690db3861bee46c7c22fe4985d317d01995c9ee5096072d61cd"
    )


# --- full の生成 (構成の定義) ---------------------------------------------


def test_full_config_loads_and_keeps_p1_context_and_seqs(generator: Any, tmp_path: Path) -> None:
    """full の生成 TOML は `load_configs` で読め、p1 の文脈長と同時実行を根拠ごと保つ。

    確かめること:

    - `--max-model-len` と `--max-num-seqs` の `Setting` が p1 と完全一致 (163840、16)
    - `weights`、`docker`、p1 から引き継いだ `env` も一致する
    - `args` は、#50 の既定 `--load-format instanttensor` を除けば p1 と一致する
    - イメージの 4 項目は smoke と同じ
    - 説明に、prefill・並列・長文脈の計測用と、実機では未確認であることを書く
    - NCCL の記録 3 つは smoke と同じ flag/value/source/quote を持つ (外さない)
    - 基の p1 の節と、次の構成の見出しのコメントは混ざらない
    """
    output = _generate_full(generator, tmp_path)
    generated_configs = load_configs(output, REPO_ROOT)

    assert set(generated_configs) == {FULL_TARGET_NAME}
    full = generated_configs[FULL_TARGET_NAME]
    p1 = load_configs(CONFIGS_PATH, REPO_ROOT)["p1-nvfp4-tp2"]
    smoke = load_configs(_generate(generator, tmp_path), REPO_ROOT)[TARGET_NAME]

    # 文脈長と同時実行は、p1 の値と根拠をそのまま保つ (新しい数値を発明しない)
    assert full.args["max-model-len"] == p1.args["max-model-len"]
    assert full.args["max-num-seqs"] == p1.args["max-num-seqs"]
    assert full.args["max-model-len"].value == "163840"
    assert full.args["max-num-seqs"].value == "16"
    # #50 の既定 --load-format instanttensor を除けば、args は p1 のままである
    assert full.args[LOAD_FORMAT_ARG_KEY].value == DEFAULT_LOAD_FORMAT_VALUE
    assert {
        key: setting for key, setting in full.args.items() if key != LOAD_FORMAT_ARG_KEY
    } == p1.args

    # イメージ、重み、docker の設定は smoke と同じ (= p1 から引き継いだもの)
    assert full.image.ref == EXPECTED_IMAGE_ID
    assert full.image.seen_as == IMAGE_TAG
    assert full.image.size_bytes == IMAGE_SIZE
    assert full.image.measured == "docs/results/2026-09-22-nope-build.md"
    assert full.weights == p1.weights
    assert full.docker == p1.docker

    kept_env = {key: setting for key, setting in full.env.items() if key in p1.env}
    assert kept_env == p1.env

    # 説明は、prefill・並列・長文脈の計測用で、実機では未確認
    assert "prefill" in full.description
    assert "並列" in full.description
    assert "長文脈" in full.description
    assert "実機では未確認" in full.description

    # NCCL の記録の 3 つは smoke と同じ flag/value/source/quote (why は用途で変わる)
    for key, flag, value in (
        ("nccl-debug", "NCCL_DEBUG", "INFO"),
        ("nccl-debug-subsys", "NCCL_DEBUG_SUBSYS", "INIT,NET"),
        ("nccl-debug-file", "NCCL_DEBUG_FILE", "/logs/nccl.%h.%p.log"),
    ):
        added = full.env[key]
        smoke_env = smoke.env[key]
        assert added.flag == flag
        assert added.value == value
        assert (added.flag, added.value, added.source, added.quote) == (
            smoke_env.flag,
            smoke_env.value,
            smoke_env.source,
            smoke_env.quote,
        )
        origin = load_configs(CONFIGS_PATH, REPO_ROOT)["netcheck-bandwidth"].env[key]
        assert added.source == origin.source
        assert added.quote == origin.quote

    # 基の節が残らず、次の構成の見出しのコメントも混ざらない
    text = output.read_text(encoding="utf-8")
    assert "[configs.p1-nvfp4-tp2" not in text
    assert "p1-fetch-nvfp4" not in text


def test_full_render_leaves_p1_values_and_a_decoy_alone(generator: Any, tmp_path: Path) -> None:
    """full では p1 の 2 値も、同じ字面の value を持つ囮テーブルも書き換えない。"""
    text_with_decoy = _insert_decoy_table(CONFIGS_PATH.read_text(encoding="utf-8"))
    image = generator.load_image(_write_inspect_json(tmp_path, _inspect_item()))

    rendered = generator.render(text_with_decoy, image, generator.FULL)

    data = tomllib.loads(rendered)
    generated = data["configs"][FULL_TARGET_NAME]
    assert generated["args"]["decoy"]["value"] == "16"
    assert generated["args"]["max-num-seqs"]["value"] == "16"
    assert generated["args"]["max-model-len"]["value"] == "163840"
    # 囮の挿入で、別のテーブルの値が替わらない
    assert generated["docker"]["shm-size"]["value"] == "16g"
    assert generated["args"]["max-num-batched-tokens"]["value"] == "2048"
    assert "p1-fetch-nvfp4" not in rendered
    assert "[configs.p1-nvfp4-tp2" not in rendered


def test_full_refuses_existing_output_and_bad_input(generator: Any, tmp_path: Path) -> None:
    """full でも、既存の出力先には書かず、不正な入力では出力を作らない。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / "tp2-full.toml"
    output.write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        generator.generate(source, output, generator.FULL)

    assert output.read_text(encoding="utf-8") == "keep\n"

    bad_source = _write_inspect_json(tmp_path, _inspect_item(Id=IMAGE_TAG))
    bad_output = tmp_path / "tp2-full-bad.toml"

    with pytest.raises(ValueError, match="イメージ ID"):
        generator.generate(bad_source, bad_output, generator.FULL)

    assert not bad_output.exists()


# --- full の起動計画 (2 台) -----------------------------------------------


def test_full_plans_for_both_nodes(generator: Any, tmp_path: Path) -> None:
    """full の 2 台の計画が、head → worker の順にでき、文脈長 163840・同時 16・IB・NCCL を持つ。"""
    head, worker = _plans_for_full_config(generator, tmp_path)

    assert [plan.node for plan in (head, worker)] == ["head", "worker"]
    assert head.container_name == f"vb-{FULL_TARGET_NAME}-head"
    assert worker.container_name == f"vb-{FULL_TARGET_NAME}-worker"

    for plan in (head, worker):
        argv = plan.argv
        assert EXPECTED_IMAGE_ID in argv
        image_at = argv.index(EXPECTED_IMAGE_ID)
        # docker の設定と環境変数は、イメージの参照より前に並ぶ
        assert argv.index("--device") < image_at
        assert argv[argv.index("--device") + 1] == "/dev/infiniband"
        for flag, value in NCCL_ENV:
            assert any(argv[index : index + 2] == (flag, value) for index in range(len(argv) - 1))
            assert argv.index(value) < image_at
        # p1 のままの文脈長と同時実行
        assert _find_pair(argv, "--max-model-len")[1] == "163840"
        assert _find_pair(argv, "--max-num-seqs")[1] == "16"
        # TP=2、2 台、順位、宛先
        assert _find_pair(argv, "--tensor-parallel-size")[1] == "2"
        assert _find_pair(argv, "--nnodes")[1] == "2"
        assert _find_pair(argv, "--master-addr")[1] == "192.168.100.10"
        # 重みのラベルは smoke と同じ
        assert plan.labels["vllm-baseline.weights"] == WEIGHTS_LABEL

    # 順位と、API を持たない指定
    assert _find_pair(head.argv, "--node-rank")[1] == "0"
    assert "--headless" not in head.argv
    assert _find_pair(worker.argv, "--node-rank")[1] == "1"
    assert "--headless" in worker.argv


def test_full_argv_matches_smoke_argv_except_context_and_seqs(
    generator: Any, tmp_path: Path
) -> None:
    """smoke の 2 台の列に、full で替わる場所だけの置換を当てると、full の列と一致する。

    替わる場所: コンテナの名前、`vllm-baseline.config=` のラベル、
    `--max-model-len` と `--max-num-seqs` の直後の値、末尾への `--load-format instanttensor`
    (#50 の既定。smoke には入らない) の追加。イメージ ID、重みのラベル、
    NCCL の記録の 3 つの `-e` の組は同じである。`config-sha256` のラベルは、
    中身が変わるので両者で違うのが正しいため、比較から除く。
    """
    smoke_plans = _plans_for_generated_config(generator, tmp_path)
    full_plans = _plans_for_full_config(generator, tmp_path)

    for smoke_plan, full_plan in zip(smoke_plans, full_plans, strict=True):
        expected = list(smoke_plan.argv)
        expected[expected.index(f"vb-{TARGET_NAME}-{smoke_plan.node}")] = full_plan.container_name
        expected[expected.index(f"vllm-baseline.config={TARGET_NAME}")] = (
            f"vllm-baseline.config={FULL_TARGET_NAME}"
        )
        expected[expected.index("--max-model-len") + 1] = "163840"
        expected[expected.index("--max-num-seqs") + 1] = "16"
        expected.extend((LOAD_FORMAT_ARG_FLAG, DEFAULT_LOAD_FORMAT_VALUE))

        kept = [arg for arg in expected if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        actual = [arg for arg in full_plan.argv if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        assert kept == actual

        # 中身が変わったので、`config-sha256` は変わる (同じでは断る)
        assert smoke_plan.labels[LABEL_CONFIG_SHA256] != full_plan.labels[LABEL_CONFIG_SHA256]


# --- CLI の入口 (full) ----------------------------------------------------


def test_main_accepts_variant_full(generator: Any, tmp_path: Path) -> None:
    """`--variant full` を付けて `main` を呼ぶと、full の構成が生成される。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / "tp2-full.toml"

    generator.main(["--variant", "full", str(source), str(output)])

    assert output.is_file()
    loaded = load_configs(output, REPO_ROOT)
    assert set(loaded) == {FULL_TARGET_NAME}


# --- full の出力の不変 (C6) -----------------------------------------------


def test_full_variant_output_is_unchanged(generator: Any, tmp_path: Path) -> None:
    """full の既定の出力と、既定を外した出力の SHA-256 が、固定値と一致する (C2、C3)。

    既定の出力は #50 の `--load-format instanttensor` を含み、外した出力は変更前と
    バイト単位で同じである。
    """
    text, image = _render_default_input(generator, tmp_path)

    default = generator.render(text, image, generator.FULL)
    opt_out = generator.render(text, image, generator.without_load_format(generator.FULL))

    assert hashlib.sha256(default.encode("utf-8")).hexdigest() == FULL_RENDER_SHA256
    assert hashlib.sha256(opt_out.encode("utf-8")).hexdigest() == FULL_AUTO_RENDER_SHA256


# --- MTP の生成 (C5) ------------------------------------------------------


def _generate_mtp(generator: Any, tmp_path: Path, spec_tokens: int) -> Path:
    """`--variant full-mtp --spec-tokens N` に当たる呼び出しで生成し、出力の道筋を返す。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / f"tp2-mtp{spec_tokens}.toml"
    generator.generate(source, output, generator.mtp_variant(spec_tokens))
    return output


def _plans_for_mtp_config(
    generator: Any, tmp_path: Path, spec_tokens: int
) -> tuple[ContainerPlan, ...]:
    """MTP の生成 TOML を、実物の `nodes.toml` で選び、2 台ぶんの計画を組み立てる。"""
    output = _generate_mtp(generator, tmp_path, spec_tokens)
    configs = load_configs(output, REPO_ROOT)
    nodes = load_nodes(NODES_PATH, REPO_ROOT)
    config = select_config(configs, mtp_target_name(spec_tokens), nodes)
    return build_plans(config, nodes, STARTED_AT)


@pytest.mark.parametrize("n", MTP_SPEC_TOKENS)
def test_mtp_config_loads_and_keeps_full_but_adds_speculation(
    generator: Any, tmp_path: Path, n: int
) -> None:
    """MTP の生成 TOML は読め、名前・説明・投機の指定以外は full と同じである (C5)。"""
    output = _generate_mtp(generator, tmp_path, n)
    name = mtp_target_name(n)
    generated = load_configs(output, REPO_ROOT)

    assert set(generated) == {name}
    mtp = generated[name]
    full = load_configs(_generate_full(generator, tmp_path), REPO_ROOT)[FULL_TARGET_NAME]

    assert mtp.allow_speculative is True
    spec = mtp.args["speculative-config"]
    assert spec.flag == "--speculative-config"
    assert spec.value == mtp_spec_value(n)
    assert spec.source is not None
    assert spec.quote is not None

    # 名前・説明・投機の指定以外は、full と同じである (不要な差を作らない)
    assert mtp.image == full.image
    assert mtp.weights == full.weights
    assert mtp.docker == full.docker
    assert mtp.env == full.env
    assert mtp.kind == full.kind
    assert mtp.nodes == full.nodes
    assert mtp.ready_timeout_s == full.ready_timeout_s
    assert mtp.served_model_name == full.served_model_name
    other_args = {key: setting for key, setting in mtp.args.items() if key != "speculative-config"}
    assert other_args == full.args
    assert mtp.description != full.description
    assert "MTP" in mtp.description

    # 基の節が残らず、次の構成の見出しのコメントも混ざらない
    text = output.read_text(encoding="utf-8")
    assert "[configs.p1-nvfp4-tp2" not in text
    assert "p1-fetch-nvfp4" not in text


@pytest.mark.parametrize("n", MTP_SPEC_TOKENS)
def test_mtp_argv_matches_full_argv_plus_the_speculative_config(
    generator: Any, tmp_path: Path, n: int
) -> None:
    """MTP の 2 台の列は、full の列の `--shutdown-timeout` の直後に 2 語を足したもの (C5)。"""
    full_plans = _plans_for_full_config(generator, tmp_path)
    mtp_plans = _plans_for_mtp_config(generator, tmp_path, n)
    name = mtp_target_name(n)

    for full_plan, mtp_plan in zip(full_plans, mtp_plans, strict=True):
        expected = list(full_plan.argv)
        expected[expected.index(f"vb-{FULL_TARGET_NAME}-{full_plan.node}")] = (
            mtp_plan.container_name
        )
        expected[expected.index(f"vllm-baseline.config={FULL_TARGET_NAME}")] = (
            f"vllm-baseline.config={name}"
        )
        at = expected.index("--shutdown-timeout") + 2
        expected[at:at] = ["--speculative-config", mtp_spec_value(n)]

        kept = [arg for arg in expected if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        actual = [arg for arg in mtp_plan.argv if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        assert kept == actual

        # 中身が変わったので、`config-sha256` は変わる (同じでは断る)
        assert full_plan.labels[LABEL_CONFIG_SHA256] != mtp_plan.labels[LABEL_CONFIG_SHA256]


@pytest.mark.parametrize("n", MTP_SPEC_TOKENS)
def test_mtp_container_names_follow_the_pattern(generator: Any, tmp_path: Path, n: int) -> None:
    """MTP のコンテナ名は `vb-p2-nope-tp2-mtp<N>-<役割>` になる (C5、C7)。"""
    head, worker = _plans_for_mtp_config(generator, tmp_path, n)

    assert [plan.node for plan in (head, worker)] == ["head", "worker"]
    assert head.container_name == f"vb-{mtp_target_name(n)}-head"
    assert worker.container_name == f"vb-{mtp_target_name(n)}-worker"


def test_mtp_render_puts_allow_speculative_only_in_the_config_header(
    generator: Any, tmp_path: Path
) -> None:
    """投機を許す項目は構成の見出しにだけ入り、囮のテーブルもほかの設定も変えない (C5)。

    SCN-C5-P1 (最上位テーブルにだけ、`served_model_name` の直後) と、SCN-C5-N1 (同じ字面を
    持つ別テーブルには足さない) を固定する。
    """
    text_with_decoy = _insert_decoy_table(CONFIGS_PATH.read_text(encoding="utf-8"))
    image = generator.load_image(_write_inspect_json(tmp_path, _inspect_item()))

    rendered = generator.render(text_with_decoy, image, generator.mtp_variant(2))

    data = tomllib.loads(rendered)
    generated = data["configs"][mtp_target_name(2)]
    assert generated["allow_speculative"] is True
    # 許可は、最上位テーブルの `served_model_name` の直後に 1 回だけ入る
    assert rendered.count("allow_speculative") == 1
    assert 'served_model_name = "glm-5-3-flash"\nallow_speculative = true\n' in rendered
    # 投機の指定は、`args` の末尾 (= `env.nccl-debug` より前) にある
    assert rendered.index(
        f"[configs.{mtp_target_name(2)}.args.speculative-config]"
    ) < rendered.index(f"[configs.{mtp_target_name(2)}.env.nccl-debug]")
    # 囮のテーブルと、同じ字面の value を持つ設定は書き換わらない
    assert generated["args"]["decoy"]["value"] == "16"
    assert generated["args"]["served-model-name"]["value"] == "glm-5-3-flash"
    assert generated["args"]["max-num-seqs"]["value"] == "16"
    assert generated["args"]["max-model-len"]["value"] == "163840"
    for section in ("args", "env", "docker"):
        for key, setting in generated.get(section, {}).items():
            assert "allow_speculative" not in setting, (section, key)
    # 区切り行 (`# ====`) より後の、次の構成の見出しのコメントは混ざらない
    assert "[configs.p1-nvfp4-tp2" not in rendered
    assert "p1-fetch-nvfp4" not in rendered


# --- MTP の CLI の入口 (C5) -----------------------------------------------


def test_main_accepts_variant_full_mtp(generator: Any, tmp_path: Path) -> None:
    """`--variant full-mtp --spec-tokens N` で、MTP の構成が生成される (C5)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / "tp2-mtp2.toml"

    generator.main(["--variant", "full-mtp", "--spec-tokens", "2", str(source), str(output)])

    loaded = load_configs(output, REPO_ROOT)
    assert set(loaded) == {mtp_target_name(2)}
    assert loaded[mtp_target_name(2)].allow_speculative is True


@pytest.mark.parametrize("bad", ["0", "-1", "abc"])
def test_main_refuses_a_bad_spec_tokens(generator: Any, tmp_path: Path, bad: str) -> None:
    """`--spec-tokens` は正の整数だけを受け、それ以外では出力を作らない (C5)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / f"tp2-mtp-{bad}.toml"

    with pytest.raises((SystemExit, ValueError)):
        generator.main(["--variant", "full-mtp", "--spec-tokens", bad, str(source), str(output)])

    assert not output.exists()


def test_main_requires_spec_tokens_for_full_mtp(generator: Any, tmp_path: Path) -> None:
    """`--variant full-mtp` に `--spec-tokens` がなければ断る (C5)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / "tp2-mtp-none.toml"

    with pytest.raises(SystemExit):
        generator.main(["--variant", "full-mtp", str(source), str(output)])

    assert not output.exists()


def test_main_refuses_spec_tokens_with_another_variant(generator: Any, tmp_path: Path) -> None:
    """`full-mtp` 以外の variant と `--spec-tokens` の組み合わせは断る (C5)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / "tp2-full-with-spec.toml"

    with pytest.raises(SystemExit):
        generator.main(["--variant", "full", "--spec-tokens", "2", str(source), str(output)])

    assert not output.exists()


# --- thread-name 付きの構成の生成 (C1、C2、C3) ----------------------------


def _generate_variant(generator: Any, tmp_path: Path, variant: Any, stem: str) -> Path:
    """`variant` で生成し、出力の道筋を返す。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / f"{stem}.toml"
    generator.generate(source, output, variant)
    return output


def _plans_for_thread_name_config(
    generator: Any, tmp_path: Path, kind: str, *, thread_names: bool
) -> tuple[ContainerPlan, ...]:
    """`kind` の変種 (thread-name の有無を選べる) の 2 台ぶんの計画を組み立てる。"""
    base = _base_variant(generator, kind)
    variant = generator.with_thread_names(base) if thread_names else base
    stem = f"plan-{'tn' if thread_names else 'base'}-{kind}"
    output = _generate_variant(generator, tmp_path, variant, stem)
    configs = load_configs(output, REPO_ROOT)
    nodes = load_nodes(NODES_PATH, REPO_ROOT)
    return build_plans(select_config(configs, variant.name, nodes), nodes, STARTED_AT)


@pytest.mark.parametrize("kind", THREAD_NAME_BASE_KINDS)
def test_thread_names_config_adds_only_the_env_and_the_name(
    generator: Any, tmp_path: Path, kind: str
) -> None:
    """`--nccl-thread-names` は、名前 (`-tn`) と env 1 つだけを足し、他は基のままにする。

    C1 (名前と env) と C2 (根拠と、名前・説明以外の一致) を固定する。
    """
    base = _base_variant(generator, kind)
    named = generator.with_thread_names(base)
    baseline = load_configs(
        _generate_variant(generator, tmp_path, base, f"base-{kind}"), REPO_ROOT
    )[base.name]
    named_configs = load_configs(
        _generate_variant(generator, tmp_path, named, f"tn-{kind}"), REPO_ROOT
    )

    assert named.name == f"{base.name}-tn"
    assert set(named_configs) == {named.name}
    tn = named_configs[named.name]

    added = tn.env[THREAD_NAME_ENV_KEY]
    assert added.flag == "NCCL_SET_THREAD_NAME"
    assert added.value == "1"
    assert {
        key: setting for key, setting in tn.env.items() if key != THREAD_NAME_ENV_KEY
    } == baseline.env

    # 名前と説明以外は、基の変種と同じである
    assert tn.description != baseline.description
    for field in (
        "args",
        "image",
        "weights",
        "docker",
        "allow_speculative",
        "kind",
        "nodes",
        "ready_timeout_s",
        "served_model_name",
    ):
        assert getattr(tn, field) == getattr(baseline, field), field


@pytest.mark.parametrize("kind", THREAD_NAME_BASE_KINDS)
def test_thread_names_env_provenance_is_the_nccl_doc(
    generator: Any, tmp_path: Path, kind: str
) -> None:
    """足す env の根拠が、NCCL の公式文書の環境変数の説明と原文で、`load_configs` で読める。

    C2 を固定する。
    """
    base = _base_variant(generator, kind)
    named = generator.with_thread_names(base)
    named_configs = load_configs(
        _generate_variant(generator, tmp_path, named, f"tn-prov-{kind}"), REPO_ROOT
    )

    added = named_configs[named.name].env[THREAD_NAME_ENV_KEY]
    assert added.flag == "NCCL_SET_THREAD_NAME"
    assert str(added.source) == NCCL_ENV_DOC
    assert added.quote is not None
    assert NCCL_QUOTE_FRAGMENT in added.quote


@pytest.mark.parametrize("kind", THREAD_NAME_BASE_KINDS)
def test_thread_names_argv_matches_base_argv_plus_one_env_pair(
    generator: Any, tmp_path: Path, kind: str
) -> None:
    """thread-name 付きの 2 台の列は、基の列に `-e NCCL_SET_THREAD_NAME=1` を 1 組だけ足したもの。

    `NCCL_DEBUG_FILE` の直後に足し、コンテナ名と `config` のラベルだけ置き換える (C1)。
    `config-sha256` は中身が変わるので両者で違うのが正しいため、比較から除く。
    """
    base = _base_variant(generator, kind)
    named = generator.with_thread_names(base)
    base_plans = _plans_for_thread_name_config(generator, tmp_path, kind, thread_names=False)
    named_plans = _plans_for_thread_name_config(generator, tmp_path, kind, thread_names=True)

    for base_plan, named_plan in zip(base_plans, named_plans, strict=True):
        expected = list(base_plan.argv)
        expected[expected.index(f"vb-{base.name}-{base_plan.node}")] = named_plan.container_name
        expected[expected.index(f"vllm-baseline.config={base.name}")] = (
            f"vllm-baseline.config={named.name}"
        )
        at = expected.index("NCCL_DEBUG_FILE=/logs/nccl.%h.%p.log") + 1
        expected[at:at] = list(THREAD_NAME_ENV)

        kept = [arg for arg in expected if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        actual = [arg for arg in named_plan.argv if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        assert kept == actual
        assert base_plan.labels[LABEL_CONFIG_SHA256] != named_plan.labels[LABEL_CONFIG_SHA256]


@pytest.mark.parametrize(
    ("extra", "expected_name"),
    [
        pytest.param((), f"{TARGET_NAME}-tn", id="smoke"),
        pytest.param(("--variant", "full"), f"{FULL_TARGET_NAME}-tn", id="full"),
        pytest.param(
            ("--variant", "full-mtp", "--spec-tokens", "2"),
            f"{mtp_target_name(2)}-tn",
            id="full-mtp",
        ),
    ],
)
def test_main_accepts_nccl_thread_names_with_each_variant(
    generator: Any, tmp_path: Path, extra: tuple[str, ...], expected_name: str
) -> None:
    """`--nccl-thread-names` は、どの `--variant` にも付けられ、`-tn` の構成を生成する (C1)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / f"{expected_name}.toml"

    generator.main([*extra, "--nccl-thread-names", str(source), str(output)])

    loaded = load_configs(output, REPO_ROOT)
    assert set(loaded) == {expected_name}
    assert loaded[expected_name].env[THREAD_NAME_ENV_KEY].flag == "NCCL_SET_THREAD_NAME"


def test_thread_names_full_plan_names_follow_the_pattern(generator: Any, tmp_path: Path) -> None:
    """full に付けたときの構成名とコンテナ名が、`-tn` の付いた名前になる (C1)。"""
    head, worker = _plans_for_thread_name_config(generator, tmp_path, "full", thread_names=True)

    assert [plan.node for plan in (head, worker)] == ["head", "worker"]
    assert head.container_name == "vb-p2-nope-tp2-full-tn-head"
    assert worker.container_name == "vb-p2-nope-tp2-full-tn-worker"
    assert head.labels[LABEL_CONFIG] == "p2-nope-tp2-full-tn"


def test_thread_names_output_without_the_flag_is_unchanged(generator: Any, tmp_path: Path) -> None:
    """付けない full の出力は固定値と一致し、`-tn` の名前は空いている (C1、C3)。

    既定つきの出力は #50 の固定値、既定を外した出力は変更前の固定値と一致する。
    """
    text, image = _render_default_input(generator, tmp_path)

    rendered = generator.render(text, image, generator.FULL)
    opt_out = generator.render(text, image, generator.without_load_format(generator.FULL))

    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == FULL_RENDER_SHA256
    assert hashlib.sha256(opt_out.encode("utf-8")).hexdigest() == FULL_AUTO_RENDER_SHA256
    assert "NCCL_SET_THREAD_NAME" not in rendered
    assert f"[configs.{FULL_TARGET_NAME}-tn]" not in rendered
    assert f"{FULL_TARGET_NAME}-tn" not in load_configs(CONFIGS_PATH, REPO_ROOT)


def test_thread_names_render_leaves_a_decoy_alone(generator: Any, tmp_path: Path) -> None:
    """同じ字面の `value = "16"` の囮テーブルがあっても、足す env は 1 つだけで、囮は触らない。

    C1 を固定する。
    """
    text_with_decoy = _insert_decoy_table(CONFIGS_PATH.read_text(encoding="utf-8"))
    image = generator.load_image(_write_inspect_json(tmp_path, _inspect_item()))
    named = generator.with_thread_names(generator.FULL)

    rendered = generator.render(text_with_decoy, image, named)

    generated = tomllib.loads(rendered)["configs"][named.name]
    assert generated["args"]["decoy"]["value"] == "16"
    assert generated["args"]["max-model-len"]["value"] == "163840"
    assert generated["env"][THREAD_NAME_ENV_KEY]["flag"] == "NCCL_SET_THREAD_NAME"
    assert rendered.count('flag = "NCCL_SET_THREAD_NAME"') == 1
    assert "[configs.p1-nvfp4-tp2" not in rendered
    assert "p1-fetch-nvfp4" not in rendered


@pytest.mark.parametrize(
    ("spec_tokens", "expected", "expected_auto"),
    [
        pytest.param(1, MTP1_RENDER_SHA256, MTP1_AUTO_RENDER_SHA256, id="mtp1"),
        pytest.param(2, MTP2_RENDER_SHA256, MTP2_AUTO_RENDER_SHA256, id="mtp2"),
    ],
)
def test_mtp_variant_output_is_unchanged(
    generator: Any, tmp_path: Path, spec_tokens: int, expected: str, expected_auto: str
) -> None:
    """mtp の既定の出力と、既定を外した出力の SHA-256 が固定値と一致する (C2、C3)。"""
    text, image = _render_default_input(generator, tmp_path)
    variant = generator.mtp_variant(spec_tokens)

    rendered = generator.render(text, image, variant)
    opt_out = generator.render(text, image, generator.without_load_format(variant))

    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == expected
    assert hashlib.sha256(opt_out.encode("utf-8")).hexdigest() == expected_auto


@pytest.mark.parametrize("kind", THREAD_NAME_BASE_KINDS)
def test_render_without_thread_names_has_no_thread_name_env(
    generator: Any, tmp_path: Path, kind: str
) -> None:
    """付けない 4 つの変種の出力に、`NCCL_SET_THREAD_NAME` は現れない (C3)。"""
    text, image = _render_default_input(generator, tmp_path)

    rendered = generator.render(text, image, _base_variant(generator, kind))

    assert "NCCL_SET_THREAD_NAME" not in rendered


# --- torch プロファイラー付きの構成の生成 (C1、C2、C3、C4) -----------------


def _plans_for_profiler_config(
    generator: Any, tmp_path: Path, kind: str, *, thread_names: bool
) -> tuple[ContainerPlan, ...]:
    """`kind` に `--torch-profiler` (必要なら `--nccl-thread-names` も) を付けた計画。"""
    variant = _base_variant(generator, kind)
    if thread_names:
        variant = generator.with_thread_names(variant)
    variant = generator.with_torch_profiler(variant)
    stem = f"plan-prof-{kind}-tn{int(thread_names)}"
    output = _generate_variant(generator, tmp_path, variant, stem)
    configs = load_configs(output, REPO_ROOT)
    nodes = load_nodes(NODES_PATH, REPO_ROOT)
    return build_plans(select_config(configs, variant.name, nodes), nodes, STARTED_AT)


@pytest.mark.parametrize("kind", PROFILER_BASE_KINDS)
def test_profiler_config_adds_only_the_arg_and_the_name(
    generator: Any, tmp_path: Path, kind: str
) -> None:
    """`--torch-profiler` は、名前 (`-prof`) と `--profiler-config` の args 1 つだけを足す (C1)。

    値は `profiler=torch` と `/logs/` の下の出力先の JSON で、`load_configs` で読める。
    名前と説明以外 (image、weights、docker、env、ほかの args、kind、nodes など) は基のままだ。
    """
    base = _base_variant(generator, kind)
    profiled = generator.with_torch_profiler(base)
    baseline = load_configs(
        _generate_variant(generator, tmp_path, base, f"base-{kind}"), REPO_ROOT
    )[base.name]
    generated = load_configs(
        _generate_variant(generator, tmp_path, profiled, f"prof-{kind}"), REPO_ROOT
    )

    assert profiled.name == f"{base.name}{PROFILER_SUFFIX}"
    assert set(generated) == {profiled.name}
    prof = generated[profiled.name]

    added = prof.args[PROFILER_ARG_KEY]
    assert added.flag == PROFILER_ARG_FLAG
    assert added.value == PROFILER_VALUE
    rest = {key: setting for key, setting in prof.args.items() if key != PROFILER_ARG_KEY}
    assert rest == baseline.args

    assert prof.description != baseline.description
    for field in (
        "image",
        "weights",
        "docker",
        "env",
        "allow_speculative",
        "kind",
        "nodes",
        "ready_timeout_s",
        "served_model_name",
    ):
        assert getattr(prof, field) == getattr(baseline, field), field


@pytest.mark.parametrize("kind", PROFILER_BASE_KINDS)
def test_profiler_arg_provenance_and_value(generator: Any, tmp_path: Path, kind: str) -> None:
    """足す args の根拠が vLLM のソースで、値が torch・`/logs/` の下の出力先・2 つの `false` (C2)。

    `torch_profiler_with_stack` と `torch_profiler_dump_cuda_time_total` は、JSON の `false` が
    `bool` の `False` として読めることまで固定する (#48)。
    """
    base = _base_variant(generator, kind)
    profiled = generator.with_torch_profiler(base)
    generated = load_configs(
        _generate_variant(generator, tmp_path, profiled, f"prof-prov-{kind}"), REPO_ROOT
    )

    added = generated[profiled.name].args[PROFILER_ARG_KEY]
    assert str(added.source) == PROFILER_SOURCE
    assert added.quote is not None
    for fragment in PROFILER_QUOTE_FRAGMENTS:
        assert fragment in added.quote

    assert added.value is not None
    value = json.loads(added.value)
    assert value["profiler"] == "torch"
    assert value["torch_profiler_dir"] == PROFILER_DIR
    assert value["torch_profiler_dir"].startswith("/logs/")
    assert value["torch_profiler_with_stack"] is False
    assert value["torch_profiler_dump_cuda_time_total"] is False


@pytest.mark.parametrize("kind", PROFILER_BASE_KINDS)
def test_profiler_argv_matches_base_argv_plus_the_arg_pair(
    generator: Any, tmp_path: Path, kind: str
) -> None:
    """`--torch-profiler` の列は、基の列の末尾に `--profiler-config` の 2 語を足したもの (C1)。

    コンテナ名と `config` のラベルだけ置き換える。`config-sha256` は中身が変わるので、
    両者で違うのが正しいため比較から除く。
    """
    base = _base_variant(generator, kind)
    profiled = generator.with_torch_profiler(base)
    base_plans = _plans_for_thread_name_config(generator, tmp_path, kind, thread_names=False)
    prof_plans = _plans_for_profiler_config(generator, tmp_path, kind, thread_names=False)

    for base_plan, prof_plan in zip(base_plans, prof_plans, strict=True):
        expected = list(base_plan.argv)
        expected[expected.index(f"vb-{base.name}-{base_plan.node}")] = prof_plan.container_name
        expected[expected.index(f"vllm-baseline.config={base.name}")] = (
            f"vllm-baseline.config={profiled.name}"
        )
        _insert_before_the_load_format(expected, PROFILER_ARG_PAIR)

        kept = [arg for arg in expected if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        actual = [arg for arg in prof_plan.argv if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        assert kept == actual
        assert base_plan.labels[LABEL_CONFIG_SHA256] != prof_plan.labels[LABEL_CONFIG_SHA256]


def test_profiler_with_thread_names_names_and_arg_order(generator: Any, tmp_path: Path) -> None:
    """2 旗を重ねると `-tn-prof` になり、args の末尾は投機 → プロファイラー → load-format (C4)。"""
    variant = generator.with_torch_profiler(generator.with_thread_names(generator.mtp_variant(2)))

    assert variant.name == "p2-nope-tp2-mtp2-tn-prof"

    generated = load_configs(
        _generate_variant(generator, tmp_path, variant, "prof-tn-mtp2"), REPO_ROOT
    )
    keys = list(generated[variant.name].args)
    assert keys[-3:] == ["speculative-config", PROFILER_ARG_KEY, LOAD_FORMAT_ARG_KEY]


@pytest.mark.parametrize("kind", PROFILER_BASE_KINDS)
def test_profiler_with_thread_names_argv_matches_base_plus_both_additions(
    generator: Any, tmp_path: Path, kind: str
) -> None:
    """`-tn` と `-prof` の列が、基の列に 2 つの追加だけを足したものになる (C3)。"""
    base = _base_variant(generator, kind)
    both = generator.with_torch_profiler(generator.with_thread_names(base))
    base_plans = _plans_for_thread_name_config(generator, tmp_path, kind, thread_names=False)
    both_plans = _plans_for_profiler_config(generator, tmp_path, kind, thread_names=True)

    for base_plan, both_plan in zip(base_plans, both_plans, strict=True):
        expected = list(base_plan.argv)
        expected[expected.index(f"vb-{base.name}-{base_plan.node}")] = both_plan.container_name
        expected[expected.index(f"vllm-baseline.config={base.name}")] = (
            f"vllm-baseline.config={both.name}"
        )
        at = expected.index("NCCL_DEBUG_FILE=/logs/nccl.%h.%p.log") + 1
        expected[at:at] = list(THREAD_NAME_ENV)
        _insert_before_the_load_format(expected, PROFILER_ARG_PAIR)

        kept = [arg for arg in expected if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        actual = [arg for arg in both_plan.argv if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        assert kept == actual
        assert base_plan.labels[LABEL_CONFIG_SHA256] != both_plan.labels[LABEL_CONFIG_SHA256]


@pytest.mark.parametrize(
    ("extra", "expected_name"),
    [
        pytest.param((), f"{TARGET_NAME}{PROFILER_SUFFIX}", id="smoke"),
        pytest.param(("--variant", "full"), f"{FULL_TARGET_NAME}{PROFILER_SUFFIX}", id="full"),
        pytest.param(
            ("--variant", "full-mtp", "--spec-tokens", "2"),
            f"{mtp_target_name(2)}{PROFILER_SUFFIX}",
            id="full-mtp",
        ),
    ],
)
def test_main_accepts_torch_profiler_with_each_variant(
    generator: Any, tmp_path: Path, extra: tuple[str, ...], expected_name: str
) -> None:
    """`--torch-profiler` は、どの `--variant` にも付けられ、`-prof` の構成を生成する (C3)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / f"{expected_name}.toml"

    generator.main([*extra, "--torch-profiler", str(source), str(output)])

    loaded = load_configs(output, REPO_ROOT)
    assert set(loaded) == {expected_name}
    assert loaded[expected_name].args[PROFILER_ARG_KEY].flag == PROFILER_ARG_FLAG


def test_main_makes_the_same_profiler_name_in_either_flag_order(
    generator: Any, tmp_path: Path
) -> None:
    """`main` は、旗の並び順に依らず `-tn-prof` の同じ出力を作る (C3)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    first = tmp_path / "prof-tn-first.toml"
    second = tmp_path / "prof-tn-second.toml"

    generator.main(
        ["--variant", "full", "--torch-profiler", "--nccl-thread-names", str(source), str(first)]
    )
    generator.main(
        ["--variant", "full", "--nccl-thread-names", "--torch-profiler", str(source), str(second)]
    )

    assert first.read_bytes() == second.read_bytes()
    assert set(load_configs(first, REPO_ROOT)) == {f"{FULL_TARGET_NAME}-tn-prof"}


def test_profiler_output_without_the_flag_is_unchanged(generator: Any, tmp_path: Path) -> None:
    """付けない 5 形の出力が不変で、`--profiler-config` と `-prof` が現れない (C4)。"""
    text, image = _render_default_input(generator, tmp_path)
    variants = (
        ("smoke", generator.SMOKE),
        ("full", generator.FULL),
        ("full-tn", generator.with_thread_names(generator.FULL)),
        ("mtp1", generator.mtp_variant(1)),
        ("mtp2", generator.mtp_variant(2)),
    )

    for label, variant in variants:
        rendered = generator.render(text, image, variant)
        assert PROFILER_ARG_FLAG not in rendered, label
        assert f"[configs.{variant.name}{PROFILER_SUFFIX}]" not in rendered, label

    full_tn = generator.render(text, image, generator.with_thread_names(generator.FULL))
    assert hashlib.sha256(full_tn.encode("utf-8")).hexdigest() == FULL_TN_RENDER_SHA256
    full_tn_opt_out = generator.render(
        text, image, generator.without_load_format(generator.with_thread_names(generator.FULL))
    )
    assert hashlib.sha256(full_tn_opt_out.encode("utf-8")).hexdigest() == FULL_TN_AUTO_RENDER_SHA256
    existing = load_configs(CONFIGS_PATH, REPO_ROOT)
    assert f"{FULL_TARGET_NAME}{PROFILER_SUFFIX}" not in existing
    assert f"{FULL_TARGET_NAME}-tn{PROFILER_SUFFIX}" not in existing


def test_profiler_render_leaves_a_decoy_alone(generator: Any, tmp_path: Path) -> None:
    """同じ字面の value の囮テーブルがあっても、足す args は 1 つだけで、囮は触らない (C4)。"""
    text_with_decoy = _insert_decoy_table(CONFIGS_PATH.read_text(encoding="utf-8"))
    image = generator.load_image(_write_inspect_json(tmp_path, _inspect_item()))
    profiled = generator.with_torch_profiler(generator.FULL)

    rendered = generator.render(text_with_decoy, image, profiled)

    generated = tomllib.loads(rendered)["configs"][profiled.name]
    assert generated["args"]["decoy"]["value"] == "16"
    assert generated["args"]["max-model-len"]["value"] == "163840"
    assert generated["args"][PROFILER_ARG_KEY]["flag"] == PROFILER_ARG_FLAG
    assert rendered.count(f'flag = "{PROFILER_ARG_FLAG}"') == 1
    assert f"{FULL_TARGET_NAME}{PROFILER_SUFFIX}" not in load_configs(CONFIGS_PATH, REPO_ROOT)
    assert "[configs.p1-nvfp4-tp2" not in rendered
    assert "p1-fetch-nvfp4" not in rendered


# --- 既定の重みの読み込み (full / full-mtp。#50。C1、C2、C3、C4) -------------


def test_full_default_load_format_table_is_inserted_once_before_the_env_comment(
    generator: Any, tmp_path: Path
) -> None:
    """full の出力に、既定の load-format テーブルが env コメントの直前に 1 つだけ入る (C1)。"""
    text, image = _render_default_input(generator, tmp_path)

    rendered = generator.render(text, image, generator.FULL)

    generated = tomllib.loads(rendered)["configs"][FULL_TARGET_NAME]
    assert list(generated["args"])[-1] == LOAD_FORMAT_ARG_KEY
    assert generated["args"][LOAD_FORMAT_ARG_KEY]["value"] == DEFAULT_LOAD_FORMAT_VALUE
    assert rendered.count(f'flag = "{LOAD_FORMAT_ARG_FLAG}"') == 1
    assert rendered.index(
        f"[configs.{FULL_TARGET_NAME}.args.{LOAD_FORMAT_ARG_KEY}]"
    ) < rendered.index(generator.FULL.env_comment)


@pytest.mark.parametrize("spec_tokens", (1, 2))
def test_mtp_default_load_format_table_follows_the_speculative_table(
    generator: Any, tmp_path: Path, spec_tokens: int
) -> None:
    """mtp の出力で、既定の load-format テーブルは投機のテーブルの後に入る (C1)。"""
    text, image = _render_default_input(generator, tmp_path)
    variant = generator.mtp_variant(spec_tokens)

    rendered = generator.render(text, image, variant)

    generated = tomllib.loads(rendered)["configs"][variant.name]
    assert list(generated["args"])[-2:] == ["speculative-config", LOAD_FORMAT_ARG_KEY]
    assert variant.name == mtp_target_name(spec_tokens)


@pytest.mark.parametrize("kind", ("full", "mtp1", "mtp2"))
def test_default_load_format_provenance_is_the_pinned_vllm_load_source(
    generator: Any, tmp_path: Path, kind: str
) -> None:
    """full / mtp の既定の load-format の根拠が、固定 commit の vLLM の LoadConfig である (C1)。"""
    variant = _base_variant(generator, kind)
    generated = load_configs(
        _generate_variant(generator, tmp_path, variant, f"default-prov-{kind}"), REPO_ROOT
    )

    added = generated[variant.name].args[LOAD_FORMAT_ARG_KEY]
    assert added.flag == LOAD_FORMAT_ARG_FLAG
    assert added.value == DEFAULT_LOAD_FORMAT_VALUE
    assert str(added.source) == DEFAULT_LOAD_FORMAT_SOURCE
    assert added.quote is not None
    assert LOAD_FORMAT_QUOTE_FRAGMENTS[DEFAULT_LOAD_FORMAT_VALUE] in added.quote
    assert added.why == DEFAULT_LOAD_FORMAT_WHY
    for fragment in ("#50", "#37", "363", "32"):
        assert fragment in added.why


def test_smoke_has_no_default_load_format(generator: Any, tmp_path: Path) -> None:
    """smoke には既定の load-format を足さない (C1 の負例、#50 の設計判断)。"""
    text, image = _render_default_input(generator, tmp_path)

    rendered = generator.render(text, image, generator.SMOKE)

    assert LOAD_FORMAT_ARG_FLAG not in rendered
    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == SMOKE_RENDER_SHA256


@pytest.mark.parametrize("kind", ("full", "mtp1", "mtp2"))
def test_default_load_format_is_appended_to_the_argv(
    generator: Any, tmp_path: Path, kind: str
) -> None:
    """既定つきの 2 台の argv は、opt-out の argv の末尾に load-format の組を足したもの (C1)。"""
    variant = _base_variant(generator, kind)
    opt_out_plans = _plans_for_variant(
        generator, tmp_path, generator.without_load_format(variant), f"default-argv-optout-{kind}"
    )
    default_plans = _plans_for_variant(generator, tmp_path, variant, f"default-argv-{kind}")

    for opt_out_plan, default_plan in zip(opt_out_plans, default_plans, strict=True):
        expected = list(opt_out_plan.argv)
        expected.extend((LOAD_FORMAT_ARG_FLAG, DEFAULT_LOAD_FORMAT_VALUE))

        kept = [arg for arg in expected if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        actual = [arg for arg in default_plan.argv if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        assert kept == actual
        assert opt_out_plan.labels[LABEL_CONFIG_SHA256] != default_plan.labels[LABEL_CONFIG_SHA256]


def test_full_default_load_format_keeps_the_config_and_container_names(
    generator: Any, tmp_path: Path
) -> None:
    """既定の付与で構成名とコンテナ名は変わらず、-lf-instanttensor も現れない (C1)。"""
    text, image = _render_default_input(generator, tmp_path)

    rendered = generator.render(text, image, generator.FULL)

    assert f"[configs.{FULL_TARGET_NAME}]" in rendered
    assert f"{FULL_TARGET_NAME}{load_format_suffix(DEFAULT_LOAD_FORMAT_VALUE)}" not in rendered

    head, worker = _plans_for_full_config(generator, tmp_path)
    assert head.container_name == f"vb-{FULL_TARGET_NAME}-head"
    assert worker.container_name == f"vb-{FULL_TARGET_NAME}-worker"


def test_full_default_load_format_render_leaves_a_decoy_alone(
    generator: Any, tmp_path: Path
) -> None:
    """同じ字面の value の囮テーブルがあっても、既定のテーブルは 1 つで囮は変わらない (C1)。"""
    text_with_decoy = _insert_decoy_table(CONFIGS_PATH.read_text(encoding="utf-8"))
    image = generator.load_image(_write_inspect_json(tmp_path, _inspect_item()))

    rendered = generator.render(text_with_decoy, image, generator.FULL)

    generated = tomllib.loads(rendered)["configs"][FULL_TARGET_NAME]
    assert generated["args"]["decoy"]["value"] == "16"
    assert generated["args"]["max-num-seqs"]["value"] == "16"
    assert rendered.count(f'flag = "{LOAD_FORMAT_ARG_FLAG}"') == 1
    assert "[configs.p1-nvfp4-tp2" not in rendered


@pytest.mark.parametrize(
    ("extra", "auto_sha256", "target"),
    [
        pytest.param(("--variant", "full"), FULL_AUTO_RENDER_SHA256, FULL_TARGET_NAME, id="full"),
        pytest.param(
            ("--variant", "full-mtp", "--spec-tokens", "2"),
            MTP2_AUTO_RENDER_SHA256,
            mtp_target_name(2),
            id="full-mtp",
        ),
    ],
)
def test_main_load_format_auto_matches_without_load_format(
    generator: Any, tmp_path: Path, extra: tuple[str, ...], auto_sha256: str, target: str
) -> None:
    """`--load-format auto` の出力は、変更前の既定 (without_load_format) と同じバイト列 (C2)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    out_main = tmp_path / "auto-main.toml"
    out_ref = tmp_path / "auto-ref.toml"
    variant = _variant_for_extra(generator, extra)

    generator.main([*extra, LOAD_FORMAT_ARG_FLAG, LOAD_FORMAT_OPT_OUT, str(source), str(out_main)])
    generator.generate(source, out_ref, generator.without_load_format(variant))

    assert out_main.read_bytes() == out_ref.read_bytes()
    assert hashlib.sha256(out_main.read_bytes()).hexdigest() == auto_sha256
    assert set(load_configs(out_main, REPO_ROOT)) == {target}


@pytest.mark.parametrize(
    ("extra", "default_sha256", "target"),
    [
        pytest.param(("--variant", "full"), FULL_RENDER_SHA256, FULL_TARGET_NAME, id="full"),
        pytest.param(
            ("--variant", "full-mtp", "--spec-tokens", "2"),
            MTP2_RENDER_SHA256,
            mtp_target_name(2),
            id="full-mtp",
        ),
    ],
)
def test_main_default_load_format_reaches_the_output(
    generator: Any, tmp_path: Path, extra: tuple[str, ...], default_sha256: str, target: str
) -> None:
    """`--load-format` を付けない `main` の出力に、既定の instanttensor が入る (C1)。

    利用者の入口 (`main`) から、既定の `--load-format instanttensor` が出力ファイルに届くことを
    確かめる。`render` / `generate` を直接呼ぶ試験とは別に、CLI の既定の経路を固定する。
    """
    source = _write_inspect_json(tmp_path, _inspect_item())
    out_main = tmp_path / "default-main.toml"
    out_ref = tmp_path / "default-ref.toml"
    variant = _variant_for_extra(generator, extra)

    generator.main([*extra, str(source), str(out_main)])
    generator.generate(source, out_ref, variant)

    assert out_main.read_bytes() == out_ref.read_bytes()
    assert hashlib.sha256(out_main.read_bytes()).hexdigest() == default_sha256
    loaded = load_configs(out_main, REPO_ROOT)
    assert set(loaded) == {target}
    assert loaded[target].args[LOAD_FORMAT_ARG_KEY].value == DEFAULT_LOAD_FORMAT_VALUE


def test_load_format_auto_is_absent_from_the_args_and_the_names(
    generator: Any, tmp_path: Path
) -> None:
    """auto を外した出力に `--load-format` も `-lf-auto` も現れず、名前は変わらない (C2)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / "auto-mtp2.toml"

    generator.main(
        [
            "--variant",
            "full-mtp",
            "--spec-tokens",
            "2",
            LOAD_FORMAT_ARG_FLAG,
            LOAD_FORMAT_OPT_OUT,
            str(source),
            str(output),
        ]
    )

    rendered = output.read_text(encoding="utf-8")
    assert LOAD_FORMAT_ARG_FLAG not in rendered
    assert LOAD_FORMAT_OPT_OUT_SUFFIX not in rendered
    assert f"[configs.{mtp_target_name(2)}]" in rendered


def test_load_format_auto_does_not_change_smoke(generator: Any, tmp_path: Path) -> None:
    """`--load-format auto` は smoke では何も変えない (C2、#50 の設計判断)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    out_default = tmp_path / "smoke-default.toml"
    out_auto = tmp_path / "smoke-auto.toml"

    generator.main([str(source), str(out_default)])
    generator.main([LOAD_FORMAT_ARG_FLAG, LOAD_FORMAT_OPT_OUT, str(source), str(out_auto)])

    assert out_default.read_bytes() == out_auto.read_bytes()
    assert hashlib.sha256(out_auto.read_bytes()).hexdigest() == SMOKE_RENDER_SHA256


@pytest.mark.parametrize(
    ("kind", "expected", "expected_auto"),
    [
        pytest.param("full", FULL_RENDER_SHA256, FULL_AUTO_RENDER_SHA256, id="full"),
        pytest.param("mtp1", MTP1_RENDER_SHA256, MTP1_AUTO_RENDER_SHA256, id="mtp1"),
        pytest.param("mtp2", MTP2_RENDER_SHA256, MTP2_AUTO_RENDER_SHA256, id="mtp2"),
    ],
)
def test_default_load_format_output_is_derived_from_the_opt_out(
    generator: Any, tmp_path: Path, kind: str, expected: str, expected_auto: str
) -> None:
    """既定つきの出力は、opt-out の出力にテーブル 1 つを挿入したものとバイト単位で同じ (C3)。"""
    text, image = _render_default_input(generator, tmp_path)
    variant = _base_variant(generator, kind)

    actual = generator.render(text, image, variant)
    derived = derive_default_load_format(generator, text, image, variant)

    assert actual == derived
    assert hashlib.sha256(actual.encode("utf-8")).hexdigest() == expected
    assert (
        hashlib.sha256(
            generator.render(text, image, generator.without_load_format(variant)).encode("utf-8")
        ).hexdigest()
        == expected_auto
    )


def test_thread_names_default_load_format_output_is_derived_from_the_opt_out(
    generator: Any, tmp_path: Path
) -> None:
    """full-tn でも、既定つきの出力は opt-out にテーブル 1 つを挿入したものと同じ (C3)。"""
    text, image = _render_default_input(generator, tmp_path)
    variant = generator.with_thread_names(generator.FULL)

    actual = generator.render(text, image, variant)
    derived = derive_default_load_format(generator, text, image, variant)

    assert actual == derived
    assert hashlib.sha256(actual.encode("utf-8")).hexdigest() == FULL_TN_RENDER_SHA256
    assert (
        hashlib.sha256(
            generator.render(text, image, generator.without_load_format(variant)).encode("utf-8")
        ).hexdigest()
        == FULL_TN_AUTO_RENDER_SHA256
    )


def test_explicit_load_format_replaces_the_default(generator: Any, tmp_path: Path) -> None:
    """既定を持つ full に別の値を明示すると、既定を置き換えて 1 つだけ入る (C4)。"""
    text, image = _render_default_input(generator, tmp_path)
    variant = generator.with_load_format(generator.FULL, "fastsafetensors")

    rendered = generator.render(text, image, variant)

    assert variant.name == f"{FULL_TARGET_NAME}-lf-fastsafetensors"
    generated = tomllib.loads(rendered)["configs"][variant.name]
    assert generated["args"][LOAD_FORMAT_ARG_KEY]["value"] == "fastsafetensors"
    assert rendered.count(f'flag = "{LOAD_FORMAT_ARG_FLAG}"') == 1


def test_default_and_explicit_load_format_do_not_double_up_in_the_argv(
    generator: Any, tmp_path: Path
) -> None:
    """既定と明示が並んで 2 つ入らず、argv の値は明示で置き換わる (C4)。"""
    variant = generator.with_load_format(generator.mtp_variant(2), "runai_streamer")
    plans = _plans_for_variant(generator, tmp_path, variant, "default-lf-argv")

    for plan in plans:
        argv = list(plan.argv)
        assert argv.count(LOAD_FORMAT_ARG_FLAG) == 1
        assert argv[argv.index(LOAD_FORMAT_ARG_FLAG) + 1] == "runai_streamer"
        assert DEFAULT_LOAD_FORMAT_VALUE not in argv


def test_load_format_auto_with_load_strategy_has_no_load_format(
    generator: Any, tmp_path: Path
) -> None:
    """`--load-format auto` と `-sls prefetch` を組にすると、strategy だけ入る (C4)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / "auto-sls.toml"

    generator.main(
        [
            "--variant",
            "full",
            LOAD_FORMAT_ARG_FLAG,
            LOAD_FORMAT_OPT_OUT,
            LOAD_STRATEGY_ARG_FLAG,
            "prefetch",
            str(source),
            str(output),
        ]
    )

    name = f"{FULL_TARGET_NAME}{load_strategy_suffix('prefetch')}"
    loaded = load_configs(output, REPO_ROOT)
    assert set(loaded) == {name}
    assert LOAD_FORMAT_ARG_KEY not in loaded[name].args
    assert loaded[name].args[LOAD_STRATEGY_ARG_KEY].value == "prefetch"


@pytest.mark.parametrize(
    ("extra", "value"),
    [
        pytest.param(("--variant", "full"), "prefetch", id="full"),
        pytest.param(("--variant", "full-mtp", "--spec-tokens", "2"), "eager", id="full-mtp"),
    ],
)
def test_main_refuses_a_load_strategy_without_a_load_format_on_a_variant_with_a_default(
    generator: Any, tmp_path: Path, extra: tuple[str, ...], value: str
) -> None:
    """既定の `--load-format instanttensor` を持つ full / full-mtp で、`--load-format` を付けずに
    `--safetensors-load-strategy` だけを指定すると、生成の前に断り、出力ファイルを作らない (#50)。

    strategy が効くのは `--load-format` を付けない読み方だけである。既定を黙って外さず、
    `-sls-<値>` の名前が、strategy の効かない (load-format つきの) 中身を指す構成も作らない。
    `--load-format auto` と組にした同じ引数は受け付ける (下の、各 `--variant` に付ける試験)。
    """
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / "refused-load-strategy.toml"

    with pytest.raises(SystemExit):
        generator.main([*extra, LOAD_STRATEGY_ARG_FLAG, value, str(source), str(output)])

    assert not output.exists()


def test_main_accepts_an_explicit_load_format_with_a_load_strategy(
    generator: Any, tmp_path: Path
) -> None:
    """明示の `--load-format instanttensor` (既定と同じ値) と `--safetensors-load-strategy` の組は、
    これまでどおり受け付ける (#50)。

    明示の `--load-format` は既定を置き換えるので、名前は `-lf-<値>-sls-<値>` になり、
    args の末尾 2 鍵は load-format → strategy の順になる。
    """
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / "explicit-lf-sls.toml"
    name = (
        f"{FULL_TARGET_NAME}{load_format_suffix(DEFAULT_LOAD_FORMAT_VALUE)}"
        f"{load_strategy_suffix('prefetch')}"
    )

    generator.main(
        [
            "--variant",
            "full",
            LOAD_FORMAT_ARG_FLAG,
            DEFAULT_LOAD_FORMAT_VALUE,
            LOAD_STRATEGY_ARG_FLAG,
            "prefetch",
            str(source),
            str(output),
        ]
    )

    loaded = load_configs(output, REPO_ROOT)
    assert set(loaded) == {name}
    assert list(loaded[name].args)[-2:] == [LOAD_FORMAT_ARG_KEY, LOAD_STRATEGY_ARG_KEY]
    assert loaded[name].args[LOAD_FORMAT_ARG_KEY].value == DEFAULT_LOAD_FORMAT_VALUE
    assert loaded[name].args[LOAD_STRATEGY_ARG_KEY].value == "prefetch"


# --- 重みの読み込み方を選ぶ構成の生成 (C1、C2、C3、C4、C5、C6) -------------


def _plans_for_variant(
    generator: Any, tmp_path: Path, variant: Any, stem: str
) -> tuple[ContainerPlan, ...]:
    """`variant` の生成 TOML を、実物の `nodes.toml` で選び、2 台ぶんの計画を組み立てる。"""
    output = _generate_variant(generator, tmp_path, variant, stem)
    configs = load_configs(output, REPO_ROOT)
    nodes = load_nodes(NODES_PATH, REPO_ROOT)
    return build_plans(select_config(configs, variant.name, nodes), nodes, STARTED_AT)


@pytest.mark.parametrize("value", LOAD_FORMAT_VALUES)
@pytest.mark.parametrize("kind", PROFILER_BASE_KINDS)
def test_load_format_config_adds_only_the_arg_and_the_name(
    generator: Any, tmp_path: Path, kind: str, value: str
) -> None:
    """`--load-format` は、名前 (`-lf-<値>`) と args 1 つだけを足し、他は基のままにする (C1)。

    名前の接尾辞は `_` を `-` に替え、args の `value` は元の値のままである
    (`runai_streamer` は `-lf-runai-streamer`)。名前と説明以外は基のままだ。
    """
    base = _base_variant(generator, kind)
    loaded = generator.with_load_format(base, value)
    baseline = load_configs(
        _generate_variant(generator, tmp_path, base, f"base-lf-{kind}"), REPO_ROOT
    )[base.name]
    generated = load_configs(
        _generate_variant(generator, tmp_path, loaded, f"load-format-{kind}-{value}"), REPO_ROOT
    )

    assert loaded.name == f"{base.name}{load_format_suffix(value)}"
    assert set(generated) == {loaded.name}
    config = generated[loaded.name]

    added = config.args[LOAD_FORMAT_ARG_KEY]
    assert added.flag == LOAD_FORMAT_ARG_FLAG
    assert added.value == value
    rest = {key: setting for key, setting in config.args.items() if key != LOAD_FORMAT_ARG_KEY}
    baseline_rest = {
        key: setting for key, setting in baseline.args.items() if key != LOAD_FORMAT_ARG_KEY
    }
    assert rest == baseline_rest

    assert config.description != baseline.description
    for field in (
        "image",
        "weights",
        "docker",
        "env",
        "allow_speculative",
        "kind",
        "nodes",
        "ready_timeout_s",
        "served_model_name",
    ):
        assert getattr(config, field) == getattr(baseline, field), field


@pytest.mark.parametrize("value", LOAD_STRATEGY_VALUES)
@pytest.mark.parametrize("kind", PROFILER_BASE_KINDS)
def test_load_strategy_config_adds_only_the_arg_and_the_name(
    generator: Any, tmp_path: Path, kind: str, value: str
) -> None:
    """`--safetensors-load-strategy` は、名前 (`-sls-<値>`) と args 1 つだけを足す (C2)。

    strategy が効くのは `--load-format` を付けない読み方だけなので、基は既定の
    `--load-format instanttensor` を外した変種 (`main` でいう `--load-format auto`) にする。
    """
    base = generator.without_load_format(_base_variant(generator, kind))
    loaded = generator.with_load_strategy(base, value)
    baseline = load_configs(
        _generate_variant(generator, tmp_path, base, f"base-sls-{kind}"), REPO_ROOT
    )[base.name]
    generated = load_configs(
        _generate_variant(generator, tmp_path, loaded, f"load-strategy-{kind}-{value}"), REPO_ROOT
    )

    assert loaded.name == f"{base.name}{load_strategy_suffix(value)}"
    assert set(generated) == {loaded.name}
    config = generated[loaded.name]

    added = config.args[LOAD_STRATEGY_ARG_KEY]
    assert added.flag == LOAD_STRATEGY_ARG_FLAG
    assert added.value == value
    rest = {key: setting for key, setting in config.args.items() if key != LOAD_STRATEGY_ARG_KEY}
    assert rest == baseline.args

    assert config.description != baseline.description
    for field in (
        "image",
        "weights",
        "docker",
        "env",
        "allow_speculative",
        "kind",
        "nodes",
        "ready_timeout_s",
        "served_model_name",
    ):
        assert getattr(config, field) == getattr(baseline, field), field


@pytest.mark.parametrize("value", LOAD_FORMAT_VALUES)
@pytest.mark.parametrize("kind", PROFILER_BASE_KINDS)
def test_load_format_argv_matches_base_argv_plus_the_arg_pair(
    generator: Any, tmp_path: Path, kind: str, value: str
) -> None:
    """`--load-format` の列は、基の列の末尾の `--load-format` を明示の値に替えたもの (C1、C4)。

    基が既定を持つ (full / mtp) ときは、`--load-format <値>` の組の値だけを置き換える。
    基が既定を持たない (smoke) ときは、末尾に 2 語を足す。コンテナ名と `config` のラベル
    だけ置き換える。`config-sha256` は中身が変わるので、両者で違うのが正しいため比較から除く。
    """
    base = _base_variant(generator, kind)
    loaded = generator.with_load_format(base, value)
    base_plans = _plans_for_variant(generator, tmp_path, base, f"argv-base-lf-{kind}")
    loaded_plans = _plans_for_variant(
        generator, tmp_path, loaded, f"argv-load-format-{kind}-{value}"
    )

    for base_plan, loaded_plan in zip(base_plans, loaded_plans, strict=True):
        assert loaded_plan.container_name == f"vb-{loaded.name}-{loaded_plan.node}"
        expected = list(base_plan.argv)
        expected[expected.index(f"vb-{base.name}-{base_plan.node}")] = loaded_plan.container_name
        expected[expected.index(f"vllm-baseline.config={base.name}")] = (
            f"vllm-baseline.config={loaded.name}"
        )
        if LOAD_FORMAT_ARG_FLAG in expected:
            expected[expected.index(LOAD_FORMAT_ARG_FLAG) + 1] = value
        else:
            expected.extend((LOAD_FORMAT_ARG_FLAG, value))

        kept = [arg for arg in expected if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        actual = [arg for arg in loaded_plan.argv if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        assert kept == actual
        assert base_plan.labels[LABEL_CONFIG_SHA256] != loaded_plan.labels[LABEL_CONFIG_SHA256]


@pytest.mark.parametrize("value", LOAD_STRATEGY_VALUES)
@pytest.mark.parametrize("kind", PROFILER_BASE_KINDS)
def test_load_strategy_argv_matches_base_argv_plus_the_arg_pair(
    generator: Any, tmp_path: Path, kind: str, value: str
) -> None:
    """`--safetensors-load-strategy` の列は、基の列の末尾にその 2 語を足したもの (C2)。

    基は、既定の `--load-format instanttensor` を外した変種にする (strategy が効くのは
    `--load-format` を付けない読み方だけ)。
    """
    base = generator.without_load_format(_base_variant(generator, kind))
    loaded = generator.with_load_strategy(base, value)
    base_plans = _plans_for_variant(generator, tmp_path, base, f"argv-base-sls-{kind}")
    loaded_plans = _plans_for_variant(
        generator, tmp_path, loaded, f"argv-load-strategy-{kind}-{value}"
    )

    for base_plan, loaded_plan in zip(base_plans, loaded_plans, strict=True):
        assert loaded_plan.container_name == f"vb-{loaded.name}-{loaded_plan.node}"
        expected = list(base_plan.argv)
        expected[expected.index(f"vb-{base.name}-{base_plan.node}")] = loaded_plan.container_name
        expected[expected.index(f"vllm-baseline.config={base.name}")] = (
            f"vllm-baseline.config={loaded.name}"
        )
        expected.extend((LOAD_STRATEGY_ARG_FLAG, value))

        kept = [arg for arg in expected if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        actual = [arg for arg in loaded_plan.argv if not arg.startswith(f"{LABEL_CONFIG_SHA256}=")]
        assert kept == actual
        assert base_plan.labels[LABEL_CONFIG_SHA256] != loaded_plan.labels[LABEL_CONFIG_SHA256]


@pytest.mark.parametrize(
    ("option", "value", "source", "fragment"),
    [
        pytest.param(
            "format",
            "instanttensor",
            LOAD_CONFIG_SOURCE,
            LOAD_FORMAT_QUOTE_FRAGMENTS["instanttensor"],
            id="instanttensor",
        ),
        pytest.param(
            "format",
            "fastsafetensors",
            FASTSAFETENSORS_DOC_SOURCE,
            LOAD_FORMAT_QUOTE_FRAGMENTS["fastsafetensors"],
            id="fastsafetensors",
        ),
        pytest.param(
            "format",
            "runai_streamer",
            LOAD_CONFIG_SOURCE,
            LOAD_FORMAT_QUOTE_FRAGMENTS["runai_streamer"],
            id="runai_streamer",
        ),
        pytest.param(
            "strategy",
            "eager",
            LOAD_CONFIG_SOURCE,
            LOAD_STRATEGY_QUOTE_FRAGMENTS["eager"],
            id="eager",
        ),
        pytest.param(
            "strategy",
            "prefetch",
            LOAD_CONFIG_SOURCE,
            LOAD_STRATEGY_QUOTE_FRAGMENTS["prefetch"],
            id="prefetch",
        ),
    ],
)
def test_load_option_provenance_is_the_pinned_vllm_source(
    generator: Any, tmp_path: Path, option: str, value: str, source: str, fragment: str
) -> None:
    """足す args の根拠が、固定 commit の vLLM のソースか公式文書の原文である (C3)。

    `fastsafetensors` だけは docstring に無いので、公式文書を出典にする。
    """
    if option == "format":
        variant = generator.with_load_format(generator.FULL, value)
        key = LOAD_FORMAT_ARG_KEY
    else:
        variant = generator.with_load_strategy(generator.FULL, value)
        key = LOAD_STRATEGY_ARG_KEY
    generated = load_configs(
        _generate_variant(generator, tmp_path, variant, f"provenance-{option}-{value}"), REPO_ROOT
    )

    added = generated[variant.name].args[key]
    assert str(added.source) == source
    assert added.quote is not None
    assert fragment in added.quote


@pytest.mark.parametrize(
    ("extra", "option_flag", "value", "expected_name"),
    [
        pytest.param(
            (),
            LOAD_FORMAT_ARG_FLAG,
            "instanttensor",
            f"{TARGET_NAME}{load_format_suffix('instanttensor')}",
            id="smoke-load-format",
        ),
        pytest.param(
            (),
            LOAD_STRATEGY_ARG_FLAG,
            "prefetch",
            f"{TARGET_NAME}{load_strategy_suffix('prefetch')}",
            id="smoke-load-strategy",
        ),
        pytest.param(
            ("--variant", "full", LOAD_FORMAT_ARG_FLAG, LOAD_FORMAT_OPT_OUT),
            LOAD_STRATEGY_ARG_FLAG,
            "prefetch",
            f"{FULL_TARGET_NAME}{load_strategy_suffix('prefetch')}",
            id="full-load-strategy",
        ),
        pytest.param(
            ("--variant", "full-mtp", "--spec-tokens", "2"),
            LOAD_FORMAT_ARG_FLAG,
            "runai_streamer",
            f"{mtp_target_name(2)}{load_format_suffix('runai_streamer')}",
            id="full-mtp-load-format",
        ),
        pytest.param(
            (
                "--variant",
                "full-mtp",
                "--spec-tokens",
                "2",
                LOAD_FORMAT_ARG_FLAG,
                LOAD_FORMAT_OPT_OUT,
            ),
            LOAD_STRATEGY_ARG_FLAG,
            "eager",
            f"{mtp_target_name(2)}{load_strategy_suffix('eager')}",
            id="full-mtp-load-strategy",
        ),
    ],
)
def test_main_accepts_load_options_with_each_variant(
    generator: Any,
    tmp_path: Path,
    extra: tuple[str, ...],
    option_flag: str,
    value: str,
    expected_name: str,
) -> None:
    """`--load-format` はどの `--variant` にも付けられる。`--safetensors-load-strategy` は、
    既定を持たない smoke には単独で、既定を持つ full / full-mtp には `--load-format auto`
    と組で付けられ、args に strategy だけが入る (C4、#50)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / f"{expected_name}.toml"

    generator.main([*extra, option_flag, value, str(source), str(output)])

    loaded = load_configs(output, REPO_ROOT)
    assert set(loaded) == {expected_name}
    key = LOAD_FORMAT_ARG_KEY if option_flag == LOAD_FORMAT_ARG_FLAG else LOAD_STRATEGY_ARG_KEY
    assert loaded[expected_name].args[key].value == value
    if option_flag == LOAD_STRATEGY_ARG_FLAG:
        assert LOAD_FORMAT_ARG_KEY not in loaded[expected_name].args


def test_combined_load_options_names_and_arg_order(generator: Any, tmp_path: Path) -> None:
    """すべての旗を重ねた名前と args の順が固定で、旗の並びに依らない (C4)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    first = tmp_path / "combined-first.toml"
    second = tmp_path / "combined-second.toml"
    name = "p2-nope-tp2-mtp2-tn-prof-lf-runai-streamer-sls-prefetch"

    generator.main(
        [
            "--variant",
            "full-mtp",
            "--spec-tokens",
            "2",
            "--nccl-thread-names",
            "--torch-profiler",
            LOAD_FORMAT_ARG_FLAG,
            "runai_streamer",
            LOAD_STRATEGY_ARG_FLAG,
            "prefetch",
            str(source),
            str(first),
        ]
    )
    generator.main(
        [
            LOAD_STRATEGY_ARG_FLAG,
            "prefetch",
            LOAD_FORMAT_ARG_FLAG,
            "runai_streamer",
            "--torch-profiler",
            "--nccl-thread-names",
            "--spec-tokens",
            "2",
            "--variant",
            "full-mtp",
            str(source),
            str(second),
        ]
    )

    assert first.read_bytes() == second.read_bytes()
    loaded = load_configs(first, REPO_ROOT)
    assert set(loaded) == {name}
    assert list(loaded[name].args)[-4:] == [
        "speculative-config",
        PROFILER_ARG_KEY,
        LOAD_FORMAT_ARG_KEY,
        LOAD_STRATEGY_ARG_KEY,
    ]

    nodes = load_nodes(NODES_PATH, REPO_ROOT)
    plans = build_plans(select_config(loaded, name, nodes), nodes, STARTED_AT)
    assert [plan.node for plan in plans] == ["head", "worker"]
    assert plans[0].container_name == f"vb-{name}-head"
    assert plans[1].container_name == f"vb-{name}-worker"


def test_load_strategy_prefetch_is_appended_after_the_speculative_arg(
    generator: Any, tmp_path: Path
) -> None:
    """既定を外した mtp2 に `prefetch` を足すと、末尾 2 鍵が投機 → strategy の順になる (C4)。

    strategy が効くのは `--load-format` を付けない読み方だけなので、基は既定を外した変種に
    する。load-format の鍵はない。
    """
    variant = generator.with_load_strategy(
        generator.without_load_format(generator.mtp_variant(2)), "prefetch"
    )
    generated = load_configs(
        _generate_variant(generator, tmp_path, variant, "sls-after-spec"), REPO_ROOT
    )

    assert variant.name == f"{mtp_target_name(2)}{load_strategy_suffix('prefetch')}"
    assert list(generated[variant.name].args)[-2:] == [
        "speculative-config",
        LOAD_STRATEGY_ARG_KEY,
    ]
    assert LOAD_FORMAT_ARG_KEY not in generated[variant.name].args


def test_load_options_output_without_the_flags_is_unchanged(generator: Any, tmp_path: Path) -> None:
    """既定を外した 5 形の出力が、変更前の固定値と一致し、読み込み方の印が現れない (C2、C3)。"""
    text, image = _render_default_input(generator, tmp_path)
    cases = (
        ("smoke", generator.SMOKE, SMOKE_RENDER_SHA256),
        ("full", generator.without_load_format(generator.FULL), FULL_AUTO_RENDER_SHA256),
        (
            "full-tn",
            generator.without_load_format(generator.with_thread_names(generator.FULL)),
            FULL_TN_AUTO_RENDER_SHA256,
        ),
        (
            "mtp1",
            generator.without_load_format(generator.mtp_variant(1)),
            MTP1_AUTO_RENDER_SHA256,
        ),
        (
            "mtp2",
            generator.without_load_format(generator.mtp_variant(2)),
            MTP2_AUTO_RENDER_SHA256,
        ),
    )

    for label, variant, expected in cases:
        rendered = generator.render(text, image, variant)
        assert LOAD_FORMAT_ARG_FLAG not in rendered, label
        assert LOAD_STRATEGY_ARG_FLAG not in rendered, label
        assert LOAD_FORMAT_SUFFIX not in rendered, label
        assert LOAD_STRATEGY_SUFFIX not in rendered, label
        assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == expected, label

    existing = load_configs(CONFIGS_PATH, REPO_ROOT)
    assert f"{FULL_TARGET_NAME}{load_format_suffix('instanttensor')}" not in existing
    assert f"{FULL_TARGET_NAME}{load_strategy_suffix('prefetch')}" not in existing


@pytest.mark.parametrize("bad", ["safetensors", "dummy", ""])
def test_main_refuses_a_bad_load_format(generator: Any, tmp_path: Path, bad: str) -> None:
    """許さない `--load-format` の値は、生成の前に断り、出力ファイルを作らない (C5)。

    `auto` は既定を外す値として受けるので、ここには含めない。
    """
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / "bad-load-format.toml"

    with pytest.raises(SystemExit):
        generator.main([LOAD_FORMAT_ARG_FLAG, bad, str(source), str(output)])

    assert not output.exists()


@pytest.mark.parametrize("bad", ["lazy", "torchao", ""])
def test_main_refuses_a_bad_load_strategy(generator: Any, tmp_path: Path, bad: str) -> None:
    """許さない `--safetensors-load-strategy` の値は、生成の前に断る (C6)。"""
    source = _write_inspect_json(tmp_path, _inspect_item())
    output = tmp_path / "bad-load-strategy.toml"

    with pytest.raises(SystemExit):
        generator.main([LOAD_STRATEGY_ARG_FLAG, bad, str(source), str(output)])

    assert not output.exists()


def test_with_load_format_refuses_a_bad_value(generator: Any) -> None:
    """`with_load_format` を直接呼んでも、許さない値は `ValueError` で断る (C6)。"""
    with pytest.raises(ValueError):
        generator.with_load_format(generator.FULL, "auto")


def test_with_load_strategy_refuses_a_bad_value(generator: Any) -> None:
    """`with_load_strategy` を直接呼んでも、許さない値は `ValueError` で断る (C6)。"""
    with pytest.raises(ValueError):
        generator.with_load_strategy(generator.FULL, "lazy")


@pytest.mark.parametrize(
    ("key", "flag", "value"),
    [
        pytest.param(LOAD_FORMAT_ARG_KEY, LOAD_FORMAT_ARG_FLAG, "instanttensor", id="load-format"),
        pytest.param(LOAD_STRATEGY_ARG_KEY, LOAD_STRATEGY_ARG_FLAG, "prefetch", id="load-strategy"),
    ],
)
def test_load_option_render_leaves_a_decoy_alone(
    generator: Any, tmp_path: Path, key: str, flag: str, value: str
) -> None:
    """同じ字面の value の囮テーブルがあっても、足す args は 1 つだけである (C1、C2)。"""
    text_with_decoy = _insert_decoy_table(CONFIGS_PATH.read_text(encoding="utf-8"))
    image = generator.load_image(_write_inspect_json(tmp_path, _inspect_item()))

    if key == LOAD_FORMAT_ARG_KEY:
        variant = generator.with_load_format(generator.FULL, value)
    else:
        variant = generator.with_load_strategy(generator.FULL, value)
    rendered = generator.render(text_with_decoy, image, variant)

    generated = tomllib.loads(rendered)["configs"][variant.name]
    assert generated["args"]["decoy"]["value"] == "16"
    assert generated["args"]["max-model-len"]["value"] == "163840"
    assert generated["args"]["max-num-seqs"]["value"] == "16"
    assert generated["args"][key]["flag"] == flag
    assert rendered.count(f'flag = "{flag}"') == 1
    assert "[configs.p1-nvfp4-tp2" not in rendered
    assert "p1-fetch-nvfp4" not in rendered
