"""確認済みの自前イメージから TP=2 の構成を生成する (初回確認用の smoke と計測用の full)。"""

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


class Variant(NamedTuple):
    """生成する構成の違いをまとめたもの。

    `arg_overrides` は `(args の鍵, 旧値, 新値, why)` の並びで、`render` が
    `[configs.{SOURCE_NAME}.args.{鍵}]` の `value` と `why` をこの順に置き換える。
    空なら置き換えず、p1 の値と根拠をそのまま残す。
    """

    name: str
    description: str
    env_comment: str
    nccl_debug_why: str
    arg_overrides: tuple[tuple[str, str, str, str], ...]


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
        choices=tuple(VARIANTS),
        default="smoke",
        help="生成する構成 (smoke: 初回確認用 4096/1、full: 計測用 163840/16)",
    )
    parser.add_argument("inspect_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    generate(args.inspect_json, args.output, VARIANTS[args.variant])


if __name__ == "__main__":
    main()
