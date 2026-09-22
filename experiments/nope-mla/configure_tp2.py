"""確認済みの自前イメージから TP=2 の縮小構成を生成する。"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_PATH = ROOT / "serving/config/configs.toml"
EXPECTED_IMAGE_ID = "sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90"
IMAGE_SEEN_AS = "vllm-nope:0961bbae-fi070"
IMAGE_MEASURED = "docs/results/2026-09-22-nope-build.md"
SOURCE_NAME = "p1-nvfp4-tp2"
TARGET_NAME = "p2-nope-tp2-smoke"
NCCL_SOURCE_NAME = "netcheck-bandwidth"


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


def render(configs_text: str, image: Mapping[str, Any]) -> str:
    """configs.toml の p1 構成から生成する TOML を返す。"""
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
    block = _replace_in_table(
        block,
        f"[configs.{SOURCE_NAME}.args.max-model-len]",
        "163840",
        "4096",
        (
            "初回は起動と短い応答だけを確かめる運用上の選択。"
            "文脈長の上限を 4096 に絞り、初回に必要な KV 容量の条件を抑える。"
            "長文脈は次の段階で確かめる"
        ),
    )
    block = _replace_in_table(
        block,
        f"[configs.{SOURCE_NAME}.args.max-num-seqs]",
        "16",
        "1",
        (
            "初回は同時に 1 本しか送らない運用上の選択。"
            "線形アテンションの状態は同時の本数に比例するので、起動と応答の確認に要らないぶんを取らない。"
            "同時実行は次の段階で確かめる"
        ),
    )
    description = (
        "NoPE 修正イメージ (vllm-nope:0961bbae-fi070) で、"
        "2 台 TP=2 の初回の起動と短い応答だけを確かめる。"
        "短い文脈 (4096) と同時実行 1。実重みでの起動は未確認"
    )
    block = block.replace(f"configs.{SOURCE_NAME}", f"configs.{TARGET_NAME}")
    block = re.sub(
        r"^description = .*$",
        f"description = {json.dumps(description, ensure_ascii=False)}",
        block,
        count=1,
        flags=re.MULTILINE,
    )
    block = block.replace(
        "# --- 環境変数 (最初の 3 つだけ。A/B で足すものは docs/results/ の実測を根拠にする) -----",
        "# --- 環境変数 (p1-nvfp4-tp2 の 3 つ + 初回の起動だけに付ける NCCL の記録の 3 つ) -----",
    )
    data = tomllib.loads(configs_text)
    source_env = data["configs"][NCCL_SOURCE_NAME]["env"]
    additions = (
        (
            "nccl-debug",
            "NCCL_DEBUG",
            "INFO",
            (
                "初回の起動だけ、NCCL の記録を採る (手順書 2.1)。"
                "経路の確定の行 (Using network …) は INFO の段でしか出ない。"
                "2 台とも IB であることを、この記録から確かめる"
            ),
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
            f"\n[configs.{TARGET_NAME}.env.{key}]\n"
            f"flag = {json.dumps(flag)}\n"
            f"value = {json.dumps(value)}\n"
            f"why = {json.dumps(why, ensure_ascii=False)}\n"
            f"source = {json.dumps(origin['source'])}\n"
            f"quote = {json.dumps(origin['quote'], ensure_ascii=False)}\n"
        )
    result = "schema_version = 1\n\n" + block
    tomllib.loads(result)
    return result


def generate(inspect_json: Path, output: Path) -> None:
    """inspect JSON から構成を生成し、既存の出力を保護して保存する。"""
    result = render(CONFIGS_PATH.read_text(encoding="utf-8"), load_image(inspect_json))
    with output.open("x", encoding="utf-8") as stream:
        stream.write(result)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inspect_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    generate(args.inspect_json, args.output)


if __name__ == "__main__":
    main()
