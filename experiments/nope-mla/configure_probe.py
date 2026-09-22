"""ビルド後の image inspect の結果から、既存の probe-nightly の派生構成を作る。"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inspect_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    image = json.loads(args.inspect_json.read_text())
    if isinstance(image, list) and len(image) == 1:
        image = image[0]
    image_id = image["Id"]
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise ValueError("完全なイメージ ID が必要")
    if image["Architecture"] != "arm64" or image["Os"] != "linux":
        raise ValueError("linux/arm64 のイメージが必要")
    if type(image["Size"]) is not int or image["Size"] <= 0:
        raise ValueError("イメージの大きさが不正")
    text = (ROOT / "serving/config/configs.toml").read_text()
    start = text.index("[configs.probe-nightly]")
    end = text.index("[configs.p1-nvfp4-tp2]")
    block = text[start:end]
    # コメントと既存の設定根拠を保ち、イメージの節だけを実測値に置き換える。
    image_start = block.index("[configs.probe-nightly.image]")
    image_end = block.index("[configs.probe-nightly.weights]")
    block = (
        block[:image_start]
        + (
            "[configs.probe-nightly.image]\n"
            f"ref = {json.dumps(image_id)}\n"
            'seen_as = "vllm-nope:0961bbae-fi070"\n'
            f"size_bytes = {image['Size']}\n"
            'measured = "docs/vllm-baseline/patched-build-procedure.md"\n\n'
        )
        + block[image_end:]
    )
    block = block.replace("configs.probe-nightly", "configs.probe-nope-local")
    block = re.sub(
        r"^description = .*$",
        'description = "NoPE 修正イメージの 1 台縮小検証"',
        block,
        count=1,
        flags=re.M,
    )
    result = "schema_version = 1\n\n" + block
    tomllib.loads(result)
    with args.output.open("x") as stream:
        stream.write(result)


if __name__ == "__main__":
    main()
