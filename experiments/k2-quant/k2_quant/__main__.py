"""変換の道具の入口。

入力の重みのディレクトリ (safetensors の shard、`model.safetensors.index.json`、
`config.json`) から、対象のテンソルを FP8 E4M3 か NVFP4A16 (重みだけ NVFP4。Issue #95) に
した新しいディレクトリを作る。CPU だけで動き、GPU とネットワークは使わない。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from k2_quant.convert import ConversionError, execute_plan, plan_conversion
from k2_quant.presets import DEFAULT_PRESET, FORMAT_NAMES, PRESET_NAMES
from k2_quant.selection import DEFAULT_PATTERN


def _build_parser() -> argparse.ArgumentParser:
    # 長い正規表現を折り返さずに出す (そのままコピーして `--pattern` に使えるように)
    parser = argparse.ArgumentParser(
        prog="k2_quant", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="今の重みのディレクトリ (config.json、model.safetensors.index.json、shard)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "新しいディレクトリ、または存在する空のディレクトリ (bind mount の宛先)。"
            "中身があれば何も書かずに終了 1"
        ),
    )
    parser.add_argument(
        "--source-repo",
        required=True,
        help="元の repo (manifest の変換条件に書く。例: RedHatAI/GLM-5.3-Flash-NVFP4)",
    )
    parser.add_argument(
        "--source-revision",
        required=True,
        help="元の revision (manifest の変換条件に書く)",
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        # 既定は解析のあとで決める。Python 3.12 の argparse は、既定と同じ値を明示しても
        # 相互排他の違反とみなさないため (`--preset k2s1 --pattern ...` が通ってしまう)
        default=None,
        help=(
            "対象の選び方の名前。k2s1 は第 1 段の既定 (下の --pattern の既定と同じ)。k2s2a は\n"
            "第 1 段の対象 + MLA の射影 + KDA のまとめていない射影 + lm_head (MLA と KDA の層は\n"
            "入力 config.json の text_config.layer_types から決め、無ければ何も書かずに終了 1)。\n"
            "k2s2b は k2s2a の対象 + KDA のまとめた層 (q・k・v・b・f_a・g_a。組を丸ごと選ぶ)。\n"
            "k2s2b の重みを vLLM で読むには、#79 の重ね合わせ (k2s2b) が前提。\n"
            "k2s3 は k2s2b と同じ対象を NVFP4A16 (重みだけ NVFP4) にする (Issue #95。読むには\n"
            "#79 の重ね合わせ k2s2b が前提で未確認)。\n"
            "--pattern と同時には使えない。manifest には、解決後の正規表現を --pattern として書く\n"
            f"(既定: {DEFAULT_PRESET})"
        ),
    )
    selection_group.add_argument(
        "--pattern",
        default=None,
        help=(
            "入力 checkpoint のモジュール名 (`.weight` を除いた名前) に `re.match` で当てて、\n"
            "変換するモジュールを選ぶ正規表現。config.json の target は、選ばれたモジュール\n"
            "から作る (この正規表現は書かない)。\n"
            "vLLM で 1 つの線形層にまとまる組 (q_a_proj と kv_a_proj_with_mqa、\n"
            "gate_proj と up_proj、KDA の q・k・v・b・f_a・g_a) の一部だけが選ばれると、\n"
            "何も書かずに終了 1。\n"
            "指定しないときは --preset の選び方を使う。--preset k2s1 (既定) の正規表現は、\n"
            "第 1 段の範囲のうち dense と共有の専門家。vLLM が FP8 を読めない部分\n"
            "(MTP の eh_proj は quant_config を受けない plain nn.Linear、ParallelLMHead は\n"
            "FP8 (W8A16) を読めない (#68)) は既定から外す:\n"
            f"{DEFAULT_PATTERN}"
        ),
    )
    parser.add_argument(
        "--format",
        choices=FORMAT_NAMES,
        default=None,
        help=(
            "数値形式。fp8 (既定) は FP8 E4M3 (重みだけ、出力チャネルごとの対称スケール)。\n"
            "nvfp4a16 は NVFP4A16 (重みだけ NVFP4。Issue #95)。--pattern と組み合わせて選ぶ。\n"
            "--preset とは同時に使えない (--preset は preset ごとの既定の形式を使う)。\n"
            "manifest の args には、nvfp4a16 のときだけ --format を書く"
        ),
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help=(
            "対象を含まない shard と通常ファイルを、写さずにハードリンクにする"
            " (入力と同じファイルシステムのときだけ。失敗したら写さずに終了 1)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.preset is not None and args.format is not None:
        parser.error("argument --format: not allowed with argument --preset")
    try:
        preset = DEFAULT_PRESET if args.preset is None else args.preset
        plan = plan_conversion(
            args.source, pattern=args.pattern, preset=preset, weight_format=args.format
        )
        execute_plan(
            plan,
            args.output,
            link=args.link,
            source_repo=args.source_repo,
            source_revision=args.source_revision,
        )
    except (ConversionError, ValueError, OSError) as exc:
        # SelectionError と ConfigError は ValueError の派生
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"converted modules: {len(plan.modules)}")
    for module in plan.modules:
        print(f"  {module}")
    rewritten = [shard.name for shard in plan.shards if shard.converted]
    untouched = [shard.name for shard in plan.shards if not shard.converted]
    print(f"rewritten shards: {len(rewritten)}")
    for name in rewritten:
        print(f"  {name}")
    verb = "linked" if args.link else "copied"
    print(f"{verb} shards: {len(untouched)}")
    for name in untouched:
        print(f"  {name}")
    print(f"{verb} files: {len(plan.other_files)}")
    for name in plan.other_files:
        print(f"  {name}")
    print(f"skipped (directories and hidden entries): {len(plan.skipped)}")
    for name in plan.skipped:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
