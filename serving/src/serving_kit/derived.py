"""変換の道具の manifest を、派生の重みのマニフェストに取り込む (`serve derived-import`)。

変換の道具 (`experiments/k2-quant`) は、変換の結果の置き場所に `manifest.json` を書く。この
module は、それを Mac で読み、serving が読む派生のマニフェスト (`types.DerivedWeightsManifest`)
に変える (issue #64)。Spark にも Hub にも触らない (ファイルと値だけを扱う)。

## 道具の manifest の形 (道具の README §3)

`conversion` (`tool`・`tool_version`・`source{repo, revision}`・`pattern`・`args`・`modules`・
dtype)、`files` (`path`・`size`・`sha256`)、`total_bytes`。道具が**書かないもの**は 3 つある。
`generated_at` は、出力を決定的にするために書かない。`commit` と、道具のリポジトリの中の道筋
(`derivation.conversion.tool`) は、コンテナの中の道具の写しからは分からず、Mac の git の文脈
でしか決まらないので、`serve derived-import` の引数で受ける。

## 意図した決めごと

1. **`args` と `target_pattern` は、入力から作る**。`derivation.conversion.args` は入力の
   `conversion.args` そのまま、`target_pattern` は `conversion.pattern` である。この module も
   `cli` も、引数の値を持たない (手で書かない)。`conversion.args` のない入力 (実際の引数の記録が
   ない古い形) は、補わずに断る
2. **形の誤りは `WeightsRefError`** で、入力の道筋を文に含める (終了コード 1)。読めない、JSON で
   ない、`conversion.tool` が `k2-quant` でない、形が違うもののすべてである
3. **2 台の一致は、最上位の `generated_at` を除いた中身**で見る。違う箇所は、入力の名前と
   `files[2].sha256` のような場所と、両方の値を挙げる。先頭の入力を基準に、残りのすべてを比べる
4. **Hub のマニフェストと同じ除外規則** (`weights.exclusion_reason`: モデルカードらしい名前と
   `.gitattributes`) を掛ける。道具は通常ファイルをすべて写すので、除いたものは
   `ImportResult.excluded_paths` に残して、計測者が見られるようにする

依存の向きは `types → … → weights → derived → cli` の一方向である (`weights` は、この module を
読み込まない)。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from serving_kit.types import (
    ConversionSpec,
    Derivation,
    DerivedWeightsManifest,
    ManifestFile,
    WeightsOrigin,
)
from serving_kit.weights import WeightsRefError, exclusion_reason

__all__ = [
    "DEFAULT_TOOL_PATH",
    "TOOL_NAME",
    "ImportResult",
    "ToolManifest",
    "compare_tool_manifests",
    "import_tool_manifest",
    "read_tool_manifest",
]

TOOL_NAME: Final = "k2-quant"
"""道具の manifest の `conversion.tool` の値 (`k2_quant.TOOL_NAME`)。この値だけを受ける。"""

DEFAULT_TOOL_PATH: Final = "experiments/k2-quant"
"""派生のマニフェストの `derivation.conversion.tool` の既定 (道具のリポジトリの中の道筋)。"""


@dataclass(frozen=True)
class ToolManifest:
    """読み込んだ道具の manifest。

    `content` は、最上位の `generated_at` を除いた、JSON の中身そのもの (2 台の比較に使う)。
    それ以外は、変換に使う項目を、型を確かめて取り出したものである。
    """

    source: Path
    content: Mapping[str, Any]
    source_repo: str
    source_revision: str
    pattern: str
    args: tuple[str, ...]
    files: tuple[ManifestFile, ...]


@dataclass(frozen=True)
class ImportResult:
    """`import_tool_manifest` の結果。`excluded_paths` は、除外規則で除いた道筋 (`path` の順)。"""

    manifest: DerivedWeightsManifest
    excluded_paths: tuple[str, ...]


def _refuse(path: Path, reason: str) -> WeightsRefError:
    return WeightsRefError(f"道具の manifest '{path}' を取り込めない: {reason}")


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def read_tool_manifest(path: Path) -> ToolManifest:
    """道具の `manifest.json` を読み、取り込みに要る形を確かめる。

    例外:
        WeightsRefError: 読めない、JSON でない、最上位や `conversion` の形が違う、
            `conversion.tool` が `k2-quant` でない、`conversion.args` がない、`files` の行が
            マニフェストの行にできない、`total_bytes` が整数でない (終了コード 1)。
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise _refuse(path, f"読めない ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise _refuse(path, f"JSON でない ({exc})") from exc
    if not isinstance(raw, dict):
        raise _refuse(path, "最上位が JSON のオブジェクトでない")
    conversion = raw.get("conversion")
    if not isinstance(conversion, dict):
        raise _refuse(path, "conversion がない")
    if conversion.get("tool") != TOOL_NAME:
        raise _refuse(
            path,
            f"この取り込みは {TOOL_NAME} の manifest だけを受ける"
            f" (conversion.tool: {conversion.get('tool')!r})",
        )
    source = conversion.get("source")
    if not isinstance(source, dict):
        raise _refuse(path, "conversion.source がない")
    repo = _text(source.get("repo"))
    revision = _text(source.get("revision"))
    pattern = _text(conversion.get("pattern"))
    if repo is None or revision is None or pattern is None:
        raise _refuse(
            path,
            "conversion.source.repo・conversion.source.revision・conversion.pattern が文字列でない",
        )
    args = conversion.get("args")
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise _refuse(
            path,
            "conversion.args (道具に渡した実際の引数の記録) がない。"
            "記録のある版 (0.2.0 以降) の道具で、変換し直す",
        )
    rows = raw.get("files")
    total = raw.get("total_bytes")
    if not isinstance(rows, list) or type(total) is not int:
        raise _refuse(path, "files (配列) か total_bytes (整数) の形が違う")
    try:
        files = tuple(
            ManifestFile(path=row["path"], size=row["size"], sha256=row["sha256"]) for row in rows
        )
    except (ValidationError, KeyError, TypeError) as exc:
        raise _refuse(path, f"files の行が path・size・sha256 の形でない ({exc})") from exc
    content = {key: value for key, value in raw.items() if key != "generated_at"}
    return ToolManifest(
        source=path,
        content=content,
        source_repo=repo,
        source_revision=revision,
        pattern=pattern,
        args=tuple(args),
        files=files,
    )


def _differences(place: str, first: Any, other: Any) -> list[str]:
    """JSON の 2 つの値を再帰的に比べ、違う場所を `place: 値 != 値` の形で並べる。"""
    if isinstance(first, dict) and isinstance(other, dict):
        found: list[str] = []
        for key in sorted(set(first) | set(other)):
            here = f"{place}.{key}" if place else str(key)
            if key not in first or key not in other:
                side = "先頭の入力にしかない" if key in first else "この入力にしかない"
                found.append(f"{here}: {side}")
            else:
                found.extend(_differences(here, first[key], other[key]))
        return found
    if isinstance(first, list) and isinstance(other, list):
        found = []
        for index in range(max(len(first), len(other))):
            here = f"{place}[{index}]"
            if index >= len(first) or index >= len(other):
                side = "先頭の入力にしかない" if index < len(first) else "この入力にしかない"
                found.append(f"{here}: {side}")
            else:
                found.extend(_differences(here, first[index], other[index]))
        return found
    if type(first) is type(other) and first == other:
        return []
    shown_first = json.dumps(first, ensure_ascii=False)
    shown_other = json.dumps(other, ensure_ascii=False)
    return [f"{place}: {shown_first} != {shown_other}"]


def compare_tool_manifests(manifests: Sequence[ToolManifest]) -> tuple[str, ...]:
    """2 台ぶん以上の manifest が、最上位の `generated_at` を除いて同じかを見る。

    先頭の入力を基準に、残りのすべてを比べる。返り値は、違う箇所の一覧
    (`<入力の名前>: files[2].sha256: <値> != <値>` の形。空なら一致)。鍵が片方にしかない、
    配列の長さが違う、型が違うことも、違う箇所に数える。
    """
    first, *others = manifests
    return tuple(
        f"{other.source.name}: {difference}"
        for other in others
        for difference in _differences("", dict(first.content), dict(other.content))
    )


def import_tool_manifest(
    tool: ToolManifest,
    *,
    name: str,
    tool_path: str,
    commit: str,
    generated_at: datetime,
) -> ImportResult:
    """道具の manifest を、派生の重みのマニフェストにする。

    `derivation.conversion.args` は `tool.args`、`target_pattern` は `tool.pattern`、`origin` は
    `tool.source_repo` と `tool.source_revision` から作る。`generated_at` は、呼ぶ側が渡す
    (この関数の中で時計を読まない)。

    例外:
        ValueError: `generated_at` が tz-aware な `datetime` でないとき。
        WeightsRefError: 名前・道具の道筋・commit・元の重みの版などが、派生のマニフェストの
            形に合わない、除いたあとに載せるファイルが 1 つもない (終了コード 1)。
    """
    if generated_at.tzinfo is None:
        raise ValueError("generated_at は UTC の tz-aware な datetime にする (types.py の決まり)")
    kept = sorted(
        (entry for entry in tool.files if exclusion_reason(entry.path) is None),
        key=lambda entry: entry.path,
    )
    excluded = sorted(
        entry.path for entry in tool.files if exclusion_reason(entry.path) is not None
    )
    try:
        manifest = DerivedWeightsManifest(
            kind="derived",
            derivation=Derivation(
                name=name,
                origin=WeightsOrigin(repo=tool.source_repo, revision=tool.source_revision),
                conversion=ConversionSpec(
                    tool=tool_path,
                    commit=commit,
                    args=tool.args,
                    target_pattern=tool.pattern,
                ),
            ),
            generated_at=generated_at,
            total_bytes=sum(entry.size for entry in kept),
            files=tuple(kept),
        )
    except ValidationError as exc:
        raise WeightsRefError(
            f"派生のマニフェスト '{name}' を組み立てられない ({tool.source}): {exc}"
        ) from exc
    return ImportResult(manifest=manifest, excluded_paths=tuple(excluded))
