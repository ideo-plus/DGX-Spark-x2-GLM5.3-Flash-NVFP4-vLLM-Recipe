"""固定した vLLM の直したファイルの写し (vllm-overlay) を、パッチから作り、確かめる。

ADR 0007 の第 2a 段 (#73)。固定した vLLM (0961bbae) の Python のファイルを直した写しを
`serving/payload/vllm-overlay/<名前>/<道筋>` に置き、`serve push` で配って、読み取り専用の
bind mount でイメージの中の同じファイルに重ねる。写しは「固定のソース + パッチ」でなければ
ならないので、パッチ (`patches/<名前>/<道筋>.patch`) と、3 つの SHA-256 (ソース、写し、パッチ)
を定義 (`<名前>.json`) に固定する。

- `refresh <名前>`: 固定のソースと、手で直した写しの差から、パッチを書き、定義の SHA-256 を
  書き換える。固定のソースの木は読むだけで、書き換えない。
- `check <名前>`: 写しとパッチの SHA-256 を定義と比べる。固定のソースがあれば、ソースの
  SHA-256 と、パッチを当てた結果が写しとバイト単位で一致することまで見る。

固定のソース (`serving/var/nope-build-0961bbae/source`) は Git 対象外で、CI にはない。
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[2]
OVERLAY_DIR = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "serving/var/nope-build-0961bbae/source"
PAYLOAD_DIR = ROOT / "serving/payload/vllm-overlay"
VLLM_COMMIT = "0961bbae2894d574be790d219651824eb199318e"
LICENSE_HEADER = (
    "# SPDX-License-Identifier: Apache-2.0",
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project",
)
"""上流の全ファイルの先頭の 2 行。写しでも変えない。"""

_SHA256_KEYS = ("source_sha256", "patched_sha256", "patch_sha256")
_HEX_DIGITS = frozenset("0123456789abcdef")


class OverlayFile(NamedTuple):
    """直した 1 つのファイル。`path` は、上流の木の中の `vllm/` で始まる相対の道筋。"""

    path: str
    patch: str
    source_sha256: str
    patched_sha256: str
    patch_sha256: str
    why: str


class Overlay(NamedTuple):
    """1 つの重ね (定義 `<名前>.json`)。`files` の並びが、mount の並びになる。"""

    name: str
    vllm_commit: str
    files: tuple[OverlayFile, ...]


def _check_relative_path(path: str) -> None:
    parts = path.split("/")
    if path.startswith("/") or parts[0] != "vllm" or ".." in parts or "" in parts:
        raise ValueError(f"道筋が vllm/ で始まる相対の道筋ではない: {path!r}")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and set(value) <= _HEX_DIGITS


def _parse_file(entry: object, name: str) -> OverlayFile:
    if not isinstance(entry, dict):
        raise ValueError(f"files の項目が object ではない: {entry!r}")
    fields: dict[str, str] = {}
    for key in OverlayFile._fields:
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"files の項目の {key} が空でない文字列ではない: {entry!r}")
        fields[key] = value
    _check_relative_path(fields["path"])
    if fields["patch"] != f"patches/{name}/{fields['path']}.patch":
        raise ValueError(f"パッチの場所が patches/{name}/<道筋>.patch ではない: {fields['patch']}")
    for key in _SHA256_KEYS:
        if not _is_sha256(fields[key]):
            raise ValueError(f"{fields['path']} の {key} が 64 桁の 16 進ではない: {fields[key]}")
    return OverlayFile(**fields)


def load_overlay(path: Path) -> Overlay:
    """定義を読み、形を確かめる。違えば `ValueError`。"""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"定義が JSON の object ではない: {path}")
    name = data.get("name")
    if not isinstance(name, str) or f"{name}.json" != path.name:
        raise ValueError(f"定義の name がファイル名と合わない: {path}: {name!r}")
    if data.get("vllm_commit") != VLLM_COMMIT:
        raise ValueError(f"定義の vllm_commit が固定の {VLLM_COMMIT} ではない: {path}")
    entries = data.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"定義の files が空でない配列ではない: {path}")
    files = tuple(_parse_file(entry, name) for entry in entries)
    paths = [file.path for file in files]
    if len(set(paths)) != len(paths):
        raise ValueError(f"定義の files の道筋が重複している: {path}")
    return Overlay(name=name, vllm_commit=VLLM_COMMIT, files=files)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_patch(original: bytes, patched: bytes, path: str) -> str:
    """`a/<道筋>` から `b/<道筋>` への unified diff (`patch -p1` で当てる)。"""
    before = original.decode("utf-8")
    after = patched.decode("utf-8")
    if not before.endswith("\n") or not after.endswith("\n"):
        raise ValueError(f"末尾が改行で終わらない: {path}")
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    if not diff:
        raise ValueError(f"写しがソースと同じで、パッチが空になる: {path}")
    return diff


def apply_patch(original: bytes, patch: Path, path: str, workdir: Path) -> bytes:
    """`workdir/<道筋>` に元のバイト列を書き、`patch -t -p1` で当てた結果を返す。"""
    target = workdir / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(original)
    subprocess.run(
        ["patch", "-t", "-p1", "-d", str(workdir), "-i", str(patch)],
        check=True,
        capture_output=True,
    )
    return target.read_bytes()


def _copy_path(overlay: Overlay, file: OverlayFile) -> Path:
    return PAYLOAD_DIR / overlay.name / file.path


def check(overlay: Overlay, *, source_dir: Path | None) -> list[str]:
    """写し・パッチ (と、あればソース) を定義と比べた問題の説明の列。空なら合格。"""
    problems: list[str] = []
    for file in overlay.files:
        copy = _copy_path(overlay, file)
        patch = OVERLAY_DIR / file.patch
        if not copy.is_file():
            problems.append(f"写しがない: {copy}")
            continue
        if not patch.is_file():
            problems.append(f"パッチがない: {patch}")
            continue
        if sha256_of(copy) != file.patched_sha256:
            problems.append(f"写しの SHA-256 が定義の patched_sha256 と違う: {copy}")
        if sha256_of(patch) != file.patch_sha256:
            problems.append(f"パッチの SHA-256 が定義の patch_sha256 と違う: {patch}")
        if tuple(copy.read_text(encoding="utf-8").splitlines()[:2]) != LICENSE_HEADER:
            problems.append(f"写しの先頭 2 行が上流の SPDX の表記ではない: {copy}")
        if source_dir is None:
            continue
        source = source_dir / file.path
        if not source.is_file():
            problems.append(f"固定のソースにファイルがない: {source}")
            continue
        if sha256_of(source) != file.source_sha256:
            problems.append(f"固定のソースの SHA-256 が定義の source_sha256 と違う: {source}")
        with tempfile.TemporaryDirectory() as workdir:
            try:
                patched = apply_patch(source.read_bytes(), patch, file.path, Path(workdir))
            except subprocess.CalledProcessError as error:
                problems.append(f"パッチが固定のソースに当たらない: {patch}: {error.stdout!r}")
                continue
        if patched != copy.read_bytes():
            problems.append(f"固定のソースにパッチを当てた結果が写しと違う: {copy}")
    return problems


def refresh(name: str) -> Overlay:
    """固定のソースと写しの差からパッチを書き、定義の SHA-256 を書き換える。"""
    if shutil.which("patch") is None:
        raise RuntimeError("patch コマンドがない")
    if not SOURCE_DIR.is_dir():
        raise RuntimeError(f"固定のソースがない: {SOURCE_DIR}")
    definition = OVERLAY_DIR / f"{name}.json"
    data: Any = json.loads(definition.read_text(encoding="utf-8"))
    overlay = load_overlay(definition)
    refreshed: list[OverlayFile] = []
    for file in overlay.files:
        source = SOURCE_DIR / file.path
        copy = _copy_path(overlay, file)
        patch = OVERLAY_DIR / file.patch
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text(
            make_patch(source.read_bytes(), copy.read_bytes(), file.path), encoding="utf-8"
        )
        refreshed.append(
            file._replace(
                source_sha256=sha256_of(source),
                patched_sha256=sha256_of(copy),
                patch_sha256=sha256_of(patch),
            )
        )
    data["files"] = [file._asdict() for file in refreshed]
    definition.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = load_overlay(definition)
    problems = check(result, source_dir=SOURCE_DIR)
    if problems:
        raise RuntimeError("作り直した結果が確かめに通らない:\n" + "\n".join(problems))
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("refresh", help="ソースと写しの差からパッチと定義を書く").add_argument(
        "name"
    )
    commands.add_parser("check", help="写しとパッチを定義と比べる").add_argument("name")
    args = parser.parse_args(argv)
    if args.command == "refresh":
        overlay = refresh(args.name)
        print(f"{overlay.name}: {len(overlay.files)} 個のファイルのパッチと定義を書いた")
        return
    overlay = load_overlay(OVERLAY_DIR / f"{args.name}.json")
    source_dir = SOURCE_DIR if SOURCE_DIR.is_dir() else None
    problems = check(overlay, source_dir=source_dir)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        raise SystemExit(1)
    scope = (
        "ソースへのパッチの適用まで" if source_dir is not None else "SHA-256 だけ (ソースがない)"
    )
    print(f"{overlay.name}: 問題なし ({scope})")


if __name__ == "__main__":
    main()
