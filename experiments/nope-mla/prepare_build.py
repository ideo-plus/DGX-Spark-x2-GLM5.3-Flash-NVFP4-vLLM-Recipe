"""固定した上流ソースと差分を Mac に用意する。既定では適用検査だけを行う。"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

BASE = "0961bbae2894d574be790d219651824eb199318e"
PATCH_BASE = "bc2ee480738d7dcc558262a0c6d81956b515b050"
PATCH_HEAD = "8d09804c877c48165c6ba69bc9dc02d09bae0b83"
REPOSITORY = "https://github.com/vllm-project/vllm.git"
PATHS = ("csrc/libtorch_stable/cache_kernels.cu", "tests/kernels/attention/test_cache.py")


def git(directory: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True)


def dependency_patch(source: Path) -> str:
    """上流の固定値が想定と違えば、置換せずに停止する。"""
    replacements = {
        "requirements/cuda.txt": (
            ("flashinfer-python==0.6.18.post1", "flashinfer-python==0.7.0"),
            ("flashinfer-cubin==0.6.18.post1", "flashinfer-cubin==0.7.0"),
        ),
        "docker/Dockerfile": (
            ("ARG FLASHINFER_VERSION=0.6.18.post1", "ARG FLASHINFER_VERSION=0.7.0"),
        ),
    }
    patches = []
    for filename, pairs in replacements.items():
        before = (source / filename).read_text()
        after = before
        for old, new in pairs:
            if after.count(old) != 1:
                raise ValueError(f"置換対象が一意でない: {filename}: {old}")
            after = after.replace(old, new)
        patches.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
            )
        )
    filename = "docker/versions.json"
    before = (source / filename).read_text()
    data = json.loads(before)
    if data["variable"]["FLASHINFER_VERSION"]["default"] != "0.6.18.post1":
        raise ValueError("versions.json の FlashInfer が想定と違う")
    data["variable"]["FLASHINFER_VERSION"]["default"] = "0.7.0"
    after = json.dumps(data, indent=2) + "\n"
    patches.extend(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )
    return "".join(patches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="新規ディレクトリ。既存なら停止")
    parser.add_argument("--apply", action="store_true", help="了承後に Mac のソースへ差分を適用")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    git(source, "init", "--quiet")
    git(source, "remote", "add", "origin", REPOSITORY)
    for revision in (BASE, PATCH_BASE, PATCH_HEAD):
        git(source, "fetch", "--quiet", "--depth=1", "origin", revision)
    git(source, "checkout", "--quiet", "--detach", BASE)
    upstream = git(source, "diff", "--binary", PATCH_BASE, PATCH_HEAD, "--", *PATHS)
    if not upstream:
        raise ValueError("上流の差分が空")
    if set(
        git(source, "diff", "--name-only", PATCH_BASE, PATCH_HEAD, "--", *PATHS).splitlines()
    ) != set(PATHS):
        raise ValueError("上流の変更ファイルが想定と違う")
    patches = {"upstream.patch": upstream, "flashinfer-version.patch": dependency_patch(source)}
    hashes = {}
    for name, patch in patches.items():
        path = output / name
        path.write_text(patch)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        git(source, "apply", "--check", str(path))
    if args.apply:
        for name in patches:
            git(source, "apply", str(output / name))
        git(source, "diff", "--check")
    manifest = {
        "base": BASE,
        "patch_base": PATCH_BASE,
        "patch_head": PATCH_HEAD,
        "flashinfer": "0.7.0",
        "applied": args.apply,
        "patch_sha256": hashes,
        "source_repository": REPOSITORY,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
