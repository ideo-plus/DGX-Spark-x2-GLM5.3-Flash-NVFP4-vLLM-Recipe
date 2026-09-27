"""K2 第 2a 段・第 2b 段の、vLLM の直したファイルの重ね (`experiments/k2-vllm-overlay/`) の試験。

固定した vLLM (0961bbae) の 3 つの Python ファイルを、lm_head・MLA の射影・KDA のまとめて
いない射影が FP8 (W8A16、チャネルごと、compressed-tensors) で読めるように直し (`k2s2a`。ADR 0007
の第 2a 段、#73)、第 2b 段 (`k2s2b`。#79) は、それに加えて KDA のまとめた層 `in_proj_qkvbfg_a` も
FP8 で読めるように直す。直したファイルの写し (`serving/payload/vllm-overlay/<名前>/<道筋>`) を、
イメージを作り直さずに読み取り専用の bind mount で重ねる。実物の vLLM は Mac に入っていない
(torch も CUDA もない) ので、直した内容は、写しのソースを `ast` で読んで確かめる (複製の分割の
読み込みだけは、写しの class を、基底を見本に替えて実行して確かめる)。実機での起動は、この試験の
対象ではない。

確かめること (1〜3、7 は、2 つの名前の両方。k2s2b にだけ当たる 5・6 は、名前を固定する):

1. 写しが、「固定のソース + パッチ」と一致する。ソース (`serving/var/nope-build-0961bbae/source`。
   Git 対象外) が手元にあれば、パッチを当てた結果が写しとバイト単位で一致することまで見る。
   ソースがない CI では、定義 (`<名前>.json`) に固定した SHA-256 で写しとパッチを固定する
2. 直したのは、ちょうど 3 つのファイルである (indexer の `attention.py` は直さない)
3. lm_head: `ParallelLMHead` が、humming の線形カーネルが読む属性を持つ
4. MLA: `quant_config` を渡し、FP8 の `weight_scale` を持つ射影を BF16 への読み替えに流さない
5. KDA: `quant_config` を外す処理を消す。まとめた層 `in_proj_qkvbfg_a` は、k2s2a では量子化せず
   (`quant_config=None`)、k2s2b では `self.quant_config` を渡す。複製の分割 (f_a・g_a。
   `replicated_shard_ids=(4, 5)`) は、重みも `weight_scale` も、`tp_rank` を 0 にして全体を読む
6. `packed_modules_mapping`: MLA の `fused_qkv_a_proj` の対応を持つ。KDA のまとめた層
   `in_proj_qkvbfg_a` の対応は、k2s2b だけが足す (checkpoint のテンソル名の形の 6 射影)
7. 写しの先頭 2 行は上流の SPDX の表記のまま、`LICENSES.md` に載っている
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import pytest

SERVING_DIR: Final[Path] = Path(__file__).resolve().parents[2]
"""`serving/` (この試験から見て 2 つ上)。"""

REPO_ROOT: Final[Path] = SERVING_DIR.parent
OVERLAY_DIR: Final[Path] = REPO_ROOT / "experiments" / "k2-vllm-overlay"
TOOL_PATH: Final[Path] = OVERLAY_DIR / "overlay.py"
PAYLOAD_ROOT: Final[Path] = SERVING_DIR / "payload" / "vllm-overlay"
"""`serve push` が Spark の `payload/vllm-overlay/` へ配る、写しの置き場。"""

SOURCE_DIR: Final[Path] = SERVING_DIR / "var" / "nope-build-0961bbae" / "source"
"""固定した vLLM (0961bbae) のソース。Git 対象外なので、CI にはない。"""

LICENSES_PATH: Final[Path] = REPO_ROOT / "LICENSES.md"

VLLM_COMMIT: Final[str] = "0961bbae2894d574be790d219651824eb199318e"
LM_HEAD_PATH: Final[str] = "vllm/model_executor/layers/vocab_parallel_embedding.py"
MODEL_PATH: Final[str] = "vllm/models/glm5next/common/model.py"
KDA_PATH: Final[str] = "vllm/models/glm5next/common/kda.py"
PATCHED_PATHS: Final[tuple[str, ...]] = (LM_HEAD_PATH, MODEL_PATH, KDA_PATH)
"""直したファイル (Issue の (a) lm_head、(b) MLA と (d) mapping、(c) KDA)。"""

STAGE_2A: Final[str] = "k2s2a"
STAGE_2B: Final[str] = "k2s2b"
OVERLAY_NAMES: Final[tuple[str, ...]] = (STAGE_2A, STAGE_2B)
"""重ねる定義の名前 (第 2a 段、第 2b 段)。共通の試験は、この 2 つの両方で確かめる。"""

LICENSE_HEADER: Final[tuple[str, str]] = (
    "# SPDX-License-Identifier: Apache-2.0",
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project",
)
"""上流の全ファイルの先頭の 2 行。写しでも変えない。"""

_SHA256_HEX: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


# --- 定義と写しの読み取り ---------------------------------------------------


def _definition_path(name: str) -> Path:
    return OVERLAY_DIR / f"{name}.json"


def _copy_dir(name: str) -> Path:
    return PAYLOAD_ROOT / name


def _read_definition(name: str) -> dict[str, Any]:
    path = _definition_path(name)
    if not path.is_file():
        pytest.fail(f"定義がない: {path}")
    data: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"定義が JSON の object ではない: {path}"
    definition: dict[str, Any] = data
    return definition


def _entry(name: str, path: str) -> dict[str, Any]:
    """定義 `name` の `files` の、`path` の項目 (ちょうど 1 つ)。"""
    matches: list[dict[str, Any]] = [
        entry for entry in _read_definition(name)["files"] if entry["path"] == path
    ]
    if len(matches) != 1:
        pytest.fail(
            f"定義の files に {path} が 1 つではない ({len(matches)} 個): {_definition_path(name)}"
        )
    return matches[0]


def _sha256(path: Path) -> str:
    if not path.is_file():
        pytest.fail(f"ファイルがない: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy(name: str, path: str) -> Path:
    return _copy_dir(name) / path


def _parse(name: str, path: str) -> ast.Module:
    """写しのソースを `ast` で読む (実行しない。vLLM は、この環境に入っていない)。"""
    copy = _copy(name, path)
    if not copy.is_file():
        pytest.fail(f"写しがない: {copy}")
    return ast.parse(copy.read_text(encoding="utf-8"), filename=str(copy))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    found = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    assert len(found) == 1, f"class {name} が 1 つではない ({len(found)} 個)"
    return found[0]


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    found = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1, f"{cls.name}.{name} が 1 つではない ({len(found)} 個)"
    return found[0]


def _keywords(call: ast.Call) -> dict[str, ast.expr]:
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}


# --- 1〜2. 写し == 固定のソース + パッチ (定義の形と、SHA-256 の固定) -----------


@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_the_definition_pins_the_commit_and_exactly_the_three_patched_files(name: str) -> None:
    """定義の `name`・`vllm_commit` と、直したファイルの集合 (ちょうど 3 つ) が固定されている。

    indexer の `attention.py` と、MTP の `mtp.py` は、この段では直さない。
    """
    # Given: コミットした定義
    data = _read_definition(name)

    # When: 直したファイルの道筋を取り出す
    paths = [entry["path"] for entry in data["files"]]

    # Then: 名前と commit が固定で、3 つのファイルがちょうど 1 度ずつ載る
    assert data["name"] == name
    assert data["vllm_commit"] == VLLM_COMMIT
    assert sorted(paths) == sorted(PATCHED_PATHS)


@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_the_definition_records_the_hashes_and_the_patch_location_of_every_file(name: str) -> None:
    """どのファイルにも、3 つの SHA-256 (ソース、写し、パッチ) と、パッチの場所と理由がある。"""
    # Given / When: 3 つのファイルの定義の項目
    entries = [_entry(name, path) for path in PATCHED_PATHS]

    # Then: SHA-256 は 64 桁の 16 進で、パッチは `patches/<名前>/<道筋>.patch`、理由は空でない
    for entry in entries:
        for key in ("source_sha256", "patched_sha256", "patch_sha256"):
            assert _SHA256_HEX.fullmatch(entry[key]), f"{entry['path']} の {key}: {entry[key]!r}"
        assert entry["patch"] == f"patches/{name}/{entry['path']}.patch"
        assert entry["why"].strip(), f"{entry['path']} の why が空"


@pytest.mark.parametrize("path", PATCHED_PATHS)
@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_the_copy_has_the_pinned_sha256(name: str, path: str) -> None:
    """写しの SHA-256 が、定義の `patched_sha256` と一致する (ソースがない CI でも動く)。

    写しを直接編集して、パッチを作り直さないと、ここで落ちる。
    """
    assert _sha256(_copy(name, path)) == _entry(name, path)["patched_sha256"]


@pytest.mark.parametrize("path", PATCHED_PATHS)
@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_the_patch_has_the_pinned_sha256(name: str, path: str) -> None:
    """パッチの SHA-256 が、定義の `patch_sha256` と一致する (ソースがない CI でも動く)。"""
    entry = _entry(name, path)
    assert _sha256(OVERLAY_DIR / entry["patch"]) == entry["patch_sha256"]


def _require_source() -> None:
    if not SOURCE_DIR.is_dir():
        pytest.skip(
            "固定のソースが手元にない (CI)。写しとパッチは、SHA-256 の固定の試験で守っている"
        )


@pytest.mark.parametrize("path", PATCHED_PATHS)
@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_the_pinned_source_file_has_the_recorded_sha256(name: str, path: str) -> None:
    """固定のソースの、その道筋のファイルが実在し、SHA-256 が `source_sha256` と一致する。"""
    _require_source()
    assert _sha256(SOURCE_DIR / path) == _entry(name, path)["source_sha256"]


def _apply_patch(name: str, path: str, workdir: Path) -> bytes:
    """ソースの写しにパッチを当てた結果 (`patch -t -p1`。パッチは `a/<道筋>`、`b/<道筋>`)。"""
    assert shutil.which("patch") is not None, "patch コマンドがない"
    target = workdir / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_DIR / path, target)
    subprocess.run(
        [
            "patch",
            "-t",
            "-p1",
            "-d",
            str(workdir),
            "-i",
            str(OVERLAY_DIR / _entry(name, path)["patch"]),
        ],
        check=True,
        capture_output=True,
    )
    return target.read_bytes()


@pytest.mark.parametrize("path", PATCHED_PATHS)
@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_the_patch_applied_to_the_pinned_source_gives_the_copy(
    name: str, path: str, tmp_path: Path
) -> None:
    """パッチを固定のソースに当てると、配る写しとバイト単位で一致する。"""
    # Given: 固定のソースがあり、写しとパッチが定義されている
    _require_source()

    # When: ソースの写しにパッチを当てる
    patched = _apply_patch(name, path, tmp_path)

    # Then: 配る写しと、1 バイトも違わない
    assert patched == _copy(name, path).read_bytes()


def _load_tool() -> Any:
    """`experiments/k2-vllm-overlay/overlay.py` を、パッケージを介さずに読む。"""
    if not TOOL_PATH.is_file():
        pytest.fail(f"道具がない: {TOOL_PATH}")
    spec = importlib.util.spec_from_file_location("k2_vllm_overlay_tool", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_the_tool_check_accepts_the_committed_definition(name: str) -> None:
    """道具の `check` が、コミットした定義と写しとパッチを問題なしとする (ソースなしでも)。"""
    tool = _load_tool()

    assert tool.check(tool.load_overlay(_definition_path(name)), source_dir=None) == []


@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_the_tool_check_accepts_the_committed_definition_against_the_source(name: str) -> None:
    """ソースがあれば、`check` はパッチの適用の一致まで見て、問題なしとする。"""
    _require_source()
    tool = _load_tool()

    assert tool.check(tool.load_overlay(_definition_path(name)), source_dir=SOURCE_DIR) == []


def test_the_tool_check_reports_a_copy_edited_without_regenerating_the_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """写しを直接編集して、パッチと定義を作り直さないと、`check` が問題を報告する。"""
    # Given: 写しの置き場の写しの 1 つを、手で書き足した状態 (定義とパッチは元のまま)
    tool = _load_tool()
    overlay = tool.load_overlay(_definition_path(STAGE_2A))
    payload = tmp_path / "payload"
    shutil.copytree(PAYLOAD_ROOT, payload)
    edited = payload / STAGE_2A / MODEL_PATH
    edited.write_bytes(edited.read_bytes() + b"# edited by hand\n")
    monkeypatch.setattr(tool, "PAYLOAD_DIR", payload)

    # When: ソースなしで `check` する
    problems = tool.check(overlay, source_dir=None)

    # Then: 問題を報告する (手を入れる前の写しは、上の試験のとおり問題なし)
    assert problems


# --- 7. ライセンスの表記 ---------------------------------------------------


@pytest.mark.parametrize("path", PATCHED_PATHS)
@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_each_copy_keeps_the_upstream_license_header(name: str, path: str) -> None:
    """写しの先頭 2 行が、上流の SPDX の表記のまま (Apache-2.0 のファイルの写し)。"""
    lines = _copy(name, path).read_text(encoding="utf-8").splitlines()

    assert tuple(lines[:2]) == LICENSE_HEADER


_HUNK: Final[re.Pattern[str]] = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def _license_header_lines_touched(patch_text: str) -> tuple[int, list[str]]:
    """パッチが、元のファイルの先頭 2 行を消す・挟む行 (と、hunk の数)。"""
    touched: list[str] = []
    hunks = 0
    old_line = 0
    for line in patch_text.splitlines():
        found = _HUNK.match(line)
        if found is not None:
            hunks += 1
            old_line = int(found.group(1))
            continue
        if hunks == 0 or line.startswith("\\"):
            continue
        if line.startswith("-"):
            if old_line <= len(LICENSE_HEADER):
                touched.append(line)
            old_line += 1
        elif line.startswith("+"):
            if old_line <= len(LICENSE_HEADER):
                touched.append(line)
        else:
            old_line += 1
    return hunks, touched


@pytest.mark.parametrize("path", PATCHED_PATHS)
@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_no_hunk_of_a_patch_touches_the_license_header(name: str, path: str) -> None:
    """パッチは、上流の先頭 2 行 (SPDX の表記) を消さず、挟まない。

    `# ruff: noqa` のような行を足して先頭を変えると、上流の写しとして扱えなくなる。
    """
    # Given: パッチの本文
    patch_text = (OVERLAY_DIR / _entry(name, path)["patch"]).read_text(encoding="utf-8")

    # When: 元のファイルの先頭 2 行に触れる行を数える
    hunks, touched = _license_header_lines_touched(patch_text)

    # Then: hunk はあり (何も読めずに通っていない)、先頭 2 行には触れない
    assert hunks >= 1, f"パッチに hunk が読めない: {name} {path}"
    assert touched == []


@pytest.mark.parametrize("path", PATCHED_PATHS)
@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_licenses_md_lists_the_copied_file_with_commit_license_and_the_patch_location(
    name: str, path: str
) -> None:
    """`LICENSES.md` の、写しのファイルの行に、固定の commit、Apache-2.0、そのパッチの場所が載る。

    行の文言や列の並びは問わない。見るのは識別子だけで、パッチの場所は、定義 (`<名前>.json`) が
    持つ値から導く (写しは上流とほぼ同じ形に保つので、変更の記録は、パッチとこの行に置く)。
    同じ道筋の行は、名前ごとにある (k2s2a の行と k2s2b の行)。パッチの場所が名前を区別する。
    """
    # Given: `LICENSES.md` の本文と、定義が持つ、そのファイルのパッチの場所
    text = LICENSES_PATH.read_text(encoding="utf-8")
    patch_location = f"experiments/k2-vllm-overlay/{_entry(name, path)['patch']}"

    # When: そのファイルの道筋を含む行を探す
    rows = [line for line in text.splitlines() if path in line]

    # Then: 1 つの行に、固定の commit、Apache-2.0、そのファイルのパッチの場所が並ぶ
    assert rows, f"LICENSES.md に {path} の行がない"
    assert any(
        VLLM_COMMIT in row and "Apache-2.0" in row and patch_location in row for row in rows
    ), rows


# --- 3. lm_head (Issue の (a)。#68) ----------------------------------------


def _lm_head_init(name: str) -> ast.FunctionDef:
    return _method(_class(_parse(name, LM_HEAD_PATH), "ParallelLMHead"), "__init__")


def _self_assignments(function: ast.FunctionDef, *, top_level_only: bool) -> dict[str, str]:
    """`self.<属性> = <式>` の、属性から式 (`ast.unparse`) への対応。"""
    statements: list[ast.AST] = list(function.body) if top_level_only else list(ast.walk(function))
    assigned: dict[str, str] = {}
    for node in statements:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                assigned[target.attr] = ast.unparse(node.value)
    return assigned


@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_lm_head_gets_the_linear_attributes_the_humming_kernel_reads(name: str) -> None:
    """FP8 の `ParallelLMHead` を humming の線形カーネルが読むための 3 つの属性が入る。

    compressed-tensors は `ParallelLMHead` を線形の層として扱う。humming は、層の
    `output_partition_sizes` と `has_bias` を読む (`LinearBase` にはあり、埋め込みにはない)。
    """
    # Given / When: `ParallelLMHead.__init__` の `self.<属性> = …` の代入
    assigned = _self_assignments(_lm_head_init(name), top_level_only=False)

    # Then: 入力の大きさ、出力の分割の大きさ (語彙の分割)、bias の有無が入る
    assert assigned["input_size_per_partition"] == "embedding_dim"
    assert assigned["output_partition_sizes"] == "[self.num_embeddings_per_partition]"
    assert assigned["has_bias"] == "bias"


@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_lm_head_attributes_are_set_even_when_the_layer_has_no_bias(name: str) -> None:
    """3 つの属性は、`if bias:` の中ではなく、常に入る (lm_head は bias を持たない)。

    `has_bias` だけを落とす、または `bias` のときだけ入れると、bias なしの lm_head で
    humming が `AttributeError` になる。
    """
    # Given / When: `__init__` の直下の (条件の中でない) 代入
    assigned = _self_assignments(_lm_head_init(name), top_level_only=True)

    # Then: 3 つとも、常に実行される位置にある
    assert {"input_size_per_partition", "output_partition_sizes", "has_bias"} <= set(assigned)


def test_the_lm_head_copy_of_the_stage_2b_is_byte_identical_to_the_stage_2a_copy() -> None:
    """第 2b 段の `vocab_parallel_embedding.py` の写しは、第 2a 段の写しと 1 バイトも違わない。

    第 2b 段が足す修正は、KDA (`kda.py`) と `packed_modules_mapping` (`model.py`) だけで、lm_head の
    修正は第 2a 段のまま持ち越す。
    """
    stage_2a = _copy(STAGE_2A, LM_HEAD_PATH)
    stage_2b = _copy(STAGE_2B, LM_HEAD_PATH)
    assert stage_2a.is_file(), f"写しがない: {stage_2a}"
    assert stage_2b.is_file(), f"写しがない: {stage_2b}"

    assert stage_2b.read_bytes() == stage_2a.read_bytes()


# --- 4. MLA (Issue の (b)) -------------------------------------------------


@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_the_decoder_layer_hands_its_quant_config_to_the_mla_attention(name: str) -> None:
    """MLA の `Glm5NextMLAAttention(...)` の `quant_config` が、`None` 固定でなく変数を渡す。"""
    # Given / When: `Glm5NextDecoderLayer.__init__` の、MLA の生成の呼び出し
    init = _method(_class(_parse(name, MODEL_PATH), "Glm5NextDecoderLayer"), "__init__")
    calls = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Glm5NextMLAAttention"
    ]

    # Then: 呼び出しは 1 つで、`quant_config=quant_config` (変数) を渡す
    assert len(calls) == 1
    value = _keywords(calls[0])["quant_config"]
    assert isinstance(value, ast.Name)
    assert value.id == "quant_config"


@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_the_overlay_does_not_touch_the_indexer_source(name: str) -> None:
    """indexer (`attention.py`) は直さない (この段では BF16 のまま)。写しにも定義にも載らない。"""
    paths = [entry["path"] for entry in _read_definition(name)["files"]]

    assert not [path for path in paths if path.endswith("/attention.py")]
    assert not (
        _copy_dir(name) / "vllm" / "models" / "glm5next" / "common" / "attention.py"
    ).exists()


_FP8_DTYPE: Final[object] = object()
"""`torch.float8_e4m3fn` の代わりの目印 (テンソルの `dtype` と比べるだけ)。"""

_LAYER: Final[str] = "model.language_model.layers.3.self_attn"

_ATTN_PROJECTIONS: Final[tuple[tuple[str, str], ...]] = (
    ("q_a_proj", "fused_qkv_a_proj"),
    ("kv_a_proj_with_mqa", "fused_qkv_a_proj"),
    ("q_b_proj", "q_b_proj"),
    ("o_proj", "o_proj"),
)
"""(checkpoint の射影の名前, モデルの層の名前)。`fused_qkv_a_proj` は、2 つの射影のまとめ。"""


def _forbid_dequantize(*args: object, **kwargs: object) -> None:
    raise AssertionError("BF16 への読み替えの計算まで進んだ")


def _fp8_attn_proj_loader(name: str) -> Callable[..., bool]:
    """写しの `_try_load_fp8_attn_proj` を、`torch` なしで呼べる形で取り出す。

    `model.py` は import できない (vLLM と torch が要る) ので、`ast` で、この関数と、それが
    読む `_FP8_ATTN_PROJS` の 2 つだけを取り出して実行する。テンソルは、`dtype` と `shape` だけを
    持つ見本で足りる (関数は、両方が揃うまで計算に進まない)。
    """
    tree = _parse(name, MODEL_PATH)
    picked: list[ast.stmt] = []
    for node in tree.body:
        is_table = isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_FP8_ATTN_PROJS"
            for target in node.targets
        )
        is_loader = isinstance(node, ast.FunctionDef) and node.name == "_try_load_fp8_attn_proj"
        if is_table or is_loader:
            picked.append(node)
    assert len(picked) == 2, "_FP8_ATTN_PROJS と _try_load_fp8_attn_proj が 1 つずつ要る"
    namespace: dict[str, Any] = {
        "torch": SimpleNamespace(float8_e4m3fn=_FP8_DTYPE),
        "_dequant_fp8_block": _forbid_dequantize,
    }
    exec(compile(ast.Module(body=picked, type_ignores=[]), MODEL_PATH, "exec"), namespace)
    loader: Callable[..., bool] = namespace["_try_load_fp8_attn_proj"]
    return loader


def _tensor(dtype: object) -> SimpleNamespace:
    return SimpleNamespace(dtype=dtype, shape=(4, 8))


def _load(
    name: str, checkpoint_name: str, tensor: SimpleNamespace, params: dict[str, object]
) -> tuple[bool, dict[str, Any], set[str]]:
    """1 つのテンソルを `_try_load_fp8_attn_proj` に渡した結果 (横取りしたか、buf、読んだ印)。"""
    buf: dict[str, Any] = {}
    loaded: set[str] = set()
    intercepted = _fp8_attn_proj_loader(name)(checkpoint_name, tensor, buf, params, loaded, 0)
    return intercepted, buf, loaded


def _w8a16_params(layer: str, target_base: str) -> dict[str, object]:
    """FP8 (W8A16、チャネルごと) の層が登録するパラメータ (`weight` と `weight_scale`)。"""
    return {
        f"{layer}.{target_base}.weight": object(),
        f"{layer}.{target_base}.weight_scale": object(),
    }


@pytest.mark.parametrize(("checkpoint", "target_base"), _ATTN_PROJECTIONS)
@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_a_weight_scale_of_an_fp8_model_projection_is_left_to_the_normal_load_path(
    name: str, checkpoint: str, target_base: str
) -> None:
    """モデルが FP8 の `weight_scale` を持つ射影の scale は、BF16 への読み替えに流さない。

    W8A16 (チャネルごと) は `weight_scale` を登録する。ここで横取りして buffer に溜めると、
    scale が FP8 のパラメータに届かない。
    """
    # Given: モデルが FP8 の `weight` と `weight_scale` を持ち、checkpoint の scale が届く
    params = _w8a16_params(_LAYER, target_base)

    # When: checkpoint の `<射影>.weight_scale` を渡す
    intercepted, buf, loaded = _load(
        name, f"{_LAYER}.{checkpoint}.weight_scale", _tensor("float32"), params
    )

    # Then: 横取りせず (通常の読み込みに任せる)、何も溜めず、読んだ印も付けない
    assert intercepted is False
    assert buf == {}
    assert loaded == set()


@pytest.mark.parametrize(("checkpoint", "target_base"), _ATTN_PROJECTIONS)
@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_an_fp8_weight_of_an_fp8_model_projection_is_not_turned_into_bf16(
    name: str, checkpoint: str, target_base: str
) -> None:
    """モデルが FP8 の `weight_scale` を持つ射影の FP8 の重みは、BF16 に読み替えない。

    `quant_config` を渡すだけで読み込みの経路を直さないと、FP8 のパラメータに、BF16 に
    戻した重みを流し込む。
    """
    # Given: モデルが FP8 の `weight` と `weight_scale` を持ち、checkpoint の FP8 の重みが届く
    params = _w8a16_params(_LAYER, target_base)

    # When: checkpoint の `<射影>.weight` (FP8) を渡す
    intercepted, buf, loaded = _load(
        name, f"{_LAYER}.{checkpoint}.weight", _tensor(_FP8_DTYPE), params
    )

    # Then: 横取りせず、何も溜めず、読んだ印も付けない
    assert intercepted is False
    assert buf == {}
    assert loaded == set()


@pytest.mark.parametrize(("checkpoint", "target_base"), _ATTN_PROJECTIONS)
@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_an_fp8_weight_of_a_bf16_model_projection_is_still_collected_for_dequantizing(
    name: str, checkpoint: str, target_base: str
) -> None:
    """モデルが BF16 のままの射影は、これまでどおり、FP8 の重みを集めて BF16 に戻す。

    変更対象外の既存の振る舞い。scale が届くまで集めておき (まだ読み込まず)、`True` を返す。
    """
    # Given: モデルは BF16 の `weight` だけを持つ (scale のパラメータがない)
    params: dict[str, object] = {f"{_LAYER}.{target_base}.weight": object()}

    # When: checkpoint の `<射影>.weight` (FP8) を渡す
    intercepted, buf, loaded = _load(
        name, f"{_LAYER}.{checkpoint}.weight", _tensor(_FP8_DTYPE), params
    )

    # Then: 横取りして集める (まだ読み込まない)
    assert intercepted is True
    assert buf != {}
    assert loaded == set()


_KDA_LAYER: Final[str] = "model.language_model.layers.0.self_attn"

_KDA_MERGED_SHARDS: Final[tuple[str, ...]] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "b_proj",
    "f_a_proj",
    "g_a_proj",
)
"""KDA の層のまとめた層 `in_proj_qkvbfg_a` の 6 つの shard (checkpoint のテンソル名の形)。"""


@pytest.mark.parametrize("suffix", ["weight", "weight_scale"])
@pytest.mark.parametrize("shard", _KDA_MERGED_SHARDS)
def test_an_fp8_kda_merged_shard_is_left_to_the_normal_load_path_in_the_stage_2b(
    shard: str, suffix: str
) -> None:
    """KDA のまとめた層の 6 つの shard の、FP8 の重みと `weight_scale` は、BF16 への読み替えに
    流さず、通常の読み込みに任せる (第 2b 段)。

    第 2b 段は、まとめた層 `in_proj_qkvbfg_a` を FP8 (W8A16) で持つので、6 つの shard の FP8 の
    重みと scale が `_try_load_fp8_attn_proj` を通る。この関数が横取りするのは MLA の射影の名前
    だけで、KDA のまとめた層の名前は横取りしない (横取りすると、`stacked_params_mapping` を通って
    `in_proj_qkvbfg_a` の FP8 のパラメータに届くはずの重みが、BF16 に戻されて届かない)。
    """
    # Given: モデルが、まとめた層の FP8 の `weight` と `weight_scale` を持つ
    params = _w8a16_params(_KDA_LAYER, "in_proj_qkvbfg_a")
    tensor = _tensor(_FP8_DTYPE if suffix == "weight" else "float32")

    # When: checkpoint の `<shard>.weight` (FP8) または `<shard>.weight_scale` を渡す
    intercepted, buf, loaded = _load(STAGE_2B, f"{_KDA_LAYER}.{shard}.{suffix}", tensor, params)

    # Then: 横取りせず、何も溜めず、読んだ印も付けない
    assert intercepted is False
    assert buf == {}
    assert loaded == set()


# --- 5. KDA (Issue の (c)) -------------------------------------------------


def _kda_init(name: str) -> ast.FunctionDef:
    return _method(_class(_parse(name, KDA_PATH), "Glm5NextLinearAttention"), "__init__")


def _projection_calls(function: ast.FunctionDef) -> dict[str, ast.Call]:
    """`prefix=f"{prefix}.<名前>"` を持つ呼び出しの、名前ごとの対応 (層の生成の呼び出し)。"""
    calls: dict[str, ast.Call] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        prefix = _keywords(node).get("prefix")
        if not isinstance(prefix, ast.JoinedStr):
            continue
        tail = prefix.values[-1]
        if (
            isinstance(tail, ast.Constant)
            and isinstance(tail.value, str)
            and tail.value.startswith(".")
        ):
            calls[tail.value.removeprefix(".")] = node
    return calls


@pytest.mark.parametrize("projection", ["f_b_proj", "g_b_proj", "o_proj"])
@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_the_unmerged_kda_projections_receive_the_quant_config(name: str, projection: str) -> None:
    """`f_b_proj`・`g_b_proj`・`o_proj` は、`self.quant_config` を受ける (FP8 で読める)。"""
    # Given / When: その射影を作る呼び出し
    call = _projection_calls(_kda_init(name))[projection]

    # Then: `quant_config=self.quant_config`
    assert ast.unparse(_keywords(call)["quant_config"]) == "self.quant_config"


def test_the_merged_kda_projection_is_not_quantized_in_the_stage_2a() -> None:
    """第 2a 段の写しは、まとめた層 `in_proj_qkvbfg_a` を量子化しない (第 2b 段で扱う)。

    `f_a`・`g_a` を複製して 1 つにまとめた層で、scale の読み込みが別に要る。第 2a 段の写しは、
    2026-09-27 の探りで使ったものなので、変えない。
    """
    # Given / When: その層を作る呼び出し
    call = _projection_calls(_kda_init(STAGE_2A))["in_proj_qkvbfg_a"]

    # Then: `quant_config=None`
    value = _keywords(call)["quant_config"]
    assert isinstance(value, ast.Constant)
    assert value.value is None


def test_the_merged_kda_projection_receives_the_quant_config_in_the_stage_2b() -> None:
    """第 2b 段の写しは、まとめた層 `in_proj_qkvbfg_a` に `self.quant_config` を渡す (FP8 で読む)。

    `None` のままだと、`in_proj_qkvbfg_a` は BF16 のまま作られ、FP8 の 6 つの shard の重みと
    `weight_scale` が届いても、BF16 の重みの形と合わない。
    """
    # Given / When: その層を作る呼び出し
    call = _projection_calls(_kda_init(STAGE_2B))["in_proj_qkvbfg_a"]

    # Then: `quant_config=self.quant_config`
    assert ast.unparse(_keywords(call)["quant_config"]) == "self.quant_config"


def test_the_merged_kda_projection_keeps_shards_4_and_5_replicated_in_the_stage_2b() -> None:
    """まとめた層の f_a (shard 4)・g_a (shard 5) は、TP の 2 台で複製する指定のまま。

    この指定が、複製の分割の重みと `weight_scale` を、`tp_rank` を 0 にして全体で読む読み込みの
    対象を決める。
    """
    # Given / When: まとめた層を作る呼び出しの `replicated_shard_ids`
    call = _projection_calls(_kda_init(STAGE_2B))["in_proj_qkvbfg_a"]

    # Then: `(4, 5)`
    assert ast.literal_eval(_keywords(call)["replicated_shard_ids"]) == (4, 5)


_MERGED_LOADER_TP_RANK: Final[int] = 1
"""見本の基底が持つ、この rank の `tp_rank` (0 と区別できる値。TP=2 の 2 台目)。"""

_REPLICATED_SHARDS: Final[tuple[int, ...]] = (4, 5)
_SPLIT_SHARDS: Final[tuple[int, ...]] = (0, 1, 2, 3)


class _BaseLoaderProbe:
    """`MergedColumnParallelLinear` の代わりの見本 (実物は vLLM と torch が要る)。

    基底の `weight_loader` / `weight_loader_v2` が呼ばれた時点の、層の `tp_rank` と、パラメータの
    `tp_rank` を記録する。実物の基底は、このパラメータの `tp_rank` で、読み込む重みを切る
    (`_ColumnvLLMParameter.load_merged_column_weight`)。
    """

    def __init__(self, input_size: int, output_sizes: list[int], **kwargs: object) -> None:
        self.tp_rank = _MERGED_LOADER_TP_RANK
        self.calls: list[tuple[str, int, int | None, int]] = []

    def weight_loader(self, param: Any, loaded_weight: object, loaded_shard_id: Any = None) -> None:
        self.calls.append(
            ("weight_loader", self.tp_rank, getattr(param, "tp_rank", None), loaded_shard_id)
        )

    def weight_loader_v2(
        self, param: Any, loaded_weight: object, loaded_shard_id: Any = None
    ) -> None:
        self.calls.append(
            ("weight_loader_v2", self.tp_rank, getattr(param, "tp_rank", None), loaded_shard_id)
        )


def _merged_linear(name: str) -> _BaseLoaderProbe:
    """写しの `_Glm5NextMergedColumnParallelLinear` を、基底を見本に替えて作る (TP=2)。

    `kda.py` は import できないので、`ast` でこの class だけを取り出して実行する。基底
    (`MergedColumnParallelLinear`) は `_BaseLoaderProbe`、注釈が読む `nn.Parameter` と
    `torch.Tensor` は目印に替える。
    """
    cls = _class(_parse(name, KDA_PATH), "_Glm5NextMergedColumnParallelLinear")
    namespace: dict[str, Any] = {
        "MergedColumnParallelLinear": _BaseLoaderProbe,
        "nn": SimpleNamespace(Parameter=object),
        "torch": SimpleNamespace(Tensor=object),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), KDA_PATH, "exec"), namespace)
    made: _BaseLoaderProbe = namespace["_Glm5NextMergedColumnParallelLinear"](
        4096,
        [4096, 4096, 4096, 32, 128, 128],
        replicated_shard_ids=(4, 5),
        tp_size=2,
    )
    return made


def _load_into(layer: _BaseLoaderProbe, loader: str, shard: int) -> SimpleNamespace:
    """`tp_rank` を持つパラメータ (FP8 の重みや scale) に、`shard` を読み込ませる。"""
    param = SimpleNamespace(tp_rank=_MERGED_LOADER_TP_RANK)
    getattr(layer, loader)(param, object(), shard)
    return param


@pytest.mark.parametrize("shard", _REPLICATED_SHARDS)
@pytest.mark.parametrize("loader", ["weight_loader", "weight_loader_v2"])
def test_replicated_kda_shards_are_read_whole_with_tp_rank_0_in_the_stage_2b(
    loader: str, shard: int
) -> None:
    """複製の分割 (f_a: shard 4、g_a: shard 5) は、基底が読む時点で、層もパラメータも `tp_rank` が
    0 になる (第 2b 段)。

    実物の基底は、パラメータの `tp_rank` から、重みや `weight_scale` を切る
    (`loaded_weight.narrow(output_dim, tp_rank * shard_size, shard_size)`)。TP=2 の 2 台目
    (`tp_rank` 1) でも 0 番目から切ると、複製した重みも scale も、どの台も同じ全体を読む。
    パラメータの `tp_rank` を 0 にしないと、FP8 の `weight_scale` (`ChannelQuantScaleParameter`) が
    1 台目の分の範囲の外を読む。`weight_loader` と `weight_loader_v2` の両方が要る。
    """
    # Given: TP=2 の 2 台目の層と、`tp_rank` を持つパラメータ
    layer = _merged_linear(STAGE_2B)

    # When: 複製の分割を読み込む
    _load_into(layer, loader, shard)

    # Then: 基底が見た層とパラメータの `tp_rank` は、どちらも 0
    assert layer.calls == [(loader, 0, 0, shard)]


@pytest.mark.parametrize("shard", _SPLIT_SHARDS)
@pytest.mark.parametrize("loader", ["weight_loader", "weight_loader_v2"])
def test_split_kda_shards_keep_their_own_tp_rank_in_the_stage_2b(loader: str, shard: int) -> None:
    """TP で分割する分割 (q・k・v・b: shard 0〜3) は、この rank の `tp_rank` のまま基底が読む。

    重みも `weight_scale` も、この rank の分だけを切る (全体を読まない)。
    """
    # Given: TP=2 の 2 台目の層と、`tp_rank` を持つパラメータ
    layer = _merged_linear(STAGE_2B)

    # When: 分割の分割を読み込む
    _load_into(layer, loader, shard)

    # Then: 基底が見た層とパラメータの `tp_rank` は、どちらもこの rank のまま
    assert layer.calls == [(loader, _MERGED_LOADER_TP_RANK, _MERGED_LOADER_TP_RANK, shard)]


@pytest.mark.parametrize("shard", [*_SPLIT_SHARDS, *_REPLICATED_SHARDS])
@pytest.mark.parametrize("loader", ["weight_loader", "weight_loader_v2"])
def test_the_merged_loader_restores_tp_rank_after_each_shard_in_the_stage_2b(
    loader: str, shard: int
) -> None:
    """どの分割を読んだあとも、層とパラメータの `tp_rank` は、この rank の値に戻る。

    複製の分割で 0 にしたままだと、次に読む分割 (q・k・v・b) が、1 台目の分を切ってしまう。
    """
    # Given: TP=2 の 2 台目の層と、`tp_rank` を持つパラメータ
    layer = _merged_linear(STAGE_2B)

    # When: 1 つの分割を読み込む
    param = _load_into(layer, loader, shard)

    # Then: 層もパラメータも、元の `tp_rank`
    assert layer.tp_rank == _MERGED_LOADER_TP_RANK
    assert param.tp_rank == _MERGED_LOADER_TP_RANK


@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_kda_no_longer_swaps_out_the_quant_config_of_the_vllm_config(name: str) -> None:
    """基底の初期化に、本物の `quant_config` が渡る (`vllm_config.quant_config = None` を消す)。

    try/finally で一時的に `None` にする処理を残すと、`self.quant_config` が `None` になり、
    どの射影も量子化を受けない。
    """
    # Given / When: `__init__` の中の、`vllm_config.quant_config` への代入
    swaps = [
        node
        for node in ast.walk(_kda_init(name))
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute)
        and target.attr == "quant_config"
        and isinstance(target.value, ast.Name)
        and target.value.id == "vllm_config"
    ]

    # Then: 1 つもない
    assert swaps == []


# --- 6. packed_modules_mapping (Issue の (d)) ------------------------------


def _packed_modules_mapping(name: str) -> dict[str, list[str]]:
    cls = _class(_parse(name, MODEL_PATH), "Glm5NextForConditionalGeneration")
    values = [
        node.value
        for node in cls.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "packed_modules_mapping"
            for target in node.targets
        )
    ]
    assert len(values) == 1, "packed_modules_mapping の代入が 1 つではない"
    mapping = ast.literal_eval(values[0])
    assert isinstance(mapping, dict)
    return mapping


@pytest.mark.parametrize("name", OVERLAY_NAMES)
def test_packed_modules_mapping_keeps_dense_gate_up_and_gains_the_mla_fused_projection(
    name: str,
) -> None:
    """既存の `gate_up_proj` を落とさず、`fused_qkv_a_proj` が、checkpoint の `q_a_proj` と
    `kv_a_proj_with_mqa` に展開される。

    compressed-tensors が `ignore` と `targets` に当てるとき、まとめた層を shard の名前に
    展開する。
    """
    mapping = _packed_modules_mapping(name)

    assert mapping["gate_up_proj"] == ["gate_proj", "up_proj"]
    assert mapping["fused_qkv_a_proj"] == ["q_a_proj", "kv_a_proj_with_mqa"]


def test_packed_modules_mapping_of_the_stage_2a_leaves_the_kda_merged_layer_out() -> None:
    """第 2a 段の写しの対応は、`gate_up_proj` と `fused_qkv_a_proj` の 2 項目だけ (KDA のまとめた
    層は足さない。第 2b 段の範囲)。
    """
    assert _packed_modules_mapping(STAGE_2A) == {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }


def test_packed_modules_mapping_of_the_stage_2b_expands_the_kda_merged_layer() -> None:
    """第 2b 段の写しの対応は、第 2a 段の 2 項目に、`in_proj_qkvbfg_a` を checkpoint のテンソル名の
    形の 6 つの shard (`q_proj`・`k_proj`・`v_proj`・`b_proj`・`f_a_proj`・`g_a_proj`) に展開する
    項目を足したもの (#79)。

    compressed-tensors は、この展開で、まとめた層の `ignore` と `targets` の当たりを、6 つの shard
    ごとに引く。`forget_gate.` を付けた名前や、並びが違う名前では、shard が別々の scheme に当たる。
    """
    assert _packed_modules_mapping(STAGE_2B) == {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "in_proj_qkvbfg_a": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"],
    }
