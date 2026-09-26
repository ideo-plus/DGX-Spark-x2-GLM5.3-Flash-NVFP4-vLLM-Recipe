"""変換の道具から serving の関門までの一連の流れの試験 (Issue #64 の C10)。

合成 checkpoint (`synthetic.build_checkpoint`) を変換の道具で 2 回 (2 台の代わり) 変換し、
その `manifest.json` を `serve derived-import` で取り込み、`configure_tp2.py --weights` で
構成を生成し、`config.load_configs` で読み、`weights.verify_weights` (偽の実行役) で照合の
記録を書き、`guards.gate_weights_verified` に掛ける。途中の段を手書きの JSON で代用しない。
確かめること:

- 取り込んだマニフェストの `derivation` と `files` は、道具の出力から作られている
- 取り込みが標準出力に出す `manifest_sha256` は、書いたファイルのバイト列の SHA-256 である
- 生成した構成の重みは派生の参照で、取り込んだマニフェストと食い違わない
- 照合の記録は取り込んだマニフェストに結び付き、関門が 2 台とも通る
- 1 ファイルの sha256 だけを差し替えたマニフェストには、同じ記録では関門が落ちる
- 2 台の変換の結果が一致しなければ、取り込みは何も書かずに終了 1 になる

実機・ネットワーク・`serving/var/` には触らない。Spark への呼び出しは偽の実行役
(`fake_runner.FakeRunner`) が受け、リポジトリ根は `tmp_path` の下の最小の形を使う。
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fake_runner import FakeRunner, Reply, Rule
from serving_kit import cli, guards, weights
from serving_kit.config import load_configs, load_nodes, select_config
from serving_kit.plan import build_plans
from serving_kit.remote import RemoteRunner
from serving_kit.types import (
    ConfigDef,
    DerivedWeightsManifest,
    DerivedWeightsRef,
    GateResult,
    ManifestFile,
    NodeDef,
    NodeRole,
)

import synthetic
from k2_quant.__main__ import main as convert_main

K2_QUANT_DIR = Path(__file__).resolve().parents[1]
"""`experiments/k2-quant/` (この試験から見て 1 つ上)。"""

REPO_ROOT = K2_QUANT_DIR.parents[1]
CONFIGS_PATH = REPO_ROOT / "serving" / "config" / "configs.toml"
NODES_PATH = REPO_ROOT / "serving" / "config" / "nodes.toml"
GENERATOR_PATH = REPO_ROOT / "experiments" / "nope-mla" / "configure_tp2.py"
P1_WEIGHTS_MANIFEST = "RedHatAI__GLM-5.3-Flash-NVFP4.manifest.json"
MEASURED_RECORDS: tuple[str, ...] = (
    "docs/results/2026-09-22-nope-build.md",
    "docs/results/2026-09-22-netcheck-bandwidth.md",
)
"""生成した構成が `measured` で指す記録 (`load_configs` は実在だけを見る)。"""

DERIVED_NAME = "k2s1"
DERIVED_COMMIT = "a" * 40
DEFAULT_TOOL_PATH = "experiments/k2-quant"
"""`serve derived-import --tool` の既定 (計画 C1)。"""

DERIVED_CONFIG_NAME = f"p2-nope-tp2-full-{DERIVED_NAME}"
RECORD_NAME = f"{DERIVED_NAME}.derived.verified.json"
"""`verify_weights` が `<record_dir>/verified/<役割>/` に書く、派生の照合の記録の名前。"""

SPARK_HOSTS: tuple[str, str] = ("spark-153d", "spark-5083")
"""2 回の変換の出力先の名前 (2 台の代わり)。"""

REPLACED_SHA256 = "f" * 64
IMPORTED_AT = datetime(2026, 9, 26, 1, 0, 0, tzinfo=UTC)
STARTED_AT = datetime(2026, 9, 26, 2, 0, 0, tzinfo=UTC)
VERIFIED_AT = datetime(2026, 9, 26, 2, 30, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Converted:
    """道具を 2 回流したあとの状態。`root` はリポジトリ根の最小の形。"""

    root: Path
    tool_manifests: tuple[Path, Path]


@dataclass(frozen=True)
class ImportRun:
    """`serve derived-import` の 1 回の結果。"""

    code: int
    out: str
    err: str


@dataclass(frozen=True)
class Imported:
    """取り込みが通ったあとの、書かれたマニフェスト。"""

    path: Path
    manifest: DerivedWeightsManifest
    out: str


def _p1_weights() -> dict[str, Any]:
    """コミット済みの `configs.toml` の `p1-nvfp4-tp2` の weights の節 (元の重みの基準)。"""
    data = tomllib.loads(CONFIGS_PATH.read_text(encoding="utf-8"))
    p1: dict[str, Any] = data["configs"]["p1-nvfp4-tp2"]["weights"]
    return p1


def _read_json(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def _kv(out: str) -> dict[str, str]:
    """標準出力の `key=value` の行を読む。"""
    pairs: dict[str, str] = {}
    for line in out.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            pairs[key] = value
    return pairs


def _repo_root(tmp_path: Path) -> Path:
    """`load_configs` に渡すリポジトリ根の最小の形 (実測の記録と、p1 の元のマニフェストの写し)。"""
    root = tmp_path / "repo"
    for record in MEASURED_RECORDS:
        (root / record).parent.mkdir(parents=True, exist_ok=True)
        (root / record).write_text("", encoding="utf-8")
    weights_dir = root / "serving" / "weights"
    weights_dir.mkdir(parents=True)
    committed = REPO_ROOT / "serving" / "weights" / P1_WEIGHTS_MANIFEST
    (weights_dir / P1_WEIGHTS_MANIFEST).write_bytes(committed.read_bytes())
    return root


def _convert(source: Path, output: Path) -> Path:
    """道具を 1 回流し (既定の `--pattern`)、書かれた `manifest.json` の道筋を返す。"""
    p1 = _p1_weights()
    code = convert_main(
        [
            "--source",
            str(source),
            "--output",
            str(output),
            "--source-repo",
            str(p1["repo"]),
            "--source-revision",
            str(p1["revision"]),
        ]
    )
    assert code == 0
    return output / synthetic.MANIFEST_NAME


def _refuse_spark(var_root: Path) -> RemoteRunner:
    """`derived-import` は Spark に触らない。実行役を求められたら落とす。"""
    raise AssertionError(f"derived-import が遠隔の実行役を求めた (var_root={var_root})")


def _derived_import(root: Path, inputs: tuple[Path, ...]) -> ImportRun:
    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(
        [
            "derived-import",
            "--name",
            DERIVED_NAME,
            "--commit",
            DERIVED_COMMIT,
            *(str(path) for path in inputs),
        ],
        runner_factory=_refuse_spark,
        stdout=out,
        stderr=err,
        repo_root=root,
        now=lambda: IMPORTED_AT,
    )
    return ImportRun(code=code, out=out.getvalue(), err=err.getvalue())


def _load_generator() -> ModuleType:
    """生成器の module を、パッケージでない道筋から読む (`experiments/nope-mla/`)。"""
    loader = SourceFileLoader("configure_tp2", str(GENERATOR_PATH))
    spec = importlib.util.spec_from_file_location("configure_tp2", GENERATOR_PATH, loader=loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inspect_json(tmp_path: Path, generator: ModuleType) -> Path:
    """生成器が受ける `docker image inspect` の形の見本 (配列 1 要素)。"""
    item = {
        "Id": generator.EXPECTED_IMAGE_ID,
        "RepoTags": [generator.IMAGE_SEEN_AS],
        "Architecture": "arm64",
        "Os": "linux",
        "Size": 23438274807,
    }
    path = tmp_path / "image-inspect.json"
    path.write_text(json.dumps([item]), encoding="utf-8")
    return path


def _verify_rules(
    config: ConfigDef, nodes: dict[NodeRole, NodeDef], manifest: DerivedWeightsManifest
) -> list[Rule]:
    """`verify_weights` の台本 (関門の読み取りは通り、2 台の `sha256sum` はマニフェストと合う)。"""
    assert isinstance(config.weights, DerivedWeightsRef)
    rules: list[Rule] = []
    for plan in build_plans(config, nodes, STARTED_AT):
        directory = weights.weights_dir_on_spark(plan.argv, config.weights.mount_at)
        lines = "".join(f"{entry.sha256}  {directory}/{entry.path}\n" for entry in manifest.files)
        rules.append(Rule(prefix=("sha256sum",), node=plan.node, replies=(Reply(stdout=lines),)))
    rules.extend(
        (
            Rule(prefix=("uname",), replies=(Reply(stdout="spark-153d\n"),)),
            Rule(prefix=("docker", "ps"), replies=(Reply(stdout=""),)),
            Rule(prefix=("test", "-d"), replies=(Reply(),)),
            Rule(kind="push", replies=(Reply(),)),
        )
    )
    return rules


def _gate(
    tmp_path: Path,
    records: dict[NodeRole, str],
    nodes: dict[NodeRole, NodeDef],
    config: ConfigDef,
    manifest: DerivedWeightsManifest,
) -> dict[NodeRole, GateResult]:
    """記録を Spark の `cat` の返事にした偽の実行役で、2 台の関門 `weights_verified` を流す。"""
    reader = FakeRunner(
        var_root=tmp_path / "gate-var",
        script=[
            Rule(prefix=("cat",), node=role, replies=(Reply(stdout=text),))
            for role, text in records.items()
        ],
    )
    return {
        role: guards.gate_weights_verified(reader, nodes[role], config, manifest)
        for role in config.nodes
    }


@pytest.fixture
def converted(tmp_path: Path) -> Converted:
    """合成 checkpoint を道具で 2 回変換し、リポジトリ根の最小の形を用意する。"""
    source = tmp_path / "origin"
    synthetic.build_checkpoint(source)
    head, worker = (
        _convert(source, tmp_path / "converted" / host / DERIVED_NAME) for host in SPARK_HOSTS
    )
    return Converted(root=_repo_root(tmp_path), tool_manifests=(head, worker))


@pytest.fixture
def imported(converted: Converted) -> Imported:
    """2 台ぶんの `manifest.json` を `serve derived-import` で取り込む。"""
    run = _derived_import(converted.root, converted.tool_manifests)
    assert run.code == cli.EXIT_OK, run.err
    path = converted.root / "serving" / "weights" / f"{DERIVED_NAME}.manifest.json"
    manifest = weights.load_manifest(path)
    assert isinstance(manifest, DerivedWeightsManifest)
    return Imported(path=path, manifest=manifest, out=run.out)


@pytest.fixture
def nodes() -> dict[NodeRole, NodeDef]:
    return load_nodes(NODES_PATH, REPO_ROOT)


@pytest.fixture
def derived_config(
    converted: Converted,
    imported: Imported,
    nodes: dict[NodeRole, NodeDef],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> ConfigDef:
    """`configure_tp2.py --variant full --weights k2s1` で生成した構成を読んで選ぶ。"""
    generator = _load_generator()
    monkeypatch.setattr(generator, "WEIGHTS_DIR", converted.root / "serving" / "weights")
    output = tmp_path / "derived.toml"
    generator.main(
        [
            "--variant",
            "full",
            "--weights",
            DERIVED_NAME,
            str(_write_inspect_json(tmp_path, generator)),
            str(output),
        ]
    )
    return select_config(load_configs(output, converted.root), DERIVED_CONFIG_NAME, nodes)


@pytest.fixture
def records(
    imported: Imported,
    derived_config: ConfigDef,
    nodes: dict[NodeRole, NodeDef],
    tmp_path: Path,
) -> dict[NodeRole, str]:
    """`verify_weights` を偽の実行役で流し、2 台ぶんの照合の記録 (JSON の本文) を返す。"""
    runner = FakeRunner(
        var_root=tmp_path / "verify-var",
        script=_verify_rules(derived_config, nodes, imported.manifest),
        default=Reply(),
    )
    record_dir = tmp_path / "records"
    outcome = weights.verify_weights(
        runner,
        derived_config,
        nodes,
        imported.manifest,
        STARTED_AT,
        confirmer=guards.make_confirmer(
            assume_yes=True, stdin=io.StringIO(""), stderr=io.StringIO()
        ),
        record_dir=record_dir,
        verified_at=VERIFIED_AT,
        # 台本の `sha256sum` の返事は全ファイルぶんなので、1 回で全ファイルを読ませる
        batch_files=len(imported.manifest.files),
    )
    assert outcome.status == "verified", outcome.detail
    return {
        role: (record_dir / "verified" / role / RECORD_NAME).read_text(encoding="utf-8")
        for role in derived_config.nodes
    }


def test_derived_import_builds_the_derived_manifest_from_the_tool_output(
    converted: Converted, imported: Imported
) -> None:
    """取り込んだマニフェストの変換の条件・元の重み・ファイルは、道具の出力から作られている。"""
    # Given: 道具が書いた manifest.json
    tool = _read_json(converted.tool_manifests[0])
    conversion = tool["conversion"]

    # When: `serve derived-import` が書いたマニフェストを読む (fixture `imported`)
    derivation = imported.manifest.derivation

    # Then: 変換の条件は道具の `args` と `pattern`、元の重みは道具の `source` と同じ
    assert derivation.name == DERIVED_NAME
    assert derivation.conversion.tool == DEFAULT_TOOL_PATH
    assert derivation.conversion.commit == DERIVED_COMMIT
    assert derivation.conversion.args == tuple(conversion["args"])
    assert derivation.conversion.target_pattern == conversion["pattern"]
    assert derivation.origin.repo == conversion["source"]["repo"]
    assert derivation.origin.revision == conversion["source"]["revision"]
    # Then: ファイルの一覧と合計は道具の `files` と `total_bytes` と同じ
    assert imported.manifest.files == tuple(
        ManifestFile(path=entry["path"], sha256=entry["sha256"], size=entry["size"])
        for entry in tool["files"]
    )
    assert imported.manifest.total_bytes == tool["total_bytes"]


def test_derived_import_reports_the_sha256_of_the_written_manifest(imported: Imported) -> None:
    """取り込みの標準出力の `manifest_sha256` は、書いたファイルのバイト列の SHA-256 である。"""
    # Given: `serve derived-import` の標準出力と、書かれたファイル (fixture `imported`)
    pairs = _kv(imported.out)

    # When: 書かれたファイルのバイト列の SHA-256 を計算する
    written = hashlib.sha256(imported.path.read_bytes()).hexdigest()

    # Then: 標準出力の値と一致し、書いたことを示す
    assert pairs["manifest_sha256"] == written
    assert pairs["status"] == "written"


def test_generated_config_refers_to_the_imported_manifest(
    imported: Imported, derived_config: ConfigDef
) -> None:
    """生成した構成の重みは派生の参照で、取り込んだマニフェストと食い違わない。"""
    # Given: 取り込んだマニフェストから生成して読んだ構成 (fixture `derived_config`)
    weights_ref = derived_config.weights

    # When: 構成の重みとマニフェストを突き合わせる
    assert isinstance(weights_ref, DerivedWeightsRef)
    mismatch = guards.weights_ref_mismatch(weights_ref, imported.manifest)

    # Then: 食い違いはない
    assert mismatch is None


def test_verification_records_are_bound_to_the_imported_manifest(
    imported: Imported, records: dict[NodeRole, str]
) -> None:
    """2 台の照合の記録は、取り込んだマニフェストの中身の SHA-256 を持つ。"""
    # Given: `verify_weights` が書いた 2 台ぶんの記録 (fixture `records`)
    # When: 記録の `manifest_sha256` を読む
    bound = {role: json.loads(text)["manifest_sha256"] for role, text in records.items()}

    # Then: 2 台とも、取り込んだマニフェストの `content_sha256` と一致する
    assert bound == {
        "head": imported.manifest.content_sha256,
        "worker": imported.manifest.content_sha256,
    }


def test_weights_gate_passes_with_the_records_of_the_imported_manifest(
    imported: Imported,
    derived_config: ConfigDef,
    nodes: dict[NodeRole, NodeDef],
    records: dict[NodeRole, str],
    tmp_path: Path,
) -> None:
    """照合の記録を Spark から読む関門 `weights_verified` は、2 台とも通る。"""
    # Given: 取り込んだマニフェストで照合した 2 台の記録 (fixture `records`)
    # When: 同じマニフェストで関門を流す
    gates = _gate(tmp_path, records, nodes, derived_config, imported.manifest)

    # Then: 2 台とも通る
    assert {role: gate.passed for role, gate in gates.items()} == {"head": True, "worker": True}


def test_weights_gate_refuses_the_records_for_a_manifest_with_a_replaced_file_hash(
    imported: Imported,
    derived_config: ConfigDef,
    nodes: dict[NodeRole, NodeDef],
    records: dict[NodeRole, str],
    tmp_path: Path,
) -> None:
    """1 ファイルの sha256 だけを差し替えたマニフェストには、前の記録では関門が落ちる。"""
    # Given: 1 ファイルの sha256 だけを差し替えたマニフェスト (道筋・大きさ・件数・合計・
    # derivation は同じなので、構成とは食い違わない)
    files = list(imported.manifest.files)
    files[0] = files[0].model_copy(update={"sha256": REPLACED_SHA256})
    replaced = imported.manifest.model_copy(update={"files": tuple(files)})
    assert files[0].sha256 != imported.manifest.files[0].sha256
    assert derived_config.weights is not None
    assert guards.weights_ref_mismatch(derived_config.weights, replaced) is None

    # When: 取り込んだマニフェストで照合した記録のまま、関門を流す
    gates = _gate(tmp_path, records, nodes, derived_config, replaced)

    # Then: 2 台とも落ち、記録が結び付くマニフェストが違うことを示す
    assert {role: gate.passed for role, gate in gates.items()} == {"head": False, "worker": False}
    for gate in gates.values():
        assert "マニフェストが違う" in gate.detail


def test_derived_import_refuses_disagreeing_manifests_of_the_two_nodes(
    converted: Converted,
) -> None:
    """2 台の変換の結果が 1 ファイルでも違えば、終了 1 で何も書かない。"""
    # Given: 2 回目の変換の結果の manifest.json の 1 ファイルの sha256 を書き換えたもの
    head, worker = converted.tool_manifests
    tampered = _read_json(worker)
    assert tampered["files"][0]["sha256"] != REPLACED_SHA256
    tampered["files"][0]["sha256"] = REPLACED_SHA256
    worker.write_text(
        json.dumps(tampered, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # When: 2 つを `serve derived-import` に渡す
    run = _derived_import(converted.root, (head, worker))

    # Then: 終了 1 で、違う箇所を示し、派生のマニフェストを書かない
    assert run.code == cli.EXIT_PRECONDITION
    assert "files[0].sha256" in run.err
    assert not (converted.root / "serving" / "weights" / f"{DERIVED_NAME}.manifest.json").exists()
