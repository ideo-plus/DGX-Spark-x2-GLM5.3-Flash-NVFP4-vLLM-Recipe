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
- 生成の前後で、実物の `configs.toml` のバイト列が同じである (生成器は読むだけで書かない)
- 生成器は、ネットワークも `subprocess` も読み込まない

試験は、`tmp_path` に inspect の JSON の見本を作り、実物の `configs.toml` と `nodes.toml` を
読み取りで使う。`serving/var/` (`.gitignore` 対象) には依存せず、実機・ネットワークには
一切つながない。生成器の module は、パッケージでない `experiments/nope-mla/` の下にあるので
`importlib` で読む。
"""

from __future__ import annotations

import ast
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
from serving_kit.plan import LABEL_CONFIG_SHA256, build_plans
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
WEIGHTS_LABEL = "RedHatAI/GLM-5.3-Flash-NVFP4@18d55bfd5a2194887738da73753975c9d3842f46"


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


def _find_pair(argv: tuple[str, ...], flag: str) -> tuple[str, str]:
    """`flag` と、その直後の値の組を返す (env 以外のフラグ付きの引数)。"""
    index = argv.index(flag)
    return argv[index], argv[index + 1]


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
