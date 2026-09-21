"""構成とノードの定義の、読み込みと検査の試験 (task 1.3)。

確かめること:

- design.md 「types / config」の検査 1〜10 のそれぞれに、断られる場合と、正しいものが
  通る場合がある
- 誤りは、項目の名前つきの日本語の文で、1 回の読み込みですべて並ぶ (requirements 3.7、
  design.md 「Error Handling」の「早く断る」「すべて並べて示す」)
- pydantic の英語の誤りの文が、そのまま出ない
- 検査 7 (直結の値) だけは `select_config` で行い、`serve` と `job` の 2 台の構成にだけ
  掛かる (requirements 4.7)
- `kind` ごとのノードの数 (tasks.md の Implementation Notes 1.2) と `schema_version`
  (design.md Data Models)
- 名前のない構成は、使える名前を添えて断られる
- `config` は `types` だけを読み込む (依存の向き)

試験用の TOML は、design.md Data Models の見本を元にした「最小の正しい構成」を作り、
そこから 1 か所ずつ壊す形で書く。
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

import pytest

from serving_kit import config as c
from serving_kit.types import CONFIG_SCHEMA_VERSION, ConfigDef, ConfigKind, NodeDef, NodeRole

# --- 見本の値 (design.md Data Models から) ------------------------------

DIGEST = "b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
REVISION = "18d55bfd" + "0" * 32
MANIFEST = "RedHatAI__GLM-5.3-Flash-NVFP4.manifest.json"
MEASURED = "docs/results/2026-09-21-netcheck-links.md"
ABSENT_MEASURED = "docs/results/2027-01-01-absent.md"
SOURCE = "https://docs.vllm.ai/en/latest/cli/serve/"
QUOTE = "vllm serve [model_tag] [options]"
IMAGE_EVIDENCE = (
    'source = "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags"\n'
    'quote = "glm53-flash-arm64-cu130 / linux/arm64 / 9666567584 B"'
)

SERVE = "p1-nvfp4-tp2"
PROBE = "probe-pinned"
FETCH = "p1-fetch-nvfp4"
INSPECT = "p1-image-licenses"
JOB = "p1-netcheck-bandwidth"

_TWO_NODE_KINDS = ("serve", "job", "fetch")
_NAMED_KINDS = ("serve", "probe")

# pydantic がそのまま出す英語の文 (これが誤りの文に混ざっていないことを確かめる)
ENGLISH_PROSE = (
    "Field required",
    "Extra inputs are not permitted",
    "Value error",
    "String should match pattern",
    "Input should be",
    "should be a valid",
    "at least 1 item",
)


# --- 試験用のリポジトリと TOML の組み立て -------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """マニフェストと実測の記録がある、リポジトリの最小の形を作る。"""
    root = tmp_path / "repo"
    (root / "docs" / "results").mkdir(parents=True, exist_ok=True)
    (root / MEASURED).write_text("直結のインターフェースの実測の要約\n", encoding="utf-8")
    (root / "serving" / "weights").mkdir(parents=True, exist_ok=True)
    (root / "serving" / "weights" / MANIFEST).write_text("{}\n", encoding="utf-8")
    (root / "serving" / "config").mkdir(parents=True, exist_ok=True)
    return root


def _toml_string(text: str) -> str:
    """TOML の文字列にする (`"` を含む値は、リテラル文字列 ('…') で書く)。"""
    if '"' in text and "'" not in text:
        return f"'{text}'"
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _setting(
    table: str,
    *,
    flag: str | None = None,
    value: str | None = None,
    why: str = "試験の理由",
    only_on: str | None = None,
    is_port: bool = False,
    evidence: str = "source",
    measured: str = MEASURED,
    extra: Sequence[str] = (),
) -> str:
    """1 つの設定のテーブルを TOML で書く。`evidence` で根拠の形を選ぶ。"""
    lines = [f"[{table}]"]
    if flag is not None:
        lines.append(f"flag = {_toml_string(flag)}")
    if value is not None:
        lines.append(f"value = {_toml_string(value)}")
    lines.append(f'why = "{why}"')
    if only_on is not None:
        lines.append(f'only_on = "{only_on}"')
    if is_port:
        lines.append("is_port = true")
    if evidence in ("source", "both"):
        lines.append(f'source = "{SOURCE}"')
        lines.append(f'quote = "{QUOTE}"')
    elif evidence == "source-only":
        lines.append(f'source = "{SOURCE}"')
    elif evidence == "quote-only":
        lines.append(f'quote = "{QUOTE}"')
    if evidence in ("measured", "both"):
        lines.append(f'measured = "{measured}"')
    lines.extend(extra)
    return "\n".join(lines) + "\n\n"


def _config(
    name: str = SERVE,
    *,
    kind: str = "serve",
    nodes: Sequence[str] | None = None,
    served: str | None = "auto",
    weights: bool | None = None,
    image_ref: str | None = None,
    revision: str = REVISION,
    manifest: str = MANIFEST,
    ready_timeout_s: str = "1800",
    head_extra: Sequence[str] = (),
    docker_extra: str = "",
    args_extra: str = "",
    env_extra: str = "",
) -> str:
    """最小の正しい構成を 1 つ書く。`*_extra` に、壊した設定を足していく。"""
    if nodes is None:
        nodes = ("head", "worker") if kind in _TWO_NODE_KINDS else ("head",)
    if served == "auto":
        served = "glm-5-3-flash" if kind in _NAMED_KINDS else None
    if weights is None:
        weights = kind in ("serve", "probe", "fetch")
    ref = image_ref if image_ref is not None else f"vllm/vllm-openai@sha256:{DIGEST}"

    head = [
        f"[configs.{name}]",
        f'kind = "{kind}"',
        'description = "試験の構成"',
        "nodes = [" + ", ".join(f'"{role}"' for role in nodes) + "]",
        f"ready_timeout_s = {ready_timeout_s}",
    ]
    if served is not None:
        head.append(f'served_model_name = "{served}"')
    head.extend(head_extra)
    text = "\n".join(head) + "\n\n"

    text += (
        "\n".join(
            [
                f"[configs.{name}.image]",
                f'ref = "{ref}"',
                'seen_as = "vllm/vllm-openai:glm53-flash-arm64-cu130"',
                "size_bytes = 9666567584",
                IMAGE_EVIDENCE,
            ]
        )
        + "\n\n"
    )

    if weights:
        text += (
            "\n".join(
                [
                    f"[configs.{name}.weights]",
                    'repo = "RedHatAI/GLM-5.3-Flash-NVFP4"',
                    f'revision = "{revision}"',
                    f'manifest = "{manifest}"',
                    'mount_at = "/models/nvfp4"',
                    'source = "https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4"',
                    'quote = "license: mit"',
                ]
            )
            + "\n\n"
        )

    text += _setting(
        f"configs.{name}.docker.mount-cache",
        flag="--mount",
        value="type=bind,source={remote_root}/cache,target=/root/.cache",
        why="JIT のキャッシュを、起動のたびに捨てない",
    )
    text += _setting(
        f"configs.{name}.args.model-path",
        value="{weights.mount_at}" if weights else "/usr/share/doc/vllm/LICENSE",
        why="vllm serve の第一の位置の引数は、モデルの場所",
    )
    text += _setting(
        f"configs.{name}.args.port",
        flag="--port",
        value="8000",
        is_port=True,
        why="既存の対象サーバー (8001) と重ならない番号",
    )
    text += _setting(
        f"configs.{name}.env.nccl-socket-ifname",
        flag="NCCL_SOCKET_IFNAME",
        value="{node.fabric_ifname}",
        why="直結の側のインターフェースを使わせる",
        evidence="measured",
    )
    return text + docker_extra + args_extra + env_extra


def _toml(*blocks: str, schema: str = f"schema_version = {CONFIG_SCHEMA_VERSION}\n\n") -> str:
    return schema + "".join(blocks)


def _write(repo: Path, text: str, filename: str = "configs.toml") -> Path:
    path = repo / "serving" / "config" / filename
    path.write_text(text, encoding="utf-8")
    return path


def _load(repo: Path, text: str) -> dict[str, ConfigDef]:
    return c.load_configs(_write(repo, text), repo)


def _refuse(repo: Path, text: str) -> str:
    with pytest.raises(c.ConfigError) as caught:
        c.load_configs(_write(repo, text), repo)
    return str(caught.value)


_NODES_BOTH = """\
[nodes.head]
ssh_host = "spark-153d"
lan_addr = "10.0.1.60"
remote_root = "/home/j5ik2o/vllm-baseline"

[nodes.worker]
ssh_host = "spark-5083"
lan_addr = "10.0.1.61"
remote_root = "/home/j5ik2o/vllm-baseline"
"""

_NODES_WITH_FABRIC = f"""\
[nodes.head]
ssh_host = "spark-153d"
lan_addr = "10.0.1.60"
remote_root = "/home/j5ik2o/vllm-baseline"
fabric_addr = "192.168.100.1"
fabric_ifname = "enp1s0f0np0"
fabric_measured = "{MEASURED}"

[nodes.worker]
ssh_host = "spark-5083"
lan_addr = "10.0.1.61"
remote_root = "/home/j5ik2o/vllm-baseline"
fabric_addr = "192.168.100.2"
fabric_ifname = "enp1s0f0np0"
fabric_measured = "{MEASURED}"
"""


def _load_nodes(repo: Path, text: str) -> dict[NodeRole, NodeDef]:
    return c.load_nodes(_write(repo, text, "nodes.toml"), repo)


def _refuse_nodes(repo: Path, text: str) -> str:
    with pytest.raises(c.ConfigError) as caught:
        c.load_nodes(_write(repo, text, "nodes.toml"), repo)
    return str(caught.value)


# --- 読み込みの土台 -----------------------------------------------------


def test_the_minimal_config_loads(repo: Path) -> None:
    configs = _load(repo, _toml(_config()))

    assert list(configs) == [SERVE]
    config = configs[SERVE]
    assert config.name == SERVE
    assert config.kind == "serve"
    assert config.nodes == ("head", "worker")
    assert config.weights is not None
    assert config.weights.revision == REVISION
    assert config.args["model-path"].flag is None
    assert config.args["port"].is_port is True
    assert config.env["nccl-socket-ifname"].measured == MEASURED


def test_args_keep_the_order_they_were_written_in(repo: Path) -> None:
    extra = _setting(
        f"configs.{SERVE}.args.max-num-seqs",
        flag="--max-num-seqs",
        value="16",
        why="線形アテンションの状態が、同時の本数に比例する",
    )
    configs = _load(repo, _toml(_config(args_extra=extra)))

    assert list(configs[SERVE].args) == ["model-path", "port", "max-num-seqs"]


def test_a_missing_file_is_named(repo: Path) -> None:
    with pytest.raises(c.ConfigError, match="見つからない"):
        c.load_configs(repo / "serving" / "config" / "absent.toml", repo)


def test_broken_toml_is_named(repo: Path) -> None:
    message = _refuse(repo, "schema_version = \n")

    assert "TOML" in message


def test_the_schema_version_is_required(repo: Path) -> None:
    message = _refuse(repo, _toml(_config(), schema=""))

    assert "schema_version" in message


@pytest.mark.parametrize("bad", ["2", "0", '"1"', "true"])
def test_a_different_schema_version_is_refused(repo: Path, bad: str) -> None:
    message = _refuse(repo, _toml(_config(), schema=f"schema_version = {bad}\n\n"))

    assert "schema_version" in message
    assert str(CONFIG_SCHEMA_VERSION) in message


def test_an_unknown_top_level_key_is_refused(repo: Path) -> None:
    schema = f"schema_version = {CONFIG_SCHEMA_VERSION}\nextras = 1\n\n"
    message = _refuse(repo, _toml(_config(), schema=schema))

    assert "extras" in message


def test_a_file_without_configs_is_refused(repo: Path) -> None:
    message = _refuse(repo, f"schema_version = {CONFIG_SCHEMA_VERSION}\n")

    assert "configs" in message


def test_an_empty_configs_table_is_refused(repo: Path) -> None:
    message = _refuse(repo, _toml("[configs]\n"))

    assert "configs" in message


def test_a_config_that_is_not_a_table_is_refused(repo: Path) -> None:
    message = _refuse(repo, _toml('[configs]\np1 = "serve"\n'))

    assert "configs.p1" in message


def test_a_written_name_that_disagrees_with_the_key_is_refused(repo: Path) -> None:
    message = _refuse(repo, _toml(_config(head_extra=('name = "another"',))))

    assert f"configs.{SERVE}.name" in message


def test_an_unknown_field_in_a_config_is_refused(repo: Path) -> None:
    message = _refuse(repo, _toml(_config(head_extra=("tensor_parallel = 2",))))

    assert f"configs.{SERVE}.tensor_parallel" in message
    assert "知らない項目" in message


def test_an_unknown_field_in_a_setting_is_refused(repo: Path) -> None:
    extra = _setting(
        f"configs.{SERVE}.args.max-num-seqs",
        flag="--max-num-seqs",
        value="16",
        extra=("reson = 2",),
    )
    message = _refuse(repo, _toml(_config(args_extra=extra)))

    assert f"configs.{SERVE}.args.max-num-seqs.reson" in message


def test_a_missing_field_is_named_in_japanese(repo: Path) -> None:
    message = _refuse(repo, _toml(_config().replace("ready_timeout_s = 1800\n", "")))

    assert f"configs.{SERVE}.ready_timeout_s: 項目がない" in message


def test_no_english_validation_prose_leaks(repo: Path) -> None:
    """pydantic の英語の文を、そのまま見せない (項目の名前つきの日本語の文に直す)。"""
    broken = _setting(
        f"configs.{SERVE}.args.max-num-seqs",
        flag="--max-num-seqs",
        value="16",
        evidence="source-only",
        extra=("reson = 1",),
    )
    text = _toml(
        _config(
            image_ref="vllm/vllm-openai:glm53-flash-arm64-cu130",
            revision="18d55bfd",
            ready_timeout_s="0",
            head_extra=("unknown_field = 1",),
            args_extra=broken,
        ).replace('kind = "serve"', 'kind = "serving"')
    )
    message = _refuse(repo, text)

    for prose in ENGLISH_PROSE:
        assert prose not in message, f"英語の文が漏れている: {prose}\n{message}"


def test_errors_from_two_configs_are_listed_together(repo: Path) -> None:
    first = _config(SERVE, image_ref="vllm/vllm-openai:latest")
    second = _config(PROBE, kind="probe", revision="18d55bfd")
    message = _refuse(repo, _toml(first, second))

    assert f"configs.{SERVE}.image.ref" in message
    assert f"configs.{PROBE}.weights.revision" in message


def test_a_setting_error_and_a_config_error_are_listed_together(repo: Path) -> None:
    """入れ子の設定が落ちても、その構成の `ConfigDef` の検査の誤りも並ぶ (Notes 1.2)。"""
    broken = _setting(
        f"configs.{SERVE}.args.max-num-seqs",
        flag="--max-num-seqs",
        value="16",
        evidence="none",
    )
    text = _toml(_config(served="RedHatAI/GLM-5.3-Flash-NVFP4", args_extra=broken))
    message = _refuse(repo, text)

    assert f"configs.{SERVE}.args.max-num-seqs: 根拠がない" in message
    # 構成そのものの決まり (型の after 検証) は、構成の名前と、項目の名前を文に持つ
    assert f"configs.{SERVE}: served_model_name" in message


# --- 検査 1: 根拠 -------------------------------------------------------


def test_two_settings_without_evidence_are_both_named(repo: Path) -> None:
    """完了の状態: 根拠のない設定を 2 つ含む構成が、2 つの項目の名前を並べて断られる。"""
    extra = _setting(
        f"configs.{SERVE}.args.max-num-seqs",
        flag="--max-num-seqs",
        value="16",
        evidence="none",
    ) + _setting(
        f"configs.{SERVE}.args.max-model-len",
        flag="--max-model-len",
        value="163840",
        evidence="none",
    )
    message = _refuse(repo, _toml(_config(args_extra=extra)))

    assert f"configs.{SERVE}.args.max-num-seqs: 根拠がない" in message
    assert f"configs.{SERVE}.args.max-model-len: 根拠がない" in message


@pytest.mark.parametrize("evidence", ["source-only", "quote-only", "both"])
def test_a_half_or_double_evidence_is_refused(repo: Path, evidence: str) -> None:
    extra = _setting(
        f"configs.{SERVE}.args.max-num-seqs",
        flag="--max-num-seqs",
        value="16",
        evidence=evidence,
    )
    message = _refuse(repo, _toml(_config(args_extra=extra)))

    assert f"configs.{SERVE}.args.max-num-seqs" in message


def test_a_measured_record_that_does_not_exist_is_refused(repo: Path) -> None:
    extra = _setting(
        f"configs.{SERVE}.args.max-num-seqs",
        flag="--max-num-seqs",
        value="16",
        evidence="measured",
        measured=ABSENT_MEASURED,
    )
    message = _refuse(repo, _toml(_config(args_extra=extra)))

    assert f"configs.{SERVE}.args.max-num-seqs.measured" in message
    assert ABSENT_MEASURED in message


def test_a_measured_record_outside_docs_is_refused(repo: Path) -> None:
    extra = _setting(
        f"configs.{SERVE}.args.max-num-seqs",
        flag="--max-num-seqs",
        value="16",
        evidence="measured",
        measured="/etc/passwd",
    )
    message = _refuse(repo, _toml(_config(args_extra=extra)))

    assert f"configs.{SERVE}.args.max-num-seqs" in message
    assert "docs/" in message


def test_an_existing_measured_record_passes(repo: Path) -> None:
    configs = _load(repo, _toml(_config()))

    assert configs[SERVE].env["nccl-socket-ifname"].measured == MEASURED


def test_the_evidence_of_the_image_is_checked_too(repo: Path) -> None:
    text = _toml(_config()).replace(IMAGE_EVIDENCE, f'measured = "{ABSENT_MEASURED}"')
    message = _refuse(repo, text)

    assert f"configs.{SERVE}.image.measured" in message


# --- 検査 2: イメージの参照 ---------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "vllm/vllm-openai:glm53-flash-arm64-cu130",
        "vllm/vllm-openai@sha256:b0501f",
        "vllm/vllm-openai@md5:b0501f99fec5136f248f78d5850977a2",
    ],
)
def test_an_image_without_a_digest_is_refused(repo: Path, bad: str) -> None:
    message = _refuse(repo, _toml(_config(image_ref=bad)))

    assert f"configs.{SERVE}.image.ref" in message
    assert "sha256" in message


def test_a_digest_image_reference_passes(repo: Path) -> None:
    configs = _load(repo, _toml(_config()))

    assert configs[SERVE].image.ref.endswith(DIGEST)


# --- 検査 3: 重み -------------------------------------------------------


@pytest.mark.parametrize("bad", ["18d55bfd", "0" * 41, "g" * 40])
def test_a_revision_that_is_not_forty_hex_digits_is_refused(repo: Path, bad: str) -> None:
    message = _refuse(repo, _toml(_config(revision=bad)))

    assert f"configs.{SERVE}.weights.revision" in message
    assert "40" in message


def test_a_manifest_that_does_not_exist_is_refused(repo: Path) -> None:
    message = _refuse(repo, _toml(_config(manifest="absent.manifest.json")))

    assert f"configs.{SERVE}.weights.manifest" in message
    assert "absent.manifest.json" in message


@pytest.mark.parametrize("bad", ["../../etc/passwd", "sub/dir.manifest.json", "."])
def test_a_manifest_outside_the_weights_directory_is_refused(repo: Path, bad: str) -> None:
    message = _refuse(repo, _toml(_config(manifest=bad)))

    assert f"configs.{SERVE}.weights.manifest" in message


def test_an_existing_manifest_passes(repo: Path) -> None:
    weights = _load(repo, _toml(_config()))[SERVE].weights

    assert weights is not None
    assert weights.manifest == MANIFEST


# --- 検査 4: 名乗るモデルの名前 -----------------------------------------


@pytest.mark.parametrize("bad", ["RedHatAI/GLM-5.3-Flash-NVFP4", "glm 5 3", "a,b"])
def test_a_model_name_with_a_slash_or_a_space_is_refused(repo: Path, bad: str) -> None:
    message = _refuse(repo, _toml(_config(served=bad)))

    assert f"configs.{SERVE}: served_model_name" in message
    assert bad in message


@pytest.mark.parametrize("kind", ["serve", "probe"])
def test_serve_and_probe_need_a_model_name(repo: Path, kind: str) -> None:
    message = _refuse(repo, _toml(_config(kind=kind, served=None)))

    assert "served_model_name" in message


@pytest.mark.parametrize("kind", ["job", "fetch", "inspect"])
def test_the_other_kinds_must_not_carry_a_model_name(repo: Path, kind: str) -> None:
    message = _refuse(repo, _toml(_config(kind=kind, served="glm-5-3-flash")))

    assert "served_model_name" in message


def test_the_five_kinds_all_load(repo: Path) -> None:
    text = _toml(
        _config(SERVE),
        _config(PROBE, kind="probe"),
        _config(JOB, kind="job"),
        _config(FETCH, kind="fetch"),
        _config(INSPECT, kind="inspect"),
    )
    configs = _load(repo, text)

    assert {config.kind for config in configs.values()} == set(get_args(ConfigKind))
    assert configs[INSPECT].served_model_name is None
    assert configs[INSPECT].weights is None


# --- 検査 5: 投機的デコード ---------------------------------------------


@pytest.mark.parametrize(
    "flag", ["--speculative-config", "--spec-method", "--spec-model", "--spec-tokens"]
)
def test_a_speculative_decoding_flag_is_refused(repo: Path, flag: str) -> None:
    extra = _setting(f"configs.{SERVE}.args.spec", flag=flag, value="3", why="試しに足す")
    message = _refuse(repo, _toml(_config(args_extra=extra)))

    assert f"configs.{SERVE}.args.spec" in message
    assert flag in message


def test_a_speculative_flag_written_as_a_positional_argument_is_refused(repo: Path) -> None:
    extra = _setting(f"configs.{SERVE}.args.spec", value="--speculative-config", why="抜け道を試す")
    message = _refuse(repo, _toml(_config(args_extra=extra)))

    assert f"configs.{SERVE}.args.spec" in message


def test_a_config_without_speculative_decoding_passes(repo: Path) -> None:
    extra = _setting(
        f"configs.{SERVE}.args.batched-tokens",
        flag="--max-num-batched-tokens",
        value="2048",
        why="投機的デコードの指定ではない",
    )
    configs = _load(repo, _toml(_config(args_extra=extra)))

    assert "batched-tokens" in configs[SERVE].args


# --- 検査 6: 置き換えの印 -----------------------------------------------


@pytest.mark.parametrize(
    "bad", ["{node.lan_addr}", "{weights.path}", "{env.HF_TOKEN}", "{worker.fabric_addr}"]
)
def test_an_unknown_replacement_marker_is_refused(repo: Path, bad: str) -> None:
    extra = _setting(
        f"configs.{SERVE}.args.master-addr", flag="--master-addr", value=bad, why="試しに足す"
    )
    message = _refuse(repo, _toml(_config(args_extra=extra)))

    assert f"configs.{SERVE}.args.master-addr" in message
    assert bad in message


def test_every_allowed_replacement_marker_passes(repo: Path) -> None:
    markers = [
        "{node.fabric_addr}",
        "{node.fabric_ifname}",
        "{node.rank}",
        "{head.fabric_addr}",
        "{head.lan_addr}",
        "{weights.mount_at}",
        "{remote_root}",
    ]
    extra = "".join(
        _setting(
            f"configs.{SERVE}.args.marker-{index}",
            flag=f"--marker-{index}",
            value=marker,
            why="使える置き換えの印",
        )
        for index, marker in enumerate(markers)
    )
    configs = _load(repo, _toml(_config(args_extra=extra)))

    assert [setting.value for setting in configs[SERVE].args.values()][2:] == markers


# JSON の値は、置き換えの印として見ない (design.md 「probe」の、層の数を減らす
# `--hf-overrides` の値が JSON になる)
JSON_VALUES = [
    '{"num_hidden_layers": 4}',
    '{"layer_types": {"0": "linear_attention", "1": "full_attention"}}',
    "{}",
    "{ }",
    '{"tokenizer_path": "{remote_root}/probe/glm"}',
]


@pytest.mark.parametrize("good", JSON_VALUES)
def test_a_json_value_is_not_read_as_a_replacement_marker(repo: Path, good: str) -> None:
    extra = _setting(
        f"configs.{SERVE}.args.hf-overrides",
        flag="--hf-overrides",
        value=good,
        why="層の数を減らす (アテンションの形は変えない)",
    )
    configs = _load(repo, _toml(_config(args_extra=extra)))

    assert configs[SERVE].args["hf-overrides"].value == good


@pytest.mark.parametrize(
    "bad", ["{foo}", "{node.fabric-addr}", '{"num_hidden_layers": {foo}}', '{"a": {"b": {bar}}}']
)
def test_an_unknown_marker_is_still_refused_next_to_json(repo: Path, bad: str) -> None:
    extra = _setting(
        f"configs.{SERVE}.args.hf-overrides", flag="--hf-overrides", value=bad, why="試しに足す"
    )
    message = _refuse(repo, _toml(_config(args_extra=extra)))

    assert f"configs.{SERVE}.args.hf-overrides" in message
    assert "知らない置き換えの印" in message


def test_a_probe_config_with_hf_overrides_loads(repo: Path) -> None:
    """design.md 「probe」の構成 (`--load-format dummy` と、層を減らす `--hf-overrides`)。"""
    extra = _setting(
        f"configs.{PROBE}.args.load-format",
        flag="--load-format",
        value="dummy",
        why="重みを取得せずに、乱数の値で起こす",
    ) + _setting(
        f"configs.{PROBE}.args.hf-overrides",
        flag="--hf-overrides",
        value='{"num_hidden_layers": 4, "layer_types": ["linear_attention"]}',
        why="アテンションの形を変えずに、層の数だけ減らす",
    )
    configs = _load(repo, _toml(_config(PROBE, kind="probe", args_extra=extra)))

    assert configs[PROBE].args["hf-overrides"].value is not None
    assert configs[PROBE].args["load-format"].value == "dummy"


# --- 検査 8: 禁じる docker のフラグ -------------------------------------


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--privileged", None),
        ("--pid", "host"),
        ("--userns", "host"),
        ("--security-opt", "seccomp=unconfined"),
        ("--volume", "/etc:/etc"),
        ("-v", "/etc:/etc"),
        ("--rm", None),
    ],
)
def test_each_forbidden_docker_flag_is_refused(repo: Path, flag: str, value: str | None) -> None:
    extra = _setting(
        f"configs.{SERVE}.docker.forbidden", flag=flag, value=value, why="根拠はあっても断る"
    )
    message = _refuse(repo, _toml(_config(docker_extra=extra)))

    assert f"configs.{SERVE}.docker.forbidden" in message
    assert flag in message


@pytest.mark.parametrize("value", ["always", "unless-stopped", "on-failure"])
def test_an_automatic_restart_is_refused(repo: Path, value: str) -> None:
    extra = _setting(
        f"configs.{SERVE}.docker.restart", flag="--restart", value=value, why="試しに足す"
    )
    message = _refuse(repo, _toml(_config(docker_extra=extra)))

    assert f"configs.{SERVE}.docker.restart" in message


def test_restart_no_is_allowed(repo: Path) -> None:
    extra = _setting(
        f"configs.{SERVE}.docker.restart",
        flag="--restart",
        value="no",
        why="自動で起こし直さないことを明示する",
    )
    configs = _load(repo, _toml(_config(docker_extra=extra)))

    assert configs[SERVE].docker["restart"].value == "no"


def test_a_forbidden_flag_written_with_an_equals_sign_is_refused(repo: Path) -> None:
    extra = _setting(
        f"configs.{SERVE}.docker.forbidden", flag="--pid=host", why="= の書き方でも断る"
    )
    message = _refuse(repo, _toml(_config(docker_extra=extra)))

    assert f"configs.{SERVE}.docker.forbidden" in message


def test_a_forbidden_flag_with_surrounding_spaces_is_refused(repo: Path) -> None:
    extra = _setting(
        f"configs.{SERVE}.docker.forbidden", flag=" --privileged ", why="前後の空白でも断る"
    )
    message = _refuse(repo, _toml(_config(docker_extra=extra)))

    assert f"configs.{SERVE}.docker.forbidden" in message
    assert "--privileged" in message


def test_a_forbidden_flag_written_as_a_positional_argument_is_refused(repo: Path) -> None:
    extra = _setting(f"configs.{SERVE}.docker.forbidden", value="--privileged", why="抜け道を試す")
    message = _refuse(repo, _toml(_config(docker_extra=extra)))

    assert f"configs.{SERVE}.docker.forbidden" in message


def test_the_allowed_docker_settings_pass(repo: Path) -> None:
    extra = _setting(
        f"configs.{SERVE}.docker.gpus", flag="--gpus", value="all", why="GPU を見せる"
    ) + _setting(
        f"configs.{SERVE}.docker.ipc", flag="--ipc", value="host", why="共有メモリを広く取る"
    )
    configs = _load(repo, _toml(_config(docker_extra=extra)))

    assert set(configs[SERVE].docker) == {"mount-cache", "gpus", "ipc"}


# --- 検査 9: --mount -----------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "type=volume,source={remote_root}/cache,target=/root/.cache",
        "type=bind,source=/etc,target=/etc",
        "type=bind,source={remote_root}/cache,src=/etc,target=/root/.cache",
        "type=bind,target=/root/.cache",
        "source={remote_root}/cache,target=/root/.cache",
        "{remote_root}/cache:/root/.cache",
        # 置き場所の外へ出る道筋 (Spark のほかの場所を、コンテナに見せない)
        "type=bind,source={remote_root}/../..,target=/host",
        "type=bind,source={remote_root}/models/../../../../etc,target=/etc",
        "type=bind,source={remote_root}/..,target=/host",
        # 置き場所の名前の、続きになっているだけの道筋
        "type=bind,source={remote_root}xyz,target=/c",
        "type=bind,source={remote_root}-other/models,target=/models",
    ],
)
def test_a_mount_outside_the_rules_is_refused(repo: Path, bad: str) -> None:
    extra = _setting(
        f"configs.{SERVE}.docker.mount-bad", flag="--mount", value=bad, why="試しに足す"
    )
    message = _refuse(repo, _toml(_config(docker_extra=extra)))

    assert f"configs.{SERVE}.docker.mount-bad" in message


def test_a_mount_without_a_value_is_refused(repo: Path) -> None:
    extra = _setting(f"configs.{SERVE}.docker.mount-bad", flag="--mount", why="値を書き忘れる")
    message = _refuse(repo, _toml(_config(docker_extra=extra)))

    assert f"configs.{SERVE}.docker.mount-bad" in message


def test_a_mount_of_the_remote_root_itself_passes(repo: Path) -> None:
    """置き場所そのものを見せる `--mount` は通る (道筋の検査が、厳しすぎないこと)。"""
    extra = _setting(
        f"configs.{SERVE}.docker.mount-root",
        flag="--mount",
        value="type=bind,source={remote_root},target=/root/vllm-baseline",
        why="置き場所そのものを見せる",
    )
    configs = _load(repo, _toml(_config(docker_extra=extra)))

    assert "mount-root" in configs[SERVE].docker


def test_a_read_only_bind_mount_under_remote_root_passes(repo: Path) -> None:
    extra = _setting(
        f"configs.{SERVE}.docker.mount-models",
        flag="--mount",
        value="type=bind,source={remote_root}/models/nvfp4,target=/models/nvfp4,readonly",
        why="重みを読み取り専用で見せる",
    )
    configs = _load(repo, _toml(_config(docker_extra=extra)))

    assert "mount-models" in configs[SERVE].docker


# --- 検査 10: --device と --cap-add -------------------------------------


@pytest.mark.parametrize("bad", ["/dev/nvidia0", "/dev/infiniband/uverbs0", "/dev/infiniband:rwm"])
def test_a_device_outside_the_list_is_refused_even_with_evidence(repo: Path, bad: str) -> None:
    extra = _setting(
        f"configs.{SERVE}.docker.device",
        flag="--device",
        value=bad,
        why="根拠はあるが、一覧にない",
        evidence="measured",
    )
    message = _refuse(repo, _toml(_config(docker_extra=extra)))

    assert f"configs.{SERVE}.docker.device" in message
    assert "一覧にないものは断る" in message


@pytest.mark.parametrize("bad", ["NET_ADMIN", "SYS_ADMIN", "ALL"])
def test_a_capability_outside_the_list_is_refused_even_with_evidence(repo: Path, bad: str) -> None:
    extra = _setting(
        f"configs.{SERVE}.docker.cap",
        flag="--cap-add",
        value=bad,
        why="根拠はあるが、一覧にない",
        evidence="measured",
    )
    message = _refuse(repo, _toml(_config(docker_extra=extra)))

    assert f"configs.{SERVE}.docker.cap" in message
    assert "一覧にないものは断る" in message


def test_the_three_allowed_device_and_capability_settings_pass(repo: Path) -> None:
    extra = (
        _setting(
            f"configs.{SERVE}.docker.device",
            flag="--device",
            value="/dev/infiniband",
            why="コンテナから RoCE のデバイスを見せる",
            evidence="measured",
        )
        + _setting(
            f"configs.{SERVE}.docker.cap-sys-nice",
            flag="--cap-add",
            value="SYS_NICE",
            why="NCCL の推奨",
        )
        + _setting(
            f"configs.{SERVE}.docker.cap-ipc-lock",
            flag="--cap-add",
            value="IPC_LOCK",
            why="メモリの登録に要る",
        )
    )
    configs = _load(repo, _toml(_config(docker_extra=extra)))

    assert {"device", "cap-sys-nice", "cap-ipc-lock"} <= set(configs[SERVE].docker)


# --- kind ごとのノードの数 (Notes 1.2) ----------------------------------


@pytest.mark.parametrize(
    ("kind", "nodes"),
    [
        ("serve", ("head",)),
        ("job", ("head",)),
        ("fetch", ("head",)),
        ("probe", ("head", "worker")),
        ("inspect", ("head", "worker")),
    ],
)
def test_a_wrong_number_of_nodes_is_refused(repo: Path, kind: str, nodes: Sequence[str]) -> None:
    message = _refuse(repo, _toml(_config(kind=kind, nodes=nodes)))

    assert f"configs.{SERVE}.nodes" in message


@pytest.mark.parametrize(
    ("kind", "nodes"),
    [
        ("serve", ("head", "worker")),
        ("job", ("head", "worker")),
        ("fetch", ("head", "worker")),
        ("probe", ("head",)),
        ("inspect", ("head",)),
    ],
)
def test_the_right_number_of_nodes_passes(repo: Path, kind: str, nodes: Sequence[str]) -> None:
    configs = _load(repo, _toml(_config(kind=kind, nodes=nodes)))

    assert configs[SERVE].nodes == tuple(nodes)


def test_the_number_of_nodes_is_named_even_when_the_config_itself_is_refused(repo: Path) -> None:
    """ノードの数の誤りが、ほかの構成の誤りに隠れない (すべて並べて示す)。"""
    text = _toml(_config(kind="serve", nodes=("head",), served="RedHatAI/GLM-5.3-Flash-NVFP4"))
    message = _refuse(repo, text)

    assert f"configs.{SERVE}.nodes" in message
    assert f"configs.{SERVE}: served_model_name" in message


def test_the_node_counts_cover_every_kind() -> None:
    assert set(c._NODE_COUNTS) == set(get_args(ConfigKind))


def test_the_node_roles_match_the_type() -> None:
    assert set(c._NODE_ROLES) == set(get_args(NodeRole))


# --- ノードの定義 -------------------------------------------------------


def test_the_minimal_nodes_file_loads(repo: Path) -> None:
    nodes = _load_nodes(repo, _NODES_BOTH)

    assert set(nodes) == {"head", "worker"}
    assert nodes["head"].role == "head"
    assert nodes["head"].ssh_host == "spark-153d"
    assert nodes["head"].fabric_addr is None


def test_the_measured_fabric_values_load(repo: Path) -> None:
    nodes = _load_nodes(repo, _NODES_WITH_FABRIC)

    assert nodes["worker"].fabric_ifname == "enp1s0f0np0"
    assert nodes["worker"].fabric_measured == MEASURED


def test_a_fabric_record_that_does_not_exist_is_refused(repo: Path) -> None:
    message = _refuse_nodes(repo, _NODES_WITH_FABRIC.replace(MEASURED, ABSENT_MEASURED))

    assert "nodes.head.fabric_measured" in message
    assert "nodes.worker.fabric_measured" in message


def test_a_fabric_record_outside_docs_is_refused(repo: Path) -> None:
    message = _refuse_nodes(repo, _NODES_WITH_FABRIC.replace(f'"{MEASURED}"', '"/etc/passwd"'))

    assert "nodes.head: fabric_measured" in message
    assert "docs/" in message


def test_an_unknown_role_is_refused(repo: Path) -> None:
    message = _refuse_nodes(repo, _NODES_BOTH + '\n[nodes.spare]\nssh_host = "spark-9999"\n')

    assert "nodes.spare" in message
    assert "head" in message
    assert "worker" in message


def test_a_written_role_that_disagrees_with_the_key_is_refused(repo: Path) -> None:
    text = _NODES_BOTH.replace("[nodes.head]", '[nodes.head]\nrole = "worker"')
    message = _refuse_nodes(repo, text)

    assert "nodes.head.role" in message


def test_an_unknown_field_in_a_node_is_refused(repo: Path) -> None:
    message = _refuse_nodes(repo, _NODES_BOTH.replace("ssh_host", "shh_host", 1))

    assert "nodes.head.shh_host" in message
    assert "知らない項目" in message


def test_a_missing_field_in_a_node_is_refused(repo: Path) -> None:
    text = _NODES_BOTH.replace('remote_root = "/home/j5ik2o/vllm-baseline"\n', "", 1)
    message = _refuse_nodes(repo, text)

    assert "nodes.head.remote_root: 項目がない" in message


def test_a_bad_address_is_refused(repo: Path) -> None:
    message = _refuse_nodes(repo, _NODES_BOTH.replace("10.0.1.60", "spark-153d.local"))

    assert "nodes.head.lan_addr" in message


@pytest.mark.parametrize("bad", ["home/j5ik2o/vllm-baseline", "/home/j5ik2o/"])
def test_a_remote_root_that_is_not_an_absolute_path_is_refused(repo: Path, bad: str) -> None:
    message = _refuse_nodes(repo, _NODES_BOTH.replace("/home/j5ik2o/vllm-baseline", bad, 1))

    assert "nodes.head: remote_root" in message
    assert bad in message


def test_the_errors_of_both_nodes_are_listed_together(repo: Path) -> None:
    message = _refuse_nodes(repo, _NODES_BOTH.replace("ssh_host", "shh_host"))

    assert "nodes.head.shh_host" in message
    assert "nodes.worker.shh_host" in message


def test_a_nodes_file_without_nodes_is_refused(repo: Path) -> None:
    message = _refuse_nodes(repo, "[nodes]\n")

    assert "nodes" in message


def test_an_unknown_top_level_key_in_the_nodes_file_is_refused(repo: Path) -> None:
    message = _refuse_nodes(repo, "extras = 1\n\n" + _NODES_BOTH)

    assert "extras" in message


def test_the_nodes_file_may_carry_the_schema_version(repo: Path) -> None:
    nodes = _load_nodes(repo, f"schema_version = {CONFIG_SCHEMA_VERSION}\n\n" + _NODES_BOTH)

    assert set(nodes) == {"head", "worker"}


def test_a_different_schema_version_in_the_nodes_file_is_refused(repo: Path) -> None:
    message = _refuse_nodes(repo, "schema_version = 2\n\n" + _NODES_BOTH)

    assert "schema_version" in message


def test_a_missing_nodes_file_is_named(repo: Path) -> None:
    with pytest.raises(c.ConfigError, match="見つからない"):
        c.load_nodes(repo / "serving" / "config" / "absent.toml", repo)


# --- 検査 7 と、名前で選ぶこと (select_config) --------------------------


def _five_configs(repo: Path) -> dict[str, ConfigDef]:
    text = _toml(
        _config(SERVE),
        _config(PROBE, kind="probe"),
        _config(JOB, kind="job"),
        _config(FETCH, kind="fetch"),
        _config(INSPECT, kind="inspect"),
    )
    return _load(repo, text)


def test_an_unknown_config_name_is_refused_with_the_names_that_work(repo: Path) -> None:
    configs = _five_configs(repo)
    nodes = _load_nodes(repo, _NODES_WITH_FABRIC)

    with pytest.raises(c.ConfigError) as caught:
        c.select_config(configs, "p1-nvfp4-tp8", nodes)

    message = str(caught.value)
    assert "p1-nvfp4-tp8" in message
    for name in (SERVE, PROBE, JOB, FETCH, INSPECT):
        assert name in message


@pytest.mark.parametrize("name", [SERVE, JOB])
def test_a_two_node_config_is_refused_while_the_fabric_values_are_empty(
    repo: Path, name: str
) -> None:
    """完了の状態: 推論サーバーの 2 台の構成は、直結の値が空だと選べない (4.7)。"""
    configs = _five_configs(repo)
    nodes = _load_nodes(repo, _NODES_BOTH)

    with pytest.raises(c.ConfigError) as caught:
        c.select_config(configs, name, nodes)

    message = str(caught.value)
    for role in ("head", "worker"):
        for field in ("fabric_addr", "fabric_ifname", "fabric_measured"):
            assert f"nodes.{role}.{field}" in message


@pytest.mark.parametrize("name", [PROBE, FETCH, INSPECT])
def test_the_other_kinds_are_selectable_while_the_fabric_values_are_empty(
    repo: Path, name: str
) -> None:
    """完了の状態: 直結の値が空でも、縮小の確認と取得の構成は選べる。"""
    configs = _five_configs(repo)
    nodes = _load_nodes(repo, _NODES_BOTH)

    assert c.select_config(configs, name, nodes).name == name


@pytest.mark.parametrize("name", [SERVE, JOB])
def test_a_two_node_config_is_selectable_once_the_fabric_values_are_measured(
    repo: Path, name: str
) -> None:
    configs = _five_configs(repo)
    nodes = _load_nodes(repo, _NODES_WITH_FABRIC)

    assert c.select_config(configs, name, nodes).name == name


def test_a_partly_measured_fabric_is_refused_with_the_missing_items(repo: Path) -> None:
    text = _NODES_WITH_FABRIC.replace('fabric_ifname = "enp1s0f0np0"\n', "", 1)
    text = text.replace(f'fabric_measured = "{MEASURED}"\n', "", 1)
    configs = _five_configs(repo)
    nodes = _load_nodes(repo, text)

    with pytest.raises(c.ConfigError) as caught:
        c.select_config(configs, SERVE, nodes)

    message = str(caught.value)
    assert "nodes.head.fabric_ifname" in message
    assert "nodes.head.fabric_measured" in message
    assert "nodes.head.fabric_addr" not in message
    assert "nodes.worker" not in message


def test_a_config_whose_node_has_no_definition_is_refused(repo: Path) -> None:
    only_head = "\n".join(_NODES_WITH_FABRIC.splitlines()[:7]) + "\n"
    configs = _five_configs(repo)
    nodes = _load_nodes(repo, only_head)

    with pytest.raises(c.ConfigError) as caught:
        c.select_config(configs, SERVE, nodes)

    assert "nodes.worker" in str(caught.value)


# --- 依存の向き ---------------------------------------------------------

ALLOWED_IMPORT_ROOTS = frozenset(
    {"__future__", "collections", "pathlib", "re", "tomllib", "typing", "pydantic", "serving_kit"}
)


def test_the_config_module_imports_only_the_types_module() -> None:
    """`config` は `types` だけを import する (design.md の Dependency direction)。"""
    tree = ast.parse(inspect.getsource(c))
    roots: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "相対の読み込みがある"
            assert node.module is not None
            roots.add(node.module.split(".")[0])
            modules.add(node.module)

    assert roots <= ALLOWED_IMPORT_ROOTS, f"許していない読み込み: {roots - ALLOWED_IMPORT_ROOTS}"
    assert {name for name in modules if name.startswith("serving_kit")} == {"serving_kit.types"}
