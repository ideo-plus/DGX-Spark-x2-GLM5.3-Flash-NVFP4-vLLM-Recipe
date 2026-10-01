"""コマンドの入口の試験 (tasks.md 5.1)。

確かめること (design.md 「入口 › cli」と「Error Handling」、requirements 1.1、1.2、1.3、
1.6、1.7、1.9、2.1、tasks.md 5.1 の完了の状態):

- **すべてのサブコマンドの `--help` が 0 で終わる**。使える名前の一覧が、design の表と合う
- **終了コードの写しは `main` の 1 か所**。例外の種類ごとに 0 / 1 / 2 / 130 になり、
  traceback を出さない (`SERVE_DEBUG=1` のときだけ出す)
- **`serve push`** が、2 台に 6 つの置き場所を作ってから、`payload/` だけを配る
  (requirements 1.1、1.2)。巻き戻しは空 (置き場所は消さない)
- **了承** (requirements 2.1): 端末でなく `--yes` もないと、状態を変える呼び出しを 1 つも
  出さずに終了コード 1 になる。計画は stderr に見せる
- 各サブコマンドが、**正しい部品を、正しい引数で**呼ぶ (呼び出しの記録を見る)。記録の回収や
  照合の時刻は、その時の時刻を渡す (使い回さない)
- 標準出力は、後の処理が読める決まった形 (`key=value` の行と、`|` で始まる表) で、`detail` を
  必ず含む。進捗、計画、警告、誤りは stderr
- 数の引数は、`nan`、`inf`、0、負、単位つき (`2h`) まで、`argparse` の段で確かめる

端から端までの重い試験 (起動 → 停止) は 5.2、安全の決まりの全体は 5.3 で書く。ここでは、
`cli` が部品をつなぐところだけを見る。実物の ssh、rsync、docker には、どの段でもつながない
(`FakeRunner` と、HTTP だけの `fake_vllm` を相手にする)。試験は、実際に眠らない。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from fake_runner import FakeRunner, Reply, Rule
from fake_vllm import FakeVllm, Fault, MessagesReply
from meminfo_sample import meminfo_rule
from serving_kit import __version__, cli
from serving_kit import autostart as autostart_mod
from serving_kit import guards as g
from serving_kit import image as im
from serving_kit import lifecycle as lc
from serving_kit import logs as lg
from serving_kit import netcheck as nc
from serving_kit import probe as pr
from serving_kit import thinking as th
from serving_kit import types as kt
from serving_kit import watch as wt
from serving_kit import weights as wg
from serving_kit.config import ConfigError
from serving_kit.guards import ApprovalError
from serving_kit.plan import (
    LABEL_CONFIG,
    LABEL_CONFIG_SHA256,
    LABEL_IMAGE,
    LABEL_KIND,
    LABEL_OWNER,
    LABEL_ROLE,
    LABEL_STARTED_AT,
    OWNER,
    OWNER_FILTER,
)
from serving_kit.remote import RemoteError, RemoteRunner
from serving_kit.types import (
    GateResult,
    InterfaceLink,
    LinkReport,
    ManifestFile,
    NodeRole,
    NodeStatus,
    PlannedPush,
    PlannedRun,
    ProbeOutcome,
    ServiceStatus,
    SmokeOutcome,
    SmokeReply,
    StartOutcome,
    StopOutcome,
    ThinkingOutcome,
    ThinkingTrial,
    VerificationRecord,
    WatchOutcome,
    WeightsManifest,
)

# --- 見本の値 -------------------------------------------------------------

DIGEST = "b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
IMAGE_REF = f"vllm/vllm-openai@sha256:{DIGEST}"
REPO = "RedHatAI/GLM-5.3-Flash-NVFP4"
REVISION = "18d55bfd" + "0" * 32
SLUG = "RedHatAI__GLM-5.3-Flash-NVFP4"
MANIFEST_NAME = f"{SLUG}.manifest.json"
MEASURED = "docs/results/2026-09-21-netcheck-links.md"
SOURCE = "https://docs.vllm.ai/en/latest/cli/serve/"
QUOTE = "vllm serve [model_tag] [options]"
REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"
SERVED_MODEL = "glm-5-3-flash"
REPO_COMMIT = "5dd44fc" + "0" * 33
BODY_MARKER = "ZZ-SMOKE-BODY-MARKER-ZZ"
"""応答の本文にだけ現れる目印 (標準出力に出ないことを見る)。"""

SERVE_CONFIG = "p1-nvfp4-tp2"
PROBE_CONFIG = "probe-pinned"
FETCH_CONFIG = "p1-fetch-nvfp4"
INSPECT_CONFIG = "p1-image-licenses"
JOB_CONFIG = "p1-netcheck-bandwidth"

UNUSED_PORT = 8123
"""HTTP に進まない試験が使う、偽の推論サーバーのない番号。"""

ROLES: tuple[NodeRole, ...] = ("head", "worker")

FIRST_NOW = datetime(2026, 9, 22, 6, 0, 0, tzinfo=UTC)

FILE_SIZES: Mapping[str, int] = {
    "config.json": 4_096,
    "tokenizer.json": 36_000_000,
    "model-00001-of-00001.safetensors": 98_000_000_000,
}

MANIFEST = WeightsManifest(
    repo=REPO,
    revision=REVISION,
    generated_at=datetime(2026, 9, 22, 1, 0, 0, tzinfo=UTC),
    total_bytes=sum(FILE_SIZES.values()),
    files=tuple(
        ManifestFile(path=path, size=size, sha256=f"{index:064x}")
        for index, (path, size) in enumerate(sorted(FILE_SIZES.items()))
    ),
)

EXPECTED_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("check",),
    ("push",),
    ("pull-image",),
    ("image-licenses",),
    ("manifest",),
    ("derived-import",),
    ("fetch",),
    ("verify",),
    ("start",),
    ("stop",),
    ("status",),
    ("smoke",),
    ("logs",),
    ("probe",),
    ("netcheck",),
    ("netcheck", "links"),
    ("netcheck", "bandwidth"),
    ("netcheck", "sanity"),
    ("netcheck", "ab"),
    ("watch",),
    ("thinking",),
    ("autostart",),
    ("autostart", "set"),
    ("autostart", "clear"),
    ("autostart", "status"),
)
"""design.md 「入口 › cli」の表の、すべてのコマンド (drift の見張り。issue #88 で `autostart`
の 3 つの子を足した)。"""


# --- 試験用のリポジトリ ---------------------------------------------------


def _toml_value(text: str) -> str:
    """TOML の文字列にする (`"` を含む値は、リテラル文字列で書く)。"""
    if '"' in text and "'" not in text:
        return f"'{text}'"
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _setting(
    table: str,
    *,
    flag: str | None = None,
    value: str | None = None,
    why: str = "試験のための設定",
    only_on: str | None = None,
    is_port: bool = False,
    measured: bool = False,
) -> str:
    lines = [f"[{table}]"]
    if flag is not None:
        lines.append(f"flag = {_toml_value(flag)}")
    if value is not None:
        lines.append(f"value = {_toml_value(value)}")
    lines.append(f'why = "{why}"')
    if only_on is not None:
        lines.append(f'only_on = "{only_on}"')
    if is_port:
        lines.append("is_port = true")
    if measured:
        lines.append(f'measured = "{MEASURED}"')
    else:
        lines.append(f'source = "{SOURCE}"')
        lines.append(f'quote = "{QUOTE}"')
    return "\n".join(lines) + "\n\n"


def _header(
    name: str,
    kind: str,
    nodes: Sequence[str],
    *,
    served: str | None = None,
    weights: bool = True,
    mount_at: str = f"/models/{SLUG}",
) -> str:
    lines = [
        f"[configs.{name}]",
        f'kind = "{kind}"',
        f'description = "試験の構成 ({kind})"',
        "nodes = [" + ", ".join(f'"{role}"' for role in nodes) + "]",
        "ready_timeout_s = 1800",
    ]
    if served is not None:
        lines.append(f'served_model_name = "{served}"')
    text = "\n".join(lines) + "\n\n"
    text += (
        "\n".join(
            [
                f"[configs.{name}.image]",
                f'ref = "{IMAGE_REF}"',
                'seen_as = "vllm/vllm-openai:glm53-flash-arm64-cu130"',
                "size_bytes = 9666567584",
                'source = "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags"',
                'quote = "glm53-flash-arm64-cu130 / linux/arm64 / 9666567584 B"',
            ]
        )
        + "\n\n"
    )
    if weights:
        text += (
            "\n".join(
                [
                    f"[configs.{name}.weights]",
                    f'repo = "{REPO}"',
                    f'revision = "{REVISION}"',
                    f'manifest = "{MANIFEST_NAME}"',
                    f'mount_at = "{mount_at}"',
                    f'source = "https://huggingface.co/{REPO}"',
                    'quote = "license: mit"',
                ]
            )
            + "\n\n"
        )
    return text


def _serve_config(port: int) -> str:
    name = SERVE_CONFIG
    text = _header(name, "serve", ROLES, served=SERVED_MODEL)
    text += _setting(
        f"configs.{name}.docker.models",
        flag="--mount",
        value="type=bind,source={remote_root}/models/" + SLUG + ",target={weights.mount_at}"
        ",readonly",
    )
    text += _setting(
        f"configs.{name}.docker.cache",
        flag="--mount",
        value="type=bind,source={remote_root}/cache,target=/root/.cache",
    )
    text += _setting(f"configs.{name}.args.model-path", value="{weights.mount_at}")
    text += _setting(
        f"configs.{name}.args.served-model-name", flag="--served-model-name", value=SERVED_MODEL
    )
    text += _setting(f"configs.{name}.args.host", flag="--host", value="{head.lan_addr}")
    text += _setting(f"configs.{name}.args.port", flag="--port", value=str(port), is_port=True)
    text += _setting(
        f"configs.{name}.args.master-port", flag="--master-port", value="29501", is_port=True
    )
    text += _setting(
        f"configs.{name}.env.host-ip",
        flag="VLLM_HOST_IP",
        value="{node.fabric_addr}",
        measured=True,
    )
    return text


def _probe_config(port: int) -> str:
    name = PROBE_CONFIG
    text = _header(name, "probe", ("head",), served=SERVED_MODEL, mount_at=f"/probe/{SLUG}")
    text += _setting(
        f"configs.{name}.docker.probe",
        flag="--mount",
        value="type=bind,source={remote_root}/probe,target=/probe,readonly",
    )
    text += _setting(f"configs.{name}.args.model-path", value="{weights.mount_at}")
    text += _setting(f"configs.{name}.args.port", flag="--port", value=str(port), is_port=True)
    return text


def _fetch_config() -> str:
    name = FETCH_CONFIG
    text = _header(name, "fetch", ROLES)
    text += _setting(f"configs.{name}.docker.entrypoint", flag="--entrypoint", value="hf")
    text += _setting(
        f"configs.{name}.docker.models",
        flag="--mount",
        value="type=bind,source={remote_root}/models,target=/models",
    )
    text += _setting(f"configs.{name}.args.download", value="download")
    text += _setting(f"configs.{name}.args.repo", value=REPO)
    text += _setting(f"configs.{name}.args.revision", flag="--revision", value=REVISION)
    text += _setting(
        f"configs.{name}.args.local-dir", flag="--local-dir", value="{weights.mount_at}"
    )
    text += _setting(f"configs.{name}.env.telemetry", flag="HF_HUB_DISABLE_TELEMETRY", value="1")
    return text


def _inspect_config() -> str:
    name = INSPECT_CONFIG
    text = _header(name, "inspect", ("head",), weights=False)
    text += _setting(f"configs.{name}.docker.entrypoint", flag="--entrypoint", value="cat")
    text += _setting(f"configs.{name}.args.license", value="/usr/share/doc/vllm/LICENSE")
    return text


def _job_config() -> str:
    name = JOB_CONFIG
    text = _header(name, "job", ROLES, weights=False)
    text += _setting(f"configs.{name}.docker.network", flag="--network", value="host")
    text += _setting(
        f"configs.{name}.docker.logs",
        flag="--mount",
        value="type=bind,source={remote_root}/logs,target=/logs",
    )
    text += _setting(f"configs.{name}.args.script", value="/payload/allreduce_bench.py")
    text += _setting(
        f"configs.{name}.args.master-port", flag="--master-port", value="29502", is_port=True
    )
    text += _setting(f"configs.{name}.env.nccl-debug", flag="NCCL_DEBUG", value="INFO")
    text += _setting(
        f"configs.{name}.env.nccl-debug-subsys", flag="NCCL_DEBUG_SUBSYS", value="INIT,NET"
    )
    text += _setting(
        f"configs.{name}.env.nccl-debug-file", flag="NCCL_DEBUG_FILE", value="/logs/nccl.log"
    )
    return text


NODES_TOML = f"""\
[nodes.head]
ssh_host = "spark-153d"
lan_addr = "127.0.0.1"
remote_root = "{REMOTE_ROOT}"
fabric_addr = "192.168.100.1"
fabric_ifname = "enp1s0f0np0"
fabric_measured = "{MEASURED}"

[nodes.worker]
ssh_host = "spark-5083"
lan_addr = "127.0.0.1"
remote_root = "{REMOTE_ROOT}"
fabric_addr = "192.168.100.2"
fabric_ifname = "enp1s0f0np0"
fabric_measured = "{MEASURED}"
"""


@dataclass(frozen=True)
class Repo:
    """試験用のリポジトリ (`serving/config/` と `serving/weights/` を持つ)。"""

    root: Path

    @property
    def serving(self) -> Path:
        return self.root / "serving"

    @property
    def configs(self) -> Path:
        return self.serving / "config" / "configs.toml"

    @property
    def nodes(self) -> Path:
        return self.serving / "config" / "nodes.toml"

    @property
    def var_root(self) -> Path:
        return self.serving / "var"

    @property
    def payload(self) -> Path:
        return self.serving / "payload"

    @property
    def manifest(self) -> Path:
        return self.serving / "weights" / MANIFEST_NAME


def make_repo(tmp_path: Path, *, port: int = UNUSED_PORT) -> Repo:
    """構成、ノード、マニフェスト、配るスクリプトの揃った、最小のリポジトリを作る。"""
    root = tmp_path / "repo"
    (root / "docs" / "results").mkdir(parents=True, exist_ok=True)
    (root / MEASURED).write_text("直結のインターフェースの実測の要約\n", encoding="utf-8")
    (root / "serving" / "config").mkdir(parents=True, exist_ok=True)
    (root / "serving" / "weights").mkdir(parents=True, exist_ok=True)
    (root / "serving" / "payload").mkdir(parents=True, exist_ok=True)
    (root / "serving" / "payload" / "allreduce_bench.py").write_text("# 試験\n", encoding="utf-8")
    repo = Repo(root=root)
    repo.manifest.write_bytes(wg.to_json_bytes(MANIFEST))
    text = "schema_version = 1\n\n"
    text += _serve_config(port)
    text += _probe_config(port)
    text += _fetch_config()
    text += _inspect_config()
    text += _job_config()
    repo.configs.write_text(text, encoding="utf-8")
    repo.nodes.write_text(NODES_TOML, encoding="utf-8")
    return repo


# --- 呼び出しの記録 -------------------------------------------------------


@dataclass
class Recorder:
    """部品の代わりに呼ばれて、引数を記録する (返す値と投げる例外は、試験が決める)。"""

    result: Any = None
    error: BaseException | None = None
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result

    @property
    def args(self) -> tuple[Any, ...]:
        return self.calls[-1][0]

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.calls[-1][1]


@dataclass
class Spy:
    """`cli` が作る実行役を、偽物に差し替えて覚えておく。"""

    script: tuple[Rule, ...] = ()
    default: Reply | None = Reply()
    made: list[FakeRunner] = field(default_factory=list)

    def __call__(self, var_root: Path) -> FakeRunner:
        runner = FakeRunner(var_root=var_root, script=self.script, default=self.default)
        self.made.append(runner)
        return runner

    @property
    def runner(self) -> FakeRunner:
        assert self.made, "実行役が 1 つも作られていない"
        return self.made[-1]


@dataclass
class Run:
    """1 回の `main` の結果。"""

    code: int
    out: str
    err: str
    spy: Spy


def stepping_clock(start: datetime = FIRST_NOW) -> Callable[[], datetime]:
    """呼ばれるたびに 1 秒進む時計 (時刻の使い回しを見つけるため)。"""
    state = {"n": 0}

    def now() -> datetime:
        moment = start + timedelta(seconds=state["n"])
        state["n"] += 1
        return moment

    return now


def run(
    argv: Sequence[str],
    repo: Repo,
    *,
    spy: Spy | None = None,
    client_factory: Callable[[float], httpx.Client] | None = None,
    confirmer: Any = None,
    stdin: str | None = None,
    now: Callable[[], datetime] | None = None,
    repo_facts: tuple[str, bool] | None = (REPO_COMMIT, False),
    var_root: Path | None = None,
) -> Run:
    """共通の引数を足して `main` を呼ぶ (標準出力と標準エラーは、文字列で受ける)。"""
    used = Spy() if spy is None else spy
    out = io.StringIO()
    err = io.StringIO()
    full = [
        *argv,
        "--configs",
        str(repo.configs),
        "--nodes",
        str(repo.nodes),
        "--var-root",
        str(repo.var_root if var_root is None else var_root),
    ]
    code = cli.main(
        full,
        runner_factory=used,
        client_factory=client_factory,
        confirmer=confirmer,
        stdin=io.StringIO("" if stdin is None else stdin),
        stdout=out,
        stderr=err,
        repo_root=repo.root,
        repo_facts=repo_facts,
        now=stepping_clock() if now is None else now,
        sleep=_no_sleep,
    )
    return Run(code=code, out=out.getvalue(), err=err.getvalue(), spy=used)


def _no_sleep(seconds: float) -> None:
    """眠らない (待ちの上限の試験が、実際の時間を待たないため)。"""
    return None


def kv(out: str) -> dict[str, str]:
    """標準出力の `key=value` の行を読む (表の行は飛ばす)。"""
    pairs: dict[str, str] = {}
    for line in out.splitlines():
        if not line or line.startswith("|"):
            continue
        assert "=" in line, f"決まった形でない標準出力の行: {line!r}"
        key, value = line.split("=", 1)
        pairs[key] = value
    return pairs


def table_rows(out: str) -> tuple[tuple[str, ...], ...]:
    """標準出力の表の行 (`|` で囲んだもの) を、欄の列にする。"""
    rows: list[tuple[str, ...]] = []
    for line in out.splitlines():
        if line.startswith("|"):
            rows.append(tuple(cell.strip() for cell in line.strip("|").split("|")))
    return tuple(rows)


@pytest.fixture(autouse=True)
def no_real_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """実物の ssh / rsync / docker / git を呼ばないことと、実際に眠らないことを固定する。"""

    def explode(argv: Sequence[str], **kwargs: Any) -> None:
        raise AssertionError(f"実物のコマンドを呼んだ: {list(argv)}")

    def no_sleep(seconds: float) -> None:
        raise AssertionError(f"試験が実際に眠った: {seconds} 秒")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(time, "sleep", no_sleep)
    yield


# --- 骨組み (task 1.1 から引き継ぐ振る舞い) --------------------------------


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """`serve --help` は 0 で終わり、使い方を標準出力に出す (design.md cli)。"""
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "serve" in capsys.readouterr().out


def test_version_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """`serve --version` は 0 で終わり、パッケージの版を標準出力に出す。"""
    with pytest.raises(SystemExit) as caught:
        cli.main(["--version"])
    assert caught.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_returns_precondition(capsys: pytest.CaptureFixture[str]) -> None:
    """サブコマンドを選ばずに呼ぶと、断って 1 で終わる。"""
    assert cli.main([]) == cli.EXIT_PRECONDITION
    assert "実行するコマンドを 1 つ選ぶこと" in capsys.readouterr().err


def test_unknown_argument_exits_with_usage_error() -> None:
    """`argparse` が扱わない引数は、`SystemExit(2)` としてそのまま伝播する。"""
    with pytest.raises(SystemExit) as caught:
        cli.main(["--no-such-flag"])
    assert caught.value.code == 2


def test_build_parser_prog_is_serve() -> None:
    parser = cli.build_parser()
    assert parser.prog == "serve"


# --- すべてのサブコマンドの --help -----------------------------------------


def subcommand_paths(parser: argparse.ArgumentParser) -> tuple[tuple[str, ...], ...]:
    """引数解析器を歩いて、すべてのサブコマンドの道筋を集める。"""
    found: list[tuple[str, ...]] = []

    def walk(current: argparse.ArgumentParser, prefix: tuple[str, ...]) -> None:
        for action in current._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for name, sub in action.choices.items():
                found.append((*prefix, name))
                walk(sub, (*prefix, name))

    walk(parser, ())
    return tuple(found)


def subcommand_parser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    """サブコマンド 1 つの引数解析器を取る (引数の説明を読むため)。"""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction) and name in action.choices:
            found = action.choices[name]
            assert isinstance(found, argparse.ArgumentParser)
            return found
    raise AssertionError(f"サブコマンド {name} がない")


def test_the_commands_are_the_ones_in_the_design_table() -> None:
    """使える名前が、design.md 「入口 › cli」の表と合う (drift の見張り)。"""
    assert set(subcommand_paths(cli.build_parser())) == set(EXPECTED_COMMANDS)


@pytest.mark.parametrize("path", EXPECTED_COMMANDS, ids=[" ".join(p) for p in EXPECTED_COMMANDS])
def test_every_subcommand_help_exits_zero(
    path: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    """すべてのサブコマンドの `--help` が 0 で終わる (tasks.md 5.1 の完了の状態)。"""
    with pytest.raises(SystemExit) as caught:
        cli.main([*path, "--help"])
    assert caught.value.code == 0
    assert capsys.readouterr().out.strip() != ""


def test_netcheck_without_a_subcommand_is_a_precondition(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`serve netcheck` だけでは、何を確かめるか決まらないので 1 で終わる。"""
    assert cli.main(["netcheck"]) == cli.EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "netcheck" in err
    assert "serve netcheck は、links / bandwidth / sanity / ab の 1 つを選ぶこと。" in err


def test_autostart_without_a_subcommand_is_a_precondition(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """[SCN-G-N1] `serve autostart` だけでは、set/clear/status のどれか決まらないので、
    `_NESTED_HANDLERS` の鍵から組み立てた案内とともに、終了コード 1 で終わる。
    """
    assert cli.main(["autostart"]) == cli.EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "autostart" in err
    assert "serve autostart は、set / clear / status の 1 つを選ぶこと。" in err


# --- 終了コードの写し (main の 1 か所) -------------------------------------

_ERRORS: tuple[tuple[BaseException, int], ...] = (
    (ConfigError("根拠のない設定"), cli.EXIT_PRECONDITION),
    (ApprovalError("了承されなかった"), cli.EXIT_PRECONDITION),
    (
        RemoteError("つながらない", node="head", ssh_host="spark-153d", argv=("docker", "version")),
        cli.EXIT_PRECONDITION,
    ),
    (ValueError("引数が足りない"), cli.EXIT_PRECONDITION),
    (wg.WeightsRefError("マニフェストが違う"), cli.EXIT_PRECONDITION),
    (im.ImageError("取得が失敗した"), cli.EXIT_FAILED),
    (wg.WeightsFetchError("取得が失敗した"), cli.EXIT_FAILED),
    (wg.WeightsMismatchError("照合が合わない"), cli.EXIT_FAILED),
    (lc.LifecycleError("起こせなかった"), cli.EXIT_FAILED),
    (pr.ProbeError("片付けが終わらなかった"), cli.EXIT_FAILED),
    (nc.NetcheckError("片付けが終わらなかった"), cli.EXIT_FAILED),
    (cli.PushError("配布できなかった"), cli.EXIT_FAILED),
    (autostart_mod.AutostartError("配布できなかった"), cli.EXIT_FAILED),
)


@pytest.mark.parametrize(("error", "expected"), _ERRORS, ids=[type(e).__name__ for e, _ in _ERRORS])
def test_main_maps_each_error_to_one_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException, expected: int
) -> None:
    """例外の種類ごとの終了コードを、`main` の 1 か所で写す (design.md Error Handling)。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(g, "run_gates", Recorder(error=error))
    result = run(["check", SERVE_CONFIG], repo)
    assert result.code == expected
    assert result.err.startswith("エラー: ") or "\nエラー: " in result.err
    assert "Traceback" not in result.err


def test_an_error_is_one_line_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """誤りは、traceback を出さずに、1 行の文で示す (tasks.md 5.1)。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(g, "run_gates", Recorder(error=ValueError("行が\n2 つある文")))
    result = run(["check", SERVE_CONFIG], repo)
    assert result.code == cli.EXIT_PRECONDITION
    shown = [line for line in result.err.splitlines() if line.startswith("エラー: ")]
    assert len(shown) == 1
    assert "\\n" in shown[0]


def test_an_interrupt_is_one_hundred_thirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """中断は 130 (design.md Error Handling)。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(g, "run_gates", Recorder(error=KeyboardInterrupt()))
    result = run(["check", SERVE_CONFIG], repo)
    assert result.code == cli.EXIT_INTERRUPTED
    assert "中断" in result.err


def test_an_unexpected_error_is_a_failure_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """予期しない例外も、1 行にして 2 で終わる (traceback は出さない)。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(g, "run_gates", Recorder(error=RuntimeError("許可の一覧にない")))
    result = run(["check", SERVE_CONFIG], repo)
    assert result.code == cli.EXIT_FAILED
    assert "予期しない" in result.err
    assert "Traceback" not in result.err


def test_the_debug_switch_shows_the_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`SERVE_DEBUG=1` のときだけ、traceback を出す口を 1 つ持つ。"""
    repo = make_repo(tmp_path)
    monkeypatch.setenv(cli.DEBUG_ENV, "1")
    monkeypatch.setattr(g, "run_gates", Recorder(error=RuntimeError("許可の一覧にない")))
    result = run(["check", SERVE_CONFIG], repo)
    assert result.code == cli.EXIT_FAILED
    assert "Traceback" in result.err


# --- 共通の引数 -----------------------------------------------------------


def test_the_default_config_paths_point_into_serving(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """`--configs` / `--nodes` を省くと、`serving/config/` の下を見る (design.md cli)。

    **主に構造で確かめる**。もとは、`check` を既定のまま呼んで、誤りの文に出る道筋を見ていた
    が、それは「既定のファイルが実在しない」ことに寄りかかった見方だった (6.2 が `configs.toml`
    と `nodes.toml` をコミットしたので、誤りは 1 段先の「直結の値がない」に変わる)。そこで
    (i) argparse が既定を決めずに `None` のままにすること、(ii) 2 つの引数の説明が既定の道筋を
    名指しすること、(iii) その道筋が `serving_dir()` の下に実在すること、の 3 つで見る。

    (iv) として、**既定の場所から実際に読んだ証拠**も 1 つ置く: 既定の `--configs` のまま、
    `--nodes` にだけ直結の値を落とした写しを渡して `check` を呼ぶと、`nodes.head.fabric_addr`
    を理由に断られる。この断りは `config.select_config` が出すもので、`_cmd_check` が
    `ctx.runner()` に触るより前である (7.3 でコミットした `nodes.toml` は埋まっているので、
    既定の `--nodes` のままでは関門に進んでしまう。写しで止める)。

    **Spark に触らないことは、構造で保証する**: 遠隔の実行役の工場を、呼ばれたら落ちるものに
    差し替えて渡す。
    """
    parser = cli.build_parser()
    args = parser.parse_args(["check", SERVE_CONFIG])
    # (i) 既定は argparse では決めない (あとで serving_dir() から組み立てる)
    assert args.configs is None
    assert args.nodes is None

    # (ii) 2 つの引数の説明が、既定の道筋を名指ししている
    check_help = subcommand_parser(parser, "check").format_help()
    for name in ("configs.toml", "nodes.toml"):
        assert f"serving/config/{name}" in check_help, name

    # (iii) その道筋が、`serving/` の下に実在する
    serving = cli.serving_dir()
    for name in ("configs.toml", "nodes.toml"):
        path = serving / "config" / name
        assert path.is_file(), path
        assert path.relative_to(serving) == Path("config") / name

    # (iv) 既定の場所から実際に読んでいる。遠隔の実行役の工場は、呼ばれたら落ちるものを渡す
    # ので、この道が Spark に触った時点で試験が落ちる (SshRunner を作らせない)
    def no_spark(var_root: Path) -> RemoteRunner:
        raise AssertionError("既定の道筋の試験は、Spark に触らない")

    unmeasured = tmp_path / "nodes.toml"
    committed = (cli.serving_dir() / "config" / "nodes.toml").read_text(encoding="utf-8")
    unmeasured.write_text(
        "\n".join(line for line in committed.splitlines() if not line.startswith("fabric_")) + "\n",
        encoding="utf-8",
    )
    code = cli.main(["check", SERVE_CONFIG, "--nodes", str(unmeasured)], runner_factory=no_spark)
    assert code == cli.EXIT_PRECONDITION
    assert "nodes.head.fabric_addr" in capsys.readouterr().err


def test_the_serving_dir_is_two_levels_above_the_package() -> None:
    """`serving/` の位置は、パッケージの位置から 2 つ上として求める。"""
    assert cli.serving_dir() == Path(cli.__file__).resolve().parents[2]
    assert cli.serving_dir().name == "serving"
    assert (cli.serving_dir() / "pyproject.toml").is_file()


def test_the_default_var_root_is_ignored_by_git() -> None:
    """記録の既定の置き場所 (`serving/var/`) が、git の管理の外にある。"""
    ignore = (cli.serving_dir().parent / ".gitignore").read_text(encoding="utf-8")
    assert "serving/var/" in ignore.splitlines()


def test_an_unknown_config_name_is_a_precondition(tmp_path: Path) -> None:
    """知らない構成の名前は、使える名前を添えて断られる (終了コード 1)。"""
    repo = make_repo(tmp_path)
    result = run(["check", "no-such-config"], repo)
    assert result.code == cli.EXIT_PRECONDITION
    assert SERVE_CONFIG in result.err


# --- 数の引数 -------------------------------------------------------------


@pytest.mark.parametrize("text", ["nan", "inf", "-inf", "0", "-1", "", "2x", "1e999"])
def test_a_degenerate_number_is_refused_before_anything_runs(tmp_path: Path, text: str) -> None:
    """数の引数は、`argparse` の段で断る (片側だけの比較では、NaN と inf が通る)。"""
    repo = make_repo(tmp_path)
    with pytest.raises(SystemExit) as caught:
        run(["start", SERVE_CONFIG, "--timeout", text, "--yes"], repo)
    assert caught.value.code == 2


@pytest.mark.parametrize(
    ("text", "seconds"), [("30", 30.0), ("30s", 30.0), ("5m", 300.0), ("2h", 7200.0)]
)
def test_seconds_accept_a_unit_suffix(text: str, seconds: float) -> None:
    """`--duration 2h` のような単位も受ける (design.md 「確認 › watch」)。"""
    assert cli.seconds(text) == seconds


@pytest.mark.parametrize("text", ["0", "-1", "1.5", "nan", "true"])
def test_a_degenerate_count_is_refused(text: str) -> None:
    """回数は、bool を除く本物の int で、1 以上だけを受ける。"""
    with pytest.raises(argparse.ArgumentTypeError):
        cli.count(text)


def test_repeats_need_at_least_two(tmp_path: Path) -> None:
    """A/B の回数 (`--repeat`) は 2 以上 (1 回ずつでは、範囲が重なるかを見られない)。"""
    with pytest.raises(argparse.ArgumentTypeError):
        cli.repeats("1")
    assert cli.repeats("3") == 3


@pytest.mark.parametrize("text", ["-1", "101", "nan", "1.5"])
def test_a_percentage_outside_the_range_is_refused(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        cli.percent(text)


def test_an_env_pair_needs_a_key_and_a_value() -> None:
    """`--env KEY=VALUE` は、両方そろっていなければ断る。"""
    assert cli.env_pair("NCCL_IB_HCA=rocep1s0f0") == ("NCCL_IB_HCA", "rocep1s0f0")
    for bad in ("NO_EQUALS", "=value", " KEY=value"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.env_pair(bad)


# --- serve check ----------------------------------------------------------


def _gates(*results: GateResult) -> Recorder:
    return Recorder(result=tuple(results))


def test_check_lists_every_gate_as_key_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`serve check` は、関門ごとの結果を `key=value` で並べる。"""
    repo = make_repo(tmp_path)
    recorder = _gates(
        GateResult(gate="reachable", node="head", passed=True, detail="入れた"),
        GateResult(gate="disk_space", node="worker", passed=True, detail="空きは足りる"),
    )
    monkeypatch.setattr(g, "run_gates", recorder)
    result = run(["check", SERVE_CONFIG], repo)
    assert result.code == cli.EXIT_OK
    pairs = kv(result.out)
    assert pairs["config"] == SERVE_CONFIG
    assert pairs["gate.1.name"] == "reachable"
    assert pairs["gate.1.node"] == "head"
    assert pairs["gate.1.passed"] == "true"
    assert pairs["gate.1.detail"] == "入れた"
    assert pairs["gate.2.name"] == "disk_space"
    assert pairs["gates_failed"] == "0"
    assert pairs["status"] == "passed"


def test_check_fails_when_any_gate_does_not_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """関門が 1 つでも不通過なら 1 (前提の不足)。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(
        g,
        "run_gates",
        _gates(GateResult(gate="gpu_idle", node="head", passed=False, detail="ほかが使っている")),
    )
    result = run(["check", SERVE_CONFIG], repo)
    assert result.code == cli.EXIT_PRECONDITION
    pairs = kv(result.out)
    assert pairs["gates_failed"] == "1"
    assert pairs["status"] == "refused"


def test_check_passes_the_manifest_of_a_config_with_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """重みを持つ構成では、コミットしたマニフェストを関門に渡す。"""
    repo = make_repo(tmp_path)
    recorder = _gates()
    monkeypatch.setattr(g, "run_gates", recorder)
    run(["check", SERVE_CONFIG], repo)
    assert recorder.kwargs["manifest"] == MANIFEST
    config = recorder.args[1]
    assert config.name == SERVE_CONFIG
    plans = recorder.args[3]
    assert tuple(one.node for one in plans) == ROLES


def test_check_passes_no_manifest_for_a_config_without_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    recorder = _gates()
    monkeypatch.setattr(g, "run_gates", recorder)
    run(["check", INSPECT_CONFIG], repo)
    assert recorder.kwargs["manifest"] is None


def test_check_makes_no_mutating_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """検査は、状態を変える呼び出しを 1 つも出さない (了承も求めない)。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(g, "run_gates", _gates())
    result = run(["check", SERVE_CONFIG], repo)
    assert [call for call in result.spy.runner.calls if call.mutating] == []


# --- serve push -----------------------------------------------------------

SPARK_DIRS = ("payload", "models", "probe", "cache", "logs", "state")


def test_push_creates_the_six_directories_then_pushes_only_the_payload(tmp_path: Path) -> None:
    """6 つの置き場所を作ってから、`payload/` だけを配る (tasks.md 5.1 の完了の状態)。"""
    repo = make_repo(tmp_path)
    result = run(["push", "--yes"], repo)
    assert result.code == cli.EXIT_OK
    runner = result.spy.runner
    assert [call.kind for call in runner.calls] == ["run", "run", "push", "push"]
    for call in runner.runs:
        assert call.argv[:2] == ("mkdir", "-p")
        assert call.argv[2:] == tuple(f"{REMOTE_ROOT}/{name}" for name in SPARK_DIRS)
    assert [(call.node, call.remote) for call in runner.pushes] == [
        ("head", "payload"),
        ("worker", "payload"),
    ]
    assert all(call.local_dir == repo.payload for call in runner.pushes)
    assert all(not call.delete for call in runner.pushes)


def test_push_makes_the_directories_on_both_nodes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result = run(["push", "--yes"], repo)
    assert [call.node for call in result.spy.runner.runs] == ["head", "worker"]


def test_push_has_an_empty_rollback(tmp_path: Path) -> None:
    """巻き戻しは空 (置き場所は消さない)。"""
    repo = make_repo(tmp_path)
    result = run(["push", "--yes"], repo)
    plan = result.spy.runner.plan
    assert plan is not None
    assert plan.rollback == ()
    assert plan.own_container_names == ()


def test_push_runs_exactly_the_plan_it_showed(tmp_path: Path) -> None:
    """**見せた計画と、流れた呼び出しが、1 対 1 で同じ**である (requirements 2.1)。

    計画と実行が別々に書かれていると、計画にだけ宛先を足す変異が、どちらの側の試験にも
    掛からない。ここで、計画の中身 (`mkdir -p` ×2 台 + `payload` への配布 ×2 台 ちょうど) と、
    実際に流れた呼び出しの宛先が同じであることを、両方から固定する。
    """
    repo = make_repo(tmp_path)
    result = run(["push", "--yes"], repo)
    plan = result.spy.runner.plan
    assert plan is not None

    runs = [one for one in plan.forward if isinstance(one, PlannedRun)]
    pushes = [one for one in plan.forward if isinstance(one, PlannedPush)]
    assert len(plan.forward) == 4
    assert [one.node for one in runs] == ["head", "worker"]
    assert all(one.argv[:2] == ("mkdir", "-p") for one in runs)
    assert {one.remote_subdir for one in pushes} == {"payload"}
    assert [one.node for one in pushes] == ["head", "worker"]
    assert all(one.local_dir == repo.payload and not one.delete for one in pushes)

    runner = result.spy.runner
    assert [(call.node, call.argv) for call in runner.runs] == [
        (one.node, one.argv) for one in runs
    ]
    assert [(call.node, call.remote, call.local_dir) for call in runner.pushes] == [
        (one.node, one.remote_subdir, one.local_dir) for one in pushes
    ]


def test_push_shows_the_plan_on_stderr(tmp_path: Path) -> None:
    """計画は stderr に見せる (標準出力は、後の処理が読む)。"""
    repo = make_repo(tmp_path)
    result = run(["push", "--yes"], repo)
    assert "mkdir" in result.err
    assert "spark-153d" in result.err
    assert "mkdir" not in result.out


def test_push_without_a_terminal_and_without_yes_changes_nothing(tmp_path: Path) -> None:
    """端末でなく `--yes` もないと、状態を変える呼び出しを 1 つも出さずに 1 で終わる。"""
    repo = make_repo(tmp_path)
    result = run(["push"], repo)
    assert result.code == cli.EXIT_PRECONDITION
    assert result.spy.runner.calls == ()
    assert "--yes" in result.err


def test_push_stops_when_the_answer_is_not_yes(tmp_path: Path) -> None:
    """端末で `yes` 以外を打つと、状態を変える呼び出しが 1 つも出ない。"""
    repo = make_repo(tmp_path)

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    out = io.StringIO()
    err = io.StringIO()
    spy = Spy()
    code = cli.main(
        [
            "push",
            "--configs",
            str(repo.configs),
            "--nodes",
            str(repo.nodes),
            "--var-root",
            str(repo.var_root),
        ],
        runner_factory=spy,
        stdin=Terminal("no\n"),
        stdout=out,
        stderr=err,
        repo_root=repo.root,
        repo_facts=(REPO_COMMIT, False),
    )
    assert code == cli.EXIT_PRECONDITION
    assert spy.runner.calls == ()


def test_push_fails_when_a_directory_cannot_be_made(tmp_path: Path) -> None:
    """`mkdir` が 0 以外で終わると、実行しての失敗 (2) になる。"""
    repo = make_repo(tmp_path)
    spy = Spy(script=(Rule(prefix=("mkdir",), replies=(Reply(exit_code=1, stderr="だめ"),)),))
    result = run(["push", "--yes"], repo, spy=spy)
    assert result.code == cli.EXIT_FAILED
    assert result.spy.runner.pushes == ()


def test_push_fails_when_the_payload_cannot_be_sent(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    spy = Spy(script=(Rule(kind="push", replies=(Reply(exit_code=23, stderr="だめ"),)),))
    result = run(["push", "--yes"], repo, spy=spy)
    assert result.code == cli.EXIT_FAILED


def test_push_says_what_it_did_in_key_value(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result = run(["push", "--yes"], repo)
    pairs = kv(result.out)
    assert pairs["status"] == "pushed"
    assert pairs["directories"] == ",".join(SPARK_DIRS)
    assert pairs["nodes"] == "head,worker"
    assert pairs["detail"] != ""


# --- 了承なしでは、どのコマンドも状態を変えない (requirements 2.1) ----------

DF_AVAIL = "        Avail\n900000000000\n"
"""`df -B1 --output=avail` の出力 (見本 `tests/fixtures/spark/*/df-avail.txt` の形)。"""

CONTAINER_IDS: Mapping[NodeRole, str] = {"head": "0123456789ab", "worker": "cdef01234567"}

STATE_CHANGING: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("push", ("push",)),
    ("pull-image", ("pull-image", SERVE_CONFIG)),
    ("image-licenses", ("image-licenses", INSPECT_CONFIG)),
    ("fetch", ("fetch", FETCH_CONFIG)),
    ("verify", ("verify", SERVE_CONFIG)),
    ("start", ("start", SERVE_CONFIG)),
    ("stop", ("stop",)),
    ("probe", ("probe", PROBE_CONFIG)),
    ("netcheck bandwidth", ("netcheck", "bandwidth", JOB_CONFIG)),
    ("netcheck sanity", ("netcheck", "sanity", JOB_CONFIG)),
    ("netcheck ab", ("netcheck", "ab", JOB_CONFIG, "--env", "NCCL_IB_HCA=rocep1s0f0")),
)
"""design.md 「入口 › cli」の表で「変える」と書いた 11 のコマンド。"""


def _record_json(role: NodeRole, scope: str) -> str:
    """その台の、重みの照合の記録 (関門 `weights_verified` が読む形)。"""
    files = wg.scoped_files(MANIFEST, scope)  # type: ignore[arg-type]
    return VerificationRecord(
        repo=REPO,
        revision=REVISION,
        scope=scope,  # type: ignore[arg-type]
        node=role,
        verified_at=datetime(2026, 9, 22, 1, 30, tzinfo=UTC),
        file_count=len(files),
        total_bytes=sum(one.size for one in files),
    ).model_dump_json()


def own_row(role: NodeRole) -> str:
    """自分のラベルで絞った一覧 (`docker ps -a --format json`) の 1 行。"""
    labels = {
        LABEL_OWNER: OWNER,
        LABEL_CONFIG: SERVE_CONFIG,
        LABEL_KIND: "serve",
        LABEL_ROLE: role,
        LABEL_IMAGE: IMAGE_REF,
        LABEL_STARTED_AT: "2026-09-22T03:00:00Z",
        LABEL_CONFIG_SHA256: "c0ffee" + "0" * 58,
    }
    return (
        json.dumps(
            {
                "ID": CONTAINER_IDS[role],
                "Names": f"vb-{SERVE_CONFIG}-{role}",
                "State": "running",
                "Image": IMAGE_REF,
                "Labels": ",".join(f"{key}={value}" for key, value in sorted(labels.items())),
            }
        )
        + "\n"
    )


def gate_rules(*, listing: Mapping[NodeRole, str] | None = None) -> tuple[Rule, ...]:
    """9 つの関門の読み取りを、すべて通す台本 (どのコマンドも、了承の段まで届く)。

    読み取りだけの台本なので、これで「了承の段まで届いたのに、状態を変える呼び出しが
    1 つも出ない」ことを確かめられる。`/proc/meminfo` の規則は、照合の記録の `cat` の規則より
    前に置く (`cat` の規則は何にでも当たる)。
    """
    rows = {} if listing is None else listing
    rules: list[Rule] = []
    for role in ROLES:
        rules.append(
            Rule(
                prefix=("docker", "ps"),
                node=role,
                replies=(Reply(stdout=rows.get(role, "")),),
            )
        )
        rules.append(meminfo_rule(node=role))
        rules.append(
            Rule(
                prefix=("cat",),
                node=role,
                when=lambda argv: argv[-1].endswith(".probe.verified.json"),
                replies=(Reply(stdout=_record_json(role, "probe_files")),),
            )
        )
        rules.append(
            Rule(prefix=("cat",), node=role, replies=(Reply(stdout=_record_json(role, "all")),))
        )
    rules.extend(
        (
            Rule(prefix=("uname",), replies=(Reply(stdout="spark-153d\n"),)),
            Rule(prefix=("nvidia-smi",), replies=(Reply(stdout=""),)),
            Rule(prefix=("test", "-e"), replies=(Reply(),)),
            Rule(
                prefix=("docker", "image", "inspect"),
                replies=(Reply(stdout=json.dumps([IMAGE_REF])),),
            ),
            Rule(prefix=("df",), replies=(Reply(stdout=DF_AVAIL),)),
            Rule(prefix=("ss",), replies=(Reply(stdout=""),)),
        )
    )
    return tuple(rules)


def gate_rows(out: str) -> list[dict[str, str]]:
    """`serve check` の標準出力から、関門の結果 (`gate.N.*`) を、並んでいる順に取り出す。"""
    pairs = kv(out)
    rows: list[dict[str, str]] = []
    index = 1
    while f"gate.{index}.name" in pairs:
        rows.append(
            {
                "name": pairs[f"gate.{index}.name"],
                "node": pairs[f"gate.{index}.node"],
                "passed": pairs[f"gate.{index}.passed"],
            }
        )
        index += 1
    return rows


def test_check_shows_the_memory_free_gate_once_per_node(tmp_path: Path) -> None:
    """`serve check` は、関門 `memory_free` の結果を、台ごとに 1 件並べる (読み取りだけの台本で
    通る。状態を変える呼び出しは出ない)。"""
    repo = make_repo(tmp_path)
    spy = Spy(script=gate_rules(), default=Reply())

    result = run(["check", SERVE_CONFIG], repo, spy=spy)

    assert result.code == cli.EXIT_OK, result.out
    shown = [
        (row["node"], row["passed"])
        for row in gate_rows(result.out)
        if row["name"] == "memory_free"
    ]
    assert shown == [("head", "true"), ("worker", "true")]
    assert [call for call in spy.runner.calls if call.mutating] == []


@pytest.mark.parametrize(("name", "argv"), STATE_CHANGING, ids=[name for name, _ in STATE_CHANGING])
def test_no_state_changing_call_without_approval(
    tmp_path: Path, name: str, argv: tuple[str, ...]
) -> None:
    """**状態を変える 11 のコマンドは、端末でなく `--yes` もなければ、何もしない**。

    関門をすべて通す台本を使うので、どのコマンドも了承の段まで届く。それでも、状態を変える
    呼び出し (`mutating`)、配布、`mkdir` が 1 つも出ず、終了コードは 1 (前提の不足) になる
    (requirements 2.1、design.md 「Error Handling」)。
    """
    repo = make_repo(tmp_path)
    # `stop` は、自分のコンテナが 1 つもないと、了承を求めずに `already_stopped` で終わる
    listing = {role: own_row(role) for role in ROLES} if name == "stop" else None
    spy = Spy(script=gate_rules(listing=listing), default=Reply())
    result = run(list(argv), repo, spy=spy)

    assert result.code == cli.EXIT_PRECONDITION
    calls = spy.made[0].calls if spy.made else ()
    assert [call for call in calls if call.mutating] == []
    assert [call for call in calls if call.kind == "push"] == []
    assert [call for call in calls if call.kind == "run" and call.argv[0] == "mkdir"] == []


# --- serve pull-image / image-licenses ------------------------------------


def test_pull_image_calls_the_part_with_the_confirmer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    recorder = Recorder(
        result=im.PullImageOutcome(
            status="pulled", config_name=SERVE_CONFIG, reference=IMAGE_REF, detail="取得した"
        )
    )
    monkeypatch.setattr(im, "pull_image", recorder)
    result = run(["pull-image", SERVE_CONFIG, "--yes"], repo)
    assert result.code == cli.EXIT_OK
    assert recorder.args[1].name == SERVE_CONFIG
    assert isinstance(recorder.kwargs["confirmer"], g.AssumeYesConfirmer)
    pairs = kv(result.out)
    assert pairs["status"] == "pulled"
    assert pairs["detail"] == "取得した"


def test_a_refused_gate_is_a_precondition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """関門が断ったら 1 で、落ちた関門を示す。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(
        im,
        "pull_image",
        Recorder(
            result=im.PullImageOutcome(
                status="refused",
                config_name=SERVE_CONFIG,
                reference=IMAGE_REF,
                gates=(
                    GateResult(gate="disk_space", node="head", passed=False, detail="空きがない"),
                ),
                detail="関門が断った",
            )
        ),
    )
    result = run(["pull-image", SERVE_CONFIG, "--yes"], repo)
    assert result.code == cli.EXIT_PRECONDITION
    pairs = kv(result.out)
    assert pairs["gate.1.name"] == "disk_space"
    assert pairs["gate.1.passed"] == "false"


def test_image_licenses_shows_the_text_on_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """読み取った表記の本文は、標準出力に出す (`report` は stderr)。"""
    repo = make_repo(tmp_path)
    reading = im.LicenseReading(
        node="head",
        container_name="vb-p1-image-licenses-head",
        exit_code=0,
        stdout="Apache License 2.0",
        stderr="",
        detail="読めた",
    )
    recorder = Recorder(
        result=im.LicensesOutcome(
            status="read", config_name=INSPECT_CONFIG, readings=(reading,), detail="1 台で読んだ"
        )
    )
    monkeypatch.setattr(im, "read_image_licenses", recorder)
    result = run(["image-licenses", INSPECT_CONFIG, "--yes"], repo)
    assert result.code == cli.EXIT_OK
    assert "Apache License 2.0" in result.out
    assert recorder.args[3] == FIRST_NOW


# --- serve manifest -------------------------------------------------------


def test_manifest_writes_the_file_without_touching_spark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`serve manifest` は Spark に触らず、`serving/weights/` に書く。"""
    repo = make_repo(tmp_path)
    repo.manifest.unlink()
    recorder = Recorder(result=wg.ManifestResult(manifest=MANIFEST, excluded_paths=("README.md",)))
    monkeypatch.setattr(wg, "build_manifest", recorder)
    result = run(["manifest", REPO, REVISION], repo)
    assert result.code == cli.EXIT_OK
    assert repo.manifest.read_bytes() == wg.to_json_bytes(MANIFEST)
    assert result.spy.made == []
    pairs = kv(result.out)
    assert pairs["path"] == str(repo.manifest)
    assert pairs["file_count"] == str(len(MANIFEST.files))
    assert pairs["excluded"] == "README.md"


def test_manifest_keeps_the_previous_timestamp_when_nothing_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """中身が同じなら、前の `generated_at` を保つ (時刻だけの差分を出さない)。"""
    repo = make_repo(tmp_path)
    remade = MANIFEST.model_copy(update={"generated_at": datetime(2027, 1, 1, tzinfo=UTC)})
    monkeypatch.setattr(
        wg, "build_manifest", Recorder(result=wg.ManifestResult(manifest=remade, excluded_paths=()))
    )
    result = run(["manifest", REPO, REVISION], repo)
    assert result.code == cli.EXIT_OK
    assert repo.manifest.read_bytes() == wg.to_json_bytes(MANIFEST)
    assert kv(result.out)["reused_generated_at"] == "true"


def test_manifest_writes_the_new_timestamp_when_the_files_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    changed = MANIFEST.model_copy(
        update={
            "generated_at": datetime(2027, 1, 1, tzinfo=UTC),
            "files": MANIFEST.files[:1],
            "total_bytes": MANIFEST.files[0].size,
        }
    )
    monkeypatch.setattr(
        wg,
        "build_manifest",
        Recorder(result=wg.ManifestResult(manifest=changed, excluded_paths=())),
    )
    result = run(["manifest", REPO, REVISION], repo)
    assert kv(result.out)["reused_generated_at"] == "false"
    assert wg.load_manifest(repo.manifest).generated_at == datetime(2027, 1, 1, tzinfo=UTC)


# --- serve derived-import -------------------------------------------------

TOOL_PATTERN = r"^model\.language_model\.layers\.(?:0|3)\.mlp\.gate_proj$"
TOOL_FILES: tuple[dict[str, Any], ...] = (
    {"path": "config.json", "size": 4_096, "sha256": "1" * 64},
    {"path": "model-00001-of-00003.safetensors", "size": 5_000, "sha256": "2" * 64},
)


def tool_manifest(
    *, files: Sequence[dict[str, Any]] = TOOL_FILES, **top_level: Any
) -> dict[str, Any]:
    """変換の道具 (`experiments/k2-quant`) が書く `manifest.json` の形 (契約の形)。"""
    payload: dict[str, Any] = {
        "conversion": {
            "tool": "k2-quant",
            "tool_version": "0.2.0",
            "source": {"repo": REPO, "revision": REVISION},
            "pattern": TOOL_PATTERN,
            "args": [
                "--source-repo",
                REPO,
                "--source-revision",
                REVISION,
                "--pattern",
                TOOL_PATTERN,
            ],
            "modules": ["model.language_model.layers.0.mlp.gate_proj"],
            "weight_dtype": "F8_E4M3",
            "scale_dtype": "F32",
            "strategy": "channel",
        },
        "files": [dict(entry) for entry in files],
        "total_bytes": sum(int(entry["size"]) for entry in files),
    }
    payload.update(top_level)
    return payload


def write_tool_manifest(directory: Path, name: str, payload: Mapping[str, Any]) -> Path:
    """道具の `manifest.json` を、Mac に写したファイルとして書く。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path


def derived_import(
    repo: Repo, inputs: Sequence[Path], *extra: str, now: Callable[[], datetime] | None = None
) -> Run:
    """`serve derived-import --name k2s1 --commit <40桁> <入力>...` を流す。"""
    argv = ["derived-import", "--name", DERIVED_NAME, "--commit", DERIVED_COMMIT, *extra]
    return run([*argv, *(str(path) for path in inputs)], repo, now=now)


def imported_manifest_path(repo: Repo) -> Path:
    return repo.serving / "weights" / DERIVED_MANIFEST_NAME


def test_derived_import_writes_a_derived_manifest_made_from_the_tool_manifest(
    tmp_path: Path,
) -> None:
    """道具の manifest から、`serving/weights/<名前>.manifest.json` の派生のマニフェストが書かれる。

    `args` は入力の `conversion.args` の列そのまま、`target_pattern` は `conversion.pattern`、
    `origin` は `conversion.source`。`commit` は引数、`tool` は既定の道具の道筋 (手で書かない)。
    """
    repo = make_repo(tmp_path)
    source = write_tool_manifest(tmp_path / "in", "head.manifest.json", tool_manifest())

    result = derived_import(repo, [source])

    assert result.code == cli.EXIT_OK, result.err
    written = wg.load_manifest(imported_manifest_path(repo))
    assert isinstance(written, kt.DerivedWeightsManifest)
    conversion = written.derivation.conversion
    assert written.derivation.name == DERIVED_NAME
    assert written.derivation.origin == kt.WeightsOrigin(repo=REPO, revision=REVISION)
    assert conversion.tool == "experiments/k2-quant"
    assert conversion.commit == DERIVED_COMMIT
    assert conversion.args == (
        "--source-repo",
        REPO,
        "--source-revision",
        REVISION,
        "--pattern",
        TOOL_PATTERN,
    )
    assert conversion.target_pattern == TOOL_PATTERN
    assert [(f.path, f.size, f.sha256) for f in written.files] == [
        (str(f["path"]), int(f["size"]), str(f["sha256"])) for f in TOOL_FILES
    ]
    assert written.total_bytes == sum(int(f["size"]) for f in TOOL_FILES)


def test_derived_import_reports_the_written_file_and_its_sha256_without_touching_spark(
    tmp_path: Path,
) -> None:
    """標準出力に、名前・道筋・件数・合計・除いたもの・SHA-256・`status=written` を出す。

    Spark には触らない (実行役を 1 つも作らない)。了承も求めない。
    """
    repo = make_repo(tmp_path)
    source = write_tool_manifest(tmp_path / "in", "head.manifest.json", tool_manifest())

    result = derived_import(repo, [source])

    assert result.code == cli.EXIT_OK, result.err
    assert result.spy.made == []
    pairs = kv(result.out)
    path = imported_manifest_path(repo)
    assert pairs["name"] == DERIVED_NAME
    assert pairs["path"] == str(path)
    assert pairs["file_count"] == str(len(TOOL_FILES))
    assert pairs["total_bytes"] == str(sum(int(f["size"]) for f in TOOL_FILES))
    assert pairs["excluded"] == ""
    assert pairs["status"] == "written"
    assert pairs["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_derived_import_records_the_tool_path_given_by_the_option(tmp_path: Path) -> None:
    """`--tool` を渡すと、変換の道具の道筋は、その値になる。"""
    repo = make_repo(tmp_path)
    source = write_tool_manifest(tmp_path / "in", "head.manifest.json", tool_manifest())

    result = derived_import(repo, [source], "--tool", "experiments/k2-quant-next")

    assert result.code == cli.EXIT_OK, result.err
    written = wg.load_manifest(imported_manifest_path(repo))
    assert isinstance(written, kt.DerivedWeightsManifest)
    assert written.derivation.conversion.tool == "experiments/k2-quant-next"


def test_derived_import_leaves_out_model_card_like_files_like_serve_manifest(
    tmp_path: Path,
) -> None:
    """モデルカードらしい名前と `.gitattributes` は、Hub のマニフェストと同じ規則で除く。"""
    repo = make_repo(tmp_path)
    files = (
        {"path": ".gitattributes", "size": 10, "sha256": "3" * 64},
        {"path": "README.md", "size": 20, "sha256": "4" * 64},
        *TOOL_FILES,
    )
    source = write_tool_manifest(tmp_path / "in", "head.manifest.json", tool_manifest(files=files))

    result = derived_import(repo, [source])

    assert result.code == cli.EXIT_OK, result.err
    written = wg.load_manifest(imported_manifest_path(repo))
    assert [f.path for f in written.files] == [str(f["path"]) for f in TOOL_FILES]
    assert written.total_bytes == sum(int(f["size"]) for f in TOOL_FILES)
    assert kv(result.out)["excluded"] == ".gitattributes,README.md"


def test_derived_import_accepts_two_manifests_that_differ_only_in_generated_at(
    tmp_path: Path,
) -> None:
    """2 台ぶんの manifest は、最上位の `generated_at` だけが違っても、中身が同じなら受け取る。"""
    repo = make_repo(tmp_path)
    head = write_tool_manifest(
        tmp_path / "in", "head.manifest.json", tool_manifest(generated_at="2026-09-25T00:00:00Z")
    )
    worker = write_tool_manifest(
        tmp_path / "in", "worker.manifest.json", tool_manifest(generated_at="2026-09-26T09:30:00Z")
    )

    result = derived_import(repo, [head, worker])

    assert result.code == cli.EXIT_OK, result.err
    assert isinstance(wg.load_manifest(imported_manifest_path(repo)), kt.DerivedWeightsManifest)
    assert kv(result.out)["inputs"] == "2"


def test_derived_import_refuses_two_manifests_that_differ_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """2 台ぶんの manifest の中身が違うと、違う箇所を示して終了コード 1 で、何も書かない。

    先頭だけを読んで残りを無視する取り込みは、2 台の変換の結果が違うことを見逃す。
    """
    repo = make_repo(tmp_path)
    head = write_tool_manifest(tmp_path / "in", "head.manifest.json", tool_manifest())
    other_files = (TOOL_FILES[0], {**TOOL_FILES[1], "sha256": "f" * 64})
    worker = write_tool_manifest(
        tmp_path / "in", "worker.manifest.json", tool_manifest(files=other_files)
    )

    result = derived_import(repo, [head, worker])

    assert result.code == cli.EXIT_PRECONDITION
    assert "files[1].sha256" in result.err
    assert "worker.manifest.json" in result.err
    assert not imported_manifest_path(repo).exists()
    assert result.spy.made == []


def test_derived_import_refuses_a_manifest_without_the_recorded_arguments(tmp_path: Path) -> None:
    """`conversion.args` がない manifest (実際の引数の記録がない形) は、補わずに断る。"""
    repo = make_repo(tmp_path)
    payload = tool_manifest()
    del payload["conversion"]["args"]
    source = write_tool_manifest(tmp_path / "in", "head.manifest.json", payload)

    result = derived_import(repo, [source])

    assert result.code == cli.EXIT_PRECONDITION
    assert "args" in result.err
    assert not imported_manifest_path(repo).exists()


def test_derived_import_does_not_overwrite_a_hub_manifest(tmp_path: Path) -> None:
    """宛先に Hub のマニフェストがあれば、終了コード 1 で、そのバイト列を変えない。"""
    repo = make_repo(tmp_path)
    destination = imported_manifest_path(repo)
    destination.write_bytes(wg.to_json_bytes(MANIFEST))
    before = destination.read_bytes()
    source = write_tool_manifest(tmp_path / "in", "head.manifest.json", tool_manifest())

    result = derived_import(repo, [source])

    assert result.code == cli.EXIT_PRECONDITION
    assert destination.read_bytes() == before


def test_derived_import_keeps_the_previous_timestamp_when_nothing_changed(tmp_path: Path) -> None:
    """中身が同じなら、前の `generated_at` を保つ (時刻だけの差分を出さない)。"""
    repo = make_repo(tmp_path)
    source = write_tool_manifest(tmp_path / "in", "head.manifest.json", tool_manifest())
    first = derived_import(repo, [source], now=stepping_clock(FIRST_NOW))
    assert first.code == cli.EXIT_OK, first.err
    written = imported_manifest_path(repo).read_bytes()

    again = derived_import(repo, [source], now=stepping_clock(FIRST_NOW + timedelta(days=30)))

    assert again.code == cli.EXIT_OK, again.err
    assert imported_manifest_path(repo).read_bytes() == written
    assert kv(again.out)["reused_generated_at"] == "true"


def test_derived_import_writes_the_new_timestamp_when_the_content_changed(tmp_path: Path) -> None:
    """中身が変われば、新しい `generated_at` で書き直す。"""
    repo = make_repo(tmp_path)
    source = write_tool_manifest(tmp_path / "in", "head.manifest.json", tool_manifest())
    derived_import(repo, [source], now=stepping_clock(FIRST_NOW))
    first = wg.load_manifest(imported_manifest_path(repo)).generated_at
    later = FIRST_NOW + timedelta(days=30)
    changed_files = (TOOL_FILES[0], {**TOOL_FILES[1], "sha256": "e" * 64})
    changed = write_tool_manifest(
        tmp_path / "in", "head.manifest.json", tool_manifest(files=changed_files)
    )

    result = derived_import(repo, [changed], now=stepping_clock(later))

    assert result.code == cli.EXIT_OK, result.err
    assert kv(result.out)["reused_generated_at"] == "false"
    assert wg.load_manifest(imported_manifest_path(repo)).generated_at >= later > first


# --- serve fetch / verify -------------------------------------------------


def _fetch_outcome(status: str = "fetched") -> wg.FetchOutcome:
    return wg.FetchOutcome(
        status="fetched" if status == "fetched" else "refused",
        config_name=FETCH_CONFIG,
        scope="all",
        detail="2 台で取得して照合した",
    )


def test_fetch_passes_a_fresh_time_for_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """起こす時刻と、照合の時刻は、その時の時刻を渡す (使い回さない)。"""
    repo = make_repo(tmp_path)
    recorder = Recorder(result=_fetch_outcome())
    monkeypatch.setattr(wg, "fetch_weights", recorder)
    result = run(["fetch", FETCH_CONFIG, "--yes"], repo)
    assert result.code == cli.EXIT_OK
    assert recorder.args[4] == FIRST_NOW
    assert recorder.kwargs["verified_at"] != recorder.args[4]
    assert recorder.kwargs["scope"] == "all"
    assert recorder.kwargs["record_dir"].is_absolute()
    assert repo.var_root in recorder.kwargs["record_dir"].parents


def test_fetch_with_probe_files_narrows_the_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    recorder = Recorder(result=_fetch_outcome())
    monkeypatch.setattr(wg, "fetch_weights", recorder)
    run(["fetch", FETCH_CONFIG, "--probe-files", "--yes"], repo)
    assert recorder.kwargs["scope"] == "probe_files"


def test_verify_uses_probe_files_for_a_probe_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """縮小の確認の構成は、safetensors を除いた範囲で照合する。"""
    repo = make_repo(tmp_path)
    recorder = Recorder(
        result=wg.VerifyOutcome(
            status="verified", config_name=PROBE_CONFIG, scope="probe_files", detail="合った"
        )
    )
    monkeypatch.setattr(wg, "verify_weights", recorder)
    result = run(["verify", PROBE_CONFIG, "--yes"], repo)
    assert result.code == cli.EXIT_OK
    assert recorder.kwargs["scope"] == "probe_files"
    assert kv(result.out)["scope"] == "probe_files"


def test_verify_refuses_a_config_without_weights(tmp_path: Path) -> None:
    """重みを持たない構成の照合は、Spark に触る前に断る (終了コード 1)。"""
    repo = make_repo(tmp_path)
    result = run(["verify", INSPECT_CONFIG, "--yes"], repo)
    assert result.code == cli.EXIT_PRECONDITION
    assert result.spy.made == []


# --- 派生の重み (手元で変換した重み) の構成 --------------------------------

DERIVED_NAME = "k2s1"
DERIVED_CONFIG = "p2-nope-tp2-full-k2s1"
DERIVED_FETCH_CONFIG = "p2-fetch-k2s1"
DERIVED_MANIFEST_NAME = f"{DERIVED_NAME}.manifest.json"
DERIVED_TOOL = "experiments/k2-quant/convert.py"
DERIVED_COMMIT = "a" * 40
DERIVED_TARGET = r"^model\.layers\.\d+\.self_attn\..*$"
DERIVED_RECORD_PATH = f"{REMOTE_ROOT}/state/{DERIVED_NAME}.derived.verified.json"
DERIVED_DIRECTORY = f"{REMOTE_ROOT}/models/{DERIVED_NAME}"


def _derived_weights_table(name: str) -> str:
    """構成 `name` の `weights` の節 (派生。元の参照と変換の条件を持つ)。"""
    lines = [
        f"[configs.{name}.weights]",
        'kind = "derived"',
        f'name = "{DERIVED_NAME}"',
        f'manifest = "{DERIVED_MANIFEST_NAME}"',
        f'mount_at = "/models/{DERIVED_NAME}"',
        "",
        f"[configs.{name}.weights.origin]",
        f'repo = "{REPO}"',
        f'revision = "{REVISION}"',
        f'manifest = "{MANIFEST_NAME}"',
        "",
        f"[configs.{name}.weights.conversion]",
        f'tool = "{DERIVED_TOOL}"',
        f'commit = "{DERIVED_COMMIT}"',
        'args = ["--dtype", "fp8"]',
        f"target_pattern = {_toml_value(DERIVED_TARGET)}",
    ]
    return "\n".join(lines) + "\n\n"


def _derived_serve_config(port: int) -> str:
    name = DERIVED_CONFIG
    text = _header(name, "serve", ROLES, served=SERVED_MODEL, weights=False)
    text += _derived_weights_table(name)
    text += _setting(
        f"configs.{name}.docker.models",
        flag="--mount",
        value="type=bind,source={remote_root}/models/" + DERIVED_NAME + ",target={weights.mount_at}"
        ",readonly",
    )
    text += _setting(
        f"configs.{name}.docker.cache",
        flag="--mount",
        value="type=bind,source={remote_root}/cache,target=/root/.cache",
    )
    text += _setting(f"configs.{name}.args.model-path", value="{weights.mount_at}")
    text += _setting(
        f"configs.{name}.args.served-model-name", flag="--served-model-name", value=SERVED_MODEL
    )
    text += _setting(f"configs.{name}.args.host", flag="--host", value="{head.lan_addr}")
    text += _setting(f"configs.{name}.args.port", flag="--port", value=str(port), is_port=True)
    text += _setting(
        f"configs.{name}.args.master-port", flag="--master-port", value="29501", is_port=True
    )
    text += _setting(
        f"configs.{name}.env.host-ip",
        flag="VLLM_HOST_IP",
        value="{node.fabric_addr}",
        measured=True,
    )
    return text


def _derived_fetch_config() -> str:
    """派生の重みを `serve fetch` に掛けようとする構成 (Hub の取得の構成の重みを差し替えたもの)。"""
    name = DERIVED_FETCH_CONFIG
    text = _header(name, "fetch", ROLES, weights=False)
    text += _derived_weights_table(name)
    text += _setting(f"configs.{name}.docker.entrypoint", flag="--entrypoint", value="hf")
    text += _setting(
        f"configs.{name}.docker.models",
        flag="--mount",
        value="type=bind,source={remote_root}/models,target=/models",
    )
    text += _setting(f"configs.{name}.args.download", value="download")
    text += _setting(f"configs.{name}.args.repo", value=REPO)
    text += _setting(f"configs.{name}.args.revision", flag="--revision", value=REVISION)
    text += _setting(
        f"configs.{name}.args.local-dir", flag="--local-dir", value="{weights.mount_at}"
    )
    text += _setting(f"configs.{name}.env.telemetry", flag="HF_HUB_DISABLE_TELEMETRY", value="1")
    return text


def _derived_manifest(origin_revision: str = REVISION) -> kt.DerivedWeightsManifest:
    """派生の構成が指す、変換の結果のマニフェスト (`origin_revision` は元の重みの版)。"""
    derivation = kt.Derivation(
        name=DERIVED_NAME,
        origin=kt.WeightsOrigin(repo=REPO, revision=origin_revision),
        conversion=kt.ConversionSpec(
            tool=DERIVED_TOOL,
            commit=DERIVED_COMMIT,
            args=("--dtype", "fp8"),
            target_pattern=DERIVED_TARGET,
        ),
    )
    return kt.DerivedWeightsManifest(
        kind="derived",
        derivation=derivation,
        generated_at=datetime(2026, 9, 26, 1, 0, 0, tzinfo=UTC),
        total_bytes=MANIFEST.total_bytes,
        files=MANIFEST.files,
    )


def make_derived_repo(tmp_path: Path, *, origin_revision: str = REVISION) -> Repo:
    """`make_repo` のリポジトリに、派生の重みの構成 2 つと、派生のマニフェストを足す。

    `origin_revision` は、派生のマニフェストが元にした重みの版 (構成に書いた版と変えると、
    マニフェストと構成が食い違う)。
    """
    repo = make_repo(tmp_path)
    with repo.configs.open("a", encoding="utf-8") as stream:
        stream.write(_derived_serve_config(UNUSED_PORT))
        stream.write(_derived_fetch_config())
    manifest_path = repo.serving / "weights" / DERIVED_MANIFEST_NAME
    manifest_path.write_bytes(wg.to_json_bytes(_derived_manifest(origin_revision)))
    return repo


def _derived_record_json(role: NodeRole) -> str:
    """その台の、派生の重みの照合の記録 (関門 `weights_verified` が読む形)。"""
    manifest = _derived_manifest()
    return kt.DerivedVerificationRecord(
        kind="derived",
        derivation=manifest.derivation,
        manifest_sha256=manifest.content_sha256,
        scope="all",
        node=role,
        verified_at=datetime(2026, 9, 26, 1, 30, tzinfo=UTC),
        file_count=len(manifest.files),
        total_bytes=manifest.total_bytes,
    ).model_dump_json()


def derived_gate_rules() -> tuple[Rule, ...]:
    """`gate_rules` の、重みの照合の記録の読み取りだけを、派生の記録の道筋のものに替えた台本。"""
    rules = [rule for rule in gate_rules() if rule.prefix != ("cat",)]
    rules.extend(
        Rule(
            prefix=("cat", DERIVED_RECORD_PATH),
            node=role,
            replies=(Reply(stdout=_derived_record_json(role)),),
        )
        for role in ROLES
    )
    return tuple(rules)


def derived_verify_rules() -> tuple[Rule, ...]:
    """`serve verify` の台本 (関門は通り、2 台の `sha256sum` はマニフェストと合う)。"""
    lines = "".join(
        f"{entry.sha256}  {DERIVED_DIRECTORY}/{entry.path}\n" for entry in MANIFEST.files
    )
    rules = [
        Rule(prefix=("sha256sum",), node=role, replies=(Reply(stdout=lines),)) for role in ROLES
    ]
    return (*rules, Rule(kind="push", replies=(Reply(),)), *gate_rules())


def weights_verified_gates(out: str) -> list[dict[str, str]]:
    """`serve check` の標準出力から、関門 `weights_verified` の結果 (台ごと) を取り出す。"""
    pairs = kv(out)
    found: list[dict[str, str]] = []
    index = 1
    while f"gate.{index}.name" in pairs:
        if pairs[f"gate.{index}.name"] == g.GATE_WEIGHTS_VERIFIED:
            found.append(
                {
                    "node": pairs[f"gate.{index}.node"],
                    "passed": pairs[f"gate.{index}.passed"],
                    "detail": pairs[f"gate.{index}.detail"],
                }
            )
        index += 1
    return found


def test_check_passes_the_weights_gate_of_a_derived_weights_config(tmp_path: Path) -> None:
    """派生の構成に、派生のマニフェストと一致する記録があれば、`serve check` は通る。"""
    repo = make_derived_repo(tmp_path)
    spy = Spy(script=derived_gate_rules(), default=Reply())

    result = run(["check", DERIVED_CONFIG], repo, spy=spy)

    assert result.code == cli.EXIT_OK, result.out
    gates = weights_verified_gates(result.out)
    assert [(gate["node"], gate["passed"]) for gate in gates] == [
        ("head", "true"),
        ("worker", "true"),
    ]


def test_check_refuses_a_derived_manifest_that_does_not_match_the_config(tmp_path: Path) -> None:
    """派生のマニフェストが、構成の元の重みと違う版を指すと、両方の同一性を示して断る。"""
    other_revision = "c" * 40
    repo = make_derived_repo(tmp_path, origin_revision=other_revision)
    spy = Spy(script=derived_gate_rules(), default=Reply())

    result = run(["check", DERIVED_CONFIG], repo, spy=spy)

    assert result.code == cli.EXIT_PRECONDITION
    assert kv(result.out)["status"] == "refused"
    gates = weights_verified_gates(result.out)
    assert [gate["passed"] for gate in gates] == ["false", "false"]
    for gate in gates:
        assert f"derived:{DERIVED_NAME}:{REPO}@{REVISION}" in gate["detail"]
        assert f"derived:{DERIVED_NAME}:{REPO}@{other_revision}" in gate["detail"]


def test_check_refuses_a_derived_config_whose_origin_manifest_is_another_weights(
    tmp_path: Path,
) -> None:
    """元の重みのマニフェストの版が `origin` と違えば、`serve check` は Spark に触らずに断る。"""
    repo = make_derived_repo(tmp_path)
    other = MANIFEST.model_copy(update={"revision": "c" * 40})
    repo.manifest.write_bytes(wg.to_json_bytes(other))
    spy = Spy(script=derived_gate_rules(), default=Reply())

    result = run(["check", DERIVED_CONFIG], repo, spy=spy)

    assert result.code == cli.EXIT_PRECONDITION
    assert f"configs.{DERIVED_CONFIG}.weights.origin.manifest" in result.err
    assert spy.made == []


def test_verify_checks_a_derived_weights_config_and_puts_the_derived_record(
    tmp_path: Path,
) -> None:
    """`serve verify` は派生の構成で `sha256sum` を流し、派生の名前の記録を `state/` に配る。"""
    repo = make_derived_repo(tmp_path)
    spy = Spy(script=derived_verify_rules(), default=Reply())

    result = run(["verify", DERIVED_CONFIG, "--yes"], repo, spy=spy)

    assert result.code == cli.EXIT_OK, result.out + result.err
    assert kv(result.out)["status"] == "verified"
    runner = spy.runner
    digests = [argv for argv in runner.argvs if argv[0] == "sha256sum"]
    assert len(digests) == len(ROLES)
    assert all(path.startswith(f"{DERIVED_DIRECTORY}/") for argv in digests for path in argv[2:])
    assert [push.node for push in runner.pushes] == list(ROLES)
    for push in runner.pushes:
        assert push.remote == "state"
        assert push.local_dir is not None
        assert sorted(item.name for item in push.local_dir.iterdir()) == [
            f"{DERIVED_NAME}.derived.verified.json"
        ]


def test_fetch_refuses_a_derived_weights_config_without_touching_spark(tmp_path: Path) -> None:
    """派生の重みには取得元がないので、`serve fetch` は終了コード 1 で断り、Spark に触らない。"""
    repo = make_derived_repo(tmp_path)
    spy = Spy(default=Reply())

    result = run(["fetch", DERIVED_FETCH_CONFIG, "--yes"], repo, spy=spy)

    assert result.code == cli.EXIT_PRECONDITION
    assert "エラー: " in result.err
    assert [call for runner in spy.made for call in runner.calls] == []


# --- serve start / stop / status / smoke ----------------------------------


def test_start_overrides_the_timeout_for_this_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--timeout` は、構成の `ready_timeout_s` を、その回だけ上書きする。"""
    repo = make_repo(tmp_path)
    recorder = Recorder(result=StartOutcome(status="ready", detail="受け付けを始めた"))
    monkeypatch.setattr(lc, "start", recorder)
    result = run(["start", SERVE_CONFIG, "--timeout", "2h", "--yes"], repo)
    assert result.code == cli.EXIT_OK
    assert recorder.kwargs["timeout_s"] == 7200.0
    assert recorder.kwargs["repo_commit"] == REPO_COMMIT
    assert recorder.kwargs["repo_dirty"] is False
    assert recorder.kwargs["var_root"] == repo.var_root
    assert recorder.kwargs["manifest"] == MANIFEST


def test_start_without_a_timeout_leaves_the_configured_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    recorder = Recorder(result=StartOutcome(status="ready"))
    monkeypatch.setattr(lc, "start", recorder)
    run(["start", SERVE_CONFIG, "--yes"], repo)
    assert recorder.kwargs["timeout_s"] is None


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("ready", cli.EXIT_OK),
        ("already_running", cli.EXIT_OK),
        ("refused", cli.EXIT_PRECONDITION),
        ("failed", cli.EXIT_FAILED),
    ],
)
def test_start_maps_its_outcome_to_an_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, expected: int
) -> None:
    repo = make_repo(tmp_path)
    monkeypatch.setattr(
        lc,
        "start",
        Recorder(result=StartOutcome(status=status, detail=f"{status} だった")),  # type: ignore[arg-type]
    )
    result = run(["start", SERVE_CONFIG, "--yes"], repo)
    assert result.code == expected
    assert kv(result.out)["detail"] == f"{status} だった"


def test_start_shows_the_log_tails_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """失敗のときの記録の末尾は、stderr に出す (標準出力に混ぜない)。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(
        lc,
        "start",
        Recorder(
            result=StartOutcome(
                status="failed", detail="時間切れ", log_tails={"head": "ZZ-TAIL-ZZ"}
            )
        ),
    )
    result = run(["start", SERVE_CONFIG, "--yes"], repo)
    assert "ZZ-TAIL-ZZ" in result.err
    assert "ZZ-TAIL-ZZ" not in result.out


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("stopped", cli.EXIT_OK),
        ("already_stopped", cli.EXIT_OK),
        ("gpu_not_released", cli.EXIT_FAILED),
    ],
)
def test_stop_maps_its_outcome_and_always_shows_the_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, expected: int
) -> None:
    """`stop` の `detail` は必ず画面に出す (3.5 の申し送り)。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(
        lc,
        "stop",
        Recorder(result=StopOutcome(status=status, detail="ZZ-STOP-DETAIL-ZZ")),  # type: ignore[arg-type]
    )
    result = run(["stop", "--yes"], repo)
    assert result.code == expected
    assert "ZZ-STOP-DETAIL-ZZ" in result.err
    assert kv(result.out)["detail"] == "ZZ-STOP-DETAIL-ZZ"


def test_status_marks_the_node_that_could_not_be_read(tmp_path: Path) -> None:
    """読めなかった台に印を付け、終了コード 1 で終わる (「何も動いていない」と読み違えない)。"""
    repo = make_repo(tmp_path)
    listing = ("docker", "ps", "-a", "--filter", f"label={OWNER_FILTER}", "--format", "json")
    spy = Spy(
        script=(
            Rule(prefix=listing, node="worker", replies=(Reply(exit_code=255, stderr="だめ"),)),
            Rule(prefix=listing, node="head", replies=(Reply(stdout=""),)),
        )
    )
    result = run(["status"], repo, spy=spy)
    assert result.code == cli.EXIT_PRECONDITION
    assert kv(result.out)["unreadable"] == "worker"
    rows = table_rows(result.out)
    assert rows[0][0] == "node"
    body = {row[0]: row for row in rows[2:]}
    assert cli.UNREADABLE_MARK in body["worker"]
    assert cli.UNREADABLE_MARK not in body["head"]
    assert [call for call in spy.runner.calls if call.mutating] == []


def test_status_is_ok_when_both_nodes_can_be_read(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result = run(["status"], repo)
    assert result.code == cli.EXIT_OK
    assert kv(result.out)["unreadable"] == ""


def test_smoke_shows_the_body_on_stderr_and_the_facts_on_stdout(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """短い要求は、**本文を stderr にだけ**、保存してよい事実を stdout に出す (10.5)。"""
    fake_vllm.set_messages_reply(MessagesReply(text=BODY_MARKER, output_tokens=7))
    repo = make_repo(tmp_path, port=_port_of(fake_vllm))
    result = run(["smoke", SERVE_CONFIG], repo)
    assert result.code == cli.EXIT_OK
    assert BODY_MARKER in result.err
    assert BODY_MARKER not in result.out
    pairs = kv(result.out)
    assert pairs["reply.en.http_status"] == "200"
    assert pairs["reply.ja.http_status"] == "200"
    assert pairs["reply.en.output_tokens"] == "7"
    assert pairs["status"] == "answered"
    assert fake_vllm.call_count("/v1/messages") == 2


def test_smoke_fails_when_a_request_does_not_answer(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    repo = make_repo(tmp_path, port=_port_of(fake_vllm))
    fake_vllm.set_messages_fault(Fault(status=500, body="だめ"))
    result = run(["smoke", SERVE_CONFIG], repo)
    assert result.code == cli.EXIT_FAILED
    assert kv(result.out)["status"] == "failed"


def test_smoke_uses_the_default_max_tokens_when_it_is_not_given(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """`--max-tokens` を書かないと、2 件とも `lifecycle.SMOKE_MAX_TOKENS` で送る。"""
    repo = make_repo(tmp_path, port=_port_of(fake_vllm))
    result = run(["smoke", SERVE_CONFIG], repo)
    assert result.code == cli.EXIT_OK
    assert _smoke_max_tokens(fake_vllm) == [lc.SMOKE_MAX_TOKENS, lc.SMOKE_MAX_TOKENS]


def test_smoke_sends_the_given_max_tokens_in_both_languages(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """`--max-tokens` に書いた上限が、英語と日本語の両方の要求の本文に入る。"""
    fake_vllm.set_messages_reply(MessagesReply(text=BODY_MARKER, output_tokens=7))
    repo = make_repo(tmp_path, port=_port_of(fake_vllm))
    result = run(["smoke", SERVE_CONFIG, "--max-tokens", "512"], repo)
    assert result.code == cli.EXIT_OK
    assert _smoke_max_tokens(fake_vllm) == [512, 512]
    assert BODY_MARKER in result.err
    assert BODY_MARKER not in result.out


@pytest.mark.parametrize("text", ["0", "-1", "1.5", "true"])
def test_a_degenerate_smoke_max_tokens_is_refused_before_anything_is_sent(
    tmp_path: Path, fake_vllm: FakeVllm, text: str
) -> None:
    """上限の値は `argparse` の段で断り、推論サーバーに要求を 1 つも送らない。"""
    repo = make_repo(tmp_path, port=_port_of(fake_vllm))
    with pytest.raises(SystemExit) as caught:
        run(["smoke", SERVE_CONFIG, "--max-tokens", text], repo)
    assert caught.value.code == 2
    assert fake_vllm.requests_for("/v1/messages") == ()


def _smoke_max_tokens(fake: FakeVllm) -> list[int]:
    """届いた短い要求 2 件の本文から、`max_tokens` を届いた順に読む。"""
    sent = fake.requests_for("/v1/messages")
    bodies = [request.body for request in sent if request.body is not None]
    assert len(bodies) == 2, "短い要求の本文が、2 件とも JSON として届いていない"
    return [int(body["max_tokens"]) for body in bodies]


def _port_of(fake: FakeVllm) -> int:
    return int(fake.base_url.rsplit(":", 1)[1])


# --- serve logs -----------------------------------------------------------


def test_logs_collects_into_the_var_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """記録の回収は、その時の時刻の置き場所に写す (requirements 1.9)。"""
    repo = make_repo(tmp_path)
    log_dir = repo.var_root / "20260922T060000Z-logs-p1-nvfp4-tp2"
    recorder = Recorder(result=log_dir)
    monkeypatch.setattr(lg, "collect_logs", recorder)
    result = run(["logs", SERVE_CONFIG], repo)
    assert result.code == cli.EXIT_OK
    assert recorder.kwargs["command"] == "logs"
    assert recorder.kwargs["config_name"] == SERVE_CONFIG
    assert recorder.kwargs["var_root"] == repo.var_root
    assert recorder.kwargs["started_at"] == FIRST_NOW
    assert kv(result.out)["log_dir"] == str(log_dir)


# --- serve probe ----------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [("ready", cli.EXIT_OK), ("inconclusive", cli.EXIT_PRECONDITION), ("failed", cli.EXIT_FAILED)],
)
def test_probe_maps_its_three_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, expected: int
) -> None:
    """`ready` = 0、`inconclusive` = 1、`failed` = 2 (4.1 の申し送り)。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(
        pr,
        "run_probe",
        Recorder(
            result=ProbeOutcome(
                status=status,  # type: ignore[arg-type]
                config_name=PROBE_CONFIG,
                detail="ZZ-PROBE-DETAIL-ZZ",
            )
        ),
    )
    result = run(["probe", PROBE_CONFIG, "--yes"], repo)
    assert result.code == expected
    assert "ZZ-PROBE-DETAIL-ZZ" in result.err
    assert kv(result.out)["detail"] == "ZZ-PROBE-DETAIL-ZZ"


def test_probe_overrides_the_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path)
    recorder = Recorder(result=ProbeOutcome(status="ready", config_name=PROBE_CONFIG))
    monkeypatch.setattr(pr, "run_probe", recorder)
    run(["probe", PROBE_CONFIG, "--timeout", "45m", "--yes"], repo)
    assert recorder.kwargs["timeout_s"] == 2700.0
    assert recorder.kwargs["record_dir"].is_absolute()


# --- serve netcheck -------------------------------------------------------


def test_netcheck_links_reads_both_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`links` は読み取りだけで、台ごとの判断を `key=value` に出す。"""
    repo = make_repo(tmp_path)
    reports = {
        role: LinkReport(
            node=role,
            interfaces=(InterfaceLink(name="enp1s0f0np0", state="UP", mtu=9000),),
            cable_count=1,
            detail=f"{role} は読めた",
        )
        for role in ROLES
    }
    recorder = Recorder(result=reports)
    monkeypatch.setattr(nc, "read_link_reports", recorder)
    result = run(["netcheck", "links"], repo)
    assert result.code == cli.EXIT_OK
    pairs = kv(result.out)
    assert pairs["cable_count.head"] == "1"
    assert pairs["detail.worker"] == "worker は読めた"
    assert "enp1s0f0np0" in result.err


def test_netcheck_bandwidth_maps_its_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    recorder = Recorder(
        result=nc.BandwidthOutcome(
            status="failed",
            config_name=JOB_CONFIG,
            nccl={},
            comparison="NVIDIA の値と比べる",
            detail="ふつうのネットワークに落ちた",
        )
    )
    monkeypatch.setattr(nc, "run_bandwidth", recorder)
    result = run(["netcheck", "bandwidth", JOB_CONFIG, "--yes"], repo)
    assert result.code == cli.EXIT_FAILED
    assert recorder.args[3] == FIRST_NOW
    assert kv(result.out)["comparison"] == "NVIDIA の値と比べる"


def test_netcheck_sanity_lists_the_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path)
    stage = nc.SanityStage(
        index=1, name="1 段", marker="sanity check is successful", passed=True, nodes_seen=ROLES
    )
    monkeypatch.setattr(
        nc,
        "run_sanity",
        Recorder(
            result=nc.SanityOutcome(
                status="passed", config_name=JOB_CONFIG, stages=(stage,), nccl={}, detail="通った"
            )
        ),
    )
    result = run(["netcheck", "sanity", JOB_CONFIG, "--yes"], repo)
    assert result.code == cli.EXIT_OK
    pairs = kv(result.out)
    assert pairs["stage.1.name"] == "1 段"
    assert pairs["stage.1.passed"] == "true"


def test_netcheck_ab_passes_the_env_and_the_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--env KEY=VALUE` は複数書ける。`--repeat` は 2 以上 (design.md 772 行)。"""
    repo = make_repo(tmp_path)
    recorder = Recorder(
        result=nc.AbReport(status="compared", config_name=JOB_CONFIG, added_env={}, detail="比べた")
    )
    monkeypatch.setattr(nc, "run_ab", recorder)
    result = run(
        [
            "netcheck",
            "ab",
            JOB_CONFIG,
            "--env",
            "NCCL_IB_HCA=rocep1s0f0",
            "--env",
            "NCCL_SOCKET_IFNAME=enp1s0f0np0",
            "--repeat",
            "4",
            "--yes",
        ],
        repo,
    )
    assert result.code == cli.EXIT_OK
    assert recorder.kwargs["extra_env"] == {
        "NCCL_IB_HCA": "rocep1s0f0",
        "NCCL_SOCKET_IFNAME": "enp1s0f0np0",
    }
    assert recorder.kwargs["repeats"] == 4


def test_netcheck_ab_needs_at_least_one_env(tmp_path: Path) -> None:
    """足す設定がなければ、A/B にならないので断る (終了コード 1)。"""
    repo = make_repo(tmp_path)
    result = run(["netcheck", "ab", JOB_CONFIG, "--yes"], repo)
    assert result.code == cli.EXIT_PRECONDITION
    assert result.spy.made == []


# --- serve watch ----------------------------------------------------------


def _watch_outcome(*, events: int) -> WatchOutcome:
    from serving_kit.types import WatchEvent, WatchSample

    sample = WatchSample(taken_at_utc=FIRST_NOW, health_ok=True, gpu_utilization_pct={})
    made = tuple(
        WatchEvent(finding="stalled", at_utc=FIRST_NOW, context=(sample,), detail="固まった")
        for _ in range(events)
    )
    return WatchOutcome(
        config_name=SERVE_CONFIG,
        started_at=FIRST_NOW,
        finished_at=FIRST_NOW,
        samples_path=Path("/tmp/samples.jsonl"),
        sample_count=3,
        events=made,
        gpu_utilization_available=True,
        detail="見張った",
    )


@pytest.mark.parametrize(("events", "expected"), [(0, cli.EXIT_OK), (1, cli.EXIT_FAILED)])
def test_watch_maps_the_events_to_an_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: int, expected: int
) -> None:
    """出来事なし = 0、出来事あり = 2 (4.5 の申し送り)。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(wt, "watch", Recorder(result=_watch_outcome(events=events)))
    result = run(["watch", SERVE_CONFIG], repo)
    assert result.code == expected
    assert kv(result.out)["events"] == str(events)


def test_watch_passes_the_numbers_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    recorder = Recorder(result=_watch_outcome(events=0))
    monkeypatch.setattr(wt, "watch", recorder)
    run(
        [
            "watch",
            SERVE_CONFIG,
            "--duration",
            "2h",
            "--interval",
            "15s",
            "--stall-window",
            "5m",
            "--stall-gpu-threshold",
            "80",
            "--unresponsive-threshold",
            "4",
        ],
        repo,
    )
    assert recorder.kwargs["duration_s"] == 7200.0
    assert recorder.kwargs["interval_s"] == 15.0
    assert recorder.kwargs["stall_window_s"] == 300.0
    assert recorder.kwargs["stall_gpu_threshold_pct"] == 80
    assert recorder.kwargs["unresponsive_threshold"] == 4
    assert recorder.kwargs["var_root"] == repo.var_root


# --- serve thinking -------------------------------------------------------


def _thinking_outcome(effective: bool | None) -> ThinkingOutcome:
    trial = ThinkingTrial(variant="none", trial_index=1, http_status=200, thinking_chars=120)
    return ThinkingOutcome(
        model=SERVED_MODEL, trials=(trial,), effective=effective, detail="確かめた"
    )


@pytest.mark.parametrize(
    ("effective", "expected"),
    [(True, cli.EXIT_OK), (False, cli.EXIT_OK), (None, cli.EXIT_PRECONDITION)],
)
def test_thinking_maps_the_judgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effective: bool | None, expected: int
) -> None:
    """判定が出れば 0、出なければ 1 (`effective` が None は「判定できなかった」)。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(th, "run_thinking", Recorder(result=_thinking_outcome(effective)))
    result = run(["thinking", SERVE_CONFIG, "--trials", "2"], repo)
    assert result.code == expected
    assert kv(result.out)["effective"] == ("" if effective is None else str(effective).lower())


def test_thinking_builds_the_base_url_from_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """宛先は、head のホストと構成の `--port` から組み立てる (4.6 の申し送り)。"""
    repo = make_repo(tmp_path, port=8000)
    recorder = Recorder(result=_thinking_outcome(True))
    monkeypatch.setattr(th, "run_thinking", recorder)
    run(["thinking", SERVE_CONFIG, "--trials", "2", "--max-tokens", "64", "--timeout", "90"], repo)
    assert recorder.kwargs["base_url"] == "http://127.0.0.1:8000"
    assert recorder.kwargs["model"] == SERVED_MODEL
    assert recorder.kwargs["trials"] == 2
    assert recorder.kwargs["max_tokens"] == 64
    assert recorder.kwargs["timeout_s"] == 90.0
    assert recorder.kwargs["var_root"] == repo.var_root


def test_thinking_talks_to_the_fake_server(tmp_path: Path, fake_vllm: FakeVllm) -> None:
    """偽の推論サーバーを相手に、5 通り × 回数の要求が出る (Spark には触らない)。"""
    repo = make_repo(tmp_path, port=_port_of(fake_vllm))
    result = run(["thinking", SERVE_CONFIG, "--trials", "1"], repo)
    assert result.code in (cli.EXIT_OK, cli.EXIT_PRECONDITION)
    assert fake_vllm.call_count("/v1/messages") == 5
    assert result.spy.made == []


# --- git の事実 -----------------------------------------------------------


def test_repo_facts_reads_the_commit_and_the_dirty_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """`serve start` に渡す commit と、未コミットの変更の有無を、`git` から読む。"""
    outputs = {("rev-parse", "HEAD"): REPO_COMMIT + "\n", ("status", "--porcelain"): " M a.py\n"}

    def fake_run(argv: Sequence[str], **kwargs: Any) -> Any:
        key = (argv[-2], argv[-1])
        return subprocess.CompletedProcess(list(argv), 0, outputs[key], "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert cli.repo_facts(Path("/repo")) == (REPO_COMMIT, True)


def test_repo_facts_says_unknown_when_git_is_not_there(monkeypatch: pytest.MonkeyPatch) -> None:
    """`git` が使えないときは、`unknown` と「変更があるかもしれない」で通す。"""

    def fake_run(argv: Sequence[str], **kwargs: Any) -> Any:
        raise FileNotFoundError("git がない")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert cli.repo_facts(Path("/repo")) == ("unknown", True)


# --- 表示の決まり ---------------------------------------------------------


def test_the_standard_output_holds_no_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """進捗と警告は stderr に出す (標準出力は、後の処理が読む形だけ)。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(
        lc,
        "start",
        Recorder(result=StartOutcome(status="ready", detail="受け付けを始めた")),
    )
    result = run(["start", SERVE_CONFIG, "--yes"], repo)
    for line in result.out.splitlines():
        assert line.startswith("|") or "=" in line


def test_a_multi_line_detail_stays_on_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`detail` が複数行でも、`key=value` の行は 1 行に収める (改行は書き換える)。"""
    repo = make_repo(tmp_path)
    monkeypatch.setattr(
        lc, "stop", Recorder(result=StopOutcome(status="stopped", detail="1 行目\n2 行目"))
    )
    result = run(["stop", "--yes"], repo)
    assert kv(result.out)["detail"] == "1 行目\\n2 行目"
    assert "1 行目\n2 行目" in result.err


def test_smoke_outcome_keys_cover_every_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    replies = (
        SmokeReply(lang="en", http_status=200, stop_reason="end_turn", output_tokens=12),
        SmokeReply(lang="ja", http_status=200, stop_reason="end_turn", replacement_char=True),
    )
    monkeypatch.setattr(
        lc, "smoke", Recorder(result=SmokeOutcome(replies=replies, detail="2 つ送った"))
    )
    result = run(["smoke", SERVE_CONFIG], repo)
    pairs = kv(result.out)
    assert pairs["reply.ja.replacement_char"] == "true"
    assert pairs["reply.en.output_tokens"] == "12"


def test_status_needs_no_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """読み取りだけのコマンドは、了承を求めない (計画を見せない)。"""
    repo = make_repo(tmp_path)
    quiet = ServiceStatus(nodes=(NodeStatus(node="head", container_state="absent"),))
    monkeypatch.setattr(lc, "read_status", Recorder(result=lc.StatusShown(service=quiet)))
    result = run(["status"], repo)
    assert result.code == cli.EXIT_OK
    assert "了承" not in result.err


def test_the_json_of_a_manifest_is_not_written_by_the_cli(tmp_path: Path) -> None:
    """`serve manifest` を呼ばないかぎり、マニフェストは書き換わらない。"""
    repo = make_repo(tmp_path)
    before = repo.manifest.read_bytes()
    run(["status"], repo)
    assert repo.manifest.read_bytes() == before
    assert json.loads(before)["repo"] == REPO
