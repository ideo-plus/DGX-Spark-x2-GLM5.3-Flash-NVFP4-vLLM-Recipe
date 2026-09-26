"""重みの取得と、2 台での照合の試験 (tasks.md 3.3)。

確かめること (design.md 「イメージと重み › weights」、tasks.md 3.3 の完了の状態):

- 偽の実行役で、**1 行だけ違う sha256 の出力**に対して、そのファイルの名前を示して失敗し、
  **取り直しの呼び出し (`docker run` / `docker pull` / `docker rm`) が 1 つも出ない**
  (requirements 3.5)
- 取得のコンテナに渡る環境変数に、トークンがない。Mac の側に `HF_TOKEN` を置いても、
  どの引数にも現れない (requirements 2.6)
- 了承しないと、`docker run` も `push` も 1 つも出ない (requirements 2.1)
- 取得のコンテナが、**切り離して**、名前とラベルつきで起こされ、`--rm` がない
- 2 台とも終わるまで待つ (片方が先に終わる、片方が失敗する)
- **もう一度打つと、動いている取得を見つけて待ち、新しく起こさない** (二重に取得しない)
- 待ちの途中の `RemoteError` と中断 (Ctrl-C) で、取得のコンテナを止めない (184 GiB の取得を、
  誤って捨てない)
- 終了した取得のコンテナは、片付けられる (3.1 の巻き戻しの決まりどおり、一覧で確かめてから)
- 縮小の確認用の範囲では、safetensors を除いたファイルだけを照合し、記録が
  `….probe.verified.json` に置かれる
- 照合の結果の記録の配布が、了承済みの計画にあり、宛先が `state/`
- 偽の実行役に記録された、コンテナを対象にする操作のすべてが、自分の一覧の ID か、了承済みの
  計画の名前 (一覧で確かめたあと) だけを対象にしている (requirements 2.3、2.4)
- ホストに何も入れない (`pip`、`apt`、`hf` をホストで呼ぶ列が、どの経路でも出ない。
  `remote` が、そもそもそういう列を断る)

実物の ssh、rsync、docker、Hugging Face Hub には、どの段でもつながない。試験は、実際に
眠らない (待つ間隔と時計は、引数で差し替える)。
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl

from fake_runner import FakeRunner, RecordedCall, Reply, Rule
from serving_kit import types as kit_types
from serving_kit import weights as w
from serving_kit.config import ConfigError
from serving_kit.guards import ApprovalError, gate_weights_verified, verification_record_path
from serving_kit.plan import (
    LABEL_CONFIG,
    LABEL_CONFIG_SHA256,
    LABEL_IMAGE,
    LABEL_KIND,
    LABEL_OWNER,
    OWNER,
    OWNER_FILTER,
    build_plans,
)
from serving_kit.remote import RemoteError
from serving_kit.types import (
    ConfigDef,
    ContainerPlan,
    ImageRef,
    ManifestFile,
    NodeDef,
    NodeRole,
    Setting,
    VerificationScope,
    WeightsManifest,
    WeightsRef,
)

# --- 見本の値 -----------------------------------------------------------

ROLES: tuple[NodeRole, ...] = ("head", "worker")

DIGEST = "b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
IMAGE_REF = f"vllm/vllm-openai@sha256:{DIGEST}"
REPO = "RedHatAI/GLM-5.3-Flash-NVFP4"
REVISION = "18d55bfd" + "0" * 32
SLUG = "RedHatAI__GLM-5.3-Flash-NVFP4"
MOUNT_AT = "/models/nvfp4"
PROBE_MOUNT_AT = "/models/probe"
REMOTE_ROOT = "/home/j5ik2o/vllm-baseline"
WEIGHTS_DIR = f"{REMOTE_ROOT}/models/{SLUG}"
PROBE_DIR = f"{REMOTE_ROOT}/probe/{SLUG}"

STARTED_AT = datetime(2026, 9, 22, 3, 0, 0, tzinfo=UTC)
VERIFIED_AT = datetime(2026, 9, 22, 4, 30, 0, tzinfo=UTC)
GENERATED_AT = datetime(2026, 9, 22, 1, 0, 0, tzinfo=UTC)

SOURCE = HttpUrl("https://huggingface.co/docs/huggingface_hub/guides/cli")
QUOTE = "hf download [-h] [--revision REVISION] ..."

CONTAINER_IDS: Mapping[NodeRole, str] = {"head": "0123456789ab", "worker": "cdef01234567"}

FILE_SIZES: Mapping[str, int] = {
    "config.json": 100,
    "model-00001-of-00002.safetensors": 4096,
    "model-00002-of-00002.safetensors": 8192,
    "tokenizer.json": 200,
    "空きのある 名前/extra.json": 7,
}
"""マニフェストに載せるファイル (`path` の順。空白を含む名前も 1 つ入れる)。"""


def digest_of(path: str) -> str:
    """このファイルの、正しい sha256 (試験の中で決め打ちにする)。"""
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def manifest() -> WeightsManifest:
    """照合の正解 (Mac でコミットしたもの)。"""
    files = tuple(
        ManifestFile(path=path, size=size, sha256=digest_of(path))
        for path, size in sorted(FILE_SIZES.items())
    )
    return WeightsManifest(
        repo=REPO,
        revision=REVISION,
        generated_at=GENERATED_AT,
        total_bytes=sum(entry.size for entry in files),
        files=files,
    )


MANIFEST = manifest()
ALL_PATHS = tuple(entry.path for entry in MANIFEST.files)
PROBE_PATHS = tuple(entry.path for entry in MANIFEST.probe_files)


def _setting(
    flag: str | None = None,
    value: str | None = None,
    *,
    only_on: NodeRole | None = None,
    is_port: bool = False,
) -> Setting:
    """根拠の付いた設定を 1 つ作る (根拠の中身は、この試験では問わない)。"""
    return Setting(
        flag=flag,
        value=value,
        why="試験のための設定",
        only_on=only_on,
        is_port=is_port,
        source=SOURCE,
        quote=QUOTE,
    )


IMAGE = ImageRef(
    ref=IMAGE_REF,
    seen_as="vllm/vllm-openai:glm53-flash-arm64-cu130",
    size_bytes=9666567584,
    source=HttpUrl("https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags"),
    quote="glm53-flash-arm64-cu130 / linux/arm64 / 9666567584 B",
)

WEIGHTS = WeightsRef(
    repo=REPO,
    revision=REVISION,
    manifest=f"{SLUG}.manifest.json",
    mount_at=MOUNT_AT,
    source=HttpUrl("https://huggingface.co/api/models/RedHatAI/GLM-5.3-Flash-NVFP4"),
    quote="List the content of a repository tree, with pagination support.",
)

PROBE_WEIGHTS = WEIGHTS.model_copy(update={"mount_at": PROBE_MOUNT_AT})

HEAD = NodeDef(
    role="head",
    ssh_host="spark-153d",
    lan_addr=IPv4Address("10.0.1.60"),
    remote_root=REMOTE_ROOT,
)
WORKER = NodeDef(
    role="worker",
    ssh_host="spark-5083",
    lan_addr=IPv4Address("10.0.1.61"),
    remote_root=REMOTE_ROOT,
)
NODES: Mapping[NodeRole, NodeDef] = {"head": HEAD, "worker": WORKER}


def fetch_config(*, ready_timeout_s: int = 14400) -> ConfigDef:
    """重みの取得の構成 (design.md Data Models の `p1-fetch-nvfp4`)。

    動かすプログラムは docker の設定 (`--entrypoint hf`)、`download` とリポジトリの名前は
    位置の引数、環境変数は `HF_HUB_DISABLE_TELEMETRY=1` だけである。
    """
    return ConfigDef(
        name="p1-fetch-nvfp4",
        kind="fetch",
        description="第一の候補の重みを、2 台に取得する",
        nodes=("head", "worker"),
        image=IMAGE,
        weights=WEIGHTS,
        docker={
            "entrypoint": _setting("--entrypoint", "hf"),
            "models": _setting(
                "--mount",
                "type=bind,source={remote_root}/models/" + SLUG + ",target={weights.mount_at}",
            ),
        },
        args={
            "download": _setting(value="download"),
            "repo": _setting(value=REPO),
            "revision": _setting("--revision", REVISION),
            "local-dir": _setting("--local-dir", "{weights.mount_at}"),
            "max-workers": _setting("--max-workers", "8"),
            "exclude": _setting("--exclude", "README.md"),
        },
        env={"telemetry": _setting("HF_HUB_DISABLE_TELEMETRY", "1")},
        ready_timeout_s=ready_timeout_s,
        served_model_name=None,
    )


def probe_fetch_config() -> ConfigDef:
    """縮小の確認用の取得の構成 (1 台。safetensors を除いて取る。親の判断)。"""
    return ConfigDef(
        name="p1-fetch-probe",
        kind="fetch",
        description="縮小の確認に使う、設定とトークナイザだけを取得する",
        nodes=("head",),
        image=IMAGE,
        weights=PROBE_WEIGHTS,
        docker={
            "entrypoint": _setting("--entrypoint", "hf"),
            "probe": _setting(
                "--mount",
                "type=bind,source={remote_root}/probe/" + SLUG + ",target={weights.mount_at}",
            ),
        },
        args={
            "download": _setting(value="download"),
            "repo": _setting(value=REPO),
            "revision": _setting("--revision", REVISION),
            "local-dir": _setting("--local-dir", "{weights.mount_at}"),
            "exclude-weights": _setting("--exclude", "*.safetensors"),
        },
        env={"telemetry": _setting("HF_HUB_DISABLE_TELEMETRY", "1")},
        ready_timeout_s=3600,
        served_model_name=None,
    )


def serve_config() -> ConfigDef:
    """推論サーバーの構成 (`serve verify` の対象にする)。"""
    return ConfigDef(
        name="p1-nvfp4-tp2",
        kind="serve",
        description="P1 の第一の構成",
        nodes=("head", "worker"),
        image=IMAGE,
        weights=WEIGHTS,
        docker={
            "models": _setting(
                "--mount",
                "type=bind,source={remote_root}/models/"
                + SLUG
                + ",target={weights.mount_at},readonly",
            ),
        },
        args={"model-path": _setting(value="{weights.mount_at}")},
        env={},
        ready_timeout_s=1800,
        served_model_name="glm-5-3-flash",
    )


FETCH_PLANS: tuple[ContainerPlan, ...] = build_plans(fetch_config(), NODES, STARTED_AT)
PLAN_OF: Mapping[NodeRole, ContainerPlan] = {plan.node: plan for plan in FETCH_PLANS}
PROBE_PLAN: ContainerPlan = build_plans(probe_fetch_config(), NODES, STARTED_AT)[0]


# --- 台本の部品 ----------------------------------------------------------

OWN_CONTAINERS_ARGV = (
    "docker",
    "ps",
    "-a",
    "--filter",
    f"label={OWNER_FILTER}",
    "--format",
    "json",
)
"""`guards.list_own_containers` が流す、ただ 1 つのコンテナの一覧の読み取り。"""


def ps_row(name: str, container_id: str, state: str, labels: Mapping[str, str]) -> str:
    """`docker ps -a --format json` の 1 行 (自分のコンテナがあるときの見本)。

    **これは実機から採った見本ではない** (tasks.md 1.6 の「ここで採れないもの」)。項目の
    名前は `test_guards.py` / `test_image.py` の見本に合わせた。
    """
    return (
        json.dumps(
            {
                "ID": container_id,
                "Names": name,
                "State": state,
                "Image": labels.get(LABEL_IMAGE, IMAGE_REF),
                "Labels": ",".join(f"{key}={value}" for key, value in sorted(labels.items())),
            }
        )
        + "\n"
    )


def ps_line(plan: ContainerPlan, *, state: str, labels: Mapping[str, str] | None = None) -> str:
    """この計画のコンテナの、一覧の 1 行。"""
    return ps_row(
        plan.container_name,
        CONTAINER_IDS[plan.node],
        state,
        plan.labels if labels is None else labels,
    )


def sha_line(path: str, directory: str, *, digest: str | None = None, binary: bool = False) -> str:
    """`sha256sum` の 1 行 (`<64 桁><空白><空白か *><道筋>`)。"""
    mark = "*" if binary else " "
    return f"{digest or digest_of(path)} {mark}{directory}/{path}\n"


def sha_output(
    paths: Sequence[str],
    directory: str,
    *,
    wrong: Mapping[str, str] | None = None,
    missing: Sequence[str] = (),
    extra_lines: Sequence[str] = (),
) -> str:
    """`sha256sum` の出力 (ないファイルの行は出ない。標準エラーに出る)。"""
    wrong = wrong or {}
    lines = [
        sha_line(path, directory, digest=wrong.get(path)) for path in paths if path not in missing
    ]
    return "".join((*lines, *extra_lines))


def missing_stderr(paths: Sequence[str], directory: str) -> str:
    return "".join(f"sha256sum: {directory}/{path}: No such file or directory\n" for path in paths)


@dataclass
class SpyConfirmer:
    """計画を見せて、決めたとおりに答える了承の口 (`guards.Confirmer` の形)。"""

    approves: bool = True
    shown: list[str] = field(default_factory=list)

    def confirm(self, plan_text: str) -> bool:
        self.shown.append(plan_text)
        return self.approves


@dataclass
class FakeClock:
    """試験用の時計。眠りは記録するだけで、実際には眠らない (時刻は、眠ったぶんだけ進む)。"""

    now: float = 0.0
    slept: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@dataclass
class FetchScript:
    """`serve fetch` の台本 (既定: 関門が通り、2 台とも 0 で終わり、照合も合う)。"""

    listings: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    states: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    run: Reply = Reply(stdout="0123456789ab\n")
    runs: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    """台ごとの `docker run` の返事 (片方だけ失敗させるため)。"""

    logs: Reply = Reply(stdout="Fetching 5 files: 100%|##########| 5/5\n")
    avail: str = "Avail\n999999999999\n"
    sha: Mapping[NodeRole, tuple[Reply, ...]] | None = None
    directory: str = WEIGHTS_DIR
    paths: Sequence[str] = ALL_PATHS
    roles: Sequence[NodeRole] = ROLES

    def rules(self) -> tuple[Rule, ...]:
        rules: list[Rule] = []
        for role in self.roles:
            plan = PLAN_OF[role]
            listings = (
                (
                    Reply(stdout=""),
                    Reply(stdout=ps_line(plan, state="running")),
                    Reply(stdout=ps_line(plan, state="exited")),
                )
                if self.listings is None
                else self.listings[role]
            )
            states = (
                (Reply(stdout="running 0\n"), Reply(stdout="exited 0\n"))
                if self.states is None
                else self.states[role]
            )
            sha = (
                (Reply(stdout=sha_output(self.paths, self.directory)),)
                if self.sha is None
                else self.sha[role]
            )
            rules.append(Rule(prefix=OWN_CONTAINERS_ARGV, node=role, replies=listings))
            rules.append(Rule(prefix=("docker", "container", "inspect"), node=role, replies=states))
            rules.append(Rule(prefix=("sha256sum",), node=role, replies=sha))
            if self.runs is not None:
                rules.append(Rule(prefix=("docker", "run"), node=role, replies=self.runs[role]))
        rules.extend(
            (
                Rule(prefix=("uname",), replies=(Reply(stdout="spark-153d\n"),)),
                Rule(prefix=("nvidia-smi",), replies=(Reply(stdout=""),)),
                Rule(prefix=("test", "-d"), replies=(Reply(),)),
                Rule(
                    prefix=("docker", "image", "inspect"),
                    replies=(Reply(stdout=json.dumps([IMAGE_REF]) + "\n"),),
                ),
                Rule(prefix=("df",), replies=(Reply(stdout=self.avail),)),
                Rule(prefix=("docker", "run"), replies=(self.run,)),
                Rule(prefix=("docker", "logs"), replies=(self.logs,)),
                Rule(prefix=("docker", "stop"), replies=(Reply(),)),
                Rule(prefix=("docker", "rm"), replies=(Reply(),)),
                Rule(kind="push", replies=(Reply(),)),
            )
        )
        return tuple(rules)


def fetch_runner(tmp_path: Path, script: FetchScript | None = None) -> FakeRunner:
    return FakeRunner(var_root=tmp_path, script=(script or FetchScript()).rules())


def verify_runner(
    tmp_path: Path,
    *,
    sha: Mapping[NodeRole, tuple[Reply, ...]] | None = None,
    layout_ok: bool = True,
    listing: Mapping[NodeRole, str] | None = None,
) -> FakeRunner:
    """`serve verify` の台本 (関門は、入れるか、取得が動いていないか、置き場所があるか)。"""
    rules: list[Rule] = []
    for role in ROLES:
        replies = (Reply(stdout=sha_output(ALL_PATHS, WEIGHTS_DIR)),) if sha is None else sha[role]
        rules.append(Rule(prefix=("sha256sum",), node=role, replies=replies))
        rules.append(
            Rule(
                prefix=OWN_CONTAINERS_ARGV,
                node=role,
                replies=(Reply(stdout="" if listing is None else listing[role]),),
            )
        )
    rules.extend(
        (
            Rule(prefix=("uname",), replies=(Reply(stdout="spark-153d\n"),)),
            Rule(prefix=("test", "-d"), replies=(Reply(exit_code=0 if layout_ok else 1),)),
            Rule(kind="push", replies=(Reply(),)),
        )
    )
    return FakeRunner(var_root=tmp_path, script=tuple(rules))


def record_source(tmp_path: Path, role: NodeRole) -> Path:
    """記録を配る元のディレクトリ (この module だけが使う場所。指摘 3)。"""
    return tmp_path / "records" / "verified" / role


def refuse_runner(tmp_path: Path, listing: Mapping[NodeRole, str]) -> FakeRunner:
    """`serve fetch` が、起こす前に断る道の台本 (入れるかと、一覧だけを読む)。"""
    rules: list[Rule] = [
        Rule(prefix=OWN_CONTAINERS_ARGV, node=role, replies=(Reply(stdout=listing[role]),))
        for role in ROLES
    ]
    rules.append(Rule(prefix=("uname",), replies=(Reply(stdout="spark-153d\n"),)))
    return FakeRunner(var_root=tmp_path, script=tuple(rules))


@pytest.fixture(autouse=True)
def no_real_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """実物の ssh / rsync / docker を呼ばないことと、実際に眠らないことを固定する。"""

    def explode(argv: Sequence[str], **kwargs: Any) -> None:
        raise AssertionError(f"実物のコマンドを呼んだ: {list(argv)}")

    def no_sleep(seconds: float) -> None:
        raise AssertionError(f"試験が実際に眠った: {seconds} 秒")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(time, "sleep", no_sleep)


# --- 呼び出しの読み取り --------------------------------------------------


def steps(runner: FakeRunner) -> tuple[str, ...]:
    """記録された呼び出しを、短い名前の列にする (順序を見るため)。"""
    names: list[str] = []
    for call in runner.calls:
        if call.kind != "run":
            names.append(call.kind)
            continue
        argv = call.argv
        if argv[0] != "docker":
            names.append(argv[0])
        elif argv[1] in ("container", "image"):
            names.append(f"{argv[1]} inspect")
        else:
            names.append(argv[1])
    return tuple(names)


def mutating_calls(runner: FakeRunner) -> tuple[RecordedCall, ...]:
    return tuple(call for call in runner.calls if call.mutating)


def argv_of(runner: FakeRunner, *prefix: str) -> tuple[tuple[str, ...], ...]:
    return tuple(argv for argv in runner.argvs if argv[: len(prefix)] == prefix)


def sha_paths(runner: FakeRunner, role: NodeRole | None = None) -> tuple[str, ...]:
    """`sha256sum` に渡った道筋の全部 (呼ばれた順)。"""
    paths: list[str] = []
    for call in runner.runs:
        if call.argv[0] != "sha256sum" or (role is not None and call.node != role):
            continue
        paths.extend(item for item in call.argv[1:] if item != "--")
    return tuple(paths)


def fetch_weights(
    runner: FakeRunner,
    config: ConfigDef,
    *,
    confirmer: SpyConfirmer | None = None,
    scope: VerificationScope = "all",
    record_dir: Path,
    clock: FakeClock | None = None,
    report: io.StringIO | None = None,
    batch_files: int = 100,
) -> Any:
    """`w.fetch_weights` を、試験の既定の引数で呼ぶ。"""
    ticker = clock or FakeClock()
    return w.fetch_weights(
        runner,
        config,
        NODES,
        MANIFEST,
        STARTED_AT,
        confirmer=confirmer or SpyConfirmer(),
        scope=scope,
        record_dir=record_dir,
        verified_at=VERIFIED_AT,
        sleep=ticker.sleep,
        clock=ticker.monotonic,
        report=report or io.StringIO(),
        batch_files=batch_files,
    )


def verify_weights(
    runner: FakeRunner,
    config: ConfigDef,
    *,
    confirmer: SpyConfirmer | None = None,
    scope: VerificationScope = "all",
    record_dir: Path,
    batch_files: int = 100,
) -> Any:
    """`w.verify_weights` を、試験の既定の引数で呼ぶ。"""
    return w.verify_weights(
        runner,
        config,
        NODES,
        MANIFEST,
        STARTED_AT,
        confirmer=confirmer or SpyConfirmer(),
        scope=scope,
        record_dir=record_dir,
        verified_at=VERIFIED_AT,
        batch_files=batch_files,
    )


# --- 置き場所の読み取り --------------------------------------------------


def test_the_weights_directory_comes_from_the_mount_whose_target_is_the_mount_point() -> None:
    argv = PLAN_OF["head"].argv
    assert w.weights_dir_on_spark(argv, MOUNT_AT) == WEIGHTS_DIR


def test_the_weights_directory_works_when_the_mount_target_is_a_parent() -> None:
    # 6.2 が、`models/` そのものを結び付ける形を選んでも読めること
    argv = (
        "docker",
        "run",
        "-d",
        "--mount",
        f"type=bind,source={REMOTE_ROOT}/models,target=/models",
        IMAGE_REF,
    )
    assert w.weights_dir_on_spark(argv, "/models/nvfp4") == f"{REMOTE_ROOT}/models/nvfp4"


def test_a_config_without_a_mount_for_the_weights_is_refused() -> None:
    argv = ("docker", "run", "-d", "--mount", "type=bind,source=/x/logs,target=/logs", IMAGE_REF)
    with pytest.raises(ConfigError) as caught:
        w.weights_dir_on_spark(argv, MOUNT_AT)
    assert MOUNT_AT in str(caught.value)


# --- sha256sum の出力の読み取り ------------------------------------------


def test_the_sha256sum_reader_handles_the_binary_mark_and_spaces_in_paths() -> None:
    text = sha_line("config.json", WEIGHTS_DIR) + sha_line(
        "空きのある 名前/extra.json", WEIGHTS_DIR, binary=True
    )
    digests, unreadable = w.parse_sha256_output(text)
    assert unreadable == ()
    assert digests == {
        f"{WEIGHTS_DIR}/config.json": digest_of("config.json"),
        f"{WEIGHTS_DIR}/空きのある 名前/extra.json": digest_of("空きのある 名前/extra.json"),
    }


def test_the_sha256sum_reader_names_the_lines_it_cannot_read() -> None:
    text = "sha256sum: おかしな行\n" + sha_line("config.json", WEIGHTS_DIR)
    digests, unreadable = w.parse_sha256_output(text)
    assert unreadable == ("sha256sum: おかしな行",)
    assert list(digests) == [f"{WEIGHTS_DIR}/config.json"]


# --- 取得: 了承と、切り離した起動 ----------------------------------------


def test_the_fetch_runs_the_gates_then_the_approval_then_the_containers(tmp_path: Path) -> None:
    runner = fetch_runner(tmp_path)
    confirmer = SpyConfirmer()
    outcome = fetch_weights(
        runner, fetch_config(), confirmer=confirmer, record_dir=tmp_path / "records"
    )

    assert outcome.status == "fetched"
    assert steps(runner)[:2] == ("uname", "ps")
    assert "run" in steps(runner)
    # 了承は 1 度だけ。計画には、取得のコンテナと、記録の配布の両方が入る
    assert len(confirmer.shown) == 1
    assert "docker run" in confirmer.shown[0]
    assert "state/" in confirmer.shown[0]
    # 関門は、まだ無い照合の記録を読まない (`cat` が 1 つも出ない)
    assert argv_of(runner, "cat") == ()


def test_nothing_runs_when_the_operator_refuses(tmp_path: Path) -> None:
    runner = fetch_runner(tmp_path)
    with pytest.raises(ApprovalError):
        fetch_weights(
            runner,
            fetch_config(),
            confirmer=SpyConfirmer(approves=False),
            record_dir=tmp_path / "records",
        )

    assert argv_of(runner, "docker", "run") == ()
    assert runner.pushes == ()
    assert mutating_calls(runner) == ()


def test_the_fetch_container_is_detached_named_and_labelled_without_rm(tmp_path: Path) -> None:
    runner = fetch_runner(tmp_path)
    fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    runs = argv_of(runner, "docker", "run")
    assert len(runs) == 2
    for argv, role in zip(runs, ROLES, strict=True):
        assert "-d" in argv, "切り離して起こしていない"
        assert "--rm" not in argv, "終了時に自動で消す指定が付いている"
        assert "--name" in argv and PLAN_OF[role].container_name in argv
        assert f"{LABEL_OWNER}={OWNER}" in argv


def test_no_token_reaches_the_fetch_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mac の側にトークンを置いても、取得のコンテナには渡らない (requirements 2.6)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_secret_value")
    runner = fetch_runner(tmp_path)
    fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    # トークンの値そのものは、どの呼び出しにも現れない
    for argv in runner.argvs:
        for item in argv:
            assert "hf_secret_value" not in item
    for argv in argv_of(runner, "docker", "run"):
        # 取得のコンテナに渡る環境変数は、構成に書いた 1 つだけ
        env = [value for flag, value in zip(argv, argv[1:], strict=False) if flag == "-e"]
        assert env == ["HF_HUB_DISABLE_TELEMETRY=1"]
        # 秘密らしい名前 (この構成の引数に、トークンらしい語は 1 つもない)
        for item in argv:
            assert "TOKEN" not in item.upper()


def test_the_gates_refuse_before_the_approval_when_the_disk_is_full(tmp_path: Path) -> None:
    runner = fetch_runner(tmp_path, FetchScript(avail="Avail\n1000\n"))
    confirmer = SpyConfirmer()
    outcome = fetch_weights(
        runner, fetch_config(), confirmer=confirmer, record_dir=tmp_path / "records"
    )

    assert outcome.status == "refused"
    assert "空き" in outcome.detail
    assert confirmer.shown == []
    assert argv_of(runner, "docker", "run") == ()
    assert mutating_calls(runner) == ()


# --- 取得: 2 台の待ち ----------------------------------------------------


def test_both_nodes_are_awaited_even_when_one_finishes_first(tmp_path: Path) -> None:
    clock = FakeClock()
    runner = fetch_runner(
        tmp_path,
        FetchScript(
            states={
                "head": (Reply(stdout="exited 0\n"),),
                "worker": (
                    Reply(stdout="running 0\n"),
                    Reply(stdout="running 0\n"),
                    Reply(stdout="exited 0\n"),
                ),
            }
        ),
    )
    outcome = fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records", clock=clock)

    assert outcome.status == "fetched"
    assert {fetched.node: fetched.exit_code for fetched in outcome.fetches} == {
        "head": 0,
        "worker": 0,
    }
    # 先に終わった台は、もう見に行かない (待ちは、終わっていない台だけを見る)
    head_states = [
        call for call in runner.runs if call.node == "head" and call.argv[1] == "container"
    ]
    assert len(head_states) == 1
    assert clock.slept, "待ちの間に、一度も眠っていない"


def test_a_failing_fetch_container_shows_the_log_tail_and_fails(tmp_path: Path) -> None:
    runner = fetch_runner(
        tmp_path,
        FetchScript(
            states={"head": (Reply(stdout="exited 1\n"),), "worker": (Reply(stdout="exited 0\n"),)},
            logs=Reply(stdout="Traceback\nOSError: Consistency check failed\n"),
        ),
    )
    with pytest.raises(w.WeightsFetchError) as caught:
        fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    message = str(caught.value)
    assert "head" in message
    assert "Consistency check failed" in message
    # 終了したコンテナは、成功でも失敗でも片付ける
    assert len(argv_of(runner, "docker", "stop")) == 2
    assert len(argv_of(runner, "docker", "rm")) == 2
    # 照合には進まない
    assert argv_of(runner, "sha256sum") == ()


def test_a_missing_hf_in_the_image_is_named(tmp_path: Path) -> None:
    runner = fetch_runner(
        tmp_path,
        FetchScript(
            states={
                "head": (Reply(stdout="exited 127\n"),),
                "worker": (Reply(stdout="exited 0\n"),),
            },
            logs=Reply(stdout='exec: "hf": executable file not found in $PATH\n'),
        ),
    )
    with pytest.raises(w.WeightsFetchError) as caught:
        fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    message = str(caught.value)
    assert "hf" in message
    assert "計測者" in message, "ホストに入れずに、計測者に尋ねることを言っていない"


def test_a_fetch_that_does_not_finish_in_time_is_not_stopped(tmp_path: Path) -> None:
    runner = fetch_runner(
        tmp_path,
        FetchScript(
            states={role: (Reply(stdout="running 0\n"),) for role in ROLES},
        ),
    )
    with pytest.raises(w.WeightsFetchError) as caught:
        fetch_weights(
            runner,
            fetch_config(ready_timeout_s=30),
            record_dir=tmp_path / "records",
            clock=FakeClock(),
        )

    assert "取得は続いている" in str(caught.value)
    assert argv_of(runner, "docker", "stop") == ()
    assert argv_of(runner, "docker", "rm") == ()


# --- 取得: 二重に取得しない ---------------------------------------------


def test_a_running_fetch_is_awaited_instead_of_started_again(tmp_path: Path) -> None:
    runner = fetch_runner(
        tmp_path,
        FetchScript(
            listings={
                role: (
                    Reply(stdout=ps_line(PLAN_OF[role], state="running")),
                    Reply(stdout=ps_line(PLAN_OF[role], state="exited")),
                )
                for role in ROLES
            },
            states={role: (Reply(stdout="exited 0\n"),) for role in ROLES},
        ),
    )
    outcome = fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    assert outcome.status == "fetched"
    assert argv_of(runner, "docker", "run") == (), "動いている取得があるのに、新しく起こした"
    assert all(fetched.attached for fetched in outcome.fetches)
    # 合流したときも、終わったコンテナは片付ける
    assert len(argv_of(runner, "docker", "stop")) == 2


def test_only_the_node_without_a_running_fetch_is_started(tmp_path: Path) -> None:
    # head では、この構成の取得が動いている。worker には、自分のコンテナが 1 つもない
    runner = fetch_runner(
        tmp_path,
        FetchScript(
            listings={
                "head": (
                    Reply(stdout=ps_line(PLAN_OF["head"], state="running")),
                    Reply(stdout=ps_line(PLAN_OF["head"], state="exited")),
                ),
                "worker": (
                    Reply(stdout=""),
                    Reply(stdout=ps_line(PLAN_OF["worker"], state="running")),
                    Reply(stdout=ps_line(PLAN_OF["worker"], state="exited")),
                ),
            },
            states={
                "head": (Reply(stdout="exited 0\n"),),
                "worker": (Reply(stdout="running 0\n"), Reply(stdout="exited 0\n")),
            },
        ),
    )
    confirmer = SpyConfirmer()
    outcome = fetch_weights(
        runner, fetch_config(), confirmer=confirmer, record_dir=tmp_path / "records"
    )

    assert outcome.status == "fetched"
    # 起こすのは worker だけ。head は、動いている取得を待つ
    runs = argv_of(runner, "docker", "run")
    assert len(runs) == 1
    assert PLAN_OF["worker"].container_name in runs[0]
    assert [call.node for call in runner.runs if call.argv[:2] == ("docker", "run")] == ["worker"]
    # 見せる計画も、起こすのは worker だけ (head は、片付けのための巻き戻しだけがある)
    shown = confirmer.shown[0]
    assert shown.count("docker run") == 1
    assert f"--name {PLAN_OF['worker'].container_name}" in shown
    assert f"docker stop -t 90 {PLAN_OF['head'].container_name}" in shown
    # 2 台とも終わるまで待ち、どちらも片付ける
    assert {fetched.node: fetched.attached for fetched in outcome.fetches} == {
        "head": True,
        "worker": False,
    }
    assert len(argv_of(runner, "docker", "stop")) == 2
    assert len(argv_of(runner, "docker", "rm")) == 2


def _other_config_line(state: str) -> str:
    """別の構成の、自分のコンテナの 1 行 (名前が違う)。"""
    labels = {
        **PLAN_OF["head"].labels,
        LABEL_CONFIG: "p1-nvfp4-tp2",
        LABEL_KIND: "serve",
    }
    return ps_row("vb-p1-nvfp4-tp2-head", "aaaabbbbcccc", state, labels)


def _different_sha_line() -> str:
    """名前は同じで、`config-sha256` が違う行。"""
    labels = {**PLAN_OF["head"].labels, LABEL_CONFIG_SHA256: "0" * 64}
    return ps_line(PLAN_OF["head"], state="running", labels=labels)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (_other_config_line("running"), "serve stop"),
        (ps_line(PLAN_OF["head"], state="exited"), "serve logs"),
        (_different_sha_line(), LABEL_CONFIG_SHA256),
    ],
    ids=["ほかの構成が動いている", "終了した自分のコンテナが残っている", "config-sha256 が違う"],
)
def test_a_conflicting_own_container_refuses_before_starting_anything(
    tmp_path: Path, line: str, expected: str
) -> None:
    runner = refuse_runner(tmp_path, {"head": line, "worker": ""})
    confirmer = SpyConfirmer()
    outcome = fetch_weights(
        runner, fetch_config(), confirmer=confirmer, record_dir=tmp_path / "records"
    )

    assert outcome.status == "refused"
    assert expected in outcome.detail
    # 1 台でも合わなければ、どの台でも起こさない
    assert argv_of(runner, "docker", "run") == ()
    assert confirmer.shown == []
    assert mutating_calls(runner) == ()


# --- 取得: 途切れと中断で、取得を止めない -------------------------------


def test_a_failure_while_starting_does_not_stop_the_fetch_that_already_runs(
    tmp_path: Path,
) -> None:
    # head は起きた。worker の `docker run` が 0 以外で返る (名前の衝突など)
    runner = fetch_runner(
        tmp_path,
        FetchScript(
            run=Reply(stdout="0123456789ab\n"),
            runs={
                "head": (Reply(stdout="0123456789ab\n"),),
                "worker": (
                    Reply(
                        exit_code=125,
                        stderr="docker: Error response from daemon: Conflict. The container name"
                        ' "/vb-p1-fetch-nvfp4-worker" is already in use',
                    ),
                ),
            },
        ),
    )
    with pytest.raises(w.WeightsFetchError) as caught:
        fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    message = str(caught.value)
    # (i) いま動いている取得の名前
    assert PLAN_OF["head"].container_name in message
    # (ii) 失敗した台と理由
    assert "worker" in message
    assert "already in use" in message
    # (iii) もう一度打つと、動いている台は待ち、起きていない台だけを起こす
    assert "serve fetch" in message
    assert "起きていない台だけを起こす" in message
    # (iv) 止めたいときは serve stop
    assert "serve stop" in message
    # 起きた取得は、止めない
    assert argv_of(runner, "docker", "stop") == ()
    assert argv_of(runner, "docker", "rm") == ()


def test_a_remote_error_while_starting_does_not_stop_the_fetch_that_already_runs(
    tmp_path: Path,
) -> None:
    dropped = RemoteError(
        "つながらなかった", node="worker", ssh_host="spark-5083", argv=("docker", "run")
    )
    runner = fetch_runner(
        tmp_path,
        FetchScript(
            runs={
                "head": (Reply(stdout="0123456789ab\n"),),
                "worker": (Reply(raises=dropped),),
            }
        ),
    )
    with pytest.raises(w.WeightsFetchError) as caught:
        fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    message = str(caught.value)
    assert PLAN_OF["head"].container_name in message
    assert "worker" in message
    assert "起きていない台だけを起こす" in message
    assert "serve stop" in message
    assert argv_of(runner, "docker", "stop") == ()
    assert argv_of(runner, "docker", "rm") == ()


def test_a_container_that_disappeared_while_waiting_keeps_the_fetch(tmp_path: Path) -> None:
    runner = fetch_runner(
        tmp_path,
        FetchScript(
            states={
                "head": (Reply(stdout="running 0\n"),),
                "worker": (Reply(exit_code=1, stderr="Error: No such container: vb-…"),),
            }
        ),
    )
    with pytest.raises(w.WeightsFetchError) as caught:
        fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    message = str(caught.value)
    assert "見つからなくなった" in message
    assert "起きていない台だけを起こす" in message
    assert "serve stop" in message
    assert argv_of(runner, "docker", "stop") == ()
    assert argv_of(runner, "docker", "rm") == ()


def test_a_remote_error_while_waiting_does_not_stop_the_fetch(tmp_path: Path) -> None:
    dropped = RemoteError(
        "つながらなかった", node="worker", ssh_host="spark-5083", argv=("docker", "container")
    )
    runner = fetch_runner(
        tmp_path,
        FetchScript(
            states={
                "head": (Reply(stdout="running 0\n"),),
                "worker": (Reply(raises=dropped),),
            }
        ),
    )
    with pytest.raises(w.WeightsFetchError) as caught:
        fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    assert "取得は続いている" in str(caught.value)
    assert "serve fetch" in str(caught.value)
    assert argv_of(runner, "docker", "stop") == ()
    assert argv_of(runner, "docker", "rm") == ()


def test_an_interrupt_while_waiting_does_not_stop_the_fetch(tmp_path: Path) -> None:
    runner = fetch_runner(
        tmp_path,
        FetchScript(
            states={
                "head": (Reply(stdout="running 0\n"),),
                "worker": (Reply(raises=KeyboardInterrupt()),),
            }
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    assert argv_of(runner, "docker", "stop") == (), "中断で、取得のコンテナを止めた"
    assert argv_of(runner, "docker", "rm") == ()


def test_the_finished_container_is_cleaned_up_only_after_the_listing_confirms_it(
    tmp_path: Path,
) -> None:
    # 一覧に名前が無い台では、`docker stop` も `docker rm` も出さない (3.1 の巻き戻しの決まり)
    runner = fetch_runner(
        tmp_path,
        FetchScript(
            listings={
                "head": (
                    Reply(stdout=""),
                    Reply(stdout=ps_line(PLAN_OF["head"], state="running")),
                    Reply(stdout=ps_line(PLAN_OF["head"], state="exited")),
                ),
                "worker": (
                    Reply(stdout=""),
                    Reply(stdout=ps_line(PLAN_OF["worker"], state="running")),
                    Reply(stdout=""),
                ),
            }
        ),
    )
    fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    stops = argv_of(runner, "docker", "stop")
    assert len(stops) == 1
    assert stops[0] == ("docker", "stop", "-t", "90", PLAN_OF["head"].container_name)


# --- 照合 ----------------------------------------------------------------


def test_one_wrong_sha256_line_names_the_file_and_refetches_nothing(tmp_path: Path) -> None:
    wrong = {"model-00002-of-00002.safetensors": "f" * 64}
    runner = verify_runner(
        tmp_path,
        sha={
            "head": (Reply(stdout=sha_output(ALL_PATHS, WEIGHTS_DIR)),),
            "worker": (Reply(stdout=sha_output(ALL_PATHS, WEIGHTS_DIR, wrong=wrong)),),
        },
    )
    with pytest.raises(w.WeightsMismatchError) as caught:
        verify_weights(runner, serve_config(), record_dir=tmp_path / "records")

    message = str(caught.value)
    assert "model-00002-of-00002.safetensors" in message
    assert "worker" in message
    # Hub の重みは、`serve fetch` を打ち直して取り直せる (この案内は変えない)
    assert "黙って取り直さない。取り直すときは、計測者が `serve fetch` を打ち直す" in message
    # 黙って取り直さない: 取り直しの呼び出しが 1 つも出ない
    assert argv_of(runner, "docker", "run") == ()
    assert argv_of(runner, "docker", "pull") == ()
    assert argv_of(runner, "docker", "rm") == ()
    # 合わなかったことは、記録に残る (関門が、この記録で断れるように)
    record = json.loads(
        (record_source(tmp_path, "worker") / f"{SLUG}.verified.json").read_text(encoding="utf-8")
    )
    assert record["mismatched"] == ["model-00002-of-00002.safetensors"]


def test_a_mismatch_after_a_fetch_does_not_refetch_anything(tmp_path: Path) -> None:
    # 取得の道でも、照合が合わなかったあとに、取り直しの呼び出しを 1 つも出さない
    wrong = {"config.json": "f" * 64}
    runner = fetch_runner(
        tmp_path,
        FetchScript(
            sha={
                "head": (Reply(stdout=sha_output(ALL_PATHS, WEIGHTS_DIR, wrong=wrong)),),
                "worker": (Reply(stdout=sha_output(ALL_PATHS, WEIGHTS_DIR)),),
            }
        ),
    )
    with pytest.raises(w.WeightsMismatchError) as caught:
        fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    assert "config.json" in str(caught.value)
    named = steps(runner)
    after = named[len(named) - 1 - named[::-1].index("sha256sum") :]
    assert "run" not in after
    assert "pull" not in after
    assert "rm" not in after


def test_a_missing_file_is_named_too(tmp_path: Path) -> None:
    missing = ("tokenizer.json",)
    runner = verify_runner(
        tmp_path,
        sha={
            "head": (
                Reply(
                    exit_code=1,
                    stdout=sha_output(ALL_PATHS, WEIGHTS_DIR, missing=missing),
                    stderr=missing_stderr(missing, WEIGHTS_DIR),
                ),
            ),
            "worker": (Reply(stdout=sha_output(ALL_PATHS, WEIGHTS_DIR)),),
        },
    )
    with pytest.raises(w.WeightsMismatchError) as caught:
        verify_weights(runner, serve_config(), record_dir=tmp_path / "records")

    assert "tokenizer.json" in str(caught.value)


def test_an_unreadable_line_makes_the_whole_batch_unverified(tmp_path: Path) -> None:
    runner = verify_runner(
        tmp_path,
        sha={
            "head": (
                Reply(stdout=sha_output(ALL_PATHS, WEIGHTS_DIR, extra_lines=("おかしな行\n",))),
            ),
            "worker": (Reply(stdout=sha_output(ALL_PATHS, WEIGHTS_DIR)),),
        },
    )
    with pytest.raises(w.WeightsMismatchError) as caught:
        verify_weights(runner, serve_config(), record_dir=tmp_path / "records")

    message = str(caught.value)
    assert "おかしな行" in message
    for path in ALL_PATHS:
        assert path in message


def test_a_matching_verification_writes_the_record_to_state(tmp_path: Path) -> None:
    runner = verify_runner(tmp_path)
    confirmer = SpyConfirmer()
    outcome = verify_weights(
        runner, serve_config(), confirmer=confirmer, record_dir=tmp_path / "records"
    )

    assert outcome.status == "verified"
    assert all(verification.ok for verification in outcome.verifications)
    # 配布は、了承済みの計画にあり、宛先は state/ で、--delete を付けない
    assert len(runner.pushes) == 2
    for push in runner.pushes:
        assert push.remote == "state"
        assert push.delete is False
        assert push.mutating is True
    assert "state/" in confirmer.shown[0]
    # 記録の道筋は、`guards.verification_record_path` が決めたものと同じ名前にする
    expected = verification_record_path(HEAD, REPO, "all").rsplit("/", 1)[-1]
    place = record_source(tmp_path, "head") / expected
    assert place.is_file()
    record = json.loads(place.read_text(encoding="utf-8"))
    assert record["scope"] == "all"
    assert record["node"] == "head"
    assert record["file_count"] == len(ALL_PATHS)
    assert record["total_bytes"] == MANIFEST.total_bytes
    assert record["mismatched"] == []


def test_the_record_is_pushed_from_a_directory_of_its_own(tmp_path: Path) -> None:
    # 5.1 が `logs.var_dir` を渡しても、回収した記録が Spark の state/ に逆流しない (指摘 3)
    record_dir = tmp_path / "records"
    (record_dir / "head").mkdir(parents=True)
    (record_dir / "container.stdout.log").write_text("よその記録", encoding="utf-8")
    (record_dir / "head" / "container.stdout.log").write_text("よその記録", encoding="utf-8")
    runner = verify_runner(tmp_path)
    verify_weights(runner, serve_config(), record_dir=record_dir)

    assert len(runner.pushes) == 2
    for push in runner.pushes:
        assert push.local_dir is not None
        assert push.local_dir != record_dir
        assert push.local_dir != record_dir / push.node
        names = sorted(item.name for item in push.local_dir.iterdir())
        assert names == [f"{SLUG}.verified.json"], f"配る元に、よそのものが入っている: {names}"
    # 呼ぶ側が置いたファイルは、消さない
    assert (record_dir / "container.stdout.log").is_file()
    assert (record_dir / "head" / "container.stdout.log").is_file()


def test_a_symlinked_record_source_is_refused(tmp_path: Path) -> None:
    # `rmtree` はリンクを消さないので、リンク先のよそのファイルが、配る元に現れてしまう
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "secret.txt").write_text("よそのもの", encoding="utf-8")
    source = record_source(tmp_path, "head")
    source.parent.mkdir(parents=True)
    source.symlink_to(elsewhere, target_is_directory=True)
    runner = verify_runner(tmp_path)

    with pytest.raises(w.WeightsError, match="シンボリックリンク"):
        verify_weights(runner, serve_config(), record_dir=tmp_path / "records")

    assert runner.pushes == ()
    assert (elsewhere / "secret.txt").is_file()


@pytest.mark.parametrize("record_dir", [Path(""), Path("records"), Path("./var/x")])
def test_a_relative_record_dir_is_refused_before_touching_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_dir: Path
) -> None:
    # 配る元は、配る前に空にする (消す操作)。相対の道筋だと、作業ディレクトリに対して効く
    monkeypatch.chdir(tmp_path)
    runner = verify_runner(tmp_path)

    with pytest.raises(ValueError, match="絶対"):
        verify_weights(runner, serve_config(), record_dir=record_dir)

    assert runner.calls == ()
    assert not (tmp_path / "verified").exists()


def test_a_stale_record_of_another_scope_is_not_pushed_again(tmp_path: Path) -> None:
    # 配る元には、今回の 1 つだけがある (前の回の、別の範囲の記録を、配り直さない)
    stale = record_source(tmp_path, "head") / f"{SLUG}.probe.verified.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")
    runner = verify_runner(tmp_path)
    verify_weights(runner, serve_config(), record_dir=tmp_path / "records")

    names = sorted(item.name for item in record_source(tmp_path, "head").iterdir())
    assert names == [f"{SLUG}.verified.json"]


def _gate_on(tmp_path: Path, text: str) -> str:
    """書いた記録を `guards.gate_weights_verified` に読ませて、その判定の文を返す。"""
    runner = FakeRunner(
        var_root=tmp_path,
        script=(Rule(prefix=("cat",), replies=(Reply(stdout=text),)),),
    )
    gate = gate_weights_verified(runner, HEAD, serve_config(), MANIFEST)
    return f"{gate.passed} {gate.detail}"


def test_the_record_this_module_writes_is_read_back_by_the_gate(tmp_path: Path) -> None:
    # 合った記録は関門を通り、合わなかった記録は関門が断る (この module と guards の継ぎ目)
    place = record_source(tmp_path, "head") / f"{SLUG}.verified.json"
    verify_weights(verify_runner(tmp_path), serve_config(), record_dir=tmp_path / "records")
    assert _gate_on(tmp_path, place.read_text(encoding="utf-8")).startswith("True")

    wrong = {"config.json": "f" * 64}
    broken = verify_runner(
        tmp_path,
        sha={
            "head": (Reply(stdout=sha_output(ALL_PATHS, WEIGHTS_DIR, wrong=wrong)),),
            "worker": (Reply(stdout=sha_output(ALL_PATHS, WEIGHTS_DIR)),),
        },
    )
    with pytest.raises(w.WeightsMismatchError):
        verify_weights(broken, serve_config(), record_dir=tmp_path / "records")
    judged = _gate_on(tmp_path, place.read_text(encoding="utf-8"))
    assert judged.startswith("False")
    assert "config.json" in judged


def test_an_unexpected_sha256_line_makes_the_whole_batch_unverified(tmp_path: Path) -> None:
    # 渡していない道筋の行が返ったら、その回のファイルを、全部「確かめられなかった」に倒す
    extra = sha_line("よその.json", "/etc")
    runner = verify_runner(
        tmp_path,
        sha={
            "head": (Reply(stdout=sha_output(ALL_PATHS, WEIGHTS_DIR, extra_lines=(extra,))),),
            "worker": (Reply(stdout=sha_output(ALL_PATHS, WEIGHTS_DIR)),),
        },
    )
    with pytest.raises(w.WeightsMismatchError) as caught:
        verify_weights(runner, serve_config(), record_dir=tmp_path / "records")

    message = str(caught.value)
    assert "/etc/よその.json" in message
    for path in ALL_PATHS:
        assert path in message


def test_verify_refuses_while_the_weights_are_being_fetched(tmp_path: Path) -> None:
    # 取得の途中のファイルの sha256 を、184 GiB ぶん計算しない (指摘 9)
    runner = verify_runner(
        tmp_path,
        listing={"head": ps_line(PLAN_OF["head"], state="running"), "worker": ""},
    )
    outcome = verify_weights(runner, serve_config(), record_dir=tmp_path / "records")

    assert outcome.status == "refused"
    assert "取得" in outcome.detail
    assert PLAN_OF["head"].container_name in outcome.detail
    assert argv_of(runner, "sha256sum") == ()
    assert runner.pushes == ()


def test_verify_runs_when_a_finished_fetch_container_is_left(tmp_path: Path) -> None:
    # 動いていない取得は、照合を止めない (読み取りだけなので、残っていてもよい)
    runner = verify_runner(
        tmp_path,
        listing={"head": ps_line(PLAN_OF["head"], state="exited"), "worker": ""},
    )
    outcome = verify_weights(runner, serve_config(), record_dir=tmp_path / "records")

    assert outcome.status == "verified"


def test_nothing_is_verified_when_the_operator_refuses(tmp_path: Path) -> None:
    runner = verify_runner(tmp_path)
    with pytest.raises(ApprovalError):
        verify_weights(
            runner,
            serve_config(),
            confirmer=SpyConfirmer(approves=False),
            record_dir=tmp_path / "records",
        )

    assert runner.pushes == ()
    assert mutating_calls(runner) == ()


def test_verify_refuses_when_the_weights_directory_is_missing(tmp_path: Path) -> None:
    runner = verify_runner(tmp_path, layout_ok=False)
    outcome = verify_weights(runner, serve_config(), record_dir=tmp_path / "records")

    assert outcome.status == "refused"
    assert "serve push" in outcome.detail
    assert argv_of(runner, "sha256sum") == ()
    assert runner.pushes == ()


def test_the_probe_scope_verifies_only_the_files_that_are_not_safetensors(tmp_path: Path) -> None:
    script = FetchScript(
        directory=PROBE_DIR,
        paths=PROBE_PATHS,
        roles=("head",),
        listings={
            "head": (
                Reply(stdout=""),
                Reply(stdout=ps_line(PROBE_PLAN, state="running")),
                Reply(stdout=ps_line(PROBE_PLAN, state="exited")),
            )
        },
        states={"head": (Reply(stdout="exited 0\n"),)},
        sha={"head": (Reply(stdout=sha_output(PROBE_PATHS, PROBE_DIR)),)},
    )
    runner = FakeRunner(var_root=tmp_path, script=script.rules())
    outcome = fetch_weights(
        runner, probe_fetch_config(), scope="probe_files", record_dir=tmp_path / "records"
    )

    assert outcome.status == "fetched"
    paths = sha_paths(runner)
    assert paths, "sha256sum が 1 度も流れていない"
    assert all(not path.endswith(".safetensors") for path in paths)
    assert set(paths) == {f"{PROBE_DIR}/{path}" for path in PROBE_PATHS}
    # 記録は、全体の照合と別のファイルに置く
    name = verification_record_path(HEAD, REPO, "probe_files").rsplit("/", 1)[-1]
    assert name == f"{SLUG}.probe.verified.json"
    record = json.loads((record_source(tmp_path, "head") / name).read_text(encoding="utf-8"))
    assert record["scope"] == "probe_files"
    assert record["file_count"] == len(PROBE_PATHS)


def test_the_paths_are_split_into_batches(tmp_path: Path) -> None:
    files = MANIFEST.files
    runner = verify_runner(
        tmp_path,
        sha={
            role: (
                Reply(stdout=sha_output([entry.path for entry in files[:2]], WEIGHTS_DIR)),
                Reply(stdout=sha_output([entry.path for entry in files[2:4]], WEIGHTS_DIR)),
                Reply(stdout=sha_output([entry.path for entry in files[4:]], WEIGHTS_DIR)),
            )
            for role in ROLES
        },
    )
    outcome = verify_weights(runner, serve_config(), record_dir=tmp_path / "records", batch_files=2)

    assert outcome.status == "verified"
    calls = [call for call in runner.runs if call.argv[0] == "sha256sum" and call.node == "head"]
    assert len(calls) == 3
    assert sha_paths(runner, "head") == tuple(f"{WEIGHTS_DIR}/{entry.path}" for entry in files)


# --- 安全の不変条件 ------------------------------------------------------

_CONTAINER_SUBCOMMANDS = frozenset(
    {"stop", "rm", "logs", "top", "kill", "start", "restart", "pause", "unpause", "wait", "port"}
)
"""コンテナを対象に取る docker のサブコマンド (許可の一覧の外のものも並べて見張る)。"""

_VALUE_FLAGS = frozenset(
    {"-t", "--time", "--tail", "--since", "--until", "-s", "--signal", "--format"}
)
"""値を取るフラグ (`docker stop -t 90 <名前>` の 90 を、対象と読み違えないため)。"""


def container_targets(argv: Sequence[str]) -> tuple[str, ...]:
    """docker の呼び出しから、コンテナを対象にしている語だけを取り出す
    (`test_image.py` と同じ読み方)。"""
    if tuple(argv[:1]) != ("docker",) or len(argv) < 2:
        return ()
    rest = list(argv[1:])
    if rest[0] == "container" and len(rest) > 1:
        if rest[1] != "inspect":
            return ()
        rest = rest[2:]
    elif rest[0] == "run":
        return tuple(value for flag, value in zip(rest, rest[1:], strict=False) if flag == "--name")
    elif rest[0] in _CONTAINER_SUBCOMMANDS:
        rest = rest[1:]
    else:
        return ()
    targets: list[str] = []
    skip = False
    for item in rest:
        if skip:
            skip = False
            continue
        if item in _VALUE_FLAGS:
            skip = True
            continue
        if item.startswith("-"):
            continue
        targets.append(item)
    return tuple(targets)


def test_every_docker_call_targets_only_our_containers(tmp_path: Path) -> None:
    runner = fetch_runner(tmp_path)
    fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    # 「一覧が返したもの = 自分のもの」を前提にすると循環するので、試験の側でも、台本に
    # 書いた行の所有のラベルを見てから、触ってよい識別子として数える
    allowed: set[str] = set()
    for role in ROLES:
        row = json.loads(ps_line(PLAN_OF[role], state="exited"))
        assert f"{LABEL_OWNER}={OWNER}" in row["Labels"]
        allowed.add(row["ID"])
        allowed.add(PLAN_OF[role].container_name)
    checked = 0
    for argv in runner.argvs:
        for target in container_targets(argv):
            checked += 1
            assert target in allowed, f"一覧にも、了承済みの計画にもない対象: {target} ({argv})"
    assert checked >= 5, "見張るべき対象が足りない (run, container inspect, logs, stop, rm)"


def test_the_container_list_is_always_filtered_by_our_label(tmp_path: Path) -> None:
    runner = fetch_runner(tmp_path)
    fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    listings = argv_of(runner, "docker", "ps")
    assert listings, "コンテナの一覧を 1 度も取っていない"
    for argv in listings:
        assert f"label={OWNER_FILTER}" in argv, f"絞らない一覧を取った: {argv}"


def test_nothing_is_installed_on_the_host(tmp_path: Path) -> None:
    runner = fetch_runner(tmp_path)
    fetch_weights(runner, fetch_config(), record_dir=tmp_path / "records")

    for argv in runner.argvs:
        assert argv[0] not in ("pip", "pip3", "apt", "apt-get", "hf", "huggingface-cli", "sudo")
    # そもそも、ホストで `hf` を呼ぶ列は、`remote` が断る (経路そのものがない)
    with pytest.raises(RuntimeError):
        runner.run(HEAD, ("hf", "download", REPO), timeout_s=1.0, mutating=False)


def test_the_fetch_refuses_a_config_of_another_kind(tmp_path: Path) -> None:
    runner = fetch_runner(tmp_path)
    with pytest.raises(ConfigError):
        fetch_weights(runner, serve_config(), record_dir=tmp_path / "records")

    assert runner.calls == ()


def test_a_config_that_does_not_mount_the_weights_is_refused_before_touching_spark(
    tmp_path: Path,
) -> None:
    # 184 GiB を取得したあとに断らない (design.md 「Error Handling」の「早く断る」)
    config = fetch_config().model_copy(
        update={"docker": {"entrypoint": _setting("--entrypoint", "hf")}}
    )
    runner = fetch_runner(tmp_path)
    with pytest.raises(ConfigError) as caught:
        fetch_weights(runner, config, record_dir=tmp_path / "records")

    assert MOUNT_AT in str(caught.value)
    assert runner.calls == ()


def test_a_manifest_that_does_not_match_the_config_is_refused(tmp_path: Path) -> None:
    runner = fetch_runner(tmp_path)
    other = MANIFEST.model_copy(update={"revision": "c" * 40})
    with pytest.raises(w.WeightsRefError):
        w.fetch_weights(
            runner,
            fetch_config(),
            NODES,
            other,
            STARTED_AT,
            confirmer=SpyConfirmer(),
            scope="all",
            record_dir=tmp_path / "records",
            verified_at=VERIFIED_AT,
        )

    assert runner.calls == ()


# --- 派生の重み (手元で変換した重み) --------------------------------------

DERIVED_NAME = "k2s1"
DERIVED_MOUNT_AT = f"/models/{DERIVED_NAME}"
DERIVED_DIR = f"{REMOTE_ROOT}/models/{DERIVED_NAME}"
DERIVED_RECORD_NAME = f"{DERIVED_NAME}.derived.verified.json"


def derivation(*, origin_revision: str = REVISION) -> kit_types.Derivation:
    """派生の同一性 (元の重み、変換の条件)。構成、マニフェスト、記録が同じものを持つ。"""
    return kit_types.Derivation(
        name=DERIVED_NAME,
        origin=kit_types.WeightsOrigin(repo=REPO, revision=origin_revision),
        conversion=kit_types.ConversionSpec(
            tool="experiments/k2-quant/convert.py",
            commit="a" * 40,
            args=("--dtype", "fp8"),
            target_pattern=r"^model\.layers\.\d+\.self_attn\..*$",
        ),
    )


def derived_manifest(*, origin_revision: str = REVISION) -> kit_types.DerivedWeightsManifest:
    """変換の結果の照合の正解 (#56 の道具が書き、Mac でコミットしたもの)。"""
    return kit_types.DerivedWeightsManifest(
        kind="derived",
        derivation=derivation(origin_revision=origin_revision),
        generated_at=GENERATED_AT,
        total_bytes=MANIFEST.total_bytes,
        files=MANIFEST.files,
    )


def derived_weights() -> kit_types.DerivedWeightsRef:
    return kit_types.DerivedWeightsRef(
        kind="derived",
        name=DERIVED_NAME,
        origin=kit_types.OriginWeightsRef(
            repo=REPO, revision=REVISION, manifest=f"{SLUG}.manifest.json"
        ),
        conversion=derivation().conversion,
        manifest=f"{DERIVED_NAME}.manifest.json",
        mount_at=DERIVED_MOUNT_AT,
    )


def derived_serve_config() -> ConfigDef:
    """派生の重みを、読み取り専用で結び付ける推論サーバーの構成。"""
    return serve_config().model_copy(
        update={
            "name": "p2-nope-tp2-full-k2s1",
            "weights": derived_weights(),
            "docker": {
                "models": _setting(
                    "--mount",
                    f"type=bind,source={{remote_root}}/models/{DERIVED_NAME},"
                    "target={weights.mount_at},readonly",
                ),
            },
        }
    )


def derived_fetch_config() -> ConfigDef:
    """派生の重みを持つ `fetch` の構成 (`kind` の検査は通るので、派生であることで断られる)。"""
    return fetch_config().model_copy(update={"weights": derived_weights()})


def derived_verify_runner(
    tmp_path: Path, *, wrong: Mapping[NodeRole, Mapping[str, str]] | None = None
) -> FakeRunner:
    """派生の置き場所 (`models/k2s1`) を読み取る `serve verify` の台本。"""
    wrong = wrong or {}
    return verify_runner(
        tmp_path,
        sha={
            role: (Reply(stdout=sha_output(ALL_PATHS, DERIVED_DIR, wrong=wrong.get(role, {}))),)
            for role in ROLES
        },
    )


def verify_derived(
    runner: FakeRunner,
    tmp_path: Path,
    *,
    config: ConfigDef | None = None,
    manifest: kit_types.DerivedWeightsManifest | WeightsManifest | None = None,
) -> Any:
    """`w.verify_weights` を、派生の構成と派生のマニフェストで呼ぶ。"""
    return w.verify_weights(
        runner,
        derived_serve_config() if config is None else config,
        NODES,
        derived_manifest() if manifest is None else manifest,
        STARTED_AT,
        confirmer=SpyConfirmer(),
        scope="all",
        record_dir=tmp_path / "records",
        verified_at=VERIFIED_AT,
        batch_files=100,
    )


def test_a_derived_verification_reads_its_own_directory_and_writes_a_derived_record(
    tmp_path: Path,
) -> None:
    runner = derived_verify_runner(tmp_path)
    outcome = verify_derived(runner, tmp_path)

    assert outcome.status == "verified"
    assert set(sha_paths(runner)) == {f"{DERIVED_DIR}/{path}" for path in ALL_PATHS}
    # 配る記録は、派生の名前のものが 1 つだけ (Hub の `<slug>.verified.json` ではない)
    for role in ROLES:
        names = sorted(item.name for item in record_source(tmp_path, role).iterdir())
        assert names == [DERIVED_RECORD_NAME]
    for push in runner.pushes:
        assert push.remote == "state"
        assert push.delete is False
    record = json.loads(
        (record_source(tmp_path, "head") / DERIVED_RECORD_NAME).read_text(encoding="utf-8")
    )
    assert record["kind"] == "derived"
    assert record["derivation"] == derivation().model_dump(mode="json")
    assert record["scope"] == "all"
    assert record["node"] == "head"
    assert record["file_count"] == len(ALL_PATHS)
    assert record["total_bytes"] == MANIFEST.total_bytes
    assert record["mismatched"] == []


def test_a_derived_record_carries_the_content_sha256_of_the_verified_manifest(
    tmp_path: Path,
) -> None:
    """照合の記録は、照合の正解にしたマニフェストの中身の SHA-256 を、2 台とも持つ。"""
    verify_derived(derived_verify_runner(tmp_path), tmp_path)

    for role in ROLES:
        record = json.loads(
            (record_source(tmp_path, role) / DERIVED_RECORD_NAME).read_text(encoding="utf-8")
        )
        assert record["manifest_sha256"] == derived_manifest().content_sha256


def test_the_derived_record_this_module_writes_is_read_back_by_the_gate(tmp_path: Path) -> None:
    # 派生の記録は、この module と guards の継ぎ目で、同じ道筋・同じ形で読める
    verify_derived(derived_verify_runner(tmp_path), tmp_path)
    text = (record_source(tmp_path, "head") / DERIVED_RECORD_NAME).read_text(encoding="utf-8")
    reader = FakeRunner(
        var_root=tmp_path,
        script=(Rule(prefix=("cat",), replies=(Reply(stdout=text),)),),
    )

    gate = gate_weights_verified(reader, HEAD, derived_serve_config(), derived_manifest())

    assert gate.passed, gate.detail


def test_a_derived_mismatch_names_the_file_records_it_and_refetches_nothing(
    tmp_path: Path,
) -> None:
    wrong: dict[NodeRole, Mapping[str, str]] = {
        "worker": {"model-00002-of-00002.safetensors": "f" * 64}
    }
    runner = derived_verify_runner(tmp_path, wrong=wrong)
    with pytest.raises(w.WeightsMismatchError) as caught:
        verify_derived(runner, tmp_path)

    message = str(caught.value)
    assert "model-00002-of-00002.safetensors" in message
    assert "worker" in message
    # 取り直しの案内は、`serve fetch` が断る派生の重みでは、変換の道具での作り直しになる
    assert "変換の道具で変換し直す" in message
    assert argv_of(runner, "docker", "run") == ()
    assert argv_of(runner, "docker", "pull") == ()
    assert argv_of(runner, "docker", "rm") == ()
    # 合わなかったことは、派生の記録に残る (関門が、この記録で断れるように)
    record = json.loads(
        (record_source(tmp_path, "worker") / DERIVED_RECORD_NAME).read_text(encoding="utf-8")
    )
    assert record["kind"] == "derived"
    assert record["mismatched"] == ["model-00002-of-00002.safetensors"]


def test_a_derived_manifest_of_another_origin_revision_is_refused_before_touching_spark(
    tmp_path: Path,
) -> None:
    runner = derived_verify_runner(tmp_path)
    with pytest.raises(w.WeightsRefError):
        verify_derived(runner, tmp_path, manifest=derived_manifest(origin_revision="c" * 40))

    assert runner.calls == ()


def test_a_derived_manifest_with_another_conversion_is_refused_before_touching_spark(
    tmp_path: Path,
) -> None:
    # 元の重みが同じでも、変換の条件 (道具のコミット) が違えば、別の重みである。
    # `Derivation.identity` は変換の条件を含まないので、identity だけでは見逃す。
    base = derived_manifest()
    conversion = base.derivation.conversion.model_copy(update={"commit": "b" * 40})
    manifest = base.model_copy(
        update={"derivation": base.derivation.model_copy(update={"conversion": conversion})}
    )
    runner = derived_verify_runner(tmp_path)

    with pytest.raises(w.WeightsRefError) as caught:
        verify_derived(runner, tmp_path, manifest=manifest)

    assert "変換の道具のコミット" in str(caught.value)
    assert runner.calls == ()


def test_a_hub_manifest_for_a_derived_config_is_refused_before_touching_spark(
    tmp_path: Path,
) -> None:
    runner = derived_verify_runner(tmp_path)
    with pytest.raises(w.WeightsRefError):
        verify_derived(runner, tmp_path, manifest=MANIFEST)

    assert runner.calls == ()


def test_the_fetch_refuses_a_derived_config_without_touching_spark(tmp_path: Path) -> None:
    # 派生の重みには、Hub の取得元がない。取得の経路に乗せず、Spark に触る前に断る
    runner = fetch_runner(tmp_path)
    with pytest.raises(ConfigError):
        w.fetch_weights(
            runner,
            derived_fetch_config(),
            NODES,
            derived_manifest(),
            STARTED_AT,
            confirmer=SpyConfirmer(),
            scope="all",
            record_dir=tmp_path / "records",
            verified_at=VERIFIED_AT,
        )

    assert runner.calls == ()
