"""安全の決まりを、コマンドの全体で固定する試験 (tasks.md 5.3)。

確かめること (requirements 2.3、2.4、2.6、2.7、5.4、7.7、8.6、8.8、10.5。design.md
「Architecture」の安全の不変条件、「Testing Strategy」、「Error Handling」。tasks.md 5.3 の
完了の状態):

すべてのサブコマンドを、偽の実行役 (`FakeRunner`) と偽の推論サーバー (`FakeVllm`) で、
`--yes` で了承した状態で流し (成功する台本と、途中で失敗する台本の両方。`start`、`probe`、
`netcheck bandwidth`、`fetch` は両方を持つ)、記録された**すべての**呼び出しを 1 つの一覧に
集めてから、決まりごとに確かめる (関数を 1 つずつ)。

1. コンテナを対象にする操作は、ラベルで絞った一覧が返した識別子か、了承済みの計画が自分で
   起こす名前だけを対象にする
2. 名前とラベルのないコンテナを起こす呼び出しがない
3. 前面で動かす、または終了時に自動で消すコンテナの起動が、どの種類にもない
4. 配布の宛先が、決まった部分 (`payload/`、`state/`) だけ
5. 許可の一覧にないコマンドがない (`argv[0]` と、`docker` のサブコマンド)
6. イメージのビルドがない
7. トークンらしい環境変数がない
8. モデルカードの取得がない
9. 記録の回収と、smoke/thinking/watch の結果のファイルに、応答の本文や、送った会話の文、
   `Authorization` が含まれない

`serve manifest` は、`cli.main` に HTTP のクライアントを差し込む口がない (`_cmd_manifest` は
`weights_mod.build_manifest` を `client=` なしで呼ぶ) ので、ここだけは
`weights_mod.build_manifest` を直に、`httpx.MockTransport` を積んだクライアントで呼ぶ
(`tests/unit/test_weights.py` の手法)。Hub には、どの試験からもつながない。

`serve probe` と `serve watch` の、縮小の確認と見張りの、端から端までの深い確認は
`test_probe_watch.py` に書く。ここでは、安全の決まりを確かめるのに要る分だけを流す。

置き場所の決まりにより、触ってよいのは `serving/tests/e2e/` の下の新規ファイルだけである。
共通の下ごしらえ (構成の TOML、`Repo`、`invoke()`、8 つの関門を通す台本) は `e2e_kit.py`
(このディレクトリの新規モジュール) に置き、ここと `test_probe_watch.py` の両方から使う。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import e2e_kit as k
import httpx
import pytest

from fake_runner import FakeRunner, RecordedCall, Reply, Rule
from fake_vllm import FakeVllm, Fault, MessagesReply
from serving_kit import cli
from serving_kit import netcheck as netcheck_mod
from serving_kit import plan as plan_mod
from serving_kit import remote as remote_mod
from serving_kit import weights as weights_mod
from serving_kit.netcheck import _round_tag  # 回の札を、実装と同じ形で作るため (private だが読む)
from serving_kit.plan import LABEL_CONFIG_SHA256, LABEL_OWNER, OWNER
from serving_kit.types import ContainerPlan, NodeRole

PlanMap = Mapping[NodeRole, ContainerPlan]
"""台ごとの、組み立てたコンテナの計画 (`k.container_plans` / `plan_mod.build_plans` の返り値)。"""

NETCHECK_NOW = datetime(2026, 9, 22, 2, 50, 59, 442940, tzinfo=UTC)
"""`netcheck bandwidth` / `sanity` の `started_at` に固定して渡す時刻 (回の札を、実行の前に
計算できるようにする。`_cmd_netcheck_*` は `ctx.now()` を 1 度だけ呼んでそのまま渡すので、
`now=` を固定すれば、`netcheck._round_tag` と同じ入力から同じ札が作れる)。"""


def _fixed_now(moment: datetime = NETCHECK_NOW) -> Callable[[], datetime]:
    return lambda: moment


# --- 世界の組み立て (一度だけ、すべてのサブコマンドを流す) --------------------


@dataclass
class World:
    """すべてのシナリオを流した結果 (1 度だけ組み立てて、決まりごとの試験がそれぞれ読む)。"""

    scenarios: dict[str, k.Invocation]
    all_calls: tuple[RecordedCall, ...]
    base: Path
    fetch_config: Any
    fetch_plans: PlanMap
    vllms: dict[str, FakeVllm]
    manifest_requests: tuple[httpx.Request, ...]
    known_names: frozenset[str]
    known_ids: frozenset[str]
    var_roots: dict[str, Path]

    def calls_of(self, *names: str) -> tuple[RecordedCall, ...]:
        calls: list[RecordedCall] = []
        for name in names:
            calls.extend(self.scenarios[name].runner.calls)
        return tuple(calls)


def _sha_line(path: str, directory: str, digest: str) -> str:
    """`sha256sum` の 1 行 (`<64 桁><空白><空白か *><道筋>`。`test_weights_fetch.py` の見本)。"""
    return f"{digest}  {directory}/{path}\n"


def _sha_output(
    paths: Sequence[str], directory: str, *, wrong: Mapping[str, str] | None = None
) -> str:
    override = wrong or {}
    by_path = {entry.path: entry.sha256 for entry in k.MANIFEST.files}
    lines = [_sha_line(path, directory, override.get(path, by_path[path])) for path in paths]
    return "".join(lines)


def _fetch_directory() -> str:
    return f"{k.REMOTE_ROOT}/models/{k.SLUG}"


def _fetch_script(plans: PlanMap, *, mismatch: bool) -> tuple[Rule, ...]:
    """`serve fetch` の台本 (`tests/unit/test_weights_fetch.py` の `FetchScript` を見本にした)。"""
    all_paths = tuple(entry.path for entry in k.MANIFEST.files)
    wrong = {all_paths[0]: "f" * 64} if mismatch else {}
    directory = _fetch_directory()
    rules: list[Rule] = []
    for role in k.ROLES:
        plan = plans[role]
        listing = (
            Reply(stdout=""),
            Reply(stdout=k.ps_row(plan, container_id=k.CONTAINER_IDS[role], state="running")),
            Reply(stdout=k.ps_row(plan, container_id=k.CONTAINER_IDS[role], state="exited")),
        )
        rules.append(Rule(prefix=k.OWN_CONTAINERS_ARGV, node=role, replies=listing))
        rules.append(
            Rule(
                prefix=("docker", "container", "inspect"),
                node=role,
                replies=(Reply(stdout="exited 0\n"),),
            )
        )
        sha = _sha_output(all_paths, directory, wrong=wrong if role == "head" else {})
        rules.append(Rule(prefix=("sha256sum",), node=role, replies=(Reply(stdout=sha),)))
    rules.extend(
        (
            Rule(prefix=("uname",), replies=(Reply(stdout="e2e-host\n"),)),
            Rule(prefix=("nvidia-smi",), replies=(Reply(stdout=""),)),
            Rule(prefix=("test", "-e"), replies=(Reply(),)),
            Rule(
                prefix=("docker", "image", "inspect"),
                replies=(Reply(stdout=json.dumps([k.IMAGE_REF]) + "\n"),),
            ),
            Rule(prefix=("df",), replies=(Reply(stdout=k.DF_AVAIL),)),
            Rule(prefix=("docker", "run"), replies=(Reply(stdout="0123456789ab\n"),)),
            Rule(prefix=("docker", "logs"), replies=(Reply(stdout="Fetching: 100%\n"),)),
            Rule(prefix=("docker", "stop"), replies=(Reply(),)),
            Rule(prefix=("docker", "rm"), replies=(Reply(),)),
            Rule(kind="push", replies=(Reply(),)),
        )
    )
    return tuple(rules)


def _verify_script(plans: PlanMap) -> tuple[Rule, ...]:
    all_paths = tuple(entry.path for entry in k.MANIFEST.files)
    directory = f"{k.REMOTE_ROOT}/models/{k.SLUG}"
    rules: list[Rule] = []
    for role in k.ROLES:
        rules.append(
            Rule(
                prefix=("sha256sum",),
                node=role,
                replies=(Reply(stdout=_sha_output(all_paths, directory)),),
            )
        )
        rules.append(Rule(prefix=k.OWN_CONTAINERS_ARGV, node=role, replies=(Reply(stdout=""),)))
    rules.extend(
        (
            Rule(prefix=("uname",), replies=(Reply(stdout="e2e-host\n"),)),
            Rule(prefix=("test", "-e"), replies=(Reply(),)),
            Rule(kind="push", replies=(Reply(),)),
        )
    )
    return tuple(rules)


def _start_ready_script(plans: PlanMap) -> tuple[Rule, ...]:
    rules: list[Rule] = []
    for role in k.ROLES:
        plan = plans[role]
        rules.append(
            Rule(
                prefix=k.OWN_CONTAINERS_ARGV,
                node=role,
                replies=(
                    Reply(stdout=""),
                    Reply(stdout=k.ps_row(plan, container_id=k.CONTAINER_IDS[role])),
                ),
            )
        )
        rules.append(Rule(prefix=("docker", "run"), node=role, replies=(Reply(),)))
    rules.extend((*k.gate_rules(k.ROLES), Rule(kind="push", replies=(Reply(),))))
    rules.append(
        Rule(prefix=("docker", "logs"), node="head", replies=(Reply(stdout="INFO ready\n"),))
    )
    return tuple(rules)


def _start_worker_exit_script(plans: PlanMap) -> tuple[Rule, ...]:
    """worker が応答の確認の前に終了する台本 (`test_start_stop.py` の見本と同じ考え方)。"""
    rules: list[Rule] = []
    for role in k.ROLES:
        plan = plans[role]
        rules.append(
            Rule(
                prefix=k.OWN_CONTAINERS_ARGV,
                node=role,
                replies=(
                    Reply(stdout=""),
                    Reply(stdout=k.ps_row(plan, container_id=k.CONTAINER_IDS[role])),
                ),
            )
        )
        rules.append(Rule(prefix=("docker", "run"), node=role, replies=(Reply(),)))
        rules.append(Rule(kind="push", node=role, replies=(Reply(),)))
        # 片付け (`lifecycle.wrap_up` → `logs.collect_logs`) が、通信の記録と起動の記録を
        # `pull` で回収する。この規則がないと、その回収の呼び出しが「台本にない呼び出し」の
        # `AssertionError` になり、それが `cli.main` に「予期しない失敗」として拾われて
        # `EXIT_FAILED` (2) になる。`start_worker_exit` の期待値もたまたま 2 なので、
        # `test_every_scenario_reached_the_expected_conclusion` の終了コードの比較だけでは
        # この食い違いを見つけられない (レビューでの指摘)
        rules.append(Rule(kind="pull", node=role, replies=(Reply(),)))
        rules.append(Rule(prefix=("docker", "stop"), node=role, replies=(Reply(),)))
        rules.append(Rule(prefix=("docker", "rm"), node=role, replies=(Reply(),)))
    rules.extend(k.gate_rules(k.ROLES))
    rules.append(
        Rule(
            prefix=("docker", "container", "inspect"),
            node="head",
            replies=(Reply(stdout="running 0\n"),),
        )
    )
    rules.append(
        Rule(
            prefix=("docker", "container", "inspect"),
            node="worker",
            replies=(Reply(stdout="exited 1\n"),),
        )
    )
    rules.append(
        Rule(prefix=("docker", "logs"), node="head", replies=(Reply(stdout="head tail\n"),))
    )
    rules.append(
        Rule(prefix=("docker", "logs"), node="worker", replies=(Reply(stdout="worker tail\n"),))
    )
    return tuple(rules)


def _stop_script(plans: PlanMap) -> tuple[Rule, ...]:
    rules: list[Rule] = []
    for role in k.ROLES:
        plan = plans[role]
        rules.append(
            Rule(
                prefix=k.OWN_CONTAINERS_ARGV,
                node=role,
                replies=(Reply(stdout=k.ps_row(plan, container_id=k.CONTAINER_IDS[role])),),
            )
        )
        rules.append(Rule(prefix=("docker", "stop"), node=role, replies=(Reply(),)))
        rules.append(Rule(prefix=("docker", "rm"), node=role, replies=(Reply(),)))
        rules.append(Rule(prefix=("nvidia-smi",), node=role, replies=(Reply(stdout=""),)))
    return tuple(rules)


def _status_script(plans: PlanMap) -> tuple[Rule, ...]:
    rules: list[Rule] = []
    for role in k.ROLES:
        plan = plans[role]
        rules.append(
            Rule(
                prefix=k.OWN_CONTAINERS_ARGV,
                node=role,
                replies=(Reply(stdout=k.ps_row(plan, container_id=k.CONTAINER_IDS[role])),),
            )
        )
        rules.append(Rule(prefix=("nvidia-smi",), node=role, replies=(Reply(stdout=""),)))
        rules.append(
            Rule(
                prefix=("ip", "-br", "link"),
                node=role,
                replies=(Reply(stdout=f"{k.FABRIC_IFNAME}          UP             <UP>\n"),),
            )
        )
    return tuple(rules)


def _image_licenses_script(plans: PlanMap) -> tuple[Rule, ...]:
    plan = plans["head"]
    listing = Rule(
        prefix=k.OWN_CONTAINERS_ARGV,
        node="head",
        replies=(
            Reply(stdout=""),
            Reply(stdout=k.ps_row(plan, container_id=k.CONTAINER_IDS["head"])),
        ),
    )
    return (
        listing,
        *k.gate_rules(("head",)),
        Rule(prefix=("docker", "run"), node="head", replies=(Reply(),)),
        Rule(
            prefix=("docker", "container", "inspect"),
            node="head",
            replies=(Reply(stdout="exited 0\n"),),
        ),
        Rule(
            prefix=("docker", "logs"), node="head", replies=(Reply(stdout="Apache License 2.0\n"),)
        ),
        Rule(prefix=("docker", "stop"), node="head", replies=(Reply(),)),
        Rule(prefix=("docker", "rm"), node="head", replies=(Reply(),)),
    )


def _pull_image_script(*, digests: Sequence[str] = (k.IMAGE_REF,)) -> tuple[Rule, ...]:
    return (
        Rule(prefix=("uname",), replies=(Reply(stdout="e2e-host\n"),)),
        Rule(prefix=("df",), replies=(Reply(stdout=k.DF_AVAIL),)),
        Rule(prefix=("docker", "pull"), replies=(Reply(),)),
        Rule(
            prefix=("docker", "image", "inspect"),
            replies=(Reply(stdout=json.dumps(list(digests)) + "\n"),),
        ),
    )


@dataclass(frozen=True)
class JobRound:
    """通信の確認のジョブの、1 回ぶんの台本 (`tests/unit/test_netcheck_jobs.py` の `Round` を
    見本にした。ジョブは、帯域と事前の確認は 1 回、A/B は 6 回、このまとまりを流す)。
    """

    plans: PlanMap
    tag: str
    nccl: Mapping[str, str]
    stdout: Mapping[str, str] = field(default_factory=dict)


def _job_rounds_script(rounds: Sequence[JobRound]) -> tuple[Rule, ...]:
    """通信の確認のジョブの台本 (`JobScript.rules()` と同じ考え方: 一覧は、最初の空に、
    回ごとの running → exited → exited を連ねた、1 つながりの列にする)。
    """
    rules: list[Rule] = []
    for role in k.ROLES:
        listings: list[str] = [""]
        states: list[Reply] = []
        outputs: list[Reply] = []
        pulls: list[Reply] = []
        for round_ in rounds:
            plan = round_.plans[role]
            listings.append(k.ps_row(plan, container_id=k.CONTAINER_IDS[role], state="running"))
            listings.extend(
                [k.ps_row(plan, container_id=k.CONTAINER_IDS[role], state="exited")] * 2
            )
            states.append(Reply(stdout="exited 0\n"))
            outputs.append(Reply(stdout=round_.stdout.get(role, "")))
            nccl_text = round_.nccl.get(role)
            writes = (
                ((f"logs/nccl-{round_.tag}.{role}.7.log", nccl_text),)
                if nccl_text is not None
                else ()
            )
            pulls.append(Reply(writes=writes) if writes else Reply(exit_code=23))
        rules.append(
            Rule(
                prefix=k.OWN_CONTAINERS_ARGV,
                node=role,
                replies=tuple(Reply(stdout=text) for text in listings),
            )
        )
        rules.append(
            Rule(prefix=("docker", "container", "inspect"), node=role, replies=tuple(states))
        )
        rules.append(
            Rule(
                prefix=("docker", "logs", "--timestamps"),
                node=role,
                replies=(Reply(stdout="全量の記録\n"),),
            )
        )
        rules.append(Rule(prefix=("docker", "logs"), node=role, replies=tuple(outputs)))
        rules.append(Rule(kind="pull", node=role, remote_prefix="logs", replies=tuple(pulls)))
    rules.extend(
        (
            Rule(prefix=("uname",), replies=(Reply(stdout="e2e-host\n"),)),
            Rule(prefix=("nvidia-smi",), replies=(Reply(stdout=""),)),
            Rule(prefix=("test", "-e"), replies=(Reply(),)),
            Rule(
                prefix=("docker", "image", "inspect"),
                replies=(Reply(stdout=json.dumps([k.IMAGE_REF]) + "\n"),),
            ),
            Rule(prefix=("df",), replies=(Reply(stdout=k.DF_AVAIL),)),
            Rule(prefix=("ss",), replies=(Reply(stdout=""),)),
            Rule(prefix=("docker", "run"), replies=(Reply(stdout="aaaa1111\n"),)),
            Rule(prefix=("docker", "stop"), replies=(Reply(),)),
            Rule(prefix=("docker", "rm"), replies=(Reply(),)),
            Rule(kind="pull", remote_prefix="state", replies=(Reply(exit_code=23),)),
        )
    )
    return tuple(rules)


def _job_bandwidth_script(
    plans: PlanMap, *, tag: str, nccl_head: str, nccl_worker: str, stdout_head: str
) -> tuple[Rule, ...]:
    return _job_rounds_script(
        [
            JobRound(
                plans=plans,
                tag=tag,
                nccl={"head": nccl_head, "worker": nccl_worker},
                stdout={"head": stdout_head, "worker": ""},
            )
        ]
    )


NCCL_IB_LOG = (
    "e2e-head:7:7 [0] NCCL INFO NCCL version 2.30.7+cuda13.0\n"
    "e2e-head:7:19 [0] NCCL INFO NET/IB : Using [0]rocep1s0f0:1/RoCE\n"
    "e2e-head:7:19 [0] NCCL INFO TOPO/NET : Made vNic 0\n"
    "e2e-head:7:19 [0] NCCL INFO Using network IB\n"
    "e2e-head:7:19 [0] NCCL INFO 8 coll channels, 0 collnet channels, 8 p2p channels\n"
    "e2e-head:7:19 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0\n"
)
NCCL_SOCKET_LOG = (
    "e2e-head:7:7 [0] NCCL INFO NCCL version 2.30.7+cuda13.0\n"
    "e2e-head:7:19 [0] NCCL INFO NET/IB : No device found.\n"
    "e2e-head:7:19 [0] NCCL INFO Using network Socket\n"
    "e2e-head:7:19 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Socket/0\n"
)


def _bench_json(size: int, busbw: float, *, world_size: int = 2) -> str:
    report = {
        "world_size": world_size,
        "dtype": "float32",
        "warmup_iters": 5,
        "measure_iters": 20,
        "torch_version": "2.9.0+cu130",
        "nccl_version": "2.30.7",
        "samples": [
            {
                "size_bytes": size,
                "time_s": {"n": 20, "mean": 0.001, "min": 0.0009, "max": 0.0012, "stdev": 0.0001},
                "algbw_gbps": busbw,
                "busbw_gbps": busbw,
            }
        ],
    }
    return json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _sanity_script(plans: PlanMap, *, stages: int, tag: str) -> tuple[Rule, ...]:
    markers = "".join(f"{stage.marker}\n" for stage in netcheck_mod.SANITY_STAGES[:stages])
    return _job_rounds_script(
        [
            JobRound(
                plans=plans,
                tag=tag,
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
                stdout={"head": markers, "worker": markers},
            )
        ]
    )


def _ab_script(
    plan_rounds: Sequence[PlanMap], *, tags: Sequence[str], busbw: Sequence[float]
) -> tuple[Rule, ...]:
    """A/B の 6 回ぶんの台本 (`--repeat 3` の既定。すべて高速の直結の経路)。"""
    return _job_rounds_script(
        [
            JobRound(
                plans=plans,
                tag=tag,
                nccl={"head": NCCL_IB_LOG, "worker": NCCL_IB_LOG},
                stdout={"head": _bench_json(1 << 30, value), "worker": ""},
            )
            for plans, tag, value in zip(plan_rounds, tags, busbw, strict=True)
        ]
    )


PROBE_READY_LOG = (
    "INFO Using FLASH_ATTN_MLA attention backend out of potential backends: ['FLASH_ATTN_MLA']\n"
    "INFO GPU KV cache size: 1,234,567 tokens,"
    " Maximum concurrency for 163,840 tokens per request: 7.54x\n"
    "INFO init engine (profile, create kv cache, warmup model) took 42.00 seconds\n"
)
PROBE_FAILURE_LOG = "".join(f"INFO booting up {i}\n" for i in range(20)) + (
    "ERROR AssertionError: pe_dim must be 64\n"
)


def _probe_script(plan: Any, *, states: tuple[Reply, ...], tails: str) -> tuple[Rule, ...]:
    role = plan.node
    return (
        Rule(
            prefix=k.OWN_CONTAINERS_ARGV,
            node=role,
            replies=(Reply(stdout=""), Reply(stdout=k.ps_row(plan, container_id="probe0001"))),
        ),
        Rule(prefix=("docker", "container", "inspect"), node=role, replies=states),
        Rule(prefix=("docker", "logs", "--timestamps"), node=role, replies=(Reply(stdout=tails),)),
        Rule(
            prefix=("cat",),
            node=role,
            replies=(Reply(stdout=k.record_json(role, "probe_files")),),
        ),
        Rule(prefix=("uname",), replies=(Reply(stdout="e2e-host\n"),)),
        Rule(prefix=("nvidia-smi",), replies=(Reply(stdout=""),)),
        Rule(prefix=("test", "-e"), replies=(Reply(),)),
        Rule(
            prefix=("docker", "image", "inspect"),
            replies=(Reply(stdout=json.dumps([k.IMAGE_REF]) + "\n"),),
        ),
        Rule(prefix=("df",), replies=(Reply(stdout=k.DF_AVAIL),)),
        Rule(prefix=("ss",), replies=(Reply(stdout=""),)),
        Rule(prefix=("docker", "run"), replies=(Reply(stdout="probe0001\n"),)),
        Rule(prefix=("docker", "stop"), replies=(Reply(),)),
        Rule(prefix=("docker", "rm"), replies=(Reply(),)),
        Rule(kind="push", replies=(Reply(),)),
        Rule(kind="pull", replies=(Reply(exit_code=23),)),
    )


def _watch_quiet_script() -> tuple[Rule, ...]:
    from serving_kit.watch import GPU_UTIL_ARGV

    return (Rule(prefix=GPU_UTIL_ARGV, replies=(Reply(stdout="10 %\n"),)),)


def _watch_metrics_sample() -> Any:
    from fake_vllm import MetricsSample

    return MetricsSample(running_requests=0, waiting_requests=0, generation_tokens_total=0)


BODY_MARKER = "ZZ-SAFETY-BODY-MARKER-ZZ"
"""偽の推論サーバーの応答にだけ現れる目印 (rule 9: 結果のファイルに出ないことを見る)。"""


def _thinking_reply_factory() -> Callable[[dict[str, Any] | None], MessagesReply]:
    """thinking の 5 通りの要求に、応答の本文に目印を持つ返事を返す (rule 9 の確認のため)。"""

    def factory(body: dict[str, Any] | None) -> MessagesReply:
        return MessagesReply(
            text=BODY_MARKER,
            thinking=BODY_MARKER,
            input_tokens=42,
            output_tokens=7,
            stop_reason="end_turn",
        )

    return factory


# --- Hub の偽物 (manifest が、モデルカードを取りに行かないことの確認) --------------------


class _HubTransport:
    """`serve manifest` が使う Hub の偽物 (README.md を含む一覧を返す)。"""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def _dispatch(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        tree_url = f"{weights_mod.HUB_BASE_URL}/api/models/{k.REPO}/tree/{k.REVISION}?recursive=1"
        if url == tree_url:
            entries = [
                {"type": "file", "path": "README.md", "size": 10, "oid": "0" * 40},
                {"type": "file", "path": ".gitattributes", "size": 5, "oid": "0" * 40},
                {"type": "file", "path": "config.json", "size": 2, "oid": "0" * 40},
            ]
            return httpx.Response(200, json=entries)
        if url == f"{weights_mod.HUB_BASE_URL}/{k.REPO}/resolve/{k.REVISION}/config.json":
            return httpx.Response(200, content=b"{}")
        raise AssertionError(f"想定していない Hub への要求: {request.method} {url}")

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._dispatch), trust_env=False)


def run_manifest(
    repo: k.Repo, monkeypatch: pytest.MonkeyPatch
) -> tuple[k.Invocation, tuple[httpx.Request, ...]]:
    """`serve manifest` は Spark に触らず、Hub にモデルカードを取りに行かない。

    `client_factory` は `manifest` に配線されていない (cli.py の `_cmd_manifest` が
    `weights_mod.build_manifest` を `client=` なしで呼ぶ) ので、`weights_mod.new_client` を
    差し替えて、実物の Hugging Face Hub の代わりに `httpx.MockTransport` を使わせる
    (`tests/unit/test_weights.py` と同じ手法)。`runner_factory` は、呼ばれたら落ちるものを
    渡し、Spark への呼び出しが 0 件であることを構造で確かめる。終了コードそのものの確認は、
    ほかのシナリオと同じく `test_every_scenario_reached_the_expected_conclusion` に任せる。
    """
    import io

    transport = _HubTransport()
    monkeypatch.setattr(weights_mod, "new_client", lambda timeout_s=None: transport.client())

    def exploding_factory(var_root: Path) -> FakeRunner:
        raise AssertionError("serve manifest は、Spark に触ってはいけない")

    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(
        [
            "manifest",
            k.REPO,
            k.REVISION,
            "--configs",
            str(repo.configs),
            "--nodes",
            str(repo.nodes),
            "--var-root",
            str(repo.var_root),
        ],
        runner_factory=exploding_factory,
        stdin=io.StringIO(""),
        stdout=out,
        stderr=err,
        repo_root=repo.root,
        repo_facts=(k.REPO_COMMIT, False),
        now=lambda: datetime.now(UTC),
        sleep=k.never_sleep,
    )
    invocation = k.Invocation(
        code=code,
        out=out.getvalue(),
        err=err.getvalue(),
        runner=FakeRunner(var_root=repo.var_root),
    )
    return invocation, tuple(transport.requests)


def _tool_manifest_payload() -> dict[str, Any]:
    """変換の道具 (`experiments/k2-quant`) が書く `manifest.json` の形 (小さな見本)。"""
    files = [
        {"path": "config.json", "size": 2, "sha256": "1" * 64},
        {"path": "model-00001-of-00001.safetensors", "size": 8, "sha256": "2" * 64},
    ]
    pattern = r"^lm_head$"
    return {
        "conversion": {
            "tool": "k2-quant",
            "tool_version": "0.2.0",
            "source": {"repo": k.REPO, "revision": k.REVISION},
            "pattern": pattern,
            "args": [
                "--source-repo",
                k.REPO,
                "--source-revision",
                k.REVISION,
                "--pattern",
                pattern,
            ],
            "modules": ["lm_head"],
            "weight_dtype": "F8_E4M3",
            "scale_dtype": "F32",
            "strategy": "channel",
        },
        "files": files,
        "total_bytes": sum(entry["size"] for entry in files),
    }


def run_derived_import(repo: k.Repo, inputs_dir: Path) -> tuple[k.Invocation, tuple[Path, ...]]:
    """`serve derived-import` は、Spark にも Hub にも触らず、`serving/weights/` に 1 つだけ書く。

    2 台ぶんの manifest (中身は同じ) を、リポジトリの外の `inputs_dir` に置いて渡す。
    `runner_factory` は、呼ばれたら落ちるものを渡す。Hub への口 (`weights_mod.new_client`) も、
    この呼び出しの間だけ、呼ばれたら落ちるものに差し替える。返すのは、呼び出しの前後で
    リポジトリの下に新しくできたファイル。
    """
    import io

    inputs_dir.mkdir(parents=True, exist_ok=True)
    inputs = []
    for name in ("spark-153d.manifest.json", "spark-5083.manifest.json"):
        path = inputs_dir / name
        path.write_text(json.dumps(_tool_manifest_payload()) + "\n", encoding="utf-8")
        inputs.append(str(path))

    def exploding_factory(var_root: Path) -> FakeRunner:
        raise AssertionError("serve derived-import は、Spark に触ってはいけない")

    def exploding_client(timeout_s: float | None = None) -> httpx.Client:
        raise AssertionError("serve derived-import は、Hub に触ってはいけない")

    before = set(k.files_under(repo.root))
    out = io.StringIO()
    err = io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(weights_mod, "new_client", exploding_client)
        code = cli.main(
            [
                "derived-import",
                "--name",
                "k2s1",
                "--commit",
                "a" * 40,
                *inputs,
                "--configs",
                str(repo.configs),
                "--nodes",
                str(repo.nodes),
                "--var-root",
                str(repo.var_root),
            ],
            runner_factory=exploding_factory,
            stdin=io.StringIO(""),
            stdout=out,
            stderr=err,
            repo_root=repo.root,
            repo_facts=(k.REPO_COMMIT, False),
            now=lambda: datetime.now(UTC),
            sleep=k.never_sleep,
        )
    written = tuple(sorted(set(k.files_under(repo.root)) - before))
    invocation = k.Invocation(
        code=code,
        out=out.getvalue(),
        err=err.getvalue(),
        runner=FakeRunner(var_root=repo.var_root),
    )
    return invocation, written


def _known_container_names(base: Path) -> set[str]:
    """このファイルの台本が「自分のコンテナ」として使う、すべての名前を組み立てる。

    5 つの構成 (`kind` ごと) の名前に加え、通信の確認の A/B が使う `-r1`〜`-r3` の付いた
    名前も要る (`plan._container_name` は `repeat_index` だけを名前に入れ、`arm` は入れない
    ので、baseline と candidate は同じ名前を使う)。
    """
    repo = k.make_repo(base / "known-names", port=k.UNUSED_PORT)
    names: set[str] = set()
    for name in (k.SERVE_CONFIG, k.PROBE_CONFIG, k.FETCH_CONFIG, k.INSPECT_CONFIG, k.JOB_CONFIG):
        config, nodes = k.load_config_and_nodes(repo, name)
        names.update(plan.container_name for plan in k.container_plans(config, nodes).values())
    job_config, job_nodes = k.load_config_and_nodes(repo, k.JOB_CONFIG)
    for repeat in (1, 2, 3):
        plans = plan_mod.build_plans(
            job_config,
            job_nodes,
            k.PLAN_BUILD_TIME,
            arm=netcheck_mod.BASELINE_ARM,
            repeat_index=repeat,
        )
        names.update(plan.container_name for plan in plans)
    return names


_KNOWN_CONTAINER_IDS: frozenset[str] = frozenset(
    {*k.CONTAINER_IDS.values(), "probe0001", "aaaa1111"}
)
"""このファイルの台本が「自分のコンテナ」の識別子として使う、すべての値 (決まり 1)。"""


def _build_world(
    base: Path, start_vllm: Callable[..., FakeVllm], monkeypatch: pytest.MonkeyPatch
) -> World:
    scenarios: dict[str, k.Invocation] = {}
    vllms: dict[str, FakeVllm] = {}
    var_roots: dict[str, Path] = {}

    def run(name: str, argv: Sequence[str], repo: k.Repo, **kwargs: Any) -> None:
        var_roots[name] = repo.var_root
        scenarios[name] = k.invoke(list(argv), repo, **kwargs)

    # --- manifest (Spark に触らない) ------------------------------------------
    repo_manifest = k.make_repo(base / "manifest", port=k.UNUSED_PORT)
    var_roots["manifest"] = repo_manifest.var_root
    manifest_invocation, manifest_requests = run_manifest(repo_manifest, monkeypatch)
    scenarios["manifest"] = manifest_invocation

    # --- push / check / pull-image / image-licenses -------------------------
    repo_push = k.make_repo(base / "push", port=k.UNUSED_PORT)
    run("push", ["push", "--yes"], repo_push, default=Reply())

    repo_check = k.make_repo(base / "check", port=k.UNUSED_PORT)
    run("check", ["check", k.SERVE_CONFIG], repo_check, script=k.gate_rules())

    repo_pull = k.make_repo(base / "pull-image", port=k.UNUSED_PORT)
    run(
        "pull_image",
        ["pull-image", k.SERVE_CONFIG, "--yes"],
        repo_pull,
        script=_pull_image_script(),
    )

    repo_licenses = k.make_repo(base / "image-licenses", port=k.UNUSED_PORT)
    inspect_config, inspect_nodes = k.load_config_and_nodes(repo_licenses, k.INSPECT_CONFIG)
    inspect_plans = k.container_plans(inspect_config, inspect_nodes)
    run(
        "image_licenses",
        ["image-licenses", k.INSPECT_CONFIG, "--yes"],
        repo_licenses,
        script=_image_licenses_script(inspect_plans),
    )

    # --- fetch / verify -------------------------------------------------------
    repo_fetch = k.make_repo(base / "fetch", port=k.UNUSED_PORT)
    fetch_config, fetch_nodes = k.load_config_and_nodes(repo_fetch, k.FETCH_CONFIG)
    fetch_plans = k.container_plans(fetch_config, fetch_nodes)
    run(
        "fetch_success",
        ["fetch", k.FETCH_CONFIG, "--yes"],
        repo_fetch,
        script=_fetch_script(fetch_plans, mismatch=False),
    )

    repo_fetch_bad = k.make_repo(base / "fetch-mismatch", port=k.UNUSED_PORT)
    run(
        "fetch_mismatch",
        ["fetch", k.FETCH_CONFIG, "--yes"],
        repo_fetch_bad,
        script=_fetch_script(fetch_plans, mismatch=True),
    )

    repo_verify = k.make_repo(base / "verify", port=k.UNUSED_PORT)
    serve_config, serve_nodes = k.load_config_and_nodes(repo_verify, k.SERVE_CONFIG)
    serve_plans = k.container_plans(serve_config, serve_nodes)
    run(
        "verify_success",
        ["verify", k.SERVE_CONFIG, "--yes"],
        repo_verify,
        script=_verify_script(serve_plans),
    )

    # --- start / stop / status / smoke / logs ---------------------------------
    serve_vllm = start_vllm()
    repo_start = k.make_repo(base / "start", port=k.port_of(serve_vllm))
    start_config, start_nodes = k.load_config_and_nodes(repo_start, k.SERVE_CONFIG)
    start_plans = k.container_plans(start_config, start_nodes)
    run(
        "start_success",
        ["start", k.SERVE_CONFIG, "--yes"],
        repo_start,
        script=_start_ready_script(start_plans),
    )

    repo_start_fail = k.make_repo(base / "start-fail", port=k.port_of(serve_vllm))
    fail_vllm = start_vllm()
    repo_start_fail = k.make_repo(base / "start-fail", port=k.port_of(fail_vllm))
    fail_config, fail_nodes = k.load_config_and_nodes(repo_start_fail, k.SERVE_CONFIG)
    fail_vllm.set_health_fault(Fault(status=None))
    fail_plans = k.container_plans(fail_config, fail_nodes)
    run(
        "start_worker_exit",
        ["start", k.SERVE_CONFIG, "--timeout", "1h", "--yes"],
        repo_start_fail,
        script=_start_worker_exit_script(fail_plans),
    )

    repo_stop = k.make_repo(base / "stop", port=k.UNUSED_PORT)
    stop_config, stop_nodes = k.load_config_and_nodes(repo_stop, k.SERVE_CONFIG)
    stop_plans = k.container_plans(stop_config, stop_nodes)
    run("stop_success", ["stop", "--yes"], repo_stop, script=_stop_script(stop_plans))

    repo_status = k.make_repo(base / "status", port=k.UNUSED_PORT)
    status_config, status_nodes = k.load_config_and_nodes(repo_status, k.SERVE_CONFIG)
    status_plans = k.container_plans(status_config, status_nodes)
    run("status_success", ["status"], repo_status, script=_status_script(status_plans))

    smoke_vllm = start_vllm()
    vllms["smoke"] = smoke_vllm
    smoke_vllm.set_messages_reply(MessagesReply(text=BODY_MARKER, stop_reason="end_turn"))
    repo_smoke = k.make_repo(base / "smoke", port=k.port_of(smoke_vllm))
    run(
        "smoke_success",
        ["smoke", k.SERVE_CONFIG, "--yes"],
        repo_smoke,
        script=(),
        expect_runner=False,
    )

    repo_logs = k.make_repo(base / "logs", port=k.UNUSED_PORT)
    logs_script = tuple(
        Rule(
            prefix=("docker", "logs", "--timestamps"),
            node=role,
            replies=(Reply(stdout="全量\n"),),
        )
        for role in k.ROLES
    ) + tuple(Rule(prefix=("cat",), node=role, replies=(Reply(exit_code=1),)) for role in k.ROLES)
    run("logs_success", ["logs", k.SERVE_CONFIG], repo_logs, script=logs_script, default=Reply())

    # --- probe (5.4 の完了の状態: 成功、失敗の両方) -----------------------------
    probe_vllm = start_vllm()
    vllms["probe"] = probe_vllm
    # `probe` の短い要求 (`lifecycle.send_smoke`) の応答にも目印を仕込む。決まり 9 は、
    # 「応答の本文が、回収した記録に出ない」ことを見るので、smoke/thinking だけでなく probe の
    # 応答にも目印がないと、probe の経路をこの決まりが確かめていないことになる (レビューでの指摘)
    probe_vllm.set_messages_reply(
        MessagesReply(text=BODY_MARKER, thinking=BODY_MARKER, stop_reason="end_turn")
    )
    repo_probe = k.make_repo(base / "probe-ready", port=k.port_of(probe_vllm))
    probe_config, probe_nodes = k.load_config_and_nodes(repo_probe, k.PROBE_CONFIG)
    probe_plans = k.container_plans(probe_config, probe_nodes)
    run(
        "probe_ready",
        ["probe", k.PROBE_CONFIG, "--yes"],
        repo_probe,
        script=_probe_script(
            probe_plans["head"], states=(Reply(stdout="running 0\n"),), tails=PROBE_READY_LOG
        ),
    )

    repo_probe_fail = k.make_repo(base / "probe-failed", port=k.UNUSED_PORT)
    probe_fail_config, probe_fail_nodes = k.load_config_and_nodes(repo_probe_fail, k.PROBE_CONFIG)
    probe_fail_plans = k.container_plans(probe_fail_config, probe_fail_nodes)
    run(
        "probe_failed",
        ["probe", k.PROBE_CONFIG, "--yes"],
        repo_probe_fail,
        script=_probe_script(
            probe_fail_plans["head"],
            states=(Reply(stdout="exited 1\n"),),
            tails=PROBE_FAILURE_LOG,
        ),
    )

    # --- netcheck ---------------------------------------------------------------
    repo_links = k.make_repo(base / "netcheck-links", port=k.UNUSED_PORT)
    links_script = tuple(
        rule
        for role in k.ROLES
        for rule in (Rule(prefix=("ip", "-br", "link"), node=role, replies=(Reply(stdout=""),)),)
    )
    run("netcheck_links", ["netcheck", "links"], repo_links, script=links_script, default=Reply())

    bandwidth_tag = _round_tag(NETCHECK_NOW, netcheck_mod.BASELINE_ARM, 1)
    sanity_tag = _round_tag(NETCHECK_NOW, "sanity", 1)

    repo_bw = k.make_repo(base / "netcheck-bandwidth", port=k.UNUSED_PORT)
    bw_config, bw_nodes = k.load_config_and_nodes(repo_bw, k.JOB_CONFIG)
    bw_plans = k.container_plans(bw_config, bw_nodes)
    run(
        "netcheck_bandwidth_success",
        ["netcheck", "bandwidth", k.JOB_CONFIG, "--yes"],
        repo_bw,
        script=_job_bandwidth_script(
            bw_plans,
            tag=bandwidth_tag,
            nccl_head=NCCL_IB_LOG,
            nccl_worker=NCCL_IB_LOG,
            stdout_head=_bench_json(1 << 20, 20.0),
        ),
        now=_fixed_now(),
    )

    repo_bw_fail = k.make_repo(base / "netcheck-bandwidth-fail", port=k.UNUSED_PORT)
    bw_fail_config, bw_fail_nodes = k.load_config_and_nodes(repo_bw_fail, k.JOB_CONFIG)
    bw_fail_plans = k.container_plans(bw_fail_config, bw_fail_nodes)
    run(
        "netcheck_bandwidth_socket_fallback",
        ["netcheck", "bandwidth", k.JOB_CONFIG, "--yes"],
        repo_bw_fail,
        script=_job_bandwidth_script(
            bw_fail_plans,
            tag=bandwidth_tag,
            nccl_head=NCCL_SOCKET_LOG,
            nccl_worker=NCCL_SOCKET_LOG,
            stdout_head=_bench_json(1 << 20, 20.0),
        ),
        now=_fixed_now(),
    )

    repo_sanity = k.make_repo(base / "netcheck-sanity", port=k.UNUSED_PORT)
    sanity_config, sanity_nodes = k.load_config_and_nodes(repo_sanity, k.JOB_CONFIG)
    sanity_plans = k.container_plans(sanity_config, sanity_nodes)
    run(
        "netcheck_sanity_success",
        ["netcheck", "sanity", k.JOB_CONFIG, "--yes"],
        repo_sanity,
        script=_sanity_script(sanity_plans, stages=4, tag=sanity_tag),
        now=_fixed_now(),
    )

    repo_ab = k.make_repo(base / "netcheck-ab", port=k.UNUSED_PORT)
    ab_config, ab_nodes = k.load_config_and_nodes(repo_ab, k.JOB_CONFIG)
    ab_order = [
        (arm, repeat)
        for repeat in range(1, 4)
        for arm in (netcheck_mod.BASELINE_ARM, netcheck_mod.CANDIDATE_ARM)
    ]
    ab_plan_rounds = [
        {
            p.node: p
            for p in plan_mod.build_plans(
                ab_config, ab_nodes, NETCHECK_NOW, arm=arm, repeat_index=repeat
            )
        }
        for arm, repeat in ab_order
    ]
    ab_tags = [_round_tag(NETCHECK_NOW, arm, repeat) for arm, repeat in ab_order]
    # A の最大 (152) < B の最小 (179) なので「採用できる」になる (見本は SEPARATED)
    ab_busbw = [150.0, 180.0, 152.0, 181.0, 151.0, 179.0]
    run(
        "netcheck_ab_success",
        [
            "netcheck",
            "ab",
            k.JOB_CONFIG,
            "--yes",
            "--env",
            "NCCL_IB_QPS_PER_CONNECTION=4",
        ],
        repo_ab,
        script=_ab_script(ab_plan_rounds, tags=ab_tags, busbw=ab_busbw),
        now=_fixed_now(),
    )

    # --- watch (読み取りだけ) -----------------------------------------------------
    watch_vllm = start_vllm()
    vllms["watch"] = watch_vllm
    watch_vllm.set_metrics(_watch_metrics_sample())
    repo_watch = k.make_repo(base / "watch", port=k.port_of(watch_vllm))
    run(
        "watch_quiet",
        [
            "watch",
            k.SERVE_CONFIG,
            "--duration",
            "1",
            "--interval",
            "1",
        ],
        repo_watch,
        script=_watch_quiet_script(),
        default=Reply(stdout=""),
        sleep=lambda seconds: None,
    )

    # --- thinking (Spark に触らない) -----------------------------------------------
    thinking_vllm = start_vllm()
    vllms["thinking"] = thinking_vllm
    thinking_vllm.set_messages_factory(_thinking_reply_factory())
    repo_thinking = k.make_repo(base / "thinking", port=k.port_of(thinking_vllm))
    run(
        "thinking_success",
        ["thinking", k.SERVE_CONFIG, "--trials", "1"],
        repo_thinking,
        expect_runner=False,
    )

    return World(
        scenarios=scenarios,
        all_calls=tuple(call for inv in scenarios.values() for call in inv.runner.calls),
        base=base,
        fetch_config=fetch_config,
        fetch_plans=fetch_plans,
        vllms=vllms,
        manifest_requests=manifest_requests,
        known_names=frozenset(_known_container_names(base)),
        known_ids=_KNOWN_CONTAINER_IDS,
        var_roots=var_roots,
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[World]:
    servers: list[FakeVllm] = []

    def start_vllm(**kwargs: Any) -> FakeVllm:
        server = FakeVllm(**kwargs)
        server.start()
        servers.append(server)
        return server

    base = tmp_path_factory.mktemp("safety")
    # `monkeypatch` の組み込みフィクスチャは関数の scope なので、module の scope のこの
    # フィクスチャでは使えない。同じ後始末 (undo) を、ここで手で行う
    mp = pytest.MonkeyPatch()
    try:
        yield _build_world(base, start_vllm, mp)
    finally:
        mp.undo()
        for server in servers:
            server.stop()


# --- 決まりごとの試験 ----------------------------------------------------------


def test_every_scenario_reached_the_expected_conclusion(world: World) -> None:
    """土台の確認: どのシナリオも、期待した終了コードで終わっている (安全の試験の前提)。"""
    expected = {
        "manifest": cli.EXIT_OK,
        "push": cli.EXIT_OK,
        "check": cli.EXIT_OK,
        "pull_image": cli.EXIT_OK,
        "image_licenses": cli.EXIT_OK,
        "fetch_success": cli.EXIT_OK,
        "fetch_mismatch": cli.EXIT_FAILED,
        "verify_success": cli.EXIT_OK,
        "start_success": cli.EXIT_OK,
        "start_worker_exit": cli.EXIT_FAILED,
        "stop_success": cli.EXIT_OK,
        "status_success": cli.EXIT_OK,
        "smoke_success": cli.EXIT_OK,
        "logs_success": cli.EXIT_OK,
        "probe_ready": cli.EXIT_OK,
        "probe_failed": cli.EXIT_FAILED,
        "netcheck_links": cli.EXIT_OK,
        "netcheck_bandwidth_success": cli.EXIT_OK,
        "netcheck_bandwidth_socket_fallback": cli.EXIT_FAILED,
        "netcheck_sanity_success": cli.EXIT_OK,
        "netcheck_ab_success": cli.EXIT_OK,
        "watch_quiet": cli.EXIT_OK,
        "thinking_success": cli.EXIT_OK,
    }
    problems = [
        f"{name}: {world.scenarios[name].code} (期待は {code})"
        f" detail={world.scenarios[name].err[-400:]!r}"
        for name, code in expected.items()
        if world.scenarios[name].code != code
    ]
    assert not problems, "\n".join(problems)

    # 終了コードの一致だけでは、台本の不足 (`AssertionError`) や `remote.CallGuard` の拒否
    # (`RuntimeError`) が、たまたま期待どおりの終了コードに写って隠れることがある
    # (レビューでの指摘: `start_worker_exit` の期待値が `EXIT_FAILED` (2) で、「台本にない
    # 呼び出し」の予期しない失敗もちょうど 2 になるため、見分けがつかなかった)。`cli.main` が
    # 予期しない例外を包んだときにだけ書く文言 (`_fail` の `unexpected=True` の枝) が、
    # どのシナリオの stderr にもないことを、ここで固定する
    unexpected = [
        name for name, invocation in world.scenarios.items() if "予期しない失敗" in invocation.err
    ]
    assert not unexpected, (
        "台本の不足や CallGuard の拒否が、予期しない失敗として隠れているシナリオがある: "
        f"{[(name, world.scenarios[name].err[-400:]) for name in unexpected]}"
    )


# --- 決まり 1: コンテナを対象にする操作の的 -----------------------------------

_CONTAINER_TARGETING_SUBCOMMANDS = frozenset({"stop", "rm", "logs"})
_VALUE_TAKING_FLAGS = frozenset({"-t", "--time", "-s", "--signal", "--format", "--tail"})


def _container_targets(call: RecordedCall) -> tuple[str, ...]:
    """`docker stop` / `rm` / `logs` / `container inspect` の対象を、位置で読む。

    `guards._targets_of` と同じ考え方 (値を取るフラグの次の語だけを飛ばし、`-` で始まらない
    語を対象として扱う)。この試験専用の、読み取り側の実装である (`guards` を import して
    使い回すと、決まりを破る実装の変異が、確かめる側の変異にもなってしまう)。
    """
    argv = call.argv
    if not argv or argv[0] != "docker" or len(argv) < 2:
        return ()
    if argv[1] in _CONTAINER_TARGETING_SUBCOMMANDS:
        rest = argv[2:]
    elif argv[1:3] == ("container", "inspect"):
        rest = argv[3:]
    else:
        return ()
    targets: list[str] = []
    skip = False
    for word in rest:
        if skip:
            skip = False
            continue
        if word in _VALUE_TAKING_FLAGS:
            skip = True
            continue
        if word.startswith("-"):
            continue
        targets.append(word)
    return tuple(targets)


def test_rule1_container_targeting_operations_hit_only_listed_or_planned_names(
    world: World,
) -> None:
    """決まり 1: 対象は、ラベルで絞った一覧の識別子か、了承済みの計画の名前だけ。

    偽の実行役に記録された、すべてのシナリオの `docker stop` / `rm` / `logs` /
    `container inspect` の対象を集め、この試験の台本がラベルつきの一覧に載せた識別子
    (`world.known_ids`)、または、この試験の 5 つの構成 (と A/B の `-r1`〜`-r3`) から
    組み立てた計画の名前 (`world.known_names`) のどちらでもないものが 1 つでもあれば断る。
    """
    targets = [(call, target) for call in world.all_calls for target in _container_targets(call)]
    assert targets, "コンテナを対象にする操作が 1 つも記録されていない (この試験は何も見ていない)"
    problems = [
        (call.node, call.argv, target)
        for call, target in targets
        if target not in world.known_ids and target not in world.known_names
    ]
    assert not problems, f"一覧にも計画にもない対象への操作がある: {problems}"


# --- 決まり 2: 名前とラベルのないコンテナを起こす呼び出しがない -----------------


def _docker_run_calls(world: World) -> tuple[RecordedCall, ...]:
    return tuple(
        call
        for call in world.all_calls
        if call.kind == "run" and call.argv[:2] == ("docker", "run")
    )


def _label_values(argv: tuple[str, ...], key: str) -> tuple[str, ...]:
    prefix = f"{key}="
    values: list[str] = []
    for flag, value in zip(argv, argv[1:], strict=False):
        if flag == "--label" and value.startswith(prefix):
            values.append(value[len(prefix) :])
    return tuple(values)


def test_rule2_every_docker_run_has_a_name_and_the_owner_label(world: World) -> None:
    """決まり 2: 名前とラベルのないコンテナを起こす呼び出しがない。

    すべての `docker run` に `--name` があり、所有のラベル (`vllm-baseline.owner=serving-kit`)
    と、`config-sha256` のラベルがあることを確かめる (design.md 「plan」の共通の 7 つのラベル)。
    """
    runs = _docker_run_calls(world)
    assert runs, "docker run が 1 つも記録されていない (この試験は何も見ていない)"
    owner_label = f"{LABEL_OWNER}={OWNER}"
    problems = []
    for call in runs:
        if "--name" not in call.argv:
            problems.append(("--name がない", call.argv))
            continue
        if owner_label not in call.argv:
            problems.append(("所有のラベルがない", call.argv))
        if not _label_values(call.argv, LABEL_CONFIG_SHA256):
            problems.append(("config-sha256 のラベルがない", call.argv))
    assert not problems, problems


# --- 決まり 3: 前面で動かす、または自動で消す起動がない ------------------------

_FOREGROUND_OR_AUTO_REMOVE_FLAGS = frozenset(
    {"-i", "--interactive", "-t", "--tty", "-a", "--attach", "--rm"}
)


def test_rule3_no_foreground_or_auto_remove_container_starts(world: World) -> None:
    """決まり 3: 前面で動かす、または終了時に自動で消すコンテナの起動が、どの種類にもない。

    すべての `docker run` が `-d` を持ち、前面で動かす指定 (`-i`/`-t`/`-a`/長い形) と
    `--rm` を、どの種類 (serve/probe/job/fetch/inspect) にも持たないことを確かめる。
    """
    runs = _docker_run_calls(world)
    assert runs, "docker run が 1 つも記録されていない"
    problems = []
    for call in runs:
        if "-d" not in call.argv:
            problems.append(("-d がない", call.argv))
        hit = _FOREGROUND_OR_AUTO_REMOVE_FLAGS & set(call.argv)
        if hit:
            problems.append((f"禁じた指定がある {sorted(hit)}", call.argv))
    assert not problems, problems


# --- 決まり 4: 配布の宛先が、決まった部分だけ ----------------------------------


def test_rule4_push_and_mkdir_targets_are_limited_to_the_fixed_places(world: World) -> None:
    """決まり 4: 配布の宛先が、決まった部分 (`payload/`、`state/`) だけ。

    `push` の呼び出しの `remote_subdir` が `payload` か `state` のどちらかで始まり、
    `mkdir` は `remote_root` の 6 つの置き場所だけを作ることを確かめる。
    """
    problems = []
    for call in world.all_calls:
        if call.kind == "push":
            top = (call.remote or "").split("/", 1)[0]
            if top not in {"payload", "state"}:
                problems.append(("push の宛先が決まった部分でない", call.node, call.remote))
        elif call.kind == "run" and call.argv[:2] == ("mkdir", "-p"):
            for path in call.argv[2:]:
                prefix = f"{k.REMOTE_ROOT}/"
                # `cli.SPARK_DIRS` をそのまま参照すると、`src` 側にその一覧を増やす変異を
                # 入れても、この試験が同じ (変異した) 一覧を読んでしまい、決まりが緩んだことに
                # 気づけない。決まった 6 つの置き場所を、ここではリテラルに固定する
                # (レビューでの指摘)
                if not path.startswith(prefix) or path[len(prefix) :] not in (
                    "payload",
                    "models",
                    "probe",
                    "cache",
                    "logs",
                    "state",
                ):
                    problems.append(("mkdir が決まった置き場所の外を作る", call.node, path))
    assert any(call.kind == "push" for call in world.all_calls), "push の呼び出しが記録されていない"
    assert any(
        call.kind == "run" and call.argv[:2] == ("mkdir", "-p") for call in world.all_calls
    ), "mkdir -p の呼び出しが記録されていない (この試験は何も見ていない)"
    assert not problems, problems


# --- 決まり 5: 許可の一覧にないコマンドがない ----------------------------------


def test_rule5_only_allowlisted_commands_and_docker_subcommands_are_used(world: World) -> None:
    """決まり 5: 許可の一覧にないコマンドがない (`remote.py` の一覧を、そのまま読む)。

    `argv[0]` が `remote._ALLOWED_COMMANDS` にあり、`docker` のサブコマンドが
    `remote._ALLOWED_DOCKER_SUBCOMMANDS` / `_ALLOWED_DOCKER_PAIRS` にあることを確かめる。
    `sudo`、`sh`、`bash`、`rm`、`curl`、`wget`、`pip`、`apt` は、この一覧のどれにもない。
    """
    runcalls = [call for call in world.all_calls if call.kind == "run" and call.argv]
    assert runcalls, "run の呼び出しが 1 つも記録されていない (この試験は何も見ていない)"
    problems = []
    for call in runcalls:
        command = call.argv[0]
        if command not in remote_mod._ALLOWED_COMMANDS:
            problems.append(("argv[0] が許可の一覧にない", call.argv))
            continue
        if command != "docker":
            continue
        rest = call.argv[1:]
        if not rest:
            problems.append(("docker にサブコマンドがない", call.argv))
            continue
        pair_ok = len(rest) >= 2 and (rest[0], rest[1]) in remote_mod._ALLOWED_DOCKER_PAIRS
        single_ok = rest[0] in remote_mod._ALLOWED_DOCKER_SUBCOMMANDS
        if not (pair_ok or single_ok):
            problems.append(("docker のサブコマンドが許可の一覧にない", call.argv))
    assert not problems, problems

    forbidden = {"sudo", "sh", "bash", "rm", "curl", "wget", "pip", "apt"}
    used = {call.argv[0] for call in world.all_calls if call.kind == "run" and call.argv}
    assert not (forbidden & used), f"禁じたコマンドが使われている: {forbidden & used}"


# --- 決まり 6: イメージのビルドがない -------------------------------------------


def test_rule6_no_image_build_commit_tag_or_push(world: World) -> None:
    """決まり 6: イメージのビルドがない (`docker build`/`commit`/`tag`/`push` がない)。"""
    forbidden_subcommands = {"build", "commit", "tag", "push"}
    runs = [
        call.argv
        for call in world.all_calls
        if call.kind == "run" and len(call.argv) > 1 and call.argv[0] == "docker"
    ]
    assert runs, "docker の run の呼び出しが 1 つも記録されていない (この試験は何も見ていない)"
    problems = [argv for argv in runs if argv[1] in forbidden_subcommands]
    assert not problems, problems


# --- 決まり 7: トークンらしい環境変数がない -------------------------------------


_SECRET_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")


def test_rule7_no_token_like_environment_variables(world: World) -> None:
    """決まり 7: トークンらしい環境変数がない (`-e`/`--env` の名前に、秘密らしい語がない)。

    `HF_TOKEN` を Mac の側の環境に置いても、`fetch` のコンテナに渡る `-e` は
    `HF_HUB_DISABLE_TELEMETRY=1` だけである (requirements 2.6、8.8)。
    """
    runs = _docker_run_calls(world)
    assert runs, "docker run が 1 つも記録されていない (この試験は何も見ていない)"
    problems = []
    for call in runs:
        for flag, value in zip(call.argv, call.argv[1:], strict=False):
            if flag not in ("-e", "--env"):
                continue
            name = value.split("=", 1)[0]
            hit = [word for word in _SECRET_WORDS if word in name.upper()]
            if hit:
                problems.append((call.node, value, hit))
    assert not problems, problems


# --- 決まり 8: モデルカードの取得がない -----------------------------------------

_MODEL_CARD_MARKS = ("readme", "model_card", "modelcard", "model-card")


def _looks_like_model_card(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    if name.endswith(".gitattributes"):
        return True
    return any(name.startswith(mark) for mark in _MODEL_CARD_MARKS)


def test_rule8_manifest_never_requests_the_model_card_from_the_hub(world: World) -> None:
    """決まり 8 (manifest): Hub への要求に、モデルカードや `.gitattributes` がない。

    `serve manifest` は Spark に触らない (`world.all_calls` に、manifest からの呼び出しは
    そもそも 1 つも入らない。`run_manifest` は Spark に触ったら落ちる実行役を渡している)。
    ここでは、Hub への要求 (`world.manifest_requests`) の道筋を見る。
    """
    assert world.manifest_requests, "Hub への要求が 1 つも記録されていない"
    problems = [
        str(request.url)
        for request in world.manifest_requests
        if _looks_like_model_card(str(request.url).split("?", 1)[0])
    ]
    assert not problems, f"モデルカードらしい道筋への要求がある: {problems}"

    manifest_calls = world.calls_of("manifest")
    assert not manifest_calls, f"serve manifest が Spark に触った: {manifest_calls}"


@dataclass
class DerivedImportRun:
    """`serve derived-import` を 1 度流した結果 (Spark に触らない他のシナリオと、別に流す)。"""

    invocation: k.Invocation
    written: tuple[str, ...]
    """呼び出しの前後で、リポジトリの下に新しくできたファイル (リポジトリの根からの相対の道筋)。"""


@pytest.fixture(scope="module")
def derived_import_run(tmp_path_factory: pytest.TempPathFactory) -> DerivedImportRun:
    """`world` (Spark を相手にする全サブコマンド) とは別に、`derived-import` だけを流す。

    このサブコマンドは Spark にも Hub にも触らないので、`world` の網 (Spark への呼び出しの一覧)
    には、何も足さない。`world` の組み立てが `derived-import` の有無に左右されないように、別の
    フィクスチャにする。
    """
    base = tmp_path_factory.mktemp("safety-derived-import")
    repo = k.make_repo(base, port=k.UNUSED_PORT)
    invocation, written = run_derived_import(repo, base / "inputs")
    return DerivedImportRun(
        invocation=invocation,
        written=tuple(path.relative_to(repo.root).as_posix() for path in written),
    )


def test_derived_import_touches_neither_spark_nor_the_hub(
    derived_import_run: DerivedImportRun,
) -> None:
    """`serve derived-import` は、Spark にも Hub にも触らない。

    操作の対象は Mac の `serving/weights/` だけである。実行役の工場と Hub への口は、呼ばれたら
    落ちるものを渡している。終了コード 0 で終わり、落ちた印の文言が標準エラーにないことで、
    どちらにも触れなかったことを確かめる。
    """
    invocation = derived_import_run.invocation

    assert invocation.code == cli.EXIT_OK, invocation.err
    assert "触ってはいけない" not in invocation.err


def test_derived_import_writes_only_the_manifest_under_serving_weights(
    derived_import_run: DerivedImportRun,
) -> None:
    """`serve derived-import` が新しく作るファイルは、`serving/weights/<名前>.manifest.json` だけ。

    `var_root` の下 (記録の置き場所) にも、`payload/` にも、何も書かない。
    """
    assert derived_import_run.written == ("serving/weights/k2s1.manifest.json",)


def test_rule8_fetch_excludes_the_model_card_from_the_download(world: World) -> None:
    """決まり 8 (fetch): 重みの取得のコンテナに渡す引数が、モデルカードを除く指定を持つ。"""
    runs = [
        call.argv
        for call in world.calls_of("fetch_success", "fetch_mismatch")
        if call.kind == "run" and call.argv[:2] == ("docker", "run")
    ]
    assert runs, "fetch の docker run が記録されていない"
    for argv in runs:
        assert "--exclude" in argv, argv
        excluded = [
            value for flag, value in zip(argv, argv[1:], strict=False) if flag == "--exclude"
        ]
        assert "README.md" in excluded, argv


# --- 決まり 9: 記録に、応答の本文・会話の文・Authorization が出ない -------------


def _message_texts(body: Mapping[str, Any] | None) -> list[str]:
    """要求の本文 (`/v1/messages` の JSON) から、会話の文だけを抜き出す (短いものは除く)。"""
    if not body:
        return []
    texts: list[str] = []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, str) and len(content) > 8:
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("thinking")
                    if isinstance(text, str) and len(text) > 8:
                        texts.append(text)
    return texts


def test_rule9_no_response_body_conversation_or_auth_leaks_into_recorded_files(
    world: World,
) -> None:
    """決まり 9: 回収した記録と、smoke/probe/thinking/watch の結果のファイルに、応答の本文、
    送った会話の文、`Authorization` が含まれない。

    `smoke`、`probe`、`thinking` の偽の応答には目印 (`BODY_MARKER`) を仕込んであり、`watch` は
    `var_root` の下に `samples.jsonl` と `result.json` を書く。これらのファイルすべてを
    読み、目印と、実際に送った会話の文 (thinking の要求の本文から読む) が出ないことを見る。
    シナリオごとに、確かめるファイルが実際に 1 件以上あることも見る (網が抜けていないか)。
    """
    leak_scenarios = ("smoke_success", "probe_ready", "thinking_success", "watch_quiet")
    # `smoke` は画面に応答を見せる仕組みなので (probe.py の見本と同じ考え方。日本語の文字化けを
    # 目で確かめるため)、`var_root` の下にファイルを 1 つも書かない。ファイルを書くと分かって
    # いる残り 3 つのシナリオだけ、「1 件以上見た」ことを確かめる
    file_producing_scenarios = ("probe_ready", "thinking_success", "watch_quiet")
    per_scenario_files: dict[str, list[Path]] = {
        name: k.files_under(world.var_roots[name]) for name in leak_scenarios
    }
    for name in file_producing_scenarios:
        assert per_scenario_files[name], (
            f"{name} の記録のファイルが見つからない (この試験は何も見ていない)"
        )
    files = [path for scenario_files in per_scenario_files.values() for path in scenario_files]
    assert files, "確かめる記録のファイルが見つからない (この試験は何も見ていない)"
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in files if path.is_file()
    )
    assert BODY_MARKER not in combined, "応答の本文の目印が、回収した記録に出ている"
    assert "authorization" not in combined.lower(), "Authorization が、回収した記録に出ている"

    # `probe` と `watch` の偽サーバーが、実際に要求を受けたことも見ておく (でなければ、上の
    # 目印の確認は「そもそも何も送っていない」ので何も確かめていないことになる)
    assert world.vllms["probe"].requests_for("/v1/messages"), (
        "probe が /v1/messages を送っていない (この試験は何も見ていない)"
    )
    assert world.vllms["watch"].requests, "watch が推論サーバーに 1 件も要求していない"

    for name, server in world.vllms.items():
        for request in server.requests:
            header_names = {key.lower() for key in request.headers}
            assert "authorization" not in header_names, f"{name} への要求に Authorization がある"
        for request in server.requests_for("/v1/messages"):
            for text in _message_texts(request.body):
                assert text not in combined, f"送った会話の文が、記録に出ている: {text[:40]!r}"
