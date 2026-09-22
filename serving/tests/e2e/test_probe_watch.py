"""縮小の確認 (`serve probe`) と見張り (`serve watch`) の、端から端までの試験 (tasks.md 5.3)。

確かめること (design.md 「確認 › probe」「確認 › watch」、requirements 5.1〜5.5、7.7、
tasks.md 5.3 の「縮小の確認と、見張りの、端から端までの試験も、ここで書く」):

- `serve probe <構成>`: `ready` (0)、`failed` (2。コンテナの終了で、時間切れを待たずに、
  片付けの順序が回収 → 停止 → 削除)、`inconclusive` (1。関門の断り、または、時間切れの手前で
  判定が出ない場合)
- **5.2 の学び**: 「時間切れを待たない」は、`detail` の文言と `docker container inspect` の
  回数で固定する (終了コードと順序だけでは固定できない)
- `serve watch <構成>`: 出来事なし (0)、`unresponsive` (2)、`stalled` (2)、中断 (130。それまでの
  要約が書き出されている)。`--duration`/`--interval` を小さくし、`cli.main` の `sleep`/`clock`/
  `now` の差し込み口を使って、実時間を待たない

どちらも `cli.main` が入口である。部品を `monkeypatch` しない。共通の下ごしらえ
(構成の TOML、`Repo`、`invoke()`) は `e2e_kit.py` (このディレクトリの新規モジュール、
`test_safety.py` と共有) から使う。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import e2e_kit as k

from fake_runner import Reply, Rule
from fake_vllm import FakeVllm, MetricsSample
from serving_kit import cli
from serving_kit.types import ContainerPlan
from serving_kit.watch import GPU_UTIL_ARGV

# --- 縮小の確認の記録の見本 ---------------------------------------------------

PROBE_READY_LOG = (
    "INFO Using FLASH_ATTN_MLA attention backend out of potential backends: ['FLASH_ATTN_MLA']\n"
    "INFO GPU KV cache size: 1,234,567 tokens,"
    " Maximum concurrency for 163,840 tokens per request: 7.54x\n"
    "INFO init engine (profile, create kv cache, warmup model) took 42.00 seconds\n"
)
PROBE_FAILURE_MARK = "pe_dim must be 64"
PROBE_FAILURE_LOG = "".join(f"INFO booting up {i}\n" for i in range(20)) + (
    f"ERROR AssertionError: {PROBE_FAILURE_MARK}\n"
)

_CONTAINER_ID = "probe0001"


def _probe_repo(base: Path, name: str, port: int) -> k.Repo:
    return k.make_repo(base / name, port=port)


def _probe_plan(repo: k.Repo) -> ContainerPlan:
    config, nodes = k.load_config_and_nodes(repo, k.PROBE_CONFIG)
    return k.container_plans(config, nodes)["head"]


def _probe_base_script(plan: ContainerPlan, *, layout_ok: bool = True) -> tuple[Rule, ...]:
    """`serve probe` が了承の前に流す、8 つの関門のうち読み取りを要るものすべて。"""
    return (
        *k.gate_rules(("head",), layout_ok=layout_ok),
        Rule(
            prefix=("cat",),
            node="head",
            replies=(Reply(stdout=k.record_json("head", "probe_files")),),
        ),
    )


def _probe_listing_rule(plan: ContainerPlan) -> Rule:
    """一覧: 起こす前は空、起こしたあとは自分の行 (`test_start_stop.py` と同じ考え方)。"""
    return Rule(
        prefix=k.OWN_CONTAINERS_ARGV,
        node="head",
        replies=(Reply(stdout=""), Reply(stdout=k.ps_row(plan, container_id=_CONTAINER_ID))),
    )


# --- `serve probe` ------------------------------------------------------------


def test_probe_ready_sends_exactly_one_request_and_exits_zero(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """起動したら `ready` (0)。短い要求はちょうど 1 つだけ送る (requirements 5.4、10.5)。"""
    repo = _probe_repo(tmp_path, "ready", k.port_of(fake_vllm))
    plan = _probe_plan(repo)
    script = (
        _probe_listing_rule(plan),
        *_probe_base_script(plan),
        Rule(prefix=("docker", "run"), node="head", replies=(Reply(stdout=_CONTAINER_ID + "\n"),)),
        Rule(
            prefix=("docker", "container", "inspect"),
            node="head",
            replies=(Reply(stdout="running 0\n"),),
        ),
        Rule(
            prefix=("docker", "logs", "--timestamps"),
            node="head",
            replies=(Reply(stdout=PROBE_READY_LOG),),
        ),
        Rule(prefix=("docker", "stop"), node="head", replies=(Reply(),)),
        Rule(prefix=("docker", "rm"), node="head", replies=(Reply(),)),
        Rule(kind="push", replies=(Reply(),)),
        Rule(kind="pull", replies=(Reply(exit_code=23),)),
    )
    result = k.invoke(["probe", k.PROBE_CONFIG, "--yes"], repo, script=script)

    assert result.code == cli.EXIT_OK, result.err
    pairs = k.kv(result.out)
    assert pairs["status"] == "ready"
    assert fake_vllm.call_count("/v1/messages") == 1

    # 起こしたコンテナが残らない (回収してから、必ず止めて消す)
    stop_calls = [c for c in result.runner.calls if c.argv[:2] == ("docker", "stop")]
    rm_calls = [c for c in result.runner.calls if c.argv[:2] == ("docker", "rm")]
    assert len(stop_calls) == 1
    assert len(rm_calls) == 1


def test_probe_failed_does_not_wait_for_the_timeout(tmp_path: Path) -> None:
    """`failed` (2)。コンテナの終了を、**1 度の読み取りで** 検出し、時間切れを待たない。

    5.2 の学び: 終了コードと呼び出しの順序だけでは「待たなかった」ことを固定できないので、
    `detail` の文言 (終了を検出したときに `lifecycle.wait_ready` が作る「時間切れを待たずに
    失敗にする」と「終了した (状態 exited、終了コード 1)」が、肯定形で含まれる) と、
    `docker container inspect` の回数 (ちょうど 1 回) の両方で固定する。
    """
    repo = _probe_repo(tmp_path, "failed", k.UNUSED_PORT)
    plan = _probe_plan(repo)
    script = (
        _probe_listing_rule(plan),
        *_probe_base_script(plan),
        Rule(prefix=("docker", "run"), node="head", replies=(Reply(stdout=_CONTAINER_ID + "\n"),)),
        # ちょうど 1 回だけ返事を書く。2 度目を読みに行けば、この台本にない呼び出しとして
        # AssertionError になり、この試験自体が落ちる (「1 回だけ読む」を、台本の形でも縛る)
        Rule(
            prefix=("docker", "container", "inspect"),
            node="head",
            replies=(Reply(stdout="exited 1\n"),),
        ),
        Rule(
            prefix=("docker", "logs", "--timestamps"),
            node="head",
            replies=(Reply(stdout=PROBE_FAILURE_LOG),),
        ),
        Rule(prefix=("docker", "stop"), node="head", replies=(Reply(),)),
        Rule(prefix=("docker", "rm"), node="head", replies=(Reply(),)),
        Rule(kind="push", replies=(Reply(),)),
        Rule(kind="pull", replies=(Reply(exit_code=23),)),
    )

    started = time.monotonic()
    result = k.invoke(["probe", k.PROBE_CONFIG, "--timeout", "1h", "--yes"], repo, script=script)
    elapsed = time.monotonic() - started

    assert result.code == cli.EXIT_FAILED, result.err
    assert elapsed < 5.0, f"時間切れを待った形跡がある ({elapsed:.1f} 秒かかった)"
    pairs = k.kv(result.out)
    assert pairs["status"] == "failed"
    assert PROBE_FAILURE_MARK in "".join(result.err.splitlines())
    # `lifecycle.wait_ready` が、コンテナの終了を検出したときに作る文言そのもの
    # (design.md 「lifecycle」の受け付けの開始の判定)。「時間切れを待たずに」失敗にしたことと、
    # 終了の状態・終了コードを、文言でも固定する (実装に存在しない文言を `not in` で書かない)
    assert "時間切れを待たずに失敗にする" in pairs["detail"]
    assert "終了した (状態 exited、終了コード 1)" in pairs["detail"]

    inspect_calls = [
        c for c in result.runner.calls if c.argv[:3] == ("docker", "container", "inspect")
    ]
    assert len(inspect_calls) == 1, (
        f"1 度の読み取りで終了を検出せず、待ちのループに入った疑いがある: {len(inspect_calls)} 回"
    )

    # 片付けの順序: 記録の回収 (docker logs。見せる末尾、分類の末尾、全量の回収の複数回ぶん)
    # → 停止 (docker stop) → 削除 (docker rm)。`docker logs` の回数そのものは、
    # `wrap_up` の実装の詳細 (見せる/分類の 2 種類の tail と、全量の回収) なので数を固定
    # せず、「すべての docker logs が、stop・rm より前に来ている」ことだけを固定する
    ordered = [
        (c.argv[0], c.argv[1])
        for c in result.runner.calls
        if c.argv[:2] in (("docker", "logs"), ("docker", "stop"), ("docker", "rm"))
    ]
    logs_indices = [index for index, pair in enumerate(ordered) if pair[1] == "logs"]
    stop_index = next(index for index, pair in enumerate(ordered) if pair[1] == "stop")
    rm_index = next(index for index, pair in enumerate(ordered) if pair[1] == "rm")
    assert logs_indices, "docker logs が 1 度も呼ばれていない"
    assert max(logs_indices) < stop_index < rm_index, ordered


def test_probe_inconclusive_when_a_gate_refuses(tmp_path: Path) -> None:
    """`inconclusive` (1)。関門が断れば、Spark に触る前に「判定できない」で終わる。"""
    repo = _probe_repo(tmp_path, "gate-refused", k.UNUSED_PORT)
    plan = _probe_plan(repo)
    script = (
        _probe_listing_rule(plan),
        *_probe_base_script(plan, layout_ok=False),
    )
    result = k.invoke(["probe", k.PROBE_CONFIG, "--yes"], repo, script=script)

    assert result.code == cli.EXIT_PRECONDITION, result.err
    pairs = k.kv(result.out)
    assert pairs["status"] == "inconclusive"
    assert [c for c in result.runner.calls if c.mutating] == []


@dataclass
class ProbeTimeoutClock:
    """試験用の時計。眠りは記録するだけで、実際には眠らない (`test_probe.py` の `FakeClock`)。"""

    now: float = 0.0
    slept: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_probe_inconclusive_when_it_times_out_without_exiting(tmp_path: Path) -> None:
    """`inconclusive` (1)。終了せずに、待ちの上限に達したら「判定できない」になる。

    `docker container inspect` はずっと `running` を返し続ける。試験は `clock`/`sleep` を
    差し込むので、実際には 1 秒もかからない (`elapsed` で、そのことも固定する)。
    """
    repo = _probe_repo(tmp_path, "timed-out", k.UNUSED_PORT)
    plan = _probe_plan(repo)
    script = (
        _probe_listing_rule(plan),
        *_probe_base_script(plan),
        Rule(prefix=("docker", "run"), node="head", replies=(Reply(stdout=_CONTAINER_ID + "\n"),)),
        Rule(
            prefix=("docker", "container", "inspect"),
            node="head",
            replies=(Reply(stdout="running 0\n"),),
        ),
        # 判定できなくても、`wrap_up` はどの結果でも記録を回収してから片付ける (見せる末尾を
        # 必ず読む。requirements 5.4)
        Rule(
            prefix=("docker", "logs", "--timestamps"),
            node="head",
            replies=(Reply(stdout="INFO まだ起動中\n"),),
        ),
        Rule(prefix=("docker", "stop"), node="head", replies=(Reply(),)),
        Rule(prefix=("docker", "rm"), node="head", replies=(Reply(),)),
        Rule(kind="push", replies=(Reply(),)),
        Rule(kind="pull", replies=(Reply(exit_code=23),)),
    )
    clock = ProbeTimeoutClock()

    started = time.monotonic()
    result = k.invoke(
        ["probe", k.PROBE_CONFIG, "--timeout", "25", "--yes"],
        repo,
        script=script,
        sleep=clock.sleep,
        clock=clock.monotonic,
    )
    elapsed = time.monotonic() - started

    assert result.code == cli.EXIT_PRECONDITION, result.err
    assert elapsed < 2.0, f"実際に眠った形跡がある ({elapsed:.1f} 秒かかった)"
    pairs = k.kv(result.out)
    assert pairs["status"] == "inconclusive"
    # `lifecycle.wait_ready` が時間切れのときに作る文言 (design.md 「lifecycle」の
    # 受け付けの開始の判定)。「終了せずに時間切れになった」ことを、文言でも固定する
    assert "秒のうちに、要求を受け付けられる状態にならなかった" in pairs["detail"]

    # 眠った回数から、待ちのループの周回数が分かる。上限 (25 秒) を、既定の間隔 (10 秒) で
    # 割り切れないところまで進めるので、コンテナの状態を読む回数は「周回数 + 1」になる
    # (最後にもう一度読んでから、時間切れと判定するため)
    inspect_calls = [
        c for c in result.runner.calls if c.argv[:3] == ("docker", "container", "inspect")
    ]
    assert len(inspect_calls) == len(clock.slept) + 1
    assert clock.now >= 25.0


# --- `serve watch` -------------------------------------------------------------


@dataclass
class WatchClock:
    """試験用の時計 (`test_watch.py` の `WatchClock` と同じ考え方)。壁の時計と単調な時計を、
    同じ歩幅で進める。`interrupt_after` を指定すると、その回数だけ眠ったところで
    `KeyboardInterrupt` を投げる。
    """

    mono: float = 0.0
    wall: datetime = field(default_factory=lambda: datetime(2026, 9, 22, 3, 0, 0, tzinfo=UTC))
    slept: list[float] = field(default_factory=list)
    interrupt_after: int | None = None

    def monotonic(self) -> float:
        return self.mono

    def now(self) -> datetime:
        return self.wall

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        if self.interrupt_after is not None and len(self.slept) >= self.interrupt_after:
            raise KeyboardInterrupt
        self.mono += seconds
        self.wall += timedelta(seconds=seconds)


def _watch_repo(base: Path, name: str, port: int) -> k.Repo:
    return k.make_repo(base / name, port=port)


def _watch_argv(*, duration: str, interval: str) -> list[str]:
    return ["watch", k.SERVE_CONFIG, "--duration", duration, "--interval", interval]


def test_watch_reports_no_events_when_the_server_stays_healthy(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """出来事なし (0)。健康なままなら、決めた時間まで観察して、静かに終わる。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, waiting_requests=0))
    repo = _watch_repo(tmp_path, "quiet", k.port_of(fake_vllm))
    clock = WatchClock()
    script = (Rule(prefix=GPU_UTIL_ARGV, replies=(Reply(stdout="5 %\n"),)),)

    result = k.invoke(
        _watch_argv(duration="5", interval="1"),
        repo,
        script=script,
        default=Reply(),
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
    )

    assert result.code == cli.EXIT_OK, result.err
    pairs = k.kv(result.out)
    assert pairs["status"] == "quiet"
    assert pairs["events"] == "0"


def test_watch_reports_unresponsive_after_three_consecutive_failures(tmp_path: Path) -> None:
    """`unresponsive` (2)。`/health` が続けて 3 回失敗すると出る。"""
    repo = _watch_repo(tmp_path, "unresponsive", k.UNUSED_PORT)
    # `/health` に一度もつながらない port (推論サーバーを起こさない)
    clock = WatchClock()
    script = (Rule(prefix=GPU_UTIL_ARGV, replies=(Reply(stdout="5 %\n"),)),)

    result = k.invoke(
        _watch_argv(duration="3", interval="1"),
        repo,
        script=script,
        default=Reply(),
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
    )

    assert result.code == cli.EXIT_FAILED, result.err
    pairs = k.kv(result.out)
    assert pairs["events"] == "1"
    assert pairs["event.1.finding"] == "unresponsive"


def test_watch_reports_stalled_when_tokens_stop_increasing(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """`stalled` (2)。処理中の要求があるのに、生成のトークンの数が窓のあいだ増えない。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=1, generation_tokens_total=100))
    repo = _watch_repo(tmp_path, "stalled", k.port_of(fake_vllm))
    clock = WatchClock()
    script = (Rule(prefix=GPU_UTIL_ARGV, replies=(Reply(stdout="97 %\n"),)),)

    result = k.invoke(
        [
            *_watch_argv(duration="5", interval="1"),
            "--stall-window",
            "3",
        ],
        repo,
        script=script,
        default=Reply(),
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
    )

    assert result.code == cli.EXIT_FAILED, result.err
    pairs = k.kv(result.out)
    assert pairs["events"] == "1"
    assert pairs["event.1.finding"] == "stalled"


def test_watch_interrupt_writes_a_partial_summary_before_stopping(
    tmp_path: Path, fake_vllm: FakeVllm
) -> None:
    """中断 (130)。それまでの観察の要約が、中断を伝える前に書き出されている。"""
    fake_vllm.set_metrics(MetricsSample(running_requests=0, waiting_requests=0))
    repo = _watch_repo(tmp_path, "interrupt", k.port_of(fake_vllm))
    clock = WatchClock(interrupt_after=2)
    script = (Rule(prefix=GPU_UTIL_ARGV, replies=(Reply(stdout="5 %\n"),)),)

    result = k.invoke(
        _watch_argv(duration="100", interval="1"),
        repo,
        script=script,
        default=Reply(),
        sleep=clock.sleep,
        clock=clock.monotonic,
        now=clock.now,
    )

    assert result.code == cli.EXIT_INTERRUPTED, result.err
    run_dirs = [path for path in repo.var_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    result_files = list(run_dirs[0].glob("result.json"))
    assert result_files, "中断のあとも、要約 (result.json) が書き出されているはず"
    written = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert written["sample_count"] == 2
    samples_files = list(run_dirs[0].glob("samples.jsonl"))
    assert samples_files
    lines = samples_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
