"""起動と、その待ちと、失敗のときの片付け (design.md 「運転 › lifecycle」)。

受け持つのは、`serve start <構成>` の中身である (tasks.md 3.4)。`serve status` / `serve stop` /
`serve smoke` は、task 3.5 が、この module に足す。

進む順 (design.md 「System Flows › 起動」):

1. 構成が使う台のすべてで、自分のラベルで絞った一覧を読み、`guards.match_running` で判定する。
   名前・構成の名前・イメージのダイジェスト・`config-sha256` が 2 台とも一致して動いていれば、
   **関門も了承も通さずに** `already_running` (終了コード 0)。名前が同じで中身が違うときは、
   違う項目を示して `refused` (終了コード 1。`serve stop` を促す)
2. 関門 (`guards.run_gates` の 8 つ)。1 つでも落ちたら `refused` で、了承も、状態を変える
   呼び出しも、1 つも出さない
3. 計画 (`plan.build_plans`) を `guards.build_approved_plan` にかけ、`guards.request_approval`
   で了承を得る。断られたら `guards.ApprovalError` (終了コード 1)
4. 起動の記録 (`types.LaunchRecord`) を Mac で作り、2 台の `state/<構成>.launch.json` に置く
5. 2 台を続けて起こす (`docker run -d`。順序は構成の `nodes` の順。design は起こす順序を
   定めていない。止めるときだけ head → worker にする)
6. 受け付けの開始を待つ。10 秒ごとに (1) head の `/health` が 200、(2) `/v1/models` の
   `data[0].id` が `served_model_name` と一致、(3) `/metrics` が読めて `vllm:spec_decode_` で
   始まる行がない、を見る。上限は構成の `ready_timeout_s` で、`timeout_s` でその回だけ
   上書きできる。待っている間、毎回、2 台のコンテナの状態も見る

**結果と例外の、終了コードへの写し方** (5.1 が、この表のとおりに写す。design.md
「Error Handling」):

- **0**: `StartOutcome.status` が `ready`、`already_running`
- **1**: `StartOutcome.status` が `refused` (関門の不通過、名前が同じで中身が違う)。
  `guards.ApprovalError`、`config.ConfigError`、`ValueError`
- **2**: `StartOutcome.status` が `failed` (時間切れ、コンテナの終了、待っても直らない
  食い違い)。`LifecycleError` (起こせなかった、名前の衝突、了承のあとに読み取りが届かな
  かった、起動の記録を置けなかった、片付けられなかった)
- **130**: `KeyboardInterrupt` (片付けてから伝える)

`RemoteError` は、この関数からは出ない。了承の**前** (関門) の「入れない」は `run_gates` が
`GateResult` (`passed=False`) に変えるので `refused` になり、了承の**あと**に届かなかったこと
は `LifecycleError` に包む (task 3.1 の `image` と同じ流儀)。

守る決まり:

- **1 つの起動の仕組み** (design.md 「Architecture Integration」): 推論サーバーも、ほかの 4 つの
  `kind` と同じ道を通る。`plan.build_plans` が組み立てた引数の列を、名前とラベルを付けて `-d` で
  起こす。前面で動かす経路、終了時に自動で消す指定、構成を持たずに起こす経路は作らない
- **了承** (requirements 2.1): 状態を変える呼び出し (起動の記録の配布、`docker run`、
  `docker stop` / `rm`) は、すべて `guards.build_approved_plan` が作った計画に入れ、
  `guards.request_approval` で了承を得てから流す
- **コンテナを対象にする操作の不変条件** (requirements 2.3、2.4): 対象にできるのは、(a)
  `guards.list_own_containers` (自分のラベルで絞った、ただ 1 つの入口) が返した行の識別子か、
  (b) 了承済みの計画が自分で起こす名前だけである。待っている間の `docker container inspect` は
  (a) の識別子に向ける。記録の読み取りは `logs` の口に任せる (同じ不変条件を守る)。止めると
  消すは、**流す直前に、その名前の行が自分の一覧にあることを `guards.rollback_commands` で
  確かめてから**流す
- **片付けてから終わる** (requirements 1.5、design.md 「Error Handling」): 失敗でも中断でも、
  記録の末尾を読み、記録を回収してから、自分のコンテナを台のすべてで止めて消す。半端に
  起きた推論サーバーを残さない (重みの取得 (task 3.3) が、失敗と中断で**止めない**のとは、
  意図して違える。184 GiB の取得と違い、起きかけのサーバーは捨ててよい)。読んだ末尾は、
  `failed` を返す経路では `StartOutcome.log_tails` に入れて 5.1 が見せ、例外の経路では
  (結果を返せないので) この module が `report` に書く
- **中断は、片付けを止めない** (レビューの指摘 1、2): 片付けの中の読み取り (見せる末尾、分類の
  末尾、記録の回収) は、どれも `KeyboardInterrupt` から守る。中断が来たら、**残りの読み取りを
  飛ばして**、必ず停止と削除に到達する。停止と削除は、中断が何度来ても、台のすべてで
  `stop` → `rm` の両方を試みる (`docker rm` (強制なし) は、動いているコンテナを消せないので、
  `stop` を飛ばすとコンテナが残る)。ただし、**一覧の読み取りで中断が来たときは、何も止めない**
  (一覧が読めていないので、その名前が自分のものだと言えない。requirements 2.3)。決まりと理由は
  `wrap_up` と `clean_up` の docstring にある
- **HTTP は head の LAN のアドレスにだけ送る** (design.md 「Security Considerations」)。
  `httpx.Client` は `trust_env=False` で作り、Mac の側のプロキシや `.netrc` を拾わない。
  応答の本文は、どのファイルにも書かない (requirements 10.5)

依存の向きにより、この module が読み込む `serving_kit` は `types`、`config`、`remote`、`plan`、
`observe`、`guards`、`logs` だけである (`probe`、`netcheck`、`watch`、`thinking`、`cli` は
読み込まない)。

**4.1 (`probe`) は、下線のない公開の口だけで書ける** (レビューの指摘 3。`probe` は
`lifecycle.py` を直せないので、使い回す助けを、すべて公開にしてある)。1 台の縮小の確認は、
`Controls` を組み立てて、`record_pushes` (了承を得る計画に足す配布) → `launch_record` →
`push_launch_records` → `start_all` → `started_targets` → `wait_ready` → `observation` →
`wrap_up` (その中で `clean_up`) の順に呼べば組み立てられる。待ち方の判定 (`check_ready`、
`Readiness.fatal`、`Waited.exited`) も、そのまま使える。`wrap_up` と `clean_up` は、**成功の
あとにも呼べる** (`probe` は、どの結果でも記録を回収して、コンテナを残さない)。起こす途中の
失敗は `RunFailed` で上がるので、`probe` は、それを捕まえて片付け、自分の結果の型に写す。

design の文からの、意図した決めごと (design が明示しない細部を、ここで決めて残す):

1. **`kind = "serve"` の構成だけを受ける**。`serve start` は推論サーバーを起こすコマンドである
   (design.md 「cli」)。縮小の確認 (`kind = "probe"`) は、結果の分け方 (`inconclusive`) も、
   記録を必ず回収することも違うので、task 4.1 が `probe` に書く
2. **HTTP のポートは、`args` の `--port` の設定から取る** (`http_port`)。待ち受けの印
   (`is_port`) だけでは選べない: design.md Data Models の見本では、`--port 8000` (HTTP) と
   `--master-port 29501` (分散の初期化) の**両方**に印が付く。印は `gate_ports_free` が「空いて
   いるか」を見るためのもので、どれが HTTP かは決めない。そこで、vLLM の口 (`vllm serve --port`)
   に当たる `--port` を、`args` の中から 1 つだけ探す。0 個でも 2 個以上でも `ConfigError`
3. **待ちの 1 周は、HTTP → コンテナの状態の順**に見る (design.md の System Flows の loop の
   並び)。受け付けが始まっていれば、その周ではコンテナの状態を見に行かない
4. **待っても直らない食い違いは、時間切れを待たずに失敗にする**: (a) `/v1/models` が読めて、
   名乗る名前が構成と違う (requirements 6.2)、(b) `/metrics` が読めて、投機的デコードの指標が
   出ている (requirements 6.7)。どちらも、構成を直さなければ変わらない。**読めないこと**
   (つながらない、200 でない、Prometheus の形でない) は、起動の途中でありうるので、上限まで待つ
5. **見せる末尾と、分類に使う末尾を分ける** (親の判断、2026-09-22)。計測者に見せるのは 80 行
   (`logs.DEFAULT_TAIL_LINES`。requirements 1.5) のままだが、失敗の種類の読み取り
   (`observe.observe_launch`) には、**2 台ぶんの長い末尾** (`OBSERVE_TAIL_LINES`) を読み直して
   掛ける。vLLM の起動の失敗は、API のサーバー・EngineCore・worker の traceback が重なって
   数百行になりうるので、`pe_dim must be 64` のような肝心の行が、最後の 80 行より手前に来て、
   見せる末尾だけでは `UNCLASSIFIED` に落ちるためである (段 0 と段 2 の判定の要。
   requirements 5.3、8.1)。2 台の両方に知っている失敗が出たときの決まりは、`observation` の
   docstring にある。全量は `logs.collect_logs` が `serving/var/` に回収するので、計測者は、
   そちらを読める。末尾を使うのは、この module が `logs` の書いたファイルの名前に寄りかから
   ないためである。`observe_launch` は「終了したか」を知らないので、**コンテナが終了していて、
   知っている失敗がどれも読めなかったときだけ**、この module が `UNCLASSIFIED` を付ける
   (tasks.md の Implementation Notes 2.2)
6. **成功のときは、記録の全量を回収しない** (design.md の System Flows の `ready` の分岐)。
   起動の記録から読む事実 (選ばれた部品、KV キャッシュの大きさ、所要。requirements 6.3) は、
   末尾を長めに (`OBSERVE_TAIL_LINES`) 読んで得る。全量が要るときは `serve logs` を打つ
7. **`vllm_version` は `/version` から埋める** (tasks.md の Implementation Notes 2.2: 記録の
   中の版の行の原文が research.md にないので、`observe` はつねに空を返す)
8. **`except` の経路で、片付けの最中に中断が来たら、中断を伝える** (task 3.1 が決めずに残した
   点を、ここで決める)。元の例外が何であれ、片付けをできるところまで (`docker rm` まで) 進めて
   から `KeyboardInterrupt` を伝えるので、終了コードは 130 になる。中断は、計測者の意思表示で
   あり、実行しての失敗より優先して伝える。**待ちを中断する 1 度目の Ctrl-C は、この長い片付けを
   始める合図なので、2 度目の Ctrl-C は、例外ではなく、ふつうの操作として扱う** (`wrap_up` は
   中断を投げずに `WrapUp.interrupted` で返し、投げるのは、見せるものを見せた呼ぶ側にする)
9. **結果の型は `types.StartOutcome` をそのまま使う**。`ServiceStatus` は、`ready` と
   `already_running` のときだけ埋める。`gpu_apps` と `fabric_link_up` は、この module が読まない
   ので空にする (`serve status` (task 3.5) が読む)
10. **回収の置き場所の時刻は、`started_at` (起こす時刻) を使う**。1 回の `start` で回収は
   高々 1 度なので、同じ秒に 2 度書く心配がない (tasks.md の Implementation Notes 2.4)
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final, TextIO

import httpx

from serving_kit.config import ConfigError
from serving_kit.guards import (
    READ_TIMEOUT_S,
    STOP_TIMEOUT_S,
    Confirmer,
    OwnContainer,
    build_approved_plan,
    list_own_containers,
    match_running,
    request_approval,
    rollback_commands,
    run_gates,
)
from serving_kit.logs import (
    DEFAULT_TAIL_LINES,
    LOG_READ_TIMEOUT_S,
    collect_logs,
    launch_record_remote_path,
    tail_logs,
    var_dir,
)
from serving_kit.observe import observe_launch
from serving_kit.plan import (
    LABEL_CONFIG,
    LABEL_CONFIG_SHA256,
    LABEL_IMAGE,
    LABEL_STARTED_AT,
    build_plans,
)
from serving_kit.remote import RemoteError, RemoteRunner
from serving_kit.types import (
    CommandResult,
    ConfigDef,
    ContainerPlan,
    ContainerState,
    GateResult,
    KnownFailure,
    LaunchObservation,
    LaunchRecord,
    NodeDef,
    NodeRole,
    NodeStatus,
    PlannedPush,
    ServiceStatus,
    StartOutcome,
    WeightsManifest,
)

__all__ = [
    "CLEANUP_TIMEOUT_S",
    "COMMAND_NAME",
    "HTTP_PORT_FLAG",
    "HTTP_TIMEOUT_S",
    "LAUNCH_RECORD_SUBDIR",
    "OBSERVE_TAIL_LINES",
    "READY_POLL_INTERVAL_S",
    "SPEC_DECODE_METRIC_PREFIX",
    "START_TIMEOUT_S",
    "Controls",
    "LifecycleError",
    "Readiness",
    "RunFailed",
    "Target",
    "Waited",
    "WrapUp",
    "check_ready",
    "clean_up",
    "http_port",
    "label_time",
    "launch_record",
    "new_client",
    "observation",
    "push_launch_records",
    "record_pushes",
    "start",
    "start_all",
    "started_targets",
    "wait_ready",
    "wrap_up",
]


# --- 決まった値 -----------------------------------------------------------

READY_POLL_INTERVAL_S: Final[float] = 10.0
"""受け付けの開始を見に行く間隔 (design.md 「lifecycle」: 10 秒ごとに見る)。"""

START_TIMEOUT_S: Final[float] = 120.0
"""`docker run -d` の時間切れ。切り離して起こすので、すぐ返る。"""

HTTP_TIMEOUT_S: Final[float] = 10.0
"""1 回の HTTP の問い合わせの時間切れ (`/health`、`/v1/models`、`/metrics`、`/version`)。"""

CLEANUP_TIMEOUT_S: Final[float] = float(STOP_TIMEOUT_S) + 30.0
"""片付け (`docker stop -t 90` と `docker rm`) の時間切れ。停止の猶予より長くする。"""

OBSERVE_TAIL_LINES: Final[int] = 2000
"""記録から事実を読むために見る行数 (成功のときの観察と、失敗の種類の分類。決めごとの 5、6)。

計測者に**見せる**行数 (`logs.DEFAULT_TAIL_LINES` = 80。requirements 1.5) とは別にする。
vLLM の起動の失敗では、API のサーバー・EngineCore・worker の traceback が重なって数百行に
なりうるので、`pe_dim must be 64` のような、知っている失敗の肝心の行が、最後の 80 行より手前に
来ることがある。見せる末尾だけで分類すると、そのとき `UNCLASSIFIED` に落ちてしまう。失敗の
種類の読み取りは、段 0 (縮小の確認) と段 2 の判定の要 (requirements 5.3、8.1) なので、分類には
つねにこの長さで読み直す (親の判断、2026-09-22)。成功のときの観察 (選ばれた部品、KV キャッシュ
の大きさ、所要) も、同じ理由で 80 行では足りない。
"""

_FAILURE_PRIORITY: Final[Mapping[KnownFailure, int]] = {
    failure: index for index, failure in enumerate(KnownFailure)
}
"""知っている失敗の優先の順序 (`types.KnownFailure` の宣言の順)。

`observe._KNOWN_FAILURE_PATTERNS` が判定に使う順序と同じものである (1 つの記録に 2 つの文面が
あるときに、`observe` が先の種類を返すのと、2 台に別の種類が出たときに、この module が先の
種類を採るのを、同じ順序でそろえる)。
"""

SPEC_DECODE_METRIC_PREFIX: Final[str] = "vllm:spec_decode_"
"""投機的デコードが入っていることを表す指標の接頭辞 (requirements 6.7)。"""

HTTP_PORT_FLAG: Final[str] = "--port"
"""HTTP のポートを表す、vLLM の口 (決めごとの 2)。"""

COMMAND_NAME: Final[str] = "start"
"""記録の置き場所の名前に使う、`serve` のサブコマンドの名前 (`logs.var_dir`)。"""

LAUNCH_RECORD_SUBDIR: Final[str] = "state"
"""起動の記録を配る先 (`remote` が許す 2 つの宛先のうちの 1 つ)。"""

_RECORD_DIRNAME: Final[str] = "launch"
"""記録を配る元の、`record_dir` の下の名前 (**この module だけが使う**。決めごとの 10)。"""

_KIND_SERVE: Final[str] = "serve"
"""`serve start` が受ける構成の種類 (決めごとの 1)。"""

_HEAD: Final[NodeRole] = "head"
"""HTTP の宛先になる役割 (design.md Data Models: `--host {head.lan_addr}`)。"""

_METRIC_PREFIX: Final[str] = "vllm:"
"""`/metrics` が読めたと言えるための、vLLM の指標の接頭辞。"""

_METRIC_RUNNING: Final[str] = "vllm:num_requests_running"
_METRIC_WAITING: Final[str] = "vllm:num_requests_waiting"
"""`ServiceStatus` に入れる、処理中と待ちの要求の数の指標の名前。"""

_STATE_FORMAT: Final[str] = "{{.State.Status}} {{.State.ExitCode}}"
"""`docker container inspect` に渡す書式 (状態と終了コードを、1 行で読む)。"""

_STATE_ABSENT: Final[str] = "absent"
"""コンテナが見つからなくなったことを表す、この module の中だけの状態。"""

_UNFINISHED_STATES: Final[frozenset[str]] = frozenset(
    {"created", "running", "restarting", "paused", "removing"}
)
"""まだ終わっていないコンテナの状態。ここにない状態 (`exited`、`dead`) は、終わりとみなす。"""

_STATE_RUNNING: Final[str] = "running"
"""`docker ps --format json` の `State` が、動いていることを表す値。"""

_NAME_CONFLICT_MARKS: Final[tuple[str, ...]] = ("conflict.", "is already in use")
"""名前の衝突らしい標準エラーの文面 (**誤りの文を親切にするためだけ**に見る。task 3.1)。"""

_CLEANUP_LABELS: Final[Mapping[str, str]] = {"stop": "止められなかった", "rm": "消せなかった"}
"""片付けの失敗を言うときの、docker のサブコマンドごとの言い方。"""

_LABEL_TIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%SZ"
"""ラベル `vllm-baseline.started-at` の時刻の形 (`plan` が書く形と同じ。試験が一致を固定する)。"""

_STOP_HINT: Final[str] = "`serve stop` で止めてから、もう一度打つ"

_SERVE_STOP_HINT: Final[str] = (
    "コンテナが残っているかもしれないので、落ち着いてから `serve stop` で片付ける"
)
"""片付けに行けなかったときに、計測者に促す文。"""

_INTERRUPTED_TAIL: Final[str] = "(中断されたので、記録の末尾を読めなかった)"
"""中断で読めなかった末尾の代わりに入れる文 (呼ぶ側が、つねに台のぶんを受け取れるように)。"""


class LifecycleError(Exception):
    """起動を実行して失敗した (design.md 「Error Handling」の終了コード 2)。

    起こせなかった、名前が衝突した、了承のあとに読み取りが届かなかった、片付けられなかった、
    のどれかである。前提の不足 (関門の不通過、了承されなかったこと) は、ここに来ない。
    """


def _default_report() -> TextIO:
    """知らせの既定の出し先 (組み立てたときの `sys.stderr`)。"""
    return sys.stderr


@dataclass(frozen=True)
class Controls:
    """差し替えられる口と、時間切れ (公開の口。4.1 も、これを組み立てて使う)。

    時計と眠りと知らせの出し先と HTTP のクライアントを引数から受けることで、試験は、実際に
    眠らず、実物の推論サーバーにもつながずに済む。

    要るのは 2 つ (`client` と `timeout_s`) だけで、残りは既定の値を持つので、
    `Controls(client=new_client(), timeout_s=float(config.ready_timeout_s))` で組み立てられる。
    `client` を閉じるのは、組み立てた側の仕事である。
    """

    client: httpx.Client
    timeout_s: float
    """受け付けの開始を待つ上限 (構成の `ready_timeout_s`、またはその回の上書き)。"""

    poll_interval_s: float = READY_POLL_INTERVAL_S
    start_timeout_s: float = START_TIMEOUT_S
    read_timeout_s: float = READ_TIMEOUT_S
    log_timeout_s: float = LOG_READ_TIMEOUT_S
    tail_lines: int = DEFAULT_TAIL_LINES
    """計測者に**見せる**末尾の行数 (分類に使う長さは `OBSERVE_TAIL_LINES` で固定)。"""

    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    report: TextIO = field(default_factory=_default_report)


@dataclass(frozen=True)
class Readiness:
    """1 回ぶんの、受け付けの開始の判定 (design.md 「lifecycle」の (1)〜(3))。"""

    detail: str
    ready: bool = False
    fatal: str | None = None
    """待っても直らない食い違い (決めごとの 4)。時間切れを待たずに失敗にする。"""
    health_ok: bool = False
    served_model: str | None = None
    max_model_len: int | None = None
    running_requests: int | None = None
    waiting_requests: int | None = None


@dataclass(frozen=True)
class Target:
    """起こしたコンテナと、その台 (対象は、一覧から来た識別子だけ)。"""

    node: NodeDef
    plan: ContainerPlan
    container: OwnContainer


@dataclass(frozen=True)
class Waited:
    """待ちの結末 (公開の口)。`failure` が空なら、受け付けが始まっている。

    `failure` には、時間切れ・コンテナの終了・待っても直らない食い違いのどれかの文が入る。
    `exited` は、終了していたコンテナの台である (`observation` が、`UNCLASSIFIED` を付けるかの
    判断に使う)。
    """

    readiness: Readiness
    exited: tuple[NodeRole, ...] = ()
    failure: str | None = None


@dataclass(frozen=True)
class WrapUp:
    """片付けの結果 (公開の口。`wrap_up` が返す)。

    - `shown`: 計測者に見せる末尾 (`Controls.tail_lines` 行)
    - `classified`: 失敗の種類の分類に使う末尾 (`OBSERVE_TAIL_LINES` 行。`classify=False` の
      ときと、中断で読めなかったときは、`shown` と同じか、その旨の文)
    - `log_dir`: 記録を回収した置き場所 (回収を始めていなければ空)
    - `problems`: 片付けられなかったこと (確かめられなかった台の理由を含む)
    - `interrupted`: 片付けの途中で受けた中断 (**呼ぶ側が、これを投げて終了コード 130 にする**)
    """

    shown: dict[NodeRole, str]
    classified: dict[NodeRole, str]
    log_dir: Path | None
    problems: tuple[str, ...]
    interrupted: KeyboardInterrupt | None


class RunFailed(Exception):
    """起こす途中の失敗 (この module の中だけ)。

    片付けを済ませてから `LifecycleError` に変える。外には出ない。
    """


# --- 小さな助け -----------------------------------------------------------


def _node_of(nodes: Mapping[NodeRole, NodeDef], config: ConfigDef, role: NodeRole) -> NodeDef:
    """構成が使う役割のノードの定義を取る (なければ、触る前に断る)。"""
    node = nodes.get(role)
    if node is None:
        raise ValueError(f"nodes.{role} の定義がない (構成 '{config.name}' が使う役割)")
    return node


def _shown_failure(result: CommandResult) -> str:
    """失敗した呼び出しの、見せる文 (標準エラーがなければ標準出力)。"""
    return result.stderr.strip() or result.stdout.strip() or "(出力なし)"


def _refused(gates: Sequence[GateResult]) -> tuple[GateResult, ...]:
    return tuple(gate for gate in gates if not gate.passed)


def _refusal_detail(refused: Sequence[GateResult]) -> str:
    """断った関門を、台と関門の名前つきで並べる。"""
    return "関門が断った: " + " / ".join(
        f"{gate.node or '-'} の {gate.gate}: {gate.detail}" for gate in refused
    )


def _also_failed(problems: Sequence[str]) -> str:
    """片付けの失敗を、ほかの誤りの文に添える。"""
    return "" if not problems else " 片付けも終わらなかった: " + " / ".join(problems)


def _report(stream: TextIO, text: str) -> None:
    """進捗と警告を知らせる (標準出力は、後の処理が読むので混ぜない)。"""
    stream.write(f"{text}\n")
    stream.flush()


def label_time(raw: str | None) -> datetime | None:
    """ラベル `vllm-baseline.started-at` の時刻を読む (読めなければ空にする)。

    書く側は `plan`、読む側はここで、同じ形 (`%Y-%m-%dT%H:%M:%SZ`、UTC) を使う。
    """
    if raw is None:
        return None
    try:
        return datetime.strptime(raw, _LABEL_TIME_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None


def http_port(config: ConfigDef) -> int:
    """推論サーバーが待ち受ける HTTP のポートを、構成から決める (決めごとの 2)。

    **決め方**: 構成の `args` の中で、フラグが `--port` の設定の値である。待ち受けの印
    (`is_port`) だけでは選べない。design.md Data Models の見本では、`--port 8000` (HTTP の口)
    と `--master-port 29501` (分散の初期化の口) の**両方**に印が付いていて、印は
    `gate_ports_free` が「その番号が空いているか」を見るためのものだからである。`--port` は、
    vLLM の `vllm serve` の HTTP の口そのものなので、これを唯一の手がかりにする。

    head に掛からない設定 (`only_on = "worker"`) は、head の HTTP の口ではないので数えない。

    例外:
        config.ConfigError: `--port` の設定が 1 つもない、または 2 つ以上あるとき
            (終了コード 1)。
    """
    found = [
        setting.value
        for setting in config.args.values()
        if setting.flag == HTTP_PORT_FLAG
        and setting.value is not None
        and setting.only_on in (None, _HEAD)
    ]
    if len(found) != 1:
        ports = ", ".join(
            f"{key} ({setting.flag or '(位置の引数)'})"
            for key, setting in config.args.items()
            if setting.is_port
        )
        raise ConfigError(
            f"構成 '{config.name}' の args から、推論サーバーの HTTP のポート"
            f" ({HTTP_PORT_FLAG} の設定) を 1 つに決められない (見つかった数: {len(found)})。"
            f"待ち受けの印の付いた設定: {ports or '(ない)'}"
        )
    return int(found[0])


def new_client(timeout_s: float = HTTP_TIMEOUT_S) -> httpx.Client:
    """受け付けの判定に使う HTTP のクライアント。

    `trust_env=False` で、Mac の側のプロキシや `.netrc` の設定を拾わない (宛先は、head の
    LAN のアドレスだけである)。転送は追わない (推論サーバーは転送を返さない)。
    """
    return httpx.Client(
        timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 5.0)),
        trust_env=False,
        follow_redirects=False,
    )


# --- HTTP の読み取り (入出力はするが、何も書かない) -----------------------


def _fetch(client: httpx.Client, url: str) -> tuple[httpx.Response | None, str]:
    """1 つの道筋を読む。200 でなければ、読めなかったこととして理由を返す。"""
    try:
        response = client.get(url)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        return None, f"{url} に届かない ({exc})"
    if response.status_code != httpx.codes.OK:
        return None, f"{url} が {response.status_code} を返した"
    return response, ""


def _parse_models(response: httpx.Response) -> tuple[str | None, int | None, str]:
    """`/v1/models` から、名乗るモデルの名前と、入力の長さの上限を読む。

    返り値:
        名前、上限、読めなかったときの理由 (読めたら空)。
    """
    try:
        payload: object = response.json()
    except ValueError as exc:
        return None, None, f"/v1/models を JSON として読めない ({exc})"
    if not isinstance(payload, dict):
        return None, None, "/v1/models の応答が JSON の object でない"
    data = payload.get("data")
    if not isinstance(data, list):
        kind = "ない" if data is None else "配列でない"
        return None, None, f"/v1/models の data が{kind}"
    if not data:
        return None, None, "/v1/models の data がまだ空である"
    first = data[0]
    if not isinstance(first, dict):
        return None, None, "/v1/models の data[0] が JSON の object でない"
    served = first.get("id")
    if not isinstance(served, str) or not served:
        return None, None, "/v1/models の data[0].id がない"
    raw_length = first.get("max_model_len")
    readable = isinstance(raw_length, int) and not isinstance(raw_length, bool)
    return served, (raw_length if readable else None), ""


def _metric_value(text: str, name: str) -> int | None:
    """Prometheus のテキストから、1 つの指標の値を読む (読めなければ空)。"""
    for line in text.splitlines():
        if not line.startswith(name) or line.startswith("#"):
            continue
        rest = line[len(name) :]
        if rest[:1] not in ("{", " "):
            continue  # 名前が前方一致しただけの、別の指標
        fields = line.rsplit(maxsplit=1)
        if len(fields) != 2:
            continue
        try:
            return int(float(fields[1]))
        except ValueError:
            return None
    return None


def _speculative_lines(text: str) -> tuple[str, ...]:
    """投機的デコードの指標の行 (`# HELP` などの注釈の行は、接頭辞に当たらない)。"""
    return tuple(line for line in text.splitlines() if line.startswith(SPEC_DECODE_METRIC_PREFIX))


def check_ready(client: httpx.Client, base_url: str, served_model_name: str) -> Readiness:
    """受け付けの開始を、1 回ぶん判定する (design.md 「lifecycle」の (1)〜(3))。

    **読めないこと**は、起動の途中でありうるので、待ち続ける理由にする。**読めて、内容が
    構成と食い違うこと**は、待っても直らないので `fatal` にする (決めごとの 4)。
    """
    _, reason = _fetch(client, f"{base_url}/health")
    if reason:
        return Readiness(detail=f"応答の確認が、まだ通らない ({reason})")
    models, reason = _fetch(client, f"{base_url}/v1/models")
    if models is None:
        return Readiness(health_ok=True, detail=f"名乗るモデルの名前を、まだ読めない ({reason})")
    served, length, problem = _parse_models(models)
    if problem or served is None:
        return Readiness(health_ok=True, detail=f"名乗るモデルの名前を、まだ読めない ({problem})")
    if served != served_model_name:
        return Readiness(
            health_ok=True,
            served_model=served,
            max_model_len=length,
            detail="名乗るモデルの名前が、構成と違う",
            fatal=(
                f"推論サーバーが名乗るモデルの名前 ('{served}') が、構成の served_model_name"
                f" ('{served_model_name}') と違う。待っても変わらないので、時間切れを待たずに"
                "失敗にする (requirements 6.2)"
            ),
        )
    metrics, reason = _fetch(client, f"{base_url}/metrics")
    if metrics is None:
        return Readiness(
            health_ok=True,
            served_model=served,
            max_model_len=length,
            detail=f"指標を、まだ読めない ({reason})",
        )
    text = metrics.text
    if not any(line.startswith(_METRIC_PREFIX) for line in text.splitlines()):
        return Readiness(
            health_ok=True,
            served_model=served,
            max_model_len=length,
            detail=f"指標の本文に、{_METRIC_PREFIX} で始まる行がまだない",
        )
    speculative = _speculative_lines(text)
    if speculative:
        return Readiness(
            health_ok=True,
            served_model=served,
            max_model_len=length,
            detail="投機的デコードの指標が出ている",
            fatal=(
                f"投機的デコードの指標が出ている ({', '.join(speculative[:3])})。最初の起動は、"
                "投機的デコードを使わない構成にする (requirements 6.7)。待っても変わらないので、"
                "時間切れを待たずに失敗にする"
            ),
        )
    return Readiness(
        ready=True,
        health_ok=True,
        served_model=served,
        max_model_len=length,
        running_requests=_metric_value(text, _METRIC_RUNNING),
        waiting_requests=_metric_value(text, _METRIC_WAITING),
        detail=f"要求を受け付けられる ({served})",
    )


def _read_version(client: httpx.Client, base_url: str) -> str | None:
    """`/version` から、推論サーバーの版を読む (読めなければ空。決めごとの 7)。"""
    response, _ = _fetch(client, f"{base_url}/version")
    if response is None:
        return None
    try:
        payload: object = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    version = payload.get("version")
    return version if isinstance(version, str) and version else None


# --- 起動の記録 -----------------------------------------------------------


def _check_record_dir(record_dir: Path) -> None:
    """`record_dir` が絶対の道筋であることを確かめる (Spark に触る前に断る)。

    配る元 (`<record_dir>/launch/<役割>/`) は、配る前に空にする。消す操作なので、相対の道筋
    (空の道筋を含む) を受けると、プロセスの作業ディレクトリに対して効いてしまう
    (task 3.3 の `weights._check_record_dir` と同じ決まり)。
    """
    if not record_dir.is_absolute():
        raise ValueError(
            f"record_dir は、絶対の道筋で渡す (配る元を、配る前に空にするため): '{record_dir}'"
        )


def _record_source_dir(record_dir: Path, role: NodeRole) -> Path:
    """起動の記録を配る元 (**この module だけが使う場所**。決めごとの 10)。

    `remote.push` は、渡したディレクトリの中身を丸ごと送るので、呼ぶ側が渡した `record_dir`
    そのものを元にしない (5.1 が `logs.var_dir` を渡すと、回収した記録が Spark の `state/` に
    逆流する)。
    """
    return record_dir / _RECORD_DIRNAME / role


def record_pushes(config: ConfigDef, record_dir: Path) -> tuple[PlannedPush, ...]:
    """起動の記録を配る計画 (宛先は `state/`。`--delete` は付けない)。

    `record_dir` は、絶対の道筋で渡す (`push_launch_records` が、配る元を、配る前に空にする)。
    """
    _check_record_dir(record_dir)
    return tuple(
        PlannedPush(
            node=role,
            local_dir=_record_source_dir(record_dir, role),
            remote_subdir=LAUNCH_RECORD_SUBDIR,
            delete=False,
            purpose=f"{role} に、起動の記録を置く (起こす直前)",
        )
        for role in config.nodes
    )


def launch_record(
    config: ConfigDef,
    plans: Sequence[ContainerPlan],
    started_at: datetime,
    *,
    repo_commit: str,
    repo_dirty: bool,
) -> LaunchRecord:
    """起動の記録を Mac で作る (requirements 3.10)。

    リポジトリの commit と、未コミットの変更の有無は**引数で受ける**。この module の中で
    `git` を呼ばない (Spark に触る道具が、Mac のリポジトリの状態を自分で調べに行かない)。
    """
    labels = plans[0].labels
    weights = config.weights
    return LaunchRecord(
        config_name=config.name,
        image_digest=labels[LABEL_IMAGE],
        weights=None if weights is None else f"{weights.repo}@{weights.revision}",
        started_at=started_at,
        plans=tuple(plans),
        config_sha256=labels[LABEL_CONFIG_SHA256],
        repo_commit=repo_commit,
        repo_dirty=repo_dirty,
    )


def _record_bytes(record: LaunchRecord) -> bytes:
    """記録を、同じ入力なら同じバイト列になる形で書き出す。"""
    text = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
    return (text + "\n").encode("utf-8")


def push_launch_records(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    record: LaunchRecord,
    record_dir: Path,
) -> None:
    """起動の記録を、2 台の `state/<構成>.launch.json` に置く。

    配布は `push(delete=False)` で、**了承済みの計画に入っていなければ `remote` が断る**。
    配る元は、この module だけが使う場所にして、配る前に空にする (よそのファイルを Spark の
    `state/` に入れない)。消す操作なので、`record_dir` は、絶対の道筋だけを受ける (相対の
    道筋だと、作業ディレクトリに対して効く。4.1 が、この公開の口を直に呼ぶときも同じ)。
    """
    _check_record_dir(record_dir)
    remote_path = PurePosixPath(launch_record_remote_path(config.name))
    if str(remote_path.parent) != LAUNCH_RECORD_SUBDIR:
        raise LifecycleError(
            f"起動の記録の道筋 ({remote_path}) が、配れる宛先 ({LAUNCH_RECORD_SUBDIR}/) の下に"
            "ない (logs.launch_record_remote_path と、この module の決まりが食い違っている)"
        )
    payload = _record_bytes(record)
    for role in config.nodes:
        node = _node_of(nodes, config, role)
        local_dir = _record_source_dir(record_dir, role)
        if local_dir.is_symlink():
            # `rmtree` はシンボリックリンクを消さないので、リンク先のよそのファイルが、配る元に
            # 現れて、Spark の `state/` に配られてしまう
            raise LifecycleError(
                f"起動の記録を配る元 ({local_dir}) が、シンボリックリンクになっている"
                " (この場所は、この道具だけが使う。リンクを外してから、やり直す)"
            )
        shutil.rmtree(local_dir, ignore_errors=True)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / remote_path.name).write_bytes(payload)
        try:
            result = runner.push(node, local_dir, LAUNCH_RECORD_SUBDIR, delete=False)
        except RemoteError as exc:
            raise LifecycleError(
                f"{role} ({node.ssh_host}) に、起動の記録を置けなかった"
                f" ({node.remote_root}/{remote_path}): {exc}"
            ) from exc
        if result.exit_code != 0:
            raise LifecycleError(
                f"{role} ({node.ssh_host}) に、起動の記録を置けなかった"
                f" ({node.remote_root}/{remote_path}): {_shown_failure(result)}"
            )


# --- 起こす ---------------------------------------------------------------


def start_all(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    controls: Controls,
) -> None:
    """2 台を続けて起こす (`docker run -d`)。

    design.md の System Flows は、起こす順序を定めていない (止めるときだけ head → worker)。
    ここでは、構成の `nodes` に書いた順に流す。切り離して起こすので、2 台の立ち上がりは、
    そのまま並行に進む。

    **`docker run` の標準エラーの文面で、名前の衝突を判定しない** (task 3.1 の決まり)。文面は
    docker の版で変わりうるので、止めてよい相手は、つねに `guards.rollback_commands` が、自分の
    ラベルで絞った一覧で決める。文面は、誤りの文を親切にするためだけに見る。
    """
    for plan in plans:
        node = _node_of(nodes, config, plan.node)
        started = runner.run(node, plan.argv, timeout_s=controls.start_timeout_s, mutating=True)
        if started.exit_code != 0:
            raise RunFailed(_start_failure_detail(node, plan, started))


def _start_failure_detail(node: NodeDef, plan: ContainerPlan, started: CommandResult) -> str:
    """`docker run` が失敗したときの、見せる文。"""
    lowered = started.stderr.lower()
    if all(mark in lowered for mark in _NAME_CONFLICT_MARKS):
        reason = (
            f"コンテナの名前が衝突した ({plan.container_name})。その名前のコンテナが、自分の"
            "ラベルで絞った一覧になければ、止めも消しもしない (自分のものでないコンテナを、"
            "名前で止めないため)"
        )
    else:
        reason = f"推論サーバーのコンテナを起こせなかった ({plan.container_name})"
    return f"{node.role} ({node.ssh_host}) で、{reason}: {_shown_failure(started)}"


def started_targets(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    controls: Controls,
) -> tuple[Target, ...]:
    """起こしたコンテナの識別子を、自分のラベルで絞った一覧から取る (不変条件の (a))。

    `docker run` が返す識別子は使わない (一覧を通った識別子だけを、あとの操作の対象にする)。
    """
    targets: list[Target] = []
    for plan in plans:
        node = _node_of(nodes, config, plan.node)
        try:
            containers = list_own_containers(runner, node, timeout_s=controls.read_timeout_s)
        except ValueError as exc:
            raise RunFailed(
                f"{plan.node} ({node.ssh_host}) で、起こした推論サーバーのコンテナを探せなかった"
                f" ({plan.container_name}): {exc}"
            ) from exc
        found = next(
            (item for item in containers if item.name == plan.container_name),
            None,
        )
        if found is None:
            raise RunFailed(
                f"{plan.node} ({node.ssh_host}) で、起こしたはずの推論サーバーのコンテナが、"
                f"自分の一覧にない ({plan.container_name})"
            )
        targets.append(Target(node=node, plan=plan, container=found))
    return tuple(targets)


# --- 待つ -----------------------------------------------------------------


def _container_state(
    runner: RemoteRunner, target: Target, timeout_s: float
) -> tuple[str, int | None]:
    """コンテナの状態と終了コードを読む (対象は、一覧から来た識別子だけ)。

    読み取りが届かなかったこと (`RemoteError`) は、そのまま外に出す (呼ぶ側が、片付けてから
    実行しての失敗にする)。**読めなかったことを「まだ動いている」に倒さない**。
    """
    argv = ("docker", "container", "inspect", "--format", _STATE_FORMAT, target.container.id)
    result = runner.run(target.node, argv, timeout_s=timeout_s, mutating=False)
    if result.exit_code != 0:
        return _STATE_ABSENT, None
    fields = result.stdout.split()
    if not fields:
        return _STATE_ABSENT, None
    code = fields[1] if len(fields) > 1 else ""
    return fields[0], int(code) if code.lstrip("-").isdecimal() else None


def _exited_detail(target: Target, status: str, exit_code: int | None) -> str:
    code = "不明" if exit_code is None else str(exit_code)
    return (
        f"{target.plan.node} ({target.node.ssh_host}) の {target.plan.container_name} が、"
        f"終了した (状態 {status}、終了コード {code})"
    )


def wait_ready(
    runner: RemoteRunner,
    targets: Sequence[Target],
    controls: Controls,
    *,
    base_url: str,
    served_model_name: str,
) -> Waited:
    """受け付けの開始を待つ (design.md 「lifecycle」の受け付けの開始の判定)。

    1 周ごとに、HTTP の 3 つを見てから、2 台のコンテナの状態を見る (決めごとの 3)。どちらかの
    コンテナが終了していたら、時間切れを待たずに失敗にする。時計と眠りは引数から来るので、
    試験は実際に眠らない。
    """
    deadline = controls.clock() + controls.timeout_s
    while True:
        readiness = check_ready(controls.client, base_url, served_model_name)
        if readiness.fatal is not None:
            return Waited(readiness=readiness, failure=readiness.fatal)
        if readiness.ready:
            return Waited(readiness=readiness)
        exited: list[str] = []
        roles: list[NodeRole] = []
        for target in targets:
            status, exit_code = _container_state(runner, target, controls.read_timeout_s)
            if status not in _UNFINISHED_STATES:
                exited.append(_exited_detail(target, status, exit_code))
                roles.append(target.plan.node)
        if exited:
            return Waited(
                readiness=readiness,
                exited=tuple(roles),
                failure=(
                    "待っている間に、推論サーバーのコンテナが終了したので、時間切れを待たずに"
                    f"失敗にする: {' / '.join(exited)}"
                ),
            )
        if controls.clock() >= deadline:
            return Waited(
                readiness=readiness,
                failure=(
                    f"{controls.timeout_s:.0f} 秒のうちに、要求を受け付けられる状態にならなかった"
                    f" (最後に見たこと: {readiness.detail})"
                ),
            )
        controls.sleep(controls.poll_interval_s)


# --- 片付け ---------------------------------------------------------------


def clean_up(
    runner: RemoteRunner,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    controls: Controls,
    *,
    interrupted: KeyboardInterrupt | None = None,
) -> tuple[tuple[str, ...], KeyboardInterrupt | None]:
    """自分のコンテナを、台のすべてで止めて消す (task 3.1 の巻き戻しの決まりに従う)。

    **前提**: 了承済みの計画が、実行役に渡っていること (`guards.request_approval`)。流す列は
    `guards.rollback_commands` が `guards.stop_argv` / `guards.remove_argv` で作るので、了承を
    得た計画の巻き戻しと完全に一致する (`remote.CallGuard` は、完全な一致で見る)。

    **保証**:

    1. 相手を決めるのは `guards.rollback_commands` で、流す直前に、自分のラベルで絞った一覧に
       その名前の行があることを確かめる (`docker stop <名前>` は、その名前のコンテナが誰のもの
       でも止めるので、名前だけを頼りにしない。requirements 2.3)。確かめられなかった台の理由
       (`serve stop` を促す文) は、そのまま `problems` に入る
    2. **一覧の読み取りで中断が来たら、`stop` も `rm` も 1 つも流さない**。一覧が読めていない
       ので、その名前が自分のコンテナだと言えない (requirements 2.3)。`serve stop` を促す文を
       `problems` と `report` に出して、中断を返す
    3. **中断が何度来ても、`stop` → `rm` の両方を、台のすべてで試みる**。`docker rm` (強制なし)
       は、動いているコンテナを消せず、`-f` は了承済みの計画にも `remote` の許可の一覧の決まり
       にもないので、`stop` を飛ばして `rm` だけ流すと、コンテナが残ってしまう。だから、中断の
       あとでも `stop` を飛ばさない
    4. **同じ呼び出しを、やり直さない** (無限に粘らない)。1 つの命令に 1 回だけ挑み、中断は
       最初の 1 つを覚えて、残りの命令に進む
    5. **急ぐことはできない**。流せるのは、了承済みの計画にある列だけなので、`stop` の猶予
       (`-t 90`) を短くする列は作れない。引数で受けた `interrupted` は、片付けの内容を変えず、
       「もう中断が来ている」ことを覚えて、最後に返すためだけに使う

    引数:
        interrupted: 片付けを始める前に、すでに受けている中断 (`wrap_up` が渡す)。

    返り値:
        片付けられなかったことと、受けた中断 (**投げるのは、呼ぶ側**)。
    """
    problems: list[str] = []
    try:
        commands, unchecked = rollback_commands(
            runner, nodes, plans, timeout_s=controls.read_timeout_s
        )
    except KeyboardInterrupt as exc:
        # 一覧が読めていないので、名前だけで止めてはならない (保証の 2)
        problems.append(
            "自分のコンテナの一覧の読み取りが中断されたので、片付けに行かない"
            f" (名前だけで止めると、よそのコンテナを止めうる)。{_SERVE_STOP_HINT}"
        )
        _report(controls.report, "警告: " + " / ".join(problems))
        return tuple(problems), interrupted or exc
    problems.extend(unchecked)
    for command in commands:
        what = _CLEANUP_LABELS[command.argv[1]]
        node = nodes[command.node]
        try:
            result = runner.run(node, command.argv, timeout_s=CLEANUP_TIMEOUT_S, mutating=True)
        except KeyboardInterrupt as exc:
            # やり直さずに、次の命令に進む (保証の 3、4)
            interrupted = interrupted or exc
            problems.append(f"{command.container}: {what} (中断された。残りの片付けは続ける)")
            continue
        except RemoteError as exc:
            problems.append(f"{command.container}: {what} (読み取りが届かなかった): {exc}")
            continue
        if result.exit_code != 0:
            problems.append(f"{command.container}: {what}: {_shown_failure(result)}")
    if problems:
        _report(
            controls.report,
            "警告: 推論サーバーのコンテナの片付けが、すべては終わらなかった: "
            + " / ".join(problems),
        )
    return tuple(problems), interrupted


def _guarded_tails(
    runner: RemoteRunner,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    controls: Controls,
    *,
    lines: int,
) -> tuple[dict[NodeRole, str], KeyboardInterrupt | None]:
    """記録の末尾を読む。中断が来たら、読めなかった旨の文で埋めて、中断を返す。

    `logs.tail_logs` は、入れない・コンテナがない・記録を読めないのどれでも例外を出さないが、
    **中断 (Ctrl-C) は素通しする**。素通しさせると、片付け (`docker stop` / `rm`) に到達せず、
    起きかけのコンテナが 2 台とも残るので、ここで受け止める (レビューの指摘 1)。
    """
    try:
        return tail_logs(runner, nodes, plans, lines=lines, timeout_s=controls.read_timeout_s), None
    except KeyboardInterrupt as exc:
        return {plan.node: _INTERRUPTED_TAIL for plan in plans}, exc


def wrap_up(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    controls: Controls,
    *,
    var_root: Path,
    started_at: datetime,
    classify: bool,
) -> WrapUp:
    """記録を読んで回収し、コンテナを片付ける (requirements 1.5)。

    順序は「**見せる**末尾 → (要るなら) **分類に使う**末尾 → 回収 → 停止 → 削除」。`docker rm`
    を回収のあとに置くのは、記録が消えないようにするためである (design.md の System Flows の
    失敗の分岐)。末尾を 2 通り読む理由は、`OBSERVE_TAIL_LINES` の説明にある (決めごとの 5)。

    **前提**: 了承済みの計画が、実行役に渡っていること (片付けは、その巻き戻しとして流す)。

    **保証**: 3 つの読み取り (見せる末尾、分類の末尾、記録の回収) は、どれも `KeyboardInterrupt`
    から守られる。中断が来たら、最初の 1 つを覚え、**残りの読み取りを飛ばして**、必ず
    `clean_up` に到達する (読めなかった末尾は、その旨の文で埋める)。読み取りは、2 台ぶんで
    13 回の遠隔の呼び出しになり、記録の全量の `docker logs` は時間切れ 300 秒なので、ここを
    守らないと、待ちを中断した Ctrl-C のあと、コンテナが 2 台とも残ってしまう
    (レビューの指摘 1、2)。**受けた中断は投げずに、`WrapUp.interrupted` で返す** (呼ぶ側が、
    見せるものを見せてから投げて、終了コード 130 にする)。

    成功のあとにも呼べる (4.1 の縮小の確認は、どの結果でも記録を回収して片付ける)。

    引数:
        classify: 失敗の種類の分類に使う、長い末尾を読むかどうか。結果を使わない経路
            (中断、実行しての失敗) では偽にして、片付けを遅らせない (レビューの指摘 4)。
    """
    shown, interrupted = _guarded_tails(runner, nodes, plans, controls, lines=controls.tail_lines)
    classified = shown
    problems: list[str] = []
    if interrupted is None and classify:
        classified, interrupted = _guarded_tails(
            runner, nodes, plans, controls, lines=OBSERVE_TAIL_LINES
        )
    log_dir: Path | None = None
    if interrupted is None:
        log_dir = var_dir(var_root, started_at, COMMAND_NAME, config.name)
        try:
            collect_logs(
                runner,
                nodes,
                plans,
                var_root=var_root,
                started_at=started_at,
                command=COMMAND_NAME,
                config_name=config.name,
                timeout_s=controls.log_timeout_s,
            )
        except KeyboardInterrupt as exc:
            # 途中まで写したものは、その置き場所に残る (道筋は `logs.var_dir` が決める)
            interrupted = exc
            problems.append(f"記録の回収が中断されたので、{log_dir} には一部だけがある")
    else:
        problems.append("中断されたので、記録を回収せずに、コンテナの片付けに進んだ")
    cleaned, interrupted = clean_up(runner, nodes, plans, controls, interrupted=interrupted)
    return WrapUp(
        shown=shown,
        classified=classified,
        log_dir=log_dir,
        problems=(*problems, *cleaned),
        interrupted=interrupted,
    )


def _show_tails(stream: TextIO, tails: Mapping[NodeRole, str]) -> None:
    """2 台の記録の末尾を見せる (例外の経路では、結果に入れて返せないため)。"""
    for role, text in tails.items():
        _report(stream, f"--- {role} の記録の末尾 ---\n{text}")


def _shown_log_dir(done: WrapUp) -> str:
    """記録の置き場所を、見せる文にする。"""
    if done.log_dir is None:
        return "記録は回収していない。"
    return f"記録は {done.log_dir} に回収した。"


def _report_log_dir(stream: TextIO, done: WrapUp) -> None:
    """回収した記録の置き場所を知らせる (中断のあとでも、計測者が場所を知れるように)。"""
    if done.log_dir is not None:
        _report(stream, f"記録は {done.log_dir} に回収した")


# --- 読み取った事実 -------------------------------------------------------


def observation(
    config: ConfigDef, tails: Mapping[NodeRole, str], exited: Sequence[NodeRole]
) -> LaunchObservation | None:
    """記録の末尾から、失敗の種類と、選ばれた部品などを読む (決めごとの 5)。

    渡す `tails` は、**分類に使う長い末尾** (`OBSERVE_TAIL_LINES`) で、構成が使う台のすべての
    ぶんである (head と worker の、どちらの記録に知っている失敗が出ても拾う)。

    どの台の読み取りを返すかの決まり:

    1. 知っている失敗が読めた台のうち、`KnownFailure` の優先の順序 (`types.KnownFailure` の
       宣言の順で、`observe` が判定に使う順序と同じ) で**先のもの**。同じ種類が 2 台に出た
       ときは、構成の `nodes` の順で先の台 (head)
    2. どの台にも知っている失敗がなく、終了した台があれば、その 1 つ目に `UNCLASSIFIED` を
       付ける (`observe_launch` は「終了したか」を知らないので、ここで付ける)
    3. どれでもなければ (時間切れなど)、構成の 1 台目の読み取りをそのまま返す
    """
    seen: dict[NodeRole, LaunchObservation] = {
        role: observe_launch(text) for role, text in tails.items()
    }
    ranked: list[tuple[int, int, NodeRole]] = []
    for order, role in enumerate(config.nodes):
        found = seen.get(role)
        if found is None or found.known_failure is None:
            continue
        ranked.append((_FAILURE_PRIORITY[found.known_failure], order, role))
    if ranked:
        return seen[min(ranked)[2]]
    for role in exited:
        found = seen.get(role)
        if found is not None:
            return found.model_copy(update={"known_failure": KnownFailure.UNCLASSIFIED})
    for role in config.nodes:
        found = seen.get(role)
        if found is not None:
            return found
    return None


def _node_status(
    config: ConfigDef, plan: ContainerPlan, container: OwnContainer | None
) -> NodeStatus:
    """1 台の状態を、自分のラベルで絞った一覧の行から作る。

    `gpu_apps` と `fabric_link_up` は、`start` が読まないので空にする (`serve status` が読む。
    決めごとの 9)。
    """
    if container is None:
        return NodeStatus(node=plan.node, container_state="absent")
    state: ContainerState = "running" if container.state == _STATE_RUNNING else "exited"
    return NodeStatus(
        node=plan.node,
        container_state=state,
        config_name=container.labels.get(LABEL_CONFIG),
        kind=config.kind,
        image_digest=container.labels.get(LABEL_IMAGE),
        config_sha256=container.labels.get(LABEL_CONFIG_SHA256),
        started_at=label_time(container.labels.get(LABEL_STARTED_AT)),
    )


def _service_status(
    config: ConfigDef,
    plans: Sequence[ContainerPlan],
    containers: Mapping[NodeRole, Sequence[OwnContainer]],
    readiness: Readiness | None = None,
) -> ServiceStatus:
    """2 台にまたがる 1 つの推論サーバーの状態を組み立てる。"""
    nodes = tuple(
        _node_status(
            config,
            plan,
            next(
                (
                    item
                    for item in containers.get(plan.node, ())
                    if item.name == plan.container_name
                ),
                None,
            ),
        )
        for plan in plans
    )
    if readiness is None:
        return ServiceStatus(nodes=nodes)
    return ServiceStatus(
        nodes=nodes,
        health_ok=readiness.health_ok,
        served_model=readiness.served_model,
        max_model_len=readiness.max_model_len,
        running_requests=readiness.running_requests,
        waiting_requests=readiness.waiting_requests,
    )


# --- 一覧の読み取りと、すでに動いているかの判定 ---------------------------


def _own_containers(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    *,
    read_timeout_s: float,
) -> dict[NodeRole, tuple[OwnContainer, ...]]:
    """構成が使う台のそれぞれで、自分のラベルで絞った一覧を 1 度だけ読む。

    この 1 回の読み取りを、`already_running` の判定と `gate_own_state` の両方で使う
    (2 度読んで食い違うことをなくす)。読めなかった台は、結果に入れない。その台の断りは、
    `guards.run_gates` が、自分で読み直して `GateResult` にする。
    """
    found: dict[NodeRole, tuple[OwnContainer, ...]] = {}
    for role in config.nodes:
        node = _node_of(nodes, config, role)
        try:
            found[role] = list_own_containers(runner, node, timeout_s=read_timeout_s)
        except (RemoteError, ValueError):
            continue
    return found


# --- 入口 -----------------------------------------------------------------


def start(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    started_at: datetime,
    *,
    confirmer: Confirmer,
    var_root: Path,
    record_dir: Path,
    repo_commit: str,
    repo_dirty: bool,
    manifest: WeightsManifest | None = None,
    timeout_s: float | None = None,
    poll_interval_s: float = READY_POLL_INTERVAL_S,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    report: TextIO | None = None,
    client: httpx.Client | None = None,
    http_timeout_s: float = HTTP_TIMEOUT_S,
    start_timeout_s: float = START_TIMEOUT_S,
    read_timeout_s: float = READ_TIMEOUT_S,
    log_timeout_s: float = LOG_READ_TIMEOUT_S,
    tail_lines: int = DEFAULT_TAIL_LINES,
) -> StartOutcome:
    """構成の推論サーバーを、2 台で起こして、受け付けの開始まで待つ (`serve start <構成>`)。

    進む順と、守る決まりは、module の docstring にある。失敗と中断のときは、2 台の記録の末尾を
    見せ、記録を回収してから、自分のコンテナを 2 台とも止めて消す (requirements 1.5)。

    引数:
        runner: 遠隔の実行役。
        config: `kind = "serve"` の構成 (`config.select_config` を通ったもの)。
        nodes: 役割ごとのノードの定義。
        started_at: 起こす時刻 (ラベルと起動の記録と、回収の置き場所の名前に使う。時差の
            付いた日時)。
        confirmer: 計画を見せて、了承を得る口。
        var_root: 記録の回収の置き場所の根 (`serving/var/`。`SshRunner` を作ったときと同じ値に
            すること。でなければ `remote.CallGuard` に宛先の外として断られる)。
        record_dir: 起動の記録を作る、Mac の側の置き場所 (絶対の道筋)。**配るのは、この下に、
            この module が作る `launch/<役割>/` だけ**で、配る前に空にする。
        repo_commit: リポジトリの commit (**呼ぶ側が調べて渡す**。この module は `git` を
            呼ばない)。
        repo_dirty: 未コミットの変更があるかどうか (同上)。
        manifest: 重みのマニフェスト (重みを使う構成では要る。`gate_weights_verified` が使う)。
        timeout_s: 受け付けの開始を待つ上限を、**その回だけ**上書きする (初回の起動は、ロードと
            JIT で長くかかりうる)。省くと、構成の `ready_timeout_s` を使う。
        poll_interval_s: 受け付けの開始を見に行く間隔。
        sleep: 眠る口 (試験は、実際に眠らないものを渡す)。既定は `time.sleep`。
        clock: 時計 (単調増加の秒)。既定は `time.monotonic`。
        report: 進捗と、例外の経路の記録の末尾を知らせる先。既定は `sys.stderr` (標準出力は、
            後の処理が読む行に使うので混ぜない)。
        client: HTTP のクライアント。省くと `new_client` で作り、この関数の中で閉じる。
        http_timeout_s: 1 回の HTTP の問い合わせの時間切れ。
        start_timeout_s: `docker run -d` の時間切れ。
        read_timeout_s: 関門と、一覧と、コンテナの状態の読み取りの時間切れ。
        log_timeout_s: 記録の全量の回収の時間切れ。
        tail_lines: 失敗のときに見せる、記録の末尾の行数。

    返り値:
        `ready` / `already_running` (終了コード 0)、`refused` (1)、`failed` (2)。

    例外:
        config.ConfigError: `kind` が `serve` でない構成、HTTP のポートを決められない構成
            (終了コード 1)。
        guards.ApprovalError: 計測者が了承しなかった (終了コード 1)。
        LifecycleError: 起こせなかった、名前が衝突した、了承のあとに読み取りが届かなかった、
            起動の記録を置けなかった、片付けられなかった (どれも、実行しての失敗で
            終了コード 2)。
        KeyboardInterrupt: 中断 (終了コード 130)。コンテナは、片付けてから伝える。
        ValueError: 構成が使う役割のノードの定義がない、`record_dir` が相対の道筋のとき。
    """
    if config.kind != _KIND_SERVE:
        raise ConfigError(
            f"推論サーバーの起動に使えるのは、kind = '{_KIND_SERVE}' の構成だけである"
            f" (構成 '{config.name}' の kind は '{config.kind}')"
        )
    if config.served_model_name is None:  # pragma: no cover - 型の検証が、先に断る
        raise ConfigError(f"構成 '{config.name}' に served_model_name がない")
    _check_record_dir(record_dir)
    port = http_port(config)
    head = _node_of(nodes, config, _HEAD)
    base_url = f"http://{head.lan_addr}:{port}"
    plans = build_plans(config, nodes, started_at)

    # 1. すでに動いているか (関門より前に、1 回の読み取りで判定する)
    containers = _own_containers(runner, config, nodes, read_timeout_s=read_timeout_s)
    match = match_running(plans, containers)
    if match.already_running:
        return StartOutcome(
            status="already_running",
            detail=(
                f"構成 '{config.name}' は、すでに {len(plans)} 台で動いている (名前、構成の名前、"
                "イメージのダイジェスト、config-sha256 が、すべて一致した)。関門も了承も"
                "通さずに、いまの状態を示して終わる"
            ),
            service=_service_status(config, plans, containers),
        )
    if match.differences:
        return StartOutcome(
            status="refused",
            detail=(
                "同じ名前のコンテナが動いているが、中身が選んだ構成と違う: "
                + " / ".join(match.differences)
                + f"。古い設定のまま計測しないように、起動しない。{_STOP_HINT}"
            ),
        )

    # 2. 関門 (読み取りだけ。1 つ落ちても、残りを流して並べる)
    gates = run_gates(
        runner,
        config,
        nodes,
        plans,
        manifest=manifest,
        containers=containers,
        timeout_s=read_timeout_s,
    )
    refused = _refused(gates)
    if refused:
        return StartOutcome(status="refused", gates=gates, detail=_refusal_detail(refused))

    # 3. 計画と了承 (ここより前に、状態を変える呼び出しは 1 つも出ていない)
    approved = build_approved_plan(plans, extra_forward=record_pushes(config, record_dir))
    request_approval(confirmer, runner, approved, nodes)

    owns_client = client is None
    controls = Controls(
        sleep=time.sleep if sleep is None else sleep,
        clock=time.monotonic if clock is None else clock,
        report=sys.stderr if report is None else report,
        client=new_client(http_timeout_s) if client is None else client,
        poll_interval_s=poll_interval_s,
        timeout_s=float(config.ready_timeout_s) if timeout_s is None else timeout_s,
        start_timeout_s=start_timeout_s,
        read_timeout_s=read_timeout_s,
        log_timeout_s=log_timeout_s,
        tail_lines=tail_lines,
    )
    try:
        return _launch(
            runner,
            config,
            nodes,
            plans,
            gates,
            controls,
            base_url=base_url,
            served_model_name=config.served_model_name,
            var_root=var_root,
            record_dir=record_dir,
            started_at=started_at,
            repo_commit=repo_commit,
            repo_dirty=repo_dirty,
        )
    finally:
        if owns_client:
            controls.client.close()


def _launch(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    gates: Sequence[GateResult],
    controls: Controls,
    *,
    base_url: str,
    served_model_name: str,
    var_root: Path,
    record_dir: Path,
    started_at: datetime,
    repo_commit: str,
    repo_dirty: bool,
) -> StartOutcome:
    """了承のあとの「置く → 起こす → 待つ → (片付ける)」。

    起こすのが失敗しても、**起こす呼び出しそのものが届かなくても** (ssh の時間切れ、切断)、
    中断されても、同じ 1 つの片付けの道を通る (task 3.1 の片付けの道の形)。遠隔の
    `docker run -d` は、ssh が切れたあとも完了しうるので、`docker run` の呼び出しも `try` の
    中に置く。

    起動の記録を置くのは `try` の**外**である (まだコンテナが 1 つもないので、片付けるものが
    ない)。置けなかったことは、そのまま `LifecycleError` になる。
    """
    record = launch_record(
        config, plans, started_at, repo_commit=repo_commit, repo_dirty=repo_dirty
    )
    push_launch_records(runner, config, nodes, record, record_dir)

    try:
        start_all(runner, config, nodes, plans, controls)
        targets = started_targets(runner, config, nodes, plans, controls)
        waited = wait_ready(
            runner,
            targets,
            controls,
            base_url=base_url,
            served_model_name=served_model_name,
        )
    except (RemoteError, RunFailed) as exc:
        # 分類の結果を使わない経路なので、長い末尾は読まない (レビューの指摘 4)
        done = wrap_up(
            runner,
            config,
            nodes,
            plans,
            controls,
            var_root=var_root,
            started_at=started_at,
            classify=False,
        )
        _show_tails(controls.report, done.shown)
        _report_log_dir(controls.report, done)
        if done.interrupted is not None:
            # 片付けの最中の中断は、実行しての失敗より優先して伝える (決めごとの 8。130)
            raise done.interrupted from exc
        reason = (
            str(exc)
            if isinstance(exc, RunFailed)
            else f"了承のあとに、読み取りが届かなかった: {exc}"
        )
        raise LifecycleError(
            f"{reason}。{_shown_log_dir(done)}{_also_failed(done.problems)}"
        ) from exc
    except BaseException as exc:
        # 中断 (Ctrl-C) と、思わぬ誤りのときも、半端に起きた推論サーバーを残さない
        done = wrap_up(
            runner,
            config,
            nodes,
            plans,
            controls,
            var_root=var_root,
            started_at=started_at,
            classify=False,
        )
        _show_tails(controls.report, done.shown)
        _report_log_dir(controls.report, done)
        if done.interrupted is not None and done.interrupted is not exc:
            # 待ちの中断 (exc) のあとに、片付けの最中にもう一度来た中断
            raise done.interrupted from exc
        raise

    if waited.failure is None:
        return _ready_outcome(
            runner, config, nodes, plans, targets, gates, controls, waited, base_url=base_url
        )

    done = wrap_up(
        runner,
        config,
        nodes,
        plans,
        controls,
        var_root=var_root,
        started_at=started_at,
        classify=True,
    )
    if done.interrupted is not None:
        _show_tails(controls.report, done.shown)
        _report_log_dir(controls.report, done)
        raise done.interrupted
    return StartOutcome(
        status="failed",
        detail=f"{waited.failure}。{_shown_log_dir(done)}"
        f"自分のコンテナを止めて消した{_also_failed(done.problems)}",
        gates=tuple(gates),
        # 分類には長い末尾を、見せるのには 80 行を使う (決めごとの 5)
        observation=observation(config, done.classified, waited.exited),
        log_tails=done.shown,
    )


def _ready_outcome(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    targets: Sequence[Target],
    gates: Sequence[GateResult],
    controls: Controls,
    waited: Waited,
    *,
    base_url: str,
) -> StartOutcome:
    """受け付けが始まったときの結果 (記録の全量は回収しない。決めごとの 6)。"""
    head = next((target for target in targets if target.plan.node == _HEAD), None)
    found: LaunchObservation | None = None
    if head is not None:
        tails = tail_logs(
            runner,
            nodes,
            (head.plan,),
            lines=OBSERVE_TAIL_LINES,
            timeout_s=controls.read_timeout_s,
        )
        found = observe_launch(tails[head.plan.node])
        version = _read_version(controls.client, base_url)
        if version is not None:
            found = found.model_copy(update={"vllm_version": version})
    containers = {target.plan.node: (target.container,) for target in targets}
    return StartOutcome(
        status="ready",
        detail=(
            f"構成 '{config.name}' の推論サーバーが、{len(plans)} 台で要求を受け付けられる"
            f"状態になった ({waited.readiness.detail})"
        ),
        gates=tuple(gates),
        service=_service_status(config, plans, containers, waited.readiness),
        observation=found,
    )
