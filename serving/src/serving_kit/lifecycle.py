"""起こす、待つ、確かめる、止める (design.md 「運転 › lifecycle」)。

受け持つのは、`serve start <構成>` (tasks.md 3.4) と、`serve status` / `serve stop` /
`serve smoke` (tasks.md 3.5) の中身である。

## 起動 (`serve start <構成>`)

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
   始まる行の有無が、構成の `allow_speculative` と一致する、を見る。上限は構成の
   `ready_timeout_s` で、`timeout_s` でその回だけ上書きできる。待っている間、毎回、2 台の
   コンテナの状態も見る

**`start` の、結果と例外の、終了コードへの写し方** (5.1 が、この表のとおりに写す。design.md
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

## 状態の確認 (`serve status`)

**読み取りだけである**。状態を変える呼び出し (`docker run` / `stop` / `rm` / `pull`、`mkdir`、
配布) を 1 つも出さないので、了承も求めない (design.md 「cli」の表)。台ごとに読むのは、
自分のラベルで絞った一覧、(終了しているときだけ) その識別子への `docker container inspect`、
`nvidia-smi` の 3 つの列、(ノードの定義に直結のインターフェースの名前があるときだけ)
`ip -br link show dev <名前>` である。よそのものについて読むのは、`nvidia-smi` の 3 つの列
だけである (requirements 2.4)。

`serve status` は構成の名前を受けない (design.md 「cli」の表は `<構成>` を `start` にだけ
付けている)。HTTP の宛先は、**動いている head のコンテナのラベル** (`vllm-baseline.config`)
から構成の定義を引いて、その `--port` で決める。

**`status` は、何も断らない**。何も動いていない、片方の台に入れない、`nvidia-smi` が
読めない、推論サーバーが応答しない、のどれでも、誤りにせずに「読めなかった」として示す
(requirements 1.6)。例外になるのは、`nodes` に見る台が 1 つもないという、呼ぶ側の誤り
(`ValueError`) だけである。**`serve status` の終了コードは、この関数では決まらない**。
5.1 が `read_status` の `unreadable` から決める (読めなかった台があれば 1。決めごとの 16)。

**`ServiceStatus` が `None` を許さない項目の扱い**: `NodeStatus.container_state` は
`absent` / `running` / `exited` の 3 つしかないので、**その台の一覧そのものを読めなかった
ときも `absent` になる** (ほかの項目は空、`gpu_apps` は空の列)。読めなかったことは、
`report` (既定は stderr) に警告として出すので、5.1 は、標準出力の表と一緒に、その警告を
計測者に見せる。`serve status` の 0 は「確かめられた」ではなく「確かめに行けた」である。

## 停止 (`serve stop`)

対象は、**2 台の、自分のラベルで絞った一覧に出たコンテナのすべて**である (`kind` を問わない。
推論サーバーも、重みの取得も、確認も、`serve stop` で止める。design.md 「weights」の
「止めるのは `serve stop`」、task 3.1 / 3.4 の「一覧を読めなかった台は `serve stop` で
片付ける」が、そのとおりになる)。構成の名前は受けない。

進む順: 台ごとに一覧を 1 度読む → 対象がなければ `already_stopped` → 対象と種類を見せて、
`guards.build_stop_plan` の計画で了承を得る → head → worker の順に `docker stop -t 90` →
2 台で `docker rm` → GPU を使っているプロセスが 0 件になるまで、最大
`GPU_RELEASE_TIMEOUT_S` 秒待つ。

**記録は回収しない** (design.md 「cli」の表の `serve stop` は停止だけで、回収は
`serve logs` の仕事である。`gate_own_state` も「`serve logs` で回収してから `serve stop` で
消す」と促す)。消すと読めなくなるので、その旨を、了承を求める前に見せる。

**`start` の片付け (`clean_up`) は使い回さない**。`clean_up` は `ContainerPlan` の列 (起こす
計画) を受けて `guards.rollback_commands` で相手を決めるが、`serve stop` は構成を受けないので
計画を作れない。代わりに、同じ決まり (中断が来ても、台のすべてで `stop` → `rm` を 1 回ずつ
試みてから中断を伝える。`docker rm` (強制なし) は動いているコンテナを消せないので、`stop` を
飛ばさない) を、この module の `_run_stop_plan` が守る。相手を流す直前に一覧で確かめ直さない
のは、**計画そのものが、その 1 回の一覧から作られている**からである。

**終了した自分のコンテナ (`exited`) も、`stop` → `rm` の列で片付ける** (`docker stop` は、
止まっているコンテナに対して成功する)。`rm` だけの道を作らないのは、片付けの筋道を 1 つに
保つためと、一覧を読んだあとに状態が変わりうるためである。

**GPU が空かないときは、止めには行かない** (requirements 2.3)。残っているプロセスは、よその
ものかもしれない。名前とメモリの量を示して `gpu_not_released` で返す。

**`stop` の、結果と例外の、終了コードへの写し方**:

- **0**: `StopOutcome.status` が `stopped`、`already_stopped`
- **1**: `guards.ApprovalError` (了承されなかった)、`ValueError` (見る台の定義がない)
- **2**: `StopOutcome.status` が `gpu_not_released`。`LifecycleError` (止められなかった、
  消せなかった、一覧を読めなかった台がある)
- **130**: `KeyboardInterrupt` (台のすべてで `stop` → `rm` を試みてから伝える)

## 短い要求での確かめ (`serve smoke <構成>`)

head の `/v1/messages` に、英語と日本語を 1 つずつ送る (Anthropic の Messages API の形。
ストリームでない応答。`SMOKE_PROMPTS` の、短くて害のない文)。**応答が意味の通る文かどうかは、
この module は判定しない** (人が画面で読む。requirements 6.5)。

**応答の本文は、画面に出すためだけのものである** (requirements 10.5)。本文を持つのは
`SmokeShown.text` (返り値の型。ファイルに書かない) だけで、保存してよい `types.SmokeReply` /
`SmokeOutcome` には、HTTP の状態、終わりの理由、入力と出力のトークンの数、置き換え文字
(U+FFFD) の有無しか入らない。**本文を返す口 (`send_smoke`) と、保存する記録の口 (`smoke` が
返す `SmokeOutcome`) を分けてある**。`SmokeOutcome.detail` にも本文を入れない。

HTTP の失敗 (つながらない、500、400、時間切れ、JSON が壊れている、`content` に `text` の
ブロックがない) は、落ちずに結果に入れる。**終了コードは、5.1 が `replies` から決める**:
すべての `http_status` が 200 なら 0、1 つでも違えば 2 (実行しての失敗)。例外は
`config.ConfigError` / `ValueError` (1) と `KeyboardInterrupt` (130) だけである。

## 守る決まり

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

**3.5 が足した口も、`probe` から使える**: 短い要求は `send_smoke` (本文を返す、下の 1 つの
口)、GPU のプロセスと直結のリンクは `read_gpu_apps` / `read_fabric_link` である。

## design の文からの、意図した決めごと

design が明示しない細部を、ここで決めて残す。

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
   名乗る名前が構成と違う (requirements 6.2)、(b) `/metrics` が読めて、投機的デコードの指標の
   有無が構成の `allow_speculative` と合わない (6.7。ADR 0006 K1)。どちらも、構成を直さなければ
   変わらない。**読めないこと** (つながらない、200 でない、Prometheus の形でない) は、起動の
   途中でありうるので、上限まで待つ
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
11. **`status` は、1 台に 1 つのコンテナを選んで示す** (`types.NodeStatus` が 1 台ぶんの型で
   あるため)。自分の一覧に複数の行があるときは、**動いている行を先に**、なければ一覧の 1 つ目を
   選ぶ (関門の `gate_own_state` は「1 つの構成ぶんだけ」を通すので、ふつうは 1 台に 1 つで
   ある)。終了コードは、選んだ行が動いていないときだけ、その識別子への
   `docker container inspect` で読む (動いているものの終了コードは意味がない)
12. **`status` の HTTP は、受け付けの判定 (`check_ready`) を使い回す**。読む 3 つの道筋
   (`/health`、`/v1/models`、`/metrics`) と、その読み方を 2 か所に持たないためである。その
   ため、`health_ok` は「200 だった」ときだけ真で、それ以外 (200 でない、つながらない) は
   `None` になり (requirements 1.6 の「読めないことは空で示す」)、**名乗る名前が構成と違う
   ときと、投機的デコードの指標の有無が構成の `allow_speculative` と合わないときは、処理中と
   待ちの要求の数を読まない** (`check_ready` が、そこで判定を打ち切る)。どちらも、構成を
   直さなければ直らない食い違いで、`served_model` には読めた名前が入るので、計測者は
   食い違いに気づける
13. **`status` は、`kind = "serve"` の動いているコンテナがあるときだけ HTTP に行く**。取得
   (`fetch`) や読み取り (`inspect`) のコンテナには、待ち受ける口がない。構成の名前が手元の
   定義にない、`--port` を 1 つに決められない、のときも、警告を出して HTTP を飛ばす
   (`serve status` は、断らない)
14. **`smoke` は、本文を返す口と、保存する記録の口を分ける** (requirements 10.5)。1 つの要求は
   `send_smoke` が `SmokeShown` (保存してよい `SmokeReply` と、画面に出すためだけの `text`) を
   返し、`smoke` は `SmokeReply` だけを `SmokeOutcome` に入れて、本文は `report` に出す。
   `SmokeShown` は `dataclass` で、`types` には置かない (`types` の型は、記録として書き出せる
   ものだけにする)
15. **`smoke` は、意味の通る文かどうかを判定しない** (requirements 6.5 は、人が確かめると
   定める)。ただし、**本文が空であること**と**置き換え文字があること**は、機械で分かるので
   `SmokeOutcome.detail` と `SmokeReply.replacement_char` に出す (`SmokeReply` に「空」の項目は
   ないので、空は `detail` の文で示す)
16. **一覧を読めなかった台の印は、`status` の外で付ける** (task 5.1 の申し送り)。`status` が
   返す `types.ServiceStatus` は design.md が決めた形で、`container_state` に「わからない」が
   ないので、一覧を読めなかった台も `absent` になる。それだけを見て「何も動いていない」と
   読み違えないように、`read_status` が `StatusShown` (状態と、読めなかった台) を返す。
   `status` は変えないので、`read_status` は、印のために一覧をもう一度読む (どちらも
   読み取りだけ)。**`status` 自身は断らないが、`serve status` の終了コードは、cli が
   `unreadable` から決める** (読めた = 0、読めなかった台がある = 1)。この module は、
   終了コードを決めない
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
from typing import Final, TextIO, cast, get_args

import httpx

from serving_kit.config import ConfigError
from serving_kit.guards import (
    READ_TIMEOUT_S,
    STOP_TIMEOUT_S,
    Confirmer,
    OwnContainer,
    build_approved_plan,
    build_stop_plan,
    list_own_containers,
    match_running,
    parse_gpu_apps,
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
    LABEL_KIND,
    LABEL_STARTED_AT,
    build_plans,
)
from serving_kit.remote import RemoteError, RemoteRunner
from serving_kit.types import (
    AnyWeightsManifest,
    CommandResult,
    ConfigDef,
    ConfigKind,
    ContainerPlan,
    ContainerState,
    GateResult,
    GpuApp,
    KnownFailure,
    Lang,
    LaunchObservation,
    LaunchRecord,
    NodeDef,
    NodeRole,
    NodeStatus,
    PlannedPush,
    PlannedRun,
    ServiceStatus,
    SmokeOutcome,
    SmokeReply,
    StartOutcome,
    StopOutcome,
)

__all__ = [
    "CLEANUP_TIMEOUT_S",
    "COMMAND_NAME",
    "GPU_APPS_ARGV",
    "GPU_POLL_INTERVAL_S",
    "GPU_RELEASE_TIMEOUT_S",
    "HTTP_PORT_FLAG",
    "HTTP_TIMEOUT_S",
    "LAUNCH_RECORD_SUBDIR",
    "OBSERVE_TAIL_LINES",
    "READY_POLL_INTERVAL_S",
    "SMOKE_MAX_TOKENS",
    "SMOKE_PROMPTS",
    "SMOKE_TIMEOUT_S",
    "SPEC_DECODE_METRIC_PREFIX",
    "START_TIMEOUT_S",
    "Controls",
    "LifecycleError",
    "Readiness",
    "RunFailed",
    "SmokeShown",
    "StatusShown",
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
    "parse_link_state",
    "push_launch_records",
    "read_fabric_link",
    "read_gpu_apps",
    "read_status",
    "record_pushes",
    "send_smoke",
    "smoke",
    "start",
    "start_all",
    "started_targets",
    "status",
    "stop",
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

GPU_RELEASE_TIMEOUT_S: Final[float] = 60.0
"""止めたあと、GPU を使っているプロセスが 0 件になるのを待つ上限 (design.md 「lifecycle」の
停止: 「最大 60 秒待つ」)。"""

GPU_POLL_INTERVAL_S: Final[float] = 5.0
"""GPU を使っているプロセスを見に行く間隔 (design は定めていない。上限の 60 秒を 12 回に
分ける。受け付けの開始を待つ間隔 (10 秒) より短くしたのは、止まるのは速いからである)。"""

SMOKE_TIMEOUT_S: Final[float] = 120.0
"""短い要求 1 つの時間切れ。生成があるので、読み取りの問い合わせ (10 秒) より長くする。"""

SMOKE_MAX_TOKENS: Final[int] = 64
"""短い要求の `max_tokens` (確かめるのは、返り切ることと文字の壊れだけなので、小さくする)。"""

SMOKE_PROMPTS: Final[Mapping[Lang, str]] = {
    "en": "Reply in one short sentence: what is the capital of France?",
    "ja": "一文で短く答えてください。日本の首都はどこですか。",
}
"""英語と日本語の、短くて害のない要求 (requirements 6.5)。

日本語を送るのは、同時の要求での文字化けの報告 (#57087) を、人が目で見て確かめられるように
するためである (置き換え文字の有無は、この module が機械で見る)。文をここに定数として置くの
は、送った内容を、構成にも記録にも残さないためである (requirements 10.5)。
"""

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
"""投機的デコードが入っていることを表す指標の接頭辞 (requirements 6.7)。

構成の `allow_speculative` が真なら、この行があることを正常とする。偽なら、ないことを正常と
する (6.7。ADR 0006 K1)。"""

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

_ROLE_ORDER: Final[tuple[NodeRole, ...]] = ("head", "worker")
"""台を見る順序 (head → worker。design.md 「lifecycle」の停止の順序と同じ)。

`serve status` と `serve stop` は構成を受けないので、構成の `nodes` ではなく、この順序で
`nodes` にある台を見る (渡された辞書の並びで、止める順序が変わらないようにする)。
"""

GPU_APPS_ARGV: Final[tuple[str, ...]] = (
    "nvidia-smi",
    "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader",
)
"""GPU を使っているプロセスを読む、ただ 1 つの問い合わせ。

よそのものについて読むのは、この 3 つの列だけである (requirements 2.2、2.4)。`guards` の
`gate_gpu_idle` が起動の前に読むのと、同じ問い合わせである。
"""

_CONFIG_KINDS: Final[frozenset[str]] = frozenset(get_args(ConfigKind))
"""ラベル `vllm-baseline.kind` として読める値 (`types.ConfigKind` の 5 つ)。"""

_LINK_UP: Final[str] = "UP"
_LINK_DOWN: Final[str] = "DOWN"
"""`ip -br link` の 2 列目 (実際の状態)。ほかの値 (`UNKNOWN` など) は、判断しない。"""

_REPLACEMENT_CHAR: Final[str] = "�"
"""置き換え文字 (日本語の文字化けの印。requirements 7.6、10.5)。"""

_TEXT_BLOCK: Final[str] = "text"
"""Anthropic の Messages API の応答で、本文を持つブロックの種類。"""


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


@dataclass(frozen=True)
class SmokeShown:
    """短い要求 1 つの結果 (公開の口。`send_smoke` が返す。決めごとの 14)。

    - `reply`: **保存してよい部分** (`types.SmokeReply`。HTTP の状態、終わりの理由、トークンの
      数、置き換え文字の有無だけ)
    - `text`: **画面に出すためだけの本文**。どのファイルにも書かない (requirements 10.5)。
      読めなかったときは空の文字列
    - `problem`: 読めなかった理由 (つながらない、200 でない、JSON が壊れている、`text` の
      ブロックがない)。**応答の本文を入れない**

    この型を `types` に置かないのは、`types` の型が、記録として書き出せるものだけであるべき
    だからである (本文を持つ型が `types` にあると、うっかり書き出せてしまう)。
    """

    reply: SmokeReply
    text: str = ""
    problem: str = ""


@dataclass(frozen=True)
class StatusShown:
    """状態と、一覧を読めなかった台 (公開の口。`read_status` が返す。決めごとの 16)。

    - `service`: `status` がそのまま返す `types.ServiceStatus`
    - `unreadable`: 自分のコンテナの一覧を読めなかった台の役割

    `types.NodeStatus.container_state` に「わからない」がないので、`status` は、一覧を読め
    なかった台も `absent` として返す。それだけを見ると「何も動いていない」と読み違えるので、
    読めなかった台を、この型で別に持って返す (`SmokeShown` と同じ形の、画面のための型)。
    """

    service: ServiceStatus
    unreadable: tuple[NodeRole, ...] = ()


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


def check_ready(
    client: httpx.Client, base_url: str, served_model_name: str, *, speculative_allowed: bool
) -> Readiness:
    """受け付けの開始を、1 回ぶん判定する (design.md 「lifecycle」の (1)〜(3))。

    **読めないこと**は、起動の途中でありうるので、待ち続ける理由にする。**読めて、内容が
    構成と食い違うこと**は、待っても直らないので `fatal` にする (決めごとの 4)。

    `speculative_allowed` は、その構成が投機的デコードを許しているかどうかである (6.7。
    ADR 0006 K1)。許す構成では投機の指標が出ているのを正常とし、出ていないのを異常とする。
    許さない構成では、出ているのを異常とする。
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
    if speculative and not speculative_allowed:
        return Readiness(
            health_ok=True,
            served_model=served,
            max_model_len=length,
            detail="投機的デコードの指標が出ている",
            fatal=(
                f"投機的デコードの指標が出ている ({', '.join(speculative[:3])})。この構成"
                " (allow_speculative = false) は投機的デコードを使わない。待っても変わらない"
                "ので、時間切れを待たずに失敗にする (requirements 6.7)"
            ),
        )
    if not speculative and speculative_allowed:
        return Readiness(
            health_ok=True,
            served_model=served,
            max_model_len=length,
            detail="投機的デコードの指標が出ていない",
            fatal=(
                "構成は投機的デコードを許している (allow_speculative = true) のに、`/metrics` に"
                f" {SPEC_DECODE_METRIC_PREFIX} の行がない。投機の設定なしで起きているので、"
                "待っても変わらない"
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
        weights=None if weights is None else weights.identity,
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


def _inspect_container(
    runner: RemoteRunner, node: NodeDef, container_id: str, timeout_s: float
) -> tuple[str, int | None]:
    """1 つのコンテナの状態と終了コードを読む (対象は、一覧から来た識別子だけ)。

    読み取りが届かなかったこと (`RemoteError`) は、そのまま外に出す (呼ぶ側が決める)。
    **読めなかったことを「まだ動いている」に倒さない**。
    """
    argv = ("docker", "container", "inspect", "--format", _STATE_FORMAT, container_id)
    result = runner.run(node, argv, timeout_s=timeout_s, mutating=False)
    if result.exit_code != 0:
        return _STATE_ABSENT, None
    fields = result.stdout.split()
    if not fields:
        return _STATE_ABSENT, None
    code = fields[1] if len(fields) > 1 else ""
    return fields[0], int(code) if code.lstrip("-").isdecimal() else None


def _container_state(
    runner: RemoteRunner, target: Target, timeout_s: float
) -> tuple[str, int | None]:
    """待っている間に見る、起こしたコンテナの状態と終了コード。

    読み取りが届かなかったこと (`RemoteError`) は、そのまま外に出す (呼ぶ側が、片付けてから
    実行しての失敗にする)。
    """
    return _inspect_container(runner, target.node, target.container.id, timeout_s)


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
    speculative_allowed: bool,
) -> Waited:
    """受け付けの開始を待つ (design.md 「lifecycle」の受け付けの開始の判定)。

    1 周ごとに、HTTP の 3 つを見てから、2 台のコンテナの状態を見る (決めごとの 3)。どちらかの
    コンテナが終了していたら、時間切れを待たずに失敗にする。時計と眠りは引数から来るので、
    試験は実際に眠らない。`speculative_allowed` は `check_ready` にそのまま渡す (6.7)。
    """
    deadline = controls.clock() + controls.timeout_s
    while True:
        readiness = check_ready(
            controls.client,
            base_url,
            served_model_name,
            speculative_allowed=speculative_allowed,
        )
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
    manifest: AnyWeightsManifest | None = None,
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
            speculative_allowed=config.allow_speculative,
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


# --- 台ごとの読み取り (`serve status` と、停止のあとの確かめで使う) --------


def read_gpu_apps(
    runner: RemoteRunner, node: NodeDef, *, timeout_s: float = READ_TIMEOUT_S
) -> tuple[GpuApp, ...] | None:
    """GPU を使っているプロセスを読む (読めなければ空を返す。断らない)。

    **よそのものについて読むのは、この 3 つの列 (pid、名前、メモリの量) だけである**
    (requirements 2.2、2.4)。関門 (`guards.gate_gpu_idle`) と同じ問い合わせだが、こちらは
    判定をせずに、読み取りをそのまま返す (`serve status` は示すだけ、`serve stop` は 0 件に
    なるのを待つだけで、どちらも断らない)。

    返り値:
        読めたプロセスの列。**読めなかったときは `None`** (「0 件」と区別する)。
    """
    try:
        result = runner.run(node, GPU_APPS_ARGV, timeout_s=timeout_s, mutating=False)
    except RemoteError:
        return None
    if result.exit_code != 0:
        return None
    try:
        return parse_gpu_apps(result.stdout)
    except ValueError:
        return None


def parse_link_state(text: str, ifname: str) -> bool | None:
    """`ip -br link` の出力から、1 つのインターフェースがつながっているかを読む。

    見るのは 2 列目 (実際の状態) である。見本 (`tests/fixtures/spark/*/ip-br-link.txt`) では、
    つながっている側が `UP`、ケーブルのない側が `DOWN` になる。仮想のインターフェースに出る
    `UNKNOWN` などは、つながっているとも切れているとも言えないので、空で返す。名前に `@if2`
    のような対の印が付くことがあるので、`@` の前までで見る。
    """
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0].split("@", 1)[0] != ifname:
            continue
        if fields[1] == _LINK_UP:
            return True
        if fields[1] == _LINK_DOWN:
            return False
        return None
    return None


def read_fabric_link(
    runner: RemoteRunner, node: NodeDef, *, timeout_s: float = READ_TIMEOUT_S
) -> bool | None:
    """直結のリンクがつながっているかを読む (ノードの定義に名前がなければ、読みに行かない)。

    `remote` が通す読み取りの形 (`ip [限られたオプション] (link|addr) [show [dev] <名前>]`) の
    うち、1 つのインターフェースだけを見る形を使う。読めなかったことは、断らずに空で返す。
    """
    ifname = node.fabric_ifname
    if ifname is None:
        return None
    argv = ("ip", "-br", "link", "show", "dev", ifname)
    try:
        result = runner.run(node, argv, timeout_s=timeout_s, mutating=False)
    except RemoteError:
        return None
    if result.exit_code != 0:
        return None
    return parse_link_state(result.stdout, ifname)


def _ordered_roles(nodes: Mapping[NodeRole, NodeDef]) -> tuple[NodeRole, ...]:
    """見る台の順序 (head → worker)。`nodes` にない役割は飛ばす。"""
    return tuple(role for role in _ROLE_ORDER if role in nodes)


def _current_container(containers: Sequence[OwnContainer]) -> OwnContainer | None:
    """1 台の状態として示す 1 つの行を選ぶ (動いている行を先に。決めごとの 11)。"""
    return next(
        (item for item in containers if item.state == _STATE_RUNNING),
        containers[0] if containers else None,
    )


def _label_kind(raw: str | None) -> ConfigKind | None:
    """ラベル `vllm-baseline.kind` を読む (知らない値は、空にする)。"""
    if raw is None or raw not in _CONFIG_KINDS:
        return None
    return cast(ConfigKind, raw)


def _shown_gpu_apps(apps: Sequence[GpuApp]) -> str:
    """GPU を使っているプロセスを、名前とメモリの量で並べる (出すのは、この 3 つだけ)。"""
    shown: list[str] = []
    for app in apps:
        pid = "pid 不明" if app.pid is None else f"pid {app.pid}"
        memory = (
            "メモリの量は読めない ([N/A])"
            if app.used_memory_mib is None
            else f"{app.used_memory_mib} MiB"
        )
        shown.append(f"{app.process_name} ({pid}, {memory})")
    return ", ".join(shown)


# --- 状態の確認 (`serve status`。読み取りだけ) -----------------------------


def _listed(
    runner: RemoteRunner, node: NodeDef, timeout_s: float, stream: TextIO
) -> tuple[OwnContainer, ...]:
    """自分のラベルで絞った一覧を読む。読めなければ、警告を出して、空として扱う。"""
    try:
        return list_own_containers(runner, node, timeout_s=timeout_s)
    except (RemoteError, ValueError) as exc:
        _report(
            stream,
            f"警告: {node.role} ({node.ssh_host}) の自分のコンテナの一覧を読めなかった。"
            f"この台の状態は、読めなかったものとして (container_state は absent で) 示す: {exc}",
        )
        return ()


def _exit_code_of(
    runner: RemoteRunner,
    node: NodeDef,
    container: OwnContainer | None,
    timeout_s: float,
    stream: TextIO,
) -> int | None:
    """終了した自分のコンテナの終了コードを読む (動いているものは読まない。決めごとの 11)。

    対象は、一覧から来た識別子だけである (requirements 2.3 の不変条件の (a))。
    """
    if container is None or container.state == _STATE_RUNNING:
        return None
    try:
        _, code = _inspect_container(runner, node, container.id, timeout_s)
    except RemoteError as exc:
        _report(
            stream,
            f"警告: {node.role} ({node.ssh_host}) の {container.name} の終了コードを"
            f"読めなかった: {exc}",
        )
        return None
    return code


def _observed_node_status(
    role: NodeRole,
    container: OwnContainer | None,
    *,
    gpu_apps: tuple[GpuApp, ...],
    fabric_link_up: bool | None,
    exit_code: int | None,
) -> NodeStatus:
    """1 台の状態を、一覧の行と、GPU と、直結のリンクの読み取りから作る (`serve status`)。

    構成の名前と種類は、**そのコンテナのラベル**から読む (`start` の `_node_status` が、選んだ
    構成から読むのとは違う。`serve status` は、構成の名前を受けないため)。コンテナがない
    **か、一覧を読めなかった**ときは `absent` になる (module の docstring の「`None` を許さない
    項目の扱い」)。
    """
    if container is None:
        return NodeStatus(
            node=role,
            container_state="absent",
            gpu_apps=gpu_apps,
            fabric_link_up=fabric_link_up,
        )
    state: ContainerState = "running" if container.state == _STATE_RUNNING else "exited"
    return NodeStatus(
        node=role,
        container_state=state,
        exit_code=exit_code,
        config_name=container.labels.get(LABEL_CONFIG),
        kind=_label_kind(container.labels.get(LABEL_KIND)),
        image_digest=container.labels.get(LABEL_IMAGE),
        config_sha256=container.labels.get(LABEL_CONFIG_SHA256),
        started_at=label_time(container.labels.get(LABEL_STARTED_AT)),
        gpu_apps=gpu_apps,
        fabric_link_up=fabric_link_up,
    )


def _status_target(
    configs: Mapping[str, ConfigDef],
    nodes: Mapping[NodeRole, NodeDef],
    container: OwnContainer | None,
    stream: TextIO,
) -> tuple[str, str, bool] | None:
    """HTTP の宛先と、名乗るはずのモデルの名前と、投機の許可を、head のコンテナのラベルから
    決める。

    決めごとの 13: 動いている `kind = "serve"` のコンテナがあるときだけ、問い合わせに行く。
    構成の名前が手元の定義にない、`--port` を 1 つに決められない、のときは、警告を出して
    飛ばす (`serve status` は、断らない)。投機の許可 (`allow_speculative`) は、`check_ready`
    に渡して、構成ごとの判定 (6.7) を同じ意味にする。
    """
    head = nodes.get(_HEAD)
    if head is None or container is None or container.state != _STATE_RUNNING:
        return None
    if container.kind != _KIND_SERVE:
        return None
    name = container.labels.get(LABEL_CONFIG)
    config = None if name is None else configs.get(name)
    if config is None:
        _report(
            stream,
            f"警告: head で動いているコンテナの構成 '{name or '(ラベルがない)'}' が、手元の"
            "定義にないので、推論サーバーの口には問い合わせない",
        )
        return None
    if config.served_model_name is None:  # pragma: no cover - config の検査 4 が、先に断る
        _report(
            stream,
            f"警告: 構成 '{config.name}' に served_model_name がないので、推論サーバーの口には"
            "問い合わせない",
        )
        return None
    try:
        port = http_port(config)
    except ConfigError as exc:
        _report(stream, f"警告: 推論サーバーの口には問い合わせない: {exc}")
        return None
    return f"http://{head.lan_addr}:{port}", config.served_model_name, config.allow_speculative


def status(
    runner: RemoteRunner,
    configs: Mapping[str, ConfigDef],
    nodes: Mapping[NodeRole, NodeDef],
    *,
    client: httpx.Client | None = None,
    report: TextIO | None = None,
    http_timeout_s: float = HTTP_TIMEOUT_S,
    read_timeout_s: float = READ_TIMEOUT_S,
) -> ServiceStatus:
    """2 台と、推論サーバーの、いまの状態を示す (`serve status`。requirements 1.6、6.5)。

    **読み取りだけで、状態を変える呼び出しを 1 つも出さない**ので、了承も求めない。台ごとに
    読むのは、自分のラベルで絞った一覧、(終了しているときだけ) その識別子への
    `docker container inspect`、`nvidia-smi` の 3 つの列、(ノードの定義に直結の
    インターフェースの名前があるときだけ) `ip -br link show dev <名前>` である。

    **どれが読めなくても、断らない** (module の docstring の「状態の確認」)。読めなかったこと
    は `report` に警告として出し、結果の項目は空にする。`container_state` は `None` を許さない
    ので、一覧そのものを読めなかった台も `absent` になる。

    引数:
        runner: 遠隔の実行役。
        configs: 読み込んだ構成の定義 (HTTP の宛先を、動いているコンテナのラベルから引く)。
        nodes: 役割ごとのノードの定義 (head → worker の順に見る)。
        client: HTTP のクライアント。省くと `new_client` で作り、この関数の中で閉じる。
        report: 警告の出し先。既定は `sys.stderr` (標準出力は、後の処理が読む行に使う)。
        http_timeout_s: 1 回の HTTP の問い合わせの時間切れ。
        read_timeout_s: 1 つの読み取りの時間切れ。

    返り値:
        2 台の `NodeStatus` と、推論サーバーの `ServiceStatus` (この関数は、何も断らない。
        `serve status` の終了コードは、cli が `read_status` の `unreadable` から決める)。

    例外:
        ValueError: `nodes` に head も worker もないとき (Spark に触る前に断る)。
    """
    roles = _ordered_roles(nodes)
    if not roles:
        raise ValueError("nodes に head も worker もない (状態を確かめる台がない)")
    stream = sys.stderr if report is None else report
    found: list[NodeStatus] = []
    seen: dict[NodeRole, OwnContainer] = {}
    for role in roles:
        node = nodes[role]
        container = _current_container(_listed(runner, node, read_timeout_s, stream))
        exit_code = _exit_code_of(runner, node, container, read_timeout_s, stream)
        apps = read_gpu_apps(runner, node, timeout_s=read_timeout_s)
        if apps is None:
            _report(
                stream,
                f"警告: {role} ({node.ssh_host}) の GPU を使っているプロセス"
                f" ({GPU_APPS_ARGV[0]}) を読めなかった (空として示す)",
            )
        found.append(
            _observed_node_status(
                role,
                container,
                gpu_apps=() if apps is None else apps,
                fabric_link_up=read_fabric_link(runner, node, timeout_s=read_timeout_s),
                exit_code=exit_code,
            )
        )
        if container is not None:
            seen[role] = container

    target = _status_target(configs, nodes, seen.get(_HEAD), stream)
    if target is None:
        return ServiceStatus(nodes=tuple(found))
    base_url, served_model_name, allow_speculative = target
    owns_client = client is None
    used = new_client(http_timeout_s) if client is None else client
    try:
        readiness = check_ready(
            used, base_url, served_model_name, speculative_allowed=allow_speculative
        )
    finally:
        if owns_client:
            used.close()
    return ServiceStatus(
        nodes=tuple(found),
        # 読めなかったこと (200 でない、つながらない) は、偽ではなく空で示す (決めごとの 12)
        health_ok=True if readiness.health_ok else None,
        served_model=readiness.served_model,
        max_model_len=readiness.max_model_len,
        running_requests=readiness.running_requests,
        waiting_requests=readiness.waiting_requests,
    )


def read_status(
    runner: RemoteRunner,
    configs: Mapping[str, ConfigDef],
    nodes: Mapping[NodeRole, NodeDef],
    *,
    client: httpx.Client | None = None,
    report: TextIO | None = None,
    http_timeout_s: float = HTTP_TIMEOUT_S,
    read_timeout_s: float = READ_TIMEOUT_S,
) -> StatusShown:
    """`status` に、一覧を読めなかった台の印を足して返す (`serve status` の入口)。

    **`status` そのものは変えない** (返す型は design.md が決めている)。読めたかどうかは、
    `status` を呼ぶ前に、台ごとに自分のラベルで絞った一覧を読んで確かめる。そのため、一覧の
    読み取りは台ごとに 2 回出る (どちらも読み取りだけで、状態を変える呼び出しは 1 つも出ない)。
    2 回の間に状態が変わりうるが、この結果は画面に見せるためのものなので、判定には使わない。

    引数と例外は `status` と同じである (`ValueError`: `nodes` に head も worker もないとき)。
    """
    roles = _ordered_roles(nodes)
    if not roles:
        raise ValueError("nodes に head も worker もない (状態を確かめる台がない)")
    stream = sys.stderr if report is None else report
    unreadable: list[NodeRole] = []
    for role in roles:
        try:
            list_own_containers(runner, nodes[role], timeout_s=read_timeout_s)
        except (RemoteError, ValueError):
            # 理由は、このあとの `status` が `_listed` で警告として出す (二重に出さない)
            unreadable.append(role)
    service = status(
        runner,
        configs,
        nodes,
        client=client,
        report=stream,
        http_timeout_s=http_timeout_s,
        read_timeout_s=read_timeout_s,
    )
    return StatusShown(service=service, unreadable=tuple(unreadable))


# --- 停止 (`serve stop`) ---------------------------------------------------


def _stop_targets_text(
    nodes: Mapping[NodeRole, NodeDef], targets: Mapping[NodeRole, Sequence[OwnContainer]]
) -> str:
    """了承を求める前に見せる、止める対象の一覧。

    種類 (`serve` / `fetch` など) と状態は、`guards.format_plan` が見せる計画の文には出ない
    ので、ここで見せる。184 GiB の取得を止めようとしていることに、計測者が気づけるように
    するためである (design.md 「weights」の Idempotency)。
    """
    lines = ["止める対象 (この道具のラベルの付いた、自分のコンテナだけ):"]
    for role in _ROLE_ORDER:
        node = nodes.get(role)
        where = role if node is None else f"{role} ({node.ssh_host})"
        lines.extend(
            f"  {where}: {container.name}"
            f" (種類 {container.kind or '(ラベルがない)'}、状態 {container.state})"
            for container in targets.get(role, ())
        )
    lines.append(
        "コンテナを消すと、その中の記録は読めなくなる"
        " (要るなら、先に `serve logs` で回収してから止める)"
    )
    return "\n".join(lines)


def _run_stop_plan(
    runner: RemoteRunner,
    nodes: Mapping[NodeRole, NodeDef],
    commands: Sequence[PlannedRun],
    stream: TextIO,
) -> tuple[tuple[str, ...], KeyboardInterrupt | None]:
    """止めて消す列を、順に 1 回ずつ流す (task 3.4 の片付けの決まりに合わせる)。

    **前提**: 了承済みの計画が、実行役に渡っていること (`guards.request_approval`)。列は
    `guards.build_stop_plan` が `stop_argv` / `remove_argv` で作るので、了承を得た計画と完全に
    一致する (`remote.CallGuard` は、完全な一致で見る)。

    **保証**:

    1. 相手は、`guards.build_stop_plan` が、自分のラベルで絞った一覧の行から作った名前だけで
       ある。流す直前に一覧を読み直さないのは、**計画そのものが、その 1 回の読み取りから
       作られている**からである (`clean_up` は、起動の**前**に書いた計画を流すので、直前に
       `guards.rollback_commands` で相手を確かめ直す)
    2. **中断が何度来ても、`stop` → `rm` の両方を、台のすべてで試みる**。`docker rm` (強制なし)
       は、動いているコンテナを消せず、`-f` は了承済みの計画にも `remote` の許可の一覧の決まり
       にもないので、`stop` を飛ばして `rm` だけ流すと、コンテナが残ってしまう
    3. **同じ呼び出しを、やり直さない** (無限に粘らない)。1 つの命令に 1 回だけ挑み、中断は
       最初の 1 つを覚えて、残りの命令に進む
    4. **急ぐことはできない**。流せるのは、了承済みの計画にある列だけなので、`stop` の猶予
       (`-t 90`) を短くする列は作れない

    返り値:
        止め切れなかったことと、受けた中断 (**投げるのは、呼ぶ側**)。
    """
    problems: list[str] = []
    interrupted: KeyboardInterrupt | None = None
    for command in commands:
        what = _CLEANUP_LABELS[command.argv[1]]
        node = nodes[command.node]
        try:
            result = runner.run(node, command.argv, timeout_s=CLEANUP_TIMEOUT_S, mutating=True)
        except KeyboardInterrupt as exc:
            # やり直さずに、次の命令に進む (保証の 2、3)
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
            stream,
            "警告: コンテナの片付けが、すべては終わらなかった: " + " / ".join(problems),
        )
    return tuple(problems), interrupted


def _wait_for_gpu(
    runner: RemoteRunner,
    nodes: Mapping[NodeRole, NodeDef],
    roles: Sequence[NodeRole],
    *,
    timeout_s: float,
    poll_interval_s: float,
    read_timeout_s: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> tuple[tuple[GpuApp, ...], tuple[NodeRole, ...]]:
    """GPU を使っているプロセスが 0 件になるまで待つ (**止めには行かない**)。

    読めなかった台は、待ち続けても読めるようにならないので、その回で「空」として数えずに、
    名前だけを返す (呼ぶ側が「確かめられなかった」と示す)。時計と眠りは引数から来るので、
    試験は実際に眠らない。

    返り値:
        残っているプロセスと、読めなかった台の役割。
    """
    deadline = clock() + timeout_s
    while True:
        remaining: list[GpuApp] = []
        unreadable: list[NodeRole] = []
        for role in roles:
            apps = read_gpu_apps(runner, nodes[role], timeout_s=read_timeout_s)
            if apps is None:
                unreadable.append(role)
                continue
            remaining.extend(apps)
        if not remaining or clock() >= deadline:
            return tuple(remaining), tuple(unreadable)
        sleep(poll_interval_s)


def _gpu_verdict(unreadable: Sequence[NodeRole]) -> str:
    """GPU の確かめの言い方 (読めなかった台があれば、空いたとは言わない)。"""
    if not unreadable:
        return "GPU を使っているプロセスは 0 件になった"
    return (
        f"{'、'.join(unreadable)} の GPU を使っているプロセスを読めなかったので、"
        "空いたかどうかは確かめられなかった (`serve status` で確かめる)"
    )


def stop(
    runner: RemoteRunner,
    nodes: Mapping[NodeRole, NodeDef],
    *,
    confirmer: Confirmer,
    report: TextIO | None = None,
    read_timeout_s: float = READ_TIMEOUT_S,
    gpu_timeout_s: float = GPU_RELEASE_TIMEOUT_S,
    gpu_poll_interval_s: float = GPU_POLL_INTERVAL_S,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> StopOutcome:
    """自分のコンテナを 2 台とも止めて消し、GPU が空くまで待つ (`serve stop`)。

    対象は、**自分のラベルで絞った一覧に出たコンテナのすべて**である (`kind` を問わない)。
    構成の名前は受けない (module の docstring の「停止」)。順序は head → worker の
    `docker stop -t 90` のあと、2 台の `docker rm` である (design.md 「lifecycle」の停止)。

    **記録は回収しない** (`serve logs` の仕事である)。消すと読めなくなるので、その旨を、
    了承を求める前に見せる。

    引数:
        runner: 遠隔の実行役。
        nodes: 役割ごとのノードの定義 (head → worker の順に止める)。
        confirmer: 計画を見せて、了承を得る口。
        report: 対象の一覧と警告の出し先。既定は `sys.stderr`。
        read_timeout_s: 一覧と `nvidia-smi` の読み取りの時間切れ。
        gpu_timeout_s: GPU を使っているプロセスが 0 件になるのを待つ上限。
        gpu_poll_interval_s: そのプロセスを見に行く間隔。
        sleep: 眠る口 (試験は、実際に眠らないものを渡す)。既定は `time.sleep`。
        clock: 時計 (単調増加の秒)。既定は `time.monotonic`。

    返り値:
        `stopped` / `already_stopped` (終了コード 0)、`gpu_not_released` (2)。

    例外:
        guards.ApprovalError: 計測者が了承しなかった (終了コード 1。状態を変える呼び出しは、
            1 つも出ない)。
        LifecycleError: 止められなかった、消せなかった、一覧を読めなかった台がある
            (終了コード 2)。
        KeyboardInterrupt: 中断 (終了コード 130)。台のすべてで `stop` → `rm` を試みてから
            伝える。
        ValueError: `nodes` に head も worker もないとき (Spark に触る前に断る)。
    """
    roles = _ordered_roles(nodes)
    if not roles:
        raise ValueError("nodes に head も worker もない (止める台がない)")
    stream = sys.stderr if report is None else report

    found: dict[NodeRole, tuple[OwnContainer, ...]] = {}
    unchecked: list[str] = []
    for role in roles:
        node = nodes[role]
        try:
            found[role] = list_own_containers(runner, node, timeout_s=read_timeout_s)
        except (RemoteError, ValueError) as exc:
            # 一覧が読めていないので、その名前が自分のものだと言えない (requirements 2.3)
            unchecked.append(
                f"{role} ({node.ssh_host}): 自分のコンテナの一覧を読めないので、この台では"
                f" 何も止めない (名前だけで止めると、よそのコンテナを止めうる): {exc}"
            )
    targets = {role: items for role, items in found.items() if items}
    if not targets:
        if unchecked:
            raise LifecycleError(
                "止める対象を確かめられなかった: "
                + " / ".join(unchecked)
                + "。入れるようになってから、もう一度 `serve stop` を打つ"
            )
        return StopOutcome(
            status="already_stopped",
            detail=(
                f"{len(roles)} 台とも、この道具のラベルの付いたコンテナは 1 つもない。"
                "止めるものがないので、了承も求めず、状態を変える呼び出しも 1 つも出さない"
            ),
        )

    _report(stream, _stop_targets_text(nodes, targets))
    plan = build_stop_plan(targets)
    request_approval(confirmer, runner, plan, nodes)
    commands = tuple(item for item in plan.forward if isinstance(item, PlannedRun))
    failures, interrupted = _run_stop_plan(runner, nodes, commands, stream)
    if interrupted is not None:
        # 中断は、計測者の意思表示なので、実行しての失敗より優先して伝える (決めごとの 8)
        raise interrupted
    if failures or unchecked:
        raise LifecycleError(
            "止め切れなかった: "
            + " / ".join((*failures, *unchecked))
            + "。`serve status` で残りを確かめてから、もう一度 `serve stop` を打つ"
        )

    remaining, unreadable = _wait_for_gpu(
        runner,
        nodes,
        roles,
        timeout_s=gpu_timeout_s,
        poll_interval_s=gpu_poll_interval_s,
        read_timeout_s=read_timeout_s,
        sleep=time.sleep if sleep is None else sleep,
        clock=time.monotonic if clock is None else clock,
    )
    count = sum(len(items) for items in targets.values())
    cleaned = (
        f"head → worker の順に、{count} 個のコンテナを止めて消した"
        f" (`docker stop -t {STOP_TIMEOUT_S}` → `docker rm`)。"
    )
    if remaining:
        return StopOutcome(
            status="gpu_not_released",
            detail=(
                f"{cleaned}{gpu_timeout_s:.0f} 秒のうちに、GPU を使っているプロセスが 0 件に"
                f"ならなかった ({_shown_gpu_apps(remaining)})。よそのプロセスかもしれないので、"
                f"止めには行かない。{_gpu_verdict(unreadable)}"
            ),
            remaining_gpu_apps=remaining,
        )
    return StopOutcome(status="stopped", detail=f"{cleaned}{_gpu_verdict(unreadable)}")


# --- 短い要求での確かめ (`serve smoke`) ------------------------------------


def _smoke_body(model: str, prompt: str, max_tokens: int) -> dict[str, object]:
    """Anthropic の Messages API の、ストリームでない要求の本文。"""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }


def _usage_count(usage: object, key: str) -> int | None:
    """`usage` から、トークンの数を読む (読めなければ空)。"""
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _reply_text(payload: Mapping[str, object]) -> tuple[str, str]:
    """応答の `content` から、本文のテキストを集める (**画面に出すためだけ**)。

    返り値:
        本文と、読めなかった理由 (読めたら空)。理由に、応答の本文を入れない。
    """
    content = payload.get("content")
    if not isinstance(content, list):
        return "", "応答に content の配列がないので、本文が空である"
    blocks = [
        block.get("text")
        for block in content
        if isinstance(block, dict) and block.get("type") == _TEXT_BLOCK
    ]
    if not blocks:
        return "", (
            f"応答の content に {_TEXT_BLOCK} のブロックがないので、本文が空である"
            " (thinking のブロックだけ、など)"
        )
    text = "".join(piece for piece in blocks if isinstance(piece, str))
    return (text, "") if text else ("", "応答の本文が空である")


def send_smoke(
    client: httpx.Client,
    base_url: str,
    *,
    model: str,
    lang: Lang,
    prompt: str,
    max_tokens: int = SMOKE_MAX_TOKENS,
) -> SmokeShown:
    """短い要求を 1 つ送って、結果を読む (公開の口。4.1 の縮小の確認も、これを使う)。

    **落ちない**。つながらない、200 でない、JSON として読めない、`text` のブロックがない、の
    どれでも、読めたところまでを `SmokeShown` に入れて返す (requirements 6.5)。判定はしない
    (意味の通る文かどうかは、人が画面で読む)。

    **応答の本文が入るのは `SmokeShown.text` だけ**で、`SmokeShown.reply` (保存してよい部分)
    にも `SmokeShown.problem` にも入らない (requirements 10.5)。

    認証のヘッダは付けない (待ち受けに認証がない。design.md 「Security Considerations」)。
    """
    url = f"{base_url}/v1/messages"
    try:
        response = client.post(url, json=_smoke_body(model, prompt, max_tokens))
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        return SmokeShown(
            reply=SmokeReply(lang=lang),
            problem=f"{url} に届かない ({type(exc).__name__})",
        )
    code = response.status_code
    if code != httpx.codes.OK:
        return SmokeShown(
            reply=SmokeReply(lang=lang, http_status=code),
            problem=f"{url} が {code} を返した",
        )
    try:
        payload: object = response.json()
    except ValueError:
        return SmokeShown(
            reply=SmokeReply(lang=lang, http_status=code),
            problem="応答を JSON として読めない",
        )
    if not isinstance(payload, dict):
        return SmokeShown(
            reply=SmokeReply(lang=lang, http_status=code),
            problem="応答が JSON の object でない",
        )
    text, problem = _reply_text(payload)
    stop_reason = payload.get("stop_reason")
    return SmokeShown(
        reply=SmokeReply(
            lang=lang,
            http_status=code,
            stop_reason=stop_reason if isinstance(stop_reason, str) else None,
            input_tokens=_usage_count(payload.get("usage"), "input_tokens"),
            output_tokens=_usage_count(payload.get("usage"), "output_tokens"),
            replacement_char=_REPLACEMENT_CHAR in text,
        ),
        text=text,
        problem=problem,
    )


def _shown_count(value: int | None) -> str:
    return "不明" if value is None else str(value)


def _smoke_facts(shown: SmokeShown) -> str:
    """1 つの応答の、**保存してよい事実だけ**を並べる (本文を入れない)。"""
    reply = shown.reply
    parts = ["届かなかった" if reply.http_status is None else f"HTTP {reply.http_status}"]
    if reply.stop_reason is not None:
        parts.append(f"終わりの理由 {reply.stop_reason}")
    parts.append(
        f"入力 {_shown_count(reply.input_tokens)} / 出力 {_shown_count(reply.output_tokens)}"
        " トークン"
    )
    if reply.replacement_char:
        parts.append("置き換え文字 (U+FFFD) を含む")
    if shown.problem:
        parts.append(shown.problem)
    return "、".join(parts)


def _shown_smoke(shown: SmokeShown) -> str:
    """1 つの応答を、画面に出す文にする (**本文が出るのは、ここだけ**)。"""
    lines = [f"--- 短い要求 ({shown.reply.lang}) ---", _smoke_facts(shown)]
    if shown.text:
        lines.append(shown.text)
    return "\n".join(lines)


def _smoke_detail(shown: Sequence[SmokeShown]) -> str:
    """結果の文 (**応答の本文を入れない**。requirements 10.5)。"""
    facts = " / ".join(f"{item.reply.lang}: {_smoke_facts(item)}" for item in shown)
    return (
        f"英語と日本語の短い要求を 1 つずつ送った ({facts})。応答の本文は画面にだけ出し、"
        "記録には残さない。意味の通る文かどうかは、計測者が画面で読んで判断する"
    )


def smoke(
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    *,
    client: httpx.Client | None = None,
    report: TextIO | None = None,
    timeout_s: float = SMOKE_TIMEOUT_S,
    max_tokens: int = SMOKE_MAX_TOKENS,
) -> SmokeOutcome:
    """英語と日本語の短い要求を 1 つずつ送る (`serve smoke <構成>`。requirements 6.5)。

    宛先は head の `/v1/messages` で、Spark の状態は変えない (HTTP だけで、遠隔の実行役を
    使わない)。**応答の本文は `report` (画面) にだけ出し、返り値にも、どのファイルにも
    書かない** (requirements 10.5)。

    引数:
        config: `kind = "serve"` の構成 (`config.select_config` を通ったもの)。
        nodes: 役割ごとのノードの定義 (head の LAN のアドレスに送る)。
        client: HTTP のクライアント。省くと `new_client` で作り、この関数の中で閉じる。
        report: 応答の本文と事実の出し先。既定は `sys.stderr`。
        timeout_s: 1 つの要求の時間切れ (生成があるので、読み取りより長くする)。
        max_tokens: 応答の長さの上限 (小さくする)。

    返り値:
        言語ごとの `SmokeReply` (本文を持たない) と、事実を並べた `detail`。**終了コードは、
        5.1 が `replies` から決める** (すべて 200 なら 0、1 つでも違えば 2)。

    例外:
        config.ConfigError: `kind` が `serve` でない構成、HTTP のポートを決められない構成。
        ValueError: head のノードの定義がないとき。
    """
    if config.kind != _KIND_SERVE:
        raise ConfigError(
            f"短い要求で確かめられるのは、kind = '{_KIND_SERVE}' の構成だけである"
            f" (構成 '{config.name}' の kind は '{config.kind}')"
        )
    if config.served_model_name is None:  # pragma: no cover - 型の検証が、先に断る
        raise ConfigError(f"構成 '{config.name}' に served_model_name がない")
    head = _node_of(nodes, config, _HEAD)
    base_url = f"http://{head.lan_addr}:{http_port(config)}"
    stream = sys.stderr if report is None else report
    owns_client = client is None
    used = new_client(timeout_s) if client is None else client
    shown: list[SmokeShown] = []
    try:
        for lang, prompt in SMOKE_PROMPTS.items():
            sent = send_smoke(
                used,
                base_url,
                model=config.served_model_name,
                lang=lang,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            shown.append(sent)
            _report(stream, _shown_smoke(sent))
    finally:
        if owns_client:
            used.close()
    return SmokeOutcome(replies=tuple(item.reply for item in shown), detail=_smoke_detail(shown))
