"""2 台の間の通信の確認 (design.md 「確認 › netcheck」)。

4 つの口を持つ。1 つ目は読み取りだけで、残りの 3 つはコンテナを起こすので、了承が要る。

- `read_link_report` / `read_link_reports` (`serve netcheck links`。tasks.md 4.3):
  直結のインターフェースを読み、ケーブルの本数を判断する。**状態を変えない**
- `run_bandwidth` (`serve netcheck bandwidth`。tasks.md 4.4): `allreduce_bench.py` を 2 台で
  流し、大きさごとの `busbw` と、使われた経路を出す。**状態を変える (了承が要る)**
- `run_sanity` (`serve netcheck sanity`): 同じ形で `vllm_sanity_check.py` を流し、4 段の成否を
  出す。**状態を変える**
- `run_ab` (`serve netcheck ab`): 足す環境変数の A/B を、交互に 3 回ずつ流して、採否を出す。
  **状態を変える**

## 結末と例外の、終了コードへの写し方 (5.1 が、この表のとおりに写す)

| 結末 / 例外 | 終了コード | 理由 |
|---|---|---|
| `BandwidthOutcome` / `SanityOutcome` の `status` が `passed` | 0 | 合格 |
| `AbReport.status` が `compared` | 0 | 比べられた (採否は、どちらも正しい答え) |
| `status` が `refused` | 1 | 関門が断った (前提の不足) |
| `guards.ApprovalError` | 1 | 計測者が了承しなかった |
| `config.ConfigError` (`plan.PlanError`)、`ValueError` | 1 | 構成と引数の誤り (触る前に断る) |
| `status` が `failed` | 2 | **不合格** (経路、4 段、確かめられなかった、時間切れ、0 以外の終了) |
| `AbReport.status` が `stopped` | 2 | 途中の回で止まった (6 回そろわなかった) |
| `NetcheckError` | 2 | 合格なのに、片付けが終わらなかった (コンテナが残りうる) |
| `KeyboardInterrupt` | 130 | 中断 (片付けてから伝える) |

**「確かめられなかった」を合格にしない** (requirements 4.4、4.7)。NCCL の記録を回収できない、
経路の行が読めない、結果の JSON が読めない、のどれでも `failed` にする。手順書は、合格でなければ
2 台での起動に進まない (design.md 「netcheck」の合否)。

## `links` — 直結のインターフェースの読み取り (tasks.md 4.3)

2 台の Spark それぞれで、**読み取りだけ**を行う (`runner.run(..., mutating=False)`。状態を
変える呼び出しを 1 つも出さない。了承も要らない):

1. `ip -br link` でインターフェースの名前と状態、`ip -br addr` でアドレスを読む
2. 管理の側 (ノードの定義の `lan_addr` が付いているインターフェース)、ループバック、
   `docker0` / `br-*` / `veth*` / `tailscale0` / 無線 (`wl*`) を除いた残りを、直結の候補に
   する (下の「候補の決め方」)
3. 候補それぞれについて、MTU (`cat /sys/class/net/<if>/mtu`)、速さと接続 (`ethtool <if>`)
   を読む
4. `ibdev2netdev` で、候補と RoCE のデバイスの対応を読む。あれば `ibv_devinfo` も読み、
   `ibdev2netdev` が挙げたデバイスが、そこにも現れることを確かめる (裏付けの読み取り。
   `ibv_devinfo` の GUID や `node_guid` は、機械を識別できる値なので、結果には入れない)
5. つながっている候補の数から、ケーブルの本数を判断する (下の「ケーブルの本数の判断」)

**入っていない道具** (`ibdev2netdev` / `ibv_devinfo` / `ethtool` が `command not found` =
終了コード 127) は、入れずに `LinkReport.tools_missing` に記録し、読み取りと判断を続ける
(requirements 4.1、design.md 「netcheck」)。**`ethtool` の標準エラーの
`netlink error: Operation not permitted` は、誤りとして扱わない** (終了コードは 0 のまま。
tasks.md の Implementation Notes 「1.6 の見本の追加」: root でなくても `ethtool` は 0 で
終わる)。片方の台に入れない (`remote.RemoteError`)、コマンドが 0 以外で終わる、出力が
読めないのそれぞれでも、落ちずに、その台のその項目を「読めなかった」に倒す。

## 本数を確定する条件 (安全の側に倒す)

**本数を確定するのは、2 台それぞれで、(i) インターフェースの一覧、(ii) アドレス (管理の側の
見分け)、(iii) つながりの判定、が、すべて読めたときだけである。読めなかったものがあれば、
読めたところまでを `detail` に示して、判断しない。**

とくに (ii) は、単に読めれば足りるのではなく、**`node.lan_addr` に一致するインターフェースを
実際に 1 つ以上見つけられたときだけ**「見分けられた」とする。`ip -br addr` が読めない
(`RemoteError`、0 以外の終了、空の出力) と、この module の管理の側の見分け方 (`_management_names`。
名前を決め打ちせず、`lan_addr` に一致するアドレスだけで見分ける) が働かず、管理の側のインター
フェースが、直結の候補に紛れ込みかねない (差し戻しの原因になった不具合: 直結の側が実際には
1 つしかつながっていない状況で、管理の側が「つながっている候補」に数えられ、`cable_count=1`
という、誤った確定の結果が返っていた)。`lan_addr` が、どのインターフェースにも付いていない
台 (ノードの定義の誤り、または別の経路で入っている) も、同じ理由で見分けの前提が崩れている
ので、同じ扱いにする。このいずれかに当たる台は、`LinkReport.interfaces` を空にし、
`cable_count` も「埋める候補」も出さない (`_unconfirmed_report`)。

## 候補の決め方

`ibdev2netdev` に現れるものを直結の候補にするのが素直だが、**`ibdev2netdev` が入っていない
台でも判断できるように**、候補は `ibdev2netdev` の有無に関わらず、構造 (名前と、管理の側の
アドレス) だけで決める。読み取れた `ibdev2netdev` は、決まった候補に RoCE のデバイスを
対応づけるためだけに使う (tasks.md 4.3 の要点)。

DGX Spark の物理のインターフェースは、NVIDIA の DGX Spark ユーザーガイド (ConnectX-7
Networking) の対応表 (research.md §e-1) のとおり、管理の 1 つ (10GbE。ノードの定義の
`lan_addr`) と、直結の側の 4 つ (`en{p1s0f0,P2p1s0f0,p1s0f1,P2p1s0f1}np{0,1}`) である。
仮想のインターフェース (`docker0`、`br-*`、`veth*`、`tailscale0`) と無線 (`wl*`) は、直結に
使わない。管理の側は、名前ではなく `lan_addr` に一致するアドレスで見分ける (実機のインター
フェースの名前 (`enP7s7`) を、この module が決め打ちしないため)。これらを除いた残りを候補と
する。

## ケーブルの本数の判断

NVIDIA の DGX Spark ユーザーガイドの原文 (research.md §e-1): **"Each QSFP port appears as
two independent Linux Ethernet interfaces. As a result, plugging in two cables shows a total
of four Linux Ethernet interfaces."** ここから、つながっている直結の候補が **2 つならケーブル
1 本、4 つなら 2 本**と判断する (tasks.md 4.3)。0、1、3、それ以外の数のときは、本数を決め
つけず、`cable_count` を空にして、`detail` に「判断できない」旨と、数えたインターフェースを
書く。2 台の判断が食い違うとき (`read_link_reports` が両方を読んだとき) は、両台の `detail`
に、そのことを書き添える。

**つながっているかどうかの判定は、`lifecycle.parse_link_state` を使い回す** (二重に持たない。
tasks.md の Implementation Notes 3.5)。`ip -br link` の 2 列目 (`UP` / `DOWN`) を主に使い、
`ethtool` の `Link detected` で裏付ける。**この 2 つの出どころが、両方読めて食い違うとき
(例: `ip` は `UP`、`ethtool` は `Link detected: no`) は、その候補のつながりが確定できない
ので、台全体のケーブルの本数を判断しない** (`connectivity_uncertain`)。`ethtool` が読めず、
`ip` の状態だけで判断した候補は、そのことを `detail` に書く (どちらで判断したかを、つねに
たどれるようにする)。`lifecycle.read_fabric_link` は、1 つの名前ごとに
`ip -br link show dev <名前>` を遠隔で流すので、ここでは使わない (直結の候補は複数あるので、
1 回だけ読んだ `ip -br link` の出力を、`parse_link_state` で名前ごとに読み直すほうが、遠隔の
呼び出しが少ない)。

## ノードの定義との突き合わせ

`node.fabric_ifname` / `fabric_addr` が埋まっていれば、読み取った結果と突き合わせ、名前が
見つからない・つながっていない・アドレスが違う、のそれぞれを `detail` に書く。**`fabric_ifname`
が、管理の側のインターフェースの名前を指しているとき (設定の誤り) は、「候補の一覧にない」
という汎用の文ではなく、「管理の側のインターフェースである」と言う。** `fabric_ifname` が
空であれば、つながっている候補の名前とアドレスを、埋める候補として `detail` に示す。
**この module は、ノードの定義のファイルを書き換えない** (7.3 が、この結果を見て、人が
`nodes.toml` を書く)。「本数を確定する条件」が崩れている台 (`_unconfirmed_report`) では、
`fabric_ifname` / `fabric_addr` の突き合わせそのものができないので、**「合っている」とは
言わず、「確かめられなかった」と言う**。

## 見せてよい値

要約に出すのは、名前・状態・MTU・速さ・アドレス・RoCE のデバイス名・ケーブルの本数だけ。
MAC アドレスや GUID など、アドレス以外の機械を識別できる値は、そもそも `InterfaceLink` に
持たせていないので、`format_link_report` にも出てこない (design.md 「netcheck」)。

## `bandwidth` / `sanity` / `ab` — 2 台のジョブ (tasks.md 4.4)

**ほかの種類 (`serve`、`probe`、`fetch`、`inspect`) と同じ 1 つの起動の仕組みを通る**
(design.md 「Architecture Integration」)。違うのは、待つ条件が「コンテナの終了」であることだけ
である。1 回の流れ:

1. 構成を確かめる (`kind = "job"`、2 台)。**Spark に触る前に**断る
2. `plan.build_plans` で 2 台ぶんの引数の列を組み立てる (A/B は、腕と回を渡す)
3. `guards.run_gates` を流す (読み取りだけ)。1 つでも断れば、`refused` で返す
4. `guards.build_approved_plan(plans=…)` で計画を作り、`guards.request_approval` で了承を得る。
   **ここより前に、状態を変える呼び出しは 1 つも出ない**
5. `lifecycle.start_all` で 2 台を `docker run -d` で起こし、`lifecycle.started_targets` で、
   自分のラベルで絞った一覧から識別子を取る
6. **2 台のコンテナの終了を待つ** (待ちの上限は、構成の `ready_timeout_s`。`timeout_s` で、その
   回だけ上書きできる)。時計と眠りは `lifecycle.Controls` から来るので、試験は実際に眠らない
7. コンテナの出力を読む (`docker logs <一覧の ID>`。標準出力と標準エラーを分けて読む)。**rank 0
   (構成の `nodes` の 1 つ目) の標準出力の 1 行の JSON** が、帯域の結果である (警告などの行が
   混ざっていてもよい)
8. 記録を回収し (`logs.collect_logs`。Spark の `logs/` の下の NCCL の記録を含む)、
   `observe.observe_nccl` で経路を読む
9. **どの結果でも、必ず止めて消す** (`lifecycle.clean_up`。3.4 の片付けの決まり)

### 待ちと片付けの、口の選び方

- 終了の判定は `docker container inspect --format '{{.State.Status}} {{.State.ExitCode}}'`
  (`remote` の許可の一覧にある読み取り) を、**一覧から来た識別子**に向けて行う。
  `lifecycle.wait_ready` は HTTP の受け付けを待つので、ジョブには使えず、`lifecycle` の側の
  同じ読み取りは下線つきの名前なので触らない (`image` と `lifecycle` にも同じ形があり、実物の
  出力の形が変わったら 3 か所を直す。CONCERNS に書いた)
- **ある台が 0 以外で終わった (または状態を読めなくなった) 時点で、ほかの台を待たずに打ち切る**。
  rendezvous のジョブでは、片方が死ねば、もう片方は固まるので、待ちの上限 (既定 1,800 秒) を
  空待ちすることになる (A/B の 6 回なら、最大 3 時間)。3.4 の「どちらかが終了したら、すぐ失敗」
  と同じ考え方である。0 で終わった台は、ほかの台を待つ
- 片付けは `lifecycle.clean_up` (止めてよい相手を、流す直前に自分の一覧で確かめる) を呼ぶ。
  回収は `logs.collect_logs` を直に呼ぶ。`lifecycle.wrap_up` を使わないのは、回収の置き場所の
  名前を `start` に決め打ちするので、**A/B の回ごとに置き場所を分けられない**ためである
  (Implementation Notes 2.4: 回収のたびに、その回のぶんを別の場所に置く)。置き場所の名前は
  `netcheck` (A/B は `netcheck-<腕>-<回>`) にする
- **起動の記録 (`state/<構成>.launch.json`) は置かない**。A/B は同じ構成で 6 回起こすので、
  1 つのファイルでは上書きになり、記録としての意味がない。どの回のものかは、回収した記録の
  置き場所 (`BandwidthRun.log_dir`) と、ラベル `vllm-baseline.run` で分かる。requirements 3.10
  は、推論サーバーの起動 (`lifecycle.start`) が満たす

### NCCL の記録は、回ごとに名前を分ける

**Spark の `<remote_root>/logs/` を消す道は、この道具のどこにもない** (`remote.push` は `logs/`
を宛先にできず、`rm` もない)。だから NCCL の記録は、回と呼び出しをまたいで積もり続ける。
`observe_nccl` の `network` は「最初に当たった行」なので、積もった記録をまとめて読むと、**前の
呼び出しの高速の経路の記録で、今回の失敗を隠してしまう**。そこで:

- この module が、**どの回にも** (A/B の最小の設定の腕にも、`bandwidth` / `sanity` の 1 回にも)
  `plan.build_plans(..., extra_env=…)` で `NCCL_DEBUG_FILE` を上書きする。値は、構成の
  `NCCL_DEBUG_FILE` の**ディレクトリ**の下の `nccl-<回の札>.%h.%p.log`。回の札は
  「起こす時刻 (UTC、**マイクロ秒まで**) + 腕 (1 回だけの確認は `sanity`) + 回の番号 (**2 桁に
  そろえる**)」で、英数字とハイフンだけ。`-e` は**あとに並ぶものが勝つ**ので、構成の値を
  上書きできる
- **記録が Spark に残らない構成は、Spark に触る前に断る** (`ConfigError`): `NCCL_DEBUG` が
  ない・`INFO` 以上でない (経路の行が出ない)、`NCCL_DEBUG_FILE` がない・絶対の道筋でない・
  その置き場所が `--mount` の `target` のどれの下でもない (コンテナが消えると記録も消える)、
  **その `--mount` が読み取り専用 (`readonly` / `ro`)** (NCCL が書けない。起こす前に断る)
- 読むのは、回収した `logs/` の下で、**名前が `nccl-<その回の札>.` で始まるファイルだけ**
  (前後を固定した一致。部分一致にすると、`…-baseline-01` が `nccl-…-baseline-10.….log` に
  当たる)。その回の札のファイルが 1 つもない台は「確かめられなかった」で、合格にしない
- **限界**: 同じ `started_at` を渡して 2 度流すと、札は同じになる。マイクロ秒まで入れてあるので、
  呼ぶ側 (5.1) が、**呼び出しごとに、その時の時刻を渡せば**分かれる (`logs.var_dir` の置き場所の
  名前は秒までなので、同じ秒の 2 度目は、同じ場所に回収される。Implementation Notes 2.4)
- 腕 A にもこの上書きが入るので、「最小の設定 = 構成のまま」ではなくなる。**変わるのは記録の
  ファイルの名前だけで、通信のふるまいには効かない** (NCCL の経路の選び方には関わらない)

### 合否 (requirements 4.4、4.7、design.md 「netcheck」の合否)

**構成のすべての台**について、次の連言を見る (research.md §e-4: 「IB から Socket に落ちた」と
明示する文字列は NCCL のソースにないので、判定は連言で行う):

1. `NcclObservation.network == "IB"` (高速の直結の経路が使われた)
2. `socket_channel_seen` が偽 (`[send] via NET/Socket` のチャンネルがない)
3. `ib_no_device` が偽 (`NET/IB : No device found.` が出ていない)
4. `Using network Socket` の行が 1 度も出ていない (**`observe_nccl` の `network` は、記録の
   最初の `Using network …` の行だけを見る**。`observe.py` は凍結されているので、`IB` →
   `Socket` の 2 行がある記録を、この module が本文 (`_SOCKET_NETWORK_LINE`) で補う)

そのうえで、

- **帯域**: rank 0 (構成の `nodes` の 1 つ目) の JSON の `world_size` が構成の台の数と一致し、
  `busbw` が 0 より大きい大きさが 1 つ以上読めること。`busbw` か `algbw` が 0 以下の項目は、
  測れていないものとして落とす
- **事前の確認**: 4 段 (PyTorch の NCCL、GLOO、vLLM の NCCL、CUDA グラフの中の NCCL) がすべて
  通ること。成功の文面は、`serving/payload/vllm_sanity_check.py` の本文が `print` する文字列
  そのもの (`SANITY_STAGES`。上流の原文なので、推測で書かない)。**各 rank が出す**ので、構成が
  使う台のすべての出力に出ていなければ、その段は通っていない
- どちらも、2 台のコンテナが終了コード 0 で終わっていること

記録が回収できない、経路の行が読めない、結果の JSON が読めない、のどれでも「確かめられなかった」
として、**合格にしない**。

### 大きさの列 (design の決まりと違うとき)

design の決まりの列は 1 MiB から 1 GiB まで 4 倍ずつの 6 つ (`DEFAULT_MESSAGE_SIZES`) だが、
A/B の短い回は `allreduce_bench.py` の `--max-bytes` などで縮められる (4.2)。この module は、
構成の引数を読み解かないので、**列が違うことは失敗にせず、結果に注記する** (公表の値と比べる
ときに読み違えないようにするため)。

### 比べる相手 (requirements 4.3、4.6)

`NVIDIA_REFERENCE` に 3 つを持ち、結果 (`BandwidthOutcome.comparison`) に、その値と、**道具が
違うこと**を書く。**どれも合否には入れない** (design.md 「netcheck」の合否は、経路と 4 段だけ。
道具が違うので、比べて示すだけである)。

| 値 | 道具 | 出どころ (research.md) |
|---|---|---|
| 189.85 Gbps | `ib_write_bw` (2 つの RoCE のデバイスの合計) | §e-1 の原文 |
| 184 Gbit/s | NVIDIA Sync のリンクの速さの試験 | §e-5 の原文 (URL つき) |
| 175 Gbps (= 21.875 GB/s) | nccl-tests の all_gather の `Avg bus bandwidth` | §e-5 (URL はない) |

こちらは自前の `allreduce_bench.py` の all-reduce の `busbw` なので、そのままの比較にはならない
(`ReferencePoint` は、**research.md が原文を引いていない値を持たない**)。

### A/B (requirements 4.5)

- 最小の設定 (腕 `baseline` = 構成のまま + 記録のファイルの名前の上書きだけ) と、足した設定
  (腕 `candidate` = `extra_env` つき) を、**交互に 3 回ずつ** (A1、B1、A2、B2、A3、B3) 流す。
  コンテナの名前は `vb-<構成>-<役割>-r<回>` で、ラベル `vllm-baseline.run` が `<腕>-<回>` になる
  (design.md 「netcheck」)
- **了承は 1 回**。6 回ぶん (12 個) の `docker run` と、その巻き戻しを、すべて並べた 1 つの計画を
  見せて、了承を得る (design.md 「remote」: A/B の繰り返しのすべての回を、了承の前に並べる)。
  腕の違う同じ回の番号は、同じ名前のコンテナになるが、**1 回ごとに片付けてから次に進む**ので、
  同時には存在しない (design.md 「netcheck」)。前の回の片付けが終わらなかったときは、次の回に
  進まずに止まる (名前の衝突を起こさない)
- 採否: **足した側の 3 回の最小が、最小の設定の側の 3 回の最大を上回ったときだけ「採用できる」**。
  比べる値は、6 回に共通する、いちばん大きいメッセージの `busbw`
- **経路 (IB か Socket か) は、採否に使わない**。要件 4.5 と design の A/B の決まりは、速さだけで
  決める。コンテナの中で高速の経路が使えないときに、足す候補 (デバイスの受け渡しなど) を A/B で
  確かめる、という使い方があるので、経路が遅い回を、採否の対象から外さない。そのかわり:
  - **回ごと・台ごとの経路の観察を `AbReport.routes` に残す** (`runs` と同じ順。要件 4.4 が、
    速さを測ったときに、どの経路が使われたかの記録を求める。7.4 が、判断の記録に書く)
  - **どれかの台で、その回の札の記録が読めなかった回は、値を比較に使わない**。その回を
    「経路を確かめられなかった」失敗として扱い、残りの回を流さずに止まる (`world_size` の
    食い違いと同じ扱い)。経路の証拠のない値で、採否を出さない
  - 採否を出すときは、`AbReport.detail` に**腕ごと・台ごとの経路のまとめ**を必ず書き、高速の
    経路でない回があれば、**採否の文のすぐそばに注意**を置く (採否は変えない)
- 途中の回が失敗したら、その回を片付けて、残りの回を流さずに、そこまでの結果と、どこで止まったか
  を返す (`AbReport.status` が `stopped`)。中断は、片付けのあとで伝える
- **`extra_env` の値は、置き換えの印を埋めず、全ノードに同じ値が渡る** (Implementation Notes
  2.1)。ノードごとに違う値が要る A/B (たとえば、2 台でインターフェースの名前が違うとき) は、
  この口では表せない。そのときは `plan` の変更を先に行うか、別の名前の構成として書く
- **`NCCL_DEBUG_FILE` と `NCCL_DEBUG` は、`extra_env` に書けない** (この module が、回ごとの
  記録のために使う)。記録の場所を変えたいときは、構成の `env` を直す
- **恒久的な設定は変えない** (requirements 2.8)。`--device` や `--cap-add` のような docker の
  設定は `extra_env` では足せないので、別の名前の構成として書く (6.2 / 7.4)。この module は、
  構成もノードの定義も書き換えない

### 6.2 への申し送り (ジョブの構成に要るもの)

- `NCCL_DEBUG = "INFO"` (または `TRACE`)、`NCCL_DEBUG_SUBSYS = "INIT,BOOTSTRAP,ENV,NET,GRAPH"`
  (research.md §e-3。既定では `NET` と `GRAPH` が出ない)
- `NCCL_DEBUG_FILE` を、**`logs/` を結び付けた `--mount` の `target` の下**の絶対の道筋にする
  (例: `<target>/nccl_%h_%p.log`)。**ファイルの名前の部分は、この module が回ごとに上書きする**
  ので、構成の値は「置き場所を決めるため」に読まれる
- `logs/` の `--mount` は、**読み取り専用 (`readonly` / `ro`) にしない** (NCCL が記録を書けない。
  この module が、起こす前に断る)。rendezvous のポートは `is_port = True` の設定として明示する
  こと (関門が、その番号が空いているかを見る)

## 依存の向きと、書かないもの

`netcheck` は `types`、`config`、`remote`、`plan`、`observe`、`guards`、`logs`、`lifecycle`
まで読み込める。`probe`、`watch`、`thinking`、`cli` (同じ層、または入口) は読み込まない。
`lifecycle` は**公開の口だけ**を使い、下線つきの名前を 1 つも触らない。`cli.py` へのつなぎ込みは
5.1 の仕事である。
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, TextIO

import httpx

from serving_kit import lifecycle
from serving_kit.config import ConfigError
from serving_kit.guards import (
    READ_TIMEOUT_S,
    Confirmer,
    build_approved_plan,
    request_approval,
    run_gates,
)
from serving_kit.lifecycle import parse_link_state
from serving_kit.logs import COMM_LOG_REMOTE_SUBDIR, LOG_READ_TIMEOUT_S, collect_logs
from serving_kit.observe import observe_nccl
from serving_kit.plan import build_plans
from serving_kit.remote import RemoteError, RemoteRunner
from serving_kit.types import (
    AbOutcome,
    BandwidthRun,
    BandwidthSample,
    CommandResult,
    ConfigDef,
    ContainerPlan,
    GateResult,
    InterfaceLink,
    LinkReport,
    NcclObservation,
    NodeDef,
    NodeRole,
    Setting,
)

__all__ = [
    "AB_REPEATS",
    "BASELINE_ARM",
    "CANDIDATE_ARM",
    "COMMAND_NAME",
    "DEFAULT_MESSAGE_SIZES",
    "JOB_POLL_INTERVAL_S",
    "NVIDIA_REFERENCE",
    "SANITY_STAGES",
    "AbReport",
    "BandwidthOutcome",
    "BandwidthReference",
    "JobEnd",
    "NetcheckError",
    "ReferencePoint",
    "RoundRoute",
    "SanityOutcome",
    "SanityStage",
    "format_link_report",
    "read_link_report",
    "read_link_reports",
    "run_ab",
    "run_bandwidth",
    "run_sanity",
]


# --- 候補の決め方 -----------------------------------------------------------

_EXCLUDED_NAMES: Final[frozenset[str]] = frozenset({"lo", "docker0", "tailscale0"})
"""名前がちょうど一致すれば、直結の候補から除くもの。"""

_EXCLUDED_PREFIXES: Final[tuple[str, ...]] = ("br-", "veth", "wl")
"""名前がこの接頭辞で始まれば、直結の候補から除くもの (ブリッジ、veth の対、無線)。"""

_MISSING_TOOL_EXIT_CODE: Final[int] = 127
"""`command not found` の終了コード (道具が入っていないことの印。tasks.md 4.3 の要点)。"""

_STATE_UP: Final[str] = "UP"
_STATE_DOWN: Final[str] = "DOWN"
_STATE_UNKNOWN: Final[str] = "UNKNOWN"
"""`InterfaceLink.state` に書く文字列 (`ip -br link` の 2 列目と同じ語)。"""

_CABLE_COUNT_BY_CONNECTED: Final[Mapping[int, int]] = {2: 1, 4: 2}
"""つながっている候補の数から、ケーブルの本数へ (research.md §e-1。module docstring の
「ケーブルの本数の判断」)。"""

_ETHTOOL_SPEED_RE: Final[re.Pattern[str]] = re.compile(r"^\s*Speed:\s*(\d+)Mb/s\s*$", re.MULTILINE)
_ETHTOOL_LINK_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*Link detected:\s*(yes|no)\b", re.MULTILINE
)
_IBDEV2NETDEV_RE: Final[re.Pattern[str]] = re.compile(
    r"^(\S+)\s+port\s+\d+\s+==>\s+(\S+)\s+\((?:Up|Down)\)\s*$", re.MULTILINE
)
_HCA_ID_PREFIX: Final[str] = "hca_id:"


def _is_excluded_by_name(name: str) -> bool:
    """名前だけで、直結の候補から除けるかどうか (管理の側は、アドレスで別に見る)。"""
    return name in _EXCLUDED_NAMES or name.startswith(_EXCLUDED_PREFIXES)


def _management_names(node: NodeDef, addr_by_name: Mapping[str, tuple[str, ...]]) -> frozenset[str]:
    """ノードの定義の `lan_addr` が付いているインターフェースの名前。"""
    lan_addr = str(node.lan_addr)
    return frozenset(
        name
        for name, addrs in addr_by_name.items()
        if any(addr.split("/", 1)[0] == lan_addr for addr in addrs)
    )


def _candidate_names(
    names: Sequence[str], node: NodeDef, addr_by_name: Mapping[str, tuple[str, ...]]
) -> list[str]:
    """直結の候補の名前を、読み取った順のまま選ぶ (module docstring の「候補の決め方」)。"""
    management = _management_names(node, addr_by_name)
    return [name for name in names if name not in management and not _is_excluded_by_name(name)]


# --- `ip -br` の出力の読み取り -----------------------------------------------


def _link_names(text: str) -> list[str]:
    """`ip -br link` の出力から、インターフェースの名前を、現れた順のまま拾う (重複なし)。

    `veth0000000@if2` のような対の印は、`lifecycle.parse_link_state` と同じく `@` の前まで
    で見る。
    """
    names: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("@", 1)[0]
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _addr_rows(text: str) -> dict[str, tuple[str, ...]]:
    """`ip -br addr` の出力から、名前ごとのアドレスの列を読む (アドレスがなければ空)。"""
    rows: dict[str, tuple[str, ...]] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("@", 1)[0]
        rows[name] = tuple(fields[2:])
    return rows


def _link_state_label(linked: bool | None) -> str:
    """`lifecycle.parse_link_state` の判定を、`InterfaceLink.state` の文字列にする。"""
    if linked is True:
        return _STATE_UP
    if linked is False:
        return _STATE_DOWN
    return _STATE_UNKNOWN


# --- `ethtool` / `ibdev2netdev` / `ibv_devinfo` の出力の読み取り -----------------


def _parse_ethtool(text: str) -> tuple[int | None, bool | None]:
    """`ethtool <if>` から、ネゴシエートした速さ (Mb/s) と、接続の有無を読む。

    つながっていないと `Speed: Unknown!` になるので、その場合は空を返す (見本
    `tests/fixtures/spark/*/ethtool-enp1s0f1np1.txt`)。
    """
    speed_match = _ETHTOOL_SPEED_RE.search(text)
    speed = int(speed_match.group(1)) if speed_match else None
    link_match = _ETHTOOL_LINK_RE.search(text)
    link_detected = link_match.group(1) == "yes" if link_match else None
    return speed, link_detected


def _parse_ibdev2netdev(text: str) -> dict[str, str]:
    """`ibdev2netdev` の出力から、インターフェースの名前と RoCE のデバイスの対応を読む。

    1 行の形は `<RoCE のデバイス> port <番号> ==> <インターフェース> (Up|Down)`
    (見本 `tests/fixtures/spark/*/ibdev2netdev.txt`)。`(Up|Down)` は、ここでは使わない
    (つながっているかどうかは `ip -br link` で決める。module docstring の「ケーブルの本数の
    判断」)。
    """
    return {ifname: device for device, ifname in _IBDEV2NETDEV_RE.findall(text)}


def _parse_ibv_devinfo_hca_ids(text: str) -> frozenset[str]:
    """`ibv_devinfo` の出力から、`hca_id` の名前の集まりを読む (裏付けの読み取りに使う)。"""
    ids: set[str] = set()
    for line in text.splitlines():
        if line.startswith(_HCA_ID_PREFIX):
            name = line[len(_HCA_ID_PREFIX) :].strip()
            if name:
                ids.add(name)
    return frozenset(ids)


# --- 遠隔の読み取りの小さな助け ----------------------------------------------


def _shown(result: CommandResult) -> str:
    """失敗した呼び出しの、見せる文 (標準エラーがなければ標準出力)。"""
    return result.stderr.strip() or result.stdout.strip() or "(出力なし)"


def _run_read(
    runner: RemoteRunner, node: NodeDef, argv: tuple[str, ...], timeout_s: float
) -> tuple[CommandResult | None, str]:
    """読み取りを 1 つ流す。届かなかったこと (`RemoteError`) は、投げずに理由の文字列で返す。"""
    try:
        return runner.run(node, argv, timeout_s=timeout_s, mutating=False), ""
    except RemoteError as exc:
        return None, str(exc)


def _read_tool(
    runner: RemoteRunner,
    node: NodeDef,
    argv: tuple[str, ...],
    timeout_s: float,
    *,
    tool_label: str,
    problems: list[str],
    tools_missing: set[str],
) -> CommandResult | None:
    """入っていないかもしれない道具 (`ibdev2netdev` / `ibv_devinfo` / `ethtool`) を読む。

    終了コード 127 (`command not found`) は「入っていない」として `tools_missing` に記録し、
    誤りにしない (requirements 4.1)。それ以外の失敗は「読めなかった」として `problems` に
    書く。
    """
    result, err = _run_read(runner, node, argv, timeout_s)
    if result is None:
        problems.append(f"{tool_label} に届かない ({err})")
        return None
    if result.exit_code == _MISSING_TOOL_EXIT_CODE:
        tools_missing.add(tool_label)
        return None
    if result.exit_code != 0:
        problems.append(f"{tool_label} を読めなかった ({_shown(result)})")
        return None
    return result


def _read_mtu(
    runner: RemoteRunner, node: NodeDef, name: str, timeout_s: float, problems: list[str]
) -> int | None:
    """`cat /sys/class/net/<if>/mtu` で MTU を読む (`remote` の許可の形を通る)。"""
    result, err = _run_read(runner, node, ("cat", f"/sys/class/net/{name}/mtu"), timeout_s)
    if result is None:
        problems.append(f"{name}: MTU に届かない ({err})")
        return None
    if result.exit_code != 0:
        problems.append(f"{name}: MTU を読めなかった ({_shown(result)})")
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        problems.append(f"{name}: MTU の出力を数として読めない ({result.stdout.strip()!r})")
        return None


def _read_ethtool(
    runner: RemoteRunner,
    node: NodeDef,
    name: str,
    timeout_s: float,
    *,
    problems: list[str],
    tools_missing: set[str],
) -> tuple[int | None, bool | None]:
    """`ethtool <if>` で速さと接続を読む。

    標準エラーの `netlink error: Operation not permitted` は誤りにしない (`_read_tool` は
    終了コードしか見ず、標準エラーの中身を検査しない。tasks.md の Implementation Notes
    「1.6 の見本の追加」)。
    """
    result = _read_tool(
        runner,
        node,
        ("ethtool", name),
        timeout_s,
        tool_label="ethtool",
        problems=problems,
        tools_missing=tools_missing,
    )
    if result is None:
        return None, None
    return _parse_ethtool(result.stdout)


# --- ケーブルの本数の判断、ノードの定義との突き合わせ -----------------------------


def _judge_cable_count(connected_names: Sequence[str]) -> tuple[int | None, str]:
    """つながっている候補の数から、ケーブルの本数を判断する (module docstring の該当節)。"""
    count = len(connected_names)
    named = "、".join(connected_names) if connected_names else "なし"
    basis = (
        f"つながっている直結の候補 {count} 個 ({named})。NVIDIA の DGX Spark ユーザーガイド "
        "(ConnectX-7 Networking): 1 つの QSFP ポートが、2 つの Linux のインターフェースとして"
        "見える (research.md §e-1)"
    )
    cables = _CABLE_COUNT_BY_CONNECTED.get(count)
    if cables is None:
        return None, (
            f"ケーブルの本数を判断できない ({basis}。2 個なら 1 本、4 個なら 2 本と判断できるが、"
            f"{count} 個では決めつけない)"
        )
    return cables, f"ケーブルは {cables} 本と判断した ({basis})"


def _fabric_note(
    node: NodeDef, interfaces: Sequence[InterfaceLink], management_names: frozenset[str]
) -> str:
    """ノードの定義の `fabric_ifname` / `fabric_addr` と、読み取った結果を突き合わせる。

    埋まっていれば食い違いを、空であれば埋める候補を返す (module docstring の「ノードの定義
    との突き合わせ」。この関数はノードの定義を書き換えない)。`fabric_ifname` が管理の側の
    名前を指しているとき (設定の誤り) は、「候補の一覧にない」という汎用の文ではなく、
    「管理の側のインターフェースである」と言う (レビューの指摘 3)。
    """
    by_name = {link.name: link for link in interfaces}
    if node.fabric_ifname is not None:
        if node.fabric_ifname in management_names:
            return (
                f"fabric_ifname ('{node.fabric_ifname}') は、管理の側のインターフェースで"
                "ある (直結には使えない)"
            )
        found = by_name.get(node.fabric_ifname)
        if found is None:
            return (
                f"ノードの定義の fabric_ifname ('{node.fabric_ifname}') に当たる直結の候補が、"
                "読み取った一覧にない"
            )
        mismatches: list[str] = []
        if found.state != _STATE_UP:
            mismatches.append(f"つながっていない (状態: {found.state})")
        if node.fabric_addr is not None:
            expected = str(node.fabric_addr)
            if not any(addr.split("/", 1)[0] == expected for addr in found.addrs):
                mismatches.append(f"アドレスが fabric_addr ('{expected}') と違う")
        if mismatches:
            return (
                f"fabric_ifname ('{node.fabric_ifname}') が、読み取った結果と食い違う: "
                + "、".join(mismatches)
            )
        return f"fabric_ifname ('{node.fabric_ifname}') は、読み取った結果と合っている"
    connected = [link for link in interfaces if link.state == _STATE_UP]
    if not connected:
        return "fabric_ifname が空で、つながっている直結の候補もない (埋める候補を示せない)"
    candidates_text = "、".join(
        f"{link.name} ({link.addrs[0]})" if link.addrs else link.name for link in connected
    )
    return f"fabric_ifname が空。埋める候補: {candidates_text}"


def _unconfirmed_report(node: NodeDef, reason: str) -> LinkReport:
    """管理の側を見分けられなかったときの結果 (レビューの指摘 1。安全の側に倒す)。

    `ip -br addr` が読めない、または `lan_addr` に一致するインターフェースが 1 つも見つから
    ないと、管理の側を、名前ではなくアドレスで見分けるこの module の仕組みが働かない。その
    状態で候補を決めると、管理の側のインターフェースが、直結の候補に紛れ込みかねない
    (レビュー担当の再現: 直結の側が実際には 1 つしかつながっていないのに、管理の側が候補に
    数えられて `cable_count=1` という、誤った確定の結果が返っていた)。そこで、直結の候補
    (`interfaces`)、「埋める候補」、ケーブルの本数のどれも出さない。`fabric_ifname` /
    `fabric_addr` の突き合わせも、確かめられない (レビューの指摘 2: 「合っている」とは
    言わない)。
    """
    parts = [reason]
    if node.fabric_ifname is not None:
        parts.append(f"fabric_ifname ('{node.fabric_ifname}') との突き合わせも確かめられなかった")
    return LinkReport(node=node.role, detail=" / ".join(parts))


# --- 公開の口 ----------------------------------------------------------------


def read_link_report(
    runner: RemoteRunner, node: NodeDef, *, timeout_s: float = READ_TIMEOUT_S
) -> LinkReport:
    """1 台ぶんの直結のリンクを読み取る (`serve netcheck links` の中身。requirements 4.1、4.2)。

    読み取りだけである (状態を変える呼び出しを 1 つも出さない。了承も要らない)。**本数を確定
    するのは、(i) インターフェースの一覧、(ii) アドレス (管理の側の見分け)、(iii) つながりの
    判定、の 3 つが、すべて読めたときだけである** (module docstring の「本数を確定する条件」。
    レビューの指摘 1 を受けた決まり)。どれか 1 つでも読めなければ、読めたところまでを
    `detail` に示して、判断しない。
    """
    problems: list[str] = []
    tools_missing: set[str] = set()

    # (i) インターフェースの一覧
    link_result, link_err = _run_read(runner, node, ("ip", "-br", "link"), timeout_s)
    if link_result is None:
        return LinkReport(
            node=node.role,
            detail=f"{node.role} ({node.ssh_host}) の ip -br link に届かない ({link_err})",
        )
    if link_result.exit_code != 0:
        return LinkReport(
            node=node.role, detail=f"ip -br link を読めなかった ({_shown(link_result)})"
        )

    # (ii) アドレス (管理の側の見分け)。読めない、または管理の側が 1 つも見つからないと、
    # 直結の候補を安全に決められない (レビュー担当の再現: 管理の側が候補に紛れ込み、誤った
    # cable_count が確定していた)。そこで、直結の候補もケーブルの本数も出さずに終える
    addr_result, addr_err = _run_read(runner, node, ("ip", "-br", "addr"), timeout_s)
    if addr_result is None:
        return _unconfirmed_report(
            node,
            "ip -br addr に届かないので、管理の側のインターフェースを見分けられず、"
            f"直結の候補もケーブルの本数も判断しない ({addr_err})",
        )
    if addr_result.exit_code != 0:
        return _unconfirmed_report(
            node,
            "ip -br addr を読めなかったので、管理の側のインターフェースを見分けられず、"
            f"直結の候補もケーブルの本数も判断しない ({_shown(addr_result)})",
        )

    addr_by_name = _addr_rows(addr_result.stdout)
    management_names = _management_names(node, addr_by_name)
    if not management_names:
        return _unconfirmed_report(
            node,
            f"ノードの定義の lan_addr ('{node.lan_addr}') に一致するインターフェースが"
            "見つからないので、管理の側を見分けられず、直結の候補もケーブルの本数も判断しない"
            " (見分けの前提が崩れている)",
        )

    names = _link_names(link_result.stdout)
    candidates = _candidate_names(names, node, addr_by_name)

    roce_by_name: dict[str, str] = {}
    ibdev_result = _read_tool(
        runner,
        node,
        ("ibdev2netdev",),
        timeout_s,
        tool_label="ibdev2netdev",
        problems=problems,
        tools_missing=tools_missing,
    )
    if ibdev_result is not None:
        roce_by_name = _parse_ibdev2netdev(ibdev_result.stdout)

    hca_ids: frozenset[str] = frozenset()
    ibv_result = _read_tool(
        runner,
        node,
        ("ibv_devinfo",),
        timeout_s,
        tool_label="ibv_devinfo",
        problems=problems,
        tools_missing=tools_missing,
    )
    if ibv_result is not None:
        hca_ids = _parse_ibv_devinfo_hca_ids(ibv_result.stdout)

    interfaces: list[InterfaceLink] = []
    connected_names: list[str] = []
    # (iii) つながりの判定。`ip` の状態と `ethtool` の `Link detected` の、どちらで判断したか
    # を `detail` に残す。両方読めて食い違うときは、この台のケーブルの本数を確定しない
    connectivity_uncertain = False
    for name in candidates:
        linked = parse_link_state(link_result.stdout, name)
        state = _link_state_label(linked)
        mtu = _read_mtu(runner, node, name, timeout_s, problems)
        speed, link_detected = _read_ethtool(
            runner, node, name, timeout_s, problems=problems, tools_missing=tools_missing
        )
        if link_detected is None:
            problems.append(
                f"{name}: ethtool の Link detected を読めなかったので、ip の状態 ({state}) "
                "だけでつながりを判断した"
            )
        elif linked is not None and link_detected != linked:
            connectivity_uncertain = True
            problems.append(
                f"{name}: ip の状態 ({state}) と ethtool の Link detected "
                f"({'yes' if link_detected else 'no'}) が食い違う"
            )
        roce_device = roce_by_name.get(name)
        if roce_device is not None and hca_ids and roce_device not in hca_ids:
            problems.append(
                f"{name}: ibdev2netdev が挙げる RoCE のデバイス ('{roce_device}') が、"
                "ibv_devinfo の一覧にない"
            )
        interfaces.append(
            InterfaceLink(
                name=name,
                state=state,
                mtu=mtu,
                speed_mbps=speed,
                addrs=addr_by_name.get(name, ()),
                roce_device=roce_device,
            )
        )
        if linked is True:
            connected_names.append(name)

    cable_count: int | None
    if connectivity_uncertain:
        cable_count = None
        problems.append(
            "つながりの判定 (ip の状態と ethtool の Link detected) が食い違うインターフェース"
            "があるので、ケーブルの本数を判断しない"
        )
    else:
        cable_count, cable_note = _judge_cable_count(connected_names)
        problems.append(cable_note)
    problems.append(_fabric_note(node, interfaces, management_names))

    return LinkReport(
        node=node.role,
        interfaces=tuple(interfaces),
        cable_count=cable_count,
        tools_missing=tuple(sorted(tools_missing)),
        detail=" / ".join(text for text in problems if text),
    )


def read_link_reports(
    runner: RemoteRunner, nodes: Mapping[NodeRole, NodeDef], *, timeout_s: float = READ_TIMEOUT_S
) -> dict[NodeRole, LinkReport]:
    """2 台ぶんの直結のリンクを読み取る (`read_link_report` を役割ごとに呼ぶ)。

    2 台の判断 (`cable_count`) が食い違うときは、両台の `detail` に、そのことを書き添える
    (tasks.md 4.3 の要点)。
    """
    reports = {
        role: read_link_report(runner, node, timeout_s=timeout_s) for role, node in nodes.items()
    }
    return _note_cable_count_mismatch(reports)


def _note_cable_count_mismatch(
    reports: Mapping[NodeRole, LinkReport],
) -> dict[NodeRole, LinkReport]:
    """2 台の `cable_count` が食い違うとき、両台の `detail` にそのことを書き添える。"""
    counted = {
        role: report.cable_count
        for role, report in reports.items()
        if report.cable_count is not None
    }
    if len({*counted.values()}) <= 1:
        return dict(reports)
    updated: dict[NodeRole, LinkReport] = {}
    for role, report in reports.items():
        others = "、".join(
            f"{other_role}: {other_count} 本"
            for other_role, other_count in sorted(counted.items())
            if other_role != role
        )
        mine = f"{report.cable_count} 本" if report.cable_count is not None else "判断できない"
        note = f"2 台の判断が食い違っている ({role}: {mine}、{others})"
        updated[role] = report.model_copy(
            update={"detail": " / ".join(text for text in (report.detail, note) if text)}
        )
    return updated


def format_link_report(report: LinkReport) -> str:
    """`LinkReport` を、人が読める形にする (5.1 が画面に、7.3 が `docs/results/` の要約に使う
    助け)。アドレス以外の機械を識別できる値 (MAC、GUID) は、そもそも `InterfaceLink` に
    持たせていないので、ここにも出てこない。
    """
    lines = [f"{report.node}:"]
    if not report.interfaces:
        lines.append("  直結の候補のインターフェースを読めなかった")
    for link in report.interfaces:
        speed = f"{link.speed_mbps}Mb/s" if link.speed_mbps is not None else "不明"
        mtu = str(link.mtu) if link.mtu is not None else "不明"
        roce = link.roce_device or "対応するデバイスがない"
        addrs = ", ".join(link.addrs) if link.addrs else "(アドレスなし)"
        lines.append(f"  {link.name}: {link.state}, MTU {mtu}, {speed}, RoCE {roce}, {addrs}")
    cable = f"{report.cable_count} 本" if report.cable_count is not None else "判断できない"
    lines.append(f"  ケーブルの本数: {cable}")
    if report.tools_missing:
        lines.append(f"  入っていない道具: {', '.join(report.tools_missing)}")
    if report.detail:
        lines.append(f"  詳細: {report.detail}")
    return "\n".join(lines)


# ===========================================================================
# ジョブ (帯域の計測、事前の確認、A/B の比較。tasks.md 4.4)
# ===========================================================================

# --- 決まった値 --------------------------------------------------------------

COMMAND_NAME: Final[str] = "netcheck"
"""記録の回収の置き場所の名前に使う、コマンドの名前 (`logs.var_dir`)。

A/B の回は `netcheck-<腕>-<回>` にして、回ごとに別の場所へ回収する (module docstring の
「待ちと片付けの、口の選び方」。Implementation Notes 2.4)。
"""

JOB_POLL_INTERVAL_S: Final[float] = 10.0
"""コンテナの終了を見に行く間隔 (`lifecycle.READY_POLL_INTERVAL_S` と同じ 10 秒)。"""

BASELINE_ARM: Final[str] = "baseline"
"""A/B の腕 A の名前 (最小の設定。構成のまま)。"""

CANDIDATE_ARM: Final[str] = "candidate"
"""A/B の腕 B の名前 (足した設定。`extra_env` つき)。"""

AB_REPEATS: Final[int] = 3
"""A/B の、腕ごとの回数の既定 (design.md 「netcheck」: 最小の設定と交互に 3 回ずつ)。"""

_AB_MIN_REPEATS: Final[int] = 2
"""A/B の回数の下限 (1 回ずつでは、範囲が重なるかどうかを見られない)。"""

_KIND_JOB: Final[str] = "job"
"""この 3 つの口が受ける構成の種類 (design.md 「netcheck」)。"""

_JOB_NODE_COUNT: Final[int] = 2
"""通信の確認のジョブの台の数 (2 台の間の通信を確かめるので、1 台では意味がない)。"""

_STATE_FORMAT: Final[str] = "{{.State.Status}} {{.State.ExitCode}}"
"""`docker container inspect` に渡す書式 (状態と終了コードを、1 行で読む)。"""

_STATE_ABSENT: Final[str] = "absent"
"""コンテナがもう無い (`docker container inspect` が 0 以外で終わった)。"""

_UNFINISHED_STATES: Final[frozenset[str]] = frozenset(
    {"created", "running", "restarting", "paused", "removing"}
)
"""まだ終わっていないコンテナの状態 (`image._UNFINISHED_STATES` と同じ集まり)。

ここにない状態 (`exited`、`dead`、読めなかったとき) は、待つのをやめて、出力を読み、片付けに
進む。**読めなかったことを「まだ動いている」に倒さない**。
"""

_FAST_NETWORK: Final[str] = "IB"
"""高速の直結の経路を表す、NCCL の記録の語 (`observe.NcclObservation.network`)。"""

_SAMPLES_KEY: Final[str] = "samples"
"""`allreduce_bench.py` の JSON の、大きさごとの結果の項目 (4.2 の出力の形)。"""

_WORLD_SIZE_KEY: Final[str] = "world_size"
"""`allreduce_bench.py` の JSON の、集団通信に入った rank の数 (4.2 の出力の形)。"""

DEFAULT_MESSAGE_SIZES: Final[tuple[int, ...]] = tuple(1 << (20 + 2 * step) for step in range(6))
"""design.md 「netcheck」の決まりの列 (1 MiB から 1 GiB まで 4 倍ずつの 6 つ)。

`allreduce_bench.py` の既定と同じ。A/B の短い回は `--max-bytes` などで縮められる (4.2) ので、
**この列と違うことは、合否には数えず、結果に注記する** (module docstring の「大きさの列」)。
"""

_NCCL_DEBUG_ENV: Final[str] = "NCCL_DEBUG"
_NCCL_DEBUG_FILE_ENV: Final[str] = "NCCL_DEBUG_FILE"
"""NCCL の記録の環境変数 (research.md §e-3、§e-4)。この module が、回ごとに上書きする。"""

_RESERVED_ENV: Final[frozenset[str]] = frozenset({_NCCL_DEBUG_ENV, _NCCL_DEBUG_FILE_ENV})
"""A/B の `extra_env` に書けない名前 (この module が使うため)。"""

_NCCL_DEBUG_ENOUGH: Final[frozenset[str]] = frozenset({"INFO", "TRACE"})
"""経路の行 (`Using network …`) が出る `NCCL_DEBUG` の値 (research.md §e-4)。

NCCL の値は `VERSION` / `WARN` / `INFO` / `TRACE` で、`INFO` 以上でなければ、判定に使う行が
出ない。
"""

_MOUNT_FLAG: Final[str] = "--mount"
_MOUNT_TARGET_KEYS: Final[frozenset[str]] = frozenset({"target", "dst", "destination"})
"""`--mount` の、コンテナの中の道筋を表す鍵 (docker が受ける 3 つの綴り)。"""

_MOUNT_READONLY_KEYS: Final[frozenset[str]] = frozenset({"readonly", "ro"})
"""`--mount` の、読み取り専用を表す鍵 (裸で書く形と `=true` の形がある)。"""

_MOUNT_FALSE_VALUES: Final[frozenset[str]] = frozenset({"false", "0"})
"""`readonly=…` が読み取り専用にならない値。"""

_TAG_TIME_FORMAT: Final[str] = "%Y%m%dT%H%M%S-%fZ"
"""回の札の時刻の形 (コロンと点を含まない。**マイクロ秒まで入れる**)。

秒までにすると、同じ秒に 2 度流したときに札が同じになり、前の回の記録を読んでしまう
(指摘 2)。`started_at` は呼ぶ側が渡すので、**呼び出しごとに、その時の時刻を渡すこと**
(`logs.var_dir` が使う置き場所の名前は秒までなので、そちらは同じ秒だと同じ場所になる。
Implementation Notes 2.4 の限界と同じ)。
"""

_TAG_REPEAT_DIGITS: Final[int] = 2
"""回の番号の桁数 (`1` を `01` にそろえる。`-1` が `-10` と取り違えにくい形にする)。"""

_TAG_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")
"""回の札に使える文字 (ファイルの名前の一部になるので、英数字とハイフンだけ)。"""

_NCCL_LOG_NAME: Final[str] = "nccl-{tag}.%h.%p.log"
"""NCCL の記録のファイルの名前 (`%h` はホスト名、`%p` は PID。NCCL が埋める)。"""

_SOCKET_NETWORK_LINE: Final[str] = "NCCL INFO Using network Socket"
"""ふつうのネットワークの経路になったことを表す行 (research.md §e-4 の原文
`NCCL INFO Using network (IB|Socket)`。NCCL の `src/init.cc:557` の書式文字列)。

`observe.observe_nccl` の `network` は、**最初の** `Using network …` の行だけを見る
(`observe.py` は凍結されているので、変えられない)。1 回の記録に `Using network IB` と
`Using network Socket` の両方があると `network` は `IB` のままになるので、この module が
本文を見て補う (指摘 4)。量指定子を持たないリテラルの `in` の検査なので、長い 1 行でも
手間は線形である (Implementation Notes 2.2 の決まり)。
"""

_SANITY_LABEL: Final[str] = "sanity"
"""事前の確認の、回の札の真ん中 (腕の代わり。1 回だけ流すので、回の番号は 1)。"""

_ROLE_JOIN: Final[str] = "、"


class NetcheckError(Exception):
    """通信の確認を実行して失敗した (終了コード 2)。

    いまのところ、**合格なのに片付けが終わらなかった**ときだけである (コンテナが残っている
    かもしれないので、合格として 0 で終わらせない)。不合格のときは、この例外にせず、片付けの
    問題を `detail` に書いて結果を返す (失敗の理由のほうが、計測者に役立つ)。
    """


# --- 比べる相手 (requirements 4.3、4.6) ---------------------------------------


@dataclass(frozen=True)
class ReferencePoint:
    """比べる相手の 1 つ (値と、測った道具と、出典と、原文)。

    **research.md が原文を引いていない値は、ここに持たない** (出典と原文と値が、つねに
    一致するようにする。requirements 11.1、11.2)。`source` は、research.md が URL を挙げて
    いればその URL、挙げていなければ research.md の節を指す (この module は、資料を取りに
    行かない)。
    """

    gbps: float
    tool: str
    source: str
    quote: str
    note: str = ""


@dataclass(frozen=True)
class BandwidthReference:
    """測った値と比べる、NVIDIA の公表の値 (requirements 4.3、4.6)。

    3 つを持つ。`measured` は公表の実測、`sync_lower_bound` と `nccl_threshold` は、NVIDIA
    自身が持つ 2 つのしきい値である (出どころも道具も違うので、分けて持つ)。**どれも合否には
    入れない** (design.md 「netcheck」の合否は、経路と事前の確認の 4 段だけ)。道具が違うので、
    そのままの比較にならないことを `tool_note` に持ち、結果にも書く
    (`BandwidthOutcome.comparison`)。
    """

    measured: ReferencePoint
    sync_lower_bound: ReferencePoint
    nccl_threshold: ReferencePoint
    tool_note: str


NVIDIA_REFERENCE: Final[BandwidthReference] = BandwidthReference(
    # 値と原文は、research.md が 2026-09-21 に一次資料から引いたもの。この module は、資料を
    # 取りに行かず、そこに書かれた値と原文だけを持つ (requirements 11.1、11.2)。
    measured=ReferencePoint(
        gbps=189.85,
        tool="ib_write_bw",
        source=(
            "research.md §e-1 (NVIDIA の性能計測の手引き)。research.md の References が挙げる"
            " URL: https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html 、"
            "https://github.com/NVIDIA/dgx-spark-playbooks"
            " (research.md は、この値を、どちらの文書に帰すかを書いていない)"
        ),
        quote="Total throughput = 92.57 + 97.28 = 189.85 Gbps",
        note=(
            "測り方は ib_write_bw、メッセージ 65536 B、デバイスは rocep1s0f0 と"
            " roceP2p1s0f0 の 2 つ (1 本のケーブルの 2 つの PCIe function) の合計"
        ),
    ),
    sync_lower_bound=ReferencePoint(
        gbps=184.0,
        tool="NVIDIA Sync のリンクの速さの試験",
        source=(
            "https://docs.nvidia.com/sync/latest/cluster-assistant.html"
            " (research.md §e-5 と References)"
        ),
        quote=(
            "NVIDIA Sync then runs a speed test across the links to check the lower bound of"
            " 184 Gbit/s."
        ),
    ),
    nccl_threshold=ReferencePoint(
        gbps=175.0,
        tool="nccl-tests の all_gather (Avg bus bandwidth)",
        source=(
            "research.md §e-5 (NVIDIA 自身の Spark のクラスタ設定スクリプトの定数。"
            "research.md は、この値に URL を挙げていない)"
        ),
        quote="# Avg bus bandwidth",
        note="21.875 GB/s (= 175 Gbps)。リングのときは 10 GB/s (= 80 Gbps)",
    ),
    tool_note=(
        "道具が違うので、そのままの比較にはならない (公表の値は ib_write_bw で測った 2 つの"
        " RoCE のデバイスの合計、しきい値は nccl-tests の all_gather の busbw、こちらは自前の"
        " allreduce_bench.py の all-reduce の busbw。NCCL の busbw の実測の例は、NVIDIA が"
        "公表していない)"
    ),
)
"""NVIDIA の公表の実測と、2 つのしきい値 (research.md §e-1、§e-5)。"""


# --- 事前の確認の 4 段 (requirements 4.6) ------------------------------------


@dataclass(frozen=True)
class SanityStage:
    """事前の確認の 1 段 (design.md 「netcheck」、research.md §e-6)。

    `marker` は、`serving/payload/vllm_sanity_check.py` の本文が `print` する文字列そのもので
    ある (上流の原文なので、推測で書かない)。`passed` と `nodes_seen` と `detail` は、流したあと
    に埋まる (`SANITY_STAGES` は、`passed` が偽の、段の定義として持つ)。
    """

    index: int
    name: str
    marker: str
    passed: bool = False
    nodes_seen: tuple[NodeRole, ...] = ()
    detail: str = ""


SANITY_STAGES: Final[tuple[SanityStage, ...]] = (
    SanityStage(
        index=1, name="PyTorch の NCCL の all-reduce", marker="PyTorch NCCL is successful!"
    ),
    SanityStage(
        index=2, name="PyTorch の GLOO の all-reduce (CPU)", marker="PyTorch GLOO is successful!"
    ),
    SanityStage(index=3, name="vLLM 独自の NCCL の all-reduce", marker="vLLM NCCL is successful!"),
    SanityStage(
        index=4,
        name="CUDA グラフの中の vLLM の NCCL",
        marker="vLLM NCCL with cuda graph is successful!",
    ),
)
"""4 段の定義 (`serving/payload/vllm_sanity_check.py` の `print` の文字列と、その順序)。

4 段目が特に大事である (2 台の DGX Spark のデッドロックの報告 #52504 は、CUDA グラフに取り込ま
れた NCCL の集団通信で起きていた。research.md §e-6)。**各 rank がこの 4 つを出す**ので、構成が
使う台のすべての出力に出ていなければ、その段は通っていない。
"""


# --- 結果の型 (`types.py` にないもの) -----------------------------------------

JobStatus = Literal["passed", "failed", "refused"]
"""ジョブの結末。`refused` は関門の断り (終了コード 1)、`failed` は不合格 (2)。"""


@dataclass(frozen=True)
class JobEnd:
    """1 台のジョブのコンテナの終わり (状態と終了コード)。"""

    node: NodeRole
    state: str
    exit_code: int | None = None


@dataclass(frozen=True)
class BandwidthOutcome:
    """帯域の計測の結末 (`serve netcheck bandwidth`)。

    測った値は `run` (`types.BandwidthRun`) に、比べる相手は `reference` と `comparison` に入る。
    `types.py` は凍結されている (ほかのタスクが使っている) ので、この結末の型と、比べる相手の型は
    この module に置く (Implementation Notes 1.2 の「要るときに足す」)。
    """

    status: JobStatus
    config_name: str
    run: BandwidthRun | None = None
    nccl: Mapping[NodeRole, NcclObservation | None] = field(default_factory=dict)
    """**台ごとの**観察 (指摘 2 の直し)。合否は、構成のすべての台について見る。

    `types.BandwidthRun.nccl` は 1 つしか持てないので、そちらには rank 0 (構成の `nodes` の
    1 つ目 = head) の観察を入れる。7.4 は、この項目から、台ごとの経路と束ねを書く。
    """

    ends: tuple[JobEnd, ...] = ()
    reference: BandwidthReference = NVIDIA_REFERENCE
    comparison: str = ""
    gates: tuple[GateResult, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class SanityOutcome:
    """事前の確認の結末 (`serve netcheck sanity`)。

    4 段のうち、どこまで通ったか (`stages`、`passed_stages`) と、どこで止まったか (時間切れ、
    コンテナの 0 以外の終了) が `detail` に出る。
    """

    status: JobStatus
    config_name: str
    stages: tuple[SanityStage, ...] = ()
    nccl: Mapping[NodeRole, NcclObservation | None] = field(default_factory=dict)
    """**台ごとの**観察 (指摘 2 の直し)。"""

    ends: tuple[JobEnd, ...] = ()
    log_dir: Path | None = None
    gates: tuple[GateResult, ...] = ()
    detail: str = ""

    @property
    def passed_stages(self) -> int:
        """通った段の数。"""
        return sum(1 for stage in self.stages if stage.passed)


@dataclass(frozen=True)
class RoundRoute:
    """A/B の 1 回ぶんの、**台ごと**の経路の観察 (`AbReport.routes`。`runs` と同じ順)。

    A/B の**採否には、経路を使わない** (要件 4.5 と design.md 「netcheck」の A/B の決まりは、
    速さだけで決める。コンテナの中で高速の経路が使えないときに、足す候補を A/B で確かめる、
    という使い方があるので、経路が遅い回を、採否の対象から外さない)。それでも、**どの経路で
    測った値なのかは、要件 4.4 が記録を求める**ので、回ごと・台ごとに、ここに残す (7.4 が、
    判断の記録に書く)。

    `socket_network_seen` は、その台の、その回の記録に `Using network Socket` の行が 1 度でも
    あったか (`observe_nccl` の `network` は、最初の行だけを見るため)。
    """

    arm: str
    repeat_index: int
    observed: Mapping[NodeRole, NcclObservation | None]
    socket_network_seen: Mapping[NodeRole, bool]


@dataclass(frozen=True)
class AbReport:
    """A/B の結末 (`serve netcheck ab`)。

    6 回そろって比べられたときだけ、`outcome` (`types.AbOutcome`) が入る。途中で止まったときは、
    そこまでの回 (`runs`) と、止まった理由 (`detail`) を返す (`types.AbOutcome` は、腕ごとに
    1 回以上を要求するので、そろわない回を入れられない)。
    """

    status: Literal["compared", "stopped", "refused"]
    config_name: str
    added_env: Mapping[str, str]
    runs: tuple[BandwidthRun, ...] = ()
    routes: tuple[RoundRoute, ...] = ()
    """回ごと・台ごとの経路の観察 (`runs` と同じ順。要件 4.4)。"""

    outcome: AbOutcome | None = None
    gates: tuple[GateResult, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class _Output:
    """1 台のコンテナの出力 (標準出力と標準エラーを分けて持つ)。"""

    stdout: str = ""
    stderr: str = ""
    problem: str = ""


@dataclass(frozen=True)
class _JobRun:
    """1 回のジョブの、流したあとの事実 (この module の中だけ)。"""

    outputs: Mapping[NodeRole, _Output] = field(default_factory=dict)
    ends: tuple[JobEnd, ...] = ()
    timed_out: bool = False
    log_dir: Path | None = None
    nccl: Mapping[NodeRole, NcclObservation | None] = field(default_factory=dict)
    socket_seen: Mapping[NodeRole, bool] = field(default_factory=dict)
    """台ごとに、その回の記録に `Using network Socket` の行があったか。"""

    problems: tuple[str, ...] = ()
    """片付けられなかったこと (`lifecycle.clean_up` が返したもの)。"""

    failure: str = ""
    """起こせなかった、了承のあとに読み取りが届かなかった、のときの文。"""


# --- 小さな助け ---------------------------------------------------------------


def _report(stream: TextIO, text: str) -> None:
    """進捗と警告を知らせる (標準出力は、後の処理が読むので混ぜない)。"""
    stream.write(f"{text}\n")
    stream.flush()


def _check_job_config(config: ConfigDef) -> None:
    """Spark に触る前に、構成が 2 台の通信の確認のものかを確かめる。"""
    if config.kind != _KIND_JOB:
        raise ConfigError(
            f"通信の確認に使えるのは、kind = '{_KIND_JOB}' の構成だけである"
            f" (構成 '{config.name}' の kind は '{config.kind}')"
        )
    if len(config.nodes) != _JOB_NODE_COUNT:
        raise ConfigError(
            f"通信の確認は 2 台で行う (構成 '{config.name}' の nodes は"
            f" {len(config.nodes)} 台: {', '.join(config.nodes)})"
        )


def _env_setting(config: ConfigDef, name: str) -> Setting | None:
    """構成の `env` から、その名前の設定を取る (環境変数の名前は `Setting.flag`)。"""
    return next((setting for setting in config.env.values() if setting.flag == name), None)


def _mounts(config: ConfigDef) -> tuple[tuple[str, bool], ...]:
    """構成の `--mount` の、コンテナの中の道筋と、読み取り専用かどうか。

    道筋は `target` / `dst` / `destination` のどれか、読み取り専用は `readonly` / `ro`
    (裸で書く形と `=true` の形) で見る (docker の `--mount` の綴り)。
    """
    mounts: list[tuple[str, bool]] = []
    for setting in config.docker.values():
        if setting.flag != _MOUNT_FLAG or setting.value is None:
            continue
        target: str | None = None
        readonly = False
        for part in setting.value.split(","):
            key, separator, value = part.partition("=")
            name = key.strip()
            if separator and name in _MOUNT_TARGET_KEYS and value:
                target = value.rstrip("/") or "/"
            elif name in _MOUNT_READONLY_KEYS and (
                not separator or value.strip().lower() not in _MOUNT_FALSE_VALUES
            ):
                readonly = True
        if target is not None:
            mounts.append((target, readonly))
    return tuple(mounts)


def _nccl_log_dir(config: ConfigDef) -> str:
    """NCCL の記録を書くディレクトリ (コンテナの中の道筋) を決める。**触る前に断る**。

    合否は NCCL の記録から出すので、記録が Spark の側に残らない構成では、確認そのものが
    成り立たない (指摘 1 の直し)。次のどれかなら、Spark に触る前に `ConfigError` にする:

    - `NCCL_DEBUG` がない、または `INFO` 以上でない (経路の行が出ない。research.md §e-4)
    - `NCCL_DEBUG_FILE` がない、絶対の道筋でない、`..` を含む
    - その道筋のディレクトリが、構成の `--mount` の `target` のどれの下でもない (コンテナが
      消えると、記録も消える)

    返り値:
        記録を書くディレクトリ (この module が、回ごとにファイルの名前を上書きするときの元)。
    """
    debug = _env_setting(config, _NCCL_DEBUG_ENV)
    if (
        debug is None
        or debug.value is None
        or debug.value.strip().upper() not in (_NCCL_DEBUG_ENOUGH)
    ):
        raise ConfigError(
            f"構成 '{config.name}' の env に {_NCCL_DEBUG_ENV} = INFO (または TRACE) がない"
            " (経路の行 (Using network …) が出ないので、合否を出せない。research.md §e-4)"
        )
    setting = _env_setting(config, _NCCL_DEBUG_FILE_ENV)
    if setting is None or setting.value is None:
        raise ConfigError(
            f"構成 '{config.name}' の env に {_NCCL_DEBUG_FILE_ENV} がない"
            " (NCCL の記録が Spark の側に残らないので、合否を出せない)"
        )
    value = setting.value
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ConfigError(
            f"構成 '{config.name}' の {_NCCL_DEBUG_FILE_ENV} は、コンテナの中の絶対の道筋に"
            f" する (.. を含めない): {value}"
        )
    directory = str(path.parent)
    mounts = _mounts(config)
    inside = [
        (target, readonly)
        for target, readonly in mounts
        if directory == target or directory.startswith(f"{target.rstrip('/')}/")
    ]
    if not inside:
        shown = "、".join(target for target, _ in mounts) if mounts else "なし"
        raise ConfigError(
            f"構成 '{config.name}' の {_NCCL_DEBUG_FILE_ENV} ({value}) の置き場所が、"
            f"--mount の target の下にない (コンテナが消えると記録も消える。target: {shown})"
        )
    # いちばん内側の結び付け (道筋の長いもの) が、その場所の書き込みを決める
    target, readonly = max(inside, key=lambda item: len(item[0]))
    if readonly:
        raise ConfigError(
            f"構成 '{config.name}' の {_NCCL_DEBUG_FILE_ENV} ({value}) の置き場所"
            f" ({target}) の --mount が、読み取り専用 (readonly / ro) である"
            " (NCCL が記録を書けないので、起こす前に断る)"
        )
    return directory


def _round_tag(started_at: datetime, label: str, repeat: int) -> str:
    """回の札 (起こす時刻 (UTC、マイクロ秒まで) + 腕 (または `sanity`) + 回の番号)。

    NCCL の記録のファイルの名前の一部になるので、英数字とハイフンだけに絞り、回の番号は
    2 桁にそろえる (`-1` と `-10` を取り違えないため)。**同じ `started_at` を渡して 2 度
    流すと、札は同じになる** (呼ぶ側が、呼び出しごとに、その時の時刻を渡すこと)。
    """
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise ValueError("started_at は、時差の付いた日時にする (回の札を UTC で作るため)")
    stamp = started_at.astimezone(UTC).strftime(_TAG_TIME_FORMAT)
    tag = f"{stamp}-{label}-{repeat:0{_TAG_REPEAT_DIGITS}d}"
    if not _TAG_RE.fullmatch(tag):
        raise ValueError(f"回の札に使えない文字がある (英数字とハイフンだけにする): {tag}")
    return tag


def _nccl_env(directory: str, tag: str) -> dict[str, str]:
    """この道具が、回ごとに上書きする `NCCL_DEBUG_FILE` (指摘 1 の直し)。

    `-e` は、あとに並ぶものが勝つので、構成の `env` の値を上書きできる。変わるのは記録の
    ファイルの名前だけで、通信のふるまいには効かない。
    """
    return {_NCCL_DEBUG_FILE_ENV: f"{directory}/{_NCCL_LOG_NAME.format(tag=tag)}"}


def _refusal_detail(refused: Sequence[GateResult]) -> str:
    """断った関門を、台と関門の名前つきで並べる。"""
    reasons = " / ".join(f"{gate.node or '-'} の {gate.gate}: {gate.detail}" for gate in refused)
    return f"関門が断ったので、通信の確認を始めなかった: {reasons}"


def _ends_text(ends: Sequence[JobEnd]) -> str:
    """2 台のコンテナの終わり方を、見せる文にする。"""
    if not ends:
        return "コンテナの終わりを読めなかった"
    return _ROLE_JOIN.join(
        f"{end.node}: {end.state} (終了コード {'不明' if end.exit_code is None else end.exit_code})"
        for end in ends
    )


def _bad_exits(ends: Sequence[JobEnd]) -> tuple[JobEnd, ...]:
    """0 以外で終わった (または終了コードを読めなかった) 台。"""
    return tuple(end for end in ends if end.exit_code != 0)


def _controls(
    config: ConfigDef,
    *,
    timeout_s: float | None,
    poll_interval_s: float,
    start_timeout_s: float,
    read_timeout_s: float,
    log_timeout_s: float,
    sleep: Callable[[float], None] | None,
    clock: Callable[[], float] | None,
    report: TextIO | None,
    client: httpx.Client,
) -> lifecycle.Controls:
    """差し替えられる口と時間切れを、`lifecycle` の公開の型にまとめる。

    ジョブは HTTP を使わないが、`lifecycle.Controls` は `client` を持つ形なので、つながない
    クライアントを渡す (作った側が閉じる)。
    """
    return lifecycle.Controls(
        client=client,
        timeout_s=float(config.ready_timeout_s) if timeout_s is None else timeout_s,
        poll_interval_s=poll_interval_s,
        start_timeout_s=start_timeout_s,
        read_timeout_s=read_timeout_s,
        log_timeout_s=log_timeout_s,
        sleep=time.sleep if sleep is None else sleep,
        clock=time.monotonic if clock is None else clock,
        report=sys.stderr if report is None else report,
    )


# --- 終了を待つ、出力を読む ---------------------------------------------------


def _container_state(
    runner: RemoteRunner, target: lifecycle.Target, timeout_s: float
) -> tuple[str, int | None]:
    """コンテナの状態と終了コードを読む (対象は、一覧から来た識別子だけ)。

    `lifecycle` にも `image` にも同じ形の読み取りがあるが、どちらも下線つきの名前なので使え
    ない (実物の出力の形が変わったら、3 か所を直す)。読み取りが届かなかったこと
    (`RemoteError`) は、そのまま外に出す (呼ぶ側が、片付けてから実行しての失敗にする)。
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


def _wait_for_exit(
    runner: RemoteRunner, targets: Sequence[lifecycle.Target], controls: lifecycle.Controls
) -> tuple[tuple[JobEnd, ...], bool]:
    """**2 台のコンテナの終了**を待つ (design.md 「Architecture Integration」の `kind = "job"`)。

    待ちの上限は `controls.timeout_s` (構成の `ready_timeout_s`、またはその回の上書き)。時計と
    眠りは `controls` から来るので、試験は実際に眠らない。

    **ある台が 0 以外で終わった (または状態を読めなくなった) 時点で、ほかの台を待たずに返る**
    (指摘 8 の直し)。rendezvous のジョブでは、片方が死ねば、もう片方は固まるので、待ちの上限
    (既定 1,800 秒) を空待ちすることになる (A/B の 6 回なら、最大 3 時間)。0 で終わった台は、
    ほかの台を待つ (2 台がそろって 0 で終わるのが、ふつうの終わり方である)。

    返り値:
        終わった台の終わり方 (構成の `nodes` の順) と、時間切れかどうか。時間切れと、0 以外で
        打ち切ったときは、まだ動いている台が `ends` に入らない (呼ぶ側が、それを見て「ほかの台
        の終了は待たなかった」と言う)。
    """
    deadline = controls.clock() + controls.timeout_s
    while True:
        ends: list[JobEnd] = []
        for target in targets:
            state, exit_code = _container_state(runner, target, controls.read_timeout_s)
            if state not in _UNFINISHED_STATES:
                ends.append(JobEnd(node=target.plan.node, state=state, exit_code=exit_code))
        if _bad_exits(ends) or len(ends) == len(targets):
            return tuple(ends), False
        if controls.clock() >= deadline:
            return tuple(ends), True
        controls.sleep(controls.poll_interval_s)


def _read_outputs(
    runner: RemoteRunner, targets: Sequence[lifecycle.Target], controls: lifecycle.Controls
) -> dict[NodeRole, _Output]:
    """コンテナの出力を、**止める前に**読む (標準出力と標準エラーを分ける)。

    対象は、自分のラベルで絞った一覧から来た識別子だけである。読めなかったことは、投げずに
    `_Output.problem` で返す (片付けには、つねに進む)。
    """
    outputs: dict[NodeRole, _Output] = {}
    for target in targets:
        argv = ("docker", "logs", target.container.id)
        try:
            result = runner.run(
                target.node, argv, timeout_s=controls.read_timeout_s, mutating=False
            )
        except RemoteError as exc:
            outputs[target.plan.node] = _Output(problem=f"コンテナの出力を読めなかった: {exc}")
            continue
        if result.exit_code != 0:
            outputs[target.plan.node] = _Output(
                problem=f"コンテナの出力を読めなかった: {_shown(result)}"
            )
            continue
        outputs[target.plan.node] = _Output(stdout=result.stdout, stderr=result.stderr)
    return outputs


# --- 片付け (記録の回収 → 停止 → 削除) ----------------------------------------


def _read_nccl(
    log_dir: Path | None, config: ConfigDef, tag: str
) -> tuple[dict[NodeRole, NcclObservation | None], dict[NodeRole, bool]]:
    """回収した NCCL の記録を、**台ごと・その回のぶんだけ**読む。

    Spark の `logs/` は `logs.collect_logs` が `<置き場所>/<役割>/logs/` に写す。**Spark の
    `logs/` を消す道はどこにもない** (`remote.push` は `logs/` を宛先にできず、`rm` もない) ので、
    NCCL の記録は、回と呼び出しをまたいで積もり続ける。だから:

    - **`nccl-<その回の札>.` で始まる名前**のファイルだけを読む (前後を固定した一致。部分一致に
      すると、`…-baseline-01` が `nccl-…-baseline-10.….log` に当たる)
    - **台ごとに** `observe_nccl` を呼ぶ (2 台ぶんを連ねると、`network` が head のものになり、
      worker だけが落ちた記録を見落とす)
    - 本文に `Using network Socket` の行が 1 度でもあるかを、別に持つ (`observe_nccl` の
      `network` は最初の行だけを見るので、`IB` → `Socket` の 2 行を見落とす)

    返り値:
        台ごとの観察と、台ごとの「`Using network Socket` の行があったか」。その回の札の
        ファイルが 1 つもなかった台は、観察が `None` (呼ぶ側が「確かめられなかった」にして、
        合格にしない)。
    """
    found: dict[NodeRole, NcclObservation | None] = dict.fromkeys(config.nodes)
    socket_seen: dict[NodeRole, bool] = dict.fromkeys(config.nodes, False)
    if log_dir is None:
        return found, socket_seen
    prefix = f"nccl-{tag}."
    for role in config.nodes:
        directory = log_dir / role / COMM_LOG_REMOTE_SUBDIR
        if not directory.is_dir():
            continue
        texts = [
            path.read_text(encoding="utf-8", errors="replace")
            for path in sorted(item for item in directory.rglob("*") if item.is_file())
            if path.name.startswith(prefix)
        ]
        if texts:
            body = "\n".join(texts)
            found[role] = observe_nccl(body)
            socket_seen[role] = _SOCKET_NETWORK_LINE in body
    return found, socket_seen


def _clean(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    controls: lifecycle.Controls,
    *,
    var_root: Path,
    started_at: datetime,
    command: str,
    interrupted: KeyboardInterrupt | None = None,
) -> tuple[Path | None, tuple[str, ...], KeyboardInterrupt | None]:
    """記録を回収してから、コンテナを止めて消す (3.4 の片付けの決まり)。

    順序は「回収 → head と worker の `stop` → 2 台の `rm`」。**中断が来ても、`clean_up` には
    必ず到達する**。すでに中断を受けているときは、`lifecycle.wrap_up` と同じく、**回収を飛ばして**
    片付けに進む (回収は 2 台ぶんで多くの遠隔の呼び出しになるので、Ctrl-C のあとに待たせない。
    Spark の `logs/` の NCCL の記録は、Spark の側に残る)。止めてよい相手を決めるのも、列を作るのも
    `lifecycle.clean_up` (`guards.rollback_commands`) なので、了承を得た計画の巻き戻しと完全に
    一致する。受けた中断は投げずに返す (呼ぶ側が、結果を見せてから投げる)。
    """
    log_dir: Path | None = None
    problems: list[str] = []
    if interrupted is None:
        try:
            log_dir = collect_logs(
                runner,
                nodes,
                plans,
                var_root=var_root,
                started_at=started_at,
                command=command,
                config_name=config.name,
                timeout_s=controls.log_timeout_s,
            )
        except KeyboardInterrupt as exc:
            interrupted = exc
            problems.append("記録の回収が中断されたので、そのまま片付けに進んだ")
    else:
        problems.append("中断されたので、記録を回収せずに、コンテナの片付けに進んだ")
    cleaned, interrupted = lifecycle.clean_up(
        runner, nodes, plans, controls, interrupted=interrupted
    )
    if log_dir is not None:
        _report(controls.report, f"記録は {log_dir} に回収した")
    return log_dir, (*problems, *cleaned), interrupted


def _run_job(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    plans: Sequence[ContainerPlan],
    controls: lifecycle.Controls,
    *,
    var_root: Path,
    started_at: datetime,
    command: str,
    tag: str,
) -> _JobRun:
    """了承のあとの「起こす → 終了を待つ → 読む → 回収 → 片付ける」を、1 回ぶん流す。

    **どの結果でも、コンテナを残さない** (3.1 の片付けの道の形を写す。`docker run` の呼び出しも
    `try` の中に入れる。ssh が切れても、遠隔の `docker run -d` は完了しうる)。中断は、片付けの
    あとで伝える。
    """
    try:
        lifecycle.start_all(runner, config, nodes, plans, controls)
        targets = lifecycle.started_targets(runner, config, nodes, plans, controls)
        ends, timed_out = _wait_for_exit(runner, targets, controls)
        outputs = _read_outputs(runner, targets, controls)
    except (RemoteError, lifecycle.RunFailed) as exc:
        log_dir, problems, interrupted = _clean(
            runner,
            config,
            nodes,
            plans,
            controls,
            var_root=var_root,
            started_at=started_at,
            command=command,
        )
        if interrupted is not None:
            raise interrupted from exc
        reason = (
            str(exc)
            if isinstance(exc, lifecycle.RunFailed)
            else f"了承のあとに、読み取りが届かなかった: {exc}"
        )
        observed, socket_seen = _read_nccl(log_dir, config, tag)
        return _JobRun(
            log_dir=log_dir,
            nccl=observed,
            socket_seen=socket_seen,
            problems=problems,
            failure=reason,
        )
    except BaseException as exc:
        # 中断 (Ctrl-C) と、思わぬ誤りのときも、ジョブのコンテナを残さない
        _, _, interrupted = _clean(
            runner,
            config,
            nodes,
            plans,
            controls,
            var_root=var_root,
            started_at=started_at,
            command=command,
            interrupted=exc if isinstance(exc, KeyboardInterrupt) else None,
        )
        if interrupted is not None and interrupted is not exc:
            raise interrupted from exc
        raise
    log_dir, problems, interrupted = _clean(
        runner,
        config,
        nodes,
        plans,
        controls,
        var_root=var_root,
        started_at=started_at,
        command=command,
    )
    if interrupted is not None:
        raise interrupted
    observed, socket_seen = _read_nccl(log_dir, config, tag)
    return _JobRun(
        outputs=outputs,
        ends=ends,
        timed_out=timed_out,
        log_dir=log_dir,
        nccl=observed,
        socket_seen=socket_seen,
        problems=problems,
    )


# --- 読み取り (JSON の 1 行、経路、4 段) --------------------------------------


def _find_report(stdout: str) -> dict[str, Any] | None:
    """rank 0 の標準出力から、`allreduce_bench.py` の結果の 1 行を探す。

    torchrun の警告など、ほかの行が混ざっていてもよい。JSON として読めて、`samples` の配列を
    持つ最初の行を、結果として採る (ほかの JSON の行と見分けるため)。
    """
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            parsed: object = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get(_SAMPLES_KEY), list):
            return parsed
    return None


@dataclass(frozen=True)
class _Bench:
    """rank 0 の JSON から読めた事実 (この module の中だけ)。"""

    samples: tuple[BandwidthSample, ...] = ()
    world_size: int | None = None
    problems: tuple[str, ...] = ()


def _read_sample(number: int, item: Mapping[str, Any]) -> tuple[BandwidthSample | None, str]:
    """`samples` の 1 件を読む (読めない項目は、理由の文で返す)。

    **`busbw` か `algbw` が 0 以下の項目は、「読めない」に倒す** (指摘 5 の直し。`0` は、
    測れていないか、時間が取れていないことを表すので、帯域として使わない)。
    """
    try:
        sample = BandwidthSample(
            size_bytes=item["size_bytes"],
            algbw_gbps=item["algbw_gbps"],
            busbw_gbps=item["busbw_gbps"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"{number} 件目を読めない ({exc})"
    if sample.busbw_gbps <= 0 or sample.algbw_gbps <= 0:
        return None, (
            f"{number} 件目 ({sample.size_bytes:,} B) の busbw が 0 以下"
            f" ({sample.busbw_gbps}) なので、測れていない"
        )
    return sample, ""


def _parse_bench(output: _Output | None, *, expected_world_size: int) -> _Bench:
    """rank 0 の標準出力の 1 行の JSON から、`world_size` と、大きさごとの帯域を読む。

    `world_size` が構成の台の数と合わなければ、2 台の all-reduce が起きていないので、読めた
    値を合否に使わない (指摘 4 の直し)。
    """
    if output is None:
        return _Bench(problems=("rank 0 のコンテナの出力を読めなかった",))
    if output.problem:
        return _Bench(problems=(output.problem,))
    report = _find_report(output.stdout)
    if report is None:
        return _Bench(
            problems=(
                "rank 0 の標準出力に、allreduce_bench.py の結果の 1 行 (samples を持つ JSON) が"
                "見つからない",
            )
        )
    problems: list[str] = []
    raw_world: object = report.get(_WORLD_SIZE_KEY)
    world_size = (
        raw_world if isinstance(raw_world, int) and not isinstance(raw_world, bool) else None
    )
    if world_size != expected_world_size:
        problems.append(
            f"結果の JSON の {_WORLD_SIZE_KEY} が {raw_world!r} で、構成の台の数"
            f" ({expected_world_size}) と合わない (2 台の all-reduce が起きていない)"
        )
    samples: list[BandwidthSample] = []
    skipped: list[str] = []
    raw: object = report[_SAMPLES_KEY]
    if not isinstance(raw, list):  # pragma: no cover - `_find_report` が配列だけを通す
        return _Bench(
            world_size=world_size,
            problems=(*problems, "結果の JSON の samples が配列でない"),
        )
    for number, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            skipped.append(f"{number} 件目が object でない")
            continue
        sample, problem = _read_sample(number, item)
        if sample is None:
            skipped.append(problem)
        else:
            samples.append(sample)
    if skipped:
        problems.append("結果の JSON の samples に、読めない項目がある: " + " / ".join(skipped))
    if not samples:
        problems.append("読めた大きさが 1 つもない (どの大きさも測れていない)")
    return _Bench(samples=tuple(samples), world_size=world_size, problems=tuple(problems))


def _sizes_note(samples: Sequence[BandwidthSample]) -> str:
    """測った大きさの列が、design の決まりの列と違うときの注記 (**合否には数えない**)。

    A/B の短い回は、`allreduce_bench.py` の `--max-bytes` などで縮められる (4.2) ので、列が
    違うことそのものは失敗にしない。ただし、公表の値と比べるときに読み違えないように、結果に
    書き残す (指摘 5 の「扱いを決める」への答え)。
    """
    sizes = tuple(sorted(sample.size_bytes for sample in samples))
    if not sizes or sizes == DEFAULT_MESSAGE_SIZES:
        return ""
    shown = "、".join(f"{size:,} B" for size in sizes)
    return (
        f"測った大きさの列 ({len(sizes)} 個: {shown}) が、design の決まりの列"
        " (1 MiB から 1 GiB まで 4 倍ずつの 6 つ) と違う (構成が --max-bytes などで縮めた回"
        "かもしれない。合否には数えないが、公表の値と比べるときは、この違いを見る)"
    )


def _route_problems(
    config: ConfigDef,
    by_node: Mapping[NodeRole, NcclObservation | None],
    socket_seen: Mapping[NodeRole, bool],
    tag: str,
) -> list[str]:
    """使われた経路を、合否の条件に照らす (requirements 4.4、research.md §e-4)。

    **構成のすべての台**について、次の連言を見る (指摘 2、3 の直し):

    1. `network == "IB"` (高速の直結の経路が使われた)
    2. `socket_channel_seen` が偽 (ふつうのネットワークの経路のチャンネルがない)
    3. `ib_no_device` が偽 (`NET/IB : No device found.` が出ていない)
    4. `Using network Socket` の行が 1 度も出ていない (`observe_nccl` の `network` は最初の
       行だけを見るので、`IB` → `Socket` の 2 行を、この module が本文で補う)

    research.md §e-4: **「IB から Socket に落ちた」と明示する文字列は NCCL のソースにないので、
    判定は連言で行う**。その回の記録が回収できなかった台、経路の行が読めなかった台は、
    「確かめられなかった」として、合格にしない。
    """
    problems: list[str] = []
    for role in config.nodes:
        found = by_node.get(role)
        if found is None:
            problems.append(
                f"{role}: この回の NCCL の記録 (名前が nccl-{tag}. で始まるファイル) を"
                "回収できなかったので、使われた経路を確かめられなかった"
            )
            continue
        if found.network is None:
            problems.append(
                f"{role}: NCCL の記録から、使われた経路の行 (Using network …) を読めなかった"
                " (確かめられなかった)"
            )
        elif found.network != _FAST_NETWORK:
            problems.append(
                f"{role}: 高速の直結の経路が使われていない (NCCL の記録の経路は {found.network})"
            )
        if found.ib_no_device:
            problems.append(f"{role}: IB のデバイスが見つかっていない (NET/IB : No device found.)")
        if found.socket_channel_seen:
            problems.append(
                f"{role}: ふつうのネットワークの経路のチャンネル ([send] via NET/Socket) が、"
                "記録にある"
            )
        if socket_seen.get(role, False) and found.network != "Socket":
            problems.append(
                f"{role}: 記録に「{_SOCKET_NETWORK_LINE}」の行がある"
                " (最初の経路の行は IB だが、途中でふつうのネットワークの経路になっている)"
            )
    return problems


def _sanity_stages(
    config: ConfigDef, outputs: Mapping[NodeRole, _Output]
) -> tuple[SanityStage, ...]:
    """4 段の成否を、2 台の出力から読む (成功の文面は、上流の原文そのもの)。

    **各 rank が 4 つの文面を出す**ので、構成が使う台のすべてに出ていなければ、その段は通って
    いないとする (集団通信なので、片方だけが通ることはない)。出力を読めなかった台は、「出て
    いない」と同じ扱いにする (確かめられないものを、通ったことにしない)。
    """
    stages: list[SanityStage] = []
    for spec in SANITY_STAGES:
        seen: list[NodeRole] = []
        missing: list[str] = []
        for role in config.nodes:
            output = outputs.get(role)
            if output is None or output.problem:
                missing.append(f"{role}: 出力を読めなかった")
                continue
            if spec.marker in output.stdout or spec.marker in output.stderr:
                seen.append(role)
            else:
                missing.append(f"{role}: 成功の文面が出ていない")
        passed = len(seen) == len(config.nodes)
        detail = (
            f"{_ROLE_JOIN.join(seen)} の出力に、成功の文面があった"
            if passed
            else f"通っていない ({_ROLE_JOIN.join(missing)})"
        )
        stages.append(
            replace(spec, passed=passed, nodes_seen=tuple(seen), detail=detail),
        )
    return tuple(stages)


def _comparison_text(samples: Sequence[BandwidthSample]) -> str:
    """測った値を、NVIDIA の公表の値と並べる (requirements 4.3、4.6)。

    **道具が違うこと**を、つねに書く。しきい値は、合否には入れない (design.md 「netcheck」の
    合否は、経路と事前の確認の 4 段だけ) ので、比べて示すだけである。
    """
    reference = NVIDIA_REFERENCE
    published = (
        f"NVIDIA の公表の実測は {reference.measured.gbps} Gbps ({reference.measured.tool})、"
        f"NVIDIA Sync の下限は {reference.sync_lower_bound.gbps:.0f} Gbit/s"
        f" ({reference.sync_lower_bound.tool})、"
        f"NVIDIA 自身のしきい値は {reference.nccl_threshold.gbps:.0f} Gbps"
        f" ({reference.nccl_threshold.tool})"
    )
    if not samples:
        return f"測れなかったので、比べられない。{published}。{reference.tool_note}"
    largest = max(samples, key=lambda sample: sample.size_bytes)
    return (
        f"いちばん大きいメッセージ ({largest.size_bytes:,} B) の busbw は"
        f" {largest.busbw_gbps:.2f} Gbps。{published}。{reference.tool_note}"
    )


def _show_outputs(stream: TextIO, done: _JobRun) -> None:
    """コンテナの出力を見せる (結果の型には入れない。長くなりうるため)。"""
    for role, output in done.outputs.items():
        body = output.problem or "\n".join(
            text.rstrip("\n") for text in (output.stdout, output.stderr) if text.strip()
        )
        _report(stream, f"--- {role} のコンテナの出力 ---\n{body or '(出力なし)'}")


def _run_problems(config: ConfigDef, done: _JobRun) -> list[str]:
    """ジョブそのものが、まっすぐ終わったかどうか (合否の前に見ること)。"""
    problems: list[str] = []
    if done.failure:
        problems.append(done.failure)
    if done.timed_out:
        problems.append(
            "待ちの上限のうちに、コンテナが終わらなかった (時間切れ。"
            f"終わっていた台: {_ends_text(done.ends) if done.ends else 'なし'})"
        )
    bad = _bad_exits(done.ends)
    if bad and not done.timed_out:
        ended = {end.node for end in done.ends}
        waiting = [role for role in config.nodes if role not in ended]
        note = (
            ""
            if not waiting
            else (
                f"。ほかの台 ({'、'.join(waiting)}) の終了は待たなかった"
                " (rendezvous の相手が死ぬと固まるので、空待ちしない)"
            )
        )
        problems.append(f"コンテナが 0 以外の終了コードで終わった ({_ends_text(bad)}){note}")
    return problems


def _verdict(
    problems: Sequence[str], done: _JobRun, *, passed: str, failed: str
) -> tuple[JobStatus, str]:
    """合否と、見せる文を決める (片付けが終わらなかったときは、合格にしない)。

    例外:
        NetcheckError: 合格なのに、片付けが終わらなかったとき (終了コード 2。コンテナが残って
            いるかもしれないことを、終了コードで分かるようにする)。
    """
    status: JobStatus = "failed" if problems else "passed"
    body = passed if not problems else f"{failed}: " + " / ".join(problems)
    if done.problems:
        trouble = "片付けが終わらなかった: " + " / ".join(done.problems)
        if status == "passed":
            raise NetcheckError(
                f"{body}。{trouble}。コンテナが残っているかもしれないので、`serve status` で"
                "確かめて、`serve stop` で片付ける"
            )
        return status, f"{body}。{trouble}"
    return status, body


# --- 公開の口: 帯域の計測 ------------------------------------------------------


def run_bandwidth(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    started_at: datetime,
    *,
    confirmer: Confirmer,
    var_root: Path,
    timeout_s: float | None = None,
    poll_interval_s: float = JOB_POLL_INTERVAL_S,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    report: TextIO | None = None,
    start_timeout_s: float = lifecycle.START_TIMEOUT_S,
    read_timeout_s: float = READ_TIMEOUT_S,
    log_timeout_s: float = LOG_READ_TIMEOUT_S,
    http_timeout_s: float = lifecycle.HTTP_TIMEOUT_S,
) -> BandwidthOutcome:
    """2 台の間の帯域を測る (`serve netcheck bandwidth`。requirements 4.3、4.4、4.6)。

    進む順と、合否の条件と、終了コードへの写し方は、module の docstring にある。**どの結果でも、
    記録を回収してから、必ず止めて消す**。

    引数:
        runner: 遠隔の実行役。
        config: `kind = "job"` の、2 台の構成 (`config.select_config` を通ったもの)。
        nodes: 役割ごとのノードの定義。
        started_at: 起こす時刻 (ラベルと、記録の回収の置き場所の名前に使う。時差の付いた日時)。
        confirmer: 計画を見せて、了承を得る口。
        var_root: 記録の回収の置き場所の根 (`serving/var/`。`SshRunner` を作ったときと同じ値に
            すること)。
        timeout_s: コンテナの終了を待つ上限を、**その回だけ**上書きする。省くと、構成の
            `ready_timeout_s` を使う。
        poll_interval_s: 終了を見に行く間隔。
        sleep: 眠る口 (試験は、実際に眠らないものを渡す)。既定は `time.sleep`。
        clock: 時計 (単調増加の秒)。既定は `time.monotonic`。
        report: 進捗と、コンテナの出力を知らせる先。既定は `sys.stderr`。
        start_timeout_s: `docker run -d` の時間切れ。
        read_timeout_s: 関門と、一覧と、コンテナの状態と出力の読み取りの時間切れ。
        log_timeout_s: 記録の全量の回収の時間切れ。
        http_timeout_s: `lifecycle.Controls` に渡すクライアントの時間切れ (ジョブは HTTP を
            使わないので、実際にはつながない)。

    返り値:
        `passed` (終了コード 0)、`refused` (1)、`failed` (2)。

    例外:
        config.ConfigError: `kind` が `job` でない構成、2 台でない構成、引数の列を組み立てられ
            ない構成 (`plan.PlanError`。どれも終了コード 1)。
        guards.ApprovalError: 計測者が了承しなかった (終了コード 1)。
        NetcheckError: 合格なのに、片付けが終わらなかった (終了コード 2)。
        KeyboardInterrupt: 中断 (終了コード 130)。コンテナは、片付けてから伝える。
        ValueError: ノードの定義がないとき (終了コード 1)。
    """
    _check_job_config(config)
    # 記録が取れない構成は、ここで断る (合否は NCCL の記録から出すため)
    directory = _nccl_log_dir(config)
    tag = _round_tag(started_at, BASELINE_ARM, 1)
    plans = build_plans(config, nodes, started_at, extra_env=_nccl_env(directory, tag))
    gates = run_gates(runner, config, nodes, plans, timeout_s=read_timeout_s)
    refused = tuple(gate for gate in gates if not gate.passed)
    if refused:
        return BandwidthOutcome(
            status="refused",
            config_name=config.name,
            gates=gates,
            comparison=_comparison_text(()),
            detail=_refusal_detail(refused),
        )
    request_approval(confirmer, runner, build_approved_plan(plans), nodes)

    client = lifecycle.new_client(http_timeout_s)
    controls = _controls(
        config,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        start_timeout_s=start_timeout_s,
        read_timeout_s=read_timeout_s,
        log_timeout_s=log_timeout_s,
        sleep=sleep,
        clock=clock,
        report=report,
        client=client,
    )
    try:
        done = _run_job(
            runner,
            config,
            nodes,
            plans,
            controls,
            var_root=var_root,
            started_at=started_at,
            command=COMMAND_NAME,
            tag=tag,
        )
    finally:
        client.close()

    bench = _parse_bench(done.outputs.get(config.nodes[0]), expected_world_size=len(config.nodes))
    run = BandwidthRun(
        arm=BASELINE_ARM,
        repeat_index=1,
        samples=bench.samples,
        # `types.BandwidthRun` は観察を 1 つしか持てないので、rank 0 (構成の 1 台目) のものを
        # 入れる。台ごとの観察は `BandwidthOutcome.nccl` にある (指摘 2)
        nccl=done.nccl.get(config.nodes[0]),
        log_dir=done.log_dir,
    )
    comparison = _comparison_text(bench.samples)
    note = _sizes_note(bench.samples)
    problems = _run_problems(config, done)
    problems.extend(_route_problems(config, done.nccl, done.socket_seen, tag))
    problems.extend(bench.problems)
    if problems:
        _show_outputs(controls.report, done)
    passed_text = (
        f"構成 '{config.name}' の帯域の計測は合格した (2 台のすべてで、高速の直結の経路が"
        f"使われ、ふつうのネットワークの経路のチャンネルがない)。{comparison}"
    )
    status, detail = _verdict(
        problems,
        done,
        passed=passed_text if not note else f"{passed_text}。{note}",
        failed=f"構成 '{config.name}' の帯域の計測は、合格にしない",
    )
    return BandwidthOutcome(
        status=status,
        config_name=config.name,
        run=run,
        nccl=done.nccl,
        ends=done.ends,
        comparison=comparison,
        gates=gates,
        detail=detail if not note or status == "passed" else f"{detail}。{note}",
    )


# --- 公開の口: 事前の確認 ------------------------------------------------------


def run_sanity(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    started_at: datetime,
    *,
    confirmer: Confirmer,
    var_root: Path,
    timeout_s: float | None = None,
    poll_interval_s: float = JOB_POLL_INTERVAL_S,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    report: TextIO | None = None,
    start_timeout_s: float = lifecycle.START_TIMEOUT_S,
    read_timeout_s: float = READ_TIMEOUT_S,
    log_timeout_s: float = LOG_READ_TIMEOUT_S,
    http_timeout_s: float = lifecycle.HTTP_TIMEOUT_S,
) -> SanityOutcome:
    """推論サーバーを立てる前の、2 台での事前の確認を流す (`serve netcheck sanity`。4.6)。

    `vllm_sanity_check.py` (上流の原文) を流し、4 段 (`SANITY_STAGES`) の成否を、2 台の出力から
    読む。引数と例外は `run_bandwidth` と同じである。

    返り値:
        `passed` (終了コード 0)、`refused` (1)、`failed` (2)。4 段のうち、どこまで通ったかと、
        どこで止まったか (時間切れ、コンテナの 0 以外の終了) が `detail` に出る。
    """
    _check_job_config(config)
    directory = _nccl_log_dir(config)
    tag = _round_tag(started_at, _SANITY_LABEL, 1)
    plans = build_plans(config, nodes, started_at, extra_env=_nccl_env(directory, tag))
    gates = run_gates(runner, config, nodes, plans, timeout_s=read_timeout_s)
    refused = tuple(gate for gate in gates if not gate.passed)
    if refused:
        return SanityOutcome(
            status="refused",
            config_name=config.name,
            stages=SANITY_STAGES,
            gates=gates,
            detail=_refusal_detail(refused),
        )
    request_approval(confirmer, runner, build_approved_plan(plans), nodes)

    client = lifecycle.new_client(http_timeout_s)
    controls = _controls(
        config,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        start_timeout_s=start_timeout_s,
        read_timeout_s=read_timeout_s,
        log_timeout_s=log_timeout_s,
        sleep=sleep,
        clock=clock,
        report=report,
        client=client,
    )
    try:
        done = _run_job(
            runner,
            config,
            nodes,
            plans,
            controls,
            var_root=var_root,
            started_at=started_at,
            command=COMMAND_NAME,
            tag=tag,
        )
    finally:
        client.close()

    stages = _sanity_stages(config, done.outputs)
    passed_count = sum(1 for stage in stages if stage.passed)
    stopped = next((stage for stage in stages if not stage.passed), None)
    summary = f"{len(stages)} 段のうち {passed_count} 段が通った"
    if stopped is not None:
        summary += f" ({stopped.index} 段目「{stopped.name}」で止まった: {stopped.detail})"
    problems = _run_problems(config, done)
    problems.extend(_route_problems(config, done.nccl, done.socket_seen, tag))
    if stopped is not None:
        problems.append(summary)
    if problems:
        _show_outputs(controls.report, done)
    status, detail = _verdict(
        problems,
        done,
        passed=(
            f"構成 '{config.name}' の事前の確認は、4 段すべてが通った"
            " (PyTorch の NCCL、GLOO、vLLM の NCCL、CUDA グラフの中の NCCL)。"
            "2 台のすべてで、高速の直結の経路が使われている"
        ),
        failed=f"構成 '{config.name}' の事前の確認は、合格にしない",
    )
    return SanityOutcome(
        status=status,
        config_name=config.name,
        stages=stages,
        nccl=done.nccl,
        ends=done.ends,
        log_dir=done.log_dir,
        gates=gates,
        detail=detail,
    )


# --- 公開の口: A/B の比較 ------------------------------------------------------


def _ab_rounds(
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    started_at: datetime,
    added_env: Mapping[str, str],
    repeats: int,
    directory: str,
) -> tuple[tuple[str, int, str, tuple[ContainerPlan, ...]], ...]:
    """A/B の回ごとの札と計画を、交互の順 (A1、B1、A2、B2、A3、B3) に組み立てる。

    計測者が足す環境変数を渡すのは、足した設定の側 (腕 B) だけである。**NCCL の記録の
    ファイルの名前の上書きは、どちらの腕にも入る** (指摘 1 の直し。腕 A は「構成のまま」では
    なくなるが、変わるのは記録のファイルの名前だけで、通信のふるまいには効かない)。
    """
    rounds: list[tuple[str, int, str, tuple[ContainerPlan, ...]]] = []
    for repeat in range(1, repeats + 1):
        for arm in (BASELINE_ARM, CANDIDATE_ARM):
            tag = _round_tag(started_at, arm, repeat)
            added = {} if arm == BASELINE_ARM else dict(added_env)
            rounds.append(
                (
                    arm,
                    repeat,
                    tag,
                    build_plans(
                        config,
                        nodes,
                        started_at,
                        arm=arm,
                        repeat_index=repeat,
                        extra_env={**added, **_nccl_env(directory, tag)},
                    ),
                )
            )
    return tuple(rounds)


def _busbw_at(run: BandwidthRun, size_bytes: int) -> float:
    """その大きさの `busbw` (ないときは呼ばない。`_common_sizes` で先に絞る)。"""
    return next(sample.busbw_gbps for sample in run.samples if sample.size_bytes == size_bytes)


def _common_sizes(runs: Sequence[BandwidthRun]) -> frozenset[int]:
    """すべての回に共通する、メッセージの大きさ (1 つでも測れていない大きさは外す)。"""
    common: frozenset[int] | None = None
    for run in runs:
        sizes = frozenset(sample.size_bytes for sample in run.samples)
        common = sizes if common is None else (common & sizes)
    return frozenset() if common is None else common


def _shown_values(values: Sequence[float]) -> str:
    return _ROLE_JOIN.join(f"{value:.2f}" for value in values)


def _route_label(found: NcclObservation | None, socket_seen: bool) -> str:
    """1 台の、1 回ぶんの経路を、短い札にする (`AbReport.detail` のまとめに使う)。"""
    if found is None:
        return "読めなかった"
    marks: list[str] = []
    if socket_seen and found.network != "Socket":
        marks.append("途中で Socket")
    if found.ib_no_device:
        marks.append("No device found")
    if found.socket_channel_seen:
        marks.append("Socket のチャンネル")
    base = found.network or "経路の行がない"
    return base if not marks else f"{base} ({'、'.join(marks)})"


def _route_summary(config: ConfigDef, routes: Sequence[RoundRoute]) -> str:
    """腕ごと・台ごとの経路のまとめ (例: `baseline: head=IB ×3 回、worker=IB ×3 回 / …`)。"""
    parts: list[str] = []
    for arm in (BASELINE_ARM, CANDIDATE_ARM):
        picked = [route for route in routes if route.arm == arm]
        shown: list[str] = []
        for role in config.nodes:
            counts: dict[str, int] = {}
            for route in picked:
                label = _route_label(
                    route.observed.get(role), route.socket_network_seen.get(role, False)
                )
                counts[label] = counts.get(label, 0) + 1
            shown.append(
                f"{role}=" + "、".join(f"{label} ×{count} 回" for label, count in counts.items())
            )
        parts.append(f"{arm}: " + "、".join(shown))
    return " / ".join(parts)


def _route_caution(config: ConfigDef, routes: Sequence[RoundRoute]) -> str:
    """高速の直結の経路でない回があれば、**採否のすぐそばに置く注意**を作る。

    採否は速さだけで決める (経路は使わない) ので、計測者が読み違えないように、採否の文の
    すぐあとに、この注意を並べる (空の文字列なら、注意はない)。
    """
    flagged: list[str] = []
    for route in routes:
        for role in config.nodes:
            found = route.observed.get(role)
            doubtful = (
                found is None
                or found.network != _FAST_NETWORK
                or found.ib_no_device
                or found.socket_channel_seen
                or route.socket_network_seen.get(role, False)
            )
            if doubtful:
                flagged.append(f"{route.arm} の {route.repeat_index} 回目の {role}")
    if not flagged:
        return ""
    return (
        "注意: 高速の直結の経路が使われていない回がある"
        f" ({'、'.join(flagged)})。採否は速さだけで決めているので、この結果を採るときは、"
        "経路も確かめる (要件 4.4)"
    )


def _shown_env(added_env: Mapping[str, str]) -> str:
    """足した環境変数を、見せる文にする (値も見せる。秘密らしい名前は `plan` が断る)。"""
    return _ROLE_JOIN.join(f"{name}={value}" for name, value in added_env.items())


def _compare(
    config: ConfigDef,
    added_env: Mapping[str, str],
    runs: Sequence[BandwidthRun],
    routes: Sequence[RoundRoute],
    gates: Sequence[GateResult],
) -> AbReport:
    """6 回の結果から、採否を決める (requirements 4.5、design.md 「netcheck」の A/B)。

    **採用できるのは、足した側 (B) の 3 回の最小が、最小の設定の側 (A) の 3 回の最大を上回った
    ときだけ**である。比べる値は、6 回に共通する、いちばん大きいメッセージの `busbw`。

    **経路は、採否に使わない** (要件 4.5 と design の A/B の決まりは、速さだけで決める。
    コンテナの中で高速の経路が使えないときに、足す候補を A/B で確かめる、という使い方がある)。
    そのかわり、腕ごと・台ごとの経路のまとめを `detail` に必ず書き、高速の経路でない回が
    あれば、**採否の文のすぐそばに注意を置く** (要件 4.4。計測者が読み違えないように)。
    """
    baseline = tuple(run for run in runs if run.arm == BASELINE_ARM)
    candidate = tuple(run for run in runs if run.arm == CANDIDATE_ARM)
    common = _common_sizes(runs)
    if not baseline or not candidate or not common:
        return AbReport(
            status="stopped",
            config_name=config.name,
            added_env=dict(added_env),
            runs=tuple(runs),
            routes=tuple(routes),
            gates=tuple(gates),
            detail=(
                "6 回に共通するメッセージの大きさがないので、比べられない"
                " (どの回も同じ大きさを測れているかを、回収した記録で確かめる)"
            ),
        )
    size = max(common)
    base_values = [_busbw_at(run, size) for run in baseline]
    cand_values = [_busbw_at(run, size) for run in candidate]
    base_max = max(base_values)
    cand_min = min(cand_values)
    adopt = cand_min > base_max
    judged = (
        f"足した側の最小 ({cand_min:.2f} Gbps) が、最小の設定の側の最大 ({base_max:.2f} Gbps) を"
        "上回ったので、採用できる"
        if adopt
        else (
            f"範囲が重なる (足した側の最小 {cand_min:.2f} Gbps が、最小の設定の側の最大"
            f" {base_max:.2f} Gbps を上回っていない) ので、採用できない"
        )
    )
    caution = _route_caution(config, routes)
    judged_with_caution = judged if not caution else f"{judged}。{caution}"
    detail = (
        f"いちばん大きい共通のメッセージ ({size:,} B) の busbw で比べた。"
        f"最小の設定 (A) の {len(base_values)} 回: {_shown_values(base_values)} Gbps、"
        f"足した設定 (B) の {len(cand_values)} 回: {_shown_values(cand_values)} Gbps。"
        f"{judged_with_caution}"
    )
    outcome = AbOutcome(
        added_env=dict(added_env),
        compared_size_bytes=size,
        baseline_runs=baseline,
        candidate_runs=candidate,
        adopt=adopt,
        detail=detail,
    )
    return AbReport(
        status="compared",
        config_name=config.name,
        added_env=dict(added_env),
        runs=tuple(runs),
        routes=tuple(routes),
        outcome=outcome,
        gates=tuple(gates),
        detail=(
            f"構成 '{config.name}' に {_shown_env(added_env)} を足した A/B を、"
            f"{len(base_values)} 回ずつ交互に流した。{judged_with_caution}。"
            f"経路 (要件 4.4): {_route_summary(config, routes)}"
        ),
    )


def run_ab(
    runner: RemoteRunner,
    config: ConfigDef,
    nodes: Mapping[NodeRole, NodeDef],
    started_at: datetime,
    *,
    extra_env: Mapping[str, str],
    confirmer: Confirmer,
    var_root: Path,
    repeats: int = AB_REPEATS,
    timeout_s: float | None = None,
    poll_interval_s: float = JOB_POLL_INTERVAL_S,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    report: TextIO | None = None,
    start_timeout_s: float = lifecycle.START_TIMEOUT_S,
    read_timeout_s: float = READ_TIMEOUT_S,
    log_timeout_s: float = LOG_READ_TIMEOUT_S,
    http_timeout_s: float = lifecycle.HTTP_TIMEOUT_S,
) -> AbReport:
    """足す設定の A/B を行う (`serve netcheck ab`。requirements 4.5)。

    最小の設定 (腕 A = 構成のまま) と、足した設定 (腕 B = `extra_env` つき) を、**交互に 3 回
    ずつ** (A1、B1、A2、B2、A3、B3) 流し、採否を決める。**了承は 1 回**で、6 回ぶん (12 個) の
    `docker run` と、その巻き戻しを、すべて並べた計画を見せる。1 回ごとに「起こす → 終了を待つ
    → 読む → 回収 → 片付ける」を済ませてから、次の回に進む (自分のコンテナは、同時に 1 つの回
    ぶんだけ)。

    **経路 (IB か Socket か) は、採否に使わない** (要件 4.5 と design.md 「netcheck」の A/B の
    決まりは、速さだけで決める。コンテナの中で高速の経路が使えないときに、足す候補を A/B で
    確かめる、という使い方があるので、経路が遅い回を採否の対象から外さない)。そのかわり、
    **回ごと・台ごとの経路の観察を `AbReport.routes` に残し** (要件 4.4)、まとめと、高速の経路で
    ない回があるときの注意を `detail` に書く。**どれかの台で、その回の札の記録が読めなかった回
    は、値を比較に使わず、その回で止まる** (経路の証拠のない値で、採否を出さない)。

    **`extra_env` の値は、置き換えの印を埋めず、全ノードに同じ値が渡る** (Implementation Notes
    2.1)。ノードごとに違う値 (たとえば、2 台でインターフェースの名前が違うとき) が要る A/B は、
    この口では表せない。そのときは `plan` の変更を先に行うか、別の名前の構成として書く。
    `--device` や `--cap-add` のような docker の設定も、ここでは足せない (別の構成として書く。
    requirements 2.8: 恒久的な設定は変えない)。

    引数:
        extra_env: 足す環境変数 (1 つ以上)。名前の決まり (秘密らしい名前を断る) は、構成の
            `env` と同じで、`plan` が掛ける。
        repeats: 腕ごとの回数 (design.md 「cli」の `--repeat`。既定は 3)。**2 回以上**にする
            (1 回ずつでは、範囲が重なるかどうかを見られない)。ほかの引数は `run_bandwidth`
            と同じ。

    返り値:
        `compared` (6 回そろって比べられた。終了コード 0。採用できる・できないの、どちらでも)、
        `stopped` (途中の回で止まった。2。経路を確かめられなかった回を含む)、`refused`
        (関門が断った。1)。

    例外:
        ValueError: `extra_env` が空のとき、`repeats` が 2 より小さいとき、ノードの定義が
            ないとき (終了コード 1)。
        config.ConfigError / guards.ApprovalError: `run_bandwidth` と同じ (どちらも 1)。
        KeyboardInterrupt: 中断 (130)。その回のコンテナは、片付けてから伝える。
    """
    _check_job_config(config)
    if not extra_env:
        raise ValueError(
            "A/B には、足す環境変数が 1 つ以上要る (--env NAME=VALUE)。"
            "足すものがなければ、同じ設定を 6 回流すだけになる"
        )
    if repeats < _AB_MIN_REPEATS:
        raise ValueError(
            f"A/B の回数は、腕ごとに {_AB_MIN_REPEATS} 回以上にする (1 回ずつでは、"
            f"範囲が重なるかどうかを見られない): {repeats}"
        )
    if reserved := sorted(set(extra_env) & _RESERVED_ENV):
        raise ValueError(
            f"A/B の足す環境変数に {'、'.join(reserved)} は書けない"
            " (この道具が、回ごとに NCCL の記録のファイルの名前を上書きするため。"
            "記録の場所を変えたいときは、構成の env を直す)"
        )
    added = dict(extra_env)
    directory = _nccl_log_dir(config)
    rounds = _ab_rounds(config, nodes, started_at, added, repeats, directory)
    all_plans = tuple(plan for _, _, _, plans in rounds for plan in plans)

    # 関門に渡す計画は、1 回目の回のぶんでよい (関門が見るのは、結び付ける置き場所と、使う
    # 待ち受けのポートで、どちらも 6 回で同じ。足す環境変数は、関門の判定に効かない)
    gates = run_gates(runner, config, nodes, rounds[0][3], timeout_s=read_timeout_s)
    refused = tuple(gate for gate in gates if not gate.passed)
    if refused:
        return AbReport(
            status="refused",
            config_name=config.name,
            added_env=added,
            gates=gates,
            detail=_refusal_detail(refused),
        )
    # 6 回ぶん (12 個) の `docker run` と、その巻き戻しを、1 つの計画として見せて了承を得る
    request_approval(confirmer, runner, build_approved_plan(all_plans), nodes)

    client = lifecycle.new_client(http_timeout_s)
    controls = _controls(
        config,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        start_timeout_s=start_timeout_s,
        read_timeout_s=read_timeout_s,
        log_timeout_s=log_timeout_s,
        sleep=sleep,
        clock=clock,
        report=report,
        client=client,
    )
    runs: list[BandwidthRun] = []
    routes: list[RoundRoute] = []
    try:
        for arm, repeat, tag, plans in rounds:
            _report(controls.report, f"A/B: {arm} の {repeat} 回目を流す")
            done = _run_job(
                runner,
                config,
                nodes,
                plans,
                controls,
                var_root=var_root,
                started_at=started_at,
                command=f"{COMMAND_NAME}-{arm}-{repeat}",
                tag=tag,
            )
            bench = _parse_bench(
                done.outputs.get(config.nodes[0]), expected_world_size=len(config.nodes)
            )
            runs.append(
                BandwidthRun(
                    arm=arm,
                    repeat_index=repeat,
                    samples=bench.samples,
                    # `types.BandwidthRun` は観察を 1 つしか持てないので、rank 0 のものを入れる。
                    # 台ごとの観察は、`AbReport.routes` に残す (要件 4.4)
                    nccl=done.nccl.get(config.nodes[0]),
                    log_dir=done.log_dir,
                )
            )
            routes.append(
                RoundRoute(
                    arm=arm,
                    repeat_index=repeat,
                    observed=dict(done.nccl),
                    socket_network_seen=dict(done.socket_seen),
                )
            )
            problems = _run_problems(config, done)
            problems.extend(bench.problems)
            # 経路の証拠のない値で、採否を出さない (どれかの台で、その回の記録が読めなければ、
            # その回を失敗として扱い、残りの回を流さずに止まる)
            unreadable = [role for role in config.nodes if done.nccl.get(role) is None]
            if unreadable:
                problems.append(
                    f"この回の NCCL の記録 (名前が nccl-{tag}. で始まるファイル) を"
                    f" {'、'.join(unreadable)} で回収できなかったので、使われた経路を"
                    "確かめられなかった (経路の証拠のない値は、比較に使わない)"
                )
            if done.problems:
                # 前の回の片付けが終わらないまま次の回に進むと、名前の衝突を起こす
                problems.append("片付けが終わらなかった: " + " / ".join(done.problems))
            if problems:
                _show_outputs(controls.report, done)
                return AbReport(
                    status="stopped",
                    config_name=config.name,
                    added_env=added,
                    runs=tuple(runs),
                    routes=tuple(routes),
                    gates=gates,
                    detail=(
                        f"A/B は、{arm} の {repeat} 回目で止まった: " + " / ".join(problems) + "。"
                        f"ここまでの {len(runs)} 回の結果だけを返す (残りの回は流していない)"
                    ),
                )
    finally:
        client.close()
    return _compare(config, added, tuple(runs), tuple(routes), gates)
