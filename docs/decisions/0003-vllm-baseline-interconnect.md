# 0003. 通信の設定の採否と、直結のインターフェースの実測

状態: 検討中 (骨組み、2026-09-22)。最初の通信の設定 (Decision 10) は決めて
`serving/config/configs.toml` に反映した。ケーブルの本数の判断、PLAN.md の訂正、帯域の
A/B の実測は、タスク 7.3〜7.4 (実機) で埋める。

関係する要件: 2.8、3.6、3.7、4.1、4.2、4.3、4.4、4.5、4.6、4.7、10.3、11.1、11.2、11.4、
11.5、11.6。仕様は `.kiro/specs/vllm-baseline/`。手順は
[`docs/vllm-baseline/procedure.md`](../vllm-baseline/procedure.md) の「関門 A」。

書かないこと (要件 10.5): **送った内容と応答の本文**(書くのは長さ、トークンの数、終わりの
理由、HTTP の状態だけ)、**認証の情報**(鍵、トークン。Spark にも置かない。要件 2.6)、
**計測者が別に起動していた構成 (`exl3-tp2`) の中身**(起動の引数、設定、差し込まれた
ファイル、記録。読んでよいのは、GPU を使っているプロセスの名前とメモリの量だけである)は、
この記録に書かない。

## 背景

要件 4 は「最小の設定から始め、A/B で良くなったものだけを採用する」ことを求める。PLAN.md
の直結の記述 (「直結リンク 2 本」「1 本あたり約 112Gb/s」「MTU 9000」) は、NVIDIA 自身の
DGX Spark ユーザーガイド (ConnectX-7 Networking) と食い違うことが、research.md §e-1 の
調査で分かった。NVIDIA の資料の原文:

> "Each DGX Spark has two QSFP ports … Each port provides up to 200 Gigabits per second
> (Gb/s) … Each QSFP port appears as two independent Linux Ethernet interfaces. As a
> result, plugging in two cables shows a total of four Linux Ethernet interfaces."

> "Full bandwidth can be achieved with just one QSFP cable."

つまり、1 本のケーブル (1 つの QSFP ポート) が、2 つの PCIe function を持ち、Linux には
2 つのインターフェース (`enp1s0f0np0`、`enP2p1s0f0np0`) として見える。「2 本」という
記述は、この事実の誤読と考えられる (PLAN.md の訂正は 7.3 で行う。要件 4.2)。

NVIDIA の 2 台直結の公式手順は、渡す環境変数がきわめて少ない (`NCCL_SOCKET_IFNAME` /
`UCX_NET_DEVICES` / `OMPI_MCA_btl_tcp_if_include` の 3 つで、すべて管理インターフェース
を指す)。一方、NVIDIA 自身の vLLM 向けの 2 台手順は、同じ変数群を QSFP 側に向けている。
この矛盾は、この構成では A/B で解く (research.md §e-3、Decision 10)。

## 決めたこと

### 1. 最初の通信の設定 (Decision 10)

`serving/config/configs.toml` の `p1-nvfp4-tp2`、`netcheck-bandwidth`、`netcheck-sanity`
の各構成に、次の 3 つだけを入れた。

| 変数 | 値 | 出典 |
|---|---|---|
| `VLLM_HOST_IP` | ノードごとの直結側のアドレス (`{node.fabric_addr}`) | `vllm/envs.py`: "used in distributed environment to determine the ip address of the current node, when the node has multiple network interfaces." |
| `NCCL_SOCKET_IFNAME` | `=` 付きで直結側のインターフェース名 (`={node.fabric_ifname}`)。先頭の `=` は前方一致でなく完全一致にするための NCCL の書式 | NCCL env docs: "To match (or not) an exact interface name, begin the prefix string with the `=` character." |
| `GLOO_SOCKET_IFNAME` | `NCCL_SOCKET_IFNAME` と同じインターフェース名 (`=` は付けない。Gloo は完全一致の書式が文書化されていない) | PyTorch distributed docs: "you can override it using the following environment variables … `GLOO_SOCKET_IFNAME`" |

`NCCL_IB_HCA`、`NCCL_IB_MERGE_NICS`、`NCCL_IB_GID_INDEX` はどれも設定しない (既定のまま)。
値は `nodes.toml` の `fabric_addr` / `fabric_ifname` を根拠にするが、**この 2 つは実測
(タスク 7.3) まで空のままにしてある**ので、この記録の時点では、2 台にまたがる構成
(`p1-nvfp4-tp2`、`netcheck-bandwidth`、`netcheck-sanity`) はまだ読み込めない (要件 4.7)。

### 2. ケーブルの本数の判断と PLAN.md の訂正 (要件 4.2)

この節は、7.3 (`serve netcheck links` の実測) で埋める。分かっている事実 (研究段階、
research.md §e-1) だけ、ここに記録する。

- **判定の基準**: つながっている直結のインターフェースが 2 つなら「ケーブル 1 本」、4 つなら
  「ケーブル 2 本」(NVIDIA の公式資料: 1 つの QSFP ポートが 2 つのインターフェースとして
  見えるため)
- **PLAN.md の訂正の要否**: PLAN.md 38 行の「直結リンク 2 本」「1 本あたり約 112Gb/s」
  「MTU 9000」は、上の「背景」に書いたとおり、NVIDIA の公式資料と食い違う疑いが強い。
  実測のあとに、`fabric_measured` の要約を根拠にして PLAN.md を直す (7.3 の仕事。ここでは
  まだ直さない)

### 3. NVIDIA の参照点 (表示だけに使い、合否には入れない)

| 値 | 何の値か | 出典 |
|---|---|---|
| 189.85 Gbps | `ib_write_bw` による、2 つの PCIe function を束ねた実測の合計 (92.57 + 97.28 Gbps) | NVIDIA の性能計測の手引き |
| 184 Gbit/s | NVIDIA Sync のクラスタアシスタントが確認する下限 | NVIDIA Sync の文書: "NVIDIA Sync then runs a speed test across the links to check the lower bound of 184 Gbit/s." |
| 175 Gbps (= 21.875 GB/s) | NVIDIA 自身が持つ、all_gather の合否のしきい値 (Spark のクラスタ設定スクリプトの定数) | 同スクリプトの定数 |

測り方 (`ib_write_bw`、all_gather) が、この構成の測り方 (自前の `torchrun` の all-reduce、
`allreduce_bench.py`) と異なるため、**これらの値は比べて示すだけで、合否の判定には使わない**
(要件 4.3、Decision 7)。

## 採らなかった案

| 案 | 採らなかった理由 |
|---|---|
| 最初から `NCCL_IB_HCA` / `NCCL_IB_MERGE_NICS` / `NCCL_IB_GID_INDEX` を設定する | NVIDIA 自身の 2 台直結の手順は `NCCL_IB_HCA` を設定していない ("NCCL discovers them — no need to name them")。`NCCL_IB_MERGE_NICS` の既定 (1) が、1 本のケーブルの 2 つの PCIe function を束ねて約 190 Gbps を出す仕組みそのもので、触ると壊れる。`NCCL_IB_GID_INDEX` は NCCL 2.21 以降「設定してはいけない」と公式が書く (イメージの NCCL は 2.30.7) |
| nccl-tests (MPI 経由) で帯域を測る | マルチノードには MPI が要り、Spark へのインストールは恒久的な設定変更に当たる (要件 2.7 / 2.8)。固定したイメージに PyTorch と NCCL が入っているので、`torchrun` を使った自前の計測 (Decision 7) のほうが、Spark に何も入れずに済む |
| `NCCL_SOCKET_IFNAME` を、A/B せずに直結側 (vLLM 手順) か管理側 (NCCL 手順) のどちらかに決め打ちする | NVIDIA 自身の資料の中で 2 つの手順が矛盾しており、どちらが速いかを決める根拠が公式資料だけでは出ない。A/B (関門 A.4) で実測してから決める (要件 4.5) |

## 影響と限界

- 最初の設定は、最適でない可能性がある (意図どおり。詰めは P5 で行う。research.md
  Decision 10 の Trade-offs)
- DGX Spark は GPUDirect RDMA に対応しない (NVIDIA の移植ガイド)。NCCL は GDR を使わず、
  ホスト経由の段取りに落ちる。ログで見るべきは `via NET/IB/GDRDMA` の有無ではなく、
  `NET/IB` が使われていて `NET/Socket` でないことである (research.md §e-2)

## 見直す条件

次のどれかをしたら、この記録を見直す。

| 変えたもの | すること |
|---|---|
| `serve netcheck links` (7.3) でケーブルの本数を確定した場合 | 「決めたこと」の節 2 と PLAN.md を、実測の要約を根拠にして埋める |
| 最初の A/B (`NCCL_SOCKET_IFNAME` の直結側 / 管理側。関門 A.4) の結果が出た場合 | 「決めたこと」の節 1 に、採否と実測の根拠 (`docs/results/…`) を追記する |
| A.4 の候補 (`--device /dev/infiniband`、`--cap-add SYS_NICE`、`--cap-add IPC_LOCK`) を A/B で確かめた場合 | 採用したものだけを、実測の根拠つきで追記する |
| 通信の確認 (`netcheck sanity`) が通らない場合 | procedure.md の「関門 A」の「止める条件」に従い、`not-working.md` に書いてから、この記録に原因の分析を追記する |

## クリーンルーム (要件 11.4、11.5、11.6)

**0002 と同じ日 (2026-09-22)、同じ性質の作業者 (この会話の文脈を持たない新しい
サブエージェント) が、research.md と道具の一次の資料だけから、通信に関わる構成の値を
書いた。** ssh・docker は使わせず、第三者のレシピは開かせていない。

**参照した資料の一覧 (要件 11.6)**: 0002 の一覧 (vLLM のソースと公式文書、Docker の公式
文書、PyTorch `run.py`) に加えて、通信に固有の資料は次のとおり (research.md §e の
Sources Consulted)。

- **NCCL 環境変数リファレンス**:
  `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html`
- **DGX Spark ユーザーガイド (ConnectX-7 Networking)**:
  `https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html`
- **DGX Spark 移植ガイド (CUDA)**:
  `https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html`
- **vLLM の並列とスケーリング / トラブルシュートの文書**:
  `https://docs.vllm.ai/en/latest/serving/parallelism_scaling/`、
  `https://docs.vllm.ai/en/latest/usage/troubleshooting/`
- **PyTorch の分散の文書**: `https://docs.pytorch.org/docs/stable/distributed.html`
- **nccl-tests の README** (帯域の測り方の比較のためだけに参照。コードは写していない):
  `https://github.com/NVIDIA/nccl-tests`
- **自分たちで測った事実**: この構成の `serve netcheck bandwidth` / `sanity` / `ab` の
  実測 (7.4 で追記)

## 実機で確かめたこと

この節は、実機の段 (7.3〜7.4) で埋める。予定している内容:

- `serve netcheck links` が読んだ、2 台のインターフェースの名前・状態・MTU・速さ・
  RoCE デバイスとの対応 (要件 4.1)
- ケーブルの本数の判断と、PLAN.md の訂正の実物の差分 (要件 4.2)
- `serve netcheck bandwidth` の帯域の実測と、上の参照点との比較 (要件 4.3)
- `serve netcheck sanity` の 4 段の合否 (要件 4.6)
- 最初の A/B (`NCCL_SOCKET_IFNAME` の直結側 / 管理側) の採否と、実測の根拠 (要件 4.5)
