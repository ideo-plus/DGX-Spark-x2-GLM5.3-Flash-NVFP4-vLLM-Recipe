# 2026-09-22 2 台の間の all-reduce の帯域 (`serve netcheck bandwidth`)

P1 (vllm-baseline) のタスク 7.4。作業用の Mac から `uv run serve netcheck bandwidth <構成> --yes` で、
固定したイメージ (`vllm/vllm-openai@sha256:b0501f99…`、NCCL 2.30.7+cuda13.3) の中の `torchrun` で
自前の all-reduce の計測 (`serving/payload/allreduce_bench.py`) を 2 台で流した。1 MiB〜1 GiB の
6 つの大きさ、温め 5 回 + 計測 20 回。`busbw = algbw × 2(n−1)/n` で、2 台では `busbw = algbw`。

NVIDIA の参照点 (表示だけで、合否には使わない): DGX Spark 2 台の 200 GbE の直結で 189.85 Gbps
(実測の公表値)、184 Gbit/s (下限)、175 Gbps (しきい値) — research.md §e。

## 回 1: 最小の設定 (`netcheck-bandwidth`) — 不合格 (経路が Socket)

設定: `NCCL_SOCKET_IFNAME==enp1s0f0np0`、`GLOO_SOCKET_IFNAME=enp1s0f0np0`、`NCCL_DEBUG=INFO`、
`NCCL_DEBUG_SUBSYS=INIT,NET`、`NCCL_DEBUG_FILE=/logs/nccl-<回の札>.%h.%p.log`。docker は
`--gpus all --ipc host --shm-size 16g --ulimit memlock=-1 --network host` と `payload/`・`logs/` の結び付け。
**`--device /dev/infiniband` は付けていない** (design.md 「最初は入れない。実測の根拠で足す」)。

| 大きさ | busbw (Gbps) |
|---|---|
| 1 MiB | 2.9 |
| 4 MiB | 9.2 |
| 16 MiB | 13.2 |
| 64 MiB | 15.7 |
| 256 MiB | 16.1 |
| 1 GiB | 15.9 |

NCCL の記録 (2 台とも同じ): `NET/IB : No device found.` → `NET/Socket : Using [0]enp1s0f0np0:192.168.100.10<0>`
→ `Using network Socket`。つまり **コンテナの中から RDMA のデバイス (`/dev/infiniband`) が見えず、
NCCL が TCP (Socket) に落ちた**。200 Gb/s のリンクの上で TCP の 16 Gbps は、直結の経路が
使われていないことの実測である (`observe_nccl` の判定: `network == "Socket"`、IB なし、
`[send] via NET/Socket` のチャンネルあり → 不合格)。記録: `serving/var/20260922T123252Z-netcheck-netcheck-bandwidth/`。

→ 足す候補 (design.md の「最初は入れない」の 3 つのうち、この実測が根拠になるもの):
`--device /dev/infiniband` (コンテナに RDMA のデバイスのノードを渡す)。それでも登録に失敗するなら
`--cap-add IPC_LOCK`。回 2 で確かめる。

## 回 2: `--device /dev/infiniband` を足した設定 (`netcheck-bandwidth-ib`) — 合格 (経路が IB)

回 1 との差は docker の `--device /dev/infiniband` の 1 つだけ (A/B)。

| 大きさ | busbw (Gbps) | 回 1 との比 |
|---|---|---|
| 1 MiB | 9.9 | 3.3× |
| 4 MiB | 58.2 | 6.4× |
| 16 MiB | 136.0 | 10.3× |
| 64 MiB | 158.6 | 10.1× |
| 256 MiB | 161.6 | 10.1× |
| 1 GiB | **186.9** | 11.7× |

NCCL の記録 (head): `NET/IB : Using [0]rocep1s0f0:1/RoCE [1]roceP2p1s0f0:1/RoCE [RO]; OOB enp1s0f0np0:192.168.100.10<0>`
→ `Using network IB` → `Connected all rings`。**1 本の QSFP ケーブルの 2 つの RoCE のデバイスを、
NCCL が既定で両方使っている** (`NCCL_IB_HCA` を指定していない)。`[send] via NET/Socket` の
チャンネルはない。2 台とも同じ。記録: `serving/var/20260922T123535Z-netcheck-netcheck-bandwidth-ib/`。

1 GiB の busbw 186.9 Gbps は、NVIDIA のしきい値 175 Gbps (nccl-tests の all_gather) を上回り、
公表の実測 189.85 Gbps (ib_write_bw) に近い (道具が違うので、そのままの比較にはならない)。

## 判断 (ADR 0003)

- **`--device /dev/infiniband` を採用する** (実測の根拠: この文書)。`netcheck-sanity` と
  `p1-nvfp4-tp2` にも、同じ `measured` を付けて足す
- `--cap-add IPC_LOCK`、`--cap-add SYS_NICE`、`NCCL_IB_HCA`、`NCCL_IB_MERGE_NICS`、
  `NCCL_IB_GID_INDEX` は、**足さない** (既定のままで IB の経路が使われ、2 つの RoCE の
  デバイスが両方使われ、しきい値を上回った。足す根拠がない)
- 2 台の直結は「QSFP ケーブル 1 本 = 200 Gb/s」であり、NCCL の busbw 186.9 Gbps はその上限に近い。
  PLAN.md の「直結リンク 2 本、1 本あたり約 112 Gb/s」は、実測に合わせて直す
