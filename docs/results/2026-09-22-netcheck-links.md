# 2026-09-22 直結のインターフェースの実測 (`serve netcheck links`)

P1 (vllm-baseline) のタスク 7.3。作業用の Mac から `uv run serve netcheck links` (読み取りだけ) で、
2 台の DGX Spark のインターフェースを読んだ。`serving/config/nodes.toml` の直結の値 (`fabric_*`) の
根拠 (`fabric_measured`) であり、PLAN.md の「ハードウェアと環境」の訂正の根拠でもある。

書かないこと: 機械に固有のアドレス (MAC、リンクローカルの IPv6) は載せない (要件 10.5)。

## 読めたこと

| 台 | インターフェース | 状態 | MTU | 速さ | RoCE のデバイス | IPv4 |
|---|---|---|---|---|---|---|
| head (spark-153d) | `enp1s0f0np0` | UP | 9000 | 200000 Mb/s | `rocep1s0f0` | 192.168.100.10/24 |
| head | `enp1s0f1np1` | DOWN (ケーブルなし) | 1500 | 不明 | `rocep1s0f1` | — |
| head | `enP2p1s0f0np0` | UP | 9000 | 200000 Mb/s | `roceP2p1s0f0` | 192.168.101.10/24 |
| head | `enP2p1s0f1np1` | DOWN (ケーブルなし) | 1500 | 不明 | `roceP2p1s0f1` | — |
| worker (spark-5083) | `enp1s0f0np0` | UP | 9000 | 200000 Mb/s | `rocep1s0f0` | 192.168.100.11/24 |
| worker | `enp1s0f1np1` | DOWN (ケーブルなし) | 1500 | 不明 | `rocep1s0f1` | — |
| worker | `enP2p1s0f0np0` | UP | 9000 | 200000 Mb/s | `roceP2p1s0f0` | 192.168.101.11/24 |
| worker | `enP2p1s0f1np1` | DOWN (ケーブルなし) | 1500 | 不明 | `roceP2p1s0f1` | — |

読み取りの道具は 2 台とも揃っていた (`ip`、`ethtool`、`ibdev2netdev`、`/sys/class/net/<if>/mtu`)。

## ケーブルの本数の判断 (要件 4.2)

**1 本**。つながっている直結の候補は 2 台とも 2 つ (`enp1s0f0np0`、`enP2p1s0f0np0`) だが、
NVIDIA の DGX Spark ユーザーガイド (ConnectX-7 Networking) のとおり、1 つの QSFP ポートが
2 つの Linux のインターフェースとして見えている (research.md §e-1)。`f1np1` (2 つめの QSFP ポート)
は 2 台とも DOWN (ケーブルなし)。したがって PLAN.md の「直結リンク 2 本」は誤りで、
**QSFP ケーブル 1 本 (200 Gb/s) が、2 つの Linux のインターフェース (2 つの RoCE のデバイス)
として見えている**、が実測である。

## `nodes.toml` に書いた直結の値

| 台 | `fabric_ifname` | `fabric_addr` |
|---|---|---|
| head | `enp1s0f0np0` | 192.168.100.10 |
| worker | `enp1s0f0np0` | 192.168.100.11 |

2 つの候補のうち `enp1s0f0np0` (192.168.100.x) を選んだ理由: 名前の並びで先で、2 台で同じ名前・
同じサブネットであり、`serve netcheck links` の「埋める候補」もこれを挙げた。もう 1 つ
(`enP2p1s0f0np0`、192.168.101.x) は同じ物理のポートの別の見え方なので、NCCL の経路としては
どちらか 1 つを `NCCL_SOCKET_IFNAME` に与えれば足りる。2 つを束ねる (`NCCL_IB_HCA` に並べる)
かどうかは、7.4 の帯域の計測と A/B で決める (ADR 0003)。
