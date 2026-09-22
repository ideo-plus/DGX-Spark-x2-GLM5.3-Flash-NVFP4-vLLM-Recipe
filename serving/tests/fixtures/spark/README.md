# 2 台の Spark から採った、読み取りのコマンドの実物の出力

`guards`、`lifecycle` の状態の確認、`netcheck links` の試験が、偽の実行役 (`tests/fake_runner.py`) の台本の見本として使う。出力の形を推測で書かないために、実機から採った (design.md の Testing Strategy、tasks.md の 1.6)。

- 採った日: 2026-09-21
- 採り方: 作業用の Mac から `ssh -o BatchMode=yes -o ConnectTimeout=5 -- <host> '<コマンド>'`。読み取りだけで、Spark の状態は変えていない
- `head/` は `spark-153d` (10.0.1.60)、`worker/` は `spark-5083` (10.0.1.61)
- 採ったときの状態: 2 台とも、GPU を使っているプロセスはない。この道具のコンテナは、まだ 1 つもない。`~/vllm-baseline/` は、まだない

## ファイルとコマンド

| ファイル | コマンド | 終了コード |
|---|---|---|
| `uname-n.txt` | `uname -n` | 0 |
| `docker-version.txt` | `docker version` | 0 |
| `docker-ps-own.txt` | `docker ps -a --filter label=vllm-baseline.owner=serving-kit --format json` | 0 |
| `nvidia-smi-compute-apps.txt` | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` | 0 |
| `nvidia-smi-gpu-util.txt` | `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` | 0 |
| `df-avail.txt` | `df -B1 --output=avail /home/j5ik2o` | 0 |
| `ip-br-link.txt` | `ip -br link` | 0 |
| `ip-br-addr.txt` | `ip -br addr` | 0 |
| `ss-ltnH.txt` | `ss -ltnH` | 0 |
| `ethtool-<if>.txt`、`ethtool-<if>.stderr.txt` | `ethtool <if>` | 0 |

| `mtu-<if>.txt` | `cat /sys/class/net/<if>/mtu` | 0 |
| `ip-link-show-enp1s0f0np0.txt` | `ip link show dev enp1s0f0np0` | 0 |
| `ibdev2netdev.txt` | `ibdev2netdev` | 0 |
| `ibv_devinfo.txt` | `ibv_devinfo` | 0 |
| `docker-image-inspect-repodigests.txt` | `docker image inspect vllm/vllm-openai@sha256:b0501f99… --format '{{json .RepoDigests}}'` | 0 |

下の 4 つは、2026-09-22 に足した (4.3 の `netcheck links` が、MTU と、RoCE のデバイスとの対応を読む形を、推測で書かないため)。`ibdev2netdev` と `ibv_devinfo` は、2 台とも入っている。

`ethtool` は、`enP7s7` (管理の側、10 GbE)、`enp1s0f0np0` と `enP2p1s0f0np0` (直結の側、つながっている)、`enp1s0f1np1` (直結の側、つながっていない) の 4 つについて採った。

## 読むときの注意

- **空のファイルは、空の出力の見本である**。`docker-ps-own.txt` (自分のコンテナがない) と `nvidia-smi-compute-apps.txt` (GPU を使っているプロセスがない) は、0 バイトで、終了コードは 0
- **`ethtool` は、root でなくても終了コード 0 で終わるが、標準エラーに `netlink error: Operation not permitted` を 1 行出す**。標準出力の `Speed:`、`Link detected:` は読める。読み取りの部品は、この標準エラーを誤りとして扱わないこと
- つながっていないインターフェースは、`Speed: Unknown!`、`Duplex: Unknown! (255)`、`Link detected: no (No cable)` になる
- 直結の側は、2 台とも、つながっているインターフェースが 2 つある (`enp1s0f0np0` と `enP2p1s0f0np0`。どちらも `Speed: 200000Mb/s`)。ケーブルの本数の判断と、PLAN.md の訂正は、7.3 で行う (ここでは、見本を採っただけ)
- `df` は、`remote_root` がまだないので、その親の `/home/j5ik2o` について採った。出力の形 (見出しの行 `Avail` と、バイト数の行) は同じ
- GB10 のユニファイドメモリで `[N/A]` になる列があるかは、ここでは確かめられていない (`--query-compute-apps` の `used_memory` は、プロセスがないので行が出ない。`utilization.gpu` は `0 %` と数で出た)。7.2 で、自分のコンテナが GPU を使っているときの出力を採って、確かめる
- **`docker-image-inspect-repodigests.txt` は、2026-09-22 (7.1) に、`serve pull-image` のあとに採った** (2 台とも、固定したダイジェストの 1 つだけの配列)。`gate_image_digest` の見本
- **ここで採れないもの** (7.2 で採って、足す): 自分のコンテナがあるときの `docker ps` の JSON、`docker container inspect` の状態、GPU を使っているプロセスがあるときの `--query-compute-apps` (7.1 の読み取りのコンテナは数秒で終わるので、採らなかった)

## 採らなかったもの

- **絞らないコンテナの一覧 (`docker ps -a`) は、採っていない、見ていない**。ラベルのないコンテナのイメージとコマンド (計測者が別に起動していた構成の中身) が入るため。コンテナの一覧は、最初から、この道具のラベルで絞って採った
- ラベルのないコンテナには、`docker inspect` も `docker logs` も向けていない

## 置き換えたもの

公開に備えて、機械を識別できる値を、形を保ったまま置き換えた。試験は、置き換えたあとの値に依存してよい。

| もとの値 | 置き換えたあと |
|---|---|
| `ibv_devinfo` の GUID (MAC から作られる) | `0200:0003:0000:00xx` (置き換えた MAC から作った値) |
| MAC アドレス (16 個) | `02:00:00:00:00:01`〜 (ローカル管理のアドレス) |
| グローバルの IPv6 アドレス (6 個) | `2001:db8::1`〜 (文書用の範囲) |
| Tailscale の IPv4 アドレス (2 個) | `100.64.0.1`、`100.64.0.2` |
| Tailscale の IPv6 アドレス (2 個) | `fd00:db8::1`、`fd00:db8::2` |
| リンクローカルの IPv6 アドレス (12 個) | `fe80::1`〜 (合成の値。実物は、仮想のインターフェースでは modified EUI-64 で MAC をそのまま含み、物理のインターフェースでも機械に固有の値なので、すべて置き換えた) |
| docker のブリッジの名前 `br-<12 桁>`、`veth<7 桁>` | `br-000000000000`、`veth0000000` |

- `ss-ltnH.txt` は、アドレスの列が右に寄せて出るので、置き換えたあとも、行の長さが変わらないように、左に空白を詰めた
- LAN のアドレス (10.0.1.60、10.0.1.61)、直結の側のアドレス (192.168.100.x、192.168.101.x)、docker の既定のブリッジのアドレス (172.17.0.1、172.18.0.1)、ホストの名前は、そのまま (LAN のアドレスとホストの名前は、CLAUDE.md にすでにある)
- 認証の情報、送った内容、応答の本文は、もともと含まれていない
- `ss-ltnH.txt` の待ち受けのポート (22、53、631 のほか、8080、5555、5556、11000 など) は、この道具のものではない。何が待ち受けているかは、調べていない (ポートの番号だけを、`gate_ports_free` の見本として採った)
- `ip -br link` は MTU を出さない。MTU は、`cat /sys/class/net/<if>/mtu` (数だけの 1 行) か、`ip link show dev <if>` の `mtu <数>` で読む (どちらも、`remote` の許可の形を通る)。直結の側 (`enp1s0f0np0`、`enP2p1s0f0np0`) は 9000、管理の側と、つながっていないものは 1500
- `ibdev2netdev` の 1 行は `<RoCE のデバイス> port <番号> ==> <インターフェース> (Up|Down)`。つながっている直結のインターフェース 2 つが、別々の RoCE のデバイス (`rocep1s0f0`、`roceP2p1s0f0`) に対応している
- `ibv_devinfo` の `node_guid` / `sys_image_guid` は、MAC から作られる (`aa:bb:cc:dd:ee:ff` → `aabb:cc03:00dd:eeff`) ので、置き換えた MAC から作った値に置き換えた。`link/ether` の MAC も、`ip-br-link.txt` と同じ対応で置き換えた (`brd ff:ff:ff:ff:ff:ff` は、そのまま)
