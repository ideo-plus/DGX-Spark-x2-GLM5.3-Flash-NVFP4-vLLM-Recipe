# vllm-autostart

Spark 2 台の上で、推論サーバーの自動起動・ヘルスチェック・立ち上げ直しを行う一式 (issue #88 P6)。
`ops/spark-power-caps` と同じ型 (README・unit・install/uninstall・sudoers) で置く。

## 設計の答え (issue #88 やること 1)

固定した vLLM (`0961bbae2894d574be790d219651824eb199318e`) のソースと、既存の
`serving_kit` のコードを根拠に、5 つの問いに答える。

### D1. 何が起動を担うか

各ノードのユーザー単位の systemd (`~/.config/systemd/user/vllm-autostart.service`,
`Type=simple`, `Restart=on-failure`, `WantedBy=default.target`) が、
`vllm_autostart.py --remote-root %h/vllm-baseline` を常駐させる。この script は標準ライブラリ
だけに依存し (`serving_kit` を import しない。Spark には配らないため)、`<remote_root>/state/
<構成>.launch.json` に置かれた、その構成のその役割の `plans[].argv` (`serve start` が
すでに了承して置いた、固定の `docker run` の引数) を、argv の形 (`docker run` であること、
`--name`、所有と構成のラベル) を確かめたうえで、**1 語も変えずに**流す。Spark の上に、
構成の定義そのもの (`configs.toml` 相当) は要らない。linger (再起動をまたいでユーザー単位の
systemd を動かし続ける設定) を有効にしておけば、Mac が無くても、Spark の再起動のあとに
自分で戻る。

### D2. どの構成を自動起動するか

Mac の `serve autostart set <構成> --yes` が、両台の `launch.json` の `config_sha256` が
Mac の計画と一致することを確かめてから、`<remote_root>/state/autostart.json` を配る
(`schema_version`, `role`, `config`, `config_sha256`, `ready_timeout_s`, `ports`,
`weights_record`, `designated_at`, `repo_commit`, `repo_dirty`)。`serve autostart clear --yes`
は `config: null` を配る。見張りは、指定が無い・`config` が `null`・`launch.json` の
`config_sha256` が指定と違う、のどれかでは、`docker run` を 1 度も流さない。

### D3. 2 台をどうそろえるか

head から worker へは ssh できないので、2 台は互いに独立した見張りとして動く。両台とも、
自ノードの `docker run` の引数に含まれる `--host`/`--port` (`configs.toml` の
`args.host` が `{head.lan_addr}` の置き換えの印を持つため、**head の LAN アドレスが両台の
引数に入る**) を使って、head の `/health` を見る。

固定した vLLM のソースで確かめた振る舞い (道筋は source root 相対):

| 問い | 確認できたこと (ファイル + シンボル名、commit `0961bbae`) |
|---|---|
| 起動順序 | `vllm/entrypoints/cli/serve.py` の `node_rank_within_dp > 0` 分岐が、worker だけを `MultiprocExecutor(vllm_config, monitor_workers=False)` + `start_worker_monitor(inline=True)` で動かす。rank 1 の分散グループへの参加 (`torch.distributed.init_process_group`、NCCL の既定タイムアウト 600 秒) と head の起動待ちループ (`wait_for_engine_startup`) は `vllm/distributed/parallel_state.py`・`vllm/config/parallel.py`・`vllm/v1/engine/utils.py` にあるが、今回はシンボル単位での再確認をしていない。**起動順序は不要で、600 秒以内に相手が来ればよい** |
| head が落ちたとき | `vllm/v1/executor/multiproc_executor.py` の `WorkerProc.worker_busy_loop` は `rpc_broadcast_mq.dequeue(indefinite=True)` で無期限に待つ (`vllm/distributed/device_communicators/shm_broadcast.py` の `MessageQueue.dequeue`)。heartbeat は無い。`monitor_death_pipe` は**同一ノードの親プロセスの終了 (`EOFError`) だけ**を見る仕組みで、別ノードの head の死は検知しない。**worker は自分では終わらない (外から止める必要がある)** |
| worker が落ちたとき | head は直接検知しない (`multiproc_executor.py` は自ノードだけを見る)。**idle の間は `/health` が 200 のまま**。次の要求で `vllm/envs.py` の `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (既定 `"300"`、TP > 1 のみ) 後に `collective_rpc("execute_model", timeout=...)` が時間切れになり、engine core が終了 → API サーバーが終了する経路がある |
| `/health` の中身 | `vllm/v1/engine/async_llm.py` の `AsyncLLM.check_health` は `self.errored` (= `engine_core.resources.engine_dead or not is_running`) だけを見る。**TP の相手の死活は見ない** |
| 片方だけの再起動 | プロセスグループも NCCL の unique id も 1 回だけ作る (`vllm/distributed/parallel_state.py`、`vllm/distributed/device_communicators/pynccl.py`)。multi-node TP の elastic な再参加は不可。**2 台そろって起こし直す必要がある** (今回のソース確認では再検証していない。ソースの読み取りに基づく結論で、実機では未確認。下記「実機で未確認のこと」参照) |

これらから、次の立ち上げ直しの手順にした:

- **worker 死亡の検知**: idle の間は head の `/health` が 200 のままなので、`/health` だけでは
  検知できない。head は、ready のあと 60 秒ごとに 1 トークンの推論プローブ
  (`POST /v1/chat/completions`, `max_tokens: 1`) を送り、3 回連続で失敗したら、head 自身を
  止めて起こし直す (worker が落ちていれば、この立ち上げ直しで両台がそろう)
- **head 死亡の検知**: worker には 2 つの検知の経路がある。
  1. **自コンテナが `exited` になった場合**: worker だけを起こし直しても、head が生きて
     いなければ 600 秒のランデブーの窓の中でしか待てず、無駄になる。worker の見張りは、
     自コンテナが `exited` になったら、head の `/health` を見て、**200 の間は起こさず待つ**
     (head がまだ生きているなら、head 側の検知で両台がそろって起こし直されるのを待った
     ほうがよい)。head が 200 でなくなったら、worker も起こし直す。600 秒待っても head が
     200 のままなら、打ち切って起こす (未確認の経路。下記)
  2. **自コンテナが running のままの場合**: idle の間は head の `/health` が 200 のままな
     ので、worker が自分の状態だけを見ていても head の死亡には気づけない。そこで worker
     は running のまま、自分でも head の `/health` を見て、**180 秒 (`UNHEALTHY_AFTER_S`。
     head 自身の 180 秒判定 (D4) と同じ定数を使う。同じ「head の `/health` が落ちた」という
     事実の判定なので、値をそろえる) 連続で 200 でなければ、自分を起こし直す**。head の
     プローブ (60 秒間隔、3 回連続失敗で立ち上げ直し) は head だけが行い、worker はプローブ
     を送らない。この経路が実機での head 死亡の検知に掛かる実際の時間は、下記「実機で
     未確認のこと」参照
- **片方だけの再起動をしない**: 上のとおり、`docker run` は、必ず「自コンテナを
  stop → rm → run」で完結させ、相手のコンテナには触れない (相手は、相手の見張りが同じ理由で
  独立に起こす)

### D4. ヘルスチェックの中身

30 秒ごとに `/health` (タイムアウト 10 秒)。**起こしてから `ready_timeout_s` までは、起動中
として扱い、`/health` の失敗を「落ちた」に数えない**。ready になったあとの head は、
`/health` が 180 秒連続で落ちる、または推論プローブが 3 回連続 (60 秒間隔、60 秒タイムアウト)
で失敗したら、自コンテナを `docker stop -t 90` → `rm` → 起こし直す。

**起動の失敗** (ready 前の終了、`ready_timeout_s` の超過) が**連続 3 回**続いたら `halted` に
し、以後 `docker run` を流さない (無限に起こし続けない)。状態ファイル
(`state/autostart.status.json`) に理由を書く。解除は `serve autostart set` の再実行
(`designated_at` が変わる) で、**見張りのプロセスを再起動しなくても**、同じインスタンスが
次の周から自動で再開する。ready になれば、失敗の回数は 0 に戻る。

状態ファイルの鍵は固定集合 (`schema_version`, `role`, `config`, `container_name`, `state`,
`reason`, `updated_at`, `started_at`, `ready_at`, `container_state`, `container_exit_code`,
`last_health_status`, `last_probe_status`, `consecutive_start_failures`) で、要求・応答の
本文や `docker logs` は含まない (要件 23。異常終了したコンテナの記録は、立ち上げ直しで
失われる。これは、この設計の限界として受け入れる)。

### D5. 自動で起こす前に、Spark の上で何を確かめるか

`serve start` の関門は 9 種 × 2 台 = 18 件 (`guards.GATE_ORDER`、issue #84 の `memory_free` を
含む)。見張りは ssh を使わないので、Spark の上だけで確かめられるものを選ぶ。

| 関門 | Spark の上で確かめるか | 確かめ方 |
|---|---|---|
| `own_state` | 確かめる | 自ラベルの `docker ps -a --filter label=vllm-baseline.owner=serving-kit --format json` |
| `gpu_idle` | 確かめる | `nvidia-smi --query-compute-apps=pid,process_name,used_memory` に行があれば断る |
| `layout` | 確かめる | argv の `--mount` (`source=`/`src=` の両方) の元が存在するか |
| `image_digest` | 確かめる | `launch.json` の `image_digest` を `docker image inspect` の `RepoDigests`/`.Id` と照合 (`guards.py` の `gate_image_digest` と同じ規則) |
| `weights_verified` | 確かめる | Mac が指定 (`autostart.json` の `weights_record`) に埋め込んだ期待値 (道筋、`scope`、`file_count`、`total_bytes`、`fields`) と、実際の照合の記録を突き合わせる。重みを持たない構成では指定が無いので検査しない |
| `ports_free` | 確かめる | 指定の `ports` を `ss -ltnH` の待ち受けの一覧と照合 |
| `memory_free` | 確かめる | `--load-format instanttensor` の argv だけ、`docker run` の前に、root 不要の固定 argv (`find <remote_root>/models -type f -exec dd if={} iflag=nocache count=0 status=none ;`) でページキャッシュを捨ててから、`cat /proc/meminfo` の `MemFree` を下限 8 GiB (`INSTANTTENSOR_MEMFREE_FLOOR_BYTES`。`guards.py` と同じ値) と比べる。足りなければ起こさず `refused` にし、理由 (`reason`) に `MemFree` の値を書く。`sudo`/`spark-drop-caches` は呼ばない (root の操作を自動で流さない。issue #88 コメント1・更新 2026-09-28) |
| `reachable` | 確かめない | 自ノードで動くので、ssh は要らない (この関門の前提そのものが無い) |
| `disk_space` | 確かめない | 起動は新たに書き込まない (重みの取得ではない)。`serve start` の時点で足りていたディスクが、その後 Spark の他の用途で急に減る想定はしていない (`guards.py:426-435` の「要る量 = 重みの合計」を、見張りが再計算する意味が薄い) |

## D6. Mac の口

- `serve autostart set <構成> --yes` — 指定する (状態を変える)
- `serve autostart clear --yes` — 解除する (状態を変える)
- `serve autostart status` — 両台の指定と状態を、`cat` で読むだけで示す (状態を変えない。
  構成ファイルを読まなくても動く)
- 指定が有効なまま `serve stop` を打つと、途中の周を見張りが観測して起こし直すことがある
  ので、`serve stop` の前に必ず `serve autostart clear --yes` で指定を解除する
  (打つ手順は `autostart-procedure.md` の「指定がある間の停止」)

設置は、`serve push` による配布 (`serving/payload/vllm-autostart/` → 両台の `payload/`) までで、
`systemctl --user enable --now` は、計測者が Spark の上で `vllm-autostart-install` を
1 コマンド打つ (`remote.py` の許可の一覧に `systemctl` を足さない。要件 21)。

## D7. 設置と外し方

root が要るのは **linger の切り替えだけ**である。`vllm-autostart.sudoers` は、次の 2 つの
固定形だけを `j5ik2o` に NOPASSWD で許す。

```
j5ik2o ALL=(root) NOPASSWD: /usr/bin/loginctl enable-linger j5ik2o, /usr/bin/loginctl disable-linger j5ik2o
```

ユーザー側 (root が要らない) は、`vllm-autostart-install` / `vllm-autostart-uninstall`
(どちらも POSIX sh) で、unit の設置・有効化・撤去・無効化を行う。

打つ手順 (設置・指定・短い実機試験・外し方) は
[`docs/vllm-baseline/autostart-procedure.md`](../../docs/vllm-baseline/autostart-procedure.md)
にある。

## 確かめない関門の理由 (再掲)

- `reachable`: 見張りは自ノードで動くので、ssh の到達性という前提そのものが無い
- `disk_space`: 起動は新たに書き込まない。`serve start` が済んだあとの、Spark 側の別要因での
  急な圧迫までは見ない

## 実機で未確認のこと

- worker が head の `/health` を待つ間、600 秒 (`WORKER_WAIT_FOR_HEAD_S`) を超えても head が
  200 を返し続ける経路 (head 側の検知が worker より遅れる場合。打ち切って起こす設計だが、
  実機での発生頻度は未確認)
- `/usr/bin/python3` の実在 (Ubuntu 24.04 の既定を前提にしている)
- `docker ps -a --filter label=… --format json` の実物の 1 行の形 (`guards.py` が読む形と
  同じと仮定している)
- head の engine 死亡経路 (worker 死亡 → head のタイムアウト → engine core 終了) の、実際の
  コンテナの終了コード
- 形成済みの分散グループ (NCCL の unique id、プロセスグループ) へ、新しい rank 1 が
  再参加したときの、実際の挙動 (ソースからは「不可」と読んだが、実機では未確認)
- linger を有効にしたあと、Spark の再起動から `docker` (dockerd) が使えるようになるまでの
  実際の待ち時間 (見張りは `docker ps` が失敗する間、待ち続ける設計だが、実測はしていない)
- `/usr/bin/loginctl` の実際の道筋 (sudoers の固定形と一致するか)
- `/usr/bin/find` の実在 (Ubuntu 24.04 の既定を前提にしている。`--find` の既定値)
- `dd iflag=nocache` によるページキャッシュ捨てが、systemd user service の環境 (手打ちの ssh
  ではなく) でも、`docs/vllm-baseline/k2-derived-weights-procedure.md` の手順で確かめたのと
  同様に効くこと
- 試験の偽の実行役が置いた、Docker の一般的な振る舞いの仮定 (単体試験の `FakeContainer`。
  issue #88 P6 の裁定・修正): 止まっているコンテナへの `docker stop` が終了コード 0、
  動いているコンテナへの `docker rm` が失敗、名前が衝突した `docker run` が失敗、開始に
  失敗したコンテナが `created` のまま残る
- 自コンテナを `docker stop` → `rm` してから、GPU とメモリ (`nvidia-smi`/`MemFree`) が実際に
  空くまでの遅れ。見張りは、次の周の前提検査で足りなければ断り、そのまま次の周でやり直す
  設計だが、実機での遅れの長さは未確認
- worker の 180 秒の head 死亡の検知と起こし直しが、新しい head が rank 1 の参加を待つ窓
  (上表の「起動順序」の行。600 秒。シンボル単位では再確認していない) に収まるかどうか

これらは、対話側が実機の設置・試験の中で確かめる。

## スコープ外

- Prometheus・Grafana の監視、熱と電源の記録 (既存の `ops/spark-power-caps` と `serve watch`
  のまま)
- 72 時間の連続稼働の試験、復旧の判定 (P8)
- `.github/workflows/ci.yml` への `systemd-analyze verify`/`visudo -cf`/`sh -n` の追加
  (書き込み範囲外。対話側かの後続の作業に申し送る)
