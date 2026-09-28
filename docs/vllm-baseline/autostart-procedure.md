# 自動起動・ヘルスチェック・立ち上げ直しの手順 (issue #88 P6)

書かないこと: 要求と応答の本文 (送った内容)、認証の情報、`docker logs` の中身。見張り
(`ops/vllm-autostart/vllm_autostart.py`) が残す記録は、状態・時刻・終了の理由・HTTP の
状態コードだけである。

設計の答え (何が起動を担うか、指定の決め方、2 台のそろえ方、ヘルスチェックの中身、Spark 上
だけで確かめる関門) の詳しい根拠は、
[`ops/vllm-autostart/README.md`](../../ops/vllm-autostart/README.md) を参照する。ここでは、
設置・指定・確認・短い実機試験・外し方の、打つ手順だけを示す。

対象は Spark の 2 台 (head: `spark-153d`、worker: `spark-5083`。ユーザーは `j5ik2o`)。
設置・有効化・実機での試験は対話側が行う (takt の作業は、コード・構成・試験の準備までが
範囲)。この手順書の、状態を変える段には「⚠ 了承を得てから」を付けてある。

## 前提の確認 (読み取りだけ)

両台にログインし、見張りが使う道具が実在することを確かめる。

```sh
command -v /usr/bin/python3 /usr/bin/docker /usr/bin/loginctl /usr/bin/find
```

## 設置

⚠ 了承を得てから、Mac から `serve push` で、`ops/vllm-autostart/` の写し
(`serving/payload/vllm-autostart/`) を両台に配る。

```bash
uv run serve push --yes
```

sudoers の断片を置く前に、構文を検査する (読み取りだけ)。

```sh
visudo -cf ops/vllm-autostart/vllm-autostart.sudoers
```

⚠ 了承を得てから、両台で `ops/vllm-autostart/vllm-autostart.sudoers` を
`/etc/sudoers.d/vllm-autostart` に置き (visudo -cf を通したあと)、linger (再起動をまたいで
ユーザー単位の systemd を動かし続ける設定) を有効にする。linger の有効化だけは、
sudoers の固定形のとおり、次の 1 コマンドで済む。

```sh
sudo loginctl enable-linger j5ik2o
```

⚠ 了承を得てから、両台で見張りの unit を設置する (`serve push` が配った
`payload/vllm-autostart/` の写しから)。

```sh
~/vllm-baseline/payload/vllm-autostart/vllm-autostart-install
```

## 自動起動の構成の指定

⚠ 了承を得てから、Mac から自動で起こす構成を指定する。両台の `launch.json` の
`config_sha256` が、この計画と一致するときだけ配られる (一致しなければ、1 度も配らず
終了コード 1 になる)。自動起動の対象は、コミット済みの構成 `glm53-tp2-mtp3-marlin`
(重み `k2s4`、MTP N=3、`--moe-backend marlin`、重ね合わせ `k2s2b` の bind mount) である。

```bash
uv run serve autostart set glm53-tp2-mtp3-marlin --yes
```

指定を止めるまで、Spark の再起動・電源断・コンテナの異常終了のあと、見張りがこの構成を
自動で起こす。

## 確認 (読み取りだけ)

Mac から、両台の指定と見張りの状態を読む。

```bash
uv run serve autostart status
```

Spark の各台でも、unit が動いていることを確かめられる。

```sh
systemctl --user is-active vllm-autostart.service
```

## 短い実機試験

起動の確認: 見張りを設置し、指定したあと、`serve autostart status` の出力の
`autostart.<役割>.status.state` が `running` になり、`ready_at` に時刻が入ることを確かめる
(`started_at` から `ready_at` までが、起動に掛かった時間である)。

⚠ 了承を得てから、Spark の worker で、自分のコンテナを異常終了させて、見張りが自動で
戻すことを確かめる (`serve stop` ではなく、Spark 上で直接コンテナを止める。`docker rm` まで
すると、見張りは意図した停止として扱い、起こし直さない)。自動起動の指定を止めたうえで
`serve stop` を打つ手順は、下の「指定がある間の停止」を見る。

```sh
docker stop vb-<構成>-worker
```

しばらく待ってから、`serve autostart status` を読み直し、`container_state` が `running` に
戻り、`consecutive_start_failures` が `0` のままであることを確かめる。head でも同じ確認を
行える (`docker stop vb-<構成>-head`)。戻るまでの時間は、新しい `started_at`/`ready_at` の
差で読む。head を止めた試験では、worker 側も head の `/health` が `UNHEALTHY_AFTER_S`
(180 秒) 連続で 200 でなくなったことを検知し、約 3 分後に worker 自身を起こし直す
(multi-node TP は 2 台そろって起こし直す必要があるため。README D3 参照)。

`glm53-tp2-mtp3-marlin` は `--load-format instanttensor` を使うので、見張りは `docker run` の
直前に、root 不要の固定コマンドでページキャッシュを捨ててから `MemFree` を確かめる
(下限 8 GiB)。下限に足りなければ、起こさずに `autostart.status.json` の `state` が `refused`
になり、`reason` に `MemFree` の値が残る。`spark-drop-caches` を見張りから自動で呼ぶことは
ない (root の操作を自動で流さない)。足りないときの対処 (⚠ 了承を得てから、両台で
`sudo /usr/local/sbin/spark-drop-caches` を流す) は
[`ops/spark-drop-caches/README.md`](../../ops/spark-drop-caches/README.md) の手順による
(対話側)。

## 指定がある間の停止

自動起動の指定 (`serve autostart set`) が有効なまま `serve stop` を打つと、その
`docker stop`/`docker rm` の途中の周を見張りが観測し、止めている最中のコンテナを自動で
起こし直すことがある (見張りの判定だけでは、`serve stop` の `docker stop` と、上の短い
実機試験で使う `docker stop` を見分けられない。worker の待ち時間に頼る方法も、未確認の
停止時間に依存するので採らない)。推論サーバーを止めるときは、必ず指定を解除してから
`serve stop` を打つこと。

⚠ 了承を得てから、Mac から自動起動の指定を解除する。

```bash
uv run serve autostart clear --yes
```

⚠ 了承を得てから、両台の見張りがこれを読んだこと (両台とも `config` が空、`state` が
`idle`) を確かめてから、`serve stop` を打つ。

```bash
uv run serve autostart status
```

`autostart.<役割>.status.config` が両台とも空になったら `serve stop` を打ってよい。指定を
解除したあとは、見張りは `docker` に一切触れない (`config: null` を読んだ周から先、
`docker ps` すら呼ばない)。そのため、この順で行えば、`serve stop` のどの時点で見張りの
周が入っても、見張りが `docker run`/`stop`/`rm` を流すことはない。

## 外し方

⚠ 了承を得てから、Mac から自動起動の指定を解除する (`config: null` を両台に配るだけ)。

```bash
uv run serve autostart clear --yes
```

⚠ 了承を得てから、両台で見張りの unit を無効化し、取り除く。

```sh
~/vllm-baseline/payload/vllm-autostart/vllm-autostart-uninstall
```

⚠ 了承を得てから、linger をもとに戻す (設置の逆。sudoers の固定形のもう一方)。

```sh
sudo loginctl disable-linger j5ik2o
```

## 実機で確かめること (見張りの実装ガイドラインに列挙した、未確認の経路)

- `/usr/bin/python3` の実在、`docker ps --format json` の実物の形
- head が落ちたときの vLLM の engine 死亡経路の終了コード
- 形成済みの分散グループへ、新しい rank 1 が再参加したときの挙動
- linger を有効にしたあと、再起動から `docker` が使えるようになるまでの待ち時間
- `/usr/bin/loginctl` の実際の道筋 (sudoers の固定形と一致するか)

これらは、対話側が実機で確認する。
