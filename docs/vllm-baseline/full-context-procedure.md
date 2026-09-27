# 文脈長 163840・同時実行 16 の TP=2 構成での prefill・並列・長文脈の計測手順

この手順は、[自前イメージで TP=2 の初回起動を確認する](patched-tp2-procedure.md) と [起動を確認したあとに流す、最初の計測の手順](initial-benchmark-procedure.md) の後に行う、**別の試行**です。対象は `p2-nope-tp2-full`（文脈長 163840、同時実行 16。`serving/config/configs.toml` の `p1-nvfp4-tp2` の値のまま）で、prefill・並列（concurrency）・長文脈（quality の needle）を測ります。文脈長 163840 が実際に起動できるか、KV 容量が足りるかは、この手順の中で初めて確かめるものであり、本書はその結果を保証しません。この文書は手順であって記録ではないので、起動や計測が成功したとは書きません。

この文書や公開記録に書かないこと: 要求・応答の本文、認証情報、`exl3-tp2` の中身。

実行位置: 以降のコマンドはすべて、このリポジトリ直下 (`DGX-Spark-GLM5.3-Flash-Recipe/`) を作業ディレクトリとして実行します。`cd` は使わず、`uv run --directory serving serve …` / `uv run --directory bench bench …` の形で呼びます。

## 1. 構成の生成と読み取り検査

```bash
uv run --directory serving python ../experiments/nope-mla/configure_tp2.py --variant full \
  ../serving/var/nope-build-0961bbae/image-inspect.json \
  ../serving/var/nope-build-0961bbae/tp2-full.toml
uv run --directory serving serve check p2-nope-tp2-full \
  --configs var/nope-build-0961bbae/tp2-full.toml
```

生成はネットワークや `subprocess` を使わず、既存出力があれば上書きしません。`--max-model-len` と `--max-num-seqs` の値と根拠は `p1-nvfp4-tp2` のままです。`full` と `full-mtp` の args の末尾には、既定で `--load-format instanttensor`（#50。根拠は vLLM の `LoadConfig` の docstring）が入ります。`serve check` は `patched-tp2-procedure.md` §6 と同じ 9 関門です。**前の試行で通ったことを根拠に省かず、全関門をもう一度検査します。**

#17 の確認（張り付くスレッドが NCCL の proxy か）のために、`--nccl-thread-names` を付けた構成も生成できます。これはどの `--variant` にも付けられ、付けたときだけ env に `NCCL_SET_THREAD_NAME=1`（NCCL の公式文書を `source` と `quote` に持つ）が入り、構成の名前が `-tn` 付きになります。付けないときの出力は変わりません。

```bash
uv run --directory serving python ../experiments/nope-mla/configure_tp2.py --variant full --nccl-thread-names \
  ../serving/var/nope-build-0961bbae/image-inspect.json \
  ../serving/var/nope-build-0961bbae/tp2-full-tn.toml
uv run --directory serving serve check p2-nope-tp2-full-tn \
  --configs var/nope-build-0961bbae/tp2-full-tn.toml
```

`p2-nope-tp2-full` との違いは、env の `NCCL_SET_THREAD_NAME=1`（根拠つき）と、構成の名前だけです。`--variant full-mtp --spec-tokens 1 --nccl-thread-names` なら `p2-nope-tp2-mtp1-tn` になります。以降の §3・§7 のコマンドは、構成名を `p2-nope-tp2-full-tn`、`--configs` を `tp2-full-tn.toml` に読み替えます（`serve check` の 9 関門は同じです）。

#37 の調査で、起動の約 7 分を占める重みの読み込みは、ディスクではなく vLLM の既定の読み方（mmap で必要なところを少しずつ読む）が律速と分かりました。その後の #37 の実測（2026-09-26）で、`--load-format instanttensor` にすると重みの読み込みが 363 秒（head）から 32 秒になり、起動全体が約 11 分から約 3 分になりました。同じ構成の `probe` の decode の tok/s・toolcall 5/5・needle 8k は、既定の読み方と同じ範囲で、壊れの印は 0 件でした。これを受けて #50 で、`full` と `full-mtp` の args には `--load-format instanttensor`（根拠は vLLM の `LoadConfig` の docstring）が既定で入ります。既定を外すには `--load-format auto` を付けます。出力は #50 より前の既定とバイト単位で同じになり、構成の名前は `p2-nope-tp2-full` のままです。

`--load-format`（`instanttensor` / `fastsafetensors` / `runai_streamer`）の 3 値を明示すると、既定を置き換えて args に根拠（`source` と `quote`）つきで入り、構成の名前が `-lf-<値>` 付きになります（`runai_streamer` は `-lf-runai-streamer`）。`--safetensors-load-strategy`（`eager` / `prefetch`）は、構成の名前が `-sls-<値>` 付きになります。ただし strategy が効くのは `--load-format` を付けない読み方だけです。既定で `--load-format instanttensor` を持つ `full` と `full-mtp` で strategy が効く構成は、`--load-format auto` と組にしたものだけで、`--load-format` を付けずに指定すると、生成の前に断ります（明示の `--load-format` と重ねた構成は作れますが、strategy は効きません。理由は §1 の最後の段落）。既定を持たない `smoke` には、単独で付けられます。どちらも `--nccl-thread-names`・`--torch-profiler` と併用できます。

```bash
uv run --directory serving python ../experiments/nope-mla/configure_tp2.py --variant full --load-format auto \
  ../serving/var/nope-build-0961bbae/image-inspect.json \
  ../serving/var/nope-build-0961bbae/tp2-full-auto.toml
uv run --directory serving serve check p2-nope-tp2-full \
  --configs var/nope-build-0961bbae/tp2-full-auto.toml
```

上の比較用の構成は、既定を外した `--load-format auto` で作ります。構成の名前は既定の `p2-nope-tp2-full` と同じなので、`--configs` のファイル名（`tp2-full.toml` と `tp2-full-auto.toml`）と `config-sha256` のラベルで区別します。同時に 2 つは起動しません。

読み込み方を比べるときは、1 構成ずつ §3 の `serve start` → `serve logs`、§7 の停止を行い、回収した起動の記録の `Loading weights took` と `Model loading took` の行を、既定（`instanttensor`）の `p2-nope-tp2-full` と `--load-format auto` で作った構成で比べます。ただし、固定した vLLM では `Loading weights took` の行を出すのは `default_loader.py` と `sharded_state_loader.py` だけで、`--load-format runai_streamer` の構成（`RunaiModelStreamerLoader`）はこの行を出しません。その構成は `Model loading took` の行で比べます。

2 回目以降の起動は、重みがページキャッシュに乗っているぶん速く読めるので、比べる前に両台のページキャッシュの状態をそろえます。ページキャッシュを捨てる操作（両台で `drop_caches` へ書く）は実機の状態を変えるので、⚠ **了承を得てから**行います。この操作には root 権限が要るので、`ops/spark-drop-caches/README.md` の手順で設置した固定のコマンドを使います（issue #84）。

```sh
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d 'sudo /usr/local/sbin/spark-drop-caches'
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-5083 'sudo /usr/local/sbin/spark-drop-caches'
```

`--safetensors-load-strategy` が効くのは、`--load-format` を付けない既定の読み方だけです（固定した vLLM で strategy を使うのは `default_loader.py` の `safetensors_weights_iterator` だけで、`instanttensor`・`fastsafetensors`・`runai_streamer` の読み方は strategy を受け取りません）。そのため `full` と `full-mtp` では、`--safetensors-load-strategy` を `--load-format auto` と組で使い（例: `--load-format auto --safetensors-load-strategy prefetch` → `p2-nope-tp2-full-sls-prefetch`）、`auto` の構成と比べます。`--load-format` を付けずに `--safetensors-load-strategy` だけを指定すると、既定の `--load-format instanttensor` が残り、strategy が効かない構成（名前は `-sls-<値>` なのに、中身は `--load-format instanttensor` つき）になってしまうので、生成器は生成の前に断ります（既定を黙って外すこともしません）。明示の `--load-format` と重ねた構成も作れますが、strategy は効かないので、比べる対象にしません。`smoke` は既定を持たないので、`--safetensors-load-strategy` だけで作れます。

## 2. GPU と X925 の上限の確認

GPU クロックの上限（`nvidia-smi -lgc 300,1800`）と X925 の周波数の上限（`cpupower -c 5-9,15-19 frequency-set -u 3000MHz`）は、どちらも再起動で既定に戻ります。issue #10 で、両台の起動のたびに入れ直す systemd の oneshot サービス（`spark-power-caps.service`）を用意しました。計測の前に、このサービスが両台で有効・実行中であることを、読み取りだけで確かめます。

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d 'systemctl is-enabled spark-power-caps.service; systemctl is-active spark-power-caps.service'
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-5083 'systemctl is-enabled spark-power-caps.service; systemctl is-active spark-power-caps.service'
```

両台とも `enabled` と `active` であれば進みます。`inactive`／`failed`／unit がないのいずれかなら、⚠ 計測者の了承を得たうえで、`ops/spark-power-caps/README.md` の手順（初回の設置、または値の変更）で入れます（上限がないまま計測してしまうおそれがあるため、了承なしには進みません）。

GB10 の `nvidia-smi` には GPU クロックの上限の設定状態が出ません。計測が終わったあと（§7 のログ回収より前）に、`serve watch` の記録（§5）から、`result.json` の `gpu_sm_clock_range_mhz`（最大値が 1800 以下。[記録](../results/2026-09-23-decode-gpu-clock-cap.md) の実測値は 1768〜1781 MHz）と `cpu_cluster_max_freq_khz.x925`（3000000 以下）で、計測中に両方の上限が効いていたことを確かめます。この 2 つの値を、今回の計測の条件として記録します。

## 3. 起動、今回分の IB、短い応答

⚠ 起動による GPU プロセス、コンテナ、ログの状態変更を了承した後に進みます。

```bash
uv run --directory serving serve start p2-nope-tp2-full \
  --configs var/nope-build-0961bbae/tp2-full.toml --timeout 3h --yes
```

進む条件は `status=ready` だけです。`already_running`／`refused`／`failed` の扱いは `patched-tp2-procedure.md` §6 と同じです。関門 `memory_free` が `MemFree` の不足を理由に断ることもあります (issue #84)。断られたら、⚠ 了承を得てから両台で `sudo /usr/local/sbin/spark-drop-caches` を実行してページキャッシュを捨て (`ops/spark-drop-caches/README.md`)、`serve start` をやり直します。続けてログを回収します。

```bash
uv run --directory serving serve logs p2-nope-tp2-full \
  --configs var/nope-build-0961bbae/tp2-full.toml
```

NCCL ログの PID が過去の試行と同じで、ファイルが上書きされることがあります。`patched-tp2-procedure.md` §7 と同じ手順で、**hostname・コンテナの開始時刻・NCCL の初期化回数**から今回分を一意に特定し、経路確定行がすべて IB であることを確かめます。特定できなければ、計測へ進まず記録して §7 へ進みます。

```bash
uv run --directory serving serve smoke p2-nope-tp2-full \
  --configs var/nope-build-0961bbae/tp2-full.toml --max-tokens 512
```

進む条件は `status=answered`、HTTP 200、意味の通る応答です (`patched-tp2-procedure.md` §7 と同じ判定)。

## 4. KV 容量の記録と進む条件

回収した起動ログから `GPU KV cache size` と `Maximum concurrency` の行を、今回の計測の条件として記録に写します。smoke 構成の観測値（221,184 トークン、16.61〜16.79 GiB、`max_num_batched_tokens` 2048、`gpu_memory_utilization` 0.9）は参考値で、この構成 (163840、同時実行 16) の値を保証しません。

`GPU KV cache size` が 163,840 トークン未満、または `Maximum concurrency` が 163,840 トークンの要求 1 本に対して 1 未満であれば、**計測へ進まず、記録して §7 のログ回収・停止へ進みます。**

## 5. 計測 (suite ごとに別ラン)

両台の GPU の温度・SM クロック・電力・使用率、ACPI 熱区域、hwmon（`mlx5`/`nvme`/`acpitz`）の温度、コアごとの CPU 使用率を、`serve watch` で**読み取りだけで**記録します。次の順で進めます。

1. 別の端末で、`serve watch` を先に起動します。

   ```bash
   uv run --directory serving serve watch p2-nope-tp2-full \
     --configs var/nope-build-0961bbae/tp2-full.toml --interval 10s --duration 2h
   ```

2. `serving/var/<起動した時刻>-watch-p2-nope-tp2-full/samples.jsonl` に 1 行目が書かれたことを確かめます。熱区域と hwmon の一覧を発見してから最初の観察をするため（`serving/README.md` の watch の節）、起動直後は少し待ちます。
3. 1 行目を確かめてから、suite ごとの `bench run` を始めます。suite が終わったら、`serve watch` を Ctrl-C で止めます（中断でも、そこまでの要約は `result.json` に書かれます）。

見るのは `result.json` の `thermal_zone_max_c`、`gpu_temperature_max_c`、`gpu_sm_clock_range_mhz`（上限 1800 が効いているか）、`gpu_power_max_w`、`hwmon_temp_max_c`、`cpu_cluster_max_freq_khz`（X925 の上限 3000 が効いているか。§2）、`thermal_over_threshold_samples`／`thermal_over_threshold_span`、`events` の `finding = thermal` と `detail`（台に届かず発見を打ち切った場合は、ここに記録されます）です。**ACPI 熱区域が 90℃ に達した区間は、基準値として扱いません。** `thermal` の出来事は終了コード 2 にもなりますが、計測そのものは止まりません。止めるかどうかは、要約と出来事を見て計測者が決めます。

suite ごとに、別々の計測ランとして、prefill → concurrency → quality（needle）の順で流します。

```bash
uv run --directory bench bench run --target p2-nope-tp2-full --suite prefill --profile quick
uv run --directory bench bench run --target p2-nope-tp2-full --suite concurrency --profile quick
uv run --directory bench bench run --target p2-nope-tp2-full --suite quality --profile quick
```

- `prefill` は狙いの入力 8k/32k/128k の 6 条件です。163840 なら 128000+16 も上限を超えず、送る前に飛びません。
- `concurrency` は `levels = [1, 2, 4, 8]`（`bench/config/profiles.toml`）です。**この結果は、対象サーバー側の `--max-num-seqs 16`（同時実行 16）の構成で測ったものであり、smoke 構成（同時実行 1、文脈長 4096）の decode と混同しません。** 2 つの計測ランを比べるときは `--set concurrency.rounds=20` を使えます（下限は正の整数。`bench/README.md`）。
- `quality` は **`needle` だけを選ぶ引数がなく**、`toolcall`・`code`・`needle` を必ず計画します (`suites/quality.py`)。needle の狙いの入力 8k/32k/128k は 163840 を超えないので、送る前に飛びません。`code` にはコードの隔離の実行環境が要ります。用意できなければ、その条件だけ理由つきで飛びます。

**長い会話の検査 (`agent`) は必須のランに含めません。** CLI で選べる単位は suite で、`--suite agent` と `agent.end_tokens`（下限は `start_tokens` 以上）で段階を絞れます。`quick` でも 300 要求・入力約 2,090 万トークンで、プレフィックスキャッシュが効かないと計算上 2.9〜5.8 時間かかり (`bench/README.md`)、この対象でのキャッシュの効きと prefill 速度は未計測です。120000+256 は 163840 を超えず、長文脈の品質は needle が覆うため、prefill・concurrency・quality の 3 ランが `completed` になり、prefill の速度から所要時間を見積もれた後に、任意の追加ランとして 1 段階から始めます。

```bash
uv run --directory bench bench run --target p2-nope-tp2-full --suite agent \
  --profile quick --set agent.end_tokens=20000
```

### 5.1 スレッドの名前の読み方（`p2-nope-tp2-full-tn` のとき、#17）

#17 で推定した「`Worker_TP` の中のメイン以外の 100% 近いスレッドは NCCL の proxy」を確かめるため、推論中にスレッドの名前を読みます。

bench の対象は足しません。`-tn` の構成は宛先・モデル名・文脈長が `p2-nope-tp2-full` と同じなので、既存の対象名で負荷を掛けます。スレッドの名前を読むための負荷なので、数分で終わる短い decode にします（`quick` の既定のままだと 35 分以上かかります）。

```bash
uv run --directory bench bench run --target p2-nope-tp2-full --suite decode --profile quick \
  --set decode.trials=2 --set decode.warmup_trials=1 --set decode.max_tokens=256
```

要約には対象名 `p2-nope-tp2-full` が残るので、記録には構成名 `p2-nope-tp2-full-tn` で測ったことを書きます。

推論中に、両台で `top -H` を**読み取りだけで**実行します。

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d 'top -H -b -n 1 | head -40'
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-5083 'top -H -b -n 1 | head -40'
```

`COMMAND` 欄のスレッドの名前と `%CPU` を記録します。期待は、`Worker_TP` の中のメイン以外の 100% 近いスレッドが、`NCCL Progress` に続く番号の名前（NCCL の `src/proxy.cc` の `"NCCL Progress%2d"`）で見えることです。別の名前なら、その名前を記録し、proxy と結論しません。名前が付かないときは、NCCL の版（2.12 以上が要る）を記録します。

## 6. 合否の見方

終了の値 0 は「計測ランが完了した」という意味で、**性能や品質の合否ではありません。** 次のいずれかがあれば、その条件を成功として扱いません。

- 試行の数が足りない印 (`insufficient_trials`。`quick` の `min_successes` は 5)
- 対象の条件が送る前に飛んだ (`skipped`)
- 要求の失敗：prefill・concurrency の速さの表は「失敗」欄が 1 以上、または印「一部が失敗」(`partial_failures`)。quality の表は「要求の失敗」欄が 1 以上
- 採点不能：quality の表の「採点できなかった」欄 (`not_scored`) が 1 以上
- 計測ランの状態が `completed` でない

`summary.md` の該当欄を見て判定します。

## 7. ログ回収・所有確認・停止

`patched-tp2-procedure.md` §8 と同じ手順で、自分のラベル (`vllm-baseline.owner=serving-kit`) のコンテナだけが対象であることを照合してから、ログを回収し、停止します。

```bash
uv run --directory serving serve logs p2-nope-tp2-full \
  --configs var/nope-build-0961bbae/tp2-full.toml
uv run --directory serving serve status --configs var/nope-build-0961bbae/tp2-full.toml
uv run --directory serving serve stop --yes
uv run --directory serving serve status --configs var/nope-build-0961bbae/tp2-full.toml
```

停止後に GPU プロセスが 0 件であることを確認します。`bench` の生データ (要求・応答の本文を含む) は Git 対象外の `results/` に残り、この文書や `docs/results/` には要求・応答の本文を載せません。公開するのは `bench publish` が写す `summary.json`／`summary.md` の 2 つだけです。

## 8. 実機でしか確かめられないこと

- `GPU KV cache size`、163,840 トークンの要求 1 本が入るか
- `--max-num-seqs 16` での起動可否
- クロック上限 1800 での prefill・並列の性能
- この対象でのプレフィックスキャッシュの効き方
- `NCCL_SET_THREAD_NAME=1` で実際にスレッドに名前が付くか（`NCCL Progress`。NCCL の版 2.12 以上が要る）
