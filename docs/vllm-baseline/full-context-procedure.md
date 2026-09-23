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

生成はネットワークや `subprocess` を使わず、既存出力があれば上書きしません。`--max-model-len` と `--max-num-seqs` の値と根拠は `p1-nvfp4-tp2` のままです。`serve check` は `patched-tp2-procedure.md` §6 と同じ 8 関門です。**前の試行で通ったことを根拠に省かず、全関門をもう一度検査します。**

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

進む条件は `status=ready` だけです。`already_running`／`refused`／`failed` の扱いは `patched-tp2-procedure.md` §6 と同じです。続けてログを回収します。

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
