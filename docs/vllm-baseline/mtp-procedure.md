# モデル付属の MTP を下書きのトークン数ごとに計測する手順 (`p2-nope-tp2-mtpN`)

この手順は、[文脈長 163840・同時実行 16 の TP=2 構成での prefill・並列・長文脈の計測手順](full-context-procedure.md) の後に行う、**別の試行**です。対象は `p2-nope-tp2-mtp1` / `p2-nope-tp2-mtp2` / `p2-nope-tp2-mtp3` / `p2-nope-tp2-mtp5`（`p2-nope-tp2-full` と同じ文脈長 163840・同時実行 16 に、モデル付属の MTP を下書き N トークンで有効にした構成）です。MTP の重みが読まれ、投機的デコードが働くか、生成が壊れないかは、この手順の中で初めて確かめるものであり、本書はその結果を保証しません。この文書は手順であって記録ではないので、起動や計測が成功したとは書きません。

この文書や公開記録に書かないこと: 送った内容と応答の本文、認証の情報、`exl3-tp2` の中身。要求・応答の本文は `results/`（Git 対象外）にだけ残り、この文書や `docs/results/` には載せません。認証の情報は、値そのものを書かず、要る場合は環境変数の名前だけを書きます。`exl3-tp2` の中身（計測者が別に起動していた構成）は調べません。

実行位置: 以降のコマンドはすべて、このリポジトリ直下 (`DGX-Spark-GLM5.3-Flash-Recipe/`) を作業ディレクトリとして実行します。`cd` は使わず、`uv run --directory serving serve …` / `uv run --directory bench bench …` のように呼びます。

## 0. 前提: この版の MTP 対応

この手順が使うイメージは、固定した vLLM のコミット `0961bbae` に NoPE の修正を加えたものです。この版のソースを読んだ結果は、次のとおりです。

- `0961bbae` は、GLM-5.3-Flash (glm5next) のモデル付属の MTP に対応しています。正式な引数は `--speculative-config` の JSON の `method` と `num_speculative_tokens`（または `--spec-method` / `--spec-model` / `--spec-tokens`）です。`config.speculative_config` の `hf_config_override` が `glm5_next` を `Glm5NextMTPModel` に写し、`num_nextn_predict_layers = 1` の重みでは `num_speculative_tokens` は 1 の倍数であれば通ります。
- **#58454 は未修正です。** `num_speculative_tokens >= 2` の投機で、文脈が `index_topk` (2048) を超えると、kpool の committed keys が壊れるという報告 (PR、Open) があります。N = 1 はこの条件に当たりません。N が 2 以上のときは §7 の確認を必ず行います。
- **#57087 は未解決です。** GB10/sm121 の TP=2 の同時処理で、非 ASCII の出力が壊れる (U+FFFD が混じる) という issue (Open、修正 PR なし) があります。MTP とは無関係に存在するので、同時実行では `concurrency/c{n}` の置き換え文字の件数を確認します（§6）。
- **#55442 はこの版に含まれません。** ModelOpt の NVFP4 の重みで MTP を有効にすると、placeholder の `shared_head.head` が量子化されて落ちる、という不具合の修正です。ただし、この手順が使う RedHatAI の重みは `quant_method: "compressed-tensors"` で、量子化の対象は experts だけなので、#55442 の経路には当たりません（これはソース上の判断であり、実機では未確認です）。
- 関連する設計の判断は、ADR 0006 の `docs/decisions/0006-path-pruning.md` にあります（このブランチには未マージのため、ここではリンクにしません）。調査の報告は `docs/research/2026-09-25-path-survey.md` です。

**イメージの作り直しは不要と判断しました。** N が 2 以上で文脈が 2048 を超える条件（§7）の結果だけは、実機で確かめるまで分かりません。

## 1. 構成の生成と読み取り検査

N ごとに、生成した構成を作ります。出力の名前は `tp2-mtpN.toml` にそろえます。

```bash
uv run --directory serving python ../experiments/nope-mla/configure_tp2.py \
  --variant full-mtp --spec-tokens 1 \
  ../serving/var/nope-build-0961bbae/image-inspect.json \
  ../serving/var/nope-build-0961bbae/tp2-mtp1.toml
uv run --directory serving serve check p2-nope-tp2-mtp1 \
  --configs var/nope-build-0961bbae/tp2-mtp1.toml
```

`--spec-tokens` は 1 以上の整数だけを受け、`--variant full-mtp` のときだけ使えます。生成した構成は、`p2-nope-tp2-full` と同じ構成に、最上位の `allow_speculative = true` と、`args` の `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`（根拠つき）を足したものです。`p2-nope-tp2-full` と同じく、args の末尾には既定の `--load-format instanttensor`（#50。根拠つき）が入ります。#37 の実測では、これで起動が約 11 分から約 3 分になりました。既定を外すには `--load-format auto` を付けます（出力は #50 より前と同じ、名前は変わりません）。`serve check` は `patched-tp2-procedure.md` §6 と同じ 9 関門です。**前の N で通ったことを根拠に省かず、N ごとに全関門をもう一度検査します。**

N = 2、3、5 も同じ形で作り、それぞれ `p2-nope-tp2-mtp2` / `mtp3` / `mtp5` を `serve check` します。

## 2. GPU と X925 の上限の確認

GPU クロックの上限（`nvidia-smi -lgc 300,1800`）と X925 の周波数の上限（`cpupower -c 5-9,15-19 frequency-set -u 3000MHz`）は、どちらも再起動で既定に戻ります。`spark-power-caps.service` が両台で有効・実行中であることを、読み取りだけで確かめます（[full-context-procedure.md](full-context-procedure.md) §2 と同じ）。

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d 'systemctl is-enabled spark-power-caps.service; systemctl is-active spark-power-caps.service'
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-5083 'systemctl is-enabled spark-power-caps.service; systemctl is-active spark-power-caps.service'
```

両台とも `enabled` と `active` であれば進みます。そうでなければ、⚠ 計測者の了承を得たうえで、[spark-power-caps の手順](../../ops/spark-power-caps/README.md)で入れます。計測が終わったあと（§9 のログ回収より前）に、`serve watch` の記録（§5）から、`result.json` の `gpu_sm_clock_range_mhz`（最大値が **1800** 以下）と `cpu_cluster_max_freq_khz.x925`（**3000000** 以下。[記録](../results/2026-09-23-decode-gpu-clock-cap.md)の実測値は 1768〜1781 MHz）で、計測中に両方の上限が効いていたことを確かめます。この 2 つの値を、今回の計測の条件として記録します。

## 3. 起動、今回分の IB、短い応答

⚠ 起動による GPU プロセス、コンテナ、ログの状態変更を了承した後に進みます。N ごとに、起動・記録・停止を 1 つのまとまりとして行います。

```bash
uv run --directory serving serve start p2-nope-tp2-mtp1 \
  --configs var/nope-build-0961bbae/tp2-mtp1.toml --timeout 3h --yes
```

進む条件は `status=ready` だけです。投機を許す構成では、起動の検査 (design 6.7) が `/metrics` に `vllm:spec_decode_` の行が出ていることを要求します。`already_running`／`refused`／`failed` の扱いは `patched-tp2-procedure.md` §6 と同じです。続けて、起動の記録から `SpeculativeConfig(` の行と `Model loading took` の行を回収し、今回の条件として記録します。

```bash
uv run --directory serving serve logs p2-nope-tp2-mtp1 \
  --configs var/nope-build-0961bbae/tp2-mtp1.toml
```

NCCL ログの今回分の特定は `patched-tp2-procedure.md` §7 と同じです（hostname・コンテナの開始時刻・NCCL の初期化回数）。経路確定行がすべて IB であることを確かめます。特定できなければ、計測へ進まず記録して §9 へ進みます。

```bash
uv run --directory serving serve smoke p2-nope-tp2-mtp1 \
  --configs var/nope-build-0961bbae/tp2-mtp1.toml --max-tokens 512
```

進む条件は `status=answered`、HTTP 200、意味の通る応答です。

## 4. KV 容量の記録と進む条件

回収した起動ログから `GPU KV cache size` と `Maximum concurrency` の行を、今回の計測の条件として記録に写します。`p2-nope-tp2-full` の実測値は 1,708,119 トークンです（[記録](../results/2026-09-23-full-context.md)を参考値として引用し、MTP のぶんだけ減ることを見込みます）。`GPU KV cache size` が 163,840 トークン未満、または `Maximum concurrency` が 163,840 トークンの要求 1 本に対して 1 未満であれば、**計測へ進まず、記録して §9 のログ回収・停止へ進みます。**

## 5. 計測 (N ごと、suite ごとに別ラン)

両台の GPU の温度・SM クロック・電力・使用率などを `serve watch` で**読み取りだけで**記録します（[full-context-procedure.md](full-context-procedure.md) §5 と同じ）。

1. 別の端末で、`serve watch` を先に起動します。

   ```bash
   uv run --directory serving serve watch p2-nope-tp2-mtp1 \
     --configs var/nope-build-0961bbae/tp2-mtp1.toml --interval 10s --duration 2h
   ```

2. `serving/var/<起動した時刻>-watch-p2-nope-tp2-mtp1/samples.jsonl` に 1 行目が書かれたことを確かめます。
3. 1 行目を確かめてから、suite ごとの `bench run` を始めます。suite が終わったら、`serve watch` を Ctrl-C で止めます。

**比較の基準（`full`、投機なし）を、N ごとの計測の前に測ります。** `p2-nope-tp2-full` に対して、同じ日・両台の上限（§2）が効いた状態で、`decode`（既定と `--set decode.max_tokens=4096`）と `concurrency` を流します。起動と停止は [full-context-procedure.md](full-context-procedure.md) §3・§7 と同じで、構成は `configure_tp2.py --variant full` が生成する `tp2-full.toml` を使います。

```bash
uv run --directory bench bench run --target p2-nope-tp2-full --suite decode --profile quick
uv run --directory bench bench run --target p2-nope-tp2-full --suite decode --profile quick \
  --set decode.max_tokens=4096
uv run --directory bench bench run --target p2-nope-tp2-full --suite concurrency --profile quick
```

`docs/results/` にある `p2-nope-tp2-full` の記録は prefill・concurrency・quality だけで、`decode` は無いので、この 3 つの値は今回測ります。結果を、下の表の N = 0 の行に写します。

N ごとに、`decode` と `concurrency` を別々のランで流します。

```bash
uv run --directory bench bench run --target p2-nope-tp2-mtp1 --suite decode --profile quick
uv run --directory bench bench run --target p2-nope-tp2-mtp1 --suite decode --profile quick \
  --set decode.max_tokens=4096
uv run --directory bench bench run --target p2-nope-tp2-mtp1 --suite concurrency --profile quick
```

**比べる値**:

- `summary.md` の `decode/code/en` と `decode/prose/ja` の `decode_tps`（tok/s）
- `metrics/deltas.jsonl` の `mean_acceptance_length`（受理長）、`spec_acceptance_rate`（投機の当たり率）、`decode_steps`、`tokens_per_step`

**1 ステップの時間**は、`decode/*`（1 本ずつ流す条件）では、2 つの定義のどちらでも出せます。同じ結果になることを確かめてから使います。

- 「試行の生成時間の合計 ÷ `decode_steps`」。合計する範囲は、`deltas.jsonl` の増分と同じ区間です。その条件のすべての試行（ウォームアップを含む）を足します
- `tokens_per_step ÷ decode_tps`

`tokens_per_step` は、1 ステップあたりの**生成**トークン数（`generation_tokens` の増分 ÷ `iteration_tokens_count` の増分）です。prefill の回の入力トークンは入りません。`decode/*` では、投機なしで 1.0 になり、MTP では受理長（`mean_acceptance_length`）とほぼ一致します。`concurrency/c{n}` では、同時に流す n 本の生成が同じステップにまとめて数えられるので、値は本数に応じて大きくなり、1.0 や受理長とは一致しません。上の 2 つの定義は、`concurrency/c{n}` の値には使いません。

N ごとの表の雛形です。`full`（投機なし。`p2-nope-tp2-full`）の値を並べて比べます。

| 対象 | N | `decode/code/en` の tok/s | `decode/prose/ja` の tok/s | 受理長 | 投機の当たり率 | 1 ステップの時間 |
|---|---|---|---|---|---|---|
| `p2-nope-tp2-full` | 0 |  |  |  |  |  |
| `p2-nope-tp2-mtp1` | 1 |  |  |  |  |  |
| `p2-nope-tp2-mtp2` | 2 |  |  |  |  |  |
| `p2-nope-tp2-mtp3` | 3 |  |  |  |  |  |
| `p2-nope-tp2-mtp5` | 5 |  |  |  |  |  |

## 6. 出力が壊れていないかの確かめ方

速さだけでなく、出力の壊れを確かめます。

- `summary.md` の **`repetition_loop`** の件数（同じ文の繰り返し）と、**`replacement_char`** の件数（U+FFFD の混入）
- `bench run --target p2-nope-tp2-mtp1 --suite quality --profile quick` の **`quality/toolcall`**。`quality` は `toolcall`・`code`・`needle` を必ず計画します。`code` はコードの隔離の実行環境が要り、用意できなければその条件だけ理由つきで飛びます

`concurrency` の文章は英語 (`_DOCUMENT_LANG = "en"`) で、出力まで日本語になる保証はありません。それでも同時処理での非 ASCII の壊れ ([#57087](https://github.com/vllm-project/vllm/issues/57087)) は、同時の本数が多いほど出やすいので、N ごとに `concurrency/c{n}` の `replacement_char` の件数を、N = 1 のときと比べます。出力が非 ASCII になるかどうかは、実機で確かめるまで分かりません。`decode/prose/ja` は 1 本ずつ順に流す条件なので、同時処理の壊れの観測には使いません。

## 7. 文脈が 2048 を超える条件の確認 (#58454)

**N が 2 以上の対象では、この確認を必ず行います。** 文脈が `index_topk` (**2048**) を超えると、kpool の committed keys が壊れるという報告 (PR #58454、未修正) があるためです ([#58454](https://github.com/vllm-project/vllm/pull/58454))。

- 入力が 8k 以上の条件（`prefill` の 8k か、`quality` の needle 8k/32k）を流します。
- 出力が 2048 を超える生成を流します。`decode` の既定の `max_tokens` では届かないので、`--set decode.max_tokens=4096` を使います（§5 の 2 つ目のコマンド）。
- `repetition_loop`、`replacement_char`、needle の正解率を、N = 1 のときと比べます。

**壊れの疑いがあれば、その N は記録して止めます。** 以降の N の計測には進まず、起きたことを記録して、イメージの作り直し（上流の修正の取り込み）を判断します。

## 8. 合否の見方

終了の値 0 は「計測ランが完了した」という意味で、**性能や品質の合否ではありません。** 次のいずれかがあれば、その条件を成功として扱いません（[full-context-procedure.md](full-context-procedure.md) §6 と同じ）。

- 試行の数が足りない印 (`insufficient_trials`)
- 対象の条件が送る前に飛んだ (`skipped`)
- 要求の失敗（速さの表の「失敗」欄が 1 以上、または印「一部が失敗」）
- 採点不能（quality の表の「採点できなかった」欄が 1 以上）
- 計測ランの状態が `completed` でない

## 9. ログ回収・所有確認・停止

N ごとに、自分のラベル (`vllm-baseline.owner=serving-kit`) のコンテナだけが対象であることを照合してから、ログを回収し、停止します。

```bash
uv run --directory serving serve logs p2-nope-tp2-mtp1 \
  --configs var/nope-build-0961bbae/tp2-mtp1.toml
uv run --directory serving serve status --configs var/nope-build-0961bbae/tp2-mtp1.toml
uv run --directory serving serve stop --yes
uv run --directory serving serve status --configs var/nope-build-0961bbae/tp2-mtp1.toml
```

停止後に GPU プロセスが 0 件であることを確認します。`bench` の生データは Git 対象外の `results/` に残り、この文書や `docs/results/` には要求・応答の本文を載せません。公開するのは `bench publish` が写す `summary.json`／`summary.md` だけです。

## 10. 実機でしか確かめられないこと

- MTP の重みが実際に読まれ、投機的デコードが働くか（起動ログの `SpeculativeConfig(` と `/metrics` の `vllm:spec_decode_`）
- MTP のぶんだけ KV 容量がどう減るか
- 下書き N ごとの受理長と、`tok/s` の伸び
- #58454 が N が 2 以上で発現するか（§7）
- `concurrency/c{n}` の #57087 の非 ASCII の壊れ（その出力が非 ASCII になるかを含めて）
