# 起動を確認したあとに流す、最初の計測の手順

この手順は、[自前イメージで TP=2 の初回起動を確認する](patched-tp2-procedure.md) の §6 の `status=ready`、§7 の IB と短い応答、§8 のログ回収・停止までが終わったあとに、**別の試行として**行うものです。同じ文書の §9 が「別の計画として扱う」とした `/v1/messages` 互換の確認にあたります。実重みでの TP=2 の起動、API 互換、IB 経路は、いずれもまだ確認できていません。この文書は手順であって記録ではないので、起動や計測が成功したとは書きません。

この文書や公開記録に書かないこと: 要求・応答の本文、認証情報、`exl3-tp2` の中身。

実行位置: 以降のコマンドはすべて、このリポジトリ直下 (`DGX-Spark-GLM5.3-Flash-Recipe/`) を作業ディレクトリとして実行します。`cd` は使わず、`uv run --directory serving serve …` / `uv run --directory bench bench …` の形で呼びます。

## 1. この試行で確かめること・確かめないこと

通常応答、SSE の完了、ツール呼び出しの解釈を確認します。並列性能、4096 を超える長文脈、品質の合否、長時間運転は次の計画に残します。

起動側は `--max-num-seqs 1` です (`experiments/nope-mla/configure_tp2.py`)。同時に 1 本しか処理しないので、ここで得た値を並列性能の測定結果として扱いません。

## 2. 起動し直す

前の試行は §8 で停止しているので、サーバーは動いていません。`patched-tp2-procedure.md` §6 に従って全 9 関門を**もう一度**検査し、同じ構成で起動します。前の試行で通ったことを根拠に関門を省きません。

```bash
uv run --directory serving serve check p2-nope-tp2-smoke --configs var/nope-build-0961bbae/tp2.toml
uv run --directory serving serve start p2-nope-tp2-smoke \
  --configs var/nope-build-0961bbae/tp2.toml --timeout 3h --yes
```

進む条件は `status=ready` だけです。続けて `patched-tp2-procedure.md` §7 の手順で、**今回のコンテナに帰属する** NCCL の記録だけを選び、`Using network IB` を確認します。ログ回収は過去の試行のファイルも含むため、hostname・PID・コンテナの開始時刻で今回分を一意に特定できなければ、要求を送らずに記録して停止します。

## 3. 非ストリーミング応答と SSE を確かめる

```bash
uv run --directory serving serve smoke p2-nope-tp2-smoke --configs var/nope-build-0961bbae/tp2.toml
uv run --directory bench bench run --target p2-nope-tp2-smoke --suite decode --profile quick
```

- `--suite decode` を**明示します**。省くと速さの 3 つ (`decode` / `prefill` / `concurrency`) が既定で流れます (`bench/src/bench_harness/cli.py` の `DEFAULT_SUITES`)。この試行では `concurrency` を使いません。`prefill` の狙いの入力は 8k / 32k / 128k なので、4096 では 6 条件とも送る前に飛びます。
- `serve smoke` で非ストリーミングの短い応答を確認した後、bench の前提確認が `POST /v1/messages` を 1 件送り、`usage` と出力の中身があることを確かめます。満たされなければ計測ランのディレクトリを作らずに終了の値 1 で止まります。ここが SSE 応答の最初の関門です。
- `bench` が送る要求は**すべて SSE** です (`MessagesRequest.stream` は `Literal[True]`)。非ストリーミングで叩く道はありません。最初のトークンまでの時間と受け取った事象の数が記録され、途中で壊れれば `stream_error` または `protocol` として残ります。
- `decode` は 6 条件 (`code`/`prose`/`json` × `en`/`ja`) で、`quick` では 6 × (10 試行 + 慣らし 2) = 72 要求です。前述の bench 前提確認の 1 要求は含みません。狙いの入力の長さを持たない条件なので、4096 でも飛びません。試行を増減するなら `--set decode.trials=N` ですが、**下限は 10** です (`types.py` の `ge=10`)。下回ると設定の誤りとして終了の値 1 になります。

## 4. ツール呼び出しを確かめる

```bash
uv run --directory bench bench run --target p2-nope-tp2-smoke --suite quality --profile quick
```

- **選べる単位は「まとまり」までで、`toolcall` だけを選ぶ引数はありません。** `--suite quality` は toolcall・code・needle の 3 種を必ず計画します (`suites/quality.py` の `plan`)。
- needle は狙いの入力が 8k / 32k / 128k なので、4096 では 15 条件とも送る前に飛びます。結果として実際に走るのは toolcall と、条件が揃えば code です。これは「toolcall を選んだ」のではなく「ほかが飛んだ」状態で、要約の飛ばした条件の欄に出ます。
- code にはコードの隔離の実行環境と公開の課題が要ります。用意できなければ、その条件には要求を 1 つも送らずに飛びます (ほかの条件は実行します)。課題を減らすなら `--set quality.code_problem_limit=N` で、**下限は 1** です (`types.py` の `ge=1`)。`0` は設定の誤りになるので書きません。
- 対象サーバー側は `--tool-call-parser glm47` と `--enable-auto-tool-choice` を渡した構成で起動している前提です (`serving/config/configs.toml`)。

## 5. 生データと公開の扱い

`bench` は**合成データの要求と応答の本文を残します**。`results/<計測ランの識別子>/` の `trials.jsonl` に応答の本文が、`bodies/<sha256>.json.gz` に送った要求の本文が入ります。`/results/` は `.gitignore` にあり、計測を始める前に道具自身がこの置き場所が git の管理の対象でないことを確かめ、そうでなければ計測ランのディレクトリを作らずに止まります。

これは `serving` 側の「本文を保持しない」条件とは**別の契約**です。同じものとして扱わないでください。公開するのは `bench publish` が写す `summary.json` と `summary.md` の 2 つだけで、この文書や `docs/results/` に要求・応答の本文は載せません。

## 6. 結果の見方

終了の値 0 は「計測ランが完了した」という意味であって、**品質の合否ではありません**。次のいずれかがあれば、成功として扱いません。

- 今回確かめる decode / toolcall が飛ばされた。対象外の needle などが飛ばされるのは想定どおりですが、それらを検査済みとは扱いません。終了の値 0 だけで全条件の成功と判断しないでください
- 「試行の数が足りない」印 (`insufficient_trials`) が出ている (`quick` の `min_successes` は 5)
- 要求そのものが失敗した試行や、採点できなかった試行がある。これらは正解の割合の**分母から外れる**ので、割合だけを見ると見落とします
- 状態が `completed` でない。未完了の計測ランは、要約と比較の先頭にその旨が出ます

## 7. ログを回収して停止する

`patched-tp2-procedure.md` §8 に従い、自分のラベル (`vllm-baseline.owner=serving-kit`) のコンテナだけが対象であることを照合してから、ログを回収して停止します。

```bash
uv run --directory serving serve logs p2-nope-tp2-smoke --configs var/nope-build-0961bbae/tp2.toml
uv run --directory serving serve status --configs var/nope-build-0961bbae/tp2.toml
uv run --directory serving serve stop --yes
```

停止後に GPU プロセスが 0 件であることを確認します。この試行でサーバーを常駐させません。

## 8. 次の計画に残すもの

- `prefill` の 8k / 32k / 128k、`quality` の needle、`agent` の 2 万〜12 万トークン。いずれも `--max-model-len` を 4096 より広げてからになります
- `concurrency` と並列性能 (`--max-num-seqs` を 1 より増やしてから)、`candidate-d` との比較
