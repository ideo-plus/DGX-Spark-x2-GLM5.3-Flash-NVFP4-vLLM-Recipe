# 1 ステップの GPU の時間の内訳を torch プロファイラーで測る手順 (`p2-nope-tp2-full-prof`)

この手順は、[文脈長 163840・同時実行 16 の TP=2 構成での prefill・並列・長文脈の計測手順](full-context-procedure.md) と [モデル付属の MTP を下書きのトークン数ごとに計測する手順](mtp-procedure.md) の後に行う、**別の試行**です。対象は `p2-nope-tp2-full-prof`（`p2-nope-tp2-full` と同じ構成に、torch プロファイラーの `--profiler-config` を足したもの）です。ADR 0006 の K2（アテンション・共有の専門家・dense・lm_head の量子化）に進むかを決めるために、1 ステップの GPU の時間の内訳を測ります。読むのはプロファイラーの値であって、調査の見積もり（`docs/research/2026-09-25-path-survey.md` §5）ではありません。この文書は手順であって記録ではないので、計測が成功したとは書きません。

この文書や公開記録に書かないこと: 送った内容と応答の本文、認証の情報、`exl3-tp2` の中身。要求・応答の本文は `results/`（Git 対象外）にだけ残り、この文書や `docs/results/` には載せません。認証の情報は、値そのものを書かず、要る場合は環境変数の名前だけを書きます。`exl3-tp2` の中身（計測者が別に起動していた構成）は調べません。

実行位置: 以降のコマンドはすべて、このリポジトリ直下を作業ディレクトリとして実行します。`cd` は使わず、`uv run --directory serving …` のように呼びます。

## 0. 前提: この版の torch プロファイラー

この手順が使うイメージは、固定した vLLM のコミット `0961bbae` に NoPE の修正を加えたものです。この版のソースを読んだ結果は、次のとおりです。

- `--profiler-config` は `vllm/config/profiler.py` の `ProfilerConfig` を受ける正式な引数です。`profiler` に `torch`、`torch_profiler_dir` にコンテナ内の絶対の道筋を書きます。`profiler` を `torch` にするときは `torch_profiler_dir` が要ります。
- 計測の開始と終了は、`vllm/entrypoints/serve/profile/api_router.py` の `POST /start_profile` と `POST /stop_profile` です。**`profiler` を `torch` にした構成でだけ、この経路が付きます。** 付けていない構成では 404 になります。
- trace は、`vllm/profiler/wrapper.py` の `torch.profiler.tensorboard_trace_handler` が、`torch_profiler_dir` の下に `<worker_name>.<time>.pt.trace.json.gz` として書きます。既定で gzip です。この手順の構成は `torch_profiler_dump_cuda_time_total=false` なので、worker は、停止のときに GPU の時間の表（`key_averages`）を書きません。一方、head のフロントエンド（`vllm/v1/engine/async_llm.py` の `AsyncLLM`。CPU だけを記録する）は、`profiler` が `torch` で `ignore_frontend` が既定の偽のとき、停止のときに CPU の表を `profiler_out_0.txt` として `torch_profiler_dir` の下に書きます（`vllm/profiler/wrapper.py` の `_stop`）。これはソースを読んだ結果で、実機では未確認です。`torch_profiler_with_stack=false` なので、Python のスタックも記録しません。
- 1 ステップの区切りの印は、trace に入りません（`vllm/v1/utils.py` の `record_function_or_nullcontext` は、`VLLM_CUSTOM_SCOPES_FOR_PROFILING` が無いと `nullcontext` になるため）。ステップ数は `/metrics` の `vllm:iteration_tokens_total_count` の増分から得ます（§3）。
- 公式の文書にも、trace が大きくなりうるので要求を少数だけ流すこと、停止のときに flush に時間がかかることが書かれています。

**実機でしか分からないこと:** GB10 / FlashInfer 0.7.0 で、実際にどのカーネル名が現れるか。rank 1（worker）が自分の `/logs/torch-profile` に trace を書くか（ソース上は `profile` の RPC が全 worker に届くが、実機では未確認）。`worker_name` の具体的な形。trace の大きさと flush の時間、NCCL と計算のストリームの重なりの有無。

## 1. 構成の生成と読み取り検査

⚠ 生成は推論サーバーの状態を変えませんが、出力先のファイルを作ります。既存の出力があれば上書きしません。

プロファイラー付きの構成を作ります。出力の名前は `tp2-full-prof.toml` にそろえます。

```bash
uv run --directory serving python ../experiments/nope-mla/configure_tp2.py \
  --variant full --torch-profiler \
  ../serving/var/nope-build-0961bbae/image-inspect.json \
  ../serving/var/nope-build-0961bbae/tp2-full-prof.toml
uv run --directory serving serve check p2-nope-tp2-full-prof \
  --configs var/nope-build-0961bbae/tp2-full-prof.toml
```

`--torch-profiler` は、どの `--variant` にも付けられます。付けたときだけ、`args` の末尾に `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/logs/torch-profile","torch_profiler_with_stack":false,"torch_profiler_dump_cuda_time_total":false}'`（根拠つき）が入り、構成の名前が `-prof` 付きになります。2 つの `false` は #48 で足したもので、trace の書き出しのメモリを減らすために、スタックの記録と、worker の `key_averages` の表（GPU の時間の表）の書き出しを切ります（根拠は `vllm/config/profiler.py` の docstring）。`/logs/torch-profile` は、`mount-logs` で結び付けた `/logs` の下なので、`serve logs` で回収できます。`--nccl-thread-names` と併用すると `p2-nope-tp2-full-tn-prof` になります。MTP と併用するときは `--variant full-mtp --spec-tokens 3 --torch-profiler` で `p2-nope-tp2-mtp3-prof` になります。付けないときの出力は、1 バイトも変わりません。`serve check` は `patched-tp2-procedure.md` §6 と同じ 9 関門です。**前の試行で通ったことを根拠に省かず、全関門をもう一度検査します。**

## 2. 起動、IB の確認、短い応答

⚠ 起動による GPU プロセス、コンテナ、ログの状態変更を了承した後に進みます。

```bash
uv run --directory serving serve start p2-nope-tp2-full-prof \
  --configs var/nope-build-0961bbae/tp2-full-prof.toml --timeout 3h --yes
```

進む条件は `status=ready` だけです。起動したら、head の `/metrics` に `vllm:iteration_tokens_total_count` が出ていることを確かめます。IB の確認と、短い応答の確かめ方は [full-context-procedure.md](full-context-procedure.md) §3 と同じです。

```bash
uv run --directory serving serve smoke p2-nope-tp2-full-prof \
  --configs var/nope-build-0961bbae/tp2-full-prof.toml --max-tokens 128
```

これは負荷の本番ではなく、起動と応答の確認です。計測の窓は §3 で開きます。

## 3. 計測の窓: /start_profile と /stop_profile

⚠ 計測は推論サーバーに負荷を掛け、GPU の記録を残します。了承を得た後に進みます。trace は大きくなるので、**計測の窓は短く**し、数分で終わる形にします。

まず、計測の前のステップ数を読みます。

```bash
curl -s http://10.0.1.60:8000/metrics | grep '^vllm:iteration_tokens_total_count'
```

次に、記録を始めます。本文は返りません。`profiler` を `torch` にした構成でだけ付く経路なので、404 なら `-prof` の構成ではありません。

```bash
curl -X POST http://10.0.1.60:8000/start_profile
```

短い負荷を流します。推奨は `serve smoke` で、英・日の固定の短い要求を 1 本ずつ流します（`--max-tokens` で上限を上書きします）。

```bash
uv run --directory serving serve smoke p2-nope-tp2-full-prof \
  --configs var/nope-build-0961bbae/tp2-full-prof.toml --max-tokens 256
```

応答の本文は、画面にだけ出ます。§6 の確認のために、画面に出た英語の応答と日本語の応答から、語を 1 つずつ覚えておきます（書き写さず、この文書にも記録にも載せません）。§2 の短い応答は、計測の窓の前なので、この確認には使えません。

代替は `bench run` です。ただし bench には decode を 1 条件に絞る口が無く、`--suite decode` は 4 条件（`code`・`prose` × `en`・`ja`）を必ず流すので、trace が大きくなります。対象は足さず、既存の `--target p2-nope-tp2-full` を使います。

```bash
uv run --directory bench bench run --target p2-nope-tp2-full --suite decode --profile probe
```

**`bench run` を流すときは、同じ窓で `serve smoke` も流します。** bench の要求は、`serve smoke` の固定の文とは別の文で、§6 では両方を数えるためです。また、§6 の応答の側の確認に使う語は、応答の本文が画面に出る `serve smoke` の応答から得ます。

負荷が終わったら、`/stop_profile` を呼ぶ前に、もう一度 `/metrics` を読みます。head が OOM で止まると、停止の後は読めないことがあるので（読めるかは実機で未確認です）、この値（負荷の終了後・停止の前）を §5 の `--steps` に使います。

```bash
curl -s http://10.0.1.60:8000/metrics | grep '^vllm:iteration_tokens_total_count'
```

次に、記録を止めます。flush に時間がかかるので、少し待ってから次へ進みます。

```bash
curl -X POST http://10.0.1.60:8000/stop_profile
curl -s http://10.0.1.60:8000/metrics | grep '^vllm:iteration_tokens_total_count'
```

ステップ数は、停止の前（負荷の終了後）の値から開始の前の値を引いた差です。この値を、§5 の `--steps` に渡します。停止の後にも読めるなら、同じ値であることを確かめます。差が 0 なら記録できていないので、負荷を増やしてやり直します。

### head の worker の OOM

2026-09-26 の初回の計測では、`/stop_profile` の trace の書き出しで、head の `VLLM::Worker_TP` が oom-killer に止められました（カーネルのログ: `Out of memory: Killed process … (VLLM::Worker_TP)`）。エンジンが止まり、`/stop_profile` は 500 になりました。head は、起動の時点で `available` が約 1 GB しかありません。この構成では上の 2 つの `false` で書き出しのメモリを減らしますが、それでも OOM になりえます。

その場合は、**worker の trace だけ**で読みます。§4 で回収した worker の `/logs/torch-profile/` の `*.pt.trace.json.gz` を §5 で集計します。TP の 2 台は同じ形の計算をするので、1 ステップの内訳は worker の trace で読めます。後始末は §7 のとおり、`serve logs` で回収してから `serve stop` します。head のエンジンが止まった後に `serve logs` が worker の trace を回収できるかは、実機で未確認です。

## 4. trace の回収

`serve logs` は、構成の全ノードの `<remote_root>/logs/` をまるごと回収します（`serving/src/serving_kit/logs.py`）。`/logs/torch-profile/` はその下なので、trace もここで回収されます。追加の回収のしかたは要りません。

```bash
uv run --directory serving serve logs p2-nope-tp2-full-prof \
  --configs var/nope-build-0961bbae/tp2-full-prof.toml
```

回収先は `serving/var/<UTC>-logs-p2-nope-tp2-full-prof/{head,worker}/logs/torch-profile/` です。`*.pt.trace.json.gz`（rank ごと）を確かめます。worker の `profiler_out_*.txt` は書かれません。head の回収先には、フロントエンドの CPU の表 `profiler_out_0.txt` が残ります（§6 で確かめる対象です）。§5 は head と worker のどちらの trace でも集計できます。head が OOM で止まったときは、worker の trace だけで読みます。

## 5. 集計

集計の道具は、回収した trace をローカルで読みます。ネットワークは使いません。

```bash
uv run --directory serving python ../experiments/k2-profile/summarize_trace.py --steps <N> \
  var/<UTC>-logs-p2-nope-tp2-full-prof/head/logs/torch-profile/<file>.pt.trace.json.gz
```

`<N>` は §3 で得たステップ数です。worker の trace も同じ形で集計します。出力は、区分（MoE 専門家 GEMM・MoE 周辺・BF16 重み GEMV/GEMM・アテンション本体・mHC・NCCL・その他・カーネルのない隙間）ごとの合計・1 ステップの平均・割合と、上位のカーネルの一覧（名前・区分・回数・合計）です。「その他」の上位に大きなカーネルがあれば、道具の `CATEGORY_RULES` を見直して再集計します。

## 6. trace に要求・応答の本文が入らないことの確認

trace に入る見込みなのは、カーネル名、演算の名前、Python の関数名とファイル名、形です（`torch_profiler_with_stack=false` なので、Python のスタックは入りません）。`record_shapes` は既定で off の見込みなので、テンソルの中身は入らない見込みです。要求・応答の本文も入らない見込みですが、これは見込みであって、実機の trace で確かめた結果ではありません。次の手順で確かめ、確かめられなかった trace は `docs/results/` に写しません。

確認は、計測の窓（§3）で流した負荷の要求の語と、応答の語を、trace の中で数えて、すべて 0 であることです。

**要求の側。** 窓で流した負荷ごとに、要求に入る語を数えます。

- `serve smoke` の固定の文の語（`capital of France`、`日本の首都`）
- `bench run` を流した窓では、bench の要求に毎回入るシステムプロンプトの語 `benchmarking assistant`（`bench/src/bench_harness/suites/decode.py` の `_SYSTEM_LINE`）

```bash
zgrep -c -F 'capital of France' \
  serving/var/<UTC>-logs-p2-nope-tp2-full-prof/head/logs/torch-profile/*.pt.trace.json.gz
zgrep -c -F '日本の首都' \
  serving/var/<UTC>-logs-p2-nope-tp2-full-prof/head/logs/torch-profile/*.pt.trace.json.gz
# bench run を流した窓だけ
zgrep -c -F 'benchmarking assistant' \
  serving/var/<UTC>-logs-p2-nope-tp2-full-prof/head/logs/torch-profile/*.pt.trace.json.gz
```

**応答の側。** §3 で覚えた、`serve smoke` の英語の応答の語と日本語の応答の語を、同じように数えます。`<…>` には、覚えた語を入れます。選んだ語は、この文書にも記録にも書きません。

```bash
zgrep -c -F '<応答の英語の語>' \
  serving/var/<UTC>-logs-p2-nope-tp2-full-prof/head/logs/torch-profile/*.pt.trace.json.gz
zgrep -c -F '<応答の日本語の語>' \
  serving/var/<UTC>-logs-p2-nope-tp2-full-prof/head/logs/torch-profile/*.pt.trace.json.gz
```

日本語の応答の語を選ぶときは、ロケールに注意します。`[ぁ-ん]` や `[一-龥]` のような文字の範囲の指定は、ロケール（`LC_ALL` や `LANG` が `C` や `POSIX`）では UTF-8 の複数バイトの文字に効かず、当たらない・別の文字に当たることがあります。語は画面の応答から目で選び、数えるときは上のとおり `-F` の固定文字列で渡します。範囲の指定を使うなら、UTF-8 のロケール（例: `LC_ALL=C.UTF-8`）を付けます。

上のコマンドは head の trace の例です。worker の trace（回収先の `worker/` の下）と `profiler_out_*.txt`（head のフロントエンドの表）にも、同じ語で同じ数え方をします。

**結論。** 両 rank の trace と `profiler_out_*.txt` のすべてに対して、上の語がどれも 0 のときだけ、「本文は入っていない」と判断します。次のときは、その trace を `docs/results/` に写さず、`serving/var/`（Git 対象外）に留めます。

- 0 でない語がある: 記録に「本文の疑い」と書きます。
- 確認できなかった: 窓で `serve smoke` を流さなかった（`bench run` だけを流した）、または応答の語を覚えていない場合です。要求の側が 0 でも、応答の側は確かめられていないので、記録に「応答の側は未確認」と書きます。窓を開け直し、`serve smoke` も流して、やり直します。

## 7. 停止

⚠ 停止は推論サーバーとコンテナの状態を変えます。了承を得た後に進みます。`serve stop` は記録を回収しないので、先に `serve logs` で回収します。

```bash
uv run --directory serving serve logs p2-nope-tp2-full-prof \
  --configs var/nope-build-0961bbae/tp2-full-prof.toml
uv run --directory serving serve status --configs var/nope-build-0961bbae/tp2-full-prof.toml
uv run --directory serving serve stop --yes
uv run --directory serving serve status --configs var/nope-build-0961bbae/tp2-full-prof.toml
```

停止後に、GPU プロセスが 0 件であることを確かめます。
