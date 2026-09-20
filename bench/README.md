# bench-harness

`bench-harness` は、DGX Spark 2 台で動く GLM-5.3-Flash の推論サーバーを、作業用の Mac から
Anthropic 互換の `/v1/messages` 経由で測る、コマンドラインの計測の道具である。生成の速さ、入力の
処理と最初のトークンまでの時間、同時処理、品質の検査、長い会話でのツール呼び出しを、同じ入力と
同じ手順で何度でも測り直せるようにする。

測り方をなぜそう決めたかは、[`docs/decisions/0001-bench-harness-measurement-method.md`](../docs/decisions/0001-bench-harness-measurement-method.md)
にある。使っている部品と課題のライセンスは [`LICENSES.md`](../LICENSES.md) にある。

| まとまり (`--suite`) | 測るもの | 条件の鍵 |
|---|---|---|
| `decode` | 生成速度 (コードと散文 × 英語と日本語) | `decode/{code,prose}/{en,ja}` |
| `prefill` | 最初のトークンまでの時間と、入力の処理速度 | `prefill/{cold,warm}/{8k,32k,128k}` |
| `concurrency` | 同時 1、2、4、8 本 | `concurrency/c{1,2,4,8}` |
| `quality` | ツール呼び出しの正確さ、コードの課題、長い入力から情報を探す課題 | `quality/…` |
| `agent` | 会話を 2 万 → 12 万トークンに伸ばしたときの、ツール呼び出しの崩れ | `agent/stage/NNNk` |

## 対象サーバーに求める前提

計測者が、対象サーバーを起動するときに満たしておくこと。

- **`POST /v1/messages` がストリームで応答し、応答にトークン数が載る** (vLLM なら v0.11.1 以降)。
  これだけが必須で、満たされないと計測ランのディレクトリを作らずに終わる
- **ツール呼び出しを測る (`--suite quality` / `--suite agent`) なら、自動のツール選択と、モデルに
  合ったパーサーが有効**であること (GLM-5.3-Flash なら `glm47`)
- **キャッシュの効きを応答から確かめたいなら、入力のトークンの内訳を返す設定が有効**であること
  (vLLM なら `--enable-prompt-tokens-details`)。なくても `/metrics` の増分で確かめられる
- **`bench calibrate` を使うなら、`POST /v1/messages/count_tokens`** があること (vLLM なら
  v0.17.0 以降)。なければ、出力 1 トークンの要求を送って、返ってきた入力のトークン数で代える
- **計測の間、ほかの利用者が対象サーバーを使わない**こと。始める前に実行中の要求の数を読み、
  0 でなければ警告を記録する

この道具が触る口は、次の 5 つだけである。**どれも推論か読み取りで、サーバーの状態を変える口は使わない。**

| 口 | いつ使うか | なければ |
|---|---|---|
| `POST /v1/messages` | すべての試行 | 計測できない |
| `POST /v1/messages/count_tokens` | `bench calibrate` だけ | 出力 1 トークンの要求で代える |
| `GET /v1/models` | 前提の確認 (入力の長さの上限 `max_model_len`) | `targets.toml` の `max_context_tokens` → 不明、の順で決める |
| `GET /version` | 前提の確認 (サーバーの版の記録) | 「不明」と記録して続ける |
| `GET /metrics` | まとまりと条件の前後、計測の間の定期的な読み取り | 内部の指標なしで続ける (得られなかった名前を残す) |

## Mac 側の用意

### 道具

```bash
# bench/ で
uv sync
```

`uv` と Python 3.12 以上が要る (`uv` が選ぶのは 3.13)。この道具は `bench/` で `uv run` する前提で、
設定ファイルの場所を `bench/config/` からの相対で探す。

### コードの課題を測るなら、隔離のイメージを作る

`--suite quality` のコードの条件だけが、コンテナの実行環境を使う。**Docker または Podman が要る。**
ないときは、コードの条件だけが理由つきで飛ぶ (ほかの条件は流れる)。

イメージは、公式の `python:3.13-slim` (ダイジェストで固定) に numpy だけを足した自前のものである。
HumanEval+ の検査のプログラムが numpy を使うため。

```bash
# bench/ で
docker build -t bench-sandbox:py3.13-numpy2.5.3 sandbox/
docker image inspect bench-sandbox:py3.13-numpy2.5.3 --format '{{.Id}}'
```

**出てきた識別子 (`sha256:…`) を、`config/profiles.toml` の `sandbox.image_digest` に書き写すこと。**
採点の側は、手元のイメージの識別子がこの値と合わなければ、使えないものとして扱う (合わないイメージでは
動かさない)。**作り直すと識別子が変わる**ので、そのたびに書き換える。

Podman を使うなら、`docker` を `podman` に読み替える。`profiles.toml` の `sandbox.runtime` は
`"auto"` (podman、docker の順に探す)、`"podman"`、`"docker"` から選ぶ。

## 設定ファイル

| ファイル | 中身 |
|---|---|
| `config/targets.toml` | 対象サーバーの定義 (接続先、モデルの名前、メモ、認証の情報の環境変数の名前、内部の指標の名前の上書き) |
| `config/profiles.toml` | 計測の設定。`quick` (開発中の素早い確認) と `full` (公開する結果を作る本番) の 2 つ |

**認証の情報は、値ではなく、値を入れた環境変数の名前を `api_key_env` に書く。** 値は実行時に読み、
生データにも要約にも標準エラーにも出さない。`targets.toml` に実在する秘密の値を書いてはならない。

```toml
[targets.own-p1]
base_url = "http://10.0.1.60:8000"
model = "glm-5.3-flash"
api_key_env = "BENCH_OWN_API_KEY"
```

**`metric_map` に書く名前は、解析したあとのサンプルの名前である。** カウンターは、対象サーバーの
出力に `_total` が見えなくても `_total` を付けて書く。ヒストグラムは `_sum` と `_count` に分けて書く。

```toml
[targets.own-p1.metric_map]
kv_usage = "custom:kv_cache_usage_ratio"              # Gauge はそのまま
prefix_hits = "custom:prefix_cache_hits_total"        # Counter は _total を付ける
iteration_tokens_sum = "custom:iteration_tokens_sum"  # Histogram は _sum と _count
iteration_tokens_count = "custom:iteration_tokens_count"
```

### thinking は、対象サーバーの既定のまま

`sampling.thinking` に書けるのは `"server_default"` だけである。`/v1/messages` から thinking を
切り替える渡し方を、2026-09-20 に実機で 5 通り試したが、どれも出力を変えなかった (くわしくは
`docs/decisions/0001-bench-harness-measurement-method.md`)。thinking を切り替えて比べたいときは、
対象サーバーの側 (起動の引数やチャットテンプレート) で切り替えた構成を、別の名前の対象として
`targets.toml` に足して測ること。

## コマンド

以下はすべて `bench/` で実行する。`--help` に、この節より詳しい説明がある。

### `bench calibrate` — 1 トークンあたりの文字数を測る

入力の長さは、内容の種類ごとの「1 トークンあたりの文字数」で狙う。**対象サーバーを変えたら、
まずこれを流して、比を設定に書き写す。**

```bash
uv run bench calibrate --target candidate-d --profile quick
```

標準出力に出るのは、`profiles.toml` にそのまま貼れる TOML の断片だけである (説明は標準エラーに出る)。

```toml
# bench calibrate --target candidate-d --profile quick
[profiles.quick.chars_per_token]
prose_en = 4.012  # 8024 文字 / 2000 トークン (count_tokens)
...
```

### `bench run` — 計測ランを 1 回流す

```bash
# 既定は、速さの 3 つ (decode, prefill, concurrency)
uv run bench run --target candidate-d --profile quick

# まとまりを名指しする (繰り返して複数選べる。書いた順に流す)
uv run bench run --target candidate-d --suite decode --suite prefill

# 品質の検査だけ
uv run bench run --target candidate-d --suite quality

# 長い会話の検査だけ
uv run bench run --target candidate-d --suite agent

# 5 つ全部 ('all' はほかの名前と混ぜられない)
uv run bench run --target candidate-d --suite all
```

**`--suite` を省いたときの既定は、速さの 3 つだけ**である。品質の検査と長い会話の検査は、実機では
1 回の計測ランが何時間もかかる (下の「どれくらい時間がかかるか」) ので、名指ししたときだけ流す。

試行の回数などの整数の項目は、`--set 経路=整数` で上書きできる (繰り返し指定でき、同じ経路を 2 回
書いたら、あとのものが効く)。

```bash
# 生成速度の試行を増やす
uv run bench run --target candidate-d --suite decode --set decode.trials=20

# 長い会話の検査を 1 段階 (2 万トークン) だけ、試行を減らして流す
uv run bench run --target candidate-d --suite agent \
  --set agent.end_tokens=20000 --set agent.trials_per_stage=10

# コードの課題を 10 問に減らす
uv run bench run --target candidate-d --suite quality --set quality.code_problem_limit=10

# 同時処理の回数を減らす
uv run bench run --target candidate-d --suite concurrency --set concurrency.rounds=2
```

公開のコードの課題 (HumanEval+) の置き場所と、取得の可否も選べる。

```bash
# 置き場所を変える (既定は bench/data-cache。git の管理の対象になり得る場所は拒む)
uv run bench run --target candidate-d --suite quality --data-cache ~/bench-data

# 取得を禁じる (置き場所になければ、コードの条件だけを理由つきで飛ばす)
uv run bench run --target candidate-d --suite quality --no-download
```

計測が終わると、要約 (`summary.json` と `summary.md`) が自動で作られる。

### `bench summarize` — 要約を作り直す

生データから作り直すだけなので、何度実行しても同じ結果になる。未完了の計測ランも要約できる。

```bash
uv run bench summarize 20260920T031500Z-candidate-d-a1b2c3
uv run bench summarize ../results/20260920T031500Z-candidate-d-a1b2c3   # 経路でもよい
```

`RUN` の引数は、経路の区切りを含まない名前なら、**先に生データの置き場所 (`--results-root`、
既定は `<リポジトリ>/results`) の下を探す**。いまいるディレクトリに同じ名前があっても、
置き場所の下が勝つ。

### `bench compare` — 2 つの計測ランを比べる

```bash
uv run bench compare 20260920T031500Z-candidate-d-a1b2c3 20260920T045500Z-candidate-d-d4e5f6

# 比較の結果を、ディレクトリの comparison.md にも書く
uv run bench compare RUN_A RUN_B --out ../results/compare-20260920
```

**収まらない条件があっても 0 で終わる。** 判定は結果であって、この道具の失敗ではない。

**比べる計測ランは、コミットしてきれいな状態で流すこと。** 道具の版は作業木の全体を見るので、
`docs/` や `.kiro/` に未コミットの変更があるだけで `.dirty` が付き、比較の先頭で警告になる
(無視されたファイルは数えない)。

### `bench publish` — 要約だけを公開の場所に写す

```bash
uv run bench publish 20260920T031500Z-candidate-d-a1b2c3
uv run bench publish RUN --docs-root ../docs/results   # 既定と同じ
```

写すのは `summary.json` と `summary.md` の 2 つだけで、生データは写さない。

**公開する前に、`summary.md` を目で見て確かめること。** この道具が守るのは「2 つのファイルだけを、
公開の場所の内側に写す」ところまでで、手で書き換えられた `summary.md` の中身の真正性までは見ない。

## 終了の値

| 値 | 意味 |
|---|---|
| 0 | 完了した。`bench compare` は、収まらない条件があってもここ。上限に達して段階を止めた計測ランもここ |
| 1 | 前提の不足 (接続できない、トークン数がない) または設定の誤り (知らない対象サーバーの名前、知らないまとまりの名前、`--set` の書式の誤り、下限を割った値、使えない `--data-cache`) |
| 2 | 連続の失敗で停止。**保存と公開の失敗もここ** (計測ランでない場所を指した、生データが読めない、要約を書けなかった、公開を断った、ディスクの書き込みに失敗した)。計測は終わったのに要約を書けなかったときも、0 にせず 2 |
| 130 | 中断 (`SIGINT`) |

## 標準出力に出る行

後続の処理が読める約束として、`bench run` と `bench summarize` の標準出力の行の形を固定してある。
**進み具合、警告、エラーは、すべて標準エラーに出る。**

```
計測ラン: <識別子>
生データ: <ディレクトリの経路>
状態: <running | completed | aborted | interrupted>
要約: <summary.md の経路>
要約 (JSON): <summary.json の経路>
```

`bench calibrate` の標準出力は、`profiles.toml` に貼れる TOML の断片だけである。
`bench publish` は、公開先のディレクトリと、写した 2 つのファイルの経路を出す。

## 結果はどこに残るか

```
results/<計測ランの識別子>/       # git の管理の対象ではない。リポジトリには入らない
├── manifest.json                # 実行の条件と状態
├── trials.jsonl                 # 試行のレコード (1 行 1 試行。送った内容と応答の本文を含む)
├── bodies/<sha256>.json.gz      # 送った要求の本文 (圧縮。内容のハッシュで重複を除く)
├── metrics/<条件>.before.prom   # 加工する前の /metrics
├── metrics/<条件>.after.prom
├── metrics/deltas.jsonl         # 条件ごとの増分と導出値
├── summary.json                 # 道具が読む要約
└── summary.md                   # 人が読む要約

docs/results/<計測ランの識別子>/  # bench publish で写した要約だけ
├── summary.json
└── summary.md
```

**`results/` は、決してリポジトリに入れない。** 送った内容と応答の本文が入っているためで、
道具自身も、計測を始める前にこの場所が git の管理の対象でないことを確かめ、そうでなければ
計測ランのディレクトリを作らずに止まる。認証の情報の値は、生データのどのファイルにも現れない
(ヘッダーにしか付けていない)。

## `summary.md` の読み方

先頭から、実行の条件 → 警告 → 飛ばした条件 → 表の読み方 → 主な結果 → 参考 → 品質の検査 →
長い会話でのツール呼び出し → 対象サーバーの内部の指標 → 使った公開の課題、の順に並ぶ。

- **未完了の計測ラン**は、先頭に「この計測ランは未完了である」と出る。状態が `completed` でない
  ものは、すべて未完了として扱う
- **主な結果と参考**は、節が分かれている。参考は同時 4 本と 8 本で、結論には数えない
- **値の名前**: `ttft_s` = 最初のトークンまでの時間 (秒)、`decode_tps` = 生成速度 (トークン/秒)、
  `prefill_tps` = 入力の処理速度 (トークン/秒)、`round_total_tps` = 1 回ぶんの合計の生成速度。
  集計に入っているのは、成功して、慣らしでない試行だけである
- **`n` の単位は行によって違う。** `ttft_s`、`decode_tps`、`prefill_tps` は試行の数 (同時処理では
  1 本が 1 つ)、`round_total_tps` は回の数。`decode_tps` は、出力が 16 トークン未満の試行を
  除いたあとの数である
- **印 (フラグ)** は、その条件で何件あったかを示すだけで、集計からは外していない
  (「出力のトークンが少なすぎる」だけが例外で、`decode_tps` の行からは除いてある)

| `summary.md` の「印」の欄 | 意味 | `summary.json` の名前 |
|---|---|---|
| 試行の数が足りない | その**行**の値の数が、`min_successes` に届かない | `insufficient_trials` |
| 一部が失敗 (成功した要求だけから求めた値) | 同時処理の 1 回ぶんの一部が失敗した | `partial_failures` |
| 早く終わった N 件 | 出力の上限に届かずに終わった | `short_outputs`、`flag_counts.short_output` |
| 長さが外れた N 件 | 実際の入力のトークン数が、狙いから許容の幅 (既定 ±5%) を超えて外れた | `length_off_target` |
| 出力が壊れている疑い N 件 | 置き換え文字 (U+FFFD) か、同じ並びの繰り返し。**表の区切り行や CSV では、壊れていなくても付く** | `suspect_outputs`、`flag_counts.replacement_char`、`flag_counts.repetition_loop` |
| 出力のトークンが少なすぎる N 件 (この行から除いた) | 出力が 16 トークン未満。`decode_tps` の行にだけ出る | `flag_counts.too_few_output_tokens` |

- **採点できなかった件数は、不正解と分けて出る。** 品質の検査の表は
  「条件 / 正解 / 採点の対象 / 正解の割合 / 95% の区間 / **採点できなかった** / 要求の失敗 / 印」で、
  割合 (`accuracy`) の分母は「採点の対象」の列 (採点できた試行) だけである。要求そのものが失敗した
  試行と、隔離の実行環境がなくて採点できなかった試行は、分母から外して「採点できなかった」に数える
  (`summary.json` では `flag_counts.not_scored`)
- **段階の表** (長い会話の検査) は「段階 / 狙いの入力 / 実際の入力 (中央値) / 試行 / 崩れ / 分母 /
  崩れた割合 / 95% の区間 / **片側 95% の上限** / 判定 / 印」で、その下に 9 種類の分類の内訳が続く。
  表の前に、しきい値、**しきい値を初めて超えた会話の長さ**、採点できた試行が残っているいちばん長い
  段階、止めた理由が並ぶ
- **「片側 95% の上限」は、両側の区間とは別の量**である。「しきい値を下回ったと言えるか」の判定に
  だけ使う。判定は 3 つ: `下回った` (片側の上限がしきい値より小さい)、`上回った` (両側の下限が
  しきい値より大きい)、`試行の数が足りない` (どちらとも言えない。崩れが 0 件なら、あと何回要るかが添う)
- **崩れが 0 件でも、「1% 未満」と言うには 1 段階あたり 299 試行が要る。** `quick` (1 段階 50 回) は
  傾向を見るためのもので、判定には `full` (1 段階 300 回) を使う

## どれくらい時間がかかるか

**品質の検査と長い会話の検査は、実機では何時間もかかる。** `--suite` を省いたときに流れないのは、
このためである。

設定 `quick` で送る要求の数 (設定から数えたもの。上限で飛ばされた条件があれば、その分は減る):

| まとまり | `quick` | `full` |
|---|---|---|
| `decode` | 48 (4 条件 × (10 + 慣らし 2)) | 88 |
| `prefill` | 36 (6 条件 × (5 + 慣らし 1)) | 66 |
| `concurrency` | 90 ((5 回 + 慣らし 1 回) × (1+2+4+8) 本) | 165 |
| `quality` | 120 (ツール 50 + コード 40 + 探す課題 30) | 423 |
| `agent` | 300 (6 段階 × 50) | 1,800 |

**実機でかかった時間 (2026-09-20、head の 8001 番の glm-5.3-flash、EXL3、TP=2、`quick` の設定):**
速さの 3 つのまとまりが 45 分 (`decode` だけで 23 分)、`quality` が 20 分半、`agent` の 6 段階が
19 分で、全部で約 1 時間半。同時処理を比較に使う 20 回 (`--set concurrency.rounds=20`) にすると、
`concurrency` だけで 30 分かかる。

**長い会話の検査は、キャッシュの効き方で大きく変わる。** `quick` は 300 要求で入力が約 2,090 万
トークン、`full` は 1,800 要求で約 1 億 2,560 万トークンを送る (保存する本文は約 0.09 GB と
約 0.56 GB)。上の 19 分は、プレフィックスキャッシュの当たり率が 0.88〜0.96 だったときの数字で
ある。**キャッシュがまったく効かない対象では**、入力の処理が毎秒 1,000〜2,000 トークンとして
`quick` で 2.9〜5.8 時間、`full` で 17〜35 時間かかりうる (これは計算で、確かめていない)。

はじめて実機で流すときは、まず 1 段階だけ流して、内訳を見ること。

```bash
uv run bench run --target candidate-d --suite agent --set agent.end_tokens=20000
```

## うまくいかないとき

### 公開のコードの課題のファイルが壊れている

ハッシュが合わないファイルは、**黙って取り直さない**。期待した値と実際の値、ファイルの場所を
添えて止まる (取り直しを自動でやると、入れ替えられたファイルが「直った」ように見えてしまう)。

```bash
# bench/ で。--data-cache で場所を変えているなら、その下のファイルを消す
rm data-cache/HumanEvalPlus-OriginFmt-v0.1.10.jsonl.gz
```

**置き場所のファイルを消してから、やり直すこと。** 中身が期待の形でない場合 (`DatasetFormatError`)
も同じである。「取得できない」(ネットワークに届かない、`--no-download` を付けた) だけは、
コードの条件を理由つきで飛ばして、ほかの条件は流れる。

### traceback が出て、計測ランが `running` のまま残った

課題の側の不具合 (`ToolTaskError`) と、課題のファイルの壊れは、**握りつぶさずに traceback のまま
外へ出す**。モデルの誤りとして数えてしまうと、正確さの数字が静かに歪むためである。この 2 つは
道具か課題の不具合なので、直してから測り直す。

計測ランは `running` のまま残る。それまでの試行は生データに残っているので、要約は作れる
(未完了として表示される)。

```bash
uv run bench summarize <識別子>
```

### コードの隔離の実行環境が使えない

要約の「飛ばした条件」に、理由つきで出る。よくある原因は 2 つ。

- **イメージがない、または識別子が合わない** — 上の「隔離のイメージを作る」のとおりに作り直し、
  `sandbox.image_digest` を書き換える。採点の途中でイメージを取りに行かせない (`--pull never`) ので、
  手元になければ、そこで使えないと判定される
- **Docker / Podman が動いていない** — 実行環境を起動してから、やり直す

### 道具を途中で止めたあとに、隔離のコンテナが残っていないか

ふつうに止めたとき (Ctrl-C、時間切れ) は、道具がコンテナを `kill` してから終わる。道具のプロセスが
強制終了されたときは、その後始末が走らない。コンテナは中の時間切れ (設定の時間切れ + 10 秒) で
自分から終わるが、念のため、次で残りがないことを見ること。残っていたら `docker kill <名前>` で止める。

```bash
docker ps -a --filter name=bench-sbx
```

### `--suite quality` の始まりで、しばらく止まったように見える

品質の検査の計画は、**コンテナの確認と、課題の読み取り (初回は 1.35 MB の取得) の間、待ちに入る**。
その間は進み具合の行が出ないが、異常ではない。

### 比較で「収まらない」条件が出た

**構成の違いだと決める前に、まず試行の回数を 20 回以上に増やして測り直すこと。** 測り方の変動係数が
3% を超えると、20 条件のうち 1 件以上で偽の「収まらない」が出る確率が 5 割を超える。
比較の結果の末尾にも、同じことが書いてある。

「判定できない」は「収まった」ではない。値の数が足りない行には、警告に理由が出る。

### 比較の警告に「対応のある比較に落ちなかった」と出た

**片方に失敗が 1 件あるだけで、対応のある比較から対応のない比較に落ちる。** 検出力が大きく下がり、
同じ差でも「収まる」と出やすくなる。失敗の原因を取り除いてから測り直すこと。

## 開発

```bash
# bench/ で
uv sync                      # 依存を解決し、開発用の仮想環境を作る
uv run ruff format --check . # 整形の確認
uv run ruff check .          # 静的検査
uv run mypy                  # 型の検査 (strict)
uv run pytest -q             # 試験
uv run bench --help          # CLI の疎通確認
```

試験は、偽のサーバー (`tests/fake_server.py`、標準ライブラリだけ) を相手にする。実機も、
本物の推論サーバーも要らない。コンテナを使う試験は、実行環境がなければ理由つきで飛ぶ。

### 壊し方の確認 (ミューテーション) の落とし穴

試験が本当に壊れ方を捕まえるかを確かめるときは、2 つの罠に気をつける。

- **`__pycache__` の古いバイトコードが使われる。** 大きさの変わらない書き換えを同じ秒のうちに
  行うと、結果が嘘になる。`PYTHONDONTWRITEBYTECODE=1` を付け、変異のたびに `__pycache__` を消し、
  変異なしで通ることを毎回確かめ直すこと
- **リポジトリの外に複製するときは、`.venv` を複製しない。** editable の導入が元のリポジトリを
  指したままになり、変異が効かない

```bash
rsync -a --exclude .venv --exclude '*cache*' bench/ /tmp/bench-mutate/
```

複製の側のコードが試験されていることを、**先にわざと壊して確かめてから**、本題の変異に進むこと。
