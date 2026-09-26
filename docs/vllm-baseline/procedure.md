# P1 の手順書 (Baseline Procedure)

この文書は、作業用の Mac から DGX Spark 2 台に、上流の公式の vLLM で GLM-5.3-Flash の推論
サーバーを立て、`bench` の計測を一通り流すまでの、**打つ順番と、進む条件と、止める条件**を
書いたものである。

- 要件は [`.kiro/specs/vllm-baseline/requirements.md`](../../.kiro/specs/vllm-baseline/requirements.md)、
  設計は [`design.md`](../../.kiro/specs/vllm-baseline/design.md)、実機でしか決まらないことの
  一覧は [`research.md`](../../.kiro/specs/vllm-baseline/research.md) にある
- 道具 (`serve`) の使い方、終了コード、Spark の上の置き場所、了承の流れは
  [`serving/README.md`](../../serving/README.md) にある。**サブコマンドと引数は、その表が正**で、
  この文書は、それを段の順に並べたものである
- リポジトリの正本は Mac にある。**Spark の上でファイルを編集しない** (配布は `serve push`)

## この文書に書かないこと (要件 10.5)

1. **送った内容と、応答の本文**。書くのは、長さ、トークンの数、終わりの理由、HTTP の状態だけ
2. **認証の情報** (鍵、トークン)。Spark にも置かない (要件 2.6)
3. **計測者が別に起動していた構成 (`exl3-tp2`) の中身** (起動の引数、設定、差し込まれた
   ファイル、記録)。道具は、自分のラベルのないコンテナに、どの操作も向けない (要件 2.3、2.4)。
   よそについて読むのは、GPU を使っているプロセスの名前とメモリの量だけである

同じ決まりが、この手順で書き足す `not-working.md`、`attempts.md`、`docs/decisions/` の ADR、
`docs/results/` の要約にも掛かる。

## 印の読み方

| 印 | 意味 |
|---|---|
| ⚠ 計測者に尋ねる | ここで止まり、計測者の答えを待つ。Claude が代わりに打つときは、**会話で了承を得てから** `--yes` を付ける (自動で付けない) |
| → 進む条件 | これを満たしたときだけ、次に進む |
| ✋ 止める条件 | これに当たったら、その場で止めて、記録を書く |
| 記録 | 書き足すファイル |

## 段と、構成の名前の対応 (要件 8.1)

| 段 | 構成の名前 | いま | 台 | 主なコマンド |
|---|---|---|---|---|
| 段 0 | `probe-pinned` | ある | 1 (head) | `serve probe probe-pinned` |
| 段 1 | `probe-nightly` | 2026-09-22 に作った (段 0 の失敗を受けて) | 1 (head) | `serve probe probe-nightly` |
| 段 2 | `p1-nvfp4-tp2` | ある | 2 | `serve start p1-nvfp4-tp2` |
| 段 3 | `p1-nvfp4-tp2-x<連番>` | この段で作る (足した指定ごとに 1 つ) | 2 | `serve start <段 3 の構成>` |
| 段 4 | `p1-w4a16-tp2` | この段で作る (第二の候補の重み) | 2 | `serve start <段 4 の構成>` |

段の間には、2 つの関門がある (段ではないが、通らなければ先に進まない)。

| 関門 | 構成の名前 | いま | 主なコマンド |
|---|---|---|---|
| 関門 A: 通信の確認 | `netcheck-bandwidth`、`netcheck-bandwidth-ib`、`netcheck-sanity` | ある | `serve netcheck links` / `bandwidth` / `sanity` / `ab` |
| 関門 B: 重みの取得 | `p1-fetch-nvfp4` | ある | `serve fetch p1-fetch-nvfp4` |

段より前の準備に使う構成は、`p1-fetch-nvfp4-probe` (段 0 に要る設定とトークナイザだけの取得)
と `p1-image-licenses` (イメージの中のライセンスの表記の読み取り) である。

要件 8.1 が定める順序は 5 つ (段 0〜段 4) で、そのあとの「打ち切り」と「まとめ」は段ではない。
まだない構成 (段 3、段 4) は、**その段に進むと決まってから作る**。作るときの決まりは、それぞれの
段の節にある (段 1 の `probe-nightly` は、段 0 の失敗を受けて 2026-09-22 に作った)。

## 全体の順序

| # | すること | 分かれ道 |
|---|---|---|
| 準備 1 | 配布 (`serve push`) | — |
| 準備 2 | イメージの取得 (`serve pull-image`) | — |
| 準備 3 | ライセンスの表記 (`serve image-licenses` → `LICENSES.md`) | 方針に合わなければ、採用せずに尋ねる |
| 準備 4 | 段 0 に要るファイルの取得 (`serve fetch --probe-files` → `serve verify`) | — |
| 段 0 | 1 台の縮小の確認 | 0 = 起動できた → 関門 A / 2 = 失敗した → 段 1 / 1 = 判定できなかった → `detail` の先頭を読む |
| 段 1 | より新しい公式のイメージで、同じ確認 (段 0 が失敗したときだけ) | 起動できた → 関門 A / 同じ場所で失敗 → 打ち切り |
| 関門 A | 通信の確認 (`links` → `nodes.toml` を埋める → `bandwidth` → `sanity` → 要れば `ab`) | 通らなければ、2 台での起動に進まない |
| 関門 B | 第一の候補の重みの取得 (`serve fetch` → `serve verify`) | 合わなければ止まる |
| 段 2 | 2 台で TP=2 の起動 → 確かめ → `bench` → 見張り → thinking | 届いた → まとめ / 届かない → 段 3 |
| 段 3 | 明示の指定を足して試す | 動いた → 段 2 の計測に戻る / どれも届かない → 段 4 を尋ねる |
| 段 4 | 第二の候補の重みで試す | 動いた → 段 2 の計測に戻る / 届かない → 打ち切り |
| 終わり | まとめ、判断の記録、片付け | — |

段 0 は 1 台だけで済むので、通信の確認より先に流す (安い順。design.md 「代替の順序」の注記)。
段 1 は、段 0 が失敗したときだけ通る枝である。

## 共通の決まり

### どこで打つか

`serve` は、Mac の `serving/` で流す。`bench` は Mac の `bench/` で流す。

```bash
cd serving   # serve …
cd bench     # bench …
```

ssh で Spark に入って手で打つのは、この手順の外である (道具が出す ssh だけを通す)。

### 試す前に `attempts.md` を見る (要件 8.3)

**どの段でも、1 つめのコマンドを打つ前に
[`attempts.md`](attempts.md) を読み、同じ構成で同じ失敗を、設定を変えずに繰り返さない。**
同じ構成をもう一度試すなら、何を変えたのかを、その行に書ける状態にしておく。

### 状態を変える操作の了承 (要件 2.1)

⚠ 計測者に尋ねる: **Spark の状態を変える 11 のコマンド
(`push`、`pull-image`、`image-licenses`、`fetch`、`verify`、`start`、`stop`、`probe`、
`netcheck bandwidth`、`netcheck sanity`、`netcheck ab`) は、打つ前に毎回尋ねる。**

道具は、関門を流したあと、これから流すコマンドと、その巻き戻しと、対象の機械を標準エラーに
全部見せて、`yes` の入力を待つ。端末でなく `--yes` もなければ、状態を変える呼び出しを 1 つも
出さずに終了コード 1 で終わる。Claude が代わりに打つときは、会話で了承を得てから `--yes` を
付ける。

### 終了コードの読み方

| 値 | 意味 | この手順での扱い |
|---|---|---|
| 0 | 正常 (すでに望む状態だった場合を含む) | → 進む |
| 1 | 前提の不足、断り (構成の誤り、関門の不通過、了承されなかった、ssh で入れない、`serve status` に読めない台がある、`serve probe` の「判定できなかった」、`serve thinking` の「判定できなかった」) | ✋ 直してやり直す。直せないなら記録して尋ねる |
| 2 | 実行しての失敗 (時間切れ、コンテナの終了、名前の衝突、照合の不一致、GPU が空かない、確認の不合格、`serve watch` が出来事を見つけた) | ✋ `not-working.md` に 1 件書く |
| 130 | 中断 (Ctrl-C) | 片付けは道具が済ませている。`serve status` で確かめる |

`serve` が**引数の使い方そのもの**を断るとき (知らないフラグ、足りない位置の引数) は、
`argparse` の決まりで 2 になる (`bench` と同じ。上の表の「実行しての失敗」とは別のもので、
Spark には 1 度も触っていない)。

`detail=` の行は、標準出力に 1 行で収めた要約で、全文は標準エラーに出る。**終了コードだけで
判断せず、`status=` と `detail=` を必ず読む。**

### 書く記録

| ファイル | いつ書くか | 何を書くか |
|---|---|---|
| [`attempts.md`](attempts.md) | 段を移るたびに必ず | 日時、段、構成の名前、結果、止まった場所、次に進む理由、記録の置き場所 (要件 8.2) |
| [`not-working.md`](not-working.md) | 失敗、不合格、飛ばしたまとまりがあったとき | 現象、再現の条件、誤りの文面、対応しそうな上流の issue、回避できたか、関わる段階 (要件 10.1) |
| `docs/results/<日付>-<何>.md` | 実測を公開の場所に残すとき (⚠ 計測者の指示があったときだけ) | 測った値と、測り方 |
| `docs/decisions/000N-*.md` | 判断したとき | `0002` イメージと重みの選定、`0003` 通信の設定の採否、`0004` `/v1/messages` の不足の扱い、`0005` 終わりの条件 (要件 10.3) |
| `serving/var/…` | 道具が自動で | 回収した記録 (git の管理の外。人が書かない) |

構成の値を書き換えたとき (段 1、段 3、段 4、`ready_timeout_s` の直し、NCCL の記録の変数の
足し外し) は、`attempts.md` の行に、**その構成で何を変えたか**を書く。

### 恒久的な設定の変更が要るとわかったとき (要件 2.8)

⚠ 計測者に尋ねる: **Spark の OS、ドライバ、ネットワークの恒久的な設定を変えない。**
変更が要るとわかったら、変更せずに、要る変更の内容と理由を `not-working.md` に書いて止まり、
計測者の判断を仰ぐ。ホストにパッケージを入れるのも、これに当たる (道具は `sudo`、`apt`、
`pip`、`systemctl` を呼べない)。

## 準備 (段より前)

### 準備 1. 配布 (`serve push`)

**前提**: 2 台に鍵認証で入れる。`nodes.toml` の `ssh_host`、`lan_addr`、`remote_root` が
埋まっている (直結の 3 つは、この時点では空でよい)。

⚠ 計測者に尋ねる (状態を変える)。

```bash
uv run serve push --yes
```

- 2 台の `remote_root` の下に 6 つの置き場所 (`payload/`、`models/`、`probe/`、`cache/`、
  `logs/`、`state/`) を作り、`serving/payload/` だけを配る
- **配布は、イメージの取得より先**である。ディスクの空きの関門が `remote_root` を見るので、
  置き場所がないと断られる

**→ 進む条件**: 終了コード 0。`status=pushed`。

**✋ 止める条件**: ssh で入れない (1)、`mkdir` か配布が失敗した (2)。

**記録**: なし (`attempts.md` は段から書く)。

### 準備 2. イメージの取得 (`serve pull-image`)

**前提**: 準備 1 が済んでいる。

7 つの構成は、**同じダイジェストのイメージを固定している**。取得は 1 回で足りるが、どの台に
入るかは、選んだ構成の台数で決まる。**2 台の構成を選ぶ** (`p1-fetch-nvfp4-probe` は 2 台)。

⚠ 計測者に尋ねる (状態を変える。約 9.7 GB × 2 台)。

```bash
uv run serve pull-image p1-fetch-nvfp4-probe --yes
uv run serve check p1-fetch-nvfp4-probe
```

**→ 進む条件**: `status=pulled`。`serve check` の表で、`gate_image_digest` の行が 2 台とも通る。

この時点の `serve check` そのものは、**終了コード 1 になる**。重みの照合の記録がまだないので、
`gate_weights_verified` が断るからである (それが準備 4 の仕事)。見るのは、8 つの関門の表のうち、
`gate_image_digest` の行である。

**✋ 止める条件**: 空きが足りない (1)、取得のあとにダイジェストが合わない (2)。

### 準備 3. ライセンスの表記 (`serve image-licenses`)

**前提**: 準備 2 が済んでいる (head にイメージがある)。

⚠ 計測者に尋ねる (状態を変える。読み取りのコンテナを 1 つ起こして、読み終えたら消す)。

```bash
uv run serve image-licenses p1-image-licenses --yes
uv run serve image-licenses p1-image-hf-version --yes   # 取得の道具 hf の有無と版 (同じ仕組みで hf version を 1 回流す)
```

- 標準出力の `key=value` のあと、空の行を 1 つ置いて、読み取った表記の本文がそのまま出る
- vLLM と、ベースのイメージのそれぞれについて、名前、版、入手先、ライセンス、用途を
  `LICENSES.md` に足す (要件 3.8)
- イメージの中に、重みの取得の道具 (`hf`) があるかも、ここで確かめる (research.md の 14b)。
  2 つめの構成 `p1-image-hf-version` は、`--entrypoint hf` で `hf version` を流すだけの
  `inspect` の構成で、版が出れば道具がある。なければ、ホストに入れずに ⚠ 計測者に尋ねる

**→ 進む条件**: `LICENSES.md` にイメージの行がある。

**✋ 止める条件**: ライセンスが MIT、Apache-2.0、BSD 系のどれでもない、または表記がない
(要件 3.9)。採用せずに、除いた理由を `docs/decisions/0002-*.md` と `LICENSES.md` に書いて、
⚠ 計測者に尋ねる。

### 準備 4. 段 0 に要るファイルの取得 (`serve fetch --probe-files`)

**前提**: 準備 2 が済んでいる。`serving/weights/RedHatAI__GLM-5.3-Flash-NVFP4.manifest.json`
がコミットされている (なければ
`uv run serve manifest RedHatAI/GLM-5.3-Flash-NVFP4 <40 桁の revision>` で作る。Spark に
触らない)。

手元で変換した重み (K2 の FP8 など) のマニフェストは、Hub から作らず、変換の道具が書いた
`manifest.json` を
`uv run serve derived-import --name <名前> --commit <40 桁の commit> <manifest.json>...`
で取り込んで、`serving/weights/<名前>.manifest.json` に書く (Spark に触らない。2 台ぶんを
渡すと、`generated_at` を除いた一致を確かめる)。手順は
[K2 の派生の重みの手順書](k2-derived-weights-procedure.md) の §2 にある。

段 0 は、重みの本体 (184.3 GiB) を 1 バイトも落とさない。要るのは、設定とトークナイザ
(約 37 MiB) だけである。

⚠ 計測者に尋ねる (状態を変える)。

```bash
uv run serve fetch p1-fetch-nvfp4-probe --probe-files --yes
uv run serve verify probe-pinned --probe-files --yes
uv run serve check probe-pinned
```

- `--probe-files` は、**照合の範囲** (safetensors を除いたファイル) を選ぶだけで、コンテナの
  引数は変えない。取得そのものは、専用の構成 (`p1-fetch-nvfp4-probe`) が `*.safetensors` を
  除いている
- 照合の結果は、Spark の `state/` に記録として残る。`probe-pinned` の重みの関門は、その記録を
  読み直す

**→ 進む条件**: `status=fetched` と `status=verified`、`mismatched=` が空。`serve check
probe-pinned` で 8 つの関門がすべて通る。

**✋ 止める条件**: 合わないファイルがある (2)。名前が出るので `not-working.md` に書く。
**黙って取り直さない。**

## 段 0: `probe-pinned` — 固定したイメージで、1 台の縮小の確認 (要件 5.1〜5.5)

**前提**: 準備 1〜4 が済んでいる。`attempts.md` を読んだ。head の GPU が空いている
(`serve check probe-pinned` の `gate_gpu_idle` が通る)。

この段が、P1 全体の分岐点である。重みを取得する前に、いちばんの懸念 (このモデルの
アテンションの形を、この GPU で受け付ける計算の部品があるか) に当たる。層を 45 から 4 に
減らし、重みを乱数 (`--load-format dummy`) にして、同じ経路を踏む。

⚠ 計測者に尋ねる (状態を変える)。

```bash
uv run serve probe probe-pinned --timeout 45m --yes
```

- 構成の `ready_timeout_s` は 600 秒だが、**見積もり**である。初回は JIT の温めがあるので、
  `--timeout` でその回だけ延ばす
- どの結果でも、道具が記録を回収してから、必ずコンテナを止めて消す。回収先は
  `serving/var/<日時>-start-probe-pinned/` である (構成の名前で見分ける)

**終了コードの読み方 (この段だけ、`serve start` と違う)**

| 値 | `status` | 意味 | 次 |
|---|---|---|---|
| 0 | `ready` | 起動できた | → 関門 A へ (段 1 は飛ばす) |
| 2 | `failed` | コンテナが終了した (知っている失敗、または `UNCLASSIFIED`) | → 段 1 へ |
| 1 | `inconclusive` | 判定が出なかった | ✋ `detail` の先頭を読む (下の表) |

**`inconclusive` (終了コード 1) を、一律に「段 2 へ進む」と読まない。**

| `detail` の先頭 | 何が起きたか | すること |
|---|---|---|
| 「関門が断ったので、縮小の確認を始めなかった」 | GPU が塞がっている、置き場所がない、照合の記録がない、など | ✋ 直して、もう一度打つ (段を進めない) |
| 「関門が断った」以外 (時間切れ、起こす途中で届かなかった、待っても直らない食い違い) | 判定が出ていない | `attempts.md` に書き、`--timeout` を延ばしてもう一度か、⚠ 計測者に尋ねる |

- 時間切れでも、`--hf-overrides` で減らした層が浅すぎて `pe_dim` の assert に**到達しない**
  ことがある。そのときは、層を増やした `--hf-overrides` に直して試す (要件 5.5。構成の値を
  変えるので `attempts.md` に書く)
- 縮小の確認そのものができないと分かった場合は、そのことを記録して、関門 A に進む (要件 5.5)

**→ 進む条件**: 終了コード 0 (`ready`)。短い要求が 1 つ返り切った (`reply.stop_reason` が出る)。
**中身のない重みなので、応答が意味の通る文であることは、ここでは確かめない** (段 2 に委ねる。
要件 5.4)。

**✋ 止める条件**: 終了コード 2 (`failed`) → 段 1 へ。終了コード 1 (`inconclusive`) で、
`detail` の先頭が「関門が断ったので」で始まる → 直してやり直す (段を進めない)。

**読み取ること (要件 5.2。標準出力の `observation.*`)**: `attention_backend` と
`attention_candidates`、`moe_backend`、`kv_cache_tokens` / `kv_cache_gib`、`model_loading_s` /
`engine_init_s`、`speculative_config_seen` (偽であること)、`known_failure`。版は `/version`
から読む。

**記録**:

- `attempts.md` に 1 行 (段 0、`probe-pinned`、結果、止まった場所、次に進む理由、
  `serving/var/…` の置き場所)
- 失敗なら `not-working.md` の 1 件目 (`known_failure` と、誤りの文面の前後)
- 上流の issue に当たる現象を再現したら、再現の事実と条件を書き、⚠ 計測者に尋ねる (報告するか)

## 段 1: `probe-nightly` — より新しい公式のイメージで、同じ確認をする (2026-09-22 に作った (段 0 の失敗を受けて))

**前提**: 段 0 が `failed` (終了コード 2) だった。`attempts.md` を読んだ。

**作った構成 (`probe-nightly`)**: 段 0 (`probe-pinned`) と同じ形で、**イメージのダイジェスト
だけ**を、2026-09-22 時点の上流の公式の nightly (commit
`0961bbae2894d574be790d219651824eb199318e`) に固定した構成である。`configs.toml` の段 0 の節を
写し、名前とイメージの節だけを変えた。

作ったときの決まり:

1. 値は、**6.2 と同じクリーンルームの決まり**で書いた。設定の 1 つ 1 つに、値、理由 (`why`)、
   根拠 (`source` + 原文の `quote`、または実測の `measured`) を添えている。公式の手引きの
   **コマンドを写さず**、道具の一次の資料を引いている (要件 11.2)
2. 第三者のレシピ、ブログ、フォーラムを開いていない (要件 11.3)。モデルカードの本文も開いて
   いない
3. **この会話の文脈を持たない新しい作業者**に、research.md と公式の資料だけを渡して書かせた
   (要件 11.4)。レビューで、根拠に辿れない設定を差し戻す (要件 11.5)
4. nightly はまだ採用の候補ではなく、段 1 の確認に使うだけである (`hf-version` /
   `image-licenses` の構成は作っていない)。採用するときは、準備 2〜3 (tasks.md 7.1 と同じ
   手順) でイメージの中の表記を読み直す
5. イメージのダイジェストは決まっているので、⚠ 計測者に尋ねる (取得は状態を変える)

```bash
uv run serve pull-image probe-nightly --yes
uv run serve probe probe-nightly --timeout 45m --yes
```

**→ 進む条件**: 終了コード 0 (`ready`) → 関門 A へ。

**✋ 止める条件**: **2 つのイメージで、同じ場所で失敗し、その原因が重みの形式にも並列の
取り方にも依らないと、起動の記録と上流のソースから示せる場合は、重みの取得に進まずに、
下の「打ち切りの手順」へ** (要件 8.7)。

**記録**: `attempts.md` に 1 行 (段 1、`probe-nightly`、イメージのダイジェスト、結果)。
`not-working.md` の該当の件に、2 つのイメージで同じかどうかを足す。`LICENSES.md` に、新しい
イメージの行を足す。

## 関門 A: 通信の確認 (要件 4.1〜4.7)

**前提**: 準備 1〜2 が済んでいる。段 0 (と、通った場合は段 1) の結果が `attempts.md` にある。

**`serve netcheck bandwidth` と `serve netcheck sanity` が通らなければ、2 台での起動に進まない**
(要件 4.7)。また、`nodes.toml` の直結の 3 つ (`fabric_addr`、`fabric_ifname`、
`fabric_measured`) が空のままでは、**2 台の構成 (`p1-nvfp4-tp2`、`netcheck-bandwidth`、
`netcheck-sanity`) は読み込めない**。見込みの値を書かず、`serve netcheck links` の実測で埋める。

### A.1 インターフェースを読む (読み取りだけ)

```bash
uv run serve netcheck links
```

- 2 台の、インターフェースの名前、状態、MTU、速さ、アドレス、RoCE のデバイスとの対応を読み、
  つながっている直結のインターフェースの数からケーブルの本数を判断する (2 つ → 1 本、
  4 つ → 2 本)
- `cable_count.<役割>` が空 (わからない) のときは、`detail.<役割>` に理由が出る。入っていない
  道具があれば `tools_missing.<役割>` に出る (**入れない**。要件 2.7)
- この道具は、どちらのインターフェースを `fabric_ifname` にするかを決めない (候補を示すだけ)

⚠ 計測者に尋ねる: **どちらの直結のインターフェースを使うか**と、要約を公開の場所
(`docs/results/<日付>-netcheck-links.md`) に書くかどうか。書いてから、`nodes.toml` の
直結の 3 つを、その要約を根拠 (`fabric_measured`) にして埋める。

- ケーブルの本数が PLAN.md のハードウェアの記述と食い違う場合は、PLAN.md を実測に合わせて
  訂正する (要件 4.2)

**→ 進む条件**: `serve check netcheck-sanity` が、**直結の値がないという理由では断られない**
(構成を読み込めて、8 つの関門の結果を並べる)。埋める前は、構成の読み込みの段で、6 つの項目の
名前を並べて終了コード 1 になる。

### A.2 帯域を測る

⚠ 計測者に尋ねる (状態を変える)。

```bash
uv run serve netcheck bandwidth netcheck-bandwidth --yes
```

- 最小の設定で、1 MiB から 1 GiB までの all-reduce を測り、`algbw` と `busbw` を出す
- `network.<役割>` が `IB` であること (高速の直結の経路が使われたこと) を見る
- `comparison=` に、NVIDIA の公表の値との比べが出る。**道具が違うので、合否には入れない**

**→ 進む条件**: `status=passed` (終了コード 0)。

**✋ 止める条件**: `status=failed` (2)。ふつうのネットワークの経路に落ちた
(`network.<役割>` が `Socket`)、記録が読めなかった、`busbw > 0` の大きさが 1 つもない。
`not-working.md` に書き、A.4 の A/B に進む。

**2026-09-22 の実測**: 最小の設定では `NET/IB : No device found.` で Socket に落ちた (16 Gbps)。
docker の設定 (`--device /dev/infiniband`) は `serve netcheck ab --env` では足せないので、
`netcheck-bandwidth` を写して、その 1 つを実測の根拠 (`measured`) つきで足した
**`netcheck-bandwidth-ib`** を作り、同じ計測を流す (A/B の B に当たる):

⚠ 計測者に尋ねる (状態を変える)。

```bash
uv run serve netcheck bandwidth netcheck-bandwidth-ib --yes
```

### A.3 事前の確認を流す

⚠ 計測者に尋ねる (状態を変える)。

```bash
uv run serve netcheck sanity netcheck-sanity --yes
```

- 推論サーバーの公式の資料が示す 4 段 (PyTorch の NCCL、PyTorch の GLOO、vLLM の NCCL、
  CUDA グラフの中の vLLM の NCCL) を流す。`stage.<番号>.passed` を 4 つとも見る
- 4 段目が、2 台の DGX Spark で報告されたデッドロックの場所そのものである

**→ 進む条件**: `status=passed`、4 段すべて `passed=True`、`nodes_seen` が 2 台ぶん。

**✋ 止める条件**: 1 段でも通らない (2)。`not-working.md` に、どの段で、どの台で止まったかを
書く。**2 台での起動に進まない。**

### A.4 設定を足す A/B (A.2 か A.3 が通らなかったとき、または足す候補があるとき)

⚠ 計測者に尋ねる (状態を変える。6 回ぶん 12 個のコンテナを 1 つの計画にして、了承は 1 回)。

```bash
uv run serve netcheck ab netcheck-bandwidth \
  --env NCCL_SOCKET_IFNAME==<管理の側のインターフェースの名前> --repeat 3 --yes
```

- `NCCL_SOCKET_IFNAME` の先頭の `=` は、前方一致ではなく完全一致にするための印である
  (NCCL の書式。`configs.toml` の同じ設定の注と同じ)。**`=` を 1 つに直さない**

- 最小の設定 (腕 A) と、足した設定 (腕 B) を、交互に 3 回ずつ流す。回ごとに、コンテナの名前と
  ラベルで、どの腕の何回目かを区別する
- **採否は、速さだけで決める**: 腕 B の 3 回の最小が、腕 A の 3 回の最大を上回ったときだけ
  「採用できる」。等しければ採用しない
- `route.<腕>.<回>.<役割>` に、回ごと・台ごとの経路が出る。経路を確かめられなかった回が
  あれば、値を使わずに止まる
- 足す環境変数は、**全ノードに同じ値**が渡る (置き換えの印は埋めない)。2 台で名前が違う値は、
  A/B にできない。`NCCL_DEBUG` と `NCCL_DEBUG_FILE` は `--env` に書けない
- コンテナの中で高速の経路が使えない場合の候補は、docker の設定 (`--device /dev/infiniband`、
  `--cap-add SYS_NICE`、`--cap-add IPC_LOCK`) である。これは `--env` では渡せないので、
  `netcheck-bandwidth` を写して設定を足した**別の名前の構成**を作り (名前は作るときに決める。
  値は 6.2 と同じ決まりで、出典つき)、`serve netcheck bandwidth` を両方流して比べる
- 採用したものだけを、実測の根拠 (`measured = "docs/results/…"`) を添えて構成に書く (要件 4.5)

⚠ 計測者に尋ねる: 恒久的な設定の変更が要るとわかった場合は、**変更せずに**記録して尋ねる
(要件 2.8)。

**記録**: `docs/decisions/0003-*.md` に、測った値、使われた経路、採否と理由。実測の要約は
⚠ 計測者の指示があったときだけ `docs/results/` に書く。`attempts.md` に 1 行。

## 関門 B: 第一の候補の重みの取得 (要件 3.4、3.5、8.1)

**前提**: 関門 A が通っている。マニフェストがコミットされている。2 台のディスクの空きが
足りる (184.3 GiB × 2 台)。

⚠ 計測者に尋ねる (状態を変える。長時間かかる)。

```bash
uv run serve fetch p1-fetch-nvfp4 --yes
uv run serve verify p1-fetch-nvfp4 --yes
```

- 2 台で並行に、固定した版を取得する。ホストに何も入れず、トークンを渡さない
- ssh が途中で切れても取得は続く。もう一度打つと、動いている取得を見つけて待つ
  (二重に取得しない)。**待ちの途中で中断しても、取得のコンテナは止めない** (止めるのは
  `serve stop`)
- 構成の `ready_timeout_s` は 28800 秒 (8 時間) の**見積もり**である。実測で直す
- 取得が動いている間の `serve verify` は、照合を始めずに断る

**→ 進む条件**: `status=fetched`、`mismatched=` が空。`serve check p1-nvfp4-tp2` で 8 つの
関門がすべて通る。

**✋ 止める条件**: 合わないファイルがある (2)。名前を示して止まり、**黙って取り直さない**。
`not-working.md` に書く。

**記録**: `LICENSES.md` に重みの行 (名前、版、入手先、ライセンス、用途。量子化版の
ライセンスは、front matter の申告による継承であることも書く)。`attempts.md` に 1 行。

## 段 2: `p1-nvfp4-tp2` — 2 台で TP=2 を起動し、計測する (要件 6、7、9)

**前提**: 関門 A と関門 B が通っている。`attempts.md` を読んだ。2 台の GPU が空いている。

### 2.1 最初の起動だけ、NCCL の記録を採る

design.md は「最初の起動だけ、NCCL の記録の 3 つの変数」を入れると定める。**足すことと外す
ことの両方が、起動し直しを伴う状態を変える操作である。**

⚠ 計測者に尋ねる (3 つを足すこと)。足したら、`configs.toml` の `p1-nvfp4-tp2` の `env` に、
次の 3 つを書く。

| 名前 | 値 |
|---|---|
| `NCCL_DEBUG` | `INFO` |
| `NCCL_DEBUG_SUBSYS` | `INIT,NET` |
| `NCCL_DEBUG_FILE` | `/logs/nccl.%h.%p.log` |

- `source` と `quote` は、**`netcheck-bandwidth` の同じ名前の設定から、そのまま写す**
  (出典は NCCL の環境変数の頁。`netcheck-bandwidth` と同じ根拠で足す)。`why` には、この起動で
  何を読みたいか (経路の確定の行と、デバイスの初期化) を書く
- `NCCL_DEBUG_SUBSYS` の値は、`netcheck-bandwidth` (`INIT,BOOTSTRAP,ENV,NET,GRAPH`) より
  狭い `INIT,NET` にする (この起動で読みたいのは、経路の確定とデバイスの初期化だけである)
- 書き先の `/logs` は、`p1-nvfp4-tp2` が読み書きできる形で結び付けている置き場所である
- この 3 つは、コンテナに渡る引数を変えるので、**`config-sha256` が変わる**

```bash
uv run serve start p1-nvfp4-tp2 --timeout 3h --yes
uv run serve logs p1-nvfp4-tp2
```

- `ready_timeout_s` は 1800 秒だが、**見積もり**である。初回は約 92 GiB のロードと JIT が
  あるので、`--timeout` でその回だけ延ばす
- 回収した記録のうち、`serving/var/<日時>-logs-p1-nvfp4-tp2/<役割>/logs/nccl.*.log` の
  `Using network …` の行を、2 台ぶん読む。**`netcheck` の合格の条件と同じで、
  `network == "IB"` (高速の直結の経路) であることを確かめる**

**→ 進む条件**: 2 台とも `IB` である。

**✋ 止める条件**: どちらかが `Socket`、または `Using network` の行が読めない。計測に進まず、
`not-working.md` に書いて、関門 A の A.4 (docker の設定を足す A/B) に戻る。

### 2.2 記録の 3 つを外して、計測の形に戻す

⚠ 計測者に尋ねる (外すことも状態を変える)。

```bash
uv run serve logs p1-nvfp4-tp2   # 2.1 で回収していなければ
uv run serve stop --yes
uv run serve start p1-nvfp4-tp2 --timeout 3h --yes
```

- 3 つを外すと `config-sha256` が変わるので、動いているコンテナとは中身が違うことになり、
  `serve start` は「名前が同じで中身が違う」として断る (終了コード 1)。**`stop` → `start` の
  順で入れ替える**
- **`serve stop` は記録を回収しない。**先に `serve logs` で回収する (了承の前に、道具も
  そう見せる)
- `serve stop` は、2 台の**すべての自分のコンテナ**を、`kind` を問わず止めて消し、GPU の
  プロセスが 0 件になるまで最大 60 秒待つ。空かなければ `gpu_not_released` (2) で、残っている
  プロセスの名前とメモリの量を示す (止めには行かない)

### 2.3 読み取って、確かめる (要件 6.3〜6.5)

```bash
uv run serve status
uv run serve smoke p1-nvfp4-tp2
```

- `serve start` の標準出力から: `observation.attention_backend` と `attention_candidates`、
  `moe_backend`、`kv_cache_tokens` / `kv_cache_gib`、`model_loading_s` / `engine_init_s`、
  `speculative_config_seen` (偽であること)
- `serve status` から: `max_model_len` (対応する入力の長さの上限)、`served_model`、
  `running_requests` / `waiting_requests`、`node.<役割>.fabric_link_up`
- `serve smoke` は、英語と日本語の短い要求を 1 つずつ送る。**応答の本文は標準エラーに出るだけ
  で、どのファイルにも保存されない。**意味の通る文かどうかは、人が画面で見て判断する
- `serve status` は、読めなかった台があれば終了コード 1 になる。「何も動いていない」と読み
  違えない

**→ 進む条件**: `status=answered` で、2 つの応答が意味の通る文である。

**✋ 止める条件**: 起動が失敗した (2)、応答が意味をなさない。`not-working.md` に、失敗した
場所、誤りの文面、**1 台での結果 (段 0) との違い**を書き、`attempts.md` に行を足して、段 3 に
進む (要件 6.6)。

**統合の作業**: 実測した所要で、構成の `ready_timeout_s` を直す。この項目はコンテナに渡る
引数に現れないので、`config-sha256` は変わらず、動いているサーバーを止めずに直せる。

**記録**: `attempts.md` に 1 行。KV キャッシュの大きさと brief.md の見積もりの食い違いは、
理由とともに `docs/decisions/0002-*.md` に書く (要件 6.4)。

### 2.4 `bench` の対象サーバーを定義して、計測の前の確認を流す (要件 7.1、7.2)

`bench/config/targets.toml` に、構成がわかる名前で足す。

```toml
[targets.vllm-nvfp4-tp2]
base_url = "http://10.0.1.60:8000"
model = "glm-5-3-flash"
max_context_tokens = 163840
notes = "P1 で自分たちが立てた対象サーバー (構成 p1-nvfp4-tp2)"
```

GPU を使ってよいプロセスの名前は、**動いているあいだに `serve check p1-nvfp4-tp2` を流す**と
読める (`gate_gpu_idle` が通らず、プロセスの名前とメモリの量を示す。読み取りだけ)。その名前を、
計測の前の確認の 2 番目の引数に使う。

```bash
cd <リポジトリの最上位>
scripts/spark-precheck.sh http://10.0.1.60:8000 '<実測したプロセスの名前>'
```

**計測のたびに、直前にこれを流し、満たさなければ計測を始めない** (要件 7.2)。

### 2.5 1 トークンあたりの文字数を測り直す (要件 7.3)

```bash
cd bench
uv run bench calibrate --target vllm-nvfp4-tp2 --profile quick
```

- 標準出力は、`bench/config/profiles.toml` に貼れる TOML の断片である
- 計測の設定の値と食い違う場合は、**どちらの値で測ったか**を記録する
- 引数は `bench/README.md` の「`bench calibrate`」の節に従う (`--target` と `--profile` だけ)

### 2.6 5 つのまとまりを `quick` で一通り流す (要件 7.4〜7.6)

**まとまりごとに、別のランで流す** (1 つが止まっても、残りを続けられるように)。

```bash
cd bench
uv run bench run --target vllm-nvfp4-tp2 --profile quick --suite decode
uv run bench run --target vllm-nvfp4-tp2 --profile quick --suite prefill
uv run bench run --target vllm-nvfp4-tp2 --profile quick --suite concurrency
uv run bench run --target vllm-nvfp4-tp2 --profile quick --suite quality
uv run bench run --target vllm-nvfp4-tp2 --profile quick --suite agent
uv run bench summarize <ランの識別子>   # 要約を作り直したいときだけ
```

- `bench run` は、終わりに要約を作る。`bench summarize` は、生データから**作り直す**ときに使う
- ほかの引数 (`--set`、`--data-cache`、`--no-download`) が要るときは、`bench/README.md` を見る。
  **実在しない引数を書かない**
- 同時処理のまとまりでは、日本語の出力の壊れの疑いの印の数を、同時の本数ごとに記録する
  (要件 7.6)
- 生データは `results/` (git の管理の外) に残る。⚠ 計測者に尋ねる: 公開する場所に要約を写すか
  (写すのは要約だけ。`bench publish <ランの識別子>` を使う。要件 7.8)

**→ 進む条件**: 5 つのまとまりのそれぞれに、要約があるか、止まった場所と理由の記録がある。

**✋ 止める条件**: あるまとまりが連続の失敗で止まった、または一部の条件が飛ばされた。
`not-working.md` に場所と理由を書き、**残りのまとまりの計測は続ける** (要件 7.5)。

上流の issue に当たる現象を再現したら、⚠ 計測者に尋ねる (上流に報告するか。要件 10.4)。

### 2.7 2 時間の連続の負荷を、外から見張る (要件 7.7)

⚠ 計測者に尋ねる: 2 台を 2 時間占有するので、始める前に声をかける。

```bash
# 端末 1 (読み取りだけ。何も止めない)
cd serving && uv run serve watch p1-nvfp4-tp2 --duration 2h --interval 30s
# 端末 2 (2.6 の所要を見て、回すまとまりを決める)
cd bench && uv run bench run --target vllm-nvfp4-tp2 --profile quick --suite decode
```

- `serve watch` は、応答の確認、生成のトークンの数、処理中と待ちの要求の数、2 台の GPU の
  使用率を、決めた間隔で読む。3 回続けて応答の確認に失敗したら「応答しない」、5 分の窓で
  トークンが増えず、要求があり、GPU が使われ続けていたら「固まった」
- 終了コードは、出来事なし = 0、**出来事あり = 2**。中断 (130) でも要約は書き出されている

**記録**: 出来事があれば、起きた時刻と前後の観察を `not-working.md` に書く (要件 10.1)。

### 2.8 `/v1/messages` の対応の状況をまとめる (要件 9)

8 つの項目のうち、`bench` の計測の結果から確かめられる 7 つは、**計測ランの ID を根拠として
書く。別の確かめを足さない** (要件 9.3)。thinking の深さの渡し方だけ、専用の道具で確かめる。

```bash
cd serving
uv run serve thinking p1-nvfp4-tp2
```

- 同じ入力、温度 0 で、5 通り (何も渡さない、`/v1/messages` の本来の口で低く、チャット
  テンプレートの項目で低く、効かないはずの値、過去の thinking を消す指定) を、3 回ずつ送る
- 記録するのは、thinking のブロックの文字数、入力と出力のトークンの数、終わりの理由だけで、
  **本文は保存しない**
- `effective=` が空 (判定できなかった) のときは終了コード 1 になる。4xx が返るなら、過去の
  thinking のブロックの形が受け付けられていない疑いがあるので、そのことを記録する

**記録**: `docs/decisions/0004-*.md` に、8 つの項目のそれぞれの判定と根拠 (ランの ID)。
扱えない項目が takt の使い方に効く場合は、変換層を自分たちで書くか、上流に返すか、そのまま
受け入れるかの案を、理由とともに書く。`bench` の側の変更は、`bench-harness` の仕様の変更として
扱い、ここでは行わない (要件 9.4、9.5)。

## 段 3: `p1-nvfp4-tp2-x<連番>` — 明示の指定を足して試す (この段で作る)

**前提**: 段 2 で、起動できなかった、または計測を流し切れなかった。`attempts.md` を読んだ。

**この段で作る構成**: 段 2 の失敗の種類 (または、計測を流し切れなかった理由) に合う指定を
下の候補から選び、**足した指定ごとに、別の名前の構成**として書く (連番は 1 から)。

| 足す候補 | いつ選ぶか |
|---|---|
| `--block-size 256` | ブロックの大きさの assert で止まった (research.md の 4) |
| `--enforce-eager` | CUDA グラフの取り込みで止まった、または対応するカーネルがない |
| `--moe-backend <値>` | エキスパートの計算の部品が選べない、または全滅した |
| `--gpu-memory-utilization` の引き下げ | 起動時のメモリの検査で止まった、KV が確保できない |

作るときの決まりは、段 1 と同じ (値、`why`、根拠。公式の手引きのコマンドを写さない。文脈を
切り離した作業者に書かせる。レビューで差し戻す)。

⚠ 計測者に尋ねる (起動は状態を変える)。

```bash
uv run serve check <段 3 の構成>
uv run serve start <段 3 の構成> --timeout 3h --yes
uv run serve smoke <段 3 の構成>
```

- **同じ失敗を、設定を変えずに繰り返さない** (要件 8.3)。試す前に `attempts.md` を見る
- 動いたら、それで失うもの (速さ、メモリ) を `bench` で測り、⚠ 計測者の指示で
  `docs/results/` に書いて、その構成の設定の根拠 (`measured`) にする (要件 8.4)

**→ 進む条件**: 起動でき、`serve smoke` が意味の通る応答を返した → 2.4 の計測に戻る。

**✋ 止める条件**: 候補をすべて試しても届かない。⚠ 計測者に尋ねる (段 4 に進むか)。
了承されなければ、下の「打ち切りの手順」へ。

**記録**: `attempts.md` に、足した指定ごとに 1 行。`not-working.md` に、失敗した場所と誤りの
文面。動いた構成があれば、失うものの実測。

## 段 4: `p1-w4a16-tp2` — 第二の候補の重みで試す (この段で作る)

**前提**: 段 3 でも届かなかった。**⚠ 計測者に尋ねる: この段に進んでよいか** (要件 8.8)。
了承されなければ、下の「打ち切りの手順」へ。

**この段で作る構成**: 第二の候補の重みの構成と、その取得の構成を足す (取得の構成の名前は、
作るときに決める。`p1-fetch-nvfp4` と同じ形で書く)。

作るときの決まり:

1. **その重みのモデルカードの本文を開かない** (要件 8.8、11.3)。そのモデルカードは
   DGX Spark 2 台の起動の手順そのもので、開かないと定めた第三者のレシピに当たる
2. 機械で読める設定のファイル (`config.json` など) と、上流のソースだけから構成を導く
3. チャットテンプレートがベンダの現行版と違うので、**現行版を明示する**
4. 値の書き方は 6.2 と同じ (根拠つき、文脈を切り離した作業者、レビューで差し戻す)

```bash
uv run serve manifest <第二の候補の repo> <40 桁の revision>   # Mac だけ。モデルカードを取得しない
uv run serve pull-image <段 4 の取得の構成> --yes               # 同じイメージなら要らない
uv run serve fetch <段 4 の取得の構成> --yes
uv run serve verify <段 4 の取得の構成> --yes
uv run serve start <段 4 の構成> --timeout 3h --yes
uv run serve smoke <段 4 の構成>
```

**→ 進む条件**: 起動でき、意味の通る応答が返る → 2.4 の計測に戻る。

**✋ 止める条件**: これでも届かない → 下の「打ち切りの手順」へ。

**記録**: マニフェストをコミットする。`LICENSES.md` に重みの行を足す。`attempts.md` に 1 行。
段 2 と同じ読み取り (要件 6.3〜6.5) を行って記録する。

## 打ち切りの手順 (要件 8.5、8.7)

次のどちらかに当たったら、試行を打ち切る。

- 段 0 と段 1 が、**同じ場所で失敗**し、その原因が重みの形式にも並列の取り方にも依らないと、
  起動の記録と上流のソースから示せる (要件 8.7)。このときは、**重みの取得に進まない**
- 決めた順序の構成をすべて試しても、パッチなしでは起動できない、または計測を流し切れない
  (要件 8.5)

すること:

1. 止まった場所、試したことの一覧、直すのに要る最小の変更の候補 (上流の未マージの変更を
   含む) を、`docs/decisions/0005-*.md` の下書きにまとめる
2. ⚠ 計測者に尋ねる: **終わりの条件を改めるかどうか**
3. **計測者の判断なしに、推論サーバーにパッチを当てない、イメージを自分でビルドしない**
   (要件 8.6)。道具にも、その経路はない

**記録**: `attempts.md` の最後の行に、打ち切りと理由。`not-working.md` に、残った件。

## 終わり (まとめと、片付け)

1. `docs/decisions/` の 4 本を仕上げる (`0002` イメージと重みの選定、`0003` 通信の設定の採否、
   `0004` `/v1/messages` の不足の扱い、`0005` 終わりの条件)。`0002` の「クリーンルーム」の節に、
   参照した資料の一覧 (要件 11.6) と、構成の値を書いた作業者が文脈を切り離されていたこと
   (要件 11.4) を書く
2. ⚠ 計測者に尋ねる: **2 台での層の分割 (PP=2) を、P1 のうちに測るか、P5 に持ち越すか**
   (要件 8.9)。TP=2 との速さの比較の候補として扱い、代替の構成とはしない
3. ⚠ 計測者に尋ねる: **推論サーバーを止めるか、動かしたままにするか** (要件 7.9)。止めるなら、
   **記録を先に回収する**

```bash
uv run serve logs p1-nvfp4-tp2   # stop は記録を回収しない
uv run serve stop --yes          # ⚠ 了承を得てから
uv run serve status              # 止まったこと、GPU が空いたことを確かめる
```

4. 一覧 (`not-working.md`)、試行の記録 (`attempts.md`)、判断の記録に、**送った内容と応答の
   本文、認証の情報、別の構成の中身が含まれていない**ことを確かめる (要件 10.5)
5. 終わりの条件 (1 つ以上の構成で起動し、`bench` の計測を一通り流せた。または、計測者が改めた
   条件) に照らして、満たしたこと、満たせなかったこと、P2 以降に持ち越すことをまとめる
   (要件 10.6)

## 計測者に尋ねる場所 (一覧)

| いつ | 何を尋ねるか | 要件 |
|---|---|---|
| Spark の状態を変える操作の前 (毎回) | 計画を見せて、流してよいか (Claude が打つときは、会話で了承を得てから `--yes`) | 2.1 |
| 恒久的な設定の変更が要るとわかったとき | 変更せずに記録して、どうするか | 2.8 |
| 計測の生データの要約を、公開する場所に写すとき | 写してよいか (写すのは要約だけ) | 7.8 |
| 計測が終わったあと | 推論サーバーを止めるか、動かしたままにするか | 7.9 |
| 打ち切り | 終わりの条件を改めるかどうか | 8.5 |
| 段 4 (第二の候補の重み) の前 | この段に進んでよいか | 8.8 |
| 起動できたあと | 2 台での層の分割 (PP=2) を、P1 で測るか P5 に持ち越すか | 8.9 |
| 上流の issue に当たる現象を再現したとき | 上流に報告するかどうか | 10.4 |

## 実機でしか決まらない 22 項目の、段への割り当て

research.md の「Open questions that only hardware can answer」を、design.md 「実機での確かめ」
の組分け (安い順) に従って割り当てた表である。「確かめ方」は、どのコマンドの何を見るかを書く。

| 番号 | 問い (要約) | 段 | 確かめ方 (どのコマンドの何を見るか) |
|---|---|---|---|
| 1 | 固定したイメージで `pe_dim must be 64` の assert が出るか (P1 全体の分岐点) | 段 0 | `serve probe probe-pinned` の `observation.known_failure` と、回収した起動の記録の全文 |
| 2 | 選ばれたアテンションの部品、または全滅した理由 | 段 0 | 同じ実行の `observation.attention_backend` / `attention_candidates` |
| 3 | sm_120 向けのバイナリが sm_121 で動くか | 段 0 | 同じ記録に「対応するカーネルがない」の誤りが出るか (`observation.known_failure`) |
| 4 | `--block-size` を指定しないと kpool の assert が出るか | 段 0 | 同じ記録のブロックの大きさの失敗。出たら段 3 の候補にする |
| 5 | 直結リンクの本数、名前、MTU、リンクの速さ | 通信の確認 | `serve netcheck links` の `cable_count.<役割>` と、人が読む報告 |
| 6 | RoCE のデバイスと、インターフェースの名前の対応 | 通信の確認 | 同じ実行の報告 (`ibdev2netdev` / `ibv_devinfo` から読んだ対応) |
| 7 | 最小の設定での 2 台の帯域 (busbw) | 通信の確認 | `serve netcheck bandwidth netcheck-bandwidth` の `sample.<大きさ>.busbw_gbps` |
| 8 | NCCL が RoCE を使っているか、デバイスが束ねられたか | 通信の確認 | 同じ実行の `network.<役割>` と、回収した NCCL の記録 |
| 9 | 事前の確認の 4 段すべてが通るか | 通信の確認 | `serve netcheck sanity netcheck-sanity` の `stage.<番号>.passed` |
| 10 | コンテナの中で `ulimit -l` が無制限になるか | 通信の確認 | 上の 2 つが通るかで判断する (`--ulimit memlock=-1` の効き) |
| 11 | `--device /dev/infiniband` なしで `NET/IB` が出るか | 通信の確認 | `netcheck bandwidth` の `network.<役割>` が `IB` か。出なければ A.4 の A/B |
| 12 | 重みの取得がシンボリックリンクになるか | 取得 | `serve fetch p1-fetch-nvfp4` のあとの `serve verify` が通れば、中身は読める (`sha256sum` はリンクをたどる)。リンクかどうかは、この道具では読めない (`remote` の許可の一覧に、ファイルを並べるコマンドがない)。配布の方式に効くので、要れば ⚠ 計測者に尋ねる |
| 13 | MTP のファイルが index に載っているか | 解決済み | 紙で確認済み (research.md §b-6)。載っている。取得は省けない |
| 14 | チャットテンプレートが読む変数 | 解決済み | 紙で確認済み (research.md §g-2)。`reasoning_effort` と `clear_thinking` だけ |
| 14b | `hf` (取得の道具) を Spark でどう動かすか | 取得 | `serve image-licenses p1-image-licenses` でイメージの中を見て、`serve fetch p1-fetch-nvfp4-probe` が通るか |
| 14c | `NCCL_SOCKET_IFNAME` は直結側と管理側のどちらが速いか | 通信の確認 | `serve netcheck ab netcheck-bandwidth --env NCCL_SOCKET_IFNAME==…` の採否。先頭の `=` は、前方一致ではなく完全一致にするための印である (NCCL の書式)。**`=` を 1 つに直さない** |
| 14d | イメージの NCCL が sm_121 で動くか | 通信の確認 | `serve netcheck sanity netcheck-sanity` の 1 段目 (`stage.1.passed`) |
| 14e | TP=2 と PP=2 のどちらが速いか | あとの比較 | 起動できたあと、⚠ 計測者に尋ねてから、同じ `bench` の条件で比べる (要件 8.9) |
| 15 | 重みのロード時間と、起動全体の所要 | 段 2 | `serve start p1-nvfp4-tp2` の `observation.model_loading_s` / `engine_init_s` (`ready_timeout_s` を直す根拠) |
| 16 | `--max-num-seqs` の既定が実際にいくつになるか | 段 2 | 同じ起動の記録の設定のダンプ (回収した `container.stdout.log`) |
| 17 | KDA の状態と KV の実際の内訳 | 段 2 | 同じ実行の `observation.kv_cache_tokens` / `kv_cache_gib` と、`serve status` の `max_model_len` |
| 18 | `--gpu-memory-utilization` をどこまで上げられるか | 段 2 | 値を上げた構成で起こし直し、`observation.kv_cache_gib` の増分を見る |
| 19 | thinking の深さが実際に変わるか、過去の thinking の掃除が入力を減らすか | bench と確かめ | `serve thinking p1-nvfp4-tp2` の `effective=` と、5 通りの結果 |
| 20 | 2 時間の連続の負荷で固まるか | bench と確かめ | `serve watch p1-nvfp4-tp2 --duration 2h` の `event.<番号>.finding` |
| 21 | 同時実行で日本語が壊れるか | bench と確かめ | `bench run --suite concurrency` の要約の、壊れの疑いの印の数 (同時の本数ごと) |
| 22 | イメージの中のライセンスの表記 | 取得 | `serve image-licenses p1-image-licenses` の出力 (`LICENSES.md` に写す) |
