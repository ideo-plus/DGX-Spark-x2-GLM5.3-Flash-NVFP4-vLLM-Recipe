# Research & Design Decisions

## Summary

- **Feature**: `bench-harness`
- **Discovery Scope**: New Feature (新規。既存のコードはない)
- **調査の方法**: クリーンルームの制約に従い、公式の文書、上流の vLLM のソース・issue・PR、Anthropic の公式文書、公式のモデルカード、各部品の公式リポジトリだけを参照した。ほかの DGX Spark 向けレシピは開いていない。調査日は 2026-09-19。基準にした vLLM は最新リリース v0.29.0 と `main`
- **Key Findings**:
  - 上流の vLLM は `/v1/messages` を既定で公開している (v0.11.1 以降、フラグ不要)。`/v1/messages/count_tokens` は v0.17.0 以降。ただし **`/v1/messages` では出力の長さを固定できない** (`ignore_eos` / `min_tokens` を本文に足しても黙って捨てられる)
  - トークン数は、ストリームの `message_start` (入力) と `message_delta` (入力と出力) の両方に載る。サーバーが `--enable-prompt-tokens-details` 付きで起動していると、`input_tokens` は「キャッシュ分を引いた値」に変わる。**入力の全長は `input_tokens + cache_read_input_tokens + cache_creation_input_tokens` で復元する必要がある**
  - vLLM は `ping` イベントを送らない。ストリームが止まったことは、チャンクの間隔の制限時間で検出するしかない
  - GLM のツール呼び出しのパーサー (`glm47`) は、途中で切れたツール呼び出しを **本文に漏らさず、丸ごと落とす**。クライアントから見ると「呼ばなかった」か「空の応答」になる
  - `/metrics` の名前は新しい名前だけを前提にしてよい (`vllm:kv_cache_usage_perc`、`vllm:prefix_cache_*`)。古い名前は v0.12.0 で削除済み
  - 依存する部品の候補はすべて MIT / BSD / Apache-2.0 / PSF。公開の課題では、LiveCodeBench のデータ (ライセンスが `cc` としか書かれていない)、ToolBench (研究・教育目的のみの記述)、Paul Graham のエッセイを使う干し草 (権利の表記なし) は使えない
  - GLM-5.3-Flash のモデルカードに載っている公開値は、エージェント系のベンチマーク (DeepSWE、Terminal-Bench 2.1 など) だけで、HumanEval や BFCL の値はない。**「量子化前の公開値と同じ課題で比べる」は、そのままでは成り立たない**
  - 作業用の Mac に入っているコンテナの実行環境は Docker Desktop だけ。Docker Desktop は、従業員 250 人以上または年商 1,000 万ドル以上の会社では有償契約が要る

## Research Log

### vLLM の `/v1/messages` の対応状況

- **Context**: takt からの呼び方を `/v1/messages` に決めた。計測もこの形式で行うので、上流がどこまで扱えるかが設計の前提になる
- **Sources Consulted**:
  - https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/anthropic/api_router.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/anthropic/serving.py
  - https://github.com/vllm-project/vllm/pull/22627 (導入の PR、2025-10-22 にマージ)
  - https://docs.vllm.ai/en/latest/serving/online_serving/
  - https://platform.claude.com/docs/en/build-with-claude/streaming
- **Findings**:
  - 経路は `POST /v1/messages` と `POST /v1/messages/count_tokens` の 2 本。無条件に登録される。モデルが生成に対応していないときは 501
  - `count_tokens` は、チャットテンプレートを実際に展開してトークンを数えるだけで、エンジンを使わない
  - ストリームのイベントは `message_start` / `content_block_start` / `content_block_delta` / `content_block_stop` / `message_delta` / `message_stop` / `error`。delta の種類は `text_delta`、`thinking_delta`、`signature_delta`、`input_json_delta`
  - `ping` は送られない (Anthropic の公式仕様は `ping` が混ざる前提)
  - ストリームの途中の例外は、HTTP のステータスではなく `event: error` で流れる
  - 認証は `Authorization: Bearer` だけ。`x-api-key` は非対応 (open issue #51572)
  - ツール呼び出しのブロックの途中で届いた本文は溜められ、**ツール呼び出しのブロックが終わったあとに別の本文ブロックとして出る** (順序が入れ替わる)
  - エラーの `error.type` は vLLM 独自の名前 (`BadRequestError` など)
- **Implications**:
  - クライアントは、イベントの順序や `ping` の有無に依存しない作りにする
  - 認証の情報は `Authorization: Bearer` と `x-api-key` の両方に付ける (サーバーを差し替えられるようにするため)
  - ブロックの順序を採点に使わない

### トークン数と、キャッシュの効きの見え方

- **Context**: 2.3、3.2、6.7 は、対象サーバーが返したトークン数を正とする
- **Sources Consulted**: `vllm/entrypoints/anthropic/serving.py` L55-91、L527-531、L850-856。`vllm/entrypoints/launchers/cli_args.py` L139-140。issue #45079
- **Findings**:
  - Anthropic の変換の層は、常に `include_usage` と `continuous_usage_stats` を強制する。そのため `message_start.message.usage.input_tokens` が最初のチャンクで得られる
  - `cache_read_input_tokens` / `cache_creation_input_tokens` は、サーバーが `--enable-prompt-tokens-details` 付きで起動しているときだけ出る。しかも `message_delta` にしか載らない
  - そのフラグが付いていると `input_tokens = prompt_tokens - cache_read - cache_creation` に補正される
- **Implications**:
  - 入力の全長は `input_tokens + cache_read + cache_creation` で求める (欠けている項は 0 と見なす)。値は `message_delta` のものを正とし、`message_delta` が来なかったときだけ `message_start` の値を使う
  - キャッシュが効いたかどうかの確認は、`cache_read_input_tokens` があればそれを、なければ `/metrics` の `prefix_cache_hits` の増分を使う。どちらも得られないときは「確認できなかった」と記録する

### 出力の長さの固定

- **Context**: 2.7 は、同じ条件のすべての試行で同じ出力の上限を使うことを求める。生成速度の比較には、出力の長さが揃っているほうがよい
- **Sources Consulted**: `vllm/entrypoints/anthropic/serving.py` の `_build_base_request` (L481-495)、`vllm/entrypoints/openai/chat_completion/protocol.py` L274-275、L703-745
- **Findings**:
  - `/v1/messages` で効くのは `max_tokens`、`stop_sequences`、`temperature`、`top_p`、`top_k`。vLLM 独自に `cache_salt`、`chat_template_kwargs`、`output_config` など
  - `ignore_eos` / `min_tokens` は `/v1/chat/completions` にはあるが、`/v1/messages` の変換では設定されない。要求のモデルは余分な項目を黙って捨てる
- **Implications**: 後述の Decision「出力の長さは指示と上限で揃える」を参照

### ツール呼び出しの扱いと、崩れ方の見え方

- **Context**: 6.3 は、応答を 9 種類に分ける。何がクライアントから観測できるかで、分類の実現性が決まる
- **Sources Consulted**:
  - `vllm/entrypoints/anthropic/serving.py` L378-465、L533-591
  - https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/__init__.py
  - `vllm/parser/abstract_parser.py` L374-381、`vllm/parser/glm47_moe.py`
  - https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/chat_template.jinja
  - issue #55541、#49248、#49412、#56263、#55754、#56868
- **Findings**:
  - `tools` があって `tool_choice` がなければ、`auto` が自動で付く。`auto` を実際に効かせるには、サーバー側に `--enable-auto-tool-choice` と `--tool-call-parser` の両方が要る
  - GLM 向けのパーサーの名前は `glm45` と `glm47` の 2 つだけ (実体は同じ)。GLM-5.3-Flash のチャットテンプレートの記法 (`<tool_call>`、`<arg_key>`、`<arg_value>`) は `glm47` の定数と一致する
  - `glm47` はエンジン型のパーサーで、不完全なツール呼び出しの記法を **落とす**。古い型のパーサーだけが、生の記法を本文に返す
  - 関連する未解決の問題: 強制の `tool_choice` で収束せず `max_tokens` まで走る (#55541)、`<arg_value>` が欠けると引数が黙って空になる (#49248)、長い文脈 (約 37 万トークン) での劣化 (#56868)
- **Implications**:
  - 「呼び出しの記法が本文に漏れた」は、GLM と `glm47` の組み合わせでは起きにくい。実際に多く観測されるのは「呼ばなかった」「空または途中で切れた」「引数が定義に合わない (空の引数)」になる見込み。分類は 9 種類とも実装するが、結果を読むときにこの偏りを念頭に置く
  - 記法の目印 (`<tool_call>` など) は、対象サーバーの定義で差し替えられるようにする。既定値は公式のチャットテンプレートから取る
  - 計測では `tool_choice` を強制しない (`auto` のまま)。#55541 を避けるためと、takt の実際の使い方に合わせるため
  - サーバーの中で落とされた記法を見るには、別の口 (生の補完) が要る。これは P4 の原因の切り分けの範囲で、この仕様には入れない

### 入力の長さの上限と、その見つけ方

- **Context**: 3.6 と 6.9 は、上限に達したら飛ばす、または止めることを求める
- **Sources Consulted**: `vllm/renderers/params.py` L498-508、`vllm/entrypoints/serve/engine/protocol.py` L105-112、`vllm/entrypoints/openai/models/serving.py` L64-72
- **Findings**:
  - `GET /v1/models` の応答に `max_model_len` が含まれる
  - 上限を超えると HTTP 400。メッセージは `This model's maximum context length is {N} tokens. ...`
- **Implications**: 上限は `GET /v1/models` の `max_model_len` → 対象サーバーの定義に書いた値 → 不明、の順で決める。400 のメッセージを正規表現で読むことはしない (サーバーの差し替えで壊れるため)。400 が返ったら、その条件を「上限に達した」として飛ばし、メッセージを理由として残す

### `/metrics` の名前

- **Context**: 7.2 と 7.3
- **Sources Consulted**:
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/metrics.py
  - https://docs.vllm.ai/en/latest/design/metrics/
  - PR #18354 (名前の変更)
- **Findings**:

  | 論理名 | Prometheus の名前 | 型 |
  |---|---|---|
  | 下書きの回数 | `vllm:spec_decode_num_drafts_total` | Counter |
  | 下書きしたトークン | `vllm:spec_decode_num_draft_tokens_total` | Counter |
  | 当たったトークン | `vllm:spec_decode_num_accepted_tokens_total` | Counter |
  | 位置ごとの当たり | `vllm:spec_decode_num_accepted_tokens_per_pos_total` (ラベル `position`) | Counter |
  | キャッシュの問い合わせ | `vllm:prefix_cache_queries_total` | Counter |
  | キャッシュの当たり | `vllm:prefix_cache_hits_total` | Counter |
  | KV の使用率 | `vllm:kv_cache_usage_perc` | Gauge |
  | 実行中の要求 | `vllm:num_requests_running` | Gauge |
  | 入力のトークン | `vllm:prompt_tokens_total` | Counter |
  | 生成のトークン | `vllm:generation_tokens_total` | Counter |
  | ステップごとのトークン | `vllm:iteration_tokens_total` (`_sum` / `_count`) | Histogram |
  | 追い出しの回数 | `vllm:num_preemptions_total` | Counter |

  - 投機的デコードの指標は、サーバーが投機的デコードを有効にしているときだけ登録される
  - 公式の式: 当たり率 = 当たったトークン ÷ 下書きしたトークン。平均の受理長 = 1 + 当たったトークン ÷ 下書きの回数
  - `vllm:gpu_cache_usage_perc` などの古い名前は v0.12.0 で削除済み
- **Implications**:
  - 生成のステップの数は `vllm:iteration_tokens_total_count` の増分、1 ステップあたりに進んだトークンの数は `_sum` の増分 ÷ `_count` の増分で求める
  - KV の使用率は Gauge なので、前後の差では意味がない。計測の間に一定の間隔で読み、最大値を取る
  - 名前の対応は、対象サーバーの定義で上書きできるようにする (比較の基準のサーバーが別の名前を使っている可能性があるため)。既定値は上の表

### プレフィックスキャッシュの動き

- **Context**: 3.4 と 3.5
- **Sources Consulted**: `vllm/config/cache.py` L130、`vllm/v1/core/kv_cache_utils.py` L632-640、L678、L859-863、https://docs.vllm.ai/en/latest/design/prefix_caching/
- **Findings**:
  - V1 のエンジンでは既定で有効
  - ハッシュされるのは、埋まったブロックだけ。ハッシュの入力は (親のブロックのハッシュ、ブロックのトークン、追加の鍵)。先頭のブロックが変われば、連鎖して後ろも全部外れる
  - `cache_salt` は vLLM 独自の項目で、先頭のブロックの追加の鍵になる
- **Implications**: 後述の Decision「キャッシュの効き方は、入力の先頭の識別子で制御する」を参照

### GLM-5.3-Flash への上流の対応

- **Context**: 計測の仕組みそのものには直接効かないが、対象サーバーの前提になる
- **Sources Consulted**: https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json、PR #53906 (2026-09-03 にマージ)、`vllm/config/speculative.py` L1045-1050、issue #55605、#57532、#55626、#57087、#57521、#54521
- **Findings**:
  - 対応は `main` にだけ入っている。**タグ付きのリリース v0.29.0 には入っていない**
  - MTP の層は 1 つ (`num_nextn_predict_layers: 1`)。下書きは実質 1 トークン
  - thinking のブロックは、サーバー側で reasoning のパーサーが有効なら、`/v1/messages` で独立したブロックとして出る
  - GB10 の TP=2 に関わる未解決の問題: 並行バッチで非 ASCII の生成が壊れる (#57087)、2 ノードの TP=2 で最初のトークンまでの時間が 3.3% 悪化 (#57521)、sm121 で greedy のデコードが非決定的 (#54521)
- **Implications**:
  - 計測ランには、サーバーの版 (`GET /version`) を記録する
  - 温度 0 でも、試行ごとに出力が変わりうる。再現性は「同じ入力を送る」ことで担保し、「同じ出力が返る」ことは前提にしない
  - #57087 は日本語の計測に直撃する。後述の Open Question 1 を参照

### 依存する部品のライセンス

- **Context**: 11.3
- **Findings** (すべて公式のリポジトリまたは PyPI で確認):

  | 部品 | 版 | ライセンス | 用途 |
  |---|---|---|---|
  | httpx | 0.28.1 | BSD-3-Clause | HTTP のクライアント |
  | httpx-sse | 0.4.3 | MIT | SSE の読み取り |
  | pydantic | 2.13.5 | MIT | 型と検証 |
  | jsonschema | 4.26.0 | MIT | ツールの引数の検証 |
  | prometheus-client | 0.26.0 | Apache-2.0 | `/metrics` のテキストの解析 |
  | pytest | 9.1.1 | MIT | 試験 |
  | pytest-asyncio | 1.4.0 | Apache-2.0 | 試験 |
  | pytest-httpserver | 1.1.5 | MIT | 試験用の偽のサーバー |
  | uv | 0.12.x | MIT OR Apache-2.0 | 環境の管理 |
  | ruff | 0.16.x | MIT | 整形と静的検査 |
  | mypy | 2.3.x | MIT | 型の検査 |

  - 統計は標準ライブラリ (`statistics`、`random`、`math`) で足りる。numpy / scipy は入れない
  - コマンドラインは標準ライブラリの `argparse` で足りる
  - 公式の `anthropic` SDK (MIT) は `base_url` を差し替えられるが、使わない。チャンクが届いた時刻を自分たちで打つ必要があり、SSE を直接読むほうが計測の意味がはっきりするため
- **Implications**: 実行時の依存は 5 つ (httpx、httpx-sse、pydantic、jsonschema、prometheus-client)

### 公開の課題のライセンス

- **Context**: 11.1、11.2、5.5
- **Findings**:

  | 課題 | ライセンス | 判定 |
  |---|---|---|
  | HumanEval | MIT | 使える |
  | HumanEval+ / MBPP+ (EvalPlus) | Apache-2.0 | 使える |
  | MBPP | CC-BY-4.0 | 使える (表示が要る) |
  | BigCodeBench | Apache-2.0 | 使えるが、多数のライブラリが要るので採らない |
  | LiveCodeBench のデータ | `cc` とだけ表記 | 使わない |
  | BFCL | Apache-2.0 | 使えるが、採点器を自作する手間が大きいので採らない |
  | tau-bench | MIT | 使えるが、この仕様の範囲を超える |
  | ToolBench | 表記が矛盾 (研究・教育目的のみ) | 使わない |
  | RULER | Apache-2.0 (コード) | 考え方だけ参考にする。干し草は自作する |
  | Needle in a Haystack (gkamradt) | コードは MIT、同梱のエッセイは権利の表記なし | 使わない |

- **Implications**: コードの課題は HumanEval+ だけを採る。ツール呼び出しと、長い入力から情報を探す課題は、自作の合成データで作る

### モデルが書いたコードの隔離

- **Context**: 5.4
- **Findings**:
  - Podman (Apache-2.0): `--network none` と `--read-only` が公式の文書で確認できる。会社の規模による条件はない
  - Docker Desktop: 機能は同じ。従業員 250 人以上または年商 1,000 万ドル以上の会社では有償契約が要る
  - Apple の `container` (Apache-2.0): ネットワークを完全に切る方法が公式の文書で確認できない
  - 作業用の Mac に今入っているのは Docker Desktop (29.7.2) だけ
- **Implications**: 実行環境は `podman` と `docker` のどちらでも動くようにし、設定で選ぶ。どちらもなければ、コードの課題を飛ばす (隔離なしでは動かさない)

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| 計測と分析を分ける 2 段のパイプライン | 計測は生データを書くだけ。要約と比較は、生データだけを読む純粋な関数 | 要約を後から作り直せる。途中で止まっても結果が残る。分析を偽のサーバーなしで試験できる。並行して実装できる | 生データの形が両者の契約になるので、先に固める必要がある | **採用** |
| 1 本のスクリプトで計測しながら集計する | 測った値をその場で集計して出力する | 最初は速く書ける | 集計の誤りを直すたびに測り直しになる。8.7 と 10.4 を満たしにくい | 不採用 |
| 既存の負荷試験の道具に載せる | 汎用の HTTP の負荷試験の道具を使う | 同時処理の制御が出来合い | SSE のチャンクごとの時刻、トークン数、ツール呼び出しの分類を扱えない。結局ほとんどを自作する | 不採用 |
| vLLM 同梱の計測スクリプトを使う | 上流の `benchmarks/` を使う | 上流と同じ物差し | `/v1/messages` を対象にしていない。ツール呼び出しの検査がない。対象サーバーを vLLM に縛る | 不採用。考え方は参考にする |

## Design Decisions

### Decision: `/v1/messages` だけを話す、1 種類のクライアントにする

- **Context**: 出力の長さを固定するには `/v1/chat/completions` が要る。2 種類のクライアントを持つ案が調査で挙がった
- **Alternatives Considered**:
  1. `/v1/messages` と `/v1/chat/completions` の 2 種類を持つ
  2. `/v1/messages` だけにする
- **Selected Approach**: 2。生成速度は、長く書かせる指示と `max_tokens` の上限で測る。上限に届かずに終わった試行には印を付ける (2.6)
- **Rationale**:
  - takt が使うのは `/v1/messages` で、測りたいのはその経路の速さ
  - `ignore_eos` で文の終わりを越えて生成させた出力は不自然で、投機的デコードの当たり率を歪める。速さの数字が実際の使い方から離れる
  - 要件 2.6 が、早く終わる試行をすでに想定している
- **Trade-offs**: 出力の長さが試行ごとに少し揺れる。速さは「トークン数 ÷ 時間」なので、長さの揺れは値にほとんど効かない
- **Follow-up**: 早く終わる試行が 2 割を超える条件があれば、指示の文を直す

### Decision: キャッシュの効き方は、入力の先頭の識別子で制御する

- **Context**: 3.4 と 3.5。調査では `cache_salt` が最も確実とされた
- **Alternatives Considered**:
  1. `cache_salt` を送る (vLLM 独自)
  2. 入力の先頭 (システムプロンプトの 1 行目) に、決まった長さの識別子を入れる
- **Selected Approach**: 2。効かない条件では試行ごとに違う識別子、効く条件では同じ識別子を使う
- **Rationale**: 対象サーバーを差し替えられること (1.1) が要件。`cache_salt` を知らないサーバーは黙って無視するので、効かない条件のつもりでキャッシュに当たる。先頭の識別子なら、プレフィックスキャッシュを持つどのサーバーでも確実に外れる
- **Trade-offs**: 識別子のぶん、入力が数トークン長くなる。識別子は決まった長さ (16 進 32 文字) にして、試行の間で長さを揃える
- **Follow-up**: 効かない条件で `cache_read_input_tokens` または `prefix_cache_hits` の増分が 0 に近いことを、結合試験と実機で確かめる

### Decision: 入力の長さは、文字数とトークン数の比で狙う

- **Context**: 3.1、5.3、6.1 は、狙ったトークン数の入力を作る必要がある。Mac にはトークナイザーがない
- **Alternatives Considered**:
  1. Hugging Face のトークナイザーを依存に入れて、手元で数える
  2. 実行のたびに `count_tokens` で数えて調整する
  3. 内容の種類ごとの「1 トークンあたりの文字数」を設定に持ち、決定的に生成する。比は `calibrate` のコマンドで測る
- **Selected Approach**: 3
- **Rationale**:
  - 1 は依存が重く、モデルの資産の取得が要る
  - 2 は、対象サーバーごとに入力が変わりうる。比較では、同じ入力を送ることが何より大事
  - 3 は、同じ設定から必ず同じ入力ができる (11.4)。実際のトークン数は応答から記録し (3.2)、外れたら印が付く (3.3)
- **Trade-offs**: 狙った長さから数 % ずれる。処理速度の計算には実際の値を使うので、結果は歪まない
- **Follow-up**: `calibrate` は `count_tokens` を使う。対象サーバーが `count_tokens` を持たないときは、出力 1 トークンの要求のトークン数で代える

### Decision: 割合の不確かさは、正確な二項の区間で出す

- **Context**: 6.5、9.6。「1% 未満」と言えるだけの試行の数があるかを示す必要がある
- **Alternatives Considered**:
  1. 正規近似
  2. Wilson の区間
  3. 正確な (Clopper–Pearson の) 区間
- **Selected Approach**: 3。二項分布の累積確率を `math.comb` で計算し、二分法で境界を求める。表示は両側 95%。しきい値を下回ったかの判定は片側 95% の上限を使う
- **Rationale**: 割合が 0 に近いところでは、正規近似は使えない。正確な区間は標準ライブラリだけで書ける。しきい値の判定は「下回った」という片側の主張なので、片側の上限が合う
- **Trade-offs**: 保守的 (区間が広め)。崩れが 0 件でも、1% 未満と言うには **1 段階あたり 299 回** の試行が要る。これは統計の要請で、避けられない
- **Follow-up**: 既定の設定は 2 種類にする。`quick` (1 段階 50 回、傾向を見る) と `full` (1 段階 300 回、判定に使う)

### Decision: 2 つの計測ランの比較は、対応のある差の再標本化で判定する

- **Context**: 9.2、9.3。P0 の終わりの条件そのもの
- **Alternatives Considered**:
  1. 中央値の差を、四分位範囲と比べる
  2. 対応のない再標本化 (それぞれの計測ランから引き直す)
  3. 対応のある再標本化 (同じ試行の番号どうしの差を引き直す)
- **Selected Approach**: 試行の番号が同じなら必ず同じ入力になる (決定的な生成) ので、両方の計測ランで成功した試行の番号が揃っているときは 3、揃わないときは 2。差の中央値の 95% 区間が 0 を含むか、差の割合が許容の幅 (既定 2%) より小さければ「収まっている」とする
- **Rationale**: 入力が違えば速さも違う (投機的デコードの当たり率が内容で変わる)。対応のない比較では、入力の違いによるばらつきに、構成の違いが埋もれる。対応のある比較なら、試行 10 回でも数 % の差を見分けられる
- **Trade-offs**: 許容の幅を入れないと、ばらつきがとても小さいときに、意味のない 0.5% の差で「収まらない」と出る。許容の幅は設定に持ち、比較の結果に記録する
- **Follow-up**: 再標本化の乱数の種は固定し、同じ比較が同じ結果を返すようにする

### Decision: ツール呼び出しの正確さと、長い会話の検査は、同じ部品で作る

- **Context**: 5.1 と 6 は、どちらも「呼ぶべきツールと引数が決まっている課題」を採点する
- **Selected Approach**: ツールの定義の目録、課題の生成、応答の分類を 1 組だけ作る。5.1 は「前置きの会話がない」場合、6 は「前置きの会話を段階的に伸ばす」場合として扱う
- **Rationale**: 採点の基準が 2 つあると、短い会話と長い会話の数字を比べられない
- **Trade-offs**: BFCL のような公開の課題との比較はできない。公開値との比較が要るのはコードの課題 (HumanEval+) だけにする

### Decision: 失敗した要求をやり直さず、生データは試行ごとに書き足す

- **Context**: 8.7、10.4、10.6
- **Selected Approach**: 生データは JSON Lines で、試行が 1 つ終わるたびに 1 行を書いてディスクに同期する。計測ランの状態は `running` → `completed` / `aborted` / `interrupted`。`running` のまま残っているものは、異常終了として未完了に扱う
- **Rationale**: 要約は生データから作り直せるので、計測の途中で落ちても、それまでの結果を要約にできる

## Risks & Mitigations

- 長い会話の検査は時間がかかる (`full` の設定で数時間) — 会話の前置きを試行の間で共有し、プレフィックスキャッシュに当てる。1 段階につき複数の会話 (既定 5 本) を使い、1 本の会話への偏りを避ける
- 生データが大きくなる (12 万トークンの要求を何百回も残す) — 要求の本文は gzip で圧縮し、内容のハッシュで重複を除いて保存する
- 計測の間に、ほかの利用者が対象サーバーを使うと、内部の指標の増分が混ざる — 始める前に `num_requests_running` を読み、0 でなければ警告して記録する
- 内容の種類ごとの「1 トークンあたりの文字数」が、モデルの更新で変わる — 実際のトークン数を必ず記録し、外れたら印を付ける。`calibrate` で測り直せる
- sm121 で greedy のデコードが非決定的 (#54521) — 出力が同じになることを前提にしない。比較は統計で行う
- vLLM の `/v1/messages` の実装は変化が速い — クライアントは、知らないイベントや項目を無視する。偽のサーバーの試験で、イベントの順序が入れ替わっても動くことを確かめる

## Open Questions

### 決めたこと (2026-09-19、計測者の判断)

1. **出力の健全性の検査を要件に足す**。Requirement 10 の 7 として追加した。上流の未解決の問題 (#57087) で壊れた出力が高速に出ると、速さの数字だけが良く見えるため。印の付いた試行は集計から外さず、件数を示す (見つけ方が機械的で、誤って拾うことがあるため)
2. **量子化前の基準は、当面は求めない**。モデルカードに HumanEval+ の公開値がなく、BF16 は 2 台に載らない。品質の検査は、構成どうしの相対比較に使う。外部の API に同じ計測を流す道は、設計を変えずに後から取れる
3. **仕様は 1 つのままにし、実装の順序で分ける**

### 残っていること

1. **コードの隔離の実行環境**。Docker Desktop の有償の条件に当たるかどうか。当たるなら、コードの課題を実装する前に Podman を入れる
2. **thinking の切り替えの渡し方**。実機で確かめる

## References

- [vLLM Anthropic API router](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/anthropic/api_router.py) — `/v1/messages` と `count_tokens` の経路
- [vLLM Anthropic serving](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/anthropic/serving.py) — ストリームの変換、トークン数、ツール呼び出しの変換
- [vLLM online serving docs](https://docs.vllm.ai/en/latest/serving/online_serving/) — 公開されている経路の一覧
- [vLLM metrics design](https://docs.vllm.ai/en/latest/design/metrics/) — V1 の指標
- [vLLM V1 metrics loggers](https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py) — 指標の名前の定義
- [vLLM spec decode metrics](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/metrics.py) — 当たり率の式
- [vLLM prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/) — ブロックのハッシュ
- [vLLM tool parsers](https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/__init__.py) — パーサーの名前
- [Anthropic streaming](https://platform.claude.com/docs/en/build-with-claude/streaming) — イベントの公式仕様
- [GLM-5.3-Flash model card](https://huggingface.co/zai-org/GLM-5.3-Flash) — ライセンス (MIT)、チャットテンプレート、公開されているベンチマーク
- [EvalPlus](https://github.com/evalplus/evalplus) / [HumanEval+](https://huggingface.co/datasets/evalplus/humanevalplus) — Apache-2.0
- [Docker Desktop license](https://docs.docker.com/subscription/desktop-license/) — 有償の条件
- [Podman license](https://github.com/containers/podman/blob/main/LICENSE) — Apache-2.0
