# 0004. `/v1/messages` の対応の状況と、不足の扱い

状態: 未検証のまま持ち越し (2026-09-22)。8 項目の判定方法と、thinking の深さの確かめ方は
決めたが、P1 の打ち切りにより実機での判定は行っていない。以下の手順は再開時の検証案として残す。

関係する要件: 9.1、9.2、9.3、9.4、9.5、10.3。仕様は `.kiro/specs/vllm-baseline/`。手順は
[`docs/vllm-baseline/procedure.md`](../vllm-baseline/procedure.md) の「2.8」。

書かないこと (要件 10.5): **送った内容と応答の本文**(書くのは長さ、トークンの数、終わりの
理由、HTTP の状態だけ)、**認証の情報**(鍵、トークン。Spark にも置かない。要件 2.6)、
**計測者が別に起動していた構成 (`exl3-tp2`) の中身**(起動の引数、設定、差し込まれた
ファイル、記録。読んでよいのは、GPU を使っているプロセスの名前とメモリの量だけである)は、
この記録に書かない。

## 背景

要件 9 は、上流の推論サーバーの `/v1/messages` が、takt の使い方に対して何を扱えて何を
扱えないかを、実機で確かめた事実として持つことを求める。P0 の実測 (`docs/decisions/
0001-bench-harness-measurement-method.md` の「thinking の切り替え」) は、5 通りの渡し方
(Anthropic 形式の `thinking`、`chat_template_kwargs.enable_thinking` など) を試して
「出力が 5 通りとも同じだった」と記録しているが、その渡し方が、vLLM の実装とチャット
テンプレートの両方から見て、そもそも効かない項目ばかりだったことが、本 P1 の調査
(research.md §g) で分かった。

vLLM のソース (`vllm/entrypoints/anthropic/protocol.py`、`serving.py`。build commit
`385dce36…`) を読んだ結果:

- **Anthropic 形式の `thinking` フィールドは実装に存在せず、pydantic の既定
  (`extra="ignore"`) で黙って捨てられる**。未マージの PR #53058 (OPEN) がこの欠落を
  指摘している
- thinking の深さを実際に動かす入口は、`output_config.effort` (`/v1/messages` 独自の口)
  と `chat_template_kwargs.reasoning_effort` (チャットテンプレート経由) の 2 つである
- `chat_template_kwargs` はそのまま下流のチャットテンプレートに渡る

GLM-5.3-Flash のチャットテンプレート (`chat_template.jinja`) を直接読んだ結果
(research.md §g-2):

- **`reasoning_effort` は `low` と `high` だけが効き、それ以外 (未指定を含む) はすべて
  `max` に落ちる**。段は実質 3 つ (low / high / max)
- **thinking は切れない** (生成プロンプトが無条件に `<think>` を開く)
- **`clear_thinking`** は thinking の深さを変えず、**履歴 (過去の assistant の
  `reasoning_content`) を空の `<think></think>` に置き換える**だけである。長い会話の
  入力トークンを減らす効果がある

## 決めたこと

### 1. 8 項目の判定方法 (要件 9.1、9.3)

| # | 項目 | 判定の方法 | `bench` で済むか (要件 9.3) |
|---|---|---|---|
| 1 | ストリーミングの応答 | `bench` の全試行が SSE を読む (`AnthropicStreamEvent.type` の各値) | 済む |
| 2 | ツール呼び出し (定義・応答・結果を含む続き) | `bench --suite quality` のツール課題と `--suite agent` | 済む |
| 3 | 長い会話 (12 万トークン、約 1,200 発話) | `bench --suite agent` の最終段 | 済む |
| 4 | 入力と出力のトークン数 | `bench` の `input_tokens` / `output_tokens` | 済む |
| 5 | キャッシュに当たったトークン数の内訳 | `cache_read_input_tokens` / `cache_creation_input_tokens` (`--enable-prompt-tokens-details` が要る) | 済む |
| 6 | トークン数だけを数える口 | `bench calibrate` の `POST /v1/messages/count_tokens` | 済む |
| 7 | thinking のブロックの出方 | `bench` が `content_block_start.type == "thinking"` と `thinking_delta` を解釈する | 済む |
| 8 | thinking の深さの渡し方 | `bench` では確かめられない (`sampling.thinking` は `server_default` のみ)。`serve thinking` で素の HTTP を送り分ける | 別に確かめる (下の節 2) |

項目 1〜7 は、`bench` のいずれかの計測ランの ID を根拠として記録する (要件 9.3。別の
確かめを足さない)。判定結果 (対応できる / できない / 条件つき) と、根拠にしたランの ID は、
段 2 (タスク 8.x) でこの表に追記する。

### 2. thinking の深さの 5 通り (要件 9.2)

同じプロンプト・同じ seed・`temperature 0` で、次の 5 通りを `serve thinking` で送り分け、
thinking のブロックの文字数と `output_tokens` を比べる。

| # | 送るもの | 期待 |
|---|---|---|
| 1 | 何も渡さない | 既定 = `Reasoning Effort: Max` |
| 2 | `{"output_config": {"effort": "low"}}` | `Reasoning Effort: Low` に変わる (`/v1/messages` 本来の口) |
| 3 | `{"chat_template_kwargs": {"reasoning_effort": "low"}}` | 2 と同じ結果になるはず (経路違い) |
| 4 | `{"output_config": {"effort": "medium"}}` | 変わらないはず (テンプレートが `max` に落とす。「API は受けるが効かない」ことの確認) |
| 5 | `{"chat_template_kwargs": {"clear_thinking": true}}` | 深さは変わらず、複数ターンの会話で `input_tokens` が減るはず |

**効いたと言える条件**: 1 と 2 の間で、thinking のブロックの文字数か `output_tokens` が
有意に変わること。4 で変わらないこと。5 で `input_tokens` が減ること。記録するのは、
thinking のブロックの**文字数**、入力と出力のトークン数、終わりの理由だけで、**本文は
保存しない** (要件 10.5)。実際の判定結果は、段 2 (タスク 8.x) でここに追記する。

### 3. 不足の扱いの案 (要件 9.4)

この節は、実機の判定 (段 2) で、扱えない・条件つきの項目が見つかった場合に埋める。用意して
おく案の型は次の 3 つ (procedure.md 2.8):

1. **変換層を自分たちで書く** (takt と `/v1/messages` の間に薄い変換を挟む)
2. **上流に返す** (vLLM の issue/PR として報告する)
3. **そのまま受け入れる** (takt 側の使い方を変える、または影響がないと判断する)

判定結果ごとに、この 3 つのどれを採るか、理由とともに、段 2 で追記する。

### 4. P0 の判断の見直し (要件 9.5)

P0 の記録 (`0001-bench-harness-measurement-method.md`) は、thinking の設定を
`server_default` 固定にしている。上の「背景」で分かったとおり、P0 が試した渡し方
(Anthropic 形式の `thinking`、`enable_thinking`) はテンプレートが読まない項目ばかりで、
**本来の口 (`output_config.effort`、`chat_template_kwargs.reasoning_effort`) を試して
いなかった**。段 2 の節 2 の表の # 2 が実際に効けば、この結果は P0 の判断を見直す理由に
なる (要件 9.5)。**その場合でも、`bench` 側の変更 (`sampling.thinking` の選択肢を増やす
など) は、この記録ではなく `bench-harness` の仕様の変更として扱い、ここでは行わない。**

## 採らなかった案

| 案 | 採らなかった理由 |
|---|---|
| Anthropic 形式の `thinking: {type: enabled/disabled}` で深さを制御する | `AnthropicMessagesRequest` にこの項目は定義されておらず、pydantic の既定で黙って捨てられる (research.md §g、未マージ PR #53058) |
| `chat_template_kwargs.enable_thinking` で切り替える | GLM-5.3-Flash のチャットテンプレートに `enable_thinking` という変数は存在しない (thinking は無条件に開く) |
| thinking の判定を `bench` の計測ランだけで済ませる | `bench` の `sampling.thinking` は `server_default` しか選べず、5 通りの送り分けを表せない。要件 9.3 は「`bench` で済むものだけ」を対象にしており、この項目は対象外 |

## 影響と限界

- `output_config.effort` の型は `Literal["low","medium","high","xhigh","max"]` で
  `"none"` がない一方、`/v1/chat/completions` の `reasoning_effort` は
  `Literal["none","minimal","low","medium","high","xhigh","max"]` である。**`medium` /
  `minimal` / `none` / `xhigh` を送ると API は 200 を返すが、テンプレート側で黙って
  `max` に落ちる** (「設定したのに変わらない」罠)。構成にも計測にも `low` / `high` /
  既定 (max) の 3 つしか使わない
- `--enable-prompt-tokens-details` を付けると `input_tokens` の意味が変わる
  (`input_tokens = prompt_tokens - cache_read - cache_creation`)。付けていない計測ランと
  直接比べない

## 見直す条件

次のどれかをしたら、この記録を見直す。

| 変えたもの | すること |
|---|---|
| 段 2 で 8 項目の判定結果が出た場合 | 節 1 の表に、判定結果と根拠のランの ID を追記する |
| thinking の深さの実測 (節 2) が「効いた」となった場合 | 節 4 に従い、P0 の判断の見直しを検討し、`bench-harness` の仕様の変更として起票する (ここでは変更しない) |
| 扱えない・条件つきの項目が見つかった場合 | 節 3 の案から選び、理由とともに記録する |
| vLLM の `/v1/messages` の実装 (build commit) が変わった場合 | 「背景」の引用元のソースを読み直し、この記録を更新する |

## クリーンルーム (要件 11.4、11.5、11.6)

**0002 と同じ日 (2026-09-22)、同じ性質の作業者 (この会話の文脈を持たない新しい
サブエージェント) が、research.md と道具の一次の資料だけから、判定の方法を書いた。**
チャットテンプレートは、ベンダの現行版 (公開リポジトリの生ファイル) を直接読んだだけで、
モデルカードの本文は開いていない。

**参照した資料の一覧 (要件 11.6)**: 0002 の一覧に加えて、`/v1/messages` に固有の資料は
次のとおり (research.md §g の Findings)。

- **vLLM のソースコード** (build commit `385dce36bcee42309924a5ece951a96db3dce7f2`。
  `https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/...`):
  `vllm/entrypoints/anthropic/api_router.py`、`vllm/entrypoints/anthropic/protocol.py`、
  `vllm/entrypoints/anthropic/serving.py`、
  `vllm/entrypoints/openai/chat_completion/protocol.py`、
  `vllm/entrypoints/launchers/cli_args.py`
- **上流の未マージ PR**: `https://github.com/vllm-project/vllm/pull/53058`
  (`thinking` フィールドの欠落)
- **GLM-5.3-Flash のチャットテンプレート** (機械可読のテンプレートファイル。モデルカードの
  本文ではない):
  `https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/chat_template.jinja`
- **自分たちで測った事実**: `serve thinking` による 5 通りの実測 (段 2 で追記)、`bench`
  のランの結果 (段 2 で追記)

## 実機で確かめたこと

P1 では起動できず、8 項目すべて未検証。対応・非対応の判定や、変換層を作る判断は行わない。
P0 の設定も変更しない。起動できた後の検証に、次の項目を持ち越す:

- 8 項目それぞれの判定結果と、根拠にした `bench` のランの ID
- thinking の 5 通りの実測 (thinking のブロックの文字数、`output_tokens`、`input_tokens`
  の変化)
- 扱えない・条件つきの項目があった場合の、不足の扱いの決定
- P0 の判断 (`sampling.thinking = "server_default"`) を見直す理由になったかどうか
