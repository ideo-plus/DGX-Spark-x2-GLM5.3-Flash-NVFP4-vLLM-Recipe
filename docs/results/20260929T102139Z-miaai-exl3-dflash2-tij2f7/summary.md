# 計測ランの要約: 20260929T102139Z-miaai-exl3-dflash2-tij2f7

> **この計測ランは未完了である** (状態: `interrupted`)。途中までの結果として読むこと (10.4)。

## 実行の条件

| 項目 | 値 |
|---|---|
| 計測ランの識別子 | `20260929T102139Z-miaai-exl3-dflash2-tij2f7` |
| 対象サーバー | `miaai-exl3-dflash2` |
| メモ | MiaAI-Lab EXL3/TR3 4bpw、DFlash2 k=7、FP8 KV、2 台 TP=2。2026-09-29 実機起動 |
| 接続先 | http://10.0.1.60:8888/ |
| モデル (定義 / サーバーの申告) | GLM-5.3-Flash-EXL3 / GLM-5.3-Flash-EXL3 |
| サーバーの版 | 0.1.dev20051+g487ecf187 |
| 状態 | `interrupted` (未完了) |
| 開始 | 2026-09-29T10:21:39.135165+00:00 |
| 終了 | 2026-09-29T10:24:35.397262+00:00 |
| 測ったまとまり | decode |
| 計測の設定 | `fast` |
| サンプリング | temperature=0.0, top_p=(なし), top_k=(なし), thinking=server_default |
| 入力の長さの上限 | 850000 トークン |
| 成功した試行の最小の数 | 5 |
| 道具の版 | `0.1.0+g8827d46f.dirty` |
| 生成器の版 | 1 |
| 要約の形の版 | 1 |
| まとまり `decode` | 試行 10 回 (慣らし 1 回)、出力の上限 256 トークン |

## 警告

- decode/code/en の text_wait_s: 時刻かトークン数が足りず、値にできなかった試行が 10 件あったので、集計から外した
- decode/code/en の text_duration_s: 時刻かトークン数が足りず、値にできなかった試行が 10 件あったので、集計から外した
- decode/code/en の text_chars_per_s: 時刻かトークン数が足りず、値にできなかった試行が 10 件あったので、集計から外した
- decode/code/ja の text_wait_s: 時刻かトークン数が足りず、値にできなかった試行が 10 件あったので、集計から外した
- decode/code/ja の text_duration_s: 時刻かトークン数が足りず、値にできなかった試行が 10 件あったので、集計から外した
- decode/code/ja の text_chars_per_s: 時刻かトークン数が足りず、値にできなかった試行が 10 件あったので、集計から外した
- decode/code/en: 本文未到達 10 件 (うち思考のみ 10 件)
- decode/code/ja: 本文未到達 10 件 (うち思考のみ 10 件)

## 本文への到達と計測件数

| 条件 | 要求 | 要求成功 | 要求失敗 | 要求成功率 | 本文あり | 本文未到達 | 思考のみ | 段階別情報不明 | 本文速度の有効件数 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `decode/code/en` | 10 | 10 | 0 | 10/10 | 0 | 10 | 10 | 0 | 0 |
| `decode/code/ja` | 10 | 10 | 0 | 10/10 | 0 | 10 | 10 | 0 | 0 |

本文ありは本文の到達件数であり、JSONの完全性や妥当性の判定ではない。

## 表の読み方

- 値の意味: `ttft_s` = 最初のトークンまでの時間 (秒)、`decode_tps` = 思考等を含む全出力の生成速度 (tok/s、最初のトークンまでの時間を含めない)、`text_wait_s` = 本文の非空データに到達するまでの秒数、`thinking_duration_s`・`text_duration_s` = 各段階の非空データの区間 (秒)、`thinking_chars`・`text_chars` = 各段階の文字数、`thinking_chars_per_s`・`text_chars_per_s` = 各段階の文字/秒、`thinking_retokenized_tps`・`text_retokenized_tps` = 生文字列の再計数トークン/秒 (生成時の段階別トークン数ではない)、`prefill_tps` = 入力の処理速度 (トークン/秒)、`round_total_tps` = 1 回ぶんの合計の生成速度 (トークン/秒)。集計に入れたのは、成功した、慣らしでない試行だけである。

- `n` の数える単位は行によって違う。`ttft_s`、`decode_tps`、`prefill_tps` は試行の数 (同時処理では 1 本が 1 つ)、`round_total_tps` は回の数である。`decode_tps` は、出力が 16 トークン未満の試行を除いたあとの数になる。

- `試行の数が足りない` の印は、行ごとに判定する (その行の値の数が、成功した試行の最小の数 5 に届かない)。

## 主な結果

### decode

| 条件 | 値 | n | 平均 | 中央値 | 最小 | 最大 | 標準偏差 | 四分位範囲 | 変動係数 | 失敗 | 印 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `decode/code/en` | `decode_tps` | 10 | 36.149 | 36.704 | 32.578 | 38.576 | 1.939 | 2.513 | 0.054 | 0 | — |
| `decode/code/en` | `ttft_s` | 10 | 0.577 | 0.599 | 0.499 | 0.618 | 0.049 | 0.079 | 0.085 | 0 | — |
| `decode/code/en` | `text_wait_s` | — | — | — | — | — | — | — | — | 0 | 試行の数が足りない |
| `decode/code/en` | `thinking_duration_s` | 10 | 7.073 | 6.948 | 6.610 | 7.827 | 0.393 | 0.490 | 0.056 | 0 | — |
| `decode/code/en` | `text_duration_s` | — | — | — | — | — | — | — | — | 0 | 試行の数が足りない |
| `decode/code/en` | `thinking_chars` | 10 | 1118.400 | 1131.000 | 1054.000 | 1168.000 | 38.719 | 52.000 | 0.035 | 0 | — |
| `decode/code/en` | `text_chars` | 10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0 | — |
| `decode/code/en` | `thinking_chars_per_s` | 10 | 158.335 | 159.979 | 149.219 | 164.854 | 5.396 | 7.285 | 0.034 | 0 | — |
| `decode/code/en` | `text_chars_per_s` | — | — | — | — | — | — | — | — | 0 | 試行の数が足りない |
| `decode/code/ja` | `decode_tps` | 10 | 33.878 | 34.694 | 29.794 | 35.916 | 2.110 | 2.145 | 0.062 | 0 | — |
| `decode/code/ja` | `ttft_s` | 10 | 0.599 | 0.613 | 0.517 | 0.627 | 0.040 | 0.011 | 0.066 | 0 | — |
| `decode/code/ja` | `text_wait_s` | — | — | — | — | — | — | — | — | 0 | 試行の数が足りない |
| `decode/code/ja` | `thinking_duration_s` | 10 | 7.555 | 7.350 | 7.100 | 8.559 | 0.502 | 0.467 | 0.066 | 0 | — |
| `decode/code/ja` | `text_duration_s` | — | — | — | — | — | — | — | — | 0 | 試行の数が足りない |
| `decode/code/ja` | `thinking_chars` | 10 | 1068.300 | 1084.000 | 966.000 | 1153.000 | 61.350 | 80.250 | 0.057 | 0 | — |
| `decode/code/ja` | `text_chars` | 10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0 | — |
| `decode/code/ja` | `thinking_chars_per_s` | 10 | 141.863 | 145.020 | 116.312 | 153.244 | 10.893 | 11.880 | 0.077 | 0 | — |
| `decode/code/ja` | `text_chars_per_s` | — | — | — | — | — | — | — | — | 0 | 試行の数が足りない |

## 参考

(この計測ランには、該当する条件がない)

## 対象サーバーの内部の指標

| 条件 | 投機の当たり率 | 平均の受理長 | 生成のステップ | 1 ステップあたりの生成トークン | プレフィックスキャッシュ | KV の使用率の最大 | 追い出し | 得られなかった指標 |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `decode/code/en` | 0.434 | 4.036 | 710.000 | 3.966 | 0.000 | 0.081 | 0.000 | — |
| `decode/code/ja` | 0.388 | 3.719 | 773.000 | 3.643 | 0.000 | 0.081 | 0.000 | — |
| `decode/prose/en` | — | — | — | — | — | 0.081 | — | spec_drafts, spec_draft_tokens, spec_accepted_tokens, prefix_queries, prefix_hits, kv_usage, running_requests, prompt_tokens, generation_tokens, iteration_tokens_sum, iteration_tokens_count, preemptions |
