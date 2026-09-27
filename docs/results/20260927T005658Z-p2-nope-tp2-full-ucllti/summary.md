# 計測ランの要約: 20260927T005658Z-p2-nope-tp2-full-ucllti

## 実行の条件

| 項目 | 値 |
|---|---|
| 計測ランの識別子 | `20260927T005658Z-p2-nope-tp2-full-ucllti` |
| 対象サーバー | `p2-nope-tp2-full` |
| メモ | 自前イメージの TP=2 の prefill・並列・長文脈の計測 (文脈長 163840、同時実行 16。起動側の構成名も p2-nope-tp2-full)。実機では未確認 |
| 接続先 | http://10.0.1.60:8000/ |
| モデル (定義 / サーバーの申告) | glm-5-3-flash / glm-5-3-flash |
| サーバーの版 | 0.1.dev1+g0961bbae2.d20260922 |
| 状態 | `completed` |
| 開始 | 2026-09-27T00:56:58.999328+00:00 |
| 終了 | 2026-09-27T01:46:03.002516+00:00 |
| 測ったまとまり | quality |
| 計測の設定 | `quick` |
| サンプリング | temperature=0.0, top_p=(なし), top_k=(なし), thinking=server_default |
| 入力の長さの上限 | 163840 トークン |
| 成功した試行の最小の数 | 5 |
| 道具の版 | `0.1.0+g23d5693f` |
| 生成器の版 | 1 |
| 要約の形の版 | 1 |
| まとまり `quality` | ツール呼び出し 50 問、探す課題は 1 つの条件につき 2 回 |

## 表の読み方

- `accuracy` (正解の割合) の分母は、**採点できた試行**だけである。要求そのものが失敗した試行と、採点できなかった試行 (コードの隔離の実行環境がない、など) は分母から外し、件数を「採点できなかった」の欄に出す (5.7)。不正解とは混ぜない。

- 割合に添えた区間は、正確な二項の区間 (Clopper-Pearson) の両側 95% である。

- 割合の行には、`試行の数が足りない` の印を付けない。不確かさは区間そのものが表しているので、印で二重に示さない。

## 主な結果

(この計測ランには、該当する条件がない)

## 参考

(この計測ランには、該当する条件がない)

## 品質の検査

| 条件 | 正解 | 採点の対象 | 正解の割合 | 95% の区間 | 採点できなかった | 要求の失敗 | 印 |
|---|--:|--:|--:|---|--:|--:|---|
| `quality/toolcall` | 38 | 50 | 0.760 | 0.618 〜 0.869 | 0 | 0 | — |
| `quality/code/humaneval+` | 25 | 40 | 0.625 | 0.458 〜 0.773 | 0 | 0 | — |
| `quality/needle/8k/d0` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/8k/d25` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/8k/d50` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/8k/d75` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/8k/d100` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/32k/d0` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/32k/d25` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/32k/d50` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/32k/d75` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/32k/d100` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/128k/d0` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/128k/d25` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/128k/d50` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/128k/d75` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |
| `quality/needle/128k/d100` | 2 | 2 | 1.000 | 0.158 〜 1.000 | 0 | 0 | — |

## 対象サーバーの内部の指標

| 条件 | 投機の当たり率 | 平均の受理長 | 生成のステップ | 1 ステップあたりの生成トークン | プレフィックスキャッシュ | KV の使用率の最大 | 追い出し | 得られなかった指標 |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `quality/toolcall` | — | — | 3127.000 | 1.000 | 0.000 | 0.011 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/code/humaneval+` | — | — | 30366.000 | 1.000 | 0.000 | 0.011 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/8k/d0` | — | — | 137.000 | 1.000 | 0.000 | 0.020 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/8k/d25` | — | — | 127.000 | 1.000 | 0.000 | 0.020 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/8k/d50` | — | — | 137.000 | 1.000 | 0.000 | 0.020 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/8k/d75` | — | — | 129.000 | 1.000 | 0.000 | 0.020 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/8k/d100` | — | — | 141.000 | 1.000 | 0.523 | 0.020 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/32k/d0` | — | — | 123.000 | 1.000 | 0.132 | 0.030 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/32k/d25` | — | — | 140.000 | 1.000 | 0.132 | 0.030 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/32k/d50` | — | — | 146.000 | 1.000 | 0.132 | 0.030 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/32k/d75` | — | — | 145.000 | 1.000 | 0.132 | 0.030 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/32k/d100` | — | — | 150.000 | 1.000 | 0.395 | 0.030 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/128k/d0` | — | — | 147.000 | 1.000 | 0.230 | 0.071 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/128k/d25` | — | — | 149.000 | 1.000 | 0.230 | 0.071 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/128k/d50` | — | — | 150.000 | 1.000 | 0.230 | 0.071 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/128k/d75` | — | — | 145.000 | 1.000 | 0.230 | 0.071 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |
| `quality/needle/128k/d100` | — | — | 175.000 | 1.000 | 0.493 | 0.071 | 0.000 | spec_drafts, spec_draft_tokens, spec_accepted_tokens |

## 使った公開の課題

| 名前 | 版 | 入手先 | ライセンス | 採点の方法 |
|---|---|---|---|---|
| HumanEval+ | v0.1.10 | https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus-OriginFmt.jsonl.gz | Apache-2.0 (EvalPlus)、MIT (OpenAI HumanEval) | 応答から取り出したコードに、課題の `test` (`check(candidate)` の定義) と `check(<entry_point>)` の呼び出しを続けて、隔離したコンテナの中で動かす。例外なく終われば正解。正解の割合を、問題の数とともに示す (5.2、5.4) |
