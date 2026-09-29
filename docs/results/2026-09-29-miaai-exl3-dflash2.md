# MiaAI EXL3 + DFlash2 の実機測定条件

## 対象

- 日付: 2026-09-29 JST
- 2 台の DGX Spark、TP=2
- MiaAI-Lab の EXL3/TR3 4bpw、DFlash2 `k=7`、FP8 KV
- サーバー: `GLM-5.3-Flash-EXL3`、API は head の `:8888`
- ベンチマーク: `fast` プロファイル。温度 0、出力 256 トークン（prefill は 16 トークン）、warmup 1 回、通常試行 10 回（prefill は 5 回）

## GPU クロック・熱条件

ベンチマーク実行中の `nvidia-smi` のスナップショットを記録する。数値は瞬間値であり、長時間の最大値ではない。

| ノード | 実 GPU クロック（graphics/SM） | 最大 graphics | 温度 | GPU 使用率 | 電力 | thermal slowdown |
|---|---:|---:|---:|---:|---:|---|
| head `spark-153d` | 約 1774 MHz | 3003 MHz | 64〜65℃ | 約 96% | 約 34 W | なし |
| worker `spark-5083` | 約 1781 MHz | 3003 MHz | 62〜63℃ | 約 96% | 約 34 W | なし |

この測定では、実効クロックは約 1.8 GHz の上限で動作しており、確認時点の `clocks_throttle_reasons` に thermal slowdown は出ていない。したがって、今回の数値は熱によるスロットリング後の値ではなく、意図したクロック制限下の値として扱う。ただし、これはスナップショットであり、安全な上限値そのものを証明するものではない。

MTP+Marlin の比較値も同じ Spark のクロック制限（GPU 約 1800 MHz、CPU 約 3.0 GHz）下で測定している。MiaAI 側の公開レシピには、ベンチマーク時の実クロック、電力上限、温度上限の記録は見当たらないため、公開値との差をクロックだけで補正しない。

## 記録

- decode: [`20260929T041916Z-miaai-exl3-dflash2-25kfbz`](20260929T041916Z-miaai-exl3-dflash2-25kfbz/summary.md)
- prefill: [`20260929T042620Z-miaai-exl3-dflash2-58w8l9`](20260929T042620Z-miaai-exl3-dflash2-58w8l9/summary.md)
- concurrency: [`20260929T044222Z-miaai-exl3-dflash2-npv4qy`](20260929T044222Z-miaai-exl3-dflash2-npv4qy/summary.md)

## 測定結果（中央値）

### Decode

| 条件 | 生成速度 | 投機の当たり率 | 平均の受理長 |
|---|---:|---:|---:|
| code / en | 34.212 tok/s | 0.430 | 4.009 |
| code / ja | 31.765 tok/s | 0.389 | 3.720 |
| prose / en | 33.423 tok/s | 0.405 | 3.835 |
| prose / ja | 26.858 tok/s | 0.299 | 3.092 |

各条件 10 回、失敗 0 回。

### Prefill（cold）

| 入力長 | 入力処理速度 | TTFT |
|---:|---:|---:|
| 8k | 1272.282 tok/s | 6.408 s |
| 32k | 1340.999 tok/s | 24.142 s |
| 128k | 1329.309 tok/s | 97.099 s |

各条件 5 回、失敗 0 回。warm 条件はプレフィックスキャッシュが効くため、実効値として別扱いにする。

### Concurrency

| 同時本数 | 1 本あたりの生成速度 | 合計の生成速度 | 投機の当たり率 | 平均の受理長 |
|---:|---:|---:|---:|---:|
| 1 | 30.538 tok/s | 30.778 tok/s | 0.352 | 3.464 |
| 2 | 22.675 tok/s | 27.908 tok/s | 0.403 | 3.824 |
| 4 | 19.753 tok/s | 30.921 tok/s | 0.418 | 3.926 |

各条件 2 回、失敗 0 回。1 本・2 本はサンプル数が 2 のため、4 本と同じ参考扱いにする。
