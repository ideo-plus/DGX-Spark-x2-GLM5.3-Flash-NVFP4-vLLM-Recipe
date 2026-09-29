# 調査レポート: 公開レシピの数値と、今の構成の比較

## 調査概要

GitHub に公開されている GLM-5.3-Flash の DGX Spark（GB10）向けレシピのうち、数値を載せているものを集めた。載っている数値を、今の構成 `glm53-tp2-mtp3-marlin` の確認の段の値と比べた。

- 調べた日: 2026-09-28
- 実機の計測はしていない。
- 読んだのは、各リポジトリの README・結果の表・issue の本文にある数値だけ。
  - コード・スクリプト・設定・パッチは読んでいない（クリーンルーム、`PLAN.md` §2）。
  - candidate D（NNNtrance）と exl3-tp2 の中身は読んでいない。
- 数値は README を要約して取った二次情報で、一次の文面を逐語で写したものではない。判断に使う数値は、出典の README で確かめ直すこと。
- 条件（出力の長さ、温度、プロンプト、thinking の有無、計測の道具）はレシピごとに違う。比べられるのは傾向までである。
- 印の意味:
  - 【事実】: 出典に書いてあること
  - 【推測】: 事実からの推定

## 今の構成の値（確認の段、`fast`）

構成は `glm53-tp2-mtp3-marlin`。2 台 TP=2、重み `k2s4`、MTP の N = 3、`--moe-backend marlin`。

| 項目 | 値 | 記録 |
|---|---:|---|
| 1 本、コード・英語 | 45.0 tok/s | [記録](../results/2026-09-28-k2-stage3.md) |
| 1 本、コード・日本語 | 43.9 tok/s | 同上 |
| 1 本、散文・英語 | 46.3 tok/s | 同上 |
| 1 本、散文・日本語 | 42.2 tok/s | 同上 |
| 同時 2 本の 1 本あたり（合計） | 27.2 tok/s（45.8） | 同上 |
| 入力の処理 32k（cold） | 1,326 tok/s | 同上 |

計測の条件: 出力 256 トークン、温度 0、1 条件 10 回の中央値。

今回の MiaAI EXL3 + DFlash2 の実機比較では、両ノードの実効 GPU クロック、温度、使用率、thermal slowdown の状態も記録する。共通条件とスナップショットは [実機測定条件](../results/2026-09-29-miaai-exl3-dflash2.md) にまとめた。

## 同じ 2 台構成との比較（1 本の生成速度、tok/s）

| レシピ | エンジン・重み | ドラフター（ライセンス） | コード | 散文 | 構造化・JSON | その他 |
|---|---|---|---:|---:|---:|---|
| **今の構成** | vLLM・NVFP4（アテンションも 4 bit） | モデル付属 MTP N=3（MIT） | **45.0** | **42.2〜46.3** | — | 同時 2 本の 1 本あたり 27.2 |
| [MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark](https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark) | vLLM・NVFP4 | MTP（MIT） | 23〜30（種類の区別なし） | — | — | 同時 2 本の 1 本あたり 16〜19。計測の方法は書かれていない |
| [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) | vLLM・NVFP4 | DFlash2（CC BY-NC-ND） | 46.9 | — | 54〜61 | 受理率はコードで 74% |
| [tonyd2wild/GLM-5.3-Flash-EXL3-on-2x-NVIDIA-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-EXL3-on-2x-NVIDIA-DGX-Spark) | vLLM・EXL3 4bpw | DFlash2（CC BY-NC-ND） | 48.6 | 19.1 | — | cold prefill 1,752（211K） |
| [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | vLLM・EXL3 4bit | DFlash2（CC BY-NC-ND） | — | 36.1 | 62.9 | prefill 1,428〜1,587。受理率は散文で約 34%、構造化で約 96% |
| [Entrpi/glm-5.3-flash-exl3-2x-spark](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark) | vLLM フォーク・EXL3 | DFlash2（CC BY-NC-ND と推定【推測】） | 42 | 30 | 51〜71 | 長文 1,490（133K） |
| [beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark](https://github.com/beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark) | SGLang・NVFP4 | DFlash2（CC BY-NC-ND） | 28.6 | 23.6 | — | 同時 8 本の合計 77.4 |

### MiaAI レシピの同一条件での再測定

同じベンチマーク基盤・同じ 2 台・同じ約 1.8 GHz の GPU 制限で、MiaAI の実機も測定した。結果の詳細は [実機測定条件と結果](../results/2026-09-29-miaai-exl3-dflash2.md) にある。

| 項目 | 結果 |
|---|---:|
| decode code/en | 34.212 tok/s |
| decode code/ja | 31.765 tok/s |
| decode prose/en | 33.423 tok/s |
| decode prose/ja | 26.858 tok/s |
| concurrency 1 本 | 30.538 tok/s |
| concurrency 2 本（1 本あたり） | 22.675 tok/s |
| concurrency 4 本（1 本あたり） | 19.753 tok/s |
| cold prefill 32k | 1340.999 tok/s |

この再測定では、decode と prefill は各条件で失敗 0 回だった。公開値との差をクロックだけで説明することはできず、DFlash2 の受理率と実装経路も併記して比較する。

## 台数が違う参考値

| レシピ | 台数 | ドラフター | コード | 散文 | 入力の処理 |
|---|---|---|---:|---:|---:|
| [mmastrac/glm-5.3-flash-4x-gx10](https://github.com/mmastrac/glm-5.3-flash-4x-gx10) | 4 | DFlash2 | 84.4 | 38.4 | 2,730（32k） |
| [alexellis/glm-5.3-flash-4x-dgx-spark-switchless](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless) | 4 | DFlash2 | 71〜75 | 30〜31 | 1,965〜2,276（64K） |
| [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark) | 4 | DFlash2 | 54.5（今の既定の構成） | — | 約 1,997 |
| [outstandly/glm53-flash-3x-dgx-spark](https://github.com/outstandly/glm53-flash-3x-dgx-spark) | 3 | DFlash2 | 51〜56 | 41 | 約 1,230 |
| [gitcommit90/glm-5.3-one-spark](https://github.com/gitcommit90/glm-5.3-one-spark) | 1（EXL3 2.05bpw） | DFlash2 | 40.1（K5） | 25〜30 | 786〜846 |
| [Weschera/glm53-flash-one-spark](https://github.com/Weschera/glm53-flash-one-spark) | 1（GGUF 約 2.7 bpw） | MTP | 25.0 | 17.9 | 約 291 |
| [sxuff/glm53-flash-single-gb10](https://github.com/sxuff/glm53-flash-single-gb10) | 1（EXL3 2.05bpw） | なし | 29（平均） | 29（平均） | 356 |

tonyd2wild の 4 台の README には、使われなくなった構成の数値（同時 48 本の合計 530 tok/s など）も混ざっている。上の表には今の既定の構成の値だけを載せた。

## わかったこと

- **業務で使えるドラフター（MIT の MTP）の 2 台構成では、公開値の中で今の構成が最も速い。**【事実】
  - 今の構成のコードは 45.0 tok/s。MiaAI-Lab NVFP4 は 23〜30 tok/s。
- **コードは、非商用の DFlash2 を使う 2 台構成（42〜48.6）とほぼ同じ水準。**【事実】
  - 今の構成は、ドラフターではなく重みの読む量を減らして届いた。重みはアテンションまで 4 bit にし、MoE は Marlin で動かしている。
- **散文では、今の構成が大きく上回る。**【事実】
  - 今の構成は 42〜46 tok/s、DFlash2 の 2 台構成は 19〜36 tok/s。
  - DFlash2 は、散文で受理率が 25〜34% と低い（mmastrac、MiaAI-Lab EXL3 の記載）。
  - 日本語の散文の公開値は、今の構成のほかに見つからなかった。
- **構造化出力・JSON では、DFlash2 のほうが速い（51〜71 tok/s）。**【事実】
  - 受理率が 96% 前後と高いためである。
- **同時 2 本の 1 本あたりは、今の構成が 27.2 tok/s。**【事実】
  - MTP の 2 台構成（16〜19）より速いが、基準の 30 には届かない。
- **入力の処理が 2,000 tok/s を超える公開値は、4 台構成だけである。**【事実】
  - 2 台構成の公開値は 1,428〜1,752 tok/s。どれも条件は長文の cold で、32k とは限らない。
  - 2 台で `PLAN.md` §3 の基準（32k で 2,000 tok/s 以上）に届いた公開例は無い。

## 集めたが数値が無いか、比べられないもの

- [barrydeen/glm53-flash-dgx-spark](https://github.com/barrydeen/glm53-flash-dgx-spark): 数値の記載なし（2 台、DFlash2）。
- [Enntity/sparkglm](https://github.com/Enntity/sparkglm): 最初のトークンまでの時間だけ。DFlash2 の実装は、独自の source-available のライセンス。
- [smazurov/sglang-gb10-glm](https://github.com/smazurov/sglang-gb10-glm): 起動の確かめだけ。
- [0xSero/GLM-5.3-Flash-DGX-Spark](https://github.com/0xSero/GLM-5.3-Flash-DGX-Spark): 忠実度の指標だけ（速さは「未着手」と記載）。
- TensorRT-LLM の経路で GLM-5.3-Flash を GB10 で動かした公開レシピは見つからなかった。

## 残った問い

| ID | 問い | 状態 | 次に確かめること |
|---|---|---|---|
| Q1 | 比べる価値の高い 2 本（MiaAI-Lab NVFP4、tonyd2wild NVFP4 DFlash2）の数値が、要約経由で正しく取れているか | 未確認 | README を直接読んで確かめる |
| Q2 | 入力の処理の基準（32k で 2,000 tok/s）が、2 台で届く値か | 未決 | 2 台で届いた公開例が無いことを、基準の見直しの材料にするかを判断する |
| Q3 | 構造化出力・JSON での今の構成の速さ | 未計測 | takt の実際の負荷（ツール呼び出しの JSON）に近い条件で測る。P8 で扱う |
