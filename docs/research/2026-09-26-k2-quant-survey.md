# 調査レポート: K2 残りの BF16 の重みを FP8 か NVFP4 にする方法

## 調査概要

GLM-5.3-Flash の今の重み（`RedHatAI/GLM-5.3-Flash-NVFP4`、rev `18d55bfd…`）では、NVFP4 になっているのはルーティングされる専門家だけである。
この調査では、残りの BF16（アテンション、共有の専門家、dense、lm_head）を FP8 か NVFP4 にする方法を、次の 4 点から調べた。

- 作る道具
- 公開されている重み
- 動いている vLLM（`0961bbae`）が sm_121 で使うカーネル
- 品質の確かめ方

調べた日は 2026-09-26。実機の操作と、この文書以外のファイルの変更はしていない。
クリーンルームの決まりに従い、他のレシピのスクリプト・パッチ・設定の文面は写していない。candidate D と `exl3-tp2` の中身も開いていない。

印の意味は次のとおり。

- 【事実】: 出典（URL かファイルの位置）で確かめたこと
- 【計算】: 事実の数値からの算術
- 【推測】: 事実からの推定

vLLM のソースの位置は、`serving/var/nope-build-0961bbae/source/`（以下 `src/`）からの相対パスで書く。

## 主要な発見

- **今の vLLM では、KDA と MLA の射影は、重みが何であっても BF16 で持つ。** 量子化した重みを作るだけでは速くならず、glm5next の修正が要る。【事実】
  - KDA は `src/vllm/models/glm5next/common/kda.py:188-194` で `quant_config` を外している。
  - MLA は `src/vllm/models/glm5next/common/model.py:330` で `quant_config=None` にしている。
  - FP8 の MLA の重みを読むと、読み込むときに BF16 に戻す（`model.py:1239-1312`）。
- **共有の専門家、dense、lm_head は、今のままでも量子化の設定を受け取る。** 重みの側で `ignore` から外せば、そのまま効く。【事実】`model.py:214-221`、`353-360`、`947-951`
- **BF16 の部分の 55% は KDA の射影である。** 1 台が 1 トークンで読む BF16 は約 8.54 GB で、そのうち KDA が 4.72 GB を占める。【計算】
  - KDA の射影を量子化した品質の公表値は、GLM では見つからない。
  - 公式の FP8 の重み（Z.AI）も、KDA は BF16 のまま残している。【事実】
- **FP8 と NVFP4A16（重みだけ）なら、較正のデータが要らない。** 今の NVFP4 の重みに入っている BF16 のテンソルから作れる。専門家のテンソルは、そのまま流用できる。【事実・推測】
- **sm_121 で速く動く見込みが最も高いのは、重みだけの Marlin（FP8 と NVFP4 のどちらも）である。**【推測】
  - CUTLASS の FP8 と FlashInfer の FP4 には、sm_121 での不具合の報告がある。【事実】
- **見積もり（投機なし）は次のとおり。**【推測】
  - すべて FP8: 1 ステップ約 50 ms（約 20 tok/s）
  - すべて NVFP4A16（lm_head は FP8）: 約 40 ms（約 25 tok/s）
  - glm5next を直さず、共有・dense・lm_head だけ FP8: 約 69 ms で、ほとんど効かない

## 1. モデルの形と、層ごとの線形層

構成の値は、`zai-org/GLM-5.3-Flash` の config による（出典 W1）。【事実】

- 45 層（MTP は別に 1 層）。KDA が 34 層、MLA が 11 層（層 3, 7, …, 43）。
- 層 0〜2 は dense の MLP。
- hidden は 4096、vocab は 154,880。
- 専門家は 288 個で、共有の専門家が 1 つ。
- MLA の値: q_lora 1536、kv_lora 512、nope 256、rope 0、v 256、64 ヘッド。
- KDA の値: 64 ヘッド × 128。
- indexer の値: 64 ヘッド × 128。

1 台（TP=2）が 1 トークンで読む BF16 は、次のとおり。【計算】
column・row の並列の層は 2 で割り、複製される層は割っていない。
計算は `src/` の層の定義（下の列）に合わせた。

| 部分 | 線形層（定義の位置） | 1 台の BF16 | 今の vLLM で量子化できるか |
|---|---|---:|---|
| KDA の射影 ×34 | `in_proj_qkvbfg_a`（q,k,v,b は分割、f_a,g_a は複製）、`f_b_proj`、`g_b_proj`、`o_proj`（`kda.py:217-298`） | 4.72 GB | できない（`kda.py:188-194`、`src/vllm/model_executor/layers/mamba/gdn/base.py:41`） |
| MLA の射影 ×11 | `fused_qkv_a_proj`（複製）、`q_b_proj`、`o_proj`（`attention.py:446-492`） | 1.20 GB | できない（`model.py:330`） |
| MLA の `kv_b_proj` ×11 | decode では BF16 の W_UK / W_UV として使う | 0.19 GB | 量子化しても BF16 に戻る（`src/vllm/model_executor/layers/attention/mla_attention.py:1226-1231`） |
| indexer ×11 | `wq_b`（複製）、`wk_weights_proj`（常に BF16）（`attention.py:250-265`） | 0.29 GB | できない（MLA の `None` を受け継ぐ） |
| 共有の専門家 ×42 | `gate_up_proj`、`down_proj`（`model.py:214-221`） | 1.06 GB | できる |
| dense の MLP ×3 | 同上（`model.py:353-360`） | 0.45 GB | できる |
| lm_head | `ParallelLMHead`（`model.py:947-951`） | 0.63 GB | できる（compressed-tensors は `compressed_tensors.py:180-187`） |
| 合計 |  | **8.54 GB** |  |

- 実測との照合【計算】
  - プロファイルの BF16 の GEMV と GEMM は、合わせて 51.9 ms / ステップ（48.10 + 1.72 + 2.09）。
  - 8.54 GB ÷ 51.9 ms で、実効の帯域は約 165 GB/s になる。
  - 前の調査の差し引き（BF16 は約 8.1 GB、`docs/research/2026-09-25-path-survey.md`）とも近い。
- 先例: 同じ vLLM の Kimi K3 は、KDA の `in_proj` に ModelOpt の FP8（`FP8_PB_WO`）を通し、128 の境界に詰め物をしている（`src/vllm/models/kimi_k3/nvidia/kda.py:557-575`）。【事実】

## 2. 量子化の道具（問い 1）

| 道具 | ライセンス | GLM-5.3-Flash（glm5next、KDA / MLA）への対応 | メモリの要り方 | 較正のデータ |
|---|---|---|---|---|
| llm-compressor | Apache-2.0（W3） | RedHatAI の NVFP4 はこの道具で作られた（W2）。v0.9.0 でアテンションの量子化が入った（W5）。GLM-5.x で KDA を量子化する例は未確認 | 層ごとに読み込む方式と、ディスクへの退避がある（W3）。1 台の GPU に 1 層が載ればよい。全体は CPU の RAM かディスクに置く | FP8 dynamic と FP8 block は不要。NVFP4（W4A4）は活性の全体のスケールのために要る（約 20 サンプルの例、W6）。NVFP4A16 は不要 |
| compressed-tensors | Apache-2.0（W4） | 形式のライブラリ。vLLM が読む | — | — |
| NVIDIA Model Optimizer | LICENSE は Apache-2.0（W7）。「プロプライエタリのライセンスがある」との issue #195 が open で、**要判断** | `nvidia/GLM-5.3-Flash-NVFP4` を作った（v0.47.0.dev）。ただし、アテンションと共有の専門家は除外している（W8） | 未確認 | NVFP4 は要る |
| Intel AutoRound | Apache-2.0（W9） | GLM-5.x・KDA への対応は未確認 | MoE 向けに RAM を減らす工夫がある | 調整（較正）が要る |
| 自前の変換（案） | 自前で書く（Apache-2.0 のライブラリだけを使う） | glm5next の名前に合わせて自分で書く | テンソルを 1 つずつ読み書きすればよい。GPU は要らない | FP8 の重みだけ、FP8 dynamic、NVFP4A16 なら不要【推測】 |

元の重みの候補（どれも MIT）【事実】

- `zai-org/GLM-5.3-Flash-BF16`: 純粋な BF16。約 321B パラメーター、約 643 GB（W10）。
- `zai-org/GLM-5.3-Flash`: BF16 と FP8 の混在で、約 328 GB（W1）。
- `RedHatAI/GLM-5.3-Flash-NVFP4`: 184.26 GiB（`serving/var/20260925T220240Z-logs-p2-nope-tp2-full-prof/head/container.stdout.log:43`）。
  - `ignore` に、すべての層のアテンション・dense・共有の専門家・lm_head・vision が入っている（W2）。
  - つまり、これらは量子化されていない。

今の重みから作れるか【推測】

- 較正の要らない方式（FP8 の重みだけ、FP8 dynamic、NVFP4A16）: 今の重みにある BF16 のテンソルを変換するだけで作れる。専門家のテンソルは 1 バイトも変えない。
  - ただし、RedHatAI の BF16 のテンソルが元の BF16 と同じものかは、未確認である。`zai-org/GLM-5.3-Flash-BF16` の同じ名前のテンソルとハッシュを比べれば確かめられる。
- NVFP4（W4A4）: 活性を較正するために、モデルを前向きに走らせる必要がある。NVFP4 の専門家を載せたままこれを行う手段は、確認できなかった。元の BF16（643 GB）から作り直すのが確実である。
- llm-compressor で glm5next を読み込むには、transformers の側にモデルの定義が要る。この定義があるかは未確認である。vLLM は `src/vllm/transformers_utils/configs/glm5_next.py` に自前の config を持っている。

## 3. 公開されている重み（問い 2）

「アテンションか dense を量子化したもの」を中心に並べた。どれもライセンスは MIT である。【事実】

| 重み | 出した者 | 量子化されている部分 | 品質の公表値 |
|---|---|---|---|
| `zai-org/GLM-5.3-Flash`（W1） | Z.AI | FP8 block（128×128）: 専門家、MLA の q_a / kv_a / q_b / o、indexer の wk。KDA は BF16（`kda.py:188` の注記「fp8 checkpoints omit their scales」）。ただし、どの層が FP8 かは vLLM の読み込みの処理（`model.py:1233-1238`）から読んだもので、全部の一覧は未確認 | 見つからない |
| `nvidia/GLM-5.3-Flash-NVFP4`（W8） | NVIDIA（ModelOpt） | 専門家と dense の MLP が NVFP4。アテンション、共有の専門家、gate、lm_head は除外。KV は FP8 | BF16 と比べてほぼ同じ（GPQA-D 0.9217 → 0.9211、IFBench 0.613 → 0.6054 など） |
| `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`（W11） | 個人・コミュニティ | 専門家が NVFP4、MTP の専門家が MXFP8。KDA 34 層と MLA 11 層（indexer を含む）は BF16 と明記 | 基準（BF16）との比較はない |
| `cyankiwi/GLM-5.3-Flash-AWQ-INT4`（W12） | 個人 | モジュールごとに INT4、FP8 block、BF16 を混ぜる（どこが何かの一覧は未確認）。動かすには vLLM のフォークのブランチ `glm53-flash-ct` が要る | 未確認 |

- **KDA の射影を FP8 か NVFP4 にした、vLLM で読める重みは見つからなかった。**【事実（見つからなかったこと）】
- MLA まで量子化した公開の重みは、Z.AI の FP8 だけである。そして、今の vLLM はそれを BF16 に戻して読む。

## 4. vLLM 0961bbae が sm_121 で使うカーネル（問い 3）

イメージは torch `2.13.0+cu130`、FlashInfer 0.7.0（`docs/results/2026-09-22-nope-build.md:77`）。
CUDA 13 なので、`12.0f`（SM12x の系統をまとめたもの）でビルドされる（`src/CMakeLists.txt:832-847`、`1007-1027`、`630-634`）。【事実】
`--linear-backend`（`src/vllm/config/kernel.py:293`）で、どのカーネルを使うかを固定できる。【事実】

| 形式（compressed-tensors の方式） | 自動で選ばれるカーネル（sm_121） | 根拠 | decode（M=1）での見立て |
|---|---|---|---|
| FP8 W8A8 dynamic（重みはチャネルごと、活性はトークンごと） | FlashInfer は対象外（テンソル単位のスケールだけ）。次の **CUTLASS の sm120 の scaled_mm** が選ばれる | `src/vllm/model_executor/kernels/linear/__init__.py:423-431`、`scaled_mm/flashinfer.py:55-65`、`scaled_mm/cutlass.py:157-170`。M ≤ 16 には 16×64×128 のタイルがある（`src/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh:131,155`） | 小さい M 用のタイルはある。ただし、線形層ごとに活性の量子化のカーネルが 1 回増える【推測】。sm_121 での `cutlass_scaled_mm` の Error Internal の報告がある（#40934、#40758）【事実】 |
| FP8 block（128×128） | FlashInfer と DeepGEMM は対象外とみられる。次の CUTLASS の blockwise sm120 が選ばれる | `__init__.py:458-469` | sm_121 で `CutlassFp8BlockScaledMMKernel` が失敗し、Triton なら動くとの報告（#43367）【事実】。KDA の合成した射影（b は 1 台あたり 32 行）は 128 の境界に合わず、詰め物が要る【推測】 |
| FP8 W8A16（重みだけ） | Humming（入っていれば）、なければ **Marlin FP8** | `__init__.py:486-489`、`scaled_mm/marlin.py:35-45`、Marlin FP8 のビルド `CMakeLists.txt:630-634` | 活性を量子化しない。小さい M 向けの Marlin で、decode に最も向く【推測】 |
| NVFP4 W4A4 | CuTe DSL は sm_10x だけ。次の **FlashInfer CUTLASS FP4** が選ばれる（条件は `has_device_capability(100)` で、121 も通る） | `__init__.py:555-570`、`nvfp4/flashinfer.py:112-123,184-197` | FlashInfer の `mm_fp4` が SM120 で壊れる報告（flashinfer #2577）【事実】。vLLM の CUTLASS FP4 の sm120 のタイルは最小 128×128×128（`src/csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu:59-73`）で、M=1 には無駄が大きい【推測】 |
| NVFP4 W4A16（NVFP4A16） | cc が 100・103 でないので **Marlin NVFP4** に固定される | `__init__.py:1119-1130` | 活性を量子化しない。decode に向く【推測】 |

- NVFP4 の注意点【事実】
  - 合成した層（KDA の 6 つの射影、MLA の `fused_qkv_a`）で、全体のスケールが射影ごとに違うと、警告を出して最大値にそろえる（`compressed_tensors_w4a4_nvfp4.py:100-113`）。
  - だから、NVFP4 にするときは、合成する射影どうしでスケールを共有させて作る必要がある。
  - FP8 のチャネルごとのスケールなら、この問題は起きない。
- 一般の注意点【事実】
  - sm_121 が「sm120 専用」の判定で外れる類の不具合の報告がある（eugr/spark-vllm-docker #143）。
  - どのカーネルが選ばれたかは、起動のログ（`Using … for NVFP4 GEMM` など）で必ず確かめる。

## 5. 読む量の減り方と、1 ステップの見積もり（問い 4）

仮定【推測】

- 重み以外の時間（プロファイルの 75.4 ms から、BF16 の 51.9 ms を引いた 23.5 ms）は変わらない。専門家の NVFP4 も、この 23.5 ms に入っている。
- 新しいカーネルの実効の帯域は、楽観の場合に今の BF16 の GEMV と同じ 165 GB/s、悲観の場合に 120 GB/s とする。悲観の場合も、BF16 のまま残る部分は 165 GB/s で数える。
  - 悲観の側は、小さい M で帯域が出ないことと、活性の量子化の分を見込んだ値である。
- `kv_b_proj` と indexer の `wk` は BF16 のまま残る。
- 1 要素あたりのバイト数: FP8 は 1、NVFP4 は 0.5625（スケール込み）。

| シナリオ | 1 台の読む量（今の BF16 部分） | 1 ステップ（楽観〜悲観） | 投機なしの tok/s |
|---|---:|---:|---:|
| 今（実測、プロファイルあり） | 8.54 GB | 75.4 ms | 13.3（プロファイルなしでは約 14） |
| S0: glm5next を直さず、共有・dense・lm_head だけ FP8 | 7.47 GB | 68.8〜71.2 ms | 約 14〜14.5 |
| S1: S0 に MLA と indexer を加えて FP8（KDA は BF16） | 6.72 GB | 64.2〜68.4 ms | 約 14.5〜15.5 |
| S2: 全部 FP8（KDA も） | 4.36 GB | 49.9〜59.4 ms | 約 17〜20 |
| S3: 全部 NVFP4A16、lm_head は FP8 | 2.67 GB | 39.7〜45.4 ms | 約 22〜25 |

- 表の値は【計算】で、前提は【推測】である。
- S2 と S3 は、`docs/results/2026-09-26-k2-profile-full.md` の見積もり（約 49 ms、約 38 ms）と合う。
- 読む量の差は、そのまま KV の余裕になる。【計算】
  - S2 では 1 台あたり約 4.2 GB 増える。今の KV の割り当ては 15.73 GiB（上記ログの 69 行目）。
- 効くのは KDA を含めたとき（S2、S3）だけである。KDA を BF16 のまま残すと、効果は 1 割未満にとどまる。【計算】

## 6. 品質の確かめ方（問い 5）

このリポジトリの `bench` で分かっていること【事実】

- 今の基準の値（投機なし）は、toolcall 37/50、humaneval+ 26/40、needle 30/30（`docs/results/2026-09-25-k1-mtp1.md:69-71`）。
- humaneval+ の不正解は、出力の上限（1024）で切れたものだった。上限を 2048 にすると 10/10 になった（`docs/results/2026-09-25-k1-n-sweep.md:18`）。
- toolcall の 95% 区間は 0.597〜0.854 と広い（`docs/results/2026-09-23-decode-gpu-clock-cap.md:137`）。
- このため、quick の設定で見分けられるのは大きな劣化だけである。

安く確かめる手順（案）【推測】

1. **問題ごとの一致を見る。** `--suite quality --profile quick` に `--set quality.code_max_tokens=2048` を付けて測り、今の構成と問題ごとに正解・不正解を突き合わせる。K1 と同じやり方で、約 20 分かかる。
2. **壊れの印を見る。** `repetition_loop` と `replacement_char` が 0 件であることと、needle 128k が正解であることを確かめる。KDA の再帰の状態は、誤差が長い文脈で積み重なるおそれがある。
3. **よりよい感度が要るとき。** 同じプロンプトで、今の構成と量子化した構成の次のトークンの分布を比べる（top-1 の一致率と、logprobs の差）。
   - `bench` には logprobs を取る機能がない。`experiments/` に小さな台本を足す必要がある。
4. 残った候補だけを `--profile full`（toolcall 200、humaneval+ 164 問、needle 4 回）で確かめる。

公表されている影響【事実】

- NVIDIA の NVFP4（専門家と dense の MLP）: GLM-5.3-Flash で BF16 とほぼ同じ（W8）。
- Z.AI 自身が、MLA の射影を FP8 block にした重みを出している（W1）。MLA の FP8 の品質の根拠としては、これが最も強い。
- KDA に近い線形アテンション（Gated DeltaNet）を NVFP4 W4A4 にしても、BF16 と種の違いの範囲で一致した報告がある（arXiv 2609.04098、5 課題の平均の差 −0.52）（W13）。
  - ただし、KDA そのものでの公表値はない。
- 一方で、RedHatAI の Kimi-K2.6-NVFP4 や NVIDIA の Qwen3-Next の NVFP4 は、アテンションを BF16 のまま残している（W14）。業界の既定は保守的である。

## 7. 推奨（問い 6）

### 本命: FP8 の重みだけ（チャネルごと、W8A16 を Marlin で）。段階を分けて KDA まで広げる

- 作り方: 今の NVFP4 の重みにある BF16 のテンソルを、自前の変換で FP8（E4M3、出力チャネルごとのスケール）にし、compressed-tensors の形で書く。
  - 較正は要らない。GPU も要らない。専門家のテンソルと config の NVFP4 の群は、そのまま残す。
- 必要な vLLM の修正（自前）
  - `kda.py:188-194` で `quant_config` を外している処理を、構成で選べるようにする。
  - `model.py:330` も同じようにする。
  - `_try_load_fp8_attn_proj`（`model.py:1247-`）が、量子化したまま読み込めるようにする。
  - Kimi K3 の経路が先例になる。
- 段階
  - (a) S0 で、形式・読み込み・Marlin FP8 を確かめる（glm5next の修正は要らない）。
  - (b) S1 で MLA を加える。
  - (c) S2 で KDA を加える。
  - 各段で、速さと第 6 節の品質を測る。
- 利点
  - 活性を量子化しないので、sm_121 の CUTLASS FP8 の報告（#40934、#40758、#43367）を避けられる。
  - 合成した層のスケールの問題もない。
- 次の手: `--linear-backend` で CUTLASS の W8A8 と比べる。
- リスク
  - Marlin FP8 の M=1 での実効の帯域が未確認（悲観の場合 S2 は約 17 tok/s）。
  - KDA の FP8 の品質の公表値がない。
  - glm5next の修正を保守し続ける負担。
  - RedHatAI の BF16 のテンソルが元の BF16 と同じかが未確認。

### 代わりの案: NVFP4A16（重みだけ NVFP4、Marlin NVFP4）。lm_head は FP8

- S3 の約 22〜25 tok/s を狙う。
- 本命と同じく較正は要らず、今の重みから作れる。
- リスク
  - 4 bit の誤差は、FP8 より大きい。KDA の再帰の状態への影響が、最も心配な点である。
  - 合成した射影では、全体のスケールを共有させる必要がある。
  - Marlin NVFP4 の、行数の少ない分割（KDA の b は 1 台あたり 32 行）での制約が未確認。
  - W4A4 にしない理由: sm_121 の FlashInfer FP4 の報告（flashinfer #2577）と、較正が要ることの 2 つ。
- 本命の S2 で品質が保てたのに速さが足りないときに進む。

## データソース

| # | ソース | 種別 | 信頼度 |
|---|---|---|---|
| W1 | https://huggingface.co/zai-org/GLM-5.3-Flash （config.json、model.safetensors.index.json、LICENSE） | HF 一次 | High（構成の値のうち q/kv_lora は、取り出した道具の値と vLLM の既定値が同じことだけで照合した） |
| W2 | https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4 （config.json、README） | HF 一次 | High |
| W3 | https://github.com/vllm-project/llm-compressor | GitHub 一次 | High |
| W4 | https://github.com/vllm-project/compressed-tensors | GitHub 一次 | High |
| W5 | https://developers.redhat.com/articles/2026/01/16/llm-compressor-090-attention-quantization-mxfp4-support-and-more | 企業ブログ | Medium |
| W6 | https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/ | 文書 | High |
| W7 | https://github.com/NVIDIA/Model-Optimizer （LICENSE、issue #195） | GitHub 一次 | High（#195 の決着は未確認） |
| W8 | https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4 （hf_quant_config.json、README） | HF 一次 | High |
| W9 | https://github.com/intel/auto-round | GitHub 一次 | High |
| W10 | https://huggingface.co/zai-org/GLM-5.3-Flash-BF16 | HF 一次 | High |
| W11 | https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark | HF | Medium |
| W12 | https://huggingface.co/cyankiwi/GLM-5.3-Flash-AWQ-INT4 | HF | Medium |
| W13 | arXiv 2609.04098 "Why Gated DeltaNet Survives 4-Bit Quantization" | 論文（本文は未読、要旨による） | Medium |
| W14 | https://huggingface.co/RedHatAI/Kimi-K2.6-NVFP4 、nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 | HF | Medium |
| W15 | vLLM の issue #40934、#40758、#43367、#43507、#43906、#55397、PR #52708。flashinfer の #2577、#2776、#4990。eugr/spark-vllm-docker #143（https://github.com/vllm-project/vllm/issues/ 以下ほか） | GitHub | Medium（題と要旨を確かめた。再現はしていない） |
| S1 | `src/`（vLLM 0961bbae）の本文中に挙げた各ファイルと行 | コードベース | High |
| S2 | `docs/results/2026-09-26-k2-profile-full.md`、`2026-09-25-k1-mtp1.md`、`2026-09-25-k1-n-sweep.md`、`2026-09-22-nope-build.md`、`serving/var/20260925T220240Z-logs-p2-nope-tp2-full-prof/head/container.stdout.log` | このリポジトリ | High |

## 残っている問い

1. 変換をどこで行うか。184 GB の重みを読み書きする場所として、Mac、Spark、別の機械のどれを使うか。takt の作業は Spark を使わない。
2. RedHatAI の BF16 のテンソルが、`zai-org/GLM-5.3-Flash-BF16` と同じか（ハッシュを比べる）。
3. Marlin FP8 と Marlin NVFP4 の、M=1 での実効の帯域（GB10 の実機で、線形層 1 つのベンチを取る）。
4. KDA の射影を量子化したときの、長い文脈（128k の needle、agent の 20k〜120k）での品質。
5. ModelOpt の issue #195 の決着（ModelOpt を使う場合だけ）。
6. glm5next を修正するか、上流へ提案するか（今の上流の main の状態は未確認）。
