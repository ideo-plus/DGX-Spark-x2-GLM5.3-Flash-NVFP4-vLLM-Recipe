# K3（modal-labs の DFlash）の実現性の調査

日付は 2026-09-27 JST。[ADR 0006](../decisions/0006-path-pruning.md) の K3 と、[道筋の調査](2026-09-25-path-survey.md) の未決（R-G5、R-G7）を確かめた。
読んだのは、Hugging Face の API と公開ファイル、GitHub の API、固定した vLLM の木（`serving/var/nope-build-0961bbae/source`、以下 `src/`）だけである。実機、SSH、Docker は使っていない。
candidate D と `exl3-tp2` の資料は開いていない。ほかのレシピのスクリプトや設定は写していない。

記号: 【事実】は出典つき、【計算】は数字からの算出、【推測】は根拠の弱い見込み。

## 結論

- 【事実】GLM-5.3-Flash 向けの DFlash 系ドラフターの公開の受理長は、どれも 3.3〜3.7 にとどまる（下の §4）。うちの MTP N = 3 の code/en の受理長 3.69 と同じ水準である。
- 【事実】固定した vLLM（0961bbae）には DFlash と DFlash2 の実装があるが、GLM-5.3-Flash の本体（`Glm5NextForCausalLM`）が `SupportsEagle3` を実装していないため、起動時に落ちる。上流の対応の PR 2 件はどちらも open である。
- 【計算・推測】1 ステップを 90〜140 ms と見ると、60 tok/s には受理長 5.4〜8.4 が要る。公開値の 3.3〜3.7 では 24〜40 tok/s で、今の 42 tok/s より遅くなりうる。
- 推奨は **条件付き（今は no-go 寄り）**。移植に手をかける前に、今の構成のまま「下書き 7 トークンを検証する 1 ステップの時間」を ngram で測り、必要な受理長を確定させる（§6）。

## 1. modal-labs/GLM-5.3-Flash-DFlash の中身とライセンス

| 項目 | 内容 | 出典 |
|---|---|---|
| リポジトリ | `modal-labs/GLM-5.3-Flash-DFlash`（作成 2026-09-20、gated なし、DL 474） | 【事実】https://huggingface.co/api/models/modal-labs/GLM-5.3-Flash-DFlash |
| 方式 | ブロック拡散（block diffusion）のドラフター。単体の言語モデルではない | 【事実】README 20〜26 行（https://huggingface.co/modal-labs/GLM-5.3-Flash-DFlash/blob/main/README.md） |
| アーキテクチャ | `architectures: ["DFlash2DraftModel"]`、`model_type: qwen3`。6 層、hidden 4096、intermediate 12288、32 ヘッド／KV 8、全層 sliding window 4096。`block_size: 8`、`target_layer_ids: [23,27,31,35,39,43]`、候補の選択器（`selector_rank 256`、`top_k 16`）と grouped conv を持つ | 【事実】config.json（同リポジトリ） |
| 大きさ | 1.389B パラメーター、BF16、`model.safetensors` 2,778,461,544 B（約 2.8 GB）。embed と lm_head は持たない（本体のものを使う） | 【事実】safetensors のヘッダーと `x-linked-size` を取得して集計 |
| 動かし方 | README は SGLang main の `--speculative-algorithm DFLASH`、`--speculative-dflash-block-size 8`、TP=4 の例だけを載せる。「draft は量子化しないこと（受理長が下がる）」 | 【事実】README 30〜47 行 |
| 受理長の公開値 | 載っていない | 【事実】README 全文 |
| 学習データ | 書かれていない | 【事実】README 全文 |
| ライセンス | card の `license: mit`。README は「MIT License, inherited from the target model」 | 【事実】README 6、51 行 |
| LICENSE の文面 | 標準の MIT の文面。著作権の行は `Copyright (c) 2026 Z.AI Co., Ltd` | 【事実】https://huggingface.co/modal-labs/GLM-5.3-Flash-DFlash/blob/main/LICENSE 3 行 |
| 本体のライセンス | `zai-org/GLM-5.3-Flash` の LICENSE も MIT で、著作権の行は同じ `Copyright (c) 2026 Z.AI Co., Ltd` | 【事実】https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE |

- 【事実】modal-labs は、ほかのドラフターでも本体の LICENSE をそのまま置いている。`GLM-5.3-NVFP4-DFlash` は「Z.ai's GLM-5.3 License, inherited from the target model」、`Kimi-K3-DFlash` は `Kimi K3 License / Copyright (c) 2026 Moonshot AI`。
  → 著作権の行が Z.AI なのは、本体の LICENSE を写した結果と読める【推測】。modal-labs 自身の著作権の表示や、別の利用規約（AUP・非商用の条項）は見当たらない【事実】。
- 商用の可否: MIT の文面には、商用の制限も利用の用途の制限もない【事実】。
  ただし次は未確定で、判断は対話側に残す。
  1. modal-labs が自分の学習した重みに MIT を付ける意思があるか（著作権者の欄が Z.AI のままで、modal の名がない）。
  2. 学習データ（他社モデルの出力を含むか、その規約）が非公開であること。
  3. 構造は `DFlash2DraftModel` で、非商用の `incoai/GLM-5.3-Flash-DFlash2` と同じ系統の構造である。構造（コード）が同じでも重みの写しとは限らない。重みの由来は確かめられない（前回の調査は「写しではない」としたが、根拠は本調査の範囲では再確認していない）。

## 2. ほかのドラフター（GLM-5.3-Flash 向け）

Hugging Face の検索（`search=GLM-5.3-Flash`、`search=dflash`、2026-09-27）で見つかったもの。

| リポジトリ | 方式・大きさ | ライセンス | 学習データ | 使えるか |
|---|---|---|---|---|
| `RedHatAI/GLM-5.3-Flash-speculator.dspark-preview` | DSpark（DFlash にマルコフ頭と確信度の頭を足したもの）。5 層、ブロック 8、BF16 約 5.1 GB。preview（3 エポック中の 2 エポック目） | MIT | 公開。Open PerfectBlend（Apache-2.0）の問いを GLM-5.3-Flash 自身で生成し直したもの | 【事実】ライセンス上は通る見込み。https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview |
| `canada-quant/GLM-5.3-Flash-DFlash2-E`（後継 `-F`、`-G`） | DFlash2。8 層、全層 full attention、約 1.84B・6.2 GB。W4A16 の本体に合わせて学習 | Apache-2.0 | 公開。公開の問い（MIT・Apache-2.0・CC-BY-4.0）に本体が答えを生成。incoai の重みは使っていないと明記 | 【事実】ライセンス上は通る見込み。「上流のままの DFlash2 のイメージでは起動できない」と自ら書いている。https://huggingface.co/canada-quant/GLM-5.3-Flash-DFlash2-E |
| `incoai/GLM-5.3-Flash-DFlash2` | DFlash2、5 層 | CC-BY-NC-ND-4.0 | — | **使えない**（非商用）【事実】 |
| `Solstice-AI/...-NVFP4-DFlash2`、`Coder40-95/...-DFlash2-TR3-v3` | DFlash2 の派生 | card にライセンスの記載なし・未確認 | — | 由来が不明なので使わない【推測】 |
| EAGLE-3、Medusa | — | — | — | GLM-5.3-Flash 向けは見つからなかった【事実】 |

## 3. vLLM の対応

### 上流の PR（GitHub API、2026-09-27 取得）

- 【事実】#56983「[Spec Decode][Model] Support DFlash2 draft models with GLM-5.3-Flash」は **open**、未マージ（https://github.com/vllm-project/vllm/pull/56983）。
  - 本文は `main` で起動を妨げる障害を 3 つ挙げる。(1) `Glm5NextForCausalLM` に `SupportsEagle3` がない、(2) draft の sliding window 層の KV のページが GLM の KV の配置と合わない、(3) 密な draft に専門家並列の設定が写される。
  - 2026-09-25 に衝突の指摘があり、2026-09-27 に作者が「KV のグループ分けの変更を切り出す」ことに同意した。
- 【事実】#55682「[Spec Decode] GLM-5.3-Flash DFlash2 aux hidden-state capture」も **open**。最終更新は 2026-09-21（https://github.com/vllm-project/vllm/pull/55682）。KV の配置は別の変更に任せている。
- 【事実】関連の PR は #55423（draft、要 rebase）、#56933（EP の修正）、#55219（KV の配置）、#54451（ROCm）。「GLM-5.3-Flash dspark」で探した PR はなかった。

### 固定した vLLM（0961bbae、2026-09-22）

- 【事実】方式 `dflash` はある: `src/vllm/config/speculative.py:69-72`。model のパスに `dflash` を含めば自動で `dflash` になる: 同 `:1319-1335`。K の追加の枠は K: 同 `:1896-1908`。
- 【事実】`DFlash2DraftModel` は登録済み（`src/vllm/model_executor/models/registry.py:635-636`）。V2 の model runner だけが DFlash2 の選択器を使う: `src/vllm/v1/worker/gpu/spec_decode/__init__.py:18-23`、V1 では「dflash2 drafts」を非対応として扱う: `src/vllm/config/vllm.py:2976-2980`。V2 は既定で選ばれる（非対応の機能がなければ）: 同 `:698-745`。
- 【事実】**GLM-5.3-Flash では落ちる。** DFlash は本体の隠れ状態を取り出すので、`set_eagle3_aux_hidden_state_layers` を呼ぶ（`src/vllm/v1/worker/gpu/model_runner.py:389-391`）。本体が `SupportsEagle3` でないと `RuntimeError` になる（`src/vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:15-20`。V1 は `src/vllm/v1/worker/gpu_model_runner.py:5435-5441`）。
  - `Glm5NextForCausalLM` の基底は `HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid` だけ（`src/vllm/models/glm5next/common/model.py:933-934`）。うちの重ね合わせ（`serving/payload/vllm-overlay/k2s2b/.../model.py:935-936`）も同じ。
  - #56983 の障害 (2)（KV の配置）と (3)（EP）も、0961bbae には未修正と見る【推測】（該当の PR が未マージのため）。
- 要る設定の形（【事実】#56983 の Test Plan の例）: `--speculative-config '{"method":"dflash","model":"<draft のパス>","num_speculative_tokens":6}'`、`VLLM_USE_V2_MODEL_RUNNER=1`。modal の draft はブロック 8 なので K は最大 7【推測】。
- 【事実】うちの `serving/` は構成の投機の指定を試験で縛っている（`serving/tests/unit/test_config_committed.py:391-400`、`test_configure_tp2.py`）。DFlash を足すときは、構成の定義と試験を同時に更新する必要がある。

## 4. 公開の受理長と速さ

| 対象 | 条件 | 受理長 | 速さ | 出典 |
|---|---|---|---|---|
| incoai DFlash2（非商用、参考のみ） | H100×8、TP8+EP、FP8、greedy、code と prose | k=6 で 3.31、k=8 で 3.33、k=16 で 3.63。MTP k=5 は 3.13 | 1 本で 218.5 tok/s（MTP k=5 は 160.7、+40%） | 【事実】#56983 本文の Performance |
| 同上 | 位置ごとの受理率 | 0.77, 0.55, 0.39, 0.28, 0.19, 0.13（k=6） | — | 【事実】同上 |
| RedHat DSpark preview | TP4、温度 0、K=8 | HumanEval 4.74、math 6.12、qa 3.11、tool_call 3.40、writing 3.13、加重平均 3.77 | — | 【事実】RedHat の card の Acceptance Results |
| canada-quant DFlash2-E | H200 TP4、thinking ON、T=1.0、K=7 | 3.57〜3.59（incoai 3.60） | c1 267.2 tok/s | 【事実】canada-quant の card、PROVENANCE.txt |
| 同上（**GB10 2 台**） | 2× DGX Spark、TP=2、eager、W4A16、K=7 | c1 3.60（incoai 3.58）、smoke 4.01（incoai 4.07） | c1 28.1 tok/s（incoai 29.0）、smoke 29.8（incoai 31.2） | 【事実】canada-quant の card の Same-rig A/B |
| canada-quant の -G | B300、同じ手順 | 3.676 | — | 【事実】canada-quant の card の冒頭 |
| modal の Kimi-K3 版 | B300 | 3.46〜6.01 | — | 【事実】前回の調査（2026-09-25-path-survey.md:107） |
| DFlash の論文 | 各種モデル | — | 「6 倍超」「EAGLE-3 の最大 2.5 倍」 | 【事実】https://arxiv.org/abs/2602.06036 の要旨 |

- 【計算】canada-quant の GB10 の c1 の 1 ステップは 3.597 ÷ 28.12 = 128 ms（incoai は 123 ms）。うちの MTP N = 3 は 3.69 ÷ 42.449 = 87 ms。重みと eager の違いがあり、直接は比べられない。
- 【事実】GLM-5.3-Flash での DFlash2 の H100 の利得は、受理長の伸び（3.13→3.31）ではなく、下書きが並列で安いことから来ている（#56983 の表）。

## 5. うちで試すときの見積もり

### 要るもの

- ダウンロード: modal の draft 約 2.8 GB【事実】。RedHat は約 5.1 GB、canada-quant は約 6.2 GB【事実】。
- 重みの形式: draft は BF16 のまま使う前提（modal の README）。本体の NVFP4・FP8 と別に読み込むので、形式の変換は要らない【推測】。
  ただし draft は本体の隠れ状態（層 23〜43）を入力にして学習されている。うちの `k2s2b`（アテンションまで量子化）では入力の分布がずれ、受理長が下がるおそれがある【推測】。canada-quant は W4A16 の本体に合わせて学習し、H200 と GB10 で受理長がほぼ同じだった（+0.3%）【事実】。
- メモリ: draft は TP=2 で割れば 1 台あたり約 1.4 GB【計算】。#56983 の設計では、KV の容量が MTP より 24.6% 減った（H100）【事実】。うちの N = 3 の KV は 1,178,387 トークンあり、1 本の計測では足りる【推測】。
- vLLM の変更: (1) glm5next に `SupportsEagle3` と隠れ状態の取り出し（mHC の `hc_post` を完了させてから `hc_contract`。単純な `hidden + residual` では誤り）、(2) sliding window の draft の KV の配置、(3) EP の修正（うちが EP を使っていなければ不要【推測】）。
  いずれも Python の変更で、今の重ね合わせ（bind mount）の仕組みで載せられる見込み【推測】。

### 速さの見込み

- 【計算】うちの `k2s2b` の 1 ステップ: 投機なし 42〜49 ms、MTP N = 2 は 73 ms、N = 3 は 87 ms、N = 4 は 100 ms（下書き 1 つで約 13〜14 ms 増える）。
- 【推測】DFlash の K=7 の 1 ステップは 90〜140 ms と見る。下書きは 1 回の前向き計算で済む（GB10 の公称の帯域 273 GB/s で 1.4 GB を読むと約 5 ms）。一方、検証は 8 トークンになり、読む専門家が増える。下限は専門家の和集合が頭打ちになる場合、上限は MTP の増え方がそのまま続く場合と、canada-quant の 123〜128 ms。

| 1 ステップ（推測） | 受理長 3.3 | 3.6 | 4.74 | 6.0 | 60 tok/s に要る受理長 |
|---|---:|---:|---:|---:|---:|
| 90 ms | 36.7 | 40.0 | 52.7 | 66.7 | 5.4 |
| 105 ms | 31.4 | 34.3 | 45.1 | 57.1 | 6.3 |
| 120 ms | 27.5 | 30.0 | 39.5 | 50.0 | 7.2 |
| 140 ms | 23.6 | 25.7 | 33.9 | 42.9 | 8.4 |

（値は tok/s、【計算】= 受理長 ÷ 1 ステップ。ブロック 8 では受理長の上限は 8。）

- 【推測】公開の汎用の受理長（3.3〜3.7）では、今の 42 tok/s を超えない。60 tok/s には、受理長 5.4 以上と 1 ステップ 90 ms 前後の両方が要る。公開値でこれに近いのは RedHat DSpark の math（6.12）だけで、HumanEval（4.74）でも足りない。

### リスク

1. 受理長の伸びが公開値で裏づけられない（上のとおり）。
2. vLLM の対応が未マージで、KV の配置の議論は続いている（#56983 のコメント、2026-09-27）。自前で重ねると保守の負担が増える。
3. 量子化した本体（`k2s2b`）で受理長が下がるおそれ【推測】。
4. GB10 での DFlash2 の公開の実績は canada-quant の 1 件だけで、それも patched な版と eager での値【事実】。
5. ライセンス: modal は著作権者の欄と学習データが未確定（§1）。RedHat は preview 版。

## 6. 推奨と次の一手

**判定: 条件付き（今は no-go 寄り）。** modal の DFlash を vLLM に移植する作業には、まだ着手しない。

最小の次の一手（実機、対話側。今の構成と重みのまま、新しい重みは要らない）:

1. **検証の費用を測る（ngram の探り）。** `{"method":"ngram","num_speculative_tokens":7,"prompt_lookup_max":4,"prompt_lookup_min":2}` で、写しの多い問い（同じコードを繰り返させる等）を 1 本流し、`/metrics` の受理長と tok/s から 1 ステップを出す。
   - ngram は本体の隠れ状態を使わないので、glm5next の今の実装で動く見込み【推測】（`SupportsEagle3` を要るのは eagle3・dflash の経路のみ）。
   - 【事実】ngram は V2 の runner の非対応の方式に入っており、指定すると V1 の runner に落ちる（`src/vllm/config/vllm.py:2924-2936`）。V1 と V2 で 1 ステップの時間が違いうるので、同じ探りを MTP N = 3 でも V1 で流し、差を補正する【推測】。
   - 8 トークンの検証の 1 ステップが分かれば、60 tok/s に要る受理長が決まる。**要る受理長が 5.5 を超えたら K3 を切る**（ADR 0006 の「受理長が 5 未満なら切る」を、測った 1 ステップで置き換える）。
2. 1 で通ったときだけ、受理長を測る。候補は、学習データが公開で MIT の RedHat DSpark と、modal の DFlash。どちらも glm5next の隠れ状態の取り出し（#55682 相当）が前提。#56983 のモデル側の変更がマージされるのを待つのが安い。
3. ライセンスの判断（modal の著作権者の欄と学習データ、RedHat の preview の扱い）は対話側で行う。

## 未確定のこと

- modal-labs が自分の重みに MIT を付けた意思、学習データ、重みの由来（incoai の重みとの関係）。
- 0961bbae で ngram の K=7 が glm5next と V1 runner で実際に動くか、V1 と V2 で 1 ステップがどれだけ違うか（コードの読みだけで、未実行）。
- うちの計測の条件（温度・thinking）で、公開の受理長（温度 0 または T=1.0）がどの程度そのまま当てはまるか。
- 量子化した本体（`k2s2b`）での DFlash の受理長の低下の大きさ。
- canada-quant の GB10 の値が CUDA グラフありでどうなるか（card には eager の A/B だけが表で載る）。
