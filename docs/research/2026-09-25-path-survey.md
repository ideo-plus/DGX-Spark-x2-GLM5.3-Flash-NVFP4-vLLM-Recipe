# 調査レポート: Issue #28 GB10（DGX Spark）2 台で GLM-5.3-Flash を動かす道筋の比較と枝切り

## 調査概要

GB10 2 台（または 1 台）で GLM-5.3-Flash を動かす道筋を、公開のレシピと一次資料の事実から比べた。比べた軸は、エンジン、重みの形式、カーネル、投機的デコード、ライセンスの 5 つ。そのうえで、探索の範囲を 3 本に絞った。

あわせて、今の約 14 tok/s と candidate D の 63〜66 tok/s の差を、要因ごとに推定した。

- 調べた日: 2026-09-25
- 実機の操作と、リポジトリのファイルの変更はしていない。
- 印の意味
  - 【事実】: 出典で確かめたこと
  - 【計算】: 事実の数値からの算術
  - 【推測】: 事実からの推定

## 主要な発見

- **投機なしの生成速度（基礎の速さ）は、エンジンと形式によらず 14〜15 tok/s にそろう。**【事実】
  - vLLM は 14.3〜15.1、SGLang は 14.2〜14.74 tok/s。
  - 今の構成は code/en 14.162、prose/ja 13.933 tok/s（`docs/results/2026-09-23-decode-gpu-clock-cap.md`）。
- **公開されている 2 台の vLLM + EXL3 の例は、どれも「専門家だけ EXL3」の重みを使っている。**【事実】
  - アテンションは BF16 のまま読んでいる。
  - 1 台が 1 トークンで読む量は、NVFP4 の 11.02 GB に対して 9.40 GB で、約 15% しか違わない【計算】。
  - だから、EXL3 と NVFP4 で速さはほとんど変わらない。
- **上流の vLLM の main は、GLM-5.3-Flash のアテンション（KDA と MLA）を常に BF16 で読む。**【事実】
  - アテンションを量子化する PR と issue は見つからなかった。
  - 一方、同じ vLLM でも Kimi K3 の経路は、KDA の射影を FP8 で動かす。
- **モデル付属の MTP（MIT）の公開値は、GB10 2 台 TP=2 で 20〜31 tok/s。**【事実】
  - 伸びは投機なしの 1.4〜2.1 倍。
  - GLM では、MTP の 1 ステップの時間が投機なしの 1.6〜2.0 倍になる【計算】。このため、下書きを深くしても伸びが打ち消されやすい。
- **ライセンスの通る深い下書き器は、modal-labs の DFlash（MIT）1 本だけ見つかった。**
  - DFlash2（CC BY-NC-ND）の重みの写しではない。
  - ただし、学習データは公表されていない。LICENSE の著作権者の欄は Z.AI になっている。
  - GLM-5.3-Flash と組み合わせて動かせる上流の実行系は、今は SGLang だけ。その SGLang は、上流のままでは GB10 で GLM-5.3-Flash を起動できない。
- **ライセンスの通る部品だけで code/en の 60 tok/s に届く根拠は、公開の事実からは得られなかった。**
  - 最も見込みの高い組み合わせでも 40〜58 tok/s【推測】。アテンションまで量子化し、modal の DFlash を足す組み合わせ。
  - prose/ja の 30 tok/s には、アテンションを量子化して MTP を足せば届く見込みがある【推測】。
- **14 と 63〜66 の差の出どころは決着していない。** 2 つの仮説が残る（後述）。決め手になるのは実機の計測と、対話側が `results/` を読むこと。

## 調査結果

### 1. 道筋ごとの比較表

- 速さは同時 1 本の値。GB10 2 台 TP=2 で測ったもの（R10 だけ 1 台）。条件は出典ごとにそろっていない。
- 非商用のドラフター（DFlash2）を使った値は載せない。

| ID | エンジン | 形式 | MTP | GB10 での実績 | 公表の速さ（tok/s） | ライセンス | 必要な修正 | 主な出典（日付） |
|---|---|---|---|---|---|---|---|---|
| R1 | vLLM | NVFP4（専門家だけ） | 使える（main） | あり（10 件以上） | 投機なし 14.3〜15.1。MTP で 20〜31 | Apache-2.0、MIT、BSD。CuTe DSL は要判断 | あり。NoPE の修正は上流の main に入った（#55277、2026-09-23）。リリース（v0.30.0、v0.30.1rc0）にはまだ入っていない。修正なしの main が GB10 で動いた報告は未発見 | NVIDIA Forums 381433（2026-08-27）、382041（2026-09-02）、H-K-B（2026-09-12〜16）、technigmaai（2026-09-25 取得）、https://github.com/vllm-project/vllm/pull/55277 |
| R1-Q | vLLM | 専門家は NVFP4、アテンション・共有の専門家・dense・lm_head は FP8 か 4 bit | 使える | なし | 公表なし（見込み: 投機なし 19〜23、MTP で 30〜43【推測】） | 同上 | あり（自前）。glm5next の量子化の扱いを変える。重みは自分で量子化する | vLLM main の glm5next のソース、計算 |
| R2 | vLLM | W4A16（Marlin） | 使える | あり（2 件） | MTP-2 で prose 26.0 | 同上。重みは MIT | R1 と同じ | NVIDIA Forums 382041、382632 |
| R3 | vLLM ＋ 上流の外の EXL3 プラグイン | EXL3（公開例は専門家だけ） | 使える | あり（フォークと overlay） | MTP-3 で 20〜24 | 主流のプラグイン（vcruz305/vllm-exl3）は AGPL-3.0 で使えない。Apache-2.0 のもの（yeasah、vllm-mach）もある。重みの一部は `shapleymcg-license-1.0` で要判断 | あり（フォーク）。上流の vLLM は EXL3 に対応しない（#19896 は not_planned） | NVIDIA Forums 382120（2026-09-02〜11）、https://github.com/vllm-project/vllm/issues/19896 |
| R4 | vLLM | NVFP4、PP=2 | — | なし | — | — | GLM-5.3-Flash の PP は main で止められている。有効にする PR #57171 は open | https://github.com/vllm-project/vllm/pull/57171 |
| R5 | SGLang v0.5.20 | NVFP4 | 使える（NEXTN）。DFlash も使える | あり（修正あり） | 投機なし 14.2〜14.74。MTP-3 で code 25.75、prose 23.51 | Apache-2.0。mathdx と CuTe DSL は要判断 | あり。上流のままでは GB10 で起動できない。DSA のバックエンドがどれも使えない（#40286）。第三者はカーネルのタイルの縮小など 4〜6 種の修正で起動した。#38430 は open | mazurov.dev（2026-09-07）、https://github.com/sgl-project/sglang/issues/40286 （2026-09-19）、#36599 |
| R6 | TensorRT-LLM | NVFP4 | PR の中だけ | なし | — | 本体は Apache-2.0。同梱のバイナリと tileiras は要判断 | GLM-5.3-Flash の PR #19136 は open。sm_121 の経路がない | TensorRT-LLM #19136 |
| R7 | llama.cpp（2 台、RPC） | GGUF | PR の中だけ | 2 台の実績なし | — | MIT | PR のブランチが要る。3 本の PR がどれも GLM5NEXT をテンソル分割の対象外にしているので、2 台は層で分ける方式だけになる | llama.cpp #27752、#27754、#27773 |
| R8 | ExLlamaV3 単体 | EXL3（アテンションまで量子化） | 使える（下書き 3） | 2 台はない | — | MIT。TabbyAPI は AGPL で使えない。MIT のサーバーもある | 複数ノードの TP が上流にない。aarch64 のビルドの修正（#393）は dev だけにある | exllamav3 #151（7 か月回答なし） |
| R9 | ktransformers | FP8 ＋ CPU | — | 動かない | — | Apache-2.0 | x86 の AVX-512 と 350 GB 以上のメモリが前提 | ktransformers の手順、#1883 |
| R10 | llama.cpp（1 台） | GGUF 約 2.7 bpw | 使える（PR） | あり | 投機なし 17.7。MTP で 15.4〜34 | MIT | PR のブランチが要る。CUDA 13.0 に固定 | DevelopersIO（2026-08-30）、ai-muninn（2026-08-30） |
| R11 | vLLM / SGLang | MXFP4（amd Quark、OneNexus、INCModel3） | 未確認 | **未発見** | — | 重みは MIT | vLLM の Triton MXFP4 MoE は、sm_121 で使えない PTX を使う（#41477）。SGLang の GLM-5.3-Flash の MXFP4 は AMD gfx950 向け（#38546） | HF の Hub API（2026-08-27〜31 公開）、https://github.com/vllm-project/vllm/issues/41477 |

参考（GB10 ではない）: vLLM main ＋ #55277 は、RTX PRO 6000 ×4、TP=4、投機なし、同時 1 本で 99.09 tok/s（TPOT 8.80 ms）【事実】。

### 2. 重みの形式ごとの状況

| 形式 | 1 要素あたりのビット（スケール込み） | GB10 2 台 TP=2 | 要点 |
|---|---|---|---|
| NVFP4 | 約 4.5 | 対応（R1、R5） | 公開の重みは専門家だけが NVFP4 で、アテンションは BF16 で読まれる |
| W4A16（AWQ / GPTQ） | 約 4 台 | 対応（R2） | 専門家は読む量の 22% にとどまるので、カーネルを替えても効かない【計算】 |
| EXL3 | 3〜4 bpw ほか | 上流の外のプラグインで対応（R3） | 2 系統ある【事実】。(a) ExLlamaV3 本体の変換は、アテンションまで量子化する。(b) 別の道具は、専門家だけを量子化する（config に `glm53_routed_experts_only` と書いてある）。2 台の公開例は (b) だけ |
| GGUF | 約 2.7〜4.5 | 2 台は層で分ける方式だけ（R7） | 1 台なら約 2.7 bpw で載る（R10）。品質は KLD 0.38（UD-Q2_K_XL） |
| MXFP4 | 4.25 | 実績なし（R11） | NVFP4 との差は、専門家の読む量で約 6%、全体ではそれ以下【計算】 |

アテンションまで量子化した場合の見込み【計算・推測】:

- 数値は TP=2 の 1 台のもの（`attention-quant-bytes-calc-output-20260925.txt`）。
- 帯域は 234 GB/s とした。
- 読み出し以外の時間は 20〜23 ms と仮定した。

| シナリオ | 1 台が読む量 | 帯域だけの上限 | 投機なしの見込み |
|---|---:|---:|---:|
| 今（NVFP4 は専門家だけ） | 11.02 GB | 21.2 | 14.2（実測） |
| 専門家だけの EXL3 4 bit | 9.40 GB | 24.9 | 約 16 |
| NVFP4-Spark（アテンションは MXFP8） | 7.47 GB | 31.3 | 約 19〜20 |
| アテンション・共有の専門家・dense・lm_head を FP8 | 6.97 GB | 33.6 | 約 19〜20 |
| KDA・MLA・共有の専門家・dense を NVFP4、lm_head を FP8 | 5.42 GB | 43.2 | 約 22〜23 |
| turboderp EXL3 4.05 bpw（アテンションは 6 bit） | 5.69 GB | 41.1 | 約 22 |
| turboderp EXL3 3.05 bpw（アテンションは 5 bit） | 4.70 GB | 49.8 | 約 24〜25 |

### 3. 投機的デコード

| 方式 | ライセンス | 使えるエンジン | GB10 2 台での公開値（同時 1 本） |
|---|---|---|---|
| モデル付属の MTP | MIT（使える） | vLLM main、SGLang（NEXTN）、ExLlamaV3、llama.cpp（PR）。TensorRT-LLM は PR の中だけ | 20〜31 tok/s |

MTP の公開値の内訳:

- NVFP4 ＋ MTP-2: prose 26〜30、code 31 tok/s。受理率は 0.61〜0.63（Forums 382041）。
- NVFP4 ＋ MTP-3: 30.5 tok/s。文脈 4,096、出力 128（technigmaai）。
- MTP-3 のときの受理長は 2.3〜2.68。
- vLLM #53969 の 21.0 tok/s は、MTP k=5 で測った値。

MTP 以外の方式:

| 方式 | ライセンス | 状態 |
|---|---|---|
| modal-labs/GLM-5.3-Flash-DFlash（ブロック 8） | MIT。学習データと LICENSE の著作権者の欄は要確認 | 動かせる上流の実行系は SGLang だけ。vLLM の main は起動時に落ちる（glm5next が `SupportsEagle3` を実装していない）。対応の PR #56983 と #55682 は open。GLM-5.3-Flash での受理長の公開値はない（同じ modal の Kimi-K3 は、B300 で 3.46〜6.01）。#56983 の設計では、KV の容量が MTP より 24.6% 少なくなる |
| DFlash2（incoai） | CC BY-NC-ND | **使えない** |
| n-gram、suffix | Apache-2.0 | 公開値なし。ハイブリッドのモデルで出力が壊れる issue（#39273）が open |
| EAGLE3、Medusa | — | GLM-5.3-Flash 向けのものは未発見 |

vLLM の MTP には、既知の問題が 2 つある。

- #58454（open）: 下書きが 2 以上で、文脈が 2048 を超えると kpool が壊れる。
- #57087: 同時処理で非 ASCII の出力が壊れる。

### 4. ライセンスの判定

| 判定 | 部品 |
|---|---|
| 使える | vLLM（Apache-2.0）、SGLang（Apache-2.0）、TensorRT-LLM の本体（Apache-2.0）、llama.cpp / ggml（MIT）、ExLlamaV3（MIT）、cuda-exl3（LICENSE の本文は MIT）、ktransformers（Apache-2.0）、GLM-5.3-Flash の重みと MTP（MIT）、RedHatAI・NVIDIA の NVFP4 と MXFP4 の各重み（MIT）、yeasah/vllm-exl3-plugin（Apache-2.0） |
| 使えない | DFlash2（CC BY-NC-ND）、TabbyAPI（AGPL）、vcruz305/vllm-exl3（AGPL-3.0） |
| 要判断 | CuTe DSL（NVIDIA の EULA。vLLM と SGLang の全道筋に関わる）、mathdx、TensorRT-LLM に同梱のバイナリと tileiras、`shapleymcg-license-1.0` の重み、modal の DFlash の学習データと LICENSE の著作権者の欄 |

### 5. 14 tok/s と 63〜66 tok/s の差の説明

要因ごとの効き方:

| 要因 | 効きそうな大きさ | 根拠 | 印 |
|---|---|---|---|
| エンジン（vLLM か SGLang か） | ほぼ 0 | 投機なしで 14.3〜15.1 と 14.2〜14.74 | 事実（条件は完全にはそろっていない） |
| MoE のカーネル（Marlin か FlashInfer CUTLASS か） | ほぼ 0 | 14.3〜14.6 と 14.16。専門家は読むバイトの 22% | 事実と計算 |
| 形式（専門家だけの EXL3 と NVFP4） | 読む量で約 15% | 9.40 GB と 11.02 GB。MTP-3 の公開値も 20〜24 と 20〜26.5 で同じ範囲 | 計算と事実 |
| 形式（アテンションまで量子化） | 投機なしで 19〜28（今の 1.35〜2 倍） | 上の表。読み出し以外の約 20 ms が残るとの仮定 | 推測 |
| TP の通信 | 今は 3〜6%。60 tok/s の世界では 14〜25% | all-reduce 92 回 × 25〜45 µs（個人のブログの遅延） | 推測 |
| 読み出し以外の時間（計算、起動、同期） | 1 トークンあたり約 20〜23 ms（今の 70.6 ms の約 3 割） | 差し引き。Qwen3.8-Flash-Next でも 16〜21 ms。帯域の効率は GLM の TP=2 で 68%、Qwen の 1 台で 71〜72% | 計算（プロファイラーの値ではない） |
| MTP | 1.4〜2.1 倍 | 公開値 20〜31 | 事実 |

candidate D について分かっていること【事実】:

- 形式は EXL3。公表値は code/en 63〜66、prose/ja 23〜25（`PLAN.md` §3）。
- 公開値で 60 前後以上の報告は、確認できた範囲ではすべて DFlash2（非商用）を使っている。
- ライセンスの通る部品だけで 60 に届いた公開例はない。
- `exl3-tp2` の平均の受理長は 2.316〜2.504 で、MTP の範囲にある（外から測った値。ADR 0001）。
- candidate D の作成者のものは、第三者の検索結果に出た GitHub のリポジトリと issue の題（メタデータ）だけを見た。本文は開いていない。
  - 題から読み取れるのは、vLLM と cuda-exl3、EXL3 4 bpw、2 ノード TP=2 であることと、MTP の語が含まれることだけ。
  - 題の中の重みの名前と同じ名前の公開の重み（brandonmusic の tr3-4bpw）は、専門家だけの EXL3 である。

差の説明は、次の 2 つの仮説が決着していない【推測】。

| 仮説 | 内容 | 合う事実 | 合わない事実・弱い点 |
|---|---|---|---|
| H-a: 深い下書き器 | 基礎の速さは今と同じ水準（15〜17 程度）。深い下書き器（おそらく非商用）で約 4 倍に伸ばし、当たりやすい code の文面で測った | 60 前後以上の公開の報告はすべて DFlash2。prose/ja の 23〜25 は MTP の prose の範囲（19.6〜30）にも入る | 題のメタデータには MTP の語がある。`exl3-tp2` の受理長は MTP の範囲 |
| H-b: 同じ MTP で、計測の条件が違う | 投機は MTP。63〜66 は `bench/` と違う条件か、違う指標で測った値 | 題のメタデータに MTP の語がある。`exl3-tp2` の受理長は 2.3〜2.5 | 基礎が 14〜17 で MTP が 1.4〜2.1 倍なら、同じ条件で 63〜66 には届かない。この仮説を採るなら、条件の違いがかなり大きいことになる |

補足:

- `PLAN.md` L47 の列名は「既存のレシピで測った値」で、`bench/` で測った値かは未確認。
- `docs/vllm-baseline/initial-benchmark-procedure.md` L80 では、candidate D との比較はまだ行われていない。
- どちらの仮説でも、「形式（EXL3）そのものが 4 倍を生んだ」という説明は、公開の事実と合わない。

### 6. 前の段階の数値の訂正

dig 1 回目の数値のうち、次の 3 件は誤りか条件の抜けだった。この報告では訂正した値で扱っている。

1. DeepSeek-V4-Flash の 41〜66 tok/s は、投機ありの値だった。
2. Qwen3.8-Flash-Next の 51.9 と 41.7 は、MTP-3 の値だった。
3. NVIDIA Forums 381703 の EXL3 + vLLM の値は、非商用のドラフターを使った値とみられる。このため数値は載せない。

もう 1 つ、analysis-1 の読みを撤回した。1 台の EXL3 2.31 bpw の値（10.79 tok/s）を仮説の傍証にした読みである。重みはアテンションまで量子化されていて、実行系がそれをどう扱っているかが分からないため。

## データソース

| # | ソース | 種別 | 信頼度 |
|---|---|---|---|
| 1 | vLLM の PR と issue（#55277、#57171、#58454、#56983、#55682、#57578、#55773、#52504、#41477、#19896、#53969）、main の `vllm/models/glm5next/`、v0.29.0 と v0.30.0 のリリースノート（GitHub API、2026-09-25 取得） | Web（上流の一次資料） | High |
| 2 | SGLang の issue と PR（#40286、#39302、#38430、#36599、#38546）、v0.5.20 のリリースノート | Web（上流の一次資料） | High |
| 3 | TensorRT-LLM #19136、llama.cpp（#27752、#27754、#27773、#26610、#28967、`benches/dgx-spark/dgx-spark.md`、Discussion #16578）、ExLlamaV3（#151、#393、`model_tp.py`）、ktransformers（#1883） | Web（上流の一次資料） | High |
| 4 | Hugging Face の Hub API と各 `config.json`（zai-org、RedHatAI、NVIDIA、turboderp、local-inference-lab、cyankiwi、INCModel3、amd、OneNexus、modal-labs） | Web（一次資料のメタデータ） | High |
| 5 | GitHub API のライセンス情報、PyPI のメタデータ（`gh-api-licenses-*.json`、`pypi-licenses-20260925.json`） | Web（一次資料） | High |
| 6 | NVIDIA Developer Forums の DGX Spark のトピック（381433、381703、382041、382120、382632） | Web（個人の報告） | Medium |
| 7 | mazurov.dev、technigmaai、H-K-B、DevelopersIO、ai-muninn、Flowtivity、Dendro Logic | Web（個人・企業のブログ） | Medium〜Low（Flowtivity は Low） |
| 8 | このリポジトリの `PLAN.md`、ADR 0001・0002・0005、`docs/results/2026-09-2x-*.md`、`.kiro/specs/vllm-baseline/design.md` 6.7 | コードベース | High |
| 9 | 計算のスクリプトと出力（`gap-calc`、`attention-quant-bytes-calc`、`crosscheck-efficiency-calc`、`v4flash-bytes-calc`、`exl3-quantcfg-agg`） | 計算（Report Directory） | Medium（仮定に依存する） |

## 結論と推奨

**結論**

- 基礎の速さは、エンジン、形式、MoE のカーネルでは変わらない。変わるのは、アテンションまで量子化して読む量を減らす場合と、投機の伸びだけである。
- ライセンスの通る部品だけで code/en の 60 tok/s に届く公開の根拠はない。一方、prose/ja の 30 tok/s には届く見込みがある。

**残す 3 本（上から順に実機で測る）**

| # | 道筋 | 期待できる速さ（推測） | 手間 | リスク | 最初に実機で測ること |
|---|---|---|---|---|---|
| K1 | R1 + MTP（vLLM、今の RedHatAI の NVFP4、TP=2、MTP k=1〜5） | code/en 25〜31、prose/ja 20〜26 | 小。上流の main に MTP がある。ただし、今の `serve` は投機の指定を断る設計（design 6.7）なので、仕様の変更が要る | #58454、#57087。基準には届かない見込み | k=1/2/3/5 ごとに、code/en と prose/ja の受理長、1 ステップの時間、tok/s。「1 ステップは 1.6〜2.0 倍」を確かめる |
| K2 | R1-Q（vLLM で、アテンション・共有の専門家・dense・lm_head も FP8 か 4 bit で読む） | 投機なし 19〜23。MTP と組んで code/en 30〜43、prose/ja 27〜38 | 中〜大。glm5next の読み込みの変更（Kimi K3 の FP8 KDA の経路が前例）と、重みの自前の量子化が要る | 品質（KDA の射影を量子化したときの品質の公表値がない）。読み出し以外の約 20 ms で頭打ちになるおそれ | (1) 今の構成の 1 トークンの時間をプロファイラーで分解し、約 20 ms の中身を見る。(2) 量子化したアテンションの KLD と、品質の検査 |
| K3 | ライセンスの通る深い下書き器（modal-labs の DFlash、MIT） | K1 の上なら 20〜35。K2 の上なら 40〜58 | 中。vLLM は #56983 のマージ待ち。今すぐ使えるのは SGLang（R5）だけで、GB10 向けの修正が要る | 学習データが書かれていない。LICENSE の著作権者の欄が Z.AI。GB10 での実績がない。KV の容量が減る | code/en と prose/ja の受理長。受理長が 5 未満なら K3 を切る |

R5（SGLang）は、K3 を早く測るための条件付きの手段として扱う。vLLM の #56983 がマージされたら、R1 に寄せて R5 は切る。

**切る道筋**

| 道筋 | 理由 | 見直す条件 |
|---|---|---|
| R2 W4A16 | K1 の変種。専門家のカーネルを替えても効かない | Marlin が必要になったとき、K1 の中で選ぶ |
| R3 vLLM + EXL3 | 上流に EXL3 がない。主流のプラグインは AGPL。公開例は専門家だけの EXL3 で、読む量は 15% しか減らない。アテンションまで量子化するなら、K2 と同じ変更が要る | K2 で FP8 や NVFP4 の品質が足りず、5〜6 bit が要るとき |
| R4 PP=2 | main で止められている（#57171 は open）。2 台の PP は decode を速くしない | — |
| R6 TensorRT-LLM | PR #19136 が未マージで、sm_121 の経路がない。要判断のライセンスが最も多い | #19136 がマージされ、SM120/121 に対応したとき |
| R7 llama.cpp の 2 台 | 3 本の PR がどれもテンソル分割を対象外にしていて、2 台では PP だけになる | — |
| R8 ExLlamaV3 単体 | TP が 1 台のホストの中だけ | 上流が複数ノードの TP に対応したとき |
| R9 ktransformers | x86 と 350 GB のメモリが前提 | — |
| R10 1 台 | 補助にとどめる。投機なし 17.7 と今より速いが、約 2.7 bpw で品質が落ちる（KLD 0.38） | 品質の基準を下げてもよいと判断されたとき |
| R11 MXFP4 | GB10 での実績がない。vLLM の Triton MXFP4 MoE は sm_121 で使えない。NVFP4 との差は読む量で約 6% 以下 | — |

**対話側への推奨（優先度の順）**

1. `results/` にある 2026-09-20 の `exl3-tp2` の計測の要約を読む（読み取りだけ）。見る値は、同時 1 本の code/en・prose/ja の `decode_tps`、受理長、1 ステップあたりのトークン数。H-a と H-b の判別に使う。
2. K1 の計測。その前に、`serve` が投機の指定を断る設計（design 6.7）を変える判断が要る。
3. 今の構成の 1 トークンの時間をプロファイラーで分解し、K2 に進むかを決める。
4. candidate D を、`bench/` の同じ条件で同時 1 本で測る（`PLAN.md` L73）。
5. ライセンスの判断を 2 つする。CuTe DSL の EULA と、modal の DFlash の学習データと著作権者の欄。

## 残存ギャップ（あれば）

| # | 項目 | 区分 | 理由・埋め方 |
|---|---|---|---|
| R-G1 | candidate D の 63〜66 の計測の条件、下書き器の方式、重みの種類（H-a と H-b のどちらか） | 調査不可（クリーンルーム） | 公開の事実とメタデータ以上は読まない規則のため。対話側が `results/` を読み、`bench/` の同じ条件で測り直す |
| R-G2 | 読み出し以外の約 20 ms の中身（計算、起動、同期、通信） | 未検証（実機が要る） | 差し引きの計算で出した値で、プロファイラーで測った値ではない |
| R-G3 | GLM の MTP の 1 ステップの重さと、日本語の prose の受理長 | 未検証（実機が要る） | K1 の最初の計測で確かめる |
| R-G4 | KDA・MLA の射影を量子化したときの品質 | 未発見 | 公表値が見つからなかった。D2 のサブタスクは途中で中断したため、`data-attention-quant.md` は §1〜§4 と §6 だけを根拠に使った。実機で KLD と品質の検査をする |
| R-G5 | modal の DFlash の GLM-5.3-Flash での受理長、学習データ、LICENSE の著作権者の欄 | 未発見・要判断 | 受理長は実機で測る。ライセンスは対話側の判断 |
| R-G6 | CuTe DSL（NVIDIA の EULA）が ADR 0002 で受け入れた範囲に入るか | 要判断 | 対話側の判断 |
| R-G7 | vLLM #56983 と #55682 がマージされるか | 未確定 | 上流を追跡する |
| R-G8 | TP の通信（8 KB の all-reduce）の実際の遅延 | 未検証 | 個人のブログの値だけ。実機で測る |
| R-G9 | 修正なしの vLLM main（#55277 以降）が GB10 で動くか | 未発見 | マージから 2 日で、報告がまだない |
| R-G10 | 第三者の検索結果に出た candidate D の作成者のリポジトリと issue の題（メタデータ）を、推測の材料に使ってよいか | 要判断（クリーンルーム） | 本文は開いていない。題の中の修正の種類の語は、この報告に書き写していない。`PLAN.md` §2 の範囲に入るかを、対話側が判断する |
| R-G11 | canada-quant の W4A16-MTP のモデルカードの本文から得られる事実 | 調査不可 | ADR 0002 Decision 11 により開いていない。本文の事実が要るかは対話側が判断する |