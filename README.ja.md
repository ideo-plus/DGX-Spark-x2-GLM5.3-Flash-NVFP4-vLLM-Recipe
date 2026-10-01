# DGX Spark x2 GLM-5.3-Flash NVFP4 vLLM Recipe

[English](README.md)

**GLM-5.3-Flash**（320B パラメータの MoE モデル、1 トークンで動くのは 18B、重みは MIT ライセンス）を、
**DGX Spark 2 台**（GB10、ユニファイドメモリ約 121.7GiB × 2）の上で **vLLM** を使い、2 台を結ぶ
1 本の QSFP/RoCE リンク越しに tensor parallel (TP=2) で動かす、クリーンルームのレシピである。
目指すのは「動く」ではなく「長時間の無人のエージェント作業を任せられる」こと。この推論サーバーは、
コーディングエージェントのワークフロー道具 `takt` のバックエンドとして、Anthropic 互換の
`/v1/messages` から呼ばれる想定である。目的・制約・段階 (P0〜P8) の全体は [`PLAN.md`](PLAN.md) を参照。

## 現在の結果

### sparkDash で測った値

[sparkDash](https://github.com/MiaAI-Lab/sparkDash) は、MiaAI-Lab と knapcio が公表値に使っている道具である。DGX Spark 2 台の TP=2、GPU クロック上限 `300〜1800 MHz`、X925 3.0 GHz で、
sparkDash 1.8.6 を使って測った。条件は `/v1/chat/completions`、温度 0、出力 400 トークン (`min_tokens` と `ignore_eos` で必ずその長さまで生成)、
各 1 回。prefill の入力は、同じ単語の繰り返しで埋めた文である。sparkDash は思考オフを求めるが、この構成の標準のチャットテンプレートはそれを読まない。
サーバーの既定の思考の深さは `low` なので、現行構成は答えの前に短い思考の 1 文を書く (sparkDash はこれも数える)。詳細は [decode の記録](docs/results/2026-10-01-default-effort-low.md) と [prefill の記録](docs/results/2026-10-01-mbt-4096.md)。

| 実装 | 構造化 同時 1 本 | 散文 同時 1 本 | コード 同時 1 本 | 構造化 同時 2 本 (1 本あたり / 合計) | prefill 32k | prefill 128k |
|---|---:|---:|---:|---:|---:|---:|
| 現行構成 (既定の思考の深さ low) | 60.64 | 43.74 | 59.03 | 49.01 / 97.11 | 1,637 | 1,614 |
| mmastrac NVFP4 + DFlash2 (既定の構成) | 69.67 | 32.16 | 66.18 | 53.93 / 107.85 | 1,861 | 1,824 |
| MiaAI-Lab EXL3 + DFlash2 (9/29 と同じ構成) | 70.52 | 32.56 | 67.30 | 55.00 / 110.00 | 1,446 | 1,500 |

mmastrac は、レシピの README の 2 台の既定の構成で測った。mmastrac の独自のチャットテンプレートは、sparkDash の思考オフを「思考を浅くする (low)」と読む。
MiaAI-Lab は、推論の設定をレシピの既定のまま (2 台の IP アドレスとネットワークの値だけ合わせた) で測った。MiaAI-Lab は思考を完全に止める。
3 つの行で、思考の条件はそろっていない。MiaAI-Lab は思考を止め、mmastrac と現行構成は短い思考の 1 文が残る。

### このリポジトリの `bench` で測った値

構成 `glm53-tp2-mtp3-marlin` を確認の段 (`fast`) で測った値。サーバーの既定の思考の深さは `low` である。
decode はすべての試行で答えの本文まで届いたので、下の decode の値は答えの本文を生成する速さである。
これらの測定では、2 台とも GPU graphics クロックを `300〜1800 MHz` (`nvidia-smi -lgc 300,1800`)、
X925 CPU コアを `3.0 GHz` に制限した。詳細な記録には、実効 graphics クロック約 1.8 GHz と
thermal slowdown がなかったことも記載している。
GPU の上限を 2200 MHz に上げ、X925 の上限を外しても、decode (1 本・同時 2 本) と cold の入力の処理は
速くならなかった。そのため、上限はこのままにしている ([9月29日の記録](docs/results/2026-09-29-gpu-clock-2200.md))。

**decode の値は答えの本文の速さである。** 思考の深さが最大だと、モデルがコードを思考の中に書いたまま、答えを返さないことが多い
([#158](https://github.com/ideo-plus/DGX-Spark-x2-GLM5.3-Flash-NVFP4-vLLM-Recipe/issues/158))。そのため、サーバーの既定を `low` にしている。深く考えさせたい要求は、`reasoning_effort` を送れば変えられる。

| 項目 | 値 | 記録 |
|---|---:|---|
| 1 本、コード・英語 | 44.3 tok/s | [10月1日の記録](docs/results/2026-10-01-default-effort-low.md) |
| 1 本、コード・日本語 | 43.0 tok/s | [10月1日の記録](docs/results/2026-10-01-default-effort-low.md) |
| 1 本、散文・英語 | 37.2 tok/s | [10月1日の記録](docs/results/2026-10-01-default-effort-low.md) |
| 1 本、散文・日本語 | 38.6 tok/s | [10月1日の記録](docs/results/2026-10-01-default-effort-low.md) |
| 1 本、JSON・英語 | 46.9 tok/s | [10月1日の記録](docs/results/2026-10-01-default-effort-low.md) |
| 1 本、JSON・日本語 | 47.9 tok/s | [10月1日の記録](docs/results/2026-10-01-default-effort-low.md) |
| 同時 2 本の1本あたり（合計） | 23.5 tok/s（40.7） | [10月1日の記録](docs/results/2026-10-01-default-effort-low.md) |
| 入力の処理 32k（cold） | 1,448 tok/s | [10月1日の記録](docs/results/2026-10-01-mbt-4096.md) |
| 入力の処理 128k（cold） | 1,429 tok/s | [10月1日の記録](docs/results/2026-10-01-mbt-4096.md) |

ほかのレシピの公開値は、[レシピの比較表](docs/research/2026-09-28-recipe-comparison.md)にまとめている。

以下の比較表には、同じ条件で測った値だけを載せる。条件はこのリポジトリの `fast` (入力の処理は 8k・32k だけ)、DGX Spark 2 台の TP=2、
GPU クロック上限 `300〜1800 MHz` である。値は tok/s の中央値で、同時 2 本は 1 本あたり、入力の処理は cold 32k である。思考の条件は違う。現行構成の decode は既定の深さ `low` での
答えの本文の値で、ほかの 2 つは各レシピの既定の思考のままで測り、256 トークンが思考の文に使われた値である。

| 実装 | コード・英語 | 散文・英語 | 同時 2 本（1 本あたり） | cold 32k 入力処理 |
|---|---:|---:|---:|---:|
| [現行構成: MTP N=3 + Marlin (KV 4 GiB、既定の深さ low)](docs/results/2026-10-01-default-effort-low.md) | 44.288 | 37.211 | 23.480 | [1,448.141](docs/results/2026-10-01-mbt-4096.md) |
| [MiaAI-Lab EXL3 + DFlash2 (9/29)](docs/results/2026-09-29-miaai-exl3-dflash2.md) | 34.212 | 33.423 | 22.675 | 1,340.999 |
| [mmastrac NVFP4 + DFlash2 (既定の構成)](docs/results/2026-09-30-mmastrac-32k-same-conditions.md) | 33.153 | 31.183 | 23.672 | 1,835.091 |

mmastrac は、レシピの README の 2 台の既定の構成 (`compose/.env` に `TP=2`、`compose/glm53.yaml` だけ、実験的な上書きなし) で測った。
この計測の最中に、2 台とも GPU ドライバーの `NV_ERR_NO_MEMORY` が出ていた。
MiaAI-Lab と mmastrac は、非商用の条件が付いた DFlash2 の下書きモデルを使う。ほかのレシピの公開値は、
プロンプト、サンプリング、思考モード、計測方法が異なるため、ここには載せない。

既定の深さ `low` での品質の確認 (ツール呼び出し 10/10、HumanEval+ 10/10、needle 8k・32k 2/2) では、壊れは見つかっていない。
値は少ないサンプル数のもので、成功の基準に対する判定は、**P8** で大量のサンプル数で行う。
記録: [`docs/results/2026-10-01-default-effort-low.md`](docs/results/2026-10-01-default-effort-low.md)。
成功の基準の全体は [`PLAN.md` §3](PLAN.md#3-成功の基準) にある。

## 仕組み

- **自前でビルドした vLLM のイメージ**: 上流 vLLM を commit `0961bbae2894d574be790d219651824eb199318e`
  に固定し、上流 PR #55277 (head `8d09804c877c48165c6ba69bc9dc02d09bae0b83`) から取り込んだ NoPE の
  カーネル修正を当てたもの。直しのたびに作り直すのではなく、1 回だけビルドする。
- 2 台の DGX Spark を結ぶ 1 本の物理 QSFP ケーブル越しの **TP=2**。NCCL からは 2 つの RoCE の
  デバイスとして見え、両方が使われる (all-reduce の busbw は実測で約 186.9 Gbps)。
- **MTP の投機的デコード、N = 3** — モデル付属の multi-token-prediction ヘッド (MIT) であり、
  非商用の第三者のドラフターではない。
- **重み**: 公開の NVFP4 の checkpoint の専門家はそのまま使い、BF16 か FP8 で残っていた重みの大半
  (アテンションの射影、共有の専門家、dense の MLP、`lm_head`、MTP の層の射影と専門家) を、
  [`experiments/k2-quant`](experiments/k2-quant/README.md) (ゼロから書いた、CPU だけで動く道具) で、
  手元で NVFP4A16 (重みだけ NVFP4) に変換する (MLA の `kv_b_proj`、indexer、MTP の `eh_proj` は BF16 のまま)。できた派生の重み `k2s4` は、`serving/weights/` の
  マニフェストで固定する。
- **既定の思考の深さ `low`** (`--default-chat-template-kwargs '{"reasoning_effort":"low"}'`)。思考の深さが最大だと、モデルがコードを
  思考の中に書いたまま、答えを返さないことが多い ([#158](https://github.com/ideo-plus/DGX-Spark-x2-GLM5.3-Flash-NVFP4-vLLM-Recipe/issues/158))。深く考えさせたい要求は、`reasoning_effort` を送れば変えられる。
  - **モデルのチャットテンプレートが知っている深さは、`low`・`high`・`max` の 3 つだけである。**
    - `medium`・`xhigh`・`minimal`・`none` など、ほかの値はすべて `max` で動く。
    - 要求に付けた値はサーバーの既定より優先されるので、`medium` を送るクライアントは最大の深さになる。
    - 送るのは `low` か `high` にする。`/v1/chat/completions` では `reasoning_effort`、`/v1/messages` では `output_config.effort` で送る ([決定の記録](docs/decisions/0004-vllm-baseline-messages-api.md))。
- **MoE のカーネル: Marlin** — すべての MoE の層で `--moe-backend marlin` を使う。構成の名前
  `glm53-tp2-mtp3-marlin` の末尾の `marlin` は、これを指す (GLM-5.3-Flash、TP=2、MTP N=3、Marlin)。
  - **何か**: [Marlin](https://github.com/IST-DASLab/marlin) (**M**ixed **A**uto-**R**egressive
    **Lin**ear kernel。IST-DASLab が作った) は、重みだけを量子化した行列の掛け算を行う GPU のカーネルである。
    4 bit の重みをメモリから読み、カーネルの中で BF16 に戻して、BF16 の活性と掛ける。
    vLLM に入っており、MoE の専門家の計算にも使える。
  - **decode に向く理由**: decode では、1 ステップに流れるトークンが少ない (1 本あたり、本物の 1 トークンと
    MTP の下書きの 3 トークン)。そのため、1 ステップの時間は、計算の量より重みを読む量で決まりやすい。
    作者の説明では、Marlin は、1 ステップに 16〜32 トークンくらいまでなら、4 bit の重みで読む量が減るぶんの速さを保つ。
  - **選んだ理由**: 公開の NVFP4 の専門家に対して、vLLM がこの GPU で選ぶ既定のカーネルは FlashInfer の CUTLASS で、
    活性も 4 bit にする (W4A4)。Marlin は活性を BF16 のまま計算し、このモデルでは decode が速かった
    ([記録](docs/results/2026-09-28-k2-stage3.md#k2s4--marlin確認の段n--3))。
  - **限界**: Marlin は、1 ステップに流れるトークンが少ない計算に向けたものである。長い入力の処理 (prefill。
    1 ステップに流れるトークンが多い) を Marlin が遅くしているかは、切り分けていない。
- その派生の重みを読み込むために要る vLLM のモデルコードの直しは、イメージを作り直さず、**読み取り
  専用の bind mount で重ねる**上書きファイルとして当てる
  ([`experiments/k2-vllm-overlay`](experiments/k2-vllm-overlay/README.md))。これにより、探りの回ごとに
  イメージを作り直さずに済む。
- root 不要でページキャッシュを捨てたうえで `--load-format instanttensor` を使う**速い起動**
  (約 4〜5 分。ふつうの `mmap` での読み込みは約 13 分)。
- 再起動をまたいでも効かせ続ける **GPU/CPU のクロックの上限** (`nvidia-smi -lgc 300,1800`、X925 の
  コアを `cpupower` で 3.0 GHz に) による熱の安定化 ([`ops/spark-power-caps`](ops/spark-power-caps/README.md))。

## リポジトリの構成

| パス | 中身 |
|---|---|
| [`bench/`](bench/README.md) | 計測の道具 (`bench`)。decode/prefill/concurrency/quality/agent の各まとまり、`probe`/`fast`/`quick`/`full` の設定 |
| [`serving/`](serving/README.md) | 運用の道具 (`serve`)。push/fetch/verify/start/stop/watch、構成は `config/`、重みのマニフェストは `weights/` |
| `experiments/k2-quant/` | 手元で重みを量子化する道具 (FP8 / NVFP4A16)。CPU だけ、クリーンルーム |
| `experiments/k2-vllm-overlay/` | 読み取り専用の bind mount で重ねる、vLLM の Python ファイルのパッチ |
| `experiments/k2-profile/`、`experiments/nope-mla/` | プロファイリングと NoPE/MLA の実験を支える道具 |
| `ops/spark-power-caps/` | 起動のたびに GPU/CPU のクロックの上限を入れ直す systemd のユニット |
| `ops/spark-drop-caches/` | 速い起動の前に使う、ページキャッシュを捨てる道具 (root 不要) |
| `docs/decisions/` | ADR。何を試し、何を測り、なぜ選んだか |
| `docs/results/` | 公開する計測の要約 (生データはリポジトリに入れず `results/` に置く) |
| `docs/vllm-baseline/` | 手順書 (基盤の立ち上げ、重みの FP8・NVFP4 への変換、上書きの立ち上げ) |
| `docs/research/`、`docs/rules/`、`docs/tasks/`、`docs/development/` | 調査のメモ、作業の決まり、タスクの記録、CI |
| `scripts/` | `spark-precheck.sh` (読み取りだけの機材の確認) と TAKT のラッパー |
| `results/` | 計測の生データ (`.gitignore` 対象。プロンプトと応答の本文はここだけに置き、リポジトリには入れない) |
| [`PLAN.md`](PLAN.md)、[`LICENSES.md`](LICENSES.md) | 計画と、ライセンス・出所の台帳 |

## クイックスタート (運用者向け)

状態を変える手順はすべて Mac から流し、DGX Spark に触る前に運用者の明示の了承が要る。ここに
無人での実行を想定したものはない。正確なコマンドは [`serving/README.md`](serving/README.md)、
段ごとの手順の全体は [`docs/vllm-baseline/procedure.md`](docs/vllm-baseline/procedure.md) を参照。

1. `serve push` — `serving/payload/` を 2 台に配る。
2. `serve pull-image` / `serve image-licenses` — 固定したイメージをダイジェストで取得し、中の
   ライセンスの表記を読む。
3. `serve manifest` + `serve fetch --probe-files` + `serve verify` — 重みのマニフェストを作り、
   小さな確認に要る分だけ取得して照合する。
4. `serve probe` — 1 台での縮小の確認として、いったんモデルを起こしてみる。
5. `serve netcheck links` / `bandwidth` / `sanity` — QSFP/RoCE のリンクを確かめ、vLLM 公式の
   事前確認を流す。
6. `serve fetch` + `serve verify` + `serve start` — 重みの本体を取得・照合し、2 台で TP=2 を起こす。
7. `serve smoke` / `serve watch` / `serve logs` / `serve stop` — 疎通を確かめ、負荷の下で見張り、
   記録を集め、後片付けする。
8. 手元で FP8・NVFP4 に変換した重みに固有の手順は、
   [`docs/vllm-baseline/k2-derived-weights-procedure.md`](docs/vllm-baseline/k2-derived-weights-procedure.md)
   と [`docs/vllm-baseline/k2-vllm-overlay-procedure.md`](docs/vllm-baseline/k2-vllm-overlay-procedure.md)
   を参照。推奨の構成 (重み `k2s4` + MTP N=3 + `--moe-backend marlin`) は
   `glm53-tp2-mtp3-marlin` である。派生の重みと重ね合わせを配って照合した後に:

   ```bash
   uv run --directory serving serve check glm53-tp2-mtp3-marlin
   uv run --directory serving serve start glm53-tp2-mtp3-marlin --yes
   ```

   `serve start --yes` は 2 台の状態を変える。了承を得てから実行する。
9. 計測には `bench` を使う ([`bench/README.md`](bench/README.md) 参照)。候補を素早く比べるなら
   `probe`、有望な候補を確かめるなら `fast`、記録するなら `quick`/`full`。

## ライセンスとクリーンルームの方針

実行時とビルド時に使うのは、MIT・Apache-2.0・BSD 系・PSF のいずれかのライセンスの部品だけである
([`PLAN.md` §2](PLAN.md#2-前提と制約) と [`LICENSES.md`](LICENSES.md) を参照。名前・版・入手先・
ライセンス・用途をすべて記録している)。非商用の条件が付いた部品 (他のレシピが使う CC BY-NC-ND の
DFlash2 ドラフターなど) は使わない。GLM-5.3-Flash の重みは、モデルカードのとおり MIT である。
唯一の意図した例外は NVIDIA の CUDA ベースイメージの Deep Learning Container の EULA で、再配布
しない・単体の製品として配らない・NVIDIA の後援をうたわない、という条件のもとで受け入れている
(詳細は `LICENSES.md`)。

このリポジトリは**クリーンルーム**で作る。ほかの GLM/Spark 向けの推論サーバーのレシピのスクリプト・
パッチ・設定ファイルは、許諾の緩いものも含めて、中身を見ながら書かない。参照してよいのは、公式の
文書・論文・モデルカード、上流の vLLM と NCCL のソース・issue・PR、そして自分たちの環境で測った
事実だけである。ちいさくない判断はすべて [`docs/decisions/`](docs/decisions/) に記録する。

## 状況・既知の限界

- **答えの本文の生成速度は、コード・英語で 44.3 tok/s で、基準の 45 tok/s にわずかに届いていない。** 同時 2 本の 1 本あたりは 23.5 tok/s (基準 23)。この MoE ではトークンごとに別の専門家を読むので、同時 2 本はメモリ帯域で頭打ちになる ([記録](docs/results/2026-10-01-concurrency-bandwidth-bound.md))。
- **cold の入力の処理は、32k で 1,448 tok/s、128k で 1,429 tok/s だった** (基準 2,000 tok/s の 72%)。KV キャッシュを 1 台あたり 4 GiB に固定し、`--max-num-batched-tokens` は 4096 にしている。MTP を使う構成では、プレフィックスキャッシュがほとんど当たらない。
- **起動には約 5 分かかる** (ページキャッシュを捨てたうえで `instanttensor` を使う)。
- **P8 (すべての成功基準に対する、大量サンプルでの最終検証) は、まだ行っていない。**
