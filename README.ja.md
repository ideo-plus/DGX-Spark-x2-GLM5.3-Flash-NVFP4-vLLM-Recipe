# DGX Spark x2 GLM-5.3-Flash NVFP4 vLLM Recipe

[English](README.md)

**GLM-5.3-Flash**（320B パラメータの MoE モデル、1 トークンで動くのは 18B、重みは MIT ライセンス）を、
**DGX Spark 2 台**（GB10、ユニファイドメモリ約 121.7GiB × 2）の上で **vLLM** を使い、2 台を結ぶ
1 本の QSFP/RoCE リンク越しに tensor parallel (TP=2) で動かす、クリーンルームのレシピである。
目指すのは「動く」ではなく「長時間の無人のエージェント作業を任せられる」こと。この推論サーバーは、
コーディングエージェントのワークフロー道具 `takt` のバックエンドとして、Anthropic 互換の
`/v1/messages` から呼ばれる想定である。目的・制約・段階 (P0〜P8) の全体は [`PLAN.md`](PLAN.md) を参照。

## 現在の結果

構成 `glm53-tp2-mtp3-marlin` を確認の段 (`fast`) で測った値。JSON は出力上限8,192トークン、
各条件10回、同じサーバーでの再計数による本文速度である。
これらの測定では、2 台とも GPU graphics クロックを `300〜1800 MHz` (`nvidia-smi -lgc 300,1800`)、
X925 CPU コアを `3.0 GHz` に制限した。詳細な記録には、実効 graphics クロック約 1.8 GHz と
thermal slowdown がなかったことも記載している。

**従来のコード・散文の生成速度は、本文ではなく思考だけを生成したときの値である。**
前回のコード・散文40試行と旧JSON計測の60試行は、すべて思考だけで256トークンを使い切った。
その旧値は本文の生成性能を示す値として扱わない。

| 項目 | 値 | 記録 |
|---|---:|---|
| 1 本、コード・英語（思考のみ） | 45.0 tok/s | [9月28日の記録](docs/results/2026-09-28-k2-stage3.md) |
| 1 本、コード・日本語（思考のみ） | 43.9 tok/s | [9月28日の記録](docs/results/2026-09-28-k2-stage3.md) |
| 1 本、散文・英語（思考のみ） | 46.3 tok/s | [9月28日の記録](docs/results/2026-09-28-k2-stage3.md) |
| 1 本、散文・日本語（思考のみ） | 42.2 tok/s | [9月28日の記録](docs/results/2026-09-28-k2-stage3.md) |
| JSON 指示・英語（本文、再計数） | 58.6 tok/s | [9月29日の記録](docs/results/2026-09-29-json-text-decode.md) |
| JSON 指示・日本語（本文、再計数） | 58.9 tok/s | [9月29日の記録](docs/results/2026-09-29-json-text-decode.md) |
| 同時 2 本の1本あたり（合計） | 27.2 tok/s（45.8） | [9月28日の記録](docs/results/2026-09-28-k2-stage3.md) |
| 入力の処理 32k（cold） | 1,326 tok/s | [9月28日の記録](docs/results/2026-09-28-k2-stage3.md) |

JSON 本文の値は再計数による速度で、配列の完成率とは別である。根拠と制約は[今回の記録](docs/results/2026-09-29-json-text-decode.md)と
[レシピの比較表](docs/research/2026-09-28-recipe-comparison.md)を参照。同時実行と入力処理は測り直していない。

以下の比較表には、同じ条件で測った値だけを載せる。条件はこのリポジトリの `fast`、DGX Spark 2 台の TP=2、
GPU クロック上限 `300〜1800 MHz` である。値は tok/s の中央値で、同時 2 本は 1 本あたり、入力の処理は cold 32k である。

| 実装 | コード・英語 | 散文・英語 | 同時 2 本（1 本あたり） | cold 32k 入力処理 |
|---|---:|---:|---:|---:|
| [現行構成: MTP N=3 + Marlin](docs/results/2026-09-28-k2-stage3.md) | **45.021** | **46.267** | **27.235** | **1,326.033** |
| [MiaAI-Lab EXL3 + DFlash2](docs/results/2026-09-29-miaai-exl3-dflash2.md) | 34.212 | 33.423 | 22.675 | 1,340.999 |

mmastrac の TP=2 は同じ条件での測定が完了したら追加する。ほかのレシピの公開値は、プロンプト、サンプリング、
思考モード、計測方法が異なるため、ここには載せない。

この段の品質の確認 (ツール呼び出し、HumanEval+、needle 8k・32k) では、壊れは見つかっていない。
値は少ないサンプル数のもので、成功の基準に対する判定は、**P8** で大量のサンプル数で行う。
記録: [`docs/results/2026-09-28-k2-stage3.md`](docs/results/2026-09-28-k2-stage3.md)。
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
- **MoE のカーネル**: すべての MoE の層で `--moe-backend marlin` (重みだけ 4 bit、活性は BF16) を使う。
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

- **同時 2 本の 1 本あたりの速さは 27.2 tok/s だった。**
- **32k の入力の処理は 1,326 tok/s だった。** MTP を使う構成では、プレフィックスキャッシュがほとんど当たらない。
- **起動には約 5 分かかる** (ページキャッシュを捨てたうえで `instanttensor` を使う)。
- **P8 (すべての成功基準に対する、大量サンプルでの最終検証) は、まだ行っていない。**
