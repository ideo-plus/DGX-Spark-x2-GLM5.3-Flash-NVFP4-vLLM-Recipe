# DGX Spark GLM-5.3-Flash Recipe

[English](README.md)

**GLM-5.3-Flash**（320B パラメータの MoE モデル、1 トークンで動くのは 18B、重みは MIT ライセンス）を、
**DGX Spark 2 台**（GB10、ユニファイドメモリ約 121.7GiB × 2）の上で **vLLM** を使い、2 台を結ぶ
1 本の QSFP/RoCE リンク越しに tensor parallel (TP=2) で動かす、クリーンルームのレシピである。
目指すのは「動く」ではなく「長時間の無人のエージェント作業を任せられる」こと。この推論サーバーは、
コーディングエージェントのワークフロー道具 `takt` のバックエンドとして、Anthropic 互換の
`/v1/messages` から呼ばれる想定である。目的・制約・段階 (P0〜P8) の全体は [`PLAN.md`](PLAN.md) を参照。

## 現在の結果 (探り `probe` の水準)

1 本の生成速度 (decode tok/s)、MTP の N = 3、同日に探り (出力 128 トークン) で測った値。

| 重み | code/en | code/ja | prose/en | prose/ja |
|---|---:|---:|---:|---:|
| 元の NVFP4 の重み (MTP N=3) | 33.99 | 29.31 | 29.75 | 27.85 |
| K2 第 2 段 FP8 (`k2s2b`) | 42.45 | 37.73 | 39.62 | 34.80 |
| K2 第 3 段 NVFP4A16 (`k2s3`) | 48.00 | 40.62 | 42.30 | 38.26 |

確認の段 (`fast`、出力 256 トークン) の `k2s3`・MTP N = 3 の値は **42.14 / 39.18 / 40.67 / 39.67**。コード・英語は、基準の 45 tok/s にまだ約 7% 届かない ([記録](docs/results/2026-09-28-k2-stage3.md))。

投機的デコードなしでは、元の重みで **約 14.0 tok/s** にとどまる。

これらは**少ないサンプル数の探りの値**であり、候補を素早く比べるためのもので、成功の判定には使わない。
確認の段 (出力 256 トークン) の値と、最終の大量サンプルでの検証は [`docs/results/`](docs/results/)
にある (例: [K2 第 2a 段](docs/results/2026-09-27-k2-stage2a.md)、
[K2 第 2b 段](docs/results/2026-09-27-k2-stage2b.md))。成功の基準に対する判定そのものは、
P0〜P7 をすべて終えたあと **P8** でまとめて行う。

成功の基準 (表と根拠の全体は [`PLAN.md` §3](PLAN.md#3-成功の基準) を参照): 生成速度 (コード・英語)
**45 tok/s** (2026-09-27 に 60 から見直した。業務で使えるドラフターでは 60 に届く見込みが無いと
分かったため。詳細は [`docs/results/2026-09-27-k3-drafter-bound.md`](docs/results/2026-09-27-k3-drafter-bound.md))、
生成速度 (散文・日本語) **30 tok/s**、エージェントの安定性 (会話 10 万トークンまで、ツール呼び出しの
誤りが 1% 未満)、`takt` の負荷を想定した 72 時間の連続稼働。

## 仕組み

- **自前でビルドした vLLM のイメージ**: 上流 vLLM を commit `0961bbae2894d574be790d219651824eb199318e`
  に固定し、上流 PR #55277 (head `8d09804c877c48165c6ba69bc9dc02d09bae0b83`) から取り込んだ NoPE の
  カーネル修正を当てたもの。直しのたびに作り直すのではなく、1 回だけビルドする。
- 2 台の DGX Spark を結ぶ 1 本の物理 QSFP ケーブル越しの **TP=2**。NCCL からは 2 つの RoCE の
  デバイスとして見え、両方が使われる (all-reduce の busbw は実測で約 186.9 Gbps)。
- **MTP の投機的デコード、N = 3** — モデル付属の multi-token-prediction ヘッド (MIT) であり、
  非商用の第三者のドラフターではない。
- **K2**: vLLM がそのままでは量子化できなかった BF16 の重み (MLA・KDA のアテンションの射影、共有の
  専門家、dense の MLP、`lm_head`) を、[`experiments/k2-quant`](experiments/k2-quant/README.md)
  (ゼロから書いた、CPU だけで動く、GPU もネットワークも使わない道具) で、手元で FP8 (第 2a・2b 段) に、
  続けて NVFP4A16 (重みだけ NVFP4、第 3 段) に変換する。
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
| `docs/vllm-baseline/` | 手順書 (基盤の立ち上げ、K2 の変換、上書きの立ち上げ) |
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
8. K2 の派生の重みに固有の手順は、
   [`docs/vllm-baseline/k2-derived-weights-procedure.md`](docs/vllm-baseline/k2-derived-weights-procedure.md)
   と [`docs/vllm-baseline/k2-vllm-overlay-procedure.md`](docs/vllm-baseline/k2-vllm-overlay-procedure.md)
   を参照。
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

- **コード・英語の 60 tok/s は、業務で使えるドラフターでは届かない。** その水準の公開値は、
  どれも非商用の深いドラフターを使っている。基準は 2026-09-27 に 45 tok/s へ見直した (詳細は
  [`docs/results/2026-09-27-k3-drafter-bound.md`](docs/results/2026-09-27-k3-drafter-bound.md))。
- **同時 2 本の 1 本あたりの速さは、まだ目標に届かない** (`k2s2b` で約 24 tok/s、目標は 30 tok/s)。
- **起動には `instanttensor` の速い経路でも約 5 分かかる** (ふつうの `mmap` での読み込みは約 13 分)。
- **`k2s3` の確認の段のコード・英語は 42.1 tok/s** で、基準の 45 tok/s に約 7% 届かない
  ([`docs/results/2026-09-28-k2-stage3.md`](docs/results/2026-09-28-k2-stage3.md))。
- **P8 (すべての成功基準に対する、大量サンプルでの最終検証) はまだ行っていない。** この README の値は
  すべて探りの水準のもので、300 試行によるエージェントの安定性の判定は、いまのところ元の重みでしか
  行っていない ([`docs/results/2026-09-24-agent-20k.md`](docs/results/2026-09-24-agent-20k.md))。
