# P1 後の調査: NoPE 対応と依存する版を確認する

2026-09-22。cc-sdd と takt は導入せず、対話で調査を進める。
P1 は終わりの条件を満たせず打ち切った。今回の調査で、その判断や過去の実測結果を変更しない。

**2026-09-23 (JST) の実機結果**: 自前イメージで GPU 単独検査と 4 層・ダミー重みの縮小起動が合格した。
API は HTTP 200 を返し、検証コンテナも削除済み。詳細は [実行記録](../results/2026-09-22-nope-build.md)。
以下は実機検証前の調査記録として残す。

書かないこと: 送った内容と応答の本文、認証の情報、`exl3-tp2` の中身。

## 第一候補は NoPE と top-k の両方を扱う変更

上流の説明を比較すると、次に適用可能性を調べる候補は
[vLLM #55277](https://github.com/vllm-project/vllm/pull/55277) がよい。
NoPE の KV 書き込みと、kpool によって広がる top-k バッファの両方を扱っているためである。
これは調査の優先順位であり、採用や実機での成功を意味しない。

2026-09-22 に GitHub の本文・会話を確認し、続いて差分と依存関係を調べた。
Mac の一時ディレクトリで差分の適用可否を検査したが、ソースへの適用、ビルド、Spark への接続は行っていない。
下表の動作報告は上流の報告であり、この環境での検証結果ではない。

| 候補 | 確認できたこと | 次に確かめる点 |
|---|---|---|
| [vLLM #55277](https://github.com/vllm-project/vllm/pull/55277) | Open。NoPE の KV 書き込みと実効 top-k 幅を修正する。FlashInfer の GLM53_NOPE 対応と関連修正に依存。RTX PRO 6000 での検証報告がある | 固定した vLLM の版への適用可否、必要な FlashInfer の版、aarch64 / GB10 でのビルド条件。9/12 の競合通知はあるが、現在の適用可否は未検証 |
| [vLLM #53969](https://github.com/vllm-project/vllm/pull/53969) | Open。KV と query の RoPE 部分をゼロ埋めする方式。GB10 2 台での動作報告がある | `index_topk=2048` では実効幅 2176 が拒否されるとの訂正コメントがある。2044 への変更を、品質検証なしに採用しない |
| [vLLM #55778](https://github.com/vllm-project/vllm/pull/55778) | Open。KV 書き込み時の RoPE 部分をゼロ埋めする方式 | この説明だけでは、query 側、実効 top-k 幅、FlashInfer の対応まで満たすか判断できない |
| [vLLM #54929](https://github.com/vllm-project/vllm/pull/54929) | 本文で GLM-5.3-Flash の NoPE を対象外と明記 | 単独で今回の障害を解消する候補から外す |

関連する [issue #57578](https://github.com/vllm-project/vllm/issues/57578) も Open。
assert の条件だけを緩めても、RoPE 部分の読み書きが正しくなるとは限らない点が指摘されている。

## 実機で試す前に、版と検証手順を固定する

次の調査では、第一候補の変更元コミット、適用先の vLLM、必要な FlashInfer を固定する。
その組み合わせで既存の縮小起動 `serve probe` を使えるかを確認し、ビルドと配布の手順を用意する。
外部の変更を読む・取り込む範囲と出典を記録し、既存のクリーンルーム実装とは区別する。

実機操作の計画には、対象ノード、追加するイメージ、使用する構成、成功・失敗の判定、停止・後片付けを含める。
パッチ適用と自前イメージのビルドは、要件 8.6 と ADR 0005 にある計測者の判断を経てから実施する。
重みの取得や TP=2 の検証は、縮小起動が通った後に検討する。

P1 側のタスク 8.8 の終了整理は完了した。上流に再現結果を報告するかの判断は残っており、投稿は未実施。

## 固定イメージのソースには差分が適用できる

調査対象は [vLLM #55277](https://github.com/vllm-project/vllm/pull/55277) の
head `8d09804c877c48165c6ba69bc9dc02d09bae0b83`。GitHub API の変更ファイル一覧から差分を組み立て、
比較対象のコミットから取得した 3 ファイルに `git apply --check --verbose` を実行した。
これは差分の文脈が一致するかの確認であり、コンパイルや動作の保証ではない。

| 比較対象 | build commit | 結果 |
|---|---|---|
| P1 固定イメージ | `385dce36bcee42309924a5ece951a96db3dce7f2` | 終了コード 0。3 ファイルすべて適用可能。試験ファイルは 13 行の位置ずれあり |
| P1 の 9/22 nightly | `0961bbae2894d574be790d219651824eb199318e` | 終了コード 1。C++ と試験は一致するが、`flashinfer_mla_sparse_sm120.py` の文脈が一致せず適用不可 |

変更対象は `csrc/libtorch_stable/cache_kernels.cu`、`tests/kernels/attention/test_cache.py`、
`vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`。
C++ の変更を含むため、Python ファイルの差し替えだけでは反映できない。

GitHub API の `mergeable` は取得時に `null`、`mergeable_state` は `unknown` だった。
上の適用結果は固定した 2 コミットに対するローカル検査であり、現在の main とのマージ可否とは区別する。

## FlashInfer 0.7.0 は依存する 2 つの修正を含む

[FlashInfer 0.7.0](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.0) は
2026-09-22 01:15:03 UTC に公開された安定版。
GitHub のコミット比較 API で、次のマージコミットから `v0.7.0` が ahead、behind が 0 であることを確認した。

| 依存する変更 | マージコミット | v0.7.0 までのコミット数 |
|---|---|---|
| [FlashInfer #4802](https://github.com/flashinfer-ai/flashinfer/pull/4802) | `453aa7c7296e9ec711fd4c1f3aa6ee061a6b69dc` | ahead 105、behind 0 |
| [FlashInfer #4947](https://github.com/flashinfer-ai/flashinfer/pull/4947) | `5cc867a9bb560bc89b91dbc738a9f63f09beb89b` | ahead 73、behind 0 |

したがって、依存修正の取り込みを待つ必要はない。以下で呼び出し側と配布物を確認した。

## nightly は Python 側の修正に相当する処理を含む

9/22 nightly の
[`_run_mqa_kernel`](https://github.com/vllm-project/vllm/blob/0961bbae2894d574be790d219651824eb199318e/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py)
は、`topk_indices_physical.shape[1]` を `max_seq_len` と `sparse_mla_top_k` の両方に渡している。
差分が適用できなかったのは関数の構造が変わっているためで、この処理を追加し直す必要はないと判断した。

同じ差分に対し、C++ と回帰テストの 2 ファイルだけを `--include` で選ぶと、nightly でも
`git apply --check --verbose` は終了コード 0 だった。修正元は上記の上流変更であり、独自に
assert の条件だけを緩める案ではない。NoPE 時の予約領域のゼロ埋めも含める。

この結果から、ビルド手順を具体化する第一候補は、**9/22 nightly のソース＋C++ 修正と回帰テスト＋
FlashInfer 0.7.0** とする。固定イメージのソースに全差分を適用する案は比較用に残す。
まだ実機で起動できると判定したわけではない。

## 配布物はあるが、イメージ全体の互換性はビルド時に確かめる

FlashInfer 0.7.0 の
[`trtllm_batch_decode_with_kv_cache_mla`](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/mla/_core.py)
は、この vLLM が渡す引数名を受け付ける。SM12x の sparse 経路では NoPE に対する
`sparse_mla_top_k_lens` の必須検査を行わない。
[`_resolve_model_type`](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/mla/_sparse_mla_sm120.py)
は幅 512 と `arbitrary_fp32` の組み合わせを GLM53_NOPE に振り分ける。
vLLM は GLM に `arbitrary_fp32` を指定しており、
[対応する top-k 幅](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/mla/_sparse_mla_sm120_plan.py) は 2176。
これはソース上の整合確認に限り、GPU での実行確認ではない。

配布インデックスで次を確認した。ファイル本体の取得やインストールは行っていない。

| 配布物 | インデックスにあるファイル | 掲載 SHA-256 |
|---|---|---|
| [Python パッケージ](https://pypi.org/pypi/flashinfer-python/0.7.0/json) | `flashinfer_python-0.7.0-py3-none-any.whl` | `11e564820cde80ec13d351b68fc44cb54a537268bca135204abafc5fadd50240` |
| [cubin](https://flashinfer.ai/whl/flashinfer-cubin/) | `flashinfer_cubin-0.7.0-py3-none-any.whl` | `f1821e11ad4ea9666a09c2b04cc16b1e34f601296dc7a7b689649281c0358a9c` |
| [CUDA 13.0 の JIT キャッシュ](https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/) | `flashinfer_jit_cache-0.7.0+cu130-cp39-abi3-manylinux_2_28_aarch64.whl` | `7fcf3fbb23283624dd605b5d02365f426eb0143ad28447b085d74345febd75de` |

nightly の [requirements/cuda.txt](https://github.com/vllm-project/vllm/blob/0961bbae2894d574be790d219651824eb199318e/requirements/cuda.txt)
は FlashInfer を `0.6.18.post1` に固定している。
ビルドでは依存の固定値と Docker 側の指定を揃える必要がある。
Python パッケージがアーキテクチャ非依存でも、依存パッケージ全体の aarch64 対応までは保証されない。

次の作業では、変更箇所と出典を記したビルド手順を Mac で用意する。対象はまず head の 1 台とし、
既存イメージを置き換えずに別名で作る計画にする。依存解決、カーネルの回帰テスト、縮小起動の順で確認し、
いずれかが失敗したら重みの取得に進まない。実行時間・空き容量・依存パッケージの最終固定値は未確認である。
