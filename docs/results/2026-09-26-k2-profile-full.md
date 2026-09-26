# K2: 1 ステップの GPU の時間の内訳（`p2-nope-tp2-full-prof`、投機なし）

日付は 2026-09-26 JST。[手順](../vllm-baseline/k2-profile-procedure.md)に沿って、投機なしの構成に torch プロファイラーを付けて測った。
2 つの上限（GPU 1800 MHz、X925 3.0 GHz）は、両台で効いていた。実機操作は対話側が行った。

## 要約

- **1 ステップの 64% は、BF16 の重みを読む GEMV（cuBLAS `gemvx`）だった。** 1 ステップ 75.4 ms のうち 48.1 ms。NVFP4 の MoE の GEMM は 15.2 ms（20%）だった。
- 重みを読むカーネル（BF16 の GEMV・GEMM と、NVFP4 の MoE）は、合わせて約 70 ms で、1 ステップの約 93% を占める。
- 重みを読む以外の時間は、合わせて約 12 ms。内訳は NCCL 3.0 ms、MoE の周辺 2.5 ms、mHC 1.7 ms、アテンションの本体 1.2 ms、その他 0.8 ms、カーネルのない隙間 2.6 ms。調査の見積もり（読み出し以外が 20〜23 ms、`docs/research/2026-09-25-path-survey.md` §5）より小さい。
- したがって、K2 の「アテンション・共有の専門家・dense・lm_head を量子化する」方向は、この内訳と合っている。BF16 の部分を FP8 にすると、1 ステップは約 49 ms（投機なしで約 20 tok/s）になる見込み。NVFP4 にすると約 38 ms（約 26 tok/s）になる見込み。どちらも、読む量が比例して減り、ほかの時間は変わらないと仮定した推定である。

## 計測の条件

- 構成は `p2-nope-tp2-full-prof`。`p2-nope-tp2-full` に `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/logs/torch-profile"}'` を足したもの。
- 計測の窓（`/start_profile` から `/stop_profile` まで）で、`serve smoke --max-tokens 256` の英・日の要求を 1 本ずつ、続けて流した（約 10 秒）。
- ステップ数は、head の `vllm:iteration_tokens_total_count` の増分で、134。trace の GPU の時間の幅は 10,103.8 ms で、1 ステップあたり 75.4 ms。プロファイラーを付けない計測（約 71 ms、[記録](2026-09-25-k1-mtp1.md)）より約 6% 長い。プロファイラーの負担と、prefill の回が入っている分と見る。

## 内訳（worker の rank 1、1 ステップあたり）

`experiments/k2-profile/summarize_trace.py` の今の区分は、実機のカーネル名に合っていなかった（#48）。下の表は、カーネル名の系統ごとに数え直したもの。

| 系統 | ms / ステップ | 割合 | 回数 / ステップ |
|---|---:|---:|---:|
| BF16 の重みの GEMV（cuBLAS `gemvx`、bf16 → bf16） | 48.10 | 63.8% | 293.0 |
| BF16 の GEMV（bf16 → fp32） | 1.72 | 2.3% | 42.0 |
| fp32 の GEMV（`gemvNSP`） | 0.24 | 0.3% | 11.0 |
| BF16 の GEMM（`cutlass_80_wmma`、`nvjet`） | 2.09 | 2.8% | 27.3 |
| NVFP4 の MoE の grouped GEMM（cutlass、BlockScaled） | 15.23 | 20.2% | 85.3 |
| MoE の周辺（expand・finalize・並べ替え・活性化・topk・fp4 への変換） | 2.48 | 3.3% | 298.4 |
| NCCL の all-reduce（`AllReduce_Sum_bf16_RING_LL`） | 3.00 | 4.0% | 93.4 |
| mHC（tilelang） | 1.72 | 2.3% | 182.7 |
| アテンションの本体（KDA の recurrent、sparse MLA、conv1d、kpool、topk、fwht） | 1.23 | 1.6% | 136.5 |
| その他のカーネル | 0.81 | 1.1% | 380.1 |
| カーネルのない隙間 | 2.61 | 3.5% | — |

ストリームの間の重なりが、合わせて 513 ms あった。そのため、割合の合計は 100% を少し超える。

## 問題と扱い

- **head の worker が OOM で落ちた。** `/stop_profile` で trace を書き出すときに、head の `VLLM::Worker_TP` が oom-killer に止められた（カーネルのログに `Out of memory: Killed process ... (VLLM::Worker_TP)` がある）。そのためエンジンが止まり、`/stop_profile` は HTTP 500 になった。head の rank 0 の GPU の trace は取れていない。TP の 2 台は同じ形の計算をするので、worker の rank 1 の trace で読んだ。書き出しのメモリを減らす設定は #48 で扱う。
- **本文の確認は、要求の側だけ。**
  - 要求の語（`capital of France`、`日本の首都`）は、trace と `profiler_out_*.txt` の中で 0 件だった。
  - 応答の側は、確認できていない。はじめは「英語の応答の語も 0 件」と書いていたが、誤りだった。台本が応答の本文（stderr に出る）を捨てていたため、応答の語は取り出せていなかった（2026-09-26 に分かった。台本は直した）。
  - trace は `serving/var/`（Git 対象外）に留め、ここには写さない。

## 次に確かめること

1. MTP の N = 3 で同じ内訳を取り、1 ステップが約 110 ms に伸びる理由（下書きの層を読む分か、検証の分か）を分ける。
2. K2 の量子化の方法を決める（BF16 の部分を FP8 か NVFP4 にする。品質の確かめ方も含めて）。
