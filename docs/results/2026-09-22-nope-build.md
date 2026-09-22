# NoPE 修正イメージのビルドと縮小検証

状態: ビルド、GPU 単独検査、縮小起動、後片付けまで完了。実行日は 2026-09-22〜23 (JST)。
以下の実行時刻とファイル名の日付は UTC。2026-09-22 に計測者から、Mac でのパッチ適用、head への配布、
ビルド、GPU 単独検査、縮小起動、検証用コンテナの後片付けまでの了承を得た。
対象は head (`spark-153d`) のみ。手順は [実行計画](../vllm-baseline/patched-build-procedure.md)。

書かないこと: 要求・応答の本文、認証情報、`exl3-tp2` の中身。

## 準備と実行記録

- 実行直前の空き容量は 2,661,103,284,224 バイト、利用可能メモリは 125,669,818,368 バイト。
  `nvidia-smi --query-compute-apps` にプロセスの行はなかった。
- 新しいイメージのタグ、GPU 単独検査のコンテナ名、配布先ディレクトリは未使用だった。
- Mac のソースに差分を適用し、準備時と SHA-256 が一致することを確認した。
  変更は上流の C++ と回帰テスト、FlashInfer の版指定を持つ 3 ファイルの計 5 ファイル。
- head の `/home/j5ik2o/vllm-baseline/payload/nope-build-0961bbae/` に配布した。
- ビルドのコマンド・開始時刻・ログ・終了結果は、Mac の `serving/var/nope-build-0961bbae/` に保存した。
  `manifest.json` は `applied: true`。GPU 単独検査と縮小起動は、ビルド成功後に行った。

ビルド対象の vLLM は `0961bbae2894d574be790d219651824eb199318e`、FlashInfer は 0.7.0。
上流の修正元と差分の SHA-256 は実行計画に記載した。

## 1 回目: ビルド用イメージのアーキテクチャが不一致

13:54:07〜13:58:17 UTC、終了コード 1。`base` ステージの最初の RUN で
`exec /bin/sh: exec format error`。BuildKit はビルド用イメージが linux/amd64 で、
要求した linux/arm64 と異なることも報告した。C++ コンパイルには到達していない。

原因は手順の `BUILD_BASE_IMAGE` 指定漏れ。上流の arm64 用 CI が使う別のイメージを確認し、
Docker Hub のタグ API から linux/arm64 のダイジェストを取得した。
2 回目は、この引数だけを追加して再試行した。元のログは `build.log` と `build-result.json`、
2 回目は `build-arm64.log` と `build-arm64-result.json` に分けた。

## 2 回目: arm64 のビルド用イメージでコンパイルを確認し、中断

指定したビルド用イメージでシェルとパッケージ管理が動き、初回のアーキテクチャ不一致は解消した。
FlashInfer 0.7.0 と JIT キャッシュの導入、ソース依存の構築、Rust 部分のビルドが完了。
C++／CUDA のビルドは途中まで進み、修正した `cache_kernels.cu` のコンパイルは通過した。
この試行ではビルド全体の完了前に、下記の並列数の変更のため中断した。

## 3 回目: 実質 8 並列でビルド成功

2 回目は 15:13:47 UTC に、対象ビルドの PID とタグ・作業ディレクトリを照合して中断した
(終了コード 130)。失敗による停止ではない。上流の `setup.py` が `MAX_JOBS / NVCC_THREADS` で
ジョブ数を決めるため、`2 / 2` では実質 1 並列になっていた。

head の 20 CPU と約 120 GB の利用可能メモリ、コンパイルキャッシュの永続マウントを確認し、
`max_jobs` だけを 16 に変更した。`nvcc_threads=2` は維持し、実質 8 並列で再開した。
ソースとパッチは同一。ログは `build-parallel8.log`、コマンドは `build-parallel8-command.json`、
終了結果は `build-parallel8-result.json` に分けた。

3 回目は 15:14:13〜15:47:57 UTC に実行し、終了コード 0 で完了した (約 33 分 43 秒)。
完成した linux/arm64 イメージは 23,438,274,807 バイト、ID は
`sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90`。

## GPU 単独検査は合格

最初の検査は `pip check` で停止し、C++ の検査本体には到達しなかった。
公式 nightly と自前イメージを比較すると、次の 2 件は同一だった。

- torch 2.13.0+cu130 のメタデータは NCCL 2.29.7 を要求するが、上流 Dockerfile は
  DeepEPv2 のために NCCL 2.30.7 を明示的に優先している。
- cuSPARSELt 0.8.1 の wheel タグは `manylinux2014_sbsa`。`pip check` は非対応と判定するが、
  ライブラリの ELF machine は 183 (AArch64) で、両イメージとも読み込みに成功した。

根拠は上流の [NCCL 指定](https://github.com/vllm-project/vllm/blob/0961bbae2894d574be790d219651824eb199318e/docker/Dockerfile)、
`dependency-audit-control.json` と `dependency-audit-treatment.json`。
上流イメージと同じ 2 件だけであることを照合し、パッケージは変更せず、C++ の単独検査を別に実行した。
`pip check` 自体が合格したとは扱わない。

15:51:07〜15:51:15 UTC の検査は終了コード 0、`cache_check=passed`。
NoPE とゼロの RoPE 入力の出力一致、予約領域のゼロ埋め、対象外の行の保持、従来の RoPE 値の保持を確認した。
検査コンテナと依存確認用コンテナはすべて、所有を照合した後に停止・削除した。
記録は `cache-check-v2-result.json` と `cache-check-v2.log`。

実際の版は torch `2.13.0+cu130`、FlashInfer `0.7.0`、vLLM `0.1.dev1+g0961bbae2.d20260922`。
vLLM の版文字列は浅い取得のソースから生成されたもので、公式 nightly の版文字列とは異なる。
比較の基準は上記の固定コミットとイメージ ID とする。

## 4 層・ダミー重みの縮小起動は合格

`serve check probe-nope-local` の 8 件の関門はすべて通過した。
`serve probe probe-nope-local --configs var/nope-build-0961bbae/probe.toml --timeout 45m --yes`
を 15:52:07〜15:55:53 UTC に実行し、終了コード 0、`status=ready` で完了した。

| 確認したこと | 結果 |
|---|---|
| アテンションのバックエンド | `FLASHINFER_MLA_SPARSE_SM120` |
| MoE のバックエンド | `FLASHINFER_CUTLASS` |
| CUDA graph | PIECEWISE と FULL の取得が完了。以前の `pe_dim must be 64 for fp8_ds_mla` は発生しなかった |
| 短い要求 | HTTP 200、入力 25 / 出力 64 トークン、`stop_reason=max_tokens` |
| エンジンの初期化 | 182.48 秒。初回起動の観測値であり、性能評価ではない |
| 検証対象 | head の 1 台、4 層、ダミー重み、最大文脈 4096、同時実行 1 |

ダミー重みの出力には置換文字が含まれた。応答の意味や品質は判定せず、本文も保存していない。
実モデルの重みの取得、TP=2、長文脈、性能、長時間の安定性、`/v1/messages` の互換性は未検証。

記録は `serving/var/20260922T155207Z-start-probe-nope-local/`、CLI の結果は
`serving/var/nope-build-0961bbae/probe.stdout.log` と `probe-result.json`。
派生構成の `config-sha256` は `5577373b91217724407066b542514ec46b98f362438aca4537324d6169bc24ca`。

15:56:16 UTC に head を読み取り確認し、Serving Kit のコンテナ、今回の補助コンテナ、
GPU を使っているプロセスがいずれも 0 件であることを確認した (`cleanup-check.json`)。
worker への配布や起動は行っていない。`exl3-tp2` の操作も行っていない。
自前イメージ、配布したソース、ビルドキャッシュ、実行記録は次の検証に使うため保持した。

## 次は実モデルでの検証を別に計画する

今回確認できたのは、固定したソースと C++ 修正、FlashInfer 0.7.0 の組み合わせで、
GB10 の NoPE の起動障害を縮小構成では回避できること。
次の候補は、同じイメージの worker への配布、本体の重みの取得・照合、TP=2 の起動と品質確認。
実機操作の範囲と取得容量を具体化してから、別途判断する。
