# NoPE 修正イメージを head でビルドし、縮小起動を確認する

状態: 2026-09-22〜23 (JST) に実施し、縮小起動まで合格。
[実行結果と途中の訂正](../results/2026-09-22-nope-build.md) を参照する。対象は head (`spark-153d`) の 1 台。
P1 の公式イメージ検証とは別の試行として、上流の C++ 修正と FlashInfer 0.7.0 を組み合わせる。
成功条件は、キャッシュの単独検査と、ダミー重みの縮小起動・短い要求が通り、検証用コンテナを片付けられること。
実モデルの品質、TP=2、性能は、この検証では判定しない。

書かないこと: 要求・応答の本文、認証情報、`exl3-tp2` の中身。

## 実行する変更と、その範囲

| 項目 | 固定する内容 |
|---|---|
| vLLM の基準 | `0961bbae2894d574be790d219651824eb199318e` |
| 上流の修正 | [vLLM #55277](https://github.com/vllm-project/vllm/pull/55277)、`bc2ee480738d7dcc558262a0c6d81956b515b050` → `8d09804c877c48165c6ba69bc9dc02d09bae0b83` の C++ と回帰テストだけ |
| FlashInfer | Python、cubin、JIT キャッシュを 0.7.0 に揃える |
| ビルド対象 | 上流 Dockerfile の `vllm-openai`、linux/arm64、CUDA 13.0.3 |
| ビルド用イメージ | `pytorch/manylinuxaarch64-builder@sha256:994bed2b225a9ff0f6fbe85c85fe84fbeac9031bb909442e18178484798529df`。Dockerfile の既定は amd64 用なので必ず上書きする |
| GPU の指定 | `torch_cuda_arch_list=12.0`。上流 CMake は CUDA 13 で SM12x の family target を使う |
| 並列数 | `max_jobs=16`、`nvcc_threads=2`。上流で割り算され、実質 8 並列になる。20 CPU と約 120 GB の利用可能メモリを確認して変更 |
| 新しいタグ | `vllm-nope:0961bbae-fi070`。既存なら上書きせず停止 |
| head の配布先 | `/home/j5ik2o/vllm-baseline/payload/nope-build-0961bbae/` |
| Mac の記録先 | `serving/var/nope-build-0961bbae/`、縮小起動の記録は `serving/var/<UTC>-start-probe-nope-local/` |

ビルドは `serve` の外で行う。`serve` のコマンド許可リストに `docker build` を追加しない。
補助スクリプトは上流の変更を Mac で準備し、Spark では編集しない。
今回だけのビルド用配布は、上記の head の専用ディレクトリに rsync する。
既存の `serve push` は両ノードに配るため、このビルド用ソースには使わない。

読み取り確認 (2026-09-22) では、head は aarch64、Docker 29.6.2、Buildx 0.35.0、
ドライバ 580.178.04。ファイルシステムの空きは約 2.66 TB、利用可能メモリは約 125.6 GB だった。
GPU のメモリ値は `nvidia-smi` で N/A だったため、GPU が空いている根拠にはしない。
ビルドの所要時間とピーク容量は未測定。最初は 100 GB 以上の空きを運用上の条件とし、
実行直前に再確認する。この値は所要容量の保証ではない。

## 1. Mac で差分を確認し、了承後にソースを用意する

リポジトリ直下で実行する。既定のモードはソース取得と `git apply --check` だけで、差分は適用しない。
出力先が既存なら停止する。GitHub への書き込みや Spark への接続は行わない。

```bash
python3 experiments/nope-mla/prepare_build.py serving/var/nope-build-check-20260922
```

`manifest.json` に基準コミット、修正元、差分の SHA-256、適用したかどうかを記録する。
差分は上流由来の 2 ファイルと、依存指定の 3 ファイル (`requirements/cuda.txt`、
`docker/Dockerfile`、`docker/versions.json`) に限る。

2026-09-22 の適用検査は合格 (`applied: false`)。生成した差分の SHA-256:

- `upstream.patch`: `3fecebaeed8b3f49475b0a5e0eaebf88966637e6089edf840491e688c1287ecc`
- `flashinfer-version.patch`: `0210f84b538b789694535334ff303fdf848ebc11b638c6c762b6b5e2ef477df1`

適用モードで再取得した際は、この 2 値と一致することを確認する。

⚠ 要件 8.6 と ADR 0005 に従い、パッチ適用と自前ビルドの了承後に実行する。

```bash
python3 experiments/nope-mla/prepare_build.py --apply serving/var/nope-build-0961bbae
cp experiments/nope-mla/check_cache.py serving/var/nope-build-0961bbae/
```

上流の依存関係には版の範囲指定やベースイメージのタグが残るため、同じソースからのビルドでも
全バイトが一致するとは限らない。実行するイメージは、ビルド後の完全な ID で固定する。

## 2. head に配布し、別名のイメージをビルドする

⚠ head への配布、イメージ取得を伴うビルド、キャッシュ作成は状態を変える。計画の了承後に実行する。
以下は Mac から打つ。配布先が既存の場合は再利用せず、内容を照合して別の試行名を決める。

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d \
  'test ! -e /home/j5ik2o/vllm-baseline/payload/nope-build-0961bbae'
rsync -a -e 'ssh -o BatchMode=yes -o ConnectTimeout=5' \
  serving/var/nope-build-0961bbae/ \
  spark-153d:/home/j5ik2o/vllm-baseline/payload/nope-build-0961bbae/
```

事前に `docker image inspect vllm-nope:0961bbae-fi070` が「存在しない」であることを確認する。
接続失敗や Docker の不調を「存在しない」と扱わない。
長いビルドは Mac の作業用ターミナルで実行し、終了コードとログを保存する。

```bash
set -o pipefail
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d \
  'cd /home/j5ik2o/vllm-baseline/payload/nope-build-0961bbae/source && docker buildx build --load --platform linux/arm64 --target vllm-openai --file docker/Dockerfile --tag vllm-nope:0961bbae-fi070 --build-arg torch_cuda_arch_list=12.0 --build-arg max_jobs=16 --build-arg nvcc_threads=2 --build-arg FLASHINFER_VERSION=0.7.0 --build-arg VLLM_BUILD_COMMIT=0961bbae2894d574be790d219651824eb199318e --build-arg VLLM_USE_PRECOMPILED=0 --build-arg USE_SCCACHE=0 --build-arg BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder@sha256:994bed2b225a9ff0f6fbe85c85fe84fbeac9031bb909442e18178484798529df --progress plain .' \
  2>&1 | tee serving/var/nope-build-0961bbae/build.log
```

ビルド失敗時はここで停止する。依存解決で版が衝突しても、`--no-deps` で押し通さない。
上流 Dockerfile が生成する中間レイヤーは残る。今回の後片付けでは `docker system prune` や既存イメージの削除を行わない。

## 3. イメージの ID と依存を照合し、C++ の単独検査を行う

読み取りだけの操作で、ビルド結果を Mac に保存する。

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d \
  'docker image inspect vllm-nope:0961bbae-fi070' \
  > serving/var/nope-build-0961bbae/image-inspect.json
python3 experiments/nope-mla/configure_probe.py \
  serving/var/nope-build-0961bbae/image-inspect.json \
  serving/var/nope-build-0961bbae/probe.toml
```

`configure_probe.py` は既存の `probe-nightly` を複製し、名前・説明・イメージの節だけを変更する。
イメージの `Id` と `Size`、linux/arm64 を確認する。タグ、短縮 ID、不正な値では生成しない。
生成後、同ディレクトリの記録とこの手順を照合してから使用する。

依存の検査は GPU 単独検査と分けて記録する。初回の実行で、`pip check` は公式 nightly と
自前イメージの両方に同じ 2 件 (NCCL の明示的な版の上書き、cuSPARSELt の `sbsa` タグ) を報告した。
NCCL の指定は上流 Dockerfile の意図と一致し、cuSPARSELt の実体は AArch64 で読み込みにも成功した。
この照合結果は [実行記録](../results/2026-09-22-nope-build.md) と
`dependency-audit-control.json` / `dependency-audit-treatment.json` にある。
この 2 件だけであることを確認した場合は、パッケージを変更せず単独検査へ進む。
別の指摘、版の違い、ライブラリの読み込み失敗があれば停止する。`pip check` の終了コード 1 を一律に無視しない。

⚠ 単独検査は GPU を使うコンテナを head に 1 つ起動する。直前に GPU の使用状況と名前の衝突を確認し、了承後に実行する。
次の `<IMAGE_ID>` は保存した JSON の `Id` (`sha256:` と 64 桁) に置き換える。タグは渡さない。

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d \
  'docker run -d --name vllm-nope-cache-check --label io.ideo-plus.nope-experiment=0961bbae --gpus all --network none --mount type=bind,src=/home/j5ik2o/vllm-baseline/payload/nope-build-0961bbae/check_cache.py,dst=/verification/check_cache.py,readonly --entrypoint /bin/bash <IMAGE_ID> -lc "python3 -m pip freeze && python3 /verification/check_cache.py"'
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d \
  'docker wait vllm-nope-cache-check'
ssh -o BatchMode=yes -o ConnectTimeout=5 spark-153d \
  'docker logs vllm-nope-cache-check' \
  > serving/var/nope-build-0961bbae/cache-check.log 2>&1
```

合格には `docker wait` が表示する終了コード 0 と、`cache_check: passed` の両方が必要。
`docker wait` コマンド自体の終了コードだけでは判定しない。
10 分で終わらなければタイムアウトとして停止する。ログを回収し、所有ラベルとイメージ ID を確認したうえで、
この検査コンテナだけを `docker stop -t 30`、`docker rm` の順で片付ける。
起動が失敗して同名のコンテナが存在した場合は、所有を確認できるまで停止・削除しない。

この検査は NoPE とゼロの RoPE 入力で出力バイトが一致すること、予約領域のゼロ埋め、
対象外の行の保持、従来の RoPE 値の保持を確認する。上流の全 GPU テストを通したとは扱わない。
`pip freeze` の記録はローカル保存に留め、公開前に取得元 URL などを点検する。

## 4. 既存の serve probe で縮小起動する

まず Mac の `serving/` から前提を読み取る。

```bash
cd serving
uv run serve check probe-nope-local --configs var/nope-build-0961bbae/probe.toml
```

ローカルイメージは `.Id` の完全一致で照合する。`serve pull-image` は使わない。
GPU の使用、設定ファイルの照合、ポートやディスクなど、既存の関門もすべて通る必要がある。

⚠ 縮小起動とその後片付けについて了承を得た後、次を実行する。

```bash
uv run serve probe probe-nope-local \
  --configs var/nope-build-0961bbae/probe.toml --timeout 45m --yes
```

合格は、`ready`、短い要求の HTTP 200 と正常な終了、記録の回収とコンテナの削除が揃った場合。
`probe` は短い要求が失敗しても `ready` を返し得るため、終了コード 0 だけでは合格にしない。
ダミー重みの文に意味があるかは判定しない。本文は保存しない。

`failed`、`inconclusive`、HTTP の失敗、片付け失敗なら停止する。新しいエラーは別の試行として記録し、
同じ条件の再実行は行わない。成功した場合も、重みの取得や worker への配布は次の判断に分ける。

## 根拠と適用範囲

- ビルド用イメージの指定は上流の [arm64 ビルド設定](https://github.com/vllm-project/vllm/blob/0961bbae2894d574be790d219651824eb199318e/.buildkite/image_build/image_build_arm64.sh) に基づく。配布元のタグ API で linux/arm64 のダイジェストを確認した。2026-09-22 の初回ビルドはこの指定がなく `exec format error` で失敗し、手順を訂正した。
- 上流の [Dockerfile](https://github.com/vllm-project/vllm/blob/0961bbae2894d574be790d219651824eb199318e/docker/Dockerfile) の target と ARG、[CMakeLists.txt](https://github.com/vllm-project/vllm/blob/0961bbae2894d574be790d219651824eb199318e/CMakeLists.txt) の CUDA 13 の対応アーキテクチャを確認した。起動コマンドはこのリポジトリの既存構成から導く。
- 依存修正と配布物の根拠は [調査メモ](patch-investigation.md)。上流変更は上流 vLLM の Apache-2.0 のソースとして取得し、差分と出典を残す。第三者のレシピは参照しない。
- ローカルイメージ ID の対応は P1 後の道具の拡張。P1 の公式イメージ限定という実施条件や、その失敗記録を書き換えない。
- 初回は計測者からパッチ適用・配布・ビルド・GPU 単独検査・縮小起動・後片付けまでの了承を得て実行した。
  新しい試行を行うときは、変更内容と実機操作の範囲を明示する。
