# 手元で変換した重み (K2 の FP8 など) を作って使う手順 (`p2-nope-tp2-full-k2s1`)

この手順は、#56 の変換の道具で、今の重み (`RedHatAI/GLM-5.3-Flash-NVFP4`、rev `18d55bfd5a2194887738da73753975c9d3842f46`) の一部を FP8 にした**派生の重み**を作り、`serving` の構成から使うまでの枠です。変換は推論用のイメージのコンテナの中で **CPU だけ**で行い、2 台それぞれで作ったマニフェストを突き合わせてから、`serving/weights/` の下にコミットします。手順の後半は [1 ステップの GPU の時間の内訳を測る手順](k2-profile-procedure.md) と [ADR 0006](../decisions/0006-path-pruning.md) の K2 に進むための下ごしらえです。

この文書に書かないこと: **送った内容**と応答の本文、**認証の情報**、`exl3-tp2` の中身。変換は手元のコンテナの中で完結し、Hub から何も取りません (匿名のままです)。`exl3-tp2` は別の計測者の構成なので調べません。この文書は手順であって記録ではないので、実機で成功したとは書きません。**⚠ の付いた操作は、計測者に了承を得てから行います。**

## 0. 前提: コミットしたマニフェストと、変換の道具

#56 の変換の道具 (この手順では `<変換の道具>` と書きます) は、変換の終わりに、変換の結果のマニフェストを JSON で書きます。`serving` 側は、この JSON を**契約**として読みます。形は次のとおりです。

```json
{
  "kind": "derived",
  "derivation": {
    "name": "k2s1",
    "origin": { "repo": "RedHatAI/GLM-5.3-Flash-NVFP4", "revision": "18d55bfd5a2194887738da73753975c9d3842f46" },
    "conversion": {
      "tool": "experiments/k2-quant/<変換の道具>",
      "commit": "<変換の道具の 40 桁の commit>",
      "args": ["--dtype", "fp8"],
      "target_pattern": "^model\\.layers\\.\\d+\\.self_attn\\..*$"
    }
  },
  "generated_at": "2026-09-26T00:00:00Z",
  "total_bytes": 0,
  "files": [ { "path": "config.json", "sha256": "<64 桁>", "size": 0 } ]
}
```

`kind = "derived"`、`derivation` (名前・元の重み・変換の条件)、`generated_at`、`total_bytes`、`files` が要ります。元の重み (`origin.repo` と `origin.revision`) は、コミット済みの `p1-nvfp4-tp2` の重みと一致していなければなりません。

この一致は、どの段でも同じものを見るわけではありません。段ごとに、確かめる内容が違います。

- **生成 (§3 の `configure_tp2.py --weights`)**: `--weights` の値と `derivation.name` が同じこと。`derivation.origin` の `repo` と `revision` が、コミット済みの `p1-nvfp4-tp2` の重みと同じこと。合わなければ、出力を作らずに断ります。**`p1-nvfp4-tp2` の重みと突き合わせるのは、この段だけです。**
- **読み込み (`load_configs`。`serve check` を含む、どのサブコマンドでも最初に通ります)**: 構成の `manifest` (`k2s1.manifest.json`) と `origin.manifest` (`RedHatAI__GLM-5.3-Flash-NVFP4.manifest.json`) が `serving/weights/` の下に実在すること。元の重みのマニフェストの中身が Hub の重みのマニフェストとして読めて、その `repo` と `revision` が、構成に書いた `origin` と同じこと。合わなければ、実機に触れる前に断ります。読み込みは `p1-nvfp4-tp2` の重みを見ません (構成に書いた `origin` と、元の重みのマニフェストの中身を突き合わせます)。
- **照合 (§4 の `serve verify`)**: 構成の `derivation` (名前・元の重み・変換の条件) と、コミットした変換の結果のマニフェストの `derivation` が同じこと。違えば、実機に触れる前に断ります。同じなら、実機の `sha256sum` の結果をマニフェストと突き合わせます。
- **関門 `weights_verified` (§5 の `serve check`、`serve start`)**: 照合と同じ `derivation` の一致に加えて、実機の照合の記録の `derivation`、ファイルの数、大きさの合計、合わないファイルの有無が、マニフェストと合うこと。**この関門は、元の重みのマニフェストを読みません。**

変換の道具 (`<変換の道具>`) の道筋は、次のように使い分けます。マニフェストの `derivation.conversion.tool` には、リポジトリの中の道筋 (`experiments/k2-quant/<変換の道具>`) を書き、`commit` にはその道具を含むコミットを書きます。`serve push` が Spark に配るのは `serving/payload/` の下だけなので、Spark の `/home/j5ik2o/vllm-baseline/payload/k2-quant/<変換の道具>` に置く道具は、`tool` と同じ `commit` の内容の写しにします。写しを `serving/payload/k2-quant/` に置いて配るか、別の方法にするかは、#56 の道具ができてから決めます (§8)。配るときは、[準備 1 の配布](procedure.md) と同じ `serve push` を、⚠ 了承を得てから打ちます。

コンテナの中では、配った `/home/j5ik2o/vllm-baseline/payload` を読み取り専用で `/tools` に結び付けます。**元の重みは `/origin` に読み取り専用で結び付け、`/origin` に入っているのは重みだけ**なので、変換の道具は `/tools` から実行します。

## 1. ⚠ 変換 (推論用のイメージのコンテナの中。CPU だけ。2 台それぞれ)

⚠ 了承を得てから実行する。この節の `ssh` と `docker run` は、Spark の上でコンテナを起こしてファイルを書く。**この節のコマンドは、計測者の了承を得てから、対話側で打ちます。**

変換は、**推論用のイメージ `sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90` のコンテナの中**で行います。GPU は渡しません (`--gpus` も `--runtime nvidia` も付けません)。ネットワークも切ります (`--network none`。Hub から何も取らないことを、コマンドで保証します)。このイメージの既定の起動は `vllm serve` なので、`--entrypoint python3` で差し替えます (`configs.toml` の取得・計測の構成が `--entrypoint` で差し替えているのと同じ理由です。`python3` は、[パッチ版のビルドの手順書](patched-build-procedure.md) がこのイメージの中で使っています)。変換の道具は `/tools` (配った payload)、元の重みは `/origin` に**読み取り専用**で結び付け、変換の結果は `/home/j5ik2o/vllm-baseline/models/k2s1` (構成の中では `{remote_root}/models/k2s1`) に書きます。

この節のコマンドは、シェルが打つものなので、`serve` の構成の置き換えの印 `{remote_root}` は使えません。`nodes.toml` の `remote_root` の実際の値 (`/home/j5ik2o/vllm-baseline`) で書いています。変換の結果の置き場所 `models/k2s1` は、`serve push` が作らず、`docker run` の `--mount type=bind` も作らない (置き場所がなければ失敗する) ので、先に作ります。

⚠ 了承を得てから実行する。置き場所を作り、2 台それぞれで変換する。

```bash
# 変換の結果の置き場所を、2 台に作る
ssh spark-153d mkdir -p /home/j5ik2o/vllm-baseline/models/k2s1
ssh spark-5083 mkdir -p /home/j5ik2o/vllm-baseline/models/k2s1

# head (spark-153d)。CPU だけで変換する。--gpus も --runtime nvidia も付けない
ssh spark-153d docker run --rm --network none \
  --entrypoint python3 \
  --mount type=bind,source=/home/j5ik2o/vllm-baseline/payload,target=/tools,readonly \
  --mount type=bind,source=/home/j5ik2o/vllm-baseline/models/glm-5-3-flash-nvfp4,target=/origin,readonly \
  --mount type=bind,source=/home/j5ik2o/vllm-baseline/models/k2s1,target=/derived \
  sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90 \
  /tools/k2-quant/<変換の道具> --input /origin --output /derived --dtype fp8

# worker (spark-5083)。同じ道具、同じ引数で、2 台それぞれのマニフェストを作る
ssh spark-5083 docker run --rm --network none \
  --entrypoint python3 \
  --mount type=bind,source=/home/j5ik2o/vllm-baseline/payload,target=/tools,readonly \
  --mount type=bind,source=/home/j5ik2o/vllm-baseline/models/glm-5-3-flash-nvfp4,target=/origin,readonly \
  --mount type=bind,source=/home/j5ik2o/vllm-baseline/models/k2s1,target=/derived \
  sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90 \
  /tools/k2-quant/<変換の道具> --input /origin --output /derived --dtype fp8
```

変換は時間がかかります (重みは約 184 GiB)。2 台で並行に進めてかまいません。

## 2. マニフェストを Mac に写し、2 台の一致を確かめてコミットする

2 台それぞれが書いたマニフェストを Mac に写し、`sha256sum` で内容を確かめてから、`cmp` で突き合わせます。ただし、マニフェストは `generated_at` (書き込んだ時刻) を持つので、2 台が別々に変換すれば**時刻の欄は必ず違います**。そのまま `cmp` すると、中身が同じでも必ず失敗します。比べるのは `generated_at` を除いた中身 (`derivation` と `total_bytes` と `files`) です。**1 バイトでも違えば、変換の条件か入力が違うので、コミットしません。** 一致したら、`serving/weights/k2s1.manifest.json` としてコミットします。取り込みの口は、この 1 ファイルだけです (実機の `sha256sum` の結果は、あとで `serve verify` が照合します)。

```bash
# 2 台ぶんを Mac に写す (rsync の道筋は、回収の場所に合わせる)
sha256sum serving/var/k2s1/spark-153d.manifest.json serving/var/k2s1/spark-5083.manifest.json
# 時刻の欄 (generated_at) を除いてから比べる
jq -S 'del(.generated_at)' serving/var/k2s1/spark-153d.manifest.json > serving/var/k2s1/norm-153d.json
jq -S 'del(.generated_at)' serving/var/k2s1/spark-5083.manifest.json > serving/var/k2s1/norm-5083.json
cmp serving/var/k2s1/norm-153d.json serving/var/k2s1/norm-5083.json
cp serving/var/k2s1/spark-153d.manifest.json serving/weights/k2s1.manifest.json
```

`serving/weights/` に置いたマニフェストが、**コミットした正解**になります。`generated_at` が 2 台で違っても、`serving` は `generated_at` を照合に使わないので、どちらのファイルをコミットしても照合の正解は同じです。

## 3. 構成の生成

派生の重みを使う構成を生成します。`--weights k2s1` を付けると、構成の名前の末尾に `-k2s1` が付き、重みの結び付けが `{remote_root}/models/k2s1` (読み取り専用) に差し替わります。出力は `tp2-full-k2s1.toml` にそろえます。

```bash
uv run --directory serving python ../experiments/nope-mla/configure_tp2.py \
  --variant full --weights k2s1 \
  ../serving/var/nope-build-0961bbae/image-inspect.json \
  ../serving/var/nope-build-0961bbae/tp2-full-k2s1.toml
```

生成した `tp2-full-k2s1.toml` は、`load_configs` で読めます。生成の時点で確かめるのは、`--weights` の値と `derivation.name` の一致と、`derivation.origin` と `p1-nvfp4-tp2` の重みの一致だけです (§0)。合わなければ、出力は作られずに断られます。変換の結果のマニフェストと元の重みのマニフェストの実在、および元の重みのマニフェストの中身は、生成のあとに `load_configs` が確かめます。

## 4. ⚠ 実機の重みを照合する (`serve verify`)

⚠ 了承を得てから実行する。`serve verify` は、Spark の上で `sha256sum` を流し、`state/k2s1.derived.verified.json` に照合の記録を置く。読み取りと記録の配布だけで、コンテナは起こさない。

`--configs` には、§3 で生成した `tp2-full-k2s1.toml` を渡します (派生の構成は、`configs.toml` にはコミットしていません)。照合の記録がなければ、§5 の `serve check` で関門 `weights_verified` が落ちます。

⚠ 了承を得てから実行する。次の `serve verify` が、実機の重みを照合し、記録を `state/` に置く。

```bash
uv run --directory serving serve verify p2-nope-tp2-full-k2s1 \
  --configs ../serving/var/nope-build-0961bbae/tp2-full-k2s1.toml \
  --yes
```

## 5. 関門をすべて通す (`serve check`)

`serve check` は、まず `load_configs` で構成を読み (マニフェストの実在と、元の重みのマニフェストの中身。§0)、そのあと 8 つの関門を流します。派生の重みでも、Hub の重みと同じ厳しさです。関門 `weights_verified` は、構成の `derivation` とマニフェストの `derivation` の一致と、照合の記録の一致を確かめます (§0)。

```bash
uv run --directory serving serve check p2-nope-tp2-full-k2s1 \
  --configs ../serving/var/nope-build-0961bbae/tp2-full-k2s1.toml
```

## 6. ⚠ 起動と短い確認 (`serve start` と `serve smoke`)

⚠ 了承を得てから実行する。`serve start` は、2 台でコンテナを起こす。`serve smoke` は、短い要求を 1 つずつ送る (応答の本文は画面に出すだけで、どこにも残さない)。

```bash
uv run --directory serving serve start p2-nope-tp2-full-k2s1 \
  --configs ../serving/var/nope-build-0961bbae/tp2-full-k2s1.toml \
  --yes

uv run --directory serving serve smoke p2-nope-tp2-full-k2s1 \
  --configs ../serving/var/nope-build-0961bbae/tp2-full-k2s1.toml
```

## 7. ⚠ 記録を回収して止める (`serve logs` と `serve stop`)

計測が済んだら、記録を回収してから止めます。`serve stop` は記録を回収しないので、必ず `serve logs` を先に打ちます。`serve logs` は、§3 で生成した構成の定義を読むので、§4〜§6 と同じ `--configs` を付けます (派生の構成は `configs.toml` にありません)。`serve stop` は構成の名前を取らず、自分のコンテナを 2 台とも止めて消します。

⚠ 了承を得てから実行する。`serve stop` は、2 台の自分のコンテナを止めて消す。

```bash
uv run --directory serving serve logs p2-nope-tp2-full-k2s1 \
  --configs ../serving/var/nope-build-0961bbae/tp2-full-k2s1.toml
uv run --directory serving serve stop --yes
```

## 8. 実機でしか分からないこと

- 変換の道具が、CPU だけで現実的な時間で終わるか。2 台の結果が一致するか。
- 変換の道具を Spark の `payload/k2-quant/` に置く方法 (`serving/payload/` に写して `serve push` で配るか、別の方法か)。#56 の道具ができてから決める。
- 派生の重みを読み込んだときの、vLLM の起動の時間と GPU のメモリ。FP8 にした層が、期待した層だけか。
- 匿名のまま Hub に触れずに、`serve check` と `serve verify` が通るか。
