# 手元で変換した重み (K2 の FP8 など) を作って使う手順 (`p2-nope-tp2-full-k2s1`)

この手順は、#56 の変換の道具で、今の重み (`RedHatAI/GLM-5.3-Flash-NVFP4`、rev `18d55bfd5a2194887738da73753975c9d3842f46`) の一部を FP8 にした**派生の重み**を作り、`serving` の構成から使うまでの枠です。変換は推論用のイメージのコンテナの中で **CPU だけ**で行い、2 台それぞれで作ったマニフェストを Mac で `serve derived-import` に渡して突き合わせ、`serving/weights/` の下にコミットします。手順の後半は [1 ステップの GPU の時間の内訳を測る手順](k2-profile-procedure.md) と [ADR 0006](../decisions/0006-path-pruning.md) の K2 に進むための下ごしらえです。

この文書に書かないこと: **送った内容**と応答の本文、**認証の情報**、`exl3-tp2` の中身。変換は手元のコンテナの中で完結し、Hub から何も取りません (匿名のままです)。`exl3-tp2` は別の計測者の構成なので調べません。この文書は手順であって記録ではないので、実機で成功したとは書きません (実機で分かったことは、日付を付けて §8 に足します)。**⚠ の付いた操作は、計測者に了承を得てから行います。**

## 0. 前提: 道具が書くマニフェストと、serving が読むマニフェスト

#56 の変換の道具 (`experiments/k2-quant`。`python3 -m k2_quant`) は、変換の終わりに、変換の結果の置き場所へ `manifest.json` を書きます。道具が書く形は次のとおりです。**`generated_at` も `commit` も書きません** (同じ入力から同じバイト列にするためと、コンテナの中の写しからは分からないためです)。

```json
{
  "conversion": {
    "tool": "k2-quant",
    "tool_version": "0.2.0",
    "source": { "repo": "RedHatAI/GLM-5.3-Flash-NVFP4", "revision": "18d55bfd5a2194887738da73753975c9d3842f46" },
    "pattern": ".*\\.layers\\.(?:0|1|2)\\.mlp\\.(?:gate|up|down)_proj$|.*\\.layers\\.\\d+\\.mlp\\.shared_experts\\.(?:gate|up|down)_proj$",
    "args": [
      "--source-repo", "RedHatAI/GLM-5.3-Flash-NVFP4",
      "--source-revision", "18d55bfd5a2194887738da73753975c9d3842f46",
      "--pattern", ".*\\.layers\\.(?:0|1|2)\\.mlp\\.(?:gate|up|down)_proj$|.*\\.layers\\.\\d+\\.mlp\\.shared_experts\\.(?:gate|up|down)_proj$"
    ],
    "modules": ["model.language_model.layers.0.mlp.gate_proj"],
    "weight_dtype": "F8_E4M3",
    "scale_dtype": "F32",
    "strategy": "channel"
  },
  "files": [ { "path": "config.json", "size": 0, "sha256": "<64 桁>" } ],
  "total_bytes": 0
}
```

`conversion.args` は、道具に渡した引数のうち、**変換の結果を決めるもの**だけです (`--source-repo`、`--source-revision`、`--pattern`。`--pattern` は渡さなくても、実際に使った既定の値を書きます)。`--source` と `--output` はマウントの位置に依存し、`--link` は出力のバイト列を変えないので、書きません。`modules` は、実際には、選ばれたモジュールの名前がすべて、名前順に並びます (見本は 1 つだけです)。

`serving` が読む形は、`serving/weights/k2s1.manifest.json` の、次の派生のマニフェストです。**道具の manifest から `serve derived-import` が作ります。手で書きません。**

```json
{
  "kind": "derived",
  "derivation": {
    "name": "k2s1",
    "origin": { "repo": "RedHatAI/GLM-5.3-Flash-NVFP4", "revision": "18d55bfd5a2194887738da73753975c9d3842f46" },
    "conversion": {
      "tool": "experiments/k2-quant",
      "commit": "<道具を含む 40 桁のコミット>",
      "args": [
        "--source-repo", "RedHatAI/GLM-5.3-Flash-NVFP4",
        "--source-revision", "18d55bfd5a2194887738da73753975c9d3842f46",
        "--pattern", ".*\\.layers\\.(?:0|1|2)\\.mlp\\.(?:gate|up|down)_proj$|.*\\.layers\\.\\d+\\.mlp\\.shared_experts\\.(?:gate|up|down)_proj$"
      ],
      "target_pattern": ".*\\.layers\\.(?:0|1|2)\\.mlp\\.(?:gate|up|down)_proj$|.*\\.layers\\.\\d+\\.mlp\\.shared_experts\\.(?:gate|up|down)_proj$"
    }
  },
  "generated_at": "2026-09-26T00:00:00Z",
  "total_bytes": 0,
  "files": [ { "path": "config.json", "sha256": "<64 桁>", "size": 0 } ]
}
```

`kind = "derived"`、`derivation` (名前・元の重み・変換の条件)、`generated_at`、`total_bytes`、`files` が要ります。取り込みの対応は、次のとおりです。

- `derivation.conversion.args` は、道具の `conversion.args` をそのまま、`target_pattern` は `conversion.pattern`、`origin` は `conversion.source` から作ります。
- `derivation.conversion.tool` は、リポジトリの中の道具の道筋 `experiments/k2-quant` (`--tool` で変えられます)、`commit` は、その道具を含むコミットです (`--commit`)。この 2 つは、Mac の git の文脈でしか決まりません。
- `files` と `total_bytes` は、道具の manifest から作ります。Hub のマニフェストと同じ規則で、モデルカードらしい名前と `.gitattributes` は載せません。

元の重み (`origin.repo` と `origin.revision`) は、コミット済みの `p1-nvfp4-tp2` の重みと一致していなければなりません。

この一致は、どの段でも同じものを見るわけではありません。段ごとに、確かめる内容が違います。

- **取り込み (§2 の `serve derived-import`)**: 2 台の manifest が、最上位の `generated_at` を除いて一致すること。道具の manifest に `conversion.args` があること (ない古い形は補わずに断ります)。宛先に Hub のマニフェストがあれば上書きしません。
- **生成 (§3 の `configure_tp2.py --weights`)**: `--weights` の値と `derivation.name` が同じこと。`derivation.origin` の `repo` と `revision` が、コミット済みの `p1-nvfp4-tp2` の重みと同じこと。合わなければ、出力を作らずに断ります。**`p1-nvfp4-tp2` の重みと突き合わせるのは、この段だけです。**
- **読み込み (`load_configs`。`serve check` を含む、どのサブコマンドでも最初に通ります)**: 構成の `manifest` (`k2s1.manifest.json`) と `origin.manifest` (`RedHatAI__GLM-5.3-Flash-NVFP4.manifest.json`) が `serving/weights/` の下に実在すること。元の重みのマニフェストの中身が Hub の重みのマニフェストとして読めて、その `repo` と `revision` が、構成に書いた `origin` と同じこと。派生の重みが `kind = "probe"` の構成に結び付いていないこと。合わなければ、実機に触れる前に断ります。読み込みは `p1-nvfp4-tp2` の重みを見ません (構成に書いた `origin` と、元の重みのマニフェストの中身を突き合わせます)。
- **照合 (§4 の `serve verify`)**: 構成の `derivation` (名前・元の重み・変換の条件) と、コミットした変換の結果のマニフェストの `derivation` が同じこと。違えば、実機に触れる前に断ります。同じなら、実機の `sha256sum` の結果をマニフェストと突き合わせ、照合の記録に、**照合に使ったマニフェストのバイト列の SHA-256** (`manifest_sha256`) を書きます。
- **関門 `weights_verified` (§5 の `serve check`、`serve start`)**: 照合と同じ `derivation` の一致に加えて、実機の照合の記録の `derivation`、ファイルの数、大きさの合計、合わないファイルの有無が、マニフェストと合うこと。さらに、記録の `manifest_sha256` が、いまコミットしているマニフェストのバイト列の SHA-256 と同じであること (`derivation`・件数・合計が同じでも、1 ファイルの sha256 が違うマニフェストに差し替えれば、古い記録では落ちます。差し替えたら `serve verify` をやり直します)。**この関門は、元の重みのマニフェストを読みません。**

## 1. ⚠ 道具を配り、変換する (推論用のイメージのコンテナの中。CPU だけ。2 台それぞれ)

道具の本体は `experiments/k2-quant/k2_quant/` にあります。Spark に配るのは `serving/payload/` の下だけ (`serve push`) なので、道具の**写し**を `serving/payload/k2-quant/k2_quant/` に置き、`serve push` で 2 台の `payload/k2-quant/k2_quant/` に配ります。写しが道具と同じであることは、試験で固定しています (`serving/tests/unit/test_payload.py`。名前の集合とバイト列が同じ)。

配ったものが、コミットした道具の内容と同じであることは、次の 3 つで確かめます。どれも、`serve` に新しい口を足さずに済ませています。

1. **Mac で、作業木がコミットと同じであること。** 道具と写しに未コミットの変更がなく、`HEAD` が §2 で `--commit` に渡すコミットであること。
2. **`serve push` が、Mac の `serving/payload/` の中身を、そのまま 2 台の `payload/` に写すこと。** 転送のあとの一致は、`rsync` の終了コードが保証します。
3. **配ったあとに、2 台の写しの `sha256sum` を、ファイル名の順に並べ替えてから、Mac の `shasum -a 256` と突き合わせること。** 手順は次のとおりです。

```bash
# 道具と、その写しに、未コミットの変更がないこと (何も出ないこと)
git status --porcelain -- experiments/k2-quant serving/payload/k2-quant

# いまのコミット。§2 の --commit に渡す 40 桁
git rev-parse HEAD
```

⚠ 了承を得てから実行する。`serve push` は、2 台の `remote_root` の下に置き場所を作り、`serving/payload/` の中身 (道具の写しを含む) を `payload/` に配る。

```bash
uv run --directory serving serve push --yes
```

⚠ 了承を得てから実行する。2 台の `ssh` で、配った写しの `sha256sum` を読み、Mac の写しと突き合わせる (読み取りだけ。何も出なければ一致)。`*.py` の展開順はロケールで違い、Spark と Mac で並び順が違うので、中身が同じでも差分が出る (§8)。そのため、両側とも、出力を `LC_ALL=C sort` でファイル名の順に並べ替えてから比べる。

```bash
diff \
  <(ssh spark-153d 'cd /home/j5ik2o/vllm-baseline/payload/k2-quant/k2_quant && sha256sum *.py | LC_ALL=C sort -k 2') \
  <(cd serving/payload/k2-quant/k2_quant && shasum -a 256 *.py | LC_ALL=C sort -k 2)
diff \
  <(ssh spark-5083 'cd /home/j5ik2o/vllm-baseline/payload/k2-quant/k2_quant && sha256sum *.py | LC_ALL=C sort -k 2') \
  <(cd serving/payload/k2-quant/k2_quant && shasum -a 256 *.py | LC_ALL=C sort -k 2)
```

変換は、**推論用のイメージ `sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90` のコンテナの中**で行います。GPU は渡しません (`--gpus` も `--runtime nvidia` も付けません)。ネットワークも切ります (`--network none`。Hub から何も取らないことを、コマンドで保証します)。このイメージの既定の起動は `vllm serve` なので、`--entrypoint python3` で差し替え、道具を `python3 -m k2_quant` として動かします (`configs.toml` の取得・計測の構成が `--entrypoint` で差し替えているのと同じ理由です。`python3` は、[パッチ版のビルドの手順書](patched-build-procedure.md) がこのイメージの中で使っています)。

道具の写しは `/tools/k2-quant` に、元の重みは `/origin` に、それぞれ**読み取り専用**で結び付け、変換の結果は `/derived` (Spark の `/home/j5ik2o/vllm-baseline/models/k2s1`。構成の中では `{remote_root}/models/k2s1`) に書きます。作業ディレクトリを `-w /tools/k2-quant` にするので、`-m k2_quant` が写しの `k2_quant` パッケージを読みます。道具の引数は、実際の CLI (`python3 -m k2_quant --help`) のとおり、`--source`・`--output`・`--source-repo`・`--source-revision` が必須で、`--pattern` は既定の範囲 (第 1 段のうち、dense と共有の専門家。`lm_head` と MTP の `shared_head.head` は、vLLM 0961bbae が FP8 (W8A16) の `ParallelLMHead` を読めない (#68。§8) ので、既定から外しています) を使うので渡しません。

`--link` は付けません。`--link` は、写さずにハードリンクにする option ですが、元の重み (`/origin`) と出力 (`/derived`) は**別々の bind mount** です。同じファイルシステムの上でも、マウントの境界をまたぐハードリンクは `EXDEV` で失敗します。したがって、対象を含まない shard も写すので、`models/` には写しのぶん (約 184 GiB) の空きが要ります。実機の空きは未確認です (§8)。

この節のコマンドは、シェルが打つものなので、`serve` の構成の置き換えの印 `{remote_root}` は使えません。`nodes.toml` の `remote_root` の実際の値 (`/home/j5ik2o/vllm-baseline`) で書いています。変換の結果の置き場所 `models/k2s1` は、`serve push` が作らず、`docker run` の `--mount type=bind` も作らない (置き場所がなければ失敗する) ので、先に**空で**作ります。道具は、存在する空のディレクトリを出力先として受け付け、中身があれば何も書かずに終了 1 になります (やり直すときは、`models/k2s1` の中身を、計測者が消してから流します)。

⚠ 了承を得てから実行する。置き場所を作り、2 台それぞれで変換する。

```bash
# 変換の結果の置き場所を、2 台に、空で作る
ssh spark-153d mkdir -p /home/j5ik2o/vllm-baseline/models/k2s1
ssh spark-5083 mkdir -p /home/j5ik2o/vllm-baseline/models/k2s1

# head (spark-153d)。CPU だけで変換する。--gpus も --runtime nvidia も付けない
# --link は付けない (別々の bind mount の間のハードリンクは EXDEV で失敗する)
ssh spark-153d docker run --rm --network none \
  --entrypoint python3 \
  --mount type=bind,source=/home/j5ik2o/vllm-baseline/payload/k2-quant,target=/tools/k2-quant,readonly \
  --mount type=bind,source=/home/j5ik2o/vllm-baseline/models/glm-5-3-flash-nvfp4,target=/origin,readonly \
  --mount type=bind,source=/home/j5ik2o/vllm-baseline/models/k2s1,target=/derived \
  -w /tools/k2-quant \
  sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90 \
  -m k2_quant --source /origin --output /derived \
  --source-repo RedHatAI/GLM-5.3-Flash-NVFP4 --source-revision 18d55bfd5a2194887738da73753975c9d3842f46

# worker (spark-5083)。同じ道具、同じ引数で、2 台それぞれのマニフェストを作る
ssh spark-5083 docker run --rm --network none \
  --entrypoint python3 \
  --mount type=bind,source=/home/j5ik2o/vllm-baseline/payload/k2-quant,target=/tools/k2-quant,readonly \
  --mount type=bind,source=/home/j5ik2o/vllm-baseline/models/glm-5-3-flash-nvfp4,target=/origin,readonly \
  --mount type=bind,source=/home/j5ik2o/vllm-baseline/models/k2s1,target=/derived \
  -w /tools/k2-quant \
  sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90 \
  -m k2_quant --source /origin --output /derived \
  --source-repo RedHatAI/GLM-5.3-Flash-NVFP4 --source-revision 18d55bfd5a2194887738da73753975c9d3842f46
```

変換は時間がかかります (重みは約 184 GiB)。2 台で並行に進めてかまいません。

## 2. マニフェストを Mac に取り込み、2 台の一致を確かめてコミットする

2 台それぞれが書いた `models/k2s1/manifest.json` を Mac に写し、`serve derived-import` に**2 つとも**渡します。`serve derived-import` は、2 台の manifest が、最上位の `generated_at` を除いて一致するときだけ、派生のマニフェストを書きます (道具の manifest は `generated_at` を書かないので、通常は 2 台の manifest がバイト単位で同じです)。**1 か所でも違えば、違う箇所 (`files[2].sha256` など) を示して終了コード 1 になり、何も書きません。** 違う場合は、変換の条件か入力が違うので、コミットしません。

⚠ 了承を得てから実行する。2 台の `models/k2s1/manifest.json` を、`rsync` で Mac の `serving/var/k2s1/` (`.gitignore` 済み) に写す (読み取りだけ)。

```bash
mkdir -p serving/var/k2s1
rsync -a spark-153d:/home/j5ik2o/vllm-baseline/models/k2s1/manifest.json serving/var/k2s1/spark-153d.manifest.json
rsync -a spark-5083:/home/j5ik2o/vllm-baseline/models/k2s1/manifest.json serving/var/k2s1/spark-5083.manifest.json
```

次は Mac だけの操作です (Spark に触りません)。`--commit` には、§1 で確かめた、道具を含むコミット (`git rev-parse HEAD`) を渡します。

```bash
uv run --directory serving serve derived-import --name k2s1 --commit <40桁> \
  ../serving/var/k2s1/spark-153d.manifest.json ../serving/var/k2s1/spark-5083.manifest.json
```

標準出力の `key=value` の行に、`file_count`・`total_bytes`・除いたファイル (`excluded`)・**`manifest_sha256`** (書いたファイルのバイト列の SHA-256) が出ます。`serving/weights/k2s1.manifest.json` が、**コミットした正解**になります。宛先が Hub のマニフェストなら、上書きせずに断ります。同じ中身で取り込み直したときは、前の `generated_at` を保つので、時刻だけの差分は出ません。

`serving/weights/k2s1.manifest.json` をコミットしてから、§3 に進みます。コミットしたあとにマニフェストを差し替えたら、§4 の `serve verify` をやり直します (照合の記録は、マニフェストのバイト列の SHA-256 に結び付いているので、古い記録では関門が落ちます)。

## 3. 構成の生成

派生の重みを使う構成を生成します。`--weights k2s1` を付けると、構成の名前の末尾に `-k2s1` が付き、重みの結び付けが `{remote_root}/models/k2s1` (読み取り専用) に差し替わります。出力は `tp2-full-k2s1.toml` にそろえます。

```bash
uv run --directory serving python ../experiments/nope-mla/configure_tp2.py \
  --variant full --weights k2s1 \
  ../serving/var/nope-build-0961bbae/image-inspect.json \
  ../serving/var/nope-build-0961bbae/tp2-full-k2s1.toml
```

生成した `tp2-full-k2s1.toml` は、`load_configs` で読めます。生成の時点で確かめるのは、`--weights` の値と `derivation.name` の一致と、`derivation.origin` と `p1-nvfp4-tp2` の重みの一致だけです (§0)。合わなければ、出力は作られずに断られます。変換の結果のマニフェストと元の重みのマニフェストの実在、および元の重みのマニフェストの中身は、生成のあとに `load_configs` が確かめます。

派生の重みは、`kind = "probe"` (縮小の確認) の構成には結び付けられません。縮小の確認は、設定とトークナイザを `serve fetch --probe-files` で `probe/<名前>/` に取得して使いますが、派生の重みには取得元がなく、この道がないためです。`load_configs` が、その理由を示して断ります。

## 4. ⚠ 実機の重みを照合する (`serve verify`)

⚠ 了承を得てから実行する。`serve verify` は、Spark の上で `sha256sum` を流し、`state/k2s1.derived.verified.json` に照合の記録を置く。読み取りと記録の配布だけで、コンテナは起こさない。

`--configs` には、§3 で生成した `tp2-full-k2s1.toml` を渡します (派生の構成は、`configs.toml` にはコミットしていません)。照合の記録がなければ、§5 の `serve check` で関門 `weights_verified` が落ちます。記録には、いまコミットしているマニフェストのバイト列の SHA-256 (`manifest_sha256`) が入ります。

⚠ 了承を得てから実行する。次の `serve verify` が、実機の重みを照合し、記録を `state/` に置く。

```bash
uv run --directory serving serve verify p2-nope-tp2-full-k2s1 \
  --configs ../serving/var/nope-build-0961bbae/tp2-full-k2s1.toml \
  --yes
```

## 5. 関門をすべて通す (`serve check`)

`serve check` は、まず `load_configs` で構成を読み (マニフェストの実在と、元の重みのマニフェストの中身。§0)、そのあと 8 つの関門を流します。派生の重みでも、Hub の重みと同じ厳しさです。関門 `weights_verified` は、構成の `derivation` とマニフェストの `derivation` の一致と、照合の記録の一致 (記録の `manifest_sha256` が、いまのマニフェストと同じことを含む) を確かめます (§0)。

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

- 変換の道具が、CPU だけで現実的な時間で終わるか。2 台の結果が一致するか。**2026-09-26 に分かった**: 変換は 2 台とも約 8.5 分で終わり、結果はバイト単位で一致した (Issue #69 の記載。変換の対象の範囲 (`lm_head` を含むか) は、この repo の記録に無く、未確認)。
- 推論用のイメージの中で、道具が動くか (道具は `torch` と `numpy` に依存します。イメージに `numpy` があるかは未確認です)。
- Spark の `models/` の空き。`--link` を使えないので、元の重みとは別に、変換の結果のぶん (約 184 GiB) が要ります。
- 配った道具の写しが、Mac の写しと一致するか (§1 の `diff`)。道具の配り方は、`serving/payload/k2-quant/` に写して `serve push` で配ると**決めました**。**2026-09-26 に分かった**: 並べ替えのない `diff` は、`*.py` の並び順 (ロケール) が Spark と Mac で違い、中身が同じでも差分が出た。並べ替えれば一致したので、§1 の `diff` は、両側とも `LC_ALL=C sort` で並べ替える形にした。
- 派生の重みを読み込んだときの、vLLM の起動の時間と GPU のメモリ。FP8 にした層が、期待した層だけか。
- 匿名のまま Hub に触れずに、`serve check` と `serve verify` が通るか。
- **2026-09-26 に分かった**: `lm_head` を含めた変換の結果 (旧 `k2s1`) は、vLLM 0961bbae が FP8 (W8A16、compressed-tensors) の `ParallelLMHead` を humming の線形カーネルで読めず、`AttributeError: 'ParallelLMHead' object has no attribute 'output_partition_sizes'` で起動できなかった (#68)。MTP の `shared_head.head` も、`SharedHead` の中の同じ `ParallelLMHead` である。そのため、`lm_head` と `shared_head.head` を、道具の既定の対象から外した (Issue #69)。例外を出したカーネルのファイルと行は、この repo の記録に無く、未確認。
- **2026-09-26 に分かった**: 変換と照合の直後は、ページキャッシュが埋まっていて、`--load-format instanttensor` での起動が `buffer_size ... exceeds device memory budget` で落ちた。`--load-format auto` なら読めた (`configure_tp2.py` の `--load-format auto`。[全コンテキストの手順書](full-context-procedure.md) を参照。§3 の `--weights k2s1` と組み合わせて生成できるかは未確認)。
