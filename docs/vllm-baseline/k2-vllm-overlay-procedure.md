# vLLM の直したファイルを重ねて起こす手順 (`--vllm-overlay k2s2a`)

この手順は、[ADR 0007](../decisions/0007-k2-stage2.md) の第 2a 段 (#73) の枠です。固定した vLLM (0961bbae) の Python のファイル 3 つを、lm_head・MLA の射影・KDA のまとめていない射影が FP8 (W8A16、チャネルごと、compressed-tensors) で読めるように直し、その写しを `serve push` で配って、**イメージを作り直さずに**、読み取り専用の bind mount (`readonly`) でコンテナの中の同じファイルに重ねます。

この文書に書かないこと: **送った内容**と応答の本文、**認証の情報**、`exl3-tp2` の中身。`exl3-tp2` は別の計測者の構成なので調べません。この文書は手順であって記録ではないので、実機で成功したとは書きません (実機で分かったことは、日付を付けて §8 に足します)。**⚠ の付いた操作は、計測者に了承を得てから行います。**

## 0. 何を重ねるか

直すファイルと、直す理由は次のとおりです。パッチは `experiments/k2-vllm-overlay/patches/k2s2a/` に、定義 (ファイルの並びと、ソース・写し・パッチの SHA-256) は `experiments/k2-vllm-overlay/k2s2a.json` にあります ([道具の説明](../../experiments/k2-vllm-overlay/README.md))。

| ファイル (イメージの中の道筋は `/usr/local/lib/python3.12/dist-packages/vllm/...`) | 直すこと |
|---|---|
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | `ParallelLMHead` に、humming の線形カーネルが読む `input_size_per_partition`・`output_partition_sizes`・`has_bias` を持たせる (#68。FP8 の `lm_head`) |
| `vllm/models/glm5next/common/model.py` | MLA に `quant_config` を渡す (`fused_qkv_a_proj`・`q_b_proj`・`o_proj`)。`packed_modules_mapping` に `fused_qkv_a_proj` → `q_a_proj`・`kv_a_proj_with_mqa` を足す。W8A16 の `weight_scale` を持つ射影を、BF16 への読み替え (`_try_load_fp8_attn_proj`) に流さない |
| `vllm/models/glm5next/common/kda.py` | `quant_config` を外す処理を消し、`o_proj`・`f_b_proj`・`g_b_proj` に渡す。まとめた層 `in_proj_qkvbfg_a` は、第 2b 段まで量子化しない |

`indexer` (`attention.py`) は直しません。MLA が `quant_config` を受けるので、`indexer` の `wq_b` と MLA の `kv_b_proj` も `quant_config` を受けますが、checkpoint の `quantization_config.ignore` に当たれば BF16 のままです。これを実機で確かめるのが §2 の段 0 と §7 です。

## 1. Mac で写しとパッチを確かめる

写し (`serving/payload/vllm-overlay/k2s2a/`) が「固定のソース + パッチ」と一致することを、試験で確かめます。固定のソース (`serving/var/nope-build-0961bbae/source`。Git 対象外) が手元にあれば、パッチを当てた結果とのバイト単位の一致まで見ます (無ければ SHA-256 の一致だけを見ます)。

```bash
git status --porcelain -- experiments/k2-vllm-overlay serving/payload/vllm-overlay
uv run --directory serving pytest -q tests/unit/test_vllm_overlay.py
```

写しを直したら、写しの側を編集してから、次でパッチと定義の SHA-256 を作り直します (固定のソースの木は書き換えません)。

```bash
uv run --directory serving python ../experiments/k2-vllm-overlay/overlay.py refresh k2s2a
uv run --directory serving python ../experiments/k2-vllm-overlay/overlay.py check k2s2a
```

## 2. 構成の生成

`--vllm-overlay k2s2a` を付けると、直したファイルごとに `--mount type=bind,source={remote_root}/payload/vllm-overlay/k2s2a/<道筋>,target=/usr/local/lib/python3.12/dist-packages/<道筋>,readonly` (根拠つき) が docker の節の末尾に入り、構成の名前の `--weights` の接尾辞の直後に `-ov-k2s2a` が付きます。付けないときの出力は変わりません。

**段 0 (第 1 段の重みのまま):** 変換の前に、第 1 段の重み `k2s1b` のまま重ねて起こし、`quant_config` を受けるようになった射影が、`ignore` で BF16 のまま起動できることを確かめます。基準は `p2-nope-tp2-full-k2s1b` です。重みの読み込み方は、第 1 段と同じ `--load-format auto` にします。

```bash
uv run --directory serving python ../experiments/nope-mla/configure_tp2.py \
  --variant full --weights k2s1b --vllm-overlay k2s2a --load-format auto \
  ../serving/var/nope-build-0961bbae/image-inspect.json \
  ../serving/var/nope-build-0961bbae/tp2-full-k2s1b-ov-k2s2a.toml
```

構成の名前は `p2-nope-tp2-full-k2s1b-ov-k2s2a` です。

**段 1 (第 2a 段の重み):** #74 の道具で第 2a 段の重みを作り、`serving/weights/k2s2a.manifest.json` を対話側がコミットしてから、`--weights k2s2a` で生成します。マニフェストが無い間は、生成器が出力を作らずに断ります。

```bash
uv run --directory serving python ../experiments/nope-mla/configure_tp2.py \
  --variant full --weights k2s2a --vllm-overlay k2s2a --load-format auto \
  ../serving/var/nope-build-0961bbae/image-inspect.json \
  ../serving/var/nope-build-0961bbae/tp2-full-k2s2a-ov-k2s2a.toml
```

構成の名前は `p2-nope-tp2-full-k2s2a-ov-k2s2a` (`tp2-full-k2s2a-ov-k2s2a.toml`) です。以下の §3〜§7 は段 0 の名前で書きます。段 1 では、構成の名前とファイル名を読み替えます。

生成器は、定義が無い、`name` や `vllm_commit` が違う、道筋が `vllm/` で始まる相対の道筋でない、写しが無いときは、出力を作らずに断ります。写しとパッチの一致は §1 の試験が見ます。

## 3. ⚠ 写しを配って突き合わせる (`serve push`)

⚠ 了承を得てから実行する。`serve push` は、2 台の `remote_root` の下に置き場所を作り、`serving/payload/` の中身 (重ねる写しを含む) を `payload/` に配る。

```bash
uv run --directory serving serve push --yes
```

⚠ 了承を得てから実行する。2 台の `ssh` で、配った写しの `sha256sum` を読み、Mac の写しと突き合わせる (読み取りだけ。何も出なければ一致)。並び順がロケールで違わないよう、両側とも `LC_ALL=C sort` でファイル名の順に並べ替えてから比べる。

```bash
diff \
  <(ssh spark-153d 'cd /home/j5ik2o/vllm-baseline/payload/vllm-overlay/k2s2a && sha256sum vllm/model_executor/layers/vocab_parallel_embedding.py vllm/models/glm5next/common/model.py vllm/models/glm5next/common/kda.py | LC_ALL=C sort -k 2') \
  <(cd serving/payload/vllm-overlay/k2s2a && shasum -a 256 vllm/model_executor/layers/vocab_parallel_embedding.py vllm/models/glm5next/common/model.py vllm/models/glm5next/common/kda.py | LC_ALL=C sort -k 2)
diff \
  <(ssh spark-5083 'cd /home/j5ik2o/vllm-baseline/payload/vllm-overlay/k2s2a && sha256sum vllm/model_executor/layers/vocab_parallel_embedding.py vllm/models/glm5next/common/model.py vllm/models/glm5next/common/kda.py | LC_ALL=C sort -k 2') \
  <(cd serving/payload/vllm-overlay/k2s2a && shasum -a 256 vllm/model_executor/layers/vocab_parallel_embedding.py vllm/models/glm5next/common/model.py vllm/models/glm5next/common/kda.py | LC_ALL=C sort -k 2)
```

## 4. ⚠ 重ねる先の道筋を、イメージの中で読み取って確かめる

重ねる先 (`/usr/local/lib/python3.12/dist-packages/vllm`) は、固定の commit の `docker/Dockerfile` (`ARG PYTHON_VERSION=3.12` と `uv pip install --system dist/*.whl`) から決めた道筋です。イメージの中の実物と一致することを、重ねない素のコンテナで読み取って確かめます。GPU は渡しません (`--gpus` も `--runtime nvidia` も付けません)。ネットワークも切ります (`--network none`)。`vllm.__file__` と同じ値 (パッケージの `__init__.py` の道筋) を、vLLM の初期化を走らせずに `importlib.util.find_spec` で読みます。

⚠ 了承を得てから実行する。2 台の `ssh` で、推論用のイメージ `sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90` の使い捨てのコンテナ (`--rm`) を起こし、vLLM の道筋と 3 つのファイルの実在を読む (読み取りだけ)。

```bash
for node in spark-153d spark-5083; do
  ssh "$node" 'docker run --rm --network none --entrypoint python3 sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90 -c "import importlib.util; print(importlib.util.find_spec(\"vllm\").origin)"'
  ssh "$node" 'docker run --rm --network none --entrypoint ls sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90 -l /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/vocab_parallel_embedding.py /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/common/model.py /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/common/kda.py'
done
```

1 行目が `/usr/local/lib/python3.12/dist-packages/vllm/__init__.py` で、3 つのファイルが `ls` で見えれば、重ねる先は正しい道筋です。違えば、§5 に進まず、生成器の `IMAGE_SITE_PACKAGES` を直します。

## 5. ⚠ 関門をすべて通す (`serve check`)

`serve check` は、まず `load_configs` で構成を読み (`--mount` の元が `{remote_root}` の下で、`type=bind` であること)、そのあと関門を流します。関門 `layout` は、`--mount` の元を `test -e` で見ます。ディレクトリだけでなく、重ねるファイルも見るので、§3 の `serve push` をしていなければ、ここで落ちます。派生の重みの照合の記録が無ければ、関門 `weights_verified` が落ちます (照合は [派生の重みの手順書](k2-derived-weights-procedure.md) の §4 のとおり、`serve verify` で行います)。

⚠ 了承を得てから実行する。次の `serve verify` は、照合の記録が無いときだけ流す。実機の重みを照合し、記録を `state/` に置く (コンテナは起こさない)。

```bash
uv run --directory serving serve verify p2-nope-tp2-full-k2s1b-ov-k2s2a \
  --configs ../serving/var/nope-build-0961bbae/tp2-full-k2s1b-ov-k2s2a.toml \
  --yes
```

⚠ 了承を得てから実行する。`serve check` は、2 台を `ssh` で読み取るだけで、状態を変えない。

```bash
uv run --directory serving serve check p2-nope-tp2-full-k2s1b-ov-k2s2a \
  --configs ../serving/var/nope-build-0961bbae/tp2-full-k2s1b-ov-k2s2a.toml
```

## 6. ⚠ 起動、短い確認、回収、停止

⚠ 了承を得てから実行する。`serve start` は、2 台でコンテナを起こす。`serve smoke` は、短い要求を 1 つずつ送る (応答の本文は画面に出すだけで、どこにも残さない)。

```bash
uv run --directory serving serve start p2-nope-tp2-full-k2s1b-ov-k2s2a \
  --configs ../serving/var/nope-build-0961bbae/tp2-full-k2s1b-ov-k2s2a.toml \
  --yes

uv run --directory serving serve smoke p2-nope-tp2-full-k2s1b-ov-k2s2a \
  --configs ../serving/var/nope-build-0961bbae/tp2-full-k2s1b-ov-k2s2a.toml
```

`serve stop` は記録を回収しないので、必ず `serve logs` を先に打ちます。回収先は `serving/var/<UTC>-logs-p2-nope-tp2-full-k2s1b-ov-k2s2a/` で、コンテナの標準出力と標準エラーの記録 (`container.stdout.log`、`container.stderr.log`) が台ごとに入ります。§7 はこの記録を読みます。

⚠ 了承を得てから実行する。`serve logs` で回収してから、`serve stop` で 2 台の自分のコンテナを止めて消す。

```bash
uv run --directory serving serve logs p2-nope-tp2-full-k2s1b-ov-k2s2a \
  --configs ../serving/var/nope-build-0961bbae/tp2-full-k2s1b-ov-k2s2a.toml
uv run --directory serving serve stop --yes
```

## 7. 起動のログで、FP8 の scheme が期待の層だけに当たったことを確かめる

1. **起動が完了すること。** scheme を取り違えた層 (たとえば、`ignore` に当たらずに NVFP4 の group に当たった射影) は、重みの形や scale が合わず、読み込みで落ちます。
2. **カーネル。** W8A16 FP8 の線形カーネルは、CUDA では `HummingFP8ScaledMMLinearKernel` が最優先で、使えなければ `MarlinFP8ScaledMMLinearKernel` です。固定の vLLM では、W8A16 の経路はカーネルを選ぶときに層の名前を渡さないので、`Selected <カーネル> for <層>` の行は出ません (この行が出るのは、層の名前を渡す別の量子化の経路です)。カーネルの取り違えや、humming が `lm_head` の属性を読めない失敗は、1. の起動の失敗として現れます。
3. **KV cache。** `GPU KV cache size` の行を、基準の `k2s1b` (`p2-nope-tp2-full-k2s1b`、1,736,006 トークン。[第 1 段の記録](../results/2026-09-26-k2-stage1.md)) と比べます。段 0 は重みが同じなので、ほぼ同じはずです。段 1 は、重みが小さくなったぶん増えるはずです。
4. **層ごとの scheme (決定的な確認)。** `Using scheme: CompressedTensorsW8A16Fp8 for <層>` は DEBUG の段でしか出ません (`VLLM_LOGGING_LEVEL`)。生成した TOML の写しの末尾に、`serving/config/configs.toml` の `probe-nightly` の `env.logging-level` の表 (根拠つき) を写した確認用の構成を作り、**1 回だけ**使います。名前は同じで、`config-sha256` のラベルで区別されます。

```bash
cp serving/var/nope-build-0961bbae/tp2-full-k2s1b-ov-k2s2a.toml \
   serving/var/nope-build-0961bbae/tp2-full-k2s1b-ov-k2s2a-debug.toml
cat >> serving/var/nope-build-0961bbae/tp2-full-k2s1b-ov-k2s2a-debug.toml <<'EOF'

[configs.p2-nope-tp2-full-k2s1b-ov-k2s2a.env.logging-level]
flag = "VLLM_LOGGING_LEVEL"
value = "DEBUG"
why = "層ごとの量子化の scheme (Using scheme: … for …) が、debug の段でしか記録に出ない (ADR 0007 の第 2a 段の確認)"
source = "https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/envs.py"
quote = "this is used for configuring the default logging level"
EOF
```

このブロックと次の `grep` は、Mac のリポジトリの根で打ちます。この構成で §5〜§6 をもう一度流し (`--configs` には `../serving/var/nope-build-0961bbae/tp2-full-k2s1b-ov-k2s2a-debug.toml` を渡します)、回収した記録から、W8A16 FP8 の scheme が当たった層を抜き出します。

```bash
grep -rhoE 'Using scheme: [A-Za-z0-9]* for [^ ]*' \
  serving/var/<UTC>-logs-p2-nope-tp2-full-k2s1b-ov-k2s2a/ \
  | grep 'CompressedTensorsW8A16Fp8' | LC_ALL=C sort | uniq -c
```

**段 0 (`k2s1b` + 重ね) で期待すること:** W8A16 FP8 の scheme が当たるのは、第 1 段で変換した dense と `shared_experts` の射影だけです。MLA・KDA の射影と `lm_head` は、`k2s1b` では変換していないので、`ignore` で BF16 のまま (scheme が当たらない) です。

**段 1 (`k2s2a` + 重ね) で期待すること:** 第 1 段の層に加えて、次に W8A16 FP8 の scheme が当たります。

- `lm_head` (`language_model.lm_head`)
- MLA 11 層の `fused_qkv_a_proj`・`q_b_proj`・`o_proj`
- KDA 34 層の `o_proj`・`f_b_proj`・`g_b_proj`

**どちらの段でも、当たってはいけない層:** `in_proj_qkvbfg_a` (KDA のまとめた層。第 2b 段)、`indexer` (`wq_b`、`wk_weights_proj`)、MLA の `kv_b_proj`。これらが W8A16 FP8 の一覧に出たら、または別の scheme (NVFP4 など) の一覧に出たら、checkpoint の `ignore` と `targets` の当たり方が想定と違うので、計測に進まずに記録します。

## 8. 実機でしか分からないこと

1. checkpoint の `quantization_config.ignore` の中身はリポジトリに無い。`quant_config` を受けるようになった `kv_b_proj`、`indexer` の `wq_b`、変換しない層の MLA・KDA の射影が `ignore` に当たらないと、compressed-tensors がクラス名で `targets` に当てて NVFP4 の group に当たり、起動が落ちうる。段 0 で先に確かめる。
2. MTP (`Glm5NextMTP`) は `packed_modules_mapping` を持たない。重ねた構成と MTP の組み合わせは、投機なしが通ってから別に確かめる。
3. イメージの中の `vllm.__file__` (§4 で確かめる)。
4. humming の線形カーネルが実際に選ばれるか、FP8 の `lm_head` が humming で正しく動くか、速さ (`k2s1b` 比) と品質。
5. NoPE の `kv_a_proj_with_mqa` の詰め物 (`load_weights` が、rope の部分を 0 で埋めて足す処理) が、FP8 の重みと、チャネルごとの `weight_scale` のまま正しく通るか (段 1 で、MLA を FP8 にしたときに初めて通る経路)。
