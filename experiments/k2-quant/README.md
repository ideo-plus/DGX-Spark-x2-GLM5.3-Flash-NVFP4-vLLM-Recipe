# k2-quant: 共有の専門家・dense を FP8 (重みだけ、チャネルごと) にする変換の道具

K2 の第 1 段 (Issue #56)。vLLM を直さずに量子化できる部分を、FP8 E4M3 の重みと、出力チャネルごとの
対称スケールに変換する。この段の目的は、「変換 → vLLM で読み込み → Marlin FP8 で読む」の流れを確かめること。
見込みの短縮は、投機なしで約 6 ms (推定。Issue の記載)。

この道具は、CPU だけで動く。GPU もネットワークも使わない。実機での実行と、vLLM での読み込みは対話側が行う
(この README を書いた時点では、どちらも未確認)。

## 1. 対象

既定の対象は、次のモジュール (接頭辞は `model.language_model.`)。実機の `config.json` の `ignore`
の名前の形に合わせている。

| 対象 | 名前 | 個数 (Issue の記載から数えた値) |
|---|---|---|
| dense の MLP | `layers.{0,1,2}.mlp.{gate,up,down}_proj` | 9 |
| 共有の専門家 | `layers.N.mlp.shared_experts.{gate,up,down}_proj` (N は 3〜45 の 43 層) | 129 |

合計は 138 個。実機の index で、この数と合うかは未確認 (実行すると `converted modules: N` が出る)。
この範囲は、Issue の「第 1 段で FP8 にする `ignore` の中の名前」のうち、vLLM が FP8 の重みとして
読める形のもの (下の「既定に含めないもの」の `eh_proj`・`lm_head`・`shared_head.head` を除く) に
合わせた。

既定に含めないもの:

- **専門家 (`mlp.experts.*`)**: NVFP4 (層 3〜44) と FP8 (層 45、MTP) のままにする。変換できる形
  (BF16・F16・F32 の `weight` だけを持つモジュール) ではないので、専門家を拾う正規表現は、道具が
  終了 1 で止まる。
- **MTP の `eh_proj`**: vLLM (`vllm/models/glm5next/common/mtp.py:49`、commit 0961bbae) の
  `eh_proj` は `self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)`
  で、`quant_config` を受けない plain `nn.Linear` なので、FP8 の `eh_proj` を読み込めない。Issue は
  第 1 段の対象に挙げているが、この理由で既定から外した (`--pattern` に分岐を足せば変換自体はできる
  が、vLLM 側で読めない)。
- **`lm_head` と MTP の `shared_head.head`** (#68): vLLM 0961bbae は、FP8 (W8A16、
  compressed-tensors) の `ParallelLMHead` を humming の線形カーネルで読めず、
  `AttributeError: 'ParallelLMHead' object has no attribute 'output_partition_sizes'` で起動できない
  (2026-09-26 に、`lm_head` を含めて変換した `k2s1` を起動して確認したと、Issue #69 の背景にある。
  この repo の記録には無い)。`lm_head` は
  `ParallelLMHead` (`model.py:947-951`。`docs/research/2026-09-26-k2-quant-survey.md` §1 の表)、
  `shared_head.head` は `SharedHead` (`vllm/model_executor/models/deepseek_mtp.py`) の中の同じ
  `ParallelLMHead`。この調査は、ソースから `lm_head` を「量子化できる」と読んでいたが、実機では
  読めなかった。例外を出したカーネルのファイルと行は、この repo の記録に無く、未確認。Issue は第 1 段の
  対象に挙げているが、この理由で既定から外した (`--pattern` に分岐を足せば変換自体はできるが、
  vLLM 0961bbae では読めない。下の「`--pattern` の書き方」)。
- `docs/vllm-baseline/mtp-procedure.md` の判断 (ModelOpt の NVFP4 の重みで MTP を有効にすると、
  placeholder の `shared_head.head` が量子化されて落ちる不具合 (#55442) は、RedHatAI の重み
  (`compressed-tensors`、量子化の対象は experts だけ) には当たらない) は、既定の変換では前提が変わらない
  (`shared_head.head` を変換しない)。`--pattern` で足したときだけ、前提が変わる。
- KDA と MLA の射影: vLLM の側で BF16 に固定されている。この段では触らない。

### `--pattern` の書き方

- 対象は、テンソル名から `.weight` を除いた「モジュール名」に、`re.match` (先頭からの一致) で当てる。
  `.weight` で終わらないテンソル (`weight_scale`、`weight_packed` など) は候補にならない。
- この正規表現は、入力 checkpoint のテンソル名に当てて、変換するモジュールを選ぶだけで、`config.json` の
  target には書かない。既定の正規表現が `.*` で始まるのは、checkpoint の名前の接頭辞
  (`model.language_model.`) を受けるため。
- `config.json` の新しい group の `targets` は、**選ばれた (変換した) モジュールから作る**。各モジュールの名前から
  `model.language_model.` を除き (`lm_head` はそのまま)、名前の順に `|` でつないで、実行時の名前の末尾に当たる
  1 つの `re:(?:.*\.)?(?:<名前>|<名前>|…)$` にする。vLLM は `config.json` の `re:` 付き target を、実行時の層名に
  `re.match` で当てる。実行時の層名の接頭辞は、本体が `language_model.model.`、MTP が `model.`、`lm_head` が
  `language_model.lm_head` で、`(?:.*\.)?` が受ける。
- そのため、利用者の正規表現が実行時には広く当たる場合 (例: `.*gate_proj$` は、実行時には専門家の `gate_proj` にも
  当たる) でも、target は変換したモジュールだけに当たる。変換していないモジュール (index に無い MTP の head など)
  には当たらない。
- `model.language_model.` の下でも、最上位の名前でもないモジュール (例: `model.visual.…`) を選ぶと、
  実行時の名前の形が分からず、target を変換した集合に限れないので、何も書かずに終了 1 で止まる。
- 例: `lm_head` を含める (実験目的)。`python -m k2_quant --help` に出る既定の正規表現の末尾に、
  分岐 `|(?:.*\.)?lm_head$` を足して、`--pattern` に渡す。分岐は `|` でつながっているので、ほかの
  分岐は変えない。MTP の head は `|.*\.layers\.\d+\.shared_head\.head$` を足す。ただし、vLLM 0961bbae は
  FP8 (W8A16) の `ParallelLMHead` を読み込めない (上の「既定に含めないもの」。#68)。
- `eh_proj` を含めたいとき (実験目的) も、既定の正規表現に `|.*\.layers\.\d+\.eh_proj$` を足して
  `--pattern` に渡す。ただし、vLLM は `eh_proj` を `quant_config` を受けない plain `nn.Linear` として
  実装しているため、FP8 に変換しても読み込めない (上の「既定に含めないもの」)。

## 2. 方式

compressed-tensors (Apache-2.0) の公開の仕様・ソースを読んで、`float-quantized` の、チャネルごとの静的な
スケールの形に合わせた。他のレシピの台本は写していない。

対象の 2 次元の重み `W` (out, in。BF16、F16、F32 のどれか) を、次の 2 つにする。

| 名前 | dtype | 形 | 内容 |
|---|---|---|---|
| `<module>.weight` | `F8_E4M3` | `(out, in)` | `clamp(W / scale, -448, 448)` を FP8 E4M3 にしたもの |
| `<module>.weight_scale` | `F32` | `(out, 1)` | `amax(\|W\|, 出力チャネルごと) / 448` (0 の行は `finfo(float32).eps`) |

- 448 は FP8 E4M3 の最大値。逆量子化は `Q.to(float32) * scale` (compressed-tensors の `_dequantize` と同じ式)。
- 誤差の上限は、試験で固定している (`tests/test_fp8.py`。要素ごとの誤差と、相対 Frobenius 誤差 ≤ 0.05)。
- 入力の `W` は書き換えない。

`config.json` の `quantization_config` には、次の config group を 1 つ足す (名前は、未使用の最小の
`group_N`。既存の group を上書きしない)。

| 鍵 | 値 |
|---|---|
| `format` | `float-quantized` |
| `targets` | `["re:(?:.*\.)?(?:<変換したモジュールの、実行時の名前の末尾。名前の順に \| でつなぐ>)$"]` (`--pattern` の正規表現ではない) |
| `weights` | `num_bits=8`、`type=float`、`strategy=channel`、`symmetric=true`、`dynamic=false`、`observer=minmax`、ほかの鍵は `null`／空 |
| `input_activations` | `null` (重みだけを量子化する) |
| `output_activations` | `null` |

変換したモジュールの名前は、`ignore` から外す。既存の group と、それ以外の `ignore`、ほかの鍵は変えない。
`quant_method` が `compressed-tensors` でない、`format` が `mixed-precision` でない、対象が既存 group の
target に当たる、対象の名前が `model.language_model.` の下でも最上位の名前でもない、のいずれかなら、
何も書かずに終了 1 で止まる。

vLLM の側 (ソースで確かめたと Issue に書いてある。この README では再確認していない):

- 共有の専門家と dense は `Glm5NextMLP` で、`quant_config` を受ける。
- `ParallelLMHead` は、compressed-tensors の scheme があれば、線形層として量子化されると読んでいたが、
  実機では、FP8 (W8A16) の `ParallelLMHead` を humming の線形カーネルで読めなかった (#68。上の
  「既定に含めないもの」)。
- 重みだけの FP8 の scheme は `compressed_tensors_w8a16_fp8.py` (`CompressedTensorsW8A16Fp8`)。

## 3. 出力

入力のディレクトリと同じ構成の、新しいディレクトリを作る。`--output` は、存在しないパスか、存在する
**空の**ディレクトリ (`docker run --mount` の宛先は、コンテナの中に必ず存在する)。中身があれば、何も
書かずに終了 1 で止まる。

- 対象を含まない shard: 中身を変えずに写す (`--link` ならハードリンク)。
- 対象を含む shard: 対象のテンソルだけを差し替える。ほかのテンソルは dtype・形・バイト列を保ち、
  `__metadata__` も保つ。
- shard 以外の通常ファイル (tokenizer など。`config.json`、index、`*.safetensors` を除く): 同じ名前で写す。
  ディレクトリと、`.` で始まる項目は写さず、実行の終わりの標準出力に列挙する。
- `model.safetensors.index.json`: 作り直す。変換したモジュールの `<module>.weight_scale` を、`weight` と
  同じ shard に足し、`metadata.total_size` を再計算する。index に載っていない shard の中身は、index に足さない。
- `config.json`: 上の group を足したもの。
- `manifest.json`: 出力のすべてのファイル (manifest 自身を除く) の `path`・`size`・`sha256` (`path` 順) と、
  `total_bytes`、変換の条件 (`tool`、`tool_version`、元の repo と revision、`pattern`、`args`、変換した
  `modules`、`weight_dtype`、`scale_dtype`、`strategy`) を書く。

`conversion.args` は、道具に渡した引数のうち、**変換の結果を決めるもの**だけの列で、
`["--source-repo", <repo>, "--source-revision", <revision>, "--pattern", <実際に使った pattern>]` になる
(`--pattern` を渡さなくても、既定の値を書く)。`--source` と `--output` はマウントの位置に依存し、`--link` は
出力のバイト列を変えないので、書かない (書くと、2 台の manifest が一致しなくなる)。serving の
`serve derived-import` は、この列をそのまま派生のマニフェストの `derivation.conversion.args` にする。
`args` のない manifest (0.1.0 までの形) は、`serve derived-import` が断る。

同じ入力・同じ引数なら、出力はバイト単位で同じになる (時刻も絶対パスも書かない。試験で固定している)。
`generated_at` と、道具を含むコミット (`commit`) も書かない。この 2 つは、`serve derived-import` が Mac で足す。

入力に `manifest.json` があると、出力の manifest と取り違えるので、終了 1 で止まる。

## 4. Mac での使い方 (開発と検証)

以降のコマンドは、リポジトリの直下を作業ディレクトリにして実行する。`cd` は使わない。

```bash
uv sync --directory experiments/k2-quant
uv run --directory experiments/k2-quant python -m k2_quant --help
```

検証 (`ruff`、`mypy`、`pytest`):

```bash
uv run --directory experiments/k2-quant ruff check .
uv run --directory experiments/k2-quant ruff format --check .
uv run --directory experiments/k2-quant mypy
uv run --directory experiments/k2-quant pytest -q
```

依存は `torch` と `numpy` (`pyproject.toml`)。`numpy` は、テンソルの生のバイト列を書き出すために要る。
試験だけが `safetensors` (参照実装) を使い、道具の本体は `safetensors` を読み込まない (safetensors の形式は
公開仕様に従って自前で読み書きする)。

試験は、小さな合成の checkpoint (数個の shard、数個の線形層。`tests/synthetic.py`) だけを使う。
実機の重み、GPU、ネットワークは使わない。`tests/test_serving_chain.py` は、変換の結果を serving の取り込み
(`serve derived-import`)・構成の生成・照合・関門まで通す (偽の実行役を相手にする)。そのため、dev の依存に
`serving-kit` (`../../serving`。editable) を持つ。

Linux (CI) では、CUDA の依存 (`cuda-toolkit` 群) を引かない CPU 版の `torch` を使う (`pyproject.toml` の
`tool.uv.sources`。`download.pytorch.org/whl/cpu`)。macOS は PyPI の `torch` のままである。CI が
`experiments/k2-quant` を検査する (`docs/development/ci.md`)。

## 5. 実機での実行の形

**この節のコマンドは、対話側が実機で行う。** 実機の状態を変える (ファイルを作る) 操作なので、⚠ **了承を
得てから**行う。道具そのものは SSH・rsync・Docker を呼ばない。手順の全体 (道具の配布、2 台の変換、
Mac への取り込み、照合) は、`docs/vllm-baseline/k2-derived-weights-procedure.md` にある。この節は、
そのうちの変換のコマンドの意味を説明する。

道具は、推論用のイメージ (`serve` が使うイメージと同じもの) のコンテナの中で、CPU だけで動かす。
`<remote_root>` は、実機の作業ディレクトリ (`serving/config/nodes.toml` の `remote_root`)。

1. 道具を実機へ配る。道具の写し (`experiments/k2-quant/k2_quant/` と同じ 7 ファイル) は
   `serving/payload/k2-quant/k2_quant/` に置いてあり、`serve push` が実機の `<remote_root>/payload/` の
   下へ配る (実機に GitHub の認証情報は置かない)。写しが道具と同じバイト列であることは、
   `serving/tests/unit/test_payload.py` が固定している。道具を変えたら、写しも同じ変更で更新する。
2. 変換の結果の置き場所 `<remote_root>/models/<新しい名前>` を、**空で**作る (`docker run --mount` は、
   宛先がなければ失敗する)。推論サーバーを止めてから、コンテナを起動する。`--gpus` は付けない。
   `--network none` でネットワークを切る (この道具はネットワークを使わない)。

```bash
docker run --rm --network none \
  --entrypoint python3 \
  --mount type=bind,source=<remote_root>/payload/k2-quant,target=/tools/k2-quant,readonly \
  --mount type=bind,source=<remote_root>/models/glm-5-3-flash-nvfp4,target=/origin,readonly \
  --mount type=bind,source=<remote_root>/models/<新しい名前>,target=/derived \
  -w /tools/k2-quant \
  <推論イメージ> \
  -m k2_quant \
  --source /origin \
  --output /derived \
  --source-repo RedHatAI/GLM-5.3-Flash-NVFP4 \
  --source-revision 18d55bfd5a2194887738da73753975c9d3842f46
```

- 上のコマンドで、コンテナの中では `python3 -m k2_quant --source /origin --output /derived …` が動く
  (Mac の `uv run … python -m k2_quant` と同じ道具)。`-w /tools/k2-quant` にするので、`-m k2_quant` が
  写しの `k2_quant` パッケージを読む。
- `--source-repo` と `--source-revision` は、manifest の変換条件 (`conversion.source`、`conversion.args`)
  に写る (元の重みは `RedHatAI/GLM-5.3-Flash-NVFP4`、rev `18d55bfd5a2194887738da73753975c9d3842f46`。
  Issue の記載)。
- **`--link` は付けない。** 元の重み (`/origin`) と出力 (`/derived`) は別々の bind mount で、同じ
  ファイルシステムの上でも、マウントの境界をまたぐハードリンクは `EXDEV` で失敗し、**写さずに**
  終了 1 になる (黙って写す動きはしない)。`--link` を付けないので、対象を含まない shard も写す。
  入力と同じ大きさのディスクが要る。`--link` は、入力と出力が同じマウントに見える場所 (Mac など) で
  使える。
- 出力先が既にあって中身があると、何も書かずに終了 1 になる (上書きしない)。存在する空のディレクトリは
  受け付ける。途中で失敗したときは、出力のディレクトリが途中まで残るので、中身を消してからやり直す。
- 出力のファイルの所有者は、コンテナの中のユーザーになる。必要なら `--user` を付ける (未確認)。
- 実行の終わりに、変換したモジュール、書き直した shard、写した (リンクした) shard とファイル、写さなかった
  項目が、標準出力に出る。書き直した shard の大きさが、`--link` のときに増えるディスクの目安になる。
- 2 台それぞれで同じコマンドを流すと、同じ `manifest.json` (バイト単位で同じ) ができる。それを Mac に
  写して `serve derived-import` に渡す (手順書 §2)。

### メモリ・ディスク・所要時間の目安

- メモリ: 対象のテンソルを 1 つずつ変換する。`fp8.py` の中間テンソル (読み込んだバイト列 + float32 への
  変換 + 割り算 + clamp) の大きさは、そのとき変換している 1 つのテンソルで決まる。変換した FP8
  (1 バイト/要素) は、その shard を書き出すまでメモリに残る。推論サーバーを止めてから実行する。
  - 既定の対象と `lm_head` を、**テンソル 1 つずつの BF16 の大きさ** (TP で分ける前) で比べる
    (**表からの計算で、実測ではない**)。元にする値は、`docs/research/2026-09-26-k2-quant-survey.md` §1 の
    表の「1 台の BF16」(TP=2 の 1 台ぶん。列・行の並列の層は 2 で割ってある)。分ける前の大きさは、
    その値を 2 倍して、テンソルの数で割って出す。
    - dense の MLP: 0.45 GB × 2 ÷ 9 (層 0〜2 の 3 層 × 3 射影) ≈ 0.10 GB
    - 共有の専門家: 1.06 GB × 2 ÷ 126 (表の 42 層 × 3 射影) ≈ 0.017 GB
      (`moe_intermediate_size=2048`、`hidden_size=4096` から、2048 × 4096 × 2 バイト ≈ 0.017 GB でも合う)
    - `lm_head`: 0.63 GB × 2 ÷ 1 ≈ 1.26 GB (`vocab_size=154880`、`hidden_size=4096` から、
      154880 × 4096 × 2 バイト ≈ 1.27 GB でも合う)
  - 既定の対象は、どのテンソルも `lm_head` より小さい。そのため、中間テンソルの大きさで見た既定のピークは、
    `--pattern` に `lm_head` を足したときより小さい見込み。ただし、次は未確認: 表の共有の専門家は 42 層
    (層 3〜44) のぶんなので、MTP (層 45) の共有の専門家の形。dense の `intermediate_size` (上の値は表からの
    逆算で、この repo に config の値の記録は無い)。shard に積み上がる FP8 のぶんを含めた、ピークの実測。
  - `lm_head` を足したときのピークは、最大の対象である `lm_head` (BF16 で約 1.3 GB) のとき。中間テンソルは、
    BF16 の大きさの約 7 倍、約 9 GB (**コードからの見積もり。実測ではない**)。
- ディスク: `--link` なら、書き直す shard のぶん。書き直す shard は、実行の出力に出る。
- 所要時間: 未計測。

### 確かめ方 (変換のあと。対話側)

- 出力の `manifest.json` の `files` が、出力のファイルと一致すること (SHA-256 と大きさ)。
- vLLM で読み込むと、起動ログに、`CompressedTensorsW8A16Fp8` と、Marlin FP8 のカーネルが選ばれたことを示す
  行が出ること (行の正確な文言は未確認)。
- `serving/config/configs.toml` に、新しい重みを使う構成を足すのは、別の依頼。この道具は足さない。

## 6. 未確認事項

- `lm_head` と `shared_head.head` は、vLLM (0961bbae) が FP8 (W8A16) の `ParallelLMHead` を humming の
  線形カーネルで読めない (#68) ので、既定の対象から外した (この点は解決済み。上の「既定に含めないもの」)。
  例外を出したカーネルのファイルと行は、この repo の記録に無く、未確認。実機の
  `model.safetensors.index.json` に MTP の head (`shared_head.head`) の重みが別にあるかは、既定の変換の
  結果に影響しない (`--pattern` で足したときだけ効く。名前の形は未確認)。`eh_proj` も、vLLM のソースで
  `quant_config` を受けない plain `nn.Linear` と確認できたので、既定の対象から外した (この点も解決済み)。
- 推論イメージの中の Python、`torch`、`numpy` のバージョン (この道具は `torch==2.13.0`、`numpy==2.5.3`、
  Python 3.13 の Mac で検証した。`pyproject.toml` の `requires-python` は 3.12 以上で、`torch~=2.13`、
  `numpy~=2.5`。イメージ側は未確認)。
- 変換した重みを vLLM (0961bbae) が読み込めるか、Marlin FP8 で読まれるか、短縮が約 6 ms になるか
  (この段の本来の確認事項。ローカルの試験では確かめられない)。
- 実機の shard の名前・個数、対象の個数 (138 個)、所要時間、メモリのピーク。
