# k2-quant: 共有の専門家・dense・lm_head を FP8 (重みだけ、チャネルごと) にする変換の道具

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
| `lm_head` | `lm_head` | 1 |
| MTP の head | `layers.\d+.shared_head.head` (別の重みとしてあるときだけ。名前は未確認) | 0 か 1 |

合計は 139 個 (MTP の head が別の重みとしてあれば 140 個)。実機の index で、この数と合うかは未確認
(実行すると `converted modules: N` が出る)。この範囲は、Issue の「第 1 段で FP8 にする `ignore` の中の名前」
のうち、vLLM が FP8 の重みとして読める形のもの (下の「既定に含めないもの」の `eh_proj` を除く) に合わせた。
MTP の head は、Issue が「別の重みとしてあるなら、それも対象にする。名前は、index から確かめる」としている
ので、その名前の形に当たれば選ぶ (index に無ければ、何も選ばれず、何も起きない)。

既定に含めないもの:

- **専門家 (`mlp.experts.*`)**: NVFP4 (層 3〜44) と FP8 (層 45、MTP) のままにする。変換できる形
  (BF16・F16・F32 の `weight` だけを持つモジュール) ではないので、専門家を拾う正規表現は、道具が
  終了 1 で止まる。
- **MTP の `eh_proj`**: vLLM (`vllm/models/glm5next/common/mtp.py:49`、commit 0961bbae) の
  `eh_proj` は `self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)`
  で、`quant_config` を受けない plain `nn.Linear` なので、FP8 の `eh_proj` を読み込めない。Issue は
  第 1 段の対象に挙げているが、この理由で既定から外した (`--pattern` に分岐を足せば変換自体はできる
  が、vLLM 側で読めない)。
- KDA と MLA の射影: vLLM の側で BF16 に固定されている。この段では触らない。

### vLLM 側の扱いが未確認のもの (`shared_head.head`)

`shared_head.head` は、Issue が第 1 段の対象に挙げている。`SharedHead`
(`vllm/model_executor/models/deepseek_mtp.py`) の中の `ParallelLMHead` で、`quant_config` を
受けるので、FP8 の重みとして読める見込みだが、実機での読み込みはこの repo の記録では確かめられていない。
`eh_proj` は、上の「既定に含めないもの」のとおり、vLLM (`vllm/models/glm5next/common/mtp.py:49`、
commit 0961bbae) のソースで `quant_config` を受けない plain `nn.Linear` と確認できたので、既定から
外し、この道具では扱わない。

- `docs/vllm-baseline/mtp-procedure.md` には、ModelOpt の NVFP4 の重みで MTP を有効にすると、placeholder の
  `shared_head.head` が量子化されて落ちる不具合 (#55442) があり、RedHatAI の重み (`compressed-tensors`、
  量子化の対象は experts だけ) には当たらない、という判断がある。これはソース上の判断で、実機では未確認。
  この道具は `shared_head.head` を変換対象に加えるので、この判断の前提が変わる。
- 読み込めなければ、`--pattern` で外して作り直す (下の「`--pattern` の書き方」)。この段の目的は、
  「変換 → 読み込み → Marlin FP8 で読む」の流れを確かめることなので、これは実機で最初に確かめる項目になる。

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
- 例: `shared_head.head` を外す。`python -m k2_quant --help` に出る既定の正規表現から、該当の分岐
  (`|.*\.layers\.\d+\.shared_head\.head$`) だけを消して、`--pattern` に渡す。分岐は `|` で
  つながっているので、ほかの分岐は変えない。
- 逆に `eh_proj` を含めたいとき (実験目的) は、既定の正規表現に `|.*\.layers\.\d+\.eh_proj$` を足して
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
- `ParallelLMHead` は、compressed-tensors の scheme があれば、線形層として量子化される。
- 重みだけの FP8 の scheme は `compressed_tensors_w8a16_fp8.py` (`CompressedTensorsW8A16Fp8`)。

## 3. 出力

入力のディレクトリと同じ構成の、新しいディレクトリを作る。

- 対象を含まない shard: 中身を変えずに写す (`--link` ならハードリンク)。
- 対象を含む shard: 対象のテンソルだけを差し替える。ほかのテンソルは dtype・形・バイト列を保ち、
  `__metadata__` も保つ。
- shard 以外の通常ファイル (tokenizer など。`config.json`、index、`*.safetensors` を除く): 同じ名前で写す。
  ディレクトリと、`.` で始まる項目は写さず、実行の終わりの標準出力に列挙する。
- `model.safetensors.index.json`: 作り直す。変換したモジュールの `<module>.weight_scale` を、`weight` と
  同じ shard に足し、`metadata.total_size` を再計算する。index に載っていない shard の中身は、index に足さない。
- `config.json`: 上の group を足したもの。
- `manifest.json`: 出力のすべてのファイル (manifest 自身を除く) の `path`・`size`・`sha256` (`path` 順) と、
  `total_bytes`、変換の条件 (`tool`、`tool_version`、元の repo と revision、`pattern`、変換した
  `modules`、`weight_dtype`、`scale_dtype`、`strategy`) を書く。

同じ入力・同じ引数なら、出力はバイト単位で同じになる (時刻も絶対パスも書かない。試験で固定している)。

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
実機の重み、GPU、ネットワークは使わない。

## 5. 実機での実行の形

**この節のコマンドは、対話側が実機で行う。** 実機の状態を変える (ファイルを作る) 操作なので、⚠ **了承を
得てから**行う。道具そのものは SSH・rsync・Docker を呼ばない。

想定は、推論用のイメージ (`serve` が使うイメージと同じもの) のコンテナの中で、CPU だけで動かす形。
`<remote_root>` は、実機の作業ディレクトリ (`serving/config/nodes.toml` の `remote_root`)。
下の重みのパスと出力の名前は例なので、実機の配置に合わせる。

1. 道具を実機へ配る (rsync。`experiments/k2-quant/` の下だけ)。実機に GitHub の認証情報は置かない。
2. 推論サーバーを止めてから、コンテナを起動する。`--gpus` は付けない。

```bash
docker run --rm \
  --entrypoint python3 \
  --mount type=bind,source=<remote_root>/models,target=/models \
  --mount type=bind,source=<remote_root>/k2-quant,target=/work,readonly \
  -w /work \
  <推論イメージ> \
  -m k2_quant \
  --source /models/glm-5-3-flash-nvfp4 \
  --output /models/<新しい名前> \
  --source-repo RedHatAI/GLM-5.3-Flash-NVFP4 \
  --source-revision 18d55bfd5a2194887738da73753975c9d3842f46 \
  --link
```

- 上のコマンドで、コンテナの中では `python3 -m k2_quant --source … --output … --source-repo …
  --source-revision … --link` が動く (Mac の `uv run … python -m k2_quant` と同じ道具)。
- `--source-repo` と `--source-revision` は、manifest の変換条件に写る (元の重みは
  `RedHatAI/GLM-5.3-Flash-NVFP4`、rev `18d55bfd5a2194887738da73753975c9d3842f46`。Issue の記載)。
- `--link` は、入力と出力が同じファイルシステムにあるときだけ使える。別のファイルシステムだと
  ハードリンクに失敗し、**写さずに** 終了 1 になる (黙って写す動きはしない)。`--link` を付けなければ、
  対象を含まない shard も写すので、入力と同じ大きさのディスクが要る。
- 出力先が既にあると、何も書かずに終了 1 になる (上書きしない)。途中で失敗したときは、出力の
  ディレクトリが途中まで残るので、消してからやり直す。
- 出力のファイルの所有者は、コンテナの中のユーザーになる。必要なら `--user` を付ける (未確認)。
- 実行の終わりに、変換したモジュール、書き直した shard、写した (リンクした) shard とファイル、写さなかった
  項目が、標準出力に出る。書き直した shard の大きさが、`--link` のときに増えるディスクの目安になる。

### メモリ・ディスク・所要時間の目安

- メモリ: 対象のテンソルを 1 つずつ変換する。ピークは、最大の対象である `lm_head`
  (`vocab_size=154880`、`hidden_size=4096`。BF16 で約 1.3 GB) のとき。`fp8.py` の中間テンソル (読み込んだ
  バイト列 + float32 への変換 + 割り算 + clamp) から見積もると、BF16 の大きさの約 7 倍、約 9 GB
  (**コードからの見積もり。実測ではない**)。変換した FP8 (1 バイト/要素) は、その shard を書き出すまで
  メモリに残る。推論サーバーを止めてから実行する。
- ディスク: `--link` なら、書き直す shard のぶん。書き直す shard は、実行の出力に出る。
- 所要時間: 未計測。

### 確かめ方 (変換のあと。対話側)

- 出力の `manifest.json` の `files` が、出力のファイルと一致すること (SHA-256 と大きさ)。
- vLLM で読み込むと、起動ログに、`CompressedTensorsW8A16Fp8` と、Marlin FP8 のカーネルが選ばれたことを示す
  行が出ること (行の正確な文言は未確認)。
- `serving/config/configs.toml` に、新しい重みを使う構成を足すのは、別の依頼。この道具は足さない。

## 6. 未確認事項

- 実機の `model.safetensors.index.json` に、MTP の head (`shared_head.head`) の重みが別にあるか。名前の形
  (既定の正規表現は `layers.\d+.shared_head.head` に当たれば選ぶ) も、実機の index で確かめる。
- `shared_head.head` を、vLLM (0961bbae) が FP8 の重みとして読めるか (上の「vLLM 側の扱いが未確認の
  もの」)。読めなければ、`--pattern` で外す。`eh_proj` は、vLLM のソースで `quant_config` を受けない
  plain `nn.Linear` と確認できたので、既定の対象から外した (この点は解決済み)。
- 推論イメージの中の Python、`torch`、`numpy` のバージョン (この道具は `torch==2.13.0`、`numpy==2.5.3`、
  Python 3.13 の Mac で検証した。`pyproject.toml` の `requires-python` は 3.12 以上で、`torch~=2.13`、
  `numpy~=2.5`。イメージ側は未確認)。
- 変換した重みを vLLM (0961bbae) が読み込めるか、Marlin FP8 で読まれるか、短縮が約 6 ms になるか
  (この段の本来の確認事項。ローカルの試験では確かめられない)。
- 実機の shard の名前・個数、対象の個数 (139 個。MTP の head があれば 140 個)、所要時間、メモリのピーク。
