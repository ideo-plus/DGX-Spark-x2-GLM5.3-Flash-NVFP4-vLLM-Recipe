# `payload/` の中の、上流の原文の取り直しの手順

`serving/payload/vllm_sanity_check.py` は、vLLM の公式のトラブルシュートの文書
(`docs/usage/troubleshooting.md`) の「Incorrect hardware/driver」の節にある、Python の
コードブロックを、1 文字も変えずに置いたものである (`serving/payload/allreduce_bench.py`
は自前のコードなので、この手順の対象ではない)。

固定したイメージを替えたとき (design.md 「Revalidation Triggers」の「イメージのダイジェスト
を替えたとき」) は、この手順で取り直す。**要約する道具 (WebFetch など) では取らない**
(原文が変わってしまうため)。**nccl-tests や、第三者のレシピ・ブログ・フォーラムは開かない**
(要件 11.3)。

## 1. 新しいイメージの build commit (40 桁) を確かめる

固定するイメージ (`vllm/vllm-openai@sha256:...`) の OCI ラベルから読む
(research.md §a-2、design.md の Technology Stack と同じ方法)。

```bash
# レジストリから取得せずに読む場合は、Docker Hub のレジストリ API か
# `docker buildx imagetools inspect` を使う (research.md §a-5)。
# 手元に取得済みのイメージがあれば:
docker image inspect <image>@sha256:<digest> \
    --format '{{index .Config.Labels "ai.vllm.build.commit"}}'
```

40 桁の commit が読めない場合だけ、GitHub の API で確かめる (公式の API。第三者のレシピでは
ない)。

```bash
curl -fsSL https://api.github.com/repos/vllm-project/vllm/commits/<短い参照や branch> \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["sha"])'
```

## 2. 生の Markdown を取得する

```bash
curl -fsSL \
    "https://raw.githubusercontent.com/vllm-project/vllm/<40 桁の commit>/docs/usage/troubleshooting.md" \
    -o troubleshooting.md
```

その commit に、このファイル自体がない場合、または「Incorrect hardware/driver」の節
(もしくは同等の 4 段の確認のコードブロック) がない場合は、推測で埋めずに、試したことを
書いて計測者に判断を仰ぐ (guess で埋めない)。

## 3. コードブロックを機械的に抜き出す

「## Incorrect hardware/driver」の節の中の、最初の &#96;&#96;&#96;python … &#96;&#96;&#96;
のコードブロック (mkdocs の `??? code` という折りたたみの中にあり、1 段ぶん 4 個の空白で
インデントされている) を、正規表現で抜き出し、共通の先頭の 4 個の空白を取り除く
(dedent)。**手で打ち直さない**。

```python
import re
from pathlib import Path

text = Path("troubleshooting.md").read_text(encoding="utf-8")
after = text[text.index("## Incorrect hardware/driver") :]
m = re.search(r"```python\n(.*?)\n( *)```\n", after, re.DOTALL)
raw_block = m.group(1)
lines = raw_block.split("\n")
indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
common = min(indents) if indents else 0
body = "\n".join(line[common:] if len(line) >= common else line for line in lines) + "\n"
```

## 4. sha256 を計算し、確認のスクリプトを書き直す

```python
import hashlib

print(hashlib.sha256(body.encode("utf-8")).hexdigest())
```

`serving/payload/vllm_sanity_check.py` の先頭のコメントのうち、**出典の URL、commit の 40 桁、
取得した日、sha256 の部分だけ**を、新しい値に書き換え、区切りの行より下を、上で抜き出した
`body` にそっくり置き換える。手で打ち直さない (抜き出した文字列を、そのままファイルに書く)。

**先頭の 2 行 (`# ruff: noqa`、`# fmt: off`) と、その下の説明のコメントは、書き換えずに、
そのまま残す**。この 2 行が、上流の原文を、静的な検査と整形から守る、本当の歯止めである
(`# fmt: on` を、どこにも足さない。区切りの行より下には、何も足さない)。先頭のコメントを
作り直す場合も、この 2 行を、1 行目と 2 行目に置く。

## 5. 試験と、`LICENSES.md` の確かめ

- `uv run pytest serving/tests/unit/test_payload.py` を流し、`ast.parse` が通ること、
  区切りの行より下の sha256 がコメントの値と一致することを確かめる
- 同じ試験が、上流の原文を守る歯止めも確かめる: ファイルの先頭の `# ruff: noqa` / `# fmt: off`
  (ruff を、どこから、どの設定で、名指しで呼んでも効く、本当の歯止め) が残っていること、
  `pyproject.toml` の `[tool.ruff]` の `extend-exclude` / `force-exclude` (ディレクトリを歩くときの
  二重の歯止め。**ruff を `serving/` の外から呼ぶと、こちらは効かない**) の値、実際に ruff を
  3 つの作業ディレクトリから名指しで呼んで、整形も検査もされないこと
- `uv run ruff format --check .` と `uv run ruff check .` が通ることを確かめる
- `LICENSES.md` の `serving-kit が配るスクリプト` の行の commit と取得した日を、新しい値に
  直す (ライセンスは vLLM が Apache-2.0 のままである限り変わらない)

## `chat-template/glm53-flash-thinking-toggle.jinja` の取り直しの手順 (issue #131)

`serving/payload/chat-template/glm53-flash-thinking-toggle.jinja` は、重みに付いている公式の
チャットテンプレートを元に、`enable_thinking`/`thinking` を読む分岐だけを足したものである
(生成の書き出しの塊だけが違い、それ以外は 1 バイトも変えていない)。

1. 次の URL から、**要約する道具 (WebFetch など) を使わずに** `curl -L` で取得する。

   ```bash
   curl -L -o chat_template.jinja \
       "https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4/resolve/18d55bfd5a2194887738da73753975c9d3842f46/chat_template.jinja"
   shasum -a 256 chat_template.jinja
   ```

2. SHA-256 が `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5`、大きさが
   10,950 バイトであることを、コミット済みのマニフェスト
   (`serving/weights/RedHatAI__GLM-5.3-Flash-NVFP4.manifest.json`、
   `serving/weights/k2s4.manifest.json` の `chat_template.jinja` の項目) と照合する。ハッシュか
   大きさが合わなければ、推測で書かずに止めて報告する。
3. 取得した内容を、1 バイトも変えずに `serving/tests/fixtures/chat-template/glm53-flash.official.jinja`
   に置き、生成の書き出しの塊 (`{%- if add_generation_prompt -%} <|assistant|>{{- '<think>' -}}
   {%- endif -%}`) だけを、`enable_thinking`/`thinking` を読む分岐に置き換えて
   `serving/payload/chat-template/glm53-flash-thinking-toggle.jinja` を作り直す。材料は、重みに
   付いている公式のテンプレートと `RedHatAI/GLM-5.3-Flash-NVFP4` のモデルカードだけであり、他の
   レシピ (MiaAI-Lab、tonyd2wild、mmastrac など) のテンプレート・スクリプト・設定は開かない
   (クリーンルームの決めごと)。
4. `uv run --directory serving pytest -q tests/unit/test_chat_template.py` を流し、写しの
   SHA-256 の固定と、新しいテンプレートが生成の書き出しの塊だけ違うことを確かめる。
5. `LICENSES.md` の「serving-kit が配る、チャットテンプレートの写し」の行の、リビジョンと
   元の SHA-256 を新しい値に直す (ライセンスは、重みの MIT のままである限り変わらない)。

## 固定した vLLM (`0961bbae`) の確認 (issue #131)

`--chat-template` の受け付け方、`/v1/messages`・`/v1/chat/completions` の両方の経路で
`chat_template_kwargs` がテンプレートとパーサーに届くこと、`glm47` のパーサーが思考オフの
出力を本文として扱うことを、固定した vLLM のソース木
(`serving/var/nope-build-0961bbae/source/`。Git 対象外) で確かめた。下の表の file:line は、
すべてこの木を実装の直前に読み直して確かめたものである (委譲の再読を含む)。

| 事実 | file:line |
|---|---|
| `--chat-template` はファイルパスか 1 行のテンプレートを受ける | `vllm/entrypoints/launchers/cli_args.py:80-82` |
| ファイルを読んで `resolved_chat_template` を作る (`_load_chat_template`) | `vllm/entrypoints/chat_utils.py:1490-1546` |
| `resolved_chat_template` が、`/v1/chat/completions` (`OpenAIServingChat`) と `/v1/messages` (`AnthropicServingMessages`) の両方の `chat_template=` に渡る | `vllm/entrypoints/generate/api_router.py:98,133,178` |
| 与えたテンプレートが、重みに付いたものより優先される (1st priority) | `vllm/renderers/hf.py:270-274` |
| `/v1/messages` (`AnthropicMessagesRequest`) は `chat_template_kwargs` を `ChatCompletionRequest` にそのまま渡す | `vllm/entrypoints/anthropic/protocol.py:171`、`vllm/entrypoints/anthropic/serving.py:494` |
| `output_config.effort` は `reasoning_effort` に写る | `vllm/entrypoints/anthropic/serving.py:515-516` |
| vLLM が `enable_thinking` を足すのは、`reasoning_effort` があり、要求が `enable_thinking` を持たないときだけ | `vllm/entrypoints/openai/chat_completion/protocol.py:584-586` |
| マージで `False` は捨てられない (`unset_values` が捨てるのは `None` と `"auto"` だけ) | `vllm/renderers/params.py:36-48` |
| 描画に届くのは、`fn_kw`・テンプレートが参照する変数名 (`find_undeclared_variables`)・HF の標準パラメータの和集合だけ。テンプレートが参照しない変数名は黙って落ちる | `vllm/renderers/hf.py:674-682,709-737` |
| パーサーには、この絞り込みより前の (絞り込まれていない) `chat_template_kwargs` が渡る | `vllm/entrypoints/openai/chat_completion/serving.py:191-201,267-275` |
| `glm47` は `Glm47MoeParserReasoningAdapter` (`glm47_moe_reasoning_parser`) に対応する | `vllm/reasoning/__init__.py:63-65` |
| パーサー (`Glm47MoeParser`) は `chat_template_kwargs.thinking`/`enable_thinking` を `bool()` で読む。`thinking_mode` は読まない | `vllm/parser/glm47_moe.py:192-195` |
| 両方とも無指定なら思考あり、どちらかが指定されていれば `bool(thinking) or bool(enable_thinking)` | `vllm/parser/glm47_moe.py:195-198` |
| 思考が無効 (`thinking_enabled=False`) なら、パーサーの初期状態が本文 (`ParserState.CONTENT`) になり、`<think>`/`</think>` を特別扱いする terminal/transition も登録されない | `vllm/parser/glm47_moe.py:81,110-125,132` |
| 思考が無効なら `extract_reasoning` が `(None, model_output)` を返し、出力全体を本文として返す | `vllm/parser/glm47_moe.py:217-223` |
| ストリームは、最初の差分でプロンプト末尾のトークンが思考の終わりかを見る (`is_reasoning_end(prompt_token_ids)`) | `vllm/parser/abstract_parser.py:689-696` |
| 非ストリームは `parser.parse(output.text, request, …)` を呼び、プロンプトのトークン列を渡さない | `vllm/entrypoints/openai/chat_completion/serving.py:969-974` |

ここから言えることは 3 つある。

- **新しいテンプレートで、テンプレートとパーサーのスイッチがそろう**: 公式テンプレートは
  `enable_thinking`/`thinking` を参照しないため、`hf.py` の絞り込みで描画に届く前に落ち、
  パーサーだけが思考を切る (sparkDash で思考が本文に漏れた原因)。新しいテンプレートは
  この 2 つの変数を参照するので、描画への到達とパーサーの初期状態がそろう
- **`thinking_mode` は、描画にもパーサーにも届かない**: `resolve_chat_template_kwargs` の
  絞り込みは変数の**参照**で決まり、パーサー (`Glm47MoeParser.__init__`) も
  `thinking`/`enable_thinking` の 2 つしか読まない
- **`/v1/messages` と `/v1/chat/completions` は、同じ `resolved_chat_template` と、同じ
  `chat_template_kwargs` の絞り込み・パーサーへの受け渡しの仕組みを共有する**: 経路ごとに
  別の実装ではない (`api_router.py` が両方の serving クラスに同じ値を渡す)

## 新しいテンプレートの思考オフの条件

生成の書き出しを `<think></think>` にするのは、次の 2 つを両方満たすときだけである。

1. `enable_thinking`/`thinking` のどちらかが指定されていて、`null` (None) でない
2. どちらの値も真でない

これは、パーサーの読み方 (`vllm/parser/glm47_moe.py:192-198`。両方が無指定なら思考あり、
それ以外は `bool(thinking) or bool(enable_thinking)`) と同じである。値が食い違う組
(`enable_thinking: true, thinking: false` など)、明示の `null`、文字列の `"false"` (真として
読む) は思考ありで、公式テンプレートと 1 文字も変わらない。テンプレートの中で
`enable_thinking`/`thinking` に `set` で代入しないのは、代入すると `vllm/renderers/hf.py:709-737`
の変数の絞り込み (テンプレートが参照する名前だけを通す) の判定が変わってしまうためである。

## 確認済みの項目

- **固定したソース木が `0961bbae2894d574be790d219651824eb199318e` と一致すること**
  (2026-09-30、対話側が確かめた): `serving/var/nope-build-0961bbae/source/` で
  `git rev-parse HEAD` が `0961bbae2894d574be790d219651824eb199318e` だった。手元の変更は
  次の 5 ファイルだけ (NoPE の修正とビルドの設定) で、`vllm/parser`、`vllm/renderers`、
  `vllm/entrypoints` に変更は無い
  - `csrc/libtorch_stable/cache_kernels.cu`
  - `docker/Dockerfile`
  - `docker/versions.json`
  - `requirements/cuda.txt`
  - `tests/kernels/attention/test_cache.py`

## 未確認の項目

- **transformers 側の実際のチャットテンプレートの描画環境**: 上のソース木の外にあり、
  読んでいない。`test_chat_template.py` は、公式と新しいテンプレートを、この木で確かめた
  同じ Jinja2 の環境で描画して比べることで、環境の差を打ち消している
- **イメージに入っている Jinja2 の版**: わからない。開発用の依存の版 (`jinja2~=3.1.6`) は、
  ソース木の `requirements/build/cuda.txt` の `>=3.1.6`、`requirements/test/cuda.txt` の
  `==3.1.6` に基づく
- **GLM-5.3 の語彙で `</think>` と `<|assistant|>` が単一トークンであること**: ストリームの
  `is_reasoning_end(prompt_token_ids)` の前提。リポジトリの中では確かめられない
- **実機での動作**: 上の file:line の確認と、ローカルの試験の合格を、実機での動作の
  確認とは扱わない。実機での起動と `serve thinking` の `chat_template_off` の通りでの
  確認は、対話側で行う
