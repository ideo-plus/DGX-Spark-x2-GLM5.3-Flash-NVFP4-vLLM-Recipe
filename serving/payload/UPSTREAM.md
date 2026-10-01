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
