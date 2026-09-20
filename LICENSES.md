# LICENSES

`bench-harness` が同梱する、または依存する部品・データ・コンテナイメージの、
名前、版、入手先、ライセンス、用途を記録する。

## 方針

- 実行時と開発時の依存は、MIT、Apache-2.0、BSD (系列)、PSF (Python Software
  Foundation License) のいずれかのライセンスのものだけを使う
- 依存を足すときは、先にこのファイルに行を足し、ライセンスが上記のいずれかで
  あることを、その部品の公式の PyPI ページまたは公式リポジトリの `LICENSE`
  ファイルで確かめてから追加する
- 版は、`bench/uv.lock` で実際に解決された版を記録する
- 取得して使うデータと、コードの隔離に使うコンテナのイメージも、同じように
  記録する。どちらもリポジトリには同梱せず、版とハッシュ (またはダイジェスト)
  を固定して取得する

## 依存する部品

版は `bench/uv.lock` で解決された値。ライセンスは、PyPI の申告
(`license_expression`) と、公式リポジトリの `LICENSE` で確かめた (2026-09-20)。

### 実行時の依存

| 名前 | 版 | 入手先 | ライセンス | 用途 |
|---|---|---|---|---|
| httpx | 0.28.1 | https://pypi.org/project/httpx/ | BSD-3-Clause | `/v1/messages` のストリームの読み取り、`/metrics` の取得を行う HTTP クライアント |
| httpx-sse | 0.4.3 | https://pypi.org/project/httpx-sse/ | MIT | Server-Sent Events (SSE) の読み取り |
| pydantic | 2.13.5 | https://pypi.org/project/pydantic/ | MIT | 設定、生データ、要約の型と検証 |
| jsonschema | 4.26.0 | https://pypi.org/project/jsonschema/ | MIT | ツールの引数を、ツールの定義に照らして検証する |
| prometheus-client | 0.26.0 | https://pypi.org/project/prometheus-client/ | Apache-2.0 AND BSD-2-Clause | `/metrics` のテキストの解析 |

- `prometheus-client` の本体は Apache-2.0 で、同梱の `prometheus_client/decorator.py`
  (Michele Simionato 氏による) だけが BSD-2-Clause である。PyPI の申告もこの
  2 つを並べている。どちらも業務で使える

### 開発時の依存

| 名前 | 版 | 入手先 | ライセンス | 用途 |
|---|---|---|---|---|
| pytest | 9.1.1 | https://pypi.org/project/pytest/ | MIT | 試験 |
| pytest-asyncio | 1.4.0 | https://pypi.org/project/pytest-asyncio/ | Apache-2.0 | 非同期の試験 |
| pytest-httpserver | 1.1.5 | https://pypi.org/project/pytest-httpserver/ | MIT | 試験用の HTTP サーバー (依存の宣言に残しているが、いまは使っていない。下記) |
| ruff | 0.16.8 | https://pypi.org/project/ruff/ | MIT | 整形と静的検査 |
| mypy | 2.3.1 | https://pypi.org/project/mypy/ | MIT | 型の検査 (strict) |

- `pytest-httpserver` は、PyPI がライセンスを申告していない。公式リポジトリの
  `LICENSE` (MIT License, Copyright (c) 2020 Zsolt Cserna) で確かめた
- `pytest-httpserver` は `bench/pyproject.toml` の開発時の依存に残っているが、
  試験はどれも使っていない (ストリームの遅れと壊れ方まで指定できる、標準
  ライブラリだけの偽のサーバー `bench/tests/fake_server.py` に置き換わった)。
  依存の宣言から外すかどうかは、別のタスクで決める

### 開発に使う外部の道具

`bench/pyproject.toml` の依存の宣言にも `bench/uv.lock` にも入らないが、
開発に必要なもの。

| 名前 | 版 | 入手先 | ライセンス | 用途 |
|---|---|---|---|---|
| uv | 0.12.11 | https://pypi.org/project/uv/ | MIT OR Apache-2.0 | 依存の解決と、仮想環境の管理 (`uv sync`、`uv run`) |
| hatchling | (版は固定していない。`[build-system] requires`) | https://pypi.org/project/hatchling/ | MIT | パッケージのビルドのバックエンド。`uv.lock` には入らない |

## 公開の課題

計測に使う公開の課題は、HumanEval+ だけである。**リポジトリには同梱せず**、
版と URL と SHA-256 を固定して取得し、確かめてから読む。置き場所は git の管理の
対象でない場所 (既定は `bench/data-cache/`) で、再配布はしない。

| 名前 | 版 | 入手先 | ライセンス | 用途 |
|---|---|---|---|---|
| HumanEval+ (EvalPlus) | v0.1.10 | https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus-OriginFmt.jsonl.gz | Apache-2.0 | 品質の検査のコードの条件。164 問の指示と検査のプログラム |
| HumanEval (OpenAI) | — | https://github.com/openai/human-eval | MIT | 上の課題の元になった問題そのもの (EvalPlus が拡張したもとの出どころ) |

- 取得するファイル: `HumanEvalPlus-OriginFmt.jsonl.gz` (1,350,689 バイト、
  SHA-256 `daa7661c8189924068069b0872a440b491edb60f8bdf431d5957adc88d18bae5`)。
  ハッシュか大きさが合わなければ、1 バイトも解釈せずに失敗する
- ライセンスは、どちらも公式リポジトリの `LICENSE` で確かめた
  (EvalPlus 本体と `humanevalplus_release` が Apache-2.0、`openai/human-eval` が MIT)
- **`evalplus` も `human-eval` も、コードは 1 行も写しておらず、依存にも加えて
  いない** (クリーンルーム、要件 11.5)。`-OriginFmt` の資産を選んだのは、各問の
  `test` がそれだけで完結した `check(candidate)` の定義になっていて、採点に
  上流のコードが要らないためである
- **`HumanEval/32` は採点の対象から外している。** 固定した版の検査のプログラムが
  `_poly(*candidate(*inp), inp)` と書かれていて (引数の順が逆)、データセット
  自身の正解の解でも落ちる。採点に回すのは残りの 163 問である

ツール呼び出しの課題と、長い入力から情報を探す課題と、長い会話は、すべて自作の
合成データで、外部の課題を使っていない (要件 11.1)。

## コンテナのイメージと、その中の部品

モデルが書いたコードを採点するために動かす、隔離のイメージである。
`bench/sandbox/Dockerfile` から**手元で作る**もので、配布も再配布もしない。

| 名前 | 版 | 入手先 | ライセンス | 用途 |
|---|---|---|---|---|
| python (公式のイメージ、`3.13-slim`) | `sha256:8d9d0b8bcf6506481eae4907c18f5e3e7902e629f5f6d684f9e7c32e85e3ddf0` (Python 3.13.15) | https://hub.docker.com/_/python | CPython の部分は PSF-2.0。同梱の Debian の部品は、それぞれのライセンス | 隔離のイメージの土台 |
| numpy | 2.5.3 | https://pypi.org/project/numpy/ | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | HumanEval+ の検査のプログラム (164 問のうち 163 問) が `np.allclose` などを使うため |

- 土台のイメージはダイジェストで固定し、numpy は版とホイールのハッシュで固定する
  (`bench/sandbox/requirements.txt`、`pip install --require-hashes`)
- **公式のイメージには、Debian の多数の部品と CPython が入っている。それぞれが
  自分のライセンスに従う。** この道具は、イメージを計測の道具として手元で動かす
  だけで、イメージもその中身も再配布しない。イメージに足しているのは numpy だけである
- numpy のライセンスは、PyPI の申告 (同梱の部品を含む表記) をそのまま写した

### コンテナの実行環境

イメージを動かす実行環境そのものも、再配布はしない。計測者の機械に入っているものを使う。

| 名前 | 版 | 入手先 | ライセンス | 用途 |
|---|---|---|---|---|
| Docker Desktop | 4.88.0 (エンジン 29.7.2) | https://www.docker.com/products/docker-desktop/ | 専有。従業員 250 人以上または年商 1,000 万ドル以上の会社では、有償契約が要る (https://docs.docker.com/subscription/desktop-license/) | この Mac での隔離の実行環境 (計測者の判断、2026-09-20) |
| Podman | — | https://github.com/containers/podman | Apache-2.0 | 対応している代わりの実行環境 (この Mac には入れていない) |

- この道具は `podman` と `docker` のどちらでも動く (`profiles.toml` の
  `sandbox.runtime`)。**どちらも入っていなければ、コードの条件だけを理由つきで
  飛ばす** (隔離なしでは動かさない)
- Docker Desktop の有償の条件に当たる場合は、Podman (Apache-2.0) に切り替える。
  設定の 1 行で済む

## 確かめ方

依存の宣言と、解決された版の一覧は、次のコマンドで出し直せる (`bench/` で実行する)。
このファイルの表と突き合わせること。

```bash
python3 - <<'PY'
import pathlib, tomllib

project = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
print("--- 宣言している部品 (pyproject.toml) ---")
for spec in project["project"]["dependencies"]:
    print("runtime:", spec)
for group, specs in project.get("dependency-groups", {}).items():
    for spec in specs:
        print(group + ":", spec)

lock = tomllib.loads(pathlib.Path("uv.lock").read_text())
print("--- 解決された版 (uv.lock) ---")
for package in sorted(lock["package"], key=lambda p: p["name"]):
    print(package["name"], "==", package.get("version"))
PY
```

`uv.lock` には、間接の依存 (`anyio`、`certifi`、`h11`、`httpcore`、`idna`、
`pydantic-core`、`rpds-py` など) も入っている。このファイルに書いているのは、
**直接宣言している部品**だけである。

隔離のイメージの識別子は、次のコマンドで確かめる。`config/profiles.toml` の
`sandbox.image_digest` と一致していなければ、採点は動かない。

```bash
docker image inspect bench-sandbox:py3.13-numpy2.5.3 --format '{{.Id}}'
```

PyPI の申告しているライセンスは、次のように読める。

```bash
python3 - <<'PY'
import json, urllib.request

for name, version in [("httpx", "0.28.1"), ("prometheus-client", "0.26.0")]:
    url = "https://pypi.org/pypi/" + name + "/" + version + "/json"
    with urllib.request.urlopen(url) as response:
        info = json.load(response)["info"]
    print(name, version, info.get("license_expression") or info.get("license"))
PY
```

申告のない部品 (`pytest-httpserver` など) は、公式リポジトリの `LICENSE` を読む。
