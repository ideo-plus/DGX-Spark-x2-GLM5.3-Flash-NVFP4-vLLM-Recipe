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
- 公開の課題 (HumanEval+ など) と、コードの隔離に使うコンテナのイメージの行は、
  タスク 8.5 で追加する (取得と選定がまだこのタスクの範囲外のため)

## 依存する部品

### 実行時の依存

| 名前 | 版 | 入手先 | ライセンス | 用途 |
|---|---|---|---|---|
| httpx | 0.28.1 | https://pypi.org/project/httpx/ | BSD-3-Clause | `/v1/messages` のストリームの読み取り、`/metrics` の取得を行う HTTP クライアント |
| httpx-sse | 0.4.3 | https://pypi.org/project/httpx-sse/ | MIT | Server-Sent Events (SSE) の読み取り |
| pydantic | 2.13.5 | https://pypi.org/project/pydantic/ | MIT | 設定、生データ、要約の型と検証 |
| jsonschema | 4.26.0 | https://pypi.org/project/jsonschema/ | MIT | ツールの引数を、ツールの定義に照らして検証する |
| prometheus-client | 0.26.0 | https://pypi.org/project/prometheus-client/ | Apache-2.0 | `/metrics` のテキストの解析 |

### 開発時の依存

| 名前 | 版 | 入手先 | ライセンス | 用途 |
|---|---|---|---|---|
| pytest | 9.1.1 | https://pypi.org/project/pytest/ | MIT | 試験 |
| pytest-asyncio | 1.4.0 | https://pypi.org/project/pytest-asyncio/ | Apache-2.0 | 非同期の試験 |
| pytest-httpserver | 1.1.5 | https://pypi.org/project/pytest-httpserver/ | MIT | 試験用の偽のサーバー |
| uv | 0.12.11 | https://pypi.org/project/uv/ | MIT OR Apache-2.0 | 依存の解決と環境の管理 |
| ruff | 0.16.8 | https://pypi.org/project/ruff/ | MIT | 整形と静的検査 |
| mypy | 2.3.1 | https://pypi.org/project/mypy/ | MIT | 型の検査 (strict) |

## 公開の課題

このタスク (1.1) の時点では、公開の課題は未取得。取得と出どころの記録は、
タスク 6.5 (HumanEval+ の取得) と、8.5 (このファイルへの追記) で行う。

## コンテナのイメージ

このタスク (1.1) の時点では、コードの隔離に使うコンテナのイメージは未選定。
選定と記録は、タスク 6.1 (実行環境の選定) と、8.5 (このファイルへの追記) で行う。
