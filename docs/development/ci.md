# GitHub Actions でローカル検査を毎回実行する

変更提案、main への push、手動実行を対象に、
[CI](../../.github/workflows/ci.yml) が `bench` と `serving` を別々に検査する。
Ubuntu 24.04、Python 3.12、uv 0.12.17 を使い、依存は各プロジェクトの `uv.lock` から取得する。
lock と pyproject が一致しなければ `uv sync --locked` が失敗する。

両方で pytest、ruff check、ruff format の確認、mypy を実行する。
serving 側では `experiments/nope-mla/` の静的検査と、takt 起動スクリプトの構文検査も行う。
Actions はコミット SHA で固定し、リポジトリへの権限は読み取りだけにする。

Spark 接続、モデル取得、GPU 検証、takt のモデル呼び出しは含めない。
bench の実コンテナを必要とする試験は、専用イメージがない環境では既存の fixture により skip される。
その skip を GPU・実機・隔離環境の合格とは扱わない。

手元では各プロジェクトのディレクトリで次を実行する。

```bash
uv sync --locked
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen mypy
uv run --frozen pytest -q
```
