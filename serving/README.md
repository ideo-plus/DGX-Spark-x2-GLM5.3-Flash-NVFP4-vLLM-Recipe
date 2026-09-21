# serving-kit

`serving-kit` は、DGX Spark 2 台に vLLM の推論サーバーを立てるためのコマンドラインの道具
(Serving Kit) である。`bench/` と同じ流儀 (src レイアウト、uv、`uv.lock` をコミット、ruff、
mypy strict、pytest) で作る。詳しい使い方 (サブコマンドの一覧、終了コード、Spark の上の
置き場所、了承の流れ) は、サブコマンドをつなぐタスク (5.1) でここに書く。

このタスク (1.1) の時点では、`serve` コマンドの入口だけが動く。

```bash
uv sync
uv run serve --help
uv run serve --version
```

依存のライセンスは、リポジトリ直下の [`LICENSES.md`](../LICENSES.md) にある。
