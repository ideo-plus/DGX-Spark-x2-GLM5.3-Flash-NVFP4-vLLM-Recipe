# bench-harness

`bench-harness` は、DGX Spark 2 台で動く GLM-5.3-Flash の推論サーバーを、作業用の
Mac から Anthropic 互換の `/v1/messages` 経由で測る、コマンドラインの計測の道具
である。生成の速さ、入力の処理と最初のトークンまでの時間、同時処理、品質の検査、
長い会話でのツール呼び出しを、同じ入力と同じ手順で何度でも測り直せるようにする。

## 開発環境

このパッケージは `uv` で管理する。`bench/` に移動してから、以下のコマンドを使う。

```bash
# 依存を解決し、開発用の仮想環境を作る
uv sync

# 整形の確認
uv run ruff format --check .

# 静的検査
uv run ruff check .

# 型の検査 (strict)
uv run mypy

# 試験
uv run pytest -q

# CLI の疎通確認
uv run bench --help
uv run bench --version
```
