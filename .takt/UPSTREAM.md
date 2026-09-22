# takt-workflows の導入元

日本語バンドルを [ideo-plus/takt-workflows](https://github.com/ideo-plus/takt-workflows) の
コミット `4eaeb89c986d023b2e6f9b3755852acc2648c812` からコピーした。
ライセンスは Apache-2.0。全文は `LICENSE.takt-workflows` に保存している。
TAKT 本体は 0.66.0 (MIT) を使う。

`facets/`、`steps/development-core-write-tests.yaml`、`flash-*.yaml` は上流由来。
ローカルの追加・変更は次のとおり。

- `spark-preparation.yaml` は `flash-default.yaml` を基に、準備作業のポリシーを計画・テスト・実装・レビューへ渡す。
- `flash-implement-dynamic.yaml` と `flash-remediation-dynamic.yaml` に、そのポリシーの参照を追加した。
- `facets/policies/spark-preparation.md` は、このリポジトリの実機操作の境界と試験方針を定める。
- `runtime.yaml` は上流の段階別割り当てを使う。モデルと接続先は各マシンの `~/.takt/runtime.yaml` に置く。

上流バンドルを入れ直すと `flash-*.yaml` のローカル変更も置き換わるため、
上記のポリシー参照を復元してから `takt workflow doctor spark-preparation` で確認する。
実行ログ、セッション情報、タスク状態は `.takt/.gitignore` で除外する。
