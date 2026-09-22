# 準備作業を takt、実機操作を対話に分ける

takt は Mac 上で構成・コード・試験・文書を作る。Spark の状態確認、配布、重み取得、起動、計測、停止は
対話側が担当する。最初の依頼は [TP=2 検証の準備](../tasks/tp2-preparation.md)。

## 導入したもの

- TAKT 0.66.0 と、固定した `takt-workflows` の日本語バンドルを使う。
- `spark-preparation` は上流の `flash-default` に、実機操作を行わないポリシーを加えたもの。
  計画、テスト、実装、レビュー、修正に同じ境界を渡す。
- `CLAUDE.md` と `AGENTS.md` に、文字列・argv・SHA-256 を使う安全条件や根拠固定のテストを維持する例外を記載した。
- 出典とローカルの変更点は [.takt/UPSTREAM.md](../../.takt/UPSTREAM.md) にある。

モデルの接続先はマシン側の `~/.takt/runtime.yaml` に置き、リポジトリには保存しない。
プロジェクトの `.takt/runtime.yaml` は profile 名による段階の割り当てだけを共有する。
導入時の T0 は、テストが OpenCode 経由の Ollama Cloud、実装が OpenCode Go。
Ollama の接続先は `localhost:11434` で、Spark は使わない。
T1 は Sonnet、T2 は Opus、計画は Fable、最終判断は Astra という既存の設定を維持した。

## 実行方法

リポジトリ直下で次を実行する。

```bash
takt workflow doctor spark-preparation
sh scripts/run-takt-preparation.sh
```

起動スクリプトは `--pipeline --skip-git` を指定する。現在のブランチのファイルを使い、
worktree の作成、commit、push、自動でのマージは行わない。未コミットの `.takt/` もこの方式では参照できる。
通常の worktree 実行を使う場合は、設定と依頼を先にコミットし、その worktree で `mise trust` を実行する。

依頼は次の 3 ファイルだけを成果物にする。

- `experiments/nope-mla/configure_tp2.py`
- `serving/tests/unit/test_configure_tp2.py`
- `docs/vllm-baseline/patched-tp2-procedure.md`

準備専用のポリシーと依頼で実機操作を禁止しているが、これは OS のネットワーク隔離ではない。
実行ログと変更差分を対話側で確認し、準備が通っても実機で成功したとは扱わない。

## 確認と引き取り

生成器が作った TOML を実際の設定ローダーと起動計画の生成処理に通し、2 台の引数と役割を確認する。
ローカルのテスト・ruff・mypy が合格したら、対話側が変更差分を確認する。
その後、取得容量と状態変更の範囲を計測者に示し、実機検証の了承を得る。

takt のセッションとレポートは `.takt/` の Git 対象外の領域に置く。
`--auto-pr` や `takt run` によるキュー全体の実行は、この準備用ランチャーでは使わない。

## 初回の実行記録 (2026-09-23)

TP=2 の依頼を実行し、生成器・試験・手順書を作成した。実装、補助レビュー、全体レビュー、
修正、修正検証まで約 129 分掛かった。修正検証で手順書に 2 件の残件が確認されたため、
対話側が引き取り、追加周回を SIGINT で止めた（終了値 130）。ワークフロー全体の承認完了ではない。
残件は、取得前のディレクトリの扱いと、取得時には作られない起動記録の欠落判定だった。
対話側でこれらと NCCL ログの帰属、転送時間の説明を修正し、不正な JSON 形状の試験を補った。
実重みの取得・実機操作は実施していない。

手元の実行ログとレビュー報告は次の場所にある（Git には含めない）。

- 進行ログ: `.takt/verification/tp2-run.log`
- 終了値と開始・終了時刻: `.takt/verification/tp2-run-result.json`
- 詳細なイベントと報告: `.takt/runs/20260922-162838-tp-2-agents/`

```bash
tail -f .takt/verification/tp2-run.log
```

`takt workflow doctor spark-preparation` は警告なしで通過した。
TAKT 0.66.0 の `takt prompt spark-preparation` は `reportContent is required for report-based judgment`
で失敗する。同じエラーは未変更の `flash-default` でも再現するが、上記の実行自体は開始・進行できた。

## 次の作業を小さく分ける

準備と CI の統合後、[API 互換・計測準備の調査](../tasks/p0-compatibility-research.md) を
組み込みの `research` で実行した。約 19 分で終了値 0。非ストリーミング応答は `serve smoke`、
SSE とツール呼び出しは `bench` で確認できること、8000 番と実際のモデル名を使う対象定義が
必要なことを確認した。レビューが指摘した `quality.code_problem_limit=0` は無効なため採用しない。
調査ログは `.takt/verification/p0-research-run.log` にある。

続いて [対象定義と初回計測手順](../tasks/tp2-benchmark-target.md) を `simple-mini` で実行し、
約 30 分でレビュー承認・終了値 0 まで完了した。設定ローダー、既存の設定テスト 30 件、ruff が通過。
対話側で、対象外の skip と今回の失敗条件を分け、非ストリーミング確認のコマンドを明記した。
変更範囲は設定と手順の 2 ファイルに限定し、既存の設定検査を使う。実機操作の禁止は依頼と
AGENTS.md で維持する。コード実装を伴う最初の依頼と同じ規模のワークフローを毎回使うのではなく、
調査・小さな設定変更・コード実装に分ける。ログは `.takt/verification/benchmark-target-run.log`。

リポジトリ直下での実行コマンドは次のとおり。いずれも実機には接続しない依頼を渡す。

```bash
takt --pipeline --skip-git --workflow research --task "$(cat docs/tasks/p0-compatibility-research.md)"
takt --pipeline --skip-git --workflow simple-mini --task "$(cat docs/tasks/tp2-benchmark-target.md)"
```

実機の短い応答確認で日本語が既定 64 トークンの上限に達したため、
[出力上限を指定する変更](../tasks/smoke-token-budget.md) も `simple-mini` で実行した。
約 26 分でレビュー承認・終了値 0。関連 209 件、ruff、mypy が通過した。
ログは `.takt/verification/smoke-budget-run.log`。実機での再試行は対話側が担当する。
