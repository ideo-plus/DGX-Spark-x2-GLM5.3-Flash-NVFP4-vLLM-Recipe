# 準備作業を takt、実機操作を対話に分ける

takt は Mac 上で構成・コード・試験・文書を作る。Spark の状態確認、配布、重み取得、起動、計測、停止は
対話側が担当する。最初の依頼は [TP=2 検証の準備](../tasks/tp2-preparation.md)。

## 導入したもの

- TAKT 0.66.0 と、固定した `takt-workflows` の日本語バンドルを使う。
- `spark-preparation` は上流の `flash-default` に、実機操作を行わないポリシーを加えたもの。
  計画、テスト、実装、レビュー、修正に同じ境界を渡す。
- `CLAUDE.md` と `AGENTS.md` に、文字列・argv・SHA-256 を使う安全条件や根拠固定のテストを維持する例外を記載した。
- 出典とローカルの変更点は [.takt/UPSTREAM.md](../../.takt/UPSTREAM.md) にある。

2026-09-26 から、このプロジェクトのプロファイル（profile 名 → provider とモデル）も、プロジェクトの `.takt/runtime.yaml` に置いて Git で管理する。
TAKT は `~/.takt/runtime.yaml` とプロジェクトのファイルを重ねて読み、同じ名前のプロファイルはプロジェクトが勝つ。
こうすると、マシン全体の設定を、ほかのリポジトリと共有しなくて済む。worktree でも同じ設定で動く。
一時的な切り替え（提供元の障害の回避など）は、その worktree の中だけで書き換え、コミットしない。
以下の 2026-09-26 より前の記述は、当時のマシン側の設定のものである。
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

## モデル割り当ての更新 (2026-09-23)

マシン側の `~/.takt/` の割り当てを次のように変えた。リポジトリの `.takt/runtime.yaml` は profile 名だけなので変更していない。

- T2 を `claude-opus-5-5` にした。
- 実装の T0（`t0-production-code`）は codex / `gpt-6-luna` にした。
  OpenCode Go には `gpt-6-luna` がなく、`Model not found` で失敗したため。
- レート制限時の切り替え先を `claude-opus-5-5` → codex / `gpt-6-sol` にした。

codex を使う profile を動かすため、`~/.takt/config.yaml` に `codex_cli_path` を加え、mise の codex 0.156.0 を指定した。
TAKT 0.66.0 に同梱の codex 0.153.4 は、ChatGPT アカウントで `gpt-6-luna`・`gpt-6-sol` を HTTP 400 で拒否した。
`gpt-6-astra` は通った。手元の CLI での単体確認では気づけないので、TAKT 経由の実行ログで確かめる。

リポジトリの外の使い捨てプロジェクトで、`flash-default` を使って小さなタスクを流した。
実行ログの `step_start` に記録された provider・model で割り当てを確かめた。

| 回 | 結果 |
|---|---|
| 1 | 同梱の codex で implement が HTTP 400。replan を 2 回繰り返して停止 |
| 2 | 全 step が割り当てどおりに動いた。final-gate で Codex の利用上限に達して停止 |
| 3 | report phase で `ReportPhaseToolCallError` が起き、TAKT のプロセスが異常終了 |
| 4 | plan から final-gate まで通過。16 分 39 秒、`Result: Success` |

2 回目の Codex の利用上限では、「You've hit your usage limit」という文言が TAKT のレート制限の判定パターンに合わなかった。
そのため `rate_limit_fallback` が働かず、そのまま止まった。
上限が戻る 2026-09-27 19:38 までは、`t0-production-code` を `claude-sonnet-5`、`t3-judge` を `claude-opus-5-5` に移している。
切り替え先の `gpt-6-sol` も外した。戻すための記述は `~/.takt/` の各ファイルにコメントで残した。
3 回目の異常終了は 1 回だけで、同じ設定の再実行では起きなかった。

## pipeline で PR まで作るとき (2026-09-23)

`takt --pipeline --auto-pr -b <ブランチ>` は、ブランチの作成、全変更のステージとコミット、push、PR の作成まで行う。
ただし TAKT 0.66.0 では、`--task` で依頼文を渡すと、依頼文の全文が次の 2 つにそのまま使われる。

- コミットメッセージ (`takt: <依頼文>`)
- PR のタイトル

[PR #13](https://github.com/ideo-plus/DGX-Spark-GLM5.3-Flash-Recipe/pull/13) の作業では、タイトルが 256 文字を超えて PR の作成に失敗した。
`pipeline.commit_message_template` は、issue を渡したときにしか効かない。

そのため、PR まで作る依頼は GitHub issue の本文に書き、`--issue <番号>` で渡す。
このとき PR のタイトルは `[#番号] <issue のタイトル>`、コミットメッセージは `feat: <issue のタイトル> (#番号)` になる。
PR へのレビューコメントは、`takt --pr <番号>` で取り込む。

## claude のアカウントを切り替えて起動する (2026-09-24)

TAKT 0.66.0 は、環境変数 `TAKT_CLAUDE_CLI_PATH`（または `~/.takt/config.yaml` の `claude_cli_path`）に指定した実行ファイルで claude を起動する。
指定しなければ、起動したシェルの `CLAUDE_CONFIG_DIR` のアカウントがそのまま使われる。
その場合、対話側のセッションと同じアカウントの上限を分け合うことになる。

アカウントの上限に当たったときは、`scripts/run-takt.sh` で別のアカウントを指定して起動する。

```bash
scripts/run-takt.sh --claude-account ~/.claude-<アカウント> --pipeline --auto-pr -b <ブランチ> -i <issue> --workflow spark-preparation
scripts/run-takt.sh --claude-account ~/.claude-<アカウント> resume
```

- `run-takt.sh` は、`TAKT_CLAUDE_CLI_PATH` に `scripts/takt-claude.sh` を絶対パスで渡す。`--claude-account` の後ろの引数は、そのまま `takt` に渡す。
- `takt-claude.sh` は、そのアカウントの `CLAUDE_CONFIG_DIR` で `claude` を起動する。
- アカウントの指定がない、または設定ディレクトリがないときは、既定のアカウントに黙って戻さず失敗する。
- `CLAUDE_CODE_OAUTH_TOKEN`、`ANTHROPIC_API_KEY`、`ANTHROPIC_AUTH_TOKEN` は外してから起動する。
  これらは `CLAUDE_CONFIG_DIR` より優先されるので、残っているとアカウントが切り替わらない。
- アカウントを替えて `takt resume` するときは、先に `takt clear` で会話の ID を消す。
  会話はアカウントの設定ディレクトリごとに保存されるので、前のアカウントの会話は見つからない。
- アカウントの名前はマシンごとの事情なので、リポジトリには書かない。
- 切り替えが効くのは claude の段だけである。codex と opencode の段は、それぞれの設定のアカウントで動く。
