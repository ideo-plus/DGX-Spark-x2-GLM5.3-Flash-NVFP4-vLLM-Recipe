# 作業の原則

- 返答と文書は日本語にする。`.kiro/specs/`、`amadeus/`、`openspec/` の Markdown も日本語にする。
- 次の行動を選択肢で示す場合は、重複のない選択肢とし、推奨を先に置く。
- GitHub の Pull Request に言及するときはリンクを付ける。
- worktree を使う場合は、その worktree で `mise trust` を実行する。

## takt から実行する準備作業

`CLAUDE.md` の「現在の開発手順」と「このリポジトリの試験の方針の例外」を守る。
takt の担当は、Mac 上のコード・構成・試験・文書の準備に限る。
SSH、rsync、Docker、実機向けの `serve` コマンド、実重みの取得は実行しない。
`serving/var/` の生ログや取得した上流ソースは、依頼に明示されたファイル以外は読まない。
公開する記録には要求・応答の本文や認証情報を含めない。`exl3-tp2` の中身は調べない。
承認済みの安全条件を緩めず、Git の commit・push・マージも依頼なしに行わない。
cc-sdd のスキルは使わず、既存仕様は判断の根拠として参照する。

## ローカルでの検証

`serving/` の標準検証は `uv run pytest`、`uv run ruff check .`、
`uv run ruff format --check .`、`uv run mypy`。
これらの試験は偽の実行役を使い、Spark に接続しない。
`bench/` の全試験には Docker を使うものがあるため、takt の準備作業では依頼された範囲だけを実行する。
