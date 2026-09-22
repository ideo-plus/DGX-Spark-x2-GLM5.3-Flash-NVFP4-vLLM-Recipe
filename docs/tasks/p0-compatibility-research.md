# TP=2 起動後の API 互換確認と初回計測を計画する

PLAN.md の P1 にある `/v1/messages` 互換と、P0 の初回計測へ進むための読み取り調査を行う。
AGENTS.md と CLAUDE.md の準備作業の境界を守る。cc-sdd は使わない。
今回は TAKT の research ワークフローを使い、コードや文書を変更せず、最終報告だけを返す。

確定事項:

- 自前イメージの 4 層・ダミー重みの起動は成功済み。実重みの TP=2 起動と API 互換は未確認。
- `docs/vllm-baseline/patched-tp2-procedure.md` の短文脈 4096・同時実行 1 の構成を、対話側で実機検証する。
- 今回の調査は起動成功を仮定した準備であり、実機の成功を宣言しない。
- 準備作業と最低限の CI は main に統合済み。

読む範囲:

- PLAN.md、HAND_OFF.md、docs/vllm-baseline/ の手順と docs/results/ の要約
- bench/README.md、bench/config/、bench/src/、必要な bench/tests/
- serving/README.md、serving/config/、serving/src/、必要な serving/tests/
- 読み取り調査なので pytest の再実行やミューテーションは不要。

禁止事項:

- SSH、rsync、Docker、実機向け serve/bench、外部 API、Web 検索、重み取得を実行しない。
- `serving/var/`、`results/`、認証情報、要求・応答の本文、`exl3-tp2` の中身を読まない。
- Git の変更・commit・push・外部投稿、他のファイルの作成や編集を行わない。

最終報告に含めること（日本語、最大 120 行）:

1. 既存ツールで確認できる API 互換の範囲（通常応答、ストリーミング、ツール呼び出し）をコードの道筋と行番号で示す。
2. 4096 文脈・同時実行 1 に収まる最初の確認と、長文脈や同時実行の構成変更が必要な計測を分ける。
3. 各確認の入力、コマンド案、出力、合否条件を示す。実行はしない。本文を保存しない現行の serving 方針と bench 側の保存仕様の違いも確認する。
4. 既存ツールだけでは不足するものがあれば、必要な変更を最大 3 件の独立した実装タスクとして提案する。無理に新しいコードを増やさない。
5. TP=2 の起動成功前に実行してはいけないこと、実機でしか確定できないことを明記する。
