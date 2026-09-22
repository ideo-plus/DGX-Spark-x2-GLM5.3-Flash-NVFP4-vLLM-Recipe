# 短い応答確認の出力上限を CLI から指定する

PLAN.md の P1 の API 確認で、実重み TP=2 は ready と IB を確認できた。
既定 64 トークンの `serve smoke` では英語が HTTP 200 / end_turn / 出力 40 で意味の通る文だった。
日本語は HTTP 200 / max_tokens / 出力 64 で text ブロックがなく、短い応答 2 件の合格条件は未達。
本文や思考内容は記録していない。原因を思考の長さだけと断定せず、上限を変えて再確認できるようにする。
実機は対話側がログ回収して停止する。このタスクからは接続しない。

AGENTS.md、CLAUDE.md の準備作業の境界と安全契約の例外を守り、cc-sdd は使わない。
変更してよいファイルは次の 4 つだけ。

1. `serving/src/serving_kit/cli.py`
   - `serve smoke <構成> --max-tokens <数>` を追加する。
   - 既存の `count` 型で正の整数を受け、既定は既存の `lifecycle.SMOKE_MAX_TOKENS`（64）。
   - 既存の `lifecycle.smoke(..., max_tokens=...)` に渡すだけにする。内部処理や応答判定は変更しない。
2. `serving/tests/unit/test_cli.py`
   - 既存の FakeVllm を使い、未指定なら両要求が 64、指定した場合は両要求がその値になることを HTTP 入力で確かめる。
   - 0・負数・整数でない値を拒否し、HTTP 要求を送らないことを確かめる。
   - 本文は stderr のみ、stdout と生成ファイルに本文を残さない既存契約を維持する。
   - 内部呼び出しの形だけを固定する試験や、無関係な既存期待値の変更はしない。
3. `serving/README.md`
   - 既定 64 を維持することと、上限を指定する例を関連箇所に短く追記する。
4. `docs/vllm-baseline/patched-tp2-procedure.md`
   - 初回の 64 トークンで本文が空なら成功扱いにせず、ログ回収・停止する既存条件を維持する。
   - 別試行で全関門・起動・今回の IB を再確認し、`--max-tokens 512` で短い要求を再送する手順を追記する。
   - 再送も上限到達・本文欠落・判定不能なら合格にしない。変更した上限値、状態、長さ、意味の判定だけを記録する。
   - 本文非保持と最終停止の条件を緩めない。重みやサーバー設定は変えない。

範囲外: lifecycle.py の変更、空の本文に対する CLI 終了コード変更、thinking 設定の切替、性能計測、他の構成や試験の変更。
SSH、rsync、Docker、実機向け serve/bench、Web 検索、重み取得、Git の commit/push/外部投稿は禁止。
serving/var/、results/、認証情報、実際の要求・応答本文、exl3-tp2 の中身を読まない。
HAND_OFF.md と docs/results/ の未コミット変更は対話側の実機記録で、今回の変更対象ではない。

検証: serving で test_cli.py と e2e/test_safety.py、ruff check/format、mypy。
同じ検証を理由なく何度も繰り返さない。全試験は対話側の統合時に CI が行う。
変更した 4 ファイル、検証結果、実機での再試行が未実施であることを日本語で報告する。
