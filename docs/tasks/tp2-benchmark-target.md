# 起動確認後に使う bench の対象定義を用意する

PLAN.md の P1 の API 互換確認と P0 の初回計測に向けた小さな準備作業。
AGENTS.md と CLAUDE.md の準備作業の境界を守る。実機の TP=2 起動は未確認であり、成功と書かない。
TAKT の `simple-mini` を使う。コードの新規実装は不要。

変更してよいファイルは 2 つだけ:

1. `bench/config/targets.toml`
   - `targets.p2-nope-tp2-smoke` を追加する。
   - base_url は `http://10.0.1.60:8000`、model は `glm-5-3-flash`、max_context_tokens は 4096。
   - notes で、実重みでの起動が未確認、同時実行 1 の初回検証用であることを示す。
   - 既存定義を変えず、秘密の値や api_key_env は追加しない。
2. `docs/vllm-baseline/initial-benchmark-procedure.md`
   - 最大 80 行の日本語。通常応答・SSE・ツール呼び出しをどの既存コマンドで確認するかを書く。
   - 先に `patched-tp2-procedure.md` の短い起動試行が成功し、ログ回収・停止まで終わった後の別試行とする。
   - 再試行は全関門を再検査し、同じ構成で起動、今回の IB を確認してから送信し、最後にログ回収・停止する。
   - 最初は `--suite decode` を明示する。既定の全速度条件や concurrency は使わない。
   - `--max-num-seqs=1` なので並列性能を測ったとは扱わない。4096 を超える prefill/needle/agent は後の計画に残す。
   - quality は既存 CLI で選べる単位を確認する。toolcall だけを選べない場合は明記し、存在しないフラグや `quality.code_problem_limit=0`（ge=1 違反）を提案しない。
   - `bench` は合成データの要求・応答を Git 管理外の results/ に保存することを明記する。serving の本文非保持と混同せず、公開文書に本文を載せない。
   - 終了コードだけでは品質の合否にならない点、データ不足・skip・API エラーを成功扱いしない点を書く。

根拠はリポジトリの PLAN.md、bench/README.md、bench/config/、bench/src/、serving/config/ と既存手順だけを使う。
SSH、rsync、Docker、実機向け serve/bench、Web 検索、重み取得、pytest の全件再実行、ミューテーションは禁止。
serving/var/、results/、認証情報、要求・応答本文、exl3-tp2 の中身は読まない。
Git の変更・commit・push・外部投稿は行わない。

検証は bench の既存設定ローダーで追加対象が読めることと、`tests/unit/test_config.py`、ruff に限定する。
設定追加だけなので新しいテストを書かず、既存試験を緩めない。実行可能な CLI 引数はソースで確認する。
完了報告では変更 2 ファイル、検証結果、実機未確認を明記する。
