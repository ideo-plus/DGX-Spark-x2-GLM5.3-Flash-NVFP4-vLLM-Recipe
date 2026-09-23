# prefill・並列・長文脈の計測に使う TP=2 構成の準備

目的は、対話側が実機で prefill、並列、長文脈の計測を行うための構成、対象定義、手順を用意すること。
実機検証そのものは行わない。
この依頼と AGENTS.md、CLAUDE.md の準備作業の境界を、計画・テスト・実装・レビュー・修正の全段階で守る。
cc-sdd のスキルは使わない。Git の commit・push・マージ、外部への投稿も行わない。

## 確定していること

- `p2-nope-tp2-smoke`（文脈長 4096、同時実行 1）で、実重み TP=2 の起動、IB 経路、短い応答、
  decode 計測、quality 計測を確認済み。結果は `docs/results/2026-09-23-decode-gpu-clock-cap.md`。
- smoke 構成での vLLM の起動ログの値は、次のとおり。
  - GPU KV cache size: 221,184 tokens
  - Available KV cache memory: 16.61〜16.79 GiB
  - `max_num_batched_tokens` 2048
  - `gpu_memory_utilization` 0.9

  これは実測の観測値で、構成を変えた後の値は保証しない。
- `serving/config/configs.toml` の `p1-nvfp4-tp2` は、`--max-model-len 163840` と `--max-num-seqs 16` を根拠付きで持つ。
  `configure_tp2.py` は、この 2 つを初回用に 4096 と 1 へ置き換えている。
- 両台の GPU には、計測者の判断で `nvidia-smi -lgc 300,1800` のクロック上限を手動で設定している。
  再起動で消える。GB10 の `nvidia-smi` には設定状態が出ない。確認は負荷中の SM クロック（1800 付近）で行う。
  全力負荷で ACPI 熱区域 80℃ に収まることを確認済み。既定クロックでは 92.6℃ まで上がった。
- bench の `quick` プロファイルでは、concurrency の水準が 1, 2, 4, 8 である。
  prefill の狙いは 8k / 32k / 128k、needle は 8k / 32k / 128k。
  agent は 2 万〜12 万トークンまで伸びる（既存手順の記述による）。

## 読む資料

- `experiments/nope-mla/configure_tp2.py` と `serving/tests/unit/test_configure_tp2.py`
- `serving/config/configs.toml` の `p1-nvfp4-tp2`
- `bench/config/targets.toml` の `p2-nope-tp2-smoke`、`bench/config/profiles.toml`、`bench/README.md`、
  bench の CLI（`bench/src/`）。suite と `--set` で変えられる項目、その下限と上限を確かめる
- `docs/vllm-baseline/patched-tp2-procedure.md`、`docs/vllm-baseline/initial-benchmark-procedure.md`
- `docs/results/2026-09-23-decode-gpu-clock-cap.md`、`HAND_OFF.md`
- 生データのうち読んでよいのは `serving/var/nope-build-0961bbae/image-inspect.json` だけ。
  他の `serving/var/`、`results/`、認証情報、要求・応答の本文、`exl3-tp2` の中身は読まない

## 成果物

書き込み先は次の 4 ファイルに限る。実行時に生まれる一時ファイルと takt のレポートは除く。
既存の `serving/config/configs.toml`、Serving Kit 本体、既存テストの期待値、ワークフローや規範ファイルは変更しない。

1. `experiments/nope-mla/configure_tp2.py`
   - 既存の `p2-nope-tp2-smoke` の生成は、引数なしのときの出力をバイト単位で変えない。
   - 新しく `p2-nope-tp2-full` を生成できるようにする（例: `--variant full`）。
   - `p2-nope-tp2-full` では、`max-model-len` と `max-num-seqs` を `p1-nvfp4-tp2` の値（163840、16）のまま残す。
     `why`・`source`・`quote` も正本から引き継ぐ。新しい数値を発明しない。
   - イメージ、重み、NCCL の記録 3 つ（`NCCL_DEBUG` など）は smoke と同じにする。
     IB の確認は起動のたびに必要なので、NCCL の記録は外さない。
   - description には、prefill・並列・長文脈の計測用で、実機では未確認であることを書く。
   - 出力先が既存なら上書きしない。入力が不正なら出力を作らない。これは既存の性質と同じ。
2. `serving/tests/unit/test_configure_tp2.py`
   - 既存の試験と期待値は変えない。
   - full の生成を試験に追加する。
     - 生成した TOML を既存の設定ローダーと起動計画の生成処理に通し、2 台ぶんの argv を確かめる。
     - 見るのは、`--max-model-len 163840`、`--max-num-seqs 16`、NCCL の 3 つ、smoke と同じイメージ ID と重み。
   - 引数なしの出力が従来と同じであることも固定する。
3. `bench/config/targets.toml`
   - `targets.p2-nope-tp2-full` を追加する。
     - `base_url` は `http://10.0.1.60:8000`、`model` は `glm-5-3-flash`、`max_context_tokens` は 163840。
     - notes で、起動側の構成名が同じ `p2-nope-tp2-full` であること、実機では未確認であることを示す。
   - 既存定義を変えない。秘密の値や `api_key_env` は追加しない。
4. `docs/vllm-baseline/full-context-procedure.md`（新規、最大 140 行の日本語）
   - 生成のコマンドを書く。出力先は `serving/var/nope-build-0961bbae/tp2-full.toml`。
   - 起動前に次を確かめる手順を書く。
     - 全関門の再検査
     - 起動、`status=ready`、今回分の IB。NCCL ログの PID が過去と同じで上書きされる場合があるので、開始時刻と初期化回数で帰属を確かめる
     - `serve smoke --max-tokens 512`
   - 起動ログの `GPU KV cache size` と `Maximum concurrency` を記録する手順を書く。
     163840 トークンの要求 1 本が入らなければ、計測へ進まず記録して停止する。
   - GPU クロック上限を確かめる手順を書く。
     - 計測前に上限が入っていることを負荷中の SM で確かめる。外れていれば計測者の了承を得て入れ直す
     - 計測中は GPU の温度・SM・電力と、ACPI 熱区域の最高値を 10 秒ごとに読み取りで記録する
     - ACPI が 90℃ に達した区間は基準値として扱わない
     - クロック上限の値を計測の条件として記録する
   - suite ごとに別の計測ランにする。順番は prefill → concurrency → quality（needle）とし、各コマンドを bench の実際の CLI で書く。
     agent を入れるかは、CLI で選べる単位と所要時間をソースで確かめて判断し、根拠を書く。
     存在しないフラグや、下限に違反する `--set` を書かない。
   - 並列の結果は同時実行 16 の構成で測ったことを明記し、smoke 構成の decode と混同しない。
   - 終了コードだけでは合否にならない点を書く。
     `insufficient_trials`、skip、要求の失敗、採点不能、`completed` 以外の状態を成功として扱わない。
   - 計測後はログ回収・所有確認・停止まで行う。`bench` の本文は Git 対象外の `results/` に置き、公開文書に載せない。

## 禁止事項と検証

根拠はリポジトリ内の資料と上記の観測値だけを使う。
SSH、rsync、Docker、実機向けの serve・bench、Web 検索、重み取得、ミューテーションは禁止。

検証は次に限る。

- `serving` で `uv run pytest tests/unit/test_configure_tp2.py tests/unit/test_readme_config_names.py tests/unit/test_procedure_doc.py`、ruff、mypy
- `bench` で `uv run pytest tests/unit/test_config.py`、ruff

既存試験を緩めない。試験を通すためだけに期待値を置き換えない。

完了報告には次を明記する。

- 変更したファイル
- 検証結果
- 実機では未確認であること
- KV 容量や起動可否など、実機でしか確かめられない事項
