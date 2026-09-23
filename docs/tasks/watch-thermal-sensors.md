# serve watch に熱と CPU の観察を加える (#12、#11 の前提)

目的は、計測中の熱源を切り分けられる記録を、Git 管理の読み取り専用の道具で残せるようにすること。
既存の `serve watch`（`serving/src/serving_kit/watch.py`）を広げる。新しい道具は作らない。
実機検証そのものは行わない。
この依頼と AGENTS.md、CLAUDE.md、`docs/rules/**/*.md` の規則を、計画・テスト・実装・レビュー・修正の全段階で守る。
Git の commit・push・PR 作成は TAKT の pipeline が行う。それ以外の Git 操作や外部投稿はしない。

## 背景

- 2026-09-23 の計測では、GPU クロックに上限 1800 をかけても、cold 128k prefill の間に
  head の ACPI 熱区域 0・4 が 89.8℃ に達した（`docs/results/2026-09-23-full-context.md`、issue #11）。
- そのときの記録は、対話側がその場で書いた Git 対象外のスクリプトで取った。
  そのうえ `docs/vllm-baseline/full-context-procedure.md` §5 が、そのスクリプトを参照している（issue #12）。
- 熱源の候補は 3 つある。
  - ConnectX-7。hwmon の `mlx5` の `asic` で、待機中でも 47℃
  - X925 の 1 コアの張り付き
  - GPU

  切り分けには、これらを同じ時刻で並べた記録が要る。
- GB10 の `clocks_event_reasons` とカウンタは当てにならない。判断は実測の温度とクロックで行う。

## 読む資料

- `serving/src/serving_kit/watch.py`、`types.py`、`cli.py`、`remote.py`（許可の一覧と `_check_cat`）、`serving/tests/unit/test_remote.py`
- `serving/tests/unit/test_watch.py`、`serving/tests/e2e/test_safety.py`
- `serving/README.md`、`docs/vllm-baseline/full-context-procedure.md`
- `docs/results/2026-09-23-decode-gpu-clock-cap.md`、`docs/results/2026-09-23-full-context.md`
- `serving/var/`、`results/`、認証情報、要求・応答の本文、`exl3-tp2` の中身は読まない

## 観察に加える項目

1 回の観察ごとに、2 台それぞれで次を読む。

1. **GPU:** 温度、SM クロック、電力、使用率を 1 回の `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` で読む。
   今の `GPU_UTIL_ARGV`（使用率だけ）は、この 1 回に置き換えてよい。「固まった」の判定に使う使用率は、同じ値を使う。
2. **熱区域:** `/sys/class/thermal/thermal_zone<N>/temp` の全区域の値を、区域番号つきで記録する。最高値だけにまとめない。
3. **hwmon:** `/sys/class/hwmon/hwmon<N>/name` が `mlx5`、`nvme`、`acpitz` のものについて、`temp*_input` の値を記録する。
   ラベルがあれば、`temp*_label` も一緒に残す。
4. **CPU:** `/proc/stat` の `cpu<N>` 行。コアごとの使用率は、前回の観察との差から計算する。
   初回は差がないので空にする。

## 制約

- 読むのに使うのは、許可の一覧にある `cat` と `nvidia-smi` だけ。`remote.py` の `_ALLOWED_COMMANDS` は広げない。
  状態を変える呼び出しを足さない。`test_safety.py` の契約を変えない。
- `cat` の読み取り先（`remote.py` の `_check_cat` と `_CAT_PREFIXES`）は、計測者の判断で次の最小限だけ広げる。
  `.kiro/specs/vllm-baseline/tasks.md` の 1.4 に従い、watch の変更より先に、独立した 1 つの変更として行う。
  - 前方一致で `/sys/class/thermal/` と `/sys/class/hwmon/` を足す。
  - `/proc/stat` を**完全一致だけで**許す。`/proc` の前方一致は許さない。
    `/proc/1/environ`、`/proc/stat/..`、`/proc/statx` などの拒否は維持する。既存の `test_remote.py` の拒否の試験も維持する。
  - `..` を含む道筋の拒否と、オプションの拒否は今のまま。
  - `_check_cat` の docstring、`_CAT_PREFIXES` の説明、エラーメッセージを、新しい読み取り先に合わせて更新する。
  - `test_remote.py` に、新しく許す場所の許可と、境界の拒否の試験を足す。
- シェルの glob は使えない（argv で実行する）。
  - 熱区域と hwmon のパスは、見張りの開始時に 1 回だけ、番号を 0 から順に `cat` して、存在するものを見つける。上限は各 32。
  - その一覧を、見張りのあいだ使う。見つけた一覧は、`result.json` に記録する。
- 1 項目が読めなくても、見張りは止めない。その項目だけを空にする。これは今の watch の方針と同じ。
- 既存の `samples.jsonl` の項目名と意味は変えない。新しい項目を足すだけにする。
  既存の `unresponsive` と `stalled` の判定の振る舞いも変えない。

## 判定と要約

- 新しい出来事 `thermal` を足す。どれかの熱区域が `--thermal-threshold`（既定 90、単位は℃）以上になった観察の、立ち上がりで 1 件にする。
  条件が続く間は 1 件のまま、という扱いは既存の判定と同じにする。
  この出来事では記録の回収（`collect_logs`）をしない。熱では推論サーバーは壊れていないため。
- `result.json` の要約に、台ごと・項目ごと（GPU 温度、SM の最小と最大、電力、各熱区域、各 hwmon）の最大値を足す。
  熱区域が閾値以上だった観察の数と、その時刻の範囲も足す。
- 見張りは読み取りだけで、計測を止めない。止めるかは、要約と出来事を見て対話側が決める。

## 成果物

書き込み先は次に限る。

- `serving/src/serving_kit/remote.py`（`cat` の読み取り先だけ）と `serving/tests/unit/test_remote.py`
- `serving/src/serving_kit/watch.py`、`serving/src/serving_kit/types.py`、`serving/src/serving_kit/cli.py`（`--thermal-threshold`）
- `serving/tests/unit/test_watch.py`。新しい試験ファイルを足してもよい
- `serving/README.md` の watch の説明
- `docs/vllm-baseline/full-context-procedure.md` §5
  - Git 対象外のスクリプトの参照を、`serve watch` に置き換える
  - 計測と並行して `serve watch` を流す手順と、見るべき要約の項目を書く
  - `test_procedure_doc.py` の契約に合わせる

試験では、偽の実行役を使って次を確かめる。

- 熱区域と hwmon の発見と、発見した上限
- コアごとの使用率の差の計算（初回は空）
- 1 項目が読めないときの扱い
- `thermal` 出来事の立ち上がりが 1 件にまとまること
- 要約の最大値
- 既存の判定が変わらないこと
- 呼び出す argv が `cat` と `nvidia-smi` だけで、状態を変えないこと

## 禁止事項と検証

SSH、rsync、Docker、実機向けの serve・bench、Web 検索、重み取得、ミューテーションは禁止。

検証は `serving` で次を行う。

- `uv run pytest`（全件）
- `uv run ruff check .`、`uv run ruff format --check .`
- `uv run mypy`

既存試験を緩めない。試験を通すためだけに期待値を置き換えない。

完了報告には次を書く。

- 変更したファイル
- 検証結果
- 実機では未確認であること
- 範囲外で見つけた汚れ
