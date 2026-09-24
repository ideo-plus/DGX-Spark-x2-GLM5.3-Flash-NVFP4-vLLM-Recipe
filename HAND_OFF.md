# HAND_OFF — 熱対策の常設と、agent 20k の初回計測の後

最終更新: 2026-09-24 08:30 JST。
熱源を X925 の張り付きと特定し、GPU と X925 の 2 つの上限を systemd で常設した。
そのうえで、agent 20k を初めて計測した。
推論サーバーは停止済み。次の実機操作は計測者の指示を待つ。

## 最初に読む要約

- **計測の条件（常設）:** 両台で `spark-power-caps.service` が次の 2 つを起動のたびに入れる。
  - GPU の上限 `-lgc 300,1800`
  - X925（CPU 5〜9、15〜19）の上限 3.0 GHz

  値は `/etc/default/spark-power-caps`、写し元は `serving/payload/spark-power-caps.default`。
  値の変更と反映は、sudoers で許した固定形でパスワードなしに行える（[README](ops/spark-power-caps/README.md)）。
- **上限の確かめ方:** 計測中の上限は、`serve watch` の `result.json` で確かめる。
  - CPU は `cpu_cluster_max_freq_khz`（x925 が 3000000）
  - GPU は `gpu_sm_clock_range_mhz`（1800 以下）

  GB10 の `nvidia-smi` には、GPU の上限の設定状態が出ない。減速の理由とカウンタも当てにならない。
- **熱:** ACPI 熱区域 0・4 の余分な熱は、処理中に X925 の 2 コアが張り付くこと（推論サーバーの待ち受けと推定）による（[記録](docs/results/2026-09-23-thermal-source.md)）。
  - X925 を 3.0 GHz にすると、head の最高は 86.9℃ から 70.8℃ に下がった。
  - 代わりに、cold 128k の prefill が約 4% 下がる。
- **結果の記録:**
  - smoke 構成（文脈長 4096、同時実行 1）の decode・quality: [記録](docs/results/2026-09-23-decode-gpu-clock-cap.md)
  - full 構成（文脈長 163840、同時実行 16）の prefill・concurrency・needle: [記録](docs/results/2026-09-23-full-context.md)
  - agent 20k: [記録](docs/results/2026-09-24-agent-20k.md)。50/50 が正しく、崩れは 0。ただし「試行の数が足りない」。熱区域の最高は 64.9℃。
- **停止の状態:** 08:10 JST 過ぎの停止後、両台ともコンテナは absent、GPU プロセスは 0 件。再開時は状態を読み直す。
- **TAKT:** 準備作業は、依頼を GitHub issue に書き、`takt --pipeline --auto-pr -b <branch> -i <番号>` で PR まで流す（下記「TAKT と CI」）。
- **計測者への依頼の仕方:** 長いコマンドや sudo の手打ちを頼まない。モバイルからは打てない。
  root が要る変更は、NOPASSWD の固定形で対話側が実行できる設計にし、了承は返事だけで済むようにする。
- **作業の規則:** `docs/rules/**/*.md` を毎回読む。Claude Code ではフックが入れる。今はボーイスカウトルールがある。

## 再開手順

1. `AGENTS.md`、`CLAUDE.md`、`docs/rules/`、このファイル、最新の記録を読み、`git status` で未コミットの差分を確認する。
2. 両台で次を読み取りで確かめる。
   - `systemctl is-active spark-power-caps.service` が `active`
   - X925 の `scaling_max_freq` が 3000000
   - コンテナが absent
3. 生成済みの構成 `serving/var/nope-build-0961bbae/tp2.toml`（smoke）と `tp2-full.toml`（full）があるか確認する（Git 対象外）。
4. 実機操作は対話側で行う。全関門を再検査して起動し、ready と**今回分**の IB を確認する。
   PID が過去と同じで NCCL ログが上書きされることがあるので、開始時刻と初期化回数で帰属を確かめる。
5. 計測中は `serve watch --interval 10s` を並べる。熱区域と hwmon を見つけるのに、起動から約 2 分半かかる。
   ACPI が 90℃ に達した区間は、基準値として扱わない。
6. 計測後はログ回収・所有確認・停止まで行い、`result.json` で上限と熱を確かめてから記録する。

以下は参照用のコマンド。作業ディレクトリはリポジトリ直下。

```bash
uv run --directory serving serve check p2-nope-tp2-full --configs var/nope-build-0961bbae/tp2-full.toml
uv run --directory serving serve start p2-nope-tp2-full --configs var/nope-build-0961bbae/tp2-full.toml --timeout 3h --yes
uv run --directory serving serve smoke p2-nope-tp2-full --configs var/nope-build-0961bbae/tp2-full.toml --max-tokens 512
uv run --directory serving serve watch p2-nope-tp2-full --configs var/nope-build-0961bbae/tp2-full.toml --interval 10s --duration 3h
uv run --directory bench bench run --target p2-nope-tp2-full --suite agent --profile quick --set agent.end_tokens=20000
```

`quality` は toolcall だけを選択できず、code・needle も計画する。`quality.code_problem_limit=0` は無効。
`serving` は本文を保存しない。一方、`bench` は PLAN.md の方針に従い、合成データの本文を Git 対象外の
`results/` に保存する。公開記録には本文を含めない。`serve smoke` の stderr（応答の本文）はファイルに保存しない。
書かないこと (要件 10.5): 送った内容と応答の本文、認証の情報、`exl3-tp2` の中身。

## 使用する構成と保存場所

| 項目 | 値・場所 |
|---|---|
| 構成名 / bench の対象名 | `p2-nope-tp2-smoke` |
| 生成済みの構成 | `serving/var/nope-build-0961bbae/tp2.toml`（smoke）、`tp2-full.toml`（full、`configure_tp2.py --variant full`）。Git 対象外 |
| イメージ ID | `sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90`（両台に存在。タグではなく ID で照合） |
| 重み | `RedHatAI/GLM-5.3-Flash-NVFP4@18d55bfd5a2194887738da73753975c9d3842f46` |
| 重みの配置 | 両台の `~/vllm-baseline/models/glm-5-3-flash-nvfp4/`。各 19 ファイル・197,881,153,655 B、照合済み |
| 構成ハッシュ | `9bd65ac3e94c8690db3861bee46c7c22fe4985d317d01995c9ee5096072d61cd` |
| API | `http://10.0.1.60:8000`、モデル名 `glm-5-3-flash`（ドットではなくハイフン） |
| 実測の要約 | `docs/results/2026-09-23-patched-tp2.md`、`2026-09-23-decode-gpu-clock-cap.md` |
| 操作の生ログ | `serving/var/nope-build-0961bbae/tp2-*.log`、`*-result.json`、`*-verdict.json` |
| 初回の応答後の回収ログ | `serving/var/20260922T215658Z-logs-p2-nope-tp2-smoke/` |
| 512 トークン再試行後の回収ログ | `serving/var/20260922T223912Z-logs-p2-nope-tp2-smoke/` |

NCCL のログは古いものも回収される。所有確認済みコンテナの hostname・開始時刻と照合する。
この環境では hostname はホスト名と同じだった。PID の再利用や同名ファイルの上書きがあるため、
過去のログに IB が書いてあるだけでは今回の合格にしない。

## TAKT と CI

使っている TAKT は 0.66.0。固定したワークフローの導入元、使い分け、検証結果は
[開発手順](docs/development/takt-preparation.md) を参照する。
`.takt/runtime.yaml` は profile 名の割り当て、モデル・接続先の実体は `~/.takt/runtime.yaml` にある。
PR まで作る依頼は GitHub issue の本文に書き、`--pipeline --auto-pr -b <branch> -i <番号>` で渡す（`--task` では全文が PR タイトルになり失敗する）。
TAKT のコミットが、依頼で分けるよう求めた変更を 1 つにまとめることがあるので、PR の前に差分を確かめる。
worktree は作っていない。今後 worktree を使う場合は `mise trust` を実行する。

2026-09-23 に `~/.takt/` の設定を次のように変えた（Git 対象外。控えは `~/.takt/*.bak-20260923`・`*.codex-20260923`）。

| 項目 | 設定 |
|---|---|
| `t2` | `claude-opus-5-5` |
| `t0-production-code` | **2026-09-24 から opencode / `opencode-go/deepseek-v4.1-flash`（計測者の指示で試行中）**。以前は claude / `claude-sonnet-5`、本来は codex / `gpt-6-luna` |
| `t3-judge` | 本来は codex / `gpt-6-astra`。**9/27 19:38 までは claude / `claude-opus-5-5`** |
| `rate_limit_fallback` | `claude-opus-5-5`。本来は続けて codex / `gpt-6-sol`（上限中はコメントアウト） |
| `codex_cli_path` | mise の codex 0.156.0 の実体。codex を更新したらパスも直す |

Codex の割り当てに戻すときは、`~/.takt/runtime.yaml` と `config.yaml` のコメント行を戻す。`codex_cli_path` は残す。

TAKT 0.66.0 で確認した問題:

- 同梱の codex 0.153.4 は、ChatGPT アカウントで `gpt-6-luna`・`gpt-6-sol` を HTTP 400 で拒否する。
  `codex_cli_path` で 0.156.0 を指定して解消した。
- Codex の「You've hit your usage limit」はレート制限と判定されず、`rate_limit_fallback` が働かずに止まる。未解決。
- report phase でモデルがツールを呼ぶと `ReportPhaseToolCallError` で Node ごと落ちることがある。1 回観測し、再実行で成功した。

スクラッチ領域の使い捨てプロジェクトで `flash-default` を流した。
割り当てどおりのモデルで plan から final-gate まで通過した（16 分 39 秒、`Result: Success`）。
リポジトリの過去の TAKT 実行と CI の記録は [開発手順](docs/development/takt-preparation.md) にある。
初回 TP=2 準備の依頼を固定した `scripts/run-takt-preparation.sh` を、次の計測を始めるつもりで再実行しない。

[CI 設定](.github/workflows/ci.yml) は `bench` / `serving` の locked sync、pytest、ruff、mypy を実行する。
Spark 接続やモデル取得は CI に含めない。

## 背景: P1 の終了とパッチ検証

**P1 は「パッチなしでは起動できない」で締めた** (計測者の判断、2026-09-22。ADR 0005)。

**その後の進捗 (2026-09-23 JST)**: 自前ビルドで GPU 単独検査と 4 層・ダミー重みの縮小起動に成功した。
縮小起動の [実行結果](docs/results/2026-09-22-nope-build.md) に続き、実重み TP=2 の起動も成功した。
両台の IB と、日本語・英語の短い応答を確認し、ログ回収・停止まで完了。
GPU プロセスと自分たちのコンテナは両台とも 0 件。
SSE、ツール呼び出し、品質・性能、長文脈は未確認。[最新の実行結果](docs/results/2026-09-23-patched-tp2.md) を参照する。

道具 (`serving/`) と根拠つきの構成と記録の土台はできていて、実機では、公式の vLLM のイメージ
2 つ (固定した 9/9 のものと、9/22 の nightly) が、GB10 (sm_121) の sparse MLA の KV カーネルの
`pe_dim == 64` の assert で同じ場所で落ちることを再現した。通信の確認 (直結の実測、帯域、
事前の 4 段の確認) は合格で、これは P1 の成果として残る。

| 見出し | 場所 |
|---|---|
| 仕様 (要件・設計・タスク・実装の記録) | `.kiro/specs/vllm-baseline/` (`tasks.md` の `## Implementation Notes` が、タスクごとの決めごとと申し送りの正本) |
| 手順書 | `docs/vllm-baseline/procedure.md` |
| 試行の記録 / 動かなかった箇所 | `docs/vllm-baseline/attempts.md` (5 行) / `not-working.md` (件 1) |
| 判断の記録 | `docs/decisions/0002` (イメージと重み、ライセンス)、`0003` (通信)、`0004` (骨組みのまま)、`0005` (終わりの条件、打ち切り) |
| 実測の要約 | `docs/results/2026-09-22-netcheck-links.md`、`2026-09-22-netcheck-bandwidth.md` |
| 道具の使い方 | `serving/README.md` (サブコマンドの表が正) |
| 生の記録 (git の外) | `serving/var/<UTC>-<コマンド>-<構成>/` |

## リポジトリと過去の仕様

- `feat/vllm-baseline` は 2026-09-23 に private のまま `main` へ統合済み
  (`0da9a774873711b3cb711af0b1b99844c7b7b7bb`)。
  一般公開前には、Kiro のスキルのライセンス、CLAUDE.md の扱い、PLAN.md の名前、
  リポジトリ直下の LICENSE の追加を整理する
- 検証: `cd serving && uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy`
  (直近 CI: serving 1,904 件、bench 1,437 件成功・27 件 skip、ruff / mypy 通過)
- `tasks.md`: 1.1〜7.4 と 8.8 は `[x]`。8.1〜8.7 は打ち切りにより未実施。
  P1 の成功条件は未達。打ち切り判断と終了記録の照合は ADR 0005 に記録済み

## 参考: 2026-09-22 終了時点の状態（現在の状態ではない）

| 項目 | head (spark-153d, 10.0.1.60) | worker (spark-5083, 10.0.1.61) |
|---|---|---|
| `~/vllm-baseline/` | `payload/ models/ probe/ cache/ logs/ state/` (作成済み) | 同じ |
| `payload/` | 配布済み (`allreduce_bench.py`、`vllm_sanity_check.py`、`UPSTREAM.md`。古い `__pycache__/` が 1 つ残っている。害はない) | 同じ |
| イメージ | 固定 (`sha256:b0501f99…`) と nightly (`sha256:865784ba…`) の 2 つ | 固定の 1 つ |
| `probe/glm-5-3-flash-nvfp4/` | 設定とトークナイザ 8 ファイル (照合済み) | 同じ |
| `models/` | **空** (重みの本体は取得していない) | 空 |
| `state/` | `RedHatAI__GLM-5.3-Flash-NVFP4.probe.verified.json` | 同じ |
| `logs/` | NCCL の記録 (回ごとの名前。消す道はない) | 同じ |
| この道具のコンテナ | 0 | 0 |
| GPU | 空き | 空き |
| `exl3-tp2` (別のレシピ) | **止めたまま** (2026-09-21 に計測者の了承で停止。中身は見ない) | 同じ |

- **`exl3-tp2` を戻すときは** `docker start exl3-tp2` を worker → head の順で (名前だけで操作する。
  `docker inspect` / `docker logs` を向けない)。これは状態を変える操作なので、計測者の了承が要る
- Spark の上でファイルを編集しない。配布は `uv run serve push --yes` (rsync)、実行は ssh
- 状態を変える操作は、毎回、計測者の了承を取る (2026-09-22 の包括的な了承は、このセッションの
  残りタスクに限ったもの)

## 過去の調査で分かったこと

1. **GB10 では、公式の vLLM で GLM-5.3-Flash を起動できない** (2 つのイメージで同じ場所)。
   `FLASHINFER_MLA_SPARSE_SM120` が唯一の候補で、KV を `fp8_ds_mla` に強制し、そのカーネルが
   `pe_dim == 64` を要求する。このモデルは `qk_rope_head_dim = 0`。`--kv-cache-dtype` /
   `--attention-backend` では回避できない (research.md §d-1)。上流の issue #57578 / #55773 は open。
   当時の候補: [PR #55277](https://github.com/vllm-project/vllm/pull/55277)、
   [PR #55778](https://github.com/vllm-project/vllm/pull/55778)、
   [PR #53969](https://github.com/vllm-project/vllm/pull/53969)、
   [PR #54929](https://github.com/vllm-project/vllm/pull/54929)。当時の状態であり、最新状態は未照合
2. **直結は QSFP ケーブル 1 本** (PLAN.md の「2 本」は誤りだった。訂正済み)。1 つのポートが
   2 つの Linux のインターフェース / 2 つの RoCE のデバイスとして見える
3. **コンテナには `--device /dev/infiniband` が要る**。ないと NCCL が TCP に落ちて 16 Gbps。
   あると IB の経路で 1 GiB の all-reduce の busbw が **186.9 Gbps** (NVIDIA のしきい値 175 を
   上回る)。`--cap-add IPC_LOCK` / `SYS_NICE`、`NCCL_IB_HCA` は要らなかった
4. vLLM の事前の 4 段の確認 (NCCL、GLOO、vLLM の NCCL、CUDA graph の中の NCCL) は 2 台で合格
5. ベースのイメージ (`nvidia/cuda`) は NVIDIA Deep Learning Container License。方針の外だが、
   CUDA を使う限り不可避で、用途が EULA の範囲に収まるので計測者の判断で受け入れた (ADR 0002、
   `LICENSES.md`)。イメージの中の `hf` は 1.30.0
6. design.md と research.md に誤りを 2 つ見つけた (`--ulimit memlock=-1` の quote の帰属、
   flashinfer autotune の既定)。ADR 0002 の「見つかった誤りの訂正」に記録

## 再開時の選択肢と保留事項

1. **推奨: agent の試行を増やして崩れの割合を判定する（`--set agent.trials_per_stage=300`）。**
   20k だけなら約 30 分。そのあと 40k〜120k の段を足す。
2. vLLM が X925 を張り付かせる待ち受けを抑える設定を、TAKT で調べる。熱の元そのものを減らせるかを確かめる。
3. code の出力上限を上げて比べる。`--set` で変えられるかを先に確かめる。
4. concurrency と agent の入力が狙いより 5〜9% 長い原因を調べる（[#9](https://github.com/ideo-plus/DGX-Spark-GLM5.3-Flash-Recipe/issues/9)）。

別途判断する事項:

- 上流への不具合報告（vLLM、TAKT の判定漏れ）。投稿は未実施であり、計測者の指示なしに送らない。
- 一般公開前の Kiro 関連ライセンス、CLAUDE.md の扱い、PLAN.md の名前、ルート LICENSE の整理。リポジトリは private のまま。
- 公式のパッチなし P1 の成功条件は未達のまま。自前イメージでの成功を理由に、過去の打ち切りタスクを実施済みに書き換えない。

パッチの来歴・ビルド・既知の依存検査の例外は
[ビルド記録](docs/results/2026-09-22-nope-build.md) と
[調査記録](docs/vllm-baseline/patch-investigation.md) に残している。
`pip check` の 2 件は公式 nightly と同じだったが、依存全体の合格とは扱っていない。
自前イメージのビルドや重み取得、TAKT 導入は完了済みなので、最初からやり直さない。

## 道具 (`serving/`) の変更時に守る契約

- 安全の決まりは `tests/e2e/test_safety.py` が 9 つの決まりとして固定している (対象は自分の
  コンテナだけ、名前とラベルのない起動なし、前面・自動削除の起動なし、配布の宛先、許可の一覧、
  ビルドなし、トークンらしい環境変数なし、モデルカードの取得なし、本文の非保持)。設計を変える
  ときは、まずこの試験を読む
- 構成 (`serving/config/configs.toml`) の設定は 1 つ 1 つに `source`+`quote` か `measured` が
  要る。`source`/`quote`/`why` は `config-sha256` の材料に入らない。argv に効く値を変えると
  `test_config_committed.py` の固定した列と sha が落ちる (意図した見張り。更新すること)
- `serve check` は、`fetch` の構成にも `weights_verified` の関門を並べる (取得の前には必ず
  不通過に見える。`serve fetch` 自身はその関門を飛ばす)。1 台の構成で `pull-image` すると head
  にしか取得されない
- `serve netcheck ab` は env の A/B だけ。docker の設定の A/B は、構成を写して 1 つ足した
  B を `netcheck bandwidth` で流す (7.4 でそうした)
- 段 1/3/4 の構成 (`probe-nightly` は作った。`p1-nvfp4-tp2-x<連番>`、`p1-w4a16-tp2` は未作成) を
  作るときは、6.2 と同じクリーンルーム (文脈を持たない作業者、一次の資料、根拠つき) で
- 手順書・記録の文書には機械の試験がある (`test_procedure_doc.py`、`test_records_doc.py`、
  `test_readme_config_names.py`)。文書を直したら `uv run pytest` を流す
