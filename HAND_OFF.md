# HAND_OFF — モデル・ハーネス更新前の引き継ぎ

最終更新: 2026-09-23 JST。現在は **計測者によるモデル・ハーネス更新のための区切り**。
更新対象・更新先の版はまだ指定されていない。次の担当は勝手に更新や実機計測を始めず、更新完了と再開の指示を確認する。

## 最初に読む要約

- 自前イメージで実重み TP=2 の起動、両台の IB、日本語・英語の短い応答まで確認した。
- 07:41:27 JST に停止を完了。その後の `serve status` でも両台の作業用コンテナは absent、GPU プロセスは 0 件だった。
  これは最後の観測値であり、再開時には状態を読み直す。
- SSE、生成速度、ツール呼び出し、品質、長文脈、長時間運転の計測は **未開始**。
- 最低限の CI は導入済み。作業は [PR #7](https://github.com/ideo-plus/DGX-Spark-GLM5.3-Flash-Recipe/pull/7) までマージ済み。
  この引き継ぎ追記の基点は `main` の `8dd7d928e9c9cde89701acf92d04b92d1b893fe8`。
- TAKT の作業はすべて終了し、継続待ちの実行はない。過去の依頼をそのまま再実行する必要はない。
- cc-sdd は導入しない。準備するコード・構成・試験・文書は Mac の TAKT、実機操作と結果の判断は対話側で担当する。
- 短い応答の再確認には `serve smoke ... --max-tokens 512` を使う。既定 64 では日本語の本文が空だった。
- 本セッションでは既存の TAKT モデル割り当てを変更していない。ユーザーが更新する「モデル・ハーネス」を、推論モデルの重みや Spark のイメージの更新と決めつけない。

## 更新後の再開手順

1. 更新が終わったことを確認し、`AGENTS.md`、`CLAUDE.md` とこのファイルを読む。
   `git status` で、この引き継ぎ追記を含む未コミット差分の有無を確認する。
2. [実機の最新記録](docs/results/2026-09-23-patched-tp2.md) と
   [初回計測の手順](docs/vllm-baseline/initial-benchmark-procedure.md) を読む。
   手順や生成器の「実重みで未確認」という記述は作成時点のもの。現在の実測状態は最新記録を優先する。
3. 下記の生成済み TOML があるか確認する。これは Git 対象外なので、別の checkout には自動で付いてこない。
   なければ [生成手順](docs/vllm-baseline/patched-tp2-procedure.md) に従う。生成器は既存出力を上書きしない。
4. 実機操作は対話側で行う。全関門を再検査し、同じ構成で起動して ready と **今回分**の IB を確認する。
   `initial-benchmark-procedure.md` の smoke コマンドには `--max-tokens 512` を追加して実行する。
5. 最初の計測は `--suite decode` を明示する。既定の速度 3 種や concurrency をまとめて実行しない。
   計測後はログ回収・所有確認・停止まで行い、結果と未確認範囲を記録する。

以下は再開時のコマンドの参照であり、更新前に実行する指示ではない。作業ディレクトリはリポジトリ直下。

```bash
uv run --directory serving serve check p2-nope-tp2-smoke --configs var/nope-build-0961bbae/tp2.toml
# 全関門通過と実行条件を確認してから起動する。以後は手順書の ready・IB 確認を省かない。
uv run --directory serving serve start p2-nope-tp2-smoke --configs var/nope-build-0961bbae/tp2.toml --timeout 3h --yes
uv run --directory serving serve smoke p2-nope-tp2-smoke --configs var/nope-build-0961bbae/tp2.toml --max-tokens 512
uv run --directory bench bench run --target p2-nope-tp2-smoke --suite decode --profile quick
```

同時実行は 1、文脈長は 4096。長文脈や並列性能の結果と混同しない。
`quality` は toolcall だけを選択できず、code・needle も計画する。`quality.code_problem_limit=0` は無効。
`serving` は本文を保存しない。一方、`bench` は PLAN.md の方針に従い、合成データの本文を Git 対象外の
`results/` に保存する。公開記録には本文を含めない。

## 使用する構成と保存場所

| 項目 | 値・場所 |
|---|---|
| 構成名 / bench の対象名 | `p2-nope-tp2-smoke` |
| 生成済みの構成 | `serving/var/nope-build-0961bbae/tp2.toml`（Git 対象外） |
| イメージ ID | `sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90`（両台に存在。タグではなく ID で照合） |
| 重み | `RedHatAI/GLM-5.3-Flash-NVFP4@18d55bfd5a2194887738da73753975c9d3842f46` |
| 重みの配置 | 両台の `~/vllm-baseline/models/glm-5-3-flash-nvfp4/`。各 19 ファイル・197,881,153,655 B、照合済み |
| 構成ハッシュ | `9bd65ac3e94c8690db3861bee46c7c22fe4985d317d01995c9ee5096072d61cd` |
| API | `http://10.0.1.60:8000`、モデル名 `glm-5-3-flash`（ドットではなくハイフン） |
| 実測の要約 | `docs/results/2026-09-23-patched-tp2.md` |
| 操作の生ログ | `serving/var/nope-build-0961bbae/tp2-*.log`、`*-result.json`、`*-verdict.json` |
| 初回の応答後の回収ログ | `serving/var/20260922T215658Z-logs-p2-nope-tp2-smoke/` |
| 512 トークン再試行後の回収ログ | `serving/var/20260922T223912Z-logs-p2-nope-tp2-smoke/` |

NCCL のログは古いものも回収される。所有確認済みコンテナの hostname・開始時刻と照合する。
この環境では hostname はホスト名と同じだった。PID の再利用や同名ファイルの上書きがあるため、
過去のログに IB が書いてあるだけでは今回の合格にしない。

## TAKT と CI

使っていた TAKT は 0.66.0。固定したワークフローの導入元、使い分け、検証結果は
[開発手順](docs/development/takt-preparation.md) を参照する。
`.takt/runtime.yaml` は profile 名の割り当て、モデル・接続先の実体は `~/.takt/runtime.yaml` にある。
更新後は現在の設定を確認し、古いモデル名やセッションを前提に再開しない。
`--pipeline --skip-git` で実行しており、worktree は作っていない。今後 worktree を使う場合は `mise trust` を実行する。

| 作業 | 結果 | ログ（Git 対象外） |
|---|---|---|
| TP=2 準備 / spark-preparation | 修正検証の残件を対話側が引き取り、終了値 130。残件は修正・検証・統合済み | `.takt/verification/tp2-run.log` |
| API 調査 / research | 正常終了 | `.takt/verification/p0-research-run.log` |
| 初回計測の準備 / simple-mini | レビュー承認・正常終了 | `.takt/verification/benchmark-target-run.log` |
| smoke の出力上限 / simple-mini | レビュー承認・正常終了 | `.takt/verification/smoke-budget-run.log` |

初回 TP=2 準備の依頼を固定した `scripts/run-takt-preparation.sh` を、次の計測を始めるつもりで再実行しない。
TAKT 0.66.0 の `takt prompt` は未変更の上流ワークフローでもプレビューに失敗したが、実行自体はできた。

[CI 設定](.github/workflows/ci.yml) は `bench` / `serving` の locked sync、pytest、ruff、mypy を実行する。
直近は serving 1,904 件、bench 1,437 件成功・27 件 skip。skip を実機や隔離環境の合格とは扱わない。
Spark 接続やモデル取得は CI に含めない。


書かないこと (要件 10.5): 送った内容と応答の本文、認証の情報、`exl3-tp2` の中身。

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

1. **推奨: モデル・ハーネス更新後、準備済みの初回計測へ進む。** 更新によって準備コードの変更が必要なら、範囲を絞った依頼を TAKT に渡す。
2. **更新を保留する場合: 現在の停止状態を維持する。** 計測の再開指示があるまで、新しい実機操作を始めない。

別途判断する事項:

- 上流への不具合報告。投稿は未実施であり、計測者の指示なしに送らない。
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
