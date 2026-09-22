# HAND_OFF — P1 (vllm-baseline) の引き継ぎ

書いた日: 2026-09-22。書いた人: Claude (cc-sdd / kiro の `/kiro-impl vllm-baseline` の自律の実行)。
進め方の更新 (2026-09-23): cc-sdd のスキルは導入しない。計測者の選択により、次の準備作業は
takt、実機操作と失敗時の判断はこの対話に分ける。先にここまでの変更を Git で統合してから、
takt の導入と作業依頼の整備へ進む。既存の仕様と実測記録は判断の根拠として使う。

書かないこと (要件 10.5): 送った内容と応答の本文、認証の情報、`exl3-tp2` の中身。

## 1. いまの状態 (一言で)

**P1 は「パッチなしでは起動できない」で締めた** (計測者の判断、2026-09-22。ADR 0005)。

**その後の進捗 (2026-09-23 JST)**: 自前ビルドで GPU 単独検査と 4 層・ダミー重みの縮小起動に成功した。
HTTP 200 を確認し、head の検証コンテナと GPU プロセスは 0 件に戻した。
実重み・TP=2・品質と性能は未検証。[実行結果](docs/results/2026-09-22-nope-build.md) に記録した。

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

## 2. リポジトリ

- 実装ブランチは `feat/vllm-baseline` (`main` から)。2026-09-23 に private のまま統合する方針。
  一般公開前には、Kiro のスキルのライセンス、CLAUDE.md の扱い、PLAN.md の名前、
  リポジトリ直下の LICENSE の追加を整理する
- 検証: `cd serving && uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy`
  (2026-09-23 時点で 1,873 件通過、ruff / mypy 通過)
- `tasks.md`: 1.1〜7.4 と 8.8 は `[x]`。8.1〜8.7 は打ち切りにより未実施。
  P1 の成功条件は未達。打ち切り判断と終了記録の照合は ADR 0005 に記録済み

## 3. DGX Spark の状態 (2026-09-22 の終わり)

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

## 4. 実機で分かったこと (要点)

1. **GB10 では、公式の vLLM で GLM-5.3-Flash を起動できない** (2 つのイメージで同じ場所)。
   `FLASHINFER_MLA_SPARSE_SM120` が唯一の候補で、KV を `fp8_ds_mla` に強制し、そのカーネルが
   `pe_dim == 64` を要求する。このモデルは `qk_rope_head_dim = 0`。`--kv-cache-dtype` /
   `--attention-backend` では回避できない (research.md §d-1)。上流の issue #57578 / #55773 は open。
   候補の PR: #55277、#55778、#53969、#54929 (どれも open。題名と状態しか見ていない)
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

## 5. 次にやること (順に)

**現在の優先順 (2026-09-23 更新)**: P1 の終了整理と、自前イメージの縮小検証は完了。次は実モデルでの検証を計画する。
調査の続きは [`docs/vllm-baseline/patch-investigation.md`](docs/vllm-baseline/patch-investigation.md) に記録する。
ビルドと縮小起動の [手順](docs/vllm-baseline/patched-build-procedure.md) と、
`experiments/nope-mla/` の補助スクリプトを用意した。Mac での差分の適用検査は合格。
`serve` は完全なローカルイメージ ID も照合できるように拡張した。
計測者から一連の実行の了承を得て、Mac でのパッチ適用と head への配布を実施した。
初回はビルド用イメージの amd64/arm64 不一致で失敗し、2 回目は実質 1 並列だったため中断した。
3 回目は実質 8 並列でビルド成功。GPU 単独検査、縮小起動、後片付けまで完了した。
イメージ ID は `sha256:9df45888d2d726a1818be1005ace819808d4a1e8b4ec01a3efa7bc7f10a40c90` (head のみ)。
経過は [実行記録](docs/results/2026-09-22-nope-build.md)、生ログは `serving/var/nope-build-0961bbae/` にある。
`pip check` の 2 件は公式 nightly でも同一で、上流の NCCL 版指定と cuSPARSELt の wheel タグに由来する。
これを依存全体の合格とは扱わず、比較とライブラリ読み込みの根拠を実行記録に残した。
以下の 1〜5 は引き継ぎ時点の候補として残す。4 の takt 導入は Git での統合後に行う。
準備するコード・構成・テスト・文書を対象とし、実機操作はこの対話で扱う。

1. **計測者が決めること (未決)**: 上流の issue #57578 に「DGX Spark (GB10) で公式イメージ 2 つで
   再現した」と報告するか (要件 10.4)。`not-working.md` 件 1 の「上流に報告するかの判断」を埋める
2. **パッチを当てたイメージでの再開** (別の仕様として起票): PR #55277 などを当てた自前の
   イメージを Spark で作り、同じ `serve probe` を流す。決めておくこと — (a) クリーンルームの
   扱い (PR の中身を読むことになる。公開時の由来の記述)、(b) 要件 8.6 の例外、(c) イメージの
   ビルドの場所 (Spark。`docker build` は道具の経路にないので、手順は別に書く)。起動できたら、
   `procedure.md` の関門 B (重みの取得 184 GiB × 2) → 段 2 から再開できる。構成 `p1-nvfp4-tp2`
   は根拠つきで用意済み (`--device /dev/infiniband` も入っている)。段 2 の最初の起動は NCCL の
   記録の 3 変数を足す (`procedure.md` 2.1)
3. **P1 の成果の要約を `docs/results/` に 1 本** (任意): 上の 4 の要点を、公開する形で
4. **takt への切り替え**: `.takt/` の導入 (`ja`)、`~/.takt/runtime.yaml` の T0 = Ollama Cloud
   など (Spark は T0 に入れない)、`LICENSES.md` に takt-workflows (Apache-2.0) の行。**先に
   `CLAUDE.md` に「このリポジトリの試験の方針の例外」を書く**: `testing-lite` の「内部構造を
   契約化しない」「完全一致文字列の不在で判断しない」は、安全の決まり (`tests/e2e/test_safety.py`)
   と根拠の固定 (`test_config_committed.py` の argv と sha、`test_procedure_doc.py`、
   `test_records_doc.py`) には当てはめない (意図して文字列で契約している)。`coding-lite` の
   「必須データへのフォールバックは REJECT」も、`cli.repo_facts` の `("unknown", True)` のような
   意図した既定値には当てはめない。実機の操作 (状態を変える操作と計測者の了承) はワークフローに
   載せず対話で行い、派生するコードの変更だけを takt に渡す
5. public に戻す前の片付け (上の 2 節)

## 6. 道具 (`serving/`) について、次の人が知っておくとよいこと

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
