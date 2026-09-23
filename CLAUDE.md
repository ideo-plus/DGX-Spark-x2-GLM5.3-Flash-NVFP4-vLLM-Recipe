## 現在の開発手順 (2026-09-23)

準備するコード・構成・テスト・文書は、作業用 Mac の takt で扱う。
DGX Spark の読み取り、配布、イメージ・重みの取得、起動、計測、停止は対話側で担当する。
takt の作業からは、SSH、rsync、Docker、実機向けの `serve` コマンドを実行しない。
実機操作が必要になったら、必要な操作と未確認事項を報告して止める。
Git の commit・push・マージは依頼に明示された場合だけ行う。

返答と作成する文書は日本語にする。下記の cc-sdd / Kiro 手順は過去の P1 の参照として残すが、
takt の作業には適用しない。既存の `.kiro/specs/` は根拠として読み、承認済みの条件を勝手に変更しない。

## ボーイスカウトルール (タスクを始める前に毎回確認する)

作業中に見つけた汚れ (古い記述、誤り、置き場所の悪い文、重複など) は、少しずつ直してよい。

- 本題を優先する。本題の差分に混ぜず、小さく独立したコミットにする。
- 影響が大きいもの、危険なもの、「このリポジトリの試験の方針の例外」に挙げた試験の契約や実機の状態に触れるものは、その場で直さず `gh issue create` で起票する。
- コードと試験の準備が要るものは、「現在の開発手順」のとおり takt に回す。
- takt の作業では、依頼が定めた書き込み範囲を優先する。範囲外で見つけた汚れは完了報告に書くだけにする。

## このリポジトリの試験の方針の例外

`testing-lite` の「内部構造を契約化しない」「完全一致文字列の不在だけで判断しない」は、
次の意図した契約を削除・緩和する理由にはしない。

- `serving/tests/e2e/test_safety.py`: 操作対象、コマンド許可、本文の非保持などの安全条件。
- `serving/tests/unit/test_config_committed.py`: 根拠付きの argv と SHA-256 の固定。
- `serving/tests/unit/test_procedure_doc.py` と `test_records_doc.py`: 手順と実測記録の契約。
- `serving/tests/unit/test_readme_config_names.py`: 手順にある構成名の実在。

契約を変える必要があるときは、根拠と観測できる振る舞いを確認し、関連する定義と検証を同時に更新する。
試験を通すためだけに期待値を置き換えない。
`coding-lite` の「必須データへのフォールバックは REJECT」も、`cli.repo_facts` の
`("unknown", True)` のように、不明であることを明示して保守的に扱う既存の契約には適用しない。

## Fable 5 の委譲方針

Fable 5 のレート制限に早く達してしまうのを避けるため、メインセッションは要件の明確化、設計、計画、監査、レビュー、最終的な統合判断に充てること。実装の段階では、見込まれるリソースの節約が調整のオーバーヘッドを上回る限り、範囲の明確な実行タスクをサブエージェントに委譲すること。

- 境界が明確な定型の実装には Sonnet を使う
- より強い推論が必要な、複雑またはリスクの高い実装には Opus を使う
- 安全にも効率的にも委譲できない、極めて難しい作業や密結合な作業には Fable 5 を直接使う。委譲のオーバーヘッドが見込まれる節約を上回る、小さく範囲の明確なタスクはメインセッションで行う
- 委譲のプロンプトには必ず、スコープ、担当するファイル、受け入れ基準、検証手順を明記する。書き込み範囲は互いに重ならないように割り当てる
- Fable 5 のメインセッションは、差分全体のレビュー、最終的な検証の確認、統合した結果を受け入れるかどうかの判断に引き続き責任を持つ

## DGX Spark への入り方

作業用の Mac の `~/.ssh/config` に `Host spark-153d spark-5083 10.0.1.60 10.0.1.61` の定義があり、鍵認証でパスワードなしで入れる (ユーザーは `j5ik2o`)。

- head (rank 0): `ssh spark-153d` (LAN 10.0.1.60)
- worker (rank 1): `ssh spark-5083` (LAN 10.0.1.61)
- Spark 上の確認や計測は、ユーザーにコマンドを打ってもらわず、`ssh spark-153d '<command>'` で直接実行する
- 非対話で使うときは `-o BatchMode=yes -o ConnectTimeout=5` を付ける
- ホスト名が解決できないときは IP で入る (同じ設定が効く)
- 状態を変える操作 (コンテナの停止・起動、パッケージやカーネルの更新、再起動) は、実行する前に確認を取る


## どこで何をするか

リポジトリの正本は作業用の Mac に置く。DGX Spark は実行するだけの場所で、Spark の上ではファイルを編集しない。

| 作業 | 場所 |
|---|---|
| リポジトリの正本、編集、コミット | Mac |
| `bench/` の実装と試験 (試験は偽のサーバーを相手にするので、Spark は要らない) | Mac |
| 計測の実行 (`bench run`)、生データ (`results/`)、要約 | Mac |
| モデルが書いたコードの隔離の実行 | Mac |
| `serving/` の実装と試験 (試験は偽の実行役と偽の推論サーバーを相手にするので、Spark は要らない) | Mac |
| 推論サーバーの起動・停止・配布の実行 (`uv run serve …`) | Mac から。Spark へは rsync で `payload/` を配り、ssh で docker を打つ |
| 推論サーバーの起動、重み、コンテナのイメージ | Spark |
| vLLM のビルドやパッチの確認 (必要になった場合) | Spark |
| `scripts/spark-precheck.sh` (読み取りだけの独立した道具。起動・停止・配布は `serving/` に移った) | Mac で書き、ssh で Spark を読み取るだけ |

- Spark へのコードの配布は rsync で行う。Spark に GitHub の認証情報を置かない
- 計測は Mac から流す。Mac と head の間は有線で、往復は平均 1.2 ミリ秒 (2026-09-19 に実測)。`bench/` は純粋な Python なので、ネットワークの影響を切り分けたいときは、あとから head の上でも流せる
- takt は Mac で動かし、Spark をモデル提供先にも作業実行先にも使わない。実機の計測は対話側から行う

# Agentic SDLC and Spec-Driven Development

Kiro-style Spec-Driven Development on an agentic SDLC

## Project Context

### Paths
- Steering: `.kiro/steering/`
- Specs: `.kiro/specs/`

### Steering vs Specification

**Steering** (`.kiro/steering/`) - Guide AI with project-wide rules and context
**Specs** (`.kiro/specs/`) - Formalize development process for individual features

### Active Specifications
- Check `.kiro/specs/` for active specifications
- Use `/kiro-spec-status [feature-name]` to check progress

## Development Guidelines
- Think in English, generate responses in English. All Markdown content written to project files (e.g., requirements.md, design.md, tasks.md, research.md, validation reports) MUST be written in the target language configured for this specification (see spec.json.language).

## Minimal Workflow
- Phase 0 (optional): `/kiro-steering`, `/kiro-steering-custom`
- Discovery: `/kiro-discovery "idea"` — determines action path, writes brief.md + roadmap.md for multi-spec projects
- Phase 1 (Specification):
  - Single spec: `/kiro-spec-quick {feature} [--auto]` or step by step:
    - `/kiro-spec-init "description"`
    - `/kiro-spec-requirements {feature}`
    - `/kiro-validate-gap {feature}` (optional: for existing codebase)
    - `/kiro-spec-design {feature} [-y]`
    - `/kiro-validate-design {feature}` (optional: design review)
    - `/kiro-spec-tasks {feature} [-y]`
  - Multi-spec: `/kiro-spec-batch` — creates all specs from roadmap.md in parallel by dependency wave
- Phase 2 (Implementation): `/kiro-impl {feature} [tasks]`
  - Without task numbers: autonomous mode (subagent per task + independent review + final validation)
  - With task numbers: manual mode (selected tasks in main context, still reviewer-gated before completion)
  - `/kiro-validate-impl {feature}` (standalone re-validation)
- Progress check: `/kiro-spec-status {feature}` (use anytime)

## Skills Structure
Skills are located in `.claude/skills/kiro-*/SKILL.md`
- Each skill is a directory with a `SKILL.md` file
- Skills run inline with access to conversation context
- Skills may delegate parallel research to subagents for efficiency
- Additional files (templates, examples) can be added to skill directories
- `kiro-review` — task-local adversarial review protocol used by reviewer subagents
- `kiro-debug` — root-cause-first debug protocol used by debugger subagents
- `kiro-verify-completion` — fresh-evidence gate before success or completion claims
- **If there is even a 1% chance a skill applies to the current task, invoke it.** Do not skip skills because the task seems simple.

## Development Rules
- 3-phase approval workflow: Requirements → Design → Tasks → Implementation
- Human review required each phase; use `-y` only for intentional fast-track
- Keep steering current and verify alignment with `/kiro-spec-status`
- Follow the user's instructions precisely, and within that scope act autonomously: gather the necessary context and complete the requested work end-to-end in this run, asking questions only when essential information is missing or the instructions are critically ambiguous.

## Steering Configuration
- Load entire `.kiro/steering/` as project memory
- Default files: `product.md`, `tech.md`, `structure.md`
- Custom files are supported (managed via `/kiro-steering-custom`)
