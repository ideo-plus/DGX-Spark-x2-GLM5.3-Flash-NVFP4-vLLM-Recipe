# Research & Design Decisions: vllm-baseline

---
**目的**: `requirements.md` の 11 件の要件を満たす設計 (Serving Kit と Baseline Procedure) の根拠を、公式の資料と上流のソースだけから積み上げて残す。

**この調査の条件 (クリーンルーム)**:

- この文書を書いた作業者は、計測者が別に起動していた構成 (`exl3-tp2`) の中身を **一度も見ていない**。文脈を持たない新しいサブエージェントとして起動された (要件 11.4)
- 参照したのは、公式の文書 (docs.vllm.ai、recipes.vllm.ai、docs.docker.com、docs.nvidia.com)、上流の vLLM / FlashInfer / NCCL / nccl-tests のソース・issue・PR・Dockerfile、Hugging Face のモデルカードと Hub API、Docker Hub のレジストリ API だけである (要件 11.1)
- 公式の手引き (recipes.vllm.ai) と NVIDIA の手順集からは **事実だけ**を使い、コマンド行と設定ファイルは写していない。個々のフラグは、その道具の一次資料 (vLLM の CLI リファレンスとソース、Docker の CLI リファレンス、NCCL の環境変数リファレンス) から導いて出典を付けた (要件 11.2)
- 第三者のレシピ、ブログ、gist、フォーラム、DGX Spark 向けの起動スクリプトを置いたリポジトリは開いていない (要件 11.3)
- ssh も docker も実行していない。私有ネットワーク (10.x / 192.168.x / 100.x) に接続していない。ハードウェアの事実は、実装のタスクで測る
- 調査日: **2026-09-21**。vLLM の `main` は `17e50b9b76…` (2026-09-20) / 手元の浅いクローンは `9b49f923…` (2026-09-20) を参照した

**印の意味**:

- **VERIFIED** — 出典の URL と、引用した原文がある
- **【推測】** — 一次資料の事実から導いたが、そのものを述べた記述はない
- **【実機で決める】** — 値を決めるのに実測が要る。何を測れば決まるかを併記する
---

## Summary

- **Feature**: `vllm-baseline`
- **Discovery Scope**: Complex Integration
- **Key Findings**:
  1. **現行の上流 vLLM は、パッチなしでは GB10 (sm_121) で GLM-5.3-Flash を起動できない見込みが高い。** このモデルは `qk_rope_head_dim = 0` (NoPE MLA) で、sm_12x で選べる唯一の sparse MLA バックエンド `FLASHINFER_MLA_SPARSE_SM120` が KV を `fp8_ds_mla` に強制正規化し、その C++ カーネルが `pe_dim == 64` を要求して落ちる (issue #57578 / #55773)。`--kv-cache-dtype` でも `--attention-backend` でも回避できない
  2. **公式のモデル別プレビュー・イメージ `vllm/vllm-openai:glm53-flash-arm64-cu130` は実在するが、同じ assert を含む。** 中身は nightly commit `385dce36…` の別名で、その commit の `cache_kernels.cu:928` に `pe_dim must be 64 for fp8_ds_mla` がある (本調査で直接確認)
  3. したがって **P1 の現実的な着地点は「パッチなしで起動」ではなく「止まる場所と、要る最小の変更の特定」** (要件 8.5) になる可能性が高い。修正の候補は上流に 3 本 (#55277 / #53969 / #55778) あり、いずれも未マージ
  4. **要件が想定する代替の順序 (第二の重み → PP=2 → 明示の指定) は、この障害には効かない。** 障害は重みの量子化形式でも並列の取り方でもなく、アテンションの幾何 (`qk_rope_head_dim = 0`) に起因するため。順序を組み替えるべき (Decision 8)
  5. それでも **1 台での起動の確認は、重みを 1 バイトも落とさずに数分でできる**。`--load-format dummy` と `--hf-overrides` で層数を削った縮小モデルを立てれば、バックエンドの選択と `concat_and_cache_mla` の経路を同じイメージで踏める (Decision 9)。180 GiB の取得の前に、いちばんの懸念に当たれる
  6. 重み: 3 つの候補 (`zai-org/GLM-5.3-Flash` 305.8 GiB、`RedHatAI/GLM-5.3-Flash-NVFP4` 184.3 GiB、`canada-quant/GLM-5.3-Flash-W4A16-MTP` 177.7 GiB) は **すべて MIT**。ただし **brief.md の「1 台あたり約 83 GiB」は誤り**で、NVFP4 は 1 台あたり約 92 GiB になる。**第二候補のモデルカードは DGX Spark の起動レシピそのもの**で、クリーンルームの条件 (要件 11.3) では開けない (Decision 11)
  6b. **PLAN.md の「直結リンク 2 本、MTU 9000」は NVIDIA の公式資料で否定された。** 2 台の直結は **ケーブル 1 本**で、1 つの QSFP ポートが Linux には 2 つのインターフェースとして見える。NVIDIA の公表実測は 92.57 + 97.28 = **189.85 Gbps**。MTU 9000 は DGX **Station** (ConnectX-8) の話で、Spark の手順の例はすべて `mtu 1500`
  6c. **thinking は切れない。** ベンダのチャットテンプレートは生成プロンプトで無条件に `<think>` を開き、`reasoning_effort` が実際に取るのは `low` / `high` / `max` の 3 段だけ。`clear_thinking` は **過去の thinking を消すだけ**で、これから出る thinking には効かない (§g)
  7. メモリの勘定で効くのは KV ではなく **KDA (線形アテンション) の状態**。45 層のうち 34 層が線形アテンションで、状態は文脈長ではなく **同時実行数に比例**する。1 本あたり約 70 MiB/ノードで、GB10 での `--max-num-seqs` の既定は 1024 になる可能性が高い。**明示的に小さく設定しないと数十 GiB を失う** (Decision 6)
  8. KV は逆に安い。MLA の層は 11 層だけで、`fp8_ds_mla` の 1 トークンあたり 656 バイト × 11 層 ≈ 7.0 KiB/トークン。十数 GiB あれば 100 万トークン級が載る
  9. `index_kpool = 4` のため **`--block-size` は 256 が要る**見込み (バックエンドが受ける 64 と 256 のうち、`index_kpool × 32 = 128` の倍数は 256 だけ)。既定の 16 では起動時の assert に当たる
  10. イメージ内の NCCL は **2.30.7** で、2 台 TP の CUDA グラフのデッドロック (issue #52504、NCCL 2.28.9 で発生・2.30.4 で解消) の版を上回っている。この既知の落とし穴は踏まずに済む見込み

---

## Research Log

### a. イメージ — どれを使い、どう固定し、どう確かめるか

- **Context**: 要件 3.2 / 3.3 が「あとから中身が変わらない識別子での固定」と「起動時の照合」を求める。要件 11.2 が「公式の手引きのコマンドを写さない」ことを求めるので、イメージの選定は一次資料 (Docker Hub のレジストリ API、vLLM の Dockerfile、vLLM の公式ドキュメント) から行う。

- **Sources Consulted**:
  - Docker Hub レジストリ API: `https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100&ordering=last_updated` (2026-09-21 取得)
  - vLLM `docker/Dockerfile` / `docker/versions.json` — https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile
  - vLLM デプロイの文書 — https://docs.vllm.ai/en/latest/deployment/docker/
  - vLLM インストールの文書 — https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
  - Docker CLI リファレンス — https://docs.docker.com/reference/cli/docker/buildx/imagetools/inspect/ 、https://docs.docker.com/reference/cli/docker/image/pull/ 、https://docs.docker.com/reference/cli/docker/image/ls/ 、https://docs.docker.com/reference/api/engine/version/v1.50/
  - recipes.vllm.ai の GLM-5.3-Flash のページ (事実のみ) — https://recipes.vllm.ai/zai-org/GLM-5.3-Flash

- **Findings**:

  **(a-1) 候補のタグとダイジェスト (VERIFIED、2026-09-21 にレジストリ API から取得)**

  | タグ | 種別 | ダイジェスト | arch | 大きさ | 最終 push |
  |---|---|---|---|---|---|
  | `glm53-flash-arm64-cu130` | 単一アーキ | `sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5` | linux/arm64 | 9,666,567,584 B | 2026-09-09T13:31:25Z |
  | `glm53-flash` | マニフェストリスト | `sha256:819ec9c063412e5730d1b0e82046ba540d1bf991f3c4f661a849aae8a0c52374` | amd64+arm64 | — | 2026-09-09T13:32:06Z |
  | `nightly-aarch64` | 単一アーキ | `sha256:54c653f0a663ac04a92c25aba60f264c6eb203635b1ad1f627eb932ae453609b` | linux/arm64 | 9,693,815,600 B | 2026-09-20T07:19:30Z |
  | `cu134-nightly-aarch64` | 単一アーキ | `sha256:2417aec1dd6929e43a684ba65b09842cb479148d8ef2a6c9be2fd11f585de9e6` | linux/arm64 | 16,583,882,610 B | 2026-09-20T07:29:52Z |
  | `latest-aarch64` (= `v0.29.0-aarch64`) | 単一アーキ | `sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a` | linux/arm64 | 9,633,833,192 B | 2026-09-09T06:06:31Z |

  brief.md が挙げた `vllm/vllm-openai:glm53-flash-arm64-cu130` は **実在する (CONFIRMED)**。`spark` / `gb10` / `sm12` を含むタグは 1 件もない (REFUTED)。`cu130` という接尾辞の付いた汎用タグ (`cu130-nightly-aarch64`、`latest-aarch64-cu130`) は 2026-04 で更新が止まっており、現在は接尾辞なしが CUDA 13.0 系である。

  **(a-2) プレビュー・イメージの正体 (VERIFIED)**

  `glm53-flash-arm64-cu130` の image config のラベル (レジストリから取得):

  ```
  ai.vllm.build.commit        = 385dce36bcee42309924a5ece951a96db3dce7f2
  ai.vllm.image.tag           = vllm/vllm-openai:nightly-385dce36bcee42309924a5ece951a96db3dce7f2
  org.opencontainers.image.revision = 385dce36bcee42309924a5ece951a96db3dce7f2
  org.opencontainers.image.source   = https://github.com/vllm-project/vllm
  maintainer                  = NVIDIA CORPORATION <cudatools@nvidia.com>
  ```

  ENV: `CUDA_VERSION=13.0.2`、`NVARCH=sbsa`、`TORCH_CUDA_ARCH_LIST=8.0 8.7 8.9 9.0 10.0 11.0 12.0`、`VLLM_ENABLE_CUDA_COMPATIBILITY=0`、`NVIDIA_REQUIRE_CUDA` に `driver>=575`。

  つまりこのタグは **独自ビルドではなく、ある nightly の別名**である。したがって「モデル別プレビューだから GLM 向けの追加の修正が入っている」という期待は成り立たない。実際、この commit のソースを直接確認したところ:

  ```
  # https://raw.githubusercontent.com/vllm-project/vllm/385dce36bcee42309924a5ece951a96db3dce7f2/csrc/libtorch_stable/cache_kernels.cu
  928:    STD_TORCH_CHECK(pe_dim == 64, "pe_dim must be 64 for fp8_ds_mla");
  939:    STD_TORCH_CHECK(pe_dim == 64, "pe_dim must be 64 for ", kv_cache_dtype);
  ```

  同 commit の `vllm/platforms/cuda.py` L131-135 も、現行 main と同じく sm_12x の MLA 候補を 2 つに固定している。→ **このイメージは §d の障害をそのまま持つ (VERIFIED)**。

  **(a-3) `TORCH_CUDA_ARCH_LIST` に 12.1 がない (VERIFIED、含意は【推測】)**

  イメージの ENV も、リポジトリの `docker/versions.json` (`TORCH_CUDA_ARCH_LIST = 7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0`) も、**sm_121 (12.1) を列挙していない**。sm_120 向けのバイナリが sm_121 で動くか (CUDA 13 のファミリ互換) について、vLLM の公式の資料は何も述べていない (UNVERIFIED)。ただし GB10 上で他のモデルが動いた報告 (#49079、#50934、#57087 など) が多数あることから、実用上は動いていると考えられる【推測】。→ **【実機で決める】: 最初の 1 台での起動で、カーネルの起動に失敗する (`no kernel image is available for execution on the device`) かどうかを見る。**

  **(a-4) イメージの中の部品の版 (VERIFIED、`docker/versions.json`)**

  | 部品 | 版 | 意味 |
  |---|---|---|
  | CUDA | 13.0.3 (イメージ ENV は 13.0.2) | Spark の CUDA 13.0 と整合 |
  | Ubuntu | 24.04 | DGX OS と同系 |
  | Python | 3.12 | — |
  | **NCCL** | **2.30.7** | issue #52504 が「2.28.9 でデッドロック、2.30.4 で解消」とした版を上回る |
  | FlashInfer | 0.6.18.post1 | recipes.vllm.ai が「NoPE sparse MLA には 0.6.17 以降が必須」「初期化のエラーが出たら 0.6.18 以降か確かめよ」と書く条件を満たす |
  | ベースイメージ | `nvidia/cuda:13.0.3-base-ubuntu24.04` | ライセンスは §a-6 |

  NCCL 2.30.7 が入っていることは大きい。issue #52504 (2× DGX Spark、TP=2、200G RoCE) の原文は「upgrading NCCL 2.28.9 → 2.30.4 fixes the deadlock with graphs re-enabled; nothing else changed」である。**この既知の落とし穴は、選んだイメージでは最初から踏まない見込み (VERIFIED な版の事実からの【推測】)**。

  **(a-5) 固定と照合の手順 (VERIFIED)**

  - **取得せずにレジストリのダイジェストを読む**: `docker buildx imagetools inspect` の説明は **"Show details of an image in the registry."** で、出力の先頭に `Digest: sha256:…` が出る (公式の出力例あり)。`--raw` は **"Show original, unformatted JSON manifest"**、`--format` の既定は `{{.Manifest}}`。
  - **ダイジェストで固定して取得する**: `docker pull` の文書に「Pull an image by digest (immutable identifier)」という節があり、**"When pulling an image by digest, you specify exactly which version of an image to pull. Doing so, allows you to \"pin\" an image to that version, and guarantee that the image you're using is always the same."** と書かれている。注記に **"Using this feature \"pins\" an image to a specific version in time. Docker does therefore not pull updated versions of an image"**。
  - **手元のイメージが定義と合うかを確かめる (要件 3.2 / 3.3)**: `docker image inspect` の CLI リファレンス本文に `RepoDigests` の語はない (docs は SILENT)。定義は Engine API のスキーマ側にある: **"List of content-addressable digests of locally available image manifests that the image is referenced from. … These digests are usually only available if the image was either pulled from a registry, or if the image was pushed to a registry, which is when the manifest is generated and its digest calculated."** → 取得したイメージなら `RepoDigests` に `vllm/vllm-openai@sha256:…` が入る。`docker images --digests` も **"Show digests"** で同じ値を出す。
  - **落とし穴**: マニフェストリスト (`glm53-flash`) のダイジェストと、arm64 の単一マニフェスト (`glm53-flash-arm64-cu130`) のダイジェストは別物である。**単一アーキのタグのダイジェストで固定する**ほうが、照合が 1 対 1 になって確実【推測】。

  **(a-6) イメージの中の vLLM の版を知る (VERIFIED)**

  3 つの手段がある。

  1. **OCI ラベル**。`docker image inspect --format '{{index .Config.Labels "ai.vllm.build.commit"}}'` で git の commit が取れる。取得前でもレジストリから読める。Dockerfile L1312-1319 にラベルの定義がある
  2. **`vllm --version`**。`vllm/entrypoints/cli/main.py` に `parser.add_argument("-v", "--version", action="version", version=importlib.metadata.version("vllm"))` がある。ただしイメージの `ENTRYPOINT` は `["vllm", "serve"]` (Dockerfile L1336) なので、`docker run <image> --version` は `vllm serve --version` になって通らない。`--entrypoint vllm` に差し替える必要がある【推測: この帰結を述べた記述はない】
  3. **`GET /version`**。`vllm/entrypoints/serve/instrumentator/basic.py` に `@router.get("/version")` があり `{"version": VLLM_VERSION}` を返す。`bench` はこれを「前提の確認 (サーバーの版の記録)」に使う

  **(a-7) ライセンス (要件 3.8 / 3.9)**

  - **vLLM 本体**: Apache-2.0 (https://github.com/vllm-project/vllm/blob/main/LICENSE、ソース先頭にも `SPDX-License-Identifier: Apache-2.0`)。**業務で使える**
  - **ベースイメージ**: `nvidia/cuda:13.0.3-base-ubuntu24.04`。イメージに `maintainer=NVIDIA CORPORATION` が残る。**vLLM の Dockerfile も文書も、NVIDIA の CUDA ベースイメージの EULA について何も述べていない (UNVERIFIED)**。Dockerfile には `org.opencontainers.image.licenses` ラベルもない
  - vLLM の文書が述べているのは任意の依存についてだけ: **"Optional dependencies are not included in order to avoid licensing issues"**
  - → **【実機で決める / 取得時に確かめる】: 取得したイメージの中のライセンス表記 (`/NGC-DL-CONTAINER-LICENSE` など) を読んで `LICENSES.md` に記録する。** 制約 (brief.md) は「NVIDIA の NGC のコンテナは独自の EULA なので使わない」としているが、`nvidia/cuda` の公式イメージは NGC のカタログとは別物なので、実物の表記を見て判断する必要がある

- **Implications for the design**:
  - 構成の定義は **タグではなくダイジェストで持つ**。`image.ref = "vllm/vllm-openai@sha256:b0501f99…"` を正とし、タグは人間向けの注記として別項目に置く
  - 起動前の照合は `docker image inspect --format '{{index .RepoDigests 0}}'` ではなく、**`{{json .RepoDigests}}` を取って一覧に含まれるかを見る** (複数のタグから参照されていると 0 番目が期待値とは限らない)【推測】
  - `--pull never` を付けて、固定したダイジェストのイメージが手元にないときに黙って取得されないようにする。Docker の文書は `never` を **"Do not pull the image, even if it's missing, and produce an error if the image does not exist in the image cache."** と定義しており、要件 3.3 の「食い違いを示して終了する」に合う
  - 第一候補は **`glm53-flash-arm64-cu130`** (arm64 + CUDA 13.0 + FlashInfer 0.6.18.post1 + NCCL 2.30.7)。第二候補は **`nightly-aarch64`** (より新しいが、日々変わるのでダイジェストで固定する)。`cu134-nightly-aarch64` は CUDA 13.4 でドライバ要件が上がる恐れがあるので、最初は選ばない【推測】

### b. 重み — どれを、どう固定し、どこに置くか

- **Context**: 要件 3.4 / 3.5 が「入手先・版・ファイルごとの検査の値での固定」と「2 台が同じであることの確認」を求める。要件 2.6 が「Spark に認証の情報を置かない」ことを求める。

- **Sources Consulted**:
  - Hugging Face Hub API — `https://huggingface.co/api/models/{repo}?blobs=true` (2026-09-21 取得)
  - 各リポジトリの `config.json` (raw で取得)
  - recipes.vllm.ai の GLM-5.3-Flash のページ (事実のみ)
  - vLLM `vllm/transformers_utils/configs/glm5_next.py` (モデルの設定の定義)

- **Findings**:

  **(b-1) 3 つの候補の実測値 (VERIFIED、Hub API)**

  | 項目 | `zai-org/GLM-5.3-Flash` | `RedHatAI/GLM-5.3-Flash-NVFP4` | `canada-quant/GLM-5.3-Flash-W4A16-MTP` |
  |---|---|---|---|
  | ライセンス (`cardData.license` / タグ) | **mit** | **mit** | **mit** |
  | 合計の大きさ | 328,366,173,469 B (**305.8 GiB**) | 197,881,158,759 B (**184.3 GiB**) | 190,856,272,640 B (**177.7 GiB**) |
  | ファイル数 / LFS | 73 / 63 | 21 / 13 | 21 / 13 |
  | safetensors | 62 shard | 10 shard + `model_mtp.safetensors` | 9 shard + `model-mtp-00001.safetensors` |
  | main の commit (`sha`) | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | `18d55bfd5a2194887738da73753975c9d3842f46` | `f087030694ea5a8d2e7f912458633a87cad3b8cd` |
  | 最終更新 | 2026-09-07 | 2026-09-18 | 2026-09-17 |
  | 量子化 | fp8 (e4m3、`activation_scheme: dynamic`) | `compressed-tensors` / `nvfp4-pack-quantized` / `format: mixed-precision` / `kv_cache_scheme: null` | (`recipe.yaml` あり。W4A16) |
  | MTP の重み | 本体に同梱 (`num_nextn_predict_layers=1`) | 別ファイル `model_mtp.safetensors` | 別ファイル `model-mtp-00001.safetensors` |

  **3 つとも MIT** なので、ライセンス上はどれも業務で使える (要件 3.9)。recipes.vllm.ai も「about 306 GiB for the default native FP8 checkpoint」と書いており、305.8 GiB という実測と一致する。

  **(b-2) brief.md の訂正が要る点**

  - brief.md: 「`RedHatAI/GLM-5.3-Flash-NVFP4` … 1 台あたり約 83 GiB」→ **誤り**。合計 184.3 GiB なので、TP=2 で **1 台あたり約 92 GiB** (埋め込みと lm_head は両ノードに複製されるので、正確には 92 より少し多い)
  - brief.md: 「第二の候補 … 1 台あたり約 89 GiB」→ 177.7 / 2 = 88.9 GiB。**おおむね正しい**
  - brief.md: 「重みの取得は 1 つにつき 160〜180 GiB」→ NVFP4 は 184.3 GiB で、この範囲をわずかに超える
  - brief.md: 「KV に回せるのは 10〜15 GiB ほどの見込み」→ 数字としては妥当だが、**KV よりも KDA の状態のほうが効く** (§d-6)

  **(b-3) アテンションの幾何は 3 つとも同じ (VERIFIED)**

  `zai-org/GLM-5.3-Flash` と `RedHatAI/GLM-5.3-Flash-NVFP4` の `config.json` を直接読んだ結果、**どちらも同じ**:

  ```
  num_hidden_layers = 45        layer_types: linear_attention 34 / deepseek_sparse_attention 11
  hidden_size = 4096            num_attention_heads = 64
  kv_lora_rank = 512            q_lora_rank = 1536
  qk_nope_head_dim = 256        qk_rope_head_dim = 0     <-- NoPE
  v_head_dim = 256              mla_use_nope = true
  index_topk = 2048             index_kpool = 4
  index_n_heads = 32            index_head_dim = 128
  n_routed_experts = 288        num_experts_per_tok = 8   n_shared_experts = 1
  moe_intermediate_size = 2048  first_k_dense_replace = 3 (mlp: dense 3 / sparse 42)
  num_nextn_predict_layers = 1  max_position_embeddings = 1,048,576
  vocab_size = 154880           architectures = ["Glm5NextForConditionalGeneration"]
  linear_attn_config = {num_heads: 64, head_dim: 128, short_conv_kernel_size: 4, gate_lower_bound: -5.0}
  ```

  → **重みの形式を替えても `qk_rope_head_dim = 0` は変わらない。** これが §d の障害の原因なので、**第二の候補の重みに替えても障害は回避できない** (Decision 8 の根拠)。

  **(b-3b) ライセンスの所在 (VERIFIED)**

  - `zai-org/GLM-5.3-Flash`: front matter `license: mit` に加えて **`LICENSE` ファイルを同梱**。冒頭は `MIT License` / `Copyright (c) 2026 Z.AI Co., Ltd` で、標準の MIT 全文。追加条項なし
  - `RedHatAI/GLM-5.3-Flash-NVFP4`: front matter に `base_model: zai-org/GLM-5.3-Flash` と `license: mit`。**`LICENSE` ファイルは無い** (`/raw/main/LICENSE` が 404)
  - `canada-quant/GLM-5.3-Flash-W4A16-MTP`: front matter `license: mit`、カード本文にも `MIT, inherited from the base model.`。**`LICENSE` ファイルは無い**
  - → 3 つとも要件 3.9 を満たす。`LICENSES.md` には「本文の同梱はベンダのリポジトリのみ。量子化版は front matter の申告による継承」と書き添える

  **(b-4) 固定の方法 (VERIFIED)**

  - **版の固定**: Hub API の `sha` (上表) がそのリポジトリの main の commit。`hf download --revision <sha>` で固定する。**`--revision` に短縮形は使えない**。Hub の文書の原文: **"When using the commit hash, it must be the full-length hash instead of a 7-character commit hash."** 実装にも `REGEX_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")` がある
  - **⚠ CLI の名前が変わっている**: `huggingface-cli` は **1.x で動かなくなった**。移行の文書の原文: **"The deprecated `huggingface-cli` has been removed, `hf` (introduced in v0.34) replaces it with a clearer resource-action CLI."** 残っているエントリポイントは警告を出して `exit(1)` する。**使うのは `hf download` / `hf cache verify`**
  - **ファイルごとの検査の値**: `https://huggingface.co/api/models/{repo}?blobs=true` が `siblings[].lfs.sha256` を返す (本調査で全ファイル取得済み。例: `RedHatAI/GLM-5.3-Flash-NVFP4` の `model-00001-of-00010.safetensors` = 20,002,894,272 B、`sha256 = 6b6846fb90f68505d54dc143c517cfb6dc5e6e0e83d1335942027a8c54369abb`)。**ただし `?blobs=true` は HF の公式 API リファレンス (OpenAPI) に記載がない**。公式に載っているのは **`GET /api/models/{repo}/tree/{rev}?recursive=1`** で、そこの `lfs.oid` が同じ sha256 である (OpenAPI の説明: **"List the content of a repository tree, with pagination support."**)。→ **クリーンルームでは `tree` API を一次の出どころにする**
  - **`/resolve/` はポインタ本文を返さない** (ここは要注意)。`HEAD .../resolve/main/<file>` が返すのは **`x-linked-etag`** ヘッダで、その値が sha256 である。`x-repo-commit` に実際に解決された commit も入る。ポインタ本文が欲しければ `/raw/` を使う
  - **照合の道具は `hf cache verify`**。文書の原文: **"Use `hf cache verify` to validate local files against their checksums on the Hub. You can verify either a cache snapshot or a regular local directory."** / **"By default, the command warns about missing or extra files. Use flags to turn these warnings into errors"** (`--fail-on-missing-files` / `--fail-on-extra-files`) / **"If mismatches are detected, the command prints a detailed list and exits with a non-zero status."** 実装は LFS のファイルを `sha256`、それ以外を `git-sha1` で照合する。→ **要件 3.5 はこの 1 コマンドで満たせる**
  - **`hf_hub_download` 自身は sha256 を検証しない**。整合性の検査は**バイト数だけ**である (`Consistency check failed: file should be of size ...`)。落としただけでは要件 3.5 を満たさない
  - **認証**: 3 つとも公開リポジトリで、本調査は **トークンなしの匿名アクセスで全ファイルの一覧と sha256 を取得できた (VERIFIED)**。→ Spark 側にトークンを置く必要はない (要件 2.6 を満たせる)。ただし「公開リポジトリならトークン不要」と明言した文は HF の文書に見つからなかった (UNVERIFIED)。文書が述べているのは逆側だけ: **"To access private or gated repositories, you must use a token."**。なお匿名は速度制限が厳しい (レート制限の頁: 匿名は IP あたり resolver 3,000 req/5min)

  **(b-4b) 取得のやり方で気をつけること (VERIFIED、いずれも設計に直結)**

  - **中断したら最初からやり直しになる。** 1.x の移行の文書の原文: **"`resume_download`, `force_filename`, and `local_dir_use_symlinks` parameters have been removed from `hf_hub_download` and `snapshot_download`."** 現行の実装はプロセスごとに一意な一時ファイルへ落とし、失敗時は `tmp_path.unlink(missing_ok=True)` で**消す**。コメントの原文: **"On failure, do not keep a partial file around: it could not be reused anyway since the temporary name is unique to this download."** → **再開は 1 回の呼び出しの中の再試行だけ**。ただし**ファイル単位では atomic rename** なので、落とし終わったシャードは残り、コマンドを再実行すれば未完了のものだけが対象になる。184 GiB を 20 GiB のシャード 11 個で落とす形はこの性質と相性がよい
  - **`HF_HUB_ENABLE_HF_TRANSFER` は無視される。** 環境変数の文書の原文: **"This is a deprecated environment variable. … This means `hf_transfer` can't be used anymore. If you are interested in higher performance, check out the `HF_XET_HIGH_PERFORMANCE` section"**。代替は **`HF_XET_HIGH_PERFORMANCE=1`** (原文: **"Set `hf-xet` to operate with increased settings to maximize network and disk resources on the machine."**)。ただし **"For advanced users on machines with high bandwidth and at least 64 GB of RAM"** という条件付き
  - **`--max-workers` の既定は 8** (**"Maximum number of workers to use for downloading files. Default is 8."**)
  - **`--local-dir` を使う。** 既定のキャッシュは `models--org--name/snapshots/<sha>/` の下に `../../blobs/<hash>` への**相対シンボリックリンク**を張る (Hub の文書の原文: **"Symlinks use relative paths: `../../blobs/{hash}`."**)。さらに 1.x では **リポジトリのフォルダの外**を指しうる (原文: **"a Xet file downloaded through `hf_xet` is stored once at `<CACHE_DIR>/blobs/<prefix>/<xet_hash>`, and the repo's `blobs/<etag>` entry is a relative symlink to it."**)。`--local-dir` なら実体のファイルが平たく置かれる (**"The downloaded files will maintain their original file structure within the specified folder."**)
  - **`CACHEDIR.TAG` に注意。** 文書の原文: **"`huggingface_hub` automatically creates a `CACHEDIR.TAG` file inside the cache directory. This tag follows the Cache Directory Tagging Standard and tells backup tools (e.g. Borg, restic, rsync) that the directory contains re-downloadable cache data and can safely be excluded from backups."** → **既定のキャッシュを rsync で写すと、道具によっては黙って飛ばされる**。`--local-dir` を使えばこの問題も起きない
  - `--local-dir` と `--cache-dir` は**同時に指定できない** (CLI がエラーにする)

  **(b-5) どこで落とすか — Mac 経由 vs 各 Spark で直接**

  | 案 | 良い点 | 悪い点 |
  |---|---|---|
  | A. Mac に 1 度落として rsync | 落とすのは 1 回。Mac が正本。ネットワークの経路が 1 つ | Mac に 184 GiB の空きが要る。Mac→Spark 2 台で計 368 GiB を LAN に流す |
  | B. 各 Spark が公開の Hub から直接落とす | Mac のディスクを使わない。2 台が並列に落とせる | 同じファイルを 2 回落とす。Spark が外に出られることが前提 |

  公開リポジトリなので **B でも Spark に認証の情報は置かれない (VERIFIED)**。ディスクの空きは 2 台とも 2.5 TB 以上 (brief.md、計測者の実測) なので、どちらも成立する。

  → **推奨は B (各 Spark で直接取得) + Mac 側で固定したマニフェストによる照合**。理由: (i) Mac のディスクの空きに依存しない、(ii) 2 台が並列なので待ち時間が半分、(iii) 要件 3.5 の「2 台にあるファイルが固定した検査の値と合うこと」は、**どちらの案でも結局 2 台で sha256 を計算して突き合わせる**ので、照合の手間は変わらない。A は、Spark が外に出られない、または取得が繰り返し失敗する場合の代替とする。

  **(b-6) MTP の重みは省けない (VERIFIED、前の見立ての訂正)**

  `RedHatAI/GLM-5.3-Flash-NVFP4` の `model_mtp.safetensors` (7,618,560,424 B) は、**`model.safetensors.index.json` の `weight_map` から参照されている** (1,753 テンソルすべてが `model.language_model.layers.45.*`)。→ **取得を省くとロードが壊れる。** ただし GPU のメモリは食わない (§d-7)。`canada-quant` 版の `model-mtp-00001.safetensors` は 14,865,311,048 B で、**量子化されていない BF16** (`weight_scale` のテンソルが 1 つもない)。

  **(b-7) 第二候補の 2 つの問題 (VERIFIED)**

  1. **モデルカードが DGX Spark の起動レシピそのものである。** 見出しに `2× DGX Spark GB10 (SM121), TP=2, ...` / `SM121 — 2× DGX Spark GB10 (desktop, 1M context)` / `# rank1 (worker) FIRST → wait 25 s → rank0 (head):` / `## Docker images` / `## Quick start` / `## SM121 kernel-level research findings` があり、front matter のタグにも `dgx-spark` がある。→ **要件 11.3「第三者のレシピ、ブログ、フォーラムにある、起動のスクリプト、パッチ、設定を開かない」に正面から当たる。** この調査ではカード本文を読んでいない (ライセンスの行だけを確認した)。したがって**「カードが要求する vLLM の引数」は意図的に未取得**である
  2. **チャットテンプレートが古い。** 3 つの `chat_template.jinja` の sha256 を実測したところ、ベンダ版と `RedHatAI` 版は**バイト単位で一致** (`0c4099f3…`)、`canada-quant` 版だけ不一致 (`34d5ee66…`、10,644 B)。ベンダの commit `03eb5366…` (2026-08-31) 時点のもので、`690b7052…` (2026-09-04、"chat template: early break in tool result reordering check") が未取り込み。**差が出るのは複数のツール呼び出しの結果の並べ替え**で、thinking の扱いは同じ。エージェントの用途では踏みうる経路

- **Implications for the design**:
  - 構成の定義には `repo` / `revision` (40 桁の commit sha) / `manifest` (`tree` API から作った、ファイル名・大きさ・sha256 の一覧) を持つ
  - マニフェストは **Mac で Hub の `tree` API から生成してリポジトリに入れる**。照合は Spark 側で `hf cache verify --local-dir <dir> --fail-on-missing-files --fail-on-extra-files` を流す (Spark 側で「正解」を作らない)
  - 照合に失敗したファイルは **黙って取り直さない**。`bench` が公開の課題に対して取っているのと同じ方針 (bench/README.md「ハッシュが合わないファイルは、黙って取り直さない」) に揃える
  - 取得は **`hf download <repo> --revision <40 桁 sha> --local-dir <dir> --max-workers 8`**。`--local-dir` を使うのは、シンボリックリンクも `CACHEDIR.TAG` も避けられるため
  - 第一候補は **`RedHatAI/GLM-5.3-Flash-NVFP4`**。recipes.vllm.ai が NVFP4 版として名指ししており、チャットテンプレートがベンダの現行版と一致し、MTP を同梱する。第二候補の扱いは Decision 11 で決める

### c. コンテナの起動の構成 — 2 台 × 1 GPU

- **Context**: 要件 1 と 2 が「決まった順序での起動」「自分が起こしたものだけを止める」「記録の回収」を求める。ここでは **公式の資料が要求しているもの**と、**こちらの推論**を明確に分ける。

- **Sources Consulted**:
  - Docker CLI リファレンス — https://docs.docker.com/reference/cli/docker/container/run/ ほか
  - Docker ホストネットワーク — https://docs.docker.com/engine/network/drivers/host/
  - Docker の実行時の権限と capability — https://docs.docker.com/engine/containers/run/#runtime-privilege-and-linux-capabilities
  - vLLM の並列とスケーリング — https://docs.vllm.ai/en/latest/serving/parallelism_scaling/
  - vLLM の Docker デプロイ — https://docs.vllm.ai/en/latest/deployment/docker/
  - NVIDIA Container Toolkit — https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html

- **Findings**: 下表の「根拠の強さ」が要件 3.6 の「根拠」に対応する。

  | フラグ | 値 | なぜ | 根拠の強さと出典 |
  |---|---|---|---|
  | `--gpus all` | `all` | GPU をコンテナに見せる | **公式が要求**。Docker: **"GPU devices to add to the container ('all' to pass all GPUs)"**。NVIDIA: **"You can specify GPUs to the Docker CLI using either the `--gpus` option starting with Docker 19.03 or the environment variable `NVIDIA_VISIBLE_DEVICES`."** |
  | `--ipc=host` | — | vLLM の TP がプロセス間で共有メモリを使う | **公式が要求**。vLLM: **"You can either use the `ipc=host` flag or `--shm-size` flag to allow the container to access the host's shared memory. vLLM uses PyTorch, which uses shared memory to share data between processes under the hood, particularly for tensor parallel inference."** |
  | `--shm-size=16g` | 16 GiB | 既定の `/dev/shm` は 64 MiB しかない | **公式が要求**(上と同じ文)+ Docker: **"If you omit the size entirely, the system uses `64m`."**。16 GiB という値は vLLM の文書の例に出る数字【事実としての採用。値の最適化は行わない】 |
  | `--ulimit memlock=-1` | 無制限 | RDMA のメモリ登録 (`ibv_reg_mr` → `mlock`) に要る | **NVIDIA が明示的に要求**。NCCL のトラブルシュート (Docker を名指し): **"Docker containers default to limited shared and pinned memory resources. When using NCCL inside a container, please make sure to adjust the shared memory size inside the container, for example by adding the following arguments to the docker launch command line: `--shm-size=1g --ulimit memlock=-1`"**。理由も NVIDIA が書いている: **"A container does not inherit the ulimits from the host (unless running in privileged mode), and changing the ulimit value within the container is not allowed. Therefore, it is preferable to set it to unlimited by running the container with: `--ulimit memlock=-1`"**。失敗の形: **"`NCCL WARN Call to ibv_create_qp failed`"** / **"`NCCL WARN Call to ibv_reg_mr failed`"**。Docker 側は `memlock` を **"Maximum locked-in-memory address space (`RLIMIT_MEMLOCK`)"** と定義するが、**`-1` の意味は docs が SILENT** |
  | `--ulimit stack=67108864` | 64 MiB | NCCL がスタックの無制限で落ちる事例がある | **NVIDIA が推奨**。NGC の手引き: **"NVIDIA recommends the use of the following flags: `docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 ...`"**。NCCL のトラブルシュートにも **"we've seen crashes if the limit is changed to `unlimited`"** |
  | `--cap-add=IPC_LOCK` | — | (上の memlock があれば**不要**) | **付けない。** NVIDIA 自身が不要と説明している (Dynamo の RDMA/InfiniBand の頁): **"IPC_LOCK capability is not required when this setup is followed. IPC_LOCK is historically needed for RDMA because `ibv_reg_mr` calls `mlock()` to pin memory pages — but `mlock()` only needs the capability if the memlock rlimit would otherwise block it."** NVIDIA 自身の DGX Spark 向けマルチノードの手順も `IPC_LOCK` を使わず `--ulimit memlock=-1` を使っている。vLLM の文書は GPUDirect RDMA の節で `IPC_LOCK` を挙げるが、**その直後の Docker の例には自分で入れていない** |
  | `--network host` | — | 2 台をまたぐ torch.distributed と NCCL が、ホストの RoCE のインターフェースと LAN を直接使う | **【推測】 (根拠は「NVIDIA 自身がそう組んでいる」という事実のみ)**。Docker: **"the container shares the host's networking namespace … the container doesn't get its own IP-address allocated"**。**RDMA のために必要だと理由つきで述べた NVIDIA の文書は存在しない (SILENT)**。NVIDIA の DGX Spark 向けマルチノードの手順が実際に `--network host` を使っているという事実と、`--master-addr` / `VLLM_HOST_IP` にホストの NIC のアドレスを渡す必要があることが根拠。**注意**: host ネットワークでは `-p` は無視される (**"port mapping doesn't take effect, and the `-p`, `--publish`, `-P`, and `--publish-all` options are ignored"**) ので、`--port` で直接開くことになる |
  | `--device /dev/infiniband:/dev/infiniband` | — | コンテナの中から RoCE の verbs のデバイスを開く | **NVIDIA が要求 (ただし別製品の文脈)**。XLIO の Docker の頁は節の見出しが "Required Configurations" で、前置きが **"the following configurations must be specified to ensure proper functionality:"**、その 1 つが **"`--device=/dev/infiniband`: This option grants the container access to available InfiniBand devices."**。**NVIDIA Container Toolkit の文書一式はこの件に完全に沈黙 (SILENT)**。代替として `NVIDIA_MOFED=enabled` がある (リリースノート: **"Add discovery of MOFED Infiniband devices if the `NVIDIA_MOFED` environment variable of the container is set to `enabled`"**) が、環境変数のリファレンスには載っていない。**NVIDIA 自身の DGX Spark の手順の中でも扱いが割れている**: TensorRT-LLM のマルチノードの手順は `--device /dev/infiniband:/dev/infiniband` を渡すが、**vLLM の手順は渡していない** |
  | `--cap-add=SYS_NICE` | — | NCCL の cuMem 経由の共有メモリが NUMA を要る | **【条件つき】**。NCCL のトラブルシュート: **"From NCCL 2.24, if CUDA driver >= 12.6 and CUDA runtime >= 12.2, it is enabled by default in favor of `/dev/shm`. However, cuMem host allocations rely on correctly configured and working NUMA support… In particular, Docker by default disables NUMA support (it can be enabled by invoking Docker with `--cap-add SYS_NICE`). From version 2.26.5, NCCL checks if cuMem host allocations work and, if needed, automatically falls back to the `/dev/shm` code."** イメージの NCCL は 2.30.7 なので自動で落ちる。→ **最初は付けない。ログに cuMem の警告が出たら足す** |
  | `--mount type=bind,source=<weights>,target=/models/<name>,readonly` | 読み取り専用 | 重みを書き換えない (要件 3.4 の固定を守る) | **公式が要求する書式**。Docker: **"Even though there is no plan to deprecate `--volume`, usage of `--mount` is recommended."** / bind の選択肢に **"readonly, ro \| If present, causes the bind mount to be mounted into the container as read-only."** |
  | `--mount type=bind,source=<cache>,target=/root/.cache` | 書き込み可 | torch.compile / DeepGEMM / FlashInfer の JIT の結果を再起動のたびに作り直さない | **【推測】**。vLLM は DeepGEMM を JIT する (`VLLM_DEEP_GEMM_WARMUP` の説明に **"this warmup increases the engine startup time by a couple of minutes"**)。キャッシュを残せば 2 回目以降が速くなる。要件 1.4 の「決めた時間のうちに受け付けられる状態になる」に効く |
  | `--name <kit>-<config>-<role>` | 例 `vb-p1nvfp4-head` | 人が読める識別 | Docker: **"Assign a name to the container"**。**許される文字の規則は docs に書かれていない (SILENT)** ので、英数字とハイフンに限る【推測】 |
  | `--label <kit>.owner=vllm-baseline` ほか | ラベル数個 | **この道具が起こしたものだけを選ぶ** (要件 2.3 / 1.7) | **公式が要求する仕組み**。Docker: **"A label is a `key=value` pair that applies metadata to a container."** / `docker ps --filter` の `label` は **"An arbitrary string representing either a key or a key-value pair. Expressed as `<key>` or `<key>=<value>`"**。**名前での絞り込みは使わない**: **"The `name` filter matches on all or part of a container's name."** — 部分一致なので、他人のコンテナを巻き込む恐れがある |
  | `--restart no` | 既定 | P1 では落ちたら落ちたままにして、落ちた事実を記録する (自動の立ち上げ直しは P6) | Docker の表に **"`--restart` \| `no` \| Restart policy to apply when a container exits"** と既定が明示されている |
  | `--pull never` | — | 固定したダイジェストのイメージが手元にないときに黙って取得しない | Docker: **"Do not pull the image, even if it's missing, and produce an error if the image does not exist in the image cache."** |
  | `-d` (detach) | — | ssh のセッションを抜けても動き続ける | Docker: **"The `--detach` (or `-d`) flag starts a container as a background process that doesn't occupy your terminal window."** |
  | `-e` で環境変数 | `VLLM_HOST_IP`、NCCL の各変数 | §e | — |

  **記録の回収 (要件 1.9)**: `docker logs` の選択肢は **"-f, --follow \| Follow log output"**、**"--since \| Show logs since timestamp … or relative (e.g. 42m for 42 minutes)"**、**"-n, --tail \| all (default) \| Number of lines to show from the end of the logs"**、**"-t, --timestamps"**。要件 1.5 の「起動の記録の終わりの部分を示す」は `--tail` で、要件 1.9 の回収は `--timestamps` 付きの全量で行う。

  **停止 (要件 1.7)**: `docker stop` は **"The main process inside the container will receive `SIGTERM`, and after a grace period, `SIGKILL`."**、**"If no signal is configured for the container, `SIGTERM` is used as default."**、`-t` の既定は **"10 seconds for Linux containers"**。vLLM 側の SIGTERM の扱いは §f。

  **(c-2) NVIDIA 自身が DGX Spark でどう組んでいるか (事実だけ、コマンドは写さない)**

  - **NCCL のクラスタリングの手順 (2/3/4 台) はコンテナを使っていない。** ベアメタルで NCCL をソースからビルドし、`mpirun` で流している。→ **コンテナの中で RoCE を使うためのフラグの要件を、NVIDIA は DGX Spark について文書化していない (SILENT)**
  - **TensorRT-LLM のマルチノードの手順はコンテナを使い**、`--gpus`、`--network host`、`--ulimit memlock=-1`、`--ulimit stack=67108864`、`--device /dev/infiniband:/dev/infiniband` を渡している。`--cap-add=IPC_LOCK` は**入っていない**
  - **vLLM の手順 (2 台) はコンテナを使うが、`/dev/infiniband` も `memlock` も `--network` も渡していない** (単一ノードの `--shm-size=16g` のみ)。渡す環境変数は `VLLM_HOST_IP` / `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` / `UCX_NET_DEVICES` / `MASTER_ADDR` で、いずれも **QSFP 側**のインターフェースと IP
  - → **NVIDIA の中でも構成が割れている。** RoCE の verbs をコンテナの中で効かせたいなら TensorRT-LLM 側の組み方が参考になる、という事実だけを採る。どちらが正しいかは実測で決める
  - **性能計測の手引きが NGC のコンテナを勧める理由** (原文): **"DGX Spark uses a long-term supported (LTS) base software stack… To access the latest CUDA features and performance improvements, users should run NVIDIA NGC containers… which are validated for DGX Spark and include newer CUDA toolkits without modifying the host system."** → この仕様は NGC ではなく vLLM 公式のイメージを使うが、「ホストを触らずコンテナで新しい CUDA を使う」という考え方は同じ

- **Implications for the design**:
  - **ラベルが、要件 2.3 (他人のものを触らない) を構造的に保証する唯一の仕組み**である。起動時に必ず付け、停止・状態確認・ログ回収は `--filter label=…` だけで対象を選ぶ。名前は人間向けの表示にしか使わない
  - `--device /dev/infiniband` と `--network host` は **DGX Spark についての公式の裏付けがない**ので、構成の定義では「実測で決めた」という根拠 (measured) を持つ項目として扱い、最初の通信の確認 (要件 4) の結果を書き込む
  - `--cap-add=IPC_LOCK` は**入れない**。入れるなら「memlock の上限が無制限にならなかった」という実測を根拠に添える

### d. vLLM の引数 — 1 台での確認と、2 台 TP=2

- **Context**: 要件 5 (1 台での確認)、要件 6 (2 台 TP=2)、要件 8 (代替の順序) の中身。ここが P1 の成否を決める。

- **Sources Consulted**:
  - vLLM の CLI リファレンス — https://docs.vllm.ai/en/latest/cli/serve/
  - vLLM の並列とスケーリング — https://docs.vllm.ai/en/latest/serving/parallelism_scaling/
  - vLLM の main のソース (`vllm/platforms/cuda.py`、`vllm/v1/attention/backend.py`、`vllm/v1/attention/backends/mla/`、`vllm/models/glm5next/`、`vllm/config/`、`vllm/v1/worker/gpu_worker.py`、`vllm/utils/mem_utils.py`、`vllm/engine/arg_utils.py`)
  - FlashInfer のソース (`flashinfer/mla/_sparse_mla_sm120*`) — https://github.com/flashinfer-ai/flashinfer
  - vLLM の issue / PR: #57578、#55773、#53963、#55277、#53969、#55778、#54929、#57156、#57087、#41725、#51921、#52291、#52504、#54666、#50934、#47365、#49079、#54521
  - recipes.vllm.ai の GLM-5.3-Flash のページ (事実のみ)

#### d-1. いちばんの懸念 — sm_121 でのアテンションのバックエンド (VERIFIED)

`vllm/platforms/cuda.py::_get_backend_priorities` の L131-135 (main、`385dce36` でも同じ):

```python
        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]
```

`TRITON_MLA` は `is_sparse()` が偽なので、`vllm/v1/attention/backend.py` L315-319 で落ちる:

```python
        if use_sparse != cls.is_sparse():
            if use_sparse:
                invalid_reasons.append("sparse not supported")
```

→ **sm_12x で GLM-5.3-Flash が取れるバックエンドは `FLASHINFER_MLA_SPARSE_SM120` の 1 つだけで、代替はない (VERIFIED)。brief.md の「選べるバックエンドは 2 つだけで、片方は sparse に非対応」という記述は正しい。**

そのバックエンドの受け入れ条件 (`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`):

- `get_supported_head_sizes() -> [512, 576]` — コメントに **"576 = 512 NoPE + 64 RoPE (with-rope layout); 512 = 512 NoPE only (no-rope layout, qk_rope_head_dim == 0)"**。→ NoPE の 512 は **受け入れられる**
- `supported_dtypes = [torch.bfloat16]`、`supported_kv_cache_dtypes = ["auto", "fp8", "fp8_e4m3", "fp8_ds_mla"]`
- `index_topk == 2048` を要求 — **GLM-5.3-Flash の `config.json` は `index_topk: 2048` なので通る (VERIFIED)**
- `get_supported_kernel_block_sizes() -> [64, 256]`

ここまでは通る。**落ちるのはその先**である。`vllm/model_executor/layers/attention/mla_attention.py` L368-372:

```python
    if backend_name == "FLASHINFER_MLA_SPARSE_SM120" and kv_cache_dtype in (
        "auto", "fp8", "fp8_e4m3",
    ):
        return "fp8_ds_mla"
```

KV の型が **無条件に `fp8_ds_mla` に書き換えられる**。そして `csrc/libtorch_stable/cache_kernels.cu` の `concat_and_cache_mla` (main の L936-945):

```cpp
  if (kv_cache_dtype == "fp8_ds_mla") {
    STD_TORCH_CHECK(kv_lora_rank == 512, "kv_lora_rank must be 512 for fp8_ds_mla");
    STD_TORCH_CHECK(pe_dim == 64, "pe_dim must be 64 for fp8_ds_mla");
    STD_TORCH_CHECK(kv_cache.size(2) == 656 / kv_cache.element_size(), ...);
```

GLM-5.3-Flash は `qk_rope_head_dim = 0` なので `pe_dim = 0` になり、ここで落ちる。issue #57578 の原文:

```
RuntimeError: Worker failed with error 'concat_and_cache_mla,
csrc/libtorch_stable/cache_kernels.cu:939, pe_dim must be 64 for fp8_ds_mla'
```

**`--kv-cache-dtype` では回避できない (VERIFIED)**。`bfloat16` を指定すると、今度はバックエンドが全滅する (issue #53963 の原文):

```
ValueError: No valid attention backend found for cuda with AttentionSelectorConfig(
  head_size=512, dtype=torch.bfloat16, kv_cache_dtype=bfloat16, use_mla=True, use_sparse=True, ...)
Reasons: {TRITON_MLA: [sparse not supported],
          FLASHINFER_MLA_SPARSE_SM120: [kv_cache_dtype not supported, kv_cache_dtype not supported]}
```

**未解決の issue と、未マージの修正 (2026-09-21 時点、いずれも `gh` で状態を再確認済み)**:

| 番号 | 種別 | 状態 | 内容 |
|---|---|---|---|
| #57578 | Issue | **OPEN** | GLM-5.3-Flash が `pe_dim must be 64` で落ちる。報告環境は **2× GB10 (sm_121) aarch64、TP=2、RoCE、`nvidia/GLM-5.3-Flash-NVFP4`、`vllm/vllm-openai:nightly-aarch64`** |
| #55773 | Issue | **OPEN** | 同じ現象の元の報告 (2026-09-07)。4× DGX Spark (sm_121a) TP=4 の報告も付いている |
| #53963 | Issue | **OPEN** | sm_120 (RTX PRO 6000) での 4 つの失敗の形。`pe_dim` / バックエンド全滅 / 幾何の不一致 / kpool indexer の block 整合 |
| #55277 | PR | **OPEN / CONFLICTING** | **C++ を直す唯一の PR**。`pe_dim ∈ {64, 0}` に緩め、RoPE のワープが `pe_dim == 0` のときは予約領域をゼロ埋めして戻る。FlashInfer 側の GLM53_NOPE 対応 (flashinfer-ai#4802 / #4947) に依存 |
| #53969 | PR | **OPEN / CONFLICTING** | Python 側でゼロ詰めするシム。**2× DGX Spark sm_121 TP=2 で 262K 文脈まで動いた報告がある**。ただし実効幅の検証が公式のチェックポイント (`index_topk=2048`, `index_kpool=4` → 実効 2176) を全部弾く既知の欠陥があり、`--hf-overrides '{"text_config": {"index_topk": 2044}}'` で回避している |
| #55778 | PR | **OPEN** | `k_pe` を 64 要素にゼロ詰めする別案 |
| #54929 | PR | **OPEN** | sm_12x 向けの Triton の sparse MLA。本文に **"Not covered: NoPE MLA models (`qk_rope_head_dim = 0`, `head_size = 512`) such as GLM-5.3-Flash"** と明記。**この件には効かない** |

**補強の証拠 (本調査で直接確認)**: FlashInfer の `main` (2026-09-19) には `_MODEL_TYPE_GLM53_NOPE` という専用の系統があり、`_StaticFamilyEnvelope(d_qk=512, bytes_per_token=528, dedicated_heads={8,16,32,64})`、`_DECODE_GLM53_NOPE_TOPK = 2176` と定義されている。`d_qk=512` は `kv_lora_rank(512) + qk_rope_head_dim(0)` と一致し、2176 は `index_topk(2048) + index_kpool(4) 由来の tail` と整合する。一方、**イメージに入っている `v0.6.18.post1` には GLM53 の系統がなく**、`_resolve_model_type(d_qk=512)` が DSv4 に落ち、その dispatch の topk は `{128, 192, 256, 512, 1024}` までで 2048/2176 を含まない。→ **FlashInfer 側も、イメージに入っている版では GLM-5.3-Flash の sm120 の形を持っていない (VERIFIED)**。

**矛盾する報告が 1 件ある (要 実機での決着)**: issue #57087 は、**2× GB10 (sm121) TP=2 で GLM-5.3-Flash の NVFP4 が実際に動いている**前提で、同時実行時の日本語 (非 ASCII) の文字化けを報告している (`v0.29.1rc1.dev145+gbb5507741`、"stock checkout"、バックエンドは `FLASHINFER_MLA_SPARSE_SM120`)。上の分析と矛盾する。→ **【実機で決める】: 固定したイメージで 1 台の縮小起動を行い、`pe_dim` の assert が出るかどうかを最初に確かめる (§d-2)。これが P1 全体の分岐点になる。**

#### d-2. (i) 1 台での起動の確認 — 重みを落とさずに行う

要件 5.1 は「1 台の Spark だけで、メモリに収まるように条件を絞って起動を試みる」と書いているが、**NVFP4 でも 184.3 GiB あり、1 台 (121.7 GiB) には収まらない**。要件 5.5 はこの場合を想定して「そのことを記録して 2 台に進む」としているが、それでは **いちばんの懸念に安く当たる**という要件 5 の目的が果たせない。

そこで、**重みを 1 バイトも落とさずに、同じイメージで同じ経路を踏む縮小の確認**を提案する。根拠はすべて一次資料:

- `--load-format dummy` — CLI の説明に **"\"dummy\" will initialize the weights with random values, which is mainly for profiling."** (`vllm/config/load.py` L47)。重みの取得が要らない
- `--hf-overrides` — `vllm/config/model.py` L320-322 に **"If a dictionary, contains arguments to be forwarded to the Hugging Face config."**。`num_hidden_layers` と `layer_types` を短くすれば、**同じアテンションの幾何のまま**小さいモデルになる
- モデルの識別には設定だけが要る。`config.json` と `tokenizer.json` だけを落とせばよい (合わせて数十 MiB)

この確認で取りたい観察 (要件 5.2):

| 観察 | どこに出るか |
|---|---|
| 選ばれたアテンションのバックエンド | `Using %s attention backend out of potential backends: %s.` (`vllm/platforms/cuda.py` L536) |
| バックエンドが全滅した場合の理由 | `No valid attention backend found for {device} with {config}. Reasons: {reasons}` (同 L500-503、ValueError) |
| `pe_dim` の assert | `concat_and_cache_mla, …/cache_kernels.cu:NNN, pe_dim must be 64 for fp8_ds_mla` |
| MoE のバックエンド | `Using '{backend}' NvFp4 MoE backend out of potential backends: {...}.` (`vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` L228-233) |
| `block_size` の不一致 | `Glm5NextIndexerCache: kpool indexer requires cache block_size to be a multiple of index_kpool * 32 (128) …` (`vllm/models/glm5next/common/attention.py`) |
| KV の大きさ | `GPU KV cache size: N tokens, Maximum concurrency for M tokens per request: X.XXx` (`vllm/v1/core/kv_cache_utils.py` L2405-2406) |
| 利用可能な KV のメモリ | `Available KV cache memory: N GiB` (`vllm/v1/worker/gpu_worker.py` L645-648) |
| 重みのロード時間 | `Model loading took %s GiB memory and %.6f seconds` |
| 起動全体 | `init engine (profile, create kv cache, warmup model) took …` |
| vLLM の版 | `GET /version`、または起動時のバナー |

**バックエンドが選ばれた理由 (落ちたほうの理由) は `logger.debug_once` にしか出ない** (`cuda.py` L495-498) ので、この確認では `VLLM_LOGGING_LEVEL=DEBUG` を付ける【推測: 環境変数名は vLLM の慣例。実機で確認する】。

#### d-3. (ii) 2 台 TP=2 の起動 — Ray を使わない方法 (VERIFIED)

vLLM の公式の文書 (https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) の "Running vLLM with MultiProcessing" の節が、head 側 (`--nnodes 2 --node-rank 0 --master-addr <HEAD_NODE_IP>`) と worker 側 (同じに `--node-rank 1 --headless` を足す) の形を示している。2 台 × 1 GPU で TP=2 に当てはめると:

| | head (rank 0) | worker (rank 1) |
|---|---|---|
| API サーバー | 起動する (`--headless` を付けない) | 起動しない (`--headless`) |
| `--nnodes` | 2 | 2 |
| `--node-rank` | 0 | 1 |
| `--tensor-parallel-size` | **2** | **2** (両ノードで同じ値) |
| `--pipeline-parallel-size` | 1 (既定) | 1 |
| `--master-addr` | head のアドレス | **同じ head のアドレス** |
| `--master-port` | 既定 29501 | 同じ |

各フラグの help の原文 (CLI リファレンス、`vllm/config/parallel.py` の docstring と一致):

- `--nnodes, -n` — **"num of nodes for multi-node distributed inference when distributed_executor_backend is mp."** 既定 `1`
- `--node-rank, -r` — **"distributed node rank for multi-node distributed inference when distributed_executor_backend is mp."** 既定 `0`
- `--master-addr` — **"distributed master address for multi-node distributed inference when distributed_executor_backend is mp."** 既定 **`127.0.0.1`** (マルチノードでは必ず上書きが要る)
- `--master-port` — **"distributed master port …"** 既定 `29501`
- `--headless` — **"Run in headless mode. See multi-node data parallel documentation for more details."** 既定 `False`
- `--distributed-executor-backend` — **"… To use \"mp\" you must also set nnodes …"**。ただし `vllm/config/parallel.py` L981-982 に `elif current_platform.is_cuda() and self.nnodes > 1: backend = "mp"` があり、**CUDA で `nnodes > 1` なら自動的に `mp` になる**ので明示は不要

**`--headless` に `--data-parallel-*` は要らない (ソースで確定)**。`vllm/entrypoints/cli/serve.py::run_headless()` は `parallel_config.node_rank_within_dp > 0` のとき **"Run headless workers (for multi-node PP/TP)."** というコメント付きの分岐に入り、`MultiprocExecutor` を起動する。help の文言 (data parallel を参照) は誤解を招くが、純粋な TP のマルチノードで使える。

分散の初期化アドレスは `vllm/distributed/parallel_state.py` L1801-1804 で `nnodes > 1` のとき `tcp://{master_addr}:{master_port}` になる。

**⚠ TP=2 と PP=2 のどちらにするか — 資料が割れている (要 判断)**

vLLM の公式の文書は、この形 (2 台 × 1 GPU、ノード間に NVLink がない) について **PP を勧めている**。原文:

> "Multi-node multi-GPU using tensor parallel and pipeline parallel inference: if the model is too large for a single node, combine tensor parallelism with pipeline parallelism. **Set `tensor_parallel_size` to the number of GPUs per node and `pipeline_parallel_size` to the number of nodes.**"

> "**Furthermore, if the GPUs on the node do not have NVLINK interconnect (e.g. L40S), leverage pipeline parallelism instead of tensor parallelism for higher throughput and lower communication overhead.**"

これをそのまま当てると **`-tp 1 -pp 2`** になる。一方、**NVIDIA 自身の DGX Spark 向けの vLLM の手順は `--tensor-parallel-size 2` を使っている** (事実)。PLAN.md と要件も TP=2 を前提にしている。

**この仕様では TP=2 で始める**。理由: (i) 要件 6 が TP=2 と書いている、(ii) PP=2 はデコードの 1 トークンごとに 2 台を直列に通るので、1〜2 本の同時実行という運用では待ち時間が素直に足し算になる【推測】、(iii) NVIDIA 自身の Spark の手順が TP=2 である。ただし **PP=2 は代替の段に残す** (Decision 8)。TP=2 の全 reduce が直結リンクの帯域で頭打ちになるなら、PP=2 のほうが速い可能性がある。**【実機で決める】: 起動できたら、同じ `bench` の条件で TP=2 と PP=2 を測って比べる (P5 に持ち越してもよい)。**

**起動の順序 (要件 1.3)**: 公式の文書は順序を明示していない (SILENT)。head の `--master-addr` が rendezvous の待ち受けになるので、**head → worker** が素直【推測】。ただし worker が先に立って待つ形でも成立するはずなので、**どちらでも動くように、両方を起動してから両方が揃うのを待つ**設計にする。

#### d-4. モデルの名前、パーサー、`/v1/messages` に効く指定 (VERIFIED)

| フラグ | 値 | 根拠 |
|---|---|---|
| `--served-model-name glm-5-3-flash` | `/` を含まない名前を **1 つだけ** (要件 6.2) | CLI: **"The model name(s) used in the API. … The model name in the model field of a response will be the first name in this list. If not specified, the model name will be the same as the `--model` argument."** ソース `vllm/entrypoints/openai/models/serving.py` が `ModelCard(id=base_model.name, …)` を返すので `/v1/models` の `id` を決める (CONFIRMED)。**別名を複数渡さないこと**: issue #51266 (OPEN) は、複数の別名を渡すと `/v1/messages` が常に最初の別名を返し、クライアントが「モデルが変わった」と誤認して直前の `thinking` ブロックを捨てる、と報告している |
| `--tool-call-parser glm47` | GLM 系のツール呼び出しのパーサー | `vllm/tool_parsers/__init__.py` に `"glm45"` と `"glm47"` が**同じ** `glm47_moe_tool_parser` / `Glm47MoeModelToolParser` として登録されている (VERIFIED)。`glm53` という名前は存在しない。recipes.vllm.ai と `RedHatAI` のカードの両方が `glm47` を使う |
| `--reasoning-parser glm47` | 同上 | `vllm/reasoning/__init__.py` に `"glm45"` / `"glm47"` が**同じ** `glm47_moe_reasoning_parser` / `Glm47MoeParserReasoningAdapter` として登録 (VERIFIED)。**⚠ 資料が割れている**: recipes.vllm.ai は `glm47`、`RedHatAI` のモデルカードは `glm45` を指定している。vLLM 側では同じ実装に写るので**機能は同じ**。この仕様では `glm47` に揃え、割れている事実を記録する。**なお、GLM-5.3-Flash (`Glm5Next`) と `glm47` パーサーの対応を述べた記述は、vLLM の文書にもソースにも存在しない (UNVERIFIED)**。`docs/models/supported_models.md` に `Glm5Next` の行はなく、`docs/features/tool_calling.md` の GLM の節は GLM-4.5 / 4.7 までである。→ **【実機で決める】: ツール呼び出しの出力が `glm47` パーサーで正しく解釈されるかを、`bench --suite quality` のツールの課題で確かめる** |
| `--no-enable-flashinfer-autotune` | 付ける | **`RedHatAI/GLM-5.3-Flash-NVFP4` のモデルカードが、このモデルの起動に必要なフラグとして挙げている** (事実)。加えて issue #52291 が「FlashInfer の autotune がマルチノードの起動を決定的にデッドロックさせる」と報告し、#53963 の報告者も「止まるのは CUTLASS のバックエンドではなく autotune の段だ」と訂正している。`vllm/config/kernel.py`: **"If True, run FlashInfer autotuning during kernel warmup."**、最適化レベル O1 以上で既定 True。→ **代替の段ではなく最初の構成に入れる**。失うものは #52291 の報告で約 35% |
| `--enable-auto-tool-choice` | — | `vllm/entrypoints/launchers/cli_args.py` L105-107: **"Enable auto tool choice for supported models. Use `--tool-call-parser` to specify which parser to use."**。同 L455-456 に `if args.enable_auto_tool_choice and not args.tool_call_parser: raise TypeError("Error: --enable-auto-tool-choice requires --tool-call-parser")` (VERIFIED)。`bench` の前提 (bench/README.md「ツール呼び出しを測るなら、自動のツール選択と、モデルに合ったパーサーが有効」) に対応 |
| `--enable-prompt-tokens-details` | — | `cli_args.py` L139-140: **"If set to True, enable `prompt_tokens_details` in usage."**。`vllm/entrypoints/anthropic/serving.py` L81-89 が `usage.prompt_tokens_details.cached_tokens` を `cache_read_input_tokens` に写すので、**これを付けないと `/v1/messages` にキャッシュの内訳が出ない (VERIFIED)**。要件 9.1 とbench の前提に直結 |

#### d-5. 既定で入っているもの (VERIFIED)

- **プレフィックスキャッシュ**: `vllm/config/cache.py` L130 が `enable_prefix_caching: bool = True`。CLI の既定は `None` だが、`EngineArgs._set_default_chunked_prefill_and_prefix_caching_args()` が `model_config.is_prefix_caching_supported` から決め、hybrid の生成モデル (GLM-5.3-Flash は KDA + sparse MLA の hybrid) では **True**。→ **明示の指定は不要**
- **チャンク化した prefill**: `vllm/config/scheduler.py` L126 が `enable_chunked_prefill: bool = True`、同じ経路で生成モデルは **True**。→ **明示の指定は不要**
- 対応する `/metrics` の名前: ソースの宣言は `vllm:prefix_cache_queries` / `vllm:prefix_cache_hits` (`vllm/v1/metrics/loggers.py` L600/L611) だが、**Prometheus のカウンターは公開時に `_total` が付く**ので、`/metrics` に出るのは **`vllm:prefix_cache_queries_total`** と **`vllm:prefix_cache_hits_total`** である。vLLM の設計の文書の原文: **"When exposing the time series for counter, a `_total` suffix will be added."** `bench/config/targets.toml` の `metric_map` の注意書き (「カウンターは `_total` を付けて書く」) と同じ扱い。**`vllm:gpu_prefix_cache_hit_rate` は存在しない** — 設計の文書に **"we now expose 'queries' and 'hits' counters rather than a 'hit rate' gauge"** とある。ゲージは `vllm:kv_cache_usage_perc` (同 L576)

#### d-6. メモリの勘定 (GB10 のユニファイドメモリ)

**`--gpu-memory-utilization` が何を意味するか (VERIFIED、ソース)**

CLI の説明は **"The fraction of GPU memory to be used for the model executor … If unspecified, will use the default value of 0.92."**。**ユニファイドメモリについて vLLM の公式ドキュメントは何も述べていない (UNVERIFIED)** が、**ソースには DGX Spark を名指しした扱いがある**。`vllm/utils/mem_utils.py` の `MemorySnapshot.measure()`:

```python
        self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)
        if current_platform.is_integrated_gpu(device.index):
            # On UMA (Unified Memory Architecture) platforms where CPU and
            # GPU share physical memory (e.g. GH200, DGX Spark, Jetson Orin),
            # cudaMemGetInfo underreports free memory because it does not
            # account for reclaimable OS memory (page cache, buffers).
            # Use psutil to get the true available memory.
            self.free_memory = psutil.virtual_memory().available
```

同ファイルには `release_device_memory_under_pressure()` (`_UMA_PRESSURE_THRESHOLD = 0.8`) もあり、UMA でシステムのメモリが逼迫したらキャッシュを OS に返す。

予算の決まり方 (`vllm/v1/worker/utils.py::request_memory` と `vllm/v1/worker/gpu_worker.py::determine_available_memory`):

```
requested_memory      = ceil(total_memory * gpu_memory_utilization)
available_kv_cache    = requested_memory - non_kv_cache_memory - cudagraph_memory_estimate
```

起動時の検査 (同 L541-550) — **free (= psutil の available) が requested を下回ると起動しない**:

```
Free memory on device (…/… GiB) on startup is less than desired GPU memory utilization (…, … GiB).
Decrease GPU memory utilization or reduce GPU memory used by other processes.
```

→ **GB10 では `total_memory` が 121.7 GiB (OS と共有する全体) を指す**。`0.92` なら 112.0 GiB を要求し、OS が使っている分を差し引いた `psutil.available` がそれを上回っている必要がある。**0.92 のままだと、重みを rsync した直後などにページキャッシュの状態次第で起動に失敗しうる**【推測】。

**KV の 1 トークンあたりの大きさ (計算、要素は VERIFIED)**

- MLA の層は 45 層中 **11 層**だけ (`layer_types` の実数)
- MLA の KV は TP で分割されない。`vllm/config/model.py::get_num_kv_heads` に **"When using MLA during decode it becomes MQA" → `return 1`** とあり、`tensor_parallel_size` で割られない (VERIFIED)。→ **2 台とも同じ量を持つ**
- `fp8_ds_mla` の 1 エントリは **656 バイト** (`cache_kernels.cu` の `kv_cache.size(2) == 656 / element_size()`)。FlashInfer 側の GLM53_NOPE は `bytes_per_token=528` (= 656 − 128 の RoPE 領域) としており、現行の 656 バイトのレイアウトでは **RoPE の 128 バイトが使われないまま確保される**
- → **1 トークンあたり 11 × 656 = 7,216 B ≈ 7.05 KiB (ノードあたり)**
- これに kpool の indexer のキャッシュ (`index_head_dim=128` を 4 トークンに 1 つへ圧縮) と tail のキャッシュが加わる。正確な値は **【実機で決める】: 起動ログの `GPU KV cache size: N tokens` と `Available KV cache memory: N GiB` の比から逆算する**

  → 16 GiB を KV に回せれば **約 230 万トークン**。`--max-model-len 163840` でも **同時 14 本分**が載る計算になる。**KV は制約にならない。**

**本当の制約は KDA (線形アテンション) の状態 — これが最大の発見**

- KDA の層は **34 層**。状態の形は `MambaStateShapeCalculator.kda_state_shape(tp_world_size, num_heads=64, head_dim=128, conv_kernel_size=4, num_spec)` (`vllm/model_executor/layers/mamba/mamba_utils.py` L303-326):
  - `recurrent_state_shape = (num_heads / tp, head_dim, head_dim)` = (32, 128, 128)
  - `conv_dim = num_heads*head_dim + 2*(num_k_heads*head_k_dim)` = 24,576 → `/tp` = 12,288、幅は `conv_kernel_size - 1 + num_spec` = 3
- 型: `kda_state_dtype()` は `mamba_ssm_cache_dtype == "auto"` のとき **`recurrent_state_dtype = torch.float32`** (同 L140-141)、conv はモデルの dtype (bf16)
- → **1 本あたり、1 ノードあたり**: 再帰の状態 34 × 32×128×128 × 4 B = 71.3 MB、conv 34 × 12,288×3 × 2 B = 2.5 MB → **合計 約 70 MiB/本**
- **`--max-num-seqs` の既定**: `vllm/engine/arg_utils.py` L2746-2775 は、`device_memory >= 70 GiB` かつデバイス名に `a100` を含まないとき **`default_max_num_seqs = 1024`** を選ぶ。GB10 の 121.7 GiB はこの条件に当たる可能性が高い【推測: `get_device_total_memory()` が UMA で何を返すかは実機で確認】
- → **既定のままだと KDA の状態だけで 1024 × 70 MiB ≈ 70 GiB を要求しかねない。**

  **この運用が想定する同時実行は 1〜2 本** (PLAN.md「同時に走らせるエージェントは 1〜2 本」) なので、**`--max-num-seqs` を明示的に小さくする (16 本 → 約 1.1 GiB、8 本 → 約 0.55 GiB)**。これは P1 の構成で最も効く 1 行になる。

  **補足**: プレフィックスキャッシュが有効だと `mamba_cache_mode` は `"align"` になる (`vllm/config/cache.py` L178-186: **"This is the default when prefix caching is enabled."**) ので、境界ごとの状態も確保される。実際の消費は **【実機で決める】: 起動ログの内訳 (`Actual usage is … for weight, … for peak activation, …`) を読む。**

**まとめ: 最初に試す値と、その根拠**

| 引数 | 値 | 根拠 |
|---|---|---|
| `--gpu-memory-utilization` | **0.90** | 既定 0.92 より控えめ。UMA では `total_memory` が OS と共有する全体を指し、`psutil.available` が要求を下回ると起動が失敗する (ソース) ため、OS 側に余裕を残す【推測】。**【実機で決める】: 起動できたら 0.92 → 0.94 と上げて `Available KV cache memory` の増分を測る** |
| `--max-num-seqs` | **16** | KDA の状態が同時実行数に比例し、1 本あたり約 70 MiB/ノード (上の計算)。運用の想定は 1〜2 本、`bench` の参考計測が 8 本まで。16 なら約 1.1 GiB で足りる |
| `--max-num-batched-tokens` | **2048** | チャンク化した prefill の 1 回分。小さいほど活性化のピークが下がる。`bench` の prefill は 128k を送るので、チャンクの回数は増えるが、メモリが先に効く【推測】。**【実機で決める】: prefill の速度と引き換えなので A/B する** |
| `--max-model-len` | **163840** | `bench` の prefill が 128k トークン、`agent` が 12 万トークンまで伸ばす (bench/README.md)。入力 128k + 出力の余裕を見て 160k。モデルの上限は 1,048,576 なので余裕はある |
| `--block-size` | **256** | `Glm5NextIndexerCache` が `block_size % (index_kpool * 32) == 0` を要求し (`vllm/models/glm5next/common/attention.py`)、`index_kpool = 4` なので 128 の倍数。バックエンドが受けるのは `[64, 256]` なので **256 のみが両方を満たす**。既定は 16 (`CacheConfig.DEFAULT_BLOCK_SIZE`) なので明示が要る。**【実機で決める】: まず指定せずに起動し、vLLM が自動で 256 を選ぶかを見る。assert が出たら 256 を明示する** |
| `--kv-cache-dtype` | **指定しない (`auto`)** | `FLASHINFER_MLA_SPARSE_SM120` では `auto` / `fp8` / `fp8_e4m3` がどれも `fp8_ds_mla` に正規化される (ソース)。指定しても結果は同じで、指定すると「効いている」と誤解する |
| `--language-model-only` | **付ける** | `vllm/config/multimodal.py` L122-124: **"If True, disables all multimodal inputs by setting all modality limits to 0."**。このモデルは `Glm5NextForConditionalGeneration` (画像・動画あり) だが、takt の用途は文章だけ。マルチモーダルの profiling のぶんのメモリを使わずに済む |
| `--shutdown-timeout` | **60** | 既定は 0 で、`vllm/config/vllm.py` L452-456 に **"Shutdown grace period for in-flight requests. Shutdown will be delayed for up to this amount of time … Any remaining requests are aborted once the timeout is reached."**。0 = 実行中の要求を即座に捨てる。`bench` の途中で止めたときに記録が壊れないように猶予を入れる |
| 投機的デコード | **付けない** | 次節 |

#### d-7. 投機的デコードを確実に切る (VERIFIED、要件 6.7)

`vllm/engine/arg_utils.py::create_speculative_config` は、`--speculative-config` / `--spec-method` / `--spec-model` / `--spec-tokens` のいずれも指定がなければ **`return None`** する。→ **MTP の重みを持つチェックポイントでも、投機的デコードは自動では有効にならない (VERIFIED)**。

- **切る方法**: 上の 4 つを一切渡さない
- **切れていることの確かめ方**: `/metrics` に `vllm:spec_decode_num_drafts`、`vllm:spec_decode_num_draft_tokens`、`vllm:spec_decode_num_accepted_tokens` (`vllm/v1/spec_decode/metrics.py` L229-231) が**現れないこと**。起動時の設定のダンプに `SpeculativeConfig(...)` が出ないこと
- **MTP の重みが読まれるか**: NVFP4 / W4A16 のどちらも MTP は**別ファイル**で、`Glm5NextMTPModel` は `vllm_config.speculative_config.draft_model_config` から作られる (`vllm/models/glm5next/common/mtp.py` L42-43: `assert vllm_config.speculative_config is not None`)。→ **投機的デコードを切っていれば MTP は読まれない【推測】。【実機で決める】: `Model loading took N GiB` が重みの大きさと合うかで確かめる**

#### d-8. 代替の順序 (要件 8.1 の見直し)

要件 8.1 は「第一の候補の重みでの TP=2 → 第二の候補の重みでの TP=2 → 2 台での層の分割 → 明示の指定の追加」と定めているが、**§d-1 の分析では、この順序のどれも `pe_dim` の障害には効かない**:

- **第二の重み**: `qk_rope_head_dim = 0` は量子化の形式に依らない (§b-3 で 2 つのチェックポイントを実測確認)
- **PP=2**: アテンションのバックエンドの選択は層ごとで、並列の取り方に依らない
- **明示の指定**: `--attention-backend` で選べるものが `FLASHINFER_MLA_SPARSE_SM120` しかない。`--kv-cache-dtype` は正規化で潰される

そこで、**根拠のある順序に組み替える** (Decision 8):

| 段 | 何を試すか | 何が分かるか | 止める条件 |
|---|---|---|---|
| 0 | 固定したイメージで、1 台・縮小・`--load-format dummy` の起動 (§d-2) | `pe_dim` の assert が出るか / バックエンドが選ばれるか。**重みの取得の前に分かる** | assert が出たら段 1 へ |
| 1 | `nightly-aarch64` (より新しい commit) で段 0 を繰り返す | 上流の修正が入ったか | 出なければ段 2 へ |
| 2 | 出なければ、重みを取得して 2 台 TP=2 (第一の重み) | 本番の構成 | 起動できれば `bench` へ |
| 3 | 起動後に不具合が出た場合のみ、明示の指定を足す (`--enforce-eager`、`--moe-backend`、`--no-enable-flashinfer-autotune`、`--block-size 256`) | 何が原因か | 段 4 |
| 4 | 第二の重み (W4A16) で 2 台 TP=2。**進む前に計測者に確認** (Decision 11) | **MoE のカーネル**の問題を切り分ける (NVFP4 特有の #54666 / #49079 / #47365 を避ける)。アテンションの問題には効かない。チャットテンプレートが古いので `--chat-template` でベンダの現行版を指す | 段 5 |
| 5 | 打ち切り。止まった場所と、要る最小の変更 (#55277 または #53969) を記録して計測者に判断を仰ぐ (要件 8.5) | — | — |

**PP=2 の位置づけを変える。** `pe_dim` の障害に対しては効かない (同じアテンションの経路を通る) ので、**障害の切り分けの段からは外す**。一方、**起動できた後の性能の選択肢としては有力**である。vLLM の公式の文書がこの形 (ノード間に NVLink がない) に PP を勧めているため (§d-3 の引用)。→ **段 3 の後に「段 3.5: 起動できた構成で TP=2 と PP=2 を測って比べる」を置く。** これは要件 8 の「代替」ではなく、要件 4.5 と同じ A/B の扱いにする (P5 に持ち越してもよい)。

#### d-9. 起動できた場合に備える、明示の指定の候補 (要件 8.4)

いずれも **A/B で効果を測ってから採用する**。原則、最初の構成には入れない。

| 指定 | いつ足すか | 根拠 | 失うもの |
|---|---|---|---|
| `--enforce-eager` | CUDA グラフでの NaN (#57156) や、グラフ内の NCCL のデッドロック (#52504) を疑うとき | CLI: **"If True, we will disable CUDA graph and always execute the model in eager mode."** / **"NOTE: This disables both torch.compile and CUDA graphs"** | 大きい。#57156 の報告では 32.6 → 17.2 tok/s |
| `--no-enable-flashinfer-autotune` | 起動が autotune の段で止まるとき (#52291: マルチノードの autotune が決定的にデッドロックする) | `vllm/config/kernel.py`: **"If True, run FlashInfer autotuning during kernel warmup."**。最適化レベル O1 以上で既定 True | カーネルが未調整になる。#52291 の報告では約 35% の低下 |
| `--moe-backend b12x` / `flashinfer_b12x` | 自動選択で Marlin になり、不正なメモリアクセスや崩れた出力が出るとき (#54666) | `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` L190-203 のコメント: **"FLASHINFER_B12X is intentionally excluded from auto-selection … use moe_backend=\"flashinfer_b12x\" to opt in explicitly."** | #49079 は `flashinfer_b12x` で活性化のピークが倍増し KV が約 55% 減ったと報告 |
| `--block-size 256` | kpool の assert が出たとき | §d-6 | なし (むしろ必要) |
| `VLLM_USE_DEEP_GEMM=0` | — | **候補から外す**。`vllm/models/glm5next/nvidia/sparse_indexer.py` に `if current_platform.is_cuda() and not has_deep_gemm(): raise RuntimeError("Sparse Attention Indexer CUDA op requires DeepGEMM to be installed.")` があり、**このモデルでは DeepGEMM が必須 (VERIFIED)**。切ると起動しない |
| `--disable-custom-all-reduce` | — | **候補から外す**。#41725 の報告に **"no-op on a 2-node group. Custom AR is already auto-disabled cross-node"** とある |

### e. NCCL と 2 台の間の通信

- **Context**: 要件 4 が「最小の設定から始め、A/B で良くなったものだけを採用する」ことを求める。

- **Sources Consulted**:
  - NCCL 環境変数リファレンス — https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
  - DGX Spark ユーザーガイド (ConnectX-7 Networking) — https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html
  - DGX Spark 移植ガイド (CUDA) — https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html
  - vLLM の並列とスケーリング / トラブルシュート — https://docs.vllm.ai/en/latest/serving/parallelism_scaling/ 、https://docs.vllm.ai/en/latest/usage/troubleshooting/
  - nccl-tests の README — https://github.com/NVIDIA/nccl-tests

- **Findings**:

  **(e-1) DGX Spark のネットワーク — PLAN.md の記述は公式資料で否定された (VERIFIED)**

  NVIDIA の DGX Spark ユーザーガイド (ConnectX-7 Networking) の原文:

  > "Each DGX Spark has two QSFP ports (sometimes called 'ConnectX-7 ports') on the back of the device. Each port provides up to 200 Gigabits per second (Gb/s), but the incoming speed is also determined by the cable that you use."
  >
  > "The NIC connects independently to the two external QSFP ports, and it connects to the SoC through two independent PCIe Gen 5 x4 links."
  >
  > "each QSFP port has two PCIe addresses to account for the two PCIe x4 links out of the NIC into the SoC."
  >
  > "**Each QSFP port appears as two independent Linux Ethernet interfaces. As a result, plugging in two cables shows a total of four Linux Ethernet interfaces.**"
  >
  > "Each Ethernet interface has a corresponding RoCE interface (typically called a 'RoCE device') for InfiniBand communication."

  公式のインターフェースの対応表:

  | QSFP ポート | PCIe アドレス | Ethernet | RoCE |
  |---|---|---|---|
  | 左 | `p1s0f0` | `enp1s0f0np0` | `rocep1s0f0` |
  | 左 | `P2p1s0f0` | `enP2p1s0f0np0` | `roceP2p1s0f0` |
  | 右 | `p1s0f1` | `enp1s0f1np1` | `rocep1s0f1` |
  | 右 | `P2p1s0f1` | `enP2p1s0f1np1` | `roceP2p1s0f1` |

  **ケーブルは 1 本**。NVIDIA Sync のクラスタアシスタントの原文: **"Two Sparks. Use one cable for a direct connection, or use one cable per device through a switch (two cables total)."** NVIDIA の 2 台接続の手順の前提条件にも **"One QSFP cable for direct 200GbE connection between two devices"** とあり、手順の本文に **"Connect the QSFP cable between both DGX Spark systems using any QSFP interface on each device. Make sure to use the same physical port on each device to prevent issues with NCCL tests."**。さらに複数の手順に共通の注記: **"Full bandwidth can be achieved with just one QSFP cable. When two QSFP cables are connected, all four interfaces must be assigned IP addresses to obtain full bandwidth."**

  **公式の実測値は約 190 Gbps**。NVIDIA の性能計測の手引きの原文: **"Total throughput = 92.57 + 97.28 = 189.85 Gbps"**。測り方は `ib_write_bw`、メッセージ 65536 B、デバイスは `rocep1s0f0` と `roceP2p1s0f0` の 2 つ (= **左ポート 1 本のケーブルの 2 つの PCIe function**)、別々のサブネットを割り当てる (**"Assign unique subnets to each active port."**)。

  → **PLAN.md の訂正 (要件 4.2)**:

  | PLAN.md の記述 | 判定 | 正しい記述 |
  |---|---|---|
  | 「直結リンク 2 本」 | **誤り** | ケーブルは 1 本。1 つの QSFP ポートが 2 つの PCIe function を持ち、Linux には 2 つのインターフェースとして見える |
  | `rocep1s0f0` と `roceP2p1s0f0` が見える | **正しい** | ただし理由は「ケーブル 2 本」ではなく「1 ポート = PCIe x4 リンク 2 本」 |
  | 「1 本あたり約 112Gb/s」 | **出典不明** | NVIDIA の公表実測は 92.57 + 97.28 = 189.85 Gbps。112 という数字は NVIDIA のどの資料にもない |
  | 「MTU 9000」 | **公式の裏づけなし** | NVIDIA は DGX Spark の MTU について何も書いていない (SILENT)。Spark の手順の `ip addr show` の例はすべて **`mtu 1500`**。**MTU 9000 は DGX Station (ConnectX-8) の手順のもの**で、そこには「2 本のケーブル、レールごとに 1 本」「GPUDirect」という別の前提がある。**この 2 つを混ぜないこと** |
  | `ethtool` が 200000Mb/s と出る | 予想される | NVIDIA の手順の例では、**同じ物理ポートの 2 つの論理インターフェースがどちらも 200000Mb/s と報告する**。足して 400G にはならない |

  **【実機で決める】(要件 4.1)**: `ip -br link` で名前・状態・MTU、`ethtool <if> | grep Speed` でネゴした速度、`ibv_devinfo` と `ibdev2netdev` で RDMA デバイスと netdev の対応を読む。リンクが上がっているインターフェースが 2 つなら「ケーブル 1 本」、4 つなら「2 本」である。

  **GB10 の公称値 (VERIFIED)**: **"128 GB LPDDR5x unified system memory, 256-bit interface, 4266 MHz, 273 GB/s bandwidth"**、CUDA コア 6,144。**SM の数は公式には非公開 (SILENT)**。compute capability は **12.1 (sm_121)** で、出典は 2 つ — `developer.nvidia.com/cuda-gpus` の表の **"12.1 | NVIDIA GB10 (DGX Spark)"** と、移植ガイドの **"For DGX Spark, this is 121-real."**。PLAN.md の「ユニファイドメモリ 121.7GiB」は使える量の実測で、公称 128 GB と矛盾しない。

  **(e-2) GPUDirect RDMA は DGX Spark では非対応 (VERIFIED)**

  NVIDIA の DGX Spark 移植ガイド (CUDA) の記述:

  > "GPUDirect RDMA technology is not supported on DGX Spark, and the mechanisms for direct I/O based on that technology, for example nvidia-peermem (for DOCA-Host), dma-buf or GDRCopy, do not work."

  理由も同ガイドが述べている: Grace Blackwell のユニファイドメモリでは、ピン留めしたデバイスメモリを CPU 複合体や PCIe のペリフェラル (NIC) からコヒーレントにアクセスできない。

  **含意**: NCCL は GDR を使えず、ホスト経由の段取りに落ちる。vLLM の文書は **"If you find `[send] via NET/IB/GDRDMA` in the logs, then NCCL is using InfiniBand with GPUDirect RDMA, which is efficient. If you find `[send] via NET/Socket` in the logs, NCCL used a raw TCP socket, which is not efficient for cross-node tensor parallelism."** と書いているが、**DGX Spark では前者は出ない**。見るべきは **`NET/IB` が使われていて `NET/Socket` でないこと**【推測: NVIDIA の非対応の記述と vLLM の記述の組み合わせ】。

  **(e-3) 最小の設定 — NVIDIA 自身の 2 台の手順は、驚くほど少ない (VERIFIED)**

  NVIDIA の NCCL の手順 (2/3/4 台に対応) の起動スクリプトのヘッダのコメントの原文:

  > "`--topology  direct (2 nodes, cable), ring (3 nodes), or switch (4 nodes).` … `direct and switch add nothing extra.`"
  >
  > "**NCCL bootstrap runs over the management interface (enP7s7); the collective data auto-routes over the CX-7 RoCE ports (NCCL discovers them — no need to name them).** Override the management interface with `MGMT_IFNAME=<iface>`."

  **2 台の直結のときに NVIDIA が渡す変数は 3 つだけで、そのすべてが管理インターフェース (10GbE RJ-45、`enP7s7`) を指している**: `NCCL_SOCKET_IFNAME` / `UCX_NET_DEVICES` / `OMPI_MCA_btl_tcp_if_include`。

  ここから読み取れる 2 つの重要な事実:

  1. **`NCCL_IB_HCA` を設定していない。** 理由もコメントに書かれている (**"NCCL discovers them — no need to name them"**)。4 台のスイッチ経由のときだけ 4 デバイスを列挙している
  2. **`NCCL_SOCKET_IFNAME` は QSFP ではなく管理インターフェース。** `NCCL_SOCKET_IFNAME` が決めるのは**ブートストラップと制御の経路**であって、データの経路ではない。データは IB/RoCE のトランスポートが自分で選ぶ

  **⚠ ただし NVIDIA の中で割れている**: NVIDIA 自身の **vLLM の手順 (2 台)** は、`NCCL_SOCKET_IFNAME` も `GLOO_SOCKET_IFNAME` も `VLLM_HOST_IP` も `MASTER_ADDR` も、**QSFP 側**のインターフェースと IP に設定している。NCCL の手順 (管理側) と正反対である。→ **どちらが速いかは A/B で決める (要件 4.5)。**

  この仕様の**最初の設定**は次のとおり。

  | 変数 | 値 | なぜ | 出典と原文 |
  |---|---|---|---|
  | `VLLM_HOST_IP` | 各ノードの**直結リンク側**のアドレス | ノードごとに違う値。未設定だと `8.8.8.8:80` への UDP connect で既定の経路の NIC を推測してしまう (`vllm/utils/network_utils.py::get_ip`)。**DGX Spark はケーブル 1 本で 4 つの IF が現れる**ので、この推測は外れやすい | `vllm/envs.py` L710-714: **"used in distributed environment to determine the ip address of the current node, when the node has multiple network interfaces. If you are using multi-node inference, you should set this differently on each node."** / vLLM の文書: **"For security, set `VLLM_HOST_IP` to an address on a private network segment. Traffic sent over this network is unencrypted…"** |
  | `--master-addr` | head の**直結リンク側**のアドレス | rendezvous とノード内キューの接続先が同じ経路に載る | `vllm/distributed/parallel_state.py` L1801-1804 (`tcp://{master_addr}:{master_port}`) |
  | `NCCL_SOCKET_IFNAME` | **`=` 付きで直結リンクの IF 名** (A 案) または **管理 IF 名** (B 案) | NCCL の**ブートストラップと制御**の経路を固定する。自動検出に任せると、上がっているが通らない IF を掴んで初期化で止まる | NCCL: **"Define to a list of prefixes to filter interfaces to be used by NCCL."** / **"`=eth0`: Use only interface `eth0`"** / **"It is therefore always recommended to add the `=` prefix to ensure an exact match."**。NCCL のトラブルシュート: **"If some interfaces are in the UP state but are not able to communicate between nodes, NCCL may try to use them anyway and therefore fail during the init functions or even hang."** → **A 案 (vLLM の手順に合わせる) で始め、B 案 (NCCL の手順に合わせる) を A/B の候補にする** |
  | `GLOO_SOCKET_IFNAME` | `NCCL_SOCKET_IFNAME` と同じ IF 名 | vLLM の事前の確認 (`test.py`) の 2 段目と、CPU 側の集団通信が Gloo を使う | PyTorch の分散の文書: **"By default, both the NCCL and Gloo backends will try to find the right network interface to use. If the automatically detected interface is not correct, you can override it using the following environment variables… `GLOO_SOCKET_IFNAME`, for example `export GLOO_SOCKET_IFNAME=eth0`"**。**⚠ Gloo については `^` の除外も `=` の完全一致も文書化されていない (SILENT)。NCCL の書式を流用しないこと** |
  | `NCCL_IB_HCA` | **設定しない** | NVIDIA 自身が 2 台の直結では設定していない (上の引用)。設定するなら `=` 付きの完全一致で | NCCL: **"Note: using `mlx5_1` without a preceding `=` will select `mlx5_1` as well as `mlx5_10` to `mlx5_19`, if they exist. It is therefore always recommended to add the `=` prefix to ensure an exact match."** |
  | `NCCL_IB_MERGE_NICS` | **設定しない (既定 1 のまま)** | **これが約 190 Gbps の仕組みそのもの**。1 本のケーブルの 2 つの PCIe function を 1 つの論理 NIC に束ねる | NCCL: **"Enable NCCL to combine dual-port IB NICs into a single logical network device. This allows NCCL to more easily aggregate dual-port NIC bandwidth."** / **"Default is 1 (enabled), define and set to 0 to disable NIC merging"**。**0 にしてはいけない** |
  | `NCCL_IB_GID_INDEX` | **設定しない** | NCCL 2.21 以降は自動で選ぶ | NCCL のトラブルシュート: **"With NCCL 2.21 and later the GID index is dynamically selected… With NCCL 2.21 and later releases, this environment variable should not be set."** イメージの NCCL は 2.30.7 |
  | `NCCL_DEBUG` | `INFO` (確認のときだけ) | 経路の確認に要る。定常運転では `WARN` | NCCL: **"Controls the debug information that is displayed from NCCL."** 値は `VERSION` / `WARN` / `INFO` / `TRACE`。ログの頁で `WARN` が **"Production minimum"** |
  | `NCCL_DEBUG_SUBSYS` | `INIT,BOOTSTRAP,ENV,NET,GRAPH` | **既定は `INIT,BOOTSTRAP,ENV` で、NET と GRAPH が入らない。** HCA の詳細も NIC の束ねも GID も、既定では出ない | NCCL: **"Allows the user to filter the `NCCL_DEBUG=INFO` output based on subsystems."** / **"The default value is INIT,BOOTSTRAP,ENV."** |
  | `NCCL_DEBUG_FILE` | `<dir>/nccl_%h_%p.log` | 2 台ぶんのログを取りこぼさずに回収する | NCCL: **"The filename format can be set to `filename.%h.%p` where `%h` is replaced with the hostname and `%p` is replaced with the process PID."** |

  **`NCCL_IB_HCA` に書く名前は 【実機で決める】**。`rocep1s0f0` は Linux の netdev 名で、IB verbs のデバイス名とは別である。`ibdev2netdev` で対応を読む (NVIDIA の手順の出力の例では、RoCE デバイス名がそのまま `rocep1s0f0` などになっている)。

  **(e-4) NCCL のログで何を見るか (VERIFIED、書式は NCCL のソースから)**

  INFO の行の形は `<hostname>:<pid>:<tid> [<cudaDev>] NCCL INFO <message>` である (`src/debug.cc`)。**`NCCL_DEBUG_SUBSYS` の名前は行の中に出ない** — `NET/IB` などは各呼び出し箇所の書式文字列の先頭の文字である。

  | 見たいこと | 探す文字列 (POSIX ERE) | 出どころ・注意 |
  |---|---|---|
  | 版 | `NCCL version [0-9.]+[^ ]*\+cuda[0-9.]+` | `src/init.cc` の `VERSION_STRING`。2.21〜2.32 で不変 |
  | **トランスポートの確定** | `NCCL INFO Using network (IB\|Socket)` | `src/init.cc:557` `INFO(NCCL_INIT, "Using network %s", …)`。**最も確実な 1 行。2.21 から 2.32 まで不変**。既定の SUBSYS で出る |
  | IB が見つからない | `NET/IB : No device found\.` | `src/transport/net_ib/init.cc`。**2.21 から 2.32 まで一字一句同一** |
  | 選ばれた IB デバイス | `NET/IB : Using` | `"NET/IB : Using%s %s; OOB %s:%s"`。`RoCE` / `IB` の別はここに出る (`NCCL_IB_LLSTR`)。**RoCE の v1/v2 の別はログに一切出ない** |
  | HCA の詳細・ポート・GID | `NET/IB: \[[0-9]+\] [^ ]+:[^ ]+:[0-9]+/(IB\|RoCE)` / `NET/IB:.* GID ([0-9]+) \(` | **`NCCL_DEBUG_SUBSYS` に `NET` が要る**。NCCL 2.26 系ではこの行は TRACE 扱いで INFO には出ない |
  | **NIC の束ね** | `NET/IB : Made virtual device \[[0-9]+\] name=([^ ]+) .* ndevs=([2-9])` / `TOPO/NET : Made vNic [0-9]+` | **`NET` と `GRAPH` が要る**。`Made virtual device` は `ndevs=1` の素のデバイスにも毎回出るので、**判定は `ndevs>=2` か name に `+` を含むこと**。`TOPO/NET : Made vNic` は ndevs==1 では出ないので、こちらが直接の証拠。**この文字列は NCCL 2.24.3 で入った** |
  | チャンネル数 | `([0-9]+) coll channels, ([0-9]+) collnet channels` | `src/init.cc` の `"%d coll channels, %d collnet channels, …"`。**2.21 から 2.32 まで不変**。既定の SUBSYS で出る |
  | チャンネルごとの経路 | `Channel [0-9]{2}/[0-9]+ : .* \[(send\|receive)\] via NET/(IB\|Socket)/[0-9]+` | 既定の SUBSYS で出る |
  | **GPUDirect RDMA が使われたか** | `via NET/IB/[0-9]+/GDRDMA` | `req.useGdr ? "/GDRDMA" : ""`。**DGX Spark では出ないのが正常** (§e-2) |

  **⚠ 「IB から Socket に落ちた」と明示する文字列は NCCL のソースに存在しない。** 判定は連言で行う: **`Using network Socket` が出ている ∧ `NET/IB : No device found.` が出ている**。NVIDIA 自身の判定の書き方 (DGX Station の手順、原文): **"Good: NCCL log shows NET/IB and mlx5_0 / mlx5_1 / Bad: NCCL log shows only NET/Socket, which means TCP fallback"**。vLLM 側の原文: **"If you find `[send] via NET/Socket` in the logs, NCCL used a raw TCP socket, which is not efficient for cross-node tensor parallelism."**

  **採取のしかた (要件 4.4 の記録)**: `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,ENV,NET,GRAPH NCCL_DEBUG_FILE=<dir>/nccl_%h_%p.log`。2 台ぶんを回収して残す。

  **(e-5) 2 台の間の帯域をどう測るか — nccl-tests か、自前か**

  nccl-tests の README (VERIFIED):

  - ライセンス: **"NCCL tests are provided under the BSD license. All source code and accompanying documentation is copyright (c) 2016-2026, NVIDIA CORPORATION."** → **業務で使える (要件 3.9)**
  - ビルド: MPI なしでも作れる (`make CUDA_HOME=… NCCL_HOME=…`)。MPI ありは `make MPI=1 MPI_HOME=…`
  - **マルチノードには MPI が要る**: **"NCCL tests rely on MPI to work on multiple processes, hence multiple nodes."** 実行は `mpirun`
  - 引数: `-b` **"minimum size to start with. Default : 32M"**、`-e` **"maximum size to end at. Default : 32M"**、`-f` **"multiplication factor between sizes. Default : disabled."**、`-g` **"number of gpus per thread. Default : 1."**、`-t` **"number of threads per process. Default : 1."**、`-n` **"number of iterations. Default : 20."**、`-w` **"number of warmup iterations (not timed). Default : 1."**、`-c` **"perform count iterations, checking correctness of results on each iteration."**
  - `algbw` と `busbw` の定義は `doc/PERFORMANCE.md` にある。原文: **"`algbw = S/t`"** / **"In all cases, we need n-1 additions and n assignments for each element. … we need 2(n-1) data transfers (x number of elements) to perform an allReduce operation."** / **"`B = S/t * (2*(n-1)/n) = algbw * (2*(n-1)/n)`"**。実装の除数は `1.0E9` なので **10 進の GB/s** である (GiB ではない)
  - **⚠ n=2 では係数が `2*(2-1)/2 = 1.0` になるので、`busbw` と `algbw` が同じ値になる。** 2 台の比較で「busbw が algbw と同じだ」と驚かないこと
  - **⚠ ビルドの既定の `NVCC_GENCODE` に sm_121 が入っていない** (CUDA 13 以上で `sm_75, sm_80, sm_90, sm_100, sm_120`)。NVIDIA 自身の手順も NCCL を `-gencode=arch=compute_121,code=sm_121` でビルドし直している。→ nccl-tests を使うなら `NVCC_GENCODE` の明示が要る
  - **公式のビルド済みバイナリは存在しない** (GitHub の Releases が 0 件)。README も取得について何も書いていない (SILENT)
  - **NVIDIA 自身が持っている合否のしきい値** (Spark のクラスタ設定スクリプトの定数、事実): all_gather の `# Avg bus bandwidth` に対して **21.875 GB/s (= 175 Gbps)**、リングのときは 10 GB/s (= 80 Gbps)。NVIDIA Sync の文書にも **"NVIDIA Sync then runs a speed test across the links to check the lower bound of 184 Gbit/s."** とある。**NCCL の busbw の実測の例は公表されていない (SILENT)** — 公表値は `ib_write_bw` の 189.85 Gbps としきい値だけ
  - `-a` は **`--average`** であって集約ではない (集約は `-m,--agg_iters`)。しかも **`(MPI=1 only)`** なので、MPI なしのビルドでは全ランクの平均が取れない
  - `-b` と `-e` の既定が**両方 32M** なので、既定のままだと 1 つの大きさしか測らない

  **問題**: 2 台で nccl-tests を動かすには **MPI が要る**。Spark に MPI を入れるのは恒久的な変更で、要件 2.7 / 2.8 に触れる (計測者の判断が要る)。

  → **推奨 (Decision 7)**: **要件 4.3 の帯域の測定は、固定した vLLM のイメージの中で `torchrun` を使った自前の all-reduce の計測で行う。** 根拠:
  - イメージには PyTorch と NCCL 2.30.7 と `torchrun` が入っている (VERIFIED、`docker/versions.json`)
  - vLLM 自身が **`torchrun --nnodes 2 --nproc-per-node=… --rdzv_backend=static --rdzv_endpoint=$MASTER_ADDR --node-rank $NODE_RANK test.py`** という形をトラブルシュートの文書で示している。static にする理由も明記: **"We use `--rdzv_backend=static` instead of `c10d` because the `c10d` rendezvous backend can fail with DNS resolution errors in multi-node setups"**
  - `busbw = algbw × 2(n−1)/n` の定義は NCCL の公式の Performance の文書にある (自分たちのコードで実装する。nccl-tests のコードは写さない)
  - Spark に何も入れない。コンテナだけで完結する

  nccl-tests は **計測者が MPI の導入を了承した場合の追加の確認**として残す (要件 2.1 / 2.8 に沿って、実行前に確認を取る)。

  **(e-6) 推論サーバーを立てる前の事前の確認 (要件 4.6、VERIFIED)**

  vLLM のトラブルシュートの文書にある `test.py` が、4 段階を確かめる:

  1. PyTorch の NCCL の all-reduce → `PyTorch NCCL is successful!`
  2. PyTorch の GLOO の all-reduce (CPU) → `PyTorch GLOO is successful!`
  3. vLLM 独自の `PyNcclCommunicator` の all-reduce → `vLLM NCCL is successful!`
  4. **CUDA グラフの中での** vLLM の NCCL → `vLLM NCCL with cuda graph is successful!`

  4 段目が特に重要である。issue #52504 が報告した 2 台 DGX Spark のデッドロックはまさに「CUDA グラフに取り込まれた NCCL の集団通信」で起きていた。**この確認が通れば、その落とし穴は踏んでいないと言える。**

  文書の注意書き: **"If the test script hangs or crashes, usually it means the hardware/drivers are broken in some sense. … As a common workaround, you can try to tune some NCCL environment variables, such as `export NCCL_P2P_DISABLE=1` to see if it helps. … Please only use these environment variables as a temporary workaround, as they might affect the performance of the system."**

  **(e-7) A/B で良くなったときだけ採用する候補 (要件 4.5)**

  **最初の構成には入れない。** それぞれ NCCL の公式の説明を添える。

  | 変数 | 既定 | 説明 (原文) | 試す理由【推測】 |
  |---|---|---|---|
  | `NCCL_NET_GDR_LEVEL` | 自動 | **"Allows the user to finely control when to use GPU Direct RDMA between a NIC and a GPU."** `LOC` = **"Never use GPU Direct RDMA (always disabled)"** | DGX Spark では GDR が非対応 (e-2) なので、`LOC` にして探索を省けば初期化が速くなるかもしれない |
  | `NCCL_IB_GID_INDEX` | `-1` | **"Defines the Global ID index used in RoCE mode."** | RoCE v2 の GID を選び損ねている場合の修正 |
  | `NCCL_IB_TC` / `NCCL_IB_SL` | `0` / `0` | **"Defines the InfiniBand traffic class field."** / **"Defines the InfiniBand Service Level."** | 輻輳制御の調整 |
  | `NCCL_IB_QPS_PER_CONNECTION` | `1` | **"Number of IB queue pairs to use for each connection between two ranks."** 値は **"Number between 1 and 128"** | 1 本のリンクを複数の QP で埋める |
  | `NCCL_IB_SPLIT_DATA_ON_QPS` | `0` | **"Controls how we use queue pairs when we create more than one."** | 上とセット |
  | `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | プラットフォーム依存 | **"Controls the minimum number of channels you want NCCL to use."** / **"Limits the number of channels NCCL can use."** | 小さいメッセージの遅延と、大きいメッセージの帯域のトレードオフ |
  | `NCCL_CROSS_NIC` | `2` | `0` = **"Always use the same NIC for the same ring/tree"**、`1` = **"Allow the use of different NICs for the same ring/tree"**、`2` = **"Try to use the same NIC … but still allow for the use of different NICs"** | 直結のインターフェースが 2 つあるときの束ね方 |
  | `NCCL_P2P_DISABLE` / `NCCL_SHM_DISABLE` | `0` / `0` | **"Disables the peer to peer (P2P) transport …"** / **"Disables the Shared Memory (SHM) transports."** | 切り分け用。#41725 では「効かなかった」と実測されている |
  | `NCCL_IB_DISABLE` | `0` | **"Prevents the IB/RoCE transport from being used by NCCL."** | 切り分け用 (これを 1 にして遅くなれば、RoCE が効いていた証拠になる) |

- **Implications for the design**:
  - 通信の設定は **3 つだけで始める** (`VLLM_HOST_IP`、`NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`、`NCCL_IB_HCA`)。それ以外は「A/B で測って良くなったものだけ」という別の表に分けて、構成の定義には `measured` の根拠がないと書けないようにする
  - `--master-addr` は **直結リンク側のアドレス**を第一候補にする。LAN 側 (10.0.1.60) を使う案もあるが、rendezvous とノード間のキューが遅い経路に載る。ただし **直結リンクが落ちているときに何も動かなくなる**ので、状態の確認 (要件 1.6) でリンクの状態も見る

### f. 起動の確認、状態、停止

- **Context**: 要件 1.4 / 1.5 / 1.6 / 1.7 / 1.8。

- **Findings (VERIFIED、ソース)**:

  **(f-1) 受け付けられる状態の判定**

  `vllm/entrypoints/serve/instrumentator/health.py`:

  ```python
  @router.get("/health", response_class=Response)
  async def health(raw_request: Request) -> Response:
      """Health check."""
      client = engine_client(raw_request)
      if client is None:
          return Response(status_code=200)
      try:
          await client.check_health()
          return Response(status_code=200)
      except EngineDeadError:
          return Response(status_code=503)
  ```

  - **`/health`**: ボディなし。エンジンが生きていれば 200、死んでいれば 503。ルートはアプリの構築後に登録されるので、**モデルのロード中は接続そのものが拒否される**。→ 「200 が返り始めた時点」を準備完了とする
  - **`/v1/models`**: `--served-model-name` が `id` になる (§d-4)。`bench` が入力長の上限 `max_model_len` を読むのに使う
  - **`/version`**: 版の記録
  - **`/ping`**: SageMaker 用だが無条件で登録される。`/health` と同じ結果

  **判定の手順**: (1) `/health` が 200 になるまで待つ、(2) `/v1/models` の `data[0].id` が構成の名前と一致することを確かめる、(3) `/metrics` が読めることを確かめる (`bench` と `scripts/spark-precheck.sh` が使う)。

  **(f-2) どれくらい待つか【推測】**

  - 重みのロード: 1 ノードあたり約 92 GiB を NVMe から読む。**【実機で決める】: 起動ログの `Model loading took N GiB memory and X.XXXXXX seconds`**
  - DeepGEMM の JIT: `VLLM_DEEP_GEMM_WARMUP` の説明に **"this warmup increases the engine startup time by a couple of minutes"**。既定は `"relax"`
  - FlashInfer の autotune: #52291 の報告では、マルチノードでここが止まる
  - → **初回は 20〜30 分を上限に置く**【推測】。キャッシュが効く 2 回目以降は短くなるはず。タイムアウトの値そのものが 【実機で決める】

  **(f-3) 止め方 (VERIFIED)**

  - **API サーバー側** (`vllm/entrypoints/launchers/launcher.py` L123-165): SIGINT / SIGTERM を受けると `shutdown_event` を立て、`engine_client.shutdown(timeout=shutdown_timeout)` を呼ぶ。`timeout == 0` なら `mode="abort"`、正なら `"drain"`。ログに `[shutdown] API server: …` が出る
  - **headless のワーカー側** (`vllm/entrypoints/cli/serve.py`): SIGTERM / SIGINT で `SystemExit` を上げ、`finally` で `engine_manager.shutdown(timeout=...)`
  - **`docker stop` は既定で SIGTERM を送り、10 秒後に SIGKILL** (Docker の文書)。vLLM の drain を待たせたいなら `-t` を `--shutdown-timeout` より長くする
  - **どちらのノードから止めるかについて、vLLM の公式の記述はない (SILENT)**。→ **【推測】: head (API サーバー) を先に止めて新規の受け付けを断ち、drain が終わってから worker を止める。** 逆順だと、head が worker の消失を検知してエラーで落ち、ログが「異常終了」に見えてしまう

  **(f-4) 何度やっても同じになること (要件 1.8)**

  - **起動**: ラベルで自分のコンテナを探し、あれば「いまの状態」を出して 0 で終わる。なければ起こす
  - **停止**: ラベルで探し、なければ「すでに止まっている」と出して 0 で終わる
  - **判定の材料**: `docker container inspect --format` (Docker: **"Format output using a custom template"**)。`.State.Running` と `.Config.Image` については **docs が SILENT** なので、実際の出力を見て使う【実機で確かめる】
  - **GPU のメモリが空いたこと (要件 1.7)**: `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` が空になることで確かめる。`scripts/spark-precheck.sh` がすでに同じ読み方をしているので、同じ形に揃える

### g. `/v1/messages` の確認項目 (要件 9)

- **Context**: 要件 9.1 が 8 つの項目を挙げ、9.3 が「`bench` の結果から確かめられるものは、別の確かめを足さない」ことを求める。

- **Findings (VERIFIED、ソース)**:

  `/v1/messages` と `/v1/messages/count_tokens` は `vllm/entrypoints/anthropic/api_router.py` に実装されている (`@router.post("/v1/messages", …)` / `@router.post("/v1/messages/count_tokens", …)`)。要求の型 `AnthropicMessagesRequest` (`vllm/entrypoints/anthropic/protocol.py` L121-176) が受け取るのは:

  ```
  model, messages, max_tokens, metadata, output_config, stop_sequences, stream,
  system, temperature, tool_choice, tools, top_k, top_p
  + vLLM 独自: cache_salt, kv_transfer_params, ec_transfer_params, vllm_xargs,
    chat_template_kwargs
  ```

  **重要な 2 点**:

  1. **Anthropic の形の `thinking` フィールドは存在しない。** だから `thinking: {type: enabled/disabled}` を送っても黙って無視される。P0 の実測 (`docs/decisions/0001-…` 「5 通りとも出力が同じだった」) は、この実装と整合する
  2. **`chat_template_kwargs` は受け取れる (VERIFIED)**。`vllm/entrypoints/anthropic/serving.py` L478 / L494 が `chat_template_kwargs=anthropic_request.chat_template_kwargs` として下流に渡す
  3. **`output_config.effort` という専用の入口がある。** `AnthropicOutputConfig.effort: Literal["low","medium","high","xhigh","max"]` (protocol.py L114-118) で、`serving.py` L515-516 が `req.reasoning_effort = output_config.effort` に写す。`ChatCompletionRequest.build_chat_params()` (`vllm/entrypoints/openai/chat_completion/protocol.py` L569-594) が `reasoning_effort` をチャットテンプレートの変数として渡す

  **(g-2) チャットテンプレートの実物を読んだ結果 (VERIFIED) — brief.md の部分的な訂正**

  `https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/chat_template.jinja` (10,950 B) の該当箇所の原文:

  ```jinja
  {%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}
  {%- if effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}
  ```
  ```jinja
  {%- set clear_thinking = clear_thinking if clear_thinking is defined else false -%}
  ```
  ```jinja
  {%- if (not clear_thinking or loop.index0 > ns.last_user_index) and reasoning_content is defined -%}
  {{ '<think>' + reasoning_content +  '</think>'}}
  {%- else -%}
  {{ '<think></think>' }}
  {%- endif -%}
  ```
  ```jinja
  {%- if add_generation_prompt -%}
      <|assistant|>{{- '<think>' -}}
  {%- endif -%}
  ```

  ベンダのモデルカードの原文も一致する:

  > "GLM-5.3-Flash supports controlling the thinking budget through the `reasoning_effort` parameter, which accepts three levels: `low`, `high`, and `max`. It defaults to `max` if not passed (or if set to any other value)."
  > "In the chat template for GLM-5.3-Flash, `clear_thinking` defaults to `false` if not passed. For chat scenarios, explicitly pass `clear_thinking=true`."

  **わかったこと**:

  1. **`reasoning_effort` が実際に効くのは `low` と `high` だけ。** それ以外 (未指定、`medium`、`none`、`minimal`、`xhigh`、`max`) は**すべて `max` に落ちる**。→ 段は実質 **3 つ (low / high / max)**
  2. **thinking は切れない。** 生成プロンプトは無条件に `<think>` を開く。`enable_thinking` という変数はテンプレートに**存在しない**。recipes.vllm.ai も **"Thinking is always on — the generation prompt opens a `<think>` block unconditionally."** と書いている
  3. **`clear_thinking` は thinking の深さを変えない。** やるのは**履歴の掃除**だけで、`true` のとき、最後の user メッセージより前の assistant の `reasoning_content` を空の `<think></think>` に置き換える。→ **長い会話の入力トークンを減らす効果がある**ので、要件 7 の `agent` のまとまりに効く可能性がある
  4. → **brief.md の「チャットテンプレートが読むのは `reasoning_effort` (low / high / 既定は max) と `clear_thinking` だけで、thinking を完全に切る手段はない」は正しい。** ただし `clear_thinking` の役割 (履歴の掃除であって深さではない) は brief.md からは読み取れない

  **⚠ API の層とテンプレートの層で受理する値が違う (設計に直結)**

  - `/v1/messages` の `output_config.effort` の型は `Literal["low","medium","high","xhigh","max"]` (`vllm/entrypoints/anthropic/protocol.py` L117)。**`"none"` は無い**
  - `/v1/chat/completions` の `reasoning_effort` の型は `Literal["none","minimal","low","medium","high","xhigh","max"]`
  - → **`medium` / `minimal` / `none` / `xhigh` を送ると API は 200 を返すが、テンプレート側で黙って `max` に落ちる。** 「設定したのに変わらない」という形の罠になる。**構成にも計測にも、`low` / `high` / 既定 (max) の 3 つしか使わない**

  **⚠ Anthropic 形式の `thinking` フィールドは黙って捨てられる (VERIFIED)**

  `AnthropicMessagesRequest` に `thinking` の項目は無く、pydantic の既定 (`extra="ignore"`) で捨てられる。未マージの PR #53058 (OPEN) の本文の原文: **"`AnthropicMessagesRequest` never declared a `thinking` field, so the documented Anthropic request parameter was silently discarded — the model sets no Pydantic `extra` policy, so the default `extra=\"ignore\"` drops it without an error."** → **P0 の実測 (`thinking: enabled/disabled` が 5 通りとも同じ出力) は、この実装で完全に説明がつく。**

  サーバー側の既定は **`--default-chat-template-kwargs`** で入れられる (`vllm/entrypoints/launchers/cli_args.py` L93-98: **"Default keyword arguments to pass to the chat template renderer. These will be merged with request-level chat_template_kwargs, with request values taking precedence."**)。→ **クライアントを直さずに `clear_thinking` の既定を変えられる唯一の口。**

  **⚠ `--enable-prompt-tokens-details` は `input_tokens` の意味を変える (VERIFIED、計測に直結)**

  `vllm/entrypoints/anthropic/serving.py` の docstring の原文: **"Anthropic defines `total_input == input_tokens + cache_read + cache_creation`. vLLM's `prompt_tokens` is the total, so `input_tokens = prompt_tokens - cache_read - cache_creation`."** → **このフラグを付けると `input_tokens` からキャッシュに当たった分が引かれる。** 付けていない計測ランと付けている計測ランで `input_tokens` を直接比べてはいけない。`bench` は 4 つの項目をすべて読むので、要約の側で足し戻せる。

  **⚠ `stop_sequences` は既定 4 個まで** (`VLLM_MAX_STOP_STRINGS`、既定 4)。5 個以上送ると 422 になる。

  **確認項目の一覧 (要件 9.1 の 8 項目)**

  | 項目 | どう確かめるか | `bench` で済むか (要件 9.3) |
  |---|---|---|
  | ストリーミングの応答 | `bench` が全試行で SSE を読む。`AnthropicStreamEvent.type` は `message_start` / `message_delta` / `message_stop` / `content_block_start` / `content_block_delta` / `content_block_stop` / `ping` / `error` (protocol.py) | **済む** |
  | ツール呼び出し (定義・応答・結果を含む続き) | `bench --suite quality` のツールの課題と `--suite agent` が、定義を送り、`tool_use` を受け、結果を含む会話を続ける | **済む** |
  | 長い会話 (12 万トークン、約 1,200 発話) | `bench --suite agent` の最終段 | **済む** |
  | 入力と出力のトークン数 | `bench` が `input_tokens` / `output_tokens` を読む (`bench/src/bench_harness/client/messages.py` の `_USAGE_FIELDS`) | **済む** |
  | キャッシュに当たったトークン数の内訳 | 同じく `cache_read_input_tokens` / `cache_creation_input_tokens`。**`--enable-prompt-tokens-details` が要る** (`serving.py` L81-89) | **済む**。付け忘れると項目が消えるので、付いていることが逆に確かめられる |
  | トークン数だけを数える口 | `bench calibrate` が `POST /v1/messages/count_tokens` を使う | **済む** |
  | thinking のブロックの出方 | `bench` は `content_block_start` の `type == "thinking"` と `thinking_delta` を解釈する (messages.py)。要約に `thinking` の量が残る | **済む** |
  | thinking の深さの渡し方 | **`bench` では確かめられない** (`sampling.thinking` は `server_default` だけ)。素の HTTP で 3 通りを送り分けて比べる | **別に確かめる (下記)** |

  **thinking の深さの確かめ方 (要件 9.2)**

  同じプロンプト・同じ seed・`temperature 0` で、次の 5 通りを送って、`thinking` のブロックの文字数と `output_tokens` を比べる。**テンプレートの実物を読んで決めた渡し方**である (上の g-2)。

  | # | 送るもの | 期待 |
  |---|---|---|
  | 1 | 何も渡さない | 既定 = `Reasoning Effort: Max` |
  | 2 | `{"output_config": {"effort": "low"}}` | `Reasoning Effort: Low` に変わる。**`/v1/messages` の本来の口** |
  | 3 | `{"chat_template_kwargs": {"reasoning_effort": "low"}}` | 2 と同じ結果になるはず (経路違い) |
  | 4 | `{"output_config": {"effort": "medium"}}` | **変わらないはず** (テンプレートが `max` に落とす)。「API は受けるが効かない」ことの確認 |
  | 5 | `{"chat_template_kwargs": {"clear_thinking": true}}` | 深さは変わらず、**長い会話で `input_tokens` が減る**はず |

  **効いたと言える条件**: 1 と 2 の間で、`thinking` のブロックの文字数か `output_tokens` が有意に変わること。4 で変わらないこと。5 で (複数ターンの会話のとき) `input_tokens` が減ること。

  P0 の記録は「5 通りとも同じだった」だが、**そのとき試した渡し方 (Anthropic 形式の `thinking`、`chat_template_kwargs.enable_thinking`) は、このテンプレートが読まない項目ばかりだった**ことが今回わかった。→ **上の #2 が効けば、P0 の判断 (`sampling.thinking = "server_default"`) を見直す理由になる** (要件 9.5)。その場合、`bench` の側の変更は `bench-harness` の仕様の変更として扱う。

  **`bench` から送れるか**: `bench` の `MessagesRequest` は `extra` を持ち、`build_request_body` が本文の**最上位**に混ぜる (`bench/src/bench_harness/client/messages.py`)。したがって道具としては `output_config` も `chat_template_kwargs` も送れる。ただし `profiles.toml` の `sampling.thinking` は `"server_default"` しか受けないので、**計測の設定から切り替える口は今はない**。P1 では素の HTTP で確かめ、必要なら `bench-harness` 側の変更として起票する。

- **Implications for the design**:
  - 要件 9 の 8 項目のうち **7 項目は `bench` を一度流せば根拠が揃う**。追加の確かめが要るのは thinking の深さだけ。Baseline Procedure はこの 1 つだけを別立てにする
  - `--enable-prompt-tokens-details` は「あると嬉しい」ではなく、**要件 9.1 を満たすために必須**

---

## Architecture Pattern Evaluation

Serving Kit の形を 4 案で比べる。判定の基準は要件から取った。

- (A) Mac から ssh で動く / (B) Spark の上でファイルを編集しない / (C) **設定 1 つ 1 つの根拠を、起動前に機械が検査できる (要件 3.6 / 3.7)** / (D) 起動と停止が何度でも同じ (要件 1.8) / (E) 自分のコンテナだけを触る (要件 2.3) / (F) 状態を変える前に確認を取る (要件 2.1) / (G) Spark なしで Mac 上で試験できる

| 案 | 中身 | 長所 | 短所・危うさ | 判定 |
|---|---|---|---|---|
| **1. POSIX シェル + 宣言的な構成ファイル** | `scripts/` に bash、構成は TOML か JSON | Mac にも Spark にも追加の依存がない。既存の `spark-precheck.sh` と同じ流儀 | **(C) が弱い**。bash で TOML を読んで「根拠の欄が空なら起動を断る」を書くと、外部の道具 (`jq` / `yq`) か手書きのパーサーが要る。(G) の試験も、bash の関数を試すのに手間がかかる。文字列の組み立てで `docker run` の引数を作ると、引用の誤りが静かに入り込む | △ |
| **2. Python の CLI (uv、pydantic、tomllib)** | `serving/` に `bench/` と同じ形のプロジェクト。`ssh` と `rsync` は `subprocess` で呼ぶ | **(C) が構造で保証できる**。pydantic の型で「根拠の欄が必須」を表現すれば、欠けた構成は読み込みの時点で落ちる。(G) も素直 (`ssh` を差し替えて、組み立てた引数列を検査する試験が書ける)。引数は文字列でなく**リスト**で組み立てるので引用の事故がない。`bench/` の慣例 (TOML + 凍結した型 + 根拠をコメントに残す) をそのまま延長できる | Mac 側に uv と Python が要る (**すでに `bench/` が要求している**)。行数は bash より増える | **◎ 採用** |
| **3. docker compose** | ノードごとに compose ファイル、`DOCKER_HOST=ssh://` か ssh 越しに `docker compose up` | 起動と停止の冪等性が組み込み | **(C) が無理**。compose の YAML に根拠の欄を持たせても、compose 自身は検査しない。さらに Docker の文書が **"Compose supports production deployments on single hosts."** と述べており、**2 台にまたがる構成は対象外**。head と worker で別々の compose を動かすことになり、compose を使う利点がほぼ消える | ✕ |
| **4. Ansible** | インベントリ + プレイブック | 冪等性と ssh が前提の設計。(A)(B)(D) は得意 | **ライセンス**: ansible-core は **GPLv3** (`https://github.com/ansible/ansible/blob/devel/COPYING` の冒頭が "GNU GENERAL PUBLIC LICENSE Version 3")。このリポジトリの方針 (MIT / Apache-2.0 / BSD 系) から外れる。道具として使うだけなら法的な問題はないが、`LICENSES.md` の方針と揃わない。**管理される側に Python が要る**: 公式の文書に **"The managed node … does not require Ansible to be installed, but requires Python to run Ansible-generated Python code."**。(C) は `assert` で書けるが読みにくい。(F) の「1 つ 1 つ確認を取る」が Ansible の流儀と噛み合わない | ✕ |

**推奨: 案 2 (Python の CLI)。** `bench/` がすでに Python 3.12+ / uv / TOML / pydantic で立っており、同じ形で `serving/` を作れば、計測者が覚えることが増えない。`scripts/spark-precheck.sh` は読み取りだけの独立した道具としてそのまま残し、Serving Kit がその出力を読む。

### 構成ファイルの形 (schema sketch)

`serving/config/configs.toml`。**すべての設定が、値と、なぜと、根拠の 3 つを持つ**。

```toml
schema_version = 1

# ---------------------------------------------------------------------------
# `[configs.<名前>]` が 1 つの構成に対応する。`<名前>` はこのテーブルの鍵から
# 取る。計測者は `serve start --config <名前>` のように選ぶ (要件 3.1)。
#
# 根拠 (provenance) の書き方は 2 通りだけで、どちらか一方が必ず要る:
#   source + quote : 公式の資料の URL と、その原文の抜粋
#   measured       : 自分たちの実測の記録の場所 (docs/results/... または docs/...)
# どちらも空の項目が 1 つでもあれば、Serving Kit はその構成での起動を断り、
# 根拠のない項目の名前を並べて終わる (要件 3.7)。
# ---------------------------------------------------------------------------

[configs.p1-nvfp4-tp2]
description = "P1 の第一の構成。上流の公式イメージ + NVFP4 + 2 台 TP=2、投機的デコードなし"
nodes = ["head", "worker"]

[configs.p1-nvfp4-tp2.image]
ref        = "vllm/vllm-openai@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
seen_as    = "vllm/vllm-openai:glm53-flash-arm64-cu130"
source     = "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags (2026-09-21 取得)"
quote      = "glm53-flash-arm64-cu130 / linux/arm64 / 9666567584 B / 2026-09-09T13:31:25Z"

[configs.p1-nvfp4-tp2.weights]
repo       = "RedHatAI/GLM-5.3-Flash-NVFP4"
revision   = "18d55bfd5a2194887738da73753975c9d3842f46"
manifest   = "serving/weights/RedHatAI__GLM-5.3-Flash-NVFP4.sha256"   # 名前・大きさ・sha256
mount_at   = "/models/glm-5-3-flash-nvfp4"
source     = "https://huggingface.co/api/models/RedHatAI/GLM-5.3-Flash-NVFP4?blobs=true"
quote      = "license=mit, sha=18d55bfd..., total=197881158759 bytes, siblings[].lfs.sha256"

# --- docker run の引数。1 つの表が 1 つのフラグ -----------------------------

[configs.p1-nvfp4-tp2.docker."--ipc"]
value  = "host"
why    = "TP がプロセス間で共有メモリを使う"
source = "https://docs.vllm.ai/en/latest/deployment/docker/"
quote  = "You can either use the `ipc=host` flag or `--shm-size` flag to allow the container to access the host's shared memory. vLLM uses PyTorch, which uses shared memory to share data between processes under the hood, particularly for tensor parallel inference."

[configs.p1-nvfp4-tp2.docker."--ulimit"]
value    = "memlock=-1:-1"
why      = "RoCE のメモリ登録が mlock を使う"
measured = "docs/results/2026-09-XX-ulimit-check.md"   # 公式の資料は -1 の意味を述べていない

# --- vllm serve の引数 ------------------------------------------------------

[configs.p1-nvfp4-tp2.serve."--tensor-parallel-size"]
value  = "2"
why    = "2 台 1 GPU ずつで 1 つのモデルを分割する"
source = "https://docs.vllm.ai/en/latest/serving/parallelism_scaling/"
quote  = "Besides Ray, Multi-node vLLM deployments can also use multiprocessing as the runtime engine."

[configs.p1-nvfp4-tp2.serve."--max-num-seqs"]
value  = "16"
why    = "KDA の状態は同時実行数に比例し、1 本あたり約 70 MiB/ノード。既定 (1024 の見込み) では数十 GiB を失う"
source = "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/mamba_utils.py#L303"
quote  = "recurrent_state_shape = (divide(num_heads, tp_world_size), head_dim, head_dim)"

# --- ノードごとに違う値 -----------------------------------------------------

[configs.p1-nvfp4-tp2.per_node.head]
node_rank = 0
headless  = false

[configs.p1-nvfp4-tp2.per_node.worker]
node_rank = 1
headless  = true

# --- 通信 (最小の設定)。A/B で採用したものだけを足す -------------------------

[configs.p1-nvfp4-tp2.env."NCCL_SOCKET_IFNAME"]
value  = "=__FILL_FROM_MEASUREMENT__"
why    = "NCCL のブートストラップを直結リンクに固定する"
source = "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html"
quote  = "Define to a list of prefixes to filter interfaces to be used by NCCL. ... `=eth0`: Use only interface `eth0`"
```

**Serving Kit の口 (案)**

| コマンド | すること | 対応する要件 |
|---|---|---|
| `serve check <構成>` | 構成の根拠の検査、イメージのダイジェストの照合、重みのマニフェストの照合、ディスクの空きの確認。**Spark の状態を変えない** | 3.2 / 3.3 / 3.6 / 3.7 / 2.5 |
| `serve push <構成>` | Spark の上で要るものだけを rsync する | 1.1 / 1.2 |
| `serve fetch <構成>` | 各 Spark で重みを取得して照合する (実行前に確認) | 3.4 / 3.5 / 2.1 |
| `serve start <構成>` | 2 台で決まった順序で起こし、`/health` が 200 になるまで待つ。失敗したらログの末尾を出して片付ける | 1.3 / 1.4 / 1.5 / 2.1 / 2.2 / 3.10 |
| `serve status` | 2 台の動作、構成の名前、GPU のメモリ、受け付けの可否 | 1.6 |
| `serve stop` | 自分のラベルのコンテナだけを止め、GPU が空いたことを確かめる | 1.7 / 2.3 |
| `serve logs [--since]` | 2 台のログを git の管理外へ写す | 1.9 |
| `serve probe <構成>` | 1 台・縮小・`--load-format dummy` の起動の確認 (§d-2) | 5.1 / 5.2 / 5.3 |
| `serve netcheck` | インターフェースの読み取り → 自前の帯域の測定 → vLLM の事前の確認 | 4.1〜4.7 |

**Mac だけでできる試験 (要件 G)**: `ssh` と `rsync` の呼び出しを差し替えられるようにして、**「この構成からはこの引数列が組み立たる」ことを試験する**。`bench` が偽のサーバーで通信の壊れ方まで試験しているのと同じ考え方。根拠の欠けた構成が起動を断られることも、試験で固定する。

---

## Design Decisions

### Decision 1: Serving Kit は Python の CLI にし、構成は根拠つきの TOML で持つ

- **Context**: 要件 3.6 / 3.7 が「設定 1 つ 1 つに根拠」と「根拠のない設定があれば起動を断る」を求める。これは単なる書式の規約ではなく、**機械が検査する制約**である
- **Alternatives Considered**: (1) bash + TOML、(2) Python の CLI、(3) docker compose、(4) Ansible
- **Selected Approach**: `serving/` に `bench/` と同じ構成 (Python 3.12+、uv、pydantic、tomllib) の CLI を作る。構成は `serving/config/configs.toml`。`ssh` / `rsync` / `docker` は引数の**リスト**で呼ぶ
- **Rationale**: 根拠の必須化を型で表現でき、Spark なしで試験できる。既存の `bench/` と同じ流儀なので、計測者の負担が増えない。compose は単一ホスト前提 (公式の記述)、Ansible は GPLv3 で方針から外れる
- **Trade-offs**: bash より行数が増える。Mac 側に Python の環境が要る (すでに要る)
- **Follow-up**: `serving/` を `bench/` と同じ `uv.lock` の流儀にし、`LICENSES.md` に依存を追記する

### Decision 2: イメージは単一アーキのタグのダイジェストで固定し、`--pull never` で起動する

- **Context**: 要件 3.2 / 3.3
- **Alternatives Considered**: (1) タグで指定する、(2) マニフェストリストのダイジェスト、(3) 単一アーキのマニフェストのダイジェスト
- **Selected Approach**: (3)。`vllm/vllm-openai@sha256:b0501f99…` を正とし、`glm53-flash-arm64-cu130` はメモとして残す。起動前に `docker image inspect` の `RepoDigests` に含まれるかを確かめ、`docker run --pull never` で動かす
- **Rationale**: Docker の文書がダイジェストでの固定を「immutable identifier」と明言している。マニフェストリストのダイジェストだと、手元に落ちた単一アーキのイメージと 1 対 1 で照合できない
- **Trade-offs**: 上流が新しい nightly を出しても自動では追わない (意図どおり)
- **Follow-up**: 取得したイメージの中のライセンス表記を読み、`LICENSES.md` に「vLLM は Apache-2.0、ベースは NVIDIA CUDA の…」と記録する

### Decision 3: 重みは各 Spark が公開の Hub から直接取得し、Mac で作ったマニフェストで照合する

- **Context**: 要件 3.4 / 3.5 / 2.6
- **Alternatives Considered**: (1) Mac に落として rsync、(2) 各 Spark で直接取得
- **Selected Approach**: (2) を既定にし、(1) を代替に残す。取得は `hf download <repo> --revision <40 桁 sha> --local-dir <dir>`、照合は `hf cache verify --local-dir <dir> --fail-on-missing-files --fail-on-extra-files`。**照合の正解 (名前・大きさ・sha256) は Mac が Hub の `tree` API から作ってリポジトリに入れる**
- **Rationale**: 公開リポジトリなので認証の情報は要らない (本調査で匿名アクセスを確認)。Mac のディスクの空きに依存しない。2 台が並列に落とせる。`--local-dir` を使うとシンボリックリンクも `CACHEDIR.TAG` も避けられ、rsync に切り替えたくなったときにも困らない。`hf cache verify` が sha256 での照合を 1 コマンドで行う (`hf_hub_download` 自身はバイト数しか見ない)
- **Trade-offs**: 同じ 184 GiB を 2 回落とす。Spark が外に出られることが前提。**中断すると未完のシャードは消えてやり直しになる** (1.x は再開を捨てた)。ただしファイル単位では atomic rename なので、再実行すれば未完のものだけが対象になる
- **Follow-up**: `hf` が入っていない Spark でどう入れるかを決める (コンテナの中で動かす案がある。ホストにパッケージを入れるのは要件 2.7 に触れる)。`HF_XET_HIGH_PERFORMANCE=1` が効くかを A/B する (条件は RAM 64 GB 以上で、Spark は満たす)

### Decision 4: 自分のコンテナはラベルで選ぶ。名前では選ばない

- **Context**: 要件 2.3 (他人のコンテナを止めない・消さない)、2.4 (他人の構成の中身を読まない)
- **Alternatives Considered**: (1) 名前の接頭辞で `docker ps --filter name=`、(2) ラベルで `--filter label=`
- **Selected Approach**: (2)。起動時に `--label` を付け、停止・状態確認・ログ回収はラベルでのみ選ぶ
- **Rationale**: Docker の文書が名前のフィルタを **"matches on all or part of a container's name"** と定義しており、**部分一致**である。計測者が別に起動していたコンテナ (`exl3-tp2`) を巻き込む恐れがある。ラベルは `<key>=<value>` の一致
- **Trade-offs**: 手で起こしたコンテナは Serving Kit の管理下に入らない (これは正しい振る舞い)
- **Follow-up**: `serve stop` が「見つからなかった」ときに、名前で探し直すような親切をしないことを試験で固定する

### Decision 5: 最初の構成に投機的デコードを入れない。入っていないことを 2 つの方法で確かめる

- **Context**: 要件 6.7、PLAN.md の P3
- **Selected Approach**: `--speculative-config` / `--spec-method` / `--spec-model` / `--spec-tokens` を一切渡さない。確認は (i) `/metrics` に `vllm:spec_decode_*` が出ないこと、(ii) 起動時の設定のダンプに `SpeculativeConfig` が出ないこと
- **Rationale**: `create_speculative_config` がこの 4 つのいずれもなければ `None` を返す (ソースで確認)。MTP を含むチェックポイントでも自動では有効にならない
- **Trade-offs**: 速さの伸びしろを P1 では取らない (意図どおり)
- **Follow-up**: MTP の重みがロードされていないことを `Model loading took N GiB` で確かめる

### Decision 6: `--max-num-seqs` を明示的に小さく設定する

- **Context**: §d-6 の計算
- **Alternatives Considered**: (1) 既定に任せる、(2) 16 にする、(3) 2 にする (運用の想定どおり)
- **Selected Approach**: (2) の 16
- **Rationale**: KDA の状態が **文脈長ではなく同時実行数**に比例し、1 本あたり約 70 MiB/ノード。GB10 の総メモリ (121.7 GiB) は vLLM の既定の分岐で「70 GiB 以上」に当たるため、既定が 1024 になる可能性が高い。`bench` の参考計測が同時 8 本まで行くので、16 なら足りて、約 1.1 GiB で済む
- **Trade-offs**: 17 本目以降は待たされる。この運用では起きない
- **Follow-up**: 起動ログの内訳で、KDA の状態が実際にどれだけ取られたかを確かめ、`--max-num-seqs` を変えたときの `Available KV cache memory` の差を測る

### Decision 7: 2 台の帯域は nccl-tests ではなく、イメージの中の torchrun で測る

- **Context**: 要件 4.3、要件 2.7 (Spark の恒久的な設定を変えない)
- **Alternatives Considered**: (1) Spark に MPI を入れて nccl-tests、(2) MPI を含むコンテナを作って nccl-tests、(3) 固定したイメージの中で torchrun を使った自前の計測
- **Selected Approach**: (3) を既定にする。(1) は計測者が了承した場合の追加として残す
- **Rationale**: nccl-tests の README が **"NCCL tests rely on MPI to work on multiple processes, hence multiple nodes."** と明記しており、2 台で測るには MPI が要る。Spark への MPI の導入は恒久的な変更。一方、固定したイメージには PyTorch と NCCL 2.30.7 と torchrun が入っており、vLLM 自身がトラブルシュートの文書で `torchrun --rdzv_backend=static` の形を示している。`busbw = algbw × 2(n−1)/n` の定義は NCCL の公式の資料にあるので、自分たちで実装できる (クリーンルーム、要件 11.2)
- **Trade-offs**: NVIDIA が公表する数値との比較が、厳密には「同じ道具での比較」でなくなる。→ アルゴリズム (all-reduce)、メッセージ長の範囲、反復数を記録して、条件を明示する
- **Follow-up**: `busbw` の定義を NCCL の公式の Performance の頁から引用して、実装のコメントに残す

### Decision 8: 代替の順序を、障害の原因に合わせて組み替える

- **Context**: 要件 8.1 の順序と、§d-1 / §d-8 の分析の食い違い
- **Alternatives Considered**: (1) 要件どおりの順序、(2) 原因に合わせた順序
- **Selected Approach**: (2)。§d-8 の段 0〜5。とくに **「重みを落とす前に、1 台の縮小起動でいちばんの懸念に当たる」** を段 0 に置く
- **Rationale**: 障害はアテンションの幾何 (`qk_rope_head_dim = 0`) に起因し、量子化の形式にも並列の取り方にも依らない。2 つのチェックポイントの `config.json` を実測して、幾何が同一であることを確認済み。要件どおりの順序で進めると、同じ失敗を 2 回繰り返して 360 GiB を無駄に落とすことになる (要件 8.3「同じ失敗を、設定を変えずに繰り返して試さない」にも反する)
- **Trade-offs**: 要件 8.1 の文言と一致しなくなる。→ **設計のレビューで、要件 8.1 の改訂を計測者に諮る**
- **Follow-up**: 段 0 の結果がどちらに転んでも、要件 10 の「動かなかった箇所」の 1 件目として記録する

### Decision 9: 1 台での確認は、重みを落とさずに `--load-format dummy` + `--hf-overrides` で行う

- **Context**: 要件 5.1 / 5.5。NVFP4 は 184.3 GiB で 1 台 (121.7 GiB) に収まらない
- **Alternatives Considered**: (1) 1 台では確認できないとして 2 台に進む (要件 5.5 の文言どおり)、(2) 縮小したモデルで確認する
- **Selected Approach**: (2)。`--load-format dummy` (乱数の重み) と `--hf-overrides` で `num_hidden_layers` と `layer_types` を短くし、同じ幾何のまま数分で起動する
- **Rationale**: 要件 5 の目的は「いちばんの懸念に、いちばん安い段階で当たる」ことである。縮小起動は、バックエンドの選択、`concat_and_cache_mla` の経路、kpool の block の整合、MoE のバックエンドの選択を**すべて同じイメージで**踏む。どちらのフラグも一次資料で確認済み
- **Trade-offs**: 乱数の重みなので、出力の意味は確かめられない (要件 5.4 は満たせない)。→ **要件 5.4 は 2 台での起動 (要件 6.5) で満たす**
- **Follow-up**: 縮小の仕方 (何層にするか、`layer_types` をどう切るか) を記録し、再現できるようにする

### Decision 10: 通信の設定は 2 つで始め、足すものは実測の根拠を持つ項目にだけ書ける

- **Context**: 要件 4.5、要件 3.6 / 3.7
- **Selected Approach**: 最小は `VLLM_HOST_IP` (ノードごと)、`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` の 2 系統だけ。**`NCCL_IB_HCA` は設定しない。`NCCL_IB_MERGE_NICS` も `NCCL_IB_GID_INDEX` も触らない。** §e-7 の候補は、構成の定義の中で **`measured` の根拠がないと書けない**ようにする
- **Rationale**: **NVIDIA 自身の 2 台の直結の手順が `NCCL_IB_HCA` を設定していない**。コメントに理由も書かれている (**"NCCL discovers them — no need to name them"**)。`NCCL_IB_MERGE_NICS` の既定 1 が、1 本のケーブルの 2 つの PCIe function を束ねて約 190 Gbps を出す仕組みそのものなので、触ると壊れる。`NCCL_IB_GID_INDEX` は NCCL 2.21 以降「設定してはいけない」と公式が書いている。#41725 では多くの NCCL の変数が「効かなかった」と実測されている
- **Trade-offs**: 最初は最適でない可能性がある (意図どおり。P5 で詰める)。`NCCL_SOCKET_IFNAME` を直結側にするか管理側にするかは、NVIDIA の中でも割れているので A/B が要る
- **Follow-up**: A/B の結果を `docs/results/` に残し、採用したものだけを `measured` 付きで構成に書く。**`NCCL_SOCKET_IFNAME` の 2 案 (直結 / 管理) を最初の A/B にする**

### Decision 11: 第二候補の重みは、モデルカードを読まずに採否を決める

- **Context**: 要件 11.3 が「第三者のレシピ、ブログ、フォーラムにある、起動のスクリプト、パッチ、設定を開かない」と定める。`canada-quant/GLM-5.3-Flash-W4A16-MTP` のモデルカード (41,952 B) は、見出しからして **DGX Spark 2 台の起動レシピそのもの** (`2× DGX Spark GB10 (SM121), TP=2, ...`、`# rank1 (worker) FIRST → wait 25 s → rank0 (head):`、`## Quick start`、`## SM121 kernel-level research findings`) である
- **Alternatives Considered**: (1) カードを読んで必要な引数を得る、(2) カードを読まずに機械可読のメタデータ (`config.json`、`recipe.yaml`、`model.safetensors.index.json`、Hub API) だけで判断する、(3) 候補から外す
- **Selected Approach**: (2)。本調査ではカードの本文を読んでいない (ライセンスの行だけを確認した)。**したがって「このカードが要求する vLLM の引数」は未取得のままにする。** 採用する場合も、引数は `config.json` と上流のソースから自分たちで導く
- **Rationale**: (1) はクリーンルームの根本を壊す。この仕様の成果物を Apache-2.0 で出すときに「ほかのレシピの中身が混ざっていない」と言えなくなる (要件 11.4 / 11.5)。(3) まで行かずに済むのは、必要な事実 (幾何、量子化の形式、大きさ、ライセンス、MTP の有無) がすべて機械可読のファイルから取れるため
- **Trade-offs**: この重みが動くための「勘どころ」を自力で見つけ直すことになる。さらに **`chat_template.jinja` がベンダの 2026-08-31 版で古い** (複数ツールの結果の並べ替えの修正が入っていない) ので、使うなら `--chat-template` でベンダの現行版を明示する必要がある
- **Follow-up**: 代替の段でこの重みに進むとき、計測者に「カードを読まずに進めてよいか」を確認する。読むと決めるなら、**その時点でクリーンルームの条件が変わったことを `docs/decisions/` に記録する**

---

## Design Synthesis (2026-09-21、設計の前の統合)

この節は、調査を行ったサブエージェントではなく、設計を書くメインのセッションが足した。メインのセッションは `exl3-tp2` の中身を見ていない。ここに書くのは、上の調査の結果の整理だけで、新しい設定の値は足していない。

**計測者の判断 (2026-09-21)**: Decision 8 (代替の順序の組み替え) は採用。要件 8.1 / 5.1 / 5.4 / 5.5 を改訂し、8.7〜8.9 を足した (commit 9ef989d)。Decision 1 (Python の CLI) は採用。Decision 11 (第二の候補の重みは、モデルカードを読まずに候補に残す) は採用。

### 1. 一般化

- **1 台の縮小の確認、通信の確認、本番の起動は、同じ 1 つの仕事の変形である**: 「固定したイメージから、ラベルを付けたコンテナを、根拠の検査を通った引数で、決めたノードの上に起こし、ある条件を待ち、記録を回収し、片付ける」。違うのは、待つ条件だけ (受け付けの開始 / 起動の成否の確定 / スクリプトの終了)。→ 引数の組み立て (純粋な関数) と、実行 (ssh 越し) を 1 組だけ作り、構成の `kind` (`serve` / `probe` / `job`) で待ち方を切り替える。確認ごとに別の起動の仕組みを作らない
- **起動の前の関門 (了承、よそのプロセス、ディスクの空き、イメージの識別子、重みの照合、根拠の検査) は、すべて「通す / 断る + 理由」を返す検査である**。→ 結果の型を 1 つにし、`serve check` と `serve start` が同じ検査の列を使う
- **起動の記録 (要件 3.10)、試行の記録 (8.2)、動かなかった箇所の再現の条件 (10.1) は、同じ 3 つ組 (構成の名前、イメージの識別子、重みの版) を持つ**。→ 起動の記録を唯一の出どころにし、文書の側はそれを引く
- **起動の記録からの読み取り (選ばれた部品、KV の大きさ) は、1 台の確認 (5.2) と 2 台の起動 (6.3) で同じ**。→ 読み取りの部品は 1 つ

### 2. 作るか、借りるか

| 対象 | 判断 | 理由 |
|---|---|---|
| 重みの取得 | **借りる**: `hf download` | Spark のホストには入れず、固定したイメージのコンテナの中で動かす (要件 2.7) |
| 重みの照合 | **作る**: Spark で `sha256sum` を流し、Mac でコミットしたマニフェストと突き合わせる | `hf cache verify` (§b-4、Decision 3) は Hub に問い合わせる照合で、要件 3.4 / 3.5 が求める「固定した検査の値」との照合にならない。Hub の側が変わっても、手元のマニフェストとの照合は変わらない。**Decision 3 の照合の部分を、この判断で置き換える** |
| 自分のコンテナの識別 | **借りる**: Docker のラベル | §c、Decision 4 |
| 2 台での事前の確認 (要件 4.6) | **借りる**: vLLM のトラブルシュートの文書の確認のスクリプトを、出典と commit を添えて、そのまま使う | 要件 4.6 が求めているのは「公式の資料が示す確認を流す」ことで、似たものを自作すると、公式の確認が通ったと言えなくなる。vLLM は Apache-2.0。要件 11.2 が写すことを禁じているのは、手引きと手順集の「起動のコマンドと設定のファイル」で、診断のスクリプトは当たらないと判断した。**設計のレビューで計測者に確かめる** |
| 2 台の帯域の測定 | **作る**: torchrun で動かす、自前の all-reduce の計測 | Decision 7。nccl-tests は MPI が要り、Spark の恒久的な変更になる |
| ssh / rsync の呼び出し | **借りる**: システムの `ssh` / `rsync` を `subprocess` で、引数のリストで呼ぶ | `~/.ssh/config` の定義がそのまま効く。paramiko は LGPL で方針から外れる。fabric はその上に載る |
| 2 時間の連続の負荷 (要件 7.7) | **負荷は借りる** (`bench` を繰り返し流す)。**見張りは作る** (応答、GPU の使われ方、生成の進みを一定の間隔で記録し、固まりを判定する) | `bench` の中身は変えない (範囲の外)。固まりの判定 (#41725 の形: GPU は使われ続けているのに進まない) は `bench` の外から見る必要がある |
| thinking の深さの確かめ (要件 9.2) | **作る**: 5 通りを送り分ける小さな確認 | `bench` は `server_default` しか送れない (§g) |

### 3. 単純化

- **Baseline Procedure はソフトウェアにしない。** 手順書 (段の順序、止める条件、記録の書式) と、Serving Kit のコマンドの組み合わせで行う。代替の順序を進める状態機械は作らない。段を進める判断は、計測者への確認を挟む (要件 8.5 / 8.8 / 8.9) ので、自動化する利点がない
- **コンテナの実行の仕組みは Docker だけ。** 差し替えの口を作らない
- **ノードは head と worker の 2 つに固定。** N 台への一般化はしない。ノードの定義 (ssh の名前、LAN のアドレス、直結のアドレスとインターフェース名) は 1 つのファイルに置き、直結の側の値は通信の確認の実測で埋める
- **構成の継承の仕組みは作らない。** 構成が増えても 5 つ前後なので、TOML に並べて書く。共通の部分を括り出すと、「設定 1 つごとに根拠」の検査が読みにくくなる
- **起動の順序を制御しない。** 2 台を続けて起こし、両方が揃うのを待つ (§d-3)。止めるときだけ head → worker の順にする (§f-3)

---

## Risks & Mitigations

| リスク | 度合い | 備え |
|---|---|---|
| **パッチなしでは sm_121 で起動できない** (§d-1)。P1 の終わりの条件を満たせない | **高** (現時点の根拠では、こうなる公算が大きい) | Decision 8 の段 0 で、重みを落とす前に安く確かめる。起動できなければ要件 8.5 の手順で、止まった場所と最小の変更 (#55277 / #53969) を記録して計測者に判断を仰ぐ。終わりの条件の改訂を早い段階で諮る |
| issue #57087 の報告 (同じ 2 台 GB10 で動いている) と分析が矛盾している | 中 | これは**良い方向の**不確かさ。段 0 の結果で決着する。動いたら、なぜ動いたか (イメージの commit の差) を記録する |
| `TORCH_CUDA_ARCH_LIST` に 12.1 がない | 中 | 段 0 で `no kernel image is available` が出るかどうかで分かる。出たら、`12.1` を含むイメージを探すか、自前ビルドの是非を計測者に諮る (要件 8.6 により、勝手にはビルドしない) |
| メモリが足りず起動しない (重み約 92 GiB + KDA + 活性化) | 中 | `--max-num-seqs 16`、`--max-num-batched-tokens 2048`、`--language-model-only`、`--gpu-memory-utilization 0.90` から始める。起動ログの内訳を読んで詰める。足りなければ第二の重み (W4A16、7 GiB 小さい) に替える |
| `--gpu-memory-utilization` の起動時の検査に落ちる (UMA で `psutil.available` が要求を下回る) | 中 | 0.90 から始める。重みの取得の直後はページキャッシュの状態が変わるので、**起動の前に一呼吸置く**か、失敗したら下げて再試行する |
| 2 台 TP が 35〜55 分で固まる (#41725) | 中 | イメージの NCCL が 2.30.7 で、#52504 の原因となった版を上回っている。要件 7.7 の 2 時間の連続の負荷で確かめる。vLLM の事前の確認の 4 段目 (CUDA グラフの中の NCCL) を先に通す |
| 同時実行で日本語が文字化けする (#57087) | 中 | 要件 7.6 のとおり `bench --suite concurrency` の「壊れている疑い」の印の数を同時本数ごとに記録する。`bench` はすでに U+FFFD と繰り返しを検出する |
| FlashInfer の autotune で起動が止まる (#52291) | 中 | 起動のタイムアウトで検知し、`--no-enable-flashinfer-autotune` を足して A/B する (要件 8.4 のとおり失うものを測る) |
| NVFP4 の MoE カーネルが自動で Marlin になり不安定 (#54666 / #50934) | 中 | 起動ログの `Using '...' NvFp4 MoE backend out of potential backends: [...]` を必ず記録する。問題が出たら `--moe-backend` を明示して A/B する |
| **vLLM 公式の手引きが、このモデルを GB10 で検証していない** | 中 | recipes.vllm.ai の対応表は `h100 / b200 / gb200 / mi355x / ascend_950pr_8x` を "verified" とし、**DGX Spark / GB10 を含まない** (同サイトは他のモデルでは `dgx_spark_gb10` の欄を持つので、欄がないのではなく検証されていない)。→ 「公式が動くと言っている構成」ではないことを前提に置き、段 0 で早く確かめる |
| **イメージの中の NCCL が sm_121 のデバイスコードを含まないかもしれない** | 中 | NVIDIA 自身の手順は NCCL を `-gencode=arch=compute_121,code=sm_121` でビルドし直している。イメージの NCCL 2.30.7 がどのアーキで焼かれているかは不明 (UNVERIFIED)。→ **vLLM の事前の確認 (`test.py`) の 1 段目が通れば、少なくとも動いている証拠になる**。通らなければ、NCCL のビルドし直しが要ることの記録になる |
| チャットテンプレートが古い重みを使ってしまう | 低 | `chat_template.jinja` の sha256 をベンダの現行版と突き合わせ、違えば `--chat-template` で明示する。3 つの候補の sha256 は §b-7 に記録した |
| PLAN.md のハードウェアの記述が実態と違う (直結リンクの本数、MTU) | 低 (**原因は特定済み**) | §e-1 のとおり、NVIDIA の公式資料で「ケーブル 1 本」「MTU は沈黙、例は 1500」「189.85 Gbps」が確認できた。要件 4.1 で実測して PLAN.md を訂正する。**MTU 9000 と「2 本のリンク」は DGX Station (ConnectX-8) の話**で、混ぜないこと |
| `--device /dev/infiniband` や `--ulimit memlock` の要否が公式に書かれていない | 低 | 付けない状態で起動し、NCCL のログで `NET/IB` が出るかを見る。出なければ足して A/B し、`measured` の根拠を付けて構成に書く |
| 温度 0 でも同じ出力が返らない (#54521、sm121/GB10、OPEN) | 低 | P0 の判断のとおり「同じ入力を送る」ことだけを前提にする (`docs/decisions/0001-…`)。品質の比較は相対で行う |

---

## Open questions that only hardware can answer (実装のタスクの確認表)

すべて **【実機で決める】**。左から順に、安い順に並べてある。

| # | 問い | 測り方 | どの要件・判断に効くか |
|---|---|---|---|
| 1 | **固定したイメージで `pe_dim must be 64` の assert が出るか** | 段 0 の縮小起動 (`--load-format dummy`)。起動ログ全文を保存 | **P1 全体の分岐点**。要件 5.3 / 8.5 |
| 2 | 選ばれたアテンションのバックエンド、または全滅した理由 | 同上。`Using %s attention backend out of potential backends: %s.` / `No valid attention backend found …` | 要件 5.2 |
| 3 | sm_120 向けのバイナリが sm_121 で動くか | 同上。`no kernel image is available for execution on the device` が出るか | イメージの選定 |
| 4 | `--block-size` を指定しないと kpool の assert が出るか | 同上 | §d-6 の `--block-size 256` の要否 |
| 5 | 直結リンクの本数、名前、MTU、リンク速度 | `ip -br link`、`ethtool`、`ibv_devinfo` | 要件 4.1 / 4.2、PLAN.md の訂正 |
| 6 | RDMA デバイス名 (`mlx5_N`) と netdev 名の対応 | `ibv_devinfo` | `NCCL_IB_HCA` の値 |
| 7 | 最小の設定での 2 台の帯域 (busbw) | Decision 7 の torchrun による計測 | 要件 4.3 |
| 8 | NCCL が RoCE を使っているか、デバイスが束ねられているか | `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH` の全文 | 要件 4.4 / 4.7 |
| 9 | vLLM の事前の確認 4 段すべてが通るか | トラブルシュートの `test.py` を torchrun で | 要件 4.6 / 4.7 |
| 10 | コンテナの中で `ulimit -l` が無制限になるか | コンテナの中で `ulimit -l` | `--ulimit memlock` の書き方 |
| 11 | `--device /dev/infiniband` なしで `NET/IB` が出るか | NCCL のログ | docker の引数 |
| 12 | 重みの取得がシンボリックリンクになるか | 取得後に `ls -l` | 配布の方式 |
| 13 | ~~MTP のファイルが index に載っているか~~ | **解決済み (紙で確認)**: 載っている。取得は省けない (§b-6) | — |
| 14 | ~~チャットテンプレートが読む変数~~ | **解決済み (紙で確認)**: `reasoning_effort` (low/high/max) と `clear_thinking` (履歴の掃除) だけ。thinking は切れない (§g-2) | — |
| 14b | `hf` (huggingface_hub の CLI) を Spark でどう動かすか | ホストに入れずに、固定したイメージか別の軽いコンテナの中で動かせるかを確かめる | 要件 2.7、Decision 3 |
| 14c | `NCCL_SOCKET_IFNAME` は直結側と管理側のどちらが速いか | 2 案で `test.py` と帯域の計測を流して比べる | Decision 10、要件 4.5 |
| 14d | イメージの NCCL が sm_121 で動くか | `test.py` の 1 段目 (PyTorch NCCL の all-reduce) が通るか | 上のリスク |
| 14e | TP=2 と PP=2 のどちらが速いか | 起動できた後、同じ `bench` の条件で比べる | §d-3、Decision 8 |
| 15 | 重みのロード時間と、起動全体の所要 | `Model loading took …` / `init engine … took …` | 要件 1.4 のタイムアウトの値 |
| 16 | `--max-num-seqs` の既定が実際にいくつになるか | 既定のまま起動して設定のダンプを読む | Decision 6 の裏づけ |
| 17 | KDA の状態と KV の実際の内訳 | `Actual usage is … for weight, … for peak activation, …` / `GPU KV cache size: N tokens` / `Available KV cache memory: N GiB` | 要件 6.3 / 6.4 |
| 18 | `--gpu-memory-utilization` をどこまで上げられるか | 0.90 → 0.92 → 0.94 と上げて `Available KV cache memory` の増分を測る | §d-6 |
| 19 | thinking の深さが `output_config.effort` で変わるか。`clear_thinking` が長い会話の入力を減らすか | §g の **5 通り**の送り分け (期待値つき) | 要件 9.2 / 9.5 |
| 20 | 2 時間の連続の負荷で固まるか | 要件 7.7 | #41725 の再現の有無 |
| 21 | 同時実行で日本語が壊れるか | `bench --suite concurrency` の印の数 | 要件 7.6、#57087 の再現の有無 |
| 22 | イメージの中のライセンス表記 | 取得後にコンテナの中のファイルを読む | 要件 3.8 / 3.9 |

---

## References

**vLLM — 文書**

- 並列とスケーリング (マルチノード、GPUDirect RDMA、ネットワークの安全) — https://docs.vllm.ai/en/latest/serving/parallelism_scaling/
- トラブルシュート (`test.py` の事前の確認) — https://docs.vllm.ai/en/latest/usage/troubleshooting/
- 分散デプロイのトラブルシュート — https://docs.vllm.ai/en/latest/serving/distributed_troubleshooting/
- Docker でのデプロイ — https://docs.vllm.ai/en/latest/deployment/docker/
- GPU のインストール — https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- `vllm serve` の CLI リファレンス — https://docs.vllm.ai/en/latest/cli/serve/
- エンジンの引数 — https://docs.vllm.ai/en/latest/configuration/engine_args/
- オンラインの提供 (口の一覧) — https://docs.vllm.ai/en/latest/serving/online_serving/

**vLLM — ソース (main、2026-09-20)**

- `vllm/platforms/cuda.py` (バックエンドの優先順位、kpool の block の整合) — https://github.com/vllm-project/vllm/blob/main/vllm/platforms/cuda.py
- `vllm/v1/attention/backend.py` (バックエンドの検証) / `vllm/v1/attention/selector.py`
- `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` / `flashinfer_mla_sparse_sm120.py`
- `vllm/model_executor/layers/attention/mla_attention.py` (`_canonicalize_sparse_mla_kv_cache_dtype`)
- `csrc/libtorch_stable/cache_kernels.cu` (`pe_dim must be 64`)
- `vllm/models/glm5next/` (モデルの実装、kpool indexer、KDA、MTP)
- `vllm/transformers_utils/configs/glm5_next.py` (設定の定義)
- `vllm/config/{cache,scheduler,parallel,model,load,multimodal,attention,vllm}.py`
- `vllm/engine/arg_utils.py` / `vllm/entrypoints/launchers/cli_args.py` / `vllm/entrypoints/cli/serve.py`
- `vllm/v1/worker/gpu_worker.py` / `vllm/v1/worker/utils.py` / `vllm/utils/mem_utils.py` (メモリの勘定、UMA)
- `vllm/model_executor/layers/mamba/mamba_utils.py` (KDA の状態の形)
- `vllm/entrypoints/anthropic/{api_router,protocol,serving}.py` (`/v1/messages`)
- `vllm/entrypoints/serve/instrumentator/{health,basic,metrics}.py`
- `vllm/v1/metrics/loggers.py` / `vllm/v1/spec_decode/metrics.py`
- `vllm/envs.py`
- `docker/Dockerfile` / `docker/versions.json`

**vLLM — issue と PR (2026-09-21 に状態を確認)**

- #57578 (OPEN) GLM-5.3-Flash が `pe_dim must be 64` で落ちる — https://github.com/vllm-project/vllm/issues/57578
- #55773 (OPEN) 同上の元の報告 — https://github.com/vllm-project/vllm/issues/55773
- #53963 (OPEN) sm_120 での 4 つの失敗の形 — https://github.com/vllm-project/vllm/issues/53963
- #55277 (OPEN / CONFLICTING) C++ を直す PR — https://github.com/vllm-project/vllm/pull/55277
- #53969 (OPEN / CONFLICTING) Python のゼロ詰めの PR — https://github.com/vllm-project/vllm/pull/53969
- #55778 (OPEN) 別のゼロ詰めの PR — https://github.com/vllm-project/vllm/pull/55778
- #54929 (OPEN) sm_12x の Triton sparse MLA (NoPE は対象外と明記) — https://github.com/vllm-project/vllm/pull/54929
- #53906 (**MERGED** 2026-09-03) GLM-5.3-Flash の対応。v0.29.0 には入っておらず、v0.29.1rc0 以降 — https://github.com/vllm-project/vllm/pull/53906
- #57156 (OPEN) sm_120/121 の CUDA グラフでの NaN — https://github.com/vllm-project/vllm/issues/57156
- #57087 (OPEN) GB10 2 台 TP=2 での非 ASCII の文字化け — https://github.com/vllm-project/vllm/issues/57087
- #41725 (OPEN) DGX Spark 2 台 TP=2 の 35〜55 分での固まり — https://github.com/vllm-project/vllm/issues/41725
- #51921 (OPEN) 4 ノード TP=4 の shm_broadcast の停止 — https://github.com/vllm-project/vllm/issues/51921
- #52291 (OPEN) FlashInfer の autotune がマルチノードでデッドロック — https://github.com/vllm-project/vllm/issues/52291
- #52504 (OPEN) NCCL 2.28.9 でのグラフのデッドロック、2.30.4 で解消 — https://github.com/vllm-project/vllm/issues/52504
- #54666 (OPEN) NVFP4 MoE の自動選択に B12X が入っていない — https://github.com/vllm-project/vllm/issues/54666
- #50934 (OPEN) GB10 の 10 日稼働後の misaligned address — https://github.com/vllm-project/vllm/issues/50934
- #47365 (OPEN) NVFP4 の flashinfer_b12x で出力が壊れる — https://github.com/vllm-project/vllm/issues/47365
- #49079 (OPEN) flashinfer_b12x で活性化のピークが倍増 — https://github.com/vllm-project/vllm/issues/49079
- #54521 (OPEN) sm121/GB10 で greedy デコードが非決定的 — https://github.com/vllm-project/vllm/issues/54521

**vLLM — 手引き (事実のみ)**

- GLM-5.3-Flash のレシピ — https://recipes.vllm.ai/zai-org/GLM-5.3-Flash
  - 「約 321B / 1 トークンあたり 18B」「45 層が KDA の線形アテンションと NoPE sparse MLA の組み合わせ」「288 のエキスパートのうち 8 を通る」「1,048,576 の文脈」
  - 「既定の FP8 のチェックポイントは約 306 GiB」「NVFP4 版は `RedHatAI/GLM-5.3-Flash-NVFP4`、Blackwell が要る」
  - 「NoPE sparse MLA には FlashInfer 0.6.17 以降が必須」「sparse-MLA の初期化のエラーが出たら、イメージの FlashInfer が 0.6.18 以降かを確かめよ」
  - パーサーの名前として `glm47` を使っている
  - **注意**: 「vLLM 0.29.0+」と書いているが、v0.29.0 のタグにはモデルの実装が入っていない (本調査で確認)

**Hugging Face**

- `zai-org/GLM-5.3-Flash` — https://huggingface.co/zai-org/GLM-5.3-Flash (MIT、`LICENSE` 同梱、305.8 GiB、`sha = eb9eb208…`)。`chat_template.jinja` の sha256 = `0c4099f3…`
- `RedHatAI/GLM-5.3-Flash-NVFP4` — https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4 (MIT、184.3 GiB、`sha = 18d55bfd…`)。テンプレートはベンダ版と一致
- `canada-quant/GLM-5.3-Flash-W4A16-MTP` — https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP (MIT、177.7 GiB、`sha = f0870306…`)。**カード本文は未読 (Decision 11)**。テンプレートの sha256 = `34d5ee66…` (ベンダの 2026-08-31 版)
- Hub のファイル一覧 (公式 API、`lfs.oid` = sha256) — `https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1`
- Hub のダウンロードの手引き / CLI / キャッシュの管理 / 環境変数 — https://huggingface.co/docs/huggingface_hub/en/guides/download 、https://huggingface.co/docs/huggingface_hub/en/guides/cli 、https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache 、https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
- `huggingface_hub` (Apache-2.0、CLI は `hf`) — https://github.com/huggingface/huggingface_hub

**NVIDIA**

- DGX Spark ユーザーガイド / ConnectX-7 Networking (ポート 2 つ、1 ポート = 2 インターフェース、命名表) — https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html
- DGX Spark ハードウェアの概要 (128 GB / 273 GB/s / 6,144 CUDA コア) — https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- DGX Spark 移植ガイド / CUDA (GPUDirect RDMA が非対応) — https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html
- DGX Spark 移植ガイド / ビルド (`121-real`) — https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/compilation.html
- DGX Spark ユーザーガイド / コンテナランタイム — https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html
- NVIDIA Sync クラスタアシスタント (ケーブル 1 本、184 Gbit/s の下限) — https://docs.nvidia.com/sync/latest/cluster-assistant.html
- CUDA GPUs の一覧 (GB10 = 12.1) — https://developer.nvidia.com/cuda-gpus
- NVIDIA/dgx-spark-playbooks (Apache-2.0。**事実だけを使い、コマンドは写していない**) — https://github.com/NVIDIA/dgx-spark-playbooks
- NCCL 環境変数リファレンス — https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- NCCL トラブルシュート / ネットワーク・実行時 (memlock、GID、cuMem と NUMA) — https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/
- NCCL ソース (ログの書式文字列) — https://github.com/NVIDIA/nccl
- NVIDIA Container Toolkit (`--gpus` と `NVIDIA_VISIBLE_DEVICES`) — https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
- NVIDIA XLIO / Docker の "Required Configurations" (`--device=/dev/infiniband`、`--ulimit memlock=-1`) — https://docs.nvidia.com/networking/display/xliov361/Setting-Up-XLIO-Within-a-Docker-Container
- NVIDIA Dynamo / RDMA・InfiniBand (`IPC_LOCK` は memlock があれば不要) — https://docs.nvidia.com/dynamo/v1.2.1/kubernetes-deployment/cloud-provider-guides/azure/rdma-infini-band
- nccl-tests (BSD、マルチノードには MPI が要る、`doc/PERFORMANCE.md` の `2(n-1)/n`) — https://github.com/NVIDIA/nccl-tests

**Docker**

- `docker run` の CLI リファレンス — https://docs.docker.com/reference/cli/docker/container/run/
- ホストネットワーク — https://docs.docker.com/engine/network/drivers/host/
- 実行時の権限と Linux の capability (`IPC_LOCK`) — https://docs.docker.com/engine/containers/run/#runtime-privilege-and-linux-capabilities
- bind マウント (読み取り専用) — https://docs.docker.com/engine/storage/bind-mounts/
- `docker container ls` のフィルタ (label / name) — https://docs.docker.com/reference/cli/docker/container/ls/
- `docker stop` — https://docs.docker.com/reference/cli/docker/container/stop/
- `docker logs` — https://docs.docker.com/reference/cli/docker/container/logs/
- `docker image pull` (ダイジェストでの固定) — https://docs.docker.com/reference/cli/docker/image/pull/
- `docker image ls --digests` — https://docs.docker.com/reference/cli/docker/image/ls/
- `docker buildx imagetools inspect` — https://docs.docker.com/reference/cli/docker/buildx/imagetools/inspect/
- Engine API (`RepoDigests` の定義) — https://docs.docker.com/reference/api/engine/version/v1.50/
- Compose の適用範囲 (単一ホスト) — https://docs.docker.com/compose/intro/features-uses/

**その他**

- FlashInfer (sm120 の sparse MLA、GLM53_NOPE の系統) — https://github.com/flashinfer-ai/flashinfer
- Ansible のライセンス (GPLv3) と管理対象ノードの要件 — https://github.com/ansible/ansible/blob/devel/COPYING 、https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html

**このリポジトリ**

- `PLAN.md` (§2 ライセンスとクリーンルーム、ハードウェアの事実、§4 P1)
- `CLAUDE.md` (どこで何をするか)
- `.kiro/specs/vllm-baseline/brief.md` / `requirements.md`
- `bench/README.md` (対象サーバーに求める前提)、`bench/config/targets.toml`、`bench/config/profiles.toml`
- `docs/decisions/0001-bench-harness-measurement-method.md` (thinking の切り替え、sm121 の非決定性)
- `scripts/spark-precheck.sh`
- `LICENSES.md`
</content>
</invoke>
