# 0002. vLLM のイメージと、重みの選定

状態: 部分的に採用 (2026-09-22)。イメージと第一候補の重みの選定そのものは確定し、
`serving/config/configs.toml` と `serving/weights/RedHatAI__GLM-5.3-Flash-NVFP4.manifest.json`
に反映済みである。イメージの中のライセンスの表記の確認 (要件 3.8、3.9) は 7.1 で済み、
ベースのイメージの NVIDIA の EULA は計測者の判断で受け入れた (下の「実機で確かめたこと」)。
実機での起動の結果 (段 0 以降) は、7.2 以降で追記する。

関係する要件: 3.2、3.4、3.8、3.9、6.4、8.8、10.3、11.1、11.2、11.3、11.4、11.5、11.6。
仕様は `.kiro/specs/vllm-baseline/`。手順は
[`docs/vllm-baseline/procedure.md`](../vllm-baseline/procedure.md)。

書かないこと (要件 10.5): **送った内容と応答の本文**(書くのは長さ、トークンの数、終わりの
理由、HTTP の状態だけ)、**認証の情報**(鍵、トークン。Spark にも置かない。要件 2.6)、
**計測者が別に起動していた構成 (`exl3-tp2`) の中身**(起動の引数、設定、差し込まれた
ファイル、記録。読んでよいのは、GPU を使っているプロセスの名前とメモリの量だけである)は、
この記録に書かない。

## 背景

要件 3.2 / 3.3 が「あとから中身が変わらない識別子での固定」と「起動時の照合」を求め、
要件 3.4 / 3.5 が重みについて同じことを求める。要件 11.2 は「公式の手引きのコマンドを
写さない」ことを求めるので、選定は一次資料 (Docker Hub のレジストリ API、vLLM のソースと
公式文書、Hugging Face Hub API、各リポジトリの `config.json`) だけから行った
(research.md §a、§b)。

DGX Spark 2 台のユニファイドメモリは 1 台あたり約 121.7 GiB (PLAN.md、実測)。第一候補の
重みは NVFP4 で 184.3 GiB、TP=2 で 1 台あたり約 92 GiB になる。この形式でも、モデルの
アテンションの幾何 (`qk_rope_head_dim = 0` の NoPE MLA + sparse indexer + 線形アテンション)
は変わらない (research.md §b-3)。この幾何を、選んだイメージの vLLM が受け付けるかどうかが、
P1 いちばんの懸念であり、重みの形式にも並列の取り方にも依らない (research.md §d-1、
Decision 8)。

## 決めたこと

### 1. イメージ

| 項目 | 値 |
|---|---|
| `ref` | `vllm/vllm-openai@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5` |
| `seen_as` (人間向けの注記。固定には使わない) | `vllm/vllm-openai:glm53-flash-arm64-cu130` |
| 大きさ | 9,666,567,584 B |
| arch | linux/arm64 |
| 最終 push | 2026-09-09T13:31:25Z |
| build commit (`ai.vllm.build.commit` ラベル) | `385dce36bcee42309924a5ece951a96db3dce7f2` |
| 中の CUDA / NCCL / FlashInfer | 13.0.3 / **2.30.7** / 0.6.18.post1 |
| ベースイメージ | `nvidia/cuda:13.0.3-base-ubuntu24.04` |

- `glm53-flash-arm64-cu130` は、モデル別の独自ビルドではなく、ある nightly の別名である
  (レジストリのラベルから確認)。GLM 向けの特別な修正は入っていない
  (research.md §a-2)。それでも選ぶ理由は、NCCL 2.30.7 を含む点にある。issue #52504
  (2× DGX Spark、TP=2、200G RoCE の CUDA グラフ内デッドロック) の原文は
  「upgrading NCCL 2.28.9 → 2.30.4 fixes the deadlock with graphs re-enabled」であり、
  このイメージは最初からその修正の上流に立つ
- 単一アーキ (arm64) のタグのダイジェストで固定する。マニフェストリスト
  (`vllm/vllm-openai:glm53-flash`) のダイジェストとは別物なので、単一アーキのほうが
  照合が 1 対 1 になる (research.md §a-5)
- `--pull never` を付け、固定したダイジェストのイメージが手元にないときに黙って取得されない
  ようにする (Docker の文書: "Do not pull the image, even if it's missing, and produce an
  error if the image does not exist in the image cache.")
- vLLM 本体は Apache-2.0 (使える)。ベースイメージ (`nvidia/cuda`) の実際のライセンス表記は、
  7.1 で読んだ (下の「実機で確かめたこと」)。**NVIDIA Deep Learning Container License** で、
  方針 (MIT / Apache-2.0 / BSD 系) の外にあるが、CUDA を使う限り不可避であり、この用途
  (自分の設備での推論。イメージを再配布しない) が EULA の許す範囲に収まることを確かめて、
  **計測者の判断 (2026-09-22) で受け入れた** (要件 3.9。`LICENSES.md` に行と理由がある)

### 2. 重み (第一候補)

| 項目 | 値 |
|---|---|
| `repo` | `RedHatAI/GLM-5.3-Flash-NVFP4` |
| `revision` (40 桁 commit sha) | `18d55bfd5a2194887738da73753975c9d3842f46` |
| ライセンス | MIT (front matter の申告による継承。`LICENSE` ファイルは同梱されていない) |
| 量子化 | `compressed-tensors` / `nvfp4-pack-quantized` / `format: mixed-precision` |
| マニフェスト (要件 6.1 でコミット済み) | `serving/weights/RedHatAI__GLM-5.3-Flash-NVFP4.manifest.json`。19 ファイル、197,881,153,655 バイト (README と `.gitattributes` を除いた合計。Hub API の生の合計は 197,881,158,759 バイト) |

- **選定の理由 (research.md §b の要約)**: 3 つの候補 (`zai-org/GLM-5.3-Flash` 305.8 GiB fp8、
  `RedHatAI/GLM-5.3-Flash-NVFP4` 184.3 GiB、`canada-quant/GLM-5.3-Flash-W4A16-MTP`
  177.7 GiB) は、すべて MIT で、アテンションの幾何が同一である
  (`config.json` を直接読んで確認)。`RedHatAI` 版を選んだのは、recipes.vllm.ai が NVFP4 版
  として名指ししており、チャットテンプレート (`chat_template.jinja`) がベンダの現行版と
  バイト単位で一致し (sha256 `0c4099f3…`)、MTP の重みを同梱するため
  (`model_mtp.safetensors`。省くとロードが壊れる。研究段階の前の見立ての訂正)
- **固定と照合の方法**: 版は 40 桁の commit sha (短縮形は使えない)。ファイルごとの検査の値は、
  公式の `tree` API (`GET /api/models/{repo}/tree/{rev}?recursive=1`) の `lfs.oid` を、
  Mac で `serve manifest` が読んでマニフェストに固定する。Spark 側の照合は `hf cache verify`
  相当 (要件 3.5)
- **取得方法**: 各 Spark が公開の Hub から直接、匿名で取得する (トークン不要。要件 2.6)。
  `--local-dir` を使い、既定のキャッシュのシンボリックリンクと `CACHEDIR.TAG` を避ける
  (research.md §b-4、§b-4b)

### 3. 第二候補の重みの扱い (要件 8.8、Decision 11)

`canada-quant/GLM-5.3-Flash-W4A16-MTP` (177.7 GiB、MIT) は、代替の順序 (要件 8.1、段 4)
で候補として残すが、**この調査では、モデルカードの本文を開いていない**。見出しからして
DGX Spark 2 台の起動レシピそのもの (`2× DGX Spark GB10 (SM121), TP=2, ...`、
`# rank1 (worker) FIRST → wait 25 s → rank0 (head):` など) であり、要件 11.3 が開かない
と定める第三者のレシピに当たる (research.md §b-7)。段 4 に進む場合も、構成は
`config.json` と上流のソースだけから導く (要件 8.8)。このモデルのチャットテンプレートは
ベンダの 2026-08-31 版で古い (複数ツールの結果の並べ替えの修正が未取り込み) ので、採用
する場合は `--chat-template` で現行版を明示する必要がある。

### 4. 見つかった誤りの訂正

構成の値を書く作業 (タスク 6.2) の中で、design.md の下書きと research.md の一部に、次の
誤りが見つかった。

- **`--ulimit memlock=-1` の出典**: design.md 863-864 行が引用していた
  「A container does not inherit the ulimits set on the host…」という quote は、
  **NCCL の公式文書には存在せず、NVIDIA XLIO の手引き由来だった**。構成
  (`configs.toml` の `docker.ulimit-memlock`) では、NCCL 自身の Docker の節
  (`troubleshooting/runtime_and_mpi_issues.html`) の quote に差し替えた。値
  (`memlock=-1`) は変えていない
- **`--ulimit stack=67108864` は入れていない**: design.md の最初の構成の表にはあるが、
  64 MiB という値の唯一の出どころが NVIDIA NGC の手引きの**コマンド**であり、要件 11.2
  (「公式の手引きのコマンドを写さない」) に当たる。NCCL の一次資料が述べているのは
  「スタックの上限を無制限にすると落ちる事例がある」ことだけで、コンテナの既定はもともと
  有限なので、この指定は積極的に上げる側になる。実測の根拠 (`measured`) が付くまでは
  足さない
- **research.md §d-4 の記述の訂正**: 「flashinfer autotune は O1 以上で既定 True」という
  記述は、ソース (`vllm/config/kernel.py` の `enable_flashinfer_autotune: bool = None`)
  と食い違う。既定は `None` (環境によって挙動が変わりうる) であり、「既定 True」ではない。
  構成では `--no-enable-flashinfer-autotune` を明示して、この曖昧さを断つことにした
  (`configs.toml` の `args.no-enable-flashinfer-autotune`)

## 採らなかった案

| 案 | 採らなかった理由 |
|---|---|
| `zai-org/GLM-5.3-Flash` (fp8、305.8 GiB) を第一候補にする | 1 台あたり約 153 GiB で、TP=2 でも重みだけで大半のユニファイドメモリを使い、KV とアクティベーションの余地が乏しい。NVFP4 版が MIT で、幾何が同一で、より小さい |
| マニフェストリスト (`vllm/vllm-openai:glm53-flash`) のダイジェストで固定する | arm64 単体のダイジェストと別物で、照合が 1 対多になる。単一アーキのタグのほうが確実 |
| `nightly-aarch64` / `cu134-nightly-aarch64` を最初から使う | 日々更新されるタグで、`cu134` 系はドライバ要件が上がる恐れがある。より新しい上流イメージは、段 0 が失敗したときの段 1 の候補として残す |
| 第二候補のモデルカードを読んで、要る vLLM の引数を得る | クリーンルームの根本を壊す (要件 11.3 / 11.4 / 11.5)。機械可読のファイル (`config.json`、`recipe.yaml`、Hub API) だけで、必要な事実 (幾何、量子化の形式、大きさ、ライセンス、MTP の有無) はすべて取れる (Decision 11) |
| `--ulimit stack=67108864` を design.md のまま入れる | 出どころが NGC の手引きの**コマンド**で、要件 11.2 に当たる。上記「見つかった誤りの訂正」を参照 |

## 影響と限界

- 選んだイメージ (`glm53-flash-arm64-cu130`) は GLM 向けの独自修正を持たない、ある nightly
  の別名である。将来、上流がこのタグを更新しても、この構成の `image.ref` (ダイジェスト) は
  自動では変わらない (意図した固定)
- `TORCH_CUDA_ARCH_LIST` に `12.1` (sm_121) が明示されていない。sm_120 向けのバイナリが
  sm_121 で動くかどうかは、上流の資料からは断定できず、段 0 の実機の結果で確かめる
  (research.md §a-3)
- 第二候補 (`canada-quant`) を選んだ場合、モデルカードを読まずに引数を導くため、「この重みが
  動くための勘どころ」を自力で見つけ直す必要がある (Decision 11 の Trade-offs)

## 見直す条件

次のどれかをしたら、この記録を見直す。

| 変えたもの | すること |
|---|---|
| 第二候補のモデルカードを読むと決めた場合 | クリーンルームの条件が変わったことを、この記録に追記する (research.md Decision 11 の Follow-up) |
| 段 0 / 段 1 で、より新しい上流イメージに固定し直した場合 (procedure.md 段 1) | この記録の「イメージ」の節に、新しいダイジェストと理由を追記する |
| `--ulimit stack` に実測の根拠 (`measured`) ができた場合 | 「採らなかった案」から「決めたこと」に移す |
| イメージの中のライセンス表記が、想定 (MIT / Apache-2.0 / BSD 系) と異なった場合 | procedure.md 準備 3 の「止める条件」に従い、採用せずに計測者に尋ね、その結果をここに書く |

## クリーンルーム (要件 11.4、11.5、11.6)

**この会話の文脈を持たない、新しいサブエージェント (Opus) が、2026-09-22 に、
research.md と道具の一次の資料だけから、構成の値 (`serving/config/configs.toml`、
`serving/config/nodes.toml`) を書いた。** ssh・docker は使わせず、第三者のレシピと、
モデルカードの本文も開かせなかった。Hugging Face Hub から取得したのは、モデルの
`config.json` 1 本だけである (アテンションの幾何とレイヤーの並びを読むため。
research.md §b-3)。第三者のレシピ、ブログ、フォーラムにある起動のスクリプト、パッチ、
設定は、1 行も参照・引用していない (要件 11.3)。

**参照した資料の一覧 (要件 11.6)**: tasks.md の Implementation Notes 6.2 の項に記録された、
その作業者の CONCERNS 16 の一覧を、URL つきで写したものである。レビュー担当が 119 件の
出典を原文と突き合わせ、112 件が一字一句一致することを確かめた (残り 7 件は design.md が
定める `image` の凝縮した 1 行)。

- **vLLM のソースコード** (build commit `385dce36bcee42309924a5ece951a96db3dce7f2`。
  `https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/...`):
  `vllm/entrypoints/cli/serve.py`、`vllm/config/model.py`、`vllm/config/parallel.py`、
  `vllm/config/scheduler.py`、`vllm/config/load.py`、`vllm/config/multimodal.py`、
  `vllm/config/kernel.py`、`vllm/config/structured_outputs.py`、`vllm/config/vllm.py`、
  `vllm/entrypoints/launchers/cli_args.py`、`vllm/utils/mem_utils.py`、`vllm/envs.py`
- **vLLM の公式文書**: `https://docs.vllm.ai/en/latest/deployment/docker/`、
  `https://docs.vllm.ai/en/latest/getting_started/installation/gpu/`
- **NVIDIA `container-images/cuda` の Dockerfile**:
  `https://gitlab.com/nvidia/container-images/cuda/-/raw/master/dist/13.0.3/ubuntu2404/base/Dockerfile`
  (ベースイメージの `NGC-DL-CONTAINER-LICENSE` の出どころ)
- **NCCL の環境変数とトラブルシュートの文書**:
  `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html`、
  `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/runtime_and_mpi_issues.html`
- **Docker の公式文書**: `https://docs.docker.com/reference/cli/docker/container/run/`、
  `https://docs.docker.com/engine/network/drivers/host/`、
  `https://docs.docker.com/engine/storage/bind-mounts/`
- **PyTorch v2.13.0 の `run.py`**:
  `https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/run.py`
- **huggingface_hub のソースと公式文書**:
  `https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/cli/download.py`、
  `https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/cli/_cli_utils.py`、
  `https://huggingface.co/docs/huggingface_hub/en/guides/download`、
  `https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables`
- **Hugging Face の OpenAPI**: `https://huggingface.co/.well-known/openapi.json`
  (`tree` API の経路の定義)
- **Docker Hub のレジストリ API**:
  `https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags`
- **モデルの `config.json` だけ** (モデルカードの本文は取得していない):
  `https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4/raw/18d55bfd5a2194887738da73753975c9d3842f46/config.json`

## 実機で確かめたこと

### 7.1 (2026-09-22): 配布、イメージの取得、ライセンスの表記

- 準備 1 (`serve push`): 2 台に 6 つの置き場所を作り、`payload/` を配った (終了コード 0)
- 準備 2 (`serve pull-image`): 2 台で `vllm/vllm-openai@sha256:b0501f99…` を取得し、
  ダイジェストが一致した (`RepoDigests` が 2 台ともその 1 つだけ)。読み取った事実: arm64、
  展開後 22,172,559,299 バイト、`ai.vllm.build.commit` = `385dce36bcee42309924a5ece951a96db3dce7f2`
  (research.md §a-2 と構成の出典が前提にした commit と一致)、`org.opencontainers.image.licenses`
  のラベルは空、entrypoint は `vllm serve`。1 台の構成で流すと head にしか取得されないので、
  2 台の構成 (`p1-fetch-nvfp4-probe`) でもう一度流した
- 準備 3 (`serve image-licenses p1-image-licenses`): `/NGC-DL-CONTAINER-LICENSE` (292 行、
  v. September 14, 2021) が読めた。**NVIDIA Deep Learning Container License** (CUDA など
  NVIDIA 独自の部品に掛かる EULA)。要件 3.9 に従い、採用の判断を計測者に尋ね、
  **受け入れる判断 (2026-09-22)** を得た。理由: (a) vLLM 本体は Apache-2.0、(b) CUDA を使う
  限り、どのイメージにもこの EULA が付き、除くと P1 で試せるイメージがない、(c) この用途
  (自分の設備での推論。コンテナを配らず、NVIDIA の API を直接さらさず、NVIDIA の GPU の上で
  だけ動かす) は、EULA の 1.b と 4.a の範囲に収まる。受け入れの条件 (再配布しない、単体の
  製品として配らない、後援をうたわない) は `LICENSES.md` に書いた
- 準備 3 の 2 つめ (`serve image-licenses p1-image-hf-version`): イメージの中に `hf`
  (huggingface_hub の CLI) が **1.30.0** で入っている (research.md の 14b は「ある」で決着。
  ホストには何も入れない)。読み取りのコンテナは、どちらも読み終えたあとに止めて消した
  (自分のコンテナは 0 に戻った)
- 段 0 / 段 1 (`serve probe`) の結果と、KV キャッシュの実測が、brief.md の見積もりと
  どれだけ合っていたか (要件 6.4)
- 第二候補の重み (段 4) に進んだ場合の、その判断の記録
