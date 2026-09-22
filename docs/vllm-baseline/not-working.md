# 動かなかった箇所の一覧 (not-working.md)

P1 (Baseline Procedure) で観察した「動かなかった箇所」を、1 か所にまとめる文書である
(要件 10.1、10.2)。書く手順は
[`procedure.md`](procedure.md) の各段の「記録」の節にある。P2 以降の担当者は、この一覧を
読んでから、同じ調査を繰り返さないこと。

## 書かないこと (要件 10.5)

1. **送った内容と、応答の本文**。書くのは、長さ、トークンの数、終わりの理由、HTTP の状態だけ
2. **認証の情報** (鍵、トークン)。Spark にも置かない (要件 2.6)
3. **計測者が別に起動していた構成 (`exl3-tp2`) の中身** (起動の引数、設定、差し込まれた
   ファイル、記録)。読んでよいのは、GPU を使っているプロセスの名前とメモリの量だけである

## 1 件の書式 (要件 10.1、10.4)

1 件は、次の項目をすべて持つ (design.md 「Baseline Procedure (文書)」)。

| 項目 | 書くこと |
|---|---|
| 番号 | 一覧の中で一意な通し番号 (1 から) |
| 現象 | 観察した現象 (何をしたら、何が起きたか) |
| 再現の条件 — 構成の名前 | `configs.toml` の構成の名前 |
| 再現の条件 — イメージのダイジェスト | `sha256:…` (構成の `image.ref` と同じ形) |
| 再現の条件 — 重みの `repo@revision` | 例: `RedHatAI/GLM-5.3-Flash-NVFP4@18d55bfd5a2194887738da73753975c9d3842f46` |
| 誤りの文面または記録の抜粋 | 全文ではなく抜粋 (要件 10.5。送った内容と応答の本文は含めない) |
| 対応しそうな上流の issue や変更 | URL。なければ「なし」 |
| 回避できたかと方法 | できた/できなかった。できたなら、その方法 |
| 関わる段階 (P2〜P6) | PLAN.md の段 (P2 重みを選ぶ、P3 投機的デコード、P4 長時間のエージェント作業での安定性、P5 通信と性能の詰め、P6 運用) のうち、関わるもの。複数可、または「なし」 |
| 上流に報告するかの判断 (要件 10.4) | 計測者に尋ねた結果。未確認なら「未定 (計測者に確認中)」 |

## 一覧

試す前に、この一覧と [`attempts.md`](attempts.md) を読むこと (要件 8.3)。

### 件 1: 固定したイメージで、GB10 の sparse MLA の KV カーネルが `pe_dim == 64` を要求して落ちる

- **番号**: 1
- **現象**: `serve probe probe-pinned` (1 台、`--load-format dummy`、`--hf-overrides` で 4 層) が、
  重みの読み込み (1.6 秒) のあと、CUDA graph の取得 (PIECEWISE) の途中で、`EngineCore` の
  `RuntimeError` によりコンテナが終了した (終了コード 1)。道具は時間切れを待たずに検知した
  (2026-09-22T10:03:53Z)。選ばれたアテンションのバックエンドは `FLASHINFER_MLA_SPARSE_SM120`
  で、**候補はそれ 1 つだけ** (`platforms/cuda.py:528`)。KV は `fp8_ds_mla` に強制された
  (`mla_attention.py:502`)。MoE は `FLASHINFER_CUTLASS`
- **再現の条件**:
  - 構成の名前: `probe-pinned`
  - イメージのダイジェスト: `sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5`
    (vLLM `0.28.1rc1.dev580+g385dce36b`、build commit `385dce36bcee42309924a5ece951a96db3dce7f2`)
  - 重みの `repo@revision`: `RedHatAI/GLM-5.3-Flash-NVFP4@18d55bfd5a2194887738da73753975c9d3842f46`
    (`config.json`、トークナイザだけ。safetensors は取得していない)
- **誤りの文面または記録の抜粋** (`serving/var/20260922T100215Z-start-probe-pinned/head/container.stdout.log`):
  - `INFO [platforms/cuda.py:528] Using FLASHINFER_MLA_SPARSE_SM120 attention backend out of potential backends: ['FLASHINFER_MLA_SPARSE_SM120'].`
  - `INFO [model_executor/.../attention/mla_attention.py:502] Using fp8_ds_mla KV cache format for FLASHINFER_MLA_SPARSE_SM120 backend.`
  - `File ".../vllm/models/glm5next/nvidia/attention.py", line 590, in forward` → … →
    `RuntimeError: concat_and_cache_mla, /workspace/csrc/libtorch_stable/cache_kernels.cu:928, pe_dim must be 64 for fp8_ds_mla`
  - 道具の判定: `observation.known_failure=pe_dim_assert`
- **対応しそうな上流の issue や変更**: https://github.com/vllm-project/vllm/issues/57578 、
  https://github.com/vllm-project/vllm/issues/55773 (どちらも 2026-09-22 時点で open)。
  原因のカーネルの assert は `csrc/libtorch_stable/cache_kernels.cu` の
  `STD_TORCH_CHECK(pe_dim == 64, "pe_dim must be 64 for fp8_ds_mla")` で、今日の `nightly`
  (commit `0961bbae2894d574be790d219651824eb199318e`) にも同じ行が残っている
  (research.md §d-1 の予測どおり。このモデルは `qk_rope_head_dim = 0` の NoPE MLA)
- **段 1 でも同じ** (2026-09-22T10:23:11Z、`probe-nightly`、イメージ
  `sha256:865784ba59b46e3e1370df8158c1bbff8769c3ab88d63551d3f25eabc1a61c45` = vLLM
  `0.29.1rc1.dev513+g0961bbae2`、commit `0961bbae2894d574be790d219651824eb199318e`): 候補の
  バックエンドは同じ 1 つ、KV は同じく `fp8_ds_mla` に強制、同じ `pe_dim must be 64 for fp8_ds_mla`
  (`cache_kernels.cu:939`) で終了。記録は `serving/var/20260922T102311Z-start-probe-nightly/`
- **回避できたかと方法**: できなかった (公式のイメージ 2 つ。research.md §d-1 のとおり、
  `--kv-cache-dtype` でも `--attention-backend` でも回避できない: この GPU で選べるバックエンドが
  1 つしかなく、そのバックエンドが KV の型を無条件に `fp8_ds_mla` に書き換える)。上流の未マージの
  変更 (2026-09-22 時点、題名と状態だけを見た): PR #55277「[Bugfix][SM120][MLA] Support NoPE
  sparse MLA (GLM-5.3-Flash) on the FlashInfer SM120 backend」、#55778「fix(mla): support NoPE head
  sizes by zeroing RoPE tail in SM120 backend」、#53969「[Bugfix] Support NoPE models on
  FLASHINFER_MLA_SPARSE_SM120 …」、#54929「[Attention] Portable Triton sparse-MLA fallback for
  SM12x」(どれも open、unmerged)。パッチを当てる・イメージを作るのは計測者の判断が要る (要件 8.6)
- **関わる段階 (P2〜P6)**: P2 (重みの形式に依らず、アテンションの KV の経路で落ちるので、
  重みを変えても同じ)、P5 (通信の詰めの前提となる 2 台の起動そのものができない)
- **上流に報告するかの判断 (要件 10.4)**: 未定 (計測者に確認中。上流の issue に当たる現象を
  自分たちの環境で再現した)

### 件 0 (雛形。書式を示すだけで、実在する記録ではない)

- **番号**: 0
- **現象**: (例) `serve probe probe-pinned` が、起動の途中でコンテナが終了して失敗した
- **再現の条件**:
  - 構成の名前: (例) `probe-pinned`
  - イメージのダイジェスト: (例) `sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5`
  - 重みの `repo@revision`: (例) `RedHatAI/GLM-5.3-Flash-NVFP4@18d55bfd5a2194887738da73753975c9d3842f46`
- **誤りの文面または記録の抜粋**: (例) `pe_dim must be 64 for fp8_ds_mla` の前後数行
- **対応しそうな上流の issue や変更**: (例) `https://github.com/vllm-project/vllm/issues/NNNNN` または「なし」
- **回避できたかと方法**: (例) できなかった / できた (`--hf-overrides` で層を増やした、など)
- **関わる段階 (P2〜P6)**: (例) P2、P5 / なし
- **上流に報告するかの判断 (要件 10.4)**: (例) 未定 (計測者に確認中) / 報告した (URL) / 報告しない (理由)
