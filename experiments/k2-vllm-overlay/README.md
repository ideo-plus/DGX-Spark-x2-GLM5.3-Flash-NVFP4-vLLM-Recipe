# k2-vllm-overlay

固定した vLLM (commit `0961bbae2894d574be790d219651824eb199318e`) の Python のファイルを直した
**写し**を、イメージを作り直さずに、読み取り専用の bind mount でコンテナの中の同じファイルに
重ねるためのパッチと定義です ([ADR 0007](../../docs/decisions/0007-k2-stage2.md) の第 2a 段、#73)。

## 置き場所

| もの | 場所 |
|---|---|
| 定義 (ファイルの並びと、ソース・写し・パッチの SHA-256) | `k2s2a.json` |
| パッチ (ファイルごとの unified diff。`a/<道筋>` → `b/<道筋>`、`patch -p1` で当てる) | `patches/k2s2a/<道筋>.patch` |
| 配る写し (`serve push` で 2 台の `payload/vllm-overlay/` に配る) | `serving/payload/vllm-overlay/k2s2a/<道筋>` |
| 固定のソース (Git 対象外。CI には無い) | `serving/var/nope-build-0961bbae/source` |
| 道具 | `overlay.py` |

定義の `files` の並びが、構成の mount の並びになります (`experiments/nope-mla/configure_tp2.py --vllm-overlay k2s2a`)。

## 直すこと (`k2s2a`)

- `vllm/model_executor/layers/vocab_parallel_embedding.py`: compressed-tensors は `ParallelLMHead` を
  線形の層として扱い、FP8 (W8A16) のとき humming の線形カーネルに渡す。humming は層の
  `output_partition_sizes` と `has_bias` を読む (`LinearBase` にはあり、埋め込みには無い。#68) ので、
  `ParallelLMHead.__init__` に `input_size_per_partition`・`output_partition_sizes`・`has_bias` を持たせる。
- `vllm/models/glm5next/common/model.py`:
  - MLA の生成 (`model.py:330`) で、`quant_config=None` をやめて `quant_config` を渡す。
  - `packed_modules_mapping` (`model.py:1046-1048`) に `fused_qkv_a_proj` → `q_a_proj`・
    `kv_a_proj_with_mqa` を足す (`ignore` と `targets` をまとめた層の shard の名前で引き当てる)。
  - `_try_load_fp8_attn_proj` は、モデルがその射影を W8A16 FP8 で持つ (`weight_scale` の
    パラメータがある) とき、BF16 への読み替えに横取りしない (`weight_scale_inv` の場合と同じ)。
- `vllm/models/glm5next/common/kda.py`: `quant_config` を一時的に外す処理 (`kda.py:188-194`) を消し、
  `o_proj`・`f_b_proj`・`g_b_proj` が `quant_config` を受けるようにする。まとめた層
  `in_proj_qkvbfg_a` は、この段では `quant_config=None` のまま (第 2b 段)。

indexer (`attention.py`) と MTP (`mtp.py`) は直しません。

## 作り方と確かめ方

写しの側を編集してから、固定のソースとの差からパッチと定義の SHA-256 を作り直します。固定のソースの
木は読むだけで、書き換えません。

```bash
uv run --directory serving python ../experiments/k2-vllm-overlay/overlay.py refresh k2s2a
uv run --directory serving python ../experiments/k2-vllm-overlay/overlay.py check k2s2a
uv run --directory serving pytest -q tests/unit/test_vllm_overlay.py
```

`check` と試験は、固定のソースがあれば、パッチを当てた結果が写しとバイト単位で一致することまで
見ます。ソースが無い CI では、定義に固定した SHA-256 で写しとパッチを固定します。写しは上流の
整形のままにするので、`serving/pyproject.toml` の ruff の対象から外しています。

実機での生成・配布・起動・確かめ方は
[手順書](../../docs/vllm-baseline/k2-vllm-overlay-procedure.md) にあります。

## クリーンルームとライセンス

上流の vLLM のソースと、このリポジトリの Issue だけを見て書きました。他のレシピのパッチは写して
いません (`PLAN.md` §2)。写しは Apache-2.0 の上流のファイルで、先頭の SPDX の表記をそのまま
保ちます (`LICENSES.md`)。上流へ返すときは、同じパッチをそのまま PR にします。
