"""GB10 上で NoPE のキャッシュ書き込みを既存の RoPE 経路と比較する。

検査する契約の出典: https://github.com/vllm-project/vllm/pull/55277
上流の pytest 全体の代替ではなく、ビルドした C++ 修正の単独検査。
"""

import importlib.metadata
import json

import torch
from vllm import _custom_ops as ops


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("この検証は GB10 (sm_121) を対象にする")
    if importlib.metadata.version("flashinfer-python") != "0.7.0":
        raise RuntimeError("FlashInfer の版が計画と違う")
    torch.manual_seed(42)
    latent = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([0, 3, 63, 65], device="cuda", dtype=torch.int64)
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    caches = []
    for rope_width in (0, 64):
        cache = torch.full((2, 64, 656), 255, device="cuda", dtype=torch.uint8)
        rope = torch.zeros(4, rope_width, device="cuda", dtype=torch.bfloat16)
        ops.concat_and_cache_mla(latent, rope, cache, slots, "fp8_ds_mla", scale)
        torch.cuda.synchronize()
        rows = cache.reshape(128, 656)
        if not bool((rows[slots, 528:] == 0).all()):
            raise RuntimeError("予約領域がゼロ埋めされていない")
        untouched = torch.ones(128, device="cuda", dtype=torch.bool)
        untouched[slots] = False
        if not bool((rows[untouched] == 255).all()):
            raise RuntimeError("対象外のキャッシュ行が変更された")
        caches.append(cache)
    torch.testing.assert_close(caches[0], caches[1], rtol=0, atol=0)
    # 既存の pe_dim=64 経路では、ゼロ以外の RoPE 値も保持する必要がある。
    rope = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    ops.concat_and_cache_mla(latent, rope, caches[1], slots, "fp8_ds_mla", scale)
    torch.cuda.synchronize()
    tail = caches[1].reshape(128, 656)[slots, 528:].contiguous().view(torch.bfloat16)
    torch.testing.assert_close(tail, rope, rtol=0, atol=0)
    print(
        json.dumps(
            {
                "cache_check": "passed",
                "torch": torch.__version__,
                "flashinfer": importlib.metadata.version("flashinfer-python"),
                "vllm": importlib.metadata.version("vllm"),
            }
        )
    )


if __name__ == "__main__":
    main()
