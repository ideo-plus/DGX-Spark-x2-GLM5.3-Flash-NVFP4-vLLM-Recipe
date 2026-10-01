# DGX Spark x2 GLM-5.3-Flash NVFP4 vLLM Recipe

[日本語](README.ja.md)

A clean-room recipe for serving **GLM-5.3-Flash** (a 320B-parameter MoE model, 18B active per
token, MIT-licensed weights) on **two NVIDIA DGX Spark** nodes (GB10, ~121.7 GiB unified memory
each) with **vLLM**, tensor-parallel across the single QSFP/RoCE link between them. The goal is
not "it runs" but "it can be trusted with long, unattended agentic work": this server is meant to
become the backend for `takt`, a coding-agent workflow tool, called through the Anthropic-compatible
`/v1/messages` API. See [`PLAN.md`](PLAN.md) for the full goal, constraints, and phased plan
(P0–P8).

## Current results

### Measured with sparkDash

[sparkDash](https://github.com/MiaAI-Lab/sparkDash) is the tool MiaAI-Lab and knapcio publish their numbers with. Measured with
sparkDash 1.8.6 on two DGX Spark nodes at TP=2 with the GPU clock capped at `300–1800 MHz` and X925 cores at 3.0 GHz: `/v1/chat/completions`, temperature 0, 400 output tokens forced with
`min_tokens` / `ignore_eos`, one run per cell. Prefill fills the prompt with a repeated word. sparkDash asks for thinking off,
which this recipe's stock chat template ignores; at the server's default reasoning effort `low`, this recipe writes one short
thinking sentence (counted by sparkDash) before answering. Details: [record](docs/results/2026-10-01-default-effort-low.md).

| Implementation | Structured c1 | Prose c1 | Code c1 | Structured c2 (per stream / total) | Prefill 32k | Prefill 128k |
|---|---:|---:|---:|---:|---:|---:|
| This recipe (default reasoning effort low) | 60.64 | 43.74 | 59.03 | 49.01 / 97.11 | 1,598 | 1,584 |
| mmastrac NVFP4 + DFlash2 (default config) | 69.67 | 32.16 | 66.18 | 53.93 / 107.85 | 1,861 | 1,824 |
| MiaAI-Lab EXL3 + DFlash2 (Sep 29 config) | 70.52 | 32.56 | 67.30 | 55.00 / 110.00 | 1,446 | 1,500 |

mmastrac runs in its README's two-node default and its own chat template reads sparkDash's thinking-off as low reasoning effort.
MiaAI-Lab runs with its default serving settings (only host and network values set for these nodes) and fully stops thinking.
Thinking is therefore not off on the same terms in the three rows: MiaAI-Lab stops it, while mmastrac and this recipe keep one short sentence.

### Measured with this repository's `bench`

Configuration `glm53-tp2-mtp3-marlin`, measured at the confirmation stage (`fast`) with the server's default
reasoning effort `low`. Every decode trial produced an answer, so the decode values below measure answer text.
Both nodes used the same clock caps for these measurements: GPU graphics clocks `300–1800 MHz`
(`nvidia-smi -lgc 300,1800`) and X925 CPU cores capped at `3.0 GHz`. The detailed measurement record
also records the observed ~1.8 GHz graphics clock and no thermal slowdown.
Raising the GPU cap to 2200 MHz and removing the X925 cap did not speed up decode (single stream or two streams)
or cold prefill, so both caps stay in place ([September 29 record](docs/results/2026-09-29-gpu-clock-2200.md)).

**Decode values measure answer text.** At maximum reasoning effort the model often writes code inside its
thinking and returns no answer ([#158](https://github.com/ideo-plus/DGX-Spark-x2-GLM5.3-Flash-NVFP4-vLLM-Recipe/issues/158)), so the server defaults to `low`; a request can still ask for
deeper thinking with `reasoning_effort`.

| Metric | Value | Record |
|---|---:|---|
| Single stream, code/en | 44.3 tok/s | [October 1](docs/results/2026-10-01-default-effort-low.md) |
| Single stream, code/ja | 43.0 tok/s | [October 1](docs/results/2026-10-01-default-effort-low.md) |
| Single stream, prose/en | 37.2 tok/s | [October 1](docs/results/2026-10-01-default-effort-low.md) |
| Single stream, prose/ja | 38.6 tok/s | [October 1](docs/results/2026-10-01-default-effort-low.md) |
| Single stream, JSON/en | 46.9 tok/s | [October 1](docs/results/2026-10-01-default-effort-low.md) |
| Single stream, JSON/ja | 47.9 tok/s | [October 1](docs/results/2026-10-01-default-effort-low.md) |
| Decode, 2 streams, per stream (aggregate) | 23.5 tok/s (40.7 aggregate) | [October 1](docs/results/2026-10-01-default-effort-low.md) |
| Prefill, 32k tokens (cold) | 1,331 tok/s | [September 30](docs/results/2026-09-30-mmastrac-32k-same-conditions.md) |
| Prefill, 128k tokens (cold) | 1,332 tok/s | [September 30](docs/results/2026-09-30-fast-prefill-long-input.md) |

Other recipes' published values are collected in the [recipe comparison](docs/research/2026-09-28-recipe-comparison.md).

The comparison below contains only measurements made under the same conditions: this repository's `fast` profile (prefill with 8k and 32k inputs only),
two DGX Spark nodes at TP=2, and the same `300–1800 MHz` GPU clock cap. Values are median tok/s; concurrency is per stream
at two simultaneous requests, and prefill is cold 32k input. Thinking differs: this recipe's decode values are answer text at its
default effort `low`, while the other two rows were measured at their own default thinking, where the 256-token budget went to thinking text.

| Implementation | Code/en | Prose/en | Concurrency 2 (per stream) | Cold prefill 32k |
|---|---:|---:|---:|---:|
| [This recipe: MTP N=3 + Marlin (KV 4 GiB, default effort low)](docs/results/2026-10-01-default-effort-low.md) | 44.288 | 37.211 | 23.480 | 1,331.163 |
| [MiaAI-Lab EXL3 + DFlash2 (Sep 29)](docs/results/2026-09-29-miaai-exl3-dflash2.md) | 34.212 | 33.423 | 22.675 | 1,340.999 |
| [mmastrac NVFP4 + DFlash2 (default config)](docs/results/2026-09-30-mmastrac-32k-same-conditions.md) | 33.153 | 31.183 | 23.672 | 1,835.091 |

mmastrac runs in its README's two-node default: `TP=2` in `compose/.env` and `compose/glm53.yaml` only, with no experimental overrides.
During these runs the GPU driver logged `NV_ERR_NO_MEMORY` on both nodes while serving.
MiaAI-Lab and mmastrac use the non-commercial DFlash2 drafter. Published values from other recipes are omitted
because their prompts, sampling, thinking mode, and timing definitions differ.

Quality checks at the default effort `low` (tool calls 10/10, HumanEval+ 10/10, needle 8k/32k 2/2) found no breakage. These are
small-sample numbers; the pass/fail judgment against every success criterion is made in phase
**P8** with large samples. Records: [`docs/results/2026-10-01-default-effort-low.md`](docs/results/2026-10-01-default-effort-low.md).
The full success criteria are in [`PLAN.md` §3](PLAN.md#3-成功の基準).

## How it works

- **Self-built vLLM image**: upstream vLLM pinned at commit `0961bbae2894d574be790d219651824eb199318e`,
  plus a NoPE kernel patch cherry-picked from upstream PR #55277 (head `8d09804c877c48165c6ba69bc9dc02d09bae0b83`),
  built once rather than rebuilt for every later fix.
- **TP=2** across the two DGX Spark nodes over the single physical QSFP cable, which NCCL sees and
  uses as two RoCE devices (measured all-reduce busbw ~186.9 Gbps).
- **MTP speculative decoding, N=3** — the model's own multi-token-prediction head (MIT-licensed),
  not a non-commercial third-party drafter.
- **Weights**: the published NVFP4 checkpoint's experts as-is, with most weights it left in BF16
  or FP8 (attention projections, shared experts, dense MLPs, `lm_head`, and the MTP layer's
  projections and experts) converted locally to NVFP4A16 (weight-only NVFP4; MLA `kv_b_proj`, the indexer, and the MTP
  `eh_proj` stay BF16) by
  [`experiments/k2-quant`](experiments/k2-quant/README.md), a from-scratch, CPU-only tool. The
  result is the derived weight set `k2s4`, pinned by a manifest in `serving/weights/`.
- **Default reasoning effort `low`** (`--default-chat-template-kwargs '{"reasoning_effort":"low"}'`). At maximum effort the model
  often writes code inside its thinking and returns no answer ([#158](https://github.com/ideo-plus/DGX-Spark-x2-GLM5.3-Flash-NVFP4-vLLM-Recipe/issues/158)). A request can still ask for deeper thinking with
  `reasoning_effort`.
- **MoE kernel: Marlin** (`--moe-backend marlin`) for all MoE layers. This is the `marlin` at the end of
  the configuration name `glm53-tp2-mtp3-marlin` (GLM-5.3-Flash, TP=2, MTP N=3, Marlin).
  - **What it is**: [Marlin](https://github.com/IST-DASLab/marlin) (**M**ixed **A**uto-**R**egressive
    **Lin**ear kernel, from IST-DASLab) is a GPU matrix-multiply kernel for weight-only quantization.
    It reads 4-bit weights from memory, dequantizes them inside the kernel, and multiplies them with
    BF16 activations. vLLM includes it and can run MoE experts with it.
  - **Why it fits decode**: in decode only a few tokens pass through each step (per request, one token
    plus the three MTP draft tokens), so the step time depends more on reading weights than on
    arithmetic. According to its authors, Marlin keeps the benefit of 4-bit weights up to about 16–32
    tokens per step.
  - **Why it is chosen**: for the published NVFP4 experts, vLLM's default kernel on this GPU is
    FlashInfer CUTLASS, which also quantizes activations to 4 bits (W4A4). Marlin keeps activations in
    BF16 and gave faster decode on this model
    ([record](docs/results/2026-09-28-k2-stage3.md#k2s4--marlin確認の段n--3)).
  - **Limit**: Marlin targets steps with few tokens. Whether it limits long-input prefill (many tokens
    per step) has not been isolated.
- vLLM model-code fixes needed to load those derived weights are applied as **read-only
  bind-mounted overlay files** on top of the unmodified image
  ([`experiments/k2-vllm-overlay`](experiments/k2-vllm-overlay/README.md)), so no image rebuild is
  needed per iteration.
- **Fast startup** with `--load-format instanttensor` after evicting the page cache without root
  (~4–5 minutes, versus ~13 minutes for a plain `mmap` load).
- **GPU/CPU clock caps** (`nvidia-smi -lgc 300,1800`, `cpupower` on the X925 cores at 3.0 GHz) kept
  in place across restarts for thermal stability ([`ops/spark-power-caps`](ops/spark-power-caps/README.md)).

## Repository layout

| Path | Contents |
|---|---|
| [`bench/`](bench/README.md) | Measurement CLI (`bench`) — decode/prefill/concurrency/quality/agent suites, `probe`/`fast`/`quick`/`full` profiles |
| [`serving/`](serving/README.md) | Operator CLI (`serve`) — push/fetch/verify/start/stop/watch, config in `config/`, weight manifests in `weights/` |
| `experiments/k2-quant/` | Local weight-quantization tool (FP8 / NVFP4A16), CPU-only, clean-room |
| `experiments/k2-vllm-overlay/` | vLLM Python file patches applied as read-only bind mounts |
| `experiments/k2-profile/`, `experiments/nope-mla/` | Profiling and NoPE/MLA experiment support code |
| `ops/spark-power-caps/` | systemd unit re-applying GPU/CPU clock caps on every boot |
| `ops/spark-drop-caches/` | Page-cache eviction helper (root not required) used before fast startup |
| `docs/decisions/` | ADRs — what was tried, measured, and why it was chosen |
| `docs/results/` | Published measurement summaries (raw data stays out of the repo, in `results/`) |
| `docs/vllm-baseline/` | Step-by-step procedures (baseline bring-up, converting weights to FP8/NVFP4, overlay bring-up) |
| `docs/research/`, `docs/rules/`, `docs/tasks/`, `docs/development/` | Investigation notes, working rules, task tracking, CI |
| `scripts/` | `spark-precheck.sh` (read-only hardware check) and TAKT wrapper scripts |
| `results/` | Raw measurement data (git-ignored; prompts and responses live here, never in the repo) |
| [`PLAN.md`](PLAN.md), [`LICENSES.md`](LICENSES.md) | The plan and the license/provenance ledger |

## Quick start (operator)

All state-changing steps run from the Mac and require explicit operator approval before touching
either DGX Spark — nothing here is meant to be run unattended. See
[`serving/README.md`](serving/README.md) for the exact commands and
[`docs/vllm-baseline/procedure.md`](docs/vllm-baseline/procedure.md) for the full phase-by-phase
procedure.

1. `serve push` — distribute `serving/payload/` to both nodes.
2. `serve pull-image` / `serve image-licenses` — fetch the pinned image by digest, print its
   embedded license notices.
3. `serve manifest` + `serve fetch --probe-files` + `serve verify` — build a weight manifest, fetch
   just what's needed for a small check, and verify it.
4. `serve probe` — bring the model up on a single node as a shrink-scale sanity check.
5. `serve netcheck links` / `bandwidth` / `sanity` — confirm the QSFP/RoCE link and run vLLM's own
   pre-flight checks.
6. `serve fetch` + `serve verify` + `serve start` — fetch and verify the full weights, then start
   TP=2 across both nodes.
7. `serve smoke` / `serve watch` / `serve logs` / `serve stop` — sanity-check, observe under load,
   collect logs, and tear down.
8. For the locally converted (FP8/NVFP4) weights specifically, see
   [`docs/vllm-baseline/k2-derived-weights-procedure.md`](docs/vllm-baseline/k2-derived-weights-procedure.md)
   and [`docs/vllm-baseline/k2-vllm-overlay-procedure.md`](docs/vllm-baseline/k2-vllm-overlay-procedure.md).
   The recommended configuration (weights `k2s4` + MTP N=3 + `--moe-backend marlin`) is
   `glm53-tp2-mtp3-marlin`; once the derived weights and overlay have been distributed
   and verified:

   ```bash
   uv run --directory serving serve check glm53-tp2-mtp3-marlin
   uv run --directory serving serve start glm53-tp2-mtp3-marlin --yes
   ```

   `serve start --yes` changes state on both nodes — get operator approval first.
9. To measure, use `bench` (see [`bench/README.md`](bench/README.md)): `probe` to compare
   candidates quickly, `fast` to confirm a promising one, `quick`/`full` to record.

## Licensing and clean-room policy

Only components under MIT, Apache-2.0, BSD-family, or PSF licenses are used at runtime and
build-time (see [`PLAN.md` §2](PLAN.md#2-前提と制約) and [`LICENSES.md`](LICENSES.md), which
records every component's name, version, source, license, and purpose). Non-commercial-licensed
components (e.g. the CC BY-NC-ND DFlash2 drafter used by other recipes) are explicitly excluded.
GLM-5.3-Flash's weights are MIT per its model card. The one deliberate exception is the NVIDIA
CUDA base image's Deep Learning Container EULA, accepted under specific conditions (no
redistribution, no standalone product, no claimed NVIDIA endorsement — see `LICENSES.md`).

This repository is built **clean-room**: no scripts, patches, or config files from other GLM/Spark
serving recipes are copied or read while writing this code, even permissively-licensed ones.
Allowed references are official docs/papers/model cards, upstream vLLM and NCCL source/issues/PRs,
and facts measured in this repo's own environment. Every non-trivial choice is recorded in
[`docs/decisions/`](docs/decisions/).

## Status and known limitations

- **Answer decode for code/en is 44.3 tok/s**, just below the 45 tok/s target. Two streams reach 23.5 tok/s per stream (target 23): with this MoE, each extra token reads its own experts, so two streams are bound by memory bandwidth ([record](docs/results/2026-10-01-concurrency-bandwidth-bound.md)).
- **Cold prefill measured 1,331 tok/s at 32k and 1,332 tok/s at 128k**, with the KV cache pinned at 4 GiB per node. The prefix cache rarely hits with MTP enabled.
- **Startup takes ~5 minutes** (`instanttensor` after evicting the page cache).
- **P8 (large-sample verification of every success criterion) has not been run yet.**
