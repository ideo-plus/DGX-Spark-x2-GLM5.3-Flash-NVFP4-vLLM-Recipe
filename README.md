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

Configuration `glm53-tp2-mtp3-marlin`, measured at the confirmation stage (`fast`). The JSON values below
are answer-phase measurements using an 8,192-token output limit and same-server output retokenization.

**The earlier code and prose values measure thinking-only output, not answer generation.**
All 40 earlier code/prose trials and all 60 trials in the latest six-condition run exhausted the
256-token limit without producing a text block. These values do not establish answer-generation
performance or a performance advantage over other recipes.

| Metric | Value | Record |
|---|---:|---|
| Single stream, code/en (thinking only) | 45.0 tok/s | [September 28](docs/results/2026-09-28-k2-stage3.md) |
| Single stream, code/ja (thinking only) | 43.9 tok/s | [September 28](docs/results/2026-09-28-k2-stage3.md) |
| Single stream, prose/en (thinking only) | 46.3 tok/s | [September 28](docs/results/2026-09-28-k2-stage3.md) |
| Single stream, prose/ja (thinking only) | 42.2 tok/s | [September 28](docs/results/2026-09-28-k2-stage3.md) |
| JSON instruction, English (answer phase, retokenized) | 58.6 tok/s | [September 29](docs/results/2026-09-29-json-text-decode.md) |
| JSON instruction, Japanese (answer phase, retokenized) | 58.9 tok/s | [September 29](docs/results/2026-09-29-json-text-decode.md) |
| Decode, 2 streams, per stream (aggregate) | 27.2 tok/s (45.8 aggregate) | [September 28](docs/results/2026-09-28-k2-stage3.md) |
| Prefill, 32k tokens (cold) | 1,326 tok/s | [September 28](docs/results/2026-09-28-k2-stage3.md) |

The JSON answer-phase values are retokenized measurements. See the [measurement record](docs/results/2026-09-29-json-text-decode.md)
and the [recipe comparison](docs/research/2026-09-28-recipe-comparison.md). Concurrency and prefill were not remeasured.

The comparison below contains only measurements made under the same conditions: this repository's `fast` profile,
two DGX Spark nodes at TP=2, and the same GPU clock cap. Values are median tok/s; concurrency is per stream
at two simultaneous requests, and prefill is cold 32k input.

| Implementation | Code/en | Prose/en | Concurrency 2 (per stream) | Cold prefill 32k |
|---|---:|---:|---:|---:|
| [This recipe: MTP N=3 + Marlin](docs/results/2026-09-28-k2-stage3.md) | **45.000** | **46.300** | **27.200** | **1,326.000** |
| [MiaAI-Lab EXL3 + DFlash2](docs/results/2026-09-29-miaai-exl3-dflash2.md) | 34.212 | 33.423 | 22.675 | 1,340.999 |

The mmastrac TP=2 result will be added after the same-condition measurement completes. Published values from
other recipes are omitted because their prompts, sampling, thinking mode, and timing definitions differ.

Quality checks at this stage (tool calls, HumanEval+, needle 8k/32k) found no breakage. These are
small-sample numbers; the pass/fail judgment against every success criterion is made in phase
**P8** with large samples. Records: [`docs/results/2026-09-28-k2-stage3.md`](docs/results/2026-09-28-k2-stage3.md).
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
- **MoE kernel**: `--moe-backend marlin` (weight-only 4-bit, BF16 activations) for all MoE layers.
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

- **2-stream concurrency is below the per-stream target** (27.2 tok/s per stream, target 30).
- **32k prefill is below the target** (1,326 tok/s, target 2,000), and the prefix cache rarely hits
  with MTP enabled.
- **Startup takes ~5 minutes** (`instanttensor` after evicting the page cache).
- **P8 (large-sample verification of every success criterion) has not been run yet.**
