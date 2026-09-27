# DGX Spark GLM-5.3-Flash Recipe

[日本語](README.ja.md)

A clean-room recipe for serving **GLM-5.3-Flash** (a 320B-parameter MoE model, 18B active per
token, MIT-licensed weights) on **two NVIDIA DGX Spark** nodes (GB10, ~121.7 GiB unified memory
each) with **vLLM**, tensor-parallel across the single QSFP/RoCE link between them. The goal is
not "it runs" but "it can be trusted with long, unattended agentic work": this server is meant to
become the backend for `takt`, a coding-agent workflow tool, called through the Anthropic-compatible
`/v1/messages` API. See [`PLAN.md`](PLAN.md) for the full goal, constraints, and phased plan
(P0–P8).

## Current results (probe-level)

Single stream, decode tokens/sec, MTP N=3, same-day probe runs with 128-token outputs:

| Weights | code/en | code/ja | prose/en | prose/ja |
|---|---:|---:|---:|---:|
| Original NVFP4 (MTP N=3) | 33.99 | 29.31 | 29.75 | 27.85 |
| K2 stage 2 FP8 (`k2s2b`) | 42.45 | 37.73 | 39.62 | 34.80 |
| K2 stage 3 NVFP4A16 (`k2s3`) | 48.00 | 40.62 | 42.30 | 38.26 |

Confirmation stage (`fast`, 256-token outputs), `k2s3` with MTP N=3: **42.14 / 39.18 / 40.67 / 39.67** — code/en is still about 7% short of the 45 tok/s criterion ([record](docs/results/2026-09-28-k2-stage3.md)).

Without speculative decoding, the original weights sit around **~14.0 tok/s**.

These are **probe-level numbers from small samples**, meant to compare candidates quickly, not to
judge success. Confirmation-stage (256-token) numbers and the final large-sample verification live
in [`docs/results/`](docs/results/) (e.g. [k2 stage 2a](docs/results/2026-09-27-k2-stage2a.md),
[k2 stage 2b](docs/results/2026-09-27-k2-stage2b.md)); the pass/fail judgment against the success
criteria itself is deferred to phase **P8**, run once at the end over all phases.

Success criteria (see [`PLAN.md` §3](PLAN.md#3-成功の基準) for the full table and rationale):
generation speed code/en **45 tok/s** (revised down from 60 on 2026-09-27 once it became clear no
license-OK drafter reaches 60 — see [`docs/results/2026-09-27-k3-drafter-bound.md`](docs/results/2026-09-27-k3-drafter-bound.md)),
prose/ja **30 tok/s**, agent stability (tool-call error rate under 1% up to 100k-token
conversations), and a 72-hour continuous run under `takt`-like load.

## How it works

- **Self-built vLLM image**: upstream vLLM pinned at commit `0961bbae2894d574be790d219651824eb199318e`,
  plus a NoPE kernel patch cherry-picked from upstream PR #55277 (head `8d09804c877c48165c6ba69bc9dc02d09bae0b83`),
  built once rather than rebuilt for every later fix.
- **TP=2** across the two DGX Spark nodes over the single physical QSFP cable, which NCCL sees and
  uses as two RoCE devices (measured all-reduce busbw ~186.9 Gbps).
- **MTP speculative decoding, N=3** — the model's own multi-token-prediction head (MIT-licensed),
  not a non-commercial third-party drafter.
- **K2**: the BF16 weights vLLM couldn't quantize out of the box (attention projections in MLA and
  KDA, shared experts, dense MLPs, `lm_head`) are converted locally to FP8 (stage 2a/2b) and then to
  NVFP4A16 (weight-only NVFP4, stage 3) with [`experiments/k2-quant`](experiments/k2-quant/README.md),
  a from-scratch, CPU-only, GPU/network-free tool.
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
| `docs/vllm-baseline/` | Step-by-step procedures (baseline bring-up, K2 conversion, overlay bring-up) |
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
8. For K2 derived weights specifically, see
   [`docs/vllm-baseline/k2-derived-weights-procedure.md`](docs/vllm-baseline/k2-derived-weights-procedure.md)
   and [`docs/vllm-baseline/k2-vllm-overlay-procedure.md`](docs/vllm-baseline/k2-vllm-overlay-procedure.md).
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

- **60 tok/s (code/en) is not reachable with license-OK drafters.** All public results at that
  level use a non-commercial deep drafter; the criterion was revised down to 45 tok/s on
  2026-09-27 — see [`docs/results/2026-09-27-k3-drafter-bound.md`](docs/results/2026-09-27-k3-drafter-bound.md).
- **2-stream concurrency is still below the per-stream target** (~24 tok/s per stream at `k2s2b`
  versus a 30 tok/s target).
- **Startup still takes ~5 minutes** even with the `instanttensor` fast path (plain `mmap` load is
  ~13 minutes).
- **code/en is 42.1 tok/s at the confirmation stage with `k2s3`**, about 7% below the 45 tok/s
  criterion ([`docs/results/2026-09-28-k2-stage3.md`](docs/results/2026-09-28-k2-stage3.md)).
- **P8 (final, large-sample validation against every success criterion) has not run yet.** All
  numbers in this README are exploratory/probe scale; the 300-trial agent-stability judgment so
  far exists only for the original weights ([`docs/results/2026-09-24-agent-20k.md`](docs/results/2026-09-24-agent-20k.md)).
