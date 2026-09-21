# ruff: noqa
# fmt: off
#
# ↑ この 2 行は、この道具の側で足したもの (上流の原文ではない)。下の本文は、上流の原文を、
#   1 文字も変えずに置いているので、静的な検査 (ruff check) にも、整形 (ruff format、black) にも
#   掛けない。`serving/pyproject.toml` の除外の設定は、ruff を `serving/` の外から呼ぶと
#   効かないので、設定に頼らない、ファイルの中の印で守る (`# fmt: on` を、下に置かないこと)。
#
# vLLM の公式のトラブルシュートの文書 (`docs/usage/troubleshooting.md`) の
# 「Incorrect hardware/driver」の節にある、確認のスクリプトをそのまま置いたもの。
# PyTorch の NCCL、PyTorch の GLOO (CPU)、vLLM 独自の NCCL、CUDA グラフの中の vLLM の NCCL
# --- の 4 段を、この順で確かめる (design.md 「netcheck」、research.md §e-6)。
#
# 出典 (commit で固定したもの。design.md の Technology Stack と research.md §a-2 が挙げる、
# 固定したイメージの build commit と同じ 40 桁):
#   - commit 付きの GitHub の blob:
#     https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/docs/usage/troubleshooting.md
#   - 生のファイル (取得に使った URL。要約する道具 (WebFetch など) では取っていない):
#     https://raw.githubusercontent.com/vllm-project/vllm/385dce36bcee42309924a5ece951a96db3dce7f2/docs/usage/troubleshooting.md
#   - 公開の文書 (最新版。commit は固定されないので、出典としては上の 2 つを正とする):
#     https://docs.vllm.ai/en/latest/usage/troubleshooting/
#   - commit (40 桁): 385dce36bcee42309924a5ece951a96db3dce7f2
#   - 取得した日: 2026-09-22
#
# 取得と抜き出しの手順 (取り直すときも同じ手順を使う。詳しくは serving/payload/UPSTREAM.md):
#   1. curl -fsSL <上の生のファイルの URL> -o troubleshooting.md
#   2. 「## Incorrect hardware/driver」の節の中の、最初の ```python ... ``` の
#      コードブロック (mkdocs の `??? code` の中。1 段ぶん 4 個の空白でインデントされている)
#      を、機械的に (正規表現で) 抜き出し、共通の先頭の 4 個の空白を取り除く (dedent)。
#      手で打ち直していない
#
# ライセンス: vLLM は Apache-2.0
#   (https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/LICENSE)。
#   このリポジトリも Apache-2.0 (PLAN.md 「このリポジトリのライセンスは Apache-2.0」)。
#   区切りの行より下の本文は、1 文字も変えていない (Apache-2.0 第 4 条 (b) が求める
#   「変更した旨の表示」は、変更していないので不要)。出典の表示は、この見出しコメントで行う
#
# この下の本文は、上流の文書のコードブロックを、変えずに置いたものである。直さない。
# 直す必要が出たら、上流の新しい commit から、同じ手順で取り直す (UPSTREAM.md)。
#
# 本文 (区切りの行より下、末尾の改行 1 個を含む、UTF-8) の SHA-256:
#   6c3a4236ea799b9eb80c8985175d093893a5b186afc0e5b77fbdee84ed8314e6
#
# 動かし方 (上流の文書が示す形をそのまま使う。ファイル名だけ、上流の `test.py` から
# `vllm_sanity_check.py` に変えている):
#
#   1 台だけで試すとき (--nproc-per-node を GPU の数に合わせる):
#     NCCL_DEBUG=TRACE torchrun --nproc-per-node=<GPU の数> vllm_sanity_check.py
#
#   2 台で試すとき (MASTER_ADDR は、両方のノードから届く、rendezvous のアドレスとポート。
#   NODE_RANK は head で 0、worker で 1):
#     NCCL_DEBUG=TRACE torchrun --nnodes 2 \
#         --nproc-per-node=1 \
#         --rdzv_backend=static \
#         --rdzv_endpoint=$MASTER_ADDR \
#         --node-rank $NODE_RANK vllm_sanity_check.py
#
#   成功すると、4 つの `... is successful!` のメッセージが順に出る。途中で止まる、または
#   落ちる場合は、ハードウェアかドライバの問題である可能性が高い、と上流の文書は述べている。
# ---8<--- 区切り: この行より下は、上流の文書のコードブロックをそのまま置いたもの (変更なし) ---8<---
# Test PyTorch NCCL
import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank() % torch.accelerator.device_count()
torch.accelerator.set_device_index(local_rank)
data = torch.FloatTensor([1,] * 128).to("cuda")
dist.all_reduce(data, op=dist.ReduceOp.SUM)
torch.accelerator.synchronize()
value = data.mean().item()
world_size = dist.get_world_size()
assert value == world_size, f"Expected {world_size}, got {value}"

print("PyTorch NCCL is successful!")

# Test PyTorch GLOO
gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
cpu_data = torch.FloatTensor([1,] * 128)
dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
value = cpu_data.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"

print("PyTorch GLOO is successful!")

if world_size <= 1:
    exit()

# Test vLLM NCCL, with cuda graph
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
# pynccl is enabled by default for 0.6.5+,
# but for 0.6.4 and below, we need to enable it manually.
# keep the code for backward compatibility when because people
# prefer to read the latest documentation.
pynccl.disabled = False

s = torch.cuda.Stream()
with torch.cuda.stream(s):
    data.fill_(1)
    out = pynccl.all_reduce(data, stream=s)
    value = out.mean().item()
    assert value == world_size, f"Expected {world_size}, got {value}"

print("vLLM NCCL is successful!")

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(cuda_graph=g, stream=s):
    out = pynccl.all_reduce(data, stream=torch.cuda.current_stream())

data.fill_(1)
g.replay()
torch.cuda.current_stream().synchronize()
value = out.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"

print("vLLM NCCL with cuda graph is successful!")

dist.destroy_process_group(gloo_group)
dist.destroy_process_group()
