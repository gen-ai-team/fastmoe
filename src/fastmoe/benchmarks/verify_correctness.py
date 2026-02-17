import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from loguru import logger
from torch.autograd import Function

from fastmoe.comm import get_ep_streams
from fastmoe.config import EPConfig, MoEScale, get_ep_cfg
from fastmoe.models.router import TopKRouter
from fastmoe.models.tiny_model import PipelineMoEBlock


class ReferenceBlock(nn.Module):
    def __init__(self, cfg: EPConfig, group):
        super().__init__()
        self.cfg = cfg
        self.group = group
        self.input_layernorm = nn.LayerNorm(cfg.moe.hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(cfg.moe.hidden_dim)
        self.shared_experts = nn.Linear(cfg.moe.hidden_dim, cfg.moe.hidden_dim)

        self.gate = TopKRouter(
            cfg.moe.hidden_dim, cfg.moe.num_experts_per_gpu * cfg.world_size, cfg.moe.top_k
        )

        self.local_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cfg.moe.hidden_dim, cfg.moe.proj_dim),
                    nn.GELU(),
                    nn.Linear(cfg.moe.proj_dim, cfg.moe.hidden_dim),
                )
                for _ in range(cfg.moe.num_experts_per_gpu)
            ]
        )

    def forward(self, x):
        # 1. Pre-Ops
        x_norm = self.input_layernorm(x)
        residual_moe = x + x_norm
        x_norm_moe = self.post_attention_layernorm(residual_moe)

        # 2. Shared
        shared = self.shared_experts(x_norm_moe)

        # 3. Router
        x_flat = x_norm_moe.view(-1, self.cfg.moe.hidden_dim)
        perm_in, perm_w, gather_idx, cap = self.gate(x_flat)

        # 4. Dispatch (Simulated via Differentiable AllToAll)
        tokens_local = len(self.local_experts) * cap
        reshaped_in = perm_in.view(self.cfg.world_size, tokens_local, self.cfg.moe.hidden_dim)

        class DiffAllToAll(Function):
            @staticmethod
            def forward(ctx, x, g):
                ctx.g = g
                y = torch.empty_like(x)
                dist.all_to_all_single(y, x, group=g)
                return y

            @staticmethod
            def backward(ctx, dy):
                dx = torch.empty_like(dy)
                dist.all_to_all_single(dx, dy, group=ctx.g)
                return dx, None

        disp = DiffAllToAll.apply(reshaped_in, self.group).view(-1, self.cfg.moe.hidden_dim)

        # 5. Experts
        inp = disp.view(self.cfg.world_size, len(self.local_experts), cap, self.cfg.moe.hidden_dim)
        inp = inp.transpose(0, 1).reshape(len(self.local_experts), -1, self.cfg.moe.hidden_dim)
        outs = [exp(inp[i]) for i, exp in enumerate(self.local_experts)]
        stack = torch.stack(outs).view(
            len(self.local_experts), self.cfg.world_size, cap, self.cfg.moe.hidden_dim
        )
        exp_out = (
            stack.transpose(0, 1)
            .contiguous()
            .view(self.cfg.world_size, tokens_local, self.cfg.moe.hidden_dim)
        )

        # 6. Combine
        comb = DiffAllToAll.apply(exp_out, self.group).view(-1, self.cfg.moe.hidden_dim)

        # 7. Post-Ops
        weighted = comb * perm_w.unsqueeze(1)
        buffer = torch.zeros_like(x_flat)
        valid = gather_idx != -1
        buffer.index_add_(0, gather_idx[valid], weighted[valid])

        return residual_moe + buffer.view_as(shared) + shared


# ==========================================
# 6. Worker
# ==========================================
def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12380"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)

    cfg = get_ep_cfg(world_size=world_size, scale=MoEScale.TINY)
    dtype = torch.float32

    pipe = PipelineMoEBlock(cfg, dist.group.WORLD, get_ep_streams()).cuda().to(dtype)
    ref = ReferenceBlock(cfg, dist.group.WORLD).cuda().to(dtype)

    with torch.no_grad():
        for p1, p2 in zip(pipe.parameters(), ref.parameters(), strict=False):
            p2.data.copy_(p1.data)

    x = torch.randn(
        cfg.moe.batch_size,
        cfg.moe.seqlen,
        cfg.moe.hidden_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    if rank == 0:
        logger.info("Running Forward...")
    y_pipe = pipe(x)

    y_ref_list = []
    for chunk in x.chunk(cfg.moe.micro_batches, dim=0):
        y_ref_list.append(ref(chunk))
    y_ref = torch.cat(y_ref_list, dim=0)

    diff = (y_pipe - y_ref).abs().max()
    if rank == 0:
        logger.info(f"Forward Max Diff: {diff:.6f}")
    assert diff < 1e-4, "Forward Mismatch!"

    if rank == 0:
        logger.info("Running Backward...")
    g = torch.randn_like(y_pipe)
    y_pipe.backward(g)
    y_ref.backward(g)

    max_grad_diff = 0.0
    for _, (p1, p2) in enumerate(zip(pipe.parameters(), ref.parameters(), strict=False)):
        if p1.grad is not None:
            d = (p1.grad - p2.grad).abs().max()
            max_grad_diff = max(max_grad_diff, d.item())

    if rank == 0:
        logger.info(f"Backward Max Grad Diff: {max_grad_diff:.6f}")
    assert max_grad_diff < 1e-3, "Backward Mismatch!"

    dist.destroy_process_group()


def run_verify_correctness():
    mp.start_processes(worker, args=(2,), nprocs=2, join=True, start_method="fork")
