import math
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from loguru import logger
from torch.autograd import Function

from fastmoe.comm import get_ep_streams
from fastmoe.config import EPConfig, MoEScale, get_ep_cfg
from fastmoe.layers.pipeline import PipelineMoELayer
from fastmoe.models.functional import permute_for_ep, unpermute_from_ep
from fastmoe.models.router import TopKRouter


# --- Helper: Differentiable All-to-All for Reference ---
class DiffAllToAll(Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        y = torch.empty_like(x)
        dist.all_to_all_single(y, x, group=group)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.empty_like(grad_output)
        dist.all_to_all_single(grad_input, grad_output, group=ctx.group)
        return grad_input, None


# --- Mock Structures ---
class MockMlp(nn.Module):
    def __init__(self, cfg: EPConfig, world_size):
        super().__init__()
        self.shared_experts = nn.Linear(cfg.moe.hidden_dim, cfg.moe.hidden_dim)
        self.gate = TopKRouter(
            cfg.moe.hidden_dim, cfg.moe.num_experts_per_gpu * world_size, cfg.moe.top_k
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cfg.moe.hidden_dim, cfg.moe.proj_dim),
                    nn.GELU(),
                    nn.Linear(cfg.moe.proj_dim, cfg.moe.hidden_dim),
                )
                for _ in range(cfg.moe.num_experts_per_gpu * world_size)
            ]
        )


class MockLayer(nn.Module):
    def __init__(self, cfg, world_size):
        super().__init__()
        self.cfg = cfg
        self.input_layernorm = nn.LayerNorm(cfg.moe.hidden_dim)
        self.self_attn = nn.Identity()
        self.post_attention_layernorm = nn.LayerNorm(cfg.moe.hidden_dim)
        self.mlp = MockMlp(cfg, world_size)


# --- Reference Block (With Autograd Support) ---
class ReferenceBlock(nn.Module):
    def __init__(self, cfg, world_size):
        super().__init__()
        self.cfg = cfg
        self.world_size = world_size
        self.input_layernorm = nn.LayerNorm(cfg.moe.hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(cfg.moe.hidden_dim)
        self.shared_experts = nn.Linear(cfg.moe.hidden_dim, cfg.moe.hidden_dim)

        self.gate = TopKRouter(
            cfg.moe.hidden_dim, cfg.moe.num_experts_per_gpu * world_size, cfg.moe.top_k
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
        x_resid = x + x_norm
        x_post_norm = self.post_attention_layernorm(x_resid)

        # 2. Shared Experts
        shared_out = self.shared_experts(x_post_norm)

        # 3. Router
        x_flat = x_post_norm.view(-1, self.cfg.moe.hidden_dim)
        topk_idx, topk_w = self.gate(x_flat)

        # 4. Permute
        batch_size = x_flat.shape[0]
        ne = self.cfg.moe.num_experts_per_gpu * self.world_size
        cap = max(int(math.ceil(batch_size * self.cfg.moe.top_k / ne * 4.0)), 4)

        perm_in, perm_w, gather_idx, _ = permute_for_ep(x_flat, topk_idx, topk_w, ne, cap)

        # 5. Dispatch (Differentiable)
        nl = len(self.local_experts)
        tokens_per_rank = nl * cap
        H = self.cfg.moe.hidden_dim

        send_disp = perm_in.view(self.world_size, tokens_per_rank, H)
        # Use Custom Function to keep graph connected
        recv_disp = DiffAllToAll.apply(send_disp, dist.group.WORLD)

        # 6. Experts
        dispatched_input = recv_disp.view(self.world_size, nl, cap, H)
        expert_in = dispatched_input.transpose(0, 1).reshape(nl, -1, H)
        expert_outs = [self.local_experts[i](expert_in[i]) for i in range(nl)]
        expert_out_stack = (
            torch.stack(expert_outs).view(nl, self.world_size, cap, H).transpose(0, 1)
        )

        # 7. Combine (Differentiable)
        send_comb = expert_out_stack.contiguous().view(self.world_size, tokens_per_rank, H)
        recv_comb = DiffAllToAll.apply(send_comb, dist.group.WORLD)
        combined_output = recv_comb.view(-1, H)

        # 8. Unpermute & Post-Ops
        routed_out = unpermute_from_ep(combined_output, gather_idx, perm_w, batch_size, H)

        # View as 3D to match residual
        routed_out = routed_out.view_as(x_resid)
        shared_out = shared_out.view_as(x_resid)

        return x_resid + routed_out + shared_out


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12399"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)

    cfg = get_ep_cfg(world_size=world_size, scale=MoEScale.TINY)
    dtype = torch.float64

    # 1. Setup Models
    mock_layer_struct = MockLayer(cfg, world_size).cuda().to(dtype)
    pipe = (
        PipelineMoELayer(
            mock_layer_struct,
            rank,
            world_size,
            dist.group.WORLD,
            get_ep_streams(),
            n_micro_batches=cfg.moe.micro_batches,
        )
        .cuda()
        .to(dtype)
    )

    ref = ReferenceBlock(cfg, world_size).cuda().to(dtype)

    # 2. Sync Weights
    with torch.no_grad():
        ref.shared_experts.weight.copy_(pipe.shared_experts.weight)
        ref.shared_experts.bias.copy_(pipe.shared_experts.bias)
        ref.gate.gate.weight.copy_(pipe.gate.gate.weight)
        for i, expert in enumerate(pipe.local_experts):
            ref.local_experts[i].load_state_dict(expert.state_dict())

    # 3. Input
    x = torch.randn(
        cfg.moe.batch_size,
        cfg.moe.seqlen,
        cfg.moe.hidden_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    # 4. Pipeline Metadata
    chunk_size = x.shape[0] // cfg.moe.micro_batches
    pipe._metas = [
        {"t0": i * chunk_size, "t1": (i + 1) * chunk_size} for i in range(cfg.moe.micro_batches)
    ]

    # 5. Forward
    y_pipe = pipe(x)
    y_ref = ref(x)

    diff = (y_pipe - y_ref).abs().max().item()
    if rank == 0:
        logger.info(f"Forward Max Diff: {diff:.6f}")
    assert diff < 1e-4, f"Forward Mismatch! {diff}"

    # 6. Backward Verification
    if rank == 0:
        logger.info("Running Backward Verification...")

    g = torch.randn_like(y_pipe)
    y_pipe.backward(g)
    y_ref.backward(g)

    # Check Shared Experts Grads
    d_shared = (pipe.shared_experts.weight.grad - ref.shared_experts.weight.grad).abs().max().item()
    if rank == 0:
        logger.info(f"Shared Expert Grad Diff: {d_shared:.6f}")
    assert d_shared < 1e-3, f"Shared Grad Mismatch: {d_shared}"

    # Check Router Grads
    d_gate = (pipe.gate.gate.weight.grad - ref.gate.gate.weight.grad).abs().max().item()
    if rank == 0:
        logger.info(f"Router Grad Diff: {d_gate:.6f}")
    assert d_gate < 1e-3, f"Router Grad Mismatch: {d_gate}"

    # Check Local Experts Grads
    for i, expert in enumerate(pipe.local_experts):
        ref_expert = ref.local_experts[i]
        for p1, p2 in zip(expert.parameters(), ref_expert.parameters(), strict=False):
            if p1.grad is not None:
                d_exp = (p1.grad - p2.grad).abs().max().item()
                assert d_exp < 1e-3, f"Expert {i} Grad Mismatch: {d_exp}"

    if rank == 0:
        logger.success("Verification Successful: Forward and Backward match!")

    dist.destroy_process_group()


def run_verify_correctness():
    mp.start_processes(worker, args=(2,), nprocs=2, join=True, start_method="fork")
