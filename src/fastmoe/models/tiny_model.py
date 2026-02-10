import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.profiler import record_function

from fastmoe.comm import Streams, get_ep_streams
from fastmoe.config import Config
from fastmoe.models.router import TopKRouter


# ==========================================
# Modules (Attention, Expert)
# ==========================================
class Attention(nn.Module):
    """
    Standard Multi-Head Attention to simulate realistic compute workloads.
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        residual = x
        x = self.norm(x)
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)
        return out + residual


class Expert(nn.Module):
    def __init__(self, dim: int, proj_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, proj_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(proj_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ==========================================
# MoE Overlap Function (Orchestrator)
# ==========================================
class MoEOverlapFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, block: "PipelineMoEBlock") -> torch.Tensor:
        ctx.block = block
        chunks = x.chunk(block.cfg.moe.micro_batches, dim=0)

        fwd_ctx = [{} for _ in range(block.cfg.moe.micro_batches)]
        outputs = [None] * block.cfg.moe.micro_batches

        ev_pre = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]
        ev_disp = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]
        ev_exp = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]
        ev_comb = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]

        total_ticks = block.cfg.moe.micro_batches + 4

        for tick in range(total_ticks):
            mb_post = tick - 4
            mb_comb = tick - 3
            mb_exp = tick - 2
            mb_disp = tick - 1
            mb_pre = tick

            block._fwd_stage_post_ops(mb_post, fwd_ctx, outputs, ev_comb, chunks)
            block._fwd_stage_combine(mb_comb, fwd_ctx, ev_exp, ev_comb)
            block._fwd_stage_experts(mb_exp, fwd_ctx, ev_disp, ev_exp)
            block._fwd_stage_dispatch(mb_disp, fwd_ctx, ev_pre, ev_disp)
            block._fwd_stage_pre_ops(mb_pre, fwd_ctx, chunks, ev_pre)

        torch.cuda.current_stream().wait_stream(block.streams[Streams.COMPUTE])
        ctx.fwd_ctx = fwd_ctx
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        block: PipelineMoEBlock = ctx.block
        fwd_ctx = ctx.fwd_ctx

        grad_chunks = grad_output.chunk(block.cfg.moe.micro_batches, dim=0)

        ev_post_bw = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]
        ev_comb_bw = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]
        ev_exp_bw = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]
        ev_disp_bw = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]

        dx_list = [None] * block.cfg.moe.micro_batches
        total_ticks = block.cfg.moe.micro_batches + 4

        for tick in range(total_ticks):
            mb_post = tick
            mb_comb = tick - 1
            mb_exp = tick - 2
            mb_disp = tick - 3
            mb_pre = tick - 4

            block._bwd_stage_post_ops(mb_post, fwd_ctx, grad_chunks, ev_post_bw)
            block._bwd_stage_combine(mb_comb, fwd_ctx, ev_post_bw, ev_comb_bw)
            block._bwd_stage_experts(mb_exp, fwd_ctx, ev_comb_bw, ev_exp_bw)
            block._bwd_stage_dispatch(mb_disp, fwd_ctx, ev_exp_bw, ev_disp_bw)
            block._bwd_stage_pre_ops(mb_pre, fwd_ctx, ev_disp_bw, dx_list)

        torch.cuda.current_stream().wait_stream(block.streams[Streams.COMPUTE])

        if not ctx.needs_input_grad[0]:
            return None, None

        return torch.cat(dx_list, dim=0), None


# ==========================================
# Configurable PipeLine Block
# ==========================================
class PipelineMoEBlock(nn.Module):
    def __init__(
        self,
        cfg: Config,
        group: dist.ProcessGroup,
        block_name: str,
        pre_op_module: Attention | None,
        post_op_module: Attention | nn.Linear | None,
        streams: dict[Streams, torch.cuda.Stream],
    ) -> None:
        super().__init__()
        self.cfg: Config = cfg
        self.block_name = block_name
        self.hidden_dim = cfg.moe.hidden_dim
        self.num_local_experts = cfg.moe.num_experts_per_gpu
        self.group = group
        self.streams = streams

        self.moe_norm = nn.LayerNorm(self.hidden_dim)
        self.pre_ops = pre_op_module if pre_op_module else nn.Identity()

        # Use TopKRouter
        # Calculate total experts in the world
        total_experts = cfg.moe.num_experts_per_gpu * cfg.world_size
        self.router = TopKRouter(
            self.hidden_dim,
            total_experts,
            cfg.moe.top_k,
            # Reusing comm_scaling_factor as Capacity Factor for convenience
            capacity_factor=cfg.moe.comm_scaling_factor,
        )

        self.experts = nn.ModuleList(
            [Expert(self.hidden_dim, cfg.moe.proj_dim) for _ in range(self.num_local_experts)]
        )
        self.post_ops = post_op_module if post_op_module else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MoEOverlapFunction.apply(x, self)

    # =========================================================================
    # FORWARD STAGES
    # =========================================================================

    def _fwd_stage_pre_ops(self, mb_idx, ctx, chunks, ev_signal):
        """Stage 1 (Fwd): Pre-Ops + Routing + Permutation [COMPUTE STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMPUTE]
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Fwd_Pre_MB{mb_idx}"
                with record_function(label):
                    x_mb = chunks[mb_idx]  # [Batch, Seq, Dim]

                    with torch.enable_grad():
                        x_proc = self.pre_ops(x_mb)
                        x_flat = x_proc.view(-1, self.hidden_dim)
                        x_normed = self.moe_norm(x_proc).view(-1, self.hidden_dim)

                        # Returns: [Experts * Capacity, Dim], [Experts * Capacity], [Experts * Capacity], int # noqa
                        permuted_inputs, permuted_weights, gather_index, capacity = self.router(
                            x_normed
                        )

                    # Save for next stages
                    # We detach 'permuted_inputs' because it crosses the stream boundary
                    ctx[mb_idx]["permuted_inputs"] = permuted_inputs.detach()

                    # Save Metadata for Combine/Backward
                    ctx[mb_idx]["gather_index"] = gather_index
                    ctx[mb_idx]["permuted_weights"] = permuted_weights
                    ctx[mb_idx]["capacity"] = capacity

                    # Save Residual
                    ctx[mb_idx]["gated_input"] = x_flat.detach()
                    # Save Input for PreOps Backward
                    ctx[mb_idx]["input_pre"] = x_mb

            ev_signal[mb_idx].record(stream)

    def _fwd_stage_dispatch(self, mb_idx, ctx, ev_wait, ev_signal):
        """Stage 2 (Fwd): Dispatch [COMM STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMM]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Fwd_Dispatch_MB{mb_idx}"
                with record_function(label):
                    # Input: [World_Experts * Capacity, Dim]
                    permuted_local = buf["permuted_inputs"]
                    capacity = buf["capacity"]

                    # We need to split this for All-to-All
                    # permuted_local is sorted by ExpertID: [Exp0, Exp1, Exp2, Exp3]
                    # If World=2, ExpPerGPU=2:
                    # Rank 0 needs [Exp0, Exp1]. Rank 1 needs [Exp2, Exp3].
                    # Size per rank = LocalExperts * Capacity

                    tokens_per_rank = self.num_local_experts * capacity

                    # [Total, D] -> [World, LocalTotal, D]
                    # Note: Tensor must be contiguous for AllToAll
                    reshaped_in = permuted_local.view(
                        self.cfg.world_size, tokens_per_rank, self.hidden_dim
                    )

                    # Prepare Output
                    # We will receive [World, LocalTotal, D] -> flatten to [World * LocalTotal, D]
                    # This contains tokens from everyone destined for MY local experts.
                    reshaped_out = torch.empty_like(reshaped_in)

                    dist.all_to_all_single(
                        reshaped_out, reshaped_in, group=self.group, async_op=False
                    )

                    # [Local_Experts * Capacity * World_Size, D]
                    buf["dispatch_output"] = reshaped_out.view(-1, self.hidden_dim)

            ev_signal[mb_idx].record(stream)

    def _fwd_stage_experts(self, mb_idx, ctx, ev_wait, ev_signal):
        """Stage 3 (Fwd): Experts [COMPUTE STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMPUTE]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Fwd_Experts_MB{mb_idx}"
                with record_function(label):
                    # Input: [World * Local_Experts * Capacity, D]
                    # We need to sort this so all tokens for LocalExpert 0 are together.
                    # Currently: [Rank0_Exp0, Rank0_Exp1, Rank1_Exp0, Rank1_Exp1]
                    # We need:   [Rank0_Exp0, Rank1_Exp0, Rank0_Exp1, Rank1_Exp1]

                    disp_out = buf["dispatch_output"]
                    capacity = buf["capacity"]

                    # Reshape to [World, LocalExperts, Capacity, D]
                    view_4d = disp_out.view(
                        self.cfg.world_size, self.num_local_experts, capacity, self.hidden_dim
                    )

                    # Transpose to [LocalExperts, World, Capacity, D]
                    # Then flatten to [LocalExperts, World*Capacity, D]
                    expert_input_grouped = view_4d.transpose(0, 1).reshape(
                        self.num_local_experts, -1, self.hidden_dim
                    )

                    # Save for Backward
                    buf["input_experts"] = expert_input_grouped.detach()

                    with torch.enable_grad():
                        # Run Experts
                        res = []
                        for i in range(self.num_local_experts):
                            # [World*Capacity, D]
                            out_i = self.experts[i](expert_input_grouped[i])
                            res.append(out_i)

                        # [LocalExperts, World*Capacity, D]
                        expert_out_grouped = torch.stack(res, dim=0)

                    # Reverse Transpose for Combine
                    # [LocalExperts, World, Capacity, D] -> [World, LocalExperts, Capacity, D]
                    expert_out_4d = expert_out_grouped.view(
                        self.num_local_experts, self.cfg.world_size, capacity, self.hidden_dim
                    ).transpose(0, 1)

                    # Flatten -> [World * LocalExperts * Capacity, D]
                    buf["expert_output"] = expert_out_4d.reshape(-1, self.hidden_dim).detach()

            ev_signal[mb_idx].record(stream)

    def _fwd_stage_combine(self, mb_idx, ctx, ev_wait, ev_signal):
        """Stage 4 (Fwd): Combine [COMM STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMM]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Fwd_Combine_MB{mb_idx}"
                with record_function(label):
                    expert_out = buf["expert_output"]
                    capacity = buf["capacity"]
                    tokens_per_rank = self.num_local_experts * capacity

                    reshaped_in = expert_out.view(
                        self.cfg.world_size, tokens_per_rank, self.hidden_dim
                    )
                    reshaped_out = torch.empty_like(reshaped_in)

                    dist.all_to_all_single(
                        reshaped_out, reshaped_in, group=self.group, async_op=False
                    )

                    # [World_Experts * Capacity, D]
                    buf["combined_output"] = reshaped_out.view(-1, self.hidden_dim)

            ev_signal[mb_idx].record(stream)

    def _fwd_stage_post_ops(self, mb_idx, ctx, outputs, ev_wait, chunks):
        """Stage 5 (Fwd): Post-Ops (Un-Permutation) [COMPUTE STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMPUTE]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Fwd_Post_MB{mb_idx}"
                with record_function(label):
                    moe_out = buf["combined_output"]  # [Total_Slots, Dim]
                    residual = buf["gated_input"]  # [Total_Tokens, Dim]

                    # Un-Permutation / Scatter
                    gather_index = buf["gather_index"]  # [Total_Slots]
                    weights = buf["permuted_weights"]  # [Total_Slots]

                    # Weighted output: Out = Expert(x) * GateWeight
                    weighted_moe = moe_out * weights.unsqueeze(1)

                    # Scatter Add back to original positions
                    # output_buffer: [Total_Tokens, Dim]
                    output_buffer = torch.zeros_like(residual)

                    # We only scatter valid slots (index != -1)
                    valid_mask = gather_index != -1
                    valid_indices = gather_index[valid_mask]
                    valid_data = weighted_moe[valid_mask]

                    # output[indices] += data
                    # We must duplicate indices into [N, D] for scatter if D > 1?
                    output_buffer.index_add_(0, valid_indices, valid_data)

                    # Compute
                    post_moe_out = residual + output_buffer

                    B_mb = chunks[mb_idx].size(0)
                    reshaped_in = post_moe_out.view(B_mb, -1, self.hidden_dim)

                    with torch.enable_grad():
                        out = self.post_ops(reshaped_in)

                    outputs[mb_idx] = out

                    # Save for Backward
                    buf["input_post"] = reshaped_in
                    buf["combined_output_for_grad"] = moe_out  # [CRITICAL] Save for Weight Grad

    # =========================================================================
    # BACKWARD STAGES
    # =========================================================================

    def _bwd_stage_post_ops(self, mb_idx, ctx, grad_chunks, ev_signal):
        """Stage 1 (Bwd): Grad Post-Ops [COMPUTE STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMPUTE]
            buf = ctx[mb_idx]

            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Bwd_Post_MB{mb_idx}"
                with record_function(label):
                    inp = buf["input_post"].detach().requires_grad_(True)

                    with torch.enable_grad():
                        out = self.post_ops(inp)

                    grads = torch.autograd.grad(
                        outputs=(out,),
                        inputs=(inp,) + tuple(self.post_ops.parameters()),
                        grad_outputs=(grad_chunks[mb_idx],),
                    )

                    d_inp = grads[0]
                    d_params = grads[1:]

                    for p, g in zip(self.post_ops.parameters(), d_params, strict=False):
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g

                    # d_inp is d(Residual + MoE)
                    # d_resid = d_inp
                    # d_moe_scattered = d_inp

                    buf["grad_residual"] = d_inp
                    buf["grad_moe_scattered"] = d_inp  # [Tokens, Dim]

            ev_signal[mb_idx].record(stream)

    def _bwd_stage_combine(self, mb_idx, ctx, ev_wait, ev_signal):
        """Stage 2 (Bwd): Grad Combine [COMM STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMM]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Bwd_Combine_MB{mb_idx}"
                with record_function(label):
                    d_moe_scattered = buf["grad_moe_scattered"].view(-1, self.hidden_dim)
                    gather_index = buf["gather_index"]
                    weights = buf["permuted_weights"]
                    capacity = buf["capacity"]

                    moe_out = buf["combined_output_for_grad"]

                    # 1. Gather Gradients (d_output -> d_weighted_moe)
                    d_weighted_moe = torch.zeros(
                        (gather_index.size(0), self.hidden_dim),
                        dtype=d_moe_scattered.dtype,
                        device=d_moe_scattered.device,
                    )

                    valid_mask = gather_index != -1
                    valid_indices = gather_index[valid_mask]
                    d_weighted_moe[valid_mask] = d_moe_scattered[valid_indices]

                    # 2. Backprop through Weight Mult
                    # d_moe_out = d_weighted * weights
                    d_moe_out = d_weighted_moe * weights.unsqueeze(1)

                    # Calculate Gradient for Gate Weights
                    # d_weights = sum(d_weighted * moe_out, dim=1)
                    # This tells the Router which expert was actually good
                    d_permuted_weights = (d_weighted_moe * moe_out).sum(dim=1)
                    buf["grad_permuted_weights"] = d_permuted_weights

                    # 4. All-to-All (Reverse Combine)
                    tokens_per_rank = self.num_local_experts * capacity
                    reshaped_in = d_moe_out.view(
                        self.cfg.world_size, tokens_per_rank, self.hidden_dim
                    )
                    reshaped_out = torch.empty_like(reshaped_in)

                    dist.all_to_all_single(
                        reshaped_out, reshaped_in, group=self.group, async_op=False
                    )

                    buf["grad_expert_out"] = reshaped_out.view(-1, self.hidden_dim)

            ev_signal[mb_idx].record(stream)

    def _bwd_stage_experts(self, mb_idx, ctx, ev_wait, ev_signal):
        """Stage 3 (Bwd): Grad Experts [COMPUTE STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMPUTE]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Bwd_Experts_MB{mb_idx}"
                with record_function(label):
                    # Input: [World*Local*Cap, Dim]
                    d_expert_out_flat = buf["grad_expert_out"]
                    expert_input_grouped = buf["input_experts"]
                    capacity = buf["capacity"]

                    # We need to reverse the transpose logic from forward
                    # Forward: [W, L, C] -> [L, W, C] -> [L, W*C]
                    # Backward Input: [W, L, C] (flattened)

                    # 1. Unflatten to [W, L, C, D]
                    d_expert_out_4d = d_expert_out_flat.view(
                        self.cfg.world_size, self.num_local_experts, capacity, self.hidden_dim
                    )
                    d_expert_out_grouped = d_expert_out_4d.transpose(0, 1).reshape(
                        self.num_local_experts, -1, self.hidden_dim
                    )

                    inp = expert_input_grouped.detach().requires_grad_(True)

                    with torch.enable_grad():
                        res = []
                        for i in range(self.num_local_experts):
                            res.append(self.experts[i](inp[i]))
                        out = torch.stack(res, dim=0)

                    grads = torch.autograd.grad(
                        outputs=(out,),
                        inputs=(inp,) + tuple(self.experts.parameters()),
                        grad_outputs=(d_expert_out_grouped,),
                    )

                    d_inp_grouped = grads[0]  # [L, W*C, D]
                    d_params = grads[1:]

                    # Accumulate Params
                    for p, g in zip(self.experts.parameters(), d_params, strict=False):
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g

                    # 4. Reverse Transpose for Dispatch Gradient
                    # [L, W*C, D] -> [L, W, C, D] -> [W, L, C, D] -> Flatten
                    d_inp_4d = d_inp_grouped.view(
                        self.num_local_experts, self.cfg.world_size, capacity, self.hidden_dim
                    )
                    d_dispatch_out = d_inp_4d.transpose(0, 1).reshape(-1, self.hidden_dim)

                    buf["grad_dispatch_out"] = d_dispatch_out

            ev_signal[mb_idx].record(stream)

    def _bwd_stage_dispatch(self, mb_idx, ctx, ev_wait, ev_signal):
        """Stage 4 (Bwd): Grad Dispatch [COMM STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMM]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Bwd_Dispatch_MB{mb_idx}"
                with record_function(label):
                    d_disp = buf["grad_dispatch_out"]
                    capacity = buf["capacity"]
                    tokens_per_rank = self.num_local_experts * capacity

                    reshaped_in = d_disp.view(self.cfg.world_size, tokens_per_rank, self.hidden_dim)
                    reshaped_out = torch.empty_like(reshaped_in)

                    dist.all_to_all_single(
                        reshaped_out, reshaped_in, group=self.group, async_op=False
                    )

                    # [Total_Slots, Dim]
                    buf["grad_permuted_input"] = reshaped_out.view(-1, self.hidden_dim)

            ev_signal[mb_idx].record(stream)

    def _bwd_stage_pre_ops(self, mb_idx, ctx, ev_wait, dx_list):
        """Stage 5 (Bwd): Grad Pre-Ops [COMPUTE STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMPUTE]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Bwd_Pre_MB{mb_idx}"
                with record_function(label):
                    d_permuted = buf["grad_permuted_input"]
                    d_resid = buf["grad_residual"]
                    x_in = buf["input_pre"]
                    gather_index = buf["gather_index"]

                    d_permuted_weights = buf["grad_permuted_weights"]

                    with torch.enable_grad():
                        x_proc = self.pre_ops(x_in)
                        x_flat = x_proc.view(-1, self.hidden_dim)
                        x_normed = self.moe_norm(x_proc).view(-1, self.hidden_dim)

                        # Re-run Router to attach Graph for Gate Gradients
                        # This creates 'permuted_weights_graph' which IS connected to 'self.router'
                        _, permuted_weights_graph, _, _ = self.router(x_normed)

                    # 1. Reverse Permutation (Data Path)
                    d_normed = torch.zeros_like(x_normed)
                    valid_mask = gather_index != -1
                    valid_indices = gather_index[valid_mask]
                    valid_grads = d_permuted[valid_mask]
                    d_normed.index_add_(0, valid_indices, valid_grads)

                    # 2. Flatten d_resid
                    d_resid_flat = d_resid.view(-1, self.hidden_dim)

                    # 3. Autograd
                    # We compute gradients for:
                    # - PreOps (via x_flat and x_normed data path)
                    # - Norm (via x_normed data path)
                    # - Router (via permuted_weights_graph)

                    grads = torch.autograd.grad(
                        outputs=(x_flat, x_normed, permuted_weights_graph),
                        grad_outputs=(d_resid_flat, d_normed, d_permuted_weights),
                        inputs=(x_in,)
                        + tuple(self.pre_ops.parameters())
                        + tuple(self.moe_norm.parameters())
                        + tuple(self.router.gate.parameters()),
                        allow_unused=True,
                    )

                    d_x = grads[0]
                    if d_x is None:
                        d_x = torch.zeros_like(x_in)

                    d_params = grads[1:]

                    # List of all params including router
                    all_params = (
                        list(self.pre_ops.parameters())
                        + list(self.moe_norm.parameters())
                        + list(self.router.gate.parameters())
                    )

                    for p, g in zip(all_params, d_params, strict=False):
                        if g is None:
                            continue
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g

                    dx_list[mb_idx] = d_x


# ==========================================
# N-Block Tiny Model
# ==========================================
class TinyModel(nn.Module):
    def __init__(
        self,
        cfg: Config,
        group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.moe.hidden_dim

        # Input Projection
        self.input_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Get the shared streams (Compute, Comm)
        self.streams: dict[Streams, torch.cuda.Stream] = get_ep_streams()
        self.blocks = nn.ModuleList()

        # Dynamic Block Construction
        # We implement the "Micro Batch Chain" where Block N computes Pre=Identity, Post=Attn(N+1).
        # Structure:
        # Block 0:   Pre=Attn(0), MoE(0), Post=Attn(1)
        # Block 1:   Pre=None,    MoE(1), Post=Attn(2)
        # ...
        # Block N-1: Pre=None,    MoE(N-1), Post=Linear(Out)

        for i in range(cfg.moe.n_blocks):
            # Pre-Op Logic:
            # Only the first block (i=0) needs to run its own Attention.
            # Subsequent blocks receive the output of Attn(i) which was computed in Block(i-1)'s Post-Op. # noqa
            if i == 0:
                pre_module = Attention(self.hidden_dim, cfg.moe.num_heads)
            else:
                pre_module = None  # Becomes nn.Identity inside the block

            # Post-Op Logic:
            # Blocks 0 to N-2 compute the *next* block's Attention.
            # The final block (N-1) computes the final Linear layer (or Identity if no head).
            if i < cfg.moe.n_blocks - 1:
                post_module = Attention(self.hidden_dim, cfg.moe.num_heads)
            else:
                # Final block post-op: Project to output or next stage
                post_module = nn.Linear(self.hidden_dim, self.hidden_dim)

            block = PipelineMoEBlock(
                cfg,
                group,
                block_name=f"B{i}",
                pre_op_module=pre_module,
                post_op_module=post_module,
                streams=self.streams,
            )
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return x
