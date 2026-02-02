import os
from enum import Enum

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.autograd import Function
from torch.profiler import record_function


# --- Configuration ---
class Giga10BConfig:
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 4,  # Reduced for verification speed
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        qk_head_dim: int = 128,
        v_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 64,
        kv_lora_rank: int = 512,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        pad_token_id: int = 0,
        rope_scaling: dict = None,
        # MoE params
        n_routed_experts: int = 8,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: int = 14336,
        n_shared_experts: int = 1,
        first_k_dense_replace: int = 1,  # First layer dense, rest MoE
        routed_scaling_factor: int = 1,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        # Infra params
        world_size: int = 1,
        micro_batches: int = 4,
        comm_scaling_factor: int = 1,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.rope_scaling = rope_scaling or {}
        self.rope_theta = 10000.0
        self.max_position_embeddings = 4096

        # MoE
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.n_shared_experts = n_shared_experts
        self.first_k_dense_replace = first_k_dense_replace
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob

        # FastMoE Compat
        self.moe = type(
            "obj",
            (object,),
            {
                "hidden_dim": hidden_size,
                "num_experts_per_gpu": n_routed_experts // world_size,
                "top_k": num_experts_per_tok,
                "proj_dim": moe_intermediate_size,
                "micro_batches": micro_batches,
                "comm_scaling_factor": comm_scaling_factor,
            },
        )
        self.world_size = world_size


# --- Utils ---
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_interleave_varlen(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    # Robustly handle both 3D [Seq, Head, Dim] and 4D [Batch, Head, Seq, Dim]
    # We ignore unsqueeze_dim argument here and infer shapes directly for safety

    if q.dim() == 3:  # (total_tokens, heads, dim)
        # Flattened/Packed case
        cos = cos.unsqueeze(1)  # [Seq, 1, Dim]
        sin = sin.unsqueeze(1)

        n, h, d = q.shape
        q_reshaped = q.view(n, h, d // 2, 2).transpose(-1, -2).reshape(n, h, d)
        n, h, d = k.shape
        k_reshaped = k.view(n, h, d // 2, 2).transpose(-1, -2).reshape(n, h, d)

    elif q.dim() == 4:  # (batch, heads, seq_len, dim)
        # Standard case
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, Seq, Dim]
        sin = sin.unsqueeze(0).unsqueeze(0)

        b, h, s, d = q.shape
        q_reshaped = q.view(b, h, s, d // 2, 2).transpose(-1, -2).reshape(b, h, s, d)
        k_reshaped = k.view(b, h, s, d // 2, 2).transpose(-1, -2).reshape(b, h, s, d)
    else:
        raise ValueError(f"Unexpected q dim: {q.dim()}")

    q_embed = (q_reshaped * cos) + (rotate_half(q_reshaped) * sin)
    k_embed = (k_reshaped * cos) + (rotate_half(k_reshaped) * sin)
    return q_embed, k_embed


class MLP(nn.Module):
    """MLP SwiGLU (Expert)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        act = self.act_fn(self.gate_proj(hidden_states))
        down_proj = self.down_proj(act * self.up_proj(hidden_states))
        return down_proj


class TopkRouter(nn.Module):
    """Original Giga10B / DeepSeek V3 Router Logic"""

    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        routed_scaling_factor: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.hidden_size = hidden_size

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))
        # Init weights for stability in verification
        nn.init.normal_(self.weight, std=0.01)

    @torch.no_grad()
    def get_topk_indices(self, scores: torch.Tensor) -> torch.Tensor:
        scores_for_choice = scores.view(
            -1, self.n_routed_experts
        ) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class Attention(nn.Module):
    """Multi-head attention Deepseek V3."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        rope_scaling: dict | None,
        attention_bias: bool,
        attention_dropout: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.qk_head_dim, bias=attention_bias
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=attention_bias
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_key_value_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=attention_bias
        )
        self.scaling = self.qk_head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_states = self.q_proj(hidden_states).view(-1, self.num_heads, self.qk_head_dim)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass))
        k_pass = k_pass.view(-1, self.num_key_value_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        cos, sin = position_embeddings
        k_rot = k_rot.unsqueeze(1)
        q_rot, k_rot = apply_rotary_pos_emb_interleave_varlen(
            q_rot, k_rot, cos, sin, unsqueeze_dim=1
        )
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=True,
        )
        return self.o_proj(attn_output.reshape(attn_output.shape[0], -1).contiguous())


class Streams(Enum):
    COMPUTE = 0
    COMM = 1


def get_ep_streams():
    return {
        Streams.COMPUTE: torch.cuda.Stream(priority=-1),
        Streams.COMM: torch.cuda.Stream(priority=0),
    }


class DifferentiableAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.empty_like(grad_output)
        dist.all_to_all_single(grad_input, grad_output, group=ctx.group)
        return grad_input, None


class FastMoERouterAdapter(nn.Module):
    """
    Adapts Giga10B Router to FastMoE Pipeline.
    """

    def __init__(self, giga_router: "TopkRouter", capacity_factor: float = 1.0):
        super().__init__()
        self.brain = giga_router
        self.capacity_factor = capacity_factor
        self.num_experts = giga_router.n_routed_experts
        self.top_k = giga_router.top_k

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            num_tokens, D = x.shape
            x_flat = x
        elif x.dim() == 3:
            B, S, D = x.shape
            num_tokens = B * S
            x_flat = x.view(-1, D)
        else:
            raise ValueError(f"Input must be 2D or 3D, got {x.shape}")

        # 1. Get Decisions from Giga10B Router
        topk_indices, topk_weights = self.brain(x_flat)

        # 2. Logistics (Capacity, Padding, Permutation)
        capacity = int((num_tokens / self.num_experts) * self.capacity_factor)
        capacity = max(capacity, 4)

        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).to(torch.int32)
        token_priority = torch.cumsum(expert_mask, dim=0) * expert_mask
        valid_mask = (token_priority > 0) & (token_priority <= capacity)

        valid_mask_flat = valid_mask.view(-1, self.num_experts)
        token_priority_flat = token_priority.view(-1, self.num_experts)

        gather_index = torch.full(
            (self.num_experts * capacity,), -1, dtype=torch.long, device=x.device
        )

        row_idx = (
            torch.arange(num_tokens * self.top_k, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.num_experts)
        )
        original_token_idx = row_idx // self.top_k

        expert_ids_range = torch.arange(self.num_experts, device=x.device).unsqueeze(0)
        dest_idx = expert_ids_range * capacity + (token_priority_flat - 1)

        active = valid_mask_flat.bool()
        gather_index.scatter_(0, dest_idx[active].flatten(), original_token_idx[active].flatten())

        weights_flat = topk_weights.view(-1).unsqueeze(1).expand(-1, self.num_experts)
        permuted_weights = torch.zeros(
            (self.num_experts * capacity,), dtype=x.dtype, device=x.device
        )
        permuted_weights.scatter_(0, dest_idx[active].flatten(), weights_flat[active].flatten())

        safe_gather_index = gather_index.clamp(min=0)
        permuted_inputs = x_flat[safe_gather_index]
        permuted_inputs.masked_fill_((gather_index == -1).unsqueeze(1), 0.0)

        return permuted_inputs, permuted_weights, gather_index, capacity


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
        block = ctx.block
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
        return torch.cat(dx_list, dim=0), None


class PipelineMoEBlock(nn.Module):
    def __init__(self, cfg: Giga10BConfig, group: dist.ProcessGroup, streams: dict):
        super().__init__()
        self.cfg = cfg
        self.group = group
        self.streams = streams
        self.hidden_dim = cfg.hidden_size
        self.num_local_experts = cfg.n_routed_experts // cfg.world_size

        # 1. Init Giga10B Router
        giga_router = TopkRouter(
            hidden_size=self.hidden_dim,
            n_routed_experts=cfg.n_routed_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            routed_scaling_factor=cfg.routed_scaling_factor,
            n_group=cfg.n_group,
            topk_group=cfg.topk_group,
            norm_topk_prob=cfg.norm_topk_prob,
        )

        # 2. Wrap with Adapter
        self.router = FastMoERouterAdapter(giga_router, capacity_factor=cfg.moe.comm_scaling_factor)

        # 3. Init Giga Experts
        self.experts = nn.ModuleList(
            [MLP(self.hidden_dim, cfg.moe_intermediate_size) for _ in range(self.num_local_experts)]
        )

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
                        + tuple(self.router.gate.parameters()),  # [NEW] Add Router Params
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


class FastMoEDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Giga10BConfig,
        layer_idx: int,
        group: dist.ProcessGroup,
        streams: dict,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.qk_head_dim,
            config.v_head_dim,
            config.qk_rope_head_dim,
            config.qk_nope_head_dim,
            config.kv_lora_rank,
            config.rope_scaling,
            config.attention_bias,
            config.attention_dropout,
            config.rms_norm_eps,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_idx >= config.first_k_dense_replace:
            # [INTEGRATION] Use PipelineMoEBlock
            self.mlp = PipelineMoEBlock(config, group, streams)
        else:
            self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, cu_seqlens, max_seqlen, position_embeddings):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens, max_seqlen, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Flatten for MoE/MLP
        batch, seq, dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, dim)

        hidden_states_flat = self.mlp(hidden_states_flat)

        hidden_states = hidden_states_flat.view(batch, seq, dim)
        hidden_states = residual + hidden_states
        return hidden_states


class FastMoEGigaModel(nn.Module):
    def __init__(self, config: Giga10BConfig, group: dist.ProcessGroup):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        streams = get_ep_streams()
        self.layers = nn.ModuleList(
            [
                FastMoEDecoderLayer(config, i, group, streams)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, cu_seqlens, max_seqlen, position_embeddings):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, max_seqlen, position_embeddings)
        return self.norm(hidden_states)


class ReferenceMoEBlock(nn.Module):
    def __init__(self, cfg: Giga10BConfig, group):
        super().__init__()
        self.cfg = cfg
        self.group = group
        self.hidden_dim = cfg.hidden_size

        # 1. Init Giga10B Router
        giga_router = TopkRouter(
            self.hidden_dim,
            cfg.n_routed_experts,
            cfg.num_experts_per_tok,
            cfg.routed_scaling_factor,
            cfg.n_group,
            cfg.topk_group,
            cfg.norm_topk_prob,
        )
        # 2. Wrap with Adapter
        self.router = FastMoERouterAdapter(giga_router, capacity_factor=cfg.moe.comm_scaling_factor)

        self.experts = nn.ModuleList(
            [
                MLP(self.hidden_dim, cfg.moe_intermediate_size)
                for _ in range(cfg.n_routed_experts // cfg.world_size)
            ]
        )

    def forward(self, x):
        permuted_inputs, permuted_weights, gather_index, capacity = self.router(x)

        tokens_per_rank = len(self.experts) * capacity
        reshaped_in = permuted_inputs.view(self.cfg.world_size, tokens_per_rank, self.hidden_dim)
        reshaped_out = DifferentiableAllToAll.apply(reshaped_in, self.group)
        dispatch_output = reshaped_out.view(-1, self.hidden_dim)

        view_4d = dispatch_output.view(
            self.cfg.world_size, len(self.experts), capacity, self.hidden_dim
        )
        expert_input_grouped = view_4d.transpose(0, 1).reshape(
            len(self.experts), -1, self.hidden_dim
        )

        res = [self.experts[i](expert_input_grouped[i]) for i in range(len(self.experts))]
        expert_out_grouped = torch.stack(res, dim=0)

        expert_out_4d = expert_out_grouped.view(
            len(self.experts), self.cfg.world_size, capacity, self.hidden_dim
        ).transpose(0, 1)
        expert_output = expert_out_4d.reshape(-1, self.hidden_dim)

        reshaped_in = expert_output.view(self.cfg.world_size, tokens_per_rank, self.hidden_dim)
        reshaped_out = DifferentiableAllToAll.apply(reshaped_in, self.group)
        moe_out = reshaped_out.view(-1, self.hidden_dim)

        weighted_moe = moe_out * permuted_weights.unsqueeze(1)
        output_buffer = torch.zeros_like(x)
        valid_mask = gather_index != -1
        output_buffer.index_add_(0, gather_index[valid_mask], weighted_moe[valid_mask])
        return output_buffer


class ReferenceDecoderLayer(nn.Module):
    def __init__(self, config: Giga10BConfig, layer_idx: int, group: dist.ProcessGroup):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Copied init from FastMoEDecoderLayer...
        self.self_attn = Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.qk_head_dim,
            config.v_head_dim,
            config.qk_rope_head_dim,
            config.qk_nope_head_dim,
            config.kv_lora_rank,
            config.rope_scaling,
            config.attention_bias,
            config.attention_dropout,
            config.rms_norm_eps,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = ReferenceMoEBlock(config, group)  # [REF]
        else:
            self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, cu_seqlens, max_seqlen, position_embeddings):
        # Same forward as FastMoEDecoderLayer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens, max_seqlen, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Flatten and chunk for Reference matching
        batch, seq, dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, dim)

        # Manually loop over microbatches to match Pipeline math
        chunks = hidden_states_flat.chunk(4, dim=0)  # Hardcoded MB=4
        out_chunks = []
        for chunk in chunks:
            out_chunks.append(self.mlp(chunk))
        hidden_states_flat = torch.cat(out_chunks, dim=0)

        hidden_states = hidden_states_flat.view(batch, seq, dim)
        hidden_states = residual + hidden_states
        return hidden_states


class ReferenceGigaModel(nn.Module):
    def __init__(self, config: Giga10BConfig, group: dist.ProcessGroup):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [ReferenceDecoderLayer(config, i, group) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, cu_seqlens, max_seqlen, position_embeddings):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, max_seqlen, position_embeddings)
        return self.norm(hidden_states)


def check_tensors(rank, name, t1, t2):
    if torch.allclose(t1, t2, atol=1e-3, rtol=1e-3):
        return True
    logger.error(f"R{rank} {name} Mismatch! Diff: {(t1 - t2).abs().max().item():.5f}")
    return False


def sync_weights(model1, model2):
    with torch.no_grad():
        for (_n1, p1), (_, p2) in zip(
            model1.named_parameters(),
            model2.named_parameters(),
            strict=False,
        ):
            p2.data.copy_(p1.data)


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12377"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)

    # 1. Config (Small)
    cfg = Giga10BConfig(
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=1024,
        num_hidden_layers=2,
        n_routed_experts=4,
        world_size=world_size,
        micro_batches=4,
        comm_scaling_factor=1.0,
    )

    # 2. Models
    fast_model = FastMoEGigaModel(cfg, dist.group.WORLD).cuda()
    ref_model = ReferenceGigaModel(cfg, dist.group.WORLD).cuda()
    sync_weights(fast_model, ref_model)

    # 3. Inputs
    B, S = 4, 32
    input_ids = torch.randint(0, cfg.vocab_size, (B, S)).cuda()
    head_dim = cfg.qk_rope_head_dim
    cos = torch.randn(S, head_dim // 2).cuda()
    sin = torch.randn(S, head_dim // 2).cuda()
    cu_seqlens = torch.arange(0, (B + 1) * S, step=S, dtype=torch.int32).cuda()

    dist.barrier()
    if rank == 0:
        logger.info(">>> Forward")

    y_fast = fast_model(input_ids, cu_seqlens, S, (cos, sin))
    y_ref = ref_model(input_ids, cu_seqlens, S, (cos, sin))
    check_tensors(rank, "Output", y_fast, y_ref)

    if rank == 0:
        logger.info(">>> Backward")
    loss_fast = y_fast.mean()
    loss_ref = y_ref.mean()
    loss_fast.backward()
    loss_ref.backward()

    grads_ok = True
    for (n, p1), (_, p2) in zip(
        fast_model.named_parameters(),
        ref_model.named_parameters(),
        strict=False,
    ):
        if p1.grad is not None and not check_tensors(rank, f"Grad {n}", p1.grad, p2.grad):
            grads_ok = False

    if grads_ok:
        logger.info(f"Rank {rank}: ✅ Success!")

    dist.destroy_process_group()


def run_experiment():
    mp.start_processes(worker, args=(2,), nprocs=2, join=True, start_method="fork")


if __name__ == "__main__":
    run_experiment()
