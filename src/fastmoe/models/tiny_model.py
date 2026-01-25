import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from fastmoe.comm import Streams, get_ep_streams
from fastmoe.config import Config


# ==========================================
# Modules
# ==========================================
class SelfAttention(nn.Module):
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
        # Input Shape: [Batch, SeqLen, HiddenDim]
        B, S, D = x.shape
        residual = x
        x = self.norm(x)

        # Projections & Reshape for Multi-head
        # View: [B, S, NumHeads, HeadDim] -> Transpose: [B, NumHeads, S, HeadDim]
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Reassemble heads
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)

        return out + residual


class Expert(nn.Module):
    """
    A single Expert (FFN). In production (DeepSeek/Mixtral), this would often be a SwiGLU.
    """

    def __init__(self, dim: int, proj_dim: int) -> None:
        super().__init__()

        # Assertion ensures we adhere to expansion ratios typical in Transformers (e.g. 4x or 8/3x)
        assert proj_dim > dim, "Projection Dimensions should be larger than dimension"

        self.fc1 = nn.Linear(dim, proj_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(proj_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Num_Tokens_Assigned_To_This_Expert, HiddenDim]
        return self.fc2(self.act(self.fc1(x)))


# ==========================================
# Configurable PipeLine Block (Shared Streams)
# ==========================================
class PipelineMoEBlock(nn.Module):
    """
    Implements the PipeLined MoE Block with N-micro-batches..

    The core idea is to break the dependency chain [Attn -> MoE -> Attn] by
    micro-batching. While we wait for the MoE communication (All-to-All) of MicroBatch `i`,
    we can compute the Attention (Pre/Post Ops) of MicroBatch `i+1` (or `i-1` depending on stage).

    Streams:
        - COMPUTE: Handles Pre-Ops (Gate, Attn L), MoE, and Post-Ops (Attn L+1).
        - COMM: Handles the heavy All-to-All transfers (Dispatch/Combine).
    """

    def __init__(
        self,
        cfg: Config,
        group: dist.ProcessGroup,
        block_name: str,
        pre_op_module: SelfAttention | None,
        post_op_module: SelfAttention | nn.Linear | None,
        streams: dict[Streams, torch.cuda.Stream],
    ) -> None:
        super().__init__()
        self.cfg: Config = cfg
        self.block_name = block_name
        self.hidden_dim = cfg.moe.hidden_dim
        # Calculate local experts: Total Experts / World Size
        self.num_local_experts = cfg.moe.num_experts_per_gpu
        self.group = group
        self.world_size = cfg.world_size

        # Unpack Shared Streams
        self.stream_compute = streams[Streams.COMPUTE]
        self.stream_comm = streams[Streams.COMM]

        # Components
        self.moe_norm = nn.LayerNorm(self.hidden_dim)

        # Pre-Ops: Usually the Self-Attention of the CURRENT layer.
        self.pre_ops = pre_op_module if pre_op_module else nn.Identity()

        # Gating: Maps tokens to experts.
        self.gate = nn.Linear(
            self.hidden_dim, cfg.moe.num_experts_per_gpu * cfg.world_size, bias=False
        )

        self.experts = nn.ModuleList(
            [Expert(self.hidden_dim, cfg.moe.proj_dim) for _ in range(self.num_local_experts)]
        )

        # Post-Ops: Usually the Self-Attention of the NEXT layer (Lancet Optimization).
        self.post_ops = post_op_module if post_op_module else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [Batch, Seq, Dim]
        _, S, D = x.shape

        # 1. Micro-batching: Split input along the Batch dimension.
        # Why: Smaller chunks allow us to overlap work. While chunk 0 is communicating, chunk 1 computes. # noqa
        chunks = x.chunk(self.cfg.moe.micro_batches, dim=0)

        # Buffers to hold intermediate tensors alive.
        # Why: CUDA streams run asynchronously. If we don't hold references in Python,
        # GC might free the tensor memory while the GPU is still reading it, causing undefined behavior. # noqa
        pipeline_buffers = [{} for _ in range(self.cfg.moe.micro_batches)]
        outputs = [None] * self.cfg.moe.micro_batches

        # CUDA Events for Synchronization
        # We need distinct events for every micro-batch at every stage boundary.
        ev_pre_ready = [torch.cuda.Event() for _ in range(self.cfg.moe.micro_batches)]
        ev_dispatch_ready = [torch.cuda.Event() for _ in range(self.cfg.moe.micro_batches)]
        ev_expert_ready = [torch.cuda.Event() for _ in range(self.cfg.moe.micro_batches)]
        ev_combine_ready = [torch.cuda.Event() for _ in range(self.cfg.moe.micro_batches)]

        # The Pipeline Loop (5 Stages)
        # Total ticks = Number of Microbatchess + Pipeline Depth (4)
        total_ticks = self.cfg.moe.micro_batches + 4

        for tick in range(total_ticks):
            # Calculate which micro-batch is in which stage at this tick
            mb_post = tick - 4
            mb_comb = tick - 3
            mb_exp = tick - 2
            mb_disp = tick - 1
            mb_pre = tick

            # Execute Stages (Reverse order helps conceptualize data flow clearing up, but execution order is async) # noqa
            self._stage_post_ops(mb_post, pipeline_buffers, outputs, ev_combine_ready, chunks, S, D)
            self._stage_combine(mb_comb, pipeline_buffers, ev_expert_ready, ev_combine_ready)
            self._stage_experts(mb_exp, pipeline_buffers, ev_dispatch_ready, ev_expert_ready)
            self._stage_dispatch(mb_disp, pipeline_buffers, ev_pre_ready, ev_dispatch_ready)
            self._stage_pre_ops(mb_pre, pipeline_buffers, chunks, ev_pre_ready, D)

        # Final Synchronization:
        # We must wait for the Compute stream to finish the last Post-Op.
        # Why: Ensures all `outputs` are populated and valid before returning to the next Python line. # noqa
        torch.cuda.current_stream().wait_stream(self.stream_compute)
        return torch.cat(outputs, dim=0)

    # ==========================================
    # Pipeline Stages (Split Logic)
    # ==========================================

    # ==========================================
    # Pipeline Stages (Split Logic)
    # ==========================================

    def _stage_pre_ops(
        self,
        mb_idx: int,
        buffers: list[dict[str, torch.Tensor]],
        chunks: tuple[torch.Tensor, ...],
        ev_signal: list[torch.cuda.Event],
        dim: int,
    ) -> None:
        """Stage 1: Pre-Ops + Gating (Compute Stream)"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            with torch.cuda.stream(self.stream_compute):
                # Helper for NVTX visualization
                is_identity = isinstance(self.pre_ops, nn.Identity)
                label = f"{self.block_name}_Pre_{'GateOnly' if is_identity else 'Attn'}_MB{mb_idx}"

                torch.cuda.nvtx.range_push(label)
                with record_function(label):
                    x_chunk = chunks[mb_idx]

                    # 1. Run Pre-Ops (e.g., Layer L Attention)
                    x_proc = self.pre_ops(x_chunk)

                    # 2. Save Residual for the MoE Add later
                    # Flatten: [Batch*Seq, Dim] for simpler linear layers
                    x_flat = x_proc.view(-1, dim)
                    buffers[mb_idx]["gated_input"] = x_flat.clone()

                    # 3. MoE Pre-Norm (Standard Pre-Norm Architecture)
                    x_normed = self.moe_norm(x_proc).view(-1, dim)
                    buffers[mb_idx]["normed_input_flat"] = x_normed

                    # 4. Gating
                    logits = self.gate(x_normed)

                    # Why TopK? Even if we don't permute in this simulation, calculating TopK
                    # incurs computational cost (sorting) which affects the timeline.
                    # In a full implementation, these indices would drive torch.scatter/gather.
                    _, _ = torch.topk(logits, k=self.cfg.moe.top_k, dim=1)

                torch.cuda.nvtx.range_pop()

            # Signal that Pre-Ops are done for this MB
            ev_signal[mb_idx].record(self.stream_compute)

    def _stage_dispatch(
        self,
        mb_idx: int,
        buffers: list[dict[str, torch.Tensor]],
        ev_wait: list[torch.cuda.Event],
        ev_signal: list[torch.cuda.Event],
    ) -> None:
        """Stage 2: Dispatch All-to-All (Comm Stream)"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            buf = buffers[mb_idx]
            # Wait for data from Pre-Ops
            self.stream_comm.wait_event(ev_wait[mb_idx])

            with torch.cuda.stream(self.stream_comm):
                label = f"{self.block_name}_Dispatch_MB{mb_idx}"
                torch.cuda.nvtx.range_push(label)
                with record_function(label):
                    real_in = buf["normed_input_flat"]

                    # Simulation: Repeat data to bloat communication size.
                    # Why: Real MoE dispatch sends sparse, shuffled data. Here we send dense copies
                    # to match the timing of a real heavy transfer without implementing the shuffle logic. # noqa
                    bloated_in = real_in.repeat(1, self.cfg.moe.comm_scaling_factor)
                    bloated_out = torch.empty_like(bloated_in)

                    dist.all_to_all_single(
                        bloated_out,
                        bloated_in,
                        group=self.group,
                        async_op=False,
                    )
                    # In reality, this output would be the shuffled tokens received from peers.
                    buffers[mb_idx]["dispatch_output"] = torch.empty_like(real_in)

                torch.cuda.nvtx.range_pop()
            ev_signal[mb_idx].record(self.stream_comm)

    def _stage_experts(
        self,
        mb_idx: int,
        buffers: list[dict[str, torch.Tensor]],
        ev_wait: list[torch.cuda.Event],
        ev_signal: list[torch.cuda.Event],
    ) -> None:
        """Stage 3: Expert Computation (Expert Stream)"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            buf = buffers[mb_idx]
            # Wait for data from Dispatch
            self.stream_compute.wait_event(ev_wait[mb_idx])

            with torch.cuda.stream(self.stream_compute):
                label = f"{self.block_name}_Experts_MB{mb_idx}"
                torch.cuda.nvtx.range_push(label)
                with record_function(label):
                    inp = buf["dispatch_output"]
                    # Split input among local experts (e.g. 2 experts per GPU)
                    splits = inp.chunk(self.num_local_experts)
                    # Serial execution of experts on this stream
                    res = [self.experts[i](splits[i]) for i in range(self.num_local_experts)]
                    buf["expert_output"] = torch.cat(res)

                torch.cuda.nvtx.range_pop()
            ev_signal[mb_idx].record(self.stream_compute)

    def _stage_combine(
        self,
        mb_idx: int,
        buffers: list[dict[str, torch.Tensor]],
        ev_wait: list[torch.cuda.Event],
        ev_signal: list[torch.cuda.Event],
    ) -> None:
        """Stage 4: Combine All-to-All (Comm Stream)"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            buf = buffers[mb_idx]
            # Wait for Experts to finish
            self.stream_comm.wait_event(ev_wait[mb_idx])

            with torch.cuda.stream(self.stream_comm):
                label = f"{self.block_name}_Combine_MB{mb_idx}"
                torch.cuda.nvtx.range_push(label)
                with record_function(label):
                    real_in = buf["expert_output"]

                    # Dead weight injection again for Combine phase
                    bloated_in = real_in.repeat(1, self.cfg.moe.comm_scaling_factor)
                    bloated_out = torch.empty_like(bloated_in)

                    dist.all_to_all_single(
                        bloated_out,
                        bloated_in,
                        group=self.group,
                        async_op=False,
                    )
                    buf["combined_output"] = torch.empty_like(real_in)

                torch.cuda.nvtx.range_pop()
            ev_signal[mb_idx].record(self.stream_comm)

    def _stage_post_ops(
        self,
        mb_idx: int,
        buffers: list[dict[str, torch.Tensor]],
        outputs: list[torch.Tensor | None],
        ev_wait: list[torch.cuda.Event],
        chunks: tuple[torch.Tensor, ...],
        seq_len: int,
        dim: int,
    ) -> None:
        """Stage 5: Post-Ops & Residual (Compute Stream)"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            buf = buffers[mb_idx]
            # Wait for Combine to finish
            self.stream_compute.wait_event(ev_wait[mb_idx])

            with torch.cuda.stream(self.stream_compute):
                op_name = self.post_ops.__class__.__name__
                label = f"{self.block_name}_Post_{op_name}_MB{mb_idx}"

                torch.cuda.nvtx.range_push(label)
                with record_function(label):
                    # 1. Residual Connection
                    # Equation: x + MoE(Norm(x))
                    moe_out = buf["combined_output"]
                    residual = buf["gated_input"]
                    post_moe_out = residual + moe_out

                    # 2. Reshape back to Sequence for Attention: [Batch, Seq, Dim]
                    mb_size = chunks[mb_idx].size(0)
                    reshaped_in = post_moe_out.view(mb_size, seq_len, dim)

                    # 3. Next Layer Ops
                    # This computes Layer L+1 Attention while Layer L comms are happening
                    outputs[mb_idx] = self.post_ops(reshaped_in)

                torch.cuda.nvtx.range_pop()


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

        # Get the shared streams (Compute, Comm, Expert)
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
                pre_module = SelfAttention(self.hidden_dim, cfg.moe.num_heads)
            else:
                pre_module = None  # Becomes nn.Identity inside the block

            # Post-Op Logic:
            # Blocks 0 to N-2 compute the *next* block's Attention.
            # The final block (N-1) computes the final Linear layer (or Identity if no head).
            if i < cfg.moe.n_blocks - 1:
                post_module = SelfAttention(self.hidden_dim, cfg.moe.num_heads)
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
