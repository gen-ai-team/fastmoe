import torch
import torch.distributed as dist
import torch.nn as nn

from fastmoe.comm import Streams, get_ep_streams
from fastmoe.config import EPConfig
from fastmoe.layers.base import BasePipelineMoE
from fastmoe.layers.common import Attention
from fastmoe.models.router import TopKRouter


class PipelineMoEBlock(BasePipelineMoE):
    def __init__(
        self,
        cfg: EPConfig,
        group: dist.ProcessGroup,
        streams: dict,
        pre_op: nn.Module | None,
        post_op: nn.Module | None,
        block_name: str | None = None,
    ) -> None:
        super().__init__(cfg.moe.micro_batches, group, streams, cfg.moe.hidden_dim)

        self.block_name = block_name

        # Initialize modules from scratch for the benchmark
        self.input_layernorm = nn.LayerNorm(self.hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_dim)
        self.pre_ops = pre_op if pre_op else nn.Identity()
        self.post_ops = post_op if post_op else nn.Identity()
        self.shared_experts = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.gate = TopKRouter(
            self.hidden_dim, cfg.moe.num_experts_per_gpu * self.world_size, cfg.moe.top_k
        )
        self.num_local_experts = cfg.moe.num_experts_per_gpu
        self.local_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, cfg.moe.proj_dim),
                    nn.GELU(),
                    nn.Linear(cfg.moe.proj_dim, self.hidden_dim),
                )
                for _ in range(self.num_local_experts)
            ]
        )

    @property
    def self_attn(self):
        return self.pre_ops  # Alias for base class logic


# ==========================================
# N-Block Tiny Model
# ==========================================
class TinyModel(nn.Module):
    def __init__(
        self,
        cfg: EPConfig,
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
                cfg=cfg,
                group=group,
                streams=self.streams,
                pre_op=pre_module,
                post_op=post_module,
                block_name=f"B{i}",
            )
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return x
