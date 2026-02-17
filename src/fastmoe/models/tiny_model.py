import torch
import torch.distributed as dist
import torch.nn as nn

from fastmoe.comm import Streams, get_ep_streams
from fastmoe.config import EPConfig
from fastmoe.layers.common import Attention
from fastmoe.layers.pipeline import PipelineMoELayer
from fastmoe.models.router import TopKRouter


class TinyMlp(nn.Module):
    """Holds the MoE components for the Layer wrapper."""

    def __init__(self, cfg: EPConfig, world_size: int):
        super().__init__()
        self.hidden_dim = cfg.moe.hidden_dim

        self.gate = TopKRouter(
            self.hidden_dim, cfg.moe.num_experts_per_gpu * world_size, cfg.moe.top_k
        )

        self.shared_experts = nn.Linear(self.hidden_dim, self.hidden_dim)

        total_experts = cfg.moe.num_experts_per_gpu * world_size
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, cfg.moe.proj_dim),
                    nn.GELU(),
                    nn.Linear(cfg.moe.proj_dim, self.hidden_dim),
                )
                for _ in range(total_experts)
            ]
        )


class TinyDecoderLayer(nn.Module):
    """Mimics a standard Transformer Decoder Layer structure."""

    def __init__(self, cfg: EPConfig, world_size: int):
        super().__init__()
        self.cfg = cfg
        self.input_layernorm = nn.LayerNorm(cfg.moe.hidden_dim)
        self.self_attn = Attention(cfg.moe.hidden_dim, cfg.moe.num_heads)
        self.post_attention_layernorm = nn.LayerNorm(cfg.moe.hidden_dim)
        self.mlp = TinyMlp(cfg, world_size)


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

        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)

        self.input_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.streams: dict[Streams, torch.cuda.Stream] = get_ep_streams()
        self.blocks = nn.ModuleList()

        for _ in range(cfg.moe.n_blocks):
            struct = TinyDecoderLayer(cfg, self.world_size)

            ep_layer = PipelineMoELayer(
                layer=struct,
                rank=self.rank,
                world_size=self.world_size,
                group=self.group,
                streams=self.streams,
                n_micro_batches=cfg.moe.micro_batches,
            )

            self.blocks.append(ep_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return x
