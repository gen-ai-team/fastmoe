import torch.distributed as dist
import torch.nn as nn

from fastmoe.layers.base import BasePipelineMoE


class PipelineMoELayer(BasePipelineMoE):
    def __init__(self, original_layer, ep_config, group=None, streams=None):
        super().__init__(ep_config.micro_batches, group, streams, original_layer.hidden_size)

        # 1. Map existing modules
        self.input_layernorm = original_layer.input_layernorm
        self.self_attn = original_layer.self_attn
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        self.shared_experts = original_layer.mlp.shared_experts
        self.gate = original_layer.mlp.gate

        # 2. Shard experts
        all_experts = original_layer.mlp.experts
        self.num_local_experts = len(all_experts) // self.world_size
        rank = dist.get_rank(group=self.group)
        self.local_experts = nn.ModuleList(
            [
                all_experts[i]
                for i in range(rank * self.num_local_experts, (rank + 1) * self.num_local_experts)
            ]
        )
