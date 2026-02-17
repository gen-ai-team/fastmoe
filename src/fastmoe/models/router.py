import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        """
        Args:
            x: [Batch, Hidden]
        Returns:
            topk_idx: [Batch, TopK]
            topk_w:   [Batch, TopK]
        """
        logits = self.gate(x)

        # 1. Top-K selection
        # We use softmax over top-k for weights (DeepSeek V3 / Mixtral style)
        topk_w, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_w = F.softmax(topk_w, dim=-1)

        return topk_idx, topk_w
