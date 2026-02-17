import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    def __init__(self, hidden_dim, num_total_experts, top_k=2, capacity_factor=1.0):
        super().__init__()
        self.num_total_experts = num_total_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(hidden_dim, num_total_experts, bias=False)

    def forward(self, x):
        # x: [Batch*Seq, Dim]
        logits = self.gate(x)
        scores = logits.sigmoid()

        # 1. Routing
        topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        # 2. Flatten for Capacity Check
        N, K = topk_indices.shape
        capacity = max(
            int(math.ceil(N * self.top_k / self.num_total_experts * self.capacity_factor)), 4
        )

        idx_flat = topk_indices.view(-1)
        w_flat = topk_weights.view(-1)

        em = F.one_hot(idx_flat, self.num_total_experts).to(torch.int32)
        pri = torch.cumsum(em, dim=0)
        active_pri = pri[torch.arange(N * K, device=x.device), idx_flat]

        # 3. Create Gather Index
        mask = active_pri <= capacity
        dest_idx = idx_flat * capacity + (active_pri - 1)

        gather_index = torch.full(
            (self.num_total_experts * capacity,), -1, dtype=torch.long, device=x.device
        )
        orig_tokens = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, K).reshape(-1)

        gather_index.scatter_(0, dest_idx[mask], orig_tokens[mask])

        # Permuted Inputs
        safe_idx = gather_index.clone()
        safe_idx[gather_index == -1] = 0
        perm_in = x[safe_idx]
        perm_in[gather_index == -1] = 0.0

        # Permuted Weights
        perm_w = torch.zeros(self.num_total_experts * capacity, dtype=x.dtype, device=x.device)
        perm_w.scatter_(0, dest_idx[mask], w_flat[mask])

        return perm_in, perm_w, gather_index, capacity
