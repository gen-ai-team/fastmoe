import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from fastmoe import consts
from fastmoe.kernels.ops import grouped_weighted_scatter_add


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        ff_dim,
        num_experts=8,
        top_k=2,
        implementation=consts.MoEImplementation.FAST,
        group: dist.ProcessGroup = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.implementation = implementation
        self.group = group

        if dist.is_initialized():
            self.world_size = dist.get_world_size(group)
            self.rank = dist.get_rank(group)
        else:
            self.world_size = 1
            self.rank = 0

        assert num_experts % self.world_size == 0, "Experts must divide evenly"
        self.num_local_experts = num_experts // self.world_size

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, ff_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(ff_dim, dim, bias=False),
                )
                for _ in range(self.num_local_experts)
            ]
        )

    def gate_and_sort(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stage 1: Local Gating and Indices Calculation
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        # Router
        logits = self.router(x_flat)
        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).type_as(x)

        # Flatten for routing
        indices_flat = topk_indices.view(-1)
        weights_flat = topk_weights.view(-1)

        # Sort tokens by expert ID to make memory access contiguous
        sort_indices = torch.argsort(indices_flat)
        expert_indices_sorted = indices_flat[sort_indices]

        # Mapping to restore original order later
        src_indices = torch.arange(x_flat.shape[0], device=x.device).repeat_interleave(self.top_k)
        reverse_map_indices = src_indices[sort_indices]
        sorted_weights = weights_flat[sort_indices]

        # Permute input data to be grouped by expert
        permuted_data = x_flat[reverse_map_indices]

        # Calculate splits for All-to-All
        # 1. Count tokens per expert [Num_Experts]
        expert_counts = torch.bincount(expert_indices_sorted, minlength=self.num_experts)

        # 2. Reshape to [World_Size, Local_Experts] to get per-rank counts
        if self.world_size > 1:
            rank_counts = expert_counts.view(self.world_size, self.num_local_experts).sum(dim=1)
        else:
            rank_counts = torch.tensor([permuted_data.shape[0]], device=x.device, dtype=torch.long)

        return permuted_data, rank_counts, reverse_map_indices, sorted_weights, (B, T)

    def dispatch_exchange(self, permuted_data, send_counts):
        """
        Stage 2: Communication (All-to-All)
        """

        if self.world_size == 1:
            return permuted_data, send_counts.tolist(), send_counts.tolist()

        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)

        send_list = send_counts.tolist()
        recv_list = recv_counts.tolist()

        total_recv = sum(recv_list)
        recv_data = torch.empty(
            total_recv, self.dim, device=permuted_data.device, dtype=permuted_data.dtype
        )

        dist.all_to_all_single(
            recv_data,
            permuted_data,
            output_split_sizes=recv_list,
            input_split_sizes=send_list,
            group=self.group,
        )

        return recv_data, recv_list, send_list

    def compute_experts(self, recv_data) -> torch.Tensor:
        """
        Stage 3: Computation
        """

        chunk_size = recv_data.shape[0] // self.num_local_experts
        remainder = recv_data.shape[0] % self.num_local_experts

        offset = 0
        results = []
        for i, expert in enumerate(self.experts):
            size = chunk_size + (1 if i < remainder else 0)
            if size > 0:
                chunk = recv_data[offset : offset + size]
                out = expert(chunk)
                results.append(out)
            else:
                results.append(
                    torch.empty(0, self.dim, device=recv_data.device, dtype=recv_data.dtype)
                )
            offset += size

        if not results:
            return torch.empty(0, self.dim, device=recv_data.device, dtype=recv_data.dtype)

        return torch.cat(results, dim=0)

    def combine_exchange(self, expert_output, recv_splits, send_splits):
        """
        Stage 4: Reverse Communication
        """

        if self.world_size == 1:
            return expert_output

        if isinstance(send_splits, torch.Tensor):
            send_splits = send_splits.tolist()

        total_back = sum(send_splits)
        final_data = torch.empty(
            total_back, self.dim, device=expert_output.device, dtype=expert_output.dtype
        )

        dist.all_to_all_single(
            final_data,
            expert_output,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
            group=self.group,
        )
        return final_data

    def unpermute(self, x_out, reverse_map_indices, sorted_weights, original_shape) -> torch.Tensor:
        """Stage 5: Local Un-permute and Scatter"""
        B, T = original_shape
        D = self.dim

        if self.implementation == consts.MoEImplementation.STANDARD:
            x_out = x_out * sorted_weights.unsqueeze(-1)
            out = torch.zeros(B * T, D, device=x_out.device, dtype=x_out.dtype)
            out.index_add_(0, reverse_map_indices, x_out)
            return out.view(B, T, D)

        elif self.implementation == consts.MoEImplementation.FAST:
            # Fast Kernel
            out = grouped_weighted_scatter_add(
                [x_out], reverse_map_indices, sorted_weights, (B * T, D)
            )
            return out.view(B, T, D)

        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        permuted, send_counts, rev_idx, weights, shape = self.gate_and_sort(x)
        recv_data, recv_splits, send_splits_list = self.dispatch_exchange(permuted, send_counts)
        expert_out = self.compute_experts(recv_data, recv_splits)
        final_data = self.combine_exchange(expert_out, recv_splits, send_splits_list)
        return self.unpermute(final_data, rev_idx, weights, shape)
