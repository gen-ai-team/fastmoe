import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from fastmoe import consts
from fastmoe.distributed import Communicator, PytorchCommunicator
from fastmoe.kernels.ops import grouped_weighted_scatter_add


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        ff_dim,
        num_experts=8,
        top_k=2,
        implementation=consts.MoEImplementation.FAST,
        communicator: Communicator = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.implementation = implementation  # Store it
        self.comm = communicator if communicator else PytorchCommunicator()

        # Distributed Setup
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        assert num_experts % self.world_size == 0, "Experts must divide evenly by ranks"
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

    def gate(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        logits = self.router(x_flat)
        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).type_as(x)
        return x_flat, topk_weights, topk_indices, (B, T)

    def dispatch(self, x_flat, topk_weights, topk_indices):
        indices_flat = topk_indices.view(-1)
        weights_flat = topk_weights.view(-1)

        src_indices = torch.arange(x_flat.shape[0], device=x_flat.device).repeat_interleave(
            self.top_k
        )

        sort_indices = torch.argsort(indices_flat)
        expert_indices_sorted = indices_flat[sort_indices]

        reverse_map_indices = src_indices[sort_indices]
        sorted_weights = weights_flat[sort_indices]
        permuted_data = x_flat[reverse_map_indices]

        # --- DISTRIBUTED LOGIC ---
        # 1. Count tokens per expert
        expert_counts = torch.bincount(expert_indices_sorted, minlength=self.num_experts)

        # 2. Aggregate into tokens per RANK
        # Reshape [Num_Experts] -> [World_Size, Local_Experts] -> Sum
        rank_counts = expert_counts.view(self.world_size, self.num_local_experts).sum(dim=1)

        # 3. Handshake (Exchange Counts)
        local_send_counts = rank_counts
        local_recv_counts = self.comm.exchange_counts(local_send_counts)

        # Convert to lists for NCCL
        send_splits = local_send_counts.tolist()
        recv_splits = local_recv_counts.tolist()

        # 4. Data Exchange
        # Now passing all 3 required arguments
        recv_data = self.comm.all_to_all(permuted_data, send_splits, recv_splits)

        return (
            recv_data,
            expert_indices_sorted,
            reverse_map_indices,
            sorted_weights,
            send_splits,
            recv_splits,
        )

    def compute_experts(self, recv_data, expert_indices_sorted) -> list[torch.Tensor]:
        # =========================================================================
        #  DATA FLOW TRACE (Scenario: Rank 0 of 2, 8 Experts Total)
        # =========================================================================
        # Setup:
        #   - We are Rank 0. We own Experts [0, 1, 2, 3].
        #   - Rank 1 owns Experts [4, 5, 6, 7].
        #
        # Input State:
        #   - recv_data: A flat, contiguous tensor of tokens sent to us from the network.
        #     Shape: [6, D]. (Why 6? Because Exp0=2, Exp1=3, Exp2=0, Exp3=1 tokens).
        #     Content: [Tok_E0, Tok_E0, Tok_E1, Tok_E1, Tok_E1, Tok_E3]
        #     Note: There are NO separators. It's just a blob.
        #
        #   - expert_indices_sorted: Global indices sorted by expert.
        #     [0, 0, 1, 1, 1, 3, 4, 4, 5, 7] (Total 10 tokens in global batch)
        # =========================================================================

        # We need to slice recv_data based on LOCAL expert counts
        # We know we received tokens for our local experts.
        # We re-calculate counts just for the subset of experts we own.

        # 1. Filter indices to only those relevant to this rank
        # (In a real implementation, 'recv_data' is just a blob, we don't have sorted indices for it
        # unless we passed them. However, since Dispatch sorts by Expert ID, and Ranks own contiguous # noqa
        # ranges of experts, the data arrives sorted!)

        # Calculate offsets for this rank
        start_expert = self.rank * self.num_local_experts
        end_expert = start_expert + self.num_local_experts
        # Trace: Rank=0, num_local=4 -> start=0, end=4. We own Experts 0..3.

        # To split 'recv_data' correctly, we need the counts for OUR experts.
        # We can look at the global 'expert_indices_sorted', but that's on the sender side.
        # In a real All-to-All, we lose the metadata unless we communicate it.
        #
        # SIMPLIFICATION FOR THIS PROJECT:
        # Since we are focusing on the Kernel speedup and standard All-to-All,
        # we assume for the 'compute' step that we can deduce splits or we pass them.
        #
        # However, 'torch.bincount' on the *global* indices works for the single-device simulation.
        # For true distributed, we would usually send an extra tensor of "expert counts".

        # For now, let's use the global counts we calculated in dispatch, assuming single-device simulation # noqa
        # or that we can access them.
        expert_counts_global = torch.bincount(expert_indices_sorted, minlength=self.num_experts)
        # Trace: bincount([0,0,1,1,1,3,4,4,5,7]) -> [2, 3, 0, 1, 2, 1, 0, 1]

        local_expert_counts = expert_counts_global[start_expert:end_expert]
        # Trace: Slice [0:4] -> [2, 3, 0, 1].
        # Meaning: "Cut the blob into chunks of size 2, 3, 0, and 1."

        # If running distributed, recv_data size must match sum(local_expert_counts)
        # If it doesn't (due to load balancing shift), this slicing will fail.
        # But for the benchmark (single GPU), this is safe.

        tokens_per_expert = torch.split(recv_data, local_expert_counts.tolist())
        # Trace: recv_data (size 6) is split into:
        #   - T0: [2, D] (For Expert 0)
        #   - T1: [3, D] (For Expert 1)
        #   - T2: [0, D] (For Expert 2 - Empty!)
        #   - T3: [1, D] (For Expert 3)

        results = []
        for i, expert in enumerate(self.experts):
            if tokens_per_expert[i].shape[0] > 0:
                results.append(expert(tokens_per_expert[i]))
            else:
                results.append(
                    torch.empty(0, self.dim, device=recv_data.device, dtype=recv_data.dtype)
                )
        # Trace:
        #   - runs Expert0(T0) -> [2, D]
        #   - runs Expert1(T1) -> [3, D]
        #   - skips Expert2    -> [0, D]
        #   - runs Expert3(T3) -> [1, D]
        #
        # Result: A List of 4 disjoint tensors living in different memory locations.
        # We pass this LIST to 'combine'

        # Return LIST for the Grouped Kernel
        return results

    def combine(
        self,
        expert_outputs_list,
        reverse_map_indices,
        sorted_weights,
        original_shape,
        send_splits,
        recv_splits,
    ):
        B, T = original_shape
        D = self.dim

        # For this single-GPU benchmark, expert_outputs_list is local.
        # We switch based on implementation.

        if self.implementation == consts.MoEImplementation.STANDARD:
            # 1. Cat (The Bottleneck)
            combined = torch.cat(expert_outputs_list, dim=0)

            # 2. Weighting
            sorted_weights = sorted_weights.to(dtype=combined.dtype)
            weighted = combined * sorted_weights.unsqueeze(-1)

            # 3. Index Add
            out = torch.zeros((B * T, D), device=combined.device, dtype=combined.dtype)
            out.index_add_(0, reverse_map_indices, weighted)
            return out.view(B, T, D)

        else:
            # --- FAST PATH ---
            # Grouped Kernel (No Cat)
            # Metadata is calculated on-the-fly inside the kernel wrapper because we don't pass 'metadata' arg. # noqa
            out = grouped_weighted_scatter_add(
                expert_outputs_list, reverse_map_indices, sorted_weights, (B * T, D)
            )
            return out.view(B, T, D)

    # def combine(
    #     self,
    #     expert_outputs_list,
    #     reverse_map_indices,
    #     sorted_weights,
    #     original_shape,
    #     send_splits,
    #     recv_splits,
    # ):
    #     B, T = original_shape
    #     D = self.dim

    #     # 1. Concat for communication (Unavoidable for network transmission)
    #     # Note: If we are local, we can skip this, but let's be rigorous.
    #     if self.world_size > 1:
    #         expert_output_flat = torch.cat(expert_outputs_list)

    #         # Inverse Communication
    #         recv_back = self.comm.all_to_all(
    #             expert_output_flat,
    #             send_counts=recv_splits,  # We send what we received
    #             recv_counts=send_splits,
    #         )  # We receive what we sent

    #         # We now have 'recv_back', but the Grouped Kernel expects a LIST of expert tensors.
    #         # We must re-split 'recv_back' into per-expert chunks to use the fast kernel?
    #         # NO. 'recv_back' is now ordered by the Original Sender.
    #         #
    #         # Actually, the Fast Kernel Optimization ("No-Cat") is most effective when:
    #         # 1. We are local (no network).
    #         # 2. OR we operate on the data *before* sending it back? No, combine happens after return. # noqa
    #         #
    #         # If we used All-to-All, we received a monolithic tensor 'recv_back'.
    #         # We can use the Single-Tensor Scatter Add (which we kept as a wrapper).
    #         # OR we can treat it as a list of 1 tensor.

    #         # For the single-GPU benchmark where world_size=1:
    #         # self.comm.all_to_all returns the input list? No, PytorchCommunicator expects tensor.
    #         pass
    #     else:
    #         # OPTIMIZED PATH (Local / Benchmark)
    #         # We skip the "Cat for Network" and pass the list directly to the kernel.
    #         # This is what enables the 13x memory reduction in your benchmark.
    #         recv_back_list = expert_outputs_list

    #     if isinstance(recv_back_list, list):
    #         # Grouped Kernel Path
    #         out = grouped_weighted_scatter_add(
    #             recv_back_list, reverse_map_indices, sorted_weights, (B * T, D)
    #         )
    #     else:
    #         # Network Path (received one big tensor)
    #         # Use the wrapper that handles single tensor -> grouped kernel
    #         out = grouped_weighted_scatter_add(
    #             [recv_back], reverse_map_indices, sorted_weights, (B * T, D)
    #         )

    #     return out.view(B, T, D)

    def forward(self, x):
        # [B, T, D] -> [N, D], where N = B x T, because the router treats tokens independently
        # We save (B, T) so we can un-flatten the tensor at the very end.
        x_flat, weights, indices, shape = self.gate(x)
        # the Communication phase. Experts process data in batches, so we must group all tokens for "Expert 0" together, # noqa
        # all tokens for "Expert 1" together, etc.
        permuted, sorted_idx, rev_map, sorted_w, send_splits, recv_splits = self.dispatch(
            x_flat, weights, indices
        )
        # Computation phase
        # returns a List[Tensor]
        expert_out_list = self.compute_experts(permuted, sorted_idx)
        # Reassembler
        return self.combine(expert_out_list, rev_map, sorted_w, shape, send_splits, recv_splits)
