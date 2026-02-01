import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k, capacity_factor=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Capacity = (Tokens / Experts) * CapacityFactor
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # Input Normalization (Handle 2D or 3D input)
        # We need num_tokens for capacity calculation and x_flat for gating.
        if x.dim() == 2:
            # Input: [Tokens, Dim]
            num_tokens, D = x.shape
            x_flat = x
        elif x.dim() == 3:
            # Input: [Batch, Seq, Dim]
            B, S, D = x.shape
            num_tokens = B * S
            x_flat = x.view(-1, D)
        else:
            raise ValueError(f"Router expects 2D or 3D input, got shape {x.shape}")

        # 2. Gating
        logits = self.gate(x_flat)  # [Tokens, Experts]
        scores = F.softmax(logits, dim=1)

        # 3. Top-K Selection
        # topk_weights: [Tokens, K], topk_indices: [Tokens, K]
        topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=1)

        # 4. Calculate Capacity
        # How many tokens *can* each expert handle?
        capacity = int((num_tokens / self.num_experts) * self.capacity_factor)
        # Ensure minimal capacity to avoid degenerate cases
        capacity = max(capacity, 4)

        # 5. Create Indexing Masks
        # Map inputs[i] -> (Expert_E, Slot_S)

        # A. Mask for each expert: [Tokens, K, Experts]
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).to(torch.int32)

        # B. Position in Expert Buffer (CumSum along tokens)
        # token_priority: [Tokens, K, Experts] - The N-th token destined for Expert E
        token_priority = torch.cumsum(expert_mask, dim=0) * expert_mask

        # C. Filter by Capacity
        # valid_mask: [Tokens, K, Experts] - 1 if token fits in capacity, 0 otherwise
        valid_mask = (token_priority > 0) & (token_priority <= capacity)

        # 6. Calculate Final Gather Indices
        # Map: (Expert_E, Slot_S) -> Input_Token_Index
        # index tensor of shape [Experts * Capacity] pointing to original tokens.

        # Flatten masks to [Tokens * K, Experts]
        valid_mask_flat = valid_mask.view(-1, self.num_experts)
        token_priority_flat = token_priority.view(-1, self.num_experts)

        # Initialize gather_index with -1 (padding)
        # Shape: [Num_Experts * Capacity]
        gather_index = torch.full(
            (self.num_experts * capacity,), -1, dtype=torch.long, device=x.device
        )

        # Map 'original_token_idx' to the destination slot.
        # original_token_idx repeats K times for the K choices.
        row_idx = (
            torch.arange(num_tokens * self.top_k, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.num_experts)
        )
        original_token_idx = row_idx // self.top_k

        # Calculate destination index in the flat expert buffer:
        # dest_idx = Expert_ID * Capacity + (Priority - 1)
        expert_ids_range = torch.arange(self.num_experts, device=x.device).unsqueeze(0)
        dest_idx = expert_ids_range * capacity + (token_priority_flat - 1)

        # Only apply valid assignments (where capacity wasn't exceeded)
        active = valid_mask_flat.bool()

        # Scatter the original token indices into the gather map
        gather_index.scatter_(0, dest_idx[active].flatten(), original_token_idx[active].flatten())

        # 7. Save Weights for Combine
        # We need to store weights in the exact same layout as the permuted tokens
        # weights_flat: [Tokens * K] (broadcasted to experts dim for indexing logic)
        weights_flat = topk_weights.view(-1).unsqueeze(1).expand(-1, self.num_experts)

        # permuted_weights: [Num_Experts * Capacity]
        permuted_weights = torch.zeros(
            (self.num_experts * capacity,), dtype=x.dtype, device=x.device
        )
        permuted_weights.scatter_(0, dest_idx[active].flatten(), weights_flat[active].flatten())

        # 8. Permute Inputs (Dispatch Preparation)
        # We use the gather_index to pull tokens.
        # Handle padding (-1 indices): Clamp to 0 temporarily, gather, then zero out.
        safe_gather_index = gather_index.clamp(min=0)

        # permuted_inputs: [Experts * Capacity, Dim]
        permuted_inputs = x_flat[safe_gather_index]

        # Apply padding mask (where index was -1, zero out the token)
        padding_mask = (gather_index == -1).unsqueeze(1)
        permuted_inputs.masked_fill_(padding_mask, 0.0)

        return permuted_inputs, permuted_weights, gather_index, capacity
