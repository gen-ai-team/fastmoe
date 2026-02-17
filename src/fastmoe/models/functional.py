import torch


def permute_for_ep(x_flat, topk_idx, topk_w, num_experts, capacity):
    """
    Permutes tokens to group them by expert.

    Args:
        x_flat: [Batch*Seq, Hidden]
        topk_idx: [Batch*Seq, TopK]
        topk_w: [Batch*Seq, TopK]
        num_experts: Total experts (world_size * local_experts)
        capacity: Max tokens per expert
    """
    B, K = topk_idx.shape
    H = x_flat.shape[1]

    # 1. Flatten indices to treat all tokens equally
    # We essentially have B*K "requests" for experts
    topk_idx_flat = topk_idx.view(-1)

    # 2. Sort by Expert ID to group them
    sorted_expert_ids, sorted_indices = torch.sort(topk_idx_flat)

    # 3. Create buffers
    perm_in = torch.zeros(num_experts, capacity, H, dtype=x_flat.dtype, device=x_flat.device)
    perm_w = torch.zeros(num_experts, capacity, dtype=topk_w.dtype, device=topk_w.device)

    # "Gather Index" maps the [NumExperts, Capacity] back to [Batch] for the backward pass
    gather_idx = torch.full((num_experts, capacity), -1, dtype=torch.long, device=x_flat.device)

    # 4. Fill the buffers
    # Fast Python approximation using masks (Vectorized)
    flat_indices = torch.arange(B, device=x_flat.device).repeat_interleave(K)

    for i in range(num_experts):
        # Find all requests for expert i
        mask = sorted_expert_ids == i
        if not mask.any():
            continue

        # Get the original token indices
        requests = sorted_indices[mask]

        # Truncate to capacity
        count = min(len(requests), capacity)
        requests = requests[:count]

        # Get original Batch indices (0..N) and TopK indices (0..K)
        # requests is index into B*K
        batch_idx = flat_indices[requests]

        perm_in[i, :count] = x_flat[batch_idx]

        # Get weights
        # We need to grab the specific weight for this k
        # Since we flattened B*K, we can index into flattened topk_w
        perm_w[i, :count] = topk_w.view(-1)[requests]

        gather_idx[i, :count] = batch_idx

    return perm_in.view(-1, H), perm_w.view(-1), gather_idx.view(-1)


def unpermute_from_ep(combined_output, gather_idx, perm_w, N, H):
    """
    Scatters expert outputs back to original token positions.
    """
    # combined: [NumExperts * Capacity, H]
    # gather_idx: [NumExperts * Capacity] (values are 0..N)
    # perm_w: [NumExperts * Capacity]

    # 1. Weight the outputs
    weighted_out = combined_output * perm_w.unsqueeze(1)

    # 2. Scatter Add
    # Output buffer: [N, H]
    buffer = torch.zeros(N, H, dtype=combined_output.dtype, device=combined_output.device)

    valid_mask = gather_idx != -1
    valid_indices = gather_idx[valid_mask]
    valid_data = weighted_out[valid_mask]

    buffer.index_add_(0, valid_indices, valid_data)

    return buffer
