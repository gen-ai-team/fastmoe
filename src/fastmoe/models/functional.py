import torch


def permute_for_ep(x_flat, topk_idx, topk_w, num_experts, capacity):
    B, K = topk_idx.shape
    H = x_flat.shape[1]
    topk_idx_flat = topk_idx.view(-1)

    sorted_expert_ids, sorted_indices = torch.sort(topk_idx_flat)

    perm_in = torch.zeros(num_experts, capacity, H, dtype=x_flat.dtype, device=x_flat.device)
    perm_w = torch.zeros(num_experts, capacity, dtype=topk_w.dtype, device=topk_w.device)
    gather_idx = torch.full((num_experts, capacity), -1, dtype=torch.long, device=x_flat.device)

    k_source_idx = torch.full((num_experts, capacity), -1, dtype=torch.long, device=x_flat.device)

    flat_indices = torch.arange(B, device=x_flat.device).repeat_interleave(K)

    for i in range(num_experts):
        mask = sorted_expert_ids == i
        if not mask.any():
            continue

        requests = sorted_indices[mask]
        count = min(len(requests), capacity)
        requests = requests[:count]

        batch_idx = flat_indices[requests]

        perm_in[i, :count] = x_flat[batch_idx]
        perm_w[i, :count] = topk_w.view(-1)[requests]
        gather_idx[i, :count] = batch_idx

        # Save the source mapping for gradients
        k_source_idx[i, :count] = requests

    return perm_in.view(-1, H), perm_w.view(-1), gather_idx.view(-1), k_source_idx.view(-1)


def unpermute_from_ep(combined_output, gather_idx, perm_w, N, H):
    """
    Scatters expert outputs back to original token positions.
    """
    # combined: [NumExperts * Capacity, H]
    # perm_w:   [NumExperts * Capacity]

    w_casted = perm_w.to(dtype=combined_output.dtype)
    weighted_out = combined_output * w_casted.unsqueeze(1)

    # Output buffer: [N, H]
    buffer = torch.zeros(N, H, dtype=combined_output.dtype, device=combined_output.device)

    valid_mask = gather_idx != -1
    valid_indices = gather_idx[valid_mask]
    valid_data = weighted_out[valid_mask]

    buffer.index_add_(0, valid_indices, valid_data)

    return buffer
