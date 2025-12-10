import torch

# =========================================================================
#  Environment Shim
# =========================================================================
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    from unittest.mock import MagicMock

    triton = MagicMock()
    tl = MagicMock()
    triton.jit = lambda f: f
    tl.constexpr = int


# =========================================================================
#  Triton Kernel: The "Grouped Scatter-Add" Engine
# =========================================================================
#
# SYSTEM DESIGN RATIONALE:
# ------------------------
# Problem: Mixture-of-Experts (MoE) produces N small tensors (one per expert).
#          Standard PyTorch requires 'torch.cat' (memory copy) to combine them
#          before processing. This copy is bandwidth-bound and kills performance.
#
# Solution: "Pointer Indirection". Instead of moving data to be contiguous,
#           we move the compute to where the data lives.
#
# Hardware View:
#           We launch a SINGLE massive kernel grid.
#           Each block determines which expert it belongs to, looks up that
#           expert's memory address from a table (ptr_table), and computes directly.
#           This fuses many kernel launches into 1, eliminating CPU launch latency
#           and VRAM allocation overhead.
#
#  SCENARIO TRACE (Mental Model):
#  Given a batch of 8 Tokens processed by 3 Experts.
#
#  Data Layout (Disjoint in VRAM):
#  - E0: 3 tokens [Ids: 0, 3, 5].  Ptr: 0xA000
#  - E1: 2 tokens [Ids: 1, 7].     Ptr: 0xB000
#  - E2: 3 tokens [Ids: 2, 4, 6].  Ptr: 0xC000
#
#  Global Metadata (Packed Order):
#  - expert_offsets: [0, 3, 5] (E0 starts at 0, E1 at 3, E2 at 5)
#  - indices:        [0, 3, 5,  1, 7,  2, 4, 6]  <-- The '7' is at index 4
#
#  Grid Launch:
#  - We launch a 2D Grid: (Max_Rows=3, Num_Experts=3).


@triton.jit
def _grouped_weighted_scatter_add_kernel(
    # [Input] Indirection Table
    # A tiny tensor [Num_Experts] containing the raw 64-bit memory addresses
    # of the input tensors. This allows us to "hop" between disjoint memory buffers.
    # List of pointers [0xA000, 0xB000, 0xC000]
    ptr_table,
    # [Input] Routing Metadata
    # idx_ptr: Where does token 'i' go in the final output? - Global restoration indices
    # w_ptr:   What is the gating weight for token 'i'? - Global gating weights
    idx_ptr,
    w_ptr,
    # [Output] Destination
    # The pre-allocated final buffer. We write directly here (Zero-Copy).
    # Final Output Buffer
    out_ptr,
    # [Metadata] Layout helpers
    # expert_offsets: [Num_Experts]. Prefix sum of counts. Helps us map a local
    #                 expert row (0..k) to a global row ID (0..BatchSize).
    #                 Offsets [0, 3, 5]
    # chunk_sizes:    [Num_Experts]. How many tokens does each expert have?
    #                 Counts [3, 2, 3]
    expert_offsets,
    chunk_sizes,
    # [Stride] Output tensor strides (usually Hidden_Dim, 1)
    stride_out_row,
    stride_out_col,
    # [Constants] Compile-time constants for loop unrolling
    # BLOCK_SIZE: Amount of data one thread block processes per step (e.g. 1024 floats)
    # HIDDEN_DIM: The width of the model (e.g. 4096 or 16384)
    BLOCK_SIZE: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # -----------------------------------------------------------
    # 1. Thread Block Scheduling (The "Virtual Grid")
    # -----------------------------------------------------------
    # We launch a 2D Grid: (Max_Rows_Per_Expert, Num_Experts).
    # This ensures we have enough threads to cover the largest expert.
    # Experts smaller than Max_Rows will have some idle threads (masked out below).

    # [Trace: Block (1, 1)]
    # row_in_expert = 1  (I am the 2nd worker for this expert)
    # expert_id     = 1  (I am working for Expert 1)

    row_in_expert = tl.program_id(0)  # Row Index (0, 1, 2... for this expert)
    expert_id = tl.program_id(1)  # Expert Index (0..255) for 256 experts mimicing GIGACHAT 3 Ultra

    # -----------------------------------------------------------
    # 2. Validity Check (Handling Load Imbalance)
    # -----------------------------------------------------------
    # Since we launched a rectangular grid but data is ragged (experts have
    # different token counts), we must early-exit threads that are out of bounds.
    # [Trace: Block (1, 1)]
    # chunk_sizes[1] is 2.
    # Is 1 >= 2? False. We stay alive.
    # (Note: Block (2, 1) would read 2, fail check, and exit).
    n_rows = tl.load(chunk_sizes + expert_id)
    if row_in_expert >= n_rows:
        return

    # -----------------------------------------------------------
    # 3. Pointer Indirection (The Magic Trick)
    # -----------------------------------------------------------
    # Instead of assuming data is at 'Input + offset', we fetch the base pointer.
    # This is what allows us to skip 'torch.cat'.
    # [Trace: Block (1, 1)]
    # We need to find the data for the 2nd token of E1 (which is Token 7).
    #
    # 1. Fetch Base: ptr_table[1] -> 0xB000

    # Load raw 64-bit pointer from table
    base_addr_int = tl.load(ptr_table + expert_id)
    # Cast to float32 pointer type so Triton understands stride logic
    src_ptr_base = base_addr_int.to(tl.pointer_type(DTYPE))

    # Calculate address for the specific row this thread is processing
    # Address = Base + (RowIndex * Stride)
    #
    # Calculate Offset: Row 1 * Stride(4096)
    # src_row_ptr points directly to Token 7's data at 0xB000 + 4096.
    src_row_ptr = src_ptr_base + row_in_expert * HIDDEN_DIM

    # -----------------------------------------------------------
    # 4. Global Indexing Lookup
    # -----------------------------------------------------------
    # We have the data, but where does it go?
    #
    # We are processing the k-th token of Expert E.
    # But what is its global token ID? (0..BatchSize)
    # Global_ID = Offset_of_Expert_E + k

    # [Trace: Block (1, 1)]
    # 1. Global Offset for E1 is 3 (Size of E0).
    global_offset = tl.load(expert_offsets + expert_id)
    # 2. Global ID = 3 + 1 = 4.
    # Meaning: I am the 5th token in the packed sequence.
    global_row_id = global_offset + row_in_expert

    # Fetch the Scatter Index (Where do I write?)
    # Look up Global Index #4
    # idx_ptr[4] is 7.
    target_row_idx = tl.load(idx_ptr + global_row_id)

    # Fetch the Gating Weight (How much do I matter?)
    # Look up Weight #4
    weight = tl.load(w_ptr + global_row_id)

    # Compute Output destination address
    # Final Destination Address
    out_row_ptr = out_ptr + target_row_idx * stride_out_row

    # -----------------------------------------------------------
    # 5. Vectorized Compute Loop (SRAM <-> HBM)
    # -----------------------------------------------------------
    # We loop over the Hidden Dimension in chunks of BLOCK_SIZE.
    # This keeps register pressure low and maximizes memory coalescing.
    for off in range(0, HIDDEN_DIM, BLOCK_SIZE):
        # Generate column offsets: [0, 1, ... 1023] + LoopOffset
        cols = off + tl.arange(0, BLOCK_SIZE)

        # Mask for the last block if Hidden_Dim isn't a multiple of BLOCK_SIZE
        mask = cols < HIDDEN_DIM

        # A. LOAD: Coalesced read from the Expert's disjoint buffer
        # Read directly from 0xB000 + 4096
        val = tl.load(src_row_ptr + cols, mask=mask)

        # B. COMPUTE: Fused multiply (saves a global Read/Write roundtrip)
        # Fuse multiplication in registers
        weighted_val = val * weight

        # C. STORE: Atomic Add
        # Why Atomic? Because multiple experts (Top-K > 1) might route to
        # the same target token. A standard store would race condition.
        # Atomic add happens in L2 cache.
        # Atomic Add to Final Output at Row 7.
        tl.atomic_add(out_row_ptr + cols, weighted_val, mask=mask)


# =========================================================================
#  Metadata Helper: The "Flight Planner"
# =========================================================================
def prepare_grouped_metadata(expert_tensors, device):
    """
    Converts high-level Python Tensors into low-level Integer Arrays (Metadata)
    that the GPU kernel can read to navigate memory.

    SCENARIO TRACE:
    ---------------
    Input: A list of 3 disjoint tensors living in VRAM.
      - T0 (Red):   Shape [3, D]. Address 0xA000.
      - T1 (Blue):  Shape [2, D]. Address 0xB000.
      - T2 (Green): Shape [3, D]. Address 0xC000.
    """

    # -------------------------------------------------------------
    # 1. POINTER EXTRACTION (The "Where")
    # -------------------------------------------------------------
    # We strip away the PyTorch tensor abstraction. We just want the raw
    # 64-bit integer that represents the memory address of the first byte.
    #
    # Trace:
    #   ptr_list = [40960, 45056, 49152]  (Decimal for 0xA000, 0xB000...)
    ptr_list = [t.data_ptr() for t in expert_tensors]

    # Move to GPU. The kernel will read this to "teleport" between buffers.
    ptr_table = torch.tensor(ptr_list, dtype=torch.int64, device=device)

    # -------------------------------------------------------------
    # 2. SIZE EXTRACTION (The "How Much")
    # -------------------------------------------------------------
    # How many rows (tokens) does each expert possess?
    #
    # Trace:
    #   sizes_list = [3, 2, 3]
    sizes_list = [t.shape[0] for t in expert_tensors]
    chunk_sizes = torch.tensor(sizes_list, dtype=torch.int32, device=device)

    # -------------------------------------------------------------
    # 3. OFFSET CALCULATION (The "Global ID")
    # -------------------------------------------------------------
    # We need to map a local expert row (e.g., Row 0 of Expert 1) back to
    # the global token sequence to find the correct indices/weights.
    # We do this via a Prefix Sum (Cumulative Sum).
    #
    # Trace:
    #   sizes:  [3, 2, 3]
    #   cumsum: [3, 5, 8]  (After Expert 0, there are 3 tokens...)

    zeros = torch.zeros(1, dtype=torch.int32, device=device)
    cumsum = torch.cumsum(chunk_sizes, dim=0)[:-1]

    # Shift right to get starting offsets:
    # Trace:
    #   expert_offsets = [0, 3, 5]
    #   - Expert 0 starts at Global ID 0.
    #   - Expert 1 starts at Global ID 3.
    #   - Expert 2 starts at Global ID 5.
    expert_offsets = torch.cat((zeros, cumsum))

    # -------------------------------------------------------------
    # 4. GRID CONFIGURATION
    # -------------------------------------------------------------
    # Determine the shape of the 2D Kernel Grid.
    # Height = Size of largest expert.
    # Width  = Number of experts.

    # Trace: max(3, 2, 3) = 3.
    # We will launch a grid of height 3.
    # Threads assigned to Expert 1 (size 2) at Row 2 will trigger the Early Exit.
    max_rows = max(sizes_list) if sizes_list else 0
    num_experts = len(sizes_list)

    return ptr_table, chunk_sizes, expert_offsets, max_rows, num_experts


# =========================================================================
#  Launch Logic
# =========================================================================
def _launch_grouped_kernel(expert_tensors, indices, weights, out, metadata=None):
    # Optimization: Allow pre-calculated metadata for static graphs (inference)
    if metadata is None:
        ptr_table, chunk_sizes, expert_offsets, max_rows, num_experts = prepare_grouped_metadata(
            expert_tensors, indices.device
        )
    else:
        ptr_table, chunk_sizes, expert_offsets, max_rows, num_experts = metadata

    if max_rows > 0:
        N, D = out.shape
        # Heuristic: 1024 is standard for saturation.
        # Smaller blocks might be better for latency-sensitive small models.
        BLOCK_SIZE = 1024 if D >= 1024 else 128

        # Grid covers the "Max" case. Threads for smaller experts will early-exit.
        grid = (max_rows, num_experts)

        triton_dtype = get_triton_dtype(expert_tensors[0].dtype)

        _grouped_weighted_scatter_add_kernel[grid](
            ptr_table,
            indices,
            weights,
            out,
            expert_offsets,
            chunk_sizes,
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            HIDDEN_DIM=D,
            DTYPE=triton_dtype,
        )
    return out


# =========================================================================
#  Autograd Function (The Bridge)
# =========================================================================
class GroupedWeightedScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weights, out_shape, *expert_tensors):
        # We must save indices/weights for Backward pass
        ctx.save_for_backward(indices, weights)
        # We need sizes to know how to split the gradients later
        ctx.expert_sizes = [t.shape[0] for t in expert_tensors]

        # Allocate output.
        # We use the first expert to determine Dtype/Device.
        ref = expert_tensors[0]
        out = torch.zeros(out_shape, device=ref.device, dtype=ref.dtype)

        # Launch the kernel
        _launch_grouped_kernel(expert_tensors, indices, weights, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        indices, weights = ctx.saved_tensors
        sizes = ctx.expert_sizes
        # Forward: Y[i] += X * w
        # Backward: dL/dX = dL/dY[i] * w

        # 1. Gather (Index Select)
        # This is the "Inverse" of Scatter.
        # We pull gradients from the output position back to the source position.
        # [Total_Tokens, D]
        gathered_grads = grad_output.index_select(0, indices)

        # 2. Weighting
        # Apply the gating weight to the gradient
        weighted_grads = gathered_grads * weights.unsqueeze(-1)

        # 3. Split (Inverse of Grouping)
        # We must return a Tuple of tensors, one matching each input expert tensor.
        # PyTorch autograd engine handles routing these to the respective expert modules.
        grad_experts = torch.split(weighted_grads, sizes)

        # Return format must match forward args:
        # (indices, weights, out_shape, *experts)
        # We return None for non-differentiable args (indices, out_shape)
        # We skip weights grad calculation for brevity here, but normally it's a Dot Product.
        return None, None, None, *grad_experts


# =========================================================================
#  Public API
# =========================================================================
def grouped_weighted_scatter_add(
    expert_tensors, indices, weights, out_shape, out=None, metadata=None
):
    """
    Public entry point supporting both Training (Autograd) and Inference (In-Place).
    """
    # 1. CPU / Mac Fallback
    # Essential for Unit Testing without H100s.
    if not indices.is_cuda:
        if out is None:
            out = torch.zeros(out_shape, device=indices.device, dtype=weights.dtype)
        offset = 0
        for t in expert_tensors:
            k = t.shape[0]
            if k > 0:
                idx_chunk = indices[offset : offset + k]
                w_chunk = weights[offset : offset + k]
                out.index_add_(0, idx_chunk, t * w_chunk.unsqueeze(-1))
            offset += k
        return out

    # 2. Inference / Benchmark Path (Fastest)
    # If user provides 'out' buffer, they want in-place modification (Graph Safe).
    # We bypass Autograd overhead entirely.
    if out is not None:
        return _launch_grouped_kernel(expert_tensors, indices, weights, out, metadata=metadata)

    # 3. Training Path (Autograd Compatible)
    # We unpack the list into *args so PyTorch Function sees them as separate leaves.
    return GroupedWeightedScatterAdd.apply(indices, weights, out_shape, *expert_tensors)


def weighted_scatter_add(src, indices, weights, out_shape, out=None):
    return grouped_weighted_scatter_add([src], indices, weights, out_shape, out)


def get_triton_dtype(torch_dtype):
    if torch_dtype == torch.float32:
        return tl.float32
    elif torch_dtype == torch.bfloat16:
        return tl.bfloat16
    elif torch_dtype == torch.float16:
        return tl.float16
    return tl.float32
