import torch

# --- PATCHING (Mock CUDA) ---
# We must patch BEFORE importing modules that might initialize CUDA things
from mocks import MockEvent, MockStream

torch.cuda.Event = MockEvent
torch.cuda.stream = lambda s: s
torch.cuda.current_stream = lambda: MockStream()
# ----------------------------

# We import the kernel wrapper to ensure it's using the CPU fallback
from fastmoe.kernels.ops import weighted_scatter_add  # noqa
from fastmoe.layers.moe import MoEFeedForward  # noqa
from fastmoe.models.tiny_model import TinyModel  # noqa


def test_moe_forward_backward_cpu():
    """
    Verifies that the MoE layer and the Scatter-Add kernel (CPU fallback)
    produce valid outputs and propagate gradients correctly.
    """
    torch.manual_seed(42)

    # Config
    B, T, D = 2, 8, 32
    ff_dim = 64
    num_experts = 4
    top_k = 2

    # Instantiate Layer directly to test isolation
    moe = MoEFeedForward(D, ff_dim, num_experts=num_experts, top_k=top_k)

    # Input Data
    x = torch.randn(B, T, D, requires_grad=True)
    target = torch.randn(B, T, D)

    print("\n--- [Test] Forward Pass ---")
    out = moe(x)

    # 1. Check Output Shape
    assert out.shape == (B, T, D), f"Shape Mismatch: {out.shape} != {(B, T, D)}"
    print("Shape check passed.")

    # 2. Check for NaNs
    assert not torch.isnan(out).any(), "Output contains NaNs"
    print("NaN check passed.")

    print("\n--- [Test] Backward Pass ---")
    loss = ((out - target) ** 2).sum()
    loss.backward()

    # 3. Check Gradient Flow to Input
    assert x.grad is not None, "Input x has no gradient"
    assert not torch.isnan(x.grad).any(), "Input gradient has NaNs"
    print("Input gradient check passed.")

    # 4. Check Gradient Flow to Router
    assert moe.router.weight.grad is not None, "Router has no gradient"
    print("Router gradient check passed.")

    # 5. Check Gradient Flow to Experts
    # Note: With random data, it is statistically possible (though unlikely)
    # that an expert receives 0 tokens.
    active_experts = 0
    for _, expert in enumerate(moe.experts):
        # The first layer of the expert MLP
        w_grad = expert[0].weight.grad
        if w_grad is not None:
            active_experts += 1
            assert not torch.isnan(w_grad).any()

    print(f"Gradients found for {active_experts}/{num_experts} experts.")
    assert active_experts > 0, "No experts received gradients!"


def test_full_model_integration():
    """
    Verifies the TinyModel structure with MoE blocks enabled.
    """
    print("\n--- [Test] Full Model Integration ---")
    in_dim, dim = 16, 32
    model = TinyModel(in_dim=in_dim, dim=dim, n_heads=4, ff_dim=64, n_layers=2, num_experts=4)

    x = torch.randn(2, 10, in_dim)  # [B, T, in_dim]
    out = model(x)

    assert out.shape == (2, 10, in_dim)
    print("Full model forward pass successful.")


def test_scatter_add_correctness():
    """
    Numerically verify the Scatter-Add CPU fallback against a naive loop.
    This ensures our 'Mock' logic in ops.py is mathematically correct.
    """
    print("\n--- [Test] Kernel Correctness (CPU) ---")
    N, D = 10, 4

    # Inputs
    src = torch.randn(N, D)
    weights = torch.rand(N)
    # Map 10 items into 5 slots
    indices = torch.randint(0, 5, (N,))
    out_shape = (5, D)

    # 1. Our CPU Fallback Implementation
    out_fast = weighted_scatter_add(src, indices, weights, out_shape)

    # 2. Naive Ground Truth
    out_gt = torch.zeros(out_shape)
    for i in range(N):
        dest_idx = indices[i].item()
        w = weights[i].item()
        val = src[i]
        out_gt[dest_idx] += val * w

    # Check
    diff = (out_fast - out_gt).abs().max()
    print(f"Max Difference: {diff.item()}")
    assert diff < 1e-5, "CPU Fallback logic is mathematically wrong"
    print("Numerical verification passed.")


if __name__ == "__main__":
    test_scatter_add_correctness()
    test_moe_forward_backward_cpu()
    test_full_model_integration()
