import torch

# --- PATCHING (Mock CUDA) ---
from mocks import MockEvent, MockStream

torch.cuda.Event = MockEvent
torch.cuda.stream = lambda s: s
torch.cuda.current_stream = lambda: MockStream()
# ----------------------------

from fastmoe.consts import MoEImplementation  # noqa
from fastmoe.kernels.ops import weighted_scatter_add  # noqa
from fastmoe.models.tiny_model import MoEFeedForward, PipelinedMoEBlock, TinyModel  # noqa


def test_moe_forward_backward_cpu():
    torch.manual_seed(42)
    B, T, D = 2, 8, 32
    ff_dim = 64
    num_experts = 4
    top_k = 2

    # Added implementation arg
    moe = MoEFeedForward(
        D, ff_dim, num_experts=num_experts, top_k=top_k, implementation=MoEImplementation.STANDARD
    )

    x = torch.randn(B, T, D, requires_grad=True)
    target = torch.randn(B, T, D)

    print("\n--- [Test] Forward Pass ---")
    out = moe(x)
    assert out.shape == (B, T, D)
    print("Shape check passed.")

    print("\n--- [Test] Backward Pass ---")
    loss = ((out - target) ** 2).sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("Input gradient check passed.")


def test_full_model_integration():
    print("\n--- [Test] Full Model Integration ---")
    in_dim, dim = 16, 32
    # Updated args to match new TinyModel signature
    model = TinyModel(
        in_dim=in_dim,
        dim=dim,
        n_heads=4,
        ff_dim=64,
        n_layers=2,
        num_experts=4,
        top_k=2,
        implementation=MoEImplementation.STANDARD,
        stream0=None,
        stream1=None,
        use_moe=True,
    )

    x = torch.randn(2, 10, in_dim)
    out = model(x)
    assert out.shape == (2, 10, in_dim)
    print("Full model forward pass successful.")


def test_pipeline_cpu_mock():
    print("\n--- [Test] Pipeline Logic (CPU) ---")
    _, dim = 16, 32
    s0 = MockStream()
    s1 = MockStream()

    # Updated PipelinedMoEBlock args
    pipe_block = PipelinedMoEBlock(
        dim=dim, n_heads=4, ff_dim=64, num_experts=4, top_k=2, stream0=s0, stream1=s1
    )

    x = torch.randn(4, 10, dim)
    out = pipe_block(x)
    assert out.shape == (4, 10, dim)
    print("Pipeline forward pass successful.")


def test_scatter_add_correctness():
    # ... (Keep existing implementation) ...
    print("\n--- [Test] Kernel Correctness (CPU) ---")
    N, D = 10, 4
    src = torch.randn(N, D)
    weights = torch.rand(N)
    indices = torch.randint(0, 5, (N,))
    out_shape = (5, D)

    out_fast = weighted_scatter_add(src, indices, weights, out_shape)

    out_gt = torch.zeros(out_shape)
    for i in range(N):
        dest_idx = indices[i].item()
        w = weights[i].item()
        val = src[i]
        out_gt[dest_idx] += val * w

    diff = (out_fast - out_gt).abs().max()
    assert diff < 1e-5
    print("Numerical verification passed.")


if __name__ == "__main__":
    test_scatter_add_correctness()
    test_moe_forward_backward_cpu()
    test_pipeline_cpu_mock()
    test_full_model_integration()
