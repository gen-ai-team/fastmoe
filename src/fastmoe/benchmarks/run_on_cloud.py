import gc

import torch
import torch.optim as optim
from loguru import logger

from fastmoe import consts
from fastmoe.config import MoEScale, MoESetup, get_config, init_app
from fastmoe.kernels.ops import grouped_weighted_scatter_add, prepare_grouped_metadata
from fastmoe.models.tiny_model import TinyModel


def verify_correctness(expert_outputs, indices, weights, out_shape):
    logger.info("Verifying numerical correctness...")
    device = expert_outputs[0].device

    # 1. Standard
    combined = torch.cat(expert_outputs, dim=0)
    weighted = combined * weights.unsqueeze(-1)
    out_std = torch.zeros(out_shape, device=device)
    out_std.index_add_(0, indices, weighted)

    # 2. FastMoE (Grouped)
    out_fast = torch.zeros(out_shape, device=device)
    grouped_weighted_scatter_add(expert_outputs, indices, weights, out_shape, out=out_fast)

    torch.cuda.synchronize()

    diff = (out_std - out_fast).abs().max()
    tol = 1e-4
    if diff > tol:
        logger.warning(f"Numerical Mismatch! Max Diff: {diff:.6f} (Tolerance: {tol})")
    else:
        logger.success(f"Verification PASSED. Max Diff: {diff:.6e}")


def benchmark_standard(expert_outputs, indices, weights, out_shape, steps):
    """Standard: Cat -> Weight -> IndexAdd"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(steps):
        combined = torch.cat(expert_outputs, dim=0)
        weighted = combined * weights.unsqueeze(-1)
        out = torch.zeros(out_shape, device=expert_outputs[0].device)
        out.index_add_(0, indices, weighted)
    end_event.record()

    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    return start_event.elapsed_time(end_event) / steps, peak_mem


def benchmark_fastmoe_grouped(expert_outputs, indices, weights, out_shape, steps):
    """
    FastMoE: Grouped Kernel + CUDA Graphs.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = expert_outputs[0].device

    out_static = torch.zeros(out_shape, device=device)

    # --- 1. PREPARE METADATA (Static Graph Setup) ---
    metadata = prepare_grouped_metadata(expert_outputs, device)

    # Warmup
    grouped_weighted_scatter_add(
        expert_outputs, indices, weights, out_shape, out=out_static, metadata=metadata
    )
    torch.cuda.synchronize()

    # --- 2. CAPTURE GRAPH ---
    g = torch.cuda.CUDAGraph()
    out_static.zero_()

    with torch.cuda.graph(g):
        grouped_weighted_scatter_add(
            expert_outputs, indices, weights, out_shape, out=out_static, metadata=metadata
        )

    # --- 3. REPLAY ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(steps):
        out_static.zero_()
        g.replay()
    end_event.record()
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    return start_event.elapsed_time(end_event) / steps, peak_mem


def run_training_experiment(implementation: str, cfg: MoESetup):
    """
    Runs a realistic training loop for N steps using AMP (Automatic Mixed Precision).
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda")

    # 1. Initialize Model
    # FF_DIM=256 reduces weight memory, but DIM=16384 keeps activation memory massive.
    model = TinyModel(
        in_dim=cfg.hidden_dim,
        dim=cfg.hidden_dim,
        n_heads=16,
        ff_dim=256,
        n_layers=2,
        num_experts=cfg.num_experts,
        implementation=implementation,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.5)

    # We use GradScaler just in case we fall back to float16, though BF16 doesn't strictly need it.
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    logger.info(f"AMP Enabled: {amp_dtype}")

    # Dummy Data
    B, T, D = cfg.batch_size, cfg.seq_len, cfg.hidden_dim
    x = torch.randn(B, T, D, device=device)
    target = x.clone()

    logger.info(f"Running {implementation.upper()} Training Loop...")

    # 3. Warmup
    for _ in range(5):
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            y = model(x)
            loss = ((y - target) ** 2).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # 4. Profile
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats()

    start_event.record()
    for step in range(cfg.active_steps):
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            y = model(x)
            loss = ((y - target) ** 2).mean()

        scaler.scale(loss).backward()
        if step % 10 == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % 10 == 0 or step == cfg.active_steps - 1:
            loss_val = loss.item()
            curr_mem = torch.cuda.memory_allocated() / (1024**3)
            # Log Grad Norm too
            logger.info(
                f"Step {step:03d}/{cfg.active_steps} | Loss: {loss_val:.4f} | GradNorm: {total_norm:.4f} | Mem: {curr_mem:.2f} GB"  # noqa
            )

    end_event.record()
    torch.cuda.synchronize()

    avg_time = start_event.elapsed_time(end_event) / cfg.active_steps
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    # Cleanup
    del model, optimizer, x, target
    return avg_time, peak_mem


def run_on_cloud():
    init_app()
    TARGET_SCALE = MoEScale.GIGACHAT_700B
    cfg = get_config(TARGET_SCALE)

    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    logger.info(f"Target: {cfg.scale.value.upper()} | Experts: {cfg.num_experts}")

    # --- PART 1: KERNEL BENCHMARK ---
    total_tokens = cfg.total_tokens
    chunk_size = total_tokens // cfg.num_experts
    expert_outputs = [
        torch.randn(chunk_size, cfg.hidden_dim, device=device) for _ in range(cfg.num_experts)
    ]
    indices = torch.randint(0, cfg.batch_size * cfg.seq_len, (total_tokens,), device=device)
    weights = torch.rand(total_tokens, device=device)
    out_shape = (cfg.batch_size * cfg.seq_len, cfg.hidden_dim)

    base_mem = torch.cuda.memory_allocated() / (1024**2)

    verify_correctness(expert_outputs, indices, weights, out_shape)

    logger.info("Benchmarking Kernel...")
    std_ms, std_mem = benchmark_standard(
        expert_outputs, indices, weights, out_shape, cfg.active_steps
    )
    fast_ms, fast_mem = benchmark_fastmoe_grouped(
        expert_outputs, indices, weights, out_shape, cfg.active_steps
    )

    logger.info("=" * 60)
    logger.info(f"KERNEL RESULTS | {cfg.scale.value.upper()}")
    logger.info(f"{'Metric':<15} | {'Standard':<12} | {'FastMoE':<12} | {'Delta':<10}")
    logger.info("-" * 60)
    logger.info(
        f"{'Latency (ms)':<15} | {std_ms:<12.3f} | {fast_ms:<12.3f} | {std_ms / fast_ms:.2f}x Faster"  # noqa
    )

    std_overhead = std_mem - base_mem
    fast_overhead = fast_mem - base_mem
    if fast_overhead < 1:
        fast_overhead = 1
    logger.info(
        f"{'Peak Mem (MB)':<15} | {std_mem:<12.0f} | {fast_mem:<12.0f} | {std_overhead / fast_overhead:.2f}x Less Ovhd"  # noqa
    )
    logger.info("=" * 60)

    del expert_outputs, indices, weights
    torch.cuda.empty_cache()

    # --- PART 2: TRAINING BENCHMARK ---
    logger.info("=" * 60)
    logger.info(f"TRAINING BENCHMARK | {TARGET_SCALE.value.upper()}")
    logger.info(f"Experts: {cfg.num_experts} | Hidden: {cfg.hidden_dim} | Seq: {cfg.seq_len}")
    logger.info("=" * 60)

    # 1. Run Standard
    try:
        std_time, std_mem = run_training_experiment(consts.MoEImplementation.STANDARD, cfg)
        logger.info(f"Standard: {std_time:.2f} ms/step | Peak Mem: {std_mem:.2f} GB")
    except torch.cuda.OutOfMemoryError:
        logger.error("Standard: OOM (Out Of Memory)!")
        std_time, std_mem = float("inf"), float("inf")

    # 2. Run FastMoE
    fast_time, fast_mem = run_training_experiment(consts.MoEImplementation.FAST, cfg)
    logger.info(f"FastMoE:  {fast_time:.2f} ms/step | Peak Mem: {fast_mem:.2f} GB")

    # 3. Comparison
    logger.info("-" * 60)
    if std_time != float("inf"):
        logger.success(f"Speedup: {std_time / fast_time:.2f}x")
        logger.success(f"Memory Reduction: {std_mem - fast_mem:.2f} GB Saved")
    else:
        logger.success("FastMoE enabled training where Standard failed (OOM).")
    logger.info("-" * 60)
