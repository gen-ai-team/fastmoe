import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
from torch.profiler import ProfilerActivity, profile, record_function

from fastmoe.config import MoEScale, get_config
from fastmoe.consts import MoEImplementation
from fastmoe.models.tiny_model import TinyModel

# =============================================================================
#  Distributed Helpers
# =============================================================================


def setup():
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        dist.init_process_group("nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    torch.cuda.set_device(local_rank)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
#  Benchmarking Logic (The Worker)
# =============================================================================


def benchmark_step(model, x, desc, profile_trace=False) -> tuple[float, float]:
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    steps = 10
    if profile_trace:
        log_path = f"./logs/{desc}"
        os.makedirs(log_path, exist_ok=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path),
        ) as prof:
            for i in range(steps):
                with record_function(f"Model Step: {i}"):
                    out = model(x)
                    out.mean().backward()
                prof.step()
        return 0.0, 0.0
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            out = model(x)
            out.mean().backward()
        end.record()
        torch.cuda.synchronize()
        avg_time = start.elapsed_time(end) / steps
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        return avg_time, peak_mem


def _worker_entrypoint():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    cfg = get_config(MoEScale.GIGACHAT_10B)
    local_batch_size = 32

    if rank == 0:
        logger.info(f"Running Distributed Benchmark on {world_size} GPUs")
        logger.info(f"Workload: Hidden={cfg.hidden_dim} | Batch={local_batch_size}")

    device = torch.device(f"cuda:{rank}")
    x = torch.randn(
        local_batch_size,
        cfg.seq_len,
        cfg.hidden_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    # 1. STANDARD
    model_std = (
        TinyModel(
            in_dim=cfg.hidden_dim,
            dim=cfg.hidden_dim,
            n_heads=cfg.n_heads,
            ff_dim=cfg.ff_dim,
            n_layers=cfg.n_layers,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            implementation=MoEImplementation.STANDARD,
            stream0=None,
            stream1=None,
            use_moe=True,
        )
        .to(device)
        .to(torch.bfloat16)
    )

    if rank == 0:
        logger.info("Benchmarking Standard...")
    ms_std, mem_std = benchmark_step(model_std, x, "standard_moe_timing", profile_trace=False)
    benchmark_step(model_std, x, "standard_moe", profile_trace=True)

    del model_std
    torch.cuda.empty_cache()

    # 2. PIPELINED
    stream0 = torch.cuda.Stream(device=device, priority=-1)
    stream1 = torch.cuda.Stream(device=device, priority=-1)

    model_pipe = (
        TinyModel(
            in_dim=cfg.hidden_dim,
            dim=cfg.hidden_dim,
            n_heads=cfg.n_heads,
            ff_dim=cfg.ff_dim,
            n_layers=cfg.n_layers,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            implementation=MoEImplementation.FAST,
            stream0=stream0,
            stream1=stream1,
            use_moe=True,
        )
        .to(device)
        .to(torch.bfloat16)
    )

    if rank == 0:
        logger.info("Benchmarking Fast (Pipelined)...")
    ms_pipe, mem_pipe = benchmark_step(model_pipe, x, "fast_moe_timing", profile_trace=False)
    benchmark_step(model_pipe, x, "fast_moe", profile_trace=True)

    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"{'Metric':<15} | {'Standard':<15} | {'Pipelined':<15} | {'Delta':<15}")
        logger.info("-" * 60)
        logger.info(
            f"{'Latency (ms)':<15} | {ms_std:<15.2f} | {ms_pipe:<15.2f} | {ms_std / ms_pipe:.2f}x Speedup"  # noqa
        )
        mem_delta = mem_std - mem_pipe
        mem_str = f"{mem_delta:.0f} MB Saved" if mem_delta > 0 else f"{abs(mem_delta):.0f} MB Cost"
        logger.info(f"{'Memory (MB)':<15} | {mem_std:<15.0f} | {mem_pipe:<15.0f} | {mem_str}")
        logger.info("=" * 60)

    cleanup()


# =============================================================================
#  Launcher
# =============================================================================


def _spawn_worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    try:
        _worker_entrypoint()
    except Exception as e:
        logger.error(f"Rank {rank} failed: {e}")
        raise e


def run_benchmark():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _worker_entrypoint()
        return
    world_size = torch.cuda.device_count()
    if world_size < 2:
        logger.warning(f"Detected {world_size} GPU. PP requires at least 2 for effective demo.")
    logger.info(f"Spawning {world_size} workers for EP Benchmark...")
    mp.spawn(_spawn_worker, args=(world_size,), nprocs=world_size, join=True)
