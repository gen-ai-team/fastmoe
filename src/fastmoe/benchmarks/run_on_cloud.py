import os
import time
import typing

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
from torch.profiler import ProfilerActivity, profile, schedule

from fastmoe.config import Config, get_cfg
from fastmoe.models.tiny_model import TinyModel

TRACE_FILENAME: typing.Final[str] = "pipelined_moe_with_comm_vs_compute_overlap.json"


def log_rank0(rank: int, msg: str | Config) -> None:
    if rank == 0:
        logger.info(msg)


# ==========================================
# Worker
# ==========================================
def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12370"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    cfg: Config = get_cfg()
    log_rank0(rank, cfg)

    model = TinyModel(cfg, dist.group.WORLD).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(cfg.moe.batch_size, cfg.moe.seqlen, cfg.moe.hidden_dim).cuda()

    def trace_handler(p):
        dist.barrier()
        if rank == 0:
            abs_path = os.path.abspath(TRACE_FILENAME)
            p.export_chrome_trace(abs_path)
            log_rank0(rank, f"\nTrace saved to: {abs_path}")

    try:
        log_rank0(rank, "Warming up...")
        for _ in range(cfg.moe.warmup_steps):
            loss = model(data).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        log_rank0(rank, "Profiling Streams...")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=cfg.moe.active_steps, repeat=1),
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True,
        ) as p:
            for _ in range(1 + 1 + cfg.moe.active_steps):
                loss = model(data).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                p.step()

        time.sleep(2)
        dist.barrier()

    finally:
        dist.destroy_process_group()


def run_experiment():
    world_size = 2
    print(f"Starting processes for {world_size} GPUs...")
    mp.start_processes(
        worker,
        args=(world_size,),
        nprocs=world_size,
        join=True,
        start_method="fork",
    )
