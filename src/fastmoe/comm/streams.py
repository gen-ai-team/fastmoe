import enum
import functools

import torch


class Streams(enum.StrEnum):
    COMM = "comm"
    COMPUTE = "compute"


@functools.lru_cache
def get_ep_streams() -> dict[Streams, torch.cuda.Stream]:
    """
    Creates the three streams required for maximal overlap in Expert Parallelism.

    We use three streams to allow three distinct types of work to happen simultaneously:
    1. COMM: Driving the NCCL All-to-All operations.
    2. COMPUTE: Running compute operations (Attention, LayerNorm, Gate, MoE).
    """
    return {
        Streams.COMM: torch.cuda.Stream(),
        Streams.COMPUTE: torch.cuda.Stream(),
    }
