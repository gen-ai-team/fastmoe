import functools
import sys
from enum import Enum

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings


class MoEScale(str, Enum):
    TINY = "tiny"


class MoESetup(BaseSettings):
    scale: MoEScale = MoEScale.TINY

    # Model
    n_blocks: int = 2
    batch_size: int = 128
    proj_dim: int = 4 * 4096
    hidden_dim: int = 4096
    num_heads: int = 32
    seqlen: int = 128

    # Experts
    num_experts_per_gpu: int = 2
    top_k: int = 2

    micro_batches: int = 2

    # Steps
    warmup_steps: int = 5
    active_steps: int = 3

    comm_scaling_factor: int = 20


class Config(BaseSettings):
    moe: MoESetup
    world_size: int

    # Defaults
    logs_dir: str = Field(default="logs")
    log_level: str = Field(default="INFO")

    class Config:
        extra = "ignore"

    def __repr__(self) -> str:
        """Produces a structured, readable log message for startup."""
        return (
            f"\n{'='*50}\n"
            f"               FastMoE Configuration\n"
            f"{'='*50}\n"
            f"System:\n"
            f"  • World Size       : {self.world_size}\n"
            f"  • Log Level        : {self.log_level}\n"
            f"  • Logs Dir         : {self.logs_dir}\n\n"
            f"Model (Scale: {self.moe.scale.value}):\n"
            f"  • Layers           : {self.moe.n_blocks}\n"
            f"  • Hidden Dim       : {self.moe.hidden_dim}\n"
            f"  • Attention Heads  : {self.moe.num_heads}\n"
            f"  • Sequence Length  : {self.moe.seqlen}\n\n"
            f"MoE & Training:\n"
            f"  • Experts per GPU  : {self.moe.num_experts_per_gpu}\n"
            f"  • Global Batch     : {self.moe.batch_size}\n"
            f"  • Micro Batches    : {self.moe.micro_batches}\n"
            f"  • Top-K            : {self.moe.top_k}\n"
            f"  • Comm. Scaling    : {self.moe.comm_scaling_factor}x (Simulation)\n"
            f"{'='*50}"
        )


def get_moe_config(scale: MoEScale = MoEScale.TINY) -> MoESetup:
    """Factory MoE configurations"""

    if scale == MoEScale.TINY:
        return MoESetup()

    raise NotImplementedError


@functools.lru_cache
def get_cfg(world_size: int, scale: MoEScale) -> Config:
    cfg = Config(
        moe=get_moe_config(scale),
        world_size=world_size,
    )

    logger.remove()
    logger.add(
        sys.stderr,
        level=cfg.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",  # noqa
    )
    return cfg
