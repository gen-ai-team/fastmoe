import functools
import sys
from enum import Enum

from loguru import logger
from pydantic_settings import BaseSettings


class MoEScale(str, Enum):
    DEBUG = "debug"
    GIGACHAT_10B = "gigachat-10b"
    GIGACHAT_700B = "gigachat-700b"


class MoESetup(BaseSettings):
    scale: MoEScale = MoEScale.DEBUG

    # Model
    batch_size: int = 4
    seq_len: int = 4096
    hidden_dim: int = 4096
    n_heads: int = 32
    ff_dim: int = 4096
    n_layers: int = 2

    # MoE
    num_experts: int = 8
    top_k: int = 2

    warmup_steps: int = 10
    active_steps: int = 50

    @property
    def total_tokens(self) -> int:
        return self.batch_size * self.seq_len * self.top_k

    @property
    def activation_volume_mb(self) -> float:
        """Size of the tensor being recombined in MB (Float32)"""
        # B * T * TopK * Hidden * 4 bytes
        return (self.total_tokens * self.hidden_dim * 4) / (1024**2)


class Config(BaseSettings):
    moe: MoESetup = MoESetup()
    logs_dir: str = "logs"
    log_level: str = "INFO"

    class Config:
        extra = "ignore"


def get_config(scale: MoEScale = MoEScale.DEBUG) -> MoESetup:
    """Factory for GigaChat MoE configurations"""

    if scale == MoEScale.DEBUG:
        return MoESetup(
            scale=MoEScale.DEBUG,
            batch_size=2,
            seq_len=128,
            hidden_dim=512,
            num_experts=4,
            top_k=2,
        )

    elif scale == MoEScale.GIGACHAT_10B:
        return MoESetup(
            scale=MoEScale.GIGACHAT_10B,
            batch_size=32,  # High batch for compute intensity
            seq_len=4096,
            hidden_dim=8192,  # Wide layers to balance H100 NVLink speed
            num_experts=8,
            top_k=2,
        )

    elif scale == MoEScale.GIGACHAT_700B:
        return MoESetup(
            scale=MoEScale.GIGACHAT_700B,
            batch_size=2,  # Reduced due to memory pressure
            seq_len=4096,
            hidden_dim=16384,  # Massive width
            num_experts=256,  # Extreme fragmentation
            top_k=4,  # Higher Top-K common at this scale
        )

    return MoESetup()


@functools.lru_cache
def init_app() -> Config:
    cfg = Config()
    logger.remove()
    logger.add(
        sys.stderr,
        level=cfg.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",  # noqa
    )
    return cfg
