from cs336_basics.model import TransformerLM
from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


# Standard model sizes to instantiate models to benchmark.
MODELS = {
    "small": ModelConfig(10000, 768, 3072, 12, 12),
    "medium": ModelConfig(10000, 1024, 4096, 24, 16),
    "large": ModelConfig(10000, 1280, 5120, 36, 20),
    "xl": ModelConfig(10000, 1600, 6400, 48, 25),
    "2.7B": ModelConfig(10000, 2560, 10240, 32, 32),
}


def instantiate_model(
    model_config: ModelConfig,
    context_length: int,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """Instantiate a model with the given config."""
    return TransformerLM(
        vocab_size=model_config.vocab_size,
        context_length=context_length,
        num_layers=model_config.num_layers,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        rope_theta=10000.0,
        device=device,
        dtype=torch.float32,
    )
