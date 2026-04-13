from cs336_basics.model import TransformerLM
from timeit import default_timer as timer
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class ModelConfig:
  vocab_size: int
  d_model: int
  d_ff: int
  num_layers: int
  num_heads: int

# Standard model sizes to instantiate models to benchmark.
BATCH_SIZE = 4
MODELS = {
  "small":  ModelConfig(10000, 768, 3072, 12, 12),
  "medium": ModelConfig(10000, 1024, 4096, 24, 16),
  "large":  ModelConfig(10000, 1280, 5120, 36, 20),
  "xl":     ModelConfig(10000, 1600, 6400, 48, 25),
  "2.7B":   ModelConfig(10000, 2560, 10240, 32, 32),
}

def instantiate_model(
  model_config: ModelConfig,
  context_length: int,
  device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
  """Instantiate a model with the given config.
  """
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


def benchmarking_script(
  model_config: ModelConfig,
  context_length: int,
  measure_also_backward: bool,
  synchronize: bool = True,
  warmup_steps: int = 5,
  measure_steps: int = 10,
  batch_size: int = 4,
) -> tuple[int, int]:
  assert torch.cuda.is_available(), "CUDA must be available for running this Benchmark"
  device = torch.device("cuda")
  print("Creating dummy data and model...")
  input = torch.randint(0, model_config.vocab_size, (batch_size, context_length), device=device)
  model = instantiate_model(model_config, context_length, device)
  
  print("Warming up...")
  for _ in range(warmup_steps):
    output = model(input)
    if measure_also_backward:
      output.sum().backward()

  print("Measuring...")
  measures_s = []
  for _ in range(measure_steps):
    # Clear gradients manually to measure 'clean' write speed (avoid readiing
    # and summing gradients from the previous step).
    for p in model.parameters():
      p.grad = None
        
    # Make sure all operations have finished before starting to measure.
    torch.cuda.synchronize()

    start = timer()
    output = model(input)
    if measure_also_backward:
      output.sum().backward()
    if synchronize:
      torch.cuda.synchronize()
    end = timer()

    measures_s.append((end - start))
  return np.mean(measures_s), np.std(measures_s)


def model_size_mb(model: torch.nn.Module) -> float:
  """Returns the total size of the model parameters in MB.
  """
  # Sum up the size of every parameter tensor
  param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
  # Sum up the size of every buffer (like RoPE frequencies or masks)
  buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
  
  total_size_mb = (param_size + buffer_size) / 1024**2
  return total_size_mb
