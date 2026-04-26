from contextlib import nullcontext

from cs336_systems.configs import ModelConfig, instantiate_model
from timeit import default_timer as timer
import numpy as np
import torch


def compute_mean_and_std_pass_times(
    model_config: ModelConfig,
    context_length: int,
    measure_also_backward: bool,
    synchronize: bool = True,
    warmup_steps: int = 5,
    measure_steps: int = 10,
    batch_size: int = 4,
    mixed_precision_dtype: torch.dtype = None,
) -> tuple[float, float]:
    """Compute the mean and std times for the forward pass.

    Compute the mean and std times for the forward pass for a model with the
    given config and context length.

    Args:
        model_config (ModelConfig): The configuration of the model to benchmark.
        context_length (int): The length of the context for the model.
        measure_also_backward (bool): Whether to also measure the backward pass.
        synchronize (bool, optional): Whether to synchronize CUDA operations.
          Defaults to True.
        warmup_steps (int, optional): The number of warmup steps. Defaults to 5.
        measure_steps (int, optional): The number of measurement steps. Defaults
          to 10.
        batch_size (int, optional): The batch size. Defaults to 4.
        mixed_precision_dtype (torch.dtype, optional): The data type to use for
          mixed precision (torch autocast). Defaults to None.

    Returns:
        tuple[float, float]: The mean and standard deviation of the forward pass
          times.
    """

    assert (
        torch.cuda.is_available()
    ), "CUDA must be available for running this Benchmark"
    device = torch.device("cuda")
    batch = torch.randint(
        0, model_config.vocab_size, (batch_size, context_length), device=device
    )
    model = instantiate_model(model_config, context_length, device)

    context = nullcontext()
    if mixed_precision_dtype:
        torch.autocast("cuda", dtype=mixed_precision_dtype)

    for _ in range(warmup_steps):
        with context:
            output = model(batch)
        if measure_also_backward:
            output.sum().backward()

    measures_s = []
    for _ in range(measure_steps):
        # Make sure all operations have finished before starting to measure.
        torch.cuda.synchronize()

        start = timer()
        with context:
            output = model(batch)
        if measure_also_backward:
            output.sum().backward()
        if synchronize:
            torch.cuda.synchronize()
        end = timer()

        measures_s.append((end - start))
    return np.mean(measures_s), np.std(measures_s)


def model_size_mb(model: torch.nn.Module) -> float:
    """Returns the total size of the model parameters in MB."""
    # Sum up the size of every parameter tensor
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    # Sum up the size of every buffer (like RoPE frequencies or masks)
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())

    total_size_mb = (param_size + buffer_size) / 1024**2
    return total_size_mb
