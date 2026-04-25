from cs336_systems.configs import MODELS, ModelConfig, instantiate_model
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
) -> tuple[int, int]:
    """Compute the mean and std times for the forward pass.

    Compute the mean and std times for the forward pass for a model with the given
    config and context length.

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

    Returns:
        tuple[int, int]: The mean and standard deviation of the forward pass
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

    for _ in range(warmup_steps):
        output = model(batch)
        if measure_also_backward:
            output.sum().backward()

    measures_s = []
    for _ in range(measure_steps):
        # Clear gradients manually to measure 'clean' write speed (avoid readiing
        # and summing gradients from the previous step).
        for p in model.parameters():
            p.grad = None

        # Make sure all operations have finished before starting to measure.
        torch.cuda.synchronize()

        start = timer()
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


def run_benchmarking(
    model_name: str, warmup_steps: int, synchronize: bool, measure_also_backward: bool
):
    """Run a simple benchmark, timing forward and, optionally, backward pass.

    Args:
        model_name (str, optional): Model to be initialized to benchmarked.
          Valid values: "small", "medium", "large", "xl", "2.7B". Defaults to
          "large".
        warmup_steps, synchronize, measure_also_backward: See
          compute_mean_and_std_pass_times.
    """
    context_length = 256
    model_config = MODELS[model_name]

    total_size_mb = model_size_mb(instantiate_model(model_config, context_length))
    print(f"--- {model_name} model ({total_size_mb:.2f} MB) ---")

    mean, std = compute_mean_and_std_pass_times(
        model_config=model_config,
        context_length=context_length,
        measure_also_backward=measure_also_backward,
        synchronize=synchronize,
        warmup_steps=warmup_steps,
    )
    print(f"{mean:.4f}s \u00b1 {std:.4f}s")
