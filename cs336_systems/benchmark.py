from contextlib import nullcontext

from cs336_basics.optimizer import AdamW
from matplotlib.style import context
from cs336_systems.configs import ModelConfig, instantiate_model
from cs336_basics.loss import cross_entropy_loss
from timeit import default_timer as timer
import numpy as np
import torch


def model_size_mb(model: torch.nn.Module) -> float:
    """Returns the total size of the model parameters in MB."""
    # Sum up the size of every parameter tensor
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    # Sum up the size of every buffer (like RoPE frequencies or masks)
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())

    total_size_mb = (param_size + buffer_size) / 1024**2
    return total_size_mb


def training_steps(
    model_config: ModelConfig,
    context_length: int = 256,
    batch_size: int = 4,
    do_backward: bool = True,
    do_optimize: bool = True,
    synchronize: bool = True,
    warmup_steps: int = 5,
    measure_steps: int = 10,
    autocast_context: torch.autocast | nullcontext = nullcontext(),
    profiler_context: torch.profiler.profile | nullcontext = nullcontext(),
    memory_profile_path: str | None = None,
) -> tuple[float, float]:
    """Compute the mean and std times for the forward pass.

    Compute the mean and std times for the forward pass for a model with the
    given config and context length.

    Args:
        model_config (ModelConfig): The configuration of the model to benchmark.
        context_length (int): The length of the context for the model.
        batch_size (int, optional): The batch size. Defaults to 4.
        do_backward (bool, optional): Whether to also measure the backward pass.
            Defaults to True.
        do_optimize (bool, optional): Whether to also perform an optimizer step.
            Defaults to True.
        synchronize (bool, optional): Whether to synchronize CUDA operations at
          the end of each measure step. Defaults to True.
        warmup_steps (int, optional): The number of warmup steps. Defaults to 5.
        measure_steps (int, optional): The number of measurement steps. Defaults
          to 10.
        autocast_context (torch.autocast | nullcontext, optional): The autocast
          context to use. Defaults to nullcontext().
        profiler_context (torch.profiler.profile | nullcontext, optional): The
          profiler context to use. Defaults to nullcontext().
        memory_profile_path (str | None, optional): If not None, enable memory
          profiling and save the memory snapshot to the specified path. Defaults
          to None.

    Returns:
        tuple[float, float]: The mean and standard deviation of the forward pass
          times.
    """
    assert (
        torch.cuda.is_available()
    ), "CUDA must be available for running this Benchmark"
    assert do_backward or not do_optimize, "Can't optimize if not doing backward"

    # Reasonable defaults.
    batch_size = 4
    device = torch.device("cuda")

    # Instantiate input, output, model, and optimizer.
    x = torch.randint(
        0, model_config.vocab_size, (batch_size, context_length), device=device
    )
    y = torch.randint(
        0, model_config.vocab_size, (batch_size, context_length), device=device
    )
    model = instantiate_model(model_config, context_length, device=device)
    optimizer = AdamW(model.parameters())

    for _ in range(warmup_steps):
        with autocast_context:
            logits = model(x)
        if do_backward:
            loss = cross_entropy_loss(logits, y)
            loss.backward()
            if do_optimize:
                optimizer.step()

    # Make sure all warmup operations completed.
    torch.cuda.synchronize()

    if memory_profile_path is not None:
        # Enable memory profiling.
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    with profiler_context as prof:
        measures_s = []
        for _ in range(measure_steps):
            start = timer()
            with autocast_context:
                with torch.profiler.record_function("forward"):
                    logits = model(x)
            if do_backward:
                with torch.profiler.record_function("backward"):
                    loss = cross_entropy_loss(logits, y)
                    loss.backward()
                if do_optimize:
                    with torch.profiler.record_function("optimizer"):
                        optimizer.step()
            if synchronize:
                torch.cuda.synchronize()
            end = timer()
            measures_s.append((end - start))

            if isinstance(profiler_context, torch.profiler.profile):
                prof.step()

    # Save a pickle file to be loaded by PyTorch's online tool.
    if memory_profile_path is not None:
        torch.cuda.memory._dump_snapshot(memory_profile_path)
        # Stop recording history.
        torch.cuda.memory._record_memory_history(enabled=None)

    return np.mean(measures_s), np.std(measures_s)
