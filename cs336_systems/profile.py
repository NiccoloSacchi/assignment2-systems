from cs336_systems.configs import instantiate_model, MODELS
from cs336_basics.optimizer import AdamW
from cs336_basics.loss import cross_entropy_loss
from datetime import datetime
from pathlib import Path
import torch


def profile_training_step(
    path: Path,
    model_name: str,
    warmup_steps: int,
    active_steps: int,
):
    """Use Torch profile to profile a training step.
    
    Data and model are randomlyinitialized.

    Args:
        path (str): Path where to save the profile traces.
        model_name (str): Name of the model to profile.
        warmup_steps (int): Number of warmup steps.
        active_steps (int): Number of active steps to profile.

    Returns:
        : _description_
    """
    assert torch.cuda.is_available(), "CUDA must be available for profiling"
    
    # Reasonable defaults.
    context_length = 256
    batch_size = 4
    device = torch.device("cuda")
    
    # Create folder for the traces.
    output_dir = path / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Instantiate input, output, model, and optimizer.
    model_config = MODELS[model_name]
    x = torch.randint(0, model_config.vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, model_config.vocab_size, (batch_size, context_length), device=device)
    model = instantiate_model(model_config, context_length, device=device)
    optimizer = AdamW(model.parameters())

    # Why so many parameters for the schedule? Profiling is usually done during
    # training: you want to spread apart the profile steps with `wait` and warmup
    # again the profiler with `warmup` every time. Our case here is simpler, we
    # just want to profile a few steps in a row after warming up properly.
    # https://www.loom.com/i/8d9519391f994f74834a48535032b3f3.
    schedule = torch.profiler.schedule(wait=0, warmup=warmup_steps, active=active_steps)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        # Useful for seeing the size of the tesnsor operations, e.g. if tensors
        # are too small then there is not enough work for the GPU.
        record_shapes=True,
        # Useful for seeing which ops are allocating large temporary buffers.
        # Introduces overhead as every malloc/free gets intercepted.
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        for _ in range(warmup_steps + active_steps):
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            loss.backward()
            optimizer.step()
            prof.step()

    print(
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=5)
    )
    trace_path = sorted(
        output_dir.glob("**/*.pt.trace.json"),
        key=lambda pth: pth.stat().st_mtime,
        reverse=True,
    )[0]
    print(f"Trace saved to {trace_path}")

    return trace_path.absolute()
