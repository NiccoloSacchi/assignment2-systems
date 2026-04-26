"""
Profile LLM training step.

Example usages:
  uv run modal run scripts/profile_llm.py \
    --model-name="medium" \
    --warmup-steps=2 \
    --active-steps=3 \
    --run-on-modal

For all options, see help:
  uv run modal run scripts/profile_llm.py -h

When running on Modal, this with will save a trace in the Modal volume, which
you can then download with:
  uv run modal volume get cs336-systems-volume \
    medium/20260426_125745/modal_2.1777208272579621503.pt.trace.json \
    ~/Downloads/trace.pt.trace.json
"""

from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import torch
from cs336_systems.benchmark import training_steps
from cs336_systems.configs import MODELS
from cs336_systems.modal_setup import app, traces_volume, TRACE_DIR


@app.function(
    # gpu="T4",    # 16GB. OOO on large.
    # gpu="L4",    # 24GB. OOO on xl.
    gpu="A100",  # 40GB. No OOO, GPU memory peaked at ~38GB.
    volumes={TRACE_DIR: traces_volume},
)
def run_func(
    path: Path,
    model_name: str,
    warmup_steps: int,
    active_steps: int,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
    synchronize: bool = False,
    do_backward: bool = True,
    do_optimize: bool = True,
):
    # return profile_training_step(**kwargs)

    # Create folder for the traces.
    output_dir = path / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Why so many parameters for the schedule? Profiling is usually done during
    # training: you want to spread apart the profile steps with `wait` and warmup
    # again the profiler with `warmup` every time. Our case here is simpler, we
    # just want to profile a few steps in a row after warming up properly.
    # https://www.loom.com/i/8d9519391f994f74834a48535032b3f3.
    schedule = torch.profiler.schedule(wait=0, warmup=warmup_steps, active=active_steps)

    profiler_context = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        # Useful for seeing the size of the tensor operations, e.g. if tensors
        # are too small then there is not enough work for the GPU.
        record_shapes=record_shapes,
        # Useful for seeing which ops are allocating large temporary buffers.
        # Introduces overhead as every malloc/free gets intercepted.
        profile_memory=profile_memory,
        with_stack=with_stack,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    )

    _ = training_steps(
        model_config=MODELS[model_name],
        context_length=256,
        batch_size=4,
        do_backward=do_backward,
        do_optimize=do_optimize,
        synchronize=synchronize,
        warmup_steps=2,
        # +1 to close the active window and trigger on_trace_ready.
        measure_steps=warmup_steps + active_steps + 1,
        autocast_context=nullcontext(),
        profiler_context=profiler_context,
    )

    trace_path = sorted(
        output_dir.glob("**/*.pt.trace.json"),
        key=lambda pth: pth.stat().st_mtime,
        reverse=True,
    )[0]
    print(f"Trace saved to {trace_path.absolute()}")


@app.local_entrypoint()
def main(
    run_on_modal: bool = True,
    model_name: str = "large",
    warmup_steps: int = 2,
    active_steps: int = 3,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
    synchronize: bool = False,
    do_backward: bool = True,
    do_optimize: bool = True,
):
    """Run benchmarking.

    Args:
        run_on_modal (bool, optional): Whether to run on Modal instead of
          locally. Defaults to True.
        Others: See profile_training_step function.
    """
    kwargs = {
        "path": TRACE_DIR / model_name,
        "model_name": model_name,
        "warmup_steps": warmup_steps,
        "active_steps": active_steps,
        "record_shapes": record_shapes,
        "profile_memory": profile_memory,
        "with_stack": with_stack,
        "synchronize": synchronize,
        "do_backward": do_backward,
        "do_optimize": do_optimize,
    }
    if run_on_modal:
        _ = run_func.remote(**kwargs)
    else:
        _ = run_func.local(**kwargs)
