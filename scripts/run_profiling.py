"""
Profile LLM training step.

Example usages:
  uv run modal run scripts/run_profiling.py \
    --model-name="medium" \
    --warmup-steps=2 \
    --active-steps=3 \
    --run-on-modal

For all options, see help:
  uv run modal run scripts/run_profiling.py -h

When running on Modal, this with will save a trace in the Modal volume, which
you can then download with:
  uv run modal volume get cs336-systems-volume \
    <trace_path> \
    ~/Downloads/trace.pt.trace.json
"""

from cs336_systems.modal_setup import app, traces_volume, TRACE_DIR
from cs336_systems.profile import profile_training_step


@app.function(
    # gpu="T4",    # 16GB. OOO on large.
    # gpu="L4",    # 24GB. OOO on xl.
    gpu="A100",  # 80GB. No OOO, GPU memory peaked at ~38GB.
    volumes={TRACE_DIR: traces_volume},
)
def run_func(**kwargs):
    return profile_training_step(**kwargs)


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
    measure_optimizer: bool = True,
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
        "measure_optimizer": measure_optimizer,
    }
    if run_on_modal:
        run_func.remote(**kwargs)
    else:
        run_func.local(**kwargs)
