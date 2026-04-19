"""
Profile LLM training step.

Example usages:
uv run modal run scripts/run_profiling.py \
    --model-name="medium" \
    --warmup-steps=5 \
    --active-steps=10 \
    --no-local
"""

from cs336_systems.modal_setup import app, traces_volume, TRACE_DIR
from cs336_systems.profile import profile_training_step

@app.function(
  # gpu="T4",    # 16GB. OOO on large.
  gpu="L4",    # 24GB. OOO on xl.
  # gpu="A100",  # 80GB. No OOO, GPU memory peaked at ~38GB.
  volumes={TRACE_DIR: traces_volume},
)
def run_func(**kwargs):
  return profile_training_step(**kwargs)

@app.local_entrypoint()
def main(
    local: bool = False,
    model_name: str = "large",
    warmup_steps: int = 5,
    active_steps: int = 10,
):
    """Run benchmarking.

    Args:
        local (bool, optional): Whether to run locally instead of on Modal.
          Defaults to False.
        Others: See profile_training_step function.
    """
    kwargs = {
      "path": TRACE_DIR / model_name,
      "model_name": model_name,
      "warmup_steps": warmup_steps,
      "active_steps": active_steps,
    }
    if local:
        out = run_func.local(**kwargs)
    else:
        out = run_func.remote(**kwargs)
    print(out)



