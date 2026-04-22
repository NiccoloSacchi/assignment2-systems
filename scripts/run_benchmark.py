"""
Run benchmarking on LLM.

Example usages:
  uv run modal run scripts/run_benchmark.py \
    --model-name="large" \
    --warmup-steps=5 \
    --no-synchronize \
    --no-measure-also-backward

  uv run modal run scripts/run_benchmark.py \
    --model-name="large" \
    --warmup-steps=5 \
    --synchronize \
    --measure-also-backward

  uv run modal run scripts/run_benchmark.py \
    --model-name="large" \
    --warmup-steps=5 \
    --synchronize \
    --measure-also-backward \
    --no-run-on-modal

For all options, see help:
  uv run modal run scripts/run_benchmark.py -h
"""

from cs336_systems.modal_setup import app
from cs336_systems.benchmark import run_benchmarking


@app.function(
    # gpu="T4",    # 16GB. OOO on large.
    gpu="L4",    # 24GB. OOO on xl.
    # gpu="A100",  # 80GB. No OOO, GPU memory peaked at ~38GB.
)
def run_func(**kwargs):
    return run_benchmarking(**kwargs)


@app.local_entrypoint()
def main(
    run_on_modal: bool = True,
    model_name: str = "large",
    warmup_steps: int = 5,
    synchronize: bool = True,
    measure_also_backward: bool = True,
):
    """Run benchmarking.

    Args:
        run_on_modal (bool, optional): Whether to run on Modal instead of locally.
          Defaults to True.
        Others: See run_benchmarking function.
    """
    kwargs = {
        "model_name": model_name,
        "warmup_steps": warmup_steps,
        "synchronize": synchronize,
        "measure_also_backward": measure_also_backward,
    }
    if run_on_modal:
        run_func.remote(**kwargs)
    else:
        run_func.local(**kwargs)
