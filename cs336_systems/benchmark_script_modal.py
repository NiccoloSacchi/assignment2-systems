"""
Run benchmarking on LLM.

Example usages:
uv run modal run cs336_systems/benchmark_script_modal.py \
    --model-name="large" \
    --warmup-steps=5 \
    --no-synchronize \
    --no-measure-also-backward

uv run modal run cs336_systems/benchmark_script_modal.py \
    --model-name="large" \
    --warmup-steps=5 \
    --synchronize \
    --measure-also-backward

uv run modal run cs336_systems/benchmark_script_modal.py \
    --model-name="large" \
    --warmup-steps=5 \
    --synchronize \
    --measure-also-backward \
    --local
"""

from modal_setup import app
from benchmark import run_benchmarking


@app.function(
    # gpu="T4",    # 16GB. OOO on large.
    gpu="L4",    # 24GB. OOO on xl.
    # gpu="A100",  # 80GB. No OOO, GPU memory peaked at ~38GB.
)
def run_func(**kwargs):
    run_benchmarking(**kwargs)


@app.local_entrypoint()
def run(
    local: bool = False,
    model_name: str = "large",
    warmup_steps: int = 5,
    synchronize: bool = True,
    measure_also_backward: bool = True,
):
    """Run benchmarking.

    Args:
        local (bool, optional): Whether to run locally instead of on Modal.
          Defaults to False.
        Others: See run_benchmarking function.
    """
    kwargs = {
        "model_name": model_name,
        "warmup_steps": warmup_steps,
        "synchronize": synchronize,
        "measure_also_backward": measure_also_backward,
    }
    if local:
        run_func.local(**kwargs)
        return
    run_func.remote(**kwargs)



