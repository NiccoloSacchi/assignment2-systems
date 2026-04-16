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
    --no-measure-also-backward

uv run modal run cs336_systems/benchmark_script_modal.py \
    --model-name="large" \
    --warmup-steps=5 \
    --synchronize \
    --measure-also-backward
"""

from modal_setup import app
from benchmark import (
  MODELS,
  instantiate_model,
  model_size_mb,
  compute_mean_and_std_pass_times,
)

@app.function(
    # gpu="T4",    # 16GB. OOO on large.
    gpu="L4",    # 24GB. OOO on xl.
    # gpu="A100",  # 80GB. No OOO, GPU memory peaked at ~38GB.
)
def run_benchmarking(model_name: str, warmup_steps: int, synchronize: bool, measure_also_backward: bool):
    context_length = 256
    model_config = MODELS[model_name]

    total_size_mb = model_size_mb(instantiate_model(model_config, context_length))
    print(f"--- {model_name} model ({total_size_mb:.2f} MB) ---")
    
    mean, std = compute_mean_and_std_pass_times(
      model_config = model_config,
      context_length = context_length,
      measure_also_backward = measure_also_backward,
      synchronize = synchronize,
      warmup_steps = warmup_steps,
    )
    print(f'{mean:.4f}s \u00b1 {std:.4f}s')


@app.local_entrypoint()
def main(
    model_name: str = "large",
    warmup_steps: int = 5,
    synchronize: bool = True,
    measure_also_backward: bool = True,
):
    """Modal takes care of parsing the below from CLI parameters.

    Args:
        model_name (str, optional): Model to be initialized to benchmarked.
          Valid values: "small", "medium", "large", "xl", "2.7B". Defaults to
          "large".
        warmup_steps (int, optional): The number of warmup steps. Defaults to 5.
        synchronize (bool, optional): Whether to run torch.cuda.synchronize()
          when timing. Useful to understand the impact of synchronizing.
          Defaults to True.
        measure_also_backward (bool, optional): Whether to time also the
          backward pass together with the forward pass. Defaults to True.
    """
    run_benchmarking.remote(model_name, warmup_steps, synchronize, measure_also_backward)
