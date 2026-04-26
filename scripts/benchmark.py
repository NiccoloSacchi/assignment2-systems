"""
Run benchmarking on LLM.

Example usages:
  uv run modal run scripts/benchmark.py \
    --model-names="small,medium,large,xl,2.7B"

For all options, see help:
  uv run modal run scripts/benchmark.py -h
"""

import itertools

import torch
from cs336_systems.configs import MODELS, instantiate_model, ModelConfig
from cs336_systems.modal_setup import app
from cs336_systems.benchmark import compute_mean_and_std_pass_times, model_size_mb


@app.function(
    # gpu="T4",    # 16GB. OOO on large.
    # gpu="L4",  # 24GB. OOO on xl.
    gpu="A100",  # 40GB. OOO on 2.7B model with mixed precision.
    # gpu="A100-80GB",  # 80GB. No OOO, GPU memory peaked at ~38GB.
)
def run_func(
    model_name: str,
    context_length: int,
    warmup_steps: int,
    synchronize: bool,
    measure_also_backward: bool,
    mixed_precision: bool,
):
    model_config = MODELS[model_name]
    # print(
    #     f"Running benchmark for {model_name} model "
    #     f"mixed_precision={mixed_precision}."
    # )
    return compute_mean_and_std_pass_times(
        model_config=model_config,
        context_length=context_length,
        measure_also_backward=measure_also_backward,
        synchronize=synchronize,
        warmup_steps=warmup_steps,
        mixed_precision_dtype=mixed_precision,
    )


@app.local_entrypoint()
def main(
    model_names: str = "large",
):
    """Run a simple benchmark, timing forward and, optionally, backward pass.

    Args:
        model_name (str, optional): Model to be initialized to benchmarked.
          Valid values: "small", "medium", "large", "xl", "2.7B". Defaults to
          "large".
    """
    model_name_values = model_names.split(",")

    # Hardcode here the experiments to run (Modal doesn't support passing lists
    # or sets as arguments, so we can't use the CLI for this).
    context_length_values = [256]
    warmup_steps_values = [5]
    synchronize_values = [True]
    measure_also_backward_values = [True]
    mixed_precision_values = [torch.bfloat16]

    experiments = list(
        itertools.product(
            context_length_values,
            warmup_steps_values,
            synchronize_values,
            measure_also_backward_values,
            mixed_precision_values,
        )
    )

    # Store calls in a dictionary keyed by model name.
    model_calls = {}
    for model_name in model_name_values:
        model_calls[model_name] = []

        for exp in experiments:
            call = run_func.spawn(model_name, *exp)
            model_calls[model_name].append((exp, call))

    # Wait and Print.
    for model_name, calls_list in model_calls.items():
        print(f"\n--- {model_name} model ---")

        for exp, call in calls_list:
            context_length, warmup, sync, backward, mp = exp
            mean, std = call.get()

            print(
                f"context_length={context_length:<4} | "
                f"backward={str(backward):<5} | "
                f"sync={str(sync):<5} | "
                f"warmup={warmup:<2} | "
                f"mixed_precision={str(mp):<20} : "
                f"{mean:>7.4f}s ± {std:.4f}s"
            )
