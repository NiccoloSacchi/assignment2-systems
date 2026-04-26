"""
Run benchmarking on LLM.

Example usages:
  uv run modal run scripts/benchmark.py \
    --model-names="small,medium,large,xl,2.7B"

For all options, see help:
  uv run modal run scripts/benchmark.py -h

You can then download files from the Modal volume with `modal volume get`,
example:
  uv run modal volume get cs336-systems-volume \
    xl/20260426_144909/memory_snapshot.pickle \
    ~/Downloads/memory_snapshot.pickle
"""

from contextlib import nullcontext
from datetime import datetime
import itertools

import torch
from cs336_systems.configs import MODELS
from cs336_systems.modal_setup import VOLUME_DIR, my_volume, app
from cs336_systems.benchmark import training_steps


@app.function(
    # gpu="T4",    # 16GB. OOO on large.
    # gpu="L4",  # 24GB. OOO on xl.
    gpu="A100",  # 40GB. OOO on 2.7B model with mixed precision.
    # gpu="A100-80GB",  # 80GB. No OOO, GPU memory peaked at ~38GB.
    volumes={VOLUME_DIR: my_volume},
)
def run_func(
    model_name: str,
    context_length: int,
    warmup_steps: int,
    synchronize: bool,
    do_backward: bool,
    do_optimize: bool,
    mixed_precision: bool,
    profile_memory: bool = False,
) -> tuple[float, float]:
    # print(
    #     f"Running benchmark for {model_name} model "
    #     f"mixed_precision={mixed_precision}."
    # )
    autocast_context = nullcontext()
    if mixed_precision:
        autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)

    memory_profile_path = None
    if profile_memory:
        memory_profile_path = (
            VOLUME_DIR
            / model_name
            / datetime.now().strftime("%Y%m%d_%H%M%S")
            / "memory_profile.pickle"
        )
        memory_profile_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Memory profile will be saved to {memory_profile_path}")
    mean, std = training_steps(
        model_config=MODELS[model_name],
        context_length=context_length,
        batch_size=4,
        do_backward=do_backward,
        do_optimize=do_optimize,
        synchronize=synchronize,
        warmup_steps=warmup_steps,
        measure_steps=3,
        autocast_context=autocast_context,
        profiler_context=nullcontext(),
        memory_profile_path=memory_profile_path,
    )
    return mean, std


@app.local_entrypoint()
def main(
    model_names: str = "large",
    profile_memory: bool = False,
):
    """Run a simple benchmark, timing forward and, optionally, backward pass.

    Args:
        model_name (str, optional): Model to be initialized to benchmarked.
          Valid values: "small", "medium", "large", "xl", "2.7B". Defaults to
          "large".
        profile_memory (bool, optional): Whether to also profile memory usage
          and save a snapshot. Defaults to False.
    """
    model_name_values = model_names.split(",")

    # Hardcode here the experiments to run (Modal doesn't support passing lists
    # or sets as arguments, so we can't use the CLI for this).
    context_length_values = [256]
    warmup_steps_values = [5]
    synchronize_values = [True]
    do_backward_values = [True]
    do_optimize_values = [False]
    mixed_precision_values = [
        None,
        # torch.bfloat16,
    ]

    experiments = list(
        itertools.product(
            context_length_values,
            warmup_steps_values,
            synchronize_values,
            do_backward_values,
            do_optimize_values,
            mixed_precision_values,
        )
    )

    # Store calls in a dictionary keyed by model name.
    model_calls = {}
    for model_name in model_name_values:
        model_calls[model_name] = []
        for exp in experiments:
            call = run_func.spawn(model_name, *exp, profile_memory=profile_memory)
            model_calls[model_name].append((exp, call))

    # Wait and Print.
    for model_name, calls_list in model_calls.items():
        print(f"\n--- {model_name} model ---")

        for exp, call in calls_list:
            context_length, warmup, sync, backward, do_optimize, mp = exp
            mean, std = call.get()

            print(
                f"context_length={context_length:<4} | "
                f"do_backward={str(backward):<5} | "
                f"do_optimize={str(do_optimize):<5} | "
                f"sync={str(sync):<5} | "
                f"warmup={warmup:<2} | "
                f"mixed_precision={str(mp):<20} : "
                f"{mean:>7.4f}s ± {std:.4f}s"
            )
