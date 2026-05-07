"""
Compute residual size when using checkpointing

Example usages:
  uv run modal run scripts/checkpoint_optimal_transformer_block.py \
    --model-name="xl" \
    --num-checkpoints=48
"""

import torch
from cs336_systems.modal_setup import app
from cs336_systems.configs import MODELS, instantiate_model


@app.function(
    gpu="A100",  # 40GB.
)  # Any GPU is fine for faster scheduling.
def run(model_name: str, num_checkpoints: int):
    device = torch.device("cuda")
    batch_size = 4
    context_length = 2048
    model_config = MODELS[model_name]

    x = torch.randint(
        0,
        model_config.vocab_size,
        (batch_size, context_length),
        device=device,
    )

    model = instantiate_model(model_config, context_length, num_checkpoints, device)
    model = torch.compile(model, fullgraph=True)

    total_size_bytes = 0

    def pack_hook(t):
        if isinstance(t, torch.nn.Parameter):
            # Skip logging parameters to avoid double counting.
            return t
        nonlocal total_size_bytes
        total_size_bytes += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
        y = model(x)
        print(f"Total size of saved tensor {total_size_bytes / (1024**2):.2f} MiB")
    print(f"Peak memory: {torch.cuda.max_memory_allocated()/(1024**2):.2f} MiB")


@app.local_entrypoint()
def main(model_name: str, num_checkpoints: int):
    run.remote(model_name, num_checkpoints)
