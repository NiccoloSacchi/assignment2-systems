"""
Triton example.

Example:
  uv run modal run scripts/triton_example.py
"""

import torch
from cs336_systems.modal_setup import app


@app.function(
    gpu="any",  # Any GPU is fine for faster scheduling.
)
def run():
    from cs336_systems.triton_example import WeightedSumFunc

    device = torch.device("cuda")
    f_weightedsum = WeightedSumFunc.apply
    batch_size = 4
    num_columns = 32
    dim = 128
    x = torch.rand((batch_size, num_columns, dim), requires_grad=True, device=device)
    w = torch.rand((dim), requires_grad=True, device=device)
    print(f"Input shapes: x.shape={x.shape}, w.shape={w.shape}")

    out = f_weightedsum(x, w)
    print("Output shape:", out.shape)

    out.mean().backward()
    print(f"Gradients shapes: x.grad.shape={x.grad.shape}, w.grad.shape={w.grad.shape}")


@app.local_entrypoint()
def main():
    run.remote()
