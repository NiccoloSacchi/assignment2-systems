"""
Compute residual size when using checkpointing

Example usages:
  uv run modal run scripts/benchmark_attention.py
"""

import torch
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from cs336_systems.modal_setup import app
from cs336_basics.layers import scaled_dot_product_attention


def mean_std(x):
    return np.mean(x), np.std(x)


@app.function(
    gpu="any",  # Any GPU is fine for faster scheduling.
    # gpu="A100",  # 40GB.
    timeout=1200,  # Timeout in seconds (1200s = 20 mins)
)
def run(compile: bool):
    device = torch.device("cuda")

    d_model_grid = [16, 32, 64, 128]
    context_length_grid = [256, 1024, 4096, 8192, 16384]
    compiled_grid = [False]
    batch_size = 8

    scaled_dot_product_attention_compiled = None
    if compile:
        # print("Compiling...")
        compiled_grid.append(True)
        scaled_dot_product_attention_compiled = torch.compile(
            scaled_dot_product_attention, fullgraph=True
        )
        # print("Compiled...")

    cols = [
        "d_model",
        "context_length",
        "compiled",
        "forward_time",
        "backward_time",
        "peak_memory_start_mib",
        "peak_memory_forward_mib",
        "peak_memory_backward_mib",
    ]
    df = pd.DataFrame(columns=cols)
    warmup_steps = 3

    for d_model in d_model_grid:
        for context_length in context_length_grid:
            for compiled in compiled_grid:
                q = torch.rand(
                    (batch_size, context_length, d_model),
                    device=device,
                    requires_grad=True,
                )
                k = torch.rand(
                    (batch_size, context_length, d_model),
                    device=device,
                    requires_grad=True,
                )
                v = torch.rand(
                    (batch_size, context_length, d_model),
                    device=device,
                    requires_grad=True,
                )
                forward_milliseconds = []
                backward_milliseconds = []
                start_peak_mem = []
                forward_peak_mem = []
                backward_peak_mem = []
                try:
                    for _ in range(100):
                        # Clear the previous loop (which might have reached OOM).
                        if "out" in locals():
                            del out
                        if "loss" in locals():
                            del loss
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

                        start_peak_mem.append(
                            torch.cuda.max_memory_allocated() / (1024**2)
                        )
                        torch.cuda.reset_peak_memory_stats()

                        # Forward pass.
                        start = timer()
                        if compiled:
                            out = scaled_dot_product_attention_compiled(q, k, v)
                        else:
                            out = scaled_dot_product_attention(q, k, v)
                        torch.cuda.synchronize()
                        forward_milliseconds.append(timer() - start)
                        forward_peak_mem.append(
                            torch.cuda.max_memory_allocated() / (1024**2)
                        )
                        torch.cuda.reset_peak_memory_stats()

                        # Backward pass.
                        # Just compute a "loss" that we can backpropagate.
                        loss = out.mean()
                        start = timer()
                        loss.backward()
                        torch.cuda.synchronize()
                        backward_milliseconds.append(timer() - start)
                        backward_peak_mem.append(
                            torch.cuda.max_memory_allocated() / (1024**2)
                        )
                except torch.cuda.OutOfMemoryError:
                    df.loc[len(df)] = [
                        d_model,
                        context_length,
                        compiled,
                        "OOM",
                        "OOM",
                        "OOM",
                        "OOM",
                        "OOM",
                    ]
                    continue
                # # Clear GPU memory.
                # del q, k, v
                # torch.cuda.empty_cache()
                # torch.cuda.reset_peak_memory_stats()

                # Discard the first warmup iterations.
                forward_milliseconds_mean, forward_milliseconds_std = mean_std(
                    forward_milliseconds[warmup_steps:]
                )
                backward_milliseconds_mean, backward_milliseconds_std = mean_std(
                    backward_milliseconds[warmup_steps:]
                )
                start_peak_mem_mean, start_peak_mem_std = mean_std(
                    start_peak_mem[warmup_steps:]
                )
                forward_peak_mem_mean, forward_peak_mem_std = mean_std(
                    forward_peak_mem[warmup_steps:]
                )
                backward_peak_mem_mean, backward_peak_mem_std = mean_std(
                    backward_peak_mem[warmup_steps:]
                )
                df.loc[len(df)] = [
                    d_model,
                    context_length,
                    compiled,
                    f"{forward_milliseconds_mean:>7.4f}ms ± {forward_milliseconds_std:.4f}ms",
                    f"{backward_milliseconds_mean:>7.4f}ms ± {backward_milliseconds_std:.4f}ms",
                    f"{start_peak_mem_mean:.2f}MiB ± {start_peak_mem_std:.2f}MiB",
                    f"{forward_peak_mem_mean:.2f}MiB ± {forward_peak_mem_std:.2f}MiB",
                    f"{backward_peak_mem_mean:.2f}MiB ± {backward_peak_mem_std:.2f}MiB",
                ]
                # print(df.iloc[-1])
    return df


@app.local_entrypoint()
def main(compile: bool = False):
    df = run.remote(compile)
    print(df.to_string(index=False))
    df.to_csv("~/Downloads/benchmark_attention.csv")
