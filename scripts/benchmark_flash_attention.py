"""
Benchmark the flash attention implementation.

Example usages:
  uv run modal run scripts/benchmark_flash_attention.py
"""

from cs336_systems.modal_setup import app
from cs336_basics.layers import scaled_dot_product_attention
import torch
import pandas as pd
import traceback


def attention_pytorch(q, k, v, is_causal):
    mask = None
    if is_causal:
        seq_len = q.shape[0]
        mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device)
        )
    return scaled_dot_product_attention(q, k, v, mask)


@app.function(
    gpu="A100-40GB",
)
def run():
    # Triton imports only work when a CUDA GPU is available.
    import triton
    from cs336_systems.flash_attention_triton import TritonFlashAttention

    assert torch.cuda.is_available(), "CUDA device required for this benchmark."

    batch_size = 1
    casual_masking = True

    # Powers of 2.
    seq_len_grid = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_model_grid = [16, 32, 64, 128]
    dtype_grid = [torch.float32, torch.bfloat16]
    attention_impl_grid = {
        "attention_pytorch": attention_pytorch,
        "flash_attention_triton": TritonFlashAttention.apply,
    }
    device = torch.device("cuda")
    results = []
    for seq_len in seq_len_grid:
        for d_model in d_model_grid:
            for dtype in dtype_grid:
                for impl_name, attention_impl in attention_impl_grid.items():
                    q = torch.randn(
                        (batch_size, seq_len, d_model),
                        device=device,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    k = torch.randn(
                        (batch_size, seq_len, d_model),
                        device=device,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    v = torch.randn(
                        (batch_size, seq_len, d_model),
                        device=device,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    do = torch.randn(
                        (batch_size, seq_len, d_model), device=device, dtype=dtype
                    )

                    # We need to make sure the output from the previous
                    # iteration does not exist anymore otherwise it will occupy
                    # GPU memory together with its residuals.
                    try:
                        del o
                    except NameError:
                        pass
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                    try:
                        fwd_ms = triton.testing.do_bench(
                            lambda: attention_impl(q, k, v, casual_masking),
                            warmup=25,
                            rep=100,
                        )
                        # Backward requires a result to call .backward() on.
                        o = attention_impl(q, k, v, casual_masking)
                        bwd_ms = triton.testing.do_bench(
                            # retain_graph=True: tell pytorch to *not* free up
                            # residuals as we call backward multiple times.
                            lambda: o.backward(do, retain_graph=True),
                            warmup=25,
                            rep=100,
                        )
                        tot_ms = fwd_ms + bwd_ms
                    except torch.cuda.OutOfMemoryError:
                        fwd_ms = bwd_ms = tot_ms = "OOM"
                    except Exception:
                        print(
                            f"[ERROR] {impl_name} failed (seq_len={seq_len}, d_model={d_model}, dtype={dtype}):"
                        )
                        traceback.print_exc()
                        fwd_ms = bwd_ms = tot_ms = float("NaN")

                    results.append(
                        {
                            "seq_len": seq_len,
                            "d_model": d_model,
                            "dtype": dtype,
                            "implementation": impl_name,
                            "forward_ms": fwd_ms,
                            "backward_ms": bwd_ms,
                            "tot_ms": tot_ms,
                            "peak_memory_mb": torch.cuda.max_memory_allocated()
                            / (1024**2),
                        }
                    )
                    print("Benchmarked: ", results[-1])
    return pd.DataFrame(results)


@app.local_entrypoint()
def main():
    df = run.remote()
    df.to_csv("~/Downloads/benchmark_flash_attention.csv")
