"""
Benchmark the flash attention implementation.

Example usages:
  uv run modal run scripts/benchmark_flash_attention.py
"""

from cs336_systems.modal_setup import app
import torch
import pandas as pd
import traceback


@app.function(
    gpu="A100-40GB",
)
def run():
    import triton
    from cs336_systems.flash_attention_pytorch import PyTorchFlashAttention
    from cs336_systems.flash_attention_triton import TritonFlashAttention

    assert torch.cuda.is_available(), "CUDA device required for this benchmark."

    batch_size = 1
    casual_masking = True

    # Powers of 2.
    seq_len_grid = [128]  # , 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_model_grid = [16]  # , 32, 64, 128]
    dtype_grid = [torch.float32]  # , torch.bfloat16
    attention_class_grid = {
        "pytorch": PyTorchFlashAttention,
        "triton": TritonFlashAttention,
    }
    results = []
    for seq_len in seq_len_grid:
        for d_model in d_model_grid:
            for dtype in dtype_grid:
                for impl_name, attention_class in attention_class_grid.items():
                    q = torch.randn(
                        (batch_size, seq_len, d_model),
                        device="cuda",
                        dtype=dtype,
                        requires_grad=True,
                    )
                    k = torch.randn(
                        (batch_size, seq_len, d_model),
                        device="cuda",
                        dtype=dtype,
                        requires_grad=True,
                    )
                    v = torch.randn(
                        (batch_size, seq_len, d_model),
                        device="cuda",
                        dtype=dtype,
                        requires_grad=True,
                    )
                    do = torch.randn(
                        (batch_size, seq_len, d_model), device="cuda", dtype=dtype
                    )
                    try:
                        t_fwd = triton.testing.do_bench(
                            lambda: attention_class.apply(q, k, v, casual_masking),
                            warmup=25,
                            rep=100,
                        )
                        # Backward requires a result to call .backward() on.
                        out_tri = attention_class.apply(q, k, v, casual_masking)
                        t_bwd = triton.testing.do_bench(
                            # retain_graph=True: tell pytorch to *not* free up
                            # residuals as we call backward multiple times.
                            lambda: out_tri.backward(do, retain_graph=True),
                            warmup=25,
                            rep=100,
                        )
                        t_tot = t_fwd + t_bwd
                    except Exception as e:
                        print(
                            f"[ERROR] {impl_name} failed (seq_len={seq_len}, d_model={d_model}, dtype={dtype}): {e}"
                        )
                        traceback.print_exc()  # This prints the full stack trace
                        t_fwd = t_bwd = t_tot = float("NaN")

                    results.append(
                        {
                            "seq_len": seq_len,
                            "d_model": d_model,
                            "dtype": dtype,
                            "implementation": impl_name,
                            "forward_time": t_fwd,
                            "backward_time": t_bwd,
                            "tot_time": t_tot,
                        }
                    )

    return pd.DataFrame(results)


@app.local_entrypoint()
def main():
    df = run.remote()
    print(df.to_string(index=False))
    df.to_csv("~/Downloads/benchmark_flash_attention.csv")
