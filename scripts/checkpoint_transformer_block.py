import torch
from torch.utils.checkpoint import checkpoint
from cs336_basics.layers import TransformerBlock
from cs336_systems.modal_setup import app


@app.function(gpu="any")  # Any GPU is fine for faster scheduling.
def run():
    total_size_bytes = 0

    def pack_hook(t):
        if isinstance(t, torch.nn.Parameter):
            # Skip logging parameters to avoid double counting.
            return t
        nonlocal total_size_bytes
        total_size_bytes += t.numel() * t.element_size()
        shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
        # print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
        return t

    device = torch.device("cuda")

    # num_layers for this model is 32
    d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_max_seq_len=context_length,
        rope_theta=10000,
        # rms_norm_eps= 1e-5,
        device=device,
        # dtype=None,
    )
    block = torch.compile(block, fullgraph=True)

    print("Without checkpoint:")
    total_size_bytes = 0
    x = torch.randn((4, context_length, d_model), requires_grad=True, device=device)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
        x = block(x)
        x = block(x)
        x = block(x)
        x = block(x)
        print(
            f"Total size of saved tensors in 4 TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB"
        )
    print(f"Peak memory: {torch.cuda.max_memory_allocated()/(1024**2):.2f} MiB")

    del x
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("With checkpoint:")
    total_size_bytes = 0
    x = torch.randn((4, context_length, d_model), requires_grad=True, device=device)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
        x = checkpoint(lambda x: block(block(x)), x, use_reentrant=False)
        x = checkpoint(lambda x: block(block(x)), x, use_reentrant=False)
        print(
            f"Total size of saved tensors in 4 TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB"
        )
    print(f"Peak memory: {torch.cuda.max_memory_allocated()/(1024**2):.2f} MiB")


@app.local_entrypoint()
def main():
    run.remote()
