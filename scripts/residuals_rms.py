import torch
from torch import nn
from cs336_systems.modal_setup import app


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x


def pack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


@app.function(gpu="any")  # Any GPU is fine for faster scheduling.
def run():
    device = torch.device("cuda")

    x = torch.randn((4, 512, 2560), requires_grad=True, device=device)

    print("Without compiling:")
    ln = RMSNorm(x.shape[-1], device=device)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = ln(x)
        y.sum().backward()

    print("Compiling:")
    ln = torch.compile(ln)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = ln(x)
        y.sum().backward()


@app.local_entrypoint()
def main():
    run.remote()
