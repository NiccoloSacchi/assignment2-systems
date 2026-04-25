"""
Profile mixed precision example.

Example:
  uv run modal run scripts/profile_mixed_precision_example.py

For all options, see help:
  uv run modal run scripts/profile_mixed_precision_example.py -h
  
When running on Modal, this with will save a trace in the Modal volume, which
you can then download with:
  uv run modal volume get cs336-systems-volume \
    <trace_path> \
    ~/Downloads/trace.pt.trace.json
"""

from datetime import datetime
from pathlib import Path
import torch
from torch import nn
from cs336_systems.modal_setup import app, traces_volume, TRACE_DIR


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False, device=device)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(10, device=device)
        self.fc2 = nn.Linear(10, 10, bias=False, device=device)
        self.fc3 = nn.Linear(10, out_features, bias=False, device=device)

    def forward(self, x):
        with torch.profiler.record_function("relu(linear1)"):
            x = self.relu(self.fc1(x))
        with torch.profiler.record_function("layer_norm"):
            x = self.ln(x)
        with torch.profiler.record_function("linear2"):
            x = self.fc2(x)
        with torch.profiler.record_function("linear3"):
            x = self.fc3(x)
        return x


@app.function(
    # gpu="T4",    # 16GB. OOO on large.
    gpu="L4",  # 24GB. OOO on xl.
    # gpu="A100",  # 80GB. No OOO, GPU memory peaked at ~38GB.
    volumes={TRACE_DIR: traces_volume},
)
def run_func():
    output_dir = (
        TRACE_DIR
        / "mixed_precision_example_profiles"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ToyModel(20, 5, device="cuda")
    input = torch.randn(5, 20, device="cuda")

    # Warm up the GPU.
    _ = model(input)
    with torch.amp.autocast(device_type="cuda"):
        _ = model(input)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=0),
    ) as prof:

        # Warm up the profiler.
        _ = model(input)
        with torch.amp.autocast(device_type="cuda"):
            _ = model(input)
        prof.step()

        with torch.profiler.record_function("without autocast"):
            _ = model(input)
        prof.step()

        with torch.profiler.record_function("with autocast"):
            with torch.amp.autocast(device_type="cuda"):
                _ = model(input)
        prof.step()

    trace_path = sorted(
        output_dir.glob("**/*.pt.trace.json"),
        key=lambda pth: pth.stat().st_mtime,
        reverse=True,
    )[0]
    print(f"Trace saved to {trace_path.absolute()}")


@app.local_entrypoint()
def main(
    run_on_modal: bool = True,
):
    if run_on_modal:
        run_func.remote()
    else:
        run_func.local()
