import os
import torch
import torch.distributed as dist
from typing import List, Any


def setup_distributed_process(rank: int, world_size: int, gpu: bool) -> torch.device:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    if gpu:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise ValueError("Unable to find CUDA devices.")
        local_rank = rank % device_count
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
        sync = torch.cuda.synchronize
    else:
        device = torch.device("cpu")
        backend = "gloo"
        sync = lambda: None

    dist.init_process_group(backend, rank=rank, world_size=world_size, device_id=device)
    return device, sync


class DDPIndividualParameters(torch.nn.Module):
    """Test with uv run pytest tests/test_ddp.py."""

    def __init__(self, module: torch.nn.Module):
        super().__init__()

        self.module = module
        self.handles: List[dist.Work] = []

        # Broadcast the initial weights from rank 0 to all other ranks.
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Register post-accumulation gradient hooks to trigger asynchronous
        # all-reduce.
        for param in self.module.parameters():
            if param.requires_grad:
                # The hook should have the following signature:
                #  hook(param: Tensor) -> None
                param.register_post_accumulate_grad_hook(self._make_hook(param=param))

    def _make_hook(self, param: torch.nn.Parameter):
        def hook(parameter: torch.nn.Parameter) -> None:
            if param.grad is not None:
                handle = dist.all_reduce(
                    param.grad.data, async_op=True, op=dist.ReduceOp.AVG
                )
                self.handles.append(handle)

        return hook

    def forward(self, *inputs, **kwargs) -> Any:
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        # Wait for every asynchronous communication call to completely
        # queue on the GPU.
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
