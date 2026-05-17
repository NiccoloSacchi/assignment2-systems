import torch
import torch.distributed as dist
from typing import List, Any


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
                with torch.profiler.record_function("grad-all-reduce"):
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
