import torch
import torch.distributed as dist
from torch.optim.optimizer import ParamsT
from typing import Type, Any, Dict


class OptimizerStateSharding(torch.optim.Optimizer):
    """Test with uv run pytest tests/test_sharded_optimizer.py"""

    def __init__(
        self,
        params: ParamsT,
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs: Any,
    ):
        # sharded_optimizer is initialized lazily in add_param_group.
        self.sharded_optimizer: torch.optim.Optimizer | None = None
        self.optimizer_cls = optimizer_cls
        self.optimizer_cls_kwargs = kwargs

        # Determine distributed world state.
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Track global parameters to their assigned rank.
        self.param_to_rank: Dict[torch.nn.Parameter, int] = {}

        # Instantiate the optimizer. It will call add_param_group on each
        # parameter/group.
        super().__init__(params, {})

    def step(self, closure=None, **kwargs) -> None:
        """
        Performs a local optimization step on sharded parameters, then
        broadcasts the updated parameters to all other ranks.
        """
        if self.sharded_optimizer:
            # Execute only if parameters were assigned to this optimizer.
            self.sharded_optimizer.step(closure, **kwargs)

        if self.world_size == 1:
            return

        # Synchronize the updated parameters across all ranks. We must iterate
        # over the global set of parameters in a deterministic order.
        for param, src_rank in self.param_to_rank.items():
            dist.broadcast(param.data, src=src_rank)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """
        Handles dynamically adding parameter groups post-initialization (e.g.,
        layer unfreezing).
        """
        # Clone hyperparameters.
        param_group_shard = {k: v for k, v in param_group.items() if k != "params"}
        param_group_shard["params"] = []
        for param in param_group["params"]:
            assigned_rank = len(self.param_to_rank) % self.world_size
            self.param_to_rank[param] = assigned_rank
            if assigned_rank == self.rank:
                param_group_shard["params"].append(param)

        super().add_param_group(param_group)

        if len(param_group_shard["params"]) == 0:
            # No parameters assigned to this rank.
            return

        if self.sharded_optimizer == None:
            self.sharded_optimizer = self.optimizer_cls(
                [param_group_shard], **self.optimizer_cls_kwargs
            )
        else:
            self.sharded_optimizer.add_param_group(param_group_shard)
