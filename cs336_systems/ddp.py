import os
import torch
import torch.distributed as dist


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

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device, sync
