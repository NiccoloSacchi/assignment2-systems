"""
Benchmark all-reduce in single-node multi-process setup.

Example usages:
  uv run modal run scripts/distributed_communication_single_node.py --no-gpu
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from timeit import default_timer as timer
from cs336_systems.modal_setup import app
from datetime import timedelta


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(
        backend, rank=rank, world_size=world_size, timeout=timedelta(minutes=15)
    )


def proc_fn(rank, gpu: bool, world_size: int, data_size_bytes: int, return_dict: dict):
    backend = "gloo"
    device = torch.device("cpu")
    sync = lambda: None
    if gpu:
        backend = "nccl"
        device = f"cuda:{rank}"
        sync = torch.cuda.synchronize
        torch.cuda.set_device(rank)
    setup(rank, world_size, backend)

    bytes_per_element = 4  # for torch.float32
    num_elements = data_size_bytes // bytes_per_element

    # Warmup.
    data = torch.empty(num_elements, dtype=torch.float32, device=device)
    dist.all_reduce(data, async_op=False)
    sync()

    # Time the reduce operation.
    data = torch.empty(num_elements, dtype=torch.float32, device=device)
    start = timer()
    dist.all_reduce(data, async_op=False)
    sync()
    duration_ms = (timer() - start) * 1000

    durations_ms = [None] * world_size
    dist.all_gather_object(durations_ms, duration_ms)
    if rank == 0:
        return_dict["mean_ms"] = np.mean(durations_ms)
        return_dict["std_ms"] = np.std(durations_ms)

    dist.destroy_process_group()


def run(gpu: bool, world_size: int, data_size_grid: int) -> list[dict]:
    manager = mp.Manager()
    results = []
    for data_size in data_size_grid:
        duration_ms = manager.dict()
        mp.spawn(
            fn=proc_fn,
            args=(gpu, world_size, data_size, duration_ms),
            nprocs=world_size,
            join=True,
        )
        duration_ms = duration_ms.copy()
        duration_ms["world_size"] = world_size
        duration_ms["data_size_mb"] = data_size / 1000**2
        results.append(duration_ms)
        print(results[-1])
    return results


timeout = 300


@app.function(timeout=timeout)
def run_0gpu(world_size, data_size):
    return run(False, world_size, data_size)


@app.function(gpu="L4:2", timeout=timeout)
def run_2gpu(world_size, data_size):
    return run(True, world_size, data_size)


@app.function(gpu="L4:4", timeout=timeout)
def run_4gpu(world_size, data_size):
    return run(True, world_size, data_size)


@app.function(gpu="L4:6", timeout=timeout)
def run_6gpu(world_size, data_size):
    return run(True, world_size, data_size)


@app.local_entrypoint()
def main(gpu: bool = False):
    world_size_grid = [
        2,
        4,
        6,
    ]
    world_size_to_fn_gpu = {
        2: run_2gpu,
        4: run_4gpu,
        6: run_6gpu,
    }
    data_size_grid = [
        10**6,  # 1 MB
        10**7,  # 10 MB
        10**8,  # 100 MB
        10**9,  # 1 GB
    ]

    # I need to call different remote functions depending on the world size to
    # use exactly the amount of GPUs it is needed.
    results = []
    for world_size in world_size_grid:
        if gpu:
            results += world_size_to_fn_gpu[world_size].remote(
                world_size, data_size_grid
            )
        else:
            results += run_0gpu.remote(world_size, data_size_grid)
    pd.DataFrame(results).to_csv(
        "~/Downloads/distributed_communication_single_node.csv"
    )
