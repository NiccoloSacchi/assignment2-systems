"""
Benchmark all-reduce in single-node multi-process setup.

Example usages: uv run modal run scripts/benchmark_ddp.py \
    --nprocs=2 \ --gpu \ --no-local \
    --grad-transfer-mode="overlap-with-backward"

You can then download files from the Modal volume with `modal volume get`,
examples:

uv run modal volume get cs336-systems-volume \
    ddp/rank=0/memory_profile.pickle \
    ~/Downloads/ddp/overlap-with-backward/rank=0/memory_profile.pickle 

uv run modal volume get cs336-systems-volume \
    ddp/overlap-with-backward/rank=0/modal_6.1778940193368957326.pt.trace.json \
    ~/Downloads/ddp/overlap-with-backward/rank=0/trace.json

uv run modal volume get cs336-systems-volume \
    ddp/naive/rank=0/modal_6.1778940109313437782.pt.trace.json \
    ~/Downloads/ddp/naive/rank=0/trace.json

You can then use https://docs.pytorch.org/memory_viz and
https://ui.perfetto.dev/ to visualize the traces.
"""

import modal
from copy import deepcopy
from cs336_systems import torch_dist
from cs336_systems.modal_setup import app
from cs336_systems.configs import MODELS, ModelConfig, instantiate_model
from cs336_systems.ddp import DDPIndividualParameters
from cs336_basics.optimizer import AdamW
from cs336_basics.loss import cross_entropy_loss
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from contextlib import nullcontext
from typing import Literal
from timeit import default_timer as timer

from cs336_systems.modal_setup import VOLUME_DIR, my_volume, app


def proc_train(
    rank: int,
    world_size: int,
    model_config: ModelConfig,
    context_length: int,
    total_batch_size: int,
    train_steps: int,
    warmup_steps: int,
    gpu: bool,
    grad_transfer_mode: Literal[
        "naive", "flat-grads", "overlap-with-backward"
    ] = "naive",
    profile: bool = False,
    profile_memory: bool = False,
):
    device, sync = torch_dist.setup(rank, world_size, gpu)

    # Seed to ensure that ranks are initialized with different initial models
    # and data.
    torch.manual_seed(rank)

    # Instantiate the model and the model parameters of rank 0 to all other
    # ranks.
    model = instantiate_model(model_config, context_length, device=device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    if grad_transfer_mode == "overlap-with-backward":
        model = DDPIndividualParameters(model)

    optimizer = AdamW(model.parameters())

    # UNCOMMENT IF YOU WANT TO CHECK THE TRAIN LOGIC WORKS THE SAME AS TRAINING
    # ON 1 PROCESS.
    # # Rank 0 creates the Single-Process Reference Model
    # ref_model = None
    # ref_optimizer = None
    # if rank == 0:
    #     ref_model = deepcopy(model)
    #     ref_optimizer = AdamW(ref_model.parameters())

    # Generate a batch of size train_steps * (total_batch_size // world_size).
    x = torch.randint(
        0,
        model_config.vocab_size,
        (train_steps, total_batch_size // world_size, context_length),
        device=device,
    )
    y = torch.randint(
        0,
        model_config.vocab_size,
        (train_steps, total_batch_size // world_size, context_length),
        device=device,
    )

    output_dir = VOLUME_DIR / "ddp" / f"{grad_transfer_mode}" / f"rank={rank}"
    output_dir.mkdir(parents=True, exist_ok=True)
    profiler_context = nullcontext()
    if profile:
        profiler_context = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=warmup_steps,
                active=train_steps - warmup_steps,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        )
    if profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    try:
        with profiler_context as prof:
            # Train.
            grad_transfer_time = 0
            start = timer()
            for i in range(train_steps):
                print(f"I am rank={rank}. Starting train step {i}...", flush=True)

                if i == warmup_steps:
                    dist.barrier()
                    grad_transfer_time = 0
                    start = timer()

                optimizer.zero_grad()
                with torch.profiler.record_function("forward"):
                    logits = model(x[i])
                    train_loss = cross_entropy_loss(logits, y[i])
                with torch.profiler.record_function("backward"):
                    train_loss.backward()

                # Average the gradients across all processes.
                if grad_transfer_mode == "overlap-with-backward":
                    model.finish_gradient_synchronization()
                else:
                    with torch.profiler.record_function("grad-all-reduce"):
                        grad_transfer_time_start = timer()
                        grads = [
                            param.grad.data
                            for param in model.parameters()
                            if param.grad is not None
                        ]
                        if grad_transfer_mode == "flat-grads":
                            flatten_grads = torch._utils._flatten_dense_tensors(grads)
                            dist.all_reduce(
                                flatten_grads, async_op=False, op=dist.ReduceOp.AVG
                            )
                            unflatten_avg_grads = torch._utils._unflatten_dense_tensors(
                                flatten_grads, grads
                            )
                            for j in range(len(grads)):
                                grads[j].copy_(unflatten_avg_grads[j])
                        else:
                            for grad in grads:
                                dist.all_reduce(
                                    grad, async_op=False, op=dist.ReduceOp.AVG
                                )
                        sync()
                    grad_transfer_time += timer() - grad_transfer_time_start

                # Gratients are up-to-date now we can optimize.
                optimizer.step()

                if isinstance(profiler_context, torch.profiler.profile):
                    prof.step()

                # UNCOMMENT IF YOU WANT TO CHECK THE TRAIN LOGIC WORKS THE SAME AS
                # TRAINING ON 1 PROCESS.
                # if rank != 0:
                #     # Other ranks send their local batch.
                #     dist.gather(x[i], gather_list=None, dst=0)
                #     dist.gather(y[i], gather_list=None, dst=0)
                # else:
                #     # Rank 0 collects batches from all other ranks.
                #     gathered_x = [torch.zeros_like(x[i]) for _ in range(world_size)]
                #     gathered_y = [torch.zeros_like(y[i]) for _ in range(world_size)]
                #     dist.gather(x[i], gather_list=gathered_x, dst=0)
                #     dist.gather(y[i], gather_list=gathered_y, dst=0)
                #     global_x = torch.cat(gathered_x, dim=0)
                #     global_y = torch.cat(gathered_y, dim=0)

                #     ref_optimizer.zero_grad()
                #     ref_logits = ref_model(global_x)
                #     ref_loss = cross_entropy_loss(ref_logits, global_y)
                #     ref_loss.backward()
                #     ref_optimizer.step()

                #     # Verify the result is the same.
                #     for p_dist, p_ref in zip(model.parameters(), ref_model.parameters()):
                #         assert torch.allclose(p_dist, p_ref, atol=1e-5), "Divergence detected!"

            tot_time = timer() - start
            if grad_transfer_mode == "overlap-with-backward":
                print(f"I am rank={rank}. Training lasted {tot_time:.2f}s.")
            else:
                print(
                    f"I am rank={rank}. I spent {grad_transfer_time / tot_time * 100:.2f}% ({grad_transfer_time:.2f}s / {tot_time:.2f}s) time on transferring gradients."
                )
    finally:
        if profile_memory:
            torch.cuda.memory._dump_snapshot(output_dir / "memory_profile.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)

        dist.destroy_process_group()

    if profile or profile_memory:
        print(f"I am rank={rank}. Traces saved in {output_dir}")


@app.cls(
    volumes={VOLUME_DIR: my_volume},
)
class Trainer:
    @modal.method()
    def train(self, nprocs, gpu, grad_transfer_mode, profile, profile_memory):
        total_batch_size = nprocs * 2
        context_length = 256
        model_config = MODELS["xl"]
        train_steps = 20
        warmup_steps = 5

        mp.spawn(
            fn=proc_train,
            args=(
                nprocs,
                model_config,
                context_length,
                total_batch_size,
                train_steps,
                warmup_steps,
                gpu,
                grad_transfer_mode,
                profile,
                profile_memory,
            ),
            nprocs=nprocs,
            join=True,
        )


@app.local_entrypoint()
def main(
    nprocs: int = 2,
    gpu: bool = False,
    local: bool = False,
    grad_transfer_mode: Literal[
        "naive", "flat-grads", "overlap-with-backward"
    ] = "naive",
    profile: bool = False,
    profile_memory: bool = False,
):
    """Simulate DDP training.

    Args:
        nprocs (int, optional): Number of processes to launch for distributed training. Defaults to 2.
        gpu (bool, optional): Whether to train on GPUs. Defaults to False.
        local (bool, optional): Whether to run locally or on Modal. Defaults to False.
        grad_transfer_mode (str, optional): How to transfer all gradients. Defaults to "naive".
    """
    assert nprocs <= 6, "Maximum 6 processes are supported."

    trainer = Trainer
    if gpu:
        trainer = trainer.with_options(gpu=f"A100-80GB:{nprocs}")

    instance = trainer()
    if local:
        instance.train.local(nprocs, gpu, grad_transfer_mode, profile, profile_memory)
    else:
        instance.train.remote(nprocs, gpu, grad_transfer_mode, profile, profile_memory)
