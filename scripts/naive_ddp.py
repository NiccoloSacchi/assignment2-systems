"""
Benchmark all-reduce in single-node multi-process setup.

Example usages:
uv run modal run scripts/naive_ddp.py \
    --nprocs=2 \
    --gpu \
    --no-local \
    --no-flat-grads

You can then download files from the Modal volume with `modal volume get`,
example:
  uv run modal volume get cs336-systems-volume \
    ddp/rank=0/memory_profile.pickle \
    ~/Downloads/memory_profile.pickle 
"""

import modal
from copy import deepcopy
from cs336_systems.modal_setup import app
from cs336_systems.configs import MODELS, ModelConfig, instantiate_model
from cs336_systems.ddp import setup_distributed_process
from cs336_basics.optimizer import AdamW
from cs336_basics.loss import cross_entropy_loss
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
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
    flat_grads: bool,
):
    profile_memory = True
    device, sync = setup_distributed_process(rank, world_size, gpu)

    # Seed to ensure that ranks are initialized with different initial models
    # and data.
    torch.manual_seed(rank)

    # Instantiate the model and the model parameters of rank 0 to all other
    # ranks.
    model = instantiate_model(model_config, context_length, device=device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

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

    if profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    try:
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
            logits = model(x[i])
            train_loss = cross_entropy_loss(logits, y[i])
            train_loss.backward()

            # Average the gradients across all processes.
            grad_transfer_time_start = timer()
            grads = [
                param.grad.data
                for param in model.parameters()
                if param.grad is not None
            ]
            if flat_grads:
                flatten_grads = torch._utils._flatten_dense_tensors(grads)
                dist.all_reduce(flatten_grads, async_op=False, op=dist.ReduceOp.AVG)
                unflatten_avg_grads = torch._utils._unflatten_dense_tensors(
                    flatten_grads, grads
                )
                for j in range(len(grads)):
                    grads[j].copy_(unflatten_avg_grads[j])
            else:
                for grad in grads:
                    dist.all_reduce(grad, async_op=False, op=dist.ReduceOp.AVG)
            sync()
            grad_transfer_time += timer() - grad_transfer_time_start

            optimizer.step()

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
        print(
            f"I am rank={rank}. I spent {grad_transfer_time / tot_time * 100:.2f}% ({grad_transfer_time:.2f}s / {tot_time:.2f}s) time on transferring gradients."
        )
    finally:
        if profile_memory:
            memory_profile_path = (
                VOLUME_DIR / "ddp" / f"rank={rank}" / "memory_profile.pickle"
            )
            memory_profile_path.parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(memory_profile_path)
            torch.cuda.memory._record_memory_history(enabled=None)

        dist.destroy_process_group()


@app.cls(
    volumes={VOLUME_DIR: my_volume},
)
class Trainer:
    @modal.method()
    def train(self, nprocs, gpu, flat_grads):
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
                flat_grads,
            ),
            nprocs=nprocs,
            join=True,
        )


@app.local_entrypoint()
def main(
    nprocs: int = 2, gpu: bool = False, local: bool = False, flat_grads: bool = False
):
    """Simulate DDP training.

    Args:
        nprocs (int, optional): Number of processes to launch for distributed training. Defaults to 2.
        gpu (bool, optional): Whether to train on GPUs. Defaults to False.
        local (bool, optional): Whether to run locally or on Modal. Defaults to False.
        flat_grads (bool, optional): Whether to flatten all gradients in 1 big tensor before transfer. Defaults to False.
    """
    assert nprocs <= 6, "Maximum 6 processes are supported."

    trainer = Trainer
    if gpu:
        trainer = trainer.with_options(gpu=f"A100-80GB:{nprocs}")

    instance = trainer()
    if local:
        instance.train.local(nprocs, gpu, flat_grads)
    else:
        instance.train.remote(nprocs, gpu, flat_grads)
