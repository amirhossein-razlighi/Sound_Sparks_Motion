from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def init_distributed_and_device() -> tuple[int, int, torch.device]:
    use_dist = (
        dist.is_available()
        and "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )

    rank = 0
    world_size = 1
    local_rank = 0

    if use_dist:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible_gpu_count <= 0:
            raise RuntimeError("Distributed launch requested but no CUDA devices are visible.")
        if local_rank < 0 or local_rank >= visible_gpu_count:
            raise RuntimeError(
                "Invalid LOCAL_RANK to visible GPU mapping: "
                f"RANK={rank} WORLD_SIZE={world_size} LOCAL_RANK={local_rank} "
                f"visible_gpus={visible_gpu_count}."
            )

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=local_rank)

    if torch.cuda.is_available():
        if use_dist:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return rank, world_size, device


def shutdown_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()
