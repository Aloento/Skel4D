from logging import Logger
import os
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel

from ..config import cfg


def save_checkpoint(
        model: torch.nn.Module | DistributedDataParallel,
        checkpoint_dir: str,
        train_steps: int
):
    if cfg.main_process:
        if cfg.distributed:
            torch.save(model.module.state_dict(), f"{checkpoint_dir}/{train_steps:07d}.pt")
        else:
            torch.save(model.state_dict(), f"{checkpoint_dir}/{train_steps:07d}.pt")
    elif cfg.distributed:
        torch.distributed.barrier()
        

def resume_checkpoint(
        model: torch.nn.Module,
        logger: Logger,
        device: torch.device,
):
    train_steps = 0 if cfg.resume_checkpoint is None else int(os.path.basename(cfg.resume_checkpoint).split('-')[0])
    logger.info(f"Resuming training from step {train_steps}" if cfg.resume_checkpoint else "Starting training from scratch")

    if cfg.resume_checkpoint is not None:
        model.load_state_dict(torch.load(cfg.resume_checkpoint, map_location="cpu"), strict=False)
        logger.info(f"Checkpoint loaded from {cfg.resume_checkpoint}")

    model = model.to(device)
    if cfg.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            find_unused_parameters=False,
        )

    return train_steps, model
