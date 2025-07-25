from logging import Logger
from time import time
import torch
import torch.distributed
from ..config import cfg


def log_epoch(
        train_steps: int,
        start_time: float,
        running_loss: float,
        log_steps: int,
        device: torch.device,
        logger: Logger
):
    log_dict = {}

    if train_steps % cfg.log_every != 0:
        return log_dict

    if cfg.distributed:
        torch.cuda.synchronize()
    
    end_time = time()
    steps_per_sec = log_steps / (end_time - start_time)

    # Reduce loss history over all processes:
    avg_loss = torch.tensor(running_loss / log_steps, device=device)

    if cfg.distributed:
        torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)

    logger.info(
        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
    )

    log_dict["train_loss"] = avg_loss
    log_dict["train_steps_per_sec"] = steps_per_sec

    return log_dict
