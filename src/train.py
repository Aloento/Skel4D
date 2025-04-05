import os
from Pipeline.mk_res_dir import mk_res_dir
from Pipeline.prepare_model import prepare_model
from Pipeline.prepare_data import prepare_data
from Utils.logger import create_logger
from Utils.seed import set_seed
from config import cfg
import torch
from diffusers.utils.import_utils import is_xformers_available
import torch.distributed
import wandb


def train():
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert is_xformers_available(), "XFormers is required for training."

    if cfg.random_seed is not None:
        set_seed(cfg.random_seed)

    if cfg.distributed:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        torch.cuda.set_device(cfg.local_rank)

    if cfg.main_process:
        experiment_dir, checkpoint_dir, samples_dir = mk_res_dir()

        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        if cfg.use_wandb:
            wandb.init(
                project="Skel4D",
                name=os.path.basename(experiment_dir),
            )
    else:
        logger = create_logger(None)

    train_steps = 0 if cfg.resume_checkpoint is None else int(os.path.basename(cfg.resume_checkpoint).split('-')[0])
    logger.info(f"Resuming training from step {train_steps}" if cfg.resume_checkpoint else "Starting training from scratch")

    model = prepare_model()

    logger.info("Model loaded")
    logger.info(f"UNet Parameters: Trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,} / {sum(p.numel() for p in model.parameters()):,}")

    loder, val_loader = prepare_data()
    