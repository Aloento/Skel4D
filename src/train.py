import os
from time import time

from .Pipeline.log_epoch import log_epoch
from .Pipeline.train_epoch import train_epoch
from .Pipeline.mk_res_dir import mk_res_dir
from .Pipeline.prepare_model import prepare_model
from .Pipeline.prepare_data import prepare_data
from .Pipeline.prepare_vae import prepare_vae
from .Pipeline.save_resume import resume_checkpoint, save_checkpoint
from .Utils.logger import create_logger
from .Utils.seed import set_seed
from .config import cfg

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
        device = torch.device(f"cuda:{cfg.local_rank}")
    else:
        device = torch.device("cuda")
        
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

    model, opt = prepare_model()
    train_steps, model = resume_checkpoint(model, logger, device)

    logger.info("Model loaded")
    logger.info(f"UNet Parameters: Trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,} / {sum(p.numel() for p in model.parameters()):,}")

    loder, val_loader, val_generator = prepare_data(logger)    
    image_processor, vae, scheduler = prepare_vae()

    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {cfg.num_steps} steps...")

    while True:
        for batch in loder:
            loss, global_nan = train_epoch(model, batch, device)

            if global_nan > 0:
                print(f"Skipping step {train_steps} due to NaN loss, loss: {loss.item()}")
                opt.zero_grad(set_to_none=True)

                del loss, global_nan
                torch.cuda.empty_cache()

                if cfg.distributed:
                    torch.distributed.barrier()
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            log_dict = log_epoch(
                train_steps,
                start_time,
                running_loss,
                log_steps,
                device,
                logger
            )

            running_loss = 0
            log_steps = 0
            start_time = time()

            del loss, global_nan

            if train_steps % cfg.ckpt_every == 0 and train_steps > 0:
                save_checkpoint(
                    model,
                    checkpoint_dir,
                    train_steps
                )
            