from os import path
import pickle
from typing import Generator
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from diffusers.image_processor import VaeImageProcessor
import torch
import wandb

from ..Utils.decode import decode_latents_and_save
from ..Utils.encode import sample_from_noise
from ..config import cfg


def visualize_epoch(
        train_steps: int,
        model: torch.nn.Module,
        val_generator: Generator,
        scheduler: EulerDiscreteScheduler,
        device: torch.device,
        vae: AutoencoderKLTemporalDecoder,
        image_processor: VaeImageProcessor,
        samples_dir: str,
        log_dict: dict,
):
    if train_steps % cfg.visualize_every != 0 or not cfg.main_process:
        return
    
    model.eval()
    val_batch = next(val_generator)

    sample_latent = sample_from_noise(
        model, scheduler,
        cond_latent=val_batch["cond_latent"].to(device),
        cond_embedding=val_batch["embedding"].to(device),
        drags=val_batch["drags"].to(device),
        max_guidance=1,
    )

    vae = vae.to(device)

    if cfg.test_dir is not None:
        test_bid = (train_steps // cfg.visualize_every) % 100
        val_batch_fpath = path.join(cfg.test_dir, f"{test_bid:05d}.pkl")
        val_batch_val = pickle.load(open(val_batch_fpath, "rb"))
        
        sample_latent_val = sample_from_noise(
            model, scheduler,
            cond_latent=val_batch_val["cond_latent"].to(device),
            cond_embedding=val_batch_val["embedding"].to(device),
            drags=val_batch_val["drags"].to(device),
            max_guidance=1,
        )
        decode_latents_and_save(
            vae, image_processor, 
            sample_latent_val[0], f"{samples_dir}/sample_{train_steps:07d}_gen_val.gif", val_batch_val["drags"][0].to(device)
        )
        decode_latents_and_save(
            vae, image_processor, 
            val_batch_val["latents"][0].to(device), f"{samples_dir}/sample_{train_steps:07d}_gt_val.gif", val_batch_val["drags"][0].to(device)
        )
    
    decode_latents_and_save(
        vae, image_processor, 
        sample_latent[0], f"{samples_dir}/sample_{train_steps:07d}_gen.gif", val_batch["drags"][0].to(device)
    )

    decode_latents_and_save(
       vae, image_processor, 
       val_batch["latents"][0].to(device), f"{samples_dir}/sample_{train_steps:07d}_gt.gif", val_batch["drags"][0].to(device)
    )

    vae = vae.to("cpu")

    log_dict["gt_video"] = wandb.Video(f"{samples_dir}/sample_{train_steps:07d}_gt.gif")
    log_dict["gen_video"] = wandb.Video(f"{samples_dir}/sample_{train_steps:07d}_gen.gif")
     
    if cfg.test_dir is not None:
        log_dict["gt_val_video"] = wandb.Video(f"{samples_dir}/sample_{train_steps:07d}_gt_val.gif")
        log_dict["gen_val_video"] = wandb.Video(f"{samples_dir}/sample_{train_steps:07d}_gen_val.gif")
