"""
Model loading utilities for Stable Diffusion
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler

from .attention_control import AttentionStore, register_attention_control


def load_ldm(device, model_type="runwayml/stable-diffusion-v1-5", feature_upsample_res=256):
    """
    Load and configure Stable Diffusion model with attention control
    
    Args:
        device: Device to load model on
        model_type: HuggingFace model identifier
        feature_upsample_res: Resolution for feature upsampling
        
    Returns:
        tuple: (model, controllers, num_gpus)
    """
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    NUM_DDIM_STEPS = 50
    scheduler.set_timesteps(NUM_DDIM_STEPS)

    ldm = StableDiffusionPipeline.from_pretrained(
        model_type, scheduler=scheduler
    ).to(device)

    if device != "cpu":
        ldm.unet = nn.DataParallel(ldm.unet)
        ldm.vae = nn.DataParallel(ldm.vae)

        controllers = {}
        for device_id in ldm.unet.device_ids:
            device = torch.device("cuda", device_id)
            controller = AttentionStore()
            controllers[device] = controller
    else:
        controllers = {}
        _device = torch.device("cpu")
        controller = AttentionStore()
        controllers[_device] = controller

    def hook_fn(module, input):
        _device = input[0].device
        register_attention_control(module, controllers[_device], feature_upsample_res=feature_upsample_res)

    if device != "cpu":
        ldm.unet.module.register_forward_pre_hook(hook_fn)
    else:
        ldm.unet.register_forward_pre_hook(hook_fn)

    num_gpus = torch.cuda.device_count()

    # Freeze all parameters
    for param in ldm.vae.parameters():
        param.requires_grad = False
    for param in ldm.text_encoder.parameters():
        param.requires_grad = False
    for param in ldm.unet.parameters():
        param.requires_grad = False

    return ldm, controllers, num_gpus
