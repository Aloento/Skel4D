"""
Utility functions for image processing and attention map handling
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image as PILImage


def init_random_noise(device, num_words=77):
    """Initialize random noise for context embedding"""
    return torch.randn(1, num_words, 1024).to(device)


def image2latent(model, image, device):
    """Convert image to latent space"""
    with torch.no_grad():
        if type(image) is PILImage:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() * 2 - 1
            image = image.permute(0, 3, 1, 2).to(device)
            if device != "cpu":
                latents = model.vae.module.encode(image)["latent_dist"].mean
            else:
                latents = model.vae.encode(image)["latent_dist"].mean
            latents = latents * 0.18215
    return latents


def find_pred_noise(ldm, image, context, noise_level=-1, device="cuda"):
    """Find predicted noise from diffusion model"""
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(0, 2, 3, 1).detach().cpu().numpy()

    with torch.no_grad():
        latent = image2latent(ldm, image, device)

    noise = torch.randn_like(latent)

    noisy_image = ldm.scheduler.add_noise(
        latent, noise, ldm.scheduler.timesteps[noise_level]
    )

    pred_noise = ldm.unet(noisy_image,
                          ldm.scheduler.timesteps[noise_level].repeat(noisy_image.shape[0]),
                          context.repeat(noisy_image.shape[0], 1, 1))["sample"]

    return noise, pred_noise


def collect_maps(
    controller,
    from_where=["up_cross"],
    upsample_res=512,
    layers=[0, 1, 2, 3],
    indices=None,
):
    """
    Collect and process attention maps from controller
    
    Returns the bilinearly upsampled attention map of size upsample_res x upsample_res
    """

    attention_maps = controller.step_store['attn']
    attention_maps_list = []
    layer_overall = -1

    for layer in range(len(attention_maps)):
        layer_overall += 1

        if layer_overall not in layers:
            continue

        data = attention_maps[layer]

        data = data.reshape(
            data.shape[0], int(data.shape[1] ** 0.5), int(data.shape[1] ** 0.5), data.shape[2]
        )

        if indices is not None:
            data = data[:, :, :, indices]

        data = data.permute(0, 3, 1, 2)

        if upsample_res != -1 and data.shape[1] ** 0.5 != upsample_res:
            # bilinearly upsample the image to attn_sizexattn_size
            data = F.interpolate(
                data,
                size=(upsample_res, upsample_res),
                mode="bilinear",
                align_corners=False,
            )

        attention_maps_list.append(data)

    # More memory-efficient way to compute mean
    if len(attention_maps_list) > 0:
        # Find the maximum size in each dimension across all attention maps
        max_dims = [0, 0, 0, 0]  # [batch, channels, height, width]
        for attn_map in attention_maps_list:
            for i in range(len(attn_map.shape)):
                max_dims[i] = max(max_dims[i], attn_map.shape[i])
        
        # Pad all attention maps to the same size
        processed_maps = []
        for attn_map in attention_maps_list:
            # Calculate padding for each dimension
            padding = []
            for i in range(len(attn_map.shape) - 1, -1, -1):  # F.pad expects padding in reverse order
                pad_size = max_dims[i] - attn_map.shape[i]
                padding.extend([0, pad_size])
            
            # Apply padding
            if any(p > 0 for p in padding):
                padded_map = F.pad(attn_map, padding, mode='constant', value=0)
            else:
                padded_map = attn_map
            processed_maps.append(padded_map)
        
        # Initialize result with first processed map
        result = processed_maps[0].clone()
        # Add remaining maps one by one
        for i in range(1, len(processed_maps)):
            result += processed_maps[i]
        # Compute mean
        result = result / len(processed_maps)
        # Average across batch dimension
        result = result.mean(dim=0)
    else:
        result = torch.tensor([])
    
    controller.reset()

    return result


def run_and_find_attn(
    ldm,
    image,
    context,
    noise_level=-1,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    layers=[0, 1, 2, 3, 4, 5],
    upsample_res=32,
    indices=None,
    controllers=None,
):
    """Run model and extract attention maps"""
    _, _ = find_pred_noise(
        ldm,
        image,
        context,
        noise_level=noise_level,
        device=device,
    )

    attention_maps = []

    for controller in controllers:
        _attention_maps = collect_maps(
            controllers[controller],
            from_where=from_where,
            upsample_res=upsample_res,
            layers=layers,
            indices=indices,
        )

        attention_maps.append(_attention_maps)
        controllers[controller].reset()

    return attention_maps
