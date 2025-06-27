"""
Visualization utilities for StableKeypoints
"""

import torch

from ..data.transforms import RandomAffineWithInverse
from .image_utils import run_and_find_attn


@torch.no_grad()
def run_image_with_context_augmented(
    ldm,
    image,
    context,
    indices,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    layers=[0, 1, 2, 3, 4, 5],
    augmentation_iterations=20,
    noise_level=-1,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    controllers=None,
    num_gpus=1,
    upsample_res=512,
):
    """Run image through model with augmentation for robust keypoint detection"""
    
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()

    num_samples = torch.zeros(len(indices), upsample_res, upsample_res).to(device)
    sum_samples = torch.zeros(len(indices), upsample_res, upsample_res).to(device)

    invertible_transform = RandomAffineWithInverse(
        degrees=augment_degrees,
        scale=augment_scale,
        translate=augment_translate,
    )

    for i in range(augmentation_iterations//num_gpus):

        augmented_img = (
            invertible_transform(torch.tensor(image)[None].repeat(num_gpus, 1, 1, 1).permute(0, 3, 1, 2))
            .permute(0, 2, 3, 1)
            .numpy()
        )

        attn_maps = run_and_find_attn(
            ldm,
            augmented_img,
            context,
            layers=layers,
            noise_level=noise_level,
            from_where=from_where,
            upsample_res=upsample_res,
            device=device,
            controllers=controllers,
            indices=indices.cpu(),
        )

        attn_maps = torch.stack([map.to("cuda:0") for map in attn_maps])

        _num_samples = invertible_transform.inverse(torch.ones_like(attn_maps))
        _sum_samples = invertible_transform.inverse(attn_maps)

        num_samples += _num_samples.sum(dim=0)
        sum_samples += _sum_samples.sum(dim=0)

    # visualize sum_samples/num_samples
    attention_maps = sum_samples / num_samples

    # replace all nans with 0s
    attention_maps[attention_maps != attention_maps] = 0

    return attention_maps
