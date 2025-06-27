"""
Embedding optimization for StableKeypoints
"""

import time
import torch
from tqdm import tqdm

from ..data.dataset import CustomDataset
from ..data.transforms import RandomAffineWithInverse
from ..utils.image_utils import init_random_noise, run_and_find_attn
from ..utils.keypoint_utils import find_top_k_gaussian, furthest_point_sampling
from .losses import sharpening_loss, equivariance_loss


def optimize_embedding(
    ldm,
    context=None,
    device="cuda",
    num_steps=2000,
    from_where=["down_cross", "mid_cross", "up_cross"],
    upsample_res=256,
    layers=[0, 1, 2, 3, 4, 5],
    lr=5e-3,
    noise_level=-1,
    num_tokens=1000,
    top_k=10,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    dataset_loc="~",
    sigma=1.0,
    sharpening_loss_weight=100,
    equivariance_attn_loss_weight=100,
    batch_size=4,
    num_gpus=1,
    max_len=-1,
    min_dist=0.05,
    furthest_point_num_samples=50,
    controllers=None,
    validation=False,
    num_subjects=1,
):
    """
    Optimize context embedding for keypoint detection
    
    Args:
        ldm: Loaded diffusion model
        context: Initial context embedding (if None, creates random)
        device: Device to run optimization on
        num_steps: Number of optimization steps
        ... (other parameters)
        
    Returns:
        Optimized context embedding
    """
    
    dataset = CustomDataset(data_root=dataset_loc, image_size=512)

    invertible_transform = RandomAffineWithInverse(
        degrees=augment_degrees,
        scale=augment_scale,
        translate=augment_translate,
    )

    if context is None:
        context = init_random_noise(device, num_words=num_tokens)

    context.requires_grad = True

    # optimize context to maximize attention at pixel_loc
    optimizer = torch.optim.Adam([context], lr=lr)

    start = time.time()

    running_equivariance_attn_loss = 0
    running_sharpening_loss = 0
    running_total_loss = 0

    # create dataloader for the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_gpus, shuffle=True, drop_last=True)
    dataloader_iter = iter(dataloader)

    for iteration in tqdm(range(int(num_steps*batch_size))):

        try:
            mini_batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            mini_batch = next(dataloader_iter)

        image = mini_batch["img"]

        attn_maps = run_and_find_attn(
            ldm,
            image,
            context,
            layers=layers,
            noise_level=noise_level,
            from_where=from_where,
            upsample_res=-1,
            device=device,
            controllers=controllers,
        )

        transformed_img = invertible_transform(image)

        attention_maps_transformed = run_and_find_attn(
            ldm,
            transformed_img,
            context,
            layers=layers,
            noise_level=noise_level,
            from_where=from_where,
            upsample_res=-1,
            device=device,
            controllers=controllers,
        )

        _sharpening_loss = []
        _loss_equivariance_attn = []

        for index, attn_map, attention_map_transformed in zip(torch.arange(num_gpus), attn_maps, attention_maps_transformed):

            top_embedding_indices = find_top_k_gaussian(
                attn_map, furthest_point_num_samples, sigma=sigma, num_subjects=num_subjects
            )

            top_embedding_indices = furthest_point_sampling(attention_map_transformed, top_k, top_embedding_indices)

            _sharpening_loss.append(sharpening_loss(attn_map[top_embedding_indices], device=device, sigma=sigma, num_subjects=num_subjects))

            _loss_equivariance_attn.append(equivariance_loss(
                attn_map[top_embedding_indices], attention_map_transformed[top_embedding_indices][None].repeat(num_gpus, 1, 1, 1), invertible_transform, index
            ))

        _sharpening_loss = torch.stack([loss.to('cuda:0') for loss in _sharpening_loss]).mean()
        _loss_equivariance_attn = torch.stack([loss.to('cuda:0') for loss in _loss_equivariance_attn]).mean()

        loss = (
            + _loss_equivariance_attn * equivariance_attn_loss_weight
            + _sharpening_loss * sharpening_loss_weight
        )

        running_equivariance_attn_loss += _loss_equivariance_attn / (batch_size//num_gpus) * equivariance_attn_loss_weight
        running_sharpening_loss += _sharpening_loss / (batch_size//num_gpus) * sharpening_loss_weight
        running_total_loss += loss / (batch_size//num_gpus)

        loss = loss / (batch_size//num_gpus)

        if iteration % 50 == 0:
            print(
                f"loss: {loss.item()}, "
                f"_loss_equivariance_attn: {running_equivariance_attn_loss.item()} "
                f"sharpening_loss: {running_sharpening_loss.item()}, "
                f"running_total_loss: {running_total_loss.item()}, "
            )
        loss.backward()
        if (iteration + 1) % (batch_size//num_gpus) == 0:
            optimizer.step()
            optimizer.zero_grad()
            running_equivariance_attn_loss = 0
            running_sharpening_loss = 0
            running_total_loss = 0

    print(f"optimization took {time.time() - start} seconds")

    return context.detach()


@torch.no_grad()
def find_best_indices(
    ldm,
    context,
    num_steps=100,
    noise_level=-1,
    upsample_res=256,
    layers=[0, 1, 2, 3, 4, 5],
    from_where=["down_cross", "mid_cross", "up_cross"],
    top_k=10,
    dataset_loc="~",
    furthest_point_num_samples=50,
    controllers=None,
    num_gpus=1,
    sigma=3,
    num_subjects=1,
):
    """Find the best indices for keypoint detection"""

    dataset = CustomDataset(data_root=dataset_loc, image_size=512)

    maps = []
    indices_list = []

    # create dataloader for the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_gpus, shuffle=True, drop_last=True)
    dataloader_iter = iter(dataloader)

    for _ in tqdm(range(num_steps//num_gpus)):

        try:
            mini_batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            mini_batch = next(dataloader_iter)

        image = mini_batch["img"]

        attention_maps = run_and_find_attn(
            ldm,
            image,
            context,
            layers=layers,
            noise_level=noise_level,
            from_where=from_where,
            upsample_res=upsample_res,
            controllers=controllers,
        )

        for attention_map in attention_maps:

            top_initial_candidates = find_top_k_gaussian(
                attention_map, furthest_point_num_samples, sigma=sigma, num_subjects=num_subjects
            )

            top_embedding_indices = furthest_point_sampling(attention_map, top_k, top_initial_candidates)

            indices_list.append(top_embedding_indices.cpu())

    # find the top_k most common indices
    indices_list = torch.cat([index for index in indices_list])
    indices, counts = torch.unique(indices_list, return_counts=True)
    indices = indices[counts.argsort(descending=True)]
    indices = indices[:top_k]

    return indices
