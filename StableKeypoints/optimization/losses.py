"""
Loss functions for StableKeypoints optimization
"""

import torch.nn.functional as F
from ..utils.keypoint_utils import gaussian_circles, find_k_max_pixels


def find_gaussian_loss_at_point(
    attn_map, pos, sigma=1.0, temperature=1e-1, device="cuda", indices=None, num_subjects=1
):
    """
    Calculate Gaussian loss at specific point
    
    Args:
        attn_map: Attention map tensor
        pos: Position between 0 and 1
        sigma: Gaussian sigma parameter
        temperature: Temperature parameter
        device: Device to run on
        indices: Optional indices to select subset
        num_subjects: Number of subjects
    """

    # attn_map is of shape (T, H, W)
    T, H, W = attn_map.shape

    # Create Gaussian circle at the given position
    target = gaussian_circles(
        pos, size=H, sigma=sigma, device=attn_map.device
    )  # Assuming H and W are the same
    target = target.to(attn_map.device)

    # possibly select a subset of indices
    if indices is not None:
        attn_map = attn_map[indices]
        target = target[indices]

    # Compute loss
    loss = F.mse_loss(attn_map, target)

    return loss


def sharpening_loss(attn_map, sigma=1.0, temperature=1e1, device="cuda", num_subjects=1):
    """Calculate sharpening loss to encourage focused attention"""

    pos = find_k_max_pixels(attn_map, num=num_subjects)/attn_map.shape[-1]

    loss = find_gaussian_loss_at_point(
        attn_map,
        pos,
        sigma=sigma,
        temperature=temperature,
        device=device,
        num_subjects=num_subjects,
    )

    return loss


def equivariance_loss(embeddings_initial, embeddings_transformed, transform, index):
    """Calculate equivariance loss for transformation consistency"""
    # untransform the embeddings_transformed
    embeddings_initial_prime = transform.inverse(embeddings_transformed)[index]

    loss = F.mse_loss(embeddings_initial, embeddings_initial_prime)

    return loss
