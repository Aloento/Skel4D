"""
Loss functions for StableKeypoints optimization
"""

import torch
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


def temporal_consistency_loss(attn_t, attn_t1, loss_type="l2", mask=None):
    """
    Calculate temporal consistency loss between attention maps of consecutive frames
    
    Args:
        attn_t: Attention maps for frame t, shape [B, K, H, W] or [K, H, W]
        attn_t1: Attention maps for frame t+1, shape [B, K, H, W] or [K, H, W]
        loss_type: Type of loss to use ("l2" or "kl")
        mask: Optional mask to weight different keypoints, shape [K] or [B, K]
        
    Returns:
        Temporal consistency loss value
    """
    if attn_t.shape != attn_t1.shape:
        raise ValueError(f"Attention maps must have same shape: {attn_t.shape} vs {attn_t1.shape}")
    
    if loss_type == "l2":
        # L2 (MSE) loss between attention maps
        loss = F.mse_loss(attn_t, attn_t1, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            if len(mask.shape) == 1:  # [K]
                if len(loss.shape) == 4:  # [B, K, H, W]
                    mask = mask.view(1, -1, 1, 1)
                else:  # [K, H, W]
                    mask = mask.view(-1, 1, 1)
            loss = loss * mask
        
        # Average over spatial dimensions and keypoints
        loss = loss.mean()
        
        # Scale the loss to make it more significant
        # Temporal consistency loss should be in similar magnitude to other losses
        loss = loss * 1000.0  # Scale factor to make temporal loss more prominent
        
    elif loss_type == "kl":
        # KL divergence loss
        # Normalize attention maps to make them proper distributions
        attn_t_norm = F.softmax(attn_t.flatten(-2), dim=-1).view_as(attn_t)
        attn_t1_norm = F.softmax(attn_t1.flatten(-2), dim=-1).view_as(attn_t1)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        attn_t_norm = attn_t_norm + eps
        attn_t1_norm = attn_t1_norm + eps
        
        # Calculate KL divergence
        loss = F.kl_div(
            attn_t1_norm.log(), 
            attn_t_norm, 
            reduction='none',
            log_target=False
        )
        
        # Apply mask if provided
        if mask is not None:
            if len(mask.shape) == 1:  # [K]
                if len(loss.shape) == 4:  # [B, K, H, W]
                    mask = mask.view(1, -1, 1, 1)
                else:  # [K, H, W]
                    mask = mask.view(-1, 1, 1)
            loss = loss * mask
        
        # Average over spatial dimensions and keypoints
        loss = loss.mean()
        
        # Scale the KL loss to make it more significant
        loss = loss * 100.0  # Scale factor for KL divergence
        
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss


def adaptive_temporal_loss(attn_t, attn_t1, motion_threshold=0.1):
    """
    Adaptive temporal consistency loss that adjusts based on motion magnitude
    
    Args:
        attn_t: Attention maps for frame t
        attn_t1: Attention maps for frame t+1
        motion_threshold: Threshold for determining significant motion
        
    Returns:
        Adaptive temporal loss value
    """
    # Calculate base temporal loss
    base_loss = temporal_consistency_loss(attn_t, attn_t1, loss_type="l2")
    
    # Calculate motion magnitude for each keypoint
    motion_magnitude = torch.mean((attn_t - attn_t1) ** 2, dim=(-2, -1))  # [B, K] or [K]
    
    # Create adaptive weights (higher weight for low motion, encouraging stability)
    # Lower weight for high motion, allowing more flexibility
    adaptive_weights = torch.exp(-motion_magnitude / motion_threshold)
    
    # Apply adaptive weights
    weighted_loss = base_loss * adaptive_weights.mean()
    
    return weighted_loss
