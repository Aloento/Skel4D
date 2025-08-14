"""
Keypoint visualization utilities for StableKeypoints training
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
from torchvision.utils import make_grid
import matplotlib.cm as cm


def create_keypoint_heatmaps(attention_maps: torch.Tensor, 
                           num_keypoints: int = 8,
                           spatial_shape: Tuple[int, int] = (40, 40)) -> torch.Tensor:
    """
    Create heatmap visualizations from StableKeypoints attention maps.
    
    Args:
        attention_maps: [num_views, spatial, num_tokens] - Individual view attention maps
        num_keypoints: Number of top keypoints to visualize per view
        spatial_shape: (height, width) of individual view
        
    Returns:
        heatmaps: [num_views * num_keypoints, 3, height, width] - RGB heatmaps for visualization
    """
    num_views, spatial, num_tokens = attention_maps.shape
    height, width = spatial_shape
    device = attention_maps.device  # Get the device of input tensors
    
    # Reshape to spatial format: [num_views, height, width, num_tokens]
    spatial_attention = attention_maps.reshape(num_views, height, width, num_tokens)
    
    # Select top keypoints based on maximum attention values
    # Sum across spatial dimensions to find most active tokens
    token_scores = attention_maps.sum(dim=1)  # [num_views, num_tokens]
    top_keypoints = torch.topk(token_scores, num_keypoints, dim=1)[1]  # [num_views, num_keypoints]
    
    heatmaps = []
    
    for view_idx in range(num_views):
        view_attention = spatial_attention[view_idx]  # [height, width, num_tokens]
        top_kp_indices = top_keypoints[view_idx]  # [num_keypoints]
        
        for kp_idx in range(num_keypoints):
            token_idx = top_kp_indices[kp_idx].item()
            
            # Extract attention map for this keypoint: [height, width]
            kp_attention = view_attention[:, :, token_idx]
            
            # Normalize to [0, 1]
            kp_attention = (kp_attention - kp_attention.min()) / (kp_attention.max() - kp_attention.min() + 1e-8)
            
            # Convert to RGB using colormap
            heatmap_np = kp_attention.detach().cpu().numpy()
            colored_heatmap = cm.hot(heatmap_np)[:, :, :3]  # Remove alpha channel
            
            # Convert to tensor and move to correct device: [3, height, width]
            heatmap_tensor = torch.from_numpy(colored_heatmap).permute(2, 0, 1).float().to(device)
            heatmaps.append(heatmap_tensor)
    
    return torch.stack(heatmaps)


def overlay_keypoints_on_images(images: torch.Tensor, 
                               attention_maps: torch.Tensor,
                               num_keypoints: int = 8,
                               alpha: float = 0.7) -> torch.Tensor:
    """
    Overlay keypoint heatmaps on original images.
    
    Args:
        images: [num_views, 3, height, width] - Original view images
        attention_maps: [num_views, spatial, num_tokens] - Attention maps
        num_keypoints: Number of keypoints to overlay
        alpha: Blending factor for overlay (0=only image, 1=only heatmap)
        
    Returns:
        overlays: [num_views, 3, height, width] - Images with keypoint overlays
    """
    num_views, channels, height, width = images.shape
    device = images.device  # Get device from input images
    
    # Ensure attention_maps are on the same device as images
    attention_maps = attention_maps.to(device)
    
    # Create heatmaps
    heatmaps = create_keypoint_heatmaps(
        attention_maps, 
        num_keypoints=num_keypoints, 
        spatial_shape=(height, width)
    )  # [num_views * num_keypoints, 3, height, width]
    
    # Reshape heatmaps back to view format and aggregate
    heatmaps = heatmaps.reshape(num_views, num_keypoints, 3, height, width)
    
    # Aggregate keypoints by taking maximum across keypoint dimension
    aggregated_heatmaps = torch.max(heatmaps, dim=1)[0]  # [num_views, 3, height, width]
    
    # Ensure both tensors are on the same device for blending
    aggregated_heatmaps = aggregated_heatmaps.to(device)
    
    # Blend with original images
    overlays = alpha * aggregated_heatmaps + (1 - alpha) * images
    
    return overlays


def create_keypoint_visualization_grid(images: torch.Tensor,
                                     attention_maps: torch.Tensor,
                                     num_keypoints_to_show: int = 4) -> torch.Tensor:
    """
    Create a comprehensive visualization grid showing images and keypoint heatmaps.
    
    Args:
        images: [num_views, 3, height, width] - Original view images  
        attention_maps: [num_views, spatial, num_tokens] - Attention maps
        num_keypoints_to_show: Number of individual keypoint heatmaps to show
        
    Returns:
        visualization_grid: [3, grid_height, grid_width] - Complete visualization
    """
    num_views = images.shape[0]
    device = images.device  # Get device from input images
    
    # Ensure attention_maps are on the same device as images
    attention_maps = attention_maps.to(device)
    
    # Create overlaid images
    overlaid_images = overlay_keypoints_on_images(images, attention_maps, num_keypoints=8, alpha=0.5)
    
    # Create individual keypoint heatmaps (top keypoints only)
    individual_heatmaps = create_keypoint_heatmaps(
        attention_maps, 
        num_keypoints=num_keypoints_to_show,
        spatial_shape=(images.shape[2], images.shape[3])
    )  # [num_views * num_keypoints_to_show, 3, height, width]
    
    # Organize visualization:
    # Row 1: Original images (6 views)
    # Row 2: Overlaid images (6 views)  
    # Row 3-N: Individual keypoint heatmaps (6 views * num_keypoints_to_show)
    
    all_visualizations = []
    
    # Add original images
    all_visualizations.extend([images[i] for i in range(num_views)])
    
    # Add overlaid images
    all_visualizations.extend([overlaid_images[i] for i in range(num_views)])
    
    # Add individual keypoint heatmaps (organized by keypoint, then by view)
    for kp_idx in range(num_keypoints_to_show):
        for view_idx in range(num_views):
            heatmap_idx = view_idx * num_keypoints_to_show + kp_idx
            all_visualizations.append(individual_heatmaps[heatmap_idx])
    
    # Create grid: (2 + num_keypoints_to_show) rows, num_views columns
    grid_tensor = torch.stack(all_visualizations)
    num_rows = 2 + num_keypoints_to_show
    
    # Use make_grid to create the final visualization
    visualization_grid = make_grid(
        grid_tensor, 
        nrow=num_views, 
        normalize=True, 
        value_range=(0, 1),
        pad_value=1.0  # White padding
    )
    
    return visualization_grid


def extract_and_visualize_keypoints_from_sk_ref(sk_ref_dict: Dict[str, torch.Tensor],
                                              target_images: torch.Tensor) -> Optional[Tuple[torch.Tensor, Dict[str, float]]]:
    """
    Extract keypoint attention maps from sk_ref_dict and create visualization.
    
    Args:
        sk_ref_dict: Dictionary mapping layer names to attention tensors
        target_images: [batch_size, num_views, 3, height, width] - Target view images
        
    Returns:
        Tuple of (visualization_grid, attention_stats) or None if no valid data
        visualization_grid: [3, grid_height, grid_width] - Complete visualization
        attention_stats: Dict with attention statistics for wandb logging
    """
    if not sk_ref_dict:
        return None
    
    from src.utils.attention_extraction import extract_sk_attention_auto_dimensions
    
    # Get device from target_images
    device = target_images.device
    
    # Filter for high-resolution layers (most informative for keypoints)
    target_layers = ['down_blocks.0', 'up_blocks.3']  # 9600 spatial resolution
    
    aggregated_attention_maps = []
    attention_stats = {}
    
    for layer_name, attention_tensor in sk_ref_dict.items():
        # Check if this is a target layer
        if any(target in layer_name for target in target_layers):
            # Ensure attention tensor is on the correct device
            attention_tensor = attention_tensor.to(device)
            # Extract SK attention maps with individual view separation
            sk_data = extract_sk_attention_auto_dimensions(attention_tensor)
            # sk_data['individual_views'] has shape [6, view_spatial, 16]
            aggregated_attention_maps.append(sk_data['individual_views'])
            
            # Compute attention statistics for this layer
            attention_maps = sk_data['individual_views']
            attention_stats[f'{layer_name}_mean_attention'] = attention_maps.mean().item()
            attention_stats[f'{layer_name}_std_attention'] = attention_maps.std().item()
            attention_stats[f'{layer_name}_max_attention'] = attention_maps.max().item()
    
    if not aggregated_attention_maps:
        return None
    
    # Average attention maps across layers: [6, view_spatial, 16]
    avg_attention_maps = torch.mean(torch.stack(aggregated_attention_maps), dim=0)
    
    # Ensure attention maps are on the same device
    avg_attention_maps = avg_attention_maps.to(device)
    
    # Compute global attention statistics
    attention_stats['global_mean_attention'] = avg_attention_maps.mean().item()
    attention_stats['global_std_attention'] = avg_attention_maps.std().item()
    attention_stats['global_max_attention'] = avg_attention_maps.max().item()
    
    # Compute keypoint concentration (entropy-based measure)
    # Lower entropy = more concentrated keypoints
    normalized_attention = F.softmax(avg_attention_maps.flatten(start_dim=1), dim=1)  # [6, spatial*16]
    attention_entropy = -(normalized_attention * torch.log(normalized_attention + 1e-10)).sum(dim=1).mean()
    attention_stats['keypoint_entropy'] = attention_entropy.item()
    
    # Compute keypoint diversity (variance of attention across tokens)
    token_variances = avg_attention_maps.var(dim=1)  # [6, 16] - variance across spatial for each token
    attention_stats['keypoint_diversity'] = token_variances.mean().item()
    
    # Use first batch item for visualization
    batch_images = target_images[0]  # [num_views, 3, height, width]
    
    # Resize images to match attention map spatial resolution if needed
    # For 1600 spatial -> 40x40, need to resize images to 40x40
    spatial_dim = avg_attention_maps.shape[1]
    if spatial_dim == 1600:  # 40x40
        target_size = (40, 40)
    elif spatial_dim == 400:  # 20x20 (for lower resolution layers)
        target_size = (20, 20)
    else:
        # Auto-detect square spatial dimensions
        side = int(spatial_dim ** 0.5)
        target_size = (side, side)
    
    # Resize images to match attention spatial resolution
    batch_images_resized = F.interpolate(
        batch_images, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
    
    # Create visualization grid
    visualization_grid = create_keypoint_visualization_grid(
        batch_images_resized,
        avg_attention_maps,
        num_keypoints_to_show=4
    )
    
    return visualization_grid, attention_stats
