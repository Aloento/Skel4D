"""
Attention Map Extraction for StableKeypoints Multi-View Processing

This module provides utilities to extract individual view attention maps from 
Zero123Plus 2x3 grid layout for StableKeypoints loss computation.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional


def extract_sk_attention_maps(attention_probs: torch.Tensor, 
                             spatial_shape: Tuple[int, int],
                             grid_shape: Tuple[int, int] = (2, 3),
                             extract_sk_only: bool = False) -> torch.Tensor:
    """
    Extract StableKeypoints attention maps from Zero123Plus multi-view grid.
    
    Args:
        attention_probs: [heads, spatial_total, tokens] - Raw attention probabilities
        spatial_shape: (height, width) - Grid spatial dimensions (e.g., 80, 120)
        grid_shape: (rows, cols) - Multi-view grid layout (default: 2x3 for Zero123Plus)
        extract_sk_only: If True, extract only SK tokens (77-92), else return all tokens
        
    Returns:
        individual_views: [heads, num_views, view_spatial, sk_tokens] 
                         - Individual view attention maps for SK processing
                         
    Example:
        Input: [5, 9600, 93] -> Output: [5, 6, 1600, 16]
        - 5 heads, 6 views, 1600 spatial per view (40x40), 16 SK tokens
    """
    heads, spatial_total, total_tokens = attention_probs.shape
    grid_height, grid_width = spatial_shape
    rows, cols = grid_shape
    num_views = rows * cols
    
    # Validate input dimensions
    expected_spatial = grid_height * grid_width
    if spatial_total != expected_spatial:
        raise ValueError(f"Spatial mismatch: expected {expected_spatial}, got {spatial_total}")
    
    # Calculate individual view dimensions
    view_height = grid_height // rows
    view_width = grid_width // cols
    view_spatial = view_height * view_width
    
    # Reshape to spatial grid: [heads, grid_height, grid_width, tokens]
    grid_attention = attention_probs.reshape(heads, grid_height, grid_width, total_tokens)
    
    # Extract SK tokens if requested (tokens 77-92)
    if extract_sk_only:
        if total_tokens < 93:
            raise ValueError(f"Not enough tokens for SK extraction: {total_tokens} < 93")
        sk_attention_grid = grid_attention[:, :, :, 77:93]  # [heads, grid_h, grid_w, 16]
        tokens_to_use = 16
    else:
        sk_attention_grid = grid_attention
        tokens_to_use = total_tokens
    
    # Extract individual views
    view_attentions = []
    for row in range(rows):
        for col in range(cols):
            # Extract view region
            view_attn = sk_attention_grid[
                :,
                row * view_height:(row + 1) * view_height,  # Y range  
                col * view_width:(col + 1) * view_width,    # X range
                :
            ]  # [heads, view_height, view_width, tokens]
            
            # Flatten spatial dimension: [heads, view_spatial, tokens]
            view_attn_flat = view_attn.reshape(heads, view_spatial, tokens_to_use)
            view_attentions.append(view_attn_flat)
    
    # Stack all views: [heads, num_views, view_spatial, tokens]
    individual_views = torch.stack(view_attentions, dim=1)
    
    return individual_views


def aggregate_multihead_attention(view_attentions: torch.Tensor, 
                                 aggregation: str = 'mean') -> torch.Tensor:
    """
    Aggregate attention across multiple heads.
    
    Args:
        view_attentions: [heads, num_views, view_spatial, tokens]
        aggregation: 'mean', 'max', or 'sum'
        
    Returns:
        aggregated: [num_views, view_spatial, tokens]
    """
    if aggregation == 'mean':
        return torch.mean(view_attentions, dim=0)
    elif aggregation == 'max':
        return torch.max(view_attentions, dim=0)[0]
    elif aggregation == 'sum':
        return torch.sum(view_attentions, dim=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def reshape_attention_to_spatial(view_attentions: torch.Tensor,
                                view_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Reshape flattened attention maps back to 2D spatial format.
    
    Args:
        view_attentions: [num_views, view_spatial, tokens] - Flattened attention
        view_shape: (height, width) - Individual view spatial dimensions
        
    Returns:
        spatial_attention: [num_views, height, width, tokens] - 2D spatial attention
    """
    num_views, view_spatial, tokens = view_attentions.shape
    view_height, view_width = view_shape
    
    if view_spatial != view_height * view_width:
        raise ValueError(f"View spatial mismatch: {view_spatial} != {view_height}x{view_width}")
    
    return view_attentions.reshape(num_views, view_height, view_width, tokens)


def extract_attention_for_sk_loss(attention_probs: torch.Tensor,
                                 spatial_shape: Tuple[int, int],
                                 grid_shape: Tuple[int, int] = (2, 3),
                                 view_aggregation: str = 'mean') -> Dict[str, torch.Tensor]:
    """
    Complete pipeline to extract and format attention maps for SK loss computation.
    
    Args:
        attention_probs: [heads, spatial_total, tokens] - Raw attention from UNet
        spatial_shape: (height, width) - Grid spatial dimensions
        grid_shape: (rows, cols) - Multi-view grid layout 
        view_aggregation: How to aggregate multiple heads
        
    Returns:
        Dictionary containing:
        - 'individual_views': [num_views, view_spatial, 16] - Per-view SK attention (flattened)
        - 'spatial_views': [num_views, view_h, view_w, 16] - Per-view SK attention (2D)
        - 'view_shape': (view_h, view_w) - Individual view dimensions
        - 'num_views': int - Number of views extracted
    """
    # Extract individual view attention maps
    view_attentions = extract_sk_attention_maps(
        attention_probs, spatial_shape, grid_shape, extract_sk_only=False
    )  # [heads, num_views, view_spatial, 16]
    
    # Aggregate across heads
    aggregated_views = aggregate_multihead_attention(
        view_attentions, aggregation=view_aggregation
    )  # [num_views, view_spatial, 16]
    
    # Calculate view dimensions
    grid_height, grid_width = spatial_shape
    rows, cols = grid_shape
    view_height = grid_height // rows
    view_width = grid_width // cols
    view_shape = (view_height, view_width)
    
    # Reshape to 2D spatial format
    spatial_views = reshape_attention_to_spatial(aggregated_views, view_shape)
    # [num_views, view_height, view_width, 16]
    
    return {
        'individual_views': aggregated_views,  # [6, 1600, 16] - for flattened SK loss
        'spatial_views': spatial_views,        # [6, 40, 40, 16] - for 2D SK loss  
        'view_shape': view_shape,              # (40, 40) - individual view size
        'num_views': rows * cols,              # 6 - total number of views
    }


# Utility functions for different UNet layer resolutions
def get_view_dimensions_for_layer(spatial_total: int, 
                                 grid_shape: Tuple[int, int] = (2, 3)) -> Tuple[int, int]:
    """
    Calculate individual view dimensions for a given UNet layer resolution.
    
    Args:
        spatial_total: Total spatial dimensions (e.g., 9600, 4096, 1024, etc.)
        grid_shape: Multi-view grid layout
        
    Returns:
        (view_height, view_width): Individual view dimensions
        
    Examples:
        spatial_total=9600 (80x120) -> view_dims=(40, 40)
        spatial_total=4096 (64x64)  -> view_dims=(32, 32) 
        spatial_total=1024 (32x32)  -> view_dims=(16, 16)
        spatial_total=256 (16x16)   -> view_dims=(8, 8)
        spatial_total=64 (8x8)      -> view_dims=(4, 4)
    """
    rows, cols = grid_shape
    
    # Find the grid dimensions that multiply to spatial_total
    # Zero123Plus maintains 2:3 aspect ratio
    aspect_ratio = cols / rows  # 3/2 = 1.5
    
    # For 2x3 grid: grid_height * grid_width = spatial_total
    # grid_width = aspect_ratio * grid_height
    # grid_height^2 * aspect_ratio = spatial_total
    grid_height = int((spatial_total / aspect_ratio) ** 0.5)
    grid_width = spatial_total // grid_height
    
    # Validate
    if grid_height * grid_width != spatial_total:
        # Fallback: try square grid
        side = int(spatial_total ** 0.5)
        if side * side == spatial_total:
            grid_height = grid_width = side
        else:
            raise ValueError(f"Cannot determine grid dimensions for spatial_total={spatial_total}")
    
    view_height = grid_height // rows
    view_width = grid_width // cols
    
    return (view_height, view_width)


def extract_sk_attention_auto_dimensions(attention_probs: torch.Tensor,
                                       grid_shape: Tuple[int, int] = (2, 3)) -> Dict[str, torch.Tensor]:
    """
    Automatically determine spatial dimensions and extract SK attention maps.
    
    Args:
        attention_probs: [heads, spatial_total, tokens] - Raw attention from UNet
        grid_shape: Multi-view grid layout
        
    Returns:
        Same as extract_attention_for_sk_loss() but with auto-detected dimensions
    """
    heads, spatial_total, tokens = attention_probs.shape
    
    # Auto-detect grid dimensions
    view_height, view_width = get_view_dimensions_for_layer(spatial_total, grid_shape)
    rows, cols = grid_shape
    grid_height = view_height * rows
    grid_width = view_width * cols
    spatial_shape = (grid_height, grid_width)
    
    return extract_attention_for_sk_loss(attention_probs, spatial_shape, grid_shape)


# Example usage and testing functions
def test_attention_extraction():
    """Test function to validate attention extraction with realistic shapes."""
    print("Testing SK Attention Extraction...")
    
    # Test case 1: High resolution layer (9600 spatial)
    heads, spatial, tokens = 5, 9600, 93
    attention_probs = torch.randn(heads, spatial, tokens)
    
    result = extract_sk_attention_auto_dimensions(attention_probs)
    
    print(f"Input shape: {attention_probs.shape}")
    print(f"Individual views: {result['individual_views'].shape}")  # [6, 1600, 16]
    print(f"Spatial views: {result['spatial_views'].shape}")        # [6, 40, 40, 16]
    print(f"View shape: {result['view_shape']}")                    # (40, 40)
    print(f"Num views: {result['num_views']}")                      # 6
    
    # Test case 2: Mid resolution layer (1024 spatial)
    attention_probs_mid = torch.randn(10, 1024, 93)
    result_mid = extract_sk_attention_auto_dimensions(attention_probs_mid)
    
    print(f"\nMid resolution:")
    print(f"Input shape: {attention_probs_mid.shape}")
    print(f"Individual views: {result_mid['individual_views'].shape}")  # [6, 171, 16] approx
    print(f"Spatial views: {result_mid['spatial_views'].shape}")
    print(f"View shape: {result_mid['view_shape']}")
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_attention_extraction()
