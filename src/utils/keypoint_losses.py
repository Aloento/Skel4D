"""
Keypoint loss computation utilities for StableKeypoints integration with Zero123Plus
"""

import torch
import torch.nn.functional as F


def find_top_k_gaussian_batch(attention_maps, top_k, sigma=1.0, epsilon=1e-5):
    """
    Find top-k embeddings that best fit Gaussian patterns.
    Similar to StableKeypoints find_top_k_gaussian but adapted for batch processing.
    
    Args:
        attention_maps: [num_tokens, height, width] attention maps
        top_k: number of top candidates to return
        sigma: Gaussian sigma for evaluation
        epsilon: small value to avoid numerical issues
        
    Returns:
        Tensor of selected token indices
    """
    num_tokens, height, width = attention_maps.shape
    device = attention_maps.device
    
    kl_distances = []
    
    # Evaluate each token's attention map for Gaussian-like patterns
    for token_idx in range(num_tokens):
        attn_map = attention_maps[token_idx]  # [height, width]
        
        # Find max pixel location
        max_pixel_location = find_max_pixel_single(attn_map.unsqueeze(0)).squeeze(0) / height  # normalize to [0,1]
        
        # Create Gaussian target at max location
        target = gaussian_circles_single(
            max_pixel_location.unsqueeze(0),  # [1, 2]
            size=height, 
            sigma=sigma, 
            device=device
        ).squeeze(0)  # [height, width]
        
        # Normalize attention map to probability distribution
        attn_map_flat = attn_map.view(-1)
        attn_map_softmax = F.softmax(attn_map_flat + epsilon, dim=-1)
        
        # Normalize target
        target_flat = target.view(-1) + epsilon
        target_flat = target_flat / target_flat.sum()
        
        # Compute KL divergence (lower is better)
        kl_div = torch.sum(target_flat * (torch.log(target_flat) - torch.log(attn_map_softmax)))
        kl_distances.append(kl_div)
    
    # Sort by KL distance (ascending - lower is better)
    kl_distances = torch.stack(kl_distances)
    sorted_indices = torch.argsort(kl_distances, descending=False)
    
    # Return top-k candidates
    return sorted_indices[:min(top_k, num_tokens)]


def furthest_point_sampling_batch(attention_maps, top_k, initial_candidates):
    """
    Apply furthest point sampling to get diverse keypoints.
    Similar to StableKeypoints furthest_point_sampling.
    
    Args:
        attention_maps: [num_tokens, height, width] attention maps
        top_k: number of final points to select
        initial_candidates: tensor of candidate token indices
        
    Returns:
        Tensor of selected token indices
    """
    if len(initial_candidates) <= top_k:
        return initial_candidates
    
    device = attention_maps.device
    height = attention_maps.shape[1]
    
    # Get max pixel locations for all candidates
    candidate_locations = []
    for idx in initial_candidates:
        max_loc = find_max_pixel_single(attention_maps[idx].unsqueeze(0)).squeeze(0) / height
        candidate_locations.append(max_loc)
    
    candidate_locations = torch.stack(candidate_locations)  # [num_candidates, 2]
    
    # Find the two points with maximum distance
    max_dist = -1
    furthest_pair = (0, 1)
    
    for i in range(len(initial_candidates)):
        for j in range(i + 1, len(initial_candidates)):
            dist = torch.sqrt(torch.sum((candidate_locations[i] - candidate_locations[j]) ** 2))
            if dist > max_dist:
                max_dist = dist
                furthest_pair = (i, j)
    
    # Initialize with furthest pair
    selected_indices = [initial_candidates[furthest_pair[0]].item(), initial_candidates[furthest_pair[1]].item()]
    selected_locations = [candidate_locations[furthest_pair[0]], candidate_locations[furthest_pair[1]]]
    
    # Iteratively add furthest points
    for _ in range(top_k - 2):
        max_min_dist = -1
        furthest_point_idx = None
        
        for i, candidate_idx in enumerate(initial_candidates):
            if candidate_idx.item() in selected_indices:
                continue
            
            # Find minimum distance to already selected points
            candidate_loc = candidate_locations[i]
            min_dist = float('inf')
            
            for selected_loc in selected_locations:
                dist = torch.sqrt(torch.sum((candidate_loc - selected_loc) ** 2))
                min_dist = min(min_dist, dist)
            
            # Keep track of point with maximum minimum distance
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                furthest_point_idx = candidate_idx.item()
                furthest_point_loc = candidate_loc
        
        if furthest_point_idx is not None:
            selected_indices.append(furthest_point_idx)
            selected_locations.append(furthest_point_loc)
    
    return torch.tensor(selected_indices, device=device)


def sharpening_loss_batch(attention_maps, sigma=1.0, temperature: float = 10.0, loss_type: str = 'mse'):
    """Sharpening loss encouraging each attention map to become a spatially localized Gaussian.

    Differences vs previous version:
    - Applies a spatial softmax with temperature to each map so optimization focuses on concentrating mass
    - Normalizes the Gaussian target to sum to 1 for comparable scale
    - Supports 'mse' (default) or 'kl' divergence objectives

    Args:
        attention_maps: [num_selected, H, W]
        sigma: Gaussian sigma parameter (in pixels)
        temperature: Softmax temperature (>0). Larger -> sharper distribution
        loss_type: 'mse' or 'kl'
    """
    if attention_maps.numel() == 0:
        return torch.tensor(0.0, device=attention_maps.device)

    num_selected, height, width = attention_maps.shape
    device = attention_maps.device
    total_loss = 0.0

    for i in range(num_selected):
        attn_map = attention_maps[i]

        # Spatial softmax normalization (encourage single peak rather than ring due to token competition)
        attn_flat = (attn_map.view(-1) * temperature).softmax(dim=-1)
        attn_norm = attn_flat.view(height, width)

        # Peak location from normalized map
        max_pixel_location = find_max_pixel_single(attn_norm.unsqueeze(0)).squeeze(0) / height

        # Normalized Gaussian target
        target = gaussian_circles_single(max_pixel_location.unsqueeze(0), size=height, sigma=sigma, device=device)
        target = target / (target.sum() + 1e-8)

        if loss_type == 'kl':
            # KL(target || attn_norm)
            attn_eps = attn_norm + 1e-8
            loss = (target * (target.add(1e-8).log() - attn_eps.log())).sum()
        else:  # mse
            loss = F.mse_loss(attn_norm, target)
        total_loss += loss

    return total_loss / num_selected


def compute_sharpening_loss_batch(attention_maps, apply_diversity_filter=True, top_k=10, furthest_point_num_samples=50, sigma=1.0):
    """
    Compute sharpening loss with StableKeypoints diversity enforcement through softmax.
    This is the main function that implements the complete SK pipeline including:
    1. Softmax normalization for diversity enforcement  
    2. Gaussian fitness filtering
    3. Furthest point sampling for spatial diversity
    4. Sharpening loss on selected diverse tokens
    
    Args:
        attention_maps: [batch_size, num_tokens, height, width] attention maps
        apply_diversity_filter: If True, apply SK diversity filtering pipeline
        top_k: Final number of keypoints to select 
        furthest_point_num_samples: Initial candidate pool size
        sigma: Gaussian sigma parameter
        
    Returns:
        Sharpening loss value
    """
    if attention_maps.dim() == 3:
        # Add batch dimension: [num_tokens, height, width] -> [1, num_tokens, height, width]
        attention_maps = attention_maps.unsqueeze(0)
    
    batch_size, num_tokens, height, width = attention_maps.shape
    device = attention_maps.device
    
    if not apply_diversity_filter:
        # Simple sharpening loss without diversity filtering
        total_loss = 0.0
        for b in range(batch_size):
            batch_loss = sharpening_loss_batch(attention_maps[b], sigma=sigma)  # [num_tokens, H, W]
            total_loss += batch_loss
        return total_loss / batch_size
    
    # Apply full StableKeypoints diversity pipeline
    total_loss = 0.0
    valid_batch_count = 0
    
    for b in range(batch_size):
        batch_attention_maps = attention_maps[b]  # [num_tokens, height, width]
        
        # Step 1: Find top candidates using Gaussian fitness with softmax normalization
        # This is where diversity is enforced through softmax as described in SK paper
        top_initial_candidates = find_top_k_gaussian_batch(
            batch_attention_maps, 
            furthest_point_num_samples, 
            sigma=sigma
        )
        
        if len(top_initial_candidates) == 0:
            continue
            
        # Step 2: Apply furthest point sampling for spatial diversity
        if len(top_initial_candidates) > top_k:
            selected_indices = furthest_point_sampling_batch(
                batch_attention_maps, 
                top_k, 
                top_initial_candidates
            )
        else:
            selected_indices = top_initial_candidates
        
        # Step 3: Apply sharpening loss only to selected diverse tokens
        if len(selected_indices) > 0:
            selected_attention_maps = batch_attention_maps[selected_indices]  # [selected_k, height, width]
            
            # Compute sharpening loss on selected tokens
            batch_loss = sharpening_loss_batch(selected_attention_maps, sigma=sigma)
            total_loss += batch_loss
            valid_batch_count += 1
    
    return total_loss / valid_batch_count if valid_batch_count > 0 else torch.tensor(0.0, device=device)


def find_max_pixel_single(attention_map):
    """
    Find max pixel location for a single attention map.
    
    Args:
        attention_map: [1, height, width] attention map
        
    Returns:
        [1, 2] coordinates of max pixel
    """
    batch_size, h, w = attention_map.shape
    map_reshaped = attention_map.view(batch_size, -1)
    max_indices = torch.argmax(map_reshaped, dim=-1)
    
    max_y = max_indices // w
    max_x = max_indices % w
    
    # Add 0.5 for pixel center and stack to get [batch_size, 2]
    max_coords = torch.stack([max_y, max_x], dim=-1).float() + 0.5
    
    return max_coords


def gaussian_circles_single(pos, size=64, sigma=1.0, device="cuda"):
    """
    Create Gaussian circles for single batch.
    
    Args:
        pos: [1, 2] position (normalized 0-1)
        size: spatial size
        sigma: Gaussian std
        device: device
        
    Returns:
        [size, size] Gaussian circle
    """
    # Scale position to pixel coordinates
    _pos = pos * size  # [1, 2]
    _pos = _pos.unsqueeze(1).unsqueeze(1)  # [1, 1, 1, 2]
    
    # Create coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32),
        torch.arange(size, device=device, dtype=torch.float32),
        indexing='ij'
    )
    grid = torch.stack([y_coords, x_coords], dim=-1) + 0.5  # [size, size, 2]
    grid = grid.unsqueeze(0)  # [1, size, size, 2]
    
    # Compute squared distances
    dist_sq = (grid[..., 0] - _pos[..., 0]) ** 2 + (grid[..., 1] - _pos[..., 1]) ** 2
    
    # Create Gaussian
    gaussian = torch.exp(-dist_sq / (2.0 * sigma**2))  # [1, size, size]
    
    return gaussian.squeeze(0)  # [size, size]

