# StableKeypoints Implementation Analysis

## 🔍 **How StableKeypoints Extracts Attention Maps**

Based on the StableKeypoints codebase analysis, here's how they handle attention extraction and keypoint diversity:

## 📋 **Key Components Overview**

### 1. **Attention Control System** (`models/attention_control.py`)

```python
# AttentionStore class collects attention maps during forward pass
class AttentionStore(AttentionControl):
    def forward(self, dict, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store["attn"].append(dict['attn'])  # ← ATTENTION STORAGE
        return dict
```

**Key Insight**: They use **hooks** in the CrossAttention layers to automatically collect attention maps during the forward pass.

### 2. **Attention Collection Pipeline** (`utils/image_utils.py`)

```python
def run_and_find_attn(ldm, image, context, ...):
    # 1. Run forward pass with noise injection
    find_pred_noise(ldm, image, context, noise_level=-1)
    
    # 2. Collect attention maps from registered controllers
    attention_maps = collect_maps(controllers[controller], 
                                 from_where=["down_cross", "mid_cross", "up_cross"],
                                 layers=[0,1,2,3,4,5])
    return attention_maps

def collect_maps(controller, layers, upsample_res=512):
    attention_maps = controller.step_store['attn']  # ← GET STORED ATTENTION
    
    # Process each layer
    for layer in layers:
        data = attention_maps[layer]  # [batch, spatial, tokens]
        
        # Reshape to spatial format
        data = data.reshape(batch, sqrt(spatial), sqrt(spatial), tokens)
        
        # Upsample to target resolution
        data = F.interpolate(data, size=(upsample_res, upsample_res))
        
    # Average across layers and batch
    result = torch.mean(torch.stack(processed_maps))
    return result  # [tokens, height, width]
```

## 🎯 **Keypoint Diversity Enforcement**

### 1. **Furthest Point Sampling** (`utils/keypoint_utils.py`)

```python
def furthest_point_sampling(attention_maps, top_k, top_initial_candidates):
    """Forces keypoints to be spatially separated"""
    
    # Find pixel locations for all candidate tokens
    max_pixel_locations = find_max_pixel(attention_maps)
    
    # Start with the two furthest points
    max_dist = -1
    for i, j in combinations(top_initial_candidates):
        dist = torch.sqrt(torch.sum((locations[i] - locations[j])**2))
        if dist > max_dist:
            furthest_pair = (i, j)
    
    selected_indices = [furthest_pair[0], furthest_pair[1]]
    
    # Iteratively add points that are furthest from already selected ones
    for _ in range(top_k - 2):
        max_min_dist = -1
        for candidate in top_initial_candidates:
            # Find minimum distance to any selected point
            min_dist_to_selected = min(distance(candidate, selected) 
                                     for selected in selected_indices)
            # Choose candidate with maximum minimum distance
            if min_dist_to_selected > max_min_dist:
                furthest_point = candidate
        selected_indices.append(furthest_point)
    
    return selected_indices
```

**Key Insight**: They use **furthest point sampling** to ensure keypoints are **spatially diverse**.

### 2. **Gaussian Ranking for Initial Selection** (`utils/keypoint_utils.py`)

```python
def find_top_k_gaussian(attention_maps, top_k, sigma=3):
    """Selects tokens that produce most Gaussian-like attention"""
    
    # Find max pixel location for each token
    max_pixel_locations = find_k_max_pixels(attention_maps, num=num_subjects)
    
    # Create target Gaussian at max location
    target = gaussian_circles(max_pixel_locations, size=image_h, sigma=sigma)
    
    # Compute KL divergence between attention and target Gaussian
    attention_softmax = F.softmax(attention_maps.flatten())
    target_softmax = target.flatten() / target.sum()
    
    kl_distances = torch.sum(target_softmax * 
                            (torch.log(target_softmax) - torch.log(attention_softmax)))
    
    # Return tokens with lowest KL divergence (most Gaussian-like)
    return torch.argsort(kl_distances)[:top_k]
```

**Key Insight**: They first filter for tokens that produce **Gaussian-like attention patterns**, then apply spatial diversity.

### 3. **Mask Radius Function** (`utils/keypoint_utils.py`)

```python
def mask_radius(map, max_coords, radius):
    """Masks attention within radius of already found keypoints"""
    
    # Create distance map from max_coords
    squared_dist = (x_coords - max_coords[:, 1])**2 + (y_coords - max_coords[:, 0])**2
    
    # Mask out pixels within radius
    mask = squared_dist > radius**2
    masked_map = map * mask.float()
    
    return masked_map
```

**Key Insight**: They **mask** already discovered keypoint locations to force subsequent keypoints to be in different spatial regions.

## 🏗️ **Training Loop Architecture** (`optimization/optimizer.py`)

### 1. **Loss Computation Strategy**

```python
# For each training step:
for iteration in range(num_steps):
    # 1. Get attention maps for current image
    attn_maps = run_and_find_attn(ldm, image, context)
    
    # 2. Get attention maps for transformed image  
    attn_maps_transformed = run_and_find_attn(ldm, transformed_image, context)
    
    # 3. Select diverse keypoint tokens
    top_indices = find_top_k_gaussian(attn_maps, furthest_point_num_samples)
    selected_indices = furthest_point_sampling(attn_maps_transformed, top_k, top_indices)
    
    # 4. Compute losses only on selected tokens
    sharpening = sharpening_loss(attn_maps[selected_indices])
    equivariance = equivariance_loss(attn_maps[selected_indices], 
                                   attn_maps_transformed[selected_indices])
    
    total_loss = sharpening * weight1 + equivariance * weight2
```

### 2. **Key Diversity Mechanisms**

1. **Token Selection**: Only optimize a subset of tokens (top_k out of num_tokens)
2. **Gaussian Filtering**: Pre-filter tokens that naturally produce focused attention
3. **Spatial Separation**: Ensure selected tokens focus on different spatial regions
4. **Masking**: Prevent multiple tokens from attending to the same location

## 🔧 **Adaptation for Your Zero123Plus Integration**

### **Current StableKeypoints Limitations for Multi-View:**
1. **Single Image Focus**: Designed for individual images, not multi-view grids
2. **No Cross-View Consistency**: No mechanism to ensure same keypoints across views
3. **Spatial Masking**: Uses 2D spatial masking, not suitable for 3D consistency

### **Your Enhanced Approach Should:**

#### 1. **Multi-View Token Selection**
```python
def select_sk_tokens_multiview(view_attentions, num_tokens=16):
    """
    Select SK tokens that are consistent across all 6 views
    
    Args:
        view_attentions: [6, spatial, num_total_tokens] - attention per view
        num_tokens: number of SK tokens to select
    """
    # 1. For each token, compute Gaussian-likeness across all views
    gaussian_scores = []
    for token_idx in range(num_total_tokens):
        view_scores = []
        for view_idx in range(6):
            score = compute_gaussian_score(view_attentions[view_idx, :, token_idx])
            view_scores.append(score)
        # Use minimum score across views (consistent Gaussian-ness)
        gaussian_scores.append(min(view_scores))
    
    # 2. Pre-filter tokens with good Gaussian scores
    gaussian_candidates = torch.argsort(gaussian_scores)[:num_tokens*3]
    
    # 3. Apply cross-view spatial diversity
    selected_tokens = furthest_point_sampling_multiview(
        view_attentions[:, :, gaussian_candidates], 
        num_tokens
    )
    
    return selected_tokens

def furthest_point_sampling_multiview(view_attentions, num_tokens):
    """Multi-view spatial diversity ensuring same semantic regions across views"""
    # Find keypoint locations in each view for each token
    token_locations = {}  # {token_idx: [6 x 2D_locations]}
    
    for token_idx in range(view_attentions.shape[2]):
        locations = []
        for view_idx in range(6):
            loc = find_max_pixel(view_attentions[view_idx, :, token_idx])
            locations.append(loc)
        token_locations[token_idx] = locations
    
    # Select tokens with maximum cross-view spatial diversity
    selected = []
    for _ in range(num_tokens):
        best_token = None
        max_min_dist = 0
        
        for candidate in token_locations:
            if candidate in selected:
                continue
                
            # Compute minimum distance to already selected tokens
            min_dist = float('inf')
            for selected_token in selected:
                # Average distance across all views
                view_distances = []
                for view_idx in range(6):
                    d = distance(token_locations[candidate][view_idx], 
                               token_locations[selected_token][view_idx])
                    view_distances.append(d)
                avg_dist = sum(view_distances) / len(view_distances)
                min_dist = min(min_dist, avg_dist)
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_token = candidate
        
        if best_token is not None:
            selected.append(best_token)
    
    return selected
```

#### 2. **Cross-View Consistency Loss**
```python
def cross_view_consistency_loss(view_attentions, selected_tokens):
    """Ensure selected tokens focus on same semantic regions across views"""
    consistency_loss = 0.0
    
    for token_idx in selected_tokens:
        # Get attention maps for this token across all views
        token_maps = view_attentions[:, :, token_idx]  # [6, spatial]
        
        # Find keypoint locations in each view
        keypoint_locs = []
        for view_idx in range(6):
            loc = find_max_pixel(token_maps[view_idx])
            keypoint_locs.append(loc)
        
        # Compute pairwise consistency (can be enhanced with camera geometry)
        for i in range(6):
            for j in range(i+1, 6):
                loc_diff = keypoint_locs[i] - keypoint_locs[j] 
                consistency_loss += torch.sum(loc_diff ** 2)
    
    return consistency_loss / (len(selected_tokens) * 15)  # 15 = 6*5/2 pairs
```

#### 3. **Integration with Your Current System**
```python
# In your SKAttnProc:
if 'ref_sk_loss' in cross_attention_kwargs:
    # Extract multi-view attention using your new module
    from src.utils.attention_extraction import extract_sk_attention_auto_dimensions
    sk_data = extract_sk_attention_auto_dimensions(attention_probs)
    
    # Apply StableKeypoints-inspired token selection
    selected_tokens = select_sk_tokens_multiview(
        sk_data['individual_views'],  # [6, spatial, 16]
        num_tokens=16
    )
    
    # Store for loss computation with diversity info
    cross_attention_kwargs['ref_sk_loss']['sk_attention_maps'].append({
        'individual_views': sk_data['individual_views'],
        'selected_tokens': selected_tokens,  # Indices of diverse tokens
        'view_shape': sk_data['view_shape'],
        'layer_name': self.layer_name
    })
```

## 🎯 **Key Takeaways for Your Implementation**

1. **Fixed vs Dynamic Selection**: StableKeypoints dynamically selects which tokens to optimize. You have **fixed 16 learnable tokens**, so you need to ensure they **learn diverse patterns**.

2. **Diversity Through Loss Design**: Instead of token selection, enforce diversity through **multi-view consistency losses** and **spatial separation penalties**.

3. **Hook-Based Collection**: Use their attention hook system as inspiration for your `SKAttnProc` implementation.

4. **Gaussian Target Loss**: Their sharpening loss is essentially **MSE between attention and target Gaussian** - very similar to your approach.

5. **Multi-Resolution Processing**: They collect from multiple UNet layers and average - you should do the same.

Your approach is actually **more sophisticated** because you're enforcing **3D geometric consistency** across multiple views, which StableKeypoints doesn't handle!
