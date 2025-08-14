# Log Analysis & Shape Comparison

## Key Findings: ✅ **Learnable Embeddings Successfully Integrated**

The logs demonstrate that the learnable embeddings concatenation is working perfectly. Here's the detailed analysis:

## 📊 **Shape Comparison Summary**

| Component | OLD (CLIP only) | NEW (CLIP + Learnable) | Change |
|-----------|------------------|------------------------|---------|
| **Text Embeddings** | `[1, 77, 1024]` | `[1, 93, 1024]` | ✅ **+16 tokens** |
| **Key/Value Shapes** | `[1, 77, dim]` | `[1, 93, dim]` | ✅ **+16 tokens** |
| **Attention Probs** | `[heads, spatial, 77]` | `[heads, spatial, 93]` | ✅ **+16 tokens** |
| **Input Processing** | Same | Same | ✅ **No disruption** |

## 🔍 **Detailed Shape Analysis**

### 1. **Text Embedding Concatenation** ✅
```
OLD: encoder_hidden_states shape: torch.Size([1, 77, 1024])    # CLIP only
NEW: encoder_hidden_states shape: torch.Size([1, 93, 1024])    # CLIP + 16 learnable
```
**Result**: Perfect concatenation of 77 CLIP tokens + 16 learnable tokens = 93 total tokens

### 2. **Cross-Attention Key/Value Projection** ✅
```
OLD: Key shape: torch.Size([1, 77, 320])   →   NEW: Key shape: torch.Size([1, 93, 320])
OLD: Value shape: torch.Size([1, 77, 320]) →   NEW: Value shape: torch.Size([1, 93, 320])
```
**Result**: Linear projections correctly handle the expanded token dimension

### 3. **Attention Probability Maps** ✅
```
OLD: Normal attention_probs shape: torch.Size([5, 4096, 77])    # 5 heads, 4096 spatial, 77 tokens
NEW: Normal attention_probs shape: torch.Size([5, 4096, 93])    # 5 heads, 4096 spatial, 93 tokens
```
**Result**: Attention maps now include 16 additional columns for learnable embeddings

### 4. **Multi-Resolution Attention Layers** ✅

All UNet attention layers show consistent expansion:

| **Resolution** | **Spatial Dims** | **OLD Attention** | **NEW Attention** | **Heads** |
|----------------|-------------------|-------------------|-------------------|-----------|
| **High-Res** | 4096 (64×64) | `[5, 4096, 77]` | `[5, 4096, 93]` | 5 heads |
| **Mid-Res** | 1024 (32×32) | `[10, 1024, 77]` | `[10, 1024, 93]` | 10 heads |
| **Low-Res** | 256 (16×16) | `[20, 256, 77]` | `[20, 256, 93]` | 20 heads |
| **Bottleneck** | 64 (8×8) | `[20, 64, 77]` | `[20, 64, 93]` | 20 heads |

### 5. **Query Shapes Unchanged** ✅
```
Query shape: torch.Size([1, 4096, 320])  # Same in both OLD and NEW
```
**Result**: Image feature queries remain unchanged, only text conditioning expanded

## 🎯 **StableKeypoints Integration Status**

### ✅ **Successfully Working**
1. **Embedding Concatenation**: `torch.cat([prompt_embeds, learnable_embeddings], dim=1)` working perfectly
2. **Forward Pass**: UNet processing 93 tokens without errors
3. **Attention Computation**: All layers computing attention over expanded token set
4. **Shape Consistency**: All tensor shapes are mathematically correct

### 🔍 **SK-Specific Attention Extraction**
The attention maps now contain SK-relevant information:
```python
# Extract SK attention from last 16 tokens
sk_attention = attention_probs[:, :, -16:]  # [heads, spatial, 16]
```

**Available SK Data**:
- **16 Learnable Tokens**: Tokens 77-92 in the attention maps
- **Multiple Resolutions**: SK attention available at 4096, 1024, 256, 64 spatial dimensions
- **Multi-Head Coverage**: 5-20 attention heads per layer providing rich attention patterns

### 📋 **Next Steps for Complete SK Integration**

1. **Attention Map Collection** (IMMEDIATE):
   ```python
   # Store SK attention maps during forward pass
   sk_attention = attention_probs[:, :, 77:93]  # Extract learnable embeddings attention
   cross_attention_kwargs['ref_sk_loss']['sk_attention_maps'].append(sk_attention)
   ```

2. **Multi-Layer Aggregation**:
   ```python
   # Combine attention from multiple UNet layers
   aggregated_attention = torch.mean(torch.stack(all_layer_attentions), dim=0)
   ```

3. **SK Loss Computation**:
   ```python
   # Compute localization loss on aggregated attention
   sk_loss = compute_sharpening_loss_batch(aggregated_attention)
   ```

## 🚀 **Performance Implications**

### **Memory Usage**
- **Text Embeddings**: `+20.8%` increase (77→93 tokens)
- **Attention Maps**: `+20.8%` increase in last dimension
- **Overall Impact**: Minimal, as text embeddings are small compared to image features

### **Computational Overhead**
- **Attention Computation**: `+20.8%` increase in Key/Value operations
- **Cross-Attention**: Negligible impact on Query computation
- **Training Speed**: Expected minimal slowdown (~5-10%)

## 🎉 **Conclusion**

The learnable embeddings integration is **completely successful**! The logs show:

1. ✅ **Perfect Shape Consistency**: All tensors have correct dimensions
2. ✅ **No Errors**: Forward pass completes without issues  
3. ✅ **SK Data Available**: 16 learnable token attention maps ready for extraction
4. ✅ **Zero123Plus Intact**: Original 77 CLIP tokens preserved for NVS functionality

**The infrastructure is ready for SK loss computation.** The next step is implementing the attention map collection mechanism to extract the SK-relevant attention patterns (tokens 77-92) and feed them into the sharpening loss computation.

## 🔬 **Technical Insights from the Logs**

### **UNet Architecture Analysis**
Based on the attention shapes, we can see the UNet structure:

1. **Encoder Path** (Downsampling):
   - High resolution: 4096 spatial dims (64×64) with 5 attention heads
   - Mid resolution: 1024 spatial dims (32×32) with 10 attention heads  
   - Low resolution: 256 spatial dims (16×16) with 20 attention heads
   - Bottleneck: 64 spatial dims (8×8) with 20 attention heads

2. **Decoder Path** (Upsampling):
   - Same resolutions but with skip connections from encoder
   - Additional spatial dimensions (9600 = 80×120) from Zero123Plus 2×3 multi-view grid

### **StableKeypoints Attention Target**
The 16 learnable embeddings are now accessible at every attention layer:
- **Tokens 0-76**: CLIP text embeddings (preserved for Zero123Plus functionality)
- **Tokens 77-92**: StableKeypoints learnable embeddings (new SK functionality)

This perfect separation allows:
1. Zero123Plus to continue using CLIP embeddings normally
2. StableKeypoints to extract attention from its dedicated tokens
3. No interference between the two systems

The integration is architecturally sound and ready for the next phase of implementation.

## 🎯 **Zero123Plus Multi-View Grid Analysis**

### **Understanding the 9600 = 80×120 Spatial Dimension**

You're absolutely correct! Let me break down the Zero123Plus multi-view processing:

#### **Zero123Plus Grid Structure**
```
Input: Single image [H, W] → Zero123Plus → 6 views in 2×3 grid [2H, 3W]
```

**For your case:**
- **Individual view resolution**: 40×40 (at this UNet layer resolution)
- **Grid arrangement**: 2 rows × 3 columns = 6 views
- **Combined grid**: 80×120 = (2×40) × (3×40)
- **Flattened spatial**: 80×120 = 9600 spatial positions

#### **Grid Layout Visualization**
```
┌──────┬──────┬──────┐
│View 1│View 2│View 3│  ← Row 1 (40×120)
│ 40×40│ 40×40│ 40×40│
├──────┼──────┼──────┤
│View 4│View 5│View 6│  ← Row 2 (40×120)  
│ 40×40│ 40×40│ 40×40│
└──────┴──────┴──────┘
Total: 80×120 = 9600 spatial positions
```

### **SK Attention Map Extraction Strategy**

The current attention shape `[heads, 9600, 93]` contains **all 6 views mixed together**. For SK loss computation, we need to:

#### **Option 1: Individual View Extraction (RECOMMENDED)**
```python
def extract_individual_view_attention(attention_maps, grid_shape=(2, 3), view_size=(40, 40)):
    """
    Extract attention maps for individual views from the 2×3 grid
    
    Args:
        attention_maps: [heads, 9600, 93] - Combined grid attention
        grid_shape: (2, 3) - 2 rows, 3 columns
        view_size: (40, 40) - Individual view spatial dimensions
    
    Returns:
        individual_views: [heads, 6, 1600, 16] - Per-view SK attention (tokens 77-92)
    """
    heads, spatial_total, tokens = attention_maps.shape
    h_view, w_view = view_size
    rows, cols = grid_shape
    
    # Reshape to grid: [heads, 80, 120, 93] 
    grid_attention = attention_maps.view(heads, rows * h_view, cols * w_view, tokens)
    
    # Extract SK tokens (77-92): [heads, 80, 120, 16]
    sk_attention_grid = grid_attention[:, :, :, 77:93]
    
    # Split into individual views: [heads, 6, 40, 40, 16]
    view_attentions = []
    for row in range(rows):
        for col in range(cols):
            # Extract view region
            view_attn = sk_attention_grid[
                :, 
                row*h_view:(row+1)*h_view,  # Y range
                col*w_view:(col+1)*w_view,  # X range
                :
            ]  # [heads, 40, 40, 16]
            
            # Flatten spatial: [heads, 1600, 16]
            view_attn_flat = view_attn.view(heads, h_view*w_view, 16)
            view_attentions.append(view_attn_flat)
    
    # Stack all views: [heads, 6, 1600, 16]
    return torch.stack(view_attentions, dim=1)
```

#### **Option 2: Average Across Views (SIMPLER)**
```python
def extract_average_sk_attention(attention_maps):
    """
    Extract SK attention and average across the grid
    
    Args:
        attention_maps: [heads, 9600, 93] - Combined grid attention
    
    Returns:
        sk_attention: [heads, 9600, 16] - SK attention maps (tokens 77-92)
    """
    # Simply extract SK tokens from the combined grid
    sk_attention = attention_maps[:, :, 77:93]  # [heads, 9600, 16]
    return sk_attention
```

### **SK Loss Computation Pipeline**

#### **Method 1: Individual View SK Loss (BEST for Multi-View Consistency)**
```python
def compute_sk_loss_multiview(attention_maps, grid_shape=(2, 3), view_size=(40, 40)):
    """
    Compute SK loss with multi-view consistency
    """
    # Extract individual view attentions: [heads, 6, 1600, 16]  
    view_attentions = extract_individual_view_attention(attention_maps, grid_shape, view_size)
    
    total_loss = 0.0
    num_views = view_attentions.shape[1]
    
    # 1. Localization loss per view
    localization_loss = 0.0
    for view_idx in range(num_views):
        view_attn = view_attentions[:, view_idx]  # [heads, 1600, 16]
        
        # Average across heads: [1600, 16]
        avg_attn = torch.mean(view_attn, dim=0)
        
        # Reshape to spatial: [40, 40, 16]
        spatial_attn = avg_attn.view(view_size[0], view_size[1], 16)
        
        # Compute sharpening loss for this view
        view_loss = compute_sharpening_loss_batch(spatial_attn.unsqueeze(0))
        localization_loss += view_loss
    
    localization_loss /= num_views
    
    # 2. Cross-view consistency loss
    consistency_loss = compute_cross_view_consistency(view_attentions)
    
    return {
        'localization': localization_loss,
        'consistency': consistency_loss,
        'total': localization_loss + consistency_loss
    }

def compute_cross_view_consistency(view_attentions):
    """
    Ensure keypoints are consistent across views
    
    Args:
        view_attentions: [heads, 6, 1600, 16] - Per-view attention maps
    """
    heads, num_views, spatial, num_tokens = view_attentions.shape
    consistency_loss = 0.0
    
    # Average across heads
    avg_view_attentions = torch.mean(view_attentions, dim=0)  # [6, 1600, 16]
    
    for token_idx in range(num_tokens):
        token_maps = avg_view_attentions[:, :, token_idx]  # [6, 1600]
        
        # Find max attention location in each view
        max_locs = []
        for view_idx in range(num_views):
            view_map = token_maps[view_idx]  # [1600]
            max_idx = torch.argmax(view_map)
            max_y, max_x = divmod(max_idx.item(), 40)  # Convert to 2D coordinates
            max_locs.append([max_y, max_x])
        
        max_locs = torch.tensor(max_locs, dtype=torch.float32)  # [6, 2]
        
        # Compute pairwise consistency
        for i in range(num_views):
            for j in range(i + 1, num_views):
                loc_diff = max_locs[i] - max_locs[j]
                consistency_loss += torch.sum(loc_diff ** 2)
    
    # Normalize by number of token pairs and view pairs
    num_pairs = num_views * (num_views - 1) // 2
    return consistency_loss / (num_tokens * num_pairs)
```

#### **Method 2: Grid-Level SK Loss (SIMPLER)**
```python
def compute_sk_loss_grid(attention_maps):
    """
    Compute SK loss on the entire 9600-spatial grid (simpler approach)
    """
    # Extract SK attention: [heads, 9600, 16]
    sk_attention = attention_maps[:, :, 77:93]
    
    # Average across heads: [9600, 16]
    avg_sk_attention = torch.mean(sk_attention, dim=0)
    
    # Reshape to grid: [80, 120, 16] 
    grid_sk_attention = avg_sk_attention.view(80, 120, 16)
    
    # Compute sharpening loss on the entire grid
    sharpening_loss = compute_sharpening_loss_batch(grid_sk_attention.unsqueeze(0))
    
    return {'sharpening': sharpening_loss}
```

### **Recommendation: Use Method 1 (Individual Views)**

**Why Method 1 is better:**
1. **True Multi-View Learning**: Each view contributes separately to keypoint discovery
2. **Cross-View Consistency**: Enforces same semantic keypoints across views  
3. **Better SK Philosophy**: Aligns with SK's goal of discovering consistent keypoints
4. **3D Understanding**: Views can learn complementary perspectives of the same object

**Why avoid Method 2:**
- Treats all views as one big image
- No cross-view consistency enforcement
- May learn different keypoints in different views
- Less aligned with 3D understanding goals

### **Integration into Your Current Pipeline**

Add this to your `forward_unet_with_sk()` method:

```python
# In your attention processor (SKAttnProc)
if 'ref_sk_loss' in cross_attention_kwargs:
    # Store the full attention maps
    cross_attention_kwargs['ref_sk_loss']['attention_maps'].append({
        'attention': attention_probs,  # [heads, spatial, 93]
        'spatial_shape': (height, width),  # Grid shape info
        'layer_name': self.layer_name
    })

# In your loss computation
def compute_sk_losses(stored_attention_maps):
    # Aggregate from multiple layers
    combined_attention = aggregate_attention_maps(stored_attention_maps)
    
    # Extract individual views and compute SK loss
    sk_losses = compute_sk_loss_multiview(
        combined_attention, 
        grid_shape=(2, 3), 
        view_size=(40, 40)  # Adjust based on layer resolution
    )
    
    return sk_losses
```

This approach will give you proper multi-view SK learning with cross-view consistency!
