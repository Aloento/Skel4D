# Unsupervised Keypoints from Pretrained Diffusion Models - Detailed Summary

## Paper Information
- **Title**: Unsupervised Keypoints from Pretrained Diffusion Models
- **Authors**: Eric Hedlin¹, Gopal Sharma¹, Shweta Mahajan¹,², Xingzhe He¹, Hossam Isack³, Abhishek Kar³, Helge Rhodin¹, Andrea Tagliasacchi⁴,⁵,⁶, Kwang Moo Yi¹
- **Affiliations**: 
  - ¹University of British Columbia
  - ²Vector Institute for AI
  - ³Google Research
  - ⁴Google DeepMind
  - ⁵Simon Fraser University
  - ⁶University of Toronto
- **Publication**: CVPR 2024 Highlight
- **ArXiv**: https://arxiv.org/abs/2312.00065
- **GitHub**: https://github.com/ubc-vision/StableKeypoints
- **Project Page**: https://stablekeypoints.github.io/

## Abstract & Core Contribution

The paper introduces a novel approach for unsupervised keypoint learning by leveraging pretrained text-to-image diffusion models (specifically Stable Diffusion). The key insight is that random text embeddings already respond consistently to semantically similar regions across images. By optimizing these text embeddings to create localized attention maps, the method can discover meaningful keypoints without any supervision.

### Key Innovation
- **No supervision required**: No ground truth keypoints, no manual annotations
- **Leverages pretrained knowledge**: Utilizes the semantic understanding embedded in large-scale diffusion models
- **Cross-attention optimization**: Optimizes text embeddings to create Gaussian-like attention patterns
- **Superior performance**: Outperforms state-of-the-art unsupervised methods, especially on unaligned data

## Technical Methodology

### 1. Attention Maps in Diffusion Networks

The method builds on latent diffusion models (LDMs) that operate in a latent space:
- **Forward process**: Gradually adds noise to latent representation z over T timesteps
- **Reverse process**: Denoises latents to recover original signal
- **Conditional generation**: Text embedding e = τθ(y) conditions the denoising process

**Cross-attention mechanism**:
- Query: Q^c_l = Φ^c_l(z_t=1) ∈ ℝ^(H×W)×D_l
- Key: K^c_l = Ψ^c_l(e) ∈ ℝ^N×D_l
- Attention map: M_l(e,X) = 𝔼_c[softmax_n(Q^c_l·K_l/√D_l)]

The final attention map aggregates across multiple layers (7-10):
M̃ = 𝔼_{l=7..10}[M_l(e,X)] ∈ ℝ^(H×W)×N

### 2. Optimization Objective

The method optimizes text embeddings using two complementary losses:

**Total Loss**:
L_total = L_localize + λ_equiv L_equiv

Where λ_equiv = 10 to balance the losses.

#### Localization Loss (L_localize)
Forces attention maps to become single-mode Gaussian distributions:

L_localize = 𝔼_n ||M̃_n - G_n||²

Where:
- μ_n = argmax_{w,h} M̃_n[h,w] (location of maximum response)
- G_n = exp(-||XY_coord - μ_n||²₂ / 2σ²) (Gaussian centered at maximum)

#### Equivariance Loss (L_equiv)
Ensures consistency across geometric transformations:

L_equiv = 𝔼_n ||T^(-1)(M_n(e,T(X))) - M_n(e,X)||²

Where T represents minor affine transformations:
- Rotations: ±15 degrees
- Translations: ±0.25×W
- Scaling: 100-120% of original size

### 3. Key Technical Features

#### Mutual Exclusivity
The softmax operation in cross-attention naturally prevents multiple tokens from attending to the same location. If embeddings become similar, their attention responses become flat (non-Gaussian), violating the localization constraint.

#### Stabilized Optimization
- Works with top-κ most localized tokens to handle spread attention maps
- Uses KL divergence to measure localization: N(κ) = ArgTop_κ{-KL(G_n, M̃_n)}
- Applies furthest point sampling for final keypoint selection

#### Test-time Enhancement
- Averages attention maps across multiple augmentations
- Upsamples low-resolution attention maps (16×16 or 32×32) to 128×128 using bicubic interpolation

## Implementation Details

### Architecture Components

The implementation consists of several key modules:

1. **Main Pipeline** (`main.py`):
   - Argument parsing and configuration
   - Dataset loading and preparation
   - Model initialization
   - Optimization orchestration
   - Evaluation and visualization

2. **Optimization Engine** (`optimize.py`):
   - Core embedding optimization logic
   - Loss computation functions
   - Attention map extraction
   - Augmentation handling

3. **Key Functions**:
   - `collect_maps()`: Extracts and processes attention maps
   - `gaussian_loss()`: Implements localization loss
   - `equivariance_loss()`: Implements transformation consistency
   - `sharpening_loss()`: Encourages localized attention
   - `optimize_embedding()`: Main optimization loop

### Training Process

1. **Initialization**: Random text embeddings (500 tokens by default)
2. **Iterative Optimization**:
   - Sample batch of images
   - Generate augmented versions
   - Extract attention maps for both original and augmented images
   - Compute localization and equivariance losses
   - Update embeddings via gradient descent
3. **Post-processing**:
   - Select top-K most consistent tokens
   - Apply furthest point sampling for final keypoints

### Hyperparameters

- **Learning rate**: 5e-3
- **Optimization steps**: 500 (human pose) to 10,000 (other datasets)
- **Batch size**: 4
- **Number of tokens**: 500
- **Upsampling resolution**: 128×128
- **Gaussian sigma**: 1.0
- **Temperature**: 1e-4

## Experimental Results

### Datasets Evaluated

1. **CelebA**: Facial images (aligned and wild variants)
2. **CUB-200-2011**: Bird images (aligned and unaligned subsets)
3. **Tai-Chi-HD**: Human pose in complex movements
4. **DeepFashion**: Fashion model images
5. **Human3.6M**: Human pose dataset

### Key Performance Metrics

- **CelebA**: Normalized ℓ₂ error by inter-ocular distance
- **CUB**: Mean ℓ₂ error normalized by image dimensions
- **Tai-Chi-HD**: Accumulated ℓ₂ error
- **DeepFashion**: Percentage of Correct Keypoints (PCK) with 6-pixel threshold
- **Human3.6M**: ℓ₂ error normalized to 128×128 resolution

### Performance Highlights

#### Superior Performance on Unaligned Data
- **CelebA Wild (K=8)**: 4.35 vs 5.24 (previous best)
- **CUB-all**: 5.4 vs 11.3 (previous best) - ~52% improvement
- **Tai-Chi-HD**: 234.89/7.02 vs 316.10/9.45 (previous best)

#### Competitive on Aligned Data
- **CelebA Aligned**: 3.60 vs 3.19 (supervised methods)
- **CUB-aligned**: 5.06 vs 3.51 (specialized methods)

#### Remarkable Small Dataset Performance
- **CUB subsets (30 images each)**: Achieves ~10.5 error vs ~20+ for baselines
- **CelebA (100 images)**: 5.33 performance comparable to full dataset methods

### Generalization Capabilities

The learned text embeddings demonstrate remarkable cross-domain generalization:

1. **Tai-Chi-HD → Human3.6M**: Achieves state-of-the-art performance (4.88 vs 16.92)
2. **CUB → Tai-Chi-HD**: Competitive performance despite domain gap (317.94 vs 535.61)
3. **Cross-species generalization**: CUB tokens respond to human heads when applied to human datasets

## Technical Advantages

### 1. No Dataset-Specific Training
Unlike traditional methods that require training from scratch on each dataset, this approach leverages pretrained diffusion model knowledge.

### 2. Robust to Data Quality
- Excels on unaligned, uncropped, and "in-the-wild" data
- Handles occlusions and appearance variations well
- No requirement for object-centric preprocessing

### 3. Semantic Consistency
- Learned keypoints carry semantic meaning across domains
- Tokens can generalize to related but different object categories
- Maintains spatial relationships and anatomical consistency

### 4. Computational Efficiency
- No need for GAN training or autoencoder architectures
- Optimization converges in reasonable time (2 hours on RTX 3090)
- Efficient inference through pretrained models

## Limitations and Considerations

### 1. Dependence on Pretrained Models
- Requires access to large-scale pretrained diffusion models
- Performance bounded by the semantic understanding of the base model

### 2. Limited Control Over Keypoint Semantics
- Cannot specify which anatomical parts should be keypoints
- Final keypoint selection depends on what the model finds "interesting"

### 3. Resolution Constraints
- Attention maps are relatively low resolution (16×16 to 32×32)
- Requires upsampling which may introduce artifacts

### 4. Computational Requirements
- Needs GPU memory for diffusion model inference
- Optimization process requires multiple forward passes

## Comparison with Related Work

### Traditional Unsupervised Methods
- **Autoencoders**: Require training from scratch, struggle with backgrounds
- **GAN-based approaches**: Training instability, domain-specific
- **Transformation-based methods**: Limited by known geometric priors

### Diffusion-Based Approaches
- **Previous work**: Focused on correspondence between image pairs
- **This work**: Discovers correspondences across entire datasets without queries

### Supervised Methods
- **Advantages of supervised**: Direct optimization for specific keypoints
- **Limitations**: Require extensive labeled data, poor generalization
- **This work**: Sometimes outperforms supervised methods on challenging data

## Usage and Command Line Interface

### Basic Usage Structure
```bash
python3 -m unsupervised_keypoints.main [arguments]
```

### Key Command Line Arguments
```python
# From main.py argument parser
parser.add_argument("--model_type", type=str, default="sd-legacy/stable-diffusion-v1-5", 
                   help="ldm model type")
parser.add_argument("--my_token", type=str, required=True, 
                   help="Hugging Face token for model download")
parser.add_argument("--dataset_loc", type=str, default="~", help="Path to dataset")
parser.add_argument("--dataset_name", 
                   choices=["celeba_aligned", "celeba_wild", "cub_aligned", "cub_001", 
                           "cub_002", "cub_003", "cub_all", "deepfashion", "taichi", 
                           "human3.6m", "unaligned_human3.6m", "custom"],
                   type=str, default="celeba_aligned", help="name of the dataset to use")
parser.add_argument("--num_steps", type=int, default=500, 
                   help="number of steps to optimize for")
parser.add_argument("--num_tokens", type=int, default=500, 
                   help="number of tokens to optimize")
parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batch for optimization")
parser.add_argument("--evaluation_method", 
                   choices=["inter_eye_distance", "visible", "mean_average_error", 
                           "pck", "orientation_invariant"],
                   help="evaluation method")
```

### Example Usage Commands
```bash
# CelebA Wild dataset
python3 -m unsupervised_keypoints.main \
    --dataset_loc /path/to/celeba \
    --dataset_name celeba_wild \
    --evaluation_method inter_eye_distance \
    --my_token YOUR_HF_TOKEN \
    --num_steps 10000 \
    --save_folder outputs/celeba_wild

# CUB-200-2011 dataset
python3 -m unsupervised_keypoints.main \
    --dataset_loc /path/to/cub \
    --dataset_name cub_all \
    --evaluation_method visible \
    --my_token YOUR_HF_TOKEN \
    --num_steps 10000

# Custom dataset (visualization only)
python3 -m unsupervised_keypoints.main \
    --dataset_loc /path/to/custom/images \
    --dataset_name custom \
    --my_token YOUR_HF_TOKEN \
    --visualize
```

## Implementation Architecture

### Core Components

```python
# Main optimization loop structure
def optimize_embedding(ldm, args, controllers, num_gpus):
    # Initialize random embeddings
    context = init_random_noise(device, num_words=num_tokens)
    
    # Setup optimizer
    optimizer = torch.optim.Adam([context], lr=args.lr)
    
    for iteration in range(num_steps):
        # Sample batch and augment
        image = sample_batch()
        transformed_img = apply_augmentation(image)
        
        # Extract attention maps
        attn_maps = extract_attention(ldm, image, context)
        attn_maps_transformed = extract_attention(ldm, transformed_img, context)
        
        # Compute losses
        localization_loss = compute_gaussian_loss(attn_maps)
        equivariance_loss = compute_equivariance(attn_maps, attn_maps_transformed)
        
        # Optimize
        total_loss = localization_loss + lambda * equivariance_loss
        total_loss.backward()
        optimizer.step()
```

### Pipeline Execution Workflow
```python
# From main.py - Complete pipeline execution
def main():
    args = parser.parse_args()
    
    # Stage 1: Load pretrained diffusion model
    ldm, controllers, num_gpus = load_ldm(args.model_type, args.my_token, args.device)
    
    # Stage 2: Optimize text embeddings to find keypoints
    embedding = optimize_embedding(ldm, args, controllers, num_gpus)
    
    # Save optimized embeddings
    torch.save(embedding, os.path.join(args.save_folder, "embedding.pt"))
    
    # Stage 3: Find best token indices using furthest point sampling
    indices = find_best_indices(ldm, embedding, args, controllers, num_gpus)
    torch.save(indices, os.path.join(args.save_folder, "indices.pt"))
    
    # Visualization stage (always runs)
    if args.visualize:
        visualize_attn_maps(ldm, embedding, indices, args, controllers, num_gpus)
    
    # Skip remaining stages for custom datasets
    if args.dataset_name == "custom":
        print("Dataset is 'custom'. Skipping precomputation, regressor training, and evaluation.")
        return
    
    # Stage 4: Precompute keypoints for all images
    source_kpts, target_kpts, visible = precompute_all_keypoints(
        ldm, embedding, indices, args, controllers, num_gpus
    )
    
    # Stage 5: Train linear regressor to map discovered keypoints to ground truth
    if args.evaluation_method == "visible" or args.evaluation_method == "mean_average_error":
        visible_reshaped = visible.unsqueeze(-1).repeat(1, 1, 2).reshape(
            visible.shape[0], visible.shape[1] * 2
        ).cpu().numpy().astype(np.float64)
        
        regressor = return_regressor_visible(
            source_kpts.cpu().numpy().reshape(source_kpts.shape[0], source_kpts.shape[1] * 2).astype(np.float64),
            target_kpts.cpu().numpy().reshape(target_kpts.shape[0], target_kpts.shape[1] * 2).astype(np.float64),
            visible_reshaped,
        )
    elif args.evaluation_method == "orientation_invariant":
        regressor = return_regressor_human36m(
            source_kpts.cpu().numpy().reshape(source_kpts.shape[0], source_kpts.shape[1] * 2).astype(np.float64),
            target_kpts.cpu().numpy().reshape(target_kpts.shape[0], target_kpts.shape[1] * 2).astype(np.float64),
        )
    else:
        regressor = return_regressor(
            source_kpts.cpu().numpy().reshape(source_kpts.shape[0], source_kpts.shape[1] * 2).astype(np.float64),
            target_kpts.cpu().numpy().reshape(target_kpts.shape[0], target_kpts.shape[1] * 2).astype(np.float64),
        )
    
    # Stage 6: Final evaluation
    evaluate(ldm, embedding, indices, regressor.to(args.device), args, controllers, num_gpus)
```

### Key Data Structures

- **Attention Maps**: Shape (T, H, W) where T is number of tokens
- **Text Embeddings**: Learnable parameters of shape (num_tokens, embedding_dim)
- **Gaussian Targets**: Reference distributions for localization loss

### Core Implementation Snippets

#### 1. Attention Map Extraction
```python
def collect_maps(
    controller,
    from_where=["up_cross"],
    upsample_res=512,
    layers=[0, 1, 2, 3],
    indices=None,
):
    """
    Returns the bilinearly upsampled attention map of size upsample_res x upsample_res 
    for the first word in the prompt
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
            # Bilinearly upsample the image to attn_size x attn_size
            data = F.interpolate(
                data,
                size=(upsample_res, upsample_res),
                mode="bilinear",
                align_corners=False,
            )
        attention_maps_list.append(data)
    
    attention_maps_list = torch.stack(attention_maps_list, dim=0).mean(dim=(0, 1))
    controller.reset()
    return attention_maps_list
```

#### 2. Gaussian Target Creation
```python
def create_gaussian_kernel(size: int, sigma: float):
    """
    Create a 2D Gaussian kernel of given size and sigma.
    """
    assert size % 2 == 1, "Size must be odd"
    center = size // 2
    x = torch.arange(0, size, dtype=torch.float32)
    y = torch.arange(0, size, dtype=torch.float32)
    x, y = torch.meshgrid(x - center, y - center)
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel
```

#### 3. Localization Loss (Sharpening Loss)
```python
def sharpening_loss(attn_map, sigma=1.0, temperature=1e1, device="cuda", num_subjects=1):
    """
    Forces attention maps to become localized Gaussian distributions
    """
    pos = eval.find_k_max_pixels(attn_map, num=num_subjects) / attn_map.shape[-1]
    loss = find_gaussian_loss_at_point(
        attn_map,
        pos,
        sigma=sigma,
        temperature=temperature,
        device=device,
        num_subjects=num_subjects,
    )
    return loss

def find_gaussian_loss_at_point(
    attn_map, pos, sigma=1.0, temperature=1e-1, device="cuda", indices=None, num_subjects=1
):
    """
    pos is a location between 0 and 1
    """
    T, H, W = attn_map.shape
    
    # Create Gaussian circle at the given position
    target = optimize_token.gaussian_circles(
        pos, size=H, sigma=sigma, device=attn_map.device
    )
    target = target.to(attn_map.device)
    
    # Possibly select a subset of indices
    if indices is not None:
        attn_map = attn_map[indices]
        target = target[indices]
    
    # Compute loss
    loss = F.mse_loss(attn_map, target)
    return loss
```

#### 4. Equivariance Loss
```python
def equivariance_loss(embeddings_initial, embeddings_transformed, transform, index):
    """
    Ensures attention maps remain consistent across geometric transformations
    """
    # Untransform the embeddings_transformed
    embeddings_initial_prime = transform.inverse(embeddings_transformed)[index]
    loss = F.mse_loss(embeddings_initial, embeddings_initial_prime)
    return loss
```

#### 5. Main Optimization Loop
```python
def optimize_embedding(ldm, args, controllers, num_gpus, context=None):
    # Initialize random text embeddings
    if context is None:
        context = ptp_utils.init_random_noise(args.device, num_words=args.num_tokens)
        context.requires_grad = True
    
    # Setup optimizer
    optimizer = torch.optim.Adam([context], lr=args.lr)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_gpus, shuffle=True, drop_last=True)
    dataloader_iter = iter(dataloader)
    
    for iteration in tqdm(range(int(int(args.num_steps) * (args.batch_size // num_gpus)))):
        try:
            mini_batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            mini_batch = next(dataloader_iter)
        
        image = mini_batch["img"]
        
        # Extract attention maps for original images
        attn_maps = ptp_utils.run_and_find_attn(
            ldm, image, context,
            layers=args.layers,
            noise_level=args.noise_level,
            device=args.device,
            controllers=controllers,
        )
        
        # Apply transformations and extract attention maps
        transformed_img = invertible_transform(image)
        attention_maps_transformed = ptp_utils.run_and_find_attn(
            ldm, transformed_img, context,
            layers=args.layers,
            noise_level=args.noise_level,
            device=args.device,
            controllers=controllers,
        )
        
        # Compute losses for each GPU/image
        _sharpening_loss = []
        _loss_equivariance_attn = []
        
        for index, attn_map, attention_map_transformed in zip(
            torch.arange(num_gpus), attn_maps, attention_maps_transformed
        ):
            # Select top-k most localized tokens
            if args.top_k_strategy == "gaussian":
                top_embedding_indices = ptp_utils.find_top_k_gaussian(
                    attn_map, args.furthest_point_num_samples, 
                    sigma=args.sigma, num_subjects=args.num_subjects
                )
            
            # Apply furthest point sampling
            top_embedding_indices = ptp_utils.furthest_point_sampling(
                attention_map_transformed, args.top_k, top_embedding_indices
            )
            
            # Compute losses
            _sharpening_loss.append(
                sharpening_loss(
                    attn_map[top_embedding_indices], 
                    device=args.device, 
                    sigma=args.sigma, 
                    num_subjects=args.num_subjects
                )
            )
            
            _loss_equivariance_attn.append(
                equivariance_loss(
                    attn_map[top_embedding_indices],
                    attention_map_transformed[top_embedding_indices][None].repeat(num_gpus, 1, 1, 1),
                    invertible_transform,
                    index
                )
            )
        
        # Aggregate losses
        _sharpening_loss = torch.stack([loss.to('cuda:0') for loss in _sharpening_loss]).mean()
        _loss_equivariance_attn = torch.stack([loss.to('cuda:0') for loss in _loss_equivariance_attn]).mean()
        
        # Combine losses
        loss = (
            _loss_equivariance_attn * args.equivariance_attn_loss_weight
            + _sharpening_loss * args.sharpening_loss_weight
        )
        
        # Backpropagation
        loss = loss / (args.batch_size // num_gpus)
        loss.backward()
        
        if (iteration + 1) % (args.batch_size // num_gpus) == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    return context.detach()
```

#### 6. Keypoint Regression and Evaluation
```python
def find_best_indices(source_kpts, target_kpts, visible=None):
    """
    Find the best mapping between discovered keypoints and ground truth using linear regression
    """
    if visible is not None:
        # Handle visibility-aware regression
        visible_reshaped = visible.unsqueeze(-1).repeat(1, 1, 2).reshape(
            visible.shape[0], visible.shape[1] * 2
        ).cpu().numpy().astype(np.float64)
        
        regressor = return_regressor_visible(
            source_kpts.cpu().numpy().reshape(source_kpts.shape[0], source_kpts.shape[1] * 2).astype(np.float64),
            target_kpts.cpu().numpy().reshape(target_kpts.shape[0], target_kpts.shape[1] * 2).astype(np.float64),
            visible_reshaped,
        )
    else:
        # Standard linear regression without bias
        regressor = return_regressor(
            source_kpts.cpu().numpy().reshape(source_kpts.shape[0], source_kpts.shape[1] * 2).astype(np.float64),
            target_kpts.cpu().numpy().reshape(target_kpts.shape[0], target_kpts.shape[1] * 2).astype(np.float64),
        )
    
    return torch.tensor(regressor).to(torch.float32)
```

#### 7. Invertible Transformations for Equivariance
```python
# From invertable_transform.py
class RandomAffineWithInverse(torch.nn.Module):
    """
    Applies random affine transformations and provides inverse transformation
    """
    def __init__(self, degrees=15, scale=(0.9, 1.1), translate=(0.1, 0.1)):
        super().__init__()
        self.degrees = degrees
        self.scale = scale
        self.translate = translate
    
    def forward(self, img):
        # Generate random transformation parameters
        angle = torch.empty(1).uniform_(-self.degrees, self.degrees).item()
        scale = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
        translate_x = torch.empty(1).uniform_(-self.translate[0], self.translate[0]).item()
        translate_y = torch.empty(1).uniform_(-self.translate[1], self.translate[1]).item()
        
        # Store parameters for inverse transformation
        self.last_params = {
            'angle': angle,
            'scale': scale,
            'translate': (translate_x, translate_y)
        }
        
        # Apply transformation
        transformed = TF.affine(
            img, angle=angle, translate=[translate_x, translate_y],
            scale=scale, shear=0
        )
        return transformed
    
    def inverse(self, attention_map):
        """
        Apply inverse transformation to attention maps
        """
        if not hasattr(self, 'last_params'):
            return attention_map
            
        params = self.last_params
        # Apply inverse transformation with opposite parameters
        inverse_transformed = TF.affine(
            attention_map,
            angle=-params['angle'],
            translate=[-params['translate'][0], -params['translate'][1]],
            scale=1.0/params['scale'],
            shear=0
        )
        return inverse_transformed
```

#### 8. Top-K Token Selection Strategies
```python
def find_top_k_gaussian(attn_map, k, sigma=1.0, num_subjects=1):
    """
    Select top-k tokens based on how well they fit Gaussian distributions
    """
    T, H, W = attn_map.shape
    scores = []
    
    for t in range(T):
        # Find maximum location
        max_pos = eval.find_k_max_pixels(attn_map[t:t+1], num=num_subjects)
        max_pos = max_pos / attn_map.shape[-1]  # Normalize to [0,1]
        
        # Create target Gaussian
        target = optimize_token.gaussian_circles(
            max_pos, size=H, sigma=sigma, device=attn_map.device
        )
        
        # Compute negative KL divergence (higher is better)
        score = -F.kl_div(
            F.log_softmax(attn_map[t].flatten(), dim=0),
            F.softmax(target.flatten(), dim=0),
            reduction='sum'
        )
        scores.append(score)
    
    # Return indices of top-k scores
    scores = torch.stack(scores)
    _, top_indices = torch.topk(scores, k)
    return top_indices

def furthest_point_sampling(attention_maps, k, candidate_indices):
    """
    Apply furthest point sampling to select diverse keypoints
    """
    if len(candidate_indices) <= k:
        return candidate_indices
    
    # Flatten attention maps for distance computation
    flattened = attention_maps[candidate_indices].view(len(candidate_indices), -1)
    
    # Initialize with random point
    selected_indices = [0]
    
    for _ in range(k - 1):
        distances = []
        for i in range(len(candidate_indices)):
            if i in selected_indices:
                distances.append(0)
            else:
                # Compute minimum distance to already selected points
                min_dist = float('inf')
                for j in selected_indices:
                    dist = F.mse_loss(flattened[i], flattened[j])
                    min_dist = min(min_dist, dist.item())
                distances.append(min_dist)
        
        # Select point with maximum minimum distance
        next_idx = np.argmax(distances)
        selected_indices.append(next_idx)
    
    return candidate_indices[selected_indices]
```

## Future Directions and Extensions

### 1. Multi-Scale Keypoints
- Leverage attention maps from different resolutions
- Hierarchical keypoint discovery

### 2. Temporal Consistency
- Extend to video data for temporal keypoint tracking
- Leverage motion priors from diffusion models

### 3. 3D Keypoint Discovery
- Apply to 3D-aware diffusion models
- Discover 3D semantic correspondences

### 4. Interactive Keypoint Refinement
- Allow user guidance during optimization
- Incorporate semantic constraints

### 5. Efficiency Improvements
- Distillation to smaller models
- Optimization acceleration techniques

## Practical Applications

### 1. Object Pose Estimation
- Robust keypoint detection for downstream pose estimation
- Handles partial occlusions and viewpoint variations

### 2. Medical Imaging
- Anatomical landmark detection without manual annotation
- Cross-patient generalization

### 3. Robotics
- Object manipulation planning using discovered keypoints
- Visual servoing applications

### 4. Content Creation
- Automatic rigging for animation
- Style transfer and image editing

## Code Organization

The implementation follows a modular structure:

```
StableKeypoints/
├── unsupervised_keypoints/
│   ├── main.py              # Main training script
│   ├── optimize.py          # Core optimization logic
│   ├── optimize_token.py    # Token-specific utilities
│   ├── eval.py             # Evaluation functions
│   ├── visualize.py        # Visualization tools
│   └── ptp_utils.py        # Attention extraction utilities
├── datasets/
│   ├── celeba.py           # CelebA dataset loader
│   ├── cub.py              # CUB dataset loader
│   ├── taichi.py           # Tai-Chi dataset loader
│   └── ...                 # Other dataset loaders
└── requirements.yaml       # Environment specification
```

## Conclusion

This work represents a significant advancement in unsupervised keypoint discovery by cleverly leveraging the semantic knowledge embedded in large-scale pretrained diffusion models. The key innovations include:

1. **Novel use of cross-attention**: Transforming text-image attention into spatial keypoint localization
2. **Effective optimization strategy**: Combining localization and equivariance losses
3. **Superior empirical results**: Outperforming existing methods, especially on challenging unaligned data
4. **Strong generalization**: Learned keypoints transfer across domains and datasets

The method opens new avenues for utilizing pretrained generative models for structured understanding tasks and demonstrates the power of leveraging large-scale pretraining for downstream computer vision applications.

The approach is particularly valuable for scenarios where:
- Labeled keypoint data is scarce or expensive to obtain
- Dealing with "in-the-wild" unprocessed images
- Requiring keypoints that generalize across related domains
- Working with new object categories without existing annotations

This work establishes a new paradigm for unsupervised keypoint learning and demonstrates how the emergent semantic understanding in large generative models can be harnessed for structured computer vision tasks.
