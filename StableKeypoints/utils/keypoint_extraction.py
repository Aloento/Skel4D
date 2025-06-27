"""
Keypoint extraction utilities
"""

import torch
from ..data.dataset import CustomDataset
from ..utils.augmentation import run_image_with_context_augmented
from ..utils.keypoint_utils import find_max_pixel


def extract_keypoints(ldm, embedding, indices, config, image_dir, controllers, num_gpus, augmentation_iterations=20):
    """
    Extract keypoint coordinates for all images.
    
    Args:
        ldm: Loaded diffusion model
        embedding: Optimized embedding
        indices: Best indices for keypoint detection
        config: Configuration object
        image_dir: Directory containing images
        controllers: Model controllers
        num_gpus: Number of GPUs
        augmentation_iterations: Number of augmentation iterations for robust detection
        
    Returns:
        List of dictionaries containing frame data: [{"frame_idx": int, "image_name": str, "img": tensor, "keypoints": array}, ...]
    """
    dataset = CustomDataset(data_root=image_dir, image_size=512)
    keypoints_data = []
    
    print(f"Extracting keypoints from {len(dataset)} images...")
    
    for frame_idx in range(len(dataset)):
        # Get image
        batch = dataset[frame_idx]
        img = batch["img"]
        image_name = batch.get("name", f"frame_{frame_idx:04d}")
        
        # Extract keypoints using the optimized embedding
        maps = []
        contexts = embedding if isinstance(embedding, list) else [embedding]
        
        for context in contexts:
            map = run_image_with_context_augmented(
                ldm,
                img,
                context,
                indices.cpu(),
                device="cuda:0",
                from_where=config.FROM_WHERE,
                layers=config.LAYERS,
                noise_level=config.NOISE_LEVEL,
                augment_degrees=config.AUGMENT_DEGREES,
                augment_scale=config.AUGMENT_SCALE,
                augment_translate=config.AUGMENT_TRANSLATE,
                augmentation_iterations=augmentation_iterations,
                controllers=controllers,
                num_gpus=num_gpus,
                upsample_res=512,
            )
            maps.append(map)
        
        # Average maps if multiple contexts
        maps = torch.stack(maps)
        final_map = torch.mean(maps, dim=0)
        
        # Find keypoint coordinates
        keypoints = find_max_pixel(final_map) / 512.0  # Normalize to [0,1]
        keypoints = keypoints.cpu().numpy()
        
        # Store frame data
        frame_data = {
            "frame_idx": frame_idx,
            "image_name": image_name,
            "img": img,
            "keypoints": keypoints
        }
        keypoints_data.append(frame_data)
        
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx + 1}/{len(dataset)} images...")
    
    return keypoints_data
