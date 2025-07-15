"""
Embedding optimization for StableKeypoints
"""

import time
import torch
from tqdm import tqdm

from ..config import Config
from ..data.dataset import CustomDataset
from ..data.temporal_dataset import TemporalDataset
from ..data.transforms import RandomAffineWithInverse
from ..utils.image_utils import init_random_noise, run_and_find_attn
from ..utils.keypoint_utils import find_top_k_gaussian, furthest_point_sampling
from .losses import sharpening_loss, equivariance_loss, temporal_consistency_loss, adaptive_temporal_loss
from .checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


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
    config=None,  # type: Config
):
    """
    Optimize context embedding for keypoint detection
    
    Args:
        ldm: Loaded diffusion model
        context: Initial context embedding (if None, creates random or loads from checkpoint)
        device: Device to run optimization on
        num_steps: Number of optimization steps
        config: Configuration object containing checkpoint settings and temporal settings
        ... (other parameters)
        
    Returns:
        Optimized context embedding
    """
    
    # Choose dataset based on whether temporal consistency is enabled
    use_temporal = config and config.TEMPORAL_CONSISTENCY_LOSS_WEIGHT > 0
    
    if use_temporal:
        print("Using temporal dataset for temporal consistency training...")
        dataset = TemporalDataset(
            data_root=dataset_loc, 
            image_size=512,
            frame_gap=config.TEMPORAL_FRAME_GAP
        )
    else:
        print("Using standard dataset...")
        dataset = CustomDataset(data_root=dataset_loc, image_size=512)

    invertible_transform = RandomAffineWithInverse(
        degrees=augment_degrees,
        scale=augment_scale,
        translate=augment_translate,
    )

    # Handle checkpoint loading and context initialization
    start_step = 0
    optimizer_state = None
    
    if config and config.LOAD_FROM_CHECKPOINT:
        checkpoint_path = config.CHECKPOINT_PATH
        if checkpoint_path is None:
            # Try to find latest checkpoint
            checkpoint_path = find_latest_checkpoint(config.CHECKPOINT_DIR)
        
        if checkpoint_path and checkpoint_path != "":
            try:
                checkpoint_data = load_checkpoint(checkpoint_path, device)
                context = checkpoint_data['embedding']
                if config.CONTINUE_TRAINING:
                    start_step = checkpoint_data.get('step', 0)
                    optimizer_state = checkpoint_data.get('optimizer_state')
                    print(f"Resuming training from step {start_step}")
                else:
                    print("Loaded embedding from checkpoint (not continuing training)")
                    return context.detach()
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting with random embedding...")
                context = None

    if context is None:
        context = init_random_noise(device, num_words=num_tokens)

    context.requires_grad = True

    # optimize context to maximize attention at pixel_loc
    optimizer = torch.optim.Adam([context], lr=lr)
    
    # Load optimizer state if resuming training
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"Could not load optimizer state: {e}")
            print("Starting with fresh optimizer")

    start = time.time()

    # Early stopping variables
    early_stopping_enabled = config.EARLY_STOPPING_ENABLED if config else False
    patience = config.EARLY_STOPPING_PATIENCE if config else 50
    min_delta = config.EARLY_STOPPING_MIN_DELTA if config else 1e-4

    best_loss = float('inf')
    patience_counter = 0
    early_stopped = False
    
    if early_stopping_enabled:
        print(f"Early stopping enabled:")
        print(f"  Patience: {patience} steps")
        print(f"  Min delta: {min_delta}")

    running_equivariance_attn_loss = 0
    running_sharpening_loss = 0
    running_temporal_loss = 0
    running_total_loss = 0

    # create dataloader for the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=num_gpus, 
        shuffle=True, 
        drop_last=True
    )
    dataloader_iter = iter(dataloader)

    for iteration in tqdm(range(start_step, int(num_steps*batch_size)), initial=start_step):

        try:
            mini_batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            mini_batch = next(dataloader_iter)
            
        if use_temporal:
            # Process temporal pair(s) - we always have temporal pairs in temporal dataset
            frame_t = mini_batch["frame_t"]
            frame_t1 = mini_batch["frame_t1"]
            
            # Get attention maps for both frames
            attn_maps_t = run_and_find_attn(
                ldm,
                frame_t,
                context,
                layers=layers,
                noise_level=noise_level,
                from_where=from_where,
                upsample_res=-1,
                device=device,
                controllers=controllers,
            )
            
            attn_maps_t1 = run_and_find_attn(
                ldm,
                frame_t1,
                context,
                layers=layers,
                noise_level=noise_level,
                from_where=from_where,
                upsample_res=-1,
                device=device,
                controllers=controllers,
            )
            
            # Use frame_t for equivariance computation
            image = frame_t
            attn_maps = attn_maps_t
        else:
            # Process single frame (original behavior)
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
            attn_maps_t = None
            attn_maps_t1 = None

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
        _temporal_loss = []

        for index, attn_map, attention_map_transformed in zip(torch.arange(num_gpus), attn_maps, attention_maps_transformed):

            top_embedding_indices = find_top_k_gaussian(
                attn_map, furthest_point_num_samples, sigma=sigma, num_subjects=num_subjects
            )

            top_embedding_indices = furthest_point_sampling(attention_map_transformed, top_k, top_embedding_indices)

            _sharpening_loss.append(sharpening_loss(attn_map[top_embedding_indices], device=device, sigma=sigma, num_subjects=num_subjects))

            _loss_equivariance_attn.append(equivariance_loss(
                attn_map[top_embedding_indices], attention_map_transformed[top_embedding_indices][None].repeat(num_gpus, 1, 1, 1), invertible_transform, index
            ))
            
            # Compute temporal consistency loss if temporal data is available
            current_sample_is_temporal = (
                use_temporal and 
                attn_maps_t is not None and 
                attn_maps_t1 is not None
            )
            
            if current_sample_is_temporal:
                attn_t_selected = attn_maps_t[index][top_embedding_indices]
                attn_t1_selected = attn_maps_t1[index][top_embedding_indices]
                
                # Choose temporal loss type from config
                temporal_loss_type = config.TEMPORAL_LOSS_TYPE if config else 'l2'
                use_adaptive = config.USE_ADAPTIVE_TEMPORAL_LOSS if config else False
                
                if use_adaptive:
                    motion_threshold = config.MOTION_THRESHOLD if config else 0.1
                    temp_loss = adaptive_temporal_loss(attn_t_selected, attn_t1_selected, motion_threshold)
                else:
                    temp_loss = temporal_consistency_loss(attn_t_selected, attn_t1_selected, temporal_loss_type)

                _temporal_loss.append(temp_loss)
            else:
                # No temporal loss for single frame samples or when temporal data is not available
                _temporal_loss.append(torch.tensor(0.0, device=device))

        _sharpening_loss = torch.stack([loss.to('cuda:0') for loss in _sharpening_loss]).mean()
        _loss_equivariance_attn = torch.stack([loss.to('cuda:0') for loss in _loss_equivariance_attn]).mean()
        _temporal_loss = torch.stack([loss.to('cuda:0') for loss in _temporal_loss]).mean()
        
        # Get temporal loss weight from config
        temporal_loss_weight = config.TEMPORAL_CONSISTENCY_LOSS_WEIGHT if config else 0.0

        loss = (
            + _loss_equivariance_attn * equivariance_attn_loss_weight
            + _sharpening_loss * sharpening_loss_weight
            + _temporal_loss * temporal_loss_weight
        )

        running_equivariance_attn_loss += _loss_equivariance_attn / (batch_size//num_gpus) * equivariance_attn_loss_weight
        running_sharpening_loss += _sharpening_loss / (batch_size//num_gpus) * sharpening_loss_weight
        running_temporal_loss += _temporal_loss / (batch_size//num_gpus) * temporal_loss_weight
        running_total_loss += loss / (batch_size//num_gpus)

        loss = loss / (batch_size//num_gpus)

        if iteration % 50 == 0:
            print(
                f"loss: {loss.item()}, "
                f"equivariance: {running_equivariance_attn_loss.item():.4f}, "
                f"sharpening: {running_sharpening_loss.item():.4f}, "
                f"temporal: {running_temporal_loss.item():.4f}, "
                f"total: {running_total_loss.item():.4f}"
            )
        loss.backward()
        if (iteration + 1) % (batch_size//num_gpus) == 0:
            optimizer.step()
            optimizer.zero_grad()
            running_equivariance_attn_loss = 0
            running_sharpening_loss = 0
            running_temporal_loss = 0
            running_total_loss = 0
            
            # Save checkpoint if enabled
            if (config and config.SAVE_CHECKPOINTS and 
                (iteration + 1) % config.CHECKPOINT_SAVE_INTERVAL == 0):
                try:
                    save_checkpoint(
                        embedding=context,
                        optimizer_state=optimizer.state_dict(),
                        step=iteration + 1,
                        loss=loss.item() if torch.is_tensor(loss) else loss,
                        config=config,
                        checkpoint_dir=config.CHECKPOINT_DIR
                    )
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")

        # Early stopping check
        if early_stopping_enabled and not early_stopped:
            if loss < best_loss - min_delta:
                best_loss = loss
                patience_counter = 0  # reset counter if we improved
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {iteration + 1} iterations")
                early_stopped = True
                break

    print(f"optimization took {time.time() - start} seconds")
    
    # Save final checkpoint if enabled
    if config and config.SAVE_CHECKPOINTS:
        try:
            final_checkpoint_path = save_checkpoint(
                embedding=context,
                optimizer_state=optimizer.state_dict(),
                step=int(num_steps*batch_size),
                loss=loss.item() if torch.is_tensor(loss) else loss,
                config=config,
                checkpoint_dir=config.CHECKPOINT_DIR
            )
            print(f"Final checkpoint saved: {final_checkpoint_path}")
        except Exception as e:
            print(f"Error saving final checkpoint: {e}")

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
            
            # Clean up GPU memory
            del attention_map
        
        # Clean up attention_maps from GPU memory
        del attention_maps
        torch.cuda.empty_cache()

    # find the top_k most common indices
    indices_list = torch.cat([index for index in indices_list])
    indices, counts = torch.unique(indices_list, return_counts=True)
    indices = indices[counts.argsort(descending=True)]
    indices = indices[:top_k]

    return indices
