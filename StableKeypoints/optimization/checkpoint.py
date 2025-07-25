"""
Checkpoint utilities for saving and loading optimized embeddings
"""

import os
import torch
import json
from datetime import datetime


def save_checkpoint(embedding, optimizer_state, step, loss, config, checkpoint_dir):
    """
    Save training checkpoint including embedding, optimizer state, and metadata
    
    Args:
        embedding: The optimized embedding tensor
        optimizer_state: Optimizer state dict
        step: Current training step
        loss: Current loss value
        config: Configuration object
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Path to saved checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f"embedding_checkpoint_step_{step}_{timestamp}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    checkpoint_data = {
        'embedding': embedding.cpu(),
        'optimizer_state': optimizer_state,
        'step': step,
        'loss': loss,
        'timestamp': timestamp,
        'config': {
            'NUM_TOKENS': config.NUM_TOKENS,
            'LEARNING_RATE': config.LEARNING_RATE,
            'NUM_KEYPOINTS': config.NUM_KEYPOINTS,
            'STABLE_DIFFUSION_MODEL': config.STABLE_DIFFUSION_MODEL,
            'FEATURE_UPSAMPLE_RES': config.FEATURE_UPSAMPLE_RES,
            'LAYERS': config.LAYERS,
            'FROM_WHERE': config.FROM_WHERE,
            'NOISE_LEVEL': config.NOISE_LEVEL,
        }
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    
    # Also save metadata as JSON for easier reading
    metadata_path = os.path.join(checkpoint_dir, f"metadata_step_{step}_{timestamp}.json")
    metadata = {
        'step': step,
        'loss': float(loss) if torch.is_tensor(loss) else loss,
        'timestamp': timestamp,
        'checkpoint_file': checkpoint_filename,
        'config': checkpoint_data['config']
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, device="cuda"):
    """
    Load checkpoint from file
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load tensors to
    
    Returns:
        Dictionary containing embedding, optimizer_state, step, loss, and config
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    
    # Move embedding to specified device
    if 'embedding' in checkpoint_data:
        checkpoint_data['embedding'] = checkpoint_data['embedding'].to(device)
    
    print(f"Loaded checkpoint from step {checkpoint_data.get('step', 'unknown')}")
    return checkpoint_data


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the given directory
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
    
    Returns:
        Path to latest checkpoint file, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("embedding_checkpoint_") and f.endswith(".pt")]
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time to get the latest
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
    
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def list_checkpoints(checkpoint_dir):
    """
    List all available checkpoints in the directory
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
    
    Returns:
        List of checkpoint information dictionaries
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    metadata_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("metadata_") and f.endswith(".json")]
    
    for metadata_file in metadata_files:
        try:
            with open(os.path.join(checkpoint_dir, metadata_file), 'r') as f:
                metadata = json.load(f)
                checkpoints.append(metadata)
        except Exception as e:
            print(f"Error reading metadata file {metadata_file}: {e}")
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x.get('step', 0), reverse=True)
    return checkpoints


def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=5):
    """
    Remove old checkpoints, keeping only the most recent ones
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of checkpoints to keep
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("embedding_checkpoint_") and f.endswith(".pt")]
    metadata_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("metadata_") and f.endswith(".json")]
    
    if len(checkpoint_files) <= keep_last_n:
        return
    
    # Sort by modification time
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    metadata_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    
    # Remove old checkpoint files
    for old_checkpoint in checkpoint_files[keep_last_n:]:
        try:
            os.remove(os.path.join(checkpoint_dir, old_checkpoint))
            print(f"Removed old checkpoint: {old_checkpoint}")
        except Exception as e:
            print(f"Error removing checkpoint {old_checkpoint}: {e}")
    
    # Remove old metadata files
    for old_metadata in metadata_files[keep_last_n:]:
        try:
            os.remove(os.path.join(checkpoint_dir, old_metadata))
            print(f"Removed old metadata: {old_metadata}")
        except Exception as e:
            print(f"Error removing metadata {old_metadata}: {e}")
