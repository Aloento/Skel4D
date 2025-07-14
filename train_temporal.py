"""
Training script with temporal consistency for StableKeypoints
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from StableKeypoints.api import StableKeypoints
from StableKeypoints.config import Config


def train_with_temporal_consistency():
    """Train StableKeypoints with temporal consistency loss"""
    print("=== Training StableKeypoints with Temporal Consistency ===")
    
    # Configure parameters
    config = Config()
    config.NUM_KEYPOINTS = 15
    config.NUM_OPTIMIZATION_STEPS = 2000
    config.BATCH_SIZE = 1
    config.FURTHEST_POINT_NUM_SAMPLES = 15
    config.IMAGE_DIR = "/home/c_capzw/c_cape3d/data/rendered/objaverse/66a62fc9ab97415f85b6322c103f8e1e/Take001"
    
    # === Temporal Consistency Settings ===
    config.TEMPORAL_CONSISTENCY_LOSS_WEIGHT = 100  # λ_temp parameter
    config.TEMPORAL_LOSS_TYPE = "kl"  # or "kl" for KL divergence
    config.TEMPORAL_FRAME_GAP = 3  # Adjacent frames
    config.USE_ADAPTIVE_TEMPORAL_LOSS = True  # Enable adaptive loss
    config.MOTION_THRESHOLD = 0.1  # For adaptive loss
    
    # === Standard Loss Weights ===
    config.SHARPENING_LOSS_WEIGHT = 100
    config.EQUIVARIANCE_ATTN_LOSS_WEIGHT = 100
    
    # === Checkpoint Settings ===
    config.LOAD_FROM_CHECKPOINT = True  # Load existing checkpoint if available
    config.CONTINUE_TRAINING = False  # Set to True to continue from checkpoint
    config.SAVE_CHECKPOINTS = True
    config.CHECKPOINT_SAVE_INTERVAL = 500  # Save every 500 steps
    
    print("Configuration:")
    print(f"  Optimization steps: {config.NUM_OPTIMIZATION_STEPS}")
    print(f"  Temporal loss weight: {config.TEMPORAL_CONSISTENCY_LOSS_WEIGHT}")
    print(f"  Temporal loss type: {config.TEMPORAL_LOSS_TYPE}")
    print(f"  Frame gap: {config.TEMPORAL_FRAME_GAP}")
    print(f"  Adaptive temporal loss: {config.USE_ADAPTIVE_TEMPORAL_LOSS}")
    print(f"  Load from checkpoint: {config.LOAD_FROM_CHECKPOINT}")
    print(f"  Continue training: {config.CONTINUE_TRAINING}")
    print()
    
    # Create StableKeypoints instance
    sk = StableKeypoints(config)
    
    # Run the complete pipeline
    results = sk.run_pipeline(
        image_dir=config.IMAGE_DIR,
        output_path="keypoints_temporal_training.gif",
        output_csv="keypoints_temporal_training.csv",
        augmentation_iterations=20
    )
    
    print(f"Training completed!")
    print(f"Generated GIF: {results['gif']}")
    print(f"Generated CSV: {results['csv']}")
    print()
    print("Key benefits of temporal consistency:")
    print("- Reduced keypoint jitter between frames")
    print("- More stable tracking across video sequences")
    print("- Better performance for downstream video generation tasks")


def compare_loss_configurations():
    """Compare different temporal loss configurations"""
    print("=== Comparing Different Temporal Loss Configurations ===")
    
    base_config = Config()
    base_config.NUM_KEYPOINTS = 15
    base_config.NUM_OPTIMIZATION_STEPS = 1000  # Reduced for comparison
    base_config.BATCH_SIZE = 1
    base_config.FURTHEST_POINT_NUM_SAMPLES = 15
    base_config.IMAGE_DIR = "/home/c_capzw/c_cape3d/data/rendered/objaverse/66a62fc9ab97415f85b6322c103f8e1e/Take001"
    base_config.LOAD_FROM_CHECKPOINT = False  # Start fresh for comparison
    base_config.SAVE_CHECKPOINTS = False  # Don't save during comparison
    
    configurations = [
        {
            "name": "Standard (No Temporal)",
            "temporal_weight": 0.0,
            "temporal_type": "l2",
            "adaptive": False,
            "output_suffix": "standard"
        },
        {
            "name": "L2 Temporal (Weight=5.0)",
            "temporal_weight": 5.0,
            "temporal_type": "l2",
            "adaptive": False,
            "output_suffix": "l2_5"
        },
        {
            "name": "L2 Temporal (Weight=15.0)",
            "temporal_weight": 15.0,
            "temporal_type": "l2",
            "adaptive": False,
            "output_suffix": "l2_15"
        },
        {
            "name": "KL Temporal (Weight=10.0)",
            "temporal_weight": 10.0,
            "temporal_type": "kl",
            "adaptive": False,
            "output_suffix": "kl_10"
        },
        {
            "name": "Adaptive L2 (Weight=10.0)",
            "temporal_weight": 10.0,
            "temporal_type": "l2",
            "adaptive": True,
            "output_suffix": "adaptive"
        }
    ]
    
    results = {}
    
    for i, config_dict in enumerate(configurations):
        print(f"\n--- Configuration {i+1}: {config_dict['name']} ---")
        
        try:
            # Create config for this test
            test_config = base_config
            test_config.TEMPORAL_CONSISTENCY_LOSS_WEIGHT = config_dict["temporal_weight"]
            test_config.TEMPORAL_LOSS_TYPE = config_dict["temporal_type"]
            test_config.USE_ADAPTIVE_TEMPORAL_LOSS = config_dict["adaptive"]
            
            # Create StableKeypoints instance
            sk = StableKeypoints(test_config)
            
            # Run pipeline
            result = sk.run_pipeline(
                image_dir=test_config.IMAGE_DIR,
                output_path=f"keypoints_comparison_{config_dict['output_suffix']}.gif",
                output_csv=f"keypoints_comparison_{config_dict['output_suffix']}.csv",
                augmentation_iterations=5
            )
            
            results[config_dict['name']] = result
            print(f"✓ Completed: {config_dict['name']}")
            
        except Exception as e:
            print(f"✗ Failed: {config_dict['name']} - {e}")
            results[config_dict['name']] = None
    
    print("\n=== Comparison Results ===")
    for config_name, result in results.items():
        if result:
            print(f"✓ {config_name}")
            print(f"  GIF: {result['gif']}")
            print(f"  CSV: {result['csv']}")
        else:
            print(f"✗ {config_name} - Failed")
    
    print("\nRecommendations:")
    print("1. Start with L2 temporal loss (weight=10.0) for most cases")
    print("2. Use adaptive temporal loss for videos with varying motion")
    print("3. Increase temporal weight if keypoints are still unstable")
    print("4. Use KL divergence for smoother attention distributions")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train StableKeypoints with temporal consistency")
    parser.add_argument("--mode", choices=["train", "compare"], default="train",
                       help="Mode: 'train' for full training, 'compare' for configuration comparison")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with reduced steps")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_with_temporal_consistency()
    elif args.mode == "compare":
        compare_loss_configurations()
    
    print("\nTemporal consistency training completed!")
    print("Check the generated GIF files to evaluate temporal stability.")
