"""
Configuration settings for StableKeypoints project
"""

class Config:
    # Model settings
    STABLE_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"
    FEATURE_UPSAMPLE_RES = 128
    NUM_DDIM_STEPS = 50
    
    # Keypoints settings
    NUM_KEYPOINTS = 15
    FURTHEST_POINT_NUM_SAMPLES = 15
    
    # Optimization settings
    NUM_OPTIMIZATION_STEPS = 2000
    BATCH_SIZE = 1
    LEARNING_RATE = 5e-3
    NUM_TOKENS = 500
    
    # Augmentation settings
    AUGMENT_DEGREES = 30
    AUGMENT_SCALE = (0.9, 1.1)
    AUGMENT_TRANSLATE = (0.1, 0.1)
    
    # Loss weights
    SHARPENING_LOSS_WEIGHT = 100
    EQUIVARIANCE_ATTN_LOSS_WEIGHT = 100
    
    # Temporal consistency settings
    TEMPORAL_CONSISTENCY_LOSS_WEIGHT = 10.0  # λ_temp parameter
    TEMPORAL_LOSS_TYPE = "l2"  # "l2" or "kl"
    TEMPORAL_FRAME_GAP = 1  # Gap between consecutive frames
    USE_ADAPTIVE_TEMPORAL_LOSS = False  # Whether to use adaptive temporal loss
    MOTION_THRESHOLD = 0.1  # Threshold for adaptive temporal loss
    
    # Attention settings
    FROM_WHERE = ["down_cross", "mid_cross", "up_cross"]
    LAYERS = [0, 1, 2, 3, 4, 5]
    NOISE_LEVEL = -1
    SIGMA = 1.0
    
    # Visualization settings
    UPSAMPLE_RES = 256
    AUGMENTATION_ITERATIONS = 10
    GIF_FPS = 5
    MAX_FRAMES = 100
    
    # Default image directory
    IMAGE_DIR = "/home/c_capzw/c_cape3d/data/rendered/objaverse/66a62fc9ab97415f85b6322c103f8e1e/Take001"
    
    # Checkpoint settings
    CHECKPOINT_DIR = "./checkpoints"
    SAVE_CHECKPOINTS = True
    CHECKPOINT_SAVE_INTERVAL = 500  # Save checkpoint every N steps
    
    # Resume/Load settings
    LOAD_FROM_CHECKPOINT = False
    CHECKPOINT_PATH = "/home/c_capzw/notebooks/Skel4D/checkpoints/embedding_checkpoint_step_2000_20250627_130735.pt"  # Path to specific checkpoint to load
    CONTINUE_TRAINING = False  # If True, continue training from checkpoint; if False, just use the embedding

    # Staged training settings
    STAGE2_STEPS = 500   # Steps for stage 2 (temporal consistency refinement)
    STAGE2_TEMPORAL_WEIGHT = 50  # Temporal loss weight for stage 2
    STAGE2_SHARPENING_WEIGHT = 50  # Reduced sharpening weight for stage 2
    STAGE2_EQUIVARIANCE_WEIGHT = 50  # Reduced equivariance weight for stage 2

    # Early stopping settings
    EARLY_STOPPING_ENABLED = True  # Enable early stopping
    EARLY_STOPPING_PATIENCE = 500   # Number of steps to wait for improvement
    EARLY_STOPPING_MIN_DELTA = 1e-4  # Minimum improvement to consider
  