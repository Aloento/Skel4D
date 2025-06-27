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
    
    # Attention settings
    FROM_WHERE = ["down_cross", "mid_cross", "up_cross"]
    LAYERS = [0, 1, 2, 3, 4, 5]
    NOISE_LEVEL = -1
    SIGMA = 1.0
    
    # Visualization settings
    UPSAMPLE_RES = 512
    AUGMENTATION_ITERATIONS = 10
    GIF_FPS = 5
    MAX_FRAMES = 100
    
    # Default image directory
    IMAGE_DIR = "/home/c_capzw/c_cape3d/data/rendered/objaverse/66a62fc9ab97415f85b6322c103f8e1e/Take001"
