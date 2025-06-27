"""
Main API for StableKeypoints
"""

from .config import Config
from .models.model_loader import load_ldm
from .optimization.optimizer import optimize_embedding, find_best_indices
from .visualization.gif_creator import create_keypoints_gif


class StableKeypoints:
    """Main class for StableKeypoints functionality"""
    
    def __init__(self, config=None):
        """
        Initialize StableKeypoints
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config if config is not None else Config()
        self.ldm = None
        self.controllers = None
        self.num_gpus = None
        self.embedding = None
        self.indices = None
    
    def load_model(self, device="cuda:0"):
        """Load the Stable Diffusion model"""
        print("Loading Stable Diffusion model...")
        self.ldm, self.controllers, self.num_gpus = load_ldm(
            device, 
            self.config.STABLE_DIFFUSION_MODEL, 
            self.config.FEATURE_UPSAMPLE_RES
        )
        print("Model loaded successfully!")
        return self
    
    def optimize(self, image_dir=None):
        """Optimize embedding for keypoint detection"""
        if self.ldm is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if image_dir is None:
            image_dir = self.config.IMAGE_DIR
        
        print("Optimizing embedding...")
        self.embedding = optimize_embedding(
            self.ldm,
            num_steps=self.config.NUM_OPTIMIZATION_STEPS,
            batch_size=self.config.BATCH_SIZE,
            top_k=self.config.NUM_KEYPOINTS,
            controllers=self.controllers,
            num_gpus=self.num_gpus,
            furthest_point_num_samples=self.config.FURTHEST_POINT_NUM_SAMPLES,
            num_tokens=self.config.NUM_TOKENS,
            dataset_loc=image_dir,
            lr=self.config.LEARNING_RATE,
            augment_degrees=self.config.AUGMENT_DEGREES,
            augment_scale=self.config.AUGMENT_SCALE,
            augment_translate=self.config.AUGMENT_TRANSLATE,
            sigma=self.config.SIGMA,
            sharpening_loss_weight=self.config.SHARPENING_LOSS_WEIGHT,
            equivariance_attn_loss_weight=self.config.EQUIVARIANCE_ATTN_LOSS_WEIGHT,
            from_where=self.config.FROM_WHERE,
            layers=self.config.LAYERS,
            noise_level=self.config.NOISE_LEVEL,
        )
        print("Embedding optimization completed!")
        return self
    
    def find_indices(self, image_dir=None):
        """Find the best indices for keypoint detection"""
        if self.embedding is None:
            raise ValueError("Embedding not optimized. Call optimize() first.")
        
        if image_dir is None:
            image_dir = self.config.IMAGE_DIR
        
        print("Finding best indices...")
        self.indices = find_best_indices(
            self.ldm,
            self.embedding,
            dataset_loc=image_dir,
            controllers=self.controllers,
            furthest_point_num_samples=self.config.FURTHEST_POINT_NUM_SAMPLES,
            top_k=self.config.NUM_KEYPOINTS,
            num_gpus=self.num_gpus,
            from_where=self.config.FROM_WHERE,
            layers=self.config.LAYERS,
            noise_level=self.config.NOISE_LEVEL,
            upsample_res=self.config.UPSAMPLE_RES,
        )
        print("Best indices found!")
        return self
    
    def create_gif(self, image_dir=None, output_path="keypoints_sequence.gif"):
        """Create GIF visualization of keypoint detection"""
        if self.embedding is None or self.indices is None:
            raise ValueError("Embedding and indices not ready. Call optimize() and find_indices() first.")
        
        if image_dir is None:
            image_dir = self.config.IMAGE_DIR
        
        print("Creating GIF visualization...")
        gif_path = create_keypoints_gif(
            self.ldm,
            self.embedding,
            self.indices,
            dataset_loc=image_dir,
            controllers=self.controllers,
            augmentation_iterations=self.config.AUGMENTATION_ITERATIONS,
            output_path=output_path,
            fps=self.config.GIF_FPS,
            max_frames=self.config.MAX_FRAMES,
            num_gpus=self.num_gpus,
            from_where=self.config.FROM_WHERE,
            layers=self.config.LAYERS,
            noise_level=self.config.NOISE_LEVEL,
            augment_degrees=self.config.AUGMENT_DEGREES,
            augment_scale=self.config.AUGMENT_SCALE,
            augment_translate=self.config.AUGMENT_TRANSLATE,
        )
        print("GIF created successfully!")
        return gif_path
    
    def run_pipeline(self, image_dir=None, output_path="keypoints_sequence.gif"):
        """Run the complete pipeline"""
        return (self
                .load_model()
                .optimize(image_dir)
                .find_indices(image_dir)
                .create_gif(image_dir, output_path))


# Convenience function for quick usage
def run_stable_keypoints(image_dir, output_path="keypoints_sequence.gif", config=None):
    """
    Convenience function to run the complete StableKeypoints pipeline
    
    Args:
        image_dir: Directory containing images
        output_path: Output path for GIF
        config: Optional configuration object
        
    Returns:
        Path to generated GIF
    """
    sk = StableKeypoints(config)
    return sk.run_pipeline(image_dir, output_path)
