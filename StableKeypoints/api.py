"""
Main API for StableKeypoints
"""

from .config import Config
from .models.model_loader import load_ldm
from .optimization.optimizer import optimize_embedding, find_best_indices
from .optimization.checkpoint import list_checkpoints, find_latest_checkpoint, load_checkpoint
from .utils.gif_utils import create_gif
from .utils.csv_utils import save_keypoints_to_csv
from .utils.keypoint_extraction import extract_keypoints


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
            config=self.config,
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
    
    def run_pipeline(self, image_dir=None, output_path="keypoints_sequence.gif", output_csv="keypoints.csv", augmentation_iterations=20):
        """Run the complete pipeline with both GIF and CSV output"""
        self.load_model()
        self.optimize(image_dir)
        self.find_indices(image_dir)
        
        if image_dir is None:
            image_dir = self.config.IMAGE_DIR
        
        # Extract keypoints data using utils function
        keypoints_data = extract_keypoints(
            self.ldm,
            self.embedding,
            self.indices,
            self.config,
            image_dir,
            self.controllers,
            self.num_gpus,
            augmentation_iterations
        )
        
        # Save to CSV
        csv_path = save_keypoints_to_csv(keypoints_data, self.indices, output_csv)
        
        # Generate GIF from extracted data
        gif_fps = getattr(self.config, 'GIF_FPS', 10)
        create_gif(keypoints_data, output_path, gif_fps)
        
        return {"gif": output_path, "csv": csv_path}
