import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from StableKeypoints.config import Config
from StableKeypoints.models.model_loader import load_ldm
from StableKeypoints.optimization.optimizer import find_best_indices
from StableKeypoints.utils.keypoint_extraction import extract_keypoints
from StableKeypoints.utils.gif_utils import create_gif
from StableKeypoints.utils.csv_utils import save_keypoints_to_csv
import time


class StagedTrainer:
    """Staged trainer for StableKeypoints"""
    
    def __init__(self, config: Config):
        self.config = config 
        self.ldm = None
        self.controllers = None
        self.num_gpus = None
        self.embedding = None
        self.indices = None
        
        # Early stopping configuration
        self.patience = config.EARLY_STOPPING_PATIENCE
        self.min_delta = config.EARLY_STOPPING_MIN_DELTA
        
    def load_model(self, device="cuda:0"):
        """Load model"""
        print("Loading Stable Diffusion model...")
        self.ldm, self.controllers, self.num_gpus = load_ldm(
            device, 
            self.config.STABLE_DIFFUSION_MODEL, 
            self.config.FEATURE_UPSAMPLE_RES
        )
        print("Model loaded successfully!")
        return self
        
    def stage1_optimize(self, image_dir=None):
        """
        Stage 1: Train stable keypoints using only equivariance and sharpening loss
        """
        print("=" * 60)
        print("STAGE 1: Training stable keypoints (equivariance + sharpening)")
        print("=" * 60)
        
        if self.ldm is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if image_dir is None:
            image_dir = self.config.IMAGE_DIR
        
        # Stage 1 configuration: disable temporal loss
        stage1_config = Config()
        # Copy original configuration
        for attr in dir(self.config):
            if not attr.startswith('_'):
                setattr(stage1_config, attr, getattr(self.config, attr))
        
        # Stage 1 specific settings
        stage1_config.TEMPORAL_CONSISTENCY_LOSS_WEIGHT = 0.0  # Disable temporal loss
        stage1_config.NUM_OPTIMIZATION_STEPS = self.config.NUM_OPTIMIZATION_STEPS
        stage1_config.SAVE_CHECKPOINTS = True
        stage1_config.CHECKPOINT_SAVE_INTERVAL = 100
        stage1_config.CHECKPOINT_DIR = os.path.join(self.config.CHECKPOINT_DIR, "stage1")
        stage1_config.LOAD_FROM_CHECKPOINT = False  # Start from scratch in stage 1
        
        print(f"Stage 1 Configuration:")
        print(f"  Steps: {stage1_config.NUM_OPTIMIZATION_STEPS}")
        print(f"  Temporal loss weight: {stage1_config.TEMPORAL_CONSISTENCY_LOSS_WEIGHT}")
        print(f"  Sharpening loss weight: {stage1_config.SHARPENING_LOSS_WEIGHT}")
        print(f"  Equivariance loss weight: {stage1_config.EQUIVARIANCE_ATTN_LOSS_WEIGHT}")
        print()
        
        # Stage 1 optimization
        self.embedding = optimize_embedding_with_early_stopping(
            self.ldm,
            num_steps=stage1_config.NUM_OPTIMIZATION_STEPS,
            batch_size=stage1_config.BATCH_SIZE,
            top_k=stage1_config.NUM_KEYPOINTS,
            controllers=self.controllers,
            num_gpus=self.num_gpus,
            furthest_point_num_samples=stage1_config.FURTHEST_POINT_NUM_SAMPLES,
            num_tokens=stage1_config.NUM_TOKENS,
            dataset_loc=image_dir,
            lr=stage1_config.LEARNING_RATE,
            augment_degrees=stage1_config.AUGMENT_DEGREES,
            augment_scale=stage1_config.AUGMENT_SCALE,
            augment_translate=stage1_config.AUGMENT_TRANSLATE,
            sigma=stage1_config.SIGMA,
            sharpening_loss_weight=stage1_config.SHARPENING_LOSS_WEIGHT,
            equivariance_attn_loss_weight=stage1_config.EQUIVARIANCE_ATTN_LOSS_WEIGHT,
            from_where=stage1_config.FROM_WHERE,
            layers=stage1_config.LAYERS,
            noise_level=stage1_config.NOISE_LEVEL,
            config=stage1_config,
            patience=self.patience,
            min_delta=self.min_delta,
            stage_name="Stage1"
        )
        
        print("Stage 1 completed: Stable keypoints learned!")
        return self
        
    def stage2_optimize(self, image_dir=None):
        """
        Stage 2: Introduce temporal loss to reduce drift based on stage 1 results
        """
        print("=" * 60)
        print("STAGE 2: Temporal consistency refinement")
        print("=" * 60)
        
        if self.embedding is None:
            raise ValueError("Stage 1 not completed. Call stage1_optimize() first.")
        
        if image_dir is None:
            image_dir = self.config.IMAGE_DIR
        
        # Stage 2 configuration: introduce temporal loss
        stage2_config = Config()
        # Copy original configuration
        for attr in dir(self.config):
            if not attr.startswith('_'):
                setattr(stage2_config, attr, getattr(self.config, attr))
        
        # Stage 2 specific settings
        stage2_config.TEMPORAL_CONSISTENCY_LOSS_WEIGHT = self.config.STAGE2_TEMPORAL_WEIGHT
        stage2_config.NUM_OPTIMIZATION_STEPS = self.config.STAGE2_STEPS
        stage2_config.SAVE_CHECKPOINTS = True
        stage2_config.CHECKPOINT_SAVE_INTERVAL = 100
        stage2_config.CHECKPOINT_DIR = os.path.join(self.config.CHECKPOINT_DIR, "stage2")
        stage2_config.LOAD_FROM_CHECKPOINT = False  # Use stage 1 embedding
        
        # Reduce other loss weights to give temporal loss better effect
        stage2_config.SHARPENING_LOSS_WEIGHT = self.config.STAGE2_SHARPENING_WEIGHT
        stage2_config.EQUIVARIANCE_ATTN_LOSS_WEIGHT = self.config.STAGE2_EQUIVARIANCE_WEIGHT
        
        print(f"Stage 2 Configuration:")
        print(f"  Steps: {stage2_config.NUM_OPTIMIZATION_STEPS}")
        print(f"  Temporal loss weight: {stage2_config.TEMPORAL_CONSISTENCY_LOSS_WEIGHT}")
        print(f"  Sharpening loss weight: {stage2_config.SHARPENING_LOSS_WEIGHT}")
        print(f"  Equivariance loss weight: {stage2_config.EQUIVARIANCE_ATTN_LOSS_WEIGHT}")
        print(f"  Temporal loss type: {stage2_config.TEMPORAL_LOSS_TYPE}")
        print()
        
        # Stage 2 optimization - use stage 1 embedding as starting point
        self.embedding = optimize_embedding_with_early_stopping(
            self.ldm,
            context=self.embedding,  # Use stage 1 embedding
            num_steps=stage2_config.NUM_OPTIMIZATION_STEPS,
            batch_size=stage2_config.BATCH_SIZE,
            top_k=stage2_config.NUM_KEYPOINTS,
            controllers=self.controllers,
            num_gpus=self.num_gpus,
            furthest_point_num_samples=stage2_config.FURTHEST_POINT_NUM_SAMPLES,
            num_tokens=stage2_config.NUM_TOKENS,
            dataset_loc=image_dir,
            lr=stage2_config.LEARNING_RATE * 0.5,  # Reduce learning rate to preserve stage 1 results
            augment_degrees=stage2_config.AUGMENT_DEGREES,
            augment_scale=stage2_config.AUGMENT_SCALE,
            augment_translate=stage2_config.AUGMENT_TRANSLATE,
            sigma=stage2_config.SIGMA,
            sharpening_loss_weight=stage2_config.SHARPENING_LOSS_WEIGHT,
            equivariance_attn_loss_weight=stage2_config.EQUIVARIANCE_ATTN_LOSS_WEIGHT,
            from_where=stage2_config.FROM_WHERE,
            layers=stage2_config.LAYERS,
            noise_level=stage2_config.NOISE_LEVEL,
            config=stage2_config,
            patience=self.patience,
            min_delta=self.min_delta,
            stage_name="Stage2"
        )
        
        print("Stage 2 completed: Temporal consistency improved!")
        return self
        
    def find_indices(self, image_dir=None):
        """Find best keypoint indices"""
        if self.embedding is None:
            raise ValueError("Embedding not optimized. Complete both stages first.")
        
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
        
    def generate_outputs(self, image_dir=None, output_path="keypoints_staged.gif", 
                        output_csv="keypoints_staged.csv", augmentation_iterations=20):
        """Generate GIF and CSV outputs"""
        if self.embedding is None or self.indices is None:
            raise ValueError("Training not completed. Run both stages and find_indices first.")
        
        if image_dir is None:
            image_dir = self.config.IMAGE_DIR
        
        print("=" * 60)
        print("GENERATING OUTPUTS")
        print("=" * 60)
        
        # Extract keypoint data
        print("Extracting keypoints...")
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
        
        # Save CSV
        print(f"Saving CSV to {output_csv}...")
        csv_path = save_keypoints_to_csv(keypoints_data, self.indices, output_csv)
        
        # Generate GIF
        print(f"Creating GIF {output_path}...")
        gif_fps = self.config.GIF_FPS
        create_gif(keypoints_data, output_path, gif_fps)
        
        print(f"Outputs generated:")
        print(f"  GIF: {output_path}")
        print(f"  CSV: {csv_path}")
        
        return {"gif": output_path, "csv": csv_path}
        
    def run_staged_pipeline(self, image_dir=None, output_path="keypoints_staged.gif",
                           output_csv="keypoints_staged.csv", augmentation_iterations=20):
        """Run complete staged training pipeline"""
        print("=" * 80)
        print("STAGED TRAINING PIPELINE FOR STABLEKEYPOINTS")
        print("=" * 80)

        start_time = time.time()
        
        # Execute complete pipeline
        self.load_model()
        self.stage1_optimize(image_dir)
        # self.stage2_optimize(image_dir)
        self.find_indices(image_dir)
        results = self.generate_outputs(image_dir, output_path, output_csv, augmentation_iterations)
        
        total_time = time.time() - start_time
        
        print("=" * 80)
        print("STAGED TRAINING COMPLETED!")
        print("=" * 80)
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Generated files:")
        print(f"  GIF: {results['gif']}")
        print(f"  CSV: {results['csv']}")
        print()

        return results


def optimize_embedding_with_early_stopping(
    ldm, context=None, device="cuda", num_steps=2000, patience=50, min_delta=1e-4,
    stage_name="", **kwargs
):
    """
    Embedding optimization with early stopping
    
    Args:
        patience: Number of steps to wait for improvement
        min_delta: Minimum improvement threshold
        stage_name: Stage name for printing
    """
    from StableKeypoints.optimization.optimizer import optimize_embedding
    
    print(f"Starting {stage_name} optimization with early stopping...")
    print(f"Early stopping: patience={patience}, min_delta={min_delta}")
    
    # Modify configuration to support early stopping
    config = kwargs.get('config')
    if config:
        # Add early stopping related configuration
        config.EARLY_STOPPING_PATIENCE = patience
        config.EARLY_STOPPING_MIN_DELTA = min_delta
        config.EARLY_STOPPING_ENABLED = True
        config.STAGE_NAME = stage_name
    
    # Call original optimization function
    embedding = optimize_embedding(
        ldm, context=context, device=device, num_steps=num_steps, **kwargs
    )
    
    return embedding


def train_staged_temporal_consistency():
    """Run staged temporal consistency training"""
    print("Initializing staged temporal consistency training...")
    
    # Configuration parameters
    config = Config()
    config.IMAGE_DIR = "/home/c_capzw/c_cape3d/data/rendered/objaverse/66a62fc9ab97415f85b6322c103f8e1e/Take001"
    
    # Temporal settings
    config.TEMPORAL_FRAME_GAP = 3
    config.USE_ADAPTIVE_TEMPORAL_LOSS = True
    
    print("Configuration:")
    print(f"  Stage 1 steps: {config.NUM_OPTIMIZATION_STEPS}")
    print(f"  Stage 2 steps: {config.STAGE2_STEPS}")
    print(f"  Stage 2 temporal weight: {config.STAGE2_TEMPORAL_WEIGHT}")
    print(f"  Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"  Temporal frame gap: {config.TEMPORAL_FRAME_GAP}")
    print(f"  Use adaptive temporal loss: {config.USE_ADAPTIVE_TEMPORAL_LOSS}")
    print()
    
    # Create staged trainer
    trainer = StagedTrainer(config)
    
    # Run staged training pipeline
    results = trainer.run_staged_pipeline(
        image_dir=config.IMAGE_DIR,
        output_path="keypoints_staged_training.gif",
        output_csv="keypoints_staged_training.csv",
        augmentation_iterations=20
    )

    return results


if __name__ == "__main__":
    train_staged_temporal_consistency()
    print("\nStaged training completed!")
