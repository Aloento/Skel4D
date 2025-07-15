# Import the modules
from StableKeypoints.api import StableKeypoints
from StableKeypoints.config import Config

def main():
    # Configure parameters
    config = Config()
    config.TEMPORAL_CONSISTENCY_LOSS_WEIGHT = 0
    config.IMAGE_DIR = "/home/c_capzw/c_cape3d/data/rendered/objaverse/66a62fc9ab97415f85b6322c103f8e1e/Take001"
    
    # Create StableKeypoints instance
    sk = StableKeypoints(config)
    
    # Run the complete pipeline (optimized to avoid duplicate computation)
    results = sk.run_pipeline(
        image_dir=config.IMAGE_DIR,
        output_path="keypoints_sequence1.gif",
        output_csv="keypoints.csv",
        augmentation_iterations=20
    )
    
    print(f"Generated GIF: {results['gif']}")
    print(f"Generated CSV: {results['csv']}")

if __name__ == "__main__":
    main()
