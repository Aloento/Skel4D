# Import the modules
from StableKeypoints.api import StableKeypoints
from StableKeypoints.config import Config

def main():
    # Configure parameters
    config = Config()
    config.NUM_KEYPOINTS = 15
    config.NUM_OPTIMIZATION_STEPS = 2000
    config.BATCH_SIZE = 1
    config.FURTHEST_POINT_NUM_SAMPLES = 15
    config.IMAGE_DIR = "/home/c_capzw/c_cape3d/data/rendered/objaverse/66a62fc9ab97415f85b6322c103f8e1e/Take001"
    
    # Create StableKeypoints instance
    sk = StableKeypoints(config)
    
    # Run the complete pipeline
    gif_path = sk.run_pipeline(
        image_dir=config.IMAGE_DIR,
        output_path="keypoints_sequence.gif"
    )
    
    print(f"Generated GIF: {gif_path}")

if __name__ == "__main__":
    main()
