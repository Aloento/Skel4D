"""
GIF creation utilities for StableKeypoints
"""

import matplotlib.pyplot as plt
import imageio
import numpy as np
from tqdm.auto import tqdm


def create_gif(keypoints_data, output_path="keypoints_sequence.gif", gif_fps=10):
    """
    Create GIF visualization of keypoint detection from extracted keypoint data
    
    Args:
        keypoints_data: List of dictionaries containing frame data with keys:
                       ["frame_idx", "image_name", "img", "keypoints"]
        output_path: Output path for the GIF file
        gif_fps: Frames per second for the GIF
        
    Returns:
        str: Path to the generated GIF file
    """
    print("Creating GIF visualization...")
    
    # Disable interactive display
    plt.ioff()
    
    # Create a temporary list to store all frames
    frames = []
    
    # Set up color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    total_frames = len(keypoints_data)
    
    for i, frame_data in enumerate(tqdm(keypoints_data, desc="Generating GIF frames")):
        img = frame_data["img"]
        keypoints = frame_data["keypoints"]
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display image and keypoints
        ax.imshow(img.numpy().transpose(1, 2, 0))
        num_points = keypoints.shape[0]
        
        for j in range(num_points):
            color = colors[j % len(colors)]
            y, x = keypoints[j]  # keypoints are [y, x] normalized
            x, y = x * 512, y * 512  # Convert to pixel coordinates
            ax.scatter(x, y, color=color, marker=f"${j}$", s=300)
        
        # Add frame number information
        ax.text(10, 30, f"Frame: {i+1}/{total_frames}", 
                fontsize=12, color='white', 
                bbox=dict(facecolor='black', alpha=0.7))
        
        ax.axis("off")
        fig.tight_layout(pad=0)
        
        # Convert figure to image
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Add image to frames list
        frames.append(image_from_plot)
        
        # Close figure to free memory
        plt.close(fig)
    
    # Generate GIF using imageio
    print(f"Generating GIF animation, saving to {output_path}...")
    imageio.mimsave(output_path, frames, fps=gif_fps)
    
    print(f"GIF animation successfully saved to {output_path}")
    return output_path
