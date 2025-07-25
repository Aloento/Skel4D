"""
CSV utilities for StableKeypoints
"""

import csv


def save_keypoints_to_csv(keypoints_data, indices, output_csv="keypoints.csv"):
    """
    Save keypoint coordinates to CSV file
    
    Args:
        keypoints_data: List of frame data from extract_keypoints()
        indices: Keypoint indices tensor/array
        output_csv: Output CSV file path
        
    Returns:
        Path to generated CSV file
    """
    # Prepare CSV data
    csv_data = []
    headers = ['frame_id', 'image_name']
    
    # Add headers for each keypoint (x, y coordinates)
    for i in range(len(indices)):
        headers.extend([f'keypoint_{i}_x', f'keypoint_{i}_y'])
    
    for frame_data in keypoints_data:
        # Prepare row data
        row_data = [frame_data["frame_idx"], frame_data["image_name"]]
        for i in range(len(frame_data["keypoints"])):
            y, x = frame_data["keypoints"][i]  # Note: find_max_pixel returns [y, x]
            row_data.extend([float(x), float(y)])  # Store as [x, y] for clarity
        
        csv_data.append(row_data)
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(csv_data)
    
    print(f"Keypoints saved to {output_csv}")
    print(f"CSV contains {len(csv_data)} frames with {len(indices)} keypoints each")
    
    return output_csv
