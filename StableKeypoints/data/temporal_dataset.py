"""
Temporal dataset for loading consecutive frames for temporal consistency training
"""

import os
import torch
from PIL import Image as PILImage
from torchvision import transforms
from torch.utils.data import Dataset


class TemporalDataset(Dataset):
    """Dataset that loads consecutive frames for temporal consistency training"""
    
    def __init__(self, data_root, image_size, frame_gap=1):
        """
        Args:
            data_root: Root directory containing video frames
            image_size: Size to resize images to
            frame_gap: Gap between consecutive frames (1 for adjacent frames)
        """
        super().__init__()
        self.data_root = data_root
        self.frame_gap = frame_gap
        
        # Get all image files and sort them
        self.image_files = [f for f in os.listdir(data_root) 
                           if os.path.isfile(os.path.join(data_root, f)) 
                           and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()
        
        # Filter out files that don't have a consecutive pair
        self.valid_indices = []
        for i in range(len(self.image_files) - frame_gap):
            self.valid_indices.append(i)
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        """
        Returns a pair of consecutive frames
        """
        base_idx = self.valid_indices[idx]
        
        # Load frame t
        img_path_t = os.path.join(self.data_root, self.image_files[base_idx])
        img_t = PILImage.open(img_path_t).convert('RGB')
        img_t = self.transform(img_t)
        
        # Load frame t+gap
        img_path_t1 = os.path.join(self.data_root, self.image_files[base_idx + self.frame_gap])
        img_t1 = PILImage.open(img_path_t1).convert('RGB')
        img_t1 = self.transform(img_t1)
        
        sample = {
            'frame_t': img_t,
            'frame_t1': img_t1,
            'img': img_t,  # For compatibility
            'kpts': torch.zeros(15, 2),  # Placeholder for keypoints
            'visibility': torch.zeros(15),  # Placeholder for visibility
            'name_t': self.image_files[base_idx],
            'name_t1': self.image_files[base_idx + self.frame_gap],
            'name': self.image_files[base_idx],  # For compatibility
        }
        return sample

    def __len__(self):
        return len(self.valid_indices)
