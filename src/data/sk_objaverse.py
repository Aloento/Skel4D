import os
import json
import numpy as np
import webdataset as wds
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from pathlib import Path
import h5py
import torchvision
from torchvision import transforms
from einops import rearrange

from src.utils.train_util import instantiate_from_config


class StableKeypointDataModuleFromConfig(pl.LightningDataModule):
    """
    DataModule that works with data from HDF5 format.
    """
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):
        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        sampler = DistributedSampler(self.datasets['train'])
        return torch.utils.data.DataLoader(
            self.datasets['train'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            sampler=sampler,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        sampler = DistributedSampler(self.datasets['validation'])
        return torch.utils.data.DataLoader(
            self.datasets['validation'], 
            batch_size=4, 
            num_workers=self.num_workers, 
            shuffle=False, 
            sampler=sampler,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets['test'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        """Custom collate function to handle the data format for Zero123++"""
        # Handle the Zero123++ format
        cond_imgs = torch.stack([item['cond_imgs'] for item in batch])  # (B, C, H, W)
        target_imgs = torch.stack([item['target_imgs'] for item in batch])  # (B, N, C, H, W)
        #skeleton_imgs = torch.stack([item['skeleton_imgs'] for item in batch])  # (B, N, C, H, W)
        
        # Keep target_imgs in 5D format as expected by the model
        return {
            'cond_imgs': cond_imgs,           # (B, C, H, W) - condition images
            'target_imgs': target_imgs,       # (B, N, C, H, W) - target images in original format
            #'skeleton_imgs': skeleton_imgs,   # (B, N, C, H, W) - skeleton guidance
            'filename': [item['filename'] for item in batch]
        }


class StableKeypointObjaverseData(Dataset):
    """
    Standalone dataset class that reads data from HDF5 format
    and provides data compatible with stable keypoint detection.
    """
    def __init__(self,
                 root_dir='',
                 debug=False,
                 validation=False,
                 max_views=6,  # Limit to 6 views for Zero123++ compatibility
                 ):
        super().__init__()
        
        self.root_dir = Path(root_dir)        
        self.max_views = max_views
        self.debug = debug
        self.validation = validation
        self.total_view = 12
        self.load_view = 6

        self.read_samples()

        self.opengl_to_colmap = torch.tensor([[  1,  0,  0,  0],
                                              [  0, -1,  0,  0],
                                              [  0,  0, -1,  0],
                                              [  0,  0,  0,  1]], dtype=torch.float32)

        # Image transforms
        image_transforms = [
            torchvision.transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Lambda(self.rearrange_func)
        ]
        self.image_transforms = transforms.Compose(image_transforms)

    def rearrange_func(self, x):
        return rearrange(x * 2. - 1., 'c h w -> h w c')

    def read_samples(self):
        hdf5_file = os.path.join(self.root_dir, 'val.hdf5' if self.validation else 'train.hdf5')        
        self.hdf5_file_path = hdf5_file
        self.h5f = None

        self.object_names = self._load_object_names()
        exlude = ['bac0fce4bc37425fb3a4596d02b95514',
                  '13262e8213d648f8926d718e5796e4c7',
                  '291d73dd9a1746c1947836bce1f446ab',
                  '30ea70178d624b87a39553108db61bca',
                  '63cb3f33ad7f4b739811409d802f921f',
                  '9c5317ed6f78424cb1e772d46308261d',
                  '8ddd8d6309244f629b5ee3eed72d94a4']
        self.object_names = [obj for obj in self.object_names if obj not in exlude]
        '''if self.validation:
            self.object_names = self.object_names[:50]
        else:
            self.object_names = self.object_names[:600]'''

    def _load_object_names(self):
        with h5py.File(self.hdf5_file_path, 'r') as f:
            return list(f.keys())

    def __len__(self):
        return len(self.object_names)

    def __enter__(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.hdf5_file_path, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.h5f is not None:
            self.h5f.close()
            self.h5f = None

    def load_im_hdf5(self, img):
        '''
        replace background pixel with random color in rendering
        '''        
        color = [1., 1., 1., 1.]   
        img = np.array(img)
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def process_img(self, img):
        # Convert the image data to a PIL Image before processing
        img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        return self.image_transforms(img)

    def pre_data(self, data):
        '''
        load the data for given filename 
        '''        
        imgs = []
        #skels = []
        w2cs = []
        intrinsics = []

        view_ids = [d.split('_')[0] for d in data.keys() if d.endswith('_image')]
        views = len(view_ids)
        index = range(views) if self.validation else torch.randperm(views)
        
        # Find the closer views
        for i in index[:self.load_view]:
            img = self.process_img(self.load_im_hdf5(data[f'{view_ids[i]}_image'])).unsqueeze(0)
            imgs.append(img)
            #skel = self.process_img(self.load_im_hdf5(data[f'{view_ids[i]}_skeleton'])).unsqueeze(0)
            #skels.append(skel)
            w2c_gl = data[f'{view_ids[i]}_camera']
            w2cs.append(w2c_gl)
            focal = .5 / np.tan(.5 * 0.8575560450553894)
            intrinsics.append(np.array([[focal, 0.0, 1.0 / 2.0],
                                        [0.0, focal, 1.0 / 2.0],
                                        [0.0, 0.0, 1.0]]))
            
        imgs = torch.cat(imgs)
        #skels = torch.cat(skels)
        intrinsics = torch.tensor(np.array(intrinsics)).to(imgs)
        w2cs = torch.tensor(np.array(w2cs)).to(imgs)
        w2cs_gl = torch.eye(4).unsqueeze(0).repeat(imgs.size(0),1,1)
        w2cs_gl[:,:3,:] = w2cs
        # camera poses in .npy files are in OpenGL convention: 
        #     x right, y up, z into the camera (backward),
        # need to transform to COLMAP / OpenCV:
        #     x right, y down, z away from the camera (forward)
        w2cs = torch.einsum('nj, bjm-> bnm', self.opengl_to_colmap, w2cs_gl)
        c2ws = torch.linalg.inv(w2cs)
        camera_centers = c2ws[:, :3, 3].clone()
        # fix the distance of the source camera to the object / world center
        assert torch.norm(camera_centers[0]) > 1e-5
        translation_scaling_factor = 2.0 / torch.norm(camera_centers[0])
        w2cs[:, :3, 3] *= translation_scaling_factor
        c2ws[:, :3, 3] *= translation_scaling_factor
        camera_centers *= translation_scaling_factor
        #return imgs, skels, w2cs, c2ws, intrinsics
        return imgs, w2cs, c2ws, intrinsics
        
    def __getitem__(self, index):
        # Ensure the HDF5 file is open
        if self.h5f is None:
            self.h5f = h5py.File(self.hdf5_file_path, 'r')

        try:
            object_group = self.h5f[self.object_names[index]]
            frame_name = np.random.choice(list(object_group.keys()))
            data = object_group[frame_name]
            imgs, w2cs, c2ws, intrinsics = self.pre_data(data)

            if len(data['joints']) == 0:
                print('error in loading data idjdjdjdjdj', flush=True)
                print('filename:', self.object_names[index], flush=True)
            
            # Check the shapes of the data
            if imgs.shape[0] != self.load_view or w2cs.shape[0] != self.load_view or c2ws.shape[0] != self.load_view or intrinsics.shape[0] != self.load_view:
                print('error in loading data', flush=True)
                print('filename:', self.object_names[index], flush=True)
                print('imgs.shape', imgs.shape, flush=True)               

            
        except:
            print('error in loading data', flush=True)
            print('filename:', self.object_names[index], flush=True)
        # Limit to max_views if we have more
        images = imgs
        
        filename = self.object_names[index]
        
        if images.shape[0] >= self.max_views:
            # Always include first view, then sample others
            indices = [0] + torch.randperm(images.shape[0] - 1)[:self.max_views-1].tolist()
            indices = sorted([i + (1 if i >= 0 else 0) for i in indices[1:]] + [0])
            
            images = images[indices]
            #skeletons = skeletons[indices]
            w2cs = w2cs[indices]
            c2ws = c2ws[indices]
            intrinsics = intrinsics[indices]

        else:
            # If fewer views, just repeat the last view to fill up to max_views
            while images.shape[0] < self.max_views:
                images = torch.cat([images, images[-1].unsqueeze(0)], dim=0)
                #skeletons = torch.cat([skeletons, skeletons[-1].unsqueeze(0)], dim=0)
                w2cs = torch.cat([w2cs, w2cs[-1].unsqueeze(0)], dim=0)
                c2ws = torch.cat([c2ws, c2ws[-1].unsqueeze(0)], dim=0)
                intrinsics = torch.cat([intrinsics, intrinsics[-1].unsqueeze(0)], dim=0)
            print(f'Warning: Fewer than {self.max_views} views for {filename}, repeating last view.', flush=True)
        
        # Convert format to match Zero123++ expectations
        n_views = images.shape[0]
        
        if n_views >= 2:
            # Convert from (H, W, C) to (C, H, W) for compatibility
            images_chw = images.permute(0, 3, 1, 2)  # (N, C, H, W)
            #skeletons_chw = skeletons.permute(0, 3, 1, 2)  # (N, C, H, W)
            
            # Convert from [-1, 1] to [0, 1] range
            images_norm = (images_chw + 1.0) / 2.0
            #skeletons_norm = (skeletons_chw + 1.0) / 2.0

            return {
                'cond_imgs': images_norm[0],        # (C, H, W) - first view as condition
                'target_imgs': images_norm,     # (N, C, H, W) - remaining views as targets
                #'skeleton_imgs': skeletons_norm, # (N, C, H, W) - skeleton for target views
                'filename': filename
            }
        else:
            asd


