import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DragVideoDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        num_max_drags: int = 30,
        num_frames: int = 14,
    ):
        super().__init__()
        
        self.h5 = h5py.File(h5_path, "r")
        self.roots = list(self.h5.keys())

        self.num_max_drags = num_max_drags
        self.num_frames = num_frames

        # required_keys = [
        #     "00b45680f31d47819cee2bab91643195_Take 001",
        #     "003f76fbaedf4074b877fcac7f8d52b7_fly",
        #     "003f76fbaedf4074b877fcac7f8d52b7_metarig|FullJump",
        #     "003f76fbaedf4074b877fcac7f8d52b7_metarig|pick",
        #     "01051d98271941bd971018f50e31fc2f_mixamo.com"
        # ]

        # required_pairs = [(key.split('_')[0], '_'.join(key.split('_')[1:])) for key in required_keys]
    
        self.obj_action_tuples = []
        for obj_idx, obj_root in enumerate(self.roots):
            for action in list(self.h5[obj_root].keys()):
                # if (obj_root, action) in required_pairs:
                self.obj_action_tuples.append((obj_idx, action))

        # self.obj_action_tuples = self.obj_action_tuples[:300]
        print(f"Total number of samples: {len(self)}")

    def __len__(self):
        return len(self.obj_action_tuples)
    
    def get_batch(self, index):
        obj_idx, action = self.obj_action_tuples[index % len(self.obj_action_tuples)]

        obj_root = self.roots[obj_idx]
        obj = self.h5[f"{obj_root}/{action}"]

        views = [obj[view] for view in obj.keys() if view.isdigit()]
        if len(views) == 0:
            raise ValueError(f"No views found in {obj_root}/{action}")
        
        selected_view = views[np.random.randint(len(views))]
        joints_2d = selected_view["joints_2ds"][:] * 2.0
        joints_filter = obj["is_joints_inside_3d"][:]
        joints_2d = joints_2d[:, joints_filter == True]
        
        # Create mask for joints_2d where both x and y are within [0, 512]
        frame_mask = ((joints_2d[..., 0] >= 0) & (joints_2d[..., 0] <= 512) & 
                      (joints_2d[..., 1] >= 0) & (joints_2d[..., 1] <= 512))
        frame_indices = np.where(frame_mask.any(axis=1))[0]
        if len(frame_indices) < self.num_frames:
            raise ValueError("Not enough valid frames: %d, %d" % (len(frame_indices), self.num_frames))

        if len(frame_indices) != self.num_frames:
            start_idx = np.random.randint(0, len(frame_indices) - self.num_frames + 1)
            frame_indices = frame_indices[start_idx:start_idx + self.num_frames]

        drags = torch.from_numpy(joints_2d[frame_indices])

        removed_drags = []
        # 1. Remove parallel drags
        for i in range(drags.shape[1]):
            if i in removed_drags:
                continue
            for j in range(i + 1, drags.shape[1]):
                if torch.norm(drags[:, i] - drags[:, j], dim=-1).sum() <= drags.shape[0] * 20:
                    removed_drags.append(j)
        drags = torch.cat([drags[:, i:i+1] for i in range(drags.shape[1]) if i not in removed_drags], dim=1)

        removed_drags = []
        # 2. Calculate the total displacement of each drag
        displacement = torch.norm(drags[1:] - drags[:-1], dim=-1).sum(dim=0)
        max_drag_idx = displacement.argmax()
        for i in range(drags.shape[1]):
            if i != max_drag_idx:
                if torch.rand(1).item() > displacement[i] / displacement.max() * 2:
                    removed_drags.append(i)
        drags = torch.cat([drags[:, i:i+1] for i in range(drags.shape[1]) if i not in removed_drags], dim=1)
        # print(f"Removed {len(removed_drags)} duplicate drags")

        if drags.shape[1] < 10:
            raise ValueError(f"Not enough joints: {drags.shape[1]}")

        if drags.shape[1] > self.num_max_drags:
            # Randomly select num_max_drags indices
            rand = torch.randperm(drags.shape[1])[:self.num_max_drags]
            drags = drags[:, rand]

        embedding = torch.from_numpy(selected_view["embeddings"][frame_indices])
        latent_means = torch.from_numpy(selected_view["latent_means"][frame_indices])
        latent_stds = torch.from_numpy(selected_view["latent_stds"][frame_indices])

        latents = []
        for latent_mean, latent_std in zip(latent_means, latent_stds):
            latents.append(latent_mean + latent_std * torch.randn_like(latent_mean))
        latents = torch.stack(latents)

        cond_latent = latent_means[0]
        embedding = embedding[0]
        
        # Normalize drags to [0, 1]
        drags = drags / 512.
        drags = torch.cat([drags[0:1].expand_as(drags), drags], dim=-1)
        drags = torch.cat([drags, torch.zeros(self.num_frames, self.num_max_drags - drags.shape[1], 4)], dim=1)

        return latents, cond_latent, embedding, drags, obj_root, action
    
    def __getitem__(self, index):
        while True:
            try:
                latents, cond_latent, embedding, drags, obj_root, action = self.get_batch(index)
                break
            except ValueError as e:
                # print(f"Error at index {index}: {e}")
                index = index + 1
        return dict(
            latents=latents.to(dtype=torch.float16).mul_(0.18215),
            cond_latent=cond_latent.to(dtype=torch.float16),
            embedding=embedding.to(dtype=torch.float16),
            drags=drags.to(dtype=torch.float16),
            name=f"{obj_root}_{action}"
        )


if __name__ == "__main__":
    dataset = DragVideoDataset("/home/c_capzw/c_cape3d/latent_states_embeddings_and_images_test.h5")

    # Find min/max values across all dimensions in the dataset
    mins, maxs = {}, {}
    for i in range(len(dataset)):
        data = dataset[i]
        print(f'Processing {i + 1}/{len(dataset)}')
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if key not in mins:
                    mins[key] = value.clone()
                    maxs[key] = value.clone()
                else:
                    mins[key] = torch.minimum(mins[key], value)
                    maxs[key] = torch.maximum(maxs[key], value)

    print("\nDataset value ranges:")
    for key in mins:
        print(f"{key}: min={mins[key].min()}, max={maxs[key].max()}, shape={dataset[0][key].shape}")
