"""
Visualization utilities for StableKeypoints
"""

import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ..data.dataset import CustomDataset
from ..data.transforms import RandomAffineWithInverse
from ..utils.image_utils import run_and_find_attn
from ..utils.keypoint_utils import find_max_pixel


@torch.no_grad()
def run_image_with_context_augmented(
    ldm,
    image,
    context,
    indices,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    layers=[0, 1, 2, 3, 4, 5],
    augmentation_iterations=20,
    noise_level=-1,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    controllers=None,
    num_gpus=1,
    upsample_res=512,
):
    """Run image through model with augmentation for robust keypoint detection"""
    
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()

    num_samples = torch.zeros(len(indices), upsample_res, upsample_res).to(device)
    sum_samples = torch.zeros(len(indices), upsample_res, upsample_res).to(device)

    invertible_transform = RandomAffineWithInverse(
        degrees=augment_degrees,
        scale=augment_scale,
        translate=augment_translate,
    )

    for i in range(augmentation_iterations//num_gpus):

        augmented_img = (
            invertible_transform(torch.tensor(image)[None].repeat(num_gpus, 1, 1, 1).permute(0, 3, 1, 2))
            .permute(0, 2, 3, 1)
            .numpy()
        )

        attn_maps = run_and_find_attn(
            ldm,
            augmented_img,
            context,
            layers=layers,
            noise_level=noise_level,
            from_where=from_where,
            upsample_res=upsample_res,
            device=device,
            controllers=controllers,
            indices=indices.cpu(),
        )

        attn_maps = torch.stack([map.to("cuda:0") for map in attn_maps])

        _num_samples = invertible_transform.inverse(torch.ones_like(attn_maps))
        _sum_samples = invertible_transform.inverse(attn_maps)

        num_samples += _num_samples.sum(dim=0)
        sum_samples += _sum_samples.sum(dim=0)

    # visualize sum_samples/num_samples
    attention_maps = sum_samples / num_samples

    # replace all nans with 0s
    attention_maps[attention_maps != attention_maps] = 0

    return attention_maps


@torch.no_grad()
def create_keypoints_gif(
    ldm,
    contexts,
    indices,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    layers=[0, 1, 2, 3, 4, 5],
    noise_level=-1,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    augmentation_iterations=20,
    dataset_loc="~",
    controllers=None,
    num_gpus=1,
    output_path="keypoints.gif",
    fps=10,
    max_frames=None  # 限制处理的图片数量，设为None处理所有图片
):
    """Create animated GIF showing keypoint detection across image sequence"""
    
    # 禁用交互式显示
    plt.ioff()
    
    dataset = CustomDataset(data_root=dataset_loc, image_size=512)
    total_frames = len(dataset) if max_frames is None else min(max_frames, len(dataset))
    
    print(f"处理 {total_frames} 张图片并生成GIF...")
    
    # 创建一个临时列表存储所有帧
    frames = []
    
    # 设置颜色循环
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for i in tqdm(range(total_frames)):
        # 获取图片
        batch = dataset[i]
        img = batch["img"]
        
        # 创建一个新的图形
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 检测关键点
        maps = []
        for j in range(len(contexts) if isinstance(contexts, list) else 1):
            context = contexts[j] if isinstance(contexts, list) else contexts
            map = run_image_with_context_augmented(
                ldm,
                img,
                context,
                indices.cpu(),
                device=device,
                from_where=from_where,
                layers=layers,
                noise_level=noise_level,
                augment_degrees=augment_degrees,
                augment_scale=augment_scale,
                augment_translate=augment_translate,
                augmentation_iterations=augmentation_iterations,
                controllers=controllers,
                num_gpus=num_gpus,
                upsample_res=512,
            )
            maps.append(map)
        
        maps = torch.stack(maps)
        map = torch.mean(maps, dim=0)
        
        # 找到关键点
        point = find_max_pixel(map) / 512.0
        point = point.cpu()
        
        # 显示图像和关键点
        ax.imshow(img.numpy().transpose(1, 2, 0))
        num_points = point.shape[0]
        
        for j in range(num_points):
            color = colors[j % len(colors)]
            x, y = point[j, 1] * 512, point[j, 0] * 512
            ax.scatter(x, y, color=color, marker=f"${j}$", s=300)
        
        # 添加帧号信息
        ax.text(10, 30, f"Frame: {i+1}/{total_frames}", 
                fontsize=12, color='white', 
                bbox=dict(facecolor='black', alpha=0.7))
        
        ax.axis("off")
        fig.tight_layout(pad=0)
        
        # 将图形转换为图像
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # 将图像添加到帧列表
        frames.append(image_from_plot)
        
        # 关闭图形以释放内存
        plt.close(fig)
    
    # 使用imageio生成GIF
    print(f"生成GIF动画，保存至 {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    
    print(f"GIF动画已成功保存至 {output_path}")
    return output_path
