#!/usr/bin/env python3
"""
Zero123Plus + StableKeypoints 演示脚本

这个脚本展示了如何使用条件探针在Zero123Plus中实现无监督关键点学习。
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import argparse
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_demo_image(size=(256, 256)):
    """创建一个演示图像（简单的几何形状）"""
    image = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(image)
    
    # 绘制一些形状作为演示
    # 圆形
    draw.ellipse([50, 50, 100, 100], fill='red', outline='black', width=2)
    
    # 矩形
    draw.rectangle([150, 80, 200, 130], fill='blue', outline='black', width=2)
    
    # 三角形（近似）
    points = [(100, 180), (130, 220), (70, 220)]
    draw.polygon(points, fill='green', outline='black', width=2)
    
    return image


def simulate_attention_and_keypoints(num_probes=8, spatial_size=16):
    """
    模拟注意力图和关键点提取过程
    
    这个函数演示了核心的关键点提取逻辑，
    使用模拟的注意力图而不是真实的Zero123Plus输出
    """
    logger.info(f"Simulating attention maps for {num_probes} probes...")
    
    # 创建模拟注意力图 [1, spatial_size*spatial_size, num_probes]
    seq_len = spatial_size * spatial_size
    attention_maps = torch.randn(1, seq_len, num_probes)
    
    # 为每个探针在不同位置添加峰值
    for i in range(num_probes):
        # 随机选择一个位置作为峰值
        peak_position = torch.randint(0, seq_len, (1,)).item()
        attention_maps[0, peak_position, i] += 5.0  # 添加强峰值
        
        # 在峰值周围添加一些邻近的激活
        y, x = divmod(peak_position, spatial_size)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < spatial_size and 0 <= nx < spatial_size:
                    neighbor_pos = ny * spatial_size + nx
                    attention_maps[0, neighbor_pos, i] += 2.0 * (1 - abs(dy) - abs(dx))
    
    return attention_maps


def extract_keypoints_from_attention(attention_maps, spatial_size, temperature=1.0):
    """
    从注意力图中提取关键点坐标
    
    这是核心的关键点提取算法，使用soft-argmax
    """
    batch_size, seq_len, num_probes = attention_maps.shape
    
    # 重塑为2D空间
    spatial_attention = attention_maps.view(batch_size, spatial_size, spatial_size, num_probes)
    spatial_attention = spatial_attention.permute(0, 3, 1, 2)  # [batch, num_probes, H, W]
    
    keypoints = torch.zeros(batch_size, num_probes, 2)
    
    for batch_idx in range(batch_size):
        for probe_idx in range(num_probes):
            attn_map = spatial_attention[batch_idx, probe_idx]  # [H, W]
            attn_flat = attn_map.flatten()  # [H*W]
            
            # 应用温度参数和softmax
            attn_soft = torch.softmax(attn_flat / temperature, dim=0)
            
            # 创建坐标网格
            y_coords, x_coords = torch.meshgrid(
                torch.arange(spatial_size, dtype=torch.float32),
                torch.arange(spatial_size, dtype=torch.float32),
                indexing='ij'
            )
            coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)  # [H*W, 2]
            
            # soft-argmax: 加权平均坐标
            weighted_coords = (attn_soft.unsqueeze(-1) * coords).sum(dim=0)  # [2]
            
            # 归一化到[0, 1]
            keypoints[batch_idx, probe_idx] = weighted_coords / (spatial_size - 1)
    
    return keypoints


def compute_demo_losses(attention_maps, spatial_size):
    """
    计算演示损失函数
    """
    batch_size, seq_len, num_probes = attention_maps.shape
    
    # 重塑为2D
    spatial_attention = attention_maps.view(batch_size, spatial_size, spatial_size, num_probes)
    spatial_attention = spatial_attention.permute(0, 3, 1, 2)  # [batch, num_probes, H, W]
    
    losses = {}
    
    # 1. 紧致性损失 (Compactness)
    compactness_loss = 0.0
    for probe_idx in range(num_probes):
        attn_map = spatial_attention[0, probe_idx]  # [H, W]
        attn_flat = attn_map.flatten()
        
        # 熵损失
        attn_norm = torch.softmax(attn_flat, dim=0)
        entropy = -(attn_norm * torch.log(attn_norm + 1e-8)).sum()
        compactness_loss += entropy
        
        # 方差损失
        variance = attn_flat.var()
        compactness_loss += variance
    
    compactness_loss /= num_probes
    losses['compactness'] = compactness_loss
    
    # 2. 多样性损失 (Diversity)
    diversity_loss = 0.0
    spatial_flat = spatial_attention[0].flatten(1)  # [num_probes, H*W]
    
    for i in range(num_probes):
        for j in range(i + 1, num_probes):
            # 计算余弦相似度
            attn_i = torch.nn.functional.normalize(spatial_flat[i], p=2, dim=0)
            attn_j = torch.nn.functional.normalize(spatial_flat[j], p=2, dim=0)
            similarity = torch.dot(attn_i, attn_j)
            diversity_loss += similarity.abs()
    
    num_pairs = num_probes * (num_probes - 1) // 2
    if num_pairs > 0:
        diversity_loss /= num_pairs
    
    losses['diversity'] = diversity_loss
    losses['total'] = compactness_loss + diversity_loss
    
    return losses


def visualize_demo_results(demo_image, keypoints, attention_maps, spatial_size, save_path=None):
    """可视化演示结果"""
    num_probes = keypoints.shape[1]
    
    # 创建颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, num_probes))
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Zero123Plus + StableKeypoints Demo Results', fontsize=16)
    
    # 1. 原始图像
    axes[0, 0].imshow(demo_image)
    axes[0, 0].set_title('Demo Image')
    axes[0, 0].axis('off')
    
    # 2. 带关键点的图像
    axes[0, 1].imshow(demo_image)
    
    # 在图像上绘制关键点
    image_size = demo_image.size
    for i, (x, y) in enumerate(keypoints[0].numpy()):
        pixel_x = x * image_size[0]
        pixel_y = y * image_size[1]
        axes[0, 1].scatter(pixel_x, pixel_y, c=[colors[i]], s=100, marker='o', 
                          edgecolors='white', linewidth=2)
        axes[0, 1].text(pixel_x + 5, pixel_y - 5, str(i), color='white', 
                       fontweight='bold', fontsize=8)
    
    axes[0, 1].set_title(f'Detected Keypoints ({num_probes})')
    axes[0, 1].axis('off')
    
    # 3. 注意力图聚合
    attention_2d = attention_maps[0].view(spatial_size, spatial_size, num_probes)
    attention_sum = attention_2d.sum(dim=2).numpy()
    
    im3 = axes[0, 2].imshow(attention_sum, cmap='hot', interpolation='bilinear')
    axes[0, 2].set_title('Aggregated Attention')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 4-6. 几个探针的注意力图
    for i in range(min(3, num_probes)):
        row = 1
        col = i
        
        attn_map = attention_2d[:, :, i].numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        im = axes[row, col].imshow(attn_map, cmap='hot', interpolation='bilinear')
        axes[row, col].set_title(f'Probe {i} Attention')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    # 隐藏未使用的子图
    for i in range(3, 3):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Zero123Plus + StableKeypoints Demo')
    parser.add_argument('--num_probes', type=int, default=8, help='Number of keypoint probes')
    parser.add_argument('--spatial_size', type=int, default=16, help='Spatial size of attention maps')
    parser.add_argument('--save_dir', type=str, default='demo_results', help='Save directory')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for soft-argmax')
    
    args = parser.parse_args()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    logger.info("🎭 Starting Zero123Plus + StableKeypoints Demo")
    logger.info(f"Parameters: {args.num_probes} probes, {args.spatial_size}x{args.spatial_size} attention")
    
    # 1. 创建演示图像
    logger.info("📸 Creating demo image...")
    demo_image = create_demo_image()
    demo_image.save(save_dir / "demo_image.png")
    
    # 2. 模拟注意力图
    logger.info("🧠 Simulating attention maps...")
    attention_maps = simulate_attention_and_keypoints(args.num_probes, args.spatial_size)
    
    # 3. 提取关键点
    logger.info("🎯 Extracting keypoints...")
    keypoints = extract_keypoints_from_attention(attention_maps, args.spatial_size, args.temperature)
    
    # 4. 计算损失
    logger.info("📊 Computing losses...")
    losses = compute_demo_losses(attention_maps, args.spatial_size)
    
    # 5. 显示结果
    logger.info("📋 Results:")
    logger.info(f"  Keypoints shape: {keypoints.shape}")
    logger.info(f"  Compactness loss: {losses['compactness']:.4f}")
    logger.info(f"  Diversity loss: {losses['diversity']:.4f}")
    logger.info(f"  Total loss: {losses['total']:.4f}")
    
    logger.info("🎯 Sample keypoints (normalized coordinates):")
    for i, (x, y) in enumerate(keypoints[0][:5]):  # 显示前5个
        logger.info(f"  Keypoint {i}: ({x:.3f}, {y:.3f})")
    
    # 6. 可视化
    logger.info("🎨 Creating visualization...")
    viz_path = save_dir / "demo_visualization.png"
    visualize_demo_results(demo_image, keypoints, attention_maps, args.spatial_size, viz_path)
    
    # 7. 保存数据
    logger.info("💾 Saving results...")
    torch.save({
        'keypoints': keypoints,
        'attention_maps': attention_maps,
        'losses': losses,
        'config': vars(args)
    }, save_dir / "demo_results.pt")
    
    logger.info(f"✅ Demo completed! Results saved to {save_dir}")
    logger.info("\n🔍 This demo shows the core concepts:")
    logger.info("  1. Attention maps from learnable probes")
    logger.info("  2. Soft-argmax keypoint extraction") 
    logger.info("  3. Compactness + diversity loss functions")
    logger.info("  4. Keypoint visualization")
    logger.info("\nIn the real implementation, attention maps come from Zero123Plus cross-attention!")


if __name__ == "__main__":
    main()
