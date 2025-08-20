"""
Zero123Plus + StableKeypoints 推理和评估脚本

用于测试训练好的探针并可视化关键点检测结果
"""

import argparse
import logging
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
import json

from src.models.zero123plus_stablekeypoints import Zero123PlusStableKeypoints

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeypointVisualizer:
    """关键点可视化工具"""
    
    def __init__(self):
        self.colors = self._generate_colors(50)  # 支持50个关键点的不同颜色
    
    def _generate_colors(self, num_colors):
        """生成不同的颜色用于可视化"""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            # HSV到RGB的简单转换
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors
    
    def visualize_keypoints_on_image(
        self, 
        image: torch.Tensor, 
        keypoints: torch.Tensor,
        radius: int = 5,
        save_path: str = None
    ) -> Image.Image:
        """
        在图像上可视化关键点
        Args:
            image: [C, H, W] 图像张量（归一化的）
            keypoints: [num_probes, 2] 关键点坐标（归一化到[0,1]）
            radius: 关键点圆圈半径
            save_path: 保存路径
        Returns:
            PIL图像
        """
        # 反归一化图像
        if image.max() <= 1.0 and image.min() >= -1.0:
            # 假设是[-1, 1]归一化
            image_np = ((image.permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
        else:
            # 假设是[0, 1]归一化
            image_np = (image.permute(1, 2, 0) * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(image_np)
        draw = ImageDraw.Draw(pil_image)
        
        # 获取图像尺寸
        width, height = pil_image.size
        
        # 绘制关键点
        for i, (x, y) in enumerate(keypoints.cpu().numpy()):
            # 转换归一化坐标到像素坐标
            pixel_x = int(x * width)
            pixel_y = int(y * height)
            
            # 选择颜色
            color = self.colors[i % len(self.colors)]
            
            # 绘制圆圈
            draw.ellipse(
                [pixel_x - radius, pixel_y - radius, pixel_x + radius, pixel_y + radius],
                fill=color,
                outline="white",
                width=2
            )
            
            # 绘制标号
            draw.text((pixel_x + radius + 2, pixel_y - radius), str(i), fill="white")
        
        if save_path:
            pil_image.save(save_path)
            logger.info(f"Saved visualization to {save_path}")
        
        return pil_image
    
    def visualize_attention_maps(
        self,
        attention_maps: torch.Tensor,
        save_dir: str = None,
        max_maps: int = 10
    ):
        """
        可视化注意力图
        Args:
            attention_maps: [batch*heads, seq_len, num_probes] 或 [batch*heads, H, W, num_probes]
            save_dir: 保存目录
            max_maps: 最大可视化的地图数量
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理注意力图的维度
        if len(attention_maps.shape) == 3:
            batch_heads, seq_len, num_probes = attention_maps.shape
            # 尝试重塑为2D
            spatial_size = int(np.sqrt(seq_len))
            if spatial_size * spatial_size == seq_len:
                attention_maps = attention_maps.view(batch_heads, spatial_size, spatial_size, num_probes)
            else:
                logger.warning(f"Cannot visualize 1D attention maps with seq_len={seq_len}")
                return
        
        batch_heads, height, width, num_probes = attention_maps.shape
        
        # 只可视化第一个batch/head的注意力图
        attention_maps = attention_maps[0]  # [H, W, num_probes]
        
        # 可视化前max_maps个探针
        num_to_viz = min(num_probes, max_maps)
        
        fig, axes = plt.subplots(2, (num_to_viz + 1) // 2, figsize=(15, 6))
        if num_to_viz == 1:
            axes = [axes]
        elif len(axes.shape) == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_to_viz):
            row = i // ((num_to_viz + 1) // 2)
            col = i % ((num_to_viz + 1) // 2)
            
            if row < axes.shape[0] and col < axes.shape[1]:
                ax = axes[row, col]
                
                # 获取第i个探针的注意力图
                attn_map = attention_maps[:, :, i].cpu().numpy()
                
                # 归一化到[0, 1]
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                
                # 显示热力图
                im = ax.imshow(attn_map, cmap='hot', interpolation='bilinear')
                ax.set_title(f'Probe {i}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 隐藏未使用的子图
        for i in range(num_to_viz, axes.shape[0] * axes.shape[1]):
            row = i // axes.shape[1]
            col = i % axes.shape[1]
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = save_dir / "attention_maps.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention maps to {save_path}")
        
        plt.show()


def load_and_preprocess_image(image_path: str, size: tuple = (256, 256)) -> torch.Tensor:
    """加载并预处理图像"""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform(image)


def run_inference(
    model: Zero123PlusStableKeypoints,
    image_path: str,
    output_dir: str = "inference_results",
    num_inference_steps: int = 20,
    guidance_scale: float = 3.0
):
    """运行推理并保存结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载图像
    logger.info(f"Loading image: {image_path}")
    image = load_and_preprocess_image(image_path)
    image = image.unsqueeze(0).to(model.device)  # 添加batch维度
    
    # 运行推理
    logger.info("Running inference...")
    with torch.no_grad():
        result, info = model.forward_with_probes(
            image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
    
    # 保存结果
    results = {}
    
    # 关键点坐标
    if 'keypoints' in info:
        keypoints = info['keypoints'][0]  # 取第一个batch
        results['keypoints'] = keypoints.cpu().numpy().tolist()
        
        logger.info(f"Detected {len(keypoints)} keypoints")
        for i, (x, y) in enumerate(keypoints.cpu().numpy()):
            logger.info(f"  Keypoint {i}: ({x:.3f}, {y:.3f})")
        
        # 可视化关键点
        visualizer = KeypointVisualizer()
        viz_image = visualizer.visualize_keypoints_on_image(
            image[0],  # 取第一个batch
            keypoints,
            save_path=output_dir / "keypoints_visualization.png"
        )
    
    # 注意力图
    if 'attention_maps' in info:
        attention_maps = info['attention_maps']
        logger.info(f"Attention maps shape: {attention_maps.shape}")
        
        # 可视化注意力图
        visualizer = KeypointVisualizer()
        visualizer.visualize_attention_maps(
            attention_maps,
            save_dir=output_dir / "attention_maps"
        )
    
    # 损失信息
    if 'losses' in info:
        losses = info['losses']
        results['losses'] = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        logger.info(f"Losses: {results['losses']}")
    
    # 保存JSON结果
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    return results


def evaluate_on_dataset(
    model: Zero123PlusStableKeypoints,
    image_dir: str,
    output_dir: str = "evaluation_results",
    max_images: int = 100
):
    """在数据集上评估模型"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集图像文件
    image_dir = Path(image_dir)
    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    
    for ext in supported_exts:
        image_paths.extend(list(image_dir.glob(f"**/*{ext}")))
        image_paths.extend(list(image_dir.glob(f"**/*{ext.upper()}")))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    logger.info(f"Evaluating on {len(image_paths)} images")
    
    # 统计信息
    all_keypoints = []
    all_losses = []
    
    for i, image_path in enumerate(image_paths):
        logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path.name}")
        
        try:
            # 为每个图像创建子目录
            image_output_dir = output_dir / f"image_{i:04d}_{image_path.stem}"
            
            # 运行推理
            results = run_inference(
                model,
                str(image_path),
                str(image_output_dir)
            )
            
            # 收集统计信息
            if 'keypoints' in results:
                all_keypoints.append(results['keypoints'])
            
            if 'losses' in results:
                all_losses.append(results['losses'])
                
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            continue
    
    # 计算总体统计
    stats = {}
    
    if all_keypoints:
        keypoints_array = np.array(all_keypoints)  # [num_images, num_probes, 2]
        
        stats['keypoints'] = {
            'mean_x': float(np.mean(keypoints_array[:, :, 0])),
            'mean_y': float(np.mean(keypoints_array[:, :, 1])),
            'std_x': float(np.std(keypoints_array[:, :, 0])),
            'std_y': float(np.std(keypoints_array[:, :, 1])),
            'num_images': len(all_keypoints),
            'num_probes': keypoints_array.shape[1]
        }
    
    if all_losses:
        loss_keys = all_losses[0].keys()
        stats['losses'] = {}
        for key in loss_keys:
            values = [loss[key] for loss in all_losses]
            stats['losses'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    # 保存统计信息
    stats_path = output_dir / "evaluation_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Evaluation completed. Stats saved to {stats_path}")
    logger.info(f"Statistics: {stats}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Zero123Plus + StableKeypoints Inference")
    parser.add_argument("--model_id", type=str, default="sudo-ai/zero123plus-v1.2",
                       help="Zero123Plus model ID")
    parser.add_argument("--probe_path", type=str, required=True,
                       help="Path to trained probe weights")
    parser.add_argument("--image_path", type=str,
                       help="Single image to process")
    parser.add_argument("--image_dir", type=str,
                       help="Directory of images to evaluate")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                       help="Output directory")
    parser.add_argument("--num_probes", type=int, default=20,
                       help="Number of probes")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--num_inference_steps", type=int, default=20,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                       help="Guidance scale")
    parser.add_argument("--max_images", type=int, default=100,
                       help="Maximum images to evaluate")
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 初始化模型
    logger.info("Initializing model...")
    model = Zero123PlusStableKeypoints(
        model_id=args.model_id,
        num_probes=args.num_probes,
        device=device
    )
    
    # 加载探针权重
    logger.info(f"Loading probe weights from {args.probe_path}")
    model.load_probes(args.probe_path)
    
    # 设置评估模式
    model.probes.eval()
    
    # 运行推理或评估
    if args.image_path:
        # 单图像推理
        run_inference(
            model,
            args.image_path,
            args.output_dir,
            args.num_inference_steps,
            args.guidance_scale
        )
    elif args.image_dir:
        # 数据集评估
        evaluate_on_dataset(
            model,
            args.image_dir,
            args.output_dir,
            args.max_images
        )
    else:
        logger.error("Please provide either --image_path or --image_dir")


if __name__ == "__main__":
    main()
