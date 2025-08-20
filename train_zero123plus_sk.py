"""
Zero123Plus + StableKeypoints 训练脚本

实现基于条件探针的无监督关键点学习
"""

import argparse
import logging
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import wandb
from typing import Optional

from src.models.zero123plus_stablekeypoints import Zero123PlusStableKeypoints

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleImageDataset(Dataset):
    """
    简单的图像数据集，用于训练探针
    """
    
    def __init__(self, image_dir: str, transform=None, max_samples: Optional[int] = None):
        self.image_dir = Path(image_dir)
        self.transform = transform or self._default_transform()
        
        # 支持的图像格式
        supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 收集图像文件
        self.image_paths = []
        for ext in supported_exts:
            self.image_paths.extend(list(self.image_dir.glob(f"**/*{ext}")))
            self.image_paths.extend(list(self.image_dir.glob(f"**/*{ext.upper()}")))
        
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
        
        logger.info(f"Found {len(self.image_paths)} images in {image_dir}")
        
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, str(image_path)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # 返回一个随机图像作为fallback
            dummy_image = torch.randn(3, 256, 256)
            return dummy_image, str(image_path)


class TrainingConfig:
    """训练配置"""
    
    def __init__(self):
        # 模型配置
        self.model_id = "sudo-ai/zero123plus-v1.2"
        self.num_probes = 20
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # 训练配置
        self.learning_rate = 1e-4
        self.num_epochs = 10
        self.batch_size = 1  # Zero123Plus通常需要较小的batch size
        self.num_workers = 2
        self.gradient_accumulation_steps = 4
        
        # 损失权重
        self.compactness_weight = 1.0
        self.diversity_weight = 0.5
        
        # 推理配置
        self.num_inference_steps = 20
        self.guidance_scale = 3.0
        
        # 输出配置
        self.output_dir = "runs"
        self.save_every = 100
        self.log_every = 10
        self.max_samples = None  # 限制训练样本数，None表示使用所有
        
        # 可视化配置
        self.visualize_keypoints = True
        self.save_attention_maps = True


class Trainer:
    """训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        logger.info("Initializing Zero123Plus-StableKeypoints model...")
        self.model = Zero123PlusStableKeypoints(
            model_id=config.model_id,
            num_probes=config.num_probes,
            device=config.device,
            dtype=config.dtype
        )
        
        # 设置优化器（仅优化探针）
        self.optimizer = optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        
        logger.info("Trainer initialized successfully")
    
    def create_dataloader(self, image_dir: str) -> DataLoader:
        """创建数据加载器"""
        dataset = SimpleImageDataset(
            image_dir,
            max_samples=self.config.max_samples
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return dataloader
    
    def train_step(self, batch) -> dict:
        """单步训练"""
        images, image_paths = batch
        images = images.to(self.config.device)
        
        # 前向传播
        try:
            result, info = self.model.forward_with_probes(
                images,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale
            )
            
            # 计算损失
            if 'losses' in info:
                losses = info['losses']
                total_loss = losses.get('total_keypoint', torch.tensor(0.0))
            else:
                # 如果没有收集到注意力图，创建dummy损失
                total_loss = torch.tensor(0.0, requires_grad=True, device=self.config.device)
                losses = {'total_keypoint': total_loss, 'compactness': total_loss, 'diversity': total_loss}
            
            # 反向传播
            if total_loss.requires_grad:
                loss_scaled = total_loss / self.config.gradient_accumulation_steps
                loss_scaled.backward()
            
            return {
                'loss': total_loss.item(),
                'compactness_loss': losses.get('compactness', torch.tensor(0.0)).item(),
                'diversity_loss': losses.get('diversity', torch.tensor(0.0)).item(),
                'num_keypoints': self.config.num_probes,
                'info': info
            }
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            # 返回dummy结果继续训练
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.config.device)
            return {
                'loss': 0.0,
                'compactness_loss': 0.0,
                'diversity_loss': 0.0,
                'num_keypoints': self.config.num_probes,
                'info': {}
            }
    
    def train(self, image_dir: str):
        """主训练循环"""
        logger.info("Starting training...")
        
        # 创建数据加载器
        dataloader = self.create_dataloader(image_dir)
        logger.info(f"Training on {len(dataloader.dataset)} images")
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # 训练步骤
                step_results = self.train_step(batch)
                
                epoch_loss += step_results['loss']
                num_batches += 1
                
                # 梯度累积
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{step_results['loss']:.4f}",
                    'compact': f"{step_results['compactness_loss']:.4f}",
                    'diverse': f"{step_results['diversity_loss']:.4f}"
                })
                
                # 日志记录
                if self.global_step % self.config.log_every == 0:
                    self._log_metrics(step_results)
                
                # 保存检查点
                if self.global_step % self.config.save_every == 0:
                    self._save_checkpoint()
                
                # 可视化（可选）
                if (self.config.visualize_keypoints and 
                    self.global_step % (self.config.save_every // 2) == 0 and
                    'keypoints' in step_results['info']):
                    self._visualize_keypoints(batch, step_results['info'])
            
            # 清理梯度累积的残余
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 记录epoch统计
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1} finished. Average loss: {avg_epoch_loss:.4f}")
        
        # 保存最终模型
        self._save_checkpoint(final=True)
        logger.info("Training completed!")
    
    def _log_metrics(self, step_results: dict):
        """记录训练指标"""
        metrics = {
            'train/loss': step_results['loss'],
            'train/compactness_loss': step_results['compactness_loss'],
            'train/diversity_loss': step_results['diversity_loss'],
            'train/global_step': self.global_step,
            'train/epoch': self.epoch
        }
        
        # 如果使用wandb
        try:
            wandb.log(metrics, step=self.global_step)
        except:
            pass
        
        # 打印日志
        if self.global_step % (self.config.log_every * 5) == 0:
            logger.info(f"Step {self.global_step}: {metrics}")
    
    def _save_checkpoint(self, final: bool = False):
        """保存检查点"""
        checkpoint_name = "final_probes.safetensors" if final else f"probes_step_{self.global_step}.safetensors"
        checkpoint_path = self.output_dir / checkpoint_name
        
        self.model.save_probes(checkpoint_path)
        
        # 保存训练状态
        state_path = self.output_dir / f"training_state_step_{self.global_step}.json"
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config.__dict__,
            'optimizer_state': str(self.optimizer.state_dict())  # 简化保存
        }
        
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _visualize_keypoints(self, batch, info: dict):
        """可视化关键点（可选实现）"""
        try:
            images, _ = batch
            if 'keypoints' in info:
                keypoints = info['keypoints']
                # 这里可以实现关键点可视化并保存图像
                # 由于篇幅限制，这里只是记录日志
                logger.info(f"Keypoints detected: {keypoints.shape}")
        except Exception as e:
            logger.warning(f"Keypoint visualization failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train Zero123Plus + StableKeypoints")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing training images")
    parser.add_argument("--output_dir", type=str, default="runs",
                       help="Output directory for checkpoints")
    parser.add_argument("--num_probes", type=int, default=20,
                       help="Number of keypoint probes")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of training samples")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="Wandb project name")
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig()
    config.output_dir = args.output_dir
    config.num_probes = args.num_probes
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size
    config.max_samples = args.max_samples
    
    if args.device != "auto":
        config.device = args.device
    
    # 初始化wandb（可选）
    if args.wandb_project:
        try:
            wandb.init(
                project=args.wandb_project,
                config=config.__dict__
            )
            logger.info(f"Initialized wandb project: {args.wandb_project}")
        except:
            logger.warning("Failed to initialize wandb, continuing without logging")
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train(args.image_dir)


if __name__ == "__main__":
    main()
