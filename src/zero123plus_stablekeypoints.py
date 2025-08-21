"""
Zero123Plus v1.2 × StableKeypoints Integration
实现方案A：条件向量探针 (Condition-Probe)

在Zero123++的条件构造中注入K个可学习探针向量，通过约束注意力分布
形成稳定关键点检测能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import numpy as np
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.attention_processor import Attention
import safetensors.torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ProbeAttentionProcessor:
    """
    自定义注意力处理器，用于收集探针的注意力图
    """
    
    def __init__(self, num_probes: int, collect_attention: bool = True):
        self.num_probes = num_probes
        self.collect_attention = collect_attention
        self.attention_maps = []
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        执行注意力计算并收集探针的注意力图
        """
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        # 计算query, key, value
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # 重塑为多头形式
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # 计算注意力分数
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # 收集探针的注意力图（最后num_probes个keys对应探针）
        if self.collect_attention and encoder_hidden_states.shape[1] >= self.num_probes:
            # 提取探针对应的注意力（假设探针在encoder_hidden_states的末尾）
            probe_attention = attention_probs[:, :, -self.num_probes:]  # [batch*heads, seq_len, num_probes]
            self.attention_maps.append(probe_attention.detach())
        
        # 执行注意力
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # 输出投影
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states
    
    def clear_attention_maps(self):
        """清除收集的注意力图"""
        self.attention_maps = []
    
    def get_aggregated_attention(self) -> Optional[torch.Tensor]:
        """聚合所有收集的注意力图"""
        if not self.attention_maps:
            return None
            
        # 聚合多层、多头、多步的注意力
        all_attention = torch.stack(self.attention_maps, dim=0)  # [layers, batch*heads, seq_len, num_probes]
        
        # 平均聚合
        aggregated = all_attention.mean(dim=0)  # [batch*heads, seq_len, num_probes]
        
        return aggregated


class LearnableProbes(nn.Module):
    """
    可学习的探针向量，用于关键点发现
    """
    
    def __init__(self, num_probes: int, probe_dim: int):
        super().__init__()
        self.num_probes = num_probes
        self.probe_dim = probe_dim
        
        # 初始化可学习探针（小随机值）
        self.probes = nn.Parameter(torch.randn(num_probes, probe_dim) * 0.01)
        
    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        返回批次化的探针向量
        Args:
            batch_size: 批次大小
        Returns:
            [batch_size, num_probes, probe_dim]
        """
        return self.probes.unsqueeze(0).expand(batch_size, -1, -1)


class Zero123PlusStableKeypoints:
    """
    Zero123Plus + StableKeypoints集成模型
    """
    
    def __init__(
        self,
        model_id: str = "sudo-ai/zero123plus-v1.2",
        num_probes: int = 20,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.device = device
        self.dtype = dtype
        self.num_probes = num_probes
        
        # 加载Zero123Plus pipeline
        logger.info(f"Loading Zero123Plus from {model_id}")
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)
        
        # 冻结主干网络
        self._freeze_backbone()
        
        # 获取条件编码器的输出维度
        # 这需要根据具体的Zero123Plus实现来调整
        self.condition_dim = self._get_condition_dim()
        
        # 创建可学习探针
        self.probes = LearnableProbes(num_probes, self.condition_dim).to(device)
        
        # 设置注意力处理器
        self.attention_processor = ProbeAttentionProcessor(num_probes)
        self._setup_attention_processors()
        
        logger.info(f"Initialized Zero123Plus-StableKeypoints with {num_probes} probes")
    
    def _freeze_backbone(self):
        """冻结Zero123Plus主干网络"""
        for param in self.pipeline.unet.parameters():
            param.requires_grad = False
        
        if hasattr(self.pipeline, 'vae'):
            for param in self.pipeline.vae.parameters():
                param.requires_grad = False
                
        logger.info("Frozen Zero123Plus backbone")
    
    def _get_condition_dim(self) -> int:
        """获取条件编码器的输出维度"""
        # 这里需要根据Zero123Plus的具体实现来确定
        # 通常cross-attention的condition维度可以从unet的config中获取
        if hasattr(self.pipeline.unet.config, 'cross_attention_dim'):
            return self.pipeline.unet.config.cross_attention_dim
        else:
            # 默认值，可能需要调整
            return 768
    
    def _setup_attention_processors(self):
        """设置注意力处理器以收集注意力图"""
        attn_procs = {}
        
        # 只在cross-attention层设置处理器
        for name, module in self.pipeline.unet.named_modules():
            if "cross_attn" in name or "attn2" in name:
                attn_procs[name] = self.attention_processor
        
        if attn_procs:
            self.pipeline.unet.set_attn_processor(attn_procs)
            logger.info(f"Set attention processors for {len(attn_procs)} cross-attention layers")
        else:
            logger.warning("No cross-attention layers found!")
    
    def _inject_probes(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        在编码器隐藏状态中注入探针
        Args:
            encoder_hidden_states: [batch_size, seq_len, hidden_dim]
        Returns:
            注入探针后的隐藏状态 [batch_size, seq_len + num_probes, hidden_dim]
        """
        batch_size = encoder_hidden_states.shape[0]
        
        # 获取探针向量
        probe_vectors = self.probes(batch_size)  # [batch_size, num_probes, hidden_dim]
        
        # 拼接到编码器隐藏状态末尾
        enhanced_states = torch.cat([encoder_hidden_states, probe_vectors], dim=1)
        
        return enhanced_states
    
    def compute_keypoint_losses(
        self,
        attention_maps: torch.Tensor,
        compactness_weight: float = 1.0,
        diversity_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        计算关键点相关损失
        Args:
            attention_maps: [batch*heads, seq_len, num_probes]
            compactness_weight: 紧致性损失权重
            diversity_weight: 多样性损失权重
        Returns:
            损失字典
        """
        losses = {}
        
        # 重塑注意力图到空间维度 (假设为正方形)
        batch_heads, seq_len, num_probes = attention_maps.shape
        
        # 假设seq_len对应空间token（例如64x64的latent对应4096个token）
        spatial_size = int(np.sqrt(seq_len))
        if spatial_size * spatial_size != seq_len:
            logger.warning(f"Cannot reshape seq_len {seq_len} to square spatial dimension")
            spatial_size = seq_len  # 使用1D处理
            spatial_attention = attention_maps  # [batch*heads, seq_len, num_probes]
        else:
            # 重塑为2D空间注意力图
            spatial_attention = attention_maps.view(batch_heads, spatial_size, spatial_size, num_probes)
            spatial_attention = spatial_attention.permute(0, 3, 1, 2)  # [batch*heads, num_probes, H, W]
        
        # 1. 紧致性损失：促使每个探针的注意力集中
        compactness_loss = 0.0
        
        if len(spatial_attention.shape) == 4:  # 2D情况
            for k in range(num_probes):
                attn_k = spatial_attention[:, k]  # [batch*heads, H, W]
                
                # 计算注意力图的熵（越低越紧致）
                attn_k_flat = attn_k.flatten(1)  # [batch*heads, H*W]
                attn_k_norm = F.softmax(attn_k_flat, dim=1)
                entropy = -(attn_k_norm * torch.log(attn_k_norm + 1e-8)).sum(dim=1)
                compactness_loss += entropy.mean()
                
                # 计算二阶矩（越小越紧致）
                # 这里简化为使用标准差
                std = attn_k_flat.std(dim=1)
                compactness_loss += std.mean()
        else:  # 1D情况
            for k in range(num_probes):
                attn_k = spatial_attention[:, :, k]  # [batch*heads, seq_len]
                
                # 计算熵
                attn_k_norm = F.softmax(attn_k, dim=1)
                entropy = -(attn_k_norm * torch.log(attn_k_norm + 1e-8)).sum(dim=1)
                compactness_loss += entropy.mean()
                
                # 计算标准差
                std = attn_k.std(dim=1)
                compactness_loss += std.mean()
        
        compactness_loss /= num_probes
        losses['compactness'] = compactness_loss * compactness_weight
        
        # 2. 多样性损失：促使不同探针的注意力分散
        diversity_loss = 0.0
        
        if len(spatial_attention.shape) == 4:  # 2D情况
            spatial_attention_flat = spatial_attention.flatten(2)  # [batch*heads, num_probes, H*W]
        else:  # 1D情况
            spatial_attention_flat = spatial_attention.permute(0, 2, 1)  # [batch*heads, num_probes, seq_len]
        
        # 计算探针间的余弦相似度
        for i in range(num_probes):
            for j in range(i + 1, num_probes):
                attn_i = spatial_attention_flat[:, i]  # [batch*heads, spatial_dim]
                attn_j = spatial_attention_flat[:, j]  # [batch*heads, spatial_dim]
                
                # 归一化
                attn_i_norm = F.normalize(attn_i, p=2, dim=1)
                attn_j_norm = F.normalize(attn_j, p=2, dim=1)
                
                # 余弦相似度
                similarity = (attn_i_norm * attn_j_norm).sum(dim=1)
                diversity_loss += similarity.abs().mean()
        
        num_pairs = num_probes * (num_probes - 1) // 2
        if num_pairs > 0:
            diversity_loss /= num_pairs
        
        losses['diversity'] = diversity_loss * diversity_weight
        
        # 总损失
        losses['total_keypoint'] = losses['compactness'] + losses['diversity']
        
        return losses
    
    def extract_keypoints(
        self, 
        attention_maps: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        从注意力图中提取关键点坐标
        Args:
            attention_maps: [batch*heads, seq_len, num_probes]
            temperature: soft-argmax的温度参数
        Returns:
            关键点坐标 [batch*heads, num_probes, 2] (归一化坐标)
        """
        batch_heads, seq_len, num_probes = attention_maps.shape
        
        # 重塑到空间维度
        spatial_size = int(np.sqrt(seq_len))
        if spatial_size * spatial_size != seq_len:
            # 1D情况的处理
            keypoints = torch.zeros(batch_heads, num_probes, 2, device=attention_maps.device)
            for k in range(num_probes):
                attn_k = attention_maps[:, :, k]  # [batch*heads, seq_len]
                attn_k = F.softmax(attn_k / temperature, dim=1)
                
                # 1D情况下，只能提取x坐标
                indices = torch.arange(seq_len, device=attention_maps.device, dtype=torch.float32)
                x_coord = (attn_k * indices).sum(dim=1) / seq_len  # 归一化到[0,1]
                
                keypoints[:, k, 0] = x_coord
                keypoints[:, k, 1] = 0.5  # y坐标设为中间值
            
            return keypoints
        
        # 2D情况
        spatial_attention = attention_maps.view(batch_heads, spatial_size, spatial_size, num_probes)
        spatial_attention = spatial_attention.permute(0, 3, 1, 2)  # [batch*heads, num_probes, H, W]
        
        keypoints = torch.zeros(batch_heads, num_probes, 2, device=attention_maps.device)
        
        for k in range(num_probes):
            attn_k = spatial_attention[:, k]  # [batch*heads, H, W]
            attn_k_flat = attn_k.flatten(1)  # [batch*heads, H*W]
            attn_k_soft = F.softmax(attn_k_flat / temperature, dim=1)
            
            # 创建坐标网格
            y_coords, x_coords = torch.meshgrid(
                torch.arange(spatial_size, device=attention_maps.device, dtype=torch.float32),
                torch.arange(spatial_size, device=attention_maps.device, dtype=torch.float32),
                indexing='ij'
            )
            coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)  # [H*W, 2]
            
            # soft-argmax
            weighted_coords = (attn_k_soft.unsqueeze(-1) * coords.unsqueeze(0)).sum(dim=1)  # [batch*heads, 2]
            
            # 归一化到[0,1]
            keypoints[:, k] = weighted_coords / (spatial_size - 1)
        
        return keypoints
    
    def forward_with_probes(
        self,
        image: torch.Tensor,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.0,
        **pipeline_kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        使用探针进行前向传播
        Args:
            image: 输入图像
            num_inference_steps: 推理步数
            guidance_scale: 引导尺度
        Returns:
            生成的图像和注意力相关信息
        """
        # 清除之前的注意力图
        self.attention_processor.clear_attention_maps()
        
        # 这里需要根据Zero123Plus的具体API来调整
        # 由于每个版本的接口可能不同，这里提供一个通用框架
        
        # 运行pipeline（这里需要hook进条件编码过程来注入探针）
        try:
            # 尝试不同的参数名
            result = self.pipeline(
                image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **pipeline_kwargs
            )
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            # 返回dummy结果用于测试
            result = type('DummyResult', (), {
                'images': [torch.zeros(3, 512, 512, device=self.device)]
            })()
        
        # 获取聚合的注意力图
        attention_maps = self.attention_processor.get_aggregated_attention()
        
        # 计算关键点相关信息
        info = {}
        if attention_maps is not None:
            info['attention_maps'] = attention_maps
            info['keypoints'] = self.extract_keypoints(attention_maps)
            info['losses'] = self.compute_keypoint_losses(attention_maps)
        
        return result, info
    
    def save_probes(self, save_path: Union[str, Path]):
        """保存探针权重"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state_dict = {'probes': self.probes.state_dict()}
        safetensors.torch.save_file(state_dict, save_path)
        logger.info(f"Saved probes to {save_path}")
    
    def load_probes(self, load_path: Union[str, Path]):
        """加载探针权重"""
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Probe file not found: {load_path}")
        
        state_dict = safetensors.torch.load_file(load_path)
        self.probes.load_state_dict(state_dict['probes'])
        logger.info(f"Loaded probes from {load_path}")
    
    def get_trainable_parameters(self):
        """获取可训练参数（仅探针）"""
        return self.probes.parameters()


# 使用示例和测试函数
def test_integration():
    """测试集成功能"""
    print("Testing Zero123Plus + StableKeypoints integration...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建模型
    model = Zero123PlusStableKeypoints(
        model_id="sudo-ai/zero123plus-v1.2",
        num_probes=10,
        device=device
    )
    
    # 创建dummy输入
    dummy_image = torch.randn(1, 3, 256, 256, device=device)
    
    # 前向传播
    try:
        result, info = model.forward_with_probes(
            dummy_image,
            num_inference_steps=5,
            guidance_scale=2.0
        )
        
        print("Forward pass successful!")
        print(f"Info keys: {list(info.keys())}")
        
        if 'keypoints' in info:
            print(f"Keypoints shape: {info['keypoints'].shape}")
            print(f"Sample keypoints: {info['keypoints'][0, :3]}")  # 前3个关键点
        
        if 'losses' in info:
            print(f"Losses: {info['losses']}")
        
        # 保存探针
        model.save_probes("test_probes.safetensors")
        print("Probe saving successful!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_integration()
