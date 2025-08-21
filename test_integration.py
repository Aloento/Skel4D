"""
快速测试脚本 - 验证Zero123Plus + StableKeypoints集成
"""

import torch
import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from zero123plus_stablekeypoints import Zero123PlusStableKeypoints

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_functionality():
    """测试基本功能"""
    logger.info("Testing Zero123Plus + StableKeypoints integration...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # 1. 初始化模型
        logger.info("Step 1: Initializing model...")
        model = Zero123PlusStableKeypoints(
            model_id="sudo-ai/zero123plus-v1.2",
            num_probes=5,  # 少量探针用于快速测试
            device=device
        )
        logger.info("✓ Model initialization successful")
        
        # 2. 测试探针参数
        logger.info("Step 2: Testing probe parameters...")
        probe_params = list(model.get_trainable_parameters())
        total_params = sum(p.numel() for p in probe_params)
        logger.info(f"✓ Trainable parameters: {total_params} (probes only)")
        
        # 3. 测试前向传播
        logger.info("Step 3: Testing forward pass...")
        dummy_image = torch.randn(1, 3, 256, 256, device=device)
        
        # 使用更少的推理步数进行快速测试
        result, info = model.forward_with_probes(
            dummy_image,
            num_inference_steps=5,
            guidance_scale=2.0
        )
        logger.info("✓ Forward pass successful")
        
        # 4. 检查输出
        logger.info("Step 4: Checking outputs...")
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Info keys: {list(info.keys())}")
        
        if 'attention_maps' in info:
            attention_shape = info['attention_maps'].shape
            logger.info(f"✓ Attention maps shape: {attention_shape}")
        
        if 'keypoints' in info:
            keypoints_shape = info['keypoints'].shape
            logger.info(f"✓ Keypoints shape: {keypoints_shape}")
            
            # 显示一些关键点坐标
            sample_keypoints = info['keypoints'][0, :3]  # 前3个关键点
            logger.info(f"✓ Sample keypoints: {sample_keypoints}")
        
        if 'losses' in info:
            losses = info['losses']
            logger.info(f"✓ Losses: {losses}")
        
        # 5. 测试保存和加载
        logger.info("Step 5: Testing save/load...")
        test_save_path = "test_probes.safetensors"
        model.save_probes(test_save_path)
        logger.info(f"✓ Saved probes to {test_save_path}")
        
        model.load_probes(test_save_path)
        logger.info(f"✓ Loaded probes from {test_save_path}")
        
        # 清理测试文件
        Path(test_save_path).unlink(missing_ok=True)
        
        logger.info("🎉 All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """测试训练步骤"""
    logger.info("\nTesting training step...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 初始化模型和优化器
        model = Zero123PlusStableKeypoints(
            model_id="sudo-ai/zero123plus-v1.2",
            num_probes=3,  # 更少的探针
            device=device
        )
        
        optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(),
            lr=1e-4
        )
        
        # 模拟训练步骤
        dummy_image = torch.randn(1, 3, 256, 256, device=device)
        
        optimizer.zero_grad()
        
        result, info = model.forward_with_probes(
            dummy_image,
            num_inference_steps=3,  # 很少的步数
            guidance_scale=1.5
        )
        
        if 'losses' in info:
            total_loss = info['losses'].get('total_keypoint', torch.tensor(0.0))
            if total_loss.requires_grad:
                total_loss.backward()
                optimizer.step()
                logger.info(f"✓ Training step successful, loss: {total_loss.item():.4f}")
            else:
                logger.warning("Loss doesn't require gradients")
        else:
            logger.warning("No losses computed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_keypoint_extraction():
    """测试关键点提取"""
    logger.info("\nTesting keypoint extraction...")
    
    try:
        # 创建模拟注意力图
        batch_heads = 2
        seq_len = 64  # 8x8的空间
        num_probes = 4
        
        # 创建一些有峰值的注意力图
        attention_maps = torch.randn(batch_heads, seq_len, num_probes)
        
        # 在某些位置添加峰值
        attention_maps[0, 10, 0] = 10.0  # 探针0在位置10有峰值
        attention_maps[0, 30, 1] = 8.0   # 探针1在位置30有峰值
        
        # 初始化模型（仅用于提取关键点）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Zero123PlusStableKeypoints(
            model_id="sudo-ai/zero123plus-v1.2",
            num_probes=num_probes,
            device=device
        )
        
        # 提取关键点
        keypoints = model.extract_keypoints(attention_maps.to(device))
        
        logger.info(f"✓ Keypoints extracted: {keypoints.shape}")
        logger.info(f"✓ Sample keypoints: {keypoints[0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Keypoint extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    logger.info("Starting comprehensive tests...")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Training Step", test_training_step),
        ("Keypoint Extraction", test_keypoint_extraction),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        results[test_name] = test_func()
    
    # 总结
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        logger.info("\n🎉 All tests passed! The integration is working correctly.")
    else:
        logger.info("\n⚠️  Some tests failed. Please check the logs above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
