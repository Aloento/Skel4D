# Copilot Instructions - Stable Keypoints + Zero123Plus Research Project

## Project Overview

This is a research project combining **StableKeypoints** unsupervised keypoint discovery with **Zero123Plus** novel view synthesis to create a self-supervised keypoint learning system. The goal is to leverage Zero123Plus's multi-view generation capabilities to train StableKeypoints without manual annotations.

### Key Components
- **StableKeypoints**: Unsupervised keypoint discovery using pretrained diffusion models
- **Zero123Plus**: Single image to consistent multi-view generation  
- **Self-Supervised Pipeline**: Novel view synthesis provides supervision for keypoint consistency
- **Slurm Integration**: Distributed training on HPC cluster

## Architecture & Data Flow

### Input Processing
```
Single Image → Zero123Plus → 7 Views (1 condition + 6 targets) → StableKeypoints Training
```

### Current Data Structure
- `cond_imgs`: (C, H, W) - Single condition image  
- `target_imgs`: (N, C, H, W) - Six target views (N=6)
- Data loaded from HDF5 files via `src.data.sk_objaverse` module

### Training Pipeline
1. **Data Loading**: HDF5-based Objaverse data through `StableKeypointObjaverseData`
2. **Multi-View Processing**: Process condition + 6 target views
3. **StableKeypoints Integration**: Activated via `use_stable_keypoints: true` in config
4. **Cross-Attention Optimization**: Learnable text embeddings optimized for keypoint discovery
5. **Loss Computation**: Localization + Equivariance + Novel View Synthesis losses

## File Structure & Key Components

### Training Scripts
- `train.py` - Main training script (USE THIS - not sk_train.py)
- `sk_train.py` - StableKeypoints standalone script (NOT USED in main pipeline)  
- `run_*_gpu.sh` - Slurm batch scripts for GPU training
- `run_*_ai.sh` - Slurm batch scripts for AI partition
- `run_*_gpu_test.sh` - Quick 20-minute test scripts for validation

### Configuration
- `configs/zero123plus-finetune-sk.yaml` - Main StableKeypoints training config (CORRECT)
- `configs/skeleton-zero123plus-finetune.yaml` - Alternative config file

### Core Modules
- `zero123plus/` - Zero123Plus model and pipeline
- `StableKeypoints/` - StableKeypoints implementation for reference
- `src/data/` - Data loading (Objaverse dataset)
  - `objaverse.py` - Standard Objaverse data loader
  - `sk_objaverse.py` - StableKeypoints-specific data loader
- `src/utils/` - Training utilities

### Data Classes (Important for Config)
- `StableKeypointDataModuleFromConfig` - Main data module class
- `StableKeypointObjaverseData` - Dataset class for individual samples

### Monitoring
- `slurm_out/` - Contains `.err` and `.out` files for job monitoring
- Check these files for training progress and errors

## Development Guidelines

### Before Making Changes
1. **Always ask before modifying code** - User prefers consultation to avoid misunderstandings
2. **Check existing implementation** - Review current code structure first
3. **Consider Slurm environment** - All training runs on cluster with batch jobs
4. **Validate pipeline flow** - Ensure changes maintain multi-view consistency
5. **No fallback solution** - User error is not an option; No branches for different shapes. No try catch blocks. Error handling must be resolved during development.

### Key Technical Considerations

#### StableKeypoints Integration
- **Cross-Attention**: Text embeddings (Key/Value) attend to image features (Query)
- **Learnable Embeddings**: Random embeddings optimized for semantic keypoint discovery
- **Loss Functions**: 
  - Localization Loss: Encourages Gaussian-like attention patterns
  - Equivariance Loss: Consistency across transformations
  - NVS Loss: Novel view synthesis reconstruction

#### Zero123Plus Specifics  
- **Multi-View Layout**: 6 views in 2x3 grid with fixed camera poses
- **Conditioning**: Reference attention + FlexDiffuse global conditioning
- **Noise Schedule**: Linear schedule for better global consistency

#### Data Processing Strategy
- **Keep Same Dataloader**: Maintain current input format
- **Individual Processing**: Process views separately before grid combination
- **Dual Forward Pass**: Separate passes for keypoint extraction and NVS training

### Common Tasks

#### Debugging Training Issues
1. Check `slurm_out/*.err` files for errors
2. Monitor `slurm_out/*.out` files for training progress
3. Verify GPU memory usage and batch sizes
4. Check data loading and preprocessing steps

#### Configuration Changes
- Update learning rates in config YAML files
- Adjust loss weights for different components
- Modify batch sizes based on GPU memory
- Set appropriate number of workers for data loading

#### Model Architecture Modifications
- Maintain compatibility with Zero123Plus pretrained weights
- Ensure StableKeypoints integration doesn't break NVS training
- Test attention mechanism changes thoroughly
- Validate multi-view consistency after modifications

### Slurm Workflow

#### Submitting Jobs
```bash
# GPU training
sbatch run_training_gpu.sh

# StableKeypoints training  
sbatch run_skel_training_gpu.sh
```

#### Monitoring Jobs
```bash
# Check job status
squeue -u $USER

# View live output
tail -f slurm_out/JOBID.out

# Check for errors
cat slurm_out/JOBID.err
```

### Testing Strategy
- `test_sk_integration.py` - Integration tests for StableKeypoints
- Always test changes on small datasets first
- Validate attention maps and keypoint extraction
- Check multi-view consistency metrics

## Current Status & Next Steps

The pipeline is almost complete. Key areas needing finalization:

1. **Training Loop Integration** - Ensuring StableKeypoints and Zero123Plus losses are properly combined
2. **Attention Map Extraction** - Verifying cross-attention maps are correctly captured from individual views
3. **Multi-View Consistency** - Validating keypoint consistency across generated views
4. **Loss Balancing** - Fine-tuning loss weights for optimal performance

## Communication Protocol

- **Always consult before code changes** - Describe proposed changes and get approval
- **Provide context for modifications** - Explain reasoning and expected impact  
- **Test incrementally** - Start with small changes and validate
- **Document changes** - Update configs and comments as needed

## Useful Commands

### Development
```bash
# Local testing
python sk_train.py

# Check config syntax
python -c "import yaml; yaml.safe_load(open('configs/skeleton-zero123plus-finetune.yaml'))"

# Monitor GPU usage
nvidia-smi -l 1
```

### Cluster Operations
```bash
# Check available partitions
sinfo

# Cancel job
scancel JOBID

# Job details
scontrol show job JOBID
```

Remember: This is a research environment focused on novel view synthesis and self-supervised keypoint learning. The integration between Zero123Plus and StableKeypoints is the core innovation being developed.
