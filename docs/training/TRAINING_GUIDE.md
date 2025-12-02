# Depth Anything V2 Training Guide

This guide explains how to fine-tune Depth Anything V2 on architectural imagery using the Transformation Portal training pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
4. [Configuration](#configuration)
5. [Training](#training)
6. [Monitoring](#monitoring)
7. [Resuming Training](#resuming-training)
8. [Model Export](#model-export)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

The training pipeline provides:

- **Fine-tuning** from pretrained Depth Anything V2 weights
- **Depth-aware augmentations** that maintain RGB-depth correspondence
- **Multiple loss functions** optimized for depth estimation
- **Mixed precision training** for faster training with less memory
- **Distributed training** support for multi-GPU setups
- **Comprehensive logging** with TensorBoard integration

## Prerequisites

### Hardware Requirements

| Training Type | VRAM | RAM | Storage |
|--------------|------|-----|---------|
| Small model  | 8GB  | 16GB | 50GB |
| Large model  | 24GB | 32GB | 100GB |
| Multi-GPU    | 16GB/GPU | 64GB | 200GB |

### Software Requirements

```bash
# Install core dependencies
pip install torch torchvision transformers

# Install training extras
pip install tqdm tensorboard pyyaml

# Distributed training is included in PyTorch. No separate package needed.
# If you need CUDA support, install the appropriate PyTorch build, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Optional: Weights & Biases
pip install wandb
```

### Verify Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Dataset Preparation

### Directory Structure

Your data should follow this structure:

```
data/
└── architectural/
    ├── train/
    │   ├── images/
    │   │   ├── render_001.jpg
    │   │   ├── render_002.png
    │   │   └── ...
    │   └── depth/
    │       ├── render_001.npy
    │       ├── render_002.npy
    │       └── ...
    └── val/
        ├── images/
        │   └── ...
        └── depth/
            └── ...
```

### Supported Formats

| Type | Formats | Notes |
|------|---------|-------|
| Images | JPG, PNG, TIFF | RGB, any bit depth |
| Depth | NPY, PNG, TIFF | Float32 or 16-bit |

### Prepare Your Data

#### Option 1: Use the preparation script

```bash
python scripts/training/prepare_training_data.py \
    --source-dir /path/to/raw/data \
    --output-dir data/architectural \
    --val-split 0.1 \
    --depth-format npy
```

#### Option 2: Create synthetic data for testing

```bash
python scripts/training/prepare_training_data.py \
    --create-sample \
    --num-samples 100 \
    --output-dir data/architectural
```

#### Option 3: Manual preparation

1. **Collect paired data**: Ensure each image has a corresponding depth map
2. **Normalize depth**: Convert to float32 with consistent units (e.g., meters)
3. **Name matching**: Image `render_001.jpg` should have depth `render_001.npy`
4. **Split data**: 80% train, 10% val, 10% test is typical

### Data Validation

```bash
# Check dataset statistics
python scripts/training/prepare_training_data.py \
    --source-dir data/architectural/train \
    --stats-only
```

## Configuration

### Configuration File

Create or modify `config/training/depth_anything_v2_large_finetune.yaml`:

```yaml
# Model configuration
model:
  name: "depth-anything/Depth-Anything-V2-Large-hf"
  pretrained: true
  freeze_encoder: false  # Set true for faster training

# Training hyperparameters
training:
  num_epochs: 50
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch = 32
  learning_rate: 1.0e-5
  weight_decay: 0.01
  warmup_epochs: 2
  mixed_precision: "fp16"

# Loss weights
loss:
  weights:
    scale_invariant: 1.0
    gradient: 0.5
    ssim: 0.3

# Data paths
data:
  train_dir: "data/architectural/train"
  val_dir: "data/architectural/val"
  image_size: [518, 518]
```

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `batch_size` | Samples per GPU | 4-16 |
| `gradient_accumulation_steps` | Accumulation steps | 4-8 |
| `learning_rate` | Initial LR | 1e-5 to 1e-4 |
| `freeze_encoder` | Freeze backbone | False for best quality |
| `mixed_precision` | FP16/BF16 | "fp16" for speed |

## Training

### Basic Training

```bash
python -m src.training.train_depth_anything_v2 \
    --config config/training/depth_anything_v2_large_finetune.yaml
```

### Training with Custom Settings

```python
from src.training import DepthTrainer, TrainingConfig
from src.training.train_depth_anything_v2 import DepthAnythingV2Wrapper

# Load model
model = DepthAnythingV2Wrapper(
    model_name="depth-anything/Depth-Anything-V2-Large-hf",
    pretrained=True,
)

# Configure training
config = TrainingConfig(
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-5,
    save_dir="checkpoints/my_experiment",
)

# Create trainer
trainer = DepthTrainer(model, config)

# Train
trainer.fit(train_loader, val_loader)
```

### Multi-GPU Training

```bash
# DataParallel (single machine, multiple GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.training.train_depth_anything_v2 \
    --config config/training/depth_anything_v2_large_finetune.yaml

# DistributedDataParallel (recommended for 4+ GPUs)
torchrun --nproc_per_node=4 -m src.training.train_depth_anything_v2 \
    --config config/training/depth_anything_v2_large_finetune.yaml \
    --distributed
```

## Monitoring

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/depth_anything_v2_large

# View at http://localhost:6006
```

### Metrics to Monitor

| Metric | Good Progress | Warning Signs |
|--------|--------------|---------------|
| Train Loss | Decreasing | Oscillating |
| Val Loss | Decreasing | Increasing (overfit) |
| Val RMSE | < 0.5 | > 1.0 |
| δ < 1.25 | > 0.9 | < 0.7 |
| Learning Rate | Smooth decay | Sudden jumps |

### Visualizations

The trainer automatically saves depth visualizations:

```
logs/depth_anything_v2_large/
└── visualizations/
    ├── epoch_001.png
    ├── epoch_002.png
    └── ...
```

## Resuming Training

### Resume from Checkpoint

```bash
python -m src.training.train_depth_anything_v2 \
    --config config/training/depth_anything_v2_large_finetune.yaml \
    --resume checkpoints/depth_anything_v2_large/checkpoint_epoch_25.pth
```

### Resume from Best Model

```bash
python -m src.training.train_depth_anything_v2 \
    --config config/training/depth_anything_v2_large_finetune.yaml \
    --resume checkpoints/depth_anything_v2_large/best_model.pth
```

## Model Export

### Load Trained Weights

```python
import torch
from src.training.train_depth_anything_v2 import DepthAnythingV2Wrapper

# Load model
model = DepthAnythingV2Wrapper(model_name="depth-anything/Depth-Anything-V2-Large-hf")

# Load trained weights
checkpoint = torch.load("checkpoints/depth_anything_v2_large/best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])

# Set to eval mode
model.eval()
```

### Export to ONNX

```python
import torch

# Export to ONNX
dummy_input = torch.randn(1, 3, 518, 518)
torch.onnx.export(
    model,
    dummy_input,
    "depth_model.onnx",
    opset_version=17,
    input_names=["image"],
    output_names=["depth"],
)
```

### Export to CoreML (Apple Silicon)

```python
import coremltools as ct

# Convert to CoreML
traced = torch.jit.trace(model, dummy_input)
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 518, 518))],
)
mlmodel.save("depth_model.mlpackage")
```

## Best Practices

### Data Quality

1. **High-quality depth maps**: Use accurate ground truth from sensors or high-quality renders
2. **Diverse scenes**: Include various lighting, materials, and viewpoints
3. **Consistent scale**: Normalize depth to consistent units across dataset
4. **Clean data**: Remove corrupted or poorly aligned pairs

### Training Strategy

1. **Start with pretrained weights**: Always fine-tune from pretrained model
2. **Warm up learning rate**: Use 1-2 epochs of warmup
3. **Monitor validation metrics**: Stop if validation loss increases
4. **Use early stopping**: Set patience to 5-10 epochs
5. **Save checkpoints frequently**: Every 5 epochs minimum

### Hyperparameter Tuning

1. **Learning rate**: Start with 1e-5, adjust based on loss curves
2. **Batch size**: Larger is better, limited by GPU memory
3. **Weight decay**: 0.01-0.1, higher for larger models
4. **Loss weights**: Scale-invariant should dominate, gradient for edges

### Memory Optimization

1. **Gradient accumulation**: Use to simulate larger batches
2. **Mixed precision**: Enable FP16 for 2x memory savings
3. **Freeze encoder**: Reduces trainable parameters significantly
4. **Gradient checkpointing**: Trade compute for memory

## Troubleshooting

### Common Issues

#### Out of Memory

```python
# Reduce batch size
training:
  batch_size: 4
  gradient_accumulation_steps: 8  # Maintain effective batch size

# Or enable gradient checkpointing
model:
  gradient_checkpointing: true
```

#### Loss is NaN

```python
# Check for invalid depth values
import numpy as np
depth = np.load("depth.npy")
print(f"NaN: {np.isnan(depth).sum()}, Inf: {np.isinf(depth).sum()}")

# Use smaller learning rate
training:
  learning_rate: 1.0e-6
```

#### Validation Loss Not Decreasing

1. Check data augmentation isn't too aggressive
2. Reduce learning rate
3. Check for data leakage between train/val
4. Increase model capacity (use larger variant)

#### Training is Slow

1. Enable mixed precision: `mixed_precision: "fp16"`
2. Increase num_workers: `num_workers: 8`
3. Enable pin_memory: `pin_memory: true`
4. Use SSD storage for data

### Performance Tips

| Optimization | Speedup | Memory Savings |
|--------------|---------|----------------|
| Mixed precision | 1.5-2x | 50% |
| Freeze encoder | 2-3x | 30% |
| Larger batch | 1.2x | - |
| Gradient accumulation | - | 50%+ |
| Multi-GPU | Nx | - |

### Getting Help

1. Check logs: `logs/depth_anything_v2_large/training.log`
2. View TensorBoard for training curves
3. Validate data with preparation script
4. Check GPU utilization with `nvidia-smi`

## References

- [Depth Anything V2 Paper](https://arxiv.org/abs/2406.09414)
- [HuggingFace Model](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)
- [Transformation Portal Documentation](../README.md)

## Example Workflows

### Quick Start (Testing)

```bash
# Create sample data
python scripts/training/prepare_training_data.py --create-sample

# Run short training
python examples/training/run_depth_training.py
```

### Production Training

```bash
# Prepare real data
python scripts/training/prepare_training_data.py \
    --source-dir /data/renders \
    --output-dir data/architectural \
    --val-split 0.1

# Full training
python -m src.training.train_depth_anything_v2 \
    --config config/training/depth_anything_v2_large_finetune.yaml

# Monitor
tensorboard --logdir logs/depth_anything_v2_large
```

### Fine-tuning for Specific Domain

```yaml
# config/training/interior_finetune.yaml
model:
  name: "depth-anything/Depth-Anything-V2-Large-hf"
  freeze_encoder: true  # Only train decoder

training:
  num_epochs: 20
  learning_rate: 5.0e-5  # Higher LR for fine-tuning

data:
  train_dir: "data/interior/train"
  val_dir: "data/interior/val"
```
