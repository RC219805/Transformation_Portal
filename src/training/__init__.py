"""
Training Pipeline for Depth Anything V2 Fine-tuning

This module provides a complete training infrastructure for fine-tuning
Depth Anything V2 models on architectural imagery.

Components:
- augmentations: Depth-aware data augmentation pipeline
- depth_dataset: Flexible dataset class for paired RGB-Depth data
- losses: Depth-specific loss functions (Scale-Invariant, Gradient, SSIM)
- metrics: Depth estimation evaluation metrics
- trainer: Training orchestration with checkpointing and logging
- train_depth_anything_v2: Main training script

Example:
    >>> from src.training import DepthTrainer, ArchitecturalDepthDataset
    >>> import yaml
    >>>
    >>> with open('config/training/depth_anything_v2_large_finetune.yaml') as f:
    ...     config = yaml.safe_load(f)
    >>>
    >>> trainer = DepthTrainer(config)
    >>> trainer.train()

Author: Transformation Portal Team
Version: 1.0.0
"""

from .augmentations import (
    DepthAwareAugmentation,
    GeometricAugmentation,
    ColorAugmentation,
    ArchitecturalAugmentation,
    get_train_augmentations,
    get_val_augmentations,
)

from .depth_dataset import (
    ArchitecturalDepthDataset,
    create_data_loaders,
    DepthDataConfig,
)

from .losses import (
    ScaleInvariantLoss,
    GradientLoss,
    SSIMLoss,
    CombinedDepthLoss,
)

from .metrics import (
    DepthMetrics,
    compute_depth_metrics,
    visualize_depth_comparison,
)

from .trainer import (
    DepthTrainer,
    TrainingConfig,
    EarlyStopping,
)

from .utils import (
    load_config,
    setup_device,
    set_seed,
    create_logger,
)

__all__ = [
    # Augmentations
    "DepthAwareAugmentation",
    "GeometricAugmentation",
    "ColorAugmentation",
    "ArchitecturalAugmentation",
    "get_train_augmentations",
    "get_val_augmentations",
    # Dataset
    "ArchitecturalDepthDataset",
    "create_data_loaders",
    "DepthDataConfig",
    # Losses
    "ScaleInvariantLoss",
    "GradientLoss",
    "SSIMLoss",
    "CombinedDepthLoss",
    # Metrics
    "DepthMetrics",
    "compute_depth_metrics",
    "visualize_depth_comparison",
    # Trainer
    "DepthTrainer",
    "TrainingConfig",
    "EarlyStopping",
    # Utils
    "load_config",
    "setup_device",
    "set_seed",
    "create_logger",
]

__version__ = "1.0.0"
