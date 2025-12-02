#!/usr/bin/env python3
"""
Training Script for Depth Anything V2 Fine-tuning

Complete training pipeline for fine-tuning Depth Anything V2 Large
on architectural imagery.

Features:
- Fine-tuning from HuggingFace pretrained weights
- Multi-GPU support (DataParallel/DistributedDataParallel)
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Configurable via YAML

Usage:
    python -m src.training.train_depth_anything_v2 \\
        --config config/training/depth_anything_v2_large_finetune.yaml

Author: Transformation Portal Team
Version: 1.0.0
"""

import argparse
import logging
import os
import sys

# Try to import dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

try:
    from transformers import AutoModelForDepthEstimation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ERROR: Transformers is required. Install with: pip install transformers")
    sys.exit(1)

from .augmentations import get_train_augmentations, get_val_augmentations
from .depth_dataset import DepthDataConfig, create_data_loaders
from .losses import CombinedDepthLoss
from .trainer import DepthTrainer, TrainingConfig
from .utils import (
    create_logger,
    load_config,
    set_seed,
    setup_device,
    validate_config,
)

logger = logging.getLogger(__name__)


class DepthAnythingV2Wrapper(nn.Module):
    """Wrapper for Depth Anything V2 model for training.

    Adapts the HuggingFace model interface for training pipeline.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Large-hf",
        pretrained: bool = True,
        freeze_encoder: bool = False,
    ):
        """Initialize model wrapper.

        Args:
            model_name: HuggingFace model name or path
            pretrained: Whether to load pretrained weights
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()

        # Load model
        if pretrained:
            # nosec B615 - revision pinning intentionally omitted for development flexibility
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_config(config)

        # Optionally freeze encoder
        if freeze_encoder:
            self._freeze_encoder()

        logger.info(f"Loaded model: {model_name}")
        logger.info(f"Pretrained: {pretrained}, Freeze encoder: {freeze_encoder}")

    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        # Freeze backbone parameters
        for name, param in self.model.named_parameters():
            if "backbone" in name or "encoder" in name:
                param.requires_grad = False

        logger.info("Froze encoder parameters")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: Input images (B, 3, H, W)

        Returns:
            Predicted depth maps (B, 1, H, W)
        """
        outputs = self.model(pixel_values=pixel_values)

        # Get predicted depth
        predicted_depth = outputs.predicted_depth

        # Ensure 4D output (B, 1, H, W)
        if predicted_depth.dim() == 3:
            predicted_depth = predicted_depth.unsqueeze(1)

        return predicted_depth


def create_model(config: dict) -> nn.Module:
    """Create depth estimation model from config.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized model
    """
    model_config = config.get("model", {})

    model = DepthAnythingV2Wrapper(
        model_name=model_config.get("name", "depth-anything/Depth-Anything-V2-Large-hf"),
        pretrained=model_config.get("pretrained", True),
        freeze_encoder=model_config.get("freeze_encoder", False),
    )

    return model


def create_loss_function(config: dict) -> nn.Module:
    """Create loss function from config.

    Args:
        config: Loss configuration dictionary

    Returns:
        Loss function
    """
    loss_config = config.get("loss", {})

    weights = loss_config.get("weights", {
        "scale_invariant": 1.0,
        "gradient": 0.5,
        "ssim": 0.3,
    })

    return CombinedDepthLoss(weights=weights)


def create_data_config(config: dict) -> DepthDataConfig:
    """Create data configuration from config.

    Args:
        config: Data configuration dictionary

    Returns:
        DepthDataConfig object
    """
    data_config = config.get("data", {})

    return DepthDataConfig(
        train_dir=data_config.get("train_dir", "data/architectural/train"),
        val_dir=data_config.get("val_dir", "data/architectural/val"),
        image_size=tuple(data_config.get("image_size", [518, 518])),
        augmentation=data_config.get("augmentation", True),
        normalize=data_config.get("normalize", True),
    )


def create_training_config(config: dict) -> TrainingConfig:
    """Create training configuration from config.

    Args:
        config: Training configuration dictionary

    Returns:
        TrainingConfig object
    """
    train_config = config.get("training", {})
    checkpoint_config = config.get("checkpointing", {})
    logging_config = config.get("logging", {})
    early_stopping_config = config.get("early_stopping", {})

    return TrainingConfig(
        # Training
        num_epochs=train_config.get("num_epochs", 50),
        batch_size=train_config.get("batch_size", 8),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
        learning_rate=train_config.get("learning_rate", 1e-5),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_epochs=train_config.get("warmup_epochs", 2),
        mixed_precision=train_config.get("mixed_precision", "fp16"),
        num_workers=train_config.get("num_workers", 8),
        pin_memory=train_config.get("pin_memory", True),
        # Checkpointing
        save_dir=checkpoint_config.get("save_dir", "checkpoints/depth_anything_v2_large"),
        log_dir=logging_config.get("log_dir", "logs/depth_anything_v2_large"),
        save_every_n_epochs=checkpoint_config.get("save_every_n_epochs", 5),
        keep_last_n=checkpoint_config.get("keep_last_n", 3),
        save_best=checkpoint_config.get("save_best", True),
        monitor_metric=checkpoint_config.get("monitor_metric", "val_rmse"),
        mode=checkpoint_config.get("mode", "min"),
        # Early stopping
        early_stopping_patience=(
            early_stopping_config.get("patience", 10)
            if early_stopping_config.get("enabled", True)
            else 0
        ),
        # Logging
        log_every_n_steps=logging_config.get("log_every_n_steps", 50),
        save_images_every_n_epochs=logging_config.get("save_images_every_n_epochs", 1),
        tensorboard=logging_config.get("tensorboard", True),
    )


def setup_distributed() -> tuple:
    """Set up distributed training if available.

    Returns:
        Tuple of (local_rank, world_size)
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        logger.info(f"Distributed training: rank {local_rank}/{world_size}")

    return local_rank, world_size


def main(args: argparse.Namespace) -> int:
    """Main training function.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success)
    """
    # Load configuration
    config = load_config(args.config)

    # Validate configuration
    warnings = validate_config(config)
    if warnings:
        logger.warning(f"Configuration warnings: {warnings}")

    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Set up logging
    log_dir = config.get("logging", {}).get("log_dir", "logs/depth_anything_v2")
    create_logger("training", log_dir=log_dir)

    # Set up device
    device, world_size = setup_device(
        use_cuda=True,
        use_mps=True,
        distributed=args.distributed,
    )

    logger.info(f"Using device: {device}")

    # Create model
    logger.info("Creating model...")
    model = create_model(config)

    # Multi-GPU support
    if torch.cuda.device_count() > 1 and not args.distributed:
        logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Create data configuration
    data_config = create_data_config(config)

    # Create augmentations
    aug_config = config.get("augmentation", {})
    image_size = data_config.image_size

    train_transform = get_train_augmentations(aug_config, image_size)
    val_transform = get_val_augmentations(image_size)

    # Create data loaders
    logger.info("Creating data loaders...")
    training_config = create_training_config(config)

    try:
        loaders = create_data_loaders(
            data_config,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=training_config.batch_size,
            num_workers=training_config.num_workers,
            pin_memory=training_config.pin_memory,
        )
    except FileNotFoundError as e:
        logger.error(f"Data directory not found: {e}")
        logger.error("Please prepare your training data first.")
        logger.error("See: scripts/training/prepare_training_data.py")
        return 1

    # Create loss function
    loss_fn = create_loss_function(config)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = DepthTrainer(
        model=model,
        config=training_config,
        loss_fn=loss_fn,
        device=device,
    )

    # Train
    logger.info("Starting training...")
    try:
        history = trainer.fit(
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            resume_from=args.resume,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1

    # Print final summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best {training_config.monitor_metric}: {trainer.best_metric:.4f}")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Checkpoints saved to: {training_config.save_dir}")
    logger.info(f"Logs saved to: {training_config.log_dir}")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train Depth Anything V2 for architectural depth estimation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/training/depth_anything_v2_large_finetune.yaml",
        help="Path to training configuration file"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use distributed training"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        sys.exit(main(args))
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
