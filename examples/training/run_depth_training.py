#!/usr/bin/env python3
"""
Example: Running Depth Anything V2 Training

This example demonstrates how to use the training pipeline to fine-tune
Depth Anything V2 on architectural imagery.

Prerequisites:
    1. Install dependencies:
       pip install torch torchvision transformers tqdm tensorboard

    2. Prepare your dataset:
       python scripts/training/prepare_training_data.py --create-sample

    3. Or use your own data in the structure:
       data/architectural/
           train/
               images/
               depth/
           val/
               images/
               depth/

Usage:
    python examples/training/run_depth_training.py

Author: Transformation Portal Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Check dependencies
try:
    import torch
    import yaml
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch pyyaml")
    sys.exit(1)


def main():
    """Run depth training example."""
    print("=" * 60)
    print("Depth Anything V2 Fine-tuning Example")
    print("=" * 60)
    print()

    # Check for CUDA/MPS availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple MPS device")
    else:
        device = "cpu"
        print("Using CPU device (training will be slow)")
    print()

    # Import training components
    try:
        from src.training import (
            DepthTrainer,
            TrainingConfig,
            DepthDataConfig,
            create_data_loaders,
            get_train_augmentations,
            get_val_augmentations,
            CombinedDepthLoss,
            set_seed,
        )
        from src.training.train_depth_anything_v2 import DepthAnythingV2Wrapper
    except ImportError as e:
        print(f"Error importing training modules: {e}")
        print("Make sure you're running from the repository root.")
        sys.exit(1)

    # Configuration
    config_path = Path("config/training/depth_anything_v2_large_finetune.yaml")

    if config_path.exists():
        print(f"Loading configuration from {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print("Using default configuration")
        config = {
            "model": {
                "name": "depth-anything/Depth-Anything-V2-Small-hf",  # Use Small for demo
                "pretrained": True,
                "freeze_encoder": False,
            },
            "training": {
                "num_epochs": 5,  # Short demo
                "batch_size": 4,
                "learning_rate": 1e-5,
            },
            "data": {
                "train_dir": "data/architectural/train",
                "val_dir": "data/architectural/val",
                "image_size": [518, 518],
            },
        }

    # Set random seed for reproducibility
    set_seed(42)
    print("Set random seed to 42")

    # Check if data exists
    train_dir = Path(config["data"]["train_dir"])
    val_dir = Path(config["data"]["val_dir"])

    if not train_dir.exists():
        print(f"\nTraining data not found at {train_dir}")
        print("Creating synthetic sample data for demonstration...")
        print("Run: python scripts/training/prepare_training_data.py --create-sample")

        # Create minimal sample data inline for demo
        create_demo_data(train_dir, num_samples=20)
        create_demo_data(val_dir, num_samples=5)
        print()

    # Create data configuration
    data_config = DepthDataConfig(
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        image_size=tuple(config["data"]["image_size"]),
    )

    # Create augmentations
    train_transform = get_train_augmentations(
        config.get("augmentation", {}),
        data_config.image_size,
    )
    val_transform = get_val_augmentations(data_config.image_size)

    # Create data loaders
    print("Creating data loaders...")
    try:
        loaders = create_data_loaders(
            data_config,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=config["training"]["batch_size"],
            num_workers=2,  # Reduced for demo
        )
        print(f"  Train batches: {len(loaders['train'])}")
        print(f"  Val batches: {len(loaders['val'])}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please prepare training data first.")
        return 1

    # Create model
    print("\nLoading model...")
    model_config = config.get("model", {})

    # Use smaller model for demo if running on CPU
    if device == "cpu":
        model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        print("  Using Small model for CPU demo")
    else:
        model_name = model_config.get("name", "depth-anything/Depth-Anything-V2-Small-hf")

    try:
        model = DepthAnythingV2Wrapper(
            model_name=model_name,
            pretrained=model_config.get("pretrained", True),
            freeze_encoder=model_config.get("freeze_encoder", False),
        )
        print(f"  Model: {model_name}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure transformers is installed: pip install transformers")
        return 1

    # Create training configuration
    train_config = TrainingConfig(
        num_epochs=config["training"].get("num_epochs", 5),
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        save_dir="checkpoints/demo",
        log_dir="logs/demo",
        save_every_n_epochs=1,
        early_stopping_patience=0,  # Disable for demo
        tensorboard=True,
    )

    # Create loss function
    loss_fn = CombinedDepthLoss(
        weights={
            "scale_invariant": 1.0,
            "gradient": 0.5,
            "ssim": 0.3,
        }
    )

    # Create trainer
    print("\nCreating trainer...")
    trainer = DepthTrainer(
        model=model,
        config=train_config,
        loss_fn=loss_fn,
        device=torch.device(device),
    )

    # Train!
    print("\nStarting training...")
    print("=" * 60)

    try:
        history = trainer.fit(
            train_loader=loaders["train"],
            val_loader=loaders["val"],
        )

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"\nFinal train loss: {history['train_loss'][-1]:.4f}")
        if history['val_loss']:
            print(f"Final val loss: {history['val_loss'][-1]:.4f}")
            print(f"Final val RMSE: {history['val_rmse'][-1]:.4f}")

        print(f"\nCheckpoints saved to: {train_config.save_dir}")
        print(f"Logs saved to: {train_config.log_dir}")
        print("\nView training progress with TensorBoard:")
        print(f"  tensorboard --logdir {train_config.log_dir}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 1

    return 0


def create_demo_data(output_dir: Path, num_samples: int = 20) -> None:
    """Create minimal demo data.

    Args:
        output_dir: Output directory
        num_samples: Number of samples
    """
    import numpy as np
    from PIL import Image

    images_dir = output_dir / "images"
    depth_dir = output_dir / "depth"

    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        # Simple synthetic image
        h, w = 256, 256  # Small for speed
        image = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)

        # Simple depth map
        y, x = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
        depth = (x + y) * 50 + np.random.randn(h, w) * 2
        depth = depth.astype(np.float32)

        Image.fromarray(image).save(images_dir / f"demo_{i:03d}.png")
        np.save(depth_dir / f"demo_{i:03d}.npy", depth)


if __name__ == "__main__":
    sys.exit(main())
