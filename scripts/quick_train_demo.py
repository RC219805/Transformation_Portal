#!/usr/bin/env python3
"""
Quick Training Demonstration
This script demonstrates the training pipeline with a minimal dataset and few epochs.
For full production training, use scripts/train_with_750picacho.sh or scripts/quickstart_training.sh

This demo:
- Generates a small synthetic dataset (50 pairs)
- Trains for 3 epochs (to demonstrate functionality)
- Validates the training loop works correctly
- Saves checkpoints

Usage:
    python scripts/quick_train_demo.py
"""

from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from enhancements.train_hyper_reality import (
    TrainingConfig,
    SyntheticDataGenerator,
    HyperRealityTrainer,
    EnhancementDataset,
    configure_device
)
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    """Run a quick training demonstration"""
    print("=" * 70)
    print("QUICK TRAINING DEMONSTRATION")
    print("=" * 70)
    print()
    print("This is a lightweight demo to validate the training infrastructure.")
    print("For full production training, use:")
    print("  - ./scripts/train_with_750picacho.sh (recommended - real data)")
    print("  - ./scripts/quickstart_training.sh (synthetic data)")
    print()

    # Configure device
    device = configure_device()
    print(f"Using device: {device}")
    print()

    # Step 1: Generate small synthetic dataset
    print("Step 1: Generating synthetic dataset (50 pairs)...")
    print("-" * 70)

    output_dir = "data/training_demo"
    generator = SyntheticDataGenerator(output_dir, num_pairs=50)
    generator.generate_training_data()

    print(f"✓ Generated 50 training pairs in {output_dir}")
    print()

    # Step 2: Configure training for quick demo
    print("Step 2: Configuring training (3 epochs for demo)...")
    print("-" * 70)

    config = TrainingConfig(
        data_dir=output_dir,
        synthetic_data=False,  # Data already generated
        batch_size=2,  # Small batch for quick demo
        num_epochs=3,  # Just 3 epochs for demonstration
        learning_rate=1e-4,
        checkpoint_dir="weights/hyper_reality_demo",
        save_frequency=1,  # Save every epoch for demo
        val_split=0.2,  # 20% validation
        val_frequency=1,
        num_workers=2,
        use_mixed_precision=False  # Disable for CPU
    )

    print("Training config:")
    print(f"  - Dataset: {config.data_dir} (50 pairs)")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Checkpoints: {config.checkpoint_dir}")
    print()

    # Step 3: Create datasets and dataloaders
    print("Step 3: Preparing datasets...")
    print("-" * 70)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    low_quality_dir = Path(output_dir) / "low_quality"
    high_quality_dir = Path(output_dir) / "high_quality"

    dataset = EnhancementDataset(low_quality_dir, high_quality_dir, transform)

    # Split into train/val
    val_size = int(config.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False  # Disable for CPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False  # Disable for CPU
    )

    print("✓ Created dataloaders:")
    print(f"  - Training samples: {train_size}")
    print(f"  - Validation samples: {val_size}")
    print()

    # Step 4: Train models
    print("Step 4: Training models (this will take a few minutes)...")
    print("-" * 70)
    print()

    try:
        trainer = HyperRealityTrainer(config)
        trainer.train(train_loader, val_loader)
        print()
        print("=" * 70)
        print("✓ TRAINING DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. For full training on real data:")
        print("     ./scripts/train_with_750picacho.sh")
        print()
        print("  2. For full synthetic data training:")
        print("     ./scripts/quickstart_training.sh")
        print()
        print("  3. Check trained models in:")
        print(f"     {config.checkpoint_dir}/")
        print()

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ TRAINING DEMONSTRATION FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        print("This may be due to:")
        print("  - Insufficient memory")
        print("  - Missing dependencies")
        print("  - GPU/CPU compatibility issues")
        print()
        print("For production training, ensure you have:")
        print("  - At least 8GB RAM")
        print("  - All ML dependencies installed (pip install -r requirements/ml.txt)")
        print("  - GPU with CUDA or Apple Silicon with MPS (recommended)")
        print()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
