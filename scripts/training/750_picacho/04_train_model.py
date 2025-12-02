#!/usr/bin/env python3
"""
Script 04: Train Model for 750 Picacho Lane.

This script runs the multi-stage training pipeline:
- Stage 1: Material Learning (512px, 20 epochs)
- Stage 2: Architectural Refinement (1024px, 20 epochs)
- Stage 3: Full-Resolution Fine-tuning (2048px, 10 epochs)

Usage:
    python scripts/training/750_picacho/04_train_model.py [options]

Author: Transformation_Portal Enhancement Team
Version: 1.0.0
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from training.property_specific.picacho_trainer import PicachoTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train 750 Picacho Lane enhancement model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/training/750_picacho_lane_protocol.yaml"),
        help="Path to training configuration YAML"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/training_750picacho"),
        help="Path to training data directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("weights/750_picacho"),
        help="Path to save checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=20,
        help="Number of epochs for Stage 1"
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=20,
        help="Number of epochs for Stage 2"
    )
    parser.add_argument(
        "--stage3-epochs",
        type=int,
        default=10,
        help="Number of epochs for Stage 3"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 60)
    print("750 PICACHO LANE MODEL TRAINING")
    print("=" * 60 + "\n")

    # Load configuration
    if args.config.exists():
        print(f"Loading configuration from: {args.config}")
        config = TrainingConfig.from_yaml(args.config)
    else:
        print("Using default configuration")
        config = TrainingConfig()

    # Override with command line arguments
    config.data_dir = args.data_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.stage1_epochs = args.stage1_epochs
    config.stage2_epochs = args.stage2_epochs
    config.stage3_epochs = args.stage3_epochs
    config.device = args.device

    # Verify data exists
    if not args.data_dir.exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        print("   Please run 03_generate_dataset.py first.")
        return 1

    print("\nConfiguration:")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Checkpoint directory: {config.checkpoint_dir}")
    print(f"  Stage 1: {config.stage1_epochs} epochs @ 512px")
    print(f"  Stage 2: {config.stage2_epochs} epochs @ 1024px")
    print(f"  Stage 3: {config.stage3_epochs} epochs @ 2048px")
    print(f"  Total epochs: {config.stage1_epochs + config.stage2_epochs + config.stage3_epochs}")
    print(f"  Device: {config.device}")

    # Initialize trainer
    trainer = PicachoTrainer(config=config)

    print(f"\nUsing device: {trainer.device}")

    # Resume from checkpoint if specified
    if args.resume and args.resume.exists():
        print(f"\nResuming from checkpoint: {args.resume}")
        # Load checkpoint logic would go here

    # Train
    print("\nStarting training...")
    try:
        report = trainer.train()

        # Summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"\n✓ Best validation loss: {report['best_val_loss']:.6f}")
        print(f"✓ Checkpoints saved to: {config.checkpoint_dir}")

        # Save training report
        import json
        report_path = config.checkpoint_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"✓ Training report saved to: {report_path}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("   Partial checkpoints may be saved.")
        return 1
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nNext step: Run 05_validate_model.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
