#!/usr/bin/env python3
"""
Model Loader for Hyper-Reality Enhancement
Handles loading pre-trained weights and checkpoint management

Author: Transformation_Portal Enhancement Team
Version: 1.0.0
"""

# pylint: disable=redefined-outer-name

import warnings
from pathlib import Path
from typing import Dict, Optional, Any
import torch

warnings.filterwarnings('ignore')


class ModelLoader:
    """Load and manage trained model weights"""

    def __init__(self, checkpoint_dir: str = "weights/hyper_reality"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = self._get_device()

    def _get_device(self):
        """Get optimal device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def load_best_model(self) -> Optional[Dict[str, Any]]:
        """Load best trained model checkpoint"""
        best_path = self.checkpoint_dir / "best_model.pth"

        if not best_path.exists():
            return None

        try:
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=True)
            return checkpoint
        except Exception as e:
            warnings.warn(f"Failed to load checkpoint from {best_path}: {e}")
            return None

    def load_checkpoint(self, epoch: int) -> Optional[Dict[str, Any]]:
        """Load specific epoch checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

        if not checkpoint_path.exists():
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            return checkpoint
        except Exception as e:
            warnings.warn(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return None

    def load_model_weights(self,
                          models: Dict[str, torch.nn.Module],
                          checkpoint: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load weights into models

        Args:
            models: Dictionary of model name -> model instance
            checkpoint: Checkpoint dict (if None, loads best model)

        Returns:
            True if weights loaded successfully, False otherwise
        """
        if checkpoint is None:
            checkpoint = self.load_best_model()

        if checkpoint is None:
            return False

        try:
            model_states = checkpoint.get('models', {})

            for name, model in models.items():
                if name in model_states:
                    model.load_state_dict(model_states[name])
                    model.to(self.device)

            return True
        except Exception as e:
            warnings.warn(f"Failed to load model weights: {e}")
            return False

    def get_available_checkpoints(self) -> list:
        """List all available checkpoints"""
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = []

        # Check for best model
        if (self.checkpoint_dir / "best_model.pth").exists():
            checkpoints.append("best_model")

        # Check for epoch checkpoints
        for path in sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth")):
            checkpoints.append(path.stem)

        return checkpoints

    def checkpoint_info(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract information from checkpoint"""
        return {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_val_loss': checkpoint.get('best_val_loss', None),
            'config': checkpoint.get('config', {}),
            'models': list(checkpoint.get('models', {}).keys()),
        }


def load_pretrained_weights(models: Dict[str, torch.nn.Module],
                            checkpoint_dir: str = "weights/hyper_reality",
                            verbose: bool = True) -> bool:
    """
    Convenience function to load pre-trained weights into models

    Args:
        models: Dictionary of model_name -> model_instance
        checkpoint_dir: Directory containing checkpoints
        verbose: Print loading status

    Returns:
        True if weights loaded successfully
    """
    loader = ModelLoader(checkpoint_dir)

    if verbose:
        checkpoints = loader.get_available_checkpoints()
        if not checkpoints:
            print("⚠️  No pre-trained weights found. Using random initialization.")
            print("   Train models with: python src/enhancements/train_hyper_reality.py")
            return False

        print(f"✓ Loading pre-trained weights from: {checkpoint_dir}")

    success = loader.load_model_weights(models)

    if verbose:
        if success:
            print("✓ Pre-trained weights loaded successfully")
        else:
            print("⚠️  Failed to load weights. Using random initialization.")

    return success


if __name__ == "__main__":
    # Test model loader
    loader = ModelLoader()
    checkpoints = loader.get_available_checkpoints()

    print(f"Available checkpoints: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"  - {cp}")

    if checkpoints:
        checkpoint = loader.load_best_model()
        if checkpoint:
            info = loader.checkpoint_info(checkpoint)
            print("\nBest model info:")
            print(f"  Epoch: {info['epoch']}")
            print(f"  Val Loss: {info['best_val_loss']}")
            print(f"  Models: {info['models']}")
