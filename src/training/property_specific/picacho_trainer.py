#!/usr/bin/env python3
"""
Multi-Stage Property-Specific Trainer for 750 Picacho Lane.

This module implements a 3-stage training pipeline optimized for
luxury real estate enhancement:
- Stage 1: Material Learning (20 epochs)
- Stage 2: Architectural Refinement (20 epochs)
- Stage 3: Full-Resolution Fine-tuning (10 epochs)

Features:
- Progressive resolution training
- Material-aware loss weighting
- Depth-guided enhancement learning
- Checkpoint management and early stopping
- Apple Silicon / CUDA optimization

Author: Transformation_Portal Enhancement Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
import logging

import numpy as np
from PIL import Image

# Optional ML imports
try:
    import torch
    from torch import nn, Tensor
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any
    nn = None  # Type stub for when PyTorch is not available
    Dataset = object
    DataLoader = object

logger = logging.getLogger(__name__)


class TrainingStage(Enum):
    """Training stages for property-specific enhancement."""
    MATERIAL_LEARNING = "stage1_material"
    ARCHITECTURAL_REFINEMENT = "stage2_architectural"
    FULL_RESOLUTION = "stage3_full_resolution"


@dataclass
class TrainingConfig:
    """Configuration for property-specific training."""
    # Model configuration
    model_name: str = "750_picacho_enhancer"
    base_model: str = "hyper_reality_v3"

    # Training stages
    stage1_epochs: int = 20
    stage2_epochs: int = 20
    stage3_epochs: int = 10
    total_epochs: int = 50

    # Learning rates per stage
    stage1_lr: float = 1e-4
    stage2_lr: float = 5e-5
    stage3_lr: float = 1e-5

    # Batch sizes per stage
    stage1_batch_size: int = 8
    stage2_batch_size: int = 4
    stage3_batch_size: int = 2

    # Resolutions per stage
    stage1_resolution: int = 512
    stage2_resolution: int = 1024
    stage3_resolution: int = 2048

    # Loss weights
    mse_weight: float = 1.0
    perceptual_weight: float = 1.0
    style_weight: float = 0.5
    depth_weight: float = 0.3
    material_weight: float = 0.5

    # Optimization
    optimizer: str = "adamw"
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    use_mixed_precision: bool = True

    # Scheduling
    scheduler: str = "cosine"
    warmup_epochs: int = 3

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("weights/750_picacho"))
    save_every: int = 5
    keep_last_n: int = 3

    # Data
    data_dir: Path = field(default_factory=lambda: Path("data/training_750picacho"))
    num_workers: int = 4
    pin_memory: bool = True

    # Hardware
    device: str = "auto"

    # Validation
    val_frequency: int = 1
    early_stopping_patience: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "stage1_epochs": self.stage1_epochs,
            "stage2_epochs": self.stage2_epochs,
            "stage3_epochs": self.stage3_epochs,
            "stage1_lr": self.stage1_lr,
            "stage2_lr": self.stage2_lr,
            "stage3_lr": self.stage3_lr,
            "checkpoint_dir": str(self.checkpoint_dir),
            "data_dir": str(self.data_dir),
        }

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "TrainingConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            TrainingConfig instance

        Raises:
            FileNotFoundError: If YAML file does not exist
            ValueError: If YAML file is malformed or invalid
        """
        import yaml

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration file: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("YAML file must contain a dictionary at the root level")

        # Create default instance for fallback values
        defaults = cls()

        # Map YAML keys to dataclass fields
        config_data = {}
        training_config = data.get("training", {})

        config_data["model_name"] = data.get("name", defaults.model_name)
        config_data["stage1_epochs"] = training_config.get("stage1", {}).get(
            "epochs", defaults.stage1_epochs
        )
        config_data["stage2_epochs"] = training_config.get("stage2", {}).get(
            "epochs", defaults.stage2_epochs
        )
        config_data["stage3_epochs"] = training_config.get("stage3", {}).get(
            "epochs", defaults.stage3_epochs
        )
        config_data["stage1_lr"] = training_config.get("stage1", {}).get(
            "learning_rate", defaults.stage1_lr
        )
        config_data["stage2_lr"] = training_config.get("stage2", {}).get(
            "learning_rate", defaults.stage2_lr
        )
        config_data["stage3_lr"] = training_config.get("stage3", {}).get(
            "learning_rate", defaults.stage3_lr
        )

        # Paths
        if "output" in data:
            config_data["checkpoint_dir"] = Path(data["output"].get(
                "checkpoint_dir", str(defaults.checkpoint_dir)
            ))
        if "data" in data:
            config_data["data_dir"] = Path(data["data"].get(
                "directory", str(defaults.data_dir)
            ))

        return cls(**config_data)


class PropertyEnhancementDataset(Dataset if TORCH_AVAILABLE else object):
    """Dataset for property-specific enhancement training."""

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        resolution: int = 512,
        include_depth: bool = True,
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Path to dataset directory
            split: Dataset split (train/val/test)
            resolution: Target resolution for images
            include_depth: Whether to load depth maps
            transform: Optional additional transforms
        """
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split
        self.resolution = resolution
        self.include_depth = include_depth
        self.transform = transform

        # Find all samples
        self.samples = self._load_samples()

        # Base transforms
        self.to_tensor = transforms.ToTensor() if TORCH_AVAILABLE else None
        self.resize = transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.LANCZOS
        ) if TORCH_AVAILABLE else None

    def _load_samples(self) -> List[Dict[str, Path]]:
        """Load sample paths from metadata."""
        samples = []

        images_dir = self.split_dir / "images"
        depth_dir = self.split_dir / "depth"

        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return samples

        # Load metadata if available
        metadata_path = self.split_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            for item in metadata:
                sample_id = item["sample_id"]
                image_path = images_dir / f"{sample_id}.png"
                depth_path = depth_dir / f"{sample_id}_depth.png"

                if image_path.exists():
                    sample = {
                        "image": image_path,
                        "metadata": item,
                    }
                    if self.include_depth and depth_path.exists():
                        sample["depth"] = depth_path
                    samples.append(sample)
        else:
            # Fallback: scan directory
            for image_path in sorted(images_dir.glob("*.png")):
                sample_id = image_path.stem
                depth_path = depth_dir / f"{sample_id}_depth.png"

                sample = {"image": image_path}
                if self.include_depth and depth_path.exists():
                    sample["depth"] = depth_path
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image"]).convert("RGB")
        if self.resize:
            image = self.resize(image)
        if self.to_tensor:
            image = self.to_tensor(image)

        result = {"image": image}

        # Load depth if available
        if "depth" in sample:
            depth = Image.open(sample["depth"])
            if self.resize:
                depth = depth.resize(
                    (self.resolution, self.resolution),
                    Image.Resampling.BILINEAR
                )
            depth_array = np.array(depth).astype(np.float32)
            # Normalize 16-bit to 0-1
            if depth_array.max() > 1:
                depth_array = depth_array / 65535.0
            if TORCH_AVAILABLE:
                result["depth"] = torch.from_numpy(depth_array).unsqueeze(0)

        # Include metadata
        if "metadata" in sample:
            result["metadata"] = sample["metadata"]

        return result


class MaterialAwareLoss(nn.Module if TORCH_AVAILABLE else object):
    """Material-aware loss function for property-specific training."""

    def __init__(
        self,
        mse_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        style_weight: float = 0.5,
        depth_weight: float = 0.3,
        material_weight: float = 0.5
    ):
        """
        Initialize loss function.

        Args:
            mse_weight: Weight for MSE loss
            perceptual_weight: Weight for perceptual (VGG) loss
            style_weight: Weight for style loss
            depth_weight: Weight for depth consistency loss
            material_weight: Weight for material-specific losses
        """
        if TORCH_AVAILABLE:
            super().__init__()

        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.depth_weight = depth_weight
        self.material_weight = material_weight

        # Initialize VGG for perceptual loss
        self.vgg = None
        if TORCH_AVAILABLE and perceptual_weight > 0:
            self._init_vgg()

    def _init_vgg(self) -> None:
        """Initialize VGG feature extractor."""
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            # Use layers for perceptual loss
            self.vgg = nn.Sequential(*list(vgg.children())[:16])
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        except Exception as e:
            logger.warning(f"Could not initialize VGG: {e}")
            self.vgg = None

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        depth_pred: Optional[Tensor] = None,
        depth_target: Optional[Tensor] = None,
        materials: Optional[List[str]] = None
    ) -> Dict[str, Tensor]:
        """
        Compute combined loss.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            depth_pred: Predicted depth [B, 1, H, W]
            depth_target: Target depth [B, 1, H, W]
            materials: List of material types in batch

        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}

        # MSE loss
        losses["mse"] = F.mse_loss(pred, target) * self.mse_weight

        # Perceptual loss
        if self.vgg is not None and self.perceptual_weight > 0:
            with torch.no_grad():
                target_features = self.vgg(target)
            pred_features = self.vgg(pred)
            losses["perceptual"] = F.mse_loss(
                pred_features, target_features
            ) * self.perceptual_weight
        else:
            losses["perceptual"] = torch.tensor(0.0, device=pred.device)

        # Style loss (Gram matrix)
        if self.style_weight > 0 and self.vgg is not None:
            with torch.no_grad():
                target_gram = self._gram_matrix(self.vgg(target))
            pred_gram = self._gram_matrix(self.vgg(pred))
            losses["style"] = F.mse_loss(pred_gram, target_gram) * self.style_weight
        else:
            losses["style"] = torch.tensor(0.0, device=pred.device)

        # Depth consistency loss
        if depth_pred is not None and depth_target is not None and self.depth_weight > 0:
            losses["depth"] = F.mse_loss(depth_pred, depth_target) * self.depth_weight
        else:
            losses["depth"] = torch.tensor(0.0, device=pred.device)

        # Material-specific losses
        if materials and self.material_weight > 0:
            losses["material"] = self._material_loss(
                pred, target, materials
            ) * self.material_weight
        else:
            losses["material"] = torch.tensor(0.0, device=pred.device)

        # Total loss
        losses["total"] = sum(losses.values())

        return losses

    def _gram_matrix(self, features: Tensor) -> Tensor:
        """Compute Gram matrix for style loss."""
        b, c, h, w = features.shape
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def _material_loss(
        self,
        pred: Tensor,
        target: Tensor,
        materials: List[str]
    ) -> Tensor:
        """Compute material-specific loss component."""
        loss = torch.tensor(0.0, device=pred.device)

        # Material-specific weighting (used for advanced loss computation)
        # Currently using simplified edge-aware loss; weights below for future extensions:
        # stone: texture=1.2, color=1.0; glass: reflection=1.3, clarity=1.2;
        # water: color=1.2, texture=0.8; wood: grain=1.2, warmth=1.1;
        # metal: highlight=1.3, contrast=1.2; fabric: texture=1.1, softness=1.0

        # Enhanced edge-aware loss for materials that benefit from texture
        if any(m in ["stone", "wood", "fabric"] for m in materials):
            # Sobel edge detection
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=pred.dtype,
                device=pred.device
            ).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

            pred_edges = F.conv2d(pred, sobel_x, padding=1, groups=3)
            target_edges = F.conv2d(target, sobel_x, padding=1, groups=3)
            loss = loss + F.mse_loss(pred_edges, target_edges) * 0.5

        return loss


class PicachoTrainer:
    """
    Multi-stage trainer for 750 Picacho Lane property-specific models.

    Implements progressive training with:
    - Stage 1: Material Learning (512px, 20 epochs)
    - Stage 2: Architectural Refinement (1024px, 20 epochs)
    - Stage 3: Full-Resolution Fine-tuning (2048px, 10 epochs)

    Attributes:
        config: Training configuration
        model: Enhancement model
        device: Compute device
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        config_path: Optional[Path] = None
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            config_path: Path to YAML configuration file
        """
        if config_path:
            self.config = TrainingConfig.from_yaml(config_path)
        else:
            self.config = config or TrainingConfig()

        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Determine device
        self.device = self._get_device()

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.scaler = None  # For mixed precision

        # Training state
        self.current_stage = TrainingStage.MATERIAL_LEARNING
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "stage": [],
            "lr": [],
        }

        logger.info(f"Initialized PicachoTrainer on device: {self.device}")

    def _get_device(self) -> str:
        """Determine compute device."""
        if self.config.device != "auto":
            return self.config.device

        if not TORCH_AVAILABLE:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _init_model(self) -> None:
        """Initialize enhancement model."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for training")

        # Try to load from enhancements module
        try:
            from enhancements.hyper_reality_enhancement import (
                CausticGenerator,
                AtmosphericSynthesizer,
                MaterialTranscendence,
                SpatialHarmonics,
                EnhancementConfig
            )

            config = EnhancementConfig()
            self.model = nn.ModuleDict({
                "caustics": CausticGenerator(config.quantum_caustics),
                "atmosphere": AtmosphericSynthesizer(config.neural_atmosphere),
                "materials": MaterialTranscendence(config.material_transcendence),
                "harmonics": SpatialHarmonics(config.spatial_harmonics),
            })

        except ImportError:
            # Fallback to simple UNet-like model
            self.model = self._create_simple_model()

        self.model = self.model.to(self.device)
        logger.info(f"Model initialized with {self._count_parameters():,} parameters")

    def _create_simple_model(self) -> "nn.Module":
        """Create simple enhancement model as fallback."""

        class SimpleEnhancer(nn.Module):
            """Simple convolutional enhancement model."""

            def __init__(self, in_channels: int = 3, features: int = 64):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(in_channels, features, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features, features * 2, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.decoder = nn.Sequential(
                    nn.Conv2d(features * 2, features, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features, in_channels, 3, padding=1),
                )

            def forward(self, x: Tensor) -> Tensor:
                features = self.encoder(x)
                output = self.decoder(features)
                return x + output * 0.3  # Residual connection

        return SimpleEnhancer()

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        if isinstance(self.model, nn.ModuleDict):
            return sum(
                p.numel() for m in self.model.values()
                for p in m.parameters() if p.requires_grad
            )
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _init_optimizer(self, lr: float) -> None:
        """Initialize optimizer and scheduler."""
        if isinstance(self.model, nn.ModuleDict):
            params = [p for m in self.model.values() for p in m.parameters()]
        else:
            params = self.model.parameters()

        if self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(params, lr=lr)

        # Scheduler
        if self.config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self._get_stage_epochs()
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )

        # Mixed precision scaler
        if self.config.use_mixed_precision and self.device == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")

    def _get_stage_epochs(self) -> int:
        """Get number of epochs for current stage."""
        stage_epochs = {
            TrainingStage.MATERIAL_LEARNING: self.config.stage1_epochs,
            TrainingStage.ARCHITECTURAL_REFINEMENT: self.config.stage2_epochs,
            TrainingStage.FULL_RESOLUTION: self.config.stage3_epochs,
        }
        return stage_epochs.get(self.current_stage, 20)

    def _get_stage_config(self) -> Dict[str, Any]:
        """Get configuration for current stage."""
        configs = {
            TrainingStage.MATERIAL_LEARNING: {
                "lr": self.config.stage1_lr,
                "batch_size": self.config.stage1_batch_size,
                "resolution": self.config.stage1_resolution,
                "epochs": self.config.stage1_epochs,
            },
            TrainingStage.ARCHITECTURAL_REFINEMENT: {
                "lr": self.config.stage2_lr,
                "batch_size": self.config.stage2_batch_size,
                "resolution": self.config.stage2_resolution,
                "epochs": self.config.stage2_epochs,
            },
            TrainingStage.FULL_RESOLUTION: {
                "lr": self.config.stage3_lr,
                "batch_size": self.config.stage3_batch_size,
                "resolution": self.config.stage3_resolution,
                "epochs": self.config.stage3_epochs,
            },
        }
        return configs[self.current_stage]

    def train(self) -> Dict[str, Any]:
        """
        Execute full training pipeline.

        Returns:
            Training results and metrics
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for training")

        logger.info("=" * 60)
        logger.info("750 PICACHO LANE PROPERTY-SPECIFIC TRAINING")
        logger.info("=" * 60)

        # Initialize model
        self._init_model()

        # Initialize loss function
        self.loss_fn = MaterialAwareLoss(
            mse_weight=self.config.mse_weight,
            perceptual_weight=self.config.perceptual_weight,
            style_weight=self.config.style_weight,
            depth_weight=self.config.depth_weight,
            material_weight=self.config.material_weight,
        )
        self.loss_fn.vgg = self.loss_fn.vgg.to(self.device) if self.loss_fn.vgg else None

        # Train each stage
        stages = [
            TrainingStage.MATERIAL_LEARNING,
            TrainingStage.ARCHITECTURAL_REFINEMENT,
            TrainingStage.FULL_RESOLUTION,
        ]

        for stage in stages:
            self.current_stage = stage
            logger.info(f"\n{'=' * 60}")
            logger.info(f"STAGE: {stage.value}")
            logger.info(f"{'=' * 60}\n")

            self._train_stage()

        # Save final model
        self._save_checkpoint(final=True)

        # Generate training report
        report = self._generate_report()

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        logger.info(f"Checkpoints saved to: {self.config.checkpoint_dir}")

        return report

    def _train_stage(self) -> None:
        """Train current stage."""
        stage_config = self._get_stage_config()

        # Initialize optimizer for this stage
        self._init_optimizer(stage_config["lr"])

        # Create dataloaders
        train_loader = self._create_dataloader(
            split="train",
            resolution=stage_config["resolution"],
            batch_size=stage_config["batch_size"]
        )
        val_loader = self._create_dataloader(
            split="val",
            resolution=stage_config["resolution"],
            batch_size=stage_config["batch_size"]
        )

        logger.info(f"Resolution: {stage_config['resolution']}px")
        logger.info(f"Batch size: {stage_config['batch_size']}")
        logger.info(f"Learning rate: {stage_config['lr']}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")

        # Training loop
        for epoch in range(stage_config["epochs"]):
            self.current_epoch = epoch

            # Train epoch
            train_loss = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["stage"].append(self.current_stage.value)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Validation
            if (epoch + 1) % self.config.val_frequency == 0:
                val_loss = self._validate(val_loader)
                self.history["val_loss"].append(val_loss)

                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train() if not isinstance(self.model, nn.ModuleDict) else None
        if isinstance(self.model, nn.ModuleDict):
            for m in self.model.values():
                m.train()

        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            image = batch["image"].to(self.device)
            depth = batch.get("depth")
            if depth is not None:
                depth = depth.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast("cuda"):
                    pred = self._forward(image, depth)
                    losses = self.loss_fn(pred, image, depth_target=depth)
                    loss = losses["total"]

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._get_parameters(),
                    self.config.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self._forward(image, depth)
                losses = self.loss_fn(pred, image, depth_target=depth)
                loss = losses["total"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._get_parameters(),
                    self.config.gradient_clip
                )
                self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            # Logging
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(
                    f"  [{self.current_stage.value}] "
                    f"Epoch {self.current_epoch + 1} "
                    f"[{batch_idx + 1}/{num_batches}] "
                    f"Loss: {avg_loss:.4f}"
                )

        return total_loss / num_batches

    def _validate(self, dataloader: DataLoader) -> float:
        """Validate model."""
        self.model.eval() if not isinstance(self.model, nn.ModuleDict) else None
        if isinstance(self.model, nn.ModuleDict):
            for m in self.model.values():
                m.eval()

        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                image = batch["image"].to(self.device)
                depth = batch.get("depth")
                if depth is not None:
                    depth = depth.to(self.device)

                pred = self._forward(image, depth)
                losses = self.loss_fn(pred, image, depth_target=depth)
                total_loss += losses["total"].item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"  Validation Loss: {avg_loss:.6f}")

        return avg_loss

    def _forward(self, image: Tensor, depth: Optional[Tensor] = None) -> Tensor:
        """Forward pass through model."""
        if isinstance(self.model, nn.ModuleDict):
            # Multi-module architecture
            enhanced = image

            # Estimate depth if not provided
            if depth is None:
                depth = self._estimate_depth(image)

            # Compute normals from depth
            normals = self._compute_normals(depth)

            # Stage 1: Caustics
            if "caustics" in self.model:
                caustics = self.model["caustics"](enhanced, depth)
                enhanced = enhanced + caustics * 0.3

            # Stage 2: Atmosphere
            if "atmosphere" in self.model:
                enhanced = self.model["atmosphere"](enhanced)

            # Stage 3: Materials
            if "materials" in self.model:
                enhanced = self.model["materials"](enhanced)

            # Stage 4: Harmonics
            if "harmonics" in self.model:
                illumination = self.model["harmonics"](normals)
                enhanced = enhanced * (1 + illumination * 0.3)

            return enhanced
        else:
            return self.model(image)

    def _estimate_depth(self, image: Tensor) -> Tensor:
        """Estimate depth from image (simple luminance-based)."""
        gray = torch.mean(image, dim=1, keepdim=True)
        return 1.0 - gray

    def _compute_normals(self, depth: Tensor) -> Tensor:
        """Compute surface normals from depth."""
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=depth.dtype,
            device=depth.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=depth.dtype,
            device=depth.device
        ).view(1, 1, 3, 3)

        dx = F.conv2d(depth, sobel_x, padding=1)
        dy = F.conv2d(depth, sobel_y, padding=1)
        dz = torch.ones_like(dx) * 0.5

        normals = torch.cat([dx, dy, dz], dim=1)
        normals = F.normalize(normals, dim=1)

        return normals

    def _get_parameters(self) -> List[Any]:
        """Get all model parameters."""
        if isinstance(self.model, nn.ModuleDict):
            return [p for m in self.model.values() for p in m.parameters()]
        return list(self.model.parameters())

    def _create_dataloader(
        self,
        split: str,
        resolution: int,
        batch_size: int
    ) -> DataLoader:
        """Create dataloader for split."""
        dataset = PropertyEnhancementDataset(
            data_dir=self.config.data_dir,
            split=split,
            resolution=resolution,
            include_depth=True
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

    def _save_checkpoint(self, is_best: bool = False, final: bool = False) -> None:
        """Save model checkpoint."""
        if isinstance(self.model, nn.ModuleDict):
            model_state = {k: v.state_dict() for k, v in self.model.items()}
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            "epoch": self.current_epoch,
            "stage": self.current_stage.value,
            "global_step": self.global_step,
            "model_state": model_state,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config.to_dict(),
        }

        if final:
            path = self.config.checkpoint_dir / "final_model.pth"
        elif is_best:
            path = self.config.checkpoint_dir / "best_model.pth"
        else:
            path = self.config.checkpoint_dir / f"checkpoint_step_{self.global_step}.pth"

        torch.save(checkpoint, path)
        logger.info(f"  ✓ Saved checkpoint: {path.name}")

    def _generate_report(self) -> Dict[str, Any]:
        """Generate training report."""
        return {
            "model_name": self.config.model_name,
            "property": "750 Picacho Lane",
            "total_epochs": sum([
                self.config.stage1_epochs,
                self.config.stage2_epochs,
                self.config.stage3_epochs
            ]),
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            "checkpoint_dir": str(self.config.checkpoint_dir),
            "device": self.device,
            "history": self.history,
        }

    def __repr__(self) -> str:
        return (
            f"PicachoTrainer(stage={self.current_stage.value}, "
            f"device={self.device})"
        )
