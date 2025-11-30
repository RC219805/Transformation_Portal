#!/usr/bin/env python3
"""
Training Pipeline for Hyper-Reality Enhancement Models
Trains neural networks on internal quality metric through supervised learning

This script provides:
- Perceptual loss training (LPIPS + MSE + Style)
- Synthetic and real data support
- Progressive training strategy
- Apple Silicon optimization
- Model checkpointing and evaluation

Author: Transformation_Portal Enhancement Team
Version: 1.0.0
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhancements.hyper_reality_enhancement import (  # noqa: E402
    CausticGenerator,
    AtmosphericSynthesizer,
    MaterialTranscendence,
    SpatialHarmonics,
    EnhancementConfig,
    configure_device
)

# Optional LPIPS import for enhanced perceptual loss
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configure device
device = configure_device()


@dataclass
class TrainingConfig:
    """Configuration for training hyper-reality models"""

    # Dataset
    data_dir: str = "data/training"
    synthetic_data: bool = True
    num_synthetic_pairs: int = 1000

    # Training hyperparameters
    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Loss weights
    mse_weight: float = 1.0
    perceptual_weight: float = 1.0
    style_weight: float = 0.5
    lpips_weight: float = 1.0  # Weight for LPIPS loss (if available)

    # Progressive training
    progressive: bool = True
    warmup_epochs: int = 5

    # Checkpointing
    checkpoint_dir: str = "weights/hyper_reality"
    save_frequency: int = 5

    # Validation
    val_split: float = 0.1
    val_frequency: int = 1

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True

    # Optimization
    use_mixed_precision: bool = True
    gradient_clip: float = 1.0


class SyntheticDataGenerator:
    """Generate synthetic training pairs for enhancement learning"""

    def __init__(self, output_dir: str, num_pairs: int = 1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_pairs = num_pairs

    def generate_training_data(self):
        """Generate synthetic low→high quality image pairs"""
        print(f"\n{'='*60}")
        print("GENERATING SYNTHETIC TRAINING DATA")
        print(f"{'='*60}\n")

        low_quality_dir = self.output_dir / "low_quality"
        high_quality_dir = self.output_dir / "high_quality"
        low_quality_dir.mkdir(exist_ok=True)
        high_quality_dir.mkdir(exist_ok=True)

        for i in tqdm(range(self.num_pairs), desc="Generating pairs"):
            # Create high-quality synthetic image
            high_quality = self._create_synthetic_image()

            # Create degraded low-quality version
            low_quality = self._degrade_image(high_quality)

            # Save pair
            high_path = high_quality_dir / f"image_{i:04d}.png"
            low_path = low_quality_dir / f"image_{i:04d}.png"

            Image.fromarray(high_quality).save(high_path)
            Image.fromarray(low_quality).save(low_path)

        print(f"\n✓ Generated {self.num_pairs} training pairs")
        print(f"  High quality: {high_quality_dir}")
        print(f"  Low quality: {low_quality_dir}")

    def _create_synthetic_image(self, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Create simplified synthetic architectural image

        Note: This generates basic architectural scenes with gradients and geometric shapes.
        For production training targeting professional architectural quality, use real
        high-quality renders (e.g., from 750 Picacho dataset) via prepare_750picacho_training_data.py

        Synthetic data is useful for:
        - Initial model development and testing
        - Rapid prototyping without requiring large datasets
        - Learning basic enhancement patterns

        Limitations:
        - Does not capture complex architectural details
        - Simplified material properties
        - Limited lighting scenarios
        """
        h, w = size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Sky gradient (blue hour aesthetic)
        sky_height = h // 3
        for y in range(sky_height):
            intensity = 1.0 - (y / sky_height) * 0.5
            img[y, :, 2] = int(180 * intensity)  # Blue
            img[y, :, 1] = int(120 * intensity)  # Green
            img[y, :, 0] = int(80 * intensity)   # Red

        # Building structure
        building_y = sky_height
        building_height = h - sky_height

        # Stucco facade with texture
        facade_color = np.array([240, 235, 220])  # Warm stucco
        noise = np.random.randn(building_height, w) * 10
        for c in range(3):
            img[building_y:, :, c] = np.clip(facade_color[c] + noise, 0, 255)

        # Windows (darker rectangles)
        for window_x in range(50, w - 50, 100):
            for window_y in range(building_y + 30, h - 30, 80):
                window_w, window_h = 60, 50
                # Glass reflection (blue tint)
                img[window_y:window_y+window_h, window_x:window_x+window_w, 2] = 140
                img[window_y:window_y+window_h, window_x:window_x+window_w, 1] = 120
                img[window_y:window_y+window_h, window_x:window_x+window_w, 0] = 100

        # Add subtle lighting variations
        x_gradient = np.linspace(0.9, 1.1, w)
        for c in range(3):
            img[building_y:, :, c] = np.clip(
                img[building_y:, :, c] * x_gradient, 0, 255
            )

        return img.astype(np.uint8)

    def _degrade_image(self, img: np.ndarray) -> np.ndarray:
        """Apply realistic degradations to create low-quality version"""
        degraded = img.copy().astype(np.float32)

        # 1. Reduce contrast
        degraded = (degraded - 128) * 0.7 + 128

        # 2. Add noise
        noise = np.random.randn(*degraded.shape) * 8
        degraded += noise

        # 3. Slight blur (loss of sharpness)
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            degraded[:, :, c] = gaussian_filter(degraded[:, :, c], sigma=0.8)

        # 4. Reduce saturation
        gray = degraded.mean(axis=2, keepdims=True)
        degraded = gray * 0.3 + degraded * 0.7

        # 5. Slight color shift
        degraded[:, :, 0] *= 0.95  # Reduce red
        degraded[:, :, 2] *= 0.98  # Slight blue reduction

        return np.clip(degraded, 0, 255).astype(np.uint8)


class EnhancementDataset(Dataset):
    """Dataset for image enhancement training"""

    def __init__(self, low_quality_dir: str, high_quality_dir: str, transform=None):
        self.low_quality_dir = Path(low_quality_dir)
        self.high_quality_dir = Path(high_quality_dir)
        self.transform = transform

        # Get all image pairs
        self.image_pairs = []
        for low_path in sorted(self.low_quality_dir.glob("*.png")):
            high_path = self.high_quality_dir / low_path.name
            if high_path.exists():
                self.image_pairs.append((low_path, high_path))

        if len(self.image_pairs) == 0:
            raise ValueError(f"No image pairs found in {low_quality_dir} and {high_quality_dir}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        low_path, high_path = self.image_pairs[idx]

        # Load images
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img


class VGGFeatureExtractor(nn.Module):
    """Extract features from pretrained VGG19 for perceptual loss"""

    def __init__(self, layers: list = None, use_input_norm: bool = True):
        """
        Args:
            layers: VGG layer indices to extract features from
                   Default: [2, 7, 12, 21, 30] (conv1_2, conv2_2, conv3_2, conv4_2, conv5_2)
            use_input_norm: Normalize input to ImageNet statistics
        """
        super().__init__()
        self.use_input_norm = use_input_norm

        # Default to standard perceptual loss layers
        if layers is None:
            layers = [2, 7, 12, 21, 30]  # relu1_2, relu2_2, relu3_2, relu4_2, relu5_2

        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except (ImportError, TypeError):
            # Fallback for older torchvision versions
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features

        # Extract required layers
        self.layers = sorted(layers)
        max_layer = max(self.layers) + 1
        self.features = nn.Sequential(*list(vgg.children())[:max_layer])

        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False

        self.features.eval()
        self.features.to(device)

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """Extract multi-scale VGG features"""
        # Normalize input
        if self.use_input_norm:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layers:
                features.append(x)

        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG19 features

    Compares feature representations at multiple VGG layers to measure
    perceptual similarity. More effective than pixel-wise MSE for image
    enhancement tasks.

    References:
        - Johnson et al., "Perceptual Losses for Real-Time Style Transfer"
        - Zhang et al., "The Unreasonable Effectiveness of Deep Features"
    """

    def __init__(self, layers: list = None, weights: list = None):
        """
        Args:
            layers: VGG layer indices for feature extraction
            weights: Weight for each layer's contribution to loss
        """
        super().__init__()

        if layers is None:
            layers = [2, 7, 12, 21, 30]

        if weights is None:
            # Default weights emphasizing mid-level features
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.weights = weights
        self.vgg = VGGFeatureExtractor(layers=layers)

    def forward(self, pred, target):
        """Compute perceptual loss between prediction and target"""
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)

        loss = 0.0
        for w, pred_feat, target_feat in zip(self.weights, pred_features, target_features):
            loss += w * F.mse_loss(pred_feat, target_feat)

        return loss / len(self.weights)


class StyleLoss(nn.Module):
    """
    Style loss using Gram matrices of pretrained VGG19 features

    Compares style patterns by matching Gram matrices at multiple VGG layers.
    Essential for texture and style transfer in image enhancement.

    References:
        - Gatys et al., "A Neural Algorithm of Artistic Style"
        - Johnson et al., "Perceptual Losses for Real-Time Style Transfer"
    """

    def __init__(self, layers: list = None, weights: list = None):
        """
        Args:
            layers: VGG layer indices for style extraction
            weights: Weight for each layer's contribution to loss
        """
        super().__init__()

        if layers is None:
            # Style layers (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
            layers = [0, 5, 10, 19, 28]

        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.weights = weights
        self.vgg = VGGFeatureExtractor(layers=layers)

    def gram_matrix(self, features):
        """Compute Gram matrix for style representation"""
        b, c, h, w = features.shape
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def forward(self, pred, target):
        """Compute style loss between prediction and target"""
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)

        loss = 0.0
        for w, pred_feat, target_feat in zip(self.weights, pred_features, target_features):
            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += w * F.mse_loss(pred_gram, target_gram)

        return loss / len(self.weights)


class HyperRealityTrainer:
    """Main training pipeline for hyper-reality models"""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize models
        self._init_models()

        # Initialize loss functions
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()

        # Initialize LPIPS loss if available
        self.lpips_fn = None
        if LPIPS_AVAILABLE:
            try:
                self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
                self.lpips_fn.eval()  # LPIPS should be in eval mode
                print("✓ LPIPS loss initialized (using VGG backbone)")
            except Exception as e:
                print(f"⚠ LPIPS initialization failed: {e}")
                self.lpips_fn = None
        else:
            print("⚠ LPIPS not available, using VGG perceptual loss only")

        # Initialize optimizer
        all_params = []
        for model in self.models.values():
            all_params.extend(model.parameters())

        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'mse': [],
            'perceptual': [],
            'style': [],
            'lpips': []
        }

    def _init_models(self):
        """Initialize all enhancement models"""
        enhancement_config = EnhancementConfig()

        self.models = {
            'caustics': CausticGenerator(enhancement_config.quantum_caustics).to(device),
            'atmosphere': AtmosphericSynthesizer(enhancement_config.neural_atmosphere).to(device),
            'materials': MaterialTranscendence(enhancement_config.material_transcendence).to(device),
            'harmonics': SpatialHarmonics(enhancement_config.spatial_harmonics).to(device),
        }

        # Set to training mode
        for model in self.models.values():
            model.train()

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop"""
        print(f"\n{'='*60}")
        print("HYPER-REALITY TRAINING PIPELINE")
        print(f"{'='*60}\n")
        print(f"Device: {device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_loader.dataset)}")
        print()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Training phase
            train_loss = self._train_epoch(train_loader, epoch)
            self.training_history['train_loss'].append(train_loss)

            # Validation phase
            if val_loader and (epoch + 1) % self.config.val_frequency == 0:
                val_loss = self._validate(val_loader, epoch)
                self.training_history['val_loss'].append(val_loss)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)

            # Save periodic checkpoints
            if (epoch + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Update learning rate
            self.scheduler.step()

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Checkpoint directory: {self.config.checkpoint_dir}")

    def _estimate_depth(self, img: torch.Tensor) -> torch.Tensor:
        """Estimate depth map from image using luminance inversion.

        This is a simplified heuristic that assumes darker regions are farther
        away. While not geometrically accurate, it provides a useful proxy for
        depth-aware effects during training. For production use with real
        architectural images, consider using Depth Anything V2 or similar
        learned depth estimation models.

        Args:
            img: Input image tensor [B, C, H, W] in [0, 1] range

        Returns:
            Estimated depth map [B, 1, H, W] where higher values = closer
        """
        gray = torch.mean(img, dim=1, keepdim=True)
        depth = 1.0 - gray
        return depth

    def _compute_normals(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute surface normals from depth map using Sobel gradients.

        Args:
            depth: Depth map tensor [B, 1, H, W]

        Returns:
            Surface normals [B, 3, H, W] as unit vectors (x, y, z components)
        """
        # Sobel filters for gradients
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3).to(depth.device)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3).to(depth.device)

        dx = F.conv2d(depth, sobel_x, padding=1)
        dy = F.conv2d(depth, sobel_y, padding=1)
        # Z-component fixed at 0.5 to ensure normals point mostly upward,
        # providing a reasonable default for surface-facing direction
        dz = torch.ones_like(dx) * 0.5

        normals = torch.cat([dx, dy, dz], dim=1)
        normals = F.normalize(normals, dim=1)

        return normals

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        total_loss = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for batch_idx, (low_img, high_img) in enumerate(pbar):
            low_img = low_img.to(device)
            high_img = high_img.to(device)

            # Forward pass through enhancement stages
            enhanced = low_img

            # Estimate depth and compute normals for depth-aware processing
            depth = self._estimate_depth(enhanced)
            normals = self._compute_normals(depth)

            # Stage 1: Caustics (with depth information)
            with torch.set_grad_enabled(True):
                caustics = self.models['caustics'](enhanced, depth)
                enhanced = enhanced + caustics * 0.3

            # Stage 2: Atmosphere
            with torch.set_grad_enabled(True):
                enhanced = self.models['atmosphere'](enhanced)

            # Stage 3: Materials
            with torch.set_grad_enabled(True):
                enhanced = self.models['materials'](enhanced)

            # Stage 4: Spatial Harmonics (illumination from normals)
            with torch.set_grad_enabled(True):
                illumination = self.models['harmonics'](normals)
                enhanced = enhanced * (1 + illumination * 0.3)

            # Compute losses
            mse = self.mse_loss(enhanced, high_img)
            perceptual = self.perceptual_loss(enhanced, high_img)
            style = self.style_loss(enhanced, high_img)

            # Compute LPIPS loss if available
            lpips_loss = torch.tensor(0.0, device=device)
            if self.lpips_fn is not None:
                # LPIPS expects input in [-1, 1] range, scale from [0, 1]
                enhanced_scaled = enhanced * 2 - 1
                high_img_scaled = high_img * 2 - 1
                lpips_loss = self.lpips_fn(enhanced_scaled, high_img_scaled).mean()

            # Combined loss
            loss = (
                self.config.mse_weight * mse +
                self.config.perceptual_weight * perceptual +
                self.config.style_weight * style +
                self.config.lpips_weight * lpips_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for model in self.models.values() for p in model.parameters()],
                self.config.gradient_clip
            )

            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            postfix = {
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse.item():.4f}',
                'percep': f'{perceptual.item():.4f}',
            }
            if self.lpips_fn is not None:
                postfix['lpips'] = f'{lpips_loss.item():.4f}'
            pbar.set_postfix(postfix)

        avg_loss = total_loss / num_batches
        return avg_loss

    def _validate(self, val_loader: DataLoader, epoch: int) -> float:
        """Validate model"""
        total_loss = 0.0
        num_batches = len(val_loader)

        # Set models to eval mode
        for model in self.models.values():
            model.eval()

        with torch.no_grad():
            for low_img, high_img in val_loader:
                low_img = low_img.to(device)
                high_img = high_img.to(device)

                # Forward pass
                enhanced = low_img

                # Estimate depth and compute normals
                depth = self._estimate_depth(enhanced)
                normals = self._compute_normals(depth)

                # Stage 1: Caustics (with depth information)
                caustics = self.models['caustics'](enhanced, depth)
                enhanced = enhanced + caustics * 0.3

                # Stage 2: Atmosphere
                enhanced = self.models['atmosphere'](enhanced)

                # Stage 3: Materials
                enhanced = self.models['materials'](enhanced)

                # Stage 4: Spatial Harmonics (illumination from normals)
                illumination = self.models['harmonics'](normals)
                enhanced = enhanced * (1 + illumination * 0.3)

                # Compute losses (consistent with training)
                mse = self.mse_loss(enhanced, high_img)
                perceptual = self.perceptual_loss(enhanced, high_img)
                style = self.style_loss(enhanced, high_img)

                # Compute LPIPS loss if available
                lpips_loss = torch.tensor(0.0, device=device)
                if self.lpips_fn is not None:
                    enhanced_scaled = enhanced * 2 - 1
                    high_img_scaled = high_img * 2 - 1
                    lpips_loss = self.lpips_fn(enhanced_scaled, high_img_scaled).mean()

                # Combined loss (same formula as training)
                loss = (
                    self.config.mse_weight * mse +
                    self.config.perceptual_weight * perceptual +
                    self.config.style_weight * style +
                    self.config.lpips_weight * lpips_loss
                )
                total_loss += loss.item()

        # Set back to train mode
        for model in self.models.values():
            model.train()

        avg_loss = total_loss / num_batches
        print(f"\n  Validation Loss: {avg_loss:.6f}")
        return avg_loss

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'config': asdict(self.config),
            'models': {name: model.state_dict() for name, model in self.models.items()},
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
        }

        # Save checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved: {best_path}")
        else:
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Hyper-Reality Enhancement Models")
    parser.add_argument("--data-dir", type=str, default="data/training",
                        help="Directory for training data")
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate synthetic training data")
    parser.add_argument("--num-pairs", type=int, default=1000,
                        help="Number of synthetic pairs to generate")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="weights/hyper_reality",
                        help="Directory for checkpoints")

    args = parser.parse_args()

    # Generate synthetic data if requested
    if args.generate_data:
        generator = SyntheticDataGenerator(args.data_dir, args.num_pairs)
        generator.generate_training_data()

    # Check if data exists
    low_quality_dir = Path(args.data_dir) / "low_quality"
    high_quality_dir = Path(args.data_dir) / "high_quality"

    if not low_quality_dir.exists() or not high_quality_dir.exists():
        print(f"\n❌ Training data not found in {args.data_dir}")
        print("   Run with --generate-data to create synthetic training data")
        return 1

    # Create datasets
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    dataset = EnhancementDataset(low_quality_dir, high_quality_dir, transform)

    # Split into train/val
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create training config
    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train
    trainer = HyperRealityTrainer(config)
    trainer.train(train_loader, val_loader)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
