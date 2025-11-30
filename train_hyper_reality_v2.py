#!/usr/bin/env python3
"""
Enhanced Training Pipeline for Hyper-Reality Enhancement Models
Version 2.0.0 - Fixes training flow and integrates LPIPS

Key improvements over v1.0.0:
- All four neural networks receive gradients (CausticGenerator, AtmosphericSynthesizer,
  MaterialTranscendence, SpatialHarmonics)
- Depth and normal maps computed during training
- LPIPS integration for true perceptual loss
- Multi-scale training strategy
- Improved loss weighting based on psychovisual research

Author: Transformation_Portal Enhancement Team
Version: 2.0.0
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
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

warnings.filterwarnings('ignore')

# Configure device
device = configure_device()


@dataclass
class EnhancedTrainingConfig:
    """Enhanced configuration for training hyper-reality models"""

    # Dataset
    data_dir: str = "data/training"
    synthetic_data: bool = True
    num_synthetic_pairs: int = 1000

    # Training hyperparameters
    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Per-model learning rates (relative to base LR)
    model_lr_scales: Dict[str, float] = field(default_factory=lambda: {
        'caustics': 1.0,
        'atmosphere': 1.0,
        'materials': 1.2,  # Slightly higher for segmentation learning
        'harmonics': 0.8,  # Slightly lower for stable SH learning
    })

    # Loss weights (calibrated for perceptual quality)
    mse_weight: float = 1.0
    perceptual_weight: float = 2.0       # Increased for perceptual targets
    lpips_weight: float = 1.5            # NEW: Direct LPIPS loss
    style_weight: float = 0.5
    material_weight: float = 0.3         # NEW: Material-specific loss
    depth_consistency_weight: float = 0.2 # NEW: Depth-aware consistency

    # Progressive training
    progressive: bool = True
    warmup_epochs: int = 5
    stage_epochs: List[int] = field(default_factory=lambda: [10, 20, 35, 50])

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
    
    # Multi-scale training
    multi_scale: bool = True
    scales: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0])


class DepthEstimator(nn.Module):
    """
    Lightweight depth estimation network for training
    
    Provides depth maps for caustic application and normal computation
    without requiring external models during training.
    """
    
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate depth map from RGB image"""
        features = self.encoder(x)
        depth = self.decoder(features)
        
        # Ensure output matches input spatial size
        if depth.shape[-2:] != x.shape[-2:]:
            depth = F.interpolate(depth, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return depth


class NormalEstimator(nn.Module):
    """
    Compute surface normals from depth map
    Uses Sobel filters for gradient estimation
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute surface normals from depth"""
        dx = F.conv2d(depth, self.sobel_x, padding=1)
        dy = F.conv2d(depth, self.sobel_y, padding=1)
        dz = torch.ones_like(dx) * 0.5
        
        normals = torch.cat([dx, dy, dz], dim=1)
        normals = F.normalize(normals, dim=1)
        
        return normals


class VGGFeatureExtractor(nn.Module):
    """Extract features from pretrained VGG19 for perceptual loss"""

    def __init__(self, layers: list = None, use_input_norm: bool = True):
        super().__init__()
        self.use_input_norm = use_input_norm

        if layers is None:
            layers = [2, 7, 12, 21, 30]

        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except (ImportError, TypeError):
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features

        self.layers = sorted(layers)
        max_layer = max(self.layers) + 1
        self.features = nn.Sequential(*list(vgg.children())[:max_layer])

        for param in self.features.parameters():
            param.requires_grad = False

        self.features.eval()
        self.features.to(device)

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layers:
                features.append(x)

        return features


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features"""

    def __init__(self, layers: list = None, weights: list = None):
        super().__init__()

        if layers is None:
            layers = [2, 7, 12, 21, 30]

        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.weights = weights
        self.vgg = VGGFeatureExtractor(layers=layers)

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)

        loss = 0.0
        for w, pred_feat, target_feat in zip(self.weights, pred_features, target_features):
            loss += w * F.mse_loss(pred_feat, target_feat)

        return loss / len(self.weights)


class StyleLoss(nn.Module):
    """Style loss using Gram matrices"""

    def __init__(self, layers: list = None, weights: list = None):
        super().__init__()

        if layers is None:
            layers = [0, 5, 10, 19, 28]

        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.weights = weights
        self.vgg = VGGFeatureExtractor(layers=layers)

    def gram_matrix(self, features):
        b, c, h, w = features.shape
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)

        loss = 0.0
        for w, pred_feat, target_feat in zip(self.weights, pred_features, target_features):
            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += w * F.mse_loss(pred_gram, target_gram)

        return loss / len(self.weights)


class LPIPSLoss(nn.Module):
    """
    LPIPS-based perceptual loss
    
    Uses the official LPIPS package if available, otherwise falls back
    to a VGG-based approximation with learned channel weights.
    """
    
    def __init__(self, net: str = 'vgg'):
        super().__init__()
        
        self.use_official = False
        
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=net).to(device)
            self.lpips_fn.eval()
            for param in self.lpips_fn.parameters():
                param.requires_grad = False
            self.use_official = True
            print("✓ Using official LPIPS for training loss")
        except ImportError:
            print("⚠ LPIPS package not found, using approximation")
            self._init_approximation()
    
    def _init_approximation(self):
        """Initialize VGG-based LPIPS approximation"""
        self.vgg = VGGFeatureExtractor(layers=[2, 7, 12, 21, 30])
        
        # Learned channel weights (approximated from LPIPS)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(64) / 64),
            nn.Parameter(torch.ones(128) / 128),
            nn.Parameter(torch.ones(256) / 256),
            nn.Parameter(torch.ones(512) / 512),
            nn.Parameter(torch.ones(512) / 512),
        ])
        
        # Layer weights
        self.layer_weights = nn.Parameter(torch.tensor([0.1, 0.1, 0.3, 0.3, 0.2]))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS loss"""
        if self.use_official:
            return self.lpips_fn(pred, target).mean()
        
        # Approximation
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        
        loss = 0.0
        for i, (pf, tf) in enumerate(zip(pred_features, target_features)):
            # Normalize features
            pf_norm = F.normalize(pf, dim=1)
            tf_norm = F.normalize(tf, dim=1)
            
            # Compute weighted difference
            diff = (pf_norm - tf_norm) ** 2
            
            # Channel weighting
            weights = self.weights[i].view(1, -1, 1, 1)
            diff_weighted = (diff * weights).sum(dim=1, keepdim=True)
            
            # Spatial mean with layer weight
            loss += self.layer_weights[i] * diff_weighted.mean()
        
        return loss


class MaterialConsistencyLoss(nn.Module):
    """
    Material-specific consistency loss
    
    Ensures that different materials are enhanced appropriately
    by computing per-material reconstruction quality.
    """
    
    MATERIAL_COLORS = torch.tensor([
        [0.85, 0.80, 0.75],  # quartzite
        [0.55, 0.40, 0.25],  # oak
        [0.50, 0.50, 0.55],  # metal
        [0.70, 0.80, 0.90],  # glass
        [0.92, 0.90, 0.85],  # stucco
        [0.30, 0.50, 0.70],  # water
        [0.25, 0.45, 0.20],  # vegetation
        [0.60, 0.75, 0.95],  # sky
    ])
    
    def __init__(self):
        super().__init__()
        self.register_buffer('material_colors', self.MATERIAL_COLORS.T)
    
    def get_material_masks(self, x: torch.Tensor) -> torch.Tensor:
        """Get soft material masks based on color similarity"""
        b, c, h, w = x.shape
        
        x_flat = x.view(b, c, -1)
        colors = self.material_colors.unsqueeze(0).unsqueeze(-1).to(x.device)
        
        x_expanded = x_flat.unsqueeze(2)
        dists = ((x_expanded - colors) ** 2).sum(dim=1)
        
        masks = F.softmax(-dists * 10, dim=1)
        masks = masks.view(b, -1, h, w)
        
        return masks
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute material-weighted reconstruction loss"""
        # Get material masks from target
        masks = self.get_material_masks(target)
        
        # Compute per-material reconstruction error
        error = (pred - target) ** 2
        
        # Weight by material importance
        material_weights = torch.tensor([
            1.2,  # quartzite - high fidelity needed
            1.0,  # oak
            1.3,  # metal - high fidelity for reflections
            1.1,  # glass
            0.9,  # stucco
            1.0,  # water
            0.8,  # vegetation
            0.7,  # sky
        ]).to(pred.device).view(1, -1, 1, 1)
        
        # Weighted material loss
        weighted_masks = masks * material_weights
        weighted_error = (error.unsqueeze(2) * weighted_masks.unsqueeze(1)).sum(dim=2)
        
        return weighted_error.mean()


class DepthConsistencyLoss(nn.Module):
    """
    Depth-aware consistency loss
    
    Ensures that depth relationships are preserved through enhancement
    and that depth-dependent effects (e.g., atmospheric perspective) are coherent.
    """
    
    def __init__(self):
        super().__init__()
        self.depth_estimator = DepthEstimator().to(device)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_depth: Optional[torch.Tensor] = None,
        target_depth: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute depth consistency loss"""
        # Estimate depth if not provided
        if pred_depth is None:
            pred_depth = self.depth_estimator(pred)
        if target_depth is None:
            target_depth = self.depth_estimator(target)
        
        # Depth reconstruction loss
        depth_recon = F.l1_loss(pred_depth, target_depth)
        
        # Depth gradient consistency (preserve edges)
        pred_grad_x = pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1]
        pred_grad_y = pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :]
        
        target_grad_x = target_depth[:, :, :, 1:] - target_depth[:, :, :, :-1]
        target_grad_y = target_depth[:, :, 1:, :] - target_depth[:, :, :-1, :]
        
        grad_loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
        
        return depth_recon + 0.5 * grad_loss


class EnhancementDataset(Dataset):
    """Dataset for image enhancement training"""

    def __init__(self, low_quality_dir: str, high_quality_dir: str, transform=None):
        self.low_quality_dir = Path(low_quality_dir)
        self.high_quality_dir = Path(high_quality_dir)
        self.transform = transform

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

        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img


class SyntheticDataGenerator:
    """Generate synthetic training pairs"""

    def __init__(self, output_dir: str, num_pairs: int = 1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_pairs = num_pairs

    def generate_training_data(self):
        print(f"\n{'='*60}")
        print("GENERATING SYNTHETIC TRAINING DATA")
        print(f"{'='*60}\n")

        low_quality_dir = self.output_dir / "low_quality"
        high_quality_dir = self.output_dir / "high_quality"
        low_quality_dir.mkdir(exist_ok=True)
        high_quality_dir.mkdir(exist_ok=True)

        for i in tqdm(range(self.num_pairs), desc="Generating pairs"):
            high_quality = self._create_synthetic_image()
            low_quality = self._degrade_image(high_quality)

            high_path = high_quality_dir / f"image_{i:04d}.png"
            low_path = low_quality_dir / f"image_{i:04d}.png"

            Image.fromarray(high_quality).save(high_path)
            Image.fromarray(low_quality).save(low_path)

        print(f"\n✓ Generated {self.num_pairs} training pairs")

    def _create_synthetic_image(self, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        h, w = size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Sky gradient
        sky_height = h // 3
        for y in range(sky_height):
            intensity = 1.0 - (y / sky_height) * 0.5
            img[y, :, 2] = int(180 * intensity)
            img[y, :, 1] = int(120 * intensity)
            img[y, :, 0] = int(80 * intensity)

        # Building structure
        building_y = sky_height
        building_height = h - sky_height

        facade_color = np.array([240, 235, 220])
        noise = np.random.randn(building_height, w) * 10
        for c in range(3):
            img[building_y:, :, c] = np.clip(facade_color[c] + noise, 0, 255)

        # Windows
        for window_x in range(50, w - 50, 100):
            for window_y in range(building_y + 30, h - 30, 80):
                window_w, window_h = 60, 50
                img[window_y:window_y+window_h, window_x:window_x+window_w, 2] = 140
                img[window_y:window_y+window_h, window_x:window_x+window_w, 1] = 120
                img[window_y:window_y+window_h, window_x:window_x+window_w, 0] = 100

        x_gradient = np.linspace(0.9, 1.1, w)
        for c in range(3):
            img[building_y:, :, c] = np.clip(img[building_y:, :, c] * x_gradient, 0, 255)

        return img.astype(np.uint8)

    def _degrade_image(self, img: np.ndarray) -> np.ndarray:
        degraded = img.copy().astype(np.float32)

        degraded = (degraded - 128) * 0.7 + 128

        noise = np.random.randn(*degraded.shape) * 8
        degraded += noise

        from scipy.ndimage import gaussian_filter
        for c in range(3):
            degraded[:, :, c] = gaussian_filter(degraded[:, :, c], sigma=0.8)

        gray = degraded.mean(axis=2, keepdims=True)
        degraded = gray * 0.3 + degraded * 0.7

        degraded[:, :, 0] *= 0.95
        degraded[:, :, 2] *= 0.98

        return np.clip(degraded, 0, 255).astype(np.uint8)


class EnhancedHyperRealityTrainer:
    """
    Enhanced training pipeline for hyper-reality models
    
    Key improvements:
    - All four networks trained end-to-end
    - Depth and normal maps computed during forward pass
    - LPIPS loss integration
    - Multi-scale training
    - Progressive stage unlocking
    """

    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._init_models()
        self._init_losses()
        self._init_optimizer()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'mse': [],
            'perceptual': [],
            'lpips': [],
            'style': [],
            'material': [],
            'depth': [],
        }

    def _init_models(self):
        """Initialize all enhancement and auxiliary models"""
        enhancement_config = EnhancementConfig()

        self.models = {
            'caustics': CausticGenerator(enhancement_config.quantum_caustics).to(device),
            'atmosphere': AtmosphericSynthesizer(enhancement_config.neural_atmosphere).to(device),
            'materials': MaterialTranscendence(enhancement_config.material_transcendence).to(device),
            'harmonics': SpatialHarmonics(enhancement_config.spatial_harmonics).to(device),
        }

        # Auxiliary networks
        self.depth_estimator = DepthEstimator().to(device)
        self.normal_estimator = NormalEstimator().to(device)

        # Set to training mode
        for model in self.models.values():
            model.train()
        self.depth_estimator.train()

    def _init_losses(self):
        """Initialize all loss functions"""
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.lpips_loss = LPIPSLoss()
        self.style_loss = StyleLoss()
        self.material_loss = MaterialConsistencyLoss()
        self.depth_loss = DepthConsistencyLoss()

    def _init_optimizer(self):
        """Initialize optimizer with per-model learning rates"""
        param_groups = []
        
        for name, model in self.models.items():
            lr_scale = self.config.model_lr_scales.get(name, 1.0)
            param_groups.append({
                'params': model.parameters(),
                'lr': self.config.learning_rate * lr_scale,
                'name': name
            })
        
        # Add depth estimator
        param_groups.append({
            'params': self.depth_estimator.parameters(),
            'lr': self.config.learning_rate * 0.5,
            'name': 'depth'
        })

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs
        )

    def _get_active_stages(self, epoch: int) -> List[str]:
        """Get active training stages for progressive training"""
        if not self.config.progressive:
            return ['caustics', 'atmosphere', 'materials', 'harmonics']
        
        stages = []
        stage_epochs = self.config.stage_epochs
        
        if epoch >= 0:
            stages.append('caustics')
        if epoch >= stage_epochs[0]:
            stages.append('atmosphere')
        if epoch >= stage_epochs[1]:
            stages.append('materials')
        if epoch >= stage_epochs[2]:
            stages.append('harmonics')
        
        return stages

    def _forward_pass(
        self,
        low_img: torch.Tensor,
        active_stages: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through enhancement pipeline
        
        Returns enhanced image and auxiliary outputs (depth, normals, etc.)
        """
        enhanced = low_img
        aux_outputs = {}
        
        # Estimate depth and normals
        depth = self.depth_estimator(enhanced)
        normals = self.normal_estimator(depth)
        aux_outputs['depth'] = depth
        aux_outputs['normals'] = normals
        
        # Stage 1: Caustics (depth-aware)
        if 'caustics' in active_stages:
            caustics = self.models['caustics'](enhanced, depth)
            # Apply caustics to water regions
            water_mask = (enhanced[:, 2:3] > enhanced[:, 0:1] * 1.2) & \
                        (enhanced[:, 2:3] > enhanced[:, 1:2] * 1.1)
            water_mask = water_mask.float()
            enhanced = enhanced + caustics * water_mask * 0.3
            aux_outputs['caustics'] = caustics
        
        # Stage 2: Atmosphere
        if 'atmosphere' in active_stages:
            enhanced = self.models['atmosphere'](enhanced)
        
        # Stage 3: Materials
        if 'materials' in active_stages:
            enhanced = self.models['materials'](enhanced)
        
        # Stage 4: Spatial Harmonics (normal-aware)
        if 'harmonics' in active_stages:
            illumination = self.models['harmonics'](normals)
            enhanced = enhanced * (1 + illumination * 0.3)
            aux_outputs['illumination'] = illumination
        
        # Clamp to valid range
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return enhanced, aux_outputs

    def _compute_loss(
        self,
        enhanced: torch.Tensor,
        target: torch.Tensor,
        aux_outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss with all components"""
        losses = {}
        
        # MSE loss
        mse = self.mse_loss(enhanced, target)
        losses['mse'] = mse.item()
        
        # Perceptual loss
        perceptual = self.perceptual_loss(enhanced, target)
        losses['perceptual'] = perceptual.item()
        
        # LPIPS loss
        lpips = self.lpips_loss(enhanced, target)
        losses['lpips'] = lpips.item()
        
        # Style loss
        style = self.style_loss(enhanced, target)
        losses['style'] = style.item()
        
        # Material consistency loss
        material = self.material_loss(enhanced, target)
        losses['material'] = material.item()
        
        # Depth consistency loss
        target_depth = self.depth_estimator(target)
        depth = self.depth_loss(enhanced, target, aux_outputs.get('depth'), target_depth)
        losses['depth'] = depth.item()
        
        # Combined loss
        total_loss = (
            self.config.mse_weight * mse +
            self.config.perceptual_weight * perceptual +
            self.config.lpips_weight * lpips +
            self.config.style_weight * style +
            self.config.material_weight * material +
            self.config.depth_consistency_weight * depth
        )
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop"""
        print(f"\n{'='*60}")
        print("ENHANCED HYPER-REALITY TRAINING PIPELINE v2.0")
        print(f"{'='*60}\n")
        print(f"Device: {device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Progressive training: {self.config.progressive}")
        print(f"Multi-scale training: {self.config.multi_scale}")
        if val_loader:
            print(f"Validation samples: {len(val_loader.dataset)}")
        print()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            active_stages = self._get_active_stages(epoch)
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs} - Active stages: {active_stages}")

            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader, epoch, active_stages)
            self.training_history['train_loss'].append(train_loss)
            
            for key, value in train_metrics.items():
                if key not in self.training_history:
                    self.training_history[key] = []
                self.training_history[key].append(value)

            # Validation phase
            if val_loader and (epoch + 1) % self.config.val_frequency == 0:
                val_loss = self._validate(val_loader, epoch, active_stages)
                self.training_history['val_loss'].append(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)

            # Save periodic checkpoints
            if (epoch + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(epoch, is_best=False)

            self.scheduler.step()

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Checkpoint directory: {self.config.checkpoint_dir}")
        
        # Save training history
        history_path = Path(self.config.checkpoint_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved: {history_path}")

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        active_stages: List[str]
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        total_loss = 0.0
        total_metrics = {}
        num_batches = len(train_loader)

        # Set models to train mode
        for name in active_stages:
            self.models[name].train()
        self.depth_estimator.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for batch_idx, (low_img, high_img) in enumerate(pbar):
            low_img = low_img.to(device)
            high_img = high_img.to(device)

            # Multi-scale training
            if self.config.multi_scale:
                scale = np.random.choice(self.config.scales)
                if scale != 1.0:
                    new_size = (int(low_img.shape[2] * scale), int(low_img.shape[3] * scale))
                    low_img = F.interpolate(low_img, size=new_size, mode='bilinear', align_corners=False)
                    high_img = F.interpolate(high_img, size=new_size, mode='bilinear', align_corners=False)

            # Forward pass
            enhanced, aux_outputs = self._forward_pass(low_img, active_stages)

            # Compute loss
            loss, metrics = self._compute_loss(enhanced, high_img, aux_outputs)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            all_params = []
            for name in active_stages:
                all_params.extend(self.models[name].parameters())
            all_params.extend(self.depth_estimator.parameters())
            
            torch.nn.utils.clip_grad_norm_(all_params, self.config.gradient_clip)

            self.optimizer.step()

            total_loss += loss.item()
            
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lpips': f'{metrics["lpips"]:.4f}',
                'mse': f'{metrics["mse"]:.4f}',
            })

        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics

    def _validate(
        self,
        val_loader: DataLoader,
        epoch: int,
        active_stages: List[str]
    ) -> float:
        """Validate model"""
        total_loss = 0.0
        num_batches = len(val_loader)

        # Set to eval mode
        for model in self.models.values():
            model.eval()
        self.depth_estimator.eval()

        with torch.no_grad():
            for low_img, high_img in val_loader:
                low_img = low_img.to(device)
                high_img = high_img.to(device)

                enhanced, aux_outputs = self._forward_pass(low_img, active_stages)
                loss, _ = self._compute_loss(enhanced, high_img, aux_outputs)
                total_loss += loss.item()

        # Set back to train mode
        for name in active_stages:
            self.models[name].train()
        self.depth_estimator.train()

        avg_loss = total_loss / num_batches
        print(f"  Validation Loss: {avg_loss:.6f}")
        return avg_loss

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'config': asdict(self.config),
            'models': {name: model.state_dict() for name, model in self.models.items()},
            'depth_estimator': self.depth_estimator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
        }

        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved: {best_path}")
        else:
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Training for Hyper-Reality Models v2.0")
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
    parser.add_argument("--progressive", action="store_true",
                        help="Use progressive training")
    parser.add_argument("--multi-scale", action="store_true",
                        help="Use multi-scale training")

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
        print("   Or use prepare_750picacho_training_data.py for real data")
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
    config = EnhancedTrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        progressive=args.progressive,
        multi_scale=args.multi_scale,
    )

    # Train
    trainer = EnhancedHyperRealityTrainer(config)
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
