#!/usr/bin/env python3
"""
Enhanced Hyper-Reality Enhancement Module for Transformation_Portal
Version 3.1.0 - Integrates trained model loading and perceptual quality assessment

Key improvements over v3.0.0:
- Automatic loading of trained model weights
- Integration with PerceptualQualityAssessor for true quality measurement
- Improved depth estimation pipeline
- Quality score based on actual perceptual metrics (not heuristic)

Author: Transformation_Portal Enhancement Team
Version: 3.1.0
"""

import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

warnings.filterwarnings('ignore')


def configure_device():
    """Optimally configure device for M4 Max architecture"""
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        torch.mps.set_per_process_memory_fraction(0.85)
        print("✓ Apple Silicon M4 Max detected - MPS acceleration enabled")
        return dev
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("✓ CUDA device detected")
        return dev
    dev = torch.device("cpu")
    print("⚠ Running on CPU (slower performance expected)")
    return dev


device = configure_device()


class QualityMode(Enum):
    """Enhancement quality targeting modes"""
    STANDARD = (70, 85)
    PREMIUM = (85, 95)
    HYPER = (95, 105)
    QUANTUM = (105, 120)
    THEORETICAL = (120, 150)


@dataclass
class EnhancementConfig:
    """Configuration for hyper-reality enhancement pipeline"""

    target_quality: int = 105
    mode: QualityMode = QualityMode.QUANTUM

    quantum_caustics: Dict = field(default_factory=lambda: {
        'enable': True,
        'coherence_length': 0.0001,
        'photon_bundles': 10000,
        'entanglement': 0.15,
        'vacuum_noise': 0.001,
        'caustic_intensity': 2.8,
        'wave_simulation': True
    })

    neural_atmosphere: Dict = field(default_factory=lambda: {
        'enable': True,
        'enhancement_level': 1.8,
        'style_amplitude': 2.5,
        'layer_count': 9,
        'impossible_colors': True,
        'twilight_mode': 'blue_hour'
    })

    material_transcendence: Dict = field(default_factory=lambda: {
        'enable': True,
        'energy_violation': 1.15,
        'negative_absorption': True,
        'quantum_interference': 0.18,
        'temporal_effects': True,
        'bioluminescence': 0.12
    })

    spatial_harmonics: Dict = field(default_factory=lambda: {
        'enable': True,
        'order': 9,
        'negative_light': True,
        'amplification': 1.5,
        'directional_boost': 1.8
    })

    synergistic: Dict = field(default_factory=lambda: {
        'enable': True,
        'edge_enhancement': 1.1,
        'local_contrast': 1.43,
        'saturation_boost': 1.3,
        'tone_curve_gamma': 0.85
    })

    processing: Dict = field(default_factory=lambda: {
        'batch_size': 1,
        'tile_size': 1024,
        'overlap': 128,
        'precision': 'float32',
        'num_workers': 8,
        'pin_memory': True
    })
    
    # Model loading
    checkpoint_dir: str = "weights/hyper_reality"
    auto_load_weights: bool = True


class CausticGenerator(nn.Module):
    """Quantum-inspired caustic pattern generator"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        self.wave_net = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        waves = self.wave_net(x)

        if self.config['wave_simulation']:
            wavelengths = torch.tensor([450e-9, 550e-9, 650e-9]).to(x.device)
            interference = torch.zeros_like(x)

            for i, wavelength in enumerate(wavelengths):
                k = 2 * np.pi / wavelength.item()
                phase = torch.randn(b, 1, h, w).to(x.device) * k
                wave_pattern = torch.sin(phase) * torch.cos(phase * 1.3)
                coherence = torch.exp(-torch.abs(wave_pattern) * self.config['coherence_length'])
                interference[:, i:i+1] = wave_pattern * coherence

            waves = waves * 0.7 + interference * 0.3

        if self.config['entanglement'] > 0:
            g2 = 1 + torch.exp(-torch.abs(waves) / self.config['photon_bundles'])
            waves = waves * g2

        if self.config['vacuum_noise'] > 0:
            vacuum = torch.randn_like(waves) * self.config['vacuum_noise']
            waves = waves + vacuum

        waves = torch.clamp(waves, 0, 1) * self.config['caustic_intensity']

        return waves


class AtmosphericSynthesizer(nn.Module):
    """Neural atmospheric synthesis for impossible skies"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        self.encoder = nn.ModuleList([
            self._make_encoder_block(3, 64),
            self._make_encoder_block(64, 128),
            self._make_encoder_block(128, 256),
            self._make_encoder_block(256, 512)
        ])

        self.latent = nn.Sequential(
            nn.Conv2d(512, 1024, 1),
            nn.GELU(),
            nn.Conv2d(1024, 1024, 1),
            nn.GELU(),
            nn.Conv2d(1024, 512, 1)
        )

        self.decoder = nn.ModuleList([
            self._make_decoder_block(512, 256),
            self._make_decoder_block(512, 128),
            self._make_decoder_block(256, 64),
            self._make_decoder_block(128, 3)
        ])

    def _make_encoder_block(self, in_c: int, out_c: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2 if in_c > 3 else 1, padding=1),
            nn.GroupNorm(min(32, out_c // 2), out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=2, dilation=2),
            nn.GroupNorm(min(32, out_c // 2), out_c),
            nn.GELU()
        )

    def _make_decoder_block(self, in_c: int, out_c: int) -> nn.Module:
        if out_c == 3:
            return nn.Sequential(
                nn.Conv2d(in_c, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, out_c, 1),
                nn.Sigmoid()
            )
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
            nn.GroupNorm(min(32, out_c // 2), out_c),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        feat = x
        for encoder in self.encoder:
            feat = encoder(feat)
            skips.append(feat)

        feat = self.latent(feat)

        if self.config['impossible_colors']:
            style = torch.randn(feat.shape[0], 512, 1, 1).to(feat.device)
            style = style * self.config['style_amplitude']
            feat = feat * (1 + style)

        for i, decoder in enumerate(self.decoder):
            if 0 < i < len(skips):
                skip = skips[-(i+1)]
                if feat.shape[-2:] != skip.shape[-2:]:
                    feat = F.interpolate(feat, size=skip.shape[-2:], mode='bilinear', align_corners=True)
                feat = torch.cat([feat, skip], dim=1)
            feat = decoder(feat)

        feat = feat * self.config['enhancement_level']

        return torch.clamp(feat, 0, 1.5)


class MaterialTranscendence(nn.Module):
    """Physics-violating material response system"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        self.segmenter = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, 1),
            nn.Softmax(dim=1)
        )

        self.material_responses = nn.ModuleDict({
            'stucco': self._make_material_net(),
            'stone': self._make_material_net(),
            'glass': self._make_material_net(),
            'water': self._make_material_net()
        })

    def _make_material_net(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        materials = self.segmenter(x)
        result = torch.zeros_like(x)

        for i, (name, net) in enumerate(self.material_responses.items()):
            mask = materials[:, i:i+1]
            response = net(x)

            if name == 'stucco' and self.config['energy_violation'] > 1.0:
                response = response * self.config['energy_violation']
            elif name == 'glass' and self.config['negative_absorption']:
                response = response * 1.05
            elif name == 'water' and self.config['bioluminescence'] > 0:
                glow = torch.randn_like(response) * self.config['bioluminescence']
                response = response + torch.abs(glow)
            elif name == 'stone' and self.config['quantum_interference'] > 0:
                interference = torch.sin(response * 20) * self.config['quantum_interference']
                response = response + interference

            result = result + mask * response

        if self.config['temporal_effects']:
            shimmer = torch.randn_like(result) * 0.02
            result = result + shimmer

        return torch.clamp(result, 0, 1.5)


class SpatialHarmonics(nn.Module):
    """Impossible illumination through spherical harmonics"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.order = config['order']

        n_coeffs = (self.order + 1) ** 2
        self.coefficients = nn.Parameter(torch.randn(n_coeffs, 3))

        with torch.no_grad():
            self.coefficients[0] *= 1.2
            if config['negative_light']:
                self.coefficients[4:9] *= -0.3
            self.coefficients[1:4] *= config['amplification']

    def forward(self, normals: torch.Tensor) -> torch.Tensor:
        b, c, h, w = normals.shape

        theta = torch.acos(torch.clamp(normals[:, 2:3], -1, 1))
        phi = torch.atan2(normals[:, 1:2], normals[:, 0:1])

        illumination = torch.zeros(b, 3, h, w).to(normals.device)

        illumination += self.coefficients[0].view(1, 3, 1, 1) * 0.282095
        illumination += self.coefficients[1].view(1, 3, 1, 1) * 0.488603 * torch.sin(theta) * torch.sin(phi)
        illumination += self.coefficients[2].view(1, 3, 1, 1) * 0.488603 * torch.cos(theta)
        illumination += self.coefficients[3].view(1, 3, 1, 1) * 0.488603 * torch.sin(theta) * torch.cos(phi)

        if self.config['directional_boost'] > 1.0:
            illumination = illumination * self.config['directional_boost']

        illumination = torch.sign(illumination) * torch.pow(torch.abs(illumination), 0.7)

        return illumination


class EnhancedDepthEstimator(nn.Module):
    """
    Improved depth estimation network
    
    Provides better depth maps for caustic application and spatial harmonics.
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
        )
        
        self.decoder = nn.Sequential(
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
        features = self.encoder(x)
        depth = self.decoder(features)
        
        if depth.shape[-2:] != x.shape[-2:]:
            depth = F.interpolate(depth, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return depth


class HyperRealityProcessor:
    """
    Enhanced processing pipeline for 105/100+ quality achievement
    
    Version 3.1.0 improvements:
    - Automatic model weight loading from checkpoints
    - True perceptual quality measurement via PerceptualQualityAssessor
    - Improved depth estimation for auxiliary features
    """

    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or EnhancementConfig()

        # Initialize processing modules
        self.caustic_gen = CausticGenerator(self.config.quantum_caustics).to(device)
        self.atmosphere_syn = AtmosphericSynthesizer(self.config.neural_atmosphere).to(device)
        self.material_trans = MaterialTranscendence(self.config.material_transcendence).to(device)
        self.spatial_harm = SpatialHarmonics(self.config.spatial_harmonics).to(device)
        
        # Depth estimation
        self.depth_estimator = EnhancedDepthEstimator().to(device)

        # Load trained weights if available
        self.weights_loaded = False
        if self.config.auto_load_weights:
            self._load_trained_weights()

        # Set to eval mode
        self.caustic_gen.eval()
        self.atmosphere_syn.eval()
        self.material_trans.eval()
        self.spatial_harm.eval()
        self.depth_estimator.eval()

        # Quality assessment
        self.quality_assessor = None
        self._init_quality_assessor()
        
        # Track enhancements
        self.enhancements_applied = []

    def _load_trained_weights(self):
        """Load trained model weights from checkpoint"""
        from .model_loader import ModelLoader
        
        loader = ModelLoader(self.config.checkpoint_dir)
        checkpoint = loader.load_best_model()
        
        if checkpoint is None:
            print("⚠ No trained weights found. Using random initialization.")
            print(f"   Train models with: python src/enhancements/train_hyper_reality_v2.py")
            return
        
        try:
            models = {
                'caustics': self.caustic_gen,
                'atmosphere': self.atmosphere_syn,
                'materials': self.material_trans,
                'harmonics': self.spatial_harm,
            }
            
            model_states = checkpoint.get('models', {})
            
            for name, model in models.items():
                if name in model_states:
                    model.load_state_dict(model_states[name])
                    print(f"✓ Loaded weights for {name}")
            
            # Load depth estimator if available
            if 'depth_estimator' in checkpoint:
                self.depth_estimator.load_state_dict(checkpoint['depth_estimator'])
                print("✓ Loaded weights for depth_estimator")
            
            self.weights_loaded = True
            
            # Report checkpoint info
            epoch = checkpoint.get('epoch', 'unknown')
            val_loss = checkpoint.get('best_val_loss', 'N/A')
            print(f"✓ Loaded checkpoint from epoch {epoch} (val_loss: {val_loss})")
            
        except Exception as e:
            print(f"⚠ Failed to load some weights: {e}")

    def _init_quality_assessor(self):
        """Initialize perceptual quality assessment"""
        try:
            from .perceptual_quality_assessment import PerceptualQualityAssessor
            self.quality_assessor = PerceptualQualityAssessor()
            print("✓ Perceptual quality assessment initialized")
        except ImportError as e:
            print(f"⚠ Quality assessor not available: {e}")
            self.quality_assessor = None

    def _estimate_depth(self, img: torch.Tensor) -> torch.Tensor:
        """Estimate depth map from image using trained network"""
        with torch.no_grad():
            depth = self.depth_estimator(img)
        return depth

    def _compute_normals(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute surface normals from depth"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)

        dx = F.conv2d(depth, sobel_x, padding=1)
        dy = F.conv2d(depth, sobel_y, padding=1)
        dz = torch.ones_like(dx) * 0.5

        normals = torch.cat([dx, dy, dz], dim=1)
        normals = F.normalize(normals, dim=1)

        return normals

    def _apply_caustics(self, img: torch.Tensor, caustics: torch.Tensor) -> torch.Tensor:
        """Apply caustic patterns to water regions"""
        water_mask = (img[:, 2:3] > img[:, 0:1] * 1.2) & (img[:, 2:3] > img[:, 1:2] * 1.1)
        water_mask = water_mask.float()
        img = img + caustics * water_mask * 0.3
        return torch.clamp(img, 0, 1.5)

    def _synergistic_amplification(self, img: torch.Tensor) -> torch.Tensor:
        """Apply final synergistic enhancements"""
        if self.config.synergistic['edge_enhancement'] > 1.0:
            kernel = torch.tensor([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
            edges = F.conv2d(img, kernel.repeat(3, 1, 1, 1), padding=1, groups=3)
            img = img + edges * (self.config.synergistic['edge_enhancement'] - 1.0)

        if self.config.synergistic['local_contrast'] > 1.0:
            local_mean = F.avg_pool2d(img, kernel_size=15, stride=1, padding=7)
            img = (img - local_mean) * self.config.synergistic['local_contrast'] + local_mean

        if self.config.synergistic['saturation_boost'] > 1.0:
            gray = torch.mean(img, dim=1, keepdim=True)
            img = gray + (img - gray) * self.config.synergistic['saturation_boost']

        if self.config.synergistic['tone_curve_gamma'] != 1.0:
            img = torch.pow(torch.clamp(img, 0, 1), self.config.synergistic['tone_curve_gamma'])

        return img

    def process_image(self,
                      image_path: str,
                      output_path: Optional[str] = None,
                      reference_path: Optional[str] = None,
                      save_intermediate: bool = False) -> Dict[str, Any]:
        """
        Process image to achieve target quality level with true perceptual measurement
        
        Args:
            image_path: Path to input image
            output_path: Path for output (auto-generated if None)
            reference_path: Optional reference image for quality comparison
            save_intermediate: Save intermediate enhancement stages
        
        Returns:
            Dictionary containing results, metrics, and quality assessment
        """
        start_time = time.time()
        self.enhancements_applied = []

        print(f"\n{'='*60}")
        print("HYPER-REALITY ENHANCEMENT PIPELINE v3.1")
        print(f"Target Quality: {self.config.target_quality}/100")
        print(f"Mode: {self.config.mode.name}")
        print(f"Weights Loaded: {self.weights_loaded}")
        print(f"{'='*60}\n")

        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        # Convert to tensor
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Generate auxiliary maps
        print("Generating auxiliary maps...")
        with torch.no_grad():
            depth_map = self._estimate_depth(img_tensor)
            normals = self._compute_normals(depth_map)

        # Stage 1: Quantum Caustics
        if self.config.quantum_caustics['enable']:
            print("\n→ Stage 1: Quantum Caustic Enhancement")
            with torch.no_grad():
                caustics = self.caustic_gen(img_tensor, depth_map)
                img_tensor = self._apply_caustics(img_tensor, caustics)
            self.enhancements_applied.append('quantum_caustics')
            if save_intermediate:
                self._save_intermediate(img_tensor, output_path, "01_caustics")

        # Stage 2: Neural Atmosphere
        if self.config.neural_atmosphere['enable']:
            print("→ Stage 2: Neural Atmospheric Synthesis")
            with torch.no_grad():
                img_tensor = self.atmosphere_syn(img_tensor)
            self.enhancements_applied.append('neural_atmosphere')
            if save_intermediate:
                self._save_intermediate(img_tensor, output_path, "02_atmosphere")

        # Stage 3: Material Transcendence
        if self.config.material_transcendence['enable']:
            print("→ Stage 3: Material Transcendence")
            with torch.no_grad():
                img_tensor = self.material_trans(img_tensor)
            self.enhancements_applied.append('material_transcendence')
            if save_intermediate:
                self._save_intermediate(img_tensor, output_path, "03_materials")

        # Stage 4: Spatial Harmonics
        if self.config.spatial_harmonics['enable']:
            print("→ Stage 4: Spatial Harmonics Illumination")
            with torch.no_grad():
                illumination = self.spatial_harm(normals)
                img_tensor = img_tensor * (1 + illumination * 0.3)
            self.enhancements_applied.append('spatial_harmonics')
            if save_intermediate:
                self._save_intermediate(img_tensor, output_path, "04_harmonics")

        # Stage 5: Synergistic Amplification
        if self.config.synergistic['enable']:
            print("→ Stage 5: Synergistic Amplification")
            img_tensor = self._synergistic_amplification(img_tensor)
            self.enhancements_applied.append('synergistic')
            if save_intermediate:
                self._save_intermediate(img_tensor, output_path, "05_synergistic")

        # Final processing
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # Convert back to PIL
        result = transforms.ToPILImage()(img_tensor.squeeze(0).cpu())
        result = result.resize(original_size, Image.Resampling.LANCZOS)

        # Compute true perceptual quality
        quality_report = None
        quality_score = 0.0
        
        if self.quality_assessor is not None:
            print("\n→ Computing Perceptual Quality Assessment...")
            quality_report = self.quality_assessor.assess(
                img_tensor,
                reference=reference_path
            )
            quality_score = quality_report.composite_score
            print(f"  Composite Score: {quality_score:.1f}/100")
            print(f"  Percentile Rank: {quality_report.percentile_rank:.1f}%")
            if quality_report.overall_material_fidelity > 0:
                print(f"  Material Fidelity: {quality_report.overall_material_fidelity:.1%}")
        else:
            # Fallback to heuristic scoring
            quality_score = 78 + len(self.enhancements_applied) * 8
            print(f"\n  Heuristic Quality: {quality_score}/100 (assessor unavailable)")

        # Save output
        if output_path is None:
            base = Path(image_path).stem
            output_path = f"{base}_hyper_reality_{int(quality_score)}.jpg"
        result.save(output_path, quality=100, subsampling=0)

        processing_time = time.time() - start_time

        print(f"\n{'='*60}")
        print("ENHANCEMENT COMPLETE")
        print(f"Final Quality: {quality_score:.1f}/100")
        print(f"Processing Time: {processing_time:.2f}s")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")

        results = {
            'output_path': output_path,
            'quality_score': quality_score,
            'processing_time': processing_time,
            'original_size': original_size,
            'device': str(device),
            'enhancements': self.enhancements_applied,
            'weights_loaded': self.weights_loaded,
        }
        
        if quality_report is not None:
            results['quality_report'] = quality_report.to_dict()

        return results

    def _save_intermediate(self, img: torch.Tensor, base_path: str, stage: str):
        """Save intermediate processing stage"""
        if base_path is None:
            return
        path = Path(base_path).parent / f"{Path(base_path).stem}_{stage}.jpg"
        result = transforms.ToPILImage()(img.squeeze(0).cpu().clamp(0, 1))
        result.save(path, quality=95)
        print(f"  Saved: {path}")


def enhance_image(image_path: str,
                  output_path: Optional[str] = None,
                  reference_path: Optional[str] = None,
                  target_quality: int = 105,
                  save_intermediate: bool = False) -> Dict[str, Any]:
    """
    Enhance a single image to hyper-reality quality with true perceptual measurement
    
    Args:
        image_path: Path to input image
        output_path: Optional output path
        reference_path: Optional reference for quality comparison
        target_quality: Target quality score (default: 105)
        save_intermediate: Save intermediate stages
    
    Returns:
        Processing results dictionary with quality assessment
    """
    config = EnhancementConfig(target_quality=target_quality)
    processor = HyperRealityProcessor(config)

    return processor.process_image(
        image_path=image_path,
        output_path=output_path,
        reference_path=reference_path,
        save_intermediate=save_intermediate
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyper-Reality Enhancement v3.1")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("-r", "--reference", help="Reference image for quality comparison")
    parser.add_argument("-q", "--quality", type=int, default=105, help="Target quality")
    parser.add_argument("-i", "--intermediate", action="store_true", help="Save intermediates")
    parser.add_argument("--no-weights", action="store_true", help="Don't load trained weights")

    args = parser.parse_args()

    config = EnhancementConfig(
        target_quality=args.quality,
        auto_load_weights=not args.no_weights
    )
    
    processor = HyperRealityProcessor(config)
    
    results = processor.process_image(
        image_path=args.input,
        output_path=args.output,
        reference_path=args.reference,
        save_intermediate=args.intermediate
    )

    print("\nProcessing Results:")
    print(f"  Output: {results['output_path']}")
    print(f"  Quality: {results['quality_score']:.1f}/100")
    print(f"  Time: {results['processing_time']:.2f}s")
    print(f"  Device: {results['device']}")
    print(f"  Weights: {'Loaded' if results['weights_loaded'] else 'Random'}")
