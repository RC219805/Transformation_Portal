"""Production-grade quality validation framework for Lux Depth V2.

Provides:
- Synthetic reference mode (degrade high-res originals, compare output vs reference)
- Real-world mode (NIMA + no-reference IQA)
- Baseline comparison (Topaz/Adobe/etc.)
- Multiple metric categories (fidelity, perceptual, aesthetic)
- Batch validation and regression testing support
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from . import metrics
from . import degradation


@dataclass
class ValidationReport:
    """Validation results for a single image or batch."""
    
    mode: str  # "synthetic" or "real"
    test_images: List[str]
    metrics_scores: Dict[str, float] = field(default_factory=dict)
    per_image_scores: List[Dict[str, object]] = field(default_factory=list)
    composite_score: Optional[float] = None
    baseline_comparison: Optional[Dict[str, object]] = None
    timestamp: Optional[str] = None
    config: Optional[Dict[str, object]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mode": self.mode,
            "test_images": self.test_images,
            "metrics_scores": self.metrics_scores,
            "per_image_scores": self.per_image_scores,
            "composite_score": self.composite_score,
            "baseline_comparison": self.baseline_comparison,
            "timestamp": self.timestamp,
            "config": self.config,
        }
    
    def save(self, output_path: Path) -> None:
        """Save report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ComparisonReport:
    """Comparison between two methods (ours vs baseline)."""
    
    our_method: str
    baseline_method: str
    test_images: List[str]
    reference_images: Optional[List[str]] = None
    
    # Aggregate scores
    our_scores: Dict[str, float] = field(default_factory=dict)
    baseline_scores: Dict[str, float] = field(default_factory=dict)
    
    # Per-image comparisons
    per_image_comparisons: List[Dict[str, object]] = field(default_factory=list)
    
    # Summary statistics
    our_wins: int = 0
    baseline_wins: int = 0
    ties: int = 0
    
    # Statistical significance (if available)
    p_values: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "our_method": self.our_method,
            "baseline_method": self.baseline_method,
            "test_images": self.test_images,
            "reference_images": self.reference_images,
            "our_scores": self.our_scores,
            "baseline_scores": self.baseline_scores,
            "per_image_comparisons": self.per_image_comparisons,
            "our_wins": self.our_wins,
            "baseline_wins": self.baseline_wins,
            "ties": self.ties,
            "p_values": self.p_values,
        }
    
    def save(self, output_path: Path) -> None:
        """Save comparison report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class QualityValidator:
    """Production-grade quality validation framework."""
    
    def __init__(
        self,
        device: str = "cpu",
        default_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize validator.
        
        Args:
            device: Device for metric computation ('cpu', 'cuda', 'mps')
            default_weights: Default metric weights for composite score
                            e.g., {"ssim": 0.3, "psnr": 0.2, "lpips": 0.3, "nima": 0.2}
        """
        self.device = device
        self.default_weights = default_weights or {
            "ssim": 0.25,
            "psnr": 0.15,
            "lpips": 0.35,
            "nima": 0.25,
        }
    
    def validate_batch(
        self,
        test_images: List[Path],
        output_dir: Path,
        baseline_dir: Optional[Path] = None,
        mode: Literal["synthetic", "real"] = "real",
        metrics_list: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> ValidationReport:
        """Validate a batch of processed images.
        
        Args:
            test_images: List of test image paths (processed outputs)
            output_dir: Directory to save validation results
            baseline_dir: Optional directory with baseline outputs for comparison
            mode: Validation mode ('synthetic' for reference-based, 'real' for no-reference)
            metrics_list: List of metrics to compute (default: all available)
            weights: Metric weights for composite score
        
        Returns:
            ValidationReport with results
        """
        if metrics_list is None:
            metrics_list = ["ssim", "psnr", "lpips", "nima"]
        
        if weights is None:
            weights = self.default_weights
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = ValidationReport(
            mode=mode,
            test_images=[str(p) for p in test_images],
            timestamp=self._get_timestamp(),
        )
        
        # Process each image
        for test_path in test_images:
            test_path = Path(test_path)
            
            # Load test image
            test_img = self._load_image(test_path)
            
            # Find reference image if in synthetic mode
            reference_img = None
            if mode == "synthetic":
                ref_path = self._find_reference(test_path)
                if ref_path and ref_path.exists():
                    reference_img = self._load_image(ref_path)
            
            # Compute metrics
            img_metrics = metrics.compute_all_metrics(
                test_img,
                reference=reference_img,
                device=self.device
            )
            
            # Filter to requested metrics
            img_metrics = {k: v for k, v in img_metrics.items() if k in metrics_list}
            
            report.per_image_scores.append({
                "image": str(test_path),
                "metrics": img_metrics,
            })
        
        # Compute aggregate scores
        report.metrics_scores = self._aggregate_scores(report.per_image_scores)
        
        # Compute composite score
        report.composite_score = self._compute_composite_score(
            report.metrics_scores,
            weights
        )
        
        # Baseline comparison if provided
        if baseline_dir:
            report.baseline_comparison = self._compare_with_baseline(
                test_images,
                baseline_dir,
                mode=mode,
                metrics_list=metrics_list
            )
        
        return report
    
    def create_synthetic_reference(
        self,
        original: Path,
        output_dir: Path,
        degradations: Optional[List[str]] = None
    ) -> Tuple[Path, Path]:
        """Create synthetic degraded/reference pair from high-quality original.
        
        Args:
            original: Path to high-quality original image
            output_dir: Directory to save degraded and reference images
            degradations: List of degradations to apply
        
        Returns:
            Tuple of (degraded_path, reference_path)
        """
        if degradations is None:
            degradations = ["downsample", "blur", "noise", "compress"]
        
        # Load original
        original_img = self._load_image(Path(original))
        
        # Create degraded pair
        degraded, reference = degradation.create_synthetic_pair(
            original_img,
            degradations=degradations
        )
        
        # Save pair
        basename = Path(original).stem
        return degradation.save_synthetic_pair(
            degraded,
            reference,
            output_dir,
            basename
        )
    
    def compare_baselines(
        self,
        ours: Path,
        baseline: Path,
        reference: Optional[Path] = None,
        metrics_list: Optional[List[str]] = None
    ) -> ComparisonReport:
        """Compare our output against baseline (Topaz/Adobe/etc.).
        
        Args:
            ours: Path to our processed image
            baseline: Path to baseline processed image
            reference: Optional ground truth reference
            metrics_list: List of metrics to compute
        
        Returns:
            ComparisonReport with comparison results
        """
        if metrics_list is None:
            metrics_list = ["ssim", "psnr", "lpips", "nima"]
        
        # Load images
        our_img = self._load_image(Path(ours))
        baseline_img = self._load_image(Path(baseline))
        reference_img = self._load_image(Path(reference)) if reference else None
        
        # Compute metrics for both
        our_metrics = metrics.compute_all_metrics(
            our_img,
            reference=reference_img,
            device=self.device
        )
        baseline_metrics = metrics.compute_all_metrics(
            baseline_img,
            reference=reference_img,
            device=self.device
        )
        
        # Filter to requested metrics
        our_metrics = {k: v for k, v in our_metrics.items() if k in metrics_list}
        baseline_metrics = {k: v for k, v in baseline_metrics.items() if k in metrics_list}
        
        # Create comparison report
        report = ComparisonReport(
            our_method="LuxDepthV2",
            baseline_method=Path(baseline).parent.name,
            test_images=[str(ours)],
            reference_images=[str(reference)] if reference else None,
            our_scores=our_metrics,
            baseline_scores=baseline_metrics,
        )
        
        # Determine winner for this comparison
        our_better = self._compare_metrics(our_metrics, baseline_metrics)
        if our_better > 0:
            report.our_wins = 1
        elif our_better < 0:
            report.baseline_wins = 1
        else:
            report.ties = 1
        
        report.per_image_comparisons.append({
            "image": str(ours),
            "our_metrics": our_metrics,
            "baseline_metrics": baseline_metrics,
            "winner": "ours" if our_better > 0 else ("baseline" if our_better < 0 else "tie"),
        })
        
        return report
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image as float [0, 1] array."""
        try:
            import cv2
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            
            # Convert to RGB if needed
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] == 4:
                img = img[..., :3]  # Drop alpha
            elif img.ndim == 3 and img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to float [0, 1]
            if img.dtype == np.uint16:
                return img.astype(np.float32) / 65535.0
            elif img.dtype == np.uint8:
                return img.astype(np.float32) / 255.0
            else:
                return img.astype(np.float32)
        
        except ImportError:
            # Fallback to PIL
            from PIL import Image
            img = np.array(Image.open(path))
            
            if img.dtype == np.uint16:
                return img.astype(np.float32) / 65535.0
            elif img.dtype == np.uint8:
                return img.astype(np.float32) / 255.0
            else:
                return img.astype(np.float32)
    
    def _find_reference(self, test_path: Path) -> Optional[Path]:
        """Find corresponding reference image for a test image."""
        # Convention: test_degraded.png -> test_reference.tif
        stem = test_path.stem
        if stem.endswith("_degraded"):
            ref_stem = stem.replace("_degraded", "_reference")
        elif stem.endswith("_upscaled16"):
            # Our output naming
            ref_stem = stem.replace("_upscaled16", "_reference")
        else:
            ref_stem = f"{stem}_reference"
        
        # Try common extensions
        parent = test_path.parent
        for ext in [".tif", ".tiff", ".png", ".jpg"]:
            ref_path = parent / f"{ref_stem}{ext}"
            if ref_path.exists():
                return ref_path
        
        return None
    
    def _aggregate_scores(self, per_image_scores: List[Dict[str, object]]) -> Dict[str, float]:
        """Aggregate per-image scores into overall scores."""
        if not per_image_scores:
            return {}
        
        all_metrics = {}
        for img_score in per_image_scores:
            for metric_name, value in img_score.get("metrics", {}).items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(float(value))
        
        # Compute mean for each metric
        return {k: float(np.mean(v)) for k, v in all_metrics.items()}
    
    def _compute_composite_score(
        self,
        metric_scores: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Compute weighted composite score from individual metrics.
        
        Higher-is-better normalization:
        - SSIM: already in [0, 1], higher is better
        - PSNR: normalize to [0, 1] with typical range [20, 50]
        - LPIPS: invert (1 - lpips) since lower is better, typical range [0, 1]
        - NIMA: normalize to [0, 1] from typical range [1, 10]
        """
        normalized_scores = {}
        
        for metric_name, score in metric_scores.items():
            if metric_name == "ssim":
                normalized_scores[metric_name] = score  # Already [0, 1]
            elif metric_name == "psnr":
                # Normalize from typical [20, 50] to [0, 1]
                normalized_scores[metric_name] = np.clip((score - 20.0) / 30.0, 0.0, 1.0)
            elif metric_name == "lpips":
                # Invert since lower is better
                normalized_scores[metric_name] = 1.0 - np.clip(score, 0.0, 1.0)
            elif metric_name == "nima":
                # Normalize from [1, 10] to [0, 1]
                normalized_scores[metric_name] = (score - 1.0) / 9.0
            else:
                # Unknown metric, assume [0, 1] higher-is-better
                normalized_scores[metric_name] = np.clip(score, 0.0, 1.0)
        
        # Compute weighted average
        composite = 0.0
        total_weight = 0.0
        for metric_name, norm_score in normalized_scores.items():
            weight = weights.get(metric_name, 0.0)
            composite += weight * norm_score
            total_weight += weight
        
        if total_weight > 0:
            composite /= total_weight
        
        return float(composite)
    
    def _compare_metrics(
        self,
        ours: Dict[str, float],
        baseline: Dict[str, float]
    ) -> int:
        """Compare two metric sets, return 1 if ours better, -1 if baseline better, 0 if tie."""
        score_diff = 0
        
        for metric_name in ours.keys():
            if metric_name not in baseline:
                continue
            
            our_val = ours[metric_name]
            base_val = baseline[metric_name]
            
            # Higher-is-better for all except LPIPS
            if metric_name == "lpips":
                # Lower is better for LPIPS
                if our_val < base_val - 0.01:
                    score_diff += 1
                elif our_val > base_val + 0.01:
                    score_diff -= 1
            else:
                # Higher is better for SSIM, PSNR, NIMA
                if our_val > base_val + 0.01:
                    score_diff += 1
                elif our_val < base_val - 0.01:
                    score_diff -= 1
        
        return np.sign(score_diff)
    
    def _compare_with_baseline(
        self,
        test_images: List[Path],
        baseline_dir: Path,
        mode: str,
        metrics_list: List[str]
    ) -> Dict[str, object]:
        """Compare test images with baseline outputs."""
        baseline_dir = Path(baseline_dir)
        
        comparisons = []
        for test_path in test_images:
            # Find corresponding baseline image
            baseline_path = baseline_dir / test_path.name
            if not baseline_path.exists():
                # Try alternate naming
                baseline_path = baseline_dir / test_path.stem.replace("_upscaled16", "") + test_path.suffix
            
            if baseline_path.exists():
                # Find reference if in synthetic mode
                reference_path = self._find_reference(test_path) if mode == "synthetic" else None
                
                # Compare
                comp = self.compare_baselines(
                    test_path,
                    baseline_path,
                    reference=reference_path,
                    metrics_list=metrics_list
                )
                comparisons.append(comp.per_image_comparisons[0])
        
        # Aggregate comparison results
        our_wins = sum(1 for c in comparisons if c.get("winner") == "ours")
        baseline_wins = sum(1 for c in comparisons if c.get("winner") == "baseline")
        ties = sum(1 for c in comparisons if c.get("winner") == "tie")
        
        return {
            "comparisons": comparisons,
            "our_wins": our_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
            "win_rate": our_wins / len(comparisons) if comparisons else 0.0,
        }
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp as ISO 8601 string."""
        import time
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
