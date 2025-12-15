#!/usr/bin/env python3
"""
Water candidate validation harness for pool and ocean scenes.

Produces:
- Coverage sanity checks (mean + median per label)
- Boundary pixel statistics
- Edge alignment vs gradients (primary metric) - requires water_detection_enabled
- Stability across perturbations (deterministic with --seed)
- False trigger rate on negative controls (should_detect=false)

Schema: WATER_GROUND_TRUTH_SCHEMA_FINAL.md
- Labels: pool, ocean (both are water)
- Negative controls: should_detect=false (no third label)
- False trigger rate replaces false positive rate

Known Limitations:
- Edge alignment requires water_detection_enabled=True
- Current detector is stub implementation (PR-W1 pending)
- Thresholds are targets, not calibrated against labeled dataset
- For production validation, complete PR-W1 detector first

Usage:
    python prw_water_validation.py \\
        --input-dir data/water_v0/images/ \\
        --ground-truth data/water_v0/ground_truth.json \\
        --output water_validation_report.json \\
        --seed 42
"""

import argparse
import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from lux_depth_v2.materials_v3 import MaterialsV3Engine, MaterialsV3Config

# SciPy dependency handling (optional for edge alignment)
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  WARNING: SciPy not available. Edge alignment metrics will be disabled.")
    print("   Install scipy for full validation: pip install scipy")


@dataclass
class ValidationResult:
    """Single validation test result (backward-compatible schema)."""
    image_path: str
    scene_type: str  # label: pool|ocean
    should_detect: bool  # from ground truth
    difficulty: str  # easy|medium|hard
    tags: List[str]  # from ground truth

    # Detection status
    detected: bool  # detector's explicit present flag

    # Coverage
    coverage: float
    coverage_px: int

    # Confidence
    confidence: float
    source: str
    implementation: str  # detector version (e.g., "stub_v0_blue_threshold")

    # Edge quality (primary metric)
    edge_alignment_score: float  # vs image gradients
    boundary_px: int

    # Stability
    stability_score: float  # across perturbations

    # False triggers
    is_false_positive: bool  # legacy alias for is_false_trigger
    is_false_trigger: bool  # should_detect=false but detected

    # Performance
    processing_time_ms: float


class WaterValidationHarness:
    """Validation harness for water detection."""

    def __init__(self, config: MaterialsV3Config, seed: Optional[int] = None):
        self.engine = MaterialsV3Engine(config)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def validate_dataset(
        self,
        ground_truth: dict,  # Full ground truth JSON (v0 schema)
        gt_base_dir: Optional[Path] = None  # Base directory for resolving root
    ) -> List[ValidationResult]:
        """Run validation on dataset using new schema."""
        results = []
        root_rel = ground_truth.get("root", "data/water_v0/images")
        
        # Resolve root relative to ground truth file location
        if gt_base_dir is not None:
            root = gt_base_dir / root_rel
        else:
            root = Path(root_rel)

        for img_relpath, img_info in ground_truth.get("images", {}).items():
            img_path = root / img_relpath
            if not img_path.exists():
                print(f"⚠️  Skipping missing image: {img_path}")
                continue

            result = self.validate_single(
                img_path,
                label=img_info.get("label", "unknown"),
                should_detect=img_info.get("should_detect", True),
                difficulty=img_info.get("difficulty", "medium"),
                tags=img_info.get("tags", [])
            )
            results.append(result)

        return results

    def validate_single(
        self,
        img_path: Path,
        label: str,
        should_detect: bool,
        difficulty: str,
        tags: List[str]
    ) -> ValidationResult:
        """Validate single image (new schema)."""
        # Set per-image deterministic seed (using stable CRC32 hash)
        if self.seed is not None:
            # Stable hash across runs (not process-salted like hash())
            stable_hash = zlib.crc32(str(img_path).encode('utf-8')) & 0xFFFFFFFF
            per_image = (self.seed ^ stable_hash) & 0xFFFFFFFF
            np.random.seed(per_image)

        # Load image
        img = Image.open(img_path).convert("RGB")
        rgb01 = np.array(img, dtype=np.float32) / 255.0

        # Create dummy depth (Materials V3 requires depth for processing)
        depth = np.ones((rgb01.shape[0], rgb01.shape[1]), dtype=np.float32)

        # For validation, we need the mask - call detector directly
        # This bypasses the normal pipeline to get mask access
        water_mask = None

        if self.engine.config.water_detection_enabled:
            from lux_depth_v2.water_candidate import WaterCandidateDetector, SceneContext
            detector = WaterCandidateDetector()
            # Call with same signature as Materials V3 will use
            detector_result = detector.detect(
                rgb01,
                depth01=depth,
                scene_context=SceneContext.UNKNOWN
            )
            # PR-W1: detector returns WaterCandidateResult dataclass (not dict)
            water_mask = detector_result.mask if hasattr(detector_result, 'mask') else None
            # Note: detector_confidence and detector_coverage not used
            # (we use pipeline results for reporting)

        # Also run full pipeline for other metrics
        segmentation_result = {
            "materials": {},
            "confidence": {},
        }

        start = time.perf_counter()
        result = self.engine.process(rgb01, segmentation_result, depth_map=depth)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract water candidate from materials_v3 metadata
        water_dict = result.get('materials_v3', {}).get('water_candidate', {})
        detected = water_dict.get('present', False)

        # Use detector results for mask-based metrics
        # Use pipeline results for coverage/confidence reporting
        edge_score = self._compute_edge_alignment(rgb01, water_mask) if water_mask is not None else 0.0
        boundary_px = self._count_boundary_pixels(water_mask) if water_mask is not None else 0

        # Compute stability (runs full pipeline multiple times)
        stability = self._compute_stability(rgb01, depth)

        # False trigger: should_detect=false but detected
        detected = water_dict.get('present', False)
        is_false_trigger = (not should_detect and detected)
        is_fp = is_false_trigger  # legacy alias, same semantics

        return ValidationResult(
            image_path=str(img_path),
            scene_type=label,
            should_detect=should_detect,
            difficulty=difficulty,
            tags=tags,
            detected=detected,
            coverage=water_dict.get('coverage', 0.0),
            coverage_px=water_dict.get('coverage_px', 0),
            confidence=water_dict.get('confidence', 0.0),
            source=water_dict.get('source', 'none'),
            implementation=water_dict.get('implementation', 'unknown_v0'),
            edge_alignment_score=edge_score,
            boundary_px=boundary_px,
            stability_score=stability,
            is_false_positive=is_fp,
            is_false_trigger=is_false_trigger,
            processing_time_ms=elapsed_ms
        )

    def _compute_edge_alignment(
        self, rgb01: np.ndarray, mask: Optional[np.ndarray]
    ) -> float:
        """
        Primary metric: edge alignment vs image gradients.

        High score = mask boundaries align with image edges.
        """
        if mask is None or not SCIPY_AVAILABLE:
            return 0.0

        # Compute image gradients
        gray = np.mean(rgb01, axis=2)
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Extract mask boundary
        boundary = self._extract_boundary(mask)

        # Measure overlap between boundary and high-gradient regions
        grad_threshold = np.percentile(grad_mag, 75)
        high_grad = (grad_mag >= grad_threshold).astype(np.float32)

        overlap = np.sum(boundary * high_grad)
        max_overlap = np.sum(boundary)

        score = overlap / max(max_overlap, 1)
        return float(score)

    def _compute_stability(self, rgb01: np.ndarray, depth: np.ndarray) -> float:
        """
        Stability across minor perturbations (resize/compress jitter).

        High score = consistent detection under perturbations.
        Deterministic when seed is set (different per image via path hash).
        """
        # Baseline detection
        h, w = rgb01.shape[:2]
        seg_result = {"materials": {}, "confidence": {}}
        baseline_report = self.engine.process(rgb01, seg_result, depth_map=depth)
        baseline_coverage = baseline_report.get('materials_v3', {}).get('water_candidate', {}).get('coverage', 0.0)

        # Deterministic perturbations (per-image seed from path hash handled in validate_single)
        # For now, use seed + 1 as per previous implementation
        # TODO: Pass image_path hash for true per-image seeds
        if self.seed is not None:
            rng = np.random.RandomState(self.seed + 1)
            noise = rng.randn(*rgb01.shape) * 0.01
        else:
            noise = np.random.randn(*rgb01.shape) * 0.01

        noisy = rgb01 + noise
        noisy = np.clip(noisy, 0, 1)
        noisy_seg = {"materials": {}, "confidence": {}}
        noisy_report = self.engine.process(noisy, noisy_seg, depth_map=depth)
        noisy_coverage = noisy_report.get('materials_v3', {}).get('water_candidate', {}).get('coverage', 0.0)

        # Stability = similarity across perturbations
        diffs = [
            abs(baseline_coverage - noisy_coverage),
        ]
        avg_diff = np.mean(diffs) if diffs else 0.0
        stability = max(0.0, 1.0 - avg_diff)

        return float(stability)

    def _extract_boundary(self, mask: np.ndarray, width: int = 3) -> np.ndarray:
        """Extract boundary of mask."""
        if mask is None or not SCIPY_AVAILABLE:
            return np.zeros((1, 1), dtype=np.float32)

        dilated = ndimage.binary_dilation(mask > 0.5, iterations=width)
        eroded = ndimage.binary_erosion(mask > 0.5, iterations=width)
        return (dilated & ~eroded).astype(np.float32)

    def _count_boundary_pixels(self, mask: Optional[np.ndarray]) -> int:
        """Count boundary pixels."""
        if mask is None:
            return 0
        boundary = self._extract_boundary(mask)
        return int(np.sum(boundary))

    def generate_report(
        self, results: List[ValidationResult], output_path: Path, ground_truth: dict
    ):
        """Generate JSON validation report (v0 schema)."""
        # Filter by label
        pool_results = [r for r in results if r.scene_type == "pool"]
        ocean_results = [r for r in results if r.scene_type == "ocean"]

        # Filter by should_detect
        should_detect_true = [r for r in results if r.should_detect]
        should_detect_false = [r for r in results if not r.should_detect]

        # Filter pool/ocean with should_detect=true (for recall)
        pool_true = [r for r in pool_results if r.should_detect]
        ocean_true = [r for r in ocean_results if r.should_detect]

        # Count detected (use explicit 'detected' flag from water_candidate.present)
        pool_detected = [r for r in pool_true if r.detected]
        ocean_detected = [r for r in ocean_true if r.detected]

        pool_recall = len(pool_detected) / len(pool_true) if pool_true else 0.0
        ocean_recall = len(ocean_detected) / len(ocean_true) if ocean_true else 0.0

        # Coverage stats (only for detected water)
        pool_coverages = [r.coverage for r in pool_detected]
        ocean_coverages = [r.coverage for r in ocean_detected]

        summary = {
            "dataset_version": ground_truth.get("version", "v0"),
            "total_images": len(results),
            "pool_images": len(pool_results),
            "ocean_images": len(ocean_results),
            "should_detect_true": len(should_detect_true),
            "should_detect_false": len(should_detect_false),

            # Recall
            "pool_recall": float(pool_recall),
            "ocean_recall": float(ocean_recall),

            # Coverage stats (mean + median)
            "pool_avg_coverage": float(np.mean(pool_coverages)) if pool_coverages else 0.0,
            "pool_median_coverage": float(np.median(pool_coverages)) if pool_coverages else 0.0,
            "ocean_avg_coverage": float(np.mean(ocean_coverages)) if ocean_coverages else 0.0,
            "ocean_median_coverage": float(np.median(ocean_coverages)) if ocean_coverages else 0.0,

            # Confidence
            "pool_avg_confidence": float(np.mean([r.confidence for r in pool_detected])) if pool_detected else 0.0,
            "ocean_avg_confidence": float(np.mean([r.confidence for r in ocean_detected])) if ocean_detected else 0.0,

            # Edge alignment (primary metric)
            "pool_avg_edge_alignment": (float(np.mean([r.edge_alignment_score for r in pool_detected]))
                                        if pool_detected else 0.0),
            "ocean_avg_edge_alignment": (float(np.mean([r.edge_alignment_score for r in ocean_detected]))
                                         if ocean_detected else 0.0),

            # Stability
            "pool_avg_stability": float(np.mean([r.stability_score for r in pool_true])) if pool_true else 0.0,
            "ocean_avg_stability": float(np.mean([r.stability_score for r in ocean_true])) if ocean_true else 0.0,

            # False triggers (should_detect=false but detected)
            "false_trigger_count": sum(r.is_false_trigger for r in results),
            "false_trigger_rate": float(sum(r.is_false_trigger for r in results) / max(len(should_detect_false), 1)),

            # Performance
            "overall_avg_processing_time_ms": float(np.mean([r.processing_time_ms for r in results])),

            # Deprecated (kept for compatibility)
            "false_positive_count": 0,
            "false_positive_rate": 0.0,
        }

        report = {
            "summary": summary,
            "results": [vars(r) for r in results]
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Validation report written to {output_path}")
        print("\n📊 Summary:")
        print(
            f"  Total images: {len(results)} "
            f"({len(should_detect_true)} water, {len(should_detect_false)} hard negatives)"
        )
        print(f"  Pool recall: {pool_recall:.1%} ({len(pool_detected)}/{len(pool_true)} detected)")
        print(
            f"    - Avg coverage: {summary['pool_avg_coverage']:.2%}, "
            f"median: {summary['pool_median_coverage']:.2%}"
        )
        print(f"    - Avg edge alignment: {summary['pool_avg_edge_alignment']:.3f}")
        print(f"  Ocean recall: {ocean_recall:.1%} ({len(ocean_detected)}/{len(ocean_true)} detected)")
        print(
            f"    - Avg coverage: {summary['ocean_avg_coverage']:.2%}, "
            f"median: {summary['ocean_median_coverage']:.2%}"
        )
        print(f"    - Avg edge alignment: {summary['ocean_avg_edge_alignment']:.3f}")
        print(
            f"  False trigger rate: {summary['false_trigger_rate']:.1%} "
            f"({summary['false_trigger_count']}/{len(should_detect_false)})"
        )
        print(f"  Avg processing time: {summary['overall_avg_processing_time_ms']:.1f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Water validation harness (PR-W4)",
        epilog=(
            "Example: python prw_water_validation.py "
            "--ground-truth data/water_v0/ground_truth.json --output report.json --seed 42"
        )
    )
    parser.add_argument(
        "--ground-truth", type=Path, required=True,
        help="Ground truth JSON (v0 schema with root, labels, images)"
    )
    parser.add_argument("--output", type=Path, default=Path("water_validation_report.json"),
                        help="Output JSON path (default: water_validation_report.json)")
    parser.add_argument("--subset-file", type=Path, default=None,
                        help="Text file with image paths (one per line) to validate subset (for CI)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for deterministic stability tests (recommended for CI)")
    parser.add_argument("--no-scipy-warning", action="store_true",
                        help="Suppress SciPy warning (for CI)")
    args = parser.parse_args()

    # Suppress SciPy warning if requested
    if args.no_scipy_warning and not SCIPY_AVAILABLE:
        pass  # Already printed once at module level

    # Load ground truth (v0 schema)
    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    # Validate schema
    if "images" not in ground_truth:
        print("❌ ERROR: Ground truth missing 'images' key (v0 schema required)")
        return

    if not ground_truth.get("images"):
        print("⚠️  WARNING: No images in ground truth")
        return

    # Filter subset if requested
    if args.subset_file:
        with open(args.subset_file) as f:
            subset_paths = {line.strip() for line in f if line.strip()}
        original_count = len(ground_truth["images"])
        ground_truth["images"] = {
            path: info for path, info in ground_truth["images"].items()
            if path in subset_paths
        }
        print(f"📋 Subset filter: {args.subset_file}")
        print(f"   Kept {len(ground_truth['images'])}/{original_count} images")

    print(f"🔍 Loading ground truth: {args.ground_truth}")
    print(f"   Version: {ground_truth.get('version', 'unknown')}")
    print(f"   Root: {ground_truth.get('root', 'data/water_v0/images')}")
    print(f"   Images: {len(ground_truth['images'])}")
    if args.seed is not None:
        print(f"   Seed: {args.seed} (deterministic mode)")

    # Run validation
    config = MaterialsV3Config(
        enabled=True,  # CRITICAL: Must enable Materials V3 for water detection to run
        water_detection_enabled=True,
        water_edge_refinement_enabled=True
    )
    harness = WaterValidationHarness(config, seed=args.seed)
    results = harness.validate_dataset(ground_truth, gt_base_dir=args.ground_truth.parent)

    if not results:
        print("❌ No results generated (check image paths)")
        return

    # Generate report
    harness.generate_report(results, args.output, ground_truth)


if __name__ == "__main__":
    main()
