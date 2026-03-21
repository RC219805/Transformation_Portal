"""Performance validation tests for Phase 2 components.

This module validates performance targets defined in planning:
- SAM2: 512x512 in <3s (GPU)
- Materials: 1024x1024 in <10s (GPU)
- 3DGS: 3-view in <30s (GPU)
- E2E: <60s total (512x512, GPU)

Performance tests are marked with @pytest.mark.benchmark and record
baseline metrics in a performance ledger for regression tracking.

Note: Performance targets assume GPU availability. CPU fallback will be
slower and is documented separately.
"""

import json
import platform
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# Performance ledger path
PERFORMANCE_LEDGER = Path(__file__).parent / "performance_ledger.json"


class PerformanceLedger:
    """Track performance metrics and detect regressions."""

    def __init__(self, ledger_path: Path = PERFORMANCE_LEDGER):
        self.ledger_path = ledger_path
        self.metrics = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load existing metrics from ledger."""
        if self.ledger_path.exists():
            with open(self.ledger_path) as f:
                return json.load(f)
        return {}

    def save(self):
        """Save metrics to ledger."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def record(self, test_name: str, duration: float, metadata: Dict[str, Any]):
        """Record a performance measurement."""
        if test_name not in self.metrics:
            self.metrics[test_name] = {
                "baseline": duration,
                "measurements": [],
                "metadata": metadata,
            }

        self.metrics[test_name]["measurements"].append(
            {
                "duration": duration,
                "timestamp": time.time(),
            }
        )

        # Keep last 100 measurements
        self.metrics[test_name]["measurements"] = self.metrics[test_name]["measurements"][-100:]

    def get_baseline(self, test_name: str) -> float:
        """Get baseline duration for a test."""
        if test_name not in self.metrics:
            return None
        return self.metrics[test_name]["baseline"]

    def check_regression(self, test_name: str, duration: float, threshold: float = 1.5):
        """Check if duration represents a regression."""
        baseline = self.get_baseline(test_name)
        if baseline is None:
            return False  # No baseline yet

        return duration > baseline * threshold


@pytest.fixture
def performance_ledger():
    """Provide performance ledger for tests."""
    return PerformanceLedger()


@pytest.fixture
def test_image_512():
    """Create a 512x512 test image."""
    return np.random.rand(512, 512, 3).astype(np.float32)


@pytest.fixture
def test_image_1024():
    """Create a 1024x1024 test image."""
    return np.random.rand(1024, 1024, 3).astype(np.float32)


@pytest.fixture
def test_mask_512():
    """Create a 512x512 test mask."""
    mask = np.zeros((512, 512), dtype=bool)
    mask[100:400, 100:400] = True
    return mask


@pytest.fixture
def test_mask_1024():
    """Create a 1024x1024 test mask."""
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[200:800, 200:800] = True
    return mask


def get_hardware_info() -> Dict[str, Any]:
    """Get hardware information for performance tracking."""
    info = {
        "platform": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # Try to detect GPU
    try:
        import torch

        info["torch_version"] = torch.__version__ if hasattr(torch, "__version__") else "stub"
        info["cuda_available"] = torch.cuda.is_available() if hasattr(torch, "cuda") else False
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
            info["mps_available"] = torch.backends.mps.is_available()
        else:
            info["mps_available"] = False

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
        elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu_name"] = "Apple Metal Performance Shaders"
    except ImportError:
        info["torch_available"] = False

    return info


@pytest.mark.benchmark
class TestMaterialsPerformance:
    """Performance tests for material generation."""

    def test_materials_heuristic_1024_performance(self, test_image_1024, test_mask_1024, performance_ledger):
        """Validate materials process 1024x1024 in <10s (target)."""
        from transformation_portal.spatial_ai.materials.heuristic_fallback import HeuristicFallback

        generator = HeuristicFallback()

        # Warmup
        _ = generator.generate_pbr_textures(
            test_image_1024,
            mask=test_mask_1024,
            material_hint="wood",
        )

        # Measure
        start = time.time()
        _ = generator.generate_pbr_textures(
            test_image_1024,
            mask=test_mask_1024,
            material_hint="wood",
        )
        duration = time.time() - start

        # Record
        metadata = {
            "image_size": "1024x1024",
            "hardware": get_hardware_info(),
            "backend": "heuristic",
        }
        performance_ledger.record("materials_heuristic_1024", duration, metadata)
        performance_ledger.save()

        # Heuristic backend should be fast (<1s typically, <20s on CI runners)
        print(f"\nMaterials 1024x1024 (heuristic): {duration:.2f}s")
        print("Target: <10s (neural), <1s (heuristic on local), <20s (CI runner variance)")

        # Warnings-only approach until L0.2 baseline comparison is implemented
        # (per benchmark CI policy: tests/benchmarks/README.md)
        if duration > 10.0:
            import warnings

            warnings.warn(
                f"Materials processing slower than target: {duration:.2f}s > 10s. "
                f"This may indicate CI runner variance or a genuine regression. "
                f"Will be enforced as blocking once baseline comparison is implemented.",
                UserWarning,
            )

        # Hard limit to catch catastrophic regressions (e.g., infinite loops)
        assert duration < 30.0, f"Materials critically slow: {duration:.2f}s > 30s (likely a bug)"

    def test_materials_performance_baseline_exists(self, performance_ledger):
        """Document materials performance baseline."""
        baseline = performance_ledger.get_baseline("materials_heuristic_1024")
        if baseline:
            print(f"\nMaterials 1024x1024 baseline: {baseline:.2f}s")  # noqa: F541
        else:
            print("\nNo materials baseline recorded yet.")


@pytest.mark.benchmark
class TestPerformanceRegression:
    """Tests for performance regression detection."""

    def test_no_significant_regressions(self, performance_ledger):
        """Check for significant performance regressions (warnings-only until baseline comparison).

        Per benchmark CI policy (tests/benchmarks/README.md): benchmarks use warnings-only
        approach in PR gating until L0.2 implements baseline comparison with % tolerance.
        This prevents CI flakiness from blocking PRs while still raising awareness.
        """
        regressions = []
        catastrophic_regressions = []

        for test_name, data in performance_ledger.metrics.items():
            if not data["measurements"]:
                continue

            baseline = data["baseline"]
            recent = data["measurements"][-1]["duration"]

            # Detect regressions > 50% (informational)
            if recent > baseline * 1.5:
                regressions.append(
                    {
                        "test": test_name,
                        "baseline": baseline,
                        "recent": recent,
                        "regression": (recent / baseline - 1) * 100,
                    }
                )

            # Detect catastrophic regressions using the 5x relative rule, but
            # apply only a small floor for very tiny baselines to reduce CI noise.
            baseline_floor_seconds = 0.1
            catastrophic_threshold = max(baseline, baseline_floor_seconds) * 5.0
            if recent > catastrophic_threshold:
                catastrophic_regressions.append(
                    {
                        "test": test_name,
                        "baseline": baseline,
                        "recent": recent,
                        "regression": (recent / baseline - 1) * 100,
                    }
                )

        if regressions:
            print("\n⚠️  Performance regressions detected (informational):")
            for reg in regressions:
                print(f"  {reg['test']}: {reg['baseline']:.2f}s → {reg['recent']:.2f}s " f"({reg['regression']:.1f}% slower)")
            print("\nNote: Regressions are warnings-only until baseline comparison is implemented.")
            print("This may indicate CI runner variance rather than a genuine regression.")
        else:
            print("\n✅ No significant performance regressions detected")

        # Only fail on catastrophic regressions (> 5x slower, likely bugs)
        if catastrophic_regressions:
            assert (
                len(catastrophic_regressions) == 0
            ), f"Catastrophic performance regressions detected (>5x slower): {catastrophic_regressions}"


@pytest.mark.benchmark
class TestPerformanceDocumentation:
    """Tests documenting expected performance characteristics."""

    def test_document_hardware_requirements(self):
        """Document hardware requirements for performance targets."""
        hw = get_hardware_info()

        print("\n=== Hardware Information ===")
        for key, value in hw.items():
            print(f"{key}: {value}")

        print("\n=== Performance Targets ===")
        print("SAM2 (512x512):")
        print("  - GPU: <3s")
        print("  - CPU: <15s")
        print("\nMaterials (1024x1024):")
        print("  - Heuristic: <1s")
        print("  - Neural (GPU): <10s")
        print("\n3DGS (3-view):")
        print("  - GPU: <30s")
        print("\nE2E Pipeline (512x512):")
        print("  - GPU: <60s")

    def test_document_optimization_strategies(self):
        """Document optimization strategies for performance."""
        print("\n=== Optimization Strategies ===")
        print("1. Model Caching:")
        print("   - Lazy load models to reduce import time")
        print("   - Cache loaded models for repeated inference")
        print("\n2. Batch Processing:")
        print("   - Process multiple images in batches when possible")
        print("   - Use GPU batch size tuning")
        print("\n3. Hardware Acceleration:")
        print("   - Prioritize CoreML on Apple Silicon")
        print("   - Use CUDA on NVIDIA GPUs")
        print("   - Fall back to CPU with warning")
        print("\n4. Resolution Scaling:")
        print("   - Downsample large images for preview")
        print("   - Full resolution for final output")
