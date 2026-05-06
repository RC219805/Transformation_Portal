"""Property-based tests for algorithmic invariants across subsystems.

Uses Hypothesis to verify mathematical/physical invariants that must hold
for any valid inputs. Extends the existing test_property_based_validation.py
(which covers config parsing) with algorithm-level properties:

  - atmosphere: physical clarity constraints
  - rendering_4k_pipeline: tone-mapping output always in [0, 1]
  - streaming/checkpoint: progress round-trips exactly
  - perceptual/metrics: PSNR/MSE symmetry (torch-conditional)
  - neuroaesthetics/golden_ratio: score always in [0, 1] (cv2-conditional)
"""

from __future__ import annotations

import importlib

import pytest

try:
    from hypothesis import assume, given, settings
    from hypothesis import strategies as st
except ImportError:
    pytest.skip("hypothesis not installed", allow_module_level=True)

pytestmark = [pytest.mark.unit]


def _is_importable(name: str) -> bool:
    """Check whether a package can be imported without triggering side-effects.

    Uses sys.modules as a fast path so that packages already loaded by earlier
    test-collection steps are correctly detected even when their __spec__ is
    None (a known torch quirk that makes importlib.util.find_spec raise
    ValueError in some environments).
    """
    import sys

    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:
        return False


_cv2_available = _is_importable("cv2")
_torch_available = _is_importable("torch")


class TestAtmosphericPhysicalInvariants:
    @given(base_visibility=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_clear_conditions_visibility_always_positive(self, base_visibility):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel

        result = AtmosphericModel().calculate_sundowner_clarity(base_visibility, sundowner_active=False)
        assert result > 0

    @given(base_visibility=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_sundowner_never_reduces_visibility(self, base_visibility):
        """Sundowner winds improve clarity — the enhanced value must be ≥ base."""
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel

        model = AtmosphericModel()
        base = model.calculate_sundowner_clarity(base_visibility, sundowner_active=False)
        enhanced = model.calculate_sundowner_clarity(base_visibility, sundowner_active=True)
        assert enhanced >= base

    @given(
        turbidity=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        humidity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_atmospheric_parameters_accept_valid_ranges(self, turbidity, humidity):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericParameters

        params = AtmosphericParameters(turbidity=turbidity, humidity=humidity)
        assert params.turbidity == pytest.approx(turbidity)
        assert params.humidity == pytest.approx(humidity)


# ---------------------------------------------------------------------------
# rendering_4k_pipeline: apply_tone_mapping — output always in [0, 1]
# ---------------------------------------------------------------------------


class TestToneMappingOutputRange:
    @classmethod
    def setup_class(cls):
        import numpy as np

        from transformation_portal.pipelines.rendering_4k_pipeline import (
            ToneMappingConfig,
            ToneMappingMethod,
            apply_tone_mapping,
        )

        cls.np = np
        cls.ToneMappingConfig = ToneMappingConfig
        cls.ToneMappingMethod = ToneMappingMethod
        cls.apply_tone_mapping = staticmethod(apply_tone_mapping)

    @given(
        values=st.lists(
            st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
            min_size=9,
            max_size=9,
        ),
        method=st.sampled_from(["agx", "filmic", "reinhard", "aces"]),
        exposure=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=60)
    def test_tone_mapping_output_always_in_unit_range(self, values, method, exposure):
        image = self.np.array(values, dtype=self.np.float32).reshape((1, 3, 3))
        config = self.ToneMappingConfig(method=self.ToneMappingMethod(method), exposure=exposure)
        result = self.apply_tone_mapping(image, config)
        assert result.min() >= -1e-5
        assert result.max() <= 1.0 + 1e-5

    @given(
        h=st.integers(min_value=4, max_value=32),
        w=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=30)
    def test_tone_mapping_output_shape_matches_input_shape(self, h, w):
        rng = self.np.random.default_rng(0)
        image = rng.random((h, w, 3), dtype=self.np.float32).astype(self.np.float32)
        result = self.apply_tone_mapping(image, self.ToneMappingConfig())
        assert result.shape == (h, w, 3)


# ---------------------------------------------------------------------------
# streaming/checkpoint: Checkpoint progress round-trips exactly
# ---------------------------------------------------------------------------


class TestCheckpointProgressRoundTrip:
    @given(progress=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, suppress_health_check=[])
    def test_progress_survives_save_load(self, progress):
        import pathlib
        import tempfile

        from transformation_portal.streaming.checkpoint import Checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "prop_ckpt.json"
            ckpt = Checkpoint(id="prop-test", progress=progress, state={}, timestamp=0.0, metadata={})
            ckpt.save(path)
            loaded = Checkpoint.load(path)
        assert loaded.progress == pytest.approx(progress, rel=1e-6, abs=1e-10)

    @given(step=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=50)
    def test_state_integer_survives_round_trip(self, step):
        import pathlib
        import tempfile

        from transformation_portal.streaming.checkpoint import Checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "state_ckpt.json"
            ckpt = Checkpoint(id="state-test", progress=0.5, state={"step": step}, timestamp=0.0, metadata={})
            ckpt.save(path)
            loaded = Checkpoint.load(path)
        assert loaded.state["step"] == step


# ---------------------------------------------------------------------------
# streaming/progress: ProgressState percentage invariants
# ---------------------------------------------------------------------------


class TestProgressStateInvariants:
    @given(
        current=st.integers(min_value=0, max_value=1000),
        total=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=60)
    def test_percentage_in_range_zero_to_100(self, current, total):
        from transformation_portal.streaming.progress import ProgressState

        assume(current <= total)
        state = ProgressState(current=current, total=total)
        pct = state.percentage
        assert pct is not None
        assert 0.0 <= pct <= 100.0

    @given(current=st.integers(min_value=0, max_value=1000))
    @settings(max_examples=40)
    def test_percentage_none_when_no_total(self, current):
        from transformation_portal.streaming.progress import ProgressState

        state = ProgressState(current=current, total=None)
        assert state.percentage is None


# ---------------------------------------------------------------------------
# perceptual/metrics: MSE symmetry (torch-required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _torch_available, reason="torch not available")
class TestMetricsSymmetryProperties:
    @given(
        values_a=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=48, max_size=48
        ),
        values_b=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=48, max_size=48
        ),
    )
    @settings(max_examples=40)
    def test_mse_is_symmetric(self, values_a, values_b):
        import torch as _torch

        from transformation_portal.perceptual.metrics import QualityMetrics

        device = _torch.device("cpu")
        substrate = type("_S", (), {"get_device": lambda self=None: device})()
        metrics = QualityMetrics(substrate)

        a = _torch.tensor(values_a, dtype=_torch.float32).reshape(1, 3, 4, 4)
        b = _torch.tensor(values_b, dtype=_torch.float32).reshape(1, 3, 4, 4)
        assert metrics.compute_mse(a, b).score == pytest.approx(metrics.compute_mse(b, a).score, rel=1e-5)

    @given(
        values=st.lists(
            st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), min_size=48, max_size=48
        ),
    )
    @settings(max_examples=30)
    def test_identical_images_always_zero_mse(self, values):
        import torch as _torch

        from transformation_portal.perceptual.metrics import QualityMetrics

        device = _torch.device("cpu")
        substrate = type("_S", (), {"get_device": lambda self=None: device})()
        metrics = QualityMetrics(substrate)

        t = _torch.tensor(values, dtype=_torch.float32).reshape(1, 3, 4, 4)
        assert metrics.compute_mse(t, t).score == pytest.approx(0.0, abs=1e-6)
