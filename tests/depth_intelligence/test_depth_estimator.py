"""Regression tests for ``depth_intelligence.DepthEstimator`` (D-26).

Covers the backend fallback on a failed model load and the memory bound of
the local-variance confidence map. No model weights are loaded: the model
class is replaced by a stand-in.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch", reason="DepthEstimator requires torch")

from transformation_portal.depth_intelligence import depth_estimator as de  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.ml]


def _substrate(device_type: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(get_device=lambda: types.SimpleNamespace(type=device_type))


def _model_class(attempts: list, fail_backends: set) -> type:
    class _Model:
        def __init__(self, *, variant, backend, precision) -> None:
            attempts.append(backend)
            if backend in fail_backends:
                raise RuntimeError(f"cannot load {backend.name}")

    return _Model


def test_cpu_load_failure_is_not_retried_on_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list = []
    monkeypatch.setattr(de, "DepthAnythingV2Model", _model_class(attempts, {de.ModelBackend.PYTORCH_CPU}))

    with pytest.raises(RuntimeError, match="cannot load PYTORCH_CPU"):
        de.DepthEstimator(_substrate("cpu"))

    assert attempts == [de.ModelBackend.PYTORCH_CPU]


def test_mps_load_failure_is_not_retried_on_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list = []
    monkeypatch.setattr(de, "DepthAnythingV2Model", _model_class(attempts, {de.ModelBackend.PYTORCH_MPS}))
    config = de.DepthConfig(backend=de.ModelBackend.PYTORCH_MPS)

    with pytest.raises(RuntimeError, match="cannot load PYTORCH_MPS"):
        de.DepthEstimator(_substrate("mps"), config)

    assert attempts == [de.ModelBackend.PYTORCH_MPS]


def test_coreml_load_failure_on_mps_substrate_falls_back_to_pytorch_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list = []
    monkeypatch.setattr(de, "DepthAnythingV2Model", _model_class(attempts, {de.ModelBackend.COREML}))

    estimator = de.DepthEstimator(_substrate("mps"))

    assert attempts == [de.ModelBackend.COREML, de.ModelBackend.PYTORCH_MPS]
    assert estimator.config.backend == de.ModelBackend.PYTORCH_MPS


def _whole_frame_confidence(depth):
    patches = torch.nn.functional.unfold(depth.unsqueeze(0).unsqueeze(0), kernel_size=5, padding=2)
    return 1.0 - torch.clamp(patches.var(dim=1).view(depth.shape), 0, 1)


@pytest.mark.parametrize("band_elements", [1, 25 * 37 * 3, 25 * 37 * 7, 1 << 30])
def test_banded_confidence_matches_the_whole_frame_unfold(monkeypatch: pytest.MonkeyPatch, band_elements: int) -> None:
    monkeypatch.setattr(de, "_CONFIDENCE_BAND_ELEMENTS", band_elements)
    generator = torch.Generator().manual_seed(7)
    depth = torch.rand(29, 37, generator=generator)
    estimator = de.DepthEstimator.__new__(de.DepthEstimator)

    confidence = estimator._compute_confidence(depth)

    assert confidence.shape == depth.shape
    assert torch.equal(confidence, _whole_frame_confidence(depth))


def test_confidence_unfold_stays_within_the_band_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = 25 * 64 * 4
    monkeypatch.setattr(de, "_CONFIDENCE_BAND_ELEMENTS", budget)
    real_unfold = torch.nn.functional.unfold
    unfolded_sizes: list = []

    def _recording_unfold(*args, **kwargs):
        out = real_unfold(*args, **kwargs)
        unfolded_sizes.append(out.numel())
        return out

    monkeypatch.setattr(torch.nn.functional, "unfold", _recording_unfold)
    depth = torch.rand(96, 64)
    estimator = de.DepthEstimator.__new__(de.DepthEstimator)

    estimator._compute_confidence(depth)

    assert unfolded_sizes
    assert max(unfolded_sizes) <= budget
