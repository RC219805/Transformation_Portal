"""CPU-only reconstruction contract tests for cold-zone ratchet stability."""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams, GaussianSplat, ReconstructionInput, Scene3D

pytestmark = pytest.mark.unit


def _camera() -> CameraParams:
    return CameraParams(
        intrinsics=np.eye(3, dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32),
        width=2,
        height=2,
    )


def _splat_kwargs() -> dict[str, np.ndarray]:
    return {
        "positions": np.zeros((2, 3), dtype=np.float32),
        "colors": np.full((2, 3), 0.5, dtype=np.float32),
        "scales": np.ones((2, 3), dtype=np.float32),
        "rotations": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 1)),
        "opacities": np.full((2, 1), 0.5, dtype=np.float32),
    }


def _input_kwargs() -> dict[str, object]:
    images = [np.zeros((2, 2, 3), dtype=np.float32), np.ones((2, 2, 3), dtype=np.float32)]
    return {
        "images": images,
        "gamma": 1.0,
        "cameras": [_camera(), _camera()],
        "tier": "apex_research",
    }


def test_camera_rejects_non_float32_extrinsics() -> None:
    with pytest.raises(ValueError, match="Extrinsics must be float32"):
        CameraParams(
            intrinsics=np.eye(3, dtype=np.float32),
            extrinsics=np.eye(4, dtype=np.float64),
            width=2,
            height=2,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("positions", np.zeros((2, 3), dtype=np.float64), "Positions must be float32"),
        ("colors", np.zeros((2, 2), dtype=np.float32), r"Colors must be \(2, 3\)"),
        ("colors", np.zeros((2, 3), dtype=np.float64), "Colors must be float32"),
        ("scales", np.zeros((2, 2), dtype=np.float32), r"Scales must be \(2, 3\)"),
        ("scales", np.zeros((2, 3), dtype=np.float64), "Scales must be float32"),
        ("rotations", np.zeros((2, 3), dtype=np.float32), r"Rotations must be \(2, 4\)"),
        ("rotations", np.zeros((2, 4), dtype=np.float64), "Rotations must be float32"),
        ("opacities", np.zeros((2,), dtype=np.float32), r"Opacities must be \(2, 1\)"),
        ("opacities", np.zeros((2, 1), dtype=np.float64), "Opacities must be float32"),
    ],
)
def test_gaussian_splat_rejects_malformed_core_arrays(field: str, value: np.ndarray, message: str) -> None:
    kwargs = _splat_kwargs()
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        GaussianSplat(**kwargs)


@pytest.mark.parametrize(
    ("sh_coefficients", "message"),
    [
        (np.zeros((2, 3), dtype=np.float32), "SH coefficients must be"),
        (np.zeros((1, 2, 3), dtype=np.float32), "SH coefficients must have 2 entries"),
        (np.zeros((2, 2, 2), dtype=np.float32), "SH coefficients must have 3 color channels"),
    ],
)
def test_gaussian_splat_rejects_malformed_spherical_harmonics(
    sh_coefficients: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GaussianSplat(**_splat_kwargs(), sh_coefficients=sh_coefficients)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"images": [np.zeros((2, 2, 3), dtype=np.float64), np.zeros((2, 2, 3), dtype=np.float32)]},
            "Image 0 must be float32",
        ),
        ({"images": [np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2, 3), dtype=np.float32)]}, r"Image 0 must be"),
        ({"depth_maps": [np.zeros((2, 2), dtype=np.float32)]}, "Number of depth maps"),
        (
            {"depth_maps": [np.zeros((2, 2), dtype=np.float64), np.zeros((2, 2), dtype=np.float32)]},
            "Depth map 0 must be float32",
        ),
        (
            {"depth_maps": [np.zeros((1, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]},
            "Depth map 0 shape",
        ),
        ({"masks": [np.zeros((2, 2), dtype=bool)]}, "Number of masks"),
        ({"masks": [np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=bool)]}, "Mask 0 must be bool"),
        ({"masks": [np.zeros((1, 2), dtype=bool), np.zeros((2, 2), dtype=bool)]}, "Mask 0 shape"),
        ({"material_maps": [{"albedo": np.zeros((2, 2, 3), dtype=np.float32)}]}, "Number of material maps"),
    ],
)
def test_reconstruction_input_rejects_malformed_optional_inputs(
    overrides: dict[str, object],
    message: str,
) -> None:
    kwargs = _input_kwargs()
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        ReconstructionInput(**kwargs)


def test_scene_rejects_negative_iteration() -> None:
    with pytest.raises(ValueError, match="Iteration must be non-negative"):
        Scene3D(
            splats=GaussianSplat(**_splat_kwargs()),
            cameras=[_camera(), _camera()],
            rmse=0.0,
            iteration=-1,
            convergence="converged",
        )
