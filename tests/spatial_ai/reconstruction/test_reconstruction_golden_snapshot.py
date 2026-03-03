"""Golden state-hash regression sentinel for reconstruction output."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is required for reconstruction golden snapshot tests")
pytestmark = [pytest.mark.ml, pytest.mark.golden]

from transformation_portal.spatial_ai.reconstruction import (  # pylint: disable=wrong-import-position
    CameraParams,
    GaussianBackend,
    ReconstructionInput,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "reconstruction" / "tiny_scene_cpu.json"
UPDATE_ENV = "UPDATE_RECONSTRUCTION_SNAPSHOT"
REQUESTED_ITERATIONS = 8
OPTIMIZATION_SEED = 123
FIXTURE_NAME = "tiny_scene_v1"
EXPECTED_BACKEND = "gaussian_splatting"


def _build_tiny_reconstruction_input() -> tuple[ReconstructionInput, CameraParams]:
    """Build deterministic 2-view fixture with sparse, bounded initialization."""
    h = 40
    w = 40

    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(y, x, indexing="ij")

    image_a = np.stack(
        [
            x_grid,
            y_grid,
            0.5 * (x_grid + y_grid),
        ],
        axis=-1,
    ).astype(np.float32)
    image_b = np.stack(
        [
            np.clip(x_grid * 0.9 + 0.08, 0.0, 1.0),
            np.clip(y_grid * 0.85 + 0.1, 0.0, 1.0),
            np.clip(0.65 * x_grid + 0.35 * y_grid, 0.0, 1.0),
        ],
        axis=-1,
    ).astype(np.float32)

    intrinsics = np.array(
        [[32.8, 0.0, w / 2.0], [0.0, 32.8, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    extrinsics_a = np.eye(4, dtype=np.float32)
    extrinsics_b = np.eye(4, dtype=np.float32)
    extrinsics_b[0, 3] = 0.1

    camera_a = CameraParams(intrinsics=intrinsics, extrinsics=extrinsics_a, width=w, height=h)
    camera_b = CameraParams(intrinsics=intrinsics, extrinsics=extrinsics_b, width=w, height=h)

    reconstruction_input = ReconstructionInput(
        images=[image_a, image_b],
        gamma=1.0,
        cameras=[camera_a, camera_b],
        tier="apex_research",
    )

    return reconstruction_input, camera_a


def _canon_f32(arr: np.ndarray) -> np.ndarray:
    """Canonical float32 normalization for stable hashing."""
    return np.ascontiguousarray(np.round(np.asarray(arr, dtype=np.float32), 6))


def _assert_scene_state_is_finite(scene) -> None:
    """Fail early if reconstruction produced non-finite state values."""
    for name, arr in [
        ("positions", scene.splats.positions),
        ("colors", scene.splats.colors),
        ("scales", scene.splats.scales),
        ("rotations", scene.splats.rotations),
        ("opacities", scene.splats.opacities),
    ]:
        if not np.isfinite(arr).all():
            raise AssertionError(f"Reconstruction scene contains non-finite values in {name}.")
    if not np.isfinite(scene.rmse):
        raise AssertionError(f"Reconstruction RMSE must be finite, got {scene.rmse!r}.")


def _state_hash(scene) -> str:
    """Compute canonical state hash for reconstruction output."""
    digest = hashlib.sha256()

    for name, arr in [
        ("positions", scene.splats.positions),
        ("colors", scene.splats.colors),
        ("scales", scene.splats.scales),
        ("rotations", scene.splats.rotations),
        ("opacities", scene.splats.opacities),
    ]:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_canon_f32(arr).tobytes())

    digest.update(b"iteration\0")
    digest.update(np.asarray([scene.iteration], dtype=np.int64).tobytes())

    digest.update(b"convergence\0")
    digest.update(scene.convergence.encode("utf-8"))

    digest.update(b"rmse\0")
    digest.update(np.asarray([round(scene.rmse, 6)], dtype=np.float32).tobytes())

    return digest.hexdigest()


def _canonical_payload(scene, requested_iterations: int) -> dict[str, object]:
    """Build canonical golden payload from runtime output."""
    _assert_scene_state_is_finite(scene)

    return {
        "schema": "reconstruction_golden.v1",
        "backend": str(scene.metadata.get("backend", "")),
        "device": "cpu",
        "fixture": FIXTURE_NAME,
        "optimization_seed": OPTIMIZATION_SEED,
        "requested_iterations": requested_iterations,
        "iteration": scene.iteration,
        "convergence": scene.convergence,
        "rmse": float(round(scene.rmse, 6)),
        "state_hash": _state_hash(scene),
    }


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    """Serialize payload deterministically for stable golden bytes."""
    text = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        separators=(",", ": "),
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"{text}\n".encode("utf-8")


def _write_golden(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(payload))


def _read_golden(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_hash_diagnostic(scene, camera0: CameraParams, backend: GaussianBackend) -> str:
    """Compute non-gating diagnostic render hash."""
    rendered = backend.render_view(scene, camera0)
    rendered_canon = np.ascontiguousarray(np.round(np.asarray(rendered, dtype=np.float32), 5))
    digest = hashlib.sha256()
    digest.update(b"render\0")
    digest.update(rendered_canon.tobytes())
    return digest.hexdigest()


def _run_reconstruction() -> tuple[object, str]:
    reconstruction_input, camera0 = _build_tiny_reconstruction_input()
    backend = GaussianBackend(
        tier="apex_research",
        device="cpu",
        optimization_seed=OPTIMIZATION_SEED,
    )
    scene = backend.reconstruct(reconstruction_input, iterations=REQUESTED_ITERATIONS)
    return scene, _render_hash_diagnostic(scene, camera0, backend)


def test_reconstruction_golden_snapshot_state_hash():
    scene, render_hash = _run_reconstruction()
    payload = _canonical_payload(scene, requested_iterations=REQUESTED_ITERATIONS)

    if payload["backend"] != EXPECTED_BACKEND:
        pytest.fail(f"Unexpected backend metadata value: {payload['backend']!r} (expected {EXPECTED_BACKEND!r})")

    if os.getenv(UPDATE_ENV) == "1":
        _write_golden(GOLDEN_PATH, payload)
        return

    if not GOLDEN_PATH.exists():
        pytest.fail(
            "Reconstruction golden snapshot missing. Generate with:\n"
            "  UPDATE_RECONSTRUCTION_SNAPSHOT=1 pytest -q "
            'tests/spatial_ai/reconstruction/test_reconstruction_golden_snapshot.py -m "ml and golden"'
        )

    expected_payload = _read_golden(GOLDEN_PATH)
    if payload != expected_payload:
        pytest.fail(
            "Reconstruction golden snapshot mismatch.\n"
            f"Expected state_hash: {expected_payload.get('state_hash')}\n"
            f"Actual state_hash:   {payload.get('state_hash')}\n"
            f"Render hash (diagnostic-only): {render_hash}\n"
            "To accept these changes, run:\n"
            "  UPDATE_RECONSTRUCTION_SNAPSHOT=1 pytest -q "
            'tests/spatial_ai/reconstruction/test_reconstruction_golden_snapshot.py -m "ml and golden"'
        )


def test_reconstruction_golden_artifact_is_byte_stable():
    scene_a, _render_hash_a = _run_reconstruction()
    scene_b, _render_hash_b = _run_reconstruction()

    payload_a = _canonical_payload(scene_a, requested_iterations=REQUESTED_ITERATIONS)
    payload_b = _canonical_payload(scene_b, requested_iterations=REQUESTED_ITERATIONS)

    assert payload_a == payload_b

    payload_bytes_a = _canonical_json_bytes(payload_a)
    payload_bytes_b = _canonical_json_bytes(payload_b)
    assert payload_bytes_a == payload_bytes_b

    if GOLDEN_PATH.exists() and os.getenv(UPDATE_ENV) != "1":
        assert payload_bytes_a == GOLDEN_PATH.read_bytes()
