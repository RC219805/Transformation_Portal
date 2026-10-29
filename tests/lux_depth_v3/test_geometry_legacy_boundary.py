"""Witness unchanged V3 coordinate limitations before a versioned migration.

These fixtures assert historical behavior, not photographic correctness. V3's
standard-input depth grid retains encoded orientation and center-crops to a
multiple of 14; V2 normalizes EXIF independently. The real Stage A then stretches
the cropped depth to the encoded source extent. Changing these witnesses needs
an explicit execution/cache recipe migration, not a new resize hidden in PR6.
No test loads a model or runs inference.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageOps

from transformation_portal.depth.backends.protocol import DepthResult, LicenseType
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.input_manager import ImageInput
from transformation_portal.lux_depth_v3.preprocessing import preprocess_image, preprocess_image_snapshot
from transformation_portal.lux_depth_v3.v2_enhance import load_image_preserve_bit_depth

pytestmark = pytest.mark.unit


def _coordinate_source(tmp_path: Path, orientation: int) -> tuple[Path, np.ndarray]:
    """Freeze asymmetric XY landmarks in a tiny lossless odd-sized source."""
    values = np.zeros((31, 45, 3), dtype=np.uint8)
    values[..., 0] = np.arange(45, dtype=np.uint8)[None, :]
    values[..., 1] = np.arange(31, dtype=np.uint8)[:, None]
    values[..., 2] = 173
    pixels = np.frombuffer(values.tobytes(), dtype=np.uint8).reshape(values.shape)
    exif = Image.Exif()
    exif[274] = orientation
    path = tmp_path / f"coordinates-exif-{orientation}.png"
    with Image.fromarray(pixels) as image:
        image.save(path, exif=exif)
    return path, pixels


@pytest.mark.parametrize("orientation", range(1, 9))
def test_legacy_v3_snapshot_preserves_encoded_crop_while_v2_normalizes_exif(tmp_path, orientation):
    source, pixels = _coordinate_source(tmp_path, orientation)
    original_bytes = source.read_bytes()
    processed, original_shape = preprocess_image(source)
    snapshot, snapshot_shape, digest = preprocess_image_snapshot(source, verify_snapshot=True)
    upright, bits, metadata = load_image_preserve_bit_depth(source)

    # Deliberate legacy witness: one pixel is removed at the top/left and two
    # at the bottom/right, even for an EXIF-mirrored or rotated photograph.
    expected_crop = pixels[1:29, 1:43].astype(np.float32) / 255
    assert not pixels.flags.writeable
    assert original_shape == snapshot_shape == (31, 45)
    assert processed.shape == (28, 42, 3)
    np.testing.assert_array_equal(processed, expected_crop)
    np.testing.assert_array_equal(snapshot, expected_crop)
    np.testing.assert_array_equal(np.rint(processed[0, 0] * 255), [1, 1, 173])
    np.testing.assert_array_equal(np.rint(processed[-1, -1] * 255), [42, 28, 173])
    assert digest == hashlib.sha256(original_bytes).hexdigest()
    assert source.read_bytes() == original_bytes

    with Image.open(source) as encoded:
        with ImageOps.exif_transpose(encoded) as reference:
            expected_upright = np.asarray(reference)
    assert bits == 8
    np.testing.assert_array_equal(upright, expected_upright)
    # Independent asymmetric corner landmarks distinguish mirror operations
    # that have identical dimensions and would pass a shape-only assertion.
    upright_first_xy = [(0, 0), (44, 0), (44, 30), (0, 30), (0, 0), (0, 30), (44, 30), (44, 0)]
    x, y = upright_first_xy[orientation - 1]
    np.testing.assert_array_equal(upright[0, 0], [x, y, 173])
    assert metadata["exif_orientation_applied"] is (orientation != 1)
    assert upright.shape == ((45, 31, 3) if orientation >= 5 else (31, 45, 3))
    if orientation != 1:
        # Mirrors are a mismatch too, even when width/height do not expose it.
        assert upright.shape != pixels.shape or not np.array_equal(upright, pixels)


@pytest.mark.parametrize("orientation", [1, 2, 6])
def test_real_stage_a_stretches_legacy_cropped_depth_over_encoded_extent(tmp_path, monkeypatch, orientation):
    """Use the production Stage A, writer, metadata, and restoration seam."""
    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module

    source, pixels = _coordinate_source(tmp_path, orientation)
    calls = []

    class CoordinateBackend:
        name = "da3"
        license_type = LicenseType.COMMERCIAL

        @staticmethod
        def ensure_available():
            return None

        @staticmethod
        def compute(image):
            encoded = np.asarray(image).copy()
            calls.append(encoded)
            # A deterministic coordinate field isolates geometry from model
            # behavior. These are fixture scalars, not calibrated depth.
            return DepthResult(
                depth_map=encoded[..., 0].astype(np.float32),
                original_image=encoded,
                metadata={},
                depth_units="relative",
                backend_id="da3",
                device="cpu",
            )

    backend = CoordinateBackend()

    class FixtureRegistry:
        @staticmethod
        def get_backend(_backend_id, _config=None, **_kwargs):
            return backend

    monkeypatch.setattr(orchestrator_module, "DepthBackendRegistry", FixtureRegistry)
    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        depth_operational_fallback_chain=("da3",),
        depth_fallback="fail",
        enable_v2=False,
        enable_materials_v3=False,
        enable_depth_cache=False,
        save_float_depth=True,
    )
    orchestrator = orchestrator_module.EnhanceOrchestrator(config, tmp_path / "output")
    orchestrator.postprocessor = SimpleNamespace(process=lambda result: result)
    depth_path = orchestrator.depth_dir / "coordinates.png"
    float_path = orchestrator.depth_dir / "coordinates.npy"
    metadata, *_ = orchestrator._compute_depth_stage(
        image_input=ImageInput(source),
        output_key=Path("coordinates"),
        depth_path=depth_path,
        float_depth_path=float_path,
        manifest_path=orchestrator.manifests_dir / "coordinates.json",
        skip_depth=False,
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], pixels[1:29, 1:43])
    restored = np.load(float_path, allow_pickle=False)
    assert restored.shape == (31, 45)
    assert metadata.stats["native_shape"] == [28, 42]
    assert metadata.stats["artifact_shape"] == [31, 45]
    # The actual production resize cannot recover the discarded edge pixels.
    # A full-extent coordinate field would instead yield [0, 22, 44].
    np.testing.assert_array_equal(restored[0, [0, 22, 44]], [1, 21.5, 42])
    assert not np.array_equal(restored, pixels[..., 0].astype(np.float32))
    assert depth_path.is_file()
