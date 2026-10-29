"""Independent geometry oracles for the opt-in corrected V3 recipe.

Fixtures use coordinate-coded synthetic images and a controlled model boundary.
They establish spatial support and source authority, never depth accuracy or
representative photographic acceptance. Historical witnesses live separately.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from transformation_portal.depth.backends.protocol import DepthResult, LicenseType
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.input_manager import ImageInput

pytestmark = pytest.mark.unit

CORRECTED_RECIPE = "normalized_crop_support_v1"


def _coordinates(shape=(31, 45), *, alpha=False):
    height, width = shape
    pixels = np.zeros((height, width, 4 if alpha else 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
    pixels[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
    pixels[..., 2] = 173
    if alpha:
        pixels[..., 3] = 255
        pixels[0, 0, 3] = 0
        pixels[-1, -1, 3] = 91
    return pixels


def _orient(pixels, orientation):
    """Independent encoded-to-canonical permutation; no production/PIL helper."""
    operations = {
        1: lambda array: array,
        2: lambda array: array[:, ::-1],
        3: lambda array: array[::-1, ::-1],
        4: lambda array: array[::-1],
        5: lambda array: array.swapaxes(0, 1),
        6: lambda array: np.rot90(array, -1),
        7: lambda array: array[::-1, ::-1].swapaxes(0, 1),
        8: lambda array: np.rot90(array, 1),
    }
    return np.ascontiguousarray(operations[orientation](pixels))


def _source(tmp_path, orientation=1, *, extension="png", shape=(31, 45), alpha=False):
    pixels = _coordinates(shape, alpha=alpha)
    exif = Image.Exif()
    exif[274] = orientation
    path = tmp_path / f"coordinates-{orientation}.{extension}"
    with Image.fromarray(pixels) as image:
        image.save(path, exif=exif)
    if extension == "jpg":
        # JPEG encoding is lossy. The oracle starts from decoded stored samples
        # rather than claiming that pre-encoding coordinate values survive it.
        with Image.open(path) as image:
            pixels = np.asarray(image).copy()
    return path, pixels


def _center_crop(pixels):
    height, width = pixels.shape[:2]
    content_height = min(height, max(14, height // 14 * 14))
    content_width = min(width, max(14, width // 14 * 14))
    top = (height - content_height) // 2
    left = (width - content_width) // 2
    return pixels[top : top + content_height, left : left + content_width], (top, left)


def _support(shape):
    height, width = shape
    support = np.zeros(shape, dtype=bool)
    content_height = min(height, max(14, height // 14 * 14))
    content_width = min(width, max(14, width // 14 * 14))
    top = (height - content_height) // 2
    left = (width - content_width) // 2
    support[top : top + content_height, left : left + content_width] = True
    return support


def _bilinear(array, shape):
    """Scalar pixel-center oracle for enlargement; no library resize calls."""
    height, width = array.shape
    output = np.empty(shape, dtype=np.float32)
    for y in range(shape[0]):
        source_y = max(0.0, min(height - 1.0, (y + 0.5) * height / shape[0] - 0.5))
        y0 = int(np.floor(source_y))
        y1 = min(height - 1, y0 + 1)
        weight_y = source_y - y0
        for x in range(shape[1]):
            source_x = max(0.0, min(width - 1.0, (x + 0.5) * width / shape[1] - 0.5))
            x0 = int(np.floor(source_x))
            x1 = min(width - 1, x0 + 1)
            weight_x = source_x - x0
            upper = array[y0, x0] * (1 - weight_x) + array[y0, x1] * weight_x
            lower = array[y1, x0] * (1 - weight_x) + array[y1, x1] * weight_x
            output[y, x] = upper * (1 - weight_y) + lower * weight_y
    return output


def _nearest(array, shape):
    """Independent nearest pixel-center sampling, with no resize helper."""
    output = np.empty(shape, dtype=array.dtype)
    for y in range(shape[0]):
        source_y = min(array.shape[0] - 1, int((y + 0.5) * array.shape[0] / shape[0]))
        for x in range(shape[1]):
            source_x = min(array.shape[1] - 1, int((x + 0.5) * array.shape[1] / shape[1]))
            output[y, x] = array[source_y, source_x]
    return output


@pytest.mark.parametrize("extension", ["png", "jpg", "tif"])
@pytest.mark.parametrize("orientation", range(1, 9))
def test_corrected_snapshot_normalizes_all_exif_before_cropping(tmp_path, extension, orientation):
    from transformation_portal.lux_depth_v3.geometry import GeometryTransform
    from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

    source, stored = _source(tmp_path, orientation, extension=extension)
    original_bytes = source.read_bytes()
    result = preprocess_geometry_snapshot(source, verify_snapshot=True)
    expected = _orient(stored, orientation)
    expected_crop, _ = _center_crop(expected)

    assert result.transform.source_sha256 == hashlib.sha256(original_bytes).hexdigest()
    assert tuple(result.transform.encoded_shape) == stored.shape[:2]
    assert tuple(result.transform.normalized_shape) == expected.shape[:2]
    assert result.transform.orientation == orientation
    np.testing.assert_array_equal(result.canonical_rgb, expected)
    np.testing.assert_array_equal(result.pixels, expected_crop.astype(np.float32) / 255)
    np.testing.assert_array_equal(result.transform.support(), _support(expected.shape[:2]))
    assert GeometryTransform.from_payload(result.transform.to_payload()) == result.transform
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize("orientation", range(1, 9))
def test_corrected_snapshot_orients_straight_alpha_with_rgb(tmp_path, orientation):
    from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

    source, stored = _source(tmp_path, orientation, alpha=True)
    result = preprocess_geometry_snapshot(source)
    expected = _orient(stored, orientation)

    np.testing.assert_array_equal(result.canonical_rgb, expected[..., :3])
    np.testing.assert_array_equal(result.alpha, expected[..., 3])
    assert np.count_nonzero(result.alpha == 0) == 1
    assert np.count_nonzero(result.alpha == 91) == 1


@pytest.mark.parametrize("orientation", [2, 6])
def test_corrected_palette_transparency_is_explicit_and_oriented(tmp_path, orientation):
    from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

    indices = np.indices((31, 45)).sum(axis=0).astype(np.uint8) % 3
    palette = [11, 22, 33, 71, 83, 97, 131, 147, 159] + [0] * (256 * 3 - 9)
    exif = Image.Exif()
    exif[274] = orientation
    source = tmp_path / "palette.png"
    with Image.fromarray(indices, mode="P") as image:
        image.putpalette(palette)
        image.save(source, transparency=bytes([0, 91, 255]), exif=exif)
    expected_rgb = np.asarray(palette, dtype=np.uint8).reshape(256, 3)[indices]
    expected_alpha = np.asarray([0, 91, 255], dtype=np.uint8)[indices]
    result = preprocess_geometry_snapshot(source)
    np.testing.assert_array_equal(result.canonical_rgb, _orient(expected_rgb, orientation))
    np.testing.assert_array_equal(result.alpha, _orient(expected_alpha, orientation))


@pytest.mark.parametrize("orientation", range(1, 9))
def test_corrected_uint16_tiff_orientation_preserves_source_precision_authority(tmp_path, orientation):
    import tifffile

    from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

    stored = np.zeros((31, 45, 4), dtype=np.uint16)
    stored[..., 0] = np.arange(45, dtype=np.uint16)[None, :] * 1000 + 17
    stored[..., 1] = np.arange(31, dtype=np.uint16)[:, None] * 1700 + 31
    stored[..., 2] = 30001
    stored[..., 3] = 65535
    stored[0, 0, 3] = 0
    stored[-1, -1, 3] = 23191
    source = tmp_path / f"native16-{orientation}.tif"
    tifffile.imwrite(
        source,
        stored,
        photometric="rgb",
        extrasamples="unassalpha",
        metadata=None,
        extratags=[(274, "H", 1, orientation, False)],
    )
    original_bytes = source.read_bytes()
    result = preprocess_geometry_snapshot(source, verify_snapshot=True)
    # The corrected V3 model still receives an explicitly bounded 8-bit proxy;
    # the unchanged 16-bit source is V2's photographic precision authority.
    expected = np.rint(_orient(stored, orientation).astype(np.float64) / 257).astype(np.uint8)
    cropped, _ = _center_crop(expected)
    assert result.source_bit_depth == 16
    assert result.canonical_rgb.dtype == np.uint8
    np.testing.assert_array_equal(result.canonical_rgb, expected[..., :3])
    np.testing.assert_array_equal(result.alpha, expected[..., 3])
    np.testing.assert_array_equal(result.pixels, cropped[..., :3].astype(np.float32) / 255)
    assert result.transform.source_sha256 == hashlib.sha256(original_bytes).hexdigest()
    assert source.read_bytes() == original_bytes
    np.testing.assert_array_equal(tifffile.imread(source), stored)


def test_corrected_mixed_crop_and_symmetric_padding_never_gains_coverage(tmp_path):
    from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

    source, stored = _source(tmp_path, shape=(29, 11))
    result = preprocess_geometry_snapshot(source)
    expected = np.pad(stored[:28], ((0, 0), (1, 2), (0, 0)), mode="edge")
    np.testing.assert_array_equal(result.pixels, expected.astype(np.float32) / 255)
    np.testing.assert_array_equal(result.transform.support(), _support((29, 11)))

    # A huge sentinel confined to model padding cannot leak into restored depth
    # or masks. The removed bottom source row has no model evidence.
    values = np.full((28, 14), 1e6, dtype=np.float32)
    values[:, 1:12] = stored[:28, :, 0].astype(np.float32) + 1
    restored, support = result.transform.restore_numeric(values)
    expected_depth = np.zeros((29, 11), dtype=np.float32)
    expected_depth[:28] = stored[:28, :, 0] + 1
    np.testing.assert_array_equal(restored, expected_depth)
    np.testing.assert_array_equal(support, _support((29, 11)))
    padding_only = np.ones((28, 14), dtype=np.float32)
    padding_only[:, 1:12] = 0
    np.testing.assert_array_equal(result.transform.restore_mask(padding_only), np.zeros((29, 11)))


def test_corrected_resized_depth_uses_pixel_centers_without_crop_stretch(tmp_path):
    from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

    source, _ = _source(tmp_path)
    result = preprocess_geometry_snapshot(source, target_size=42)
    assert result.pixels.shape == (28, 42, 3)
    field = np.add.outer(np.arange(28, dtype=np.float32) * 100, np.arange(42, dtype=np.float32))
    restored, support = result.transform.restore_numeric(field)

    # Pillow rounds its separable float32 passes; the independent scalar
    # interpolation agrees within two float32 ULPs at this field's upper end.
    np.testing.assert_allclose(restored, _bilinear(field, (31, 45)), atol=0.0005, rtol=0)
    np.testing.assert_array_equal(support, np.ones((31, 45), dtype=bool))
    # Landmark values make a half-pixel/axis-swap error visible independently
    # of round-trip agreement between two implementations of the same mistake.
    np.testing.assert_array_equal(restored[[0, -1], [0, -1]], [0, 2741])


def test_corrected_reduced_crop_excludes_unobserved_native_border_and_interpolated_zero(tmp_path):
    from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

    source, _ = _source(tmp_path)
    # Legacy aspect resize truncates 31×45 to 24×35 at long edge 35, then its
    # center crop retains 14×28 at (top 5, left 3). Geometry records that loss.
    result = preprocess_geometry_snapshot(source, target_size=35)
    assert result.pixels.shape == (14, 28, 3)
    depth = np.add.outer(np.arange(14, dtype=np.float32) * 100, np.arange(28, dtype=np.float32)) + 1000
    crop_values = np.zeros((24, 35), dtype=np.float32)
    crop_values[5:19, 3:31] = depth
    crop_support = np.zeros((24, 35), dtype=bool)
    crop_support[5:19, 3:31] = True
    expected_support = _nearest(crop_support, (31, 45))
    interpolated_support = _bilinear(crop_support.astype(np.float32), (31, 45))
    expected = np.zeros((31, 45), dtype=np.float32)
    np.divide(
        _bilinear(crop_values, (31, 45)),
        interpolated_support,
        out=expected,
        where=expected_support,
    )

    actual, support = result.transform.restore_numeric(depth)
    np.testing.assert_array_equal(support, expected_support)
    np.testing.assert_allclose(actual, expected, atol=0.0005, rtol=0)
    assert not np.any(actual[~support])
    assert actual[support].min() >= 1000 - 2 * np.spacing(np.float32(1000))
    labels = np.indices((14, 28))[1] >= 14
    label_canvas = np.zeros((24, 35), dtype=bool)
    label_canvas[5:19, 3:31] = labels
    np.testing.assert_array_equal(result.transform.restore_mask(labels), _nearest(label_canvas, (31, 45)))


@pytest.mark.parametrize("shape", [(27, 42), (28, 41), (28, 42, 1)])
def test_corrected_restoration_rejects_shape_only_model_approximations(tmp_path, shape):
    from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

    source, _ = _source(tmp_path)
    result = preprocess_geometry_snapshot(source)
    with pytest.raises(ValueError):
        result.transform.restore_numeric(np.ones(shape, dtype=np.float32))
    with pytest.raises(ValueError):
        result.transform.restore_mask(np.ones(shape, dtype=np.float32))


@pytest.mark.parametrize(
    "mutation",
    ["source", "orientation", "normalized-shape", "crop-origin", "padding", "float-derived-shape", "bool-padding", "unknown"],
)
def test_geometry_payload_rejects_conflicting_or_untyped_transform_authority(tmp_path, mutation):
    from transformation_portal.lux_depth_v3.geometry import GeometryTransform
    from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

    source, _ = _source(tmp_path)
    result = preprocess_geometry_snapshot(source)
    payload = deepcopy(result.transform.to_payload())
    if mutation == "source":
        payload["source_sha256"] = "not-a-source-digest"
    elif mutation == "orientation":
        payload["orientation"] = 6
    elif mutation == "normalized-shape":
        payload["normalized_shape"][0] += 1
    elif mutation == "crop-origin":
        payload["crop_origin"][0] += 1
    elif mutation == "padding":
        payload["padding"][0] += 1
    elif mutation == "float-derived-shape":
        payload["model_shape"][0] = float(payload["model_shape"][0])
    elif mutation == "bool-padding":
        payload["padding"][0] = False
    else:
        payload["unknown"] = "not authorized"
    with pytest.raises(ValueError):
        GeometryTransform.from_payload(payload)


def test_geometry_result_cannot_mutate_pixels_behind_source_authority(tmp_path):
    from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

    source, _ = _source(tmp_path, alpha=True)
    result = preprocess_geometry_snapshot(source)
    for samples in (result.pixels, result.canonical_rgb, result.alpha):
        assert samples is not None
        assert not samples.flags.writeable
        with pytest.raises(ValueError):
            samples.flat[0] = 255
        with pytest.raises(ValueError):
            samples.setflags(write=True)


@pytest.mark.parametrize("orientation", [0, 9, 65535])
def test_corrected_snapshot_rejects_unknown_orientation(tmp_path, orientation):
    from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

    source, _ = _source(tmp_path, orientation)
    with pytest.raises(ValueError):
        preprocess_geometry_snapshot(source)


def test_corrected_snapshot_rejects_raw_before_loading_optional_runtime(tmp_path, monkeypatch):
    from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

    source = tmp_path / "unmapped.dng"
    source.write_bytes(b"controlled raw fixture; no decoder is authorized")
    monkeypatch.setattr(
        "transformation_portal.lux_depth_v3.ingest_adapter.decode_for_lux_depth",
        lambda *_args, **_kwargs: pytest.fail("corrected RAW must reject before decode"),
    )
    with pytest.raises(ValueError, match="RAW|raw"):
        preprocess_geometry_snapshot(source)


def test_corrected_snapshot_reuses_predecode_descriptor_validation_and_rejects_symlinks(tmp_path):
    from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

    source, _ = _source(tmp_path)
    observations = []

    def reject_opened_source(value):
        observations.append(value.st_size)
        raise ValueError("fixture rejected source descriptor before decode")

    with pytest.raises(ValueError, match="fixture rejected source descriptor"):
        preprocess_geometry_snapshot(source, opened_file_stat_validator=reject_opened_source)
    assert observations == [source.stat().st_size]
    alias = tmp_path / "alias.png"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="regular|symlink"):
        preprocess_geometry_snapshot(alias)


def test_prepared_legacy_replay_and_corrected_profile_cannot_share_depth_cache(tmp_path):
    from transformation_portal.core.execution_identity_v3 import (
        BackendRuntimeIdentity,
        ExecutionIdentityV3,
        MaterializedExecutionIdentityV3,
    )
    from transformation_portal.lux_depth_v3.depth_cache import DEPTH_CACHE_SCHEMA, DepthCache
    from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
    from transformation_portal.stage_graph.registry import StageRegistryIdentifier

    root = tmp_path / "inputs"
    root.mkdir()
    source, _ = _source(root, orientation=6)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    common = {
        "depth_backend": "da2",
        "depth_operational_fallback_chain": ("da2",),
        "allow_synthetic_fallback": False,
        "enable_depth_cache": True,
        "enable_v2": True,
    }
    legacy_default = prepare_lux_execution(EnhanceConfig(**common), root, [source])
    legacy_explicit = prepare_lux_execution(
        EnhanceConfig(geometry_recipe="legacy_encoded_crop_stretch_v1", **common), root, [source]
    )
    corrected = prepare_lux_execution(EnhanceConfig(geometry_recipe=CORRECTED_RECIPE, **common), root, [source])
    assert legacy_default.canonical_plan_bytes == legacy_explicit.canonical_plan_bytes
    assert corrected.canonical_plan_bytes != legacy_default.canonical_plan_bytes

    def materialize(prepared):
        node = next(node for node in prepared.plan.nodes if node.stage_registry_id is StageRegistryIdentifier.LUX_DEPTH)
        seed = ExecutionIdentityV3.from_plan(
            prepared.plan,
            stage_node_id=node.node_id,
            candidate_id="da2",
            input_id=prepared.plan.inputs[0].input_id,
        )
        closure = {
            "dependency_lock_sha256": "4" * 64,
            "interpreter_identity_sha256": "5" * 64,
            "platform_identity_sha256": "6" * 64,
            "accelerator_identity_sha256": "7" * 64,
            "source_identity_sha256": "8" * 64,
        }
        runtime = BackendRuntimeIdentity.from_seed(seed, materialized_weights_sha256="3" * 64, **closure)
        return seed, MaterializedExecutionIdentityV3.from_plan(
            prepared.plan,
            stage_node_id=node.node_id,
            candidate_id="da2",
            input_id=prepared.plan.inputs[0].input_id,
            executed_backend="da2",
            input_content_sha256=source_digest,
            backend_runtime_identities=(runtime,),
            **closure,
        )

    legacy_seed, legacy_identity = materialize(legacy_default)
    explicit_seed, explicit_identity = materialize(legacy_explicit)
    corrected_seed, corrected_identity = materialize(corrected)
    assert legacy_seed.to_canonical_bytes() == explicit_seed.to_canonical_bytes()
    assert legacy_identity.to_canonical_bytes() == explicit_identity.to_canonical_bytes()
    assert legacy_seed.to_canonical_bytes() != corrected_seed.to_canonical_bytes()
    assert legacy_identity.input_content_sha256 == corrected_identity.input_content_sha256
    assert legacy_identity.backend_runtime_identities == corrected_identity.backend_runtime_identities
    assert legacy_identity.cache_key(DEPTH_CACHE_SCHEMA) != corrected_identity.cache_key(DEPTH_CACHE_SCHEMA)

    # Runtime/model/source bytes are identical, but the two image coordinate
    # recipes have incompatible model frames and may never reuse one entry.
    cache = DepthCache(tmp_path / "cache")
    legacy_depth = np.arange(28 * 42, dtype=np.float32).reshape(28, 42)
    corrected_depth = np.arange(42 * 28, dtype=np.float32).reshape(42, 28)
    assert cache.store(legacy_identity, legacy_depth)
    assert cache.get(corrected_identity) is None
    assert cache.store(corrected_identity, corrected_depth)
    np.testing.assert_array_equal(cache.get(legacy_identity), legacy_depth)
    np.testing.assert_array_equal(cache.get(corrected_identity), corrected_depth)


@pytest.mark.parametrize("orientation,wrong_grid", [(value, False) for value in range(1, 9)] + [(6, True)])
def test_corrected_stage_a_writes_only_oriented_native_crop_support(tmp_path, monkeypatch, orientation, wrong_grid):
    """Exercise production Stage A, postprocessing seam and output writers."""
    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module
    from transformation_portal.lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError

    source, stored = _source(tmp_path, orientation)
    observed = []

    class CoordinateBackend:
        name = "da3"
        license_type = LicenseType.COMMERCIAL

        @staticmethod
        def ensure_available():
            return None

        @staticmethod
        def compute(image):
            encoded = np.asarray(image).copy()
            observed.append(encoded)
            depth = encoded[..., 0].astype(np.float32) + 1
            return DepthResult(
                depth_map=depth[:-1] if wrong_grid else depth,
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
        geometry_recipe=CORRECTED_RECIPE,
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
    arguments = {
        "image_input": ImageInput(source),
        "output_key": Path("coordinates"),
        "depth_path": depth_path,
        "float_depth_path": float_path,
        "manifest_path": orchestrator.manifests_dir / "coordinates.json",
        "skip_depth": False,
    }
    if wrong_grid:
        with pytest.raises(LuxExecutionPlanAuthorityError, match="model grid"):
            orchestrator._compute_depth_stage(**arguments)
        assert len(observed) == 1
        assert not depth_path.exists()
        assert not float_path.exists()
        assert not arguments["manifest_path"].exists()
        return
    metadata, *_ = orchestrator._compute_depth_stage(**arguments)

    canonical = _orient(stored, orientation)
    crop, (top, left) = _center_crop(canonical)
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], crop)
    expected = np.zeros(canonical.shape[:2], dtype=np.float32)
    expected[top : top + crop.shape[0], left : left + crop.shape[1]] = crop[..., 0].astype(np.float32) + 1
    np.testing.assert_array_equal(np.load(float_path, allow_pickle=False), expected)
    assert metadata.stats["native_shape"] == list(crop.shape[:2])
    assert metadata.stats["artifact_shape"] == list(canonical.shape[:2])
    assert depth_path.is_file()
