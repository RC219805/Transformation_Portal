"""Independent photographic coordinates through the real V5 graph and files.

Only the model/runtime boundary is controlled. Expected orientation uses source
pixel coordinates, and restoration uses a scalar pixel-center sampler rather
than the production image transforms or restoration helpers.
"""

from __future__ import annotations

import copy
import io
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image

from transformation_portal.core.depth_evidence import build_depth_evidence
from transformation_portal.core.image_artifact import ImageMaster, metadata_payload
from transformation_portal.lux_depth_v4 import evidence as shared_evidence
from transformation_portal.lux_depth_v4 import pipeline as shared_pipeline
from transformation_portal.lux_depth_v4.photography import create_proxy, decode_master
from transformation_portal.lux_depth_v5 import pipeline
from transformation_portal.lux_depth_v5.evidence import verify_execution_evidence_v3
from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request, prepare
from transformation_portal.lux_depth_v5.photography import align_depth

pytestmark = pytest.mark.unit


def _oriented(source, orientation):
    """Place each source coordinate according to the EXIF coordinate table."""
    height, width = source.shape[:2]
    shape = (height, width) if orientation <= 4 else (width, height)
    expected = np.empty((*shape, *source.shape[2:]), dtype=source.dtype)
    for row in range(height):
        for column in range(width):
            locations = {
                1: (row, column),
                2: (row, width - 1 - column),
                3: (height - 1 - row, width - 1 - column),
                4: (height - 1 - row, column),
                5: (column, row),
                6: (column, height - 1 - row),
                7: (width - 1 - column, height - 1 - row),
                8: (width - 1 - column, row),
            }
            expected[locations[orientation]] = source[row, column]
    return expected


def _rgba16():
    row, column = np.indices((73, 119))
    pixels = np.stack((10000 + 173 * row, 13000 + 127 * column, 9000 + 31 * row + 43 * column), axis=-1)
    alpha = np.full((73, 119), 65535)
    alpha[4:11, 13:27] = 17000
    alpha[64:, 101:] = 0
    return np.concatenate((pixels, alpha[..., None]), axis=-1).astype(np.uint16)


def _linear(encoded):
    encoded = encoded.astype(np.float32)
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)


def _model_arrays(shape):
    row, column = np.indices(shape)
    native = (1 + 3 * column + 7 * row).astype(np.float32)
    sky = np.zeros(shape, bool)
    sky[:3, :7] = True
    native[8:11, 19:23] = 0
    if shape == (28, 42):
        native[26:] = 1_000_000
    elif shape == (42, 28):
        native[:, 26:] = 1_000_000
    return native, sky


def _sample(values, y, x):
    """Scalar bilinear interpolation with edge extension and no image library."""
    y, x = max(0, min(y, values.shape[0] - 1)), max(0, min(x, values.shape[1] - 1))
    y0, x0 = math.floor(y), math.floor(x)
    y1, x1 = min(y0 + 1, values.shape[0] - 1), min(x0 + 1, values.shape[1] - 1)
    dy, dx = y - y0, x - x0
    return (1 - dy) * ((1 - dx) * values[y0, x0] + dx * values[y0, x1]) + dy * (
        (1 - dx) * values[y1, x0] + dx * values[y1, x1]
    )


def _aligned_oracle(native, sky, resized, shape):
    height, width = resized
    values = native[:height, :width]
    valid = (values > 0) & ~sky[:height, :width]
    low, high = np.percentile(values[valid], [1, 99])
    relative = np.where(valid, np.clip((values - low) / (high - low), 0, 1), 0)
    result, support = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    usable = np.zeros(shape, bool)
    for row in range(shape[0]):
        y = (row + 0.5) * height / shape[0] - 0.5
        nearest_y = min(height - 1, math.floor((row + 0.5) * height / shape[0]))
        for column in range(shape[1]):
            x = (column + 0.5) * width / shape[1] - 0.5
            nearest_x = min(width - 1, math.floor((column + 0.5) * width / shape[1]))
            weight = _sample(valid, y, x)
            if valid[nearest_y, nearest_x] and weight > 0:
                usable[row, column] = True
                result[row, column] = _sample(relative, y, x) / weight
                support[row, column] = weight
    return result, usable, support


@pytest.fixture
def controlled_worker(monkeypatch):
    class Runtime:
        sha256 = "a" * 64
        source_sha256 = "b" * 64

        def verify(self):
            pass

    class Session:
        def __init__(self, _python, plan, *, cancellation):
            self.plan = plan
            payload = plan.to_payload()
            backend = {
                "model_canonical_key": payload["model"]["canonical_key"],
                "model_repo_id": payload["model"]["repo_id"],
                "model_lock_revision": payload["model"]["revision"],
                "actual_device": payload["device"],
            }
            self.runtime = SimpleNamespace(
                runtime_identity_sha256="c" * 64,
                to_mapping=lambda: {"backend_identity": copy.deepcopy(backend)},
            )

        def compute(self, proxy):
            native, sky = _model_arrays(proxy.shape[:2])
            return {"native_depth": native, "sky_mask": sky}, {
                "native_semantics": "da3_metric_uncalibrated",
                "precision": "fp32",
                "inference_recipe": "tp.da3.explicit_precision_sky.v1",
                "sky_available": True,
            }

        def checkpoint(self):
            pass

        def verify(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(pipeline._ExecutionProfile, "session_type", Session)
    monkeypatch.setattr(shared_pipeline, "PhotographyRuntime", Runtime)
    monkeypatch.setattr(
        shared_pipeline,
        "require_process_supervisor",
        lambda: SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0))),
    )
    monkeypatch.setattr(
        shared_evidence.DA3RuntimeIdentityEvidence,
        "from_mapping",
        lambda value: SimpleNamespace(cacheable=True, to_mapping=lambda: copy.deepcopy(value)),
    )


@pytest.mark.parametrize("orientation", range(1, 9))
def test_all_exif_native_rgba_depth_and_delivery_roundtrip(tmp_path, controlled_worker, orientation):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = _rgba16()
    tifffile.imwrite(
        inputs / "asymmetric.tif",
        source,
        photometric="rgb",
        extrasamples="unassalpha",
        extratags=[(274, "H", 1, orientation, False)],
    )
    request = LuxDepthV5Request(
        inputs, tmp_path / "output", input_color="srgb", target_size=42, strength=0.3, clarity=0, refinement="bilinear"
    )
    result = pipeline.run(prepare(request))
    verified = verify_execution_evidence_v3(result.output_root, expected_plan_sha256=result.plan_fingerprint_sha256)
    assert {item.path for item in verified.artifacts} == set(result.artifact_paths)
    root = result.output_root / "input-0000"
    expected = _oriented(source, orientation)
    original = np.load(root / "source-master.npy", allow_pickle=False)
    np.testing.assert_allclose(original, _linear(expected[..., :3].astype(np.float32) / 65535), atol=1e-7)
    np.testing.assert_array_equal(np.load(root / "alpha.npy", allow_pickle=False), expected[..., 3] / np.float32(65535))
    descriptor = json.loads((root / "photograph.json").read_bytes())
    transform = descriptor["depth"]["geometry"]
    expected_resized = (26, 42) if orientation <= 4 else (42, 26)
    assert transform["original_shape"] == list(expected.shape[:2])
    assert transform["resized_shape"] == list(expected_resized)
    native = np.load(root / "native-depth.npy", allow_pickle=False)
    sky = np.load(root / "native-sky.npy", allow_pickle=False)
    np.testing.assert_array_equal(native, _model_arrays(native.shape)[0])
    relative, valid, support = _aligned_oracle(native, sky, expected_resized, expected.shape[:2])
    np.testing.assert_array_equal(np.load(root / "aligned-depth-valid.npy", allow_pickle=False), valid)
    np.testing.assert_allclose(np.load(root / "relative-depth.npy", allow_pickle=False), relative, atol=3e-7)
    np.testing.assert_allclose(np.load(root / "depth-support-score.npy", allow_pickle=False), support, atol=3e-7)
    finished = np.load(root / "master.npy", allow_pickle=False)
    protected = ~valid | (expected[..., 3] != 65535)
    np.testing.assert_array_equal(finished[protected], original[protected])
    assert np.any(finished[~protected] != original[~protected])
    delivery = tifffile.imread(root / "delivery.tif")
    assert delivery.shape == expected.shape and delivery.dtype == np.uint16
    np.testing.assert_array_equal(delivery[..., 3], expected[..., 3])
    with tifffile.TiffFile(root / "delivery.tif") as image:
        assert image.pages[0].tags[274].value == 1
    assert descriptor["source"]["metadata"]["source_orientation"] == orientation


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
@pytest.mark.parametrize("orientation", range(1, 9))
def test_raster_ingress_uses_source_coordinates_once(image_format, orientation):
    source = (_rgba16() // 257).astype(np.uint8)
    if image_format == "JPEG":
        source = source[..., :3]
    stream = io.BytesIO()
    exif = Image.Exif()
    exif[274] = orientation
    Image.fromarray(source).save(stream, format=image_format, exif=exif, **({"quality": 98} if image_format == "JPEG" else {}))
    # Decode the stored JPEG samples, not its lossy pre-encoding source. Reading
    # does not call exif_transpose; the coordinate table is the sole oracle.
    with Image.open(io.BytesIO(stream.getvalue())) as stored:
        decoded = np.asarray(stored)
    expected = _oriented(decoded, orientation)
    master = decode_master(stream.getvalue(), source_name=f"source.{image_format.lower()}", input_color="srgb")
    np.testing.assert_allclose(master.pixels, _linear(expected[..., :3].astype(np.float32) / 255), atol=1e-7)
    if image_format == "PNG":
        np.testing.assert_array_equal(master.alpha, expected[..., 3] / np.float32(255))
    else:
        assert master.alpha is None
    proxy = create_proxy(master, 42)
    assert proxy.transform.original_shape == expected.shape[:2]


@pytest.mark.parametrize("mutation", ["source", "transform", "missing_master", "missing_proxy"])
def test_unbound_depth_rejected_before_native_finishing_or_write(tmp_path, monkeypatch, mutation):
    master = ImageMaster(np.full((73, 119, 3), 0.2, np.float32), "a" * 64, 16)
    proxy = create_proxy(master, 42)
    native, sky = _model_arrays(proxy.transform.padded_shape)
    evidence = build_depth_evidence(native, sky, proxy, master.source_sha256)
    if mutation == "source":
        evidence = replace(evidence, source_sha256="f" * 64)
    elif mutation == "transform":
        wrong = replace(evidence.transform, original_shape=(72, 119))
        evidence = replace(evidence, transform=wrong)
    else:
        metadata = metadata_payload(evidence.metadata)
        metadata.pop("proxy_master_content_hash" if mutation == "missing_master" else "proxy_content_hash")
        evidence = replace(evidence, metadata=metadata)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unbound geometry reached native editing or publication")

    monkeypatch.setattr(pipeline, "enhance_master_v5", forbidden)
    monkeypatch.setattr(shared_pipeline, "write_delivery", forbidden)
    with pytest.raises(ValueError, match="same source, master, and geometry"):
        aligned = align_depth(evidence, master, proxy, refinement="bilinear")
        pipeline.enhance_master_v5(master, aligned)
        shared_pipeline.write_delivery(master, tmp_path / "forbidden.tif")
    assert not (tmp_path / "forbidden.tif").exists()
