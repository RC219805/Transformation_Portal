from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from transformation_portal.core.image_artifact import ImageMaster, artifact_content_hash
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.photography import linear_to_srgb, output_srgb_icc, srgb_to_linear
from transformation_portal.photographic_delivery import (
    DeliveryImage,
    ExportPreset,
    NativeWorkingMaster,
    RevisionRecipe,
    cohesion_diagnostics,
    create_revision,
    export_delivery,
    inspect_image,
    package_delivery,
)

pytestmark = pytest.mark.unit


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _npy(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _native(*, schema="tp.lux.graded_photograph.v2", alpha=False, icc=False):
    pixels = np.tile(np.array([-0.03, 0.000001, 1.8], dtype=np.float32), (6, 8, 1))
    alpha_pixels = np.full((6, 8), 1.0, dtype=np.float32) if alpha else None
    profile = output_srgb_icc() if icc else None
    original = ImageMaster(pixels, "a" * 64, 8, alpha_pixels, {"source_format": "JPEG", "input_color": "srgb"}, profile)
    descriptor = {"schema": schema, "input_id": "input-0000", "master": original.to_payload()}
    descriptor_bytes = canonicalize_json(descriptor)
    source = _npy(pixels)
    alpha_bytes = _npy(alpha_pixels) if alpha else None
    native = NativeWorkingMaster(
        descriptor_sha256=_sha(descriptor_bytes),
        descriptor_bytes=descriptor_bytes,
        alpha_sha256=_sha(alpha_bytes) if alpha else None,
        alpha_bytes=alpha_bytes,
        source_icc_sha256=_sha(profile) if profile else None,
        source_icc_bytes=profile,
    )
    image = DeliveryImage("master.npy", _sha(source), "Living room", source_bytes=source, native=native)
    return image, pixels, descriptor, original


@pytest.mark.parametrize(
    "schema", ["tp.lux.photograph.v2", "tp.lux.photograph.v3", "tp.lux.graded_photograph.v1", "tp.lux.graded_photograph.v2"]
)
def test_retained_v5_v6_native_working_samples_are_float_authority(schema: str) -> None:
    image, pixels, descriptor, original = _native(schema=schema, icc=True, alpha=True)
    master = image.decode()
    np.testing.assert_array_equal(master.pixels, pixels)
    assert master.source_bit_depth == 32
    assert master.source_icc is None  # Source ICC is provenance, never a tag on working linear samples.
    info = inspect_image(image)
    assert info["source_sha256"] == image.source_sha256
    assert info["bit_depth"] == 32 and info["original_source_bit_depth"] == 8
    assert info["pixel_sha256"] == _sha(pixels.tobytes())
    assert info["source_identity_sha256"] == _sha(canonicalize_json(image.source_bundle))
    assert info["retained_descriptor_content_sha256"] == original.content_hash()
    assert info["master_content_sha256"] == master.content_hash() != original.content_hash()
    assert info["original_source_icc_verified"] is True
    assert info["source_icc_sha256"] is None
    json.dumps(info)


def test_missing_original_icc_is_declared_provenance_without_manufactured_bytes() -> None:
    image, pixels, descriptor, original = _native(icc=True)
    descriptor_bytes = image.native.descriptor_bytes
    image = DeliveryImage(
        "master.npy",
        image.source_sha256,
        "Living room",
        source_bytes=image.source_bytes,
        native=NativeWorkingMaster(_sha(descriptor_bytes), descriptor_bytes=descriptor_bytes),
    )
    info = inspect_image(image)
    assert info["original_source_icc_sha256"] == _sha(output_srgb_icc())
    assert info["original_source_icc_verified"] is False
    assert image.decode().source_icc is None
    assert info["retained_descriptor_content_sha256"] == artifact_content_hash(
        descriptor["master"], {"pixels": pixels, "alpha": None}
    )
    assert info["master_content_sha256"] != original.content_hash()


def test_descriptor_and_alpha_components_change_identity_even_with_identical_npy() -> None:
    image, pixels, descriptor, _ = _native(alpha=True)
    descriptor["master"]["metadata"]["description"] = "Revised provenance"
    changed_descriptor = canonicalize_json(descriptor)
    alpha = _npy(np.full((6, 8), 0.5, dtype=np.float32))
    other = DeliveryImage(
        "master.npy",
        image.source_sha256,
        "Living room",
        source_bytes=image.source_bytes,
        native=NativeWorkingMaster(
            _sha(changed_descriptor), descriptor_bytes=changed_descriptor, alpha_sha256=_sha(alpha), alpha_bytes=alpha
        ),
    )
    assert other.source_sha256 == image.source_sha256
    assert other.source_identity_sha256 != image.source_identity_sha256
    assert inspect_image(other)["master_content_sha256"] != inspect_image(image)["master_content_sha256"]


@pytest.mark.parametrize(
    "damage",
    [
        "int16",
        "float64",
        "fortran",
        "trailing",
        "nonfinite",
        "shape",
        "alpha_missing",
        "alpha_range",
        "wrong_color",
        "wrong_schema",
        "icc_mismatch",
    ],
)
def test_native_contract_fails_closed_before_rendering(damage: str) -> None:
    image, pixels, descriptor, _ = _native(alpha=True)
    data, alpha, profile = image.source_bytes, image.native.alpha_bytes, None
    if damage in {"int16", "float64"}:
        data = _npy(pixels.astype(damage))
    elif damage == "fortran":
        data = _npy(np.asfortranarray(pixels))
    elif damage == "trailing":
        data += b"unselected trailing payload"
    elif damage == "nonfinite":
        pixels[0, 0, 0] = np.inf
        data = _npy(pixels)
    elif damage == "shape":
        descriptor["master"]["shape"] = [9, 9]
    elif damage == "alpha_missing":
        alpha = None
    elif damage == "alpha_range":
        alpha = _npy(np.full((6, 8), 1.1, dtype=np.float32))
    elif damage == "wrong_color":
        descriptor["master"]["color_space"] = "srgb"
    elif damage == "wrong_schema":
        descriptor["schema"] = "tp.lux.arbitrary.v1"
    elif damage == "icc_mismatch":
        profile = output_srgb_icc()
    descriptor_bytes = canonicalize_json(descriptor)
    changed = DeliveryImage(
        "master.npy",
        _sha(data),
        "Kitchen",
        source_bytes=data,
        native=NativeWorkingMaster(
            _sha(descriptor_bytes),
            descriptor_bytes=descriptor_bytes,
            alpha_sha256=_sha(alpha) if alpha else None,
            alpha_bytes=alpha,
            source_icc_sha256=_sha(profile) if profile else None,
            source_icc_bytes=profile,
        ),
    )
    with pytest.raises(ValueError):
        inspect_image(changed)


def test_native_path_components_revalidate_exact_hashes_without_symlink_access(tmp_path: Path) -> None:
    image, _, _, _ = _native(alpha=True)
    master, descriptor, alpha = (tmp_path / name for name in ("master.npy", "photograph.json", "alpha.npy"))
    master.write_bytes(image.source_bytes)
    descriptor.write_bytes(image.native.descriptor_bytes)
    alpha.write_bytes(image.native.alpha_bytes)
    lazy = DeliveryImage(
        "master.npy",
        image.source_sha256,
        "Kitchen",
        source_path=master,
        native=NativeWorkingMaster(
            image.native.descriptor_sha256,
            descriptor_path=descriptor,
            alpha_sha256=image.native.alpha_sha256,
            alpha_path=alpha,
        ),
    )
    assert inspect_image(lazy)["source_identity_sha256"] == image.source_identity_sha256
    descriptor.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        lazy.decode()
    descriptor.unlink()
    descriptor.symlink_to(alpha)
    with pytest.raises(ValueError, match="symlink"):
        lazy.decode()


def test_native_oversized_header_is_rejected_before_numpy_parses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    image, _, _, _ = _native()
    raw = b"\x93NUMPY\x02\x00" + (5000).to_bytes(4, "little") + b" " * 5000
    malformed = DeliveryImage("master.npy", _sha(raw), "Living room", source_bytes=raw, native=image.native)

    def must_not_parse(*args, **kwargs):
        raise AssertionError("Oversized untrusted headers must fail before NumPy parsing")

    monkeypatch.setattr(np.lib.format, "read_array_header_2_0", must_not_parse)
    with pytest.raises(ValueError, match="bounded NPY header"):
        malformed.decode()


def test_native_master_bundle_exports_exact_arrays_and_unclipped_usable_tiffs(tmp_path: Path) -> None:
    image, pixels, _, _ = _native(icc=True, alpha=True)
    root = tmp_path / "exports"
    inventory = export_delivery(
        [image],
        root,
        property_name="16 Seaview",
        export_presets=[ExportPreset.from_payload("photographic_master"), ExportPreset.from_payload("retouch_print")],
    )
    arrays = [row for row in inventory["files"] if row["path"].endswith(".npy")]
    assert len(arrays) == 1 and (root / arrays[0]["path"]).read_bytes() == image.source_bytes
    for row in inventory["files"]:
        assert row["source_identity_sha256"] == image.source_identity_sha256
        assert row["source_master_sha256"] == image.source_sha256
        assert row["source_bit_depth"] == row["bit_depth"] == 32
        assert row["precision_recovery"] is False and row["container_promotion"] is False
        if row["path"].endswith(".tif"):
            samples = tifffile.imread(root / row["path"])
            np.testing.assert_allclose(samples[..., :3], linear_to_srgb(pixels), rtol=0, atol=1e-7)
            assert samples.min() < 0 and samples.max() > 1
    components = {row["component"]: (root / row["path"]).read_bytes() for row in inventory["native_components"]}
    assert components == image.native.component_snapshots()
    retained = DeliveryImage(
        "master.npy",
        arrays[0]["sha256"],
        "Living room",
        source_path=root / arrays[0]["path"],
        native=NativeWorkingMaster(
            descriptor_sha256=_sha(components["descriptor"]),
            descriptor_bytes=components["descriptor"],
            alpha_sha256=_sha(components["alpha"]),
            alpha_bytes=components["alpha"],
            source_icc_sha256=_sha(components["source_icc"]),
            source_icc_bytes=components["source_icc"],
        ),
    )
    assert retained.source_identity_sha256 == image.source_identity_sha256
    np.testing.assert_array_equal(retained.decode().pixels, pixels)
    publish = tmp_path / "publish"
    publish.mkdir()
    receipt = package_delivery(root, publish / "native.zip")
    assert receipt["status"] == "verified"
    assert receipt["source_file_count"] == len(inventory["files"]) + len(components) + 3


def test_native_revision_and_cohesion_bind_full_component_identity() -> None:
    image, pixels, _, _ = _native()
    result = create_revision(image, RevisionRecipe(exposure_stops=-0.1, reference_sha256=image.source_identity_sha256))
    expected = pixels * np.float32(2**-0.1)
    rendered = tifffile.imread(io.BytesIO(result["source_bytes"]))
    np.testing.assert_allclose(srgb_to_linear(rendered), expected, rtol=3e-6, atol=1e-8)
    assert rendered.min() < 0 and rendered.max() > 1
    receipt = result["receipt"]
    assert receipt["source_identity_sha256"] == image.source_identity_sha256
    assert receipt["source_master_sha256"] == image.source_sha256
    assert receipt["source_bit_depth"] == 32 and receipt["approval_transferred"] is False
    diagnostics = cohesion_diagnostics(
        [image],
        [{"name": "Interior", "reference_sha256": image.source_identity_sha256, "members": [image.source_identity_sha256]}],
    )
    row = diagnostics["groups"][0]["images"][0]
    assert row["source_identity_sha256"] == image.source_identity_sha256
    assert row["source_master_sha256"] == image.source_sha256
