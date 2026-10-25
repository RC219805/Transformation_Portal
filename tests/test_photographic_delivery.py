from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from transformation_portal.core.photographic_export import jpeg_bytes, resize_long_edge, unsharp_mask
from transformation_portal.lux_depth_v4.photography import output_srgb_icc
from transformation_portal.photographic_delivery import (
    DeliveryImage,
    ExportPreset,
    RevisionRecipe,
    cohesion_diagnostics,
    create_revision,
    destination_presets,
    export_delivery,
    inspect_image,
    package_delivery,
)

pytestmark = pytest.mark.unit


def _tiff(pixels: np.ndarray) -> bytes:
    stream = io.BytesIO()
    tifffile.imwrite(stream, pixels, photometric="rgb", metadata={"color_space": "srgb"}, iccprofile=output_srgb_icc())
    return stream.getvalue()


def _image(pixels: np.ndarray | None = None, *, scene: str = "Living room", **kwargs) -> DeliveryImage:
    if pixels is None:
        pixels = np.full((18, 24, 3), 18000, dtype=np.uint16)
    data = _tiff(pixels)
    return DeliveryImage("delivery.tif", hashlib.sha256(data).hexdigest(), scene, source_bytes=data, **kwargs)


def test_explicit_ingestion_preserves_near_black_integer_precision_and_reports_color() -> None:
    image = _image(np.ones((4, 5, 3), dtype=np.uint16))
    master = image.decode()
    assert float(master.pixels.max()) < 0.000002
    info = inspect_image(image)
    assert (info["width"], info["height"], info["bit_depth"]) == (5, 4, 16)
    assert info["source_icc_sha256"] == hashlib.sha256(output_srgb_icc()).hexdigest()
    assert info["working_color"] == "extended_linear_srgb"
    json.dumps(info)


def test_path_backed_source_revalidates_hash_and_does_not_accept_links(tmp_path: Path) -> None:
    original = _image()
    source = tmp_path / "master.tif"
    source.write_bytes(original.source_bytes)
    image = DeliveryImage("master.tif", original.source_sha256, "Kitchen", source_path=source)
    assert image.snapshot() == original.source_bytes
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        image.snapshot()
    source.unlink()
    source.symlink_to(tmp_path / "missing.tif")
    with pytest.raises(ValueError, match="symlink"):
        image.snapshot()


@pytest.mark.parametrize("name", [preset.name for preset in destination_presets()])
def test_named_presets_are_explicit_and_round_trip_constraints(name: str) -> None:
    preset = ExportPreset.from_payload(name)
    assert preset.name == name
    assert preset.to_payload()["intended_use"]
    assert ExportPreset.from_payload({"name": name}) == preset


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "website", "long_edge": 0},
        {"name": "website", "long_edge": True},
        {"name": "mls", "jpeg_quality": 101},
        {"name": "website", "sharpen_amount": 2},
        {"name": "photographic_master", "long_edge": 100},
        {"name": "retouch_print", "sharpen_amount": 0.1},
        {"name": "website", "extra": "ignored?"},
        {"name": "universal_mls"},
    ],
)
def test_invalid_destination_overrides_fail_closed(payload) -> None:
    with pytest.raises(ValueError):
        ExportPreset.from_payload(payload)


def test_named_exports_preserve_original_master_and_link_every_derivative(tmp_path: Path) -> None:
    image = _image(metadata={"title": "Living room", "creator": "Property photographer", "copyright": "Local owner"})
    output = tmp_path / "exports"
    inventory = export_delivery([image], output, property_name="16 Seaview", export_presets=destination_presets())
    assert len(inventory["files"]) == 6
    assert inventory["sources"][0]["source_master_sha256"] == image.source_sha256
    for row in inventory["files"]:
        destination = output / row["path"]
        assert "16-seaview-living-room-v1-" in destination.name
        assert row["source_master_sha256"] == image.source_sha256
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == row["sha256"]
        assert row["recipe_sha256"]
        assert row["approval_transferred"] is False
        assert row["precision_recovery"] is False
        assert (row["width"], row["height"]) == (24, 18)
        if row["preset"] == "photographic_master":
            assert destination.read_bytes() == image.source_bytes
            assert row["bit_depth"] == 16
        elif row["preset"] == "retouch_print":
            assert row["container_promotion"] is True
            assert tifffile.imread(destination).dtype == np.float32
            decoded = DeliveryImage(destination.name, row["sha256"], "retouch", source_path=destination).decode()
            np.testing.assert_allclose(decoded.pixels, image.decode().pixels, atol=1e-7)
        elif row["preset"] == "print":
            assert tifffile.imread(destination).dtype == np.uint16
        else:
            with Image.open(destination) as exported:
                assert exported.info["icc_profile"] == output_srgb_icc()
                assert exported.getexif()[315] == "Property photographer"
                assert b"Local owner" in exported.info["xmp"]
    assert (output / "contact-sheet.jpg").is_file()
    assert inventory["contact_sheet"]["sources"][0]["source_master_sha256"] == image.source_sha256
    assert "float32 extended retouch" in (output / "inventory.txt").read_text()
    package_spec = json.loads((output / "package-spec.json").read_bytes())
    assert {row["source"] for row in package_spec["files"]} == {
        *[row["path"] for row in inventory["files"]],
        "contact-sheet.jpg",
        "inventory.json",
        "inventory.txt",
    }


def test_website_resize_is_configurable_and_never_upscales(tmp_path: Path) -> None:
    image = _image(np.full((60, 100, 3), 50000, dtype=np.uint16))
    inventory = export_delivery(
        [image],
        tmp_path / "small",
        property_name="Property",
        export_presets=[ExportPreset.from_payload({"name": "website", "long_edge": 40})],
    )
    row = inventory["files"][0]
    assert (row["width"], row["height"]) == (40, 24)
    inventory = export_delivery(
        [image],
        tmp_path / "native",
        property_name="Property",
        export_presets=[ExportPreset.from_payload({"name": "website", "long_edge": 200})],
    )
    assert inventory["files"][0]["width"] == 100


def test_contact_sheet_records_the_requested_sdr_render_and_exact_master_link(tmp_path: Path) -> None:
    image = _image()
    inventory = export_delivery(
        [image],
        tmp_path / "explicit-render",
        property_name="House",
        export_presets=[ExportPreset.from_payload({"name": "website", "render": "clip_srgb"})],
    )
    contact_source = inventory["contact_sheet"]["sources"][0]
    assert contact_source["source_master_sha256"] == image.source_sha256
    assert contact_source["derivative_sha256"] == inventory["files"][0]["sha256"]
    assert contact_source["render"]["recipe"]["mode"] == "clip_srgb"
    assert inventory["contact_sheet"]["recipe"]["scope"].startswith("Bounded contact preview")


def test_extended_master_and_retouch_do_not_silently_clip_or_reduce_precision(tmp_path: Path) -> None:
    pixels = np.array([[[-0.1, 0.00125, 2.5], [0.2, 0.3, 0.4]]], dtype=np.float32)
    image = _image(pixels)
    inventory = export_delivery(
        [image],
        tmp_path / "extended",
        property_name="Property",
        export_presets=[ExportPreset.from_payload("photographic_master"), ExportPreset.from_payload("retouch_print")],
    )
    row = next(row for row in inventory["files"] if row["preset"] == "retouch_print")
    stored = tifffile.imread(tmp_path / "extended" / row["path"])
    assert row["container_promotion"] is False
    assert stored.dtype == np.float32
    np.testing.assert_allclose(stored, pixels, rtol=1e-6, atol=1e-7)
    assert stored.min() < 0 and stored.max() > 1


def test_export_cannot_replace_existing_delivery_or_follow_output_symlink(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    protected = existing / "finalized.tif"
    protected.write_bytes(b"protected")
    with pytest.raises(ValueError, match="fresh or empty"):
        export_delivery([_image()], existing, property_name="House", export_presets=destination_presets())
    assert protected.read_bytes() == b"protected"
    link = tmp_path / "link"
    link.symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        export_delivery([_image()], link, property_name="House", export_presets=destination_presets())


def test_verified_packager_integration_and_copy_validate_all_members(tmp_path: Path) -> None:
    output = tmp_path / "export"
    export_delivery([_image()], output, property_name="House", export_presets=[ExportPreset.from_payload("website")])
    destination = tmp_path / "packages"
    destination.mkdir()
    copy = tmp_path / "copy"
    copy.mkdir()
    archive_path = destination / "house.zip"
    receipt = package_delivery(output, archive_path, copy_to=copy)
    assert receipt["status"] == "verified"
    assert receipt["packager_source_sha256"]
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("house-photographic-delivery/_package/manifest.json"))
        for item in manifest["files"]:
            payload = archive.read(f"house-photographic-delivery/{item['archive_path']}")
            assert hashlib.sha256(payload).hexdigest() == item["sha256"]
            assert len(payload) == item["size_bytes"]
    assert (copy / "house.zip").read_bytes() == archive_path.read_bytes()
    for suffix in (".sha256", ".receipt.json"):
        assert (copy / f"house.zip{suffix}").read_bytes() == Path(f"{archive_path}{suffix}").read_bytes()


def test_changed_derivative_prevents_package_publication(tmp_path: Path) -> None:
    output = tmp_path / "export"
    inventory = export_delivery(
        [_image()], output, property_name="House", export_presets=[ExportPreset.from_payload("website")]
    )
    (output / inventory["files"][0]["path"]).write_bytes(b"changed after export")
    packages = tmp_path / "packages"
    packages.mkdir()
    with pytest.raises(ValueError, match="Verified packaging failed"):
        package_delivery(output, packages / "must-not-publish.zip")
    assert list(packages.iterdir()) == []


def test_explicit_revision_is_precise_hash_bound_and_unapproved() -> None:
    image = _image(np.full((16, 24, 3), 20000, dtype=np.uint16))
    reference = "a" * 64
    recipe = RevisionRecipe.from_payload({"exposure_stops": -0.10, "reference_sha256": reference})
    result = create_revision(image, recipe)
    receipt = result["receipt"]
    assert receipt["source_master_sha256"] == image.source_sha256
    assert receipt["revision_sha256"] == hashlib.sha256(result["source_bytes"]).hexdigest()
    assert receipt["reference_sha256"] == reference
    assert receipt["approval_transferred"] is False
    assert receipt["review_status"] == "unreviewed"
    assert receipt["output_bit_depth"] == 32
    revised = DeliveryImage(
        "revision.tif", receipt["revision_sha256"], "Living room", source_bytes=result["source_bytes"]
    ).decode()
    np.testing.assert_allclose(revised.pixels, image.decode().pixels * np.exp2(-0.1), rtol=2e-6)
    with Image.open(io.BytesIO(result["preview_bytes"])) as preview:
        assert preview.format == "PNG"
    assert create_revision(image, recipe)["source_bytes"] == result["source_bytes"]


def test_material_revision_affects_only_explicit_bounded_region_and_records_recipe() -> None:
    image = _image(np.full((80, 80, 3), 20000, dtype=np.uint16))
    recipe = RevisionRecipe.from_payload(
        {"materials": [{"name": "plaster", "region": [0.2, 0.2, 0.4, 0.4], "exposure_stops": 0.25}]}
    )
    result = create_revision(image, recipe)
    revised = DeliveryImage(
        "revision.tif", result["receipt"]["revision_sha256"], "Room", source_bytes=result["source_bytes"]
    ).decode()
    baseline = image.decode().pixels
    assert np.all(revised.pixels[24, 24] > baseline[24, 24])
    np.testing.assert_allclose(revised.pixels[-1, -1], baseline[-1, -1], atol=1e-7)
    assert result["receipt"]["recipe"]["materials"][0]["name"] == "plaster"


@pytest.mark.parametrize(
    "payload",
    [
        {"exposure_stops": 2},
        {"exposure_stops": True},
        {"white_balance": [1, 1, 2]},
        {"saturation": float("nan")},
        {"automatic_matching": True},
        {"unknown": 1},
        {"materials": [{"name": "wall", "region": [0.5, 0, 0.1, 1]}]},
        {"materials": [{"name": "wall", "region": [0, 0, 1, 1], "white_balance": [1, 1, 1.5]}]},
    ],
)
def test_unbounded_or_implicit_revision_rejected(payload) -> None:
    with pytest.raises(ValueError):
        RevisionRecipe.from_payload(payload)


def test_cohesion_flags_outliers_without_adjusting_or_neutralizing_twilight() -> None:
    reference = _image(np.full((8, 12, 3), 12000, dtype=np.uint16), scene="Warm room")
    outlier = _image(np.full((8, 12, 3), 45000, dtype=np.uint16), scene="Bright room")
    group = {
        "name": "Warm interior",
        "reference_sha256": reference.source_sha256,
        "members": [reference.source_sha256, outlier.source_sha256],
    }
    before = outlier.snapshot()
    result = cohesion_diagnostics([reference, outlier], [group])
    assert result["automatic_adjustment"] is False
    assert "exposure_outlier" in result["groups"][0]["images"][1]["flags"]
    assert result["groups"][0]["images"][0]["flags"] == []
    assert outlier.snapshot() == before
    with pytest.raises(ValueError, match="Every photograph"):
        cohesion_diagnostics([reference, outlier], [{**group, "members": [reference.source_sha256]}])


def test_shared_editorial_primitives_keep_precision_and_require_explicit_sdr() -> None:
    samples = np.linspace(0.3, 0.302, 80 * 100 * 3, dtype=np.float32).reshape((80, 100, 3))
    resized = resize_long_edge(samples, 40)
    assert resized.dtype == np.float32 and len(np.unique(resized)) > 256
    sharpened = unsharp_mask(resized)
    assert sharpened.dtype == np.float32
    with pytest.raises(ValueError, match="explicit bounded SDR"):
        jpeg_bytes(np.full((1, 1, 3), 1.01, dtype=np.float32), icc_profile=output_srgb_icc())


def test_managed_exports_do_not_load_optional_editorial_or_model_runtimes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import transformation_portal.photographic_delivery; import sys; "
            "assert not {'rawpy', 'reportlab', 'torch'} & set(sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
