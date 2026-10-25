"""Color preparation is execution authority, separate from ICC label retention."""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.config_resolver import compute_config_fingerprint
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution, runtime_config_from_execution_plan
from transformation_portal.lux_depth_v3.v2_enhance import V2EnhancementError, enhance_image
from transformation_portal.lux_depth_v3.v2_presets import V2EnhancementConfig
from transformation_portal.lux_depth_v4.photography import output_srgb_icc

pytestmark = pytest.mark.unit


def test_ambiguous_input_is_blocked_without_output_and_explicit_assumption_is_recorded(tmp_path: Path) -> None:
    source = tmp_path / "untagged.png"
    Image.fromarray(np.full((8, 8, 3), 64, dtype=np.uint8)).save(source)
    with pytest.raises(V2EnhancementError, match="Ambiguous input color"):
        enhance_image(source, tmp_path / "blocked.png", output_bit_depth=16)
    assert sorted(path.name for path in tmp_path.iterdir()) == [source.name]
    receipt = enhance_image(source, tmp_path / "explicit.tif", output_bit_depth=16, input_color="auto_assume_srgb")
    assert receipt["color"]["input_policy"] == "auto_assume_srgb"
    assert receipt["color"]["working_color"] == "bounded_encoded_srgb"
    assert receipt["color"]["preparation"]["resolution"] == "untagged_input_srgb_assumption"
    assert receipt["processing_version"] == "1.2.0"


def test_native_near_black_reencoding_quantizes_once_and_retains_uint16_levels(tmp_path: Path) -> None:
    source = tmp_path / "near-black.tif"
    pixels = np.arange(96, dtype=np.uint16).reshape(8, 4, 3)
    tifffile.imwrite(source, pixels, photometric="rgb", metadata={"color_space": "srgb"}, iccprofile=output_srgb_icc())
    receipt = enhance_image(
        source, tmp_path / "master.tif", output_bit_depth=16, config=V2EnhancementConfig.from_preset("none")
    )
    np.testing.assert_array_equal(tifffile.imread(receipt["output"]), pixels)
    assert receipt["color"]["output_icc_sha256"] == hashlib.sha256(output_srgb_icc()).hexdigest()


def test_valid_adobe_rgb_is_converted_not_relabelled(tmp_path: Path) -> None:
    imagecodecs = pytest.importorskip("imagecodecs")
    if not imagecodecs.CMS.available:
        pytest.skip("Native CMS runtime unavailable")
    source = tmp_path / "adobe-rgb.tif"
    profile = imagecodecs.cms_profile("adobergb")
    pixels = np.array([[[0.3, 0.4, 0.5], [0.5, 0.3, 0.2]]], dtype=np.float32)
    samples = np.rint(pixels * 65535).astype(np.uint16)
    tifffile.imwrite(source, samples, photometric="rgb", iccprofile=profile)
    expected = imagecodecs.cms_transform(
        samples.astype(np.float32) / 65535,
        profile,
        output_srgb_icc(),
        colorspace="rgb",
        planar=False,
        outcolorspace="rgb",
        outplanar=False,
        outdtype=np.float32,
        intent=imagecodecs.CMS.INTENT.RELATIVE_COLORIMETRIC,
        flags=imagecodecs.CMS.FLAGS.NOOPTIMIZE | imagecodecs.CMS.FLAGS.NOCACHE,
    )
    receipt = enhance_image(
        source, tmp_path / "converted.tif", output_bit_depth=16, config=V2EnhancementConfig.from_preset("none")
    )
    actual = tifffile.imread(receipt["output"]).astype(np.float32) / 65535
    np.testing.assert_allclose(actual, expected, atol=2 / 65535, rtol=0)
    assert not np.allclose(actual, pixels, atol=0.01)
    with tifffile.TiffFile(receipt["output"]) as output:
        assert output.pages[0].tags[34675].value == output_srgb_icc()
    assert receipt["color"]["source_icc_preserved"] is False


def test_wide_gamut_overrange_fails_closed_and_exact_copy_remains_unchanged(tmp_path: Path) -> None:
    imagecodecs = pytest.importorskip("imagecodecs")
    if not imagecodecs.CMS.available:
        pytest.skip("Native CMS runtime unavailable")
    source = tmp_path / "wide.tif"
    tifffile.imwrite(
        source,
        np.full((3, 4, 3), [65535, 0, 0], dtype=np.uint16),
        photometric="rgb",
        iccprofile=imagecodecs.cms_profile("adobergb"),
    )
    before = source.read_bytes()
    with pytest.raises(V2EnhancementError, match="Unified/V5/V6"):
        enhance_image(source, tmp_path / "blocked.tif", output_bit_depth=16)
    assert len(list(tmp_path.iterdir())) == 1
    copy = tmp_path / "exact.tif"
    receipt = enhance_image(source, copy, config=V2EnhancementConfig.from_preset("none"))
    assert receipt["status"] == "passthrough" and copy.read_bytes() == before


def test_palette_expansion_keeps_invalid_color_evidence_blocking(tmp_path: Path) -> None:
    source = tmp_path / "palette.png"
    Image.new("P", (8, 8)).save(source)
    encoded = source.read_bytes()
    # Explicitly unknown cICP transfer must not disappear during RGB expansion.
    payload = b"\xff\xff\xff\xff"
    chunk = struct.pack(">I", len(payload)) + b"cICP" + payload + struct.pack(">I", zlib.crc32(b"cICP" + payload))
    source.write_bytes(encoded[:33] + chunk + encoded[33:])
    with pytest.raises(V2EnhancementError, match="color|metadata"):
        enhance_image(source, tmp_path / "blocked.png", input_color="auto_assume_srgb")


def test_float_master_is_not_silently_reduced_to_v3_integer_output(tmp_path: Path) -> None:
    source = tmp_path / "float-master.tif"
    tifffile.imwrite(source, np.full((3, 4, 3), 0.251235, dtype=np.float32), photometric="rgb", iccprofile=output_srgb_icc())
    with pytest.raises(V2EnhancementError, match="Unified/V5/V6.*float precision"):
        enhance_image(source, tmp_path / "blocked.tif", output_bit_depth=16)
    assert len(list(tmp_path.iterdir())) == 1


def test_v2_plan_and_fingerprint_bind_explicit_color_and_processing_contract(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "scene.jpg"
    source.write_bytes(b"Planning does not decode model inputs")
    settings = dict(depth_backend="synthetic", allow_synthetic_fallback=True, enable_v2=True, enable_materials_v3=False)
    first = prepare_lux_execution(EnhanceConfig(**settings, v2_input_color="auto"), root, [source])
    second = prepare_lux_execution(EnhanceConfig(**settings, v2_input_color="auto_assume_srgb"), root, [source])
    node = next(node for node in second.plan.nodes if node.stage_registry_id.value == "tp.stage.lux.v2.v1")
    assert node.configuration["schema"] == "tp.stage.config.lux.v2.v2"
    assert node.configuration["input_color"] == "auto_assume_srgb"
    assert node.configuration["processing_version"] == "1.2.0"
    assert node.configuration["color_contract"] == "encoded_srgb_v1"
    assert first.plan.plan_fingerprint_sha256 != second.plan.plan_fingerprint_sha256
    assert runtime_config_from_execution_plan(second.plan).v2_input_color == "auto_assume_srgb"
    a = compute_config_fingerprint(EnhanceConfig(v2_input_color="auto"))
    b = compute_config_fingerprint(EnhanceConfig(v2_input_color="auto_assume_srgb"))
    assert a.v2_only().to_sha256() != b.v2_only().to_sha256()
    assert a.depth_only().to_sha256() == b.depth_only().to_sha256()
