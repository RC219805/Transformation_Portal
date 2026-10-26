"""Color preparation is execution authority, separate from ICC label retention."""

from __future__ import annotations

import hashlib
import io
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import tifffile
from PIL import Image, PngImagePlugin

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.config_resolver import build_materials_fingerprint_payload, compute_config_fingerprint
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution, runtime_config_from_execution_plan
from transformation_portal.lux_depth_v3.v2_enhance import (
    V2EnhancementError,
    decode_v2_source,
    enhance_image,
    prepare_materials_srgb_proxy,
)
from transformation_portal.lux_depth_v3.v2_presets import V2EnhancementConfig
from transformation_portal.lux_depth_v4.photography import linear_to_srgb, output_srgb_icc

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


@pytest.mark.parametrize("suffix", ["webp", "bmp"])
@pytest.mark.parametrize("policy", ["srgb", "auto_assume_srgb"])
def test_legacy_formats_use_lossless_color_carrier_with_explicit_encoding(tmp_path: Path, suffix: str, policy: str) -> None:
    source = tmp_path / f"legacy.{suffix}"
    pixels = np.full((8, 10, 3), [64, 96, 128], dtype=np.uint8)
    Image.fromarray(pixels).save(source, lossless=True)
    before = source.read_bytes()
    result = enhance_image(
        source,
        tmp_path / "out.tif",
        input_color=policy,
        output_bit_depth=16,
        config=V2EnhancementConfig.from_preset("none"),
    )
    np.testing.assert_array_equal(tifffile.imread(result["output"]), pixels.astype(np.uint16) * 257)
    master = decode_v2_source(before, source_name=source.name, input_color=policy)
    assert master.source_sha256 == hashlib.sha256(before).hexdigest()
    assert master.metadata["source_format"] == suffix.upper()
    assert master.metadata["source_normalization"]["pixel_quantization"] is False
    assert source.read_bytes() == before
    with pytest.raises(V2EnhancementError, match="Ambiguous"):
        enhance_image(source, tmp_path / "blocked.png", input_color="auto", output_bit_depth=8)


def _bmp_with_profile(profile: bytes, *, color_space: int = 0x4D424544) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), (80, 100, 120)).save(stream, format="BMP")
    ordinary = stream.getvalue()
    header = bytearray(ordinary[:54] + b"\0" * 84)
    struct.pack_into("<I", header, 14, 124)
    struct.pack_into("<I", header, 2, len(ordinary) + 84 + len(profile))
    struct.pack_into("<I", header, 10, 138)
    struct.pack_into("<I", header, 70, color_space)
    struct.pack_into("<II", header, 126, len(ordinary) + 84 - 14, len(profile))
    return bytes(header) + ordinary[54:] + profile


@pytest.mark.parametrize("suffix", ["webp", "bmp"])
def test_legacy_embedded_icc_is_converted_and_unknown_profiles_fail_closed(tmp_path: Path, suffix: str) -> None:
    cms = pytest.importorskip("imagecodecs")
    if not cms.CMS.available:
        pytest.skip("Native CMS runtime unavailable")
    profile = cms.cms_profile("adobergb")
    source = tmp_path / f"profile.{suffix}"
    if suffix == "webp":
        Image.new("RGB", (8, 8), (80, 100, 120)).save(source, lossless=True, icc_profile=profile)
    else:
        source.write_bytes(_bmp_with_profile(profile))
    result = enhance_image(
        source, tmp_path / "converted.tif", output_bit_depth=16, config=V2EnhancementConfig.from_preset("none")
    )
    expected = linear_to_srgb(decode_v2_source(source.read_bytes(), source_name=source.name).pixels)
    actual = tifffile.imread(result["output"]).astype(np.float32) / 65535
    np.testing.assert_allclose(actual, expected, atol=1 / 65535, rtol=0)
    assert result["color"]["preparation"]["source_color"] == "icc_rgb"
    if suffix == "webp":
        Image.new("RGB", (8, 8)).save(source, lossless=True, icc_profile=b"unknown profile")
    else:
        source.write_bytes(_bmp_with_profile(b"unknown profile"))
    with pytest.raises(V2EnhancementError, match="ICC|profile"):
        enhance_image(source, tmp_path / "bad.tif", output_bit_depth=16, input_color="auto_assume_srgb")


def test_bmp_calibration_is_not_discarded_by_an_assumption(tmp_path: Path) -> None:
    source = tmp_path / "unknown.bmp"
    source.write_bytes(_bmp_with_profile(b"", color_space=0))
    with pytest.raises(V2EnhancementError, match="BMP color metadata"):
        enhance_image(source, tmp_path / "bad.png", output_bit_depth=8, input_color="auto_assume_srgb")


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("kind", ["srgb", "linear", "gray_icc", "calibrated_png"])
def test_actual_materials_writer_to_v2_uses_validated_srgb(tmp_path: Path, bits: int, kind: str) -> None:
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    samples = np.full((8, 10, 3), 96 / 255, dtype=np.float32)
    source = tmp_path / "source.png"
    policy = "auto"
    if kind == "gray_icc":
        cms = pytest.importorskip("imagecodecs")
        if not cms.CMS.available:
            pytest.skip("Native CMS runtime unavailable")
        Image.new("L", (10, 8), 96).save(source, icc_profile=cms.cms_profile("gray", gamma=2.2))
    elif kind == "calibrated_png":
        cms = pytest.importorskip("imagecodecs")
        if not cms.CMS.available:
            pytest.skip("Native CMS runtime unavailable")
        chunks = PngImagePlugin.PngInfo()
        chunks.add(b"gAMA", struct.pack(">I", 50000))
        chunks.add(b"cHRM", struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000))
        Image.new("RGB", (10, 8), (96, 96, 96)).save(source, pnginfo=chunks)
    elif kind == "linear":
        Image.new("RGB", (10, 8), (96, 96, 96)).save(source)
        policy = "linear_srgb"
    else:
        Image.new("RGB", (10, 8), (96, 96, 96)).save(source, icc_profile=output_srgb_icc())
    original_proxy = samples.copy()
    expected, _ = prepare_materials_srgb_proxy(
        samples, source=source.read_bytes(), source_name=source.name, input_color=policy
    )
    # Validate against full precision source color authority independently of the
    # existing proxy conversion, including a grayscale profile and calibrated PNG.
    source_master = decode_v2_source(source.read_bytes(), source_name=source.name, input_color=policy)
    np.testing.assert_allclose(expected, linear_to_srgb(source_master.pixels), atol=2e-6, rtol=0)
    engine_inputs = []

    def process(**kwargs):
        engine_inputs.append(kwargs["image"].copy())
        return {"enhanced_image": kwargs["image"], "materials_v3_metadata": {"segmentation_metadata": {}}}

    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator.config = EnhanceConfig(enable_materials_v3=True, output_bit_depth=bits, v2_input_color=policy)
    orchestrator.output_root = tmp_path / "artifacts"
    orchestrator.materials_v3_engine = SimpleNamespace(process=process)
    with (
        patch("transformation_portal.lux_depth_v3.segmentation_backend.segment_materials", return_value={}) as segment,
        patch(
            "transformation_portal.lux_depth_v3.segmentation_backend.get_last_segmentation_runtime_metadata", return_value={}
        ),
    ):
        result, _, handoff = orchestrator._run_materials_v3_stage(
            preprocessed_array=samples,
            source_path=source,
            depth_map=np.zeros((8, 10), np.float32),
            output_key=Path("scene"),
        )
    assert handoff is not None and result is not None
    np.testing.assert_array_equal(samples, original_proxy)  # Depth/model preprocessing was not changed.
    np.testing.assert_array_equal(segment.call_args.args[0], (original_proxy * 255).astype(np.uint8))
    np.testing.assert_allclose(engine_inputs[0], expected, atol=0, rtol=0)
    preparation = result["materials_v3_metadata"]["segmentation_metadata"]["color_preparation"]
    assert preparation["geometry"] == [8, 10]
    assert preparation["output_color"] == "bounded_encoded_srgb"
    assert preparation["depth_preprocessing_changed"] is False
    assert preparation["segmentation_preprocessing_changed"] is False
    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    runner_policies = []

    def run_v2(**kwargs):
        runner_policies.append(kwargs["input_color"])
        return enhance_image(
            kwargs["input_path"],
            kwargs["output_dir"] / "final.tif",
            output_bit_depth=kwargs["output_bit_depth"],
            input_color=kwargs["input_color"],
            config=V2EnhancementConfig.from_preset("none"),
        )

    orchestrator.v2_runner = SimpleNamespace(run=run_v2)
    orchestrator.v2_dir = tmp_path / "v2"
    orchestrator.depth_dir = tmp_path / "depth"
    orchestrator.should_skip_v2 = lambda *args, **kwargs: False
    receipt, _, _ = orchestrator._run_v2_stage(
        image_input=ImageInput(path=handoff),
        depth_path=None,
        output_key=Path("scene"),
        v2_log_path=tmp_path / "log.txt",
        manifest_path=tmp_path / "manifest.json",
        skip_depth=False,
        materials_v3_result=result,
        materials_input_path=handoff,
    )
    assert runner_policies == ["auto"]  # Also applies to an original linear-sRGB source.
    maximum = 65535 if bits == 16 else 255
    output = tifffile.imread(receipt["output"]) if bits == 16 else np.asarray(Image.open(receipt["output"]))
    np.testing.assert_allclose(output.astype(np.float32) / maximum, expected, atol=1.1 / maximum, rtol=0)


@pytest.mark.parametrize("kind", ["ambiguous", "bad_icc", "wide_gamut", "hdr"])
def test_materials_color_does_not_launder_unknown_or_extended_source(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source.tif"
    samples = np.full((8, 10, 3), 0.4, dtype=np.float32)
    if kind == "hdr":
        tifffile.imwrite(source, np.full((8, 10, 3), 1.4, np.float32), photometric="rgb", iccprofile=output_srgb_icc())
    elif kind == "wide_gamut":
        cms = pytest.importorskip("imagecodecs")
        if not cms.CMS.available:
            pytest.skip("Native CMS runtime unavailable")
        tifffile.imwrite(
            source, np.full((8, 10, 3), [65535, 0, 0], np.uint16), photometric="rgb", iccprofile=cms.cms_profile("adobergb")
        )
    else:
        tifffile.imwrite(
            source,
            np.full((8, 10, 3), 20000, np.uint16),
            photometric="rgb",
            iccprofile=b"unknown" if kind == "bad_icc" else None,
        )
    with pytest.raises((ValueError, V2EnhancementError)):
        prepare_materials_srgb_proxy(
            samples,
            source=source.read_bytes(),
            source_name=source.name,
            input_color="auto_assume_srgb" if kind == "bad_icc" else "auto",
        )


def test_raw_materials_proxy_requires_established_linear_contract_without_preview() -> None:
    proxy = np.full((8, 10, 3), 0.2, np.float32)
    contract = {"contract": "legacy_linear_srgb", "mode": "auto", "preview_escape_enabled": False}
    encoded, receipt = prepare_materials_srgb_proxy(
        proxy, source=None, source_name="source.cr2", input_color="auto", raw_contract=contract
    )
    np.testing.assert_allclose(encoded, linear_to_srgb(proxy), atol=0, rtol=0)
    assert receipt["source_contract"] == contract
    with pytest.raises(V2EnhancementError, match="preview is depth-only"):
        prepare_materials_srgb_proxy(
            proxy,
            source=None,
            source_name="source.cr2",
            input_color="auto",
            raw_contract={**contract, "preview_escape_enabled": True},
        )


def test_materials_plan_and_cache_bind_source_color_even_without_v2(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "scene.jpg"
    source.write_bytes(b"No model or pixel decoding during planning")
    settings = dict(depth_backend="synthetic", allow_synthetic_fallback=True, enable_materials_v3=True, enable_v2=False)
    auto = EnhanceConfig(**settings, v2_input_color="auto")
    asserted = EnhanceConfig(**settings, v2_input_color="srgb")
    first = prepare_lux_execution(auto, root, [source])
    second = prepare_lux_execution(asserted, root, [source])
    material = next(node for node in second.plan.nodes if node.stage_registry_id.value == "tp.stage.lux.materials_v3.v1")
    assert material.configuration["schema"] == "tp.stage.config.lux.materials_v3.v2"
    assert material.configuration["input_color"] == "srgb"
    assert material.configuration["color_contract"] == "encoded_srgb_v1"
    assert material.configuration["processing_version"] == "1.2.0"
    assert runtime_config_from_execution_plan(second.plan).v2_input_color == "srgb"
    assert first.plan.plan_fingerprint_sha256 != second.plan.plan_fingerprint_sha256
    assert build_materials_fingerprint_payload(auto) != build_materials_fingerprint_payload(asserted)
    # Legacy Stage A reuse includes Materials artifacts, so it must invalidate
    # when their pixels change even though native depth inference is unchanged.
    assert (
        compute_config_fingerprint(auto).depth_only().to_sha256()
        != compute_config_fingerprint(asserted).depth_only().to_sha256()
    )


@pytest.mark.parametrize("mismatch", ["source_policy", "handoff"])
def test_materials_v2_plan_cannot_silently_project_inconsistent_color(mismatch: str, tmp_path: Path) -> None:
    from transformation_portal.core.execution_plan import CanonicalExecutionPlan, with_execution_plan_fingerprint
    from transformation_portal.lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError

    source = tmp_path / "scene.jpg"
    source.write_bytes(b"Planning only")
    prepared = prepare_lux_execution(
        EnhanceConfig(depth_backend="synthetic", allow_synthetic_fallback=True, enable_materials_v3=True, enable_v2=True),
        tmp_path,
        [source],
    )
    payload = prepared.plan.to_payload()
    node = next(node for node in payload["nodes"] if node["stage_registry_id"] == "tp.stage.lux.v2.v1")
    node["configuration"]["input_color" if mismatch == "source_policy" else "materials_color_handoff"] = (
        "srgb" if mismatch == "source_policy" else "none"
    )
    altered = CanonicalExecutionPlan.from_payload(with_execution_plan_fingerprint(payload))
    with pytest.raises(LuxExecutionPlanAuthorityError, match="Materials"):
        runtime_config_from_execution_plan(altered)
