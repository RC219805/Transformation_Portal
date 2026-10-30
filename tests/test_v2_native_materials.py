"""Native SDR Materials uses original precision across the V3/V2 boundary."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import tifffile

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3 import native_materials as native_module
from transformation_portal.lux_depth_v3 import v2_enhance as v2_module
from transformation_portal.lux_depth_v3.geometry import GeometryTransform
from transformation_portal.lux_depth_v3.geometry_handoff import load_geometry_handoff, write_geometry_handoff
from transformation_portal.lux_depth_v3.native_materials import (
    apply_native_materials,
    freeze_native_materials,
    native_materials_binding_hash,
)
from transformation_portal.lux_depth_v3.v2_enhance import V2EnhancementError, enhance_image, load_native_materials_binding
from transformation_portal.lux_depth_v3.v2_presets import V2EnhancementConfig, processing_version_for_geometry
from transformation_portal.lux_depth_v3.v2_runner import V2Runner
from transformation_portal.lux_depth_v4.photography import decode_master, linear_to_srgb
from transformation_portal.materials_v4.artifacts import write_evidence
from transformation_portal.materials_v4.contracts import MaterialEvidence, RegionEvidence
from transformation_portal.materials_v4.engine import MaterialOperation, ResponsePolicy
from transformation_portal.stage_graph.stage import StageContext
from transformation_portal.stage_graph.stages.enhancement import EnhancementStage

pytestmark = pytest.mark.unit


def _case(tmp_path, *, orientation=1, strength=0.001, alpha=True, full_ramp=False):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "native.tif"
    rows, columns = np.indices((256, 256) if full_ramp else (31, 45))
    ramp = np.arange(rows.size, dtype=np.uint16).reshape(rows.shape)
    rgb = (
        np.stack((ramp, np.flipud(ramp), np.fliplr(ramp)), axis=-1)
        if full_ramp
        else np.stack((20000 + ramp, 21000 + columns, 22000 + rows), axis=-1).astype(np.uint16)
    )
    samples = rgb
    if alpha:
        opacity = np.full(rows.shape, 65535, np.uint16)
        opacity[5:8, 13:17] = 0
        opacity[11:14, 21:26] = 32791
        samples = np.dstack((rgb, opacity))
    tifffile.imwrite(
        source,
        samples,
        photometric="rgb",
        extrasamples="unassalpha" if alpha else None,
        metadata=None,
        extratags=[(274, "H", 1, orientation, False)],
    )
    master = decode_master(source.read_bytes(), source_name=source.name, input_color="srgb")
    material_root = tmp_path / "materials"
    material_root.mkdir()
    mask = np.zeros(master.shape, np.float32)
    mask[3:-3, 4:-4] = 1
    mask[9:17, 8:19] = 0.75
    evidence = MaterialEvidence(
        master.source_sha256, master.shape, (RegionEvidence("water", "water", mask, semantic_confidence=0.99),)
    )
    evidence_path = material_root / "evidence.json"
    write_evidence(evidence, evidence_path)
    manifest = material_root / "manifest.json"
    manifest.write_bytes(
        canonicalize_json(
            {
                "schema": "tp.lux.materials_manifest.v1",
                "inputs": [
                    {
                        "path": source.name,
                        "source_sha256": master.source_sha256,
                        "evidence_path": evidence_path.name,
                        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                        "shape": list(master.shape),
                    }
                ],
            }
        )
    )
    policy = ResponsePolicy(operations=(MaterialOperation("water", "linear_gain_v1", strength),), min_coverage_px=1)
    policy_path = material_root / "policy.json"
    policy_path.write_bytes(canonicalize_json(policy.to_payload()))
    binding = freeze_native_materials(manifest, policy_path, inputs, [source], "srgb")
    binding_path = tmp_path / "native-binding.json"
    binding_path.write_bytes(canonicalize_json(binding))
    transform = GeometryTransform(master.source_sha256, rows.shape, orientation)
    depth = np.full(master.shape, 0.5, np.float32)
    depth[~transform.support()] = 0
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, depth, allow_pickle=False)
    handoff = tmp_path / "geometry.json"
    write_geometry_handoff(
        handoff,
        source_sha256=master.source_sha256,
        v2_input_path=source,
        plan_fingerprint_sha256=None,
        transform=transform,
        depth_map_path=depth_path,
        depth_normalization={"method": "controlled_fixture"},
    )
    return source, master, mask, binding, binding_path, transform, depth_path, handoff


def _run(case, tmp_path, **kwargs):
    source, _, _, _, binding_path, _, depth_path, handoff = case
    return enhance_image(
        source,
        tmp_path / "output.tif",
        config=V2EnhancementConfig.from_preset("none"),
        input_color="srgb",
        geometry_recipe="normalized_crop_support_v1",
        geometry_handoff_path=handoff,
        depth_map_path=depth_path,
        native_materials_binding_path=binding_path,
        native_materials_source=source.name,
        **kwargs,
    )


def _with_unselected_sources(case, *, count=1):
    source, master, _, binding, binding_path, _, _, _ = case
    material_root = source.parent.parent / "materials"
    manifest = material_root / "manifest.json"
    payload = json.loads(manifest.read_bytes())
    files = [source]
    for index in range(count):
        other = source.with_name(f"unselected-{index}.tif")
        other.write_bytes(source.read_bytes())
        files.append(other)
        evidence = MaterialEvidence(
            master.source_sha256,
            master.shape,
            (RegionEvidence("other", "water", np.full(master.shape, 1 / (index + 3), np.float32), semantic_confidence=0.99),),
        )
        evidence_path = material_root / f"unselected-{index}-evidence.json"
        write_evidence(evidence, evidence_path)
        payload["inputs"].append(
            {
                "path": other.name,
                "source_sha256": master.source_sha256,
                "evidence_path": evidence_path.name,
                "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                "shape": list(master.shape),
            }
        )
    manifest.write_bytes(canonicalize_json(payload))
    binding = freeze_native_materials(manifest, material_root / "policy.json", source.parent, files, "srgb")
    binding_path.write_bytes(canonicalize_json(binding))
    return (*case[:3], binding, *case[4:])


@pytest.mark.parametrize("unselected_count", [1, 4])
def test_per_image_native_loads_only_selected_bundle_at_loader_stage_and_cache(tmp_path, monkeypatch, unselected_count):
    case = _with_unselected_sources(_case(tmp_path), count=unselected_count)
    selected_path = case[0].name
    selected_evidence = case[3]["sources"][0]["evidence"]["path"]
    original_load = native_module.load_bound_evidence
    observed = []

    def selected_only(root, record, limits):
        observed.append(record["path"])
        assert record["path"] == selected_evidence, "Per-image processing decoded unrelated native masks"
        return original_load(root, record, limits)

    monkeypatch.setattr(native_module, "load_bound_evidence", selected_only)
    load_native_materials_binding(case[4], source_path=selected_path)
    assert observed == [selected_evidence]
    _run(case, tmp_path)
    assert observed == [selected_evidence] * 3
    context = _cache_context(case, case[3])
    stage = EnhancementStage(version="1.4.0", geometry_recipe="normalized_crop_support_v1")
    stage.get_cache_key(context)
    stage.compute(context)
    assert observed == [selected_evidence] * 6


def test_multisource_native_child_requires_selector_before_decoding_any_bundle(tmp_path, monkeypatch):
    case = _with_unselected_sources(_case(tmp_path))

    def forbidden(*_args, **_kwargs):
        pytest.fail("Ambiguous child selection reached a numeric mask bundle")

    monkeypatch.setattr(native_module, "load_bound_evidence", forbidden)
    with pytest.raises(V2EnhancementError, match="explicit source selector"):
        load_native_materials_binding(case[4])


def test_native_child_subprocess_never_decodes_unselected_bundle(tmp_path, monkeypatch):
    case = _with_unselected_sources(_case(tmp_path, strength=0))
    source, _, _, binding, binding_path, _, depth_path, handoff = case
    unselected_bundle = source.parent.parent / "materials" / binding["sources"][1]["evidence"]["path"]
    unselected_manifest = json.loads(unselected_bundle.read_bytes())
    unselected_mask = unselected_bundle.parent / unselected_manifest["regions"][0]["mask"]["path"]
    unselected_mask.write_bytes(b"Unselected mask must never be decoded by this child")
    runner = V2Runner()
    monkeypatch.setenv("PYTHONPATH", str(runner.repo_root / "src"))
    result = runner.run(
        input_path=source,
        depth_dir=None,
        output_dir=tmp_path / "selected-child",
        preset="none",
        input_color="srgb",
        geometry_recipe="normalized_crop_support_v1",
        geometry_handoff_path=handoff,
        depth_map_path=depth_path,
        native_materials_binding_path=binding_path,
        native_materials_source=source.name,
    )
    np.testing.assert_array_equal(tifffile.imread(result["output"]), tifffile.imread(source))
    assert result["native_materials"]["selected_source_path"] == source.name


@pytest.mark.parametrize("orientation", range(1, 9))
def test_native_sdr_edits_original_16bit_pixels_and_preserves_alpha_geometry(tmp_path, orientation):
    case = _case(tmp_path, orientation=orientation)
    source, master, mask, binding, _, transform, _, _ = case
    source_bytes = source.read_bytes()
    report = _run(case, tmp_path)
    actual = tifffile.imread(report["output"])
    allowed = (mask > 0) & transform.support() & (master.alpha > 0)
    expected = master.pixels.copy()
    # Independent native linear gain oracle; no proxy/model-resolution result.
    expected[allowed] = (
        master.pixels[allowed].astype(np.float64) + master.pixels[allowed].astype(np.float64) * 0.001 * mask[allowed, None]
    ).astype(np.float32)
    encoded = np.rint(np.clip(linear_to_srgb(expected), 0, 1) * 65535).astype(np.uint16)
    np.testing.assert_array_equal(actual[..., :3], encoded)
    np.testing.assert_array_equal(actual[..., 3], np.rint(master.alpha * 65535).astype(np.uint16))
    original = np.rint(linear_to_srgb(master.pixels) * 65535).astype(np.uint16)
    np.testing.assert_array_equal(actual[..., :3][~allowed], original[~allowed])
    assert np.any(actual[..., :3][allowed] != original[allowed])
    assert len(np.unique(actual[..., 0])) > 1000
    assert report["processing_version"] == report["config"]["version"] == "1.4.0"
    assert report["native_materials"]["binding_sha256"] == binding["binding_sha256"]
    assert report["native_materials"]["selected_source_path"] == source.name
    assert report["native_materials"]["precision_recovery"] is False
    assert report["native_materials"]["execution"]["protected_changed_pixels"] == 0
    assert source.read_bytes() == source_bytes


def test_zero_native_strength_preserves_every_16bit_code_without_proxy_quantization(tmp_path):
    case = _case(tmp_path, strength=0, full_ramp=True)
    original = tifffile.imread(case[0])
    report = _run(case, tmp_path)
    np.testing.assert_array_equal(tifffile.imread(report["output"]), original)
    assert report["native_materials"]["execution"]["changed_pixels"] == 0
    assert len(np.unique(original[..., 0])) == 65536


@pytest.mark.parametrize("mutation", ["same_size_bytes", "larger_file", "larger_geometry"])
def test_native_source_drift_fails_before_photographic_decode_or_output(tmp_path, monkeypatch, mutation):
    case = _case(tmp_path)
    source = case[0]
    original_size = source.stat().st_size
    if mutation == "same_size_bytes":
        data = bytearray(source.read_bytes())
        data[-1] ^= 1
        source.write_bytes(data)
        assert source.stat().st_size == original_size
    elif mutation == "larger_file":
        source.write_bytes(source.read_bytes() + b"unadmitted data")
        assert source.stat().st_size > original_size
    else:
        # Alter only the native dimensions in the TIFF header, keeping its
        # encoded extent unchanged. Neither new geometry nor bytes are admitted.
        with tifffile.TiffFile(source, mode="r+") as image:
            image.pages[0].tags[256].overwrite(4500)
            image.pages[0].tags[257].overwrite(3100)
        assert source.stat().st_size == original_size

    def forbidden(*_args, **_kwargs):
        pytest.fail("Changed source reached photographic pixel decoding before frozen byte verification")

    monkeypatch.setattr(v2_module, "decode_v2_source", forbidden)
    with pytest.raises((ValueError, V2EnhancementError), match="Native Materials"):
        _run(case, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.parametrize("full_ramp", [False, True])
def test_native_real_decoder_receives_exact_admitted_pixel_limit(tmp_path, monkeypatch, full_ramp):
    from transformation_portal.lux_depth_v4 import photography

    case = _case(tmp_path, strength=0, orientation=6, full_ramp=full_ramp)
    original_decoder = photography.decode_master
    limits = []

    def bounded_decoder(source, **kwargs):
        limits.append(kwargs["max_pixels"])
        return original_decoder(source, **kwargs)

    monkeypatch.setattr(photography, "decode_master", bounded_decoder)
    report = _run(case, tmp_path)
    assert limits == [int(np.prod(case[3]["sources"][0]["shape"]))]
    expected = np.rot90(tifffile.imread(case[0]), -1)
    np.testing.assert_array_equal(tifffile.imread(report["output"]), expected)


def test_native_stage_and_additional_v2_adjustments_preserve_protected_native_samples(tmp_path):
    case = _case(tmp_path)
    source, master, _, binding, binding_path, transform, depth_path, handoff = case
    native_baseline, _ = apply_native_materials(master, binding, source_path=source.name, support=transform.support())
    report = enhance_image(
        source,
        tmp_path / "enhanced.tif",
        input_color="srgb",
        geometry_recipe="normalized_crop_support_v1",
        geometry_handoff_path=handoff,
        depth_map_path=depth_path,
        native_materials_binding_path=binding_path,
        native_materials_source=source.name,
    )
    actual = tifffile.imread(report["output"])
    protected = ~transform.support() | (master.alpha != 1)
    expected = np.rint(linear_to_srgb(native_baseline.pixels) * 65535).astype(np.uint16)
    np.testing.assert_array_equal(actual[..., :3][protected], expected[protected])
    np.testing.assert_array_equal(actual[..., 3], np.rint(master.alpha * 65535).astype(np.uint16))
    assert report["stage_metadata"]["has_depth"] is True
    assert report["enhancement_metadata"]["material_masks_supplied"] is False


def test_native_runner_roundtrip_retains_exact_binding_and_selected_source(tmp_path, monkeypatch):
    case = _case(tmp_path, strength=0)
    source, _, _, binding, binding_path, _, depth_path, handoff = case
    runner = V2Runner()
    # A reused development interpreter may retain another checkout's editable
    # install. The controlled child must import the exact source under test.
    monkeypatch.setenv("PYTHONPATH", str(runner.repo_root / "src"))
    result = runner.run(
        input_path=source,
        depth_dir=None,
        output_dir=tmp_path / "child-output",
        preset="none",
        asset_key="native",
        input_color="srgb",
        geometry_recipe="normalized_crop_support_v1",
        geometry_handoff_path=handoff,
        depth_map_path=depth_path,
        native_materials_binding_path=binding_path,
        native_materials_source=source.name,
    )
    assert result["status"] == "success" and result["processing_version"] == "1.4.0"
    assert result["native_materials"]["binding_sha256"] == binding["binding_sha256"]
    assert result["native_materials"]["selected_source_path"] == source.name
    np.testing.assert_array_equal(tifffile.imread(result["output"]), tifffile.imread(source))


@pytest.mark.parametrize("mutation", ["recipe", "digest", "unknown", "duplicate_field", "symlink", "legacy_geometry", "masks"])
def test_invalid_native_authority_fails_before_output(tmp_path, mutation):
    case = _case(tmp_path)
    binding_path = case[4]
    if mutation in {"recipe", "digest", "unknown"}:
        payload = copy.deepcopy(case[3])
        if mutation == "recipe":
            payload["recipe"] = "unreviewed"
        elif mutation == "digest":
            payload["binding_sha256"] = "f" * 64
        else:
            payload["trust_me"] = True
        binding_path.write_bytes(canonicalize_json(payload))
    elif mutation == "duplicate_field":
        binding_path.write_text('{"recipe":"native_sdr_v1","recipe":"native_sdr_v1"}')
    elif mutation == "symlink":
        target = binding_path.with_name("target.json")
        binding_path.rename(target)
        binding_path.symlink_to(target)
    kwargs = {"masks_file": tmp_path / "unused.npz"} if mutation == "masks" else {}
    if mutation == "legacy_geometry":
        with pytest.raises(ValueError, match="Native Materials requires"):
            enhance_image(case[0], tmp_path / "forbidden" / "output.tif", native_materials_binding_path=binding_path)
    else:
        with pytest.raises((ValueError, V2EnhancementError, OSError)):
            _run(case, tmp_path / "forbidden", **kwargs)
    assert not (tmp_path / "forbidden").exists()


def _cache_context(case, binding):
    source, master, _, _, _, _, depth_path, handoff = case
    geometry = load_geometry_handoff(handoff, input_bytes=source.read_bytes(), depth_map_path=depth_path)
    output, receipt = apply_native_materials(master, binding, source_path=source.name, support=geometry.support)
    image = np.clip(linear_to_srgb(output.pixels), 0, 1)
    geometry = geometry.with_image(image, output.alpha)
    context = StageContext(device="cpu")
    context.set_artifact("image", image)
    context.set_artifact("depth_map", geometry.depth_map)
    context.set_artifact("geometry_handoff", geometry)
    context.set_artifact("native_materials", {"binding": binding, "receipt": receipt, "master": output})
    return context


def test_native_cache_identity_includes_frozen_policy_and_actual_execution(tmp_path):
    case = _case(tmp_path, strength=0)
    first = _cache_context(case, case[3])
    changed = copy.deepcopy(case[3])
    changed["policy"]["min_confidence"] = 0.9
    changed["binding_sha256"] = native_materials_binding_hash(changed)
    second = _cache_context(case, changed)
    np.testing.assert_array_equal(first.get_artifact("image"), second.get_artifact("image"))
    stage = EnhancementStage(version="1.4.0", geometry_recipe="normalized_crop_support_v1")
    assert stage.get_cache_key(first) != stage.get_cache_key(second)
    native = first.get_artifact("native_materials")
    native["master"] = replace(native["master"], pixels=native["master"].pixels + np.float32(0.00001))
    with pytest.raises(ValueError, match="actual native execution"):
        stage.get_cache_key(first)


@pytest.mark.parametrize("mutation", ["selected_source", "binding_digest", "support", "policy", "execution_plan"])
def test_native_cache_refuses_claimed_identity_without_matching_execution(tmp_path, mutation):
    case = _case(tmp_path, strength=0)
    context = _cache_context(case, case[3])
    receipt = context.get_artifact("native_materials")["receipt"]
    if mutation == "selected_source":
        receipt["selected_source_path"] = "other.tif"
    elif mutation == "binding_digest":
        receipt["binding_sha256"] = "f" * 64
    elif mutation == "support":
        receipt["support_sha256"] = "f" * 64
    elif mutation == "policy":
        receipt["execution"]["response_plan"]["policy"]["max_abs_delta"] = 0.1
    else:
        receipt["execution"]["plan_sha256"] = "f" * 64
    stage = EnhancementStage(version="1.4.0", geometry_recipe="normalized_crop_support_v1")
    with pytest.raises(ValueError, match="Native Materials"):
        stage.get_cache_key(context)


def test_native_processing_version_does_not_promote_legacy_or_geometry_only_defaults():
    assert processing_version_for_geometry() == "1.2.0"
    assert processing_version_for_geometry("normalized_crop_support_v1") == "1.3.0"
    assert processing_version_for_geometry("normalized_crop_support_v1", native_materials=True) == "1.4.0"
    with pytest.raises(ValueError, match="Native Materials requires"):
        processing_version_for_geometry(native_materials=True)
