"""Native Materials authority retains source precision, geometry and abstention."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

from transformation_portal.lux_depth_v3 import native_materials as native
from transformation_portal.lux_depth_v4.photography import decode_master
from transformation_portal.materials_v4.artifacts import write_evidence
from transformation_portal.materials_v4.contracts import MaterialEvidence, RegionEvidence
from transformation_portal.materials_v4.engine import MaterialOperation, ResponsePolicy

pytestmark = pytest.mark.unit


@pytest.fixture
def prepared_native(tmp_path):
    return _prepare_native(tmp_path)


def _prepare_native(tmp_path, *, bits=16, orientation=1, alpha=False):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    path = inputs / "native.tif"
    rows, columns = np.indices((31, 45))
    samples = np.stack((30000 + columns, 31000 + rows, 32000 + rows + columns), axis=-1).astype(np.uint16)
    if bits == 8:
        samples = (samples // 257).astype(np.uint8)
    if alpha:
        opacity = np.full(samples.shape[:2], np.iinfo(samples.dtype).max, samples.dtype)
        opacity[5:9, 11:17] = 0
        opacity[15:19, 21:27] //= 2
        samples = np.concatenate((samples, opacity[..., None]), axis=2)
    tifffile.imwrite(
        path,
        samples,
        photometric="rgb",
        metadata=None,
        extrasamples="unassalpha" if alpha else None,
        extratags=[(274, "H", 1, orientation, False)],
    )
    master = decode_master(path.read_bytes(), source_name=path.name, input_color="linear_srgb")
    masks = tmp_path / "materials"
    masks.mkdir()
    mask = np.zeros(master.shape, np.float32)
    mask[3:25, 7:31] = 1
    evidence = MaterialEvidence(
        master.source_sha256,
        master.shape,
        (RegionEvidence("water", "water", mask, semantic_confidence=0.99),),
    )
    write_evidence(evidence, masks / "evidence.json")
    manifest = masks / "materials.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "tp.lux.materials_manifest.v1",
                "inputs": [
                    {
                        "path": path.name,
                        "source_sha256": master.source_sha256,
                        "evidence_path": "evidence.json",
                        "evidence_sha256": hashlib.sha256((masks / "evidence.json").read_bytes()).hexdigest(),
                        "shape": list(master.shape),
                    }
                ],
            }
        )
    )
    policy = ResponsePolicy(operations=(MaterialOperation("water", "linear_gain_v1", 0.0001),), min_coverage_px=1)
    policy_path = masks / "policy.json"
    policy_path.write_text(json.dumps(policy.to_payload()))
    binding = native.freeze_native_materials(manifest, policy_path, inputs, [path], "linear_srgb")
    return path, master, mask, binding


def test_native_response_preserves_sub_eight_bit_gradient_and_support(prepared_native):
    _, master, mask, binding = prepared_native
    support = np.ones(master.shape, bool)
    support[:, 24:] = False
    result, receipt = native.apply_native_materials(master, binding, support=support)
    allowed = (mask > 0) & support
    np.testing.assert_array_equal(result.pixels[~allowed], master.pixels[~allowed])
    delta = result.pixels[allowed] - master.pixels[allowed]
    assert 0 < float(delta.min()) < float(delta.max()) < 1 / 255
    assert len(np.unique(result.pixels[5, 7:24, 0])) == 17
    assert result.pixels.dtype == np.float32 and result.shape == master.shape
    assert result.source_bit_depth == 16
    assert receipt["source_evidence_sha256"] != receipt["effective_evidence_sha256"]
    assert receipt["execution"]["protected_changed_pixels"] == 0
    assert receipt["precision_recovery"] is False


def test_evidence_location_is_portable_but_every_numeric_byte_is_bound(prepared_native, tmp_path):
    _, master, _, binding = prepared_native
    destination = tmp_path / "relocated"
    shutil.copytree(binding["evidence_root"], destination)
    moved = {**binding, "evidence_root": str(destination)}
    assert native.native_materials_binding_hash(moved) == binding["binding_sha256"]
    native.revalidate_native_materials(moved)
    first = native.apply_native_materials(master, binding, support=np.ones(master.shape, bool))[0]
    second = native.apply_native_materials(master, moved, support=np.ones(master.shape, bool))[0]
    np.testing.assert_array_equal(first.pixels, second.pixels)


@pytest.mark.parametrize("mutation", ["schema", "recipe", "digest", "unknown", "limits", "path", "source_bits"])
def test_closed_binding_rejects_tampering(prepared_native, mutation):
    _, _, _, original = prepared_native
    binding = copy.deepcopy(original)
    if mutation == "schema":
        binding["schema"] = "tp.lux.native_materials.v0"
    elif mutation == "recipe":
        binding["recipe"] = "automatic"
    elif mutation == "digest":
        binding["binding_sha256"] = "a" * 64
    elif mutation == "unknown":
        binding["trust_all"] = True
    elif mutation == "limits":
        binding["limits"]["max_working_bytes"] *= 2
    elif mutation == "path":
        binding["sources"][0]["path"] = "../native.tif"
    elif mutation == "source_bits":
        binding["sources"][0]["source_bit_depth"] = 32
    with pytest.raises(ValueError):
        native.validate_native_materials(binding)


def test_changed_source_fails_before_execution(prepared_native):
    path, _, _, binding = prepared_native
    changed = bytearray(path.read_bytes())
    changed[-1] ^= 1
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="source changed"):
        native.revalidate_native_materials(binding, input_root=path.parent)


def test_changed_mask_fails_before_native_operation(prepared_native, monkeypatch):
    _, master, _, binding = prepared_native
    evidence_root = Path(binding["evidence_root"])
    masks = list(evidence_root.glob("*.npy"))
    assert masks
    changed = bytearray(masks[0].read_bytes())
    changed[-1] ^= 1
    masks[0].write_bytes(changed)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Changed evidence reached a native operation")

    monkeypatch.setattr(native, "apply_response", forbidden)
    with pytest.raises(ValueError, match="Mask file changed"):
        native.apply_native_materials(master, binding, support=np.ones(master.shape, bool))


@pytest.mark.parametrize("mutation", ["source", "frame", "support_dtype", "support_frame"])
def test_source_and_support_frame_are_explicit(prepared_native, mutation):
    _, master, _, binding = prepared_native
    support = np.ones(master.shape, bool)
    if mutation == "source":
        master = replace(master, source_sha256="b" * 64)
    elif mutation == "frame":
        master = replace(master, pixels=master.pixels.transpose(1, 0, 2))
    elif mutation == "support_dtype":
        support = support.astype(np.float32)
    else:
        support = support.T
    with pytest.raises(ValueError, match="source/frame|geometry support"):
        native.apply_native_materials(master, binding, support=support)


def test_duplicate_identical_sources_require_original_relative_selector(prepared_native):
    _, master, _, original = prepared_native
    binding = copy.deepcopy(original)
    binding["sources"].append({**copy.deepcopy(binding["sources"][0]), "path": "second/native.tif"})
    manifest_path = Path(binding["evidence_root"]) / binding["manifest"]["path"]
    payload = json.loads(manifest_path.read_bytes())
    payload["inputs"].append({**payload["inputs"][0], "path": "second/native.tif"})
    manifest_bytes = json.dumps(payload).encode()
    manifest_path.write_bytes(manifest_bytes)
    binding["manifest"].update(sha256=hashlib.sha256(manifest_bytes).hexdigest(), size_bytes=len(manifest_bytes))
    binding["binding_sha256"] = native.native_materials_binding_hash(binding)
    with pytest.raises(ValueError, match="exactly one"):
        native.apply_native_materials(master, binding, support=np.ones(master.shape, bool))
    _, receipt = native.apply_native_materials(master, binding, "second/native.tif", np.ones(master.shape, bool))
    assert receipt["selected_source_path"] == "second/native.tif"


def test_low_memory_is_rejected_before_kernel_or_mask_decode(prepared_native, monkeypatch):
    _, master, _, original = prepared_native
    binding = copy.deepcopy(original)
    binding["sources"][0]["size_bytes"] = binding["limits"]["max_input_bytes"]
    binding["sources"][0]["shape"] = [8192, 8192]
    binding["sources"][0]["evidence"]["shape"] = [8192, 8192]
    binding["sources"][0]["mask_bytes"] = 8192 * 8192 * 4
    binding["binding_sha256"] = native.native_materials_binding_hash(binding)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Over-budget operation allocated masks or executed")

    monkeypatch.setattr(native, "load_bound_evidence", forbidden)
    monkeypatch.setattr(native, "apply_response", forbidden)
    with pytest.raises(ValueError, match="allocation exceeds"):
        native.apply_native_materials(master, binding, support=np.ones(master.shape, bool))


def test_native_response_cannot_silently_clip_out_of_sdr(prepared_native, monkeypatch):
    _, master, _, binding = prepared_native

    def extended_result(master, _evidence, _plan):
        return replace(master, pixels=np.full_like(master.pixels, 1.01)), {}

    monkeypatch.setattr(native, "apply_response", extended_result)
    with pytest.raises(ValueError, match="bounded SDR"):
        native.apply_native_materials(master, binding, support=np.ones(master.shape, bool))


def test_untrusted_inferred_observations_are_not_promoted(prepared_native, monkeypatch):
    _, master, _, binding = prepared_native
    original_load = native.load_bound_evidence

    def inferred(*args, **kwargs):
        evidence = original_load(*args, **kwargs)
        return replace(evidence, regions=tuple(replace(region, provenance="inferred") for region in evidence.regions))

    monkeypatch.setattr(native, "load_bound_evidence", inferred)
    result, receipt = native.apply_native_materials(master, binding, support=np.ones(master.shape, bool))
    np.testing.assert_array_equal(result.pixels, master.pixels)
    assert receipt["execution"]["regions"][0]["reason"] == "untrusted_inference"


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("orientation", range(1, 9))
def test_native_orientation_and_straight_alpha_are_preserved(tmp_path, bits, orientation):
    _, master, mask, binding = _prepare_native(tmp_path, bits=bits, orientation=orientation, alpha=True)
    result, receipt = native.apply_native_materials(master, binding, support=np.ones(master.shape, bool))
    assert binding["sources"][0]["shape"] == list(master.shape)
    assert binding["sources"][0]["source_bit_depth"] == bits
    assert binding["sources"][0]["alpha"] is True
    protected = (master.alpha == 0) | (mask == 0)
    np.testing.assert_array_equal(result.pixels[protected], master.pixels[protected])
    np.testing.assert_array_equal(result.alpha, master.alpha)
    assert receipt["alpha_mode"] == "straight"


def test_modified_native_baseline_cannot_claim_original_source_identity(prepared_native):
    _, master, _, binding = prepared_native
    changed = replace(master, pixels=master.pixels + np.float32(0.0001))
    with pytest.raises(ValueError, match="exactly one matching"):
        native.apply_native_materials(changed, binding, support=np.ones(master.shape, bool))


def test_forged_omission_cannot_erase_evidence_listed_in_frozen_manifest(prepared_native):
    _, master, _, original = prepared_native
    binding = copy.deepcopy(original)
    binding["sources"][0].update(evidence=None, mask_bytes=0)
    binding["binding_sha256"] = native.native_materials_binding_hash(binding)
    with pytest.raises(ValueError, match="manifest source map"):
        native.apply_native_materials(master, binding, support=np.ones(master.shape, bool))


def test_frozen_tile_scratch_is_included_inside_total_allocation(prepared_native):
    _, _, _, original = prepared_native
    binding = copy.deepcopy(original)
    binding["policy"]["max_working_bytes"] = binding["limits"]["max_working_bytes"]
    binding["binding_sha256"] = native.native_materials_binding_hash(binding)
    with pytest.raises(ValueError, match="allocation exceeds"):
        native.validate_native_materials(binding)


def test_insufficient_tile_policy_is_rejected_before_execution(prepared_native):
    _, _, _, original = prepared_native
    binding = copy.deepcopy(original)
    binding["policy"]["max_working_bytes"] = 1
    binding["binding_sha256"] = native.native_materials_binding_hash(binding)
    with pytest.raises(ValueError, match="tile scratch exceeds"):
        native.validate_native_materials(binding)


@pytest.mark.parametrize("mutation", ["float", "allocation"])
def test_unsupported_precision_or_allocation_fails_before_native_decode(prepared_native, monkeypatch, mutation):
    path, _, _, binding = prepared_native
    if mutation == "float":
        tifffile.imwrite(path, np.full((31, 45, 3), 0.5, np.float32), photometric="rgb", metadata=None)
    else:
        monkeypatch.setattr(native, "_header", lambda *_args: ((8192, 8192), 16, False))

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unsupported source allocated native photographic pixels")

    monkeypatch.setattr(native, "decode_master", forbidden)
    root = Path(binding["evidence_root"])
    with pytest.raises(ValueError, match="integer 8/16|allocation exceeds"):
        native.freeze_native_materials(root / "materials.json", root / "policy.json", path.parent, [path], "linear_srgb")
