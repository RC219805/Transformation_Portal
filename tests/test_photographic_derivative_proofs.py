"""Exact retained derivative identities stay separate from human approval."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import tifffile

from transformation_portal.core import verified_delivery
from transformation_portal.lux_depth_v4.photography import output_srgb_icc
from transformation_portal.photographic_delivery import DeliveryImage, ExportPreset, export_delivery, exports, package_delivery
from transformation_portal.photographic_delivery.models import canonical_bytes
from transformation_portal.photographic_delivery.proofs import package_proof_delivery, prepare_derivative_proofs

pytestmark = pytest.mark.unit


@pytest.fixture
def proof_case(tmp_path: Path) -> dict:
    source = io.BytesIO()
    tifffile.imwrite(
        source,
        np.full((12, 18, 3), 20000, dtype=np.uint16),
        photometric="rgb",
        metadata={"color_space": "srgb"},
        iccprofile=output_srgb_icc(),
    )
    raw = source.getvalue()
    image = DeliveryImage("source.tif", hashlib.sha256(raw).hexdigest(), "Living room", source_bytes=raw)
    root = tmp_path / "retained"
    export_delivery(
        [image],
        root,
        property_name="Proof house",
        export_presets=[ExportPreset.from_payload("website"), ExportPreset.from_payload("print")],
    )
    output = tmp_path / "packages"
    output.mkdir()
    return {
        "root": root,
        "inventory": (root / "inventory.json").read_bytes(),
        "spec": (root / "package-spec.json").read_bytes(),
        "output": output / "proof.zip",
    }


def _expected(case: dict) -> dict[str, str]:
    return {proof.path: proof.sha256 for proof in prepare_derivative_proofs(case["inventory"])}


def _package(case: dict, **kwargs) -> dict:
    return package_proof_delivery(
        case["inventory"],
        case["spec"],
        case["root"],
        case["output"],
        expected_proofs=kwargs.pop("expected_proofs", _expected(case)),
        **kwargs,
    )


def test_proof_identity_is_deterministic_and_deeply_immutable(proof_case: dict) -> None:
    first = prepare_derivative_proofs(proof_case["inventory"])
    second = prepare_derivative_proofs(proof_case["inventory"])
    inventory = json.loads(proof_case["inventory"])
    assert len(first) == len(inventory["files"]) == 2
    assert first == second
    by_path = {row["path"]: row for row in inventory["files"]}
    for proof in first:
        payload = proof.to_payload()
        assert payload["schema"] == "tp.photo.derivative_proof.v1"
        assert payload["scope"] == "inspection_identity_only"
        assert payload["inventory_sha256"] == hashlib.sha256(proof_case["inventory"]).hexdigest()
        assert payload["derivative_record"] == by_path[proof.path]
        assert proof.sha256 == hashlib.sha256(proof.canonical_payload).hexdigest()
        assert payload["derivative_record"]["approval_transferred"] is False
        payload["derivative_record"]["recipe"]["jpeg_quality"] = 50
        assert proof.to_payload()["derivative_record"]["recipe"]["jpeg_quality"] != 50
        with pytest.raises(FrozenInstanceError):
            proof.canonical_payload = b"changed"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("width", 13),
        ("height", 11),
        ("sha256", "a" * 64),
        ("size_bytes", 100),
        ("icc_sha256", "b" * 64),
        ("color_domain", "changed-domain"),
        ("intended_use", "Another intended consumer"),
    ],
)
def test_every_derivative_record_change_creates_a_new_identity(proof_case: dict, field: str, replacement) -> None:
    before = _expected(proof_case)
    inventory = json.loads(proof_case["inventory"])
    row = inventory["files"][0]
    row[field] = replacement
    changed = prepare_derivative_proofs(canonical_bytes(inventory))
    assert next(proof.sha256 for proof in changed if proof.path == row["path"]) != before[row["path"]]


def test_recipe_render_and_whole_inventory_changes_are_bound(proof_case: dict) -> None:
    before = _expected(proof_case)
    inventory = json.loads(proof_case["inventory"])
    inventory["files"][0]["recipe"]["jpeg_quality"] = 91
    inventory["files"][0]["recipe_sha256"] = hashlib.sha256(canonical_bytes(inventory["files"][0]["recipe"])).hexdigest()
    inventory["files"][0]["render"]["inspection_note"] = "same pixels, different recorded rendering context"
    changed = prepare_derivative_proofs(canonical_bytes(inventory))
    assert all(proof.sha256 != before[proof.path] for proof in changed)
    inventory = json.loads(proof_case["inventory"])
    inventory["property_name"] = "Changed inventory context"
    assert all(proof.sha256 != before[proof.path] for proof in prepare_derivative_proofs(canonical_bytes(inventory)))


def test_source_and_presentation_identity_changes_need_new_inspection(proof_case: dict) -> None:
    expected = _expected(proof_case)
    inventory = json.loads(proof_case["inventory"])
    source = inventory["sources"][0]
    source["source_master_sha256"] = "a" * 64
    source["source_identity_sha256"] = "a" * 64
    source["source_bundle"]["master_sha256"] = "a" * 64
    source["presentation_policy_sha256"] = "b" * 64
    for row in inventory["files"]:
        for key in ("source_master_sha256", "source_identity_sha256", "source_bundle", "presentation_policy_sha256"):
            row[key] = source[key]
    proof_case["inventory"] = canonical_bytes(inventory)
    assert all(proof.sha256 != expected[proof.path] for proof in prepare_derivative_proofs(proof_case["inventory"]))
    with pytest.raises(ValueError):
        _package(proof_case, expected_proofs=expected)
    assert list(proof_case["output"].parent.iterdir()) == []


def test_composite_native_identity_cannot_fall_back_to_master_hash(proof_case: dict) -> None:
    inventory = json.loads(proof_case["inventory"])
    source = inventory["sources"][0]
    source["source_bundle"]["descriptor_sha256"] = "a" * 64
    source["source_bundle"]["alpha_sha256"] = "b" * 64
    source["source_bundle"]["source_icc_sha256"] = "c" * 64
    identity = hashlib.sha256(canonical_bytes(source["source_bundle"])).hexdigest()
    source["source_identity_sha256"] = identity
    for row in inventory["files"]:
        row["source_bundle"] = source["source_bundle"]
        row["source_identity_sha256"] = identity
    assert all(
        proof.to_payload()["derivative_record"]["source_identity_sha256"] == identity
        for proof in prepare_derivative_proofs(canonical_bytes(inventory))
    )
    source["source_identity_sha256"] = source["source_master_sha256"]
    for row in inventory["files"]:
        row["source_identity_sha256"] = source["source_master_sha256"]
    with pytest.raises(ValueError):
        prepare_derivative_proofs(canonical_bytes(inventory))


@pytest.mark.parametrize("placement", ["source", "derivative"])
@pytest.mark.parametrize("identity", [[], {}])
def test_non_scalar_source_identity_is_a_validation_error(proof_case: dict, placement: str, identity) -> None:
    inventory = json.loads(proof_case["inventory"])
    row = inventory["sources"][0] if placement == "source" else inventory["files"][0]
    row["source_identity_sha256"] = identity
    with pytest.raises(ValueError):
        prepare_derivative_proofs(canonical_bytes(inventory))


@pytest.mark.parametrize("component", ["alpha_sha256", "source_icc_sha256"])
def test_ordinary_source_cannot_claim_orphan_native_components(proof_case: dict, component: str) -> None:
    inventory = json.loads(proof_case["inventory"])
    source = inventory["sources"][0]
    assert source["source_bundle"]["descriptor_sha256"] is None
    source["source_bundle"][component] = "a" * 64
    for row in inventory["files"]:
        row["source_bundle"] = source["source_bundle"]
    with pytest.raises(ValueError):
        prepare_derivative_proofs(canonical_bytes(inventory))


@pytest.mark.parametrize("descriptor", ["", "a" * 63, "A" * 64, []])
def test_native_descriptor_binding_requires_a_valid_digest(proof_case: dict, descriptor) -> None:
    inventory = json.loads(proof_case["inventory"])
    source = inventory["sources"][0]
    source["source_bundle"]["descriptor_sha256"] = descriptor
    for row in inventory["files"]:
        row["source_bundle"] = source["source_bundle"]
    with pytest.raises(ValueError):
        prepare_derivative_proofs(canonical_bytes(inventory))


@pytest.mark.parametrize("placement", ["source", "derivative"])
def test_presentation_binding_must_be_explicit_even_for_legacy_sources(proof_case: dict, placement: str) -> None:
    inventory = json.loads(proof_case["inventory"])
    row = inventory["sources"][0] if placement == "source" else inventory["files"][0]
    assert row.pop("presentation_policy_sha256") is None
    with pytest.raises(ValueError):
        prepare_derivative_proofs(canonical_bytes(inventory))


def test_coherent_recipe_hash_cannot_authorize_a_truncated_recipe(proof_case: dict) -> None:
    inventory = json.loads(proof_case["inventory"])
    row = inventory["files"][0]
    row["recipe"].pop("shoulder")
    row["recipe_sha256"] = hashlib.sha256(canonical_bytes(row["recipe"])).hexdigest()
    with pytest.raises(ValueError):
        prepare_derivative_proofs(canonical_bytes(inventory))


def test_coherent_recipe_hash_cannot_authorize_out_of_contract_quality(proof_case: dict) -> None:
    inventory = json.loads(proof_case["inventory"])
    row = inventory["files"][0]
    row["recipe"]["jpeg_quality"] = 49
    row["recipe_sha256"] = hashlib.sha256(canonical_bytes(row["recipe"])).hexdigest()
    with pytest.raises(ValueError):
        prepare_derivative_proofs(canonical_bytes(inventory))


def test_known_native_precision_companion_retains_its_full_recipe(proof_case: dict) -> None:
    inventory = json.loads(proof_case["inventory"])
    source = inventory["sources"][0]
    source["source_bundle"]["descriptor_sha256"] = "a" * 64
    identity = hashlib.sha256(canonical_bytes(source["source_bundle"])).hexdigest()
    source["source_identity_sha256"] = identity
    source["source_format"] = "NPY"
    source["bit_depth"] = 32
    row = inventory["files"][0]
    inventory["files"] = [row]
    row.update(
        path="photographic_master/retained-precision.tif",
        source_identity_sha256=identity,
        source_bundle=source["source_bundle"],
        source_bit_depth=32,
        preset="photographic_master",
        bit_depth=32,
        container_promotion=False,
        color_domain="extended_encoded_srgb",
        render=None,
    )
    row["recipe"] = {
        **ExportPreset.from_payload("photographic_master").to_payload(),
        "companion": "analytic_extended_srgb_transfer_no_clipping",
    }
    row["recipe_sha256"] = hashlib.sha256(canonical_bytes(row["recipe"])).hexdigest()
    proof = prepare_derivative_proofs(canonical_bytes(inventory))[0]
    assert proof.to_payload()["derivative_record"]["recipe"] == row["recipe"]
    assert proof.to_payload()["derivative_record"]["source_identity_sha256"] == identity
    row["recipe"]["companion"] = "unversioned-replacement"
    row["recipe_sha256"] = hashlib.sha256(canonical_bytes(row["recipe"])).hexdigest()
    with pytest.raises(ValueError):
        prepare_derivative_proofs(canonical_bytes(inventory))


@pytest.mark.parametrize("raw", [b"null", b"[]", b"{", b'{"schema":"x","schema":"y"}', b'{"value":NaN}'])
def test_ambiguous_or_malformed_inventory_is_rejected(raw: bytes) -> None:
    with pytest.raises(ValueError):
        prepare_derivative_proofs(raw)


def test_inventory_snapshot_requires_bounded_immutable_bytes(proof_case: dict) -> None:
    for raw in (bytearray(proof_case["inventory"]), "{}", b" " * (8 * 1024 * 1024 + 1)):
        with pytest.raises(ValueError):
            prepare_derivative_proofs(raw)


@pytest.mark.parametrize(
    "mutation",
    ["schema", "duplicate_path", "duplicate_source", "unknown_source", "source_binding", "recipe_hash", "boolean_size"],
)
def test_invalid_or_ambiguous_bindings_fail_closed(proof_case: dict, mutation: str) -> None:
    inventory = json.loads(proof_case["inventory"])
    row = inventory["files"][0]
    if mutation == "schema":
        inventory["schema"] = "tp.photo.delivery_inventory.v999"
    elif mutation == "duplicate_path":
        inventory["files"].append(dict(row))
    elif mutation == "duplicate_source":
        inventory["sources"].append(dict(inventory["sources"][0]))
    elif mutation == "unknown_source":
        row["source_identity_sha256"] = "c" * 64
    elif mutation == "source_binding":
        row["source_master_sha256"] = "d" * 64
    elif mutation == "recipe_hash":
        row["recipe"]["jpeg_quality"] = 90
    else:
        row["size_bytes"] = True
    with pytest.raises(ValueError):
        prepare_derivative_proofs(canonical_bytes(inventory))


@pytest.mark.parametrize("kind", ["missing", "extra", "invalid", "changed"])
def test_packaging_requires_the_exact_expected_proof_set(proof_case: dict, kind: str) -> None:
    expected = _expected(proof_case)
    path = next(iter(expected))
    if kind == "missing":
        expected.pop(path)
    elif kind == "extra":
        expected["uninspected.jpg"] = "f" * 64
    elif kind == "invalid":
        expected[path] = True
    else:
        expected[path] = "f" * 64
    with pytest.raises(ValueError):
        _package(proof_case, expected_proofs=expected)
    assert list(proof_case["output"].parent.iterdir()) == []


def test_malformed_expected_mapping_is_rejected_before_publication(proof_case: dict) -> None:
    class MalformedExpectedProofs(Mapping):
        def __getitem__(self, key):
            return "a" * 64

        def __iter__(self):
            yield ["nonhashable"]

        def __len__(self):
            return 1

    with pytest.raises(ValueError):
        _package(proof_case, expected_proofs=MalformedExpectedProofs())
    assert list(proof_case["output"].parent.iterdir()) == []


def test_non_digest_object_cannot_match_expected_identity_by_custom_equality(proof_case: dict) -> None:
    class PermissiveDigest:
        def __eq__(self, other):
            return True

    expected = _expected(proof_case)
    expected[next(iter(expected))] = PermissiveDigest()
    with pytest.raises(ValueError):
        _package(proof_case, expected_proofs=expected)
    assert list(proof_case["output"].parent.iterdir()) == []


@pytest.mark.parametrize("kind", ["missing", "null"])
def test_full_export_package_requires_its_contact_sheet(proof_case: dict, kind: str) -> None:
    inventory = json.loads(proof_case["inventory"])
    if kind == "missing":
        inventory.pop("contact_sheet")
    else:
        inventory["contact_sheet"] = None
    proof_case["inventory"] = canonical_bytes(inventory)
    with pytest.raises(ValueError, match="contact sheet"):
        _package(proof_case)
    assert list(proof_case["output"].parent.iterdir()) == []


def test_packaging_preserves_exact_inspected_bytes_without_encoding(proof_case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _expected(proof_case)
    inventory = json.loads(proof_case["inventory"])
    retained = {row["path"]: (proof_case["root"] / row["path"]).read_bytes() for row in inventory["files"]}

    def forbidden(*args, **kwargs):
        pytest.fail("Proof packaging must never regenerate an inspected derivative")

    monkeypatch.setattr(exports, "export_delivery", forbidden)
    monkeypatch.setattr(exports, "jpeg_bytes", forbidden)
    receipt = _package(proof_case)
    assert receipt["status"] == "verified"
    assert receipt["scope"] == "byte_integrity_not_semantic_verification"
    assert receipt["proof_scope"] == "inspection_identity_only"
    assert receipt["derivative_proof_sha256"] == expected
    with zipfile.ZipFile(proof_case["output"]) as archive:
        for path, data in retained.items():
            assert archive.read(f"proof-house-photographic-delivery/{path}") == data
        packaged_inventory = json.loads(archive.read("proof-house-photographic-delivery/inventory.json"))
        assert packaged_inventory["approval_scope"] == inventory["approval_scope"]
        assert all(row["approval_transferred"] is False for row in packaged_inventory["files"])


@pytest.mark.parametrize("member", ["derivative", "inventory"])
def test_changes_after_proof_validation_prevent_all_publication(
    proof_case: dict, monkeypatch: pytest.MonkeyPatch, member: str, tmp_path: Path
) -> None:
    original = verified_delivery.build_package
    copy = tmp_path / "copy"
    copy.mkdir()

    def mutate_then_build(manifest_path, source_root, output, copy_to=None):
        target = "inventory.json" if member == "inventory" else json.loads(proof_case["inventory"])["files"][0]["path"]
        (Path(source_root) / target).write_bytes(b"changed after inspection")
        return original(manifest_path, source_root, output, copy_to)

    monkeypatch.setattr(verified_delivery, "build_package", mutate_then_build)
    with pytest.raises(ValueError):
        _package(proof_case, copy_to=copy)
    assert list(proof_case["output"].parent.iterdir()) == []
    assert list(copy.iterdir()) == []


def test_optional_copy_retains_the_same_verified_archive_and_sidecars(proof_case: dict, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    copy.mkdir()
    receipt = _package(proof_case, copy_to=copy)
    assert receipt["derivative_proof_sha256"] == _expected(proof_case)
    assert receipt["copied_to"] == str(copy)
    for path in proof_case["output"].parent.iterdir():
        assert (copy / path.name).read_bytes() == path.read_bytes()
    assert len(list(copy.iterdir())) == 3


def test_packaging_uses_frozen_manifest_instead_of_mutable_original(proof_case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    original = verified_delivery.build_package

    def replace_original_then_build(manifest_path, source_root, output, copy_to=None):
        assert Path(manifest_path) != Path(source_root) / "package-spec.json"
        assert json.loads(Path(manifest_path).read_bytes()) == json.loads(proof_case["spec"])
        (Path(source_root) / "package-spec.json").write_bytes(b"invalid replacement")
        return original(manifest_path, source_root, output, copy_to)

    monkeypatch.setattr(verified_delivery, "build_package", replace_original_then_build)
    assert _package(proof_case)["status"] == "verified"


@pytest.mark.parametrize("kind", ["inventory", "missing", "extra", "redirect", "digest"])
def test_manifest_cannot_diverge_from_inspected_inventory(proof_case: dict, kind: str) -> None:
    spec = json.loads(proof_case["spec"])
    derivative_path = json.loads(proof_case["inventory"])["files"][0]["path"]
    row = next(row for row in spec["files"] if row["source"] == derivative_path)
    if kind == "inventory":
        next(row for row in spec["files"] if row["source"] == "inventory.json")["sha256"] = "a" * 64
    elif kind == "missing":
        spec["files"].remove(row)
    elif kind == "extra":
        secret = proof_case["root"] / "not-inspected.txt"
        secret.write_bytes(b"secret")
        spec["files"].append(
            {
                "source": secret.name,
                "archive_path": secret.name,
                "sha256": hashlib.sha256(b"secret").hexdigest(),
                "size_bytes": 6,
            }
        )
    elif kind == "redirect":
        row["archive_path"] = "different-name.jpg"
    else:
        row["sha256"] = "b" * 64
    proof_case["spec"] = canonical_bytes(spec)
    with pytest.raises(ValueError):
        _package(proof_case)
    assert list(proof_case["output"].parent.iterdir()) == []


def test_existing_generic_packaging_keeps_source_only_behavior(proof_case: dict) -> None:
    receipt = package_delivery(proof_case["root"], proof_case["output"])
    assert receipt["status"] == "verified"
    assert "proof_scope" not in receipt
    assert "derivative_proof_sha256" not in receipt
