"""Opt-in canonical native Materials qualification against owned local services."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tests import test_managed_materials_canonical_qualification as controlled_authority
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.photography import output_srgb_icc
from transformation_portal.photographic_delivery import managed_qualification as managed

pytestmark = [pytest.mark.integration, pytest.mark.slow]
fixture_authority = controlled_authority.fixture_authority
request_case = controlled_authority.request_case


@pytest.mark.parametrize("scenario", sorted(managed.SCENARIOS))
def test_real_canonical_s3_managed_cohort_faults_and_independent_restart_verifier(authority, tmp_path, scenario):
    container = os.environ.get("TP_MANAGED_QUALIFICATION_S3_CONTAINER")
    publication = os.environ.get("TP_MANAGED_QUALIFICATION_S3_PUBLICATION_ROOT")
    if not container or not publication or not os.environ.get("TP_MANAGED_QUALIFICATION_DATABASE_URL"):
        pytest.skip("Canonical qualification requires owned loopback MinIO/Redis and a dedicated migrated *_test database")
    profile = managed._profile_identity()
    if profile is None:
        pytest.skip("The current source/runtime does not match the governed managed resource profile")
    retained, evidence = tmp_path / "retained", tmp_path / "evidence"
    retained.mkdir()
    evidence.mkdir()
    source = retained / "source.tif"
    samples = np.full((4000, 5000, 4), 24000, np.uint16)
    samples[..., 3] = 65535
    samples[:, :8, 3] = 0
    samples[:, 8:16, 3] = 32768
    tifffile.imwrite(
        source, samples, photometric="rgb", extrasamples="unassalpha", rowsperstrip=64, iccprofile=output_srgb_icc()
    )
    del samples
    recipe = {
        "schema": "tp.photo.materials_recipe.v1",
        "input_stage": "display_referred",
        "regions": [
            {"id": f"preservation-{index}", "label": "stone", "region": rectangle, "strength": 0}
            for index, rectangle in enumerate(([0, 0, 0.5, 0.5], [0.5, 0, 1, 0.5], [0, 0.5, 0.5, 1], [0.5, 0.5, 1, 1]))
        ],
    }
    manifest = retained / "manifest.json"
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest.write_bytes(
        canonicalize_json(
            {"schema": managed.MANIFEST_SCHEMA, "sources": [{"path": str(source), "sha256": digest, "recipe": recipe}]}
        )
    )
    report = managed.qualify(
        manifest,
        evidence / "fresh",
        profile,
        provider="canonical-s3",
        canonical_plan=authority.plan,
        canonical_bindings=authority.bindings,
        s3_publication_root=Path(publication),
        s3_container=container,
        repeats=2 if scenario == "success" else 1,
        scenario=scenario,
    )
    assert report["status"] == "measured", report["blocking_reasons"]
    result = report["complete_managed_result"]
    assert result["provider"] == "canonical-s3" and result["canonical_fixture_import"]["redis_locator_acquired"] is True
    expected = 2 if scenario == "success" else 1 if scenario == "cancel_transport" else 0
    assert len(result["results"]) == len(report["independent_verification"]["results"]) == expected
    assert all(
        row["products_verified"] == 5 and row["maximum_16bit_code_delta"] <= 1
        for row in report["independent_verification"]["results"]
    )
    assert all(
        result["final_admission"][key] == 0
        for key in ("active", "outstanding", "queued", "memory_reserved_bytes", "scratch_reserved_bytes")
    )
    if expected:
        assert result["transport_ownership_verified"] is True
    if scenario == "success":
        assert report["encoded_products_repeatable"] is True
        phases = [row["phase"] for row in result["phases"]]
        assert "same_tenant_inspection_completed_while_native_held" in phases
        assert "same_tenant_inspection_completed_while_outer_transport_blocked" in phases
