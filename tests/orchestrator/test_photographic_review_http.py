"""Real image/revision/export/ZIP flow through the new authenticated route seam."""

from __future__ import annotations

import hashlib
import io
import json
import uuid
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from starlette.responses import JSONResponse

from transformation_portal.api.routes.photographic_review import create_photographic_review_router
from transformation_portal.lux_depth_v4.photography import output_srgb_icc
from transformation_portal.orchestrator.photographic_review_access import ReviewAccessError, ReviewContext, ReviewRuntime
from transformation_portal.orchestrator.photographic_review_store import MemoryReviewStore

pytestmark = pytest.mark.unit


@pytest.fixture(params=["v1", "v2"])
def review_case(monkeypatch, request):
    from transformation_portal.orchestrator import photographic_review_service as service
    from transformation_portal.orchestrator import photographic_review_store as stores

    store = MemoryReviewStore()
    monkeypatch.setattr(stores, "get_review_store", lambda: store)
    monkeypatch.setattr(service, "get_review_store", lambda: store)
    pixels = np.empty((64, 96, 3), dtype=np.uint16)
    pixels[..., 0] = np.arange(96, dtype=np.uint16)[None, :] * 600
    pixels[..., 1] = np.arange(64, dtype=np.uint16)[:, None] * 800
    pixels[..., 2] = 12000
    encoded = io.BytesIO()
    tifffile.imwrite(
        encoded,
        pixels,
        photometric="rgb",
        metadata=None,
        extratags=[(34675, "B", len(output_srgb_icc()), output_srgb_icc(), False)],
    )
    raw = encoded.getvalue()
    original = io.BytesIO()
    Image.fromarray((pixels / 257).astype(np.uint8)).save(original, format="PNG", icc_profile=output_srgb_icc())
    path = "v6/input-0000/delivery.tif"
    original_path = "original/input-0000.png"
    blobs = {path: raw, original_path: original.getvalue()}
    refs = [
        {
            "artifact_path": key,
            "sha256": hashlib.sha256(value).hexdigest(),
            "size_bytes": len(value),
            "scene_id": "input-0000",
            "label": "Living room",
            "source_name": "final.tif" if key == path else "original.png",
            "role": "final" if key == path else "original",
            "input_color": "auto",
            "kind": "indexed",
        }
        for key, value in blobs.items()
    ]
    assets = {}
    calls = []
    discarded = []
    controls = SimpleNamespace(fail_after_publish=None)
    job_id = "job_photo_" + uuid.uuid4().hex
    base = f"/{request.param}/jobs/{job_id}/photographic-review"
    canonical_base = f"/v1/jobs/{job_id}/photographic-review"
    other_base = f"/{'v2' if request.param == 'v1' else 'v1'}/jobs/{job_id}/photographic-review"

    async def materialize(ref, destination, maximum_bytes):
        value = assets[ref["storage_path"]] if ref.get("kind") == "review_asset" else blobs[ref["artifact_path"]]
        digest = hashlib.sha256(value).hexdigest()
        if digest != ref["sha256"] or len(value) != ref["size_bytes"] or len(value) > maximum_bytes:
            raise ReviewAccessError("Artifact changed")
        with destination.open("xb") as stream:
            stream.write(value)
        return {"sha256": digest, "size_bytes": len(value)}

    async def publish(storage, source, mime):
        assert storage.startswith("photographic-review/") and storage not in assets
        if controls.fail_after_publish == len(calls) + 1:
            raise OSError("fixture publication interrupted")
        assets[storage] = source.read_bytes()
        calls.append((storage, mime))

    async def discard(paths):
        discarded.extend(paths)
        for storage in paths:
            assets.pop(storage, None)

    context = ReviewContext(
        job_id,
        "tenant_a",
        {"username": "reviewer", "assurance": "test_authenticated_operator"},
        refs,
        materialize,
        publish,
        discard,
    )

    def error(status, **kwargs):
        return JSONResponse(
            {
                "schema": "tp.orchestrator.error.v1",
                "success": False,
                "data": None,
                "error": {key: value for key, value in kwargs.items() if key != "retriable"},
            },
            status_code=status,
            headers={"Cache-Control": "no-store"},
        )

    async def authorize(request, requested_job):
        if request.headers.get("x-api-key") != "fixture-key":
            return error(401, code="UNAUTHORIZED", message="authentication required", details={})
        if requested_job != job_id:
            return error(404, code="NOT_FOUND", message="job not found", details={})
        return context

    application = FastAPI()
    application.include_router(create_photographic_review_router(ReviewRuntime(authorize, error)))
    with TestClient(application, headers={"x-api-key": "fixture-key"}) as client:
        yield SimpleNamespace(
            client=client,
            base=base,
            canonical_base=canonical_base,
            other_base=other_base,
            job_id=job_id,
            path=path,
            original_path=original_path,
            sha=refs[0]["sha256"],
            original_sha=refs[1]["sha256"],
            blobs=blobs,
            assets=assets,
            pixels=pixels,
            store=store,
            context=context,
            calls=calls,
            discarded=discarded,
            controls=controls,
        )


def _approve(case, path=None, sha=None, version=0, status="approved"):
    return case.client.post(
        case.base + "/decisions",
        json={
            "artifact_path": path or case.path,
            "sha256": sha or case.sha,
            "status": status,
            "expected_version": version,
            "note": "Checked native window edges",
        },
    )


def test_catalogue_native_crop_and_fit_are_hash_bound_display_derivatives(review_case):
    case = review_case
    response = case.client.get(case.base)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    data = response.json()["data"]
    assert data["state_version"] == 0
    scene = data["scenes"][0]
    assert {v["role"] for v in scene["variants"]} == {"original", "final"}
    final = next(v for v in scene["variants"] if v["role"] == "final")
    assert (final["width"], final["height"], final["bit_depth"]) == (96, 64, 16)
    assert (
        next(v for v in scene["variants"] if v["role"] == "original")["intended_use"]
        == "Retained input / comparison reference"
    )
    assert "storage_path" not in response.text and "private_original" not in response.text
    params = {"artifact_path": case.path, "sha256": case.sha, "x": 11, "y": 7, "width": 13, "height": 9}
    crop = case.client.get(case.base + "/native-crop", params=params)
    assert crop.status_code == 200
    assert crop.headers["x-tp-source-sha256"] == case.sha
    assert (crop.headers["x-tp-native-width"], crop.headers["x-tp-native-height"]) == ("13", "9")
    observed = np.asarray(Image.open(io.BytesIO(crop.content)))
    expected = np.rint(case.pixels[7:16, 11:24].astype(np.float64) / 257).astype(np.uint8)
    assert observed.shape == (9, 13, 3)
    assert np.max(np.abs(observed.astype(int) - expected.astype(int))) <= 1
    fit = case.client.get(case.base + "/render", params={"artifact_path": case.path, "sha256": case.sha, "max_edge": 32})
    assert fit.status_code == 200 and max(Image.open(io.BytesIO(fit.content)).size) == 32
    assert hashlib.sha256(case.blobs[case.path]).hexdigest() == case.sha


@pytest.mark.parametrize(
    "changes", [{"sha256": "0" * 64}, {"artifact_path": "../private.tif"}, {"x": 90, "width": 13}, {"width": 2049}]
)
def test_native_inspection_rejects_changed_unknown_or_unbounded_requests(review_case, changes):
    case = review_case
    params = {"artifact_path": case.path, "sha256": case.sha, "x": 0, "y": 0, "width": 13, "height": 9, **changes}
    response = case.client.get(case.base + "/native-crop", params=params)
    assert response.status_code in {400, 404, 409, 422}
    assert not case.assets


def test_decisions_are_explicit_exact_hash_and_compare_and_swap(review_case):
    case = review_case
    assert _approve(case).status_code == 200
    stale = _approve(case, version=0, status="needs_revision")
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "REVIEW_CONFLICT"
    changed = _approve(case, sha="0" * 64, version=1)
    assert changed.status_code == 409
    original = _approve(case, path=case.original_path, sha=case.original_sha, version=1)
    assert original.status_code == 400
    state = case.client.get(case.base).json()["data"]
    assert state["decisions"][0]["status"] == "approved"
    assert state["decisions"][0]["inspection"]["bit_depth"] == 16


def test_lighting_reference_cohesion_and_quality_do_not_change_approval(review_case):
    case = review_case
    groups = {"groups": [{"name": "Daylight", "reference_sha256": case.sha, "members": [case.sha]}], "expected_version": 0}
    response = case.client.post(case.base + "/lighting-groups", json=groups)
    assert response.status_code == 200, response.text
    assert response.json()["data"]["decisions"] == []
    result = case.client.get(
        case.base + "/quality",
        params={
            "artifact_path": case.path,
            "sha256": case.sha,
            "reference_path": case.original_path,
            "reference_sha256": case.original_sha,
        },
    )
    assert result.status_code == 200, result.text
    assert result.json()["data"]["scope"] == "reference_relative_not_photographic_acceptance"
    assert result.json()["data"]["algorithm_revision"] == "spatial_v2"
    assert case.client.get(case.base).json()["data"]["decisions"] == []


def test_revision_preview_commit_approval_export_and_revocation_flow(review_case):
    case = review_case
    assert _approve(case).status_code == 200
    request = {
        "artifact_path": case.path,
        "sha256": case.sha,
        "scene_id": "input-0000",
        "expected_version": 1,
        "recipe": {"exposure_stops": -0.10, "white_balance": [1, 1, 1], "saturation": 1},
        "preview_only": True,
    }
    preview = case.client.post(case.base + "/revisions", json=request)
    assert preview.status_code == 200, preview.text
    assert preview.json()["data"]["preview_data_url"].startswith("data:image/png;base64,")
    assert not case.assets and case.client.get(case.base).json()["data"]["state_version"] == 1
    committed = case.client.post(case.base + "/revisions", json={**request, "preview_only": False})
    assert committed.status_code == 200, committed.text
    new = committed.json()["data"]["revision"]
    assert new["review_status"] == "unreviewed" and new["sha256"] != case.sha
    assert "storage_path" not in committed.text
    scene = case.client.get(case.base).json()["data"]["scenes"][0]
    assert {v["role"] for v in scene["variants"]} == {"original", "final", "previous_revision"}
    payload = {
        "property_name": "Seaview House",
        "presets": ["photographic_master", "website"],
        "approved_sha256": [new["sha256"]],
        "expected_version": 2,
    }
    blocked = case.client.post(case.base + "/delivery", json=payload)
    assert blocked.status_code == 400
    assert _approve(case, new["artifact_path"], new["sha256"], version=2).status_code == 200
    exported = case.client.post(case.base + "/delivery", json={**payload, "expected_version": 3})
    assert exported.status_code == 200, exported.text
    delivery = exported.json()["data"]
    assert delivery["filename"].startswith("seaview-house-approved-")
    assert delivery["download_url"].startswith(case.canonical_base + "/files/")
    download_alias = case.base + delivery["download_url"].removeprefix(case.canonical_base)
    downloaded = case.client.get(download_alias)
    assert downloaded.status_code == 200
    assert hashlib.sha256(downloaded.content).hexdigest() == delivery["sha256"]
    assert case.client.get(delivery["download_url"]).content == downloaded.content
    with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
        assert archive.testzip() is None
        name = next(name for name in archive.namelist() if name.endswith("/approval-record.json"))
        approvals = json.loads(archive.read(name))
        assert approvals["masters"][0]["sha256"] == new["sha256"]
        assert approvals["masters"][0]["actor"]["username"] == "reviewer"
        sums_name = next(name for name in archive.namelist() if name.endswith("/_package/SHA256SUMS.txt"))
        for line in archive.read(sums_name).decode().splitlines():
            digest, member = line.split("  ", 1)
            assert hashlib.sha256(archive.read(member)).hexdigest() == digest
    assert _approve(case, new["artifact_path"], new["sha256"], version=4, status="needs_revision").status_code == 200
    assert case.client.get(delivery["download_url"]).status_code == 409
    assert case.client.get(download_alias).status_code == 409
    assert hashlib.sha256(case.blobs[case.path]).hexdigest() == case.sha


def test_recorded_view_recovers_saved_revision_policy_and_audits_without_changing_approval_scope(review_case):
    import base64

    case = review_case
    query = {"artifact_path": case.path, "sha256": case.sha}
    assert case.client.get(case.base + "/render", params={**query, "view": "recorded"}).status_code == 409
    legacy = case.client.get(case.base + "/render", params=query)
    explicit = case.client.get(case.base + "/render", params={**query, "view": "clip_srgb"})
    assert legacy.content == explicit.content
    assert legacy.headers["x-tp-presentation-view"] == "clip_srgb"
    assert legacy.headers["x-tp-presentation-policy-sha256"] == "none"
    assert case.client.get(case.base + "/render", params={**query, "view": "guessed"}).status_code == 422
    request = {
        **query,
        "scene_id": "input-0000",
        "expected_version": 0,
        "recipe": {"exposure_stops": -0.1},
        "presentation_policy_sha256": None,
    }
    preview = case.client.post(case.base + "/revisions", json={**request, "preview_only": True}).json()["data"]
    saved = case.client.post(case.base + "/revisions", json={**request, "preview_only": False})
    assert saved.status_code == 200, saved.text
    current = next(
        item for item in case.client.get(case.base).json()["data"]["scenes"][0]["variants"] if item["role"] == "final"
    )
    policy_sha = current["presentation_policy_sha256"]
    assert current["presentation"]["available_views"] == ["recorded", "clip_srgb"]
    query = {"artifact_path": current["artifact_path"], "sha256": current["sha256"], "view": "recorded"}
    fit = case.client.get(case.base + "/render", params=query)
    assert fit.status_code == 200, fit.text
    assert fit.headers["x-tp-presentation-policy-sha256"] == policy_sha
    expected = np.asarray(Image.open(io.BytesIO(base64.b64decode(preview["preview_data_url"].split(",", 1)[1]))))
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(fit.content))), expected)
    crop = case.client.get(case.base + "/native-crop", params={**query, "x": 11, "y": 7, "width": 13, "height": 9})
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(crop.content))), expected[7:16, 11:24])
    approval = {
        "artifact_path": current["artifact_path"],
        "sha256": current["sha256"],
        "status": "approved",
        "expected_version": 1,
        "inspection_presentation": {"view": "recorded", "policy_sha256": "0" * 64},
    }
    assert case.client.post(case.base + "/decisions", json=approval).status_code == 409
    approval["inspection_presentation"]["policy_sha256"] = policy_sha
    decision = case.client.post(case.base + "/decisions", json=approval)
    assert decision.status_code == 200, decision.text
    recorded = decision.json()["data"]["decisions"][0]
    assert recorded["inspection_presentation"] == approval["inspection_presentation"]
    assert recorded["source_identity_sha256"] == current["sha256"]
    stale = case.client.post(
        case.base + "/revisions",
        json={
            **request,
            "artifact_path": current["artifact_path"],
            "sha256": current["sha256"],
            "expected_version": 2,
            "presentation_policy_sha256": "0" * 64,
        },
    )
    assert stale.status_code == 409


def test_authentication_unknown_job_and_no_implicit_approval(review_case):
    case = review_case
    assert case.client.get(case.base, headers={"x-api-key": "invalid"}).status_code == 401
    assert case.client.get(case.base.replace(case.job_id, "job_other")).status_code == 404
    payload = {"property_name": "House", "presets": ["website"], "approved_sha256": [case.sha], "expected_version": 0}
    assert case.client.post(case.base + "/delivery", json=payload).status_code == 400
    assert not case.assets


def test_delivery_history_recovers_committed_export_without_catalogue_or_image_reads(review_case, monkeypatch):
    from transformation_portal.orchestrator import photographic_review_service as service

    case = review_case
    assert _approve(case).status_code == 200
    payload = {
        "property_name": "History",
        "presets": ["website"],
        "approved_sha256": [case.sha],
        "expected_version": 1,
        "request_correlation_id": "original-attempt",
    }
    exported = case.client.post(case.base + "/delivery", json=payload)
    assert exported.status_code == 200, exported.text

    async def forbidden(*args, **kwargs):
        raise AssertionError("History must not load catalogue pixels")

    monkeypatch.setattr(service, "load_image", forbidden)
    found = case.client.get(case.base + "/deliveries?request_correlation_id=original-attempt")
    assert found.status_code == 200, found.text
    page = found.json()["data"]
    assert page["state_version"] == 2 and page["snapshot_count"] == 1
    row = page["items"][0]
    assert row["id"] == exported.json()["data"]["id"]
    assert row["request_fingerprint_sha256"] == exported.json()["data"]["request_fingerprint_sha256"]
    assert row["intent"]["presets"][0]["name"] == "website"
    assert row["approval_version"] == 1 and row["file_validation"] == "not_checked"
    assert row["sources"][0]["sha256"] == case.sha
    assert row["applied_derivatives_schema"] == "tp.photo.delivery_applied.v1"
    assert row["applied_derivatives"][0]["operation"] == "render_sdr"
    assert row["applied_derivatives"][0]["render"]["mode"] == "perceptual_srgb"
    assert "storage_path" not in found.text and "private_original" not in found.text
    assert found.headers["cache-control"] == "no-store"
    assert (
        case.client.get(case.other_base + "/deliveries").json()["data"]
        == case.client.get(case.base + "/deliveries").json()["data"]
    )
    assert case.client.get(case.base + "/deliveries", headers={"x-api-key": "invalid"}).status_code == 401
    assert case.client.get(case.base + "/deliveries?limit=65").status_code == 422
    assert case.client.get(case.base + "/deliveries?cursor=invalid").status_code == 400


@pytest.mark.parametrize(
    ("method", "suffix"),
    [
        ("GET", ""),
        ("GET", "/render"),
        ("GET", "/native-crop"),
        ("GET", "/quality"),
        ("POST", "/decisions"),
        ("POST", "/lighting-groups"),
        ("POST", "/revisions"),
        ("POST", "/delivery"),
        ("GET", "/deliveries"),
        ("GET", "/files/" + "0" * 64 + "/delivery.zip"),
    ],
)
def test_each_versioned_operation_retains_auth_and_job_rejection(review_case, method, suffix):
    case = review_case
    params = {
        "artifact_path": case.path,
        "sha256": case.sha,
        "reference_path": case.original_path,
        "reference_sha256": case.original_sha,
        "x": 0,
        "y": 0,
        "width": 8,
        "height": 8,
    }
    bodies = {
        "/decisions": {"artifact_path": case.path, "sha256": case.sha, "status": "selected", "expected_version": 0},
        "/lighting-groups": {
            "groups": [{"name": "Daylight", "reference_sha256": case.sha, "members": [case.sha]}],
            "expected_version": 0,
        },
        "/revisions": {
            "artifact_path": case.path,
            "sha256": case.sha,
            "scene_id": "input-0000",
            "expected_version": 0,
            "recipe": {},
            "preview_only": True,
        },
        "/delivery": {"property_name": "House", "presets": ["website"], "approved_sha256": [case.sha], "expected_version": 0},
    }
    kwargs = {"json": bodies[suffix]} if method == "POST" else {"params": params}
    denied = case.client.request(method, case.base + suffix, headers={"x-api-key": "invalid"}, **kwargs)
    assert denied.status_code == 401 and denied.json()["error"]["code"] == "UNAUTHORIZED"
    absent = case.client.request(method, case.base.replace(case.job_id, "job_other") + suffix, **kwargs)
    assert absent.status_code == 404 and absent.json()["error"]["code"] == "NOT_FOUND"
    assert not case.assets and not case.calls


def test_aliases_share_exact_review_state_and_compare_and_swap(review_case):
    case = review_case
    assert _approve(case).status_code == 200
    current = case.client.get(case.base).json()
    alias = case.client.get(case.other_base).json()
    assert alias == current
    assert alias["schema"] == "tp.photographic_review.v1"
    assert alias["data"]["state_version"] == 1
    stale = case.client.post(
        case.other_base + "/decisions",
        json={"artifact_path": case.path, "sha256": case.sha, "status": "needs_revision", "expected_version": 0},
    )
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "REVIEW_CONFLICT"


def test_decoder_validation_never_exposes_internal_paths(review_case, monkeypatch):
    from transformation_portal.orchestrator import photographic_review_service as service

    def invalid(_):
        raise ValueError("decoder failed at /private/operator/secret-profile.icc https://internal.invalid/key")

    monkeypatch.setattr(service, "_metadata", invalid)
    response = review_case.client.get(review_case.base)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_ARGUMENT"
    assert "secret-profile" not in response.text and "internal.invalid" not in response.text


def test_revision_conflict_discards_only_uncommitted_publication(review_case, monkeypatch):
    from transformation_portal.orchestrator.photographic_review_store import ReviewConflict

    case = review_case
    original_mutate = case.store.mutate

    async def conflict(job, tenant, version, change, event):
        if event["kind"] == "revision":
            raise ReviewConflict("fixture concurrent review change")
        return await original_mutate(job, tenant, version, change, event)

    monkeypatch.setattr(case.store, "mutate", conflict)
    response = case.client.post(
        case.base + "/revisions",
        json={
            "artifact_path": case.path,
            "sha256": case.sha,
            "scene_id": "input-0000",
            "expected_version": 0,
            "recipe": {"exposure_stops": -0.1},
            "preview_only": False,
        },
    )
    assert response.status_code == 409
    assert len(case.discarded) == 1 and not case.assets
    assert case.client.get(case.base).json()["data"]["state_version"] == 0


def test_partial_delivery_publication_is_rolled_back(review_case):
    case = review_case
    assert _approve(case).status_code == 200
    case.controls.fail_after_publish = 2
    response = case.client.post(
        case.base + "/delivery",
        json={
            "property_name": "Seaview",
            "presets": ["website"],
            "approved_sha256": [case.sha],
            "expected_version": 1,
            "scene_labels": {"input-0000": "Living room"},
        },
    )
    assert response.status_code == 503
    assert len(case.discarded) == 1 and not case.assets
    assert case.client.get(case.base).json()["data"]["state_version"] == 1


def test_scene_names_and_composite_identity_are_bound_to_delivery(review_case):
    case = review_case
    assert _approve(case).status_code == 200
    exported = case.client.post(
        case.base + "/delivery",
        json={
            "property_name": "Seaview",
            "presets": ["website"],
            "approved_sha256": [case.sha],
            "expected_version": 1,
            "scene_labels": {"input-0000": "Ocean living room"},
        },
    )
    assert exported.status_code == 200, exported.text
    delivery = exported.json()["data"]
    assert any("ocean-living-room" in row["path"] for row in delivery["inventory"]["files"])
    case.context.references[0]["source_identity_sha256"] = "0" * 64
    denied = case.client.get(delivery["download_url"])
    assert denied.status_code == 409
    assert denied.json()["error"]["code"] == "APPROVAL_REQUIRED"


def test_identical_bytes_in_distinct_scenes_do_not_share_displayed_decisions(review_case):
    case = review_case
    other = dict(case.context.references[0], artifact_path="v6/input-0001/delivery.tif", scene_id="input-0001")
    case.context.references.append(other)
    case.blobs[other["artifact_path"]] = case.blobs[case.path]
    assert _approve(case).status_code == 200
    scenes = case.client.get(case.base).json()["data"]["scenes"]
    assert next(s for s in scenes if s["scene_id"] == "input-0000")["decision"]["artifact_path"] == case.path
    assert next(s for s in scenes if s["scene_id"] == "input-0001")["decision"] is None


def test_delivery_disk_preflight_accounts_for_decoded_tiffs_and_native_companions():
    from transformation_portal.orchestrator.photographic_review_service import delivery_disk_budget
    from transformation_portal.photographic_delivery import ExportPreset

    decision = {"inspection": {"width": 6000, "height": 4000}}
    compressed = {"size_bytes": 1024}
    website = [ExportPreset.from_payload("website")]
    retouch = [ExportPreset.from_payload("retouch_print")]
    web_budget = delivery_disk_budget([decision], [compressed], website)
    float_budget = delivery_disk_budget([decision], [compressed], retouch)
    assert float_budget > web_budget + 1024**3
    native = {"size_bytes": 1024, "native": {"descriptor": {"size_bytes": 8192}, "alpha": {"size_bytes": 96_000_000}}}
    assert delivery_disk_budget([decision], [native], retouch) > float_budget + 96_000_000
    with pytest.raises(ValueError, match="geometry"):
        delivery_disk_budget([{"inspection": {"width": 100000, "height": 4000}}], [compressed], retouch)


def test_insufficient_delivery_disk_rejects_before_source_materialization(review_case, monkeypatch):
    from transformation_portal.orchestrator import photographic_review_service as service

    case = review_case
    assert _approve(case).status_code == 200
    monkeypatch.setattr(service.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    response = case.client.post(
        case.base + "/delivery",
        json={
            "property_name": "Seaview",
            "presets": ["retouch_print"],
            "approved_sha256": [case.sha],
            "expected_version": 1,
        },
    )
    assert response.status_code == 400 and "require" in response.json()["error"]["message"]
    assert not case.assets and not case.calls
