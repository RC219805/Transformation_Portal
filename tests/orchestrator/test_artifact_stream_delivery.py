"""Same-origin artifact delivery changes transport without granting access."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app
from tests import test_app_orchestrator_contract_http as http_contract
from tests.orchestrator import test_pilot_control_plane_contract as pilot_contract
from transformation_portal.orchestrator.artifact_store.base import ArtifactNotFoundError, ArtifactStoreError

pytestmark = pytest.mark.unit
_reset_pilot_state = pilot_contract._reset_pilot_state
_STREAM_HEADER = {"X-TP-Artifact-Delivery": "stream"}


@pytest.fixture(params=[False, True], ids=["legacy", "committed"])
def artifact_case(request, monkeypatch, tmp_path):
    committed = request.param
    monkeypatch.setattr(app, "_uses_dispatch_authority", lambda: committed)
    monkeypatch.setattr(app, "_record_pilot_audit", pilot_contract._audit_noop)
    relative = "outputs/preview.png"
    storage_path = "generations/committed/preview.png" if committed else relative
    local = tmp_path / "outputs" / "preview.png"
    local.parent.mkdir()
    local.write_bytes(b"local bytes must never replace missing S3 authority")
    job = app.Job(
        id="job_stream_delivery",
        created_at=app._now(),
        state="succeeded",
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(tmp_path)}},
        artifact_store_mirrored=True,
        artifact_store_backend="s3",
        artifacts={
            "items": [{"path": relative, "relative_path": relative}],
            "indexed_count": 1,
            "lifecycle": {"mirror_status": "mirrored"},
        },
        artifact_lookup={relative: local},
    )
    http_contract._seed_job(job)
    calls = []

    class Store:
        backend = "s3"
        failure = None

        async def head(self, job_id, path):
            calls.append(("head", job_id, path))
            return SimpleNamespace(content_type="image/png")

        async def open_bytes(self, job_id, path):
            calls.append(("open", job_id, path))
            if self.failure is not None:
                raise self.failure

            async def chunks():
                yield b"verified-"
                yield b"preview"

            return chunks()

        async def presign_get(self, job_id, path, **kwargs):
            calls.append(("presign", job_id, path))
            assert kwargs["content_type"] == "image/png"
            assert kwargs["cache_control"] == "no-store"
            return "https://private-bucket.example/preview?signature=private"

    class Records:
        async def get_locator(self, job_id):
            assert job_id == job.id
            return SimpleNamespace(tenant_id="default")

        async def committed_manifest(self, job_id, tenant_id):
            assert job_id == job.id and tenant_id == "default"
            return {"files": [{"path": relative, "storage_path": storage_path, "content_type": "image/png"}]}

    store = Store()
    monkeypatch.setattr(app, "_artifact_store", lambda: store)
    monkeypatch.setattr(app, "get_operational_record_store", Records)
    client = TestClient(app.app, headers={"x-api-key": "contract-secret"})
    try:
        yield SimpleNamespace(
            client=client, job=job, store=store, calls=calls, committed=committed, relative=relative, storage_path=storage_path
        )
    finally:
        client.close()


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("delivery", [None, "stream", "unknown"])
def test_stream_selector_keeps_default_redirect_and_private_delivery_headers(artifact_case, version, delivery):
    case = artifact_case
    response = case.client.get(
        f"/{version}/jobs/{case.job.id}/artifacts/{case.relative}",
        headers={"X-TP-Artifact-Delivery": delivery} if delivery else {},
        follow_redirects=False,
    )
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    if delivery == "stream":
        assert response.status_code == 200
        assert response.content == b"verified-preview"
        assert response.headers["content-type"] == "image/png"
        assert "location" not in response.headers
        assert "content-disposition" not in response.headers
        assert ("open", case.job.id, case.storage_path) in case.calls
        assert all(call[0] != "presign" for call in case.calls)
    else:
        assert response.status_code == 307
        assert response.headers["location"].startswith("https://private-bucket.example/")
        assert ("presign", case.job.id, case.storage_path) in case.calls
        assert all(call[0] != "open" for call in case.calls)


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("failure", [ArtifactStoreError, ArtifactNotFoundError])
def test_s3_stream_failure_never_falls_back_to_legacy_local_bytes(artifact_case, version, failure):
    case = artifact_case
    case.store.failure = failure("private credentials and storage path must not leak")
    response = case.client.get(
        f"/{version}/jobs/{case.job.id}/artifacts/{case.relative}", headers=_STREAM_HEADER, follow_redirects=False
    )
    expected = 404 if failure is ArtifactNotFoundError and not case.committed else 503
    assert response.status_code == expected
    assert response.json()["error"]["code"] == ("NOT_FOUND" if expected == 404 else "ARTIFACT_STORE_UNAVAILABLE")
    assert "private credentials" not in response.text
    assert "local bytes" not in response.text
    assert all(call[0] != "presign" for call in case.calls)


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("denial", ["api_key", "unknown_artifact", "foreign_tenant", "internal_storage_path"])
def test_stream_selector_does_not_bypass_artifact_authorization(artifact_case, monkeypatch, version, denial):
    case = artifact_case
    path = f"/{version}/jobs/{case.job.id}/artifacts/{case.relative}"
    headers = dict(_STREAM_HEADER)
    if denial == "api_key":
        headers["x-api-key"] = "invalid"
    elif denial == "unknown_artifact":
        path = f"/{version}/jobs/{case.job.id}/artifacts/outputs/unlisted.png"
    elif denial == "internal_storage_path":
        path = f"/{version}/jobs/{case.job.id}/artifacts/generations/committed/preview.png"
    else:
        monkeypatch.setattr(app, "PILOT_CONTROL_PLANE_ENABLED", True)
        case.job.effective_request = {"tenant_id": "tenant_b"}
        http_contract._sync_seeded_job(case.job)
        headers.update(pilot_contract._identity_headers("GET", path))
    response = case.client.get(path, headers=headers, follow_redirects=False)
    assert response.status_code == (401 if denial == "api_key" else 404)
    assert case.calls == []
