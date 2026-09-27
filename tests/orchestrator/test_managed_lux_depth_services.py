"""Unified HTTP workflows retain exact authority through real managed services.

Only native inference is controlled. Postgres admission, Redis delivery, the
external worker, verification, publication fencing and artifact downloads run
normally with the legacy flags disabled. This does not prove native quality.
"""

from __future__ import annotations

import pytest

from tests.orchestrator import test_managed_photography_services as inference_services
from tests.orchestrator import test_managed_v6_services as process_services

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
request_case = inference_services.request_case
service_urls = process_services.service_urls


@pytest.mark.parametrize("api_prefix", ["/v1", "/v2"])
@pytest.mark.parametrize("workflow", ["process", "infer"])
async def test_unified_external_worker_preserves_admission_and_verified_delivery(
    service_urls, request_case, tmp_path, monkeypatch, api_prefix, workflow
):
    if workflow == "process":
        await process_services._managed_v6_round_trip(
            service_urls, request_case, tmp_path, monkeypatch, api_prefix, unified=True
        )
    else:
        monkeypatch.setenv("TP_PHOTOGRAPHY_TEST_DATABASE_URL", service_urls[0])
        await inference_services._managed_photography_round_trip(request_case, tmp_path, monkeypatch, api_prefix, unified=True)


@pytest.mark.parametrize("revocation", ["canceled", "expired", "epoch", "holder"])
async def test_unified_verified_output_cannot_override_revoked_dispatch_authority(
    service_urls, request_case, tmp_path, monkeypatch, revocation
):
    await process_services._managed_v6_fenced_publication(
        service_urls, request_case, tmp_path, monkeypatch, revocation, unified=True
    )
