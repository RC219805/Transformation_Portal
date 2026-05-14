"""Route-inventory contract gate for the FastAPI portal origin.

The ``app.py`` origin is ~9k lines and its public surface is a small set
of FastAPI routes. The two large contract suites
(``test_app_orchestrator_runtime.py``, ``test_app_orchestrator_contract_http.py``)
exercise most behaviour, but new routes can land without anyone noticing
that no contract test was ever written for them. This file is the
*inventory* gate — it walks ``app.routes`` and asserts every method+path
pair appears in an explicit registry tagged with a coarse family.

What this file enforces (the cheap, statically-verifiable invariants):

* **Drift**: every live route must be registered, every registered route
  must still be live. Renames / additions / removals show up in the diff.
* **Family**: every route belongs to a coarse family
  (``healthcheck``, ``portal_html``, ``portal_assets``,
  ``portal_telemetry``, ``readiness``, ``presets``, ``config``,
  ``uploads``, ``jobs``). Family ``None`` is forbidden.
* **v1/v2 parity** for the dual-versioned jobs API: a route that exists
  under ``/v1/jobs/...`` must have the same shape under ``/v2/jobs/...``
  (and vice versa). Asymmetric drift is the most common regression here.

What this file deliberately does NOT enforce:

* Per-route test-file coverage. Statically grepping for `/v1/...` strings
  inside a 6k-line test file gives false negatives (parametrized paths,
  fixture indirection) and false positives (paths in docstrings). The
  honest place for that is a coverage report, not a static scan. If you
  want to know whether a route has a contract / rejection test, run the
  coverage report and inspect ``app.py`` line coverage for the handler.
"""

from __future__ import annotations

import importlib
from typing import Iterable

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.regression]


# Methods FastAPI routes against. Starlette auto-adds ``HEAD`` for every
# ``GET`` route; we ignore it.
_TRACKED_METHODS = frozenset({"GET", "POST", "PUT", "DELETE", "PATCH"})


# Allowed families — additions require human review. Keep the set small
# and don't reach for "misc" / "other".
_ALLOWED_FAMILIES = frozenset(
    {
        "healthcheck",
        "portal_html",
        "portal_assets",
        "portal_telemetry",
        "readiness",
        "presets",
        "config",
        "uploads",
        "jobs",
    }
)


# (method, path) → family. Sorted lexicographically by (path, method) so
# additions/removals land as a clean diff. Path templates use FastAPI's
# ``{name}`` / ``{name:path}`` form verbatim. The portal-video routes
# embed the literal asset filename because ``@app.get(f"/portal/video/{NAME}")``
# resolves the f-string at decoration time.
ROUTE_REGISTRY: dict[tuple[str, str], str] = {
    ("GET", "/"): "portal_html",
    ("GET", "/healthz"): "healthcheck",
    ("GET", "/portal"): "portal_html",
    ("GET", "/portal/assets/{asset_path:path}"): "portal_assets",
    ("GET", "/portal/bootstrap"): "portal_html",
    ("GET", "/portal/video/dna-portal-video-2.mp4"): "portal_assets",
    ("GET", "/ready"): "healthcheck",
    ("GET", "/v1/config-metadata"): "config",
    ("POST", "/v1/config-preview"): "config",
    ("GET", "/v1/jobs"): "jobs",
    ("POST", "/v1/jobs"): "jobs",
    ("GET", "/v1/jobs/{job_id}"): "jobs",
    ("DELETE", "/v1/jobs/{job_id}/artifacts"): "jobs",
    ("GET", "/v1/jobs/{job_id}/artifacts/{artifact_path:path}"): "jobs",
    ("POST", "/v1/jobs/{job_id}/cancel"): "jobs",
    ("GET", "/v1/jobs/{job_id}/events"): "jobs",
    ("POST", "/v1/portal/events"): "portal_telemetry",
    ("POST", "/v1/portal/rum"): "portal_telemetry",
    ("GET", "/v1/portal/video/dna-portal-video-2.mp4"): "portal_assets",
    ("GET", "/v1/presets"): "presets",
    ("GET", "/v1/readiness"): "readiness",
    ("POST", "/v1/uploads/staging"): "uploads",
    ("GET", "/v2/jobs"): "jobs",
    ("POST", "/v2/jobs"): "jobs",
    ("GET", "/v2/jobs/{job_id}"): "jobs",
    ("DELETE", "/v2/jobs/{job_id}/artifacts"): "jobs",
    ("GET", "/v2/jobs/{job_id}/artifacts/{artifact_path:path}"): "jobs",
    ("POST", "/v2/jobs/{job_id}/cancel"): "jobs",
    ("GET", "/v2/jobs/{job_id}/events"): "jobs",
}


def _iter_app_routes() -> Iterable[tuple[str, str]]:
    """Yield ``(method, path)`` tuples for every concrete HTTP route on ``app``."""
    orchestrator_app = importlib.import_module("app").app
    for route in orchestrator_app.routes:
        # Starlette also tracks Mount objects, lifespan handlers, etc;
        # we only care about HTTP-level concrete routes that have ``methods``.
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or not path:
            continue
        for method in methods:
            if method in _TRACKED_METHODS:
                yield (method, path)


@pytest.fixture(scope="module")
def discovered_routes() -> set[tuple[str, str]]:
    return set(_iter_app_routes())


class TestRouteInventory:
    def test_every_live_route_is_registered(self, discovered_routes):
        # New route landed in app.py? Add it to ROUTE_REGISTRY.
        registered = set(ROUTE_REGISTRY)
        unexpected = sorted(discovered_routes - registered)
        assert not unexpected, (
            "The following live routes are NOT in ROUTE_REGISTRY:\n  "
            + "\n  ".join(f"{m} {p}" for m, p in unexpected)
            + "\n\nAdd a registry entry tagging the route's family. New "
            "routes also need at least one happy-path contract test in "
            "test_app_orchestrator_contract_http.py and (for state-mutating "
            "routes) a 4xx test in test_app_rejection_paths.py."
        )

    def test_every_registered_route_is_live(self, discovered_routes):
        # Route was removed/renamed? Remove the registry entry too.
        registered = set(ROUTE_REGISTRY)
        stale = sorted(registered - discovered_routes)
        assert not stale, (
            "The following ROUTE_REGISTRY entries no longer match a live "
            "route:\n  " + "\n  ".join(f"{m} {p}" for m, p in stale) + "\n\nRemove or update the entry."
        )

    def test_every_route_has_an_allowed_family(self):
        # Family classification is the cheap organizational lens; missing
        # or unknown values mean the registry was edited carelessly.
        bad: list[str] = []
        for (method, path), family in ROUTE_REGISTRY.items():
            if not family:
                bad.append(f"{method} {path}: family is empty")
            elif family not in _ALLOWED_FAMILIES:
                bad.append(f"{method} {path}: family={family!r} not in _ALLOWED_FAMILIES")
        assert not bad, "Unrecognized route families:\n  " + "\n  ".join(bad)

    def test_jobs_family_routes_have_v1_v2_parity(self, discovered_routes):
        # The jobs API is dual-versioned; a v1 route landing without a
        # matching v2 route (or the reverse) is almost always a regression.
        v1_jobs = {(m, p) for (m, p) in discovered_routes if p.startswith("/v1/jobs")}
        v2_jobs = {(m, p) for (m, p) in discovered_routes if p.startswith("/v2/jobs")}
        v1_relative = {(m, p.removeprefix("/v1")) for (m, p) in v1_jobs}
        v2_relative = {(m, p.removeprefix("/v2")) for (m, p) in v2_jobs}
        only_v1 = sorted(v1_relative - v2_relative)
        only_v2 = sorted(v2_relative - v1_relative)
        assert not only_v1 and not only_v2, (
            f"Jobs API v1/v2 parity broken. "
            f"Only-v1: {only_v1!r}. Only-v2: {only_v2!r}. "
            "Add the missing route OR document the intentional asymmetry."
        )

    def test_jobs_routes_use_consistent_methods_across_versions(self, discovered_routes):
        # Pin the method set per relative path across v1/v2: if /v1/jobs
        # supports GET+POST then /v2/jobs must too (and only those).
        def _method_map(prefix: str) -> dict[str, set[str]]:
            result: dict[str, set[str]] = {}
            for method, path in discovered_routes:
                if not path.startswith(prefix):
                    continue
                relative = path.removeprefix(prefix)
                result.setdefault(relative, set()).add(method)
            return result

        v1 = _method_map("/v1/jobs")
        v2 = _method_map("/v2/jobs")
        mismatches = []
        for relative, v1_methods in v1.items():
            v2_methods = v2.get(relative, set())
            if v1_methods != v2_methods:
                mismatches.append(f"  /jobs{relative}: v1={sorted(v1_methods)} v2={sorted(v2_methods)}")
        assert not mismatches, "Jobs API v1 and v2 expose different method sets:\n" + "\n".join(mismatches)
