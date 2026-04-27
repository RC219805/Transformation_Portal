"""Schema name vocabulary for orchestrator v1 envelopes.

Every JSON response from the orchestrator includes a top-level ``schema`` field
identifying its wire contract. This module promotes those literal strings to a
typed ``Literal`` so handlers cannot drift onto unregistered names.

The list is exhaustive as of the inventory done for PR A. To add a new schema:
extend the ``Literal`` and document the route(s) that emit it.
"""

from __future__ import annotations

from typing import Literal

SchemaName = Literal[
    "tp.orchestrator.config_metadata.v1",
    "tp.orchestrator.config_preview.v1",
    "tp.orchestrator.error.v1",
    "tp.orchestrator.job.v1",
    "tp.orchestrator.job_status.v1",
    "tp.orchestrator.jobs.v1",
    "tp.orchestrator.portal_event.v1",
    "tp.orchestrator.portal_rum.v1",
    "tp.orchestrator.portal_rum_ingest.v1",
    "tp.orchestrator.presets.v1",
    "tp.orchestrator.readiness.v1",
    "tp.orchestrator.upload_staging.v1",
]
"""Closed set of orchestrator v1 wire-contract identifiers."""

ERROR_SCHEMA: Literal["tp.orchestrator.error.v1"] = "tp.orchestrator.error.v1"
"""The universal error schema used by `_error_response` in app.py."""
