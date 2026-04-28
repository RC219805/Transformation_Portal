"""Typed request/response models for the orchestrator's config/presets routes.

Three routes get ``response_model=`` annotations in PR D of the Phase 1.2
sequence:

- ``GET  /v1/presets``         → ``ApiEnvelope[PresetsData]``
  (``tp.orchestrator.presets.v1``)
- ``GET  /v1/config-metadata`` → ``ApiEnvelope[ConfigMetadataData]``
  (``tp.orchestrator.config_metadata.v1``)
- ``POST /v1/config-preview``  → ``ApiEnvelope[ConfigPreviewData]``
  (``tp.orchestrator.config_preview.v1``)

Every route handler returns ``JSONResponse(_api_envelope(...))`` directly,
so ``response_model`` is **OpenAPI-only** — no runtime serialisation by
FastAPI. Wire shapes are unchanged by this PR; the models exist to type
the OpenAPI schema and provide a stable surface for typed callers.

A note on conservative typing: ``_lux_config_metadata`` (app.py:3869) and
``_build_lux_config_preview`` (app.py:4250) return deep, churning,
pipeline-specific dicts. Fully typing every nested shape would chain this
module to large swaths of pipeline logic and force a model bump on every
internal change. The pragmatic choice here is to type the **stable
top-level shape** and use ``dict[str, Any]`` (with ``extra="allow"``) for
the inner pipeline-specific structures.

``ConfigPreviewRequest`` is defined for type-discipline / future use but is
**not yet wired** as the handler parameter — same reasoning as
``JobCreateRequest`` (Phase 1.2 PR C). Wiring it would shift FastAPI's
422 to the orchestrator's 400 envelope (the existing
``RequestValidationError`` handler at app.py:7879 already does that
conversion for /v[12]/* paths), but the conversion drops specific
error-reason codes. That trade-off deserves its own PR.
"""

from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field

from transformation_portal.api.v1.envelopes import ApiEnvelope

# ---------------------------------------------------------------------------
# /v1/presets payload (tp.orchestrator.presets.v1)
# ---------------------------------------------------------------------------


class PresetEntry(BaseModel):
    """A single preset within a pipeline's preset catalog.

    Mirrors the dicts in ``app.py:PRESET_CATALOG`` (line 1061). The
    ``recommended_args`` payload is pipeline-specific and modelled as
    ``dict[str, Any]`` rather than a typed sub-class — the keys are the
    union of every pipeline-specific dispatch arg, and adding new presets
    or pipeline knobs shouldn't require a model bump.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    label: str
    stability: str
    description: str
    is_research: bool
    recommended_args: dict[str, Any] = Field(default_factory=dict)
    advanced_sections: list[str] = Field(default_factory=list)


class PipelinePresetGroup(BaseModel):
    """A pipeline plus its presets, used by the multi-pipeline shape of
    ``/v1/presets`` (no ``pipeline`` query param supplied)."""

    model_config = ConfigDict(extra="forbid")

    pipeline: str
    presets: list[PresetEntry]


class PresetsAllPipelinesData(BaseModel):
    """Payload when ``/v1/presets`` is called WITHOUT ``?pipeline=``."""

    model_config = ConfigDict(extra="forbid")

    pipelines: list[PipelinePresetGroup]


class PresetsSinglePipelineData(BaseModel):
    """Payload when ``/v1/presets`` is called WITH ``?pipeline=foo``."""

    model_config = ConfigDict(extra="forbid")

    pipeline: str
    presets: list[PresetEntry]


# Both shapes share the same envelope schema (tp.orchestrator.presets.v1).
# The Union's order matters for Pydantic discriminator-less validation:
# we put PresetsSinglePipelineData first because its required fields
# (``pipeline`` + ``presets``) don't overlap with the other shape's
# required field (``pipelines``), so unambiguous.
PresetsData = Union[PresetsSinglePipelineData, PresetsAllPipelinesData]


# ---------------------------------------------------------------------------
# /v1/config-metadata payload (tp.orchestrator.config_metadata.v1)
# ---------------------------------------------------------------------------


class ConfigMetadataData(BaseModel):
    """Payload for ``tp.orchestrator.config_metadata.v1``.

    Top-level shape from ``_lux_config_metadata`` (app.py:3869). The nested
    structures (``fields``, ``backend_catalog``, ``model_catalog``,
    ``debug_bundle_policy``) carry pipeline-specific shapes that churn
    frequently — they're modelled as ``dict[str, Any]`` rather than
    fully-typed nested classes. ``extra="allow"`` accommodates new
    top-level fields added by future PRs without a model bump.

    Documented top-level fields are based on assertions in
    ``tests/test_app_orchestrator_contract_http.py::test_config_metadata_contract_for_lux_depth_pipeline``.
    """

    model_config = ConfigDict(extra="allow")

    pipeline: str
    advanced_sections: list[str] = Field(default_factory=list)
    estimate_bands: dict[str, list[str]] = Field(default_factory=dict)
    backend_catalog: dict[str, Any] = Field(default_factory=dict)
    # The following are present in current handler output but kept Optional
    # because not every pipeline necessarily emits them.
    fields: dict[str, Any] | None = None
    model_catalog: dict[str, Any] | None = None
    debug_bundle_policy: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# /v1/config-preview payload (tp.orchestrator.config_preview.v1)
# ---------------------------------------------------------------------------


class ConfigPreviewData(BaseModel):
    """Payload for ``tp.orchestrator.config_preview.v1``.

    Output shape from ``_build_config_preview`` (app.py:5190), which
    delegates per pipeline to ``_build_lux_config_preview`` or
    ``_build_archive_config_preview``. Each pipeline produces a different
    subset of top-level fields; nothing is universally required.

    The fields documented here are common across pipelines, sourced from
    contract assertions in
    ``tests/test_app_orchestrator_contract_http.py::test_lux_config_preview_returns_execution_args_and_repair_warning_for_repo_local_shorthand``.
    All are Optional; ``extra="allow"`` accommodates pipeline-specific keys.
    """

    model_config = ConfigDict(extra="allow")

    pipeline: str | None = None
    field_errors: list[dict[str, Any]] | None = None
    field_warnings: list[dict[str, Any]] | None = None
    normalized_args: dict[str, Any] | None = None
    execution_args: dict[str, Any] | None = None
    readiness: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Request models (defined for OpenAPI / future use; not yet wired)
# ---------------------------------------------------------------------------


class ConfigPreviewRequest(BaseModel):
    """Typed request body for ``POST /v1/config-preview``.

    NOT YET wired as the handler parameter — see module docstring for the
    error-reason-code trade-off. Defined so a future PR can adopt it.

    The ``args`` payload is pipeline-specific (validated downstream by
    ``_build_*_config_preview``); kept as ``dict[str, Any]`` rather than a
    discriminated union.
    """

    model_config = ConfigDict(extra="allow")

    pipeline: str
    args: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Envelope aliases — what FastAPI sees in response_model=
# ---------------------------------------------------------------------------


PresetsEnvelope = ApiEnvelope[PresetsData]
"""Convenience alias for the typed envelope wrapping the presets payload."""

ConfigMetadataEnvelope = ApiEnvelope[ConfigMetadataData]
"""Convenience alias for the typed envelope wrapping config-metadata data."""

ConfigPreviewEnvelope = ApiEnvelope[ConfigPreviewData]
"""Convenience alias for the typed envelope wrapping config-preview data."""

__all__ = [
    "ConfigMetadataData",
    "ConfigMetadataEnvelope",
    "ConfigPreviewData",
    "ConfigPreviewEnvelope",
    "ConfigPreviewRequest",
    "PipelinePresetGroup",
    "PresetEntry",
    "PresetsAllPipelinesData",
    "PresetsData",
    "PresetsEnvelope",
    "PresetsSinglePipelineData",
]
