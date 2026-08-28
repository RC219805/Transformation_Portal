"""Single-resolution invocation contract shared by ``plan`` and ``run``.

Implements the P0-1 repair (issue #2065): one immutable ``ResolvedInvocation``
built by exactly one license-enforcing model resolution, which the ``--plan``
CLI mode serializes and the run path consumes without re-resolution.

Invariant (program-wide): ``plan`` and ``run`` share the same resolution path,
and ``run`` consumes the exact resolved object ``plan`` emits. Consumers must
read the authoritative contract from this object instead of re-deriving model
identity from compatibility fields such as ``config.model_variant`` — the
legacy variant mapping can otherwise resolve a commercial selection back to a
research-licensed model.

The plan distinguishes:
- ``planned_backend``: the backend the plan selects,
- ``candidate_fallback_chain``: the permitted fallback edges,
- ``executed_backend``: runtime-only; recorded in manifests during execution
  and deliberately absent from this object and its serialization, because
  backend availability cannot be fully resolved before runtime preflight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from .model_resolution import ModelRequest, ResolvedModel, resolve_model_contract

if TYPE_CHECKING:
    from .config import EnhanceConfig

RESOLVED_INVOCATION_SCHEMA = "tp.lux.resolved_invocation.v1"


@dataclass(frozen=True)
class LicenseAcknowledgements:
    """Individually attributable license acknowledgements from the request."""

    non_commercial_ok: bool
    apple_depth_pro_research: bool
    research_tools: bool

    def to_payload(self) -> Dict[str, Any]:
        return {
            "non_commercial_ok": self.non_commercial_ok,
            "apple_depth_pro_research": self.apple_depth_pro_research,
            "research_tools": self.research_tools,
        }


@dataclass(frozen=True)
class ResolvedInvocation:
    """Immutable, authoritative result of the single shared resolution pass.

    ``resolved_model`` is the license-enforced model contract for the DA3
    backend family; it is ``None`` for backends whose model identity is
    governed outside the DA3 registry (e.g. ``depth_pro``), whose license
    gates run in their existing validation paths.
    """

    schema: str
    planned_backend: str
    candidate_fallback_chain: Tuple[str, ...]
    resolved_model: Optional[ResolvedModel]
    license_acknowledgements: LicenseAcknowledgements
    license_enforced: bool
    quality_tier: str
    preset_requested: Optional[str]
    preset_resolved: Optional[str]
    stages: Tuple[str, ...]
    requested_artifacts: Tuple[str, ...]
    input_dir: str
    input_files: Tuple[str, ...]
    config_fingerprint_sha256: str
    warnings: Tuple[str, ...]

    def to_payload(self) -> Dict[str, Any]:
        """Serialize to a plain payload for canonical JSON emission.

        ``executed_backend`` must never appear here — it is runtime state
        recorded in manifests, not plannable state.
        """
        model_payload: Optional[Dict[str, Any]] = None
        if self.resolved_model is not None:
            resolved = self.resolved_model
            spec = resolved.spec
            usage_class = getattr(spec, "usage_class", None)
            model_payload = {
                "requested_selector": resolved.requested_selector,
                "canonical_key": resolved.canonical_key,
                "repo_id": getattr(spec, "repo_id", None),
                "revision": resolved.revision,
                "license_id": getattr(spec, "license_id", None),
                "usage_class": getattr(usage_class, "value", usage_class),
                "requires_non_commercial_ok": bool(getattr(spec, "requires_non_commercial_ok", False)),
                "accelerator_kind": getattr(resolved.accelerator_kind, "value", str(resolved.accelerator_kind)),
                "legacy_model_variant_name": resolved.legacy_model_variant_name,
            }
        return {
            "schema": self.schema,
            # ADR-051 conflict-matrix note: this plan surface is a bounded
            # pre-designation spike; its serialization is provisional until
            # the Plan/DAG representation row of ADR-051 is decided.
            "stability": "provisional",
            "planned_backend": self.planned_backend,
            "candidate_fallback_chain": list(self.candidate_fallback_chain),
            "resolved_model": model_payload,
            "license_acknowledgements": self.license_acknowledgements.to_payload(),
            "license_evaluation": {
                "enforced": self.license_enforced,
                "status": ("allowed" if self.license_enforced else "deferred_to_backend_gates"),
            },
            "quality_tier": self.quality_tier,
            "preset_requested": self.preset_requested,
            "preset_resolved": self.preset_resolved,
            "stages": list(self.stages),
            "requested_artifacts": list(self.requested_artifacts),
            "input_dir": self.input_dir,
            "input_files": list(self.input_files),
            "config_fingerprint_sha256": self.config_fingerprint_sha256,
            "warnings": list(self.warnings),
        }

    def to_canonical_json(self) -> str:
        from ..ingest.canonical_json import canonicalize_json

        return canonicalize_json(self.to_payload()).decode("utf-8")


def resolved_invocation_schema_path() -> Path:
    """Path to the provisional tp.lux.resolved_invocation.v1 JSON Schema."""
    return Path(__file__).resolve().parents[3] / "schemas" / "lux" / "resolved_invocation.schema.json"


def validate_resolved_invocation_payload(payload: Dict[str, Any]) -> None:
    """Validate a serialized plan against the provisional v1 JSON Schema.

    Raises jsonschema.ValidationError on mismatch. Consumers parsing plan
    JSON from an untrusted producer MUST validate before use; note that a
    schema-valid payload still carries NO licensing authority — the model
    contract must be revalidated via
    ``model_resolution.validate_authoritative_model_contract`` at every
    consumption boundary.
    """
    import json

    import jsonschema

    with open(resolved_invocation_schema_path(), "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.validate(payload, schema)


def authoritative_model_contract(config: Any) -> Optional[ResolvedModel]:
    """Return the authoritative model contract carried on a config, if any.

    Consumers (ConfigResolver, depth backends) call this instead of
    re-resolving model identity from compatibility fields.
    """
    invocation = getattr(config, "resolved_invocation", None)
    if invocation is None:
        return None
    return getattr(invocation, "resolved_model", None)


def _planned_stages(config: "EnhanceConfig") -> Tuple[str, ...]:
    # Mirrors the production execution order inside the orchestrator:
    # depth runs first, Materials V3 and PBR run within the depth stage's
    # scope (materials before PBR), then V2 enhancement, then optional
    # scene reconstruction, then output/manifest assembly.
    stages: List[str] = ["preprocess", "depth"]
    if getattr(config, "enable_materials_v3", False):
        stages.append("materials_v3")
    if getattr(config, "generate_pbr", False):
        stages.append("pbr")
    if getattr(config, "enable_v2", False) and getattr(config, "v2_preset", None):
        stages.append("v2")
    if getattr(config, "enable_reconstruction", False):
        stages.append("reconstruction")
    stages.append("output")
    return tuple(stages)


def _requested_artifacts(config: "EnhanceConfig") -> Tuple[str, ...]:
    # Only artifacts the pipeline actually produces are listed; the inert
    # emit_marketing/emit_report switches surface as warnings instead
    # (dispositions tracked in issues #2067/#2068).
    artifacts: List[str] = [
        "depth_u16_png",
        "depth_metadata_json",
        "combined_manifest_json",
        "batch_manifest_json",
    ]
    if getattr(config, "save_float_depth", False):
        artifacts.append("depth_float_npy")
    if getattr(config, "enable_materials_v3", False):
        artifacts.append("materials_v3_masks")
    if getattr(config, "generate_pbr", False):
        artifacts.append("pbr_maps")
    if getattr(config, "enable_v2", False) and getattr(config, "v2_preset", None):
        artifacts.append("v2_enhanced_image")
    if getattr(config, "enable_reconstruction", False):
        artifacts.append("reconstruction_bundle")
    if getattr(config, "emit_run_card", False):
        artifacts.append("run_card")
    if getattr(config, "emit_master16", False) or getattr(config, "emit_upscaled16", False):
        # The two flags currently act as one joint bit-depth switch; the
        # honest plannable artifact is the 16-bit intermediate lane.
        artifacts.append("bit_depth_16_intermediates")
    return tuple(artifacts)


def _plan_warnings(config: "EnhanceConfig") -> Tuple[str, ...]:
    warnings: List[str] = []
    if getattr(config, "emit_marketing", False):
        warnings.append("--emit-marketing currently produces no deliverable (disposition tracked in issue #2067)")
    if getattr(config, "emit_master16", False) and getattr(config, "emit_upscaled16", False):
        warnings.append(
            "--emit-master16 and --emit-upscaled16 currently act as a single "
            "bit-depth switch (disposition tracked in issue #2068)"
        )
    return tuple(warnings)


def build_resolved_invocation(
    config: "EnhanceConfig",
    input_dir: Path,
    input_files: Sequence[Path],
) -> ResolvedInvocation:
    """Perform the single shared resolution pass and freeze its result.

    This is THE license-enforcing model resolution for the DA3 backend
    family: ``ModelLicenseError`` / ``UnknownModelError`` raised here are the
    same errors a real run would surface, and callers (both ``--plan`` and
    ``run``) must not resolve again.

    Performs no model loading, no inference, and no filesystem writes.
    """
    # Local imports keep this module import-light and cycle-free.
    from ..depth.backends.registry import DepthBackendRegistry
    from .config_resolver import compute_config_fingerprint
    from .pipeline_coordinator import (
        normalize_backend_id,
        resolve_requested_backend,
        resolve_runtime_backend_chain,
    )

    # Backend selection parity (P0-1): use the SAME platform-aware requested-
    # backend resolution and registry validation the runtime uses, so a plan
    # cannot select a backend the run would not (e.g. Apple Silicon Depth Pro
    # opt-in), advertise an unregistered backend, or skip a license gate the
    # runtime registry enforces.
    explicit_backend_request = normalize_backend_id(getattr(config, "depth_backend", None)) is not None
    planned_backend = resolve_requested_backend(getattr(config, "depth_backend", None), config)
    planned_backend = DepthBackendRegistry().validate_backend_request(planned_backend, config)

    # Candidate-chain truth: an explicit backend request is strict at runtime
    # (no silent downgrade), so the plan must not advertise fallback edges
    # startup will never attempt. Only a defaulted selection carries the
    # configured operational chain.
    if explicit_backend_request:
        candidate_chain: Tuple[str, ...] = (planned_backend,)
    else:
        candidate_chain = tuple(resolve_runtime_backend_chain(planned_backend, config))

    resolved_model: Optional[ResolvedModel] = None
    if planned_backend == "da3":
        resolved_model = resolve_model_contract(
            ModelRequest(
                model_key=getattr(config, "model_key", None),
                raw_model_id=getattr(config, "raw_model_id", None),
                model_variant=getattr(config, "model_variant", None),
                use_coreml_backend=bool(getattr(config, "use_coreml_backend", False)),
                non_commercial_ok=bool(getattr(config, "non_commercial_ok", False)),
                enforce_license=True,
            )
        )
    # License evaluation happened for every planned backend: the DA3 family
    # through resolve_model_contract, every other backend through the
    # registry's license validation above.
    license_enforced = True

    # ONE identity across plan, cache, manifests, and run cards: the plan
    # fingerprint is the SAME ConfigFingerprint algorithm the runtime uses,
    # with the authoritative resolved identity flowing into it (the runtime
    # path picks the identity up from the carried invocation, so plan and
    # runtime fingerprints are equal for the same contract).
    fingerprint = compute_config_fingerprint(config, resolved_model_contract=resolved_model)
    fingerprint_sha256 = fingerprint.to_sha256()
    preset = getattr(config, "preset", None)
    preset_resolved = preset.value if preset is not None else f"quality_tier:{config.quality_tier}"

    input_dir_resolved = Path(input_dir)
    relative_files: List[str] = []
    for file_path in input_files:
        candidate = Path(file_path)
        try:
            relative_files.append(candidate.relative_to(input_dir_resolved).as_posix())
        except ValueError:
            relative_files.append(candidate.as_posix())

    return ResolvedInvocation(
        schema=RESOLVED_INVOCATION_SCHEMA,
        planned_backend=planned_backend,
        candidate_fallback_chain=candidate_chain,
        resolved_model=resolved_model,
        license_acknowledgements=LicenseAcknowledgements(
            non_commercial_ok=bool(getattr(config, "non_commercial_ok", False)),
            apple_depth_pro_research=bool(getattr(config, "accept_apple_depth_pro_research_license", False)),
            research_tools=bool(getattr(config, "accept_research_tools_license", False)),
        ),
        license_enforced=license_enforced,
        quality_tier=str(getattr(config, "quality_tier", "standard")),
        preset_requested=getattr(config, "preset_requested", None),
        preset_resolved=preset_resolved,
        stages=_planned_stages(config),
        requested_artifacts=_requested_artifacts(config),
        input_dir=str(input_dir_resolved),
        input_files=tuple(sorted(relative_files)),
        config_fingerprint_sha256=fingerprint_sha256,
        warnings=_plan_warnings(config),
    )
