"""Stable string constants for APEX gate failure / warning codes.

These codes form the contract between the Lux Depth V3 orchestrator (which
emits them in run cards, manifests, and evidence sidecars) and the
``transformation_portal.evals`` layer (which consumes them in the APEX
promotion bundle). Locating them here — alongside the orchestrator that emits
them — keeps the dependency direction one-way: ``evals`` imports from
``lux_depth_v3``, not the reverse.

The ``apex_evidence_bundle`` module re-exports these names for backward
compatibility, so existing imports from
``transformation_portal.evals.apex_evidence_bundle`` continue to work.
"""

from __future__ import annotations

#: Strict-gate failure code raised when masks exist and every implemented
#: Materials V3 pixel op is blocked, leaving ``applied_ops_count == 0``.
APEX_MATERIALS_PIXEL_OPS_EMPTY = "APEX_MATERIALS_PIXEL_OPS_EMPTY"

#: Non-fatal warning code emitted when the only blocker on every implemented
#: pixel op is ``below_confidence_threshold``. The orchestrator emits the
#: output without applying pixel ops and surfaces this code on the run card
#: instead of failing the strict gate.
APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE = "APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE"

__all__ = [
    "APEX_MATERIALS_PIXEL_OPS_EMPTY",
    "APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE",
]
