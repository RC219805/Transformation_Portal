"""Futuristic failure prevention utilities for the LUT automation suite.

This module introduces a playful-yet-practical :class:`PropheticOrchestrator`
that mirrors the repository's taste for evocative metaphors.  The orchestrator
wraps two lightweight collaborators:

``CausalityEngine``
    Extracts a normalized list of weak points from a predicted failure
    description.  The class accepts flexible input structures so the helper can
    plug into dashboards or bespoke forecasting code without ceremony.

``QuantumProbabilityField``
    Tracks how assertively the codebase has reinforced each weak point.  The
    implementation is intentionally deterministic—the "quantum" naming is
    purely aesthetic—yet it provides ergonomic helpers for tests and
    instrumentation.

The orchestrator pulls the two utilities together to proactively generate
"temporal antibodies" (countermeasures) that neutralize the predicted failure
before it can manifest.  While whimsical, the helpers are fully typed and keep
state that downstream tooling can inspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence


@dataclass(frozen=True)
class WeakPoint:
    """Represents a vulnerable component within the causal chain."""

    component: str
    failure_mode: str
    severity: Optional[str] = None
    metadata: Optional[Mapping[str, object]] = None

    def signature(self) -> str:
        """Return a stable identifier for reporting and debugging."""

        if self.severity:
            return f"{self.component}:{self.failure_mode}:{self.severity}"
        return f"{self.component}:{self.failure_mode}"


@dataclass(frozen=True)
class TemporalAntibody:
    """Represents a preventative measure aimed at a specific weak point."""

    target: WeakPoint
    countermeasure: str
    confidence: float


class CausalityEngine:
    """Trace the underlying causes of a predicted failure.

    The engine accepts flexible input (strings, mappings, dataclasses, or raw
    sequences) and always returns a list of :class:`WeakPoint` instances.  This
    keeps higher level orchestration code agnostic to the exact forecasting
    format.
    """

    def trace_failure_origins(self, predicted_failure: object) -> List[WeakPoint]:
        """Normalize *predicted_failure* into a list of weak points."""

        weak_entries: Iterable[object]

        if isinstance(predicted_failure, Mapping):
            weak_entries = predicted_failure.get("weak_points") or predicted_failure.get("causes") or ()
        elif isinstance(predicted_failure, Sequence) and not isinstance(predicted_failure, (str, bytes, bytearray)):
            weak_entries = predicted_failure
        else:
            weak_entries = [predicted_failure]

        normalized: List[WeakPoint] = []
        for entry in weak_entries:
            normalized.append(self._normalize_entry(entry))
        return normalized

    def _normalize_entry(self, entry: object) -> WeakPoint:
        if isinstance(entry, WeakPoint):
            return entry
        if isinstance(entry, Mapping):
            component = str(
                entry.get("component")
                or entry.get("system")
                or entry.get("target")
                or entry.get("name")
                or "unknown"
            ).strip()
            failure_mode = str(
                entry.get("failure_mode")
                or entry.get("issue")
                or entry.get("mode")
                or entry.get("description")
                or entry.get("reason")
                or "unspecified"
            ).strip()
            severity = entry.get("severity") or entry.get("criticality")
            metadata_keys = {"component", "system", "target", "name", "failure_mode", "issue", "mode", "description", "reason", "severity", "criticality"}
            metadata: MutableMapping[str, object] = {
                str(key): value
                for key, value in entry.items()
                if key not in metadata_keys
            }
            return WeakPoint(
                component=component or "unknown",
                failure_mode=failure_mode or "unspecified",
                severity=str(severity) if severity is not None else None,
                metadata=dict(metadata) or None,
            )
        if isinstance(entry, str):
            component, _, failure_mode = entry.partition(":")
            component = component.strip() or "unknown"
            failure_mode = failure_mode.strip() or "unspecified"
            if not failure_mode:
                failure_mode = "unspecified"
            return WeakPoint(component=component, failure_mode=failure_mode)
        return WeakPoint(component="unknown", failure_mode=str(entry))


class QuantumProbabilityField:
    """Maintain success probabilities for each weak point."""

    def __init__(self) -> None:
        self._branch_probabilities: MutableMapping[WeakPoint, float] = {}

    def strengthen_reality_branch(
        self, weak_point: WeakPoint, *, success_probability: float
    ) -> float:
        """Record a high-confidence outcome for *weak_point*.

        Parameters
        ----------
        weak_point:
            The vulnerability that is being mitigated.
        success_probability:
            A value between 0 and 1 indicating how robust the mitigation is.
        """

        if not 0.0 <= success_probability <= 1.0:
            raise ValueError("success_probability must be between 0 and 1")
        current = self._branch_probabilities.get(weak_point, 0.0)
        updated = max(current, success_probability)
        self._branch_probabilities[weak_point] = updated
        return updated

    def probability_of(self, weak_point: WeakPoint) -> float:
        """Return the recorded success probability for *weak_point*."""

        return self._branch_probabilities.get(weak_point, 0.0)

    def snapshot(self) -> Mapping[WeakPoint, float]:
        """Return a shallow copy of the probability ledger."""

        return dict(self._branch_probabilities)


class PropheticOrchestrator:
    """Orchestrate proactive fixes for predicted failures."""

    def __init__(
        self,
        timeline_analyzer: Optional[CausalityEngine] = None,
        probability_weaver: Optional[QuantumProbabilityField] = None,
    ) -> None:
        self.timeline_analyzer = timeline_analyzer or CausalityEngine()
        self.probability_weaver = probability_weaver or QuantumProbabilityField()
        self._deployed_antibodies: List[TemporalAntibody] = []

    def prevent_future_failure(self, predicted_failure: object) -> List[TemporalAntibody]:
        """Neutralize *predicted_failure* before it manifests.

        The method performs three steps:

        1. Use :class:`CausalityEngine` to derive the causal chain.
        2. Reinforce each weak point by setting its success probability to 0.9999.
        3. Generate and deploy temporal antibodies describing the preventative work.

        Returns
        -------
        list[TemporalAntibody]
            The countermeasures that were deployed.
        """

        causal_chain = self.timeline_analyzer.trace_failure_origins(predicted_failure)
        for weak_point in causal_chain:
            self.probability_weaver.strengthen_reality_branch(
                weak_point, success_probability=0.9999
            )
        antibodies = self.generate_anti_patterns(predicted_failure, causal_chain)
        self.deploy_temporal_antibodies(antibodies)
        return antibodies

    def generate_anti_patterns(
        self, predicted_failure: object, causal_chain: Optional[Sequence[WeakPoint]] = None
    ) -> List[TemporalAntibody]:
        """Create actionable countermeasures for each weak point."""

        if causal_chain is None:
            causal_chain = self.timeline_analyzer.trace_failure_origins(predicted_failure)

        antibodies: List[TemporalAntibody] = []
        for weak_point in causal_chain:
            severity = (weak_point.severity or "medium").lower()
            if severity in {"critical", "high"}:
                confidence = 0.995
            elif severity in {"low", "minor"}:
                confidence = 0.9
            else:
                confidence = 0.96

            countermeasure = (
                f"Install anticipatory guardrails for {weak_point.component} to neutralize "
                f"{weak_point.failure_mode}."
            )
            antibodies.append(
                TemporalAntibody(
                    target=weak_point,
                    countermeasure=countermeasure,
                    confidence=confidence,
                )
            )
        return antibodies

    def deploy_temporal_antibodies(self, antibodies: Iterable[TemporalAntibody]) -> None:
        """Record the deployed antibodies for future inspection."""

        self._deployed_antibodies.extend(antibodies)

    @property
    def deployed_antibodies(self) -> List[TemporalAntibody]:
        """Return a copy of the deployed antibodies list."""

        return list(self._deployed_antibodies)
