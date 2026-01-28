"""Dreaming pipeline abstractions.

This module models an asynchronous dreaming loop that periodically invents new
techniques while no foreground work is scheduled.  The design is intentionally
light-weight so it can be used in tests and interactive sessions without
requiring any of the large dependencies that power the production rendering
pipelines in this repository.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Iterable, List, Sequence


@dataclass
class DreamSequence:
    """A simple container describing the outcome of a generated dream."""

    idea: str
    coherence: float
    failure_modes: List[str] = field(default_factory=list)

    def is_coherent(self, threshold: float = 0.6) -> bool:
        """Return ``True`` when the dream appears actionable."""

        return self.coherence >= threshold


@dataclass
class Technique:
    """Represents a crystallised idea that can be integrated into a pipeline."""

    name: str
    description: str


class StandardPipeline:
    """Collects techniques discovered by the dreaming loop."""

    def __init__(self) -> None:
        self._integrated: List[Technique] = []
        self._pending: List[Technique] = []

    def integrate(self, technique: Technique) -> None:
        """Register a new technique and mark it for future processing."""

        self._integrated.append(technique)
        self._pending.append(technique)

    def detect_conflicts(self, technique: Technique) -> List[Technique]:
        """Return techniques that would conflict with ``technique``."""

        return [
            existing for existing in self._integrated if existing.name == technique.name
        ]

    def resolve_conflicts(self, technique: Technique) -> None:
        """Remove conflicting techniques that are superseded by ``technique``."""

        self._integrated = [
            existing for existing in self._integrated if existing.name != technique.name
        ]
        self._pending = [
            existing for existing in self._pending if existing.name != technique.name
        ]

    def active_jobs(self) -> bool:
        """Whether the pipeline has techniques waiting to be processed."""

        return bool(self._pending)

    def complete_next_job(self) -> Technique | None:
        """Pop the next technique awaiting attention."""

        if not self._pending:
            return None
        return self._pending.pop(0)

    @property
    def techniques(self) -> Sequence[Technique]:
        return tuple(self._integrated)


class DreamState:
    """Turns coherent dream sequences into implementable techniques."""

    def crystallize(self, dream_sequence: DreamSequence) -> Technique:
        return Technique(
            name=f"Technique::{dream_sequence.idea}",
            description=(
                "A structured approach distilled from the subconscious "
                f"exploration of {dream_sequence.idea}."
            ),
        )


class InnovationEngine:
    """Generates dream sequences asynchronously."""

    def __init__(self) -> None:
        self._iteration = 0

    async def generate_vision(self) -> DreamSequence:
        # Sleep very briefly so that ``sleep_cycle`` yields control in event
        # loops during tests.
        await asyncio.sleep(0)
        idea = f"concept_{self._iteration}"
        coherence = 0.5 + (self._iteration % 3) * 0.25
        failure_modes: List[str] = []
        if coherence < 0.6:
            failure_modes = [f"insufficient_clarity_{self._iteration}"]
        self._iteration += 1
        return DreamSequence(
            idea=idea, coherence=coherence, failure_modes=failure_modes
        )


class BoundaryKnowledge:
    """Tracks the limits discovered through failed dreams."""

    def __init__(self) -> None:
        self._constraints: List[str] = []

    @property
    def constraints(self) -> Sequence[str]:
        return tuple(self._constraints)

    def expand(self, failure_modes: Iterable[str]) -> None:
        for mode in failure_modes:
            if mode not in self._constraints:
                self._constraints.append(mode)


class DreamingPipeline:
    def __init__(self) -> None:
        self.conscious_processor = StandardPipeline()
        self.unconscious_processor = DreamState()
        self.rem_cycles = InnovationEngine()
        self.boundary_knowledge = BoundaryKnowledge()

    def active_jobs(self) -> bool:
        return self.conscious_processor.active_jobs()

    async def sleep_cycle(self) -> None:
        while not self.active_jobs():
            dream_sequence = await self.rem_cycles.generate_vision()

            if dream_sequence.is_coherent():
                new_technique = self.unconscious_processor.crystallize(dream_sequence)
                self.conscious_processor.integrate(new_technique)

            self.boundary_knowledge.expand(dream_sequence.failure_modes)


@dataclass
class ArchitecturalHypothesis:
    """Represents a speculative pipeline improvement."""

    technique: Technique
    mutation_notes: str
    originating_dream: DreamSequence


@dataclass
class EvaluationResult:
    """Stores the outcome of testing a hypothesis."""

    hypothesis: ArchitecturalHypothesis
    score: float
    diagnostics: Dict[str, float]


class QuantumOptimizer:
    """Transforms the dreaming pipeline into an autonomous optimizer."""

    def __init__(
        self,
        pipeline: DreamingPipeline,
        *,
        max_iterations: int = 25,
        exploration_batch_size: int = 3,
        target_score: float | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.max_iterations = max_iterations
        self.exploration_batch_size = exploration_batch_size
        self.target_score = target_score
        self._iteration = 0
        self._best_result: EvaluationResult | None = None
        self._history: List[EvaluationResult] = []
        self._technique_scores: Dict[str, float] = {}

    @property
    def history(self) -> Sequence[EvaluationResult]:
        """Immutable view of all evaluation results."""

        return tuple(self._history)

    def convergence_achieved(self) -> bool:
        """Return ``True`` when optimisation can safely conclude."""

        if (
            self.target_score is not None
            and self._best_result
            and self._best_result.score >= self.target_score
        ):
            return True
        return self._iteration >= self.max_iterations

    async def evolve_pipeline(
        self,
        performance_metrics: Callable[
            [ArchitecturalHypothesis],
            float | Awaitable[float] | Dict[str, float] | Awaitable[Dict[str, float]],
        ],
    ) -> None:
        """Continuously optimise the pipeline until convergence."""

        while not self.convergence_achieved():
            hypotheses = await self.generate_architectural_mutations()
            if not hypotheses:
                break
            results = await self.test_parallel_realities(
                hypotheses, performance_metrics
            )
            self.adopt_superior_architecture(results)
            self.crystallize_learning()
            self._iteration += 1

    async def generate_architectural_mutations(self) -> List[ArchitecturalHypothesis]:
        """Expand the search frontier with freshly crystallised techniques."""

        dreams = await asyncio.gather(
            *(
                self.pipeline.rem_cycles.generate_vision()
                for _ in range(self.exploration_batch_size)
            )
        )
        hypotheses: List[ArchitecturalHypothesis] = []
        for dream, counter in zip(dreams, itertools.count(1)):
            if not dream.is_coherent():
                self.pipeline.boundary_knowledge.expand(dream.failure_modes)
                continue
            technique = self.pipeline.unconscious_processor.crystallize(dream)
            mutation_notes = (
                f"mutation_{self._iteration}_{counter}: derived from {dream.idea}"
            )
            hypotheses.append(
                ArchitecturalHypothesis(
                    technique=technique,
                    mutation_notes=mutation_notes,
                    originating_dream=dream,
                )
            )
        return hypotheses

    async def test_parallel_realities(
        self,
        hypotheses: Sequence[ArchitecturalHypothesis],
        performance_metrics: Callable[
            [ArchitecturalHypothesis],
            float | Awaitable[float] | Dict[str, float] | Awaitable[Dict[str, float]],
        ],
    ) -> List[EvaluationResult]:
        """Evaluate hypotheses concurrently."""

        async def _evaluate(hypothesis: ArchitecturalHypothesis) -> EvaluationResult:
            raw_metrics = performance_metrics(hypothesis)
            if inspect.isawaitable(raw_metrics):
                raw_metrics = await raw_metrics  # type: ignore[assignment]

            if isinstance(raw_metrics, dict):
                diagnostics = raw_metrics
                score = float(sum(raw_metrics.values()) / max(len(raw_metrics), 1))
            else:
                score = float(raw_metrics)
                diagnostics = {"score": score}

            return EvaluationResult(
                hypothesis=hypothesis, score=score, diagnostics=diagnostics
            )

        evaluations = await asyncio.gather(
            *(_evaluate(hypothesis) for hypothesis in hypotheses)
        )
        self._history.extend(evaluations)
        return evaluations

    def adopt_superior_architecture(self, results: Sequence[EvaluationResult]) -> None:
        """Select and integrate the best performing hypothesis."""

        if not results:
            return

        best_result = max(results, key=lambda result: result.score)
        technique_name = best_result.hypothesis.technique.name
        prior_score = self._technique_scores.get(technique_name)

        if prior_score is not None and prior_score >= best_result.score:
            return

        conflicts = self.pipeline.conscious_processor.detect_conflicts(
            best_result.hypothesis.technique
        )
        if conflicts:
            self.pipeline.conscious_processor.resolve_conflicts(
                best_result.hypothesis.technique
            )

        self.pipeline.conscious_processor.integrate(best_result.hypothesis.technique)
        self._technique_scores[technique_name] = best_result.score
        self._best_result = best_result

    def crystallize_learning(self) -> None:
        """Record learnings from the latest optimisation step."""

        if not self._best_result:
            return

        dream = self._best_result.hypothesis.originating_dream
        self.pipeline.boundary_knowledge.expand(dream.failure_modes)
