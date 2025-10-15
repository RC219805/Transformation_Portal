"""Synthetic viewers for subjective video quality evaluation.

This module implements a whimsical-yet-pragmatic take on the
``SyntheticViewer`` concept outlined in the project brief.  The original
description referenced components such as a ``DigitalConsciousness`` and an
``ExperientialMemory`` that *feel* the aesthetics of a luxury real-estate
video.  The implementation below keeps that flavour while grounding the
behaviour in deterministic, testable Python code.

The high-level flow is:

* A ``DigitalConsciousness`` normalises the incoming ``video_stream`` into a
  sequence of :class:`JourneyMoment` instances.
* Those moments are wrapped in an :class:`EmotionalJourney`, which exposes
  summary statistics (means, variability, rhythm).
* An ``ExperientialMemory`` stores a light-weight snapshot of each traversal.
* The ``TrainedOnMillionsOfLuxuryViewings`` cortex converts the journey into
  an :class:`ACUScore` – a synthetic "Aesthetic Consensus Unit" score.
* ``SyntheticViewer`` orchestrates the above and invites a few archetypal
  clones to reach a consensus rating.

The end result is a friendly object that can take a list of frame-level
metrics (``technical``, ``emotional``, ``memorability`` and ``desire``) and
return a blended score that lives in the ``[0.0, 1.0]`` range.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Iterator, List, Mapping, MutableSequence, Sequence


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp ``value`` to the inclusive ``[minimum, maximum]`` range."""

    return max(minimum, min(value, maximum))


@dataclass(frozen=True)
class ACUScore:
    """Aggregated aesthetic score for a video experience."""

    technical: float
    emotional: float
    memorability: float
    desire_quotient: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "technical", _clamp(self.technical))
        object.__setattr__(self, "emotional", _clamp(self.emotional))
        object.__setattr__(self, "memorability", _clamp(self.memorability))
        object.__setattr__(self, "desire_quotient", _clamp(self.desire_quotient))

    @property
    def overall(self) -> float:
        """Return the arithmetic mean of the individual score components."""

        return fmean(
            (
                self.technical,
                self.emotional,
                self.memorability,
                self.desire_quotient,
            )
        )

    def as_dict(self) -> Mapping[str, float]:
        """Represent the score as a dictionary for JSON serialisation."""

        return {
            "technical": self.technical,
            "emotional": self.emotional,
            "memorability": self.memorability,
            "desire_quotient": self.desire_quotient,
            "overall": self.overall,
        }


@dataclass(frozen=True)
class JourneyMoment:
    """Frame or beat of the emotional journey in the ``[0, 1]`` range."""

    technical: float
    emotional: float
    memorability: float
    desire: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "technical", _clamp(self.technical))
        object.__setattr__(self, "emotional", _clamp(self.emotional))
        object.__setattr__(self, "memorability", _clamp(self.memorability))
        object.__setattr__(self, "desire", _clamp(self.desire))


class EmotionalJourney:
    """Sequence wrapper exposing summary statistics for moments."""

    def __init__(self, moments: Sequence[JourneyMoment]):
        if not moments:
            raise ValueError("An emotional journey requires at least one moment")
        self._moments: Sequence[JourneyMoment] = tuple(moments)

    def __iter__(self) -> Iterator[JourneyMoment]:
        return iter(self._moments)

    @property
    def moments(self) -> Sequence[JourneyMoment]:
        return self._moments

    def _mean(self, attr: str) -> float:
        return fmean(getattr(moment, attr) for moment in self._moments)

    def _variability(self, attr: str) -> float:
        values = [getattr(moment, attr) for moment in self._moments]
        return max(values) - min(values) if len(values) > 1 else 0.0

    def _rhythm(self, attr: str) -> float:
        if len(self._moments) < 2:
            return 0.0
        deltas = [
            abs(getattr(self._moments[idx + 1], attr) - getattr(self._moments[idx], attr))
            for idx in range(len(self._moments) - 1)
        ]
        return fmean(deltas)

    def summary(self) -> Mapping[str, float]:
        """Return mean metrics and dynamism helpers for the journey."""

        return {
            "technical": self._mean("technical"),
            "emotional": self._mean("emotional"),
            "memorability": self._mean("memorability"),
            "desire": self._mean("desire"),
            "emotional_variability": self._variability("emotional"),
            "memorability_variability": self._variability("memorability"),
            "desire_rhythm": self._rhythm("desire"),
            "length": float(len(self._moments)),
        }


class DigitalConsciousness:
    """Normalises raw video impressions into an :class:`EmotionalJourney`."""

    def __init__(self, archetype: str):
        self.archetype = archetype

    def traverse(self, video_stream: Iterable[Mapping[str, float] | JourneyMoment] | EmotionalJourney) -> EmotionalJourney:
        if isinstance(video_stream, EmotionalJourney):
            return video_stream

        moments: List[JourneyMoment] = []
        for entry in video_stream:
            if isinstance(entry, JourneyMoment):
                moments.append(entry)
                continue
            if not isinstance(entry, Mapping):  # pragma: no cover - safety net
                raise TypeError(
                    "video_stream entries must be mappings or JourneyMoment instances"
                )

            moment = JourneyMoment(
                technical=float(entry.get("technical", 0.0)),
                emotional=float(entry.get("emotional", entry.get("emotion", 0.0))),
                memorability=float(entry.get("memorability", entry.get("memory", 0.0))),
                desire=float(entry.get("desire", entry.get("desire_quotient", 0.0))),
            )
            moments.append(moment)

        return EmotionalJourney(moments)


@dataclass(frozen=True)
class ArchetypeProfile:
    """Weighting profile used by the aesthetic cortex."""

    technical_emphasis: float = 1.0
    emotional_emphasis: float = 1.0
    memorability_emphasis: float = 1.0
    desire_emphasis: float = 1.0
    variability_bonus: float = 0.05
    rhythm_bonus: float = 0.05
    baseline_bias: float = 0.0


class ExperientialMemory:
    """Stores light-weight snapshots of journeys the viewer has taken."""

    def __init__(self):
        self._snapshots: MutableSequence[Mapping[str, float]] = []

    @property
    def snapshots(self) -> Sequence[Mapping[str, float]]:
        return tuple(self._snapshots)

    def remember(self, journey: EmotionalJourney) -> Mapping[str, float]:
        snapshot = journey.summary()
        self._snapshots.append(snapshot)
        return snapshot


class TrainedOnMillionsOfLuxuryViewings:
    """Maps an ``EmotionalJourney`` to an :class:`ACUScore`."""

    def __init__(self):
        self._profiles = {
            "default": ArchetypeProfile(),
            "minimalist_millennial": ArchetypeProfile(
                technical_emphasis=0.96,
                emotional_emphasis=1.08,
                memorability_emphasis=1.02,
                desire_emphasis=1.05,
                variability_bonus=0.04,
                rhythm_bonus=0.06,
                baseline_bias=0.01,
            ),
            "traditional_luxury_connoisseur": ArchetypeProfile(
                technical_emphasis=1.05,
                emotional_emphasis=0.98,
                memorability_emphasis=1.04,
                desire_emphasis=1.02,
                variability_bonus=0.03,
                rhythm_bonus=0.04,
            ),
            "futurist_tech_executive": ArchetypeProfile(
                technical_emphasis=1.07,
                emotional_emphasis=1.02,
                memorability_emphasis=0.97,
                desire_emphasis=1.08,
                variability_bonus=0.05,
                rhythm_bonus=0.07,
                baseline_bias=0.015,
            ),
        }

    def score(self, journey: EmotionalJourney, archetype: str) -> ACUScore:
        summary = journey.summary()
        profile = self._profiles.get(archetype, self._profiles["default"])

        technical = self._score_channel(
            summary["technical"],
            profile.technical_emphasis,
            summary["emotional_variability"],
            profile,
        )
        emotional = self._score_channel(
            summary["emotional"],
            profile.emotional_emphasis,
            summary["emotional_variability"],
            profile,
        )
        memorability = self._score_channel(
            summary["memorability"],
            profile.memorability_emphasis,
            summary["memorability_variability"],
            profile,
        )
        desire = self._score_channel(
            summary["desire"],
            profile.desire_emphasis,
            summary["desire_rhythm"],
            profile,
        )

        return ACUScore(
            technical=technical,
            emotional=emotional,
            memorability=memorability,
            desire_quotient=desire,
        )

    @staticmethod
    def _score_channel(
        base_value: float, emphasis: float, dynamism: float, profile: ArchetypeProfile
    ) -> float:
        raw = base_value * emphasis
        raw += dynamism * profile.variability_bonus
        raw += profile.baseline_bias
        return _clamp(raw)


class SyntheticViewer:
    """Synthesised viewer that reports an ``ACUScore`` for a video."""

    def __init__(self, archetype: str = "default") -> None:
        self.archetype = archetype
        self.consciousness = DigitalConsciousness(archetype)
        self.memory = ExperientialMemory()
        self.aesthetic_cortex = TrainedOnMillionsOfLuxuryViewings()

    def clone(self, archetype: str) -> "SyntheticViewer":
        """Return a fresh viewer instance for a different archetype."""

        return SyntheticViewer(archetype)

    def _score_from_journey(self, journey: EmotionalJourney) -> ACUScore:
        self.memory.remember(journey)
        return self.aesthetic_cortex.score(journey, self.archetype)

    def reach_aesthetic_consensus(self, scores: Sequence[ACUScore]) -> ACUScore:
        if not scores:
            raise ValueError("At least one score is required for consensus")

        return ACUScore(
            technical=fmean(score.technical for score in scores),
            emotional=fmean(score.emotional for score in scores),
            memorability=fmean(score.memorability for score in scores),
            desire_quotient=fmean(score.desire_quotient for score in scores),
        )

    def experience_content(
        self, video_stream: Iterable[Mapping[str, float] | JourneyMoment] | EmotionalJourney
    ) -> ACUScore:
        """Experience a video stream and return the consensus ``ACUScore``."""

        journey = self.consciousness.traverse(video_stream)
        primary_score = self._score_from_journey(journey)

        perspectives = [
            self.clone(archetype="minimalist_millennial"),
            self.clone(archetype="traditional_luxury_connoisseur"),
            self.clone(archetype="futurist_tech_executive"),
        ]

        perspective_scores = [viewer._score_from_journey(journey) for viewer in perspectives]

        return self.reach_aesthetic_consensus([primary_score, *perspective_scores])

