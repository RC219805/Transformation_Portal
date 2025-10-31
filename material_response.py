"""Material Response principle support utilities.

The marketing material for the 800 Picacho Lane collection repeatedly alludes
to a proprietary "Material Response" technology.  The production codebase does
not actually perform physically based rendering, but we can still document what
the principle *means* so tests and documentation have a concrete artefact to
reference.  This module provides such an artefact.
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

# Precompiled regex patterns for performance
_KEYWORD_PATTERN = re.compile(r"[a-z0-9]+")
_TEXTURE_PATTERN = re.compile(r"[a-z]+")


def _is_sequence(value: object) -> bool:
    """Return ``True`` when ``value`` should be treated as a sequence."""

    return isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray))


def _coerce_matrix(data: Sequence[Sequence[float]]) -> List[List[float]]:
    """Convert ``data`` into a rectangular list-of-lists of floats."""

    if not _is_sequence(data):
        raise TypeError("material data must be a sequence")

    if len(data) == 0:
        raise ValueError("material data cannot be empty")

    if _is_sequence(data[0]):
        rows = []
        row_length = None
        for row in data:
            if not _is_sequence(row):
                raise TypeError("material rows must be sequences")
            coerced_row = [float(value) for value in row]
            if row_length is None:
                row_length = len(coerced_row)
                if row_length == 0:
                    raise ValueError("material rows cannot be empty")
            elif len(coerced_row) != row_length:
                raise ValueError("material data must form a rectangular grid")
            rows.append(coerced_row)
        return rows

    coerced = [float(value) for value in data]
    if len(coerced) == 0:
        raise ValueError("material data cannot be empty")
    return [coerced]


def _flatten(matrix: Sequence[Sequence[float]]) -> List[float]:
    """Return a flattened representation of ``matrix``."""

    return [value for row in matrix for value in row]


def _median(values: Sequence[float]) -> float:
    """Compute the median of ``values``."""

    ordered = sorted(values)
    length = len(ordered)
    if length == 0:
        raise ValueError("median of empty sequence is undefined")
    middle = length // 2
    if length % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2.0
    return ordered[middle]


def _fft_frequency(index: int, size: int) -> float:
    """Return the normalised FFT frequency for ``index``."""

    if size <= 0:
        raise ValueError("size must be positive")
    half = size // 2
    if index <= half:
        return index / size
    return (index - size) / size


def _dft2(matrix: Sequence[Sequence[float]]) -> List[List[complex]]:
    """Compute the 2D discrete Fourier transform for ``matrix``."""

    rows = len(matrix)
    cols = len(matrix[0])
    result = [[0j for _ in range(cols)] for _ in range(rows)]

    for u in range(rows):
        for v in range(cols):
            total = 0j
            for x in range(rows):
                for y in range(cols):
                    angle = -2.0 * math.pi * ((u * x / rows) + (v * y / cols))
                    total += matrix[x][y] * complex(math.cos(angle), math.sin(angle))
            result[u][v] = total

    return result


def _energy_by_band(
    dft: Sequence[Sequence[complex]],
    band: str,
    cutoff: float,
    radii: Sequence[Sequence[float]],
) -> float:
    """Return the total squared magnitude energy for the requested band."""

    total = 0.0
    rows = len(dft)
    cols = len(dft[0])

    for u in range(rows):
        for v in range(cols):
            radius = radii[u][v]
            in_band = radius >= cutoff if band == "high" else radius <= cutoff
            if in_band:
                coefficient = dft[u][v]
                total += (coefficient.real ** 2) + (coefficient.imag ** 2)

    return total


def _radial_frequency_grid(shape: Tuple[int, int]) -> List[List[float]]:
    """Return the radial frequency grid for ``shape``."""

    rows, cols = shape
    grid = []
    for u in range(rows):
        row = []
        freq_u = _fft_frequency(u, rows)
        for v in range(cols):
            freq_v = _fft_frequency(v, cols)
            row.append(math.hypot(freq_u, freq_v))
        grid.append(row)
    return grid


def _linear_regression(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Return the slope and intercept for ``xs``/``ys``."""

    if len(xs) != len(ys):
        raise ValueError("xs and ys must have matching lengths")
    n = len(xs)
    if n == 0:
        raise ValueError("linear regression requires data")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)

    if math.isclose(denominator, 0.0):
        return 0.0, mean_y

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


@dataclass(frozen=True)
class MaterialResponseExample:
    """Structured example describing how the principle manifests."""

    material: str
    lighting: str
    challenge: str
    response: str
    outcome: str

    def as_dict(self) -> Dict[str, str]:
        """Return the example in a serialisable dictionary form."""

        return {
            "material": self.material,
            "lighting": self.lighting,
            "challenge": self.challenge,
            "response": self.response,
            "outcome": self.outcome,
        }


class MaterialResponsePrinciple:
    """Captures the intent behind the Material Response marketing claims.

    The principle is intentionally high-level: it frames how surface-specific
    adjustments should be reasoned about even when a pipeline primarily applies
    tone and colour transforms.  Tests that refer to this principle can inspect
    the structured data returned here without coupling to any particular image
    processing implementation.
    """

    name: str = "Material Response"
    focus: str = (
        "Honor the unique light interaction of each surface instead of applying "
        "purely global transforms."
    )

    tenets: List[str] = [
        "Respect energy conservation in highlights so reflective materials retain believable sheen.",
        "Preserve midtone texture to keep organic materials tactile and dimensional.",
        "Blend transitions between adjacent materials so adjustments feel authored rather than procedural.",
    ]

    def describe(self) -> str:
        """Return a human readable description of the principle."""

        return f"{self.name}: {self.focus}"

    def guidelines(self) -> List[str]:
        """Return the set of guiding tenets."""

        return list(self.tenets)

    def generate_examples(self) -> List[Dict[str, str]]:
        """Produce canonical examples demonstrating the principle in action.

        The examples intentionally cover a variety of materials.  Returning
        dictionaries keeps the data ergonomic for documentation tooling and
        avoids a dependency on this module's dataclass by callers.
        """

        examples: Iterable[MaterialResponseExample] = [
            MaterialResponseExample(
                material="polished marble foyer",
                lighting="late-afternoon sun grazing through clerestory windows",
                challenge="Specular highlights risk clipping and flattening the stone veining.",
                response="Apply highlight recovery before clarity so micro-contrast is boosted only after "
                "energy is preserved.",
                outcome="Marble retains luminous sheen with detailed veining rather than a white patch.",
            ),
            MaterialResponseExample(
                material="brushed brass fixtures",
                lighting="mixed tungsten sconces and cool daylight",
                challenge="Mixed colour temperatures create muddy warm-cool interference on the anisotropic metal.",
                response="Balance tint locally while moderating saturation to keep the brushed texture defined.",
                outcome="Brass reads as intentionally warm while grooves stay articulated.",
            ),
            MaterialResponseExample(
                material="velvet chaise",
                lighting="soft bounced fill with narrow specular kicker",
                challenge="Global contrast lift would crush the velvet's directionality and make it plastic.",
                response="Target midtone contrast and vibrance to maintain pile shading while enriching dye density.",
                outcome="Velvet keeps tactile depth and directional sheen without waxy highlights.",
            ),
        ]

        return [example.as_dict() for example in examples]


_STOP_WORDS = {
    "the",
    "and",
    "or",
    "so",
    "that",
    "a",
    "an",
    "to",
    "of",
    "in",
    "between",
    "with",
    "for",
    "be",
    "is",
    "are",
    "as",
    "keep",
    "so",
    "rather",
}


def _extract_keywords(text: str) -> List[str]:
    """Return a list of meaningful lowercase keywords from ``text``."""

    tokens = _KEYWORD_PATTERN.findall(text.lower())
    return [token for token in tokens if token not in _STOP_WORDS and len(token) > 2]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Return ``value`` limited to the inclusive ``[minimum, maximum]`` range."""

    if minimum > maximum:
        raise ValueError(
            "minimum cannot be greater than maximum: "
            f"minimum={minimum!r}, maximum={maximum!r}"
        )

    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class MaterialAestheticProfile:
    """Structured description of how a material should be perceived."""

    name: str
    texture: str
    rarity: float
    craftsmanship: float
    innovation: float

    def __post_init__(self) -> None:
        for attribute in ("rarity", "craftsmanship", "innovation"):
            value = getattr(self, attribute)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{attribute} must be between 0.0 and 1.0 inclusive; received {value!r}"
                )


@dataclass(frozen=True)
class LightingProfile:
    """Simplified representation of the lighting environment."""

    warmth: float
    intensity: float
    diffusion: float

    def __post_init__(self) -> None:
        for attribute in ("warmth", "intensity", "diffusion"):
            value = getattr(self, attribute)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{attribute} must be between 0.0 and 1.0 inclusive; received {value!r}"
                )


@dataclass(frozen=True)
class ViewerProfile:
    """Describes the viewer in terms of aesthetic preferences."""

    cultural_background: str
    novelty_preference: float
    heritage_affinity: float

    def __post_init__(self) -> None:
        for attribute in ("novelty_preference", "heritage_affinity"):
            value = getattr(self, attribute)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{attribute} must be between 0.0 and 1.0 inclusive; received {value!r}"
                )

        if not self.cultural_background:
            raise ValueError("cultural_background must be a non-empty string")


@dataclass(frozen=True)
class EmotionalResonance:
    """Container capturing the viewer's predicted emotional response."""

    awe: float
    comfort: float
    focus: float
    cultural_background: str

    def __post_init__(self) -> None:
        for attribute in ("awe", "comfort", "focus"):
            value = getattr(self, attribute)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{attribute} must be between 0.0 and 1.0 inclusive; received {value!r}"
                )

        if not self.cultural_background:
            raise ValueError("cultural_background must be a non-empty string")

    def as_dict(self) -> Dict[str, float]:
        """Return the resonance as a serialisable mapping."""

        return {
            "awe": self.awe,
            "comfort": self.comfort,
            "focus": self.focus,
            "cultural_background": self.cultural_background,
        }


@dataclass(frozen=True)
class ContextualResonance:
    """Emotion scores contextualised for a cultural narrative."""

    scores: Dict[str, float]
    narrative: str

    def __post_init__(self) -> None:
        required_keys = {"awe", "comfort", "focus"}
        missing = required_keys.difference(self.scores)
        if missing:
            raise ValueError(
                f"scores must include {sorted(required_keys)}; missing keys: {sorted(missing)}"
            )

        for key, value in self.scores.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Score {key!r} must be between 0.0 and 1.0 inclusive; received {value!r}"
                )

        if not self.narrative:
            raise ValueError("narrative must be a non-empty string")


class NeuroAestheticEngine:
    """Lightweight psychophysical model for interpreting material descriptors."""

    _TEXTURE_AFFECTIONS: Mapping[str, Mapping[str, float]] = {
        "velvet": {"comfort": 0.18, "awe": 0.07},
        "marble": {"awe": 0.2, "focus": 0.08},
        "brushed": {"focus": 0.12, "awe": 0.04},
        "polished": {"awe": 0.15},
        "matte": {"focus": -0.05, "comfort": 0.05},
        "handcrafted": {"comfort": 0.12, "awe": 0.05},
        "organic": {"comfort": 0.1},
        "lacquer": {"awe": 0.06, "focus": 0.05},
    }

    def predict_limbic_response(
        self, texture: str, warmth: float, cultural_background: str
    ) -> EmotionalResonance:
        """Return an :class:`EmotionalResonance` derived from qualitative inputs."""

        if not 0.0 <= warmth <= 1.0:
            raise ValueError("warmth must be between 0.0 and 1.0 inclusive")

        tokens = _TEXTURE_PATTERN.findall(texture.lower())
        awe = 0.45
        comfort = 0.45
        focus = 0.45

        for token in tokens:
            if token in self._TEXTURE_AFFECTIONS:
                weights = self._TEXTURE_AFFECTIONS[token]
                awe += weights.get("awe", 0.0)
                comfort += weights.get("comfort", 0.0)
                focus += weights.get("focus", 0.0)

        # Warmth emphasises comfort, whereas cooler setups heighten focus.
        comfort += 0.2 * (1.0 - abs(warmth - 0.6))
        focus += 0.18 * (0.5 - abs(warmth - 0.4))
        awe += 0.15 * warmth

        # Cultural familiarity slightly reinforces comfort and awe.
        cultural_modifier = 0.02 if cultural_background else 0.0
        comfort += cultural_modifier
        awe += cultural_modifier

        return EmotionalResonance(
            awe=_clamp(awe),
            comfort=_clamp(comfort),
            focus=_clamp(focus),
            cultural_background=cultural_background.lower(),
        )


class GlobalLuxurySemantics:
    """Contextualises emotion scores for regional expectations of luxury."""

    _CULTURAL_WEIGHTS: Mapping[str, Mapping[str, float]] = {
        "mediterranean": {"comfort": 0.06, "awe": 0.05},
        "scandinavian": {"comfort": 0.08, "focus": 0.07, "awe": -0.03},
        "japanese": {"focus": 0.09, "comfort": -0.02, "awe": 0.04},
        "middle eastern": {"awe": 0.08, "comfort": 0.04},
        "american": {"focus": 0.03, "comfort": 0.02},
    }
    _CULTURAL_ALIAS_MAP: Mapping[str, str] = {
        "med": "mediterranean",
        "mediterranean coast": "mediterranean",
        "scandi": "scandinavian",
        "nordic": "scandinavian",
        "jp": "japanese",
        "levantine": "middle eastern",
        "us": "american",
        "usa": "american",
    }

    def recontextualize(
        self, material: MaterialAestheticProfile, resonance: EmotionalResonance
    ) -> ContextualResonance:
        """Return resonance tuned to cultural and material narratives."""

        background_key = resonance.cultural_background.strip().lower()
        background = self._CULTURAL_ALIAS_MAP.get(background_key, background_key)
        weights = self._CULTURAL_WEIGHTS.get(background, {})

        awe = _clamp(resonance.awe + weights.get("awe", 0.0) + 0.12 * material.rarity)
        comfort = _clamp(
            resonance.comfort
            + weights.get("comfort", 0.0)
            + 0.1 * material.craftsmanship
        )
        focus = _clamp(
            resonance.focus
            + weights.get("focus", 0.0)
            + 0.08 * (1.0 - material.innovation)
        )

        narrative = (
            f"{material.name} channels a {background or 'global'} sensibility by "
            f"balancing awe ({awe:.2f}), comfort ({comfort:.2f}) and focus ({focus:.2f})."
        )

        return ContextualResonance(scores={"awe": awe, "comfort": comfort, "focus": focus}, narrative=narrative)


class FutureStatePredictor:
    """Projects how a material treatment will age alongside design trends."""

    def project(
        self, material: MaterialAestheticProfile, resonance: ContextualResonance
    ) -> float:
        """Return a 0-1 score indicating forward-looking relevance."""

        awe = resonance.scores["awe"]
        focus = resonance.scores["focus"]

        rarity_weight = 0.35 * material.rarity
        innovation_weight = 0.4 * material.innovation
        emotional_weight = 0.25 * (0.6 * awe + 0.4 * focus)

        projection = rarity_weight + innovation_weight + emotional_weight
        return _clamp(projection)


class CognitiveMaterialResponse:
    """High-level facade orchestrating the psychophysical subsystems."""

    def __init__(self) -> None:
        self.perception_model = NeuroAestheticEngine()
        self.cultural_context = GlobalLuxurySemantics()
        self.temporal_relevance = FutureStatePredictor()

    def process(
        self,
        material: MaterialAestheticProfile,
        lighting: LightingProfile,
        viewer_profile: ViewerProfile,
    ) -> Dict[str, object]:
        """Return a holistic appraisal of the material treatment."""

        # Not just physics - but psychophysics
        emotional_resonance = self.perception_model.predict_limbic_response(
            material.texture,
            lighting.warmth,
            viewer_profile.cultural_background,
        )

        return self.optimize_for_consciousness(material, emotional_resonance)

    def optimize_for_consciousness(
        self, material: MaterialAestheticProfile, emotional_resonance: EmotionalResonance
    ) -> Dict[str, object]:
        """Blend cultural and temporal heuristics into actionable guidance."""

        contextualized = self.cultural_context.recontextualize(material, emotional_resonance)
        future_alignment = self.temporal_relevance.project(material, contextualized)
        luxury_index = self._composite_index(material, contextualized, future_alignment)

        recommendations = self._recommendations(material, contextualized)

        return {
            "material": material.name,
            "texture": material.texture,
            "emotional_resonance": contextualized.scores,
            "narrative": contextualized.narrative,
            "luxury_index": luxury_index,
            "future_alignment": future_alignment,
            "recommendations": recommendations,
        }

    @staticmethod
    def _composite_index(
        material: MaterialAestheticProfile,
        resonance: ContextualResonance,
        future_alignment: float,
    ) -> float:
        craftsmanship_weight = 0.35 * material.craftsmanship
        rarity_weight = 0.25 * material.rarity
        emotional_weight = 0.25 * (0.5 * resonance.scores["awe"] + 0.5 * resonance.scores["comfort"])
        future_weight = 0.15 * future_alignment

        return _clamp(craftsmanship_weight + rarity_weight + emotional_weight + future_weight)

    @staticmethod
    def _recommendations(
        material: MaterialAestheticProfile, resonance: ContextualResonance
    ) -> List[str]:
        recommendations: List[str] = []
        awe = resonance.scores["awe"]
        comfort = resonance.scores["comfort"]
        focus = resonance.scores["focus"]

        if awe < 0.6:
            recommendations.append(
                "Introduce controlled specular accents to elevate perceived grandeur."
            )
        if comfort < 0.55:
            recommendations.append(
                "Blend warmer fill lighting or tactile styling to soften the presentation."
            )
        if focus < 0.5:
            recommendations.append(
                "Shape negative space to emphasise the material's structural rhythm."
            )

        if material.innovation > 0.65 and awe >= 0.6:
            recommendations.append(
                "Document the treatment narrative for launch collateral while momentum is high."
            )

        if not recommendations:
            recommendations.append("Maintain current treatment; responses align with luxury objectives.")

        return recommendations


def violates(decision: str, tenet: str) -> bool:
    """Heuristically determine whether ``decision`` conflicts with ``tenet``."""

    decision_lower = decision.lower()
    tenet_lower = tenet.lower()

    # Specific heuristics for the three marketing tenets.  These catch the most
    # common contradictions that show up in the architectural documentation and
    # associated tests.
    tenet_specific_checks = (
        (
            {"energy", "highlight"},
            {
                "clip the highlights",
                "blow out the highlights",
                "overexpose highlights",
                "ignore energy conservation",
                "treat highlights as unlimited",
            },
        ),
        (
            {"midtone", "texture"},
            {
                "flatten the midtones",
                "blur midtone texture",
                "remove midtone detail",
                "plastic midtones",
                "eliminate texture",
            },
        ),
        (
            {"blend", "transitions"},
            {
                "hard mask transition",
                "no blending between materials",
                "treat each material in isolation",
                "apply a binary mask",
            },
        ),
    )

    for keywords, forbidden_phrases in tenet_specific_checks:
        if keywords.issubset(set(_extract_keywords(tenet_lower))):
            if any(phrase in decision_lower for phrase in forbidden_phrases):
                return True

    # Generic negation handling: if the decision explicitly ignores or bypasses
    # aspects that the tenet cares about we treat it as a violation.
    negating_words = {
        "ignore",
        "bypass",
        "discard",
        "skip",
        "flatten",
        "eliminate",
        "remove",
        "crush",
        "erase",
    }
    keywords = _extract_keywords(tenet_lower)

    for negation in negating_words:
        if negation in decision_lower:
            if any(f"{negation} {keyword}" in decision_lower for keyword in keywords):
                return True
            # Fall back to a proximity check if the explicit "negation keyword"
            # pattern was not matched.  This keeps the heuristic resilient when
            # the decision references the concept in a different grammatical
            # order (e.g. "midtone texture gets flattened")
            for keyword in keywords:
                if keyword in decision_lower:
                    return True

    return False


class MarketingClaimValidator:
    """Validates that implementation decisions align with marketing principles."""

    def assess_decision(
        self, decision: str, principle: MaterialResponsePrinciple
    ) -> bool:
        """Return ``True`` when ``decision`` honours ``principle``'s tenets."""

        for tenet in principle.guidelines():
            if violates(decision, tenet):
                return False
        return True


__all__ = [
    "MaterialResponsePrinciple",
    "MaterialResponseExample",
    "MaterialAestheticProfile",
    "LightingProfile",
    "ViewerProfile",
    "EmotionalResonance",
    "ContextualResonance",
    "NeuroAestheticEngine",
    "GlobalLuxurySemantics",
    "FutureStatePredictor",
    "CognitiveMaterialResponse",
    "MarketingClaimValidator",
    "MaterialResponseValidator",
    "compose_operations",
    "apply_transformation_tensor",
    "violates",
]


class MaterialResponseValidator:
    """Quantitative heuristics for validating material treatments.

    The validator deliberately keeps the implementations lightweight and free of
    heavy numerical dependencies.  The goal is to surface signal that is
    directionally correct for tests rather than to provide production-grade
    analysis of BRDFs or fractal geometry.
    """

    def measure_specular_preservation(
        self, before: Sequence[Sequence[float]], after: Sequence[Sequence[float]]
    ) -> float:
        """Return the energy ratio for the high-frequency Fourier band.

        ``before`` and ``after`` are expected to be array-like objects that can
        be coerced into rectangular grids of floats.  The method computes the sum
        of squared magnitudes in the high-frequency region of the Fourier
        spectrum and reports ``after / before``.  When the reference energy is
        zero the ratio gracefully falls back to ``1.0`` so tests can reason about
        a neutral baseline.
        """

        return self._fourier_energy_ratio(before, after, band="high")

    # ------------------------------------------------------------------
    # Internal helpers

    @staticmethod
    def _fourier_energy_ratio(
        before: Sequence[Sequence[float]],
        after: Sequence[Sequence[float]],
        *,
        band: str,
    ) -> float:
        """Return the ratio of Fourier-band energy between ``after`` and ``before``.

        The helper performs minimal validation and defaults to returning ``1.0``
        if the reference energy is zero to keep the metric stable for synthetic
        fixtures used in the tests.
        """

        if band not in {"high", "low"}:
            raise ValueError("band must be 'high' or 'low'")

        before_matrix = _coerce_matrix(before)
        after_matrix = _coerce_matrix(after)

        if len(before_matrix) != len(after_matrix) or len(before_matrix[0]) != len(after_matrix[0]):
            raise ValueError("before and after arrays must share the same shape")

        dft_before = _dft2(before_matrix)
        dft_after = _dft2(after_matrix)

        radii = _radial_frequency_grid((len(before_matrix), len(before_matrix[0])))
        cutoff = _median(_flatten(radii))

        energy_before = _energy_by_band(dft_before, band, cutoff, radii)
        energy_after = _energy_by_band(dft_after, band, cutoff, radii)

        if math.isclose(energy_before, 0.0, rel_tol=1e-9, abs_tol=1e-12):
            return 1.0

        return float(energy_after / energy_before)

    @staticmethod
    def _calculate_hausdorff_dimension(surface: Sequence[Sequence[float]]) -> float:
        """Estimate fractal dimension using a simple box-counting approach."""

        matrix = _coerce_matrix(surface)
        if len(matrix) == 0 or len(matrix[0]) == 0:
            raise ValueError("surface must contain data")

        min_value = min(_flatten(matrix))
        normalised = [[value - min_value for value in row] for row in matrix]

        max_value = max(_flatten(normalised))
        if not math.isclose(max_value, 0.0, rel_tol=1e-9, abs_tol=1e-12):
            normalised = [[value / max_value for value in row] for row in normalised]

        threshold = _median(_flatten(normalised))
        binary = [[value > threshold for value in row] for row in normalised]

        rows = len(binary)
        cols = len(binary[0])
        min_dim = min(rows, cols)
        if min_dim <= 0:
            raise ValueError("surface must contain data")

        max_exponent = int(math.floor(math.log2(min_dim)))
        if max_exponent <= 1:
            return 1.0

        sizes = [2 ** exponent for exponent in range(1, max_exponent)]
        counts = [MaterialResponseValidator._boxcount(binary, size) for size in sizes]

        eps = 1e-9
        xs = [math.log(size + eps) for size in sizes]
        ys = [math.log(count + eps) for count in counts]
        slope, _ = _linear_regression(xs, ys)
        dimension = max(-slope, 0.0)
        return float(dimension)

    @staticmethod
    def _boxcount(binary: Sequence[Sequence[bool]], size: int) -> int:
        """Count non-empty boxes of the given ``size`` for ``binary`` data."""

        if size <= 0:
            raise ValueError("size must be positive")

        rows = len(binary)
        cols = len(binary[0]) if rows else 0
        if rows == 0 or cols == 0:
            return 0

        trimmed_rows = rows - (rows % size)
        trimmed_cols = cols - (cols % size)
        if trimmed_rows == 0 or trimmed_cols == 0:
            return 0

        count = 0
        for row in range(0, trimmed_rows, size):
            for col in range(0, trimmed_cols, size):
                occupied = False
                for dr in range(size):
                    if occupied:
                        break
                    for dc in range(size):
                        if binary[row + dr][col + dc]:
                            occupied = True
                            break
                if occupied:
                    count += 1

        return count


OperationLike = Any


def compose_operations(*operations: OperationLike, size: int = 3, dtype: np.dtype | None = None) -> np.ndarray:
    """Return a composite linear transformation for channel-wise operations.

    Parameters
    ----------
    operations:
        Arbitrary operation descriptors that can be converted into channel
        transformation matrices.  Supported inputs include:

        * ``None`` – ignored.
        * Scalars – interpreted as uniform channel scales.
        * 1-D sequences – treated as per-channel scale factors.
        * 2-D square matrices – applied directly.
        * Mappings with ``"matrix"``/``"mix"`` keys (matrix) or ``"scale"``/
          ``"diag"`` keys (diagonal factors).  Optional ``"name"``/``"type"``
          metadata is used to coalesce duplicated operations.
        * Callables returning any of the above when invoked with ``size``.

    size:
        Channel dimensionality.  Defaults to ``3`` for RGB imagery.

    dtype:
        Optional :mod:`numpy` dtype for the returned tensor.  ``float32`` is
        used by default to match the codebase's image processing conventions.

    The helper performs conflict detection by coalescing repeated named
    operations.  When two operations share the same ``name``/``type`` metadata
    they are multiplied together in the order encountered so that the overall
    effect matches the sequential application that previously existed in the
    pipeline.
    """

    matrix_dtype = np.dtype(dtype) if dtype is not None else np.float32
    identity = np.eye(size, dtype=np.float64)

    if not operations:
        return identity.astype(matrix_dtype, copy=False)

    def _normalise(operation: OperationLike) -> Tuple[np.ndarray, str | None]:
        if operation is None:
            raise ValueError("compose_operations received an empty operation placeholder")

        name: str | None = None

        if callable(operation):  # type: ignore[call-arg]
            resolved = operation(size)  # type: ignore[misc]
            name = getattr(operation, "__name__", None)
            return _ensure_matrix(resolved, size), name

        if isinstance(operation, Mapping):
            maybe_name = (
                operation.get("name")
                or operation.get("type")
                or operation.get("label")
                or operation.get("id")
                or operation.get("kind")
            )
            if maybe_name is not None:
                name = str(maybe_name)

            if "matrix" in operation:
                return _ensure_matrix(operation["matrix"], size), name
            if "mix" in operation:
                return _ensure_matrix(operation["mix"], size), name
            if "scale" in operation or "diag" in operation:
                diag_source = operation.get("scale", operation.get("diag"))
                return _coerce_diagonal(diag_source, size), name
            if "weights" in operation:
                return _coerce_diagonal(operation["weights"], size), name
            raise ValueError(
                "operation mapping must contain 'matrix', 'mix', 'scale', 'diag', or 'weights'"
            )

        return _ensure_matrix(operation, size), None

    composed: List[np.ndarray] = []
    name_index: Dict[str, int] = {}

    for operation in operations:
        if operation is None:
            continue
        matrix, name = _normalise(operation)
        if name is not None:
            index = name_index.get(name)
            if index is not None:
                composed[index] = composed[index] @ matrix
                continue
            name_index[name] = len(composed)
        composed.append(matrix)

    if not composed:
        return identity.astype(matrix_dtype, copy=False)

    result = identity
    for matrix in composed:
        if matrix.shape != (size, size):
            raise ValueError(
                "operation matrix must be square with side length matching the channel size"
            )
        result = result @ matrix

    return result.astype(matrix_dtype, copy=False)


def _coerce_diagonal(operation: object, size: int) -> np.ndarray:
    if operation is None:
        raise ValueError("diagonal operation cannot be None")

    diag = np.asarray(operation, dtype=np.float64)
    if diag.ndim == 0:
        return np.eye(size, dtype=np.float64) * float(diag)
    if diag.ndim != 1:
        raise ValueError("diagonal operation must be a scalar or a 1-D sequence")
    if diag.shape[0] != size:
        raise ValueError(
            f"diagonal operation expects {size} coefficients; received {diag.shape[0]}"
        )
    return np.diag(diag)


def _ensure_matrix(operation: object, size: int) -> np.ndarray:
    matrix = np.asarray(operation, dtype=np.float64)

    if matrix.ndim == 0:
        # Treat scalar matrices as uniform scale factors across channels.
        return np.eye(size, dtype=np.float64) * float(matrix)

    if matrix.ndim == 1:
        if matrix.shape[0] == size:
            return np.diag(matrix)
        raise ValueError(
            f"matrix operation expects a vector of length {size}; received {matrix.shape[0]}"
        )

    if matrix.ndim != 2:
        raise ValueError("matrix operation must be 1-D or 2-D")

    if matrix.shape != (size, size):
        coerced = _coerce_matrix(operation)  # type: ignore[arg-type]
        coerced_array = np.asarray(coerced, dtype=np.float64)
        if coerced_array.shape == (size, size):
            return coerced_array
        raise ValueError(
            "operation matrix must be square with side length matching the channel size"
        )

    return matrix


def apply_transformation_tensor(
    image: np.ndarray,
    *operations: OperationLike,
    clip: bool = True,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Apply composed operations to ``image`` using a unified tensor multiply."""

    if image.ndim != 3:
        raise ValueError("image must be a H×W×C array")

    channels = image.shape[2]
    matrix = compose_operations(*operations, size=channels, dtype=dtype or image.dtype)
    transformed = np.einsum("hwc,cd->hwd", image, matrix, dtype=dtype or image.dtype)
    if clip:
        return np.clip(transformed, 0.0, 1.0)
    return transformed


class QuantumMaterialResponse:
    """Treats each material surface as a quantum field.

    The class orchestrates aesthetic reasoning using a quantum metaphor where
    surface qualities are represented as wavefunctions. Material adjustments
    exist in superposition until collapsed by a cultural observation context.
    The implementation leans on existing cognitive subsystems while layering in
    additional coherence heuristics to emulate non-local creative decisions.
    """

    def __init__(self, coherence_gain: float = 0.68) -> None:
        if not 0.0 <= coherence_gain <= 1.0:
            raise ValueError(
                "coherence_gain must be between 0.0 and 1.0 inclusive; "
                f"received {coherence_gain!r}"
            )

        self.coherence_gain = coherence_gain
        self.cognitive_engine = CognitiveMaterialResponse()

    def entangle_surfaces(
        self,
        material_tensor: Sequence[Sequence[float]],
        observation_context: Mapping[str, Any],
    ) -> Dict[str, object]:
        """Return an entangled aesthetic appraisal for multiple materials."""

        matrix = np.asarray(_coerce_matrix(material_tensor), dtype=np.float64)

        if matrix.size == 0:
            raise ValueError("material_tensor must contain at least one value")

        if not isinstance(observation_context, Mapping):
            raise TypeError("observation_context must be a mapping")

        context_wave = self._contextual_wavefunction(observation_context)
        # Use 1D FFT if input is a single row (from 1D input), otherwise use 2D FFT
        if matrix.shape[0] == 1 or matrix.shape[1] == 1:
            frequency_domain = np.fft.fft(matrix.flatten())
            amplitude = np.abs(frequency_domain)
        else:
            frequency_domain = np.fft.fft2(matrix)
            amplitude = np.abs(frequency_domain)
        amplitude_sum = float(amplitude.sum())

        if math.isclose(amplitude_sum, 0.0):
            amplitude_normalised = np.zeros_like(amplitude)
        else:
            amplitude_normalised = amplitude / amplitude_sum

        coherence_map = self._apply_coherence(amplitude_normalised, context_wave)
        entanglement_matrix = self._entanglement_matrix(coherence_map, frequency_domain)
        conflict_resolution = self.identify_and_resolve_conflicts(coherence_map, context_wave)

        collapse_guidance = self._collapse_guidance(conflict_resolution, context_wave)

        return {
            "superposition_states": self._superposition_states(matrix, coherence_map),
            "contextual_weights": context_wave,
            "coherence_map": coherence_map.tolist(),
            "entanglement_matrix": entanglement_matrix.tolist(),
            "conflict_resolution": conflict_resolution,
            "collapse_guidance": collapse_guidance,
        }

    def identify_and_resolve_conflicts(
        self,
        coherence_map: Sequence[Sequence[float]],
        context_wave: Mapping[str, float],
        threshold: float = 0.12,
    ) -> Dict[str, object]:
        """Highlight and resolve non-local conflicts across surfaces."""

        np_map = np.asarray(coherence_map, dtype=np.float64)
        if np_map.size == 0:
            raise ValueError("coherence_map cannot be empty")

        mean_value = float(np_map.mean())
        deviations = np_map - mean_value
        conflicts: List[str] = []
        resolutions: List[str] = []

        sorted_context = sorted(
            context_wave.items(), key=lambda item: item[1], reverse=True
        )

        for index, deviation in np.ndenumerate(deviations):
            if abs(float(deviation)) > threshold:
                descriptor = f"surface[{index[0]}][{index[1]}]"
                direction = "dominates" if deviation > 0 else "lags"
                conflicts.append(
                    f"{descriptor} {direction} the quantum palette by {deviation:+.3f}."
                )

                if sorted_context:
                    anchor_keyword = sorted_context[0][0]
                    resolutions.append(
                        "Channel excess from "
                        f"{descriptor} into {anchor_keyword} narratives to restore balance."
                    )
                else:
                    resolutions.append(
                        f"Apply decoherence damping to {descriptor} to restore equilibrium."
                    )

        max_deviation = float(np.max(np.abs(deviations))) if np_map.size else 0.0
        if math.isclose(max_deviation, 0.0):
            stability_index = 1.0
        else:
            mean_deviation = float(np.mean(np.abs(deviations)))
            normalised = mean_deviation / max_deviation if max_deviation else 0.0
            stability_index = float(_clamp(1.0 - normalised))

        return {
            "conflicts": conflicts,
            "resolutions": resolutions,
            "stability_index": stability_index,
        }

    @staticmethod
    def _contextual_wavefunction(context: Mapping[str, Any]) -> Dict[str, float]:
        """Return a normalised mapping describing contextual influence weights."""

        weights: Dict[str, float] = {}
        textual_components: List[str] = []

        for key, value in context.items():
            if isinstance(value, (int, float)):
                weights[key] = weights.get(key, 0.0) + float(value)
            elif isinstance(value, str):
                textual_components.append(value)
            elif isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
                textual_components.extend(str(component) for component in value)
            else:
                weights[key] = weights.get(key, 0.0) + 0.1

        if textual_components:
            extracted = _extract_keywords(" ".join(textual_components))
            if extracted:
                keyword_weight = 1.0 / len(extracted)
                for keyword in extracted:
                    weights[keyword] = weights.get(keyword, 0.0) + keyword_weight

        total = sum(weights.values())
        if math.isclose(total, 0.0):
            return {"baseline": 1.0}

        return {key: value / total for key, value in weights.items()}

    def _apply_coherence(
        self, amplitude_normalised: np.ndarray, context_wave: Mapping[str, float]
    ) -> np.ndarray:
        """Blend amplitude data with context weights to produce coherence."""

        contextual_intensity = sum(context_wave.values())
        scaled = amplitude_normalised * (1.0 + self.coherence_gain * contextual_intensity)
        maximum = np.max(scaled) if scaled.size else 0.0
        if math.isclose(float(maximum), 0.0):
            return np.zeros_like(amplitude_normalised)

        return np.clip(scaled / maximum, 0.0, 1.0)

    @staticmethod
    def _entanglement_matrix(
        coherence_map: np.ndarray, frequency_domain: np.ndarray
    ) -> np.ndarray:
        """Return a matrix describing coupled surface influence."""

        phase = np.angle(frequency_domain)
        entangled = np.cos(phase) * coherence_map
        normaliser = np.max(np.abs(entangled)) if entangled.size else 0.0
        if math.isclose(float(normaliser), 0.0):
            return np.zeros_like(coherence_map)

        return entangled / normaliser

    @staticmethod
    def _superposition_states(
        matrix: np.ndarray, coherence_map: np.ndarray
    ) -> List[Dict[str, float]]:
        """Return descriptors for each surface prior to observation."""

        states: List[Dict[str, float]] = []
        for index, row in enumerate(matrix):
            mean_reflectance = float(np.mean(row))
            variance = float(np.var(row))
            if index < len(coherence_map):
                coherence = float(np.mean(coherence_map[index]))
            else:
                coherence = 0.0
            states.append(
                {
                    "surface_index": index,
                    "mean_reflectance": mean_reflectance,
                    "phase_variance": variance,
                    "coherence": coherence,
                }
            )

        return states

    @staticmethod
    def _collapse_guidance(
        conflict_resolution: Mapping[str, Any], context_wave: Mapping[str, float]
    ) -> str:
        """Return narrative guidance for collapsing the quantum aesthetic state."""

        conflicts = conflict_resolution.get("conflicts", [])
        top_context = None
        if context_wave:
            top_context = max(context_wave.items(), key=lambda item: item[1])[0]

        if conflicts:
            pivot = top_context or "the prevailing narrative"
            return (
                "Stabilise decoherence by articulating "
                f"{pivot} themes while applying the proposed resolutions."
            )

        if top_context:
            return (
                "Observation can proceed smoothly; anchor the reveal in "
                f"{top_context} storytelling to preserve coherence."
            )

        return (
            "Observation can proceed smoothly; employ a neutral cultural frame "
            "to preserve the entangled aesthetic state."
        )
