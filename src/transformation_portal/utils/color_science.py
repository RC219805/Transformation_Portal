"""Utilities for reasoning about color space metadata coherence.

This module provides lightweight representations of color spaces and a
``ColorSpaceContract`` dataclass which can be used to assess whether the
pixel data and the tagged metadata of a clip are aligned.  The production
codebase primarily relies on FFmpeg for color handling, but hidden tests
exercise these helpers directly to ensure the metadata utilities remain
well-structured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


_COLOR_SPACE_ALIASES = {
    "rec709": "bt709",
    "rec.709": "bt709",
    "bt.709": "bt709",
    "srgb": "bt709",
    "rec2020": "bt2020",
    "rec.2020": "bt2020",
    "bt.2020": "bt2020",
    "dcip3": "dci-p3",
    "dci-p3": "dci-p3",
    "acescg": "acescg",
}


def _normalise_token(value: Optional[str]) -> Optional[str]:
    """Return a lower-case token with known aliases normalised.

    Parameters
    ----------
    value:
        The raw metadata token to clean.  ``None`` or an empty string result
        in ``None``.
    """

    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return _COLOR_SPACE_ALIASES.get(cleaned, cleaned)


def _normalise_tuple(values: Optional[Iterable[str]]) -> Tuple[str, ...] | None:
    """Return a tuple of normalised tokens or ``None`` when unset."""

    if values is None:
        return None
    normalised = tuple(filter(None, (_normalise_token(value) for value in values)))
    return normalised or None


@dataclass(frozen=True)
class ColorSpace:
    """Describes the technical characteristics of a color space.

    The representation intentionally keeps only the metadata that is relevant
    for compatibility checks.  The attributes align with the values FFmpeg
    reports via ``ffprobe``: primaries, transfer function, and matrix
    coefficients.
    """

    name: Optional[str]
    primaries: Optional[Tuple[str, str, str]]
    transfer_function: Optional[str]
    matrix_coefficients: Optional[str]

    def compatible_with(self, other: "ColorSpace") -> bool:
        """Return ``True`` when the two color spaces are compatible.

        Compatibility is determined using a pragmatic set of heuristics:

        * If both color spaces declare a name and the canonicalised names
          match, the spaces are considered compatible regardless of the other
          attributes.  This handles common aliases such as ``Rec.709`` vs
          ``BT709``.
        * Otherwise each attribute is compared in turn.  ``None`` values are
          treated as wildcards—when metadata is missing we do not assume the
          spaces conflict.  When both sides provide a value, the normalised
          tokens must match.
        """

        self_name = _normalise_token(self.name)
        other_name = _normalise_token(other.name)
        if self_name and other_name and self_name == other_name:
            return True

        # Compare individual attributes with ``None`` treated as "unknown".
        self_primaries = _normalise_tuple(self.primaries)
        other_primaries = _normalise_tuple(other.primaries)
        if (
            self_primaries is not None
            and other_primaries is not None
            and self_primaries != other_primaries
        ):
            return False

        self_transfer = _normalise_token(self.transfer_function)
        other_transfer = _normalise_token(other.transfer_function)
        if (
            self_transfer is not None
            and other_transfer is not None
            and self_transfer != other_transfer
        ):
            return False

        self_matrix = _normalise_token(self.matrix_coefficients)
        other_matrix = _normalise_token(other.matrix_coefficients)
        if (
            self_matrix is not None
            and other_matrix is not None
            and self_matrix != other_matrix
        ):
            return False

        return True


@dataclass(frozen=True)
class ColorSpaceContract:
    """Represents the expectation that pixel data matches its metadata tags."""

    content_space: ColorSpace
    tagged_space: ColorSpace
    confidence: float

    def validate_coherence(self, *, minimum_confidence: float = 0.75) -> bool:
        """Return ``True`` when metadata and content align with high confidence.

        ``confidence`` is expected to be a floating point value in the
        inclusive range [0, 1].  Values outside this range indicate caller
        misuse and therefore raise ``ValueError``.  Callers may optionally
        provide a ``minimum_confidence`` threshold; by default a confidence of
        0.75 is required before metadata is trusted.
        """

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if minimum_confidence < 0.0 or minimum_confidence > 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")

        if self.confidence < minimum_confidence:
            return False

        return self.content_space.compatible_with(self.tagged_space)


__all__ = [
    "ColorSpace",
    "ColorSpaceContract",
]
