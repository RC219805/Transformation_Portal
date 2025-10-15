"""Utilities for normalising temporal evolution roadmaps.

The repository's strategic planning documents often materialise as deeply
nested YAML structures.  This module provides small dataclasses that convert
those raw mappings into objects with ergonomic helpers for rendering human
readable summaries.  The entry point :meth:`TemporalEvolutionRoadmap.from_mapping`
accepts either the raw ``temporal_evolution`` block or a dictionary containing
it and validates the structure so downstream tooling can rely on consistent
shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, MutableSequence, Sequence


@dataclass(frozen=True)
class EvolutionDirective:
    """A single directive inside a temporal evolution discipline.

    Attributes
    ----------
    summary:
        Short label describing the directive.  When the source data uses a
        mapping (``{"Monitor": "…"}``), the mapping key becomes the summary.
    detail:
        Optional explanatory text.  This is taken from the mapping value when
        present, otherwise plain string entries are treated as summaries without
        extra detail.
    """

    summary: str
    detail: str | None = None

    def to_bullet(self) -> str:
        """Return a Markdown bullet point representation of the directive."""

        if self.detail:
            return f"- **{self.summary}**: {self.detail}"
        return f"- {self.summary}"

    def serialise(self) -> str | dict[str, str]:
        """Return a JSON/YAML friendly representation of the directive."""

        if self.detail is None:
            return self.summary
        return {self.summary: self.detail}


@dataclass(frozen=True)
class EvolutionDiscipline:
    """A named collection of :class:`EvolutionDirective` instances."""

    name: str
    directives: Sequence[EvolutionDirective]

    @property
    def human_name(self) -> str:
        """Return a prettified version of :attr:`name` for display."""

        return self.name.replace("_", " ").title()

    def to_markdown(self) -> str:
        """Render the discipline and its directives as Markdown."""

        lines = [f"### {self.human_name}"]
        lines.extend(directive.to_bullet() for directive in self.directives)
        return "\n".join(lines)

    def serialise(self) -> list[str | dict[str, str]]:
        """Return a serialisable list representation of the directives."""

        return [directive.serialise() for directive in self.directives]


@dataclass(frozen=True)
class TemporalEvolutionRoadmap:
    """Structured representation of the ``temporal_evolution`` document."""

    disciplines: Sequence[EvolutionDiscipline]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "TemporalEvolutionRoadmap":
        """Create a roadmap from a mapping derived from YAML or JSON.

        The input may either be the ``temporal_evolution`` block itself or a
        dictionary that contains it.  The method validates that each discipline
        is defined as a list of string or ``{summary: detail}`` entries and
        raises :class:`TypeError` when unexpected shapes are encountered.
        """

        root: object
        if "temporal_evolution" in payload:
            root = payload["temporal_evolution"]
        else:
            root = payload

        if not isinstance(root, Mapping):
            raise TypeError("temporal_evolution payload must be a mapping")

        disciplines: MutableSequence[EvolutionDiscipline] = []
        for raw_name, directives in root.items():
            if not isinstance(raw_name, str):
                raise TypeError("Discipline names must be strings")

            name = raw_name.strip()
            if not name:
                raise TypeError("Discipline names must not be empty")

            normalized = _normalise_directives(name, directives)
            disciplines.append(EvolutionDiscipline(name=name, directives=tuple(normalized)))

        return cls(disciplines=tuple(disciplines))

    def to_markdown(self) -> str:
        """Render the full roadmap as Markdown."""

        return "\n\n".join(discipline.to_markdown() for discipline in self.disciplines)

    def serialise(self) -> dict[str, list[str | dict[str, str]]]:
        """Return a serialisable representation of the roadmap."""

        return {discipline.name: discipline.serialise() for discipline in self.disciplines}


def _normalise_directives(name: str, directives: object) -> List[EvolutionDirective]:
    if isinstance(directives, (str, bytes)) or not isinstance(directives, Sequence):
        raise TypeError(
            f"Directives for discipline '{name}' must be a sequence of steps"
        )

    result: List[EvolutionDirective] = []
    for entry in directives:  # type: ignore[union-attr]
        if isinstance(entry, Mapping):
            if len(entry) != 1:
                raise TypeError(
                    f"Directive mappings for discipline '{name}' must contain a single entry"
                )
            summary, detail = next(iter(entry.items()))
            if not isinstance(summary, str) or not isinstance(detail, str):
                raise TypeError(
                    f"Directive mapping for discipline '{name}' must map strings to strings"
                )
            stripped_summary = summary.strip()
            if not stripped_summary:
                raise TypeError(
                    f"Directive mapping for discipline '{name}' must include a summary"
                )
            result.append(EvolutionDirective(stripped_summary, detail.strip()))
        elif isinstance(entry, str):
            stripped = entry.strip()
            if not stripped:
                continue
            result.append(EvolutionDirective(stripped))
        else:
            raise TypeError(
                f"Unsupported directive type for discipline '{name}': {type(entry)!r}"
            )

    return result


__all__ = [
    "EvolutionDirective",
    "EvolutionDiscipline",
    "TemporalEvolutionRoadmap",
]