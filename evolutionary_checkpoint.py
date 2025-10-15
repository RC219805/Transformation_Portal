"""Evolutionary guardrails for the codebase's long-term trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class EvolutionaryCheckpoint:
    """Represents an evolutionary deadline for a particular workflow.

    The checkpoint keeps track of a *horizon* (a :class:`~datetime.date` after
    which a migration must be pursued) and the ``mutation_path`` that should be
    followed once the horizon has been crossed.

    The :meth:`evolve_or_alert` method returns human readable guidance that can
    be surfaced in dashboards or CI logs.  It also accepts an optional
    ``today`` override to make the class simple to test without having to rely
    on the ambient system clock.
    """

    horizon: date
    mutation_path: str

    def evolve_or_alert(self, *, today: date | None = None) -> str:
        """Return guidance about whether evolution is required.

        Parameters
        ----------
        today:
            Optional date to use instead of :func:`datetime.date.today`.  This
            is primarily useful for deterministic testing.
        """

        reference_date = today or date.today()
        if reference_date > self.horizon:
            return f"EVOLUTION REQUIRED: Migrate to {self.mutation_path}"
        return f"STABLE: Current form viable until {self.horizon.isoformat()}"
