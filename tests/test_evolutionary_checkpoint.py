"""Tests for :mod:`evolutionary_checkpoint`."""

from __future__ import annotations

from datetime import date

import pytest

from evolutionary_checkpoint import EvolutionaryCheckpoint


def test_evolution_required_message_when_horizon_has_passed() -> None:
    checkpoint = EvolutionaryCheckpoint(
        horizon=date(2024, 1, 1), mutation_path="lux/v2/pipeline"
    )

    message = checkpoint.evolve_or_alert(today=date(2024, 1, 2))

    assert message == "EVOLUTION REQUIRED: Migrate to lux/v2/pipeline"


def test_evolution_not_required_message_when_within_horizon() -> None:
    checkpoint = EvolutionaryCheckpoint(
        horizon=date(2024, 1, 10), mutation_path="lux/v3/pipeline"
    )

    message = checkpoint.evolve_or_alert(today=date(2024, 1, 5))

    assert (
        message
        == "STABLE: Current form viable until 2024-01-10"
    )


class _FrozenDate(date):
    """Helper date that lets us control :func:`date.today`."""

    @classmethod
    def today(cls) -> "_FrozenDate":
        return cls(2024, 1, 9)


def test_today_defaults_to_current_date(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = EvolutionaryCheckpoint(
        horizon=date(2024, 1, 10), mutation_path="lux/v3/pipeline"
    )

    monkeypatch.setattr("evolutionary_checkpoint.date", _FrozenDate)

    message = checkpoint.evolve_or_alert()

    assert message == "STABLE: Current form viable until 2024-01-10"
