"""Reject vulnerable FastVLM dependency pins in the isolated runtime closure."""

from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

pytestmark = [pytest.mark.unit, pytest.mark.security]

REQUIREMENTS = Path(__file__).resolve().parents[2] / "config/fastvlm_runtime_requirements.txt"


@pytest.mark.parametrize(
    ("distribution", "minimum_version"),
    [
        ("fsspec", "2026.6.0"),  # GHSA-27vj-qcqg-25rc: ReferenceFileSystem template injection.
        ("multidict", "6.9.1"),  # GHSA-54p9-h82j-f925: items-view reference leak.
    ],
)
def test_fastvlm_dependency_security_floor(distribution: str, minimum_version: str) -> None:
    requirements = [
        Requirement(line)
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    matching = [requirement for requirement in requirements if requirement.name.lower() == distribution]

    assert len(matching) == 1, f"FastVLM must contain one exact {distribution} pin"
    specifiers = list(matching[0].specifier)
    assert len(specifiers) == 1 and specifiers[0].operator == "=="
    assert Version(specifiers[0].version) >= Version(minimum_version)
