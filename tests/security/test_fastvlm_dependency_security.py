"""Reject vulnerable FastVLM dependency pins in the isolated runtime closure."""

from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

pytestmark = [pytest.mark.unit, pytest.mark.security]

REQUIREMENTS = Path(__file__).resolve().parents[2] / "config/fastvlm_runtime_requirements.txt"


@pytest.mark.parametrize(
    ("distribution", "minimum_version"),
    [
        ("aiohttp", "3.14.4"),  # GHSA-2g87-pp6c-29x5 and related parser/resource fixes.
        ("brotli", "1.2.1"),  # GHSA-9v8h-qqgm-4m35: decoder memory safety.
        ("certifi", "2026.10.19"),  # GHSA-6q48-6c7w-9gw2: compromised emSign roots.
        ("filelock", "4.2.1"),  # Minimum patched version for the October 2026 filelock advisories.
        ("pillow", "13.0.0"),  # GHSA-8g5g-8vmr-9782 and related image decoder fixes.
        ("setuptools", "84.1.0"),  # GHSA-xmj3-9gf7-h52w.
        ("yarl", "1.25.1"),  # Required by aiohttp 3.14.4.
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


def test_fastvlm_captioning_closure_excludes_incompatible_demo_ui_dependencies() -> None:
    names = {
        canonicalize_name(Requirement(line).name)
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    # Upstream Gradio demos impose Pillow<13; governed captioning uses mlx_vlm.generate.
    assert names.isdisjoint({"gradio", "gradio-client", "hf-gradio"})


def test_fastvlm_pydantic_core_matches_the_reviewed_pydantic_release() -> None:
    requirements = [
        Requirement(line)
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    # Pydantic 2.13.4 requires this exact core version; update the pair together.
    for distribution, version in (("pydantic", "2.13.4"), ("pydantic-core", "2.46.4")):
        matching = [requirement for requirement in requirements if canonicalize_name(requirement.name) == distribution]
        assert len(matching) == 1, f"FastVLM must contain one exact {distribution} pin"
        assert str(matching[0].specifier) == f"=={version}"
