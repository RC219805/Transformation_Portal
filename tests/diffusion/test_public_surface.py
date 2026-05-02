"""Public-surface smoke tests for ``transformation_portal.diffusion``.

The diffusion package wraps FLUX.1 / ControlNet for architectural
enhancement. Its sub-modules eagerly import ``torch`` and ``cv2``; cv2
is part of the core lockfile, but torch is an optional ML dependency.
These smoke tests skip when torch is unavailable so they remain green
on the offline/core CI lane and exercise the public surface on the ML
lane.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.ml]


@pytest.fixture(scope="module")
def diffusion_pkg():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    import transformation_portal.diffusion as pkg

    return pkg


def test_package_importable(diffusion_pkg):
    assert diffusion_pkg is not None


def test_declared_all_resolves(diffusion_pkg):
    for name in diffusion_pkg.__all__:
        assert hasattr(diffusion_pkg, name), f"declared in __all__ but missing: {name}"


def test_public_classes_are_classes(diffusion_pkg):
    from transformation_portal.diffusion import (
        ArchitecturalPromptBuilder,
        FLUXControlNet,
        FLUXPipeline,
    )

    for cls in (FLUXPipeline, FLUXControlNet, ArchitecturalPromptBuilder):
        assert isinstance(cls, type)


def test_architectural_prompt_builder_offline_construction(diffusion_pkg):
    # Prompt builder is a dataclass-like helper that must work without a GPU.
    from transformation_portal.diffusion import ArchitecturalPromptBuilder

    builder = ArchitecturalPromptBuilder()
    assert builder is not None
