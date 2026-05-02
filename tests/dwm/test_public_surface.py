"""Public-surface smoke tests for ``transformation_portal.dwm``.

DWM (Diffusion World Model) predicts pipeline outcomes via latent-space
diffusion. ``model.py`` eagerly imports ``torch``, so the smoke tests
skip when torch is unavailable.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.ml]


@pytest.fixture(scope="module")
def dwm_pkg():
    pytest.importorskip("torch")
    import transformation_portal.dwm as pkg

    return pkg


def test_package_importable(dwm_pkg):
    assert dwm_pkg is not None


def test_declared_all_resolves(dwm_pkg):
    for name in dwm_pkg.__all__:
        assert hasattr(dwm_pkg, name), f"declared in __all__ but missing: {name}"


def test_public_classes_are_classes(dwm_pkg):
    from transformation_portal.dwm import DiffusionSchedule, DiffusionWorldModel

    for cls in (DiffusionWorldModel, DiffusionSchedule):
        assert isinstance(cls, type)
