"""Public-surface smoke tests for ``transformation_portal.depth_intelligence``.

The package's ``__init__.py`` historically imported five submodules, four
of which never existed — making the package unimportable. The current
``__init__.py`` re-exports only the symbols backed by ``depth_estimator``
which is itself eager-torch. Smoke tests are gated on torch availability
and pin the importable surface so any future regression (or accidental
restoration of the broken imports) is caught.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.ml]


@pytest.fixture(scope="module")
def depth_intel_pkg():
    pytest.importorskip("torch")
    import transformation_portal.depth_intelligence as pkg

    return pkg


def test_package_importable(depth_intel_pkg):
    assert depth_intel_pkg.__version__


def test_declared_all_matches_existing_modules(depth_intel_pkg):
    # If __all__ ever grows new names, the corresponding module must exist
    # and the symbol must be importable.
    for name in depth_intel_pkg.__all__:
        assert hasattr(depth_intel_pkg, name), f"declared in __all__ but missing: {name}"


def test_currently_exposes_depth_estimator_surface(depth_intel_pkg):
    from transformation_portal.depth_intelligence import (
        DepthConfig,
        DepthEstimator,
        DepthMap,
    )

    assert isinstance(DepthEstimator, type)
    # DepthConfig and DepthMap are dataclasses providing the contract surface.
    import dataclasses

    assert dataclasses.is_dataclass(DepthConfig)
    assert dataclasses.is_dataclass(DepthMap)
