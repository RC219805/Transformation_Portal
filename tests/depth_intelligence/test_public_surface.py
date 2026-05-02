"""Public-surface smoke tests for ``transformation_portal.depth_intelligence``.

The package's ``__init__.py`` historically imported five submodules, four
of which never existed — making the package unimportable. It now exposes
its public symbols via PEP 562 lazy exports (mirroring
``transformation_portal.depth``) so the package itself is importable in
core/offline environments. Resolving any symbol still pulls in
``torch``/``numpy``/``PIL`` via ``depth_estimator``; those checks are
gated on ``pytest.importorskip("torch")``.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_package_imports_without_ml_stack():
    # The package itself must load lazily — no torch needed at import.
    import transformation_portal.depth_intelligence as pkg

    assert pkg.__version__
    assert "DepthEstimator" in pkg.__all__


@pytest.mark.unit
def test_unknown_attribute_raises_attribute_error():
    import transformation_portal.depth_intelligence as pkg

    with pytest.raises(AttributeError):
        pkg.NotARealSymbol  # noqa: B018 — exercising __getattr__


@pytest.mark.unit
def test_dir_advertises_declared_all():
    import transformation_portal.depth_intelligence as pkg

    listed = set(dir(pkg))
    for name in pkg.__all__:
        assert name in listed, f"declared in __all__ but missing from dir(): {name}"


@pytest.mark.ml
def test_lazy_resolution_returns_expected_classes():
    pytest.importorskip("torch")
    import dataclasses

    from transformation_portal.depth_intelligence import (
        DepthConfig,
        DepthEstimator,
        DepthMap,
    )

    assert isinstance(DepthEstimator, type)
    assert dataclasses.is_dataclass(DepthConfig)
    assert dataclasses.is_dataclass(DepthMap)


@pytest.mark.ml
def test_declared_all_resolves_under_full_stack():
    pytest.importorskip("torch")
    import transformation_portal.depth_intelligence as pkg

    for name in pkg.__all__:
        assert getattr(pkg, name) is not None, f"declared in __all__ but resolution failed: {name}"
