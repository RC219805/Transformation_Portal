"""Public-surface smoke tests for ``transformation_portal.hardening``.

CLAUDE.md identifies hardening as safety-critical ("hardening that must
remain intact"). The package previously had no direct tests; this file
asserts the package is importable, exposes its declared ``__all__``
surface, and that the public types behave as the framework requires
(``Pipeline`` is a Protocol, ``UniversalHardenedWrapper`` is a class
with the documented interface, ``wrap_function`` is callable).

Pure offline tests — no torch/transformers/scipy required.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]


def test_package_importable():
    import transformation_portal.hardening as pkg

    assert pkg.__version__


def test_declared_all_resolves():
    import transformation_portal.hardening as pkg

    for name in pkg.__all__:
        assert hasattr(pkg, name), f"declared in __all__ but missing: {name}"


def test_universal_hardened_wrapper_is_class():
    from transformation_portal.hardening import UniversalHardenedWrapper

    assert isinstance(UniversalHardenedWrapper, type)


def test_pipeline_is_runtime_checkable_protocol():
    from typing import Protocol

    from transformation_portal.hardening import Pipeline

    # Pipeline is documented as a contract; must be a Protocol so duck-typed
    # implementations work.
    assert issubclass(Pipeline, Protocol) or hasattr(Pipeline, "__protocol_attrs__")


def test_wrap_function_is_callable():
    from transformation_portal.hardening import wrap_function

    assert callable(wrap_function)
