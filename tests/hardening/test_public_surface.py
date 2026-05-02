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
    from transformation_portal.hardening import Pipeline

    # ``issubclass(Pipeline, Protocol)`` would raise TypeError because
    # ``typing.Protocol`` itself is not runtime-checkable. Use the typing
    # module's private flags that are stable across 3.11/3.12 instead:
    # ``_is_protocol`` is set by ``Protocol`` and ``_is_runtime_protocol``
    # is set by ``@runtime_checkable``. Both must be true for duck-typed
    # implementations to pass ``isinstance`` checks against ``Pipeline``.
    assert getattr(Pipeline, "_is_protocol", False) is True, "Pipeline must be a typing.Protocol"
    assert getattr(Pipeline, "_is_runtime_protocol", False) is True, "Pipeline must be @runtime_checkable"

    # Positive isinstance check: a duck-typed object exposing the
    # documented `process(...)` method satisfies the protocol.
    class _DuckPipeline:
        def process(self, input_path, **kwargs):  # noqa: ARG002 - signature pin
            return None

    assert isinstance(_DuckPipeline(), Pipeline)


def test_wrap_function_is_callable():
    from transformation_portal.hardening import wrap_function

    assert callable(wrap_function)
