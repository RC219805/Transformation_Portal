"""Public-surface smoke tests for ``transformation_portal.interfaces``.

The interfaces package provides the abstract base classes and Protocols
referenced by ADR-001 ("module interface contracts"). These smoke tests
confirm the package imports cleanly offline, every name in ``__all__``
resolves, and the abstractness contract on each ABC is enforced
(``__abstractmethods__`` non-empty), so an accidental drop of
``@abstractmethod`` is caught as a real failure rather than a silent
contract drift.
"""

from __future__ import annotations

import abc

import pytest

pytestmark = [pytest.mark.unit]


def test_package_importable():
    import transformation_portal.interfaces as pkg

    assert pkg.__version__


def test_declared_all_resolves():
    import transformation_portal.interfaces as pkg

    for name in pkg.__all__:
        assert hasattr(pkg, name), f"declared in __all__ but missing: {name}"


def _assert_is_abc_with_abstract_methods(cls: type) -> None:
    """Pin the ABC contract: must be ABCMeta-flavored and have at least one
    declared abstract method. We avoid instantiating the class (which would
    work-by-side-effect with ``pytest.raises(TypeError)`` but also trip the
    pylint ``abstract-class-instantiated`` error)."""
    assert isinstance(cls, abc.ABCMeta), f"{cls.__name__} must use ABCMeta (or subclass)"
    abstracts = getattr(cls, "__abstractmethods__", frozenset())
    assert abstracts, f"{cls.__name__} declares no abstract methods — abstractness contract is gone"


def test_processor_interfaces_are_abstract():
    from transformation_portal.interfaces import ImageProcessor, VideoProcessor

    for cls in (ImageProcessor, VideoProcessor):
        _assert_is_abc_with_abstract_methods(cls)


def test_pipeline_interfaces_are_abstract():
    from transformation_portal.interfaces import BatchPipeline, Pipeline, PipelineStage

    for cls in (Pipeline, PipelineStage, BatchPipeline):
        _assert_is_abc_with_abstract_methods(cls)


def test_error_classes_subclass_exception():
    from transformation_portal.interfaces import (
        EnhancementError,
        EstimationError,
        PipelineError,
        ProcessingError,
        SegmentationError,
    )

    for err in (
        ProcessingError,
        PipelineError,
        EnhancementError,
        SegmentationError,
        EstimationError,
    ):
        assert issubclass(err, Exception)


def test_material_type_is_enum_like():
    from transformation_portal.interfaces import MaterialType

    # MaterialType should be enumerable and have at least one member; we
    # don't pin specific values so this stays robust to taxonomy growth.
    members = list(MaterialType)
    assert members
    for member in members:
        assert isinstance(member, MaterialType)
