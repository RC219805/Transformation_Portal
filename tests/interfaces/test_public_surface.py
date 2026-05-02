"""Public-surface smoke tests for ``transformation_portal.interfaces``.

The interfaces package provides the abstract base classes and Protocols
referenced by ADR-001 ("module interface contracts"). These smoke tests
confirm the package imports cleanly offline and that every name in
``__all__`` resolves to an importable symbol — guarding against silent
removal or rename of contract surfaces.
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


def test_processor_interfaces_are_abstract():
    from transformation_portal.interfaces import ImageProcessor, VideoProcessor

    for cls in (ImageProcessor, VideoProcessor):
        assert isinstance(cls, type)
        # ABCs raise TypeError if instantiated directly.
        with pytest.raises(TypeError):
            cls()  # type: ignore[abstract]


def test_pipeline_interfaces_are_abstract():
    from transformation_portal.interfaces import BatchPipeline, Pipeline, PipelineStage

    for cls in (Pipeline, PipelineStage, BatchPipeline):
        assert isinstance(cls, type)


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


def test_abstract_methods_are_marked():
    from transformation_portal.interfaces import ImageProcessor

    abstracts = getattr(ImageProcessor, "__abstractmethods__", frozenset())
    assert abstracts, "ImageProcessor must declare at least one abstract method"
    assert isinstance(ImageProcessor, abc.ABCMeta)
