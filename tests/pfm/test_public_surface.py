"""Public-surface smoke tests for ``transformation_portal.pfm``.

PFM (Pipeline Foundation Model) bundles dataset/tokenizer/model utilities.
``tokenizer.py`` and ``model.py`` eagerly import ``torch``, so the smoke
tests skip when torch is not installed. Once a follow-up makes torch
imports lazy (per CLAUDE.md), these can be promoted to ``unit`` marks.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.ml]


@pytest.fixture(scope="module")
def pfm_pkg():
    pytest.importorskip("torch")
    import transformation_portal.pfm as pkg

    return pkg


def test_package_importable(pfm_pkg):
    assert pfm_pkg is not None


def test_declared_all_resolves(pfm_pkg):
    for name in pfm_pkg.__all__:
        assert hasattr(pfm_pkg, name), f"declared in __all__ but missing: {name}"


def test_data_pipeline_records_are_dataclasses(pfm_pkg):
    import dataclasses

    from transformation_portal.pfm import RunRecord, StepRecord

    assert dataclasses.is_dataclass(RunRecord)
    assert dataclasses.is_dataclass(StepRecord)


def test_build_sequence_is_callable(pfm_pkg):
    from transformation_portal.pfm import build_sequence

    assert callable(build_sequence)


def test_tokenizer_is_class(pfm_pkg):
    from transformation_portal.pfm import PFMTokenizer

    assert isinstance(PFMTokenizer, type)
