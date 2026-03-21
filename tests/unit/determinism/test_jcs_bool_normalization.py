"""Unit tests for determinism/JCS boolean normalization."""

import json

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from transformation_portal.determinism.ingest import probe_subnormals_preserved
from transformation_portal.determinism.jcs import dumps


@pytest.mark.unit
def test_probe_subnormals_preserved_returns_python_bool():
    """FTZ/DAZ probe must return a native bool for downstream evidence serialization."""
    result = probe_subnormals_preserved()
    assert isinstance(result, bool)


@pytest.mark.unit
def test_jcs_rejects_numpy_bool_scalar():
    """Document current JCS contract: NumPy scalar booleans are not supported JSON types."""
    with pytest.raises(TypeError, match="Unsupported type for JCS serialization"):
        dumps({"value": np.bool_(True)})


@pytest.mark.unit
def test_jcs_serializes_ingest_ftz_probe_after_bool_normalization():
    """Ingest FTZ probe evidence should serialize after explicit bool normalization."""
    ingest_record = {
        "ftz_daz_probe": {
            "subnormals_preserved": bool(probe_subnormals_preserved()),
            "policy": "fail_closed",
        }
    }

    payload = dumps(ingest_record)
    parsed = json.loads(payload)
    assert isinstance(parsed["ftz_daz_probe"]["subnormals_preserved"], bool)
