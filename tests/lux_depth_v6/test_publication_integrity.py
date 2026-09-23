"""Late output mutations must never acquire V6 completion authority."""

from __future__ import annotations

import pytest

from tests.lux_depth_v5.test_pipeline import execute
from transformation_portal.lux_depth_v3.execution_evidence import ArtifactEvidenceError
from transformation_portal.lux_depth_v6 import evidence
from transformation_portal.lux_depth_v6.pipeline import run
from transformation_portal.lux_depth_v6.plan import LuxDepthV6Request, prepare

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("operation", ["execute", "verify"])
@pytest.mark.parametrize("mutation", ["changed", "removed", "extra"])
def test_outputs_remain_bound_during_final_source_verification(request_case, tmp_path, monkeypatch, operation, mutation):
    parent = execute(request_case).output_root
    prepared = prepare(LuxDepthV6Request(parent, tmp_path / "v6"))
    if operation == "verify":
        run(prepared)
    validate_source = evidence.validate_source

    def validate_then_mutate(*args, **kwargs):
        validate_source(*args, **kwargs)
        product = prepared.output_root / "input-0000/baseline.npy"
        if mutation == "changed":
            raw = product.read_bytes()
            product.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        elif mutation == "removed":
            product.unlink()
        else:
            (prepared.output_root / "undeclared.txt").write_bytes(b"late undeclared product")

    monkeypatch.setattr(evidence, "validate_source", validate_then_mutate)
    with pytest.raises((ValueError, OSError, ArtifactEvidenceError)):
        if operation == "execute":
            run(prepared)
        else:
            evidence.verify_execution_evidence(prepared.output_root, source_root=parent)
    if operation == "execute":
        assert not (prepared.output_root / "evidence.json").exists()
