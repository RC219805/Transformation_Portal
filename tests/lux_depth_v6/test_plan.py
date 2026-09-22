"""Frozen implementation authority and complete preview/output admission."""

from dataclasses import replace
from pathlib import Path

import pytest

from transformation_portal.lux_depth_v6 import plan as plan_module
from transformation_portal.lux_depth_v6 import products
from transformation_portal.lux_depth_v6.pipeline import run
from transformation_portal.lux_depth_v6.plan import (
    MAX_PLAN_BYTES,
    LuxDepthV6Request,
    OutputLimits,
    digest,
    prepare,
    processing_identity,
)
from transformation_portal.lux_depth_v6.source import prepare_source

pytestmark = pytest.mark.unit


def test_processing_identity_binds_product_encoder_source():
    assert processing_identity()["modules"][products.__name__] == digest(Path(products.__file__).read_bytes())


def test_product_source_change_invalidates_prepared_execution(completed, tmp_path, monkeypatch):
    prepared = prepare(LuxDepthV6Request(completed.result.output_root, tmp_path / "v6"))
    original = Path(products.__file__).read_bytes()
    replacement = tmp_path / "changed_products.py"
    replacement.write_bytes(original + b"\n# changed processing implementation\n")
    monkeypatch.setattr(products, "__file__", str(replacement))
    with pytest.raises(ValueError, match="processing"):
        run(prepared)
    assert not prepared.output_root.exists()


def test_source_limit_changes_invalidate_prepared_execution(completed, tmp_path):
    prepared = prepare(LuxDepthV6Request(completed.result.output_root, tmp_path / "v6"))
    altered_limits = replace(prepared.source.limits, memory_mib=prepared.source.limits.memory_mib + 1)
    altered = replace(prepared, source=replace(prepared.source, limits=altered_limits))
    with pytest.raises(ValueError, match="source"):
        run(altered)
    assert not prepared.output_root.exists()


def test_multi_image_rgba_reservation_includes_every_png_preview(completed, tmp_path, monkeypatch):
    source = prepare_source(completed.result.output_root)
    # Three float RGB masters, float alpha, and 16-bit RGBA TIFF already consume
    # 48 bytes/pixel. Each 1600px RGBA preview can need another ~10 MiB. The old
    # 2 MiB/image allowance plus one global plan allowance cannot cover a batch.
    images = tuple(replace(source.images[0], input_id=f"input-{index:04d}", shape=(1600, 1600)) for index in range(4))
    source = replace(source, images=images)
    monkeypatch.setattr(plan_module, "prepare_source", lambda *_args, **_kwargs: source)
    insufficient_budget = 4 * (1600 * 1600 * 48 + 2 * 1024**2) + MAX_PLAN_BYTES
    request = LuxDepthV6Request(source.root, tmp_path / "v6", output_limits=OutputLimits(max_output_bytes=insufficient_budget))
    with pytest.raises(ValueError, match="reservation"):
        prepare(request)
    assert not request.output_dir.exists()


@pytest.mark.parametrize("field,value", [("max_pixels", 1), ("max_input_bytes", 1), ("memory_mib", 1), ("memory_mib", 128)])
def test_rebuilt_plan_cannot_bypass_source_admission(completed, tmp_path, field, value):
    from transformation_portal.ingest.canonical_json import canonicalize_json
    from transformation_portal.lux_depth_v6.plan import GradePlan

    prepared = prepare(LuxDepthV6Request(completed.result.output_root, tmp_path / "v6"))
    impossible = replace(prepared.source.limits, **{field: value})
    altered_source = replace(prepared.source, limits=impossible)
    payload = prepared.plan.to_payload()
    payload["source"]["limits"] = impossible.to_payload()
    with pytest.raises(ValueError, match="budget|ceiling|admission"):
        altered = replace(prepared, plan=GradePlan(canonicalize_json(payload)), source=altered_source)
        run(altered)
    assert not prepared.output_root.exists()


def test_source_root_replaced_by_link_is_rejected_before_execution(completed, tmp_path):
    from transformation_portal.lux_depth_v3.execution_evidence import ArtifactEvidenceError

    prepared = prepare(LuxDepthV6Request(completed.result.output_root, tmp_path / "v6"))
    retained = tmp_path / "moved-parent"
    completed.result.output_root.rename(retained)
    completed.result.output_root.symlink_to(retained, target_is_directory=True)
    with pytest.raises((ValueError, OSError, ArtifactEvidenceError)):
        run(prepared)
    assert not prepared.output_root.exists()


def test_late_product_write_failure_preserves_partial_output_without_completion(completed, tmp_path, monkeypatch):
    from transformation_portal.lux_depth_v3.execution_evidence import ArtifactEvidenceError
    from transformation_portal.lux_depth_v6 import pipeline

    prepared = prepare(LuxDepthV6Request(completed.result.output_root, tmp_path / "v6"))
    real_write = pipeline._secure_atomic_write_bytes

    def fail_master_write(root, relative, data, **kwargs):
        if relative == "input-0000/master.npy":
            raise OSError("simulated delivery disk failure")
        return real_write(root, relative, data, **kwargs)

    monkeypatch.setattr(pipeline, "_secure_atomic_write_bytes", fail_master_write)
    with pytest.raises(ArtifactEvidenceError) as error:
        run(prepared)
    assert isinstance(error.value.__cause__, OSError)
    assert "simulated delivery disk failure" in str(error.value.__cause__)
    assert (prepared.output_root / "input-0000/baseline.npy").is_file()
    assert not (prepared.output_root / "evidence.json").exists()
