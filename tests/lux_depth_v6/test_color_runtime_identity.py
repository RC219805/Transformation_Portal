"""Current CMS identity is required to execute, while historical plans still parse."""

from dataclasses import replace

import pytest

from tests.lux_depth_v6 import test_depth_pro_pipeline as native_fixtures
from tests.lux_depth_v6 import test_managed as managed_fixtures
from transformation_portal.core.execution_plan_v5 import ExecutionPlanV5
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.color_runtime import COLOR_PROCESSING_MODULES
from transformation_portal.lux_depth_v6 import depth_pro, managed, pipeline, plan

pytestmark = pytest.mark.unit
native_case = native_fixtures.native_case
prepared = managed_fixtures.fixture_prepared
refingerprint = managed_fixtures.refingerprint


@pytest.fixture(params=["grade", "depth_pro"])
def prepared_case(request, tmp_path):
    if request.param == "depth_pro":
        native_case = request.getfixturevalue("native_case")
        return depth_pro, depth_pro.NativeDepthProPlan, depth_pro.prepare(native_case)
    completed = request.getfixturevalue("completed")
    return pipeline, plan.GradePlan, plan.prepare(plan.LuxDepthV6Request(completed.result.output_root, tmp_path / "grade"))


def test_historical_processing_identity_parses_but_cannot_execute_as_current(prepared_case):
    executor, plan_type, prepared = prepared_case
    payload = prepared.plan.to_payload()
    payload["processing"].pop("color_preparation")
    for module in COLOR_PROCESSING_MODULES:
        payload["processing"]["modules"].pop(module)
    historical = plan_type(canonicalize_json(payload))
    assert historical.to_payload() == payload
    with pytest.raises(ValueError, match="processing|implementation"):
        executor.run(replace(prepared, plan=historical))
    assert not prepared.output_root.exists()


@pytest.mark.parametrize("mutation", ["cms_bytes", "wheel_bytes", "wrong_engine", "version_disagrees"])
def test_cms_identity_drift_fails_before_output(prepared_case, mutation):
    executor, plan_type, prepared = prepared_case
    payload = prepared.plan.to_payload()
    cms = payload["processing"]["color_preparation"]
    if mutation == "wrong_engine":
        cms["lcms_version"] = "unrecognized 1.0"
    elif mutation == "version_disagrees":
        cms["imagecodecs_version"] = "0.0.0"
    elif mutation == "cms_bytes":
        cms["cms_extension_sha256"] = "ab" * 32
    else:
        cms["wheel_files_sha256"] = "ab" * 32
    with pytest.raises(ValueError, match="CMS|processing|implementation"):
        changed = plan_type(canonicalize_json(payload))
        executor.run(replace(prepared, plan=changed))
    assert not prepared.output_root.exists()


def test_composite_historical_identity_parses_but_cannot_execute_as_current(prepared):
    payload = prepared.plan.to_payload()
    payload["processing"].pop("color_preparation")
    for module in COLOR_PROCESSING_MODULES:
        payload["processing"]["modules"].pop(module)
    historical = ExecutionPlanV5(refingerprint(payload))
    assert historical.to_payload() == payload
    with pytest.raises(ValueError, match="processing identity"):
        managed.run(replace(prepared, plan=historical))
    assert not prepared.output_root.exists()


@pytest.mark.parametrize("mutation", ["wrong_engine", "version_disagrees", "path_escape", "unbounded_inventory"])
def test_composite_color_runtime_schema_remains_closed(prepared, mutation):
    payload = prepared.plan.to_payload()
    cms = payload["processing"]["color_preparation"]
    if mutation == "wrong_engine":
        cms["lcms_version"] = "unrecognized 1.0"
    elif mutation == "version_disagrees":
        cms["imagecodecs_version"] = "0.0.0"
    elif mutation == "path_escape":
        cms["bundled_lcms"] = {"../liblcms2.so": "ab" * 32}
    else:
        cms["bundled_lcms"] = {f"imagecodecs/liblcms2-{number}.so": "ab" * 32 for number in range(9)}
    with pytest.raises(ValueError):
        ExecutionPlanV5(refingerprint(payload))
    assert not prepared.output_root.exists()
