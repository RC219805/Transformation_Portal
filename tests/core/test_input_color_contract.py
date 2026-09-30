"""Color policy stays explicit across CLI, frozen plans, and replay identities."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.lux_depth_v4 import test_materials_v4 as material_fixture
from tests.lux_depth_v6 import test_depth_pro_pipeline as native_fixture
from transformation_portal.core.execution_plan_v2 import parse_execution_plan
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.lifecycle import prepare as prepare_v4
from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request
from transformation_portal.lux_depth_v5.lifecycle import prepare as prepare_v5
from transformation_portal.lux_depth_v6 import depth_pro

pytestmark = pytest.mark.unit
material_case = material_fixture.material_case
native_case = native_fixture.native_case
_MODES = ("auto", "auto_assume_srgb", "srgb", "linear_srgb")


@pytest.mark.parametrize("version", ["v2", "v3", "v4", "v4_materials"])
@pytest.mark.parametrize("mode", [*_MODES, "assume_any_profile"])
def test_color_mode_roundtrips_in_legacy_and_current_plan_families(material_case, version, mode):
    request = replace(material_case.request, input_color=mode)
    if version in {"v2", "v4"}:
        request = replace(request, materials_manifest=None)
    prepare = prepare_v4
    if version.startswith("v4"):
        request = LuxDepthV5Request(**vars(request))
        prepare = prepare_v5
    if mode not in _MODES:
        with pytest.raises(ValueError, match="Invalid V4 configuration"):
            prepare(request)
    else:
        prepared = prepare(request)
        payload = prepared.plan.to_payload()
        assert payload["schema"] == f"tp.execution.plan.{version.split('_')[0]}"
        assert payload["configuration"]["input_color"] == mode
        assert payload["nodes"][0]["configuration"]["input_color"] == mode
        assert parse_execution_plan(prepared.canonical_plan_bytes).canonical_bytes == prepared.canonical_plan_bytes
    assert not request.output_dir.exists()


@pytest.mark.parametrize("route", ["v4", "v5", "infer", "process"])
@pytest.mark.parametrize("mode", [None, "auto_assume_srgb", "assume_any_profile"])
def test_cli_color_mode_reaches_exact_plan_without_changing_default(material_case, capfd, route, mode):
    module = "lux_depth" if route in {"infer", "process"} else f"lux_depth_{route}"
    main = importlib.import_module(f"transformation_portal.{module}.__main__").main
    args = [route] if module == "lux_depth" else []
    args += [
        "--input-dir",
        str(material_case.request.input_dir),
        "--output-dir",
        str(material_case.request.output_dir),
        "--plan",
    ]
    if mode is not None:
        args += ["--input-color", mode]
    if mode == "assume_any_profile":
        with pytest.raises(SystemExit) as failure:
            main(args)
        assert failure.value.code == 2
    else:
        assert main(args) == 0
        payload = json.loads(capfd.readouterr().out)
        inference = payload["inference"] if route == "process" else payload
        assert inference["configuration"]["input_color"] == (mode or "auto")
        assert inference["nodes"][0]["configuration"]["input_color"] == (mode or "auto")
        assert parse_execution_plan(canonicalize_json(payload)).to_payload() == payload
    assert not material_case.request.output_dir.exists()


@pytest.mark.parametrize("mode", [*_MODES, "assume_any_profile"])
def test_depth_pro_plan_color_enum_is_additive_and_closed(native_case, mode):
    payload = depth_pro.prepare(native_case).plan.to_payload()
    payload["input_color"] = mode
    if mode not in _MODES:
        with pytest.raises(ValueError, match="color"):
            depth_pro.NativeDepthProPlan(canonicalize_json(payload))
    else:
        plan = depth_pro.NativeDepthProPlan(canonicalize_json(payload))
        assert plan.to_payload()["input_color"] == mode
        assert plan.to_payload()["schema"] == "tp.lux.depth_pro.plan.v1"


@pytest.mark.parametrize("route", ["unified", "v6"])
def test_depth_pro_cli_binds_assumption_mode_and_replay(native_case, capfd, route):
    module = "lux_depth" if route == "unified" else "lux_depth_v6"
    main = importlib.import_module(f"transformation_portal.{module}.__main__").main
    args = ["depth-pro"] if route == "unified" else ["--depth-backend", "depth-pro"]
    args += [
        "--input-dir",
        str(native_case.input_dir),
        "--output-dir",
        str(native_case.output_dir),
        "--depth-pro-python",
        str(native_case.python_executable),
        "--depth-pro-checkpoint",
        str(native_case.checkpoint),
        "--non-commercial-ok",
        "--accept-apple-depth-pro-research-license",
        "--input-color",
        "auto_assume_srgb",
    ]
    assert main(args + ["--plan"]) == 0
    planned = depth_pro.NativeDepthProPlan(capfd.readouterr().out.strip().encode())
    assert planned.to_payload()["input_color"] == "auto_assume_srgb"
    expected = depth_pro.prepare(replace(native_case, input_color="auto_assume_srgb"))
    assert planned.canonical_bytes == expected.canonical_plan_bytes
    result = depth_pro.run(expected)
    assert depth_pro.verify(result.output_root, source_root=native_case.input_dir).to_payload()["input_count"] == 1


@pytest.mark.parametrize("module_name", ["depth_pro", "plan"])
def test_replay_identity_binds_color_preparation_implementation(module_name):
    module = importlib.import_module(f"transformation_portal.lux_depth_v6.{module_name}")
    preparation = importlib.import_module("transformation_portal.lux_depth_v4.color_preparation")
    assert module.processing_identity()["modules"][preparation.__name__] == module.digest(
        Path(preparation.__file__).read_bytes()
    )
