"""Unified CLI contracts with real plans, graphs, products, and semantic replay.

Only neural inference and native runtime observations use established controlled
fixtures. CLI execution, source admission, and artifact verification are real.
"""

from __future__ import annotations

import hashlib
import json
import signal

import numpy as np
import pytest
import tifffile

from tests.lux_depth_v5 import test_pipeline as v5_fixture
from tests.lux_depth_v6 import test_depth_pro_pipeline as native_fixture
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth.__main__ import main
from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request, prepare
from transformation_portal.materials_v4.artifacts import write_evidence
from transformation_portal.materials_v4.contracts import MaterialEvidence, RegionEvidence
from transformation_portal.materials_v4.engine import ResponsePolicy

pytestmark = pytest.mark.unit
request_case = v5_fixture.request_case
native_case = native_fixture.native_case


def _arguments(command, request):
    return [
        command,
        "--input-dir",
        str(request.input_dir),
        "--output-dir",
        str(request.output_dir),
        "--input-color",
        "srgb",
        "--target-size",
        "56",
    ]


def test_infer_plan_is_exact_native_plan_and_preserves_explicit_options(request_case, capsys, tmp_path):
    arguments = _arguments("infer", request_case) + [
        "--strength",
        "0.4",
        "--clarity",
        "0.1",
        "--precision",
        "fp16",
        "--refinement",
        "bilinear",
        "--preview-maps",
        "--runtime-python",
        "/explicit/da3/bin/python",
        "--raw-python",
        "/explicit/raw/bin/python",
        "--cache-dir",
        str(tmp_path / "retained-cache"),
        "--max-pixels",
        "2000",
        "--max-input-bytes",
        "50000",
        "--max-output-bytes",
        "5000000",
        "--memory-mib",
        "512",
        "--wall-time-seconds",
        "300",
    ]
    expected = prepare(
        LuxDepthV5Request(
            request_case.input_dir,
            request_case.output_dir,
            input_color="srgb",
            target_size=56,
            strength=0.4,
            clarity=0.1,
            precision="fp16",
            refinement="bilinear",
            preview_maps=True,
            runtime_python="/explicit/da3/bin/python",
            raw_python="/explicit/raw/bin/python",
            cache_dir=tmp_path / "retained-cache",
            max_pixels=2000,
            max_input_bytes=50000,
            max_output_bytes=5000000,
            memory_mib=512,
            wall_time_seconds=300,
        )
    )
    assert main([*arguments, "--plan"]) == 0
    assert capsys.readouterr().out.encode() == expected.canonical_plan_bytes
    assert not request_case.output_dir.exists()
    assert not (tmp_path / "retained-cache").exists()
    assert v5_fixture.SessionFixture.calls == 0


def test_process_runs_composite_and_verifies_exact_plan(request_case, capsys):
    arguments = _arguments("process", request_case) + ["--exposure-stops", "0.3", "--depth-refinement", "guided_bilinear_v4"]
    assert main([*arguments, "--plan"]) == 0
    planned = capsys.readouterr().out.encode()
    payload = json.loads(planned)
    assert payload["schema"] == "tp.execution.plan.v5"
    assert payload["inference"]["configuration"]["depth"]["precision"] == "fp32"
    assert payload["finishing"]["grade"]["exposure_stops"] == 0.3
    assert not request_case.output_dir.exists()
    assert v5_fixture.SessionFixture.calls == 0
    assert main(arguments) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["plan_sha256"] == hashlib.sha256(planned).hexdigest()
    assert summary["input_count"] == 1
    assert summary["production_acceptance"] == "not_established"
    output = request_case.output_dir
    assert (output / "source-v5/input-0000/native-depth.npy").is_file()
    assert (output / "v6/input-0000/depth-relative.tif").is_file()
    assert tifffile.imread(output / "v6/input-0000/delivery.tif").dtype == np.uint16
    assert main(["verify", "--output-dir", str(output), "--expected-plan-sha256", summary["plan_sha256"]]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True


def test_infer_materials_plan_run_and_rehashed_verification(request_case, capsys, tmp_path):
    source = request_case.input_dir / "ramp.tif"
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    material_root = tmp_path / "materials"
    material_root.mkdir()
    mask = np.zeros((28, 42), dtype=np.float32)
    mask[:, :21] = 1
    evidence = MaterialEvidence(source_digest, (28, 42), (RegionEvidence("glass-1", "glass", mask, 0.99),))
    bundle = material_root / "evidence.json"
    write_evidence(evidence, bundle)
    manifest = material_root / "materials.json"
    manifest.write_bytes(
        canonicalize_json(
            {
                "schema": "tp.lux.materials_manifest.v1",
                "inputs": [
                    {
                        "path": source.name,
                        "source_sha256": source_digest,
                        "evidence_path": bundle.name,
                        "evidence_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                        "shape": [28, 42],
                    }
                ],
            }
        )
    )
    policy = ResponsePolicy(max_abs_delta=0.001)
    policy_path = material_root / "policy.json"
    policy_path.write_bytes(canonicalize_json(policy.to_payload()))
    arguments = _arguments("infer", request_case) + [
        "--strength",
        "0",
        "--materials-manifest",
        str(manifest),
        "--materials-policy",
        str(policy_path),
    ]
    assert main([*arguments, "--plan"]) == 0
    planned = capsys.readouterr().out.encode()
    assert json.loads(planned)["configuration"]["materials_v4"] == policy.to_payload()
    assert main(arguments) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["plan_sha256"] == hashlib.sha256(planned).hexdigest()
    receipt = json.loads((request_case.output_dir / "input-0000/photograph.json").read_bytes())["materials"]
    assert receipt["status"] == "applied"
    assert 0 < receipt["max_abs_delta"] <= 0.001
    assert main(["verify", "--output-dir", str(request_case.output_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    (request_case.output_dir / "input-0000/master.npy").write_bytes(b"tampered")
    assert main(["verify", "--output-dir", str(request_case.output_dir)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "lux-depth:" in captured.err


@pytest.mark.parametrize("depth_maps", [True, False])
def test_finish_plans_executes_and_replays_retained_generation(request_case, tmp_path, capsys, depth_maps):
    parent = v5_fixture.execute(request_case).output_root
    target = tmp_path / "finished"
    arguments = ["finish", "--input-dir", str(parent), "--output-dir", str(target), "--saturation", "1.1"]
    if not depth_maps:
        arguments.append("--no-depth-maps")
    assert main([*arguments, "--plan"]) == 0
    plan = capsys.readouterr().out.encode()
    assert json.loads(plan)["schema"] == ("tp.lux.grade.plan.v2" if depth_maps else "tp.lux.grade.plan.v1")
    assert not target.exists()
    assert main(arguments) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["plan_sha256"] == hashlib.sha256(plan).hexdigest()
    assert main(["verify", "--output-dir", str(target), "--source-root", str(parent)]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    assert (target / "input-0000/depth-relative.tif").exists() is depth_maps


def test_depth_pro_uses_native_research_authority_and_verification(native_case, capsys):
    arguments = [
        "depth-pro",
        "--input-dir",
        str(native_case.input_dir),
        "--output-dir",
        str(native_case.output_dir),
        "--depth-pro-python",
        str(native_case.python_executable),
        "--depth-pro-checkpoint",
        str(native_case.checkpoint),
        "--input-color",
        "srgb",
    ]
    assert main([*arguments, "--plan"]) == 1
    assert "acknowledgements" in capsys.readouterr().err
    assert not native_case.output_dir.exists()
    arguments += ["--non-commercial-ok", "--accept-apple-depth-pro-research-license"]
    assert main([*arguments, "--plan"]) == 0
    planned = capsys.readouterr().out.encode()
    assert json.loads(planned)["schema"] == "tp.lux.depth_pro.plan.v1"
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["plan_sha256"] == hashlib.sha256(planned).hexdigest()
    assert main(["verify", "--output-dir", str(native_case.output_dir), "--input-dir", str(native_case.input_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signal_cancellation_restores_handlers_and_refuses_completion(request_case, monkeypatch, capsys, signum):
    compute = v5_fixture.SessionFixture.compute
    previous = {value: signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)}

    def interrupted(self, proxy):
        signal.raise_signal(signum)
        return compute(self, proxy)

    monkeypatch.setattr(v5_fixture.SessionFixture, "compute", interrupted)
    assert main(_arguments("infer", request_case)) == 128 + signum
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cancelled" in captured.err.lower()
    assert not (request_case.output_dir / "execution-evidence.json").exists()
    assert {value: signal.getsignal(value) for value in previous} == previous


@pytest.mark.parametrize(
    "command,extra",
    [
        ("process", ["--materials-manifest", "unused.json"]),
        ("infer", ["--exposure-stops", "1"]),
        ("infer", ["--refinement", "guided_bilinear_v4"]),
        ("finish", ["--precision", "fp16"]),
        ("verify", ["--strength", "0.2"]),
    ],
)
def test_inapplicable_options_are_errors_not_silently_discarded(tmp_path, command, extra, capsys):
    arguments = [command, "--output-dir", str(tmp_path / "output")]
    if command != "verify":
        arguments += ["--input-dir", str(tmp_path / "input")]
    with pytest.raises(SystemExit) as caught:
        main([*arguments, *extra])
    assert caught.value.code == 2
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "output").exists()


def test_policy_is_bounded_before_any_backend_or_output(request_case, tmp_path, capsys):
    policy = tmp_path / "policy.json"
    policy.write_bytes(b" " * 65537)
    assert main([*_arguments("infer", request_case), "--materials-policy", str(policy), "--plan"]) == 1
    assert capsys.readouterr().out == ""
    assert not request_case.output_dir.exists()
    assert v5_fixture.SessionFixture.calls == 0


def test_legacy_help_passes_through_to_v3(capsys):
    assert main(["legacy", "--help"]) == 0
    captured = capsys.readouterr()
    assert "--quality-tier" in captured.out
    assert "--output-bit-depth" in captured.out


def test_finish_rejects_unused_refinement(request_case, tmp_path, capsys):
    assert (
        main(
            [
                "finish",
                "--input-dir",
                str(request_case.input_dir),
                "--output-dir",
                str(tmp_path / "finished"),
                "--no-depth-maps",
                "--depth-refinement",
                "bilinear",
                "--plan",
            ]
        )
        == 1
    )
    assert "requires --depth-maps" in capsys.readouterr().err
    assert not (tmp_path / "finished").exists()
